"""Optional end-to-end siamese finetuning (higher ceiling than frozen features).

A single backbone is shared across both images; a small head produces a scalar
score and the pair logit is ``s1 - s2``. Designed for Kaggle 2x T4: AMP fp16 +
``DataParallel`` across both GPUs. This is the slow, high-ceiling path; the
frozen-feature pipeline runs independently and the two can be blended.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..config import Config
from ..utils import roc_auc


class SiameseScorer:
    """Wraps a timm backbone + MLP head into a per-image scalar scorer."""

    def __init__(self, backbone: str, img_size: int, pretrained: bool, dropout: float = 0.3):
        import torch.nn as nn
        import timm
        try:
            self.backbone = timm.create_model(backbone, pretrained=pretrained,
                                               num_classes=0, img_size=img_size)
        except TypeError:
            self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 1)
        )

    def module(self):
        import torch.nn as nn

        backbone, head = self.backbone, self.head

        class _M(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = backbone
                self.head = head

            def forward(self, x):
                return self.head(self.backbone(x)).squeeze(1)

        return _M()


def _make_transform(img_size: int, training: bool):
    from torchvision import transforms as T
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    if training:
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.1, 0.1, 0.1, 0.0),
            T.ToTensor(), T.Normalize(mean, std),
        ])
    return T.Compose([
        T.Resize(int(img_size * 1.14)), T.CenterCrop(img_size),
        T.ToTensor(), T.Normalize(mean, std),
    ])


def finetune_cv(bytes1, bytes2, y, te_bytes1, te_bytes2, folds, cfg: Config,
                device: str) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """Train the siamese model per fold; return (oof_logit, test_logit, aucs)."""
    import torch
    from torch.utils.data import DataLoader
    from ..data import PairImageDataset

    tr_tf = _make_transform(cfg.finetune_img_size, training=True)
    ev_tf = _make_transform(cfg.finetune_img_size, training=False)
    use_dp = torch.cuda.device_count() > 1

    oof = np.zeros(len(y), dtype=np.float32)
    test = np.zeros(len(te_bytes1), dtype=np.float32)
    aucs: List[float] = []

    te_ds = PairImageDataset(te_bytes1, te_bytes2, ev_tf, labels=None)
    te_loader = DataLoader(te_ds, batch_size=cfg.finetune_batch_size, shuffle=False,
                           num_workers=cfg.num_workers, pin_memory=True)

    for fold, (tr, va) in enumerate(folds):
        scorer = SiameseScorer(cfg.finetune_backbone, cfg.finetune_img_size, cfg.pretrained)
        model = scorer.module().to(device)
        if use_dp:
            model = torch.nn.DataParallel(model)

        b1 = [bytes1[i] for i in tr]; b2 = [bytes2[i] for i in tr]
        tr_ds = PairImageDataset(b1, b2, tr_tf, labels=y[tr], swap_aug=cfg.bt_swap_aug)
        tr_loader = DataLoader(tr_ds, batch_size=cfg.finetune_batch_size, shuffle=True,
                               num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
        vb1 = [bytes1[i] for i in va]; vb2 = [bytes2[i] for i in va]
        va_ds = PairImageDataset(vb1, vb2, ev_tf, labels=y[va])
        va_loader = DataLoader(va_ds, batch_size=cfg.finetune_batch_size, shuffle=False,
                               num_workers=cfg.num_workers, pin_memory=True)

        base = model.module if use_dp else model
        param_groups = [
            {"params": base.backbone.parameters(), "lr": cfg.finetune_lr},
            {"params": base.head.parameters(), "lr": cfg.finetune_head_lr},
        ]
        opt = torch.optim.AdamW(param_groups, weight_decay=1e-2)
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device == "cuda")
        lossf = torch.nn.BCEWithLogitsLoss()

        best_auc, best_state = -1.0, None
        for epoch in range(cfg.finetune_epochs):
            freeze = epoch < cfg.finetune_freeze_epochs
            for p in base.backbone.parameters():
                p.requires_grad = not freeze
            model.train()
            for x1, x2, yb in tr_loader:
                x1 = x1.to(device, non_blocking=True)
                x2 = x2.to(device, non_blocking=True)
                yb = yb.float().to(device)
                opt.zero_grad()
                with torch.cuda.amp.autocast(enabled=cfg.amp and device == "cuda"):
                    logit = model(x1) - model(x2)
                    loss = lossf(logit, yb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

            model.eval()
            vp = []
            with torch.no_grad():
                for x1, x2, _ in va_loader:
                    x1 = x1.to(device); x2 = x2.to(device)
                    with torch.cuda.amp.autocast(enabled=cfg.amp and device == "cuda"):
                        logit = model(x1) - model(x2)
                    vp.append(logit.float().cpu().numpy())
            vp = np.concatenate(vp)
            a = roc_auc(y[va], vp)
            print(f"  [FT] fold {fold} epoch {epoch}: AUC={a:.5f}")
            if a > best_auc:
                best_auc = a
                best_state = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
                oof[va] = vp

        if best_state is not None:
            base.load_state_dict(best_state)
        model.eval()
        tp = []
        with torch.no_grad():
            for x1, x2, _ in te_loader:
                x1 = x1.to(device); x2 = x2.to(device)
                with torch.cuda.amp.autocast(enabled=cfg.amp and device == "cuda"):
                    logit = model(x1) - model(x2)
                tp.append(logit.float().cpu().numpy())
        test += np.concatenate(tp) / len(folds)
        aucs.append(best_auc)
        del model, base
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"  [FT] OOF AUC={roc_auc(y, oof):.5f}")
    return oof, test, aucs
