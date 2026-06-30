"""Bradley-Terry head over frozen features.

A shared MLP maps each image embedding to a scalar "quality" score; the pair
logit is ``s(f1) - s(f2)``. This is antisymmetric by construction and directly
optimizes ranking (= ROC AUC). Trained per CV fold on (F1, F2, y).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..utils import roc_auc


def _build_scorer(in_dim: int, hidden: List[int], dropout: float):
    import torch.nn as nn
    layers: List[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
        d = h
    layers += [nn.Linear(d, 1)]
    return nn.Sequential(*layers)


def _train_one_fold(F1_tr, F2_tr, y_tr, F1_va, F2_va, y_va, cfg, device) -> Tuple[np.ndarray, object]:
    import torch
    import torch.nn as nn

    in_dim = F1_tr.shape[1]
    model = _build_scorer(in_dim, cfg.bt_hidden, cfg.bt_dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.bt_lr, weight_decay=cfg.bt_weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.bt_epochs)
    lossf = nn.BCEWithLogitsLoss()

    t1 = torch.from_numpy(F1_tr); t2 = torch.from_numpy(F2_tr)
    ty = torch.from_numpy(y_tr.astype(np.float32))
    v1 = torch.from_numpy(F1_va).to(device); v2 = torch.from_numpy(F2_va).to(device)

    n = len(y_tr)
    bs = cfg.bt_batch_size
    best_auc, best_state, patience = -1.0, None, 0

    for epoch in range(cfg.bt_epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) == 1:  # BatchNorm1d needs >1 sample in train mode
                continue
            a = t1[idx].to(device); b = t2[idx].to(device)
            yb = ty[idx].to(device)
            if cfg.bt_swap_aug:
                swap = torch.rand(len(idx), device=device) < 0.5
                a2 = torch.where(swap[:, None], b, a)
                b2 = torch.where(swap[:, None], a, b)
                yb = torch.where(swap, 1.0 - yb, yb)
                a, b = a2, b2
            logit = (model(a) - model(b)).squeeze(1)
            loss = lossf(logit, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            vlogit = (model(v1) - model(v2)).squeeze(1).cpu().numpy()
        auc = roc_auc(y_va, vlogit)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.bt_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_auc, model


def predict_scores(model, F, device, batch: int = 4096) -> np.ndarray:
    """Per-image scalar score s(f) for the whole array."""
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(F), batch):
            x = torch.from_numpy(F[i:i + batch]).to(device)
            out.append(model(x).squeeze(1).cpu().numpy())
    return np.concatenate(out)


def train_bt_cv(F1, F2, y, F1_te, F2_te, folds, cfg, device) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """Returns (oof_logits, test_logits, per_fold_auc)."""
    oof = np.zeros(len(y), dtype=np.float32)
    test = np.zeros(len(F1_te), dtype=np.float32)
    aucs: List[float] = []
    for fold, (tr, va) in enumerate(folds):
        _, model = _train_one_fold(
            F1[tr], F2[tr], y[tr], F1[va], F2[va], y[va], cfg, device)
        s1v = predict_scores(model, F1[va], device)
        s2v = predict_scores(model, F2[va], device)
        oof[va] = s1v - s2v
        s1t = predict_scores(model, F1_te, device)
        s2t = predict_scores(model, F2_te, device)
        test += (s1t - s2t) / len(folds)
        a = roc_auc(y[va], oof[va])
        aucs.append(a)
        print(f"  [BT] fold {fold}: AUC={a:.5f}")
    print(f"  [BT] OOF AUC={roc_auc(y, oof):.5f}")
    return oof, test, aucs
