"""Frozen-backbone feature extraction with caching.

For each backbone we produce one L2-normalized embedding per image, for both
slots and both splits, and cache them as ``.npy``. Re-running is cheap because
extraction is skipped when a cache file already exists.
"""
from __future__ import annotations

import hashlib
import os
from typing import List

import numpy as np

from .config import BackboneSpec, Config
from .utils import timer


def _safe_name(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:10] + "_" + s.replace("/", "_").replace(".", "-")[:40]


def cache_path(cfg: Config, backbone: str, split: str, col: str) -> str:
    return os.path.join(cfg.cache_dir, f"feat__{_safe_name(backbone)}__{split}__{col}.npy")


def build_model_and_transform(spec: BackboneSpec, pretrained: bool, device: str):
    """Create a timm backbone (classifier head removed) + matching transform."""
    import timm
    from timm.data import resolve_model_data_config, create_transform

    kwargs = dict(pretrained=pretrained, num_classes=0)
    if spec.img_size is not None:
        kwargs["img_size"] = spec.img_size
    try:
        model = timm.create_model(spec.name, **kwargs)
    except TypeError:
        # some models don't accept img_size; fall back to native
        kwargs.pop("img_size", None)
        model = timm.create_model(spec.name, **kwargs)

    if spec.weight_path:
        import torch
        state = torch.load(spec.weight_path, map_location="cpu")
        state = state.get("state_dict", state) if isinstance(state, dict) else state
        model.load_state_dict(state, strict=False)

    model = model.eval().to(device)

    data_cfg = resolve_model_data_config(model)
    if spec.img_size is not None:
        data_cfg["input_size"] = (3, spec.img_size, spec.img_size)
    transform = create_transform(**data_cfg, is_training=False)
    return model, transform


def _extract_stream(model, parquet: str, col: str, transform, batch_size: int,
                    device: str, amp: bool, tta: bool, desc: str,
                    num_decode_threads: int = 8) -> np.ndarray:
    """Stream a parquet binary column through the model, batch by batch.

    Memory-safe: only one row-group chunk of bytes and one model batch of
    tensors are held at a time (no full-column list, no DataLoader workers).
    Progress is reported per image via tqdm.
    """
    import io
    import torch
    import pyarrow.parquet as pq
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image

    autocast = torch.cuda.amp.autocast if device == "cuda" else _nullcast
    pf = pq.ParquetFile(parquet)

    def decode_tf(b):
        return transform(Image.open(io.BytesIO(b)).convert("RGB"))

    def run_batch(tensors) -> np.ndarray:
        x = torch.stack(tensors).to(device, non_blocking=True)
        with torch.no_grad(), autocast(enabled=amp):
            e = model(x).float()
            if tta:
                e = (e + model(torch.flip(x, dims=[3])).float()) / 2.0
        return torch.nn.functional.normalize(e, dim=1).cpu().numpy().astype(np.float32)

    try:
        from tqdm.auto import tqdm
        pbar = tqdm(total=pf.metadata.num_rows, desc=desc, leave=True)
    except Exception:
        pbar = None

    feats: List[np.ndarray] = []
    carry: List["torch.Tensor"] = []
    # PIL releases the GIL during JPEG/WEBP decode, so threads parallelize the
    # CPU-bound decode while the GPU stays fed; bytes are streamed in chunks.
    with ThreadPoolExecutor(max_workers=num_decode_threads) as ex:
        for rb in pf.iter_batches(batch_size=128, columns=[col]):
            carry.extend(ex.map(decode_tf, rb.to_pydict()[col]))
            del rb
            while len(carry) >= batch_size:
                feats.append(run_batch(carry[:batch_size]))
                if pbar is not None:
                    pbar.update(batch_size)
                carry = carry[batch_size:]
        if carry:
            feats.append(run_batch(carry))
            if pbar is not None:
                pbar.update(len(carry))
    if pbar is not None:
        pbar.close()
    return np.concatenate(feats, axis=0)


class _nullcast:
    def __init__(self, enabled=False):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def extract_for_backbone(cfg: Config, spec: BackboneSpec) -> None:
    """Extract & cache features for one backbone across all splits/columns.

    Streams each parquet column straight through the model to keep RAM bounded
    (the raw image bytes for a column can be several GB).
    """
    import gc
    import torch

    device = cfg.device
    targets = [
        ("train", cfg.train_parquet, cfg.img1_col),
        ("train", cfg.train_parquet, cfg.img2_col),
        ("test", cfg.test_parquet, cfg.img1_col),
        ("test", cfg.test_parquet, cfg.img2_col),
    ]
    if all(os.path.exists(cache_path(cfg, spec.name, s, c)) for s, _, c in targets):
        print(f"[features] {spec.name}: all cached, skip")
        return

    model, transform = build_model_and_transform(spec, cfg.pretrained, device)
    model.eval()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    for split, parquet, col in targets:
        out = cache_path(cfg, spec.name, split, col)
        if os.path.exists(out):
            print(f"  cached {out}")
            continue
        with timer(f"extract {spec.name} {split}/{col}"):
            feats = _extract_stream(model, parquet, col, transform, spec.batch_size,
                                    device, cfg.amp, cfg.use_tta_hflip,
                                    desc=f"{spec.name.split('.')[0]} {split}/{col}",
                                    num_decode_threads=cfg.decode_threads)
            np.save(out, feats)
            print(f"  saved {out}  shape={feats.shape}")
            del feats
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


def extract_all(cfg: Config) -> None:
    for spec in cfg.backbones:
        extract_for_backbone(cfg, spec)


def load_features(cfg: Config, split: str):
    """Concatenate cached features from all backbones for a split.

    Returns ``(F1, F2)`` with shape ``[N, sum_of_dims]`` each (slot 1 / slot 2),
    plus the list of per-backbone dims for optional per-backbone modeling.
    """
    f1_parts, f2_parts, dims = [], [], []
    for spec in cfg.backbones:
        p1 = cache_path(cfg, spec.name, split, cfg.img1_col)
        p2 = cache_path(cfg, spec.name, split, cfg.img2_col)
        a1 = np.load(p1)
        a2 = np.load(p2)
        f1_parts.append(a1)
        f2_parts.append(a2)
        dims.append(a1.shape[1])
    F1 = np.concatenate(f1_parts, axis=1).astype(np.float32)
    F2 = np.concatenate(f2_parts, axis=1).astype(np.float32)
    return F1, F2, dims
