"""Turn per-image embeddings (F1, F2) + metadata into a pairwise table for GBDT.

The GBDT consumes an explicit "contrast" of the pair. We avoid asymmetric raw
concatenation by default (``compact`` mode) so the model sees antisymmetric
signal; ``full`` mode additionally appends both raw vectors.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def _cosine(F1: np.ndarray, F2: np.ndarray) -> np.ndarray:
    num = (F1 * F2).sum(axis=1)
    den = (np.linalg.norm(F1, axis=1) * np.linalg.norm(F2, axis=1)) + 1e-8
    return (num / den).astype(np.float32)


def build_meta_features(meta: dict) -> np.ndarray:
    """Codec/size prior. image_2 is always WEBP, so we encode image_1's format
    and log byte sizes plus their difference/ratio."""
    len1 = meta["len1"].astype(np.float32)
    len2 = meta["len2"].astype(np.float32)
    fmt1 = meta["fmt1"]
    is_jpeg = (fmt1 == "JPEG").astype(np.float32)
    is_png = (fmt1 == "PNG").astype(np.float32)
    is_webp = (fmt1 == "WEBP").astype(np.float32)
    l1 = np.log1p(len1)
    l2 = np.log1p(len2)
    feats = np.stack([
        l1, l2, l1 - l2, l1 / (l2 + 1.0),
        is_jpeg, is_png, is_webp,
    ], axis=1).astype(np.float32)
    return feats


def build_pair_features(F1: np.ndarray, F2: np.ndarray, meta: Optional[dict] = None,
                        mode: str = "compact") -> np.ndarray:
    """Construct the GBDT design matrix.

    compact: [F1-F2, F1*F2, |F1-F2|, cos, meta]
    full:    compact + [F1, F2]
    """
    diff = (F1 - F2).astype(np.float32)
    prod = (F1 * F2).astype(np.float32)
    adiff = np.abs(diff)
    cos = _cosine(F1, F2)[:, None]
    parts = [diff, prod, adiff, cos]
    if mode == "full":
        parts += [F1, F2]
    if meta is not None:
        parts.append(build_meta_features(meta))
    return np.concatenate(parts, axis=1).astype(np.float32)
