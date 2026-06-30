"""Cross-validation splits, rank utilities and OOF blend search."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .utils import roc_auc


def make_folds(y: np.ndarray, n_folds: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Stratified K-fold indices (sklearn if available, else a numpy fallback)."""
    try:
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        return [(tr, va) for tr, va in skf.split(np.zeros(len(y)), y)]
    except Exception:
        rng = np.random.RandomState(seed)
        folds_va = [[] for _ in range(n_folds)]
        for cls in np.unique(y):
            idx = np.where(y == cls)[0]
            rng.shuffle(idx)
            for k, chunk in enumerate(np.array_split(idx, n_folds)):
                folds_va[k].extend(chunk.tolist())
        out = []
        allidx = np.arange(len(y))
        for k in range(n_folds):
            va = np.array(sorted(folds_va[k]))
            tr = np.setdiff1d(allidx, va)
            out.append((tr, va))
        return out


def rankavg(preds: List[np.ndarray]) -> np.ndarray:
    """Average predictions on the rank scale (robust to differing score ranges)."""
    ranks = []
    for p in preds:
        order = np.argsort(np.argsort(p))
        ranks.append(order / (len(p) - 1 + 1e-9))
    return np.mean(ranks, axis=0)


def search_blend_weights(oof_list: List[np.ndarray], y: np.ndarray,
                         step: float = 0.1) -> Tuple[List[float], float]:
    """Grid-search convex blend weights over rank-transformed OOF preds."""
    k = len(oof_list)
    ranks = [np.argsort(np.argsort(p)).astype(np.float64) / (len(p) - 1) for p in oof_list]
    if k == 1:
        return [1.0], roc_auc(y, ranks[0])

    best_w, best_auc = None, -1.0

    def gen(prefix, remaining, slots):
        if slots == 1:
            yield prefix + [round(remaining, 4)]
            return
        n = int(round(remaining / step))
        for i in range(n + 1):
            w = round(i * step, 4)
            yield from gen(prefix + [w], remaining - w, slots - 1)

    for w in gen([], 1.0, k):
        if any(x < 0 for x in w):
            continue
        blend = sum(wi * ri for wi, ri in zip(w, ranks))
        a = roc_auc(y, blend)
        if a > best_auc:
            best_auc, best_w = a, w
    return best_w, best_auc
