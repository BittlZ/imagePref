"""Small shared helpers: seeding, devices, timing, metric."""
from __future__ import annotations

import os
import random
import time
import contextlib
import numpy as np


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def get_device(prefer: str = "cuda") -> str:
    try:
        import torch
        if prefer == "cuda" and torch.cuda.is_available():
            return "cuda"
        if prefer == "mps" and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def n_gpus() -> int:
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


@contextlib.contextmanager
def timer(name: str):
    t0 = time.time()
    print(f"[{name}] start", flush=True)
    yield
    print(f"[{name}] done in {time.time() - t0:.1f}s", flush=True)


def roc_auc(y_true, y_score) -> float:
    """ROC AUC via the rank (Mann-Whitney) formulation; no sklearn dependency."""
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    # average ranks to handle ties
    ranks = np.empty(len(y_score), dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    i = 0
    n = len(sorted_scores)
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
