"""Write a competition submission file."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd


def write_submission(test_scores: np.ndarray, sample_submission_path: str,
                     out_path: str, target_col: str = "is_image1_better") -> str:
    """Write predictions in the sample-submission format.

    ROC AUC only needs the ranking, so we emit the (rank-normalized) score as a
    pseudo-probability. The index column is taken from the sample file.
    """
    sub = pd.read_csv(sample_submission_path)
    assert len(sub) == len(test_scores), (len(sub), len(test_scores))
    ranks = np.argsort(np.argsort(test_scores)).astype(np.float64)
    sub[target_col] = ranks / (len(ranks) - 1)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sub.to_csv(out_path, index=False)
    print(f"[submit] wrote {out_path}  ({len(sub)} rows)")
    return out_path
