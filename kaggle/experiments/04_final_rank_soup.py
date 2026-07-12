"""Kaggle experiment exported from notebooks/ivan_german_image_preference_solution.ipynb.

This script is kept for review/reuse; run it inside a Kaggle notebook with the competition data and code dataset attached.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

TARGET = "is_image1_better"

def find_csv(name):
    hits = list(Path("/kaggle/working").rglob(name)) + list(Path("/kaggle/input").rglob(name))
    if not hits:
        raise FileNotFoundError(f"Не нашел {name}")
    print(name, "->", hits[0])
    return hits[0]

def rank01(x):
    x = np.asarray(x, dtype=np.float64)
    return np.argsort(np.argsort(x)).astype(np.float64) / (len(x) - 1)

def load_ranked(name):
    path = find_csv(name)
    df = pd.read_csv(path)
    return df, rank01(df[TARGET].values)

base_df, dino = load_ranked("submission_dinov2.csv")
_, quality = load_ranked("submission_quality_ensemble.csv")
_, siglip = load_ranked("submission_siglip.csv")

blend = rank01(
    0.75 * dino +
    0.20 * quality +
    0.05 * siglip
)

out = base_df.copy()
out[TARGET] = blend

out_path = Path("/kaggle/working/submission_soup_dino75_quality20_siglip05.csv")
out.to_csv(out_path, index=False)

report = {
    "submission": str(out_path),
    "weights": {
        "dino": 0.75,
        "quality": 0.20,
        "siglip": 0.05,
    },
}
Path("/kaggle/working/report_soup_dino75_quality20_siglip05.json").write_text(json.dumps(report, indent=2))

print(json.dumps(report, indent=2))
print("wrote:", out_path)