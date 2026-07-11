"""Kaggle experiment exported from notebooks/ivan_german_image_preference_solution.ipynb.

This script is kept for review/reuse; run it inside a Kaggle notebook with the competition data and code dataset attached.
"""

from pathlib import Path
import sys, io, json, zipfile, shutil, gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

runner = next(Path("/kaggle/input").rglob("full_train.py"))
CODE_ROOT = runner.parents[1]
PKG_ROOT = CODE_ROOT / "image_preference"
sys.path.insert(0, str(PKG_ROOT))

from imgpref.config import auto_config, BackboneSpec
from imgpref.pipeline import run_train
from imgpref.submit import write_submission
from imgpref.cv import search_blend_weights

WORK_DIR = Path("/kaggle/working/work")
CACHE_DIR = WORK_DIR / "cache"
WORK_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

if not (WORK_DIR / "preds.npz").exists():
    zip_names = [
        "imgpref_dinov2_work.zip",
        "imgpref_dinov2_checkpoint.zip",
        "imgpref_siglip_work.zip",
        "imgpref_work_final.zip",
    ]
    hits = []
    for name in zip_names:
        hits += list(Path("/kaggle/input").rglob(name))
    if hits:
        print("restoring:", hits[0])
        with zipfile.ZipFile(hits[0]) as zf:
            zf.extractall(WORK_DIR)

cfg = auto_config(config_path=str(PKG_ROOT / "configs" / "default.yaml"))
cfg.work_dir = str(WORK_DIR)
cfg.cache_dir = str(CACHE_DIR)
cfg.finetune_enabled = False

for spec in [
    BackboneSpec("vit_large_patch16_siglip_384.webli", img_size=384, batch_size=24),
    BackboneSpec("vit_large_patch14_reg4_dinov2.lvd142m", img_size=392, batch_size=16),
]:
    if spec.name not in [s.name for s in cfg.backbones]:
        cfg.backbones.append(spec)

cfg.ensure_dirs()

if not (WORK_DIR / "preds.npz").exists():
    print("preds.npz not found, retraining heads from cached backbone features...")
    run_train(cfg)

RESAMPLE = getattr(Image, "Resampling", Image).BILINEAR

def image_quality_features(b):
    with Image.open(io.BytesIO(b)) as img:
        fmt = img.format or "UNK"
        w, h = img.size
        img = img.convert("RGB")
        img.thumbnail((256, 256), RESAMPLE)
        arr = np.asarray(img, dtype=np.float32) / 255.0

    r, g, bl = arr[..., 0], arr[..., 1], arr[..., 2]
    mx, mn = arr.max(axis=2), arr.min(axis=2)
    sat = (mx - mn) / (mx + 1e-6)
    gray = 0.299 * r + 0.587 * g + 0.114 * bl

    hist = np.histogram(gray, bins=32, range=(0, 1))[0].astype(np.float32)
    p = hist / (hist.sum() + 1e-8)
    entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum() / 5.0)

    if gray.shape[0] > 2 and gray.shape[1] > 2:
        lap = (
            -4 * gray[1:-1, 1:-1]
            + gray[:-2, 1:-1] + gray[2:, 1:-1]
            + gray[1:-1, :-2] + gray[1:-1, 2:]
        )
        sharp = float(lap.var())
        dx = np.abs(np.diff(gray, axis=1))
        dy = np.abs(np.diff(gray, axis=0))
        grad_mean = float((dx.mean() + dy.mean()) / 2)
        edge_density = float(((dx > 0.08).mean() + (dy > 0.08).mean()) / 2)
    else:
        sharp, grad_mean, edge_density = 0.0, 0.0, 0.0

    rg = r - g
    yb = 0.5 * (r + g) - bl
    colorfulness = float(
        np.sqrt(rg.var() + yb.var()) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    )

    feats = [
        np.log1p(len(b)), np.log1p(w), np.log1p(h), np.log1p(w * h),
        w / (h + 1e-6), h / (w + 1e-6),
        float(gray.mean()), float(gray.std()), float(np.percentile(gray, 5)),
        float(np.percentile(gray, 95)), float((gray < 0.05).mean()), float((gray > 0.95).mean()),
        float(sat.mean()), float(sat.std()), float((sat < 0.05).mean()), float((sat > 0.80).mean()),
        float(r.mean()), float(g.mean()), float(bl.mean()),
        float(r.std()), float(g.std()), float(bl.std()),
        sharp, grad_mean, edge_density, entropy, colorfulness,
        float((arr < 0.02).mean()), float((arr > 0.98).mean()),
        float(fmt == "JPEG"), float(fmt == "PNG"), float(fmt == "WEBP"),
    ]
    return np.asarray(feats, dtype=np.float32)

def extract_quality(parquet_path, split, has_target):
    out_path = WORK_DIR / f"quality_{split}_v1.npz"
    if out_path.exists():
        print("cached:", out_path)
        z = np.load(out_path)
        return z["q1"], z["q2"], z["y"] if has_target else None

    cols = [cfg.img1_col, cfg.img2_col] + ([cfg.target_col] if has_target else [])
    pf = pq.ParquetFile(parquet_path)
    q1_parts, q2_parts, y_parts = [], [], []

    with ThreadPoolExecutor(max_workers=8) as ex:
        pbar = tqdm(total=pf.metadata.num_rows, desc=f"quality {split}")
        for batch in pf.iter_batches(batch_size=128, columns=cols):
            d = batch.to_pydict()
            q1_parts.append(np.stack(list(ex.map(image_quality_features, d[cfg.img1_col]))))
            q2_parts.append(np.stack(list(ex.map(image_quality_features, d[cfg.img2_col]))))
            if has_target:
                y_parts.extend(d[cfg.target_col])
            pbar.update(len(d[cfg.img1_col]))
        pbar.close()

    q1 = np.vstack(q1_parts).astype(np.float32)
    q2 = np.vstack(q2_parts).astype(np.float32)
    y = np.asarray(y_parts, dtype=np.int64) if has_target else np.array([], dtype=np.int64)
    np.savez_compressed(out_path, q1=q1, q2=q2, y=y)
    print("saved:", out_path, q1.shape, q2.shape)
    return q1, q2, y if has_target else None

def pair_quality(q1, q2):
    diff = q1 - q2
    return np.concatenate([q1, q2, diff, np.abs(diff), q1 * q2], axis=1).astype(np.float32)

q1, q2, y = extract_quality(cfg.train_parquet, "train", True)
q1t, q2t, _ = extract_quality(cfg.test_parquet, "test", False)

X = pair_quality(q1, q2)
Xt = pair_quality(q1t, q2t)
print("quality pair features:", X.shape, Xt.shape)

folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, y))
q_oof = np.zeros(len(y), dtype=np.float32)
q_test = np.zeros(len(Xt), dtype=np.float32)
aucs = []

try:
    from catboost import CatBoostClassifier
    model_name = "catboost_quality"
    for fold, (tr, va) in enumerate(folds):
        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=3000,
            learning_rate=0.03,
            depth=5,
            l2_leaf_reg=10,
            random_seed=42 + fold,
            bootstrap_type="Bernoulli",
            subsample=0.85,
            od_type="Iter",
            od_wait=200,
            allow_writing_files=False,
            verbose=False,
        )
        model.fit(X[tr], y[tr], eval_set=(X[va], y[va]), use_best_model=True)
        q_oof[va] = model.predict_proba(X[va])[:, 1]
        q_test += model.predict_proba(Xt)[:, 1] / len(folds)
        auc = roc_auc_score(y[va], q_oof[va])
        aucs.append(auc)
        print(f"[quality/catboost] fold {fold}: AUC={auc:.5f}")
except Exception as e:
    print("CatBoost unavailable or failed, fallback to LightGBM:", repr(e))
    import lightgbm as lgb
    model_name = "lightgbm_quality"
    params = dict(
        objective="binary",
        metric="auc",
        learning_rate=0.025,
        num_leaves=31,
        feature_fraction=0.8,
        bagging_fraction=0.85,
        bagging_freq=1,
        min_child_samples=30,
        lambda_l2=10.0,
        verbosity=-1,
    )
    for fold, (tr, va) in enumerate(folds):
        dtr = lgb.Dataset(X[tr], label=y[tr])
        dva = lgb.Dataset(X[va], label=y[va])
        model = lgb.train(
            params, dtr, num_boost_round=3000, valid_sets=[dva],
            callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)],
        )
        q_oof[va] = model.predict(X[va], num_iteration=model.best_iteration)
        q_test += model.predict(Xt, num_iteration=model.best_iteration) / len(folds)
        auc = roc_auc_score(y[va], q_oof[va])
        aucs.append(auc)
        print(f"[quality/lgbm] fold {fold}: AUC={auc:.5f}")

quality_auc = roc_auc_score(y, q_oof)
print(f"[quality] OOF AUC={quality_auc:.5f}")

base = np.load(WORK_DIR / "preds.npz", allow_pickle=True)
base_oof = [base["oof"][i] for i in range(base["oof"].shape[0])]
base_test = [base["test"][i] for i in range(base["test"].shape[0])]
base_names = [str(x) for x in base["names"]]

all_oof = base_oof + [q_oof]
all_test = base_test + [q_test]
all_names = base_names + [model_name]

weights, ensemble_auc = search_blend_weights(all_oof, y, step=0.05)

def rank01(p):
    return np.argsort(np.argsort(p)).astype(np.float64) / (len(p) - 1)

ensemble_test = sum(w * rank01(p) for w, p in zip(weights, all_test))

quality_sub = write_submission(
    q_test,
    cfg.sample_submission,
    "/kaggle/working/submission_quality_only.csv",
    cfg.target_col,
)
ensemble_sub = write_submission(
    ensemble_test,
    cfg.sample_submission,
    "/kaggle/working/submission_quality_ensemble.csv",
    cfg.target_col,
)

report = {
    "quality_model": model_name,
    "quality_fold_auc": [float(x) for x in aucs],
    "quality_oof_auc": float(quality_auc),
    "ensemble_names": all_names,
    "ensemble_weights": {name: float(w) for name, w in zip(all_names, weights)},
    "ensemble_oof_auc": float(ensemble_auc),
    "quality_submission": quality_sub,
    "ensemble_submission": ensemble_sub,
}
Path("/kaggle/working/report_quality.json").write_text(json.dumps(report, indent=2))
np.savez_compressed(WORK_DIR / "quality_preds.npz", oof=q_oof, test=q_test, y=y)
archive = shutil.make_archive("/kaggle/working/imgpref_quality_work", "zip", root_dir=WORK_DIR)

print(json.dumps(report, indent=2))
print("archive:", archive)