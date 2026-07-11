"""Kaggle experiment exported from notebooks/ivan_german_image_preference_solution.ipynb.

This script is kept for review/reuse; run it inside a Kaggle notebook with the competition data and code dataset attached.
"""

from pathlib import Path
import sys, json, zipfile, shutil, gc
import numpy as np

runner = next(Path("/kaggle/input").rglob("full_train.py"))
CODE_ROOT = runner.parents[1]
PKG_ROOT = CODE_ROOT / "image_preference"
sys.path.insert(0, str(PKG_ROOT))

from imgpref.config import auto_config, BackboneSpec
from imgpref.features import cache_path, load_features
from imgpref.data import read_label_and_meta
from imgpref.pair_features import build_pair_features
from imgpref.pipeline import run_train
from imgpref.submit import write_submission
from imgpref.cv import search_blend_weights
from imgpref.models.bt_head import train_bt_cv
from imgpref.utils import seed_everything, get_device, roc_auc

WORK_DIR = Path("/kaggle/working/work")
CACHE_DIR = WORK_DIR / "cache"
WORK_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

if len(list(CACHE_DIR.glob("feat__*.npy"))) < 16:
    for name in ["imgpref_dinov2_work.zip", "imgpref_quality_work.zip", "imgpref_siglip_work.zip", "imgpref_work_final.zip"]:
        hits = list(Path("/kaggle/input").rglob(name))
        if hits:
            print("restoring:", hits[0])
            with zipfile.ZipFile(hits[0]) as zf:
                zf.extractall(WORK_DIR)
            break

cfg = auto_config(config_path=str(PKG_ROOT / "configs" / "default.yaml"))
cfg.work_dir = str(WORK_DIR)
cfg.cache_dir = str(CACHE_DIR)
cfg.finetune_enabled = False
cfg.backbones = [
    BackboneSpec("convnextv2_base.fcmae_ft_in22k_in1k_384", img_size=384, batch_size=48),
    BackboneSpec("vit_large_patch14_clip_224.openai", img_size=224, batch_size=64),
    BackboneSpec("vit_large_patch16_siglip_384.webli", img_size=384, batch_size=24),
    BackboneSpec("vit_large_patch14_reg4_dinov2.lvd142m", img_size=392, batch_size=16),
]
cfg.ensure_dirs()

missing = []
for spec in cfg.backbones:
    for split, col in [("train", cfg.img1_col), ("train", cfg.img2_col), ("test", cfg.img1_col), ("test", cfg.img2_col)]:
        p = Path(cache_path(cfg, spec.name, split, col))
        if not p.exists():
            missing.append(str(p))
if missing:
    raise FileNotFoundError("Missing cached features:\n" + "\n".join(missing[:20]))

base_preds_path = Path("/kaggle/working/fast_stack_base_preds.npz")
if not base_preds_path.exists():
    print("training standard DINO4 heads...")
    base_report = run_train(cfg)
    shutil.copyfile(WORK_DIR / "preds.npz", base_preds_path)
    Path("/kaggle/working/report_fast_stack_base.json").write_text(json.dumps(base_report, indent=2))
else:
    print("using cached:", base_preds_path)

base = np.load(base_preds_path, allow_pickle=True)
base_names = [str(x) for x in base["names"]]
base_oofs = [base["oof"][i].astype(np.float32) for i in range(base["oof"].shape[0])]
base_tests = [base["test"][i].astype(np.float32) for i in range(base["test"].shape[0])]

print("loading features...")
F1, F2, dims = load_features(cfg, "train")
F1t, F2t, _ = load_features(cfg, "test")
print("dims:", dims, "train:", F1.shape, "test:", F1t.shape)

print("loading labels/meta...")
y, meta_tr = read_label_and_meta(cfg.train_parquet, cfg.target_col, cfg.img1_col, cfg.img2_col, has_target=True)
_, meta_te = read_label_and_meta(cfg.test_parquet, cfg.target_col, cfg.img1_col, cfg.img2_col, has_target=False)

from sklearn.model_selection import StratifiedKFold
folds = [(tr, va) for tr, va in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(np.zeros(len(y)), y)]

print("training BT seed bag...")
device = get_device(cfg.device)
bt_oofs, bt_tests = [], []
for seed in [7, 2026]:
    seed_everything(seed)
    cfg.seed = seed
    oof, test, aucs = train_bt_cv(F1, F2, y, F1t, F2t, folds, cfg, device)
    bt_oofs.append(oof)
    bt_tests.append(test)

bt_bag_oof = np.mean(bt_oofs, axis=0).astype(np.float32)
bt_bag_test = np.mean(bt_tests, axis=0).astype(np.float32)
print("[bt_bag] OOF AUC=", roc_auc(y, bt_bag_oof))

def swap_meta(meta):
    return {
        "len1": meta["len2"],
        "len2": meta["len1"],
        "fmt1": meta["fmt2"],
        "fmt2": meta["fmt1"],
    }

print("building antisymmetric pair features...")
Xf = build_pair_features(F1, F2, meta_tr, mode="compact")
Xr = build_pair_features(F2, F1, swap_meta(meta_tr), mode="compact")
Xtf = build_pair_features(F1t, F2t, meta_te, mode="compact")
Xtr = build_pair_features(F2t, F1t, swap_meta(meta_te), mode="compact")
print("X:", Xf.shape, Xtf.shape)

print("training antisymmetric LightGBM...")
import lightgbm as lgb

anti_oof = np.zeros(len(y), dtype=np.float32)
anti_test = np.zeros(len(Xtf), dtype=np.float32)
anti_aucs = []

params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.025,
    "num_leaves": 31,
    "feature_fraction": 0.55,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "min_child_samples": 60,
    "lambda_l2": 15.0,
    "verbosity": -1,
    "force_col_wise": True,
}

for fold, (tr, va) in enumerate(folds):
    X_train = np.concatenate([Xf[tr], Xr[tr]], axis=0)
    y_train = np.concatenate([y[tr], 1 - y[tr]], axis=0)

    dtr = lgb.Dataset(X_train, label=y_train)
    dva = lgb.Dataset(Xf[va], label=y[va])

    model = lgb.train(
        {**params, "seed": 100 + fold},
        dtr,
        num_boost_round=2500,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)],
    )

    pf = model.predict(Xf[va], num_iteration=model.best_iteration)
    pr = model.predict(Xr[va], num_iteration=model.best_iteration)
    anti_oof[va] = 0.5 * (pf + (1.0 - pr))

    ptf = model.predict(Xtf, num_iteration=model.best_iteration)
    ptr = model.predict(Xtr, num_iteration=model.best_iteration)
    anti_test += 0.5 * (ptf + (1.0 - ptr)) / len(folds)

    auc = roc_auc(y[va], anti_oof[va])
    anti_aucs.append(float(auc))
    print(f"[antisym_lgbm] fold {fold}: AUC={auc:.5f}, best_iter={model.best_iteration}")

    del X_train, y_train, dtr, dva, model
    gc.collect()

anti_auc = roc_auc(y, anti_oof)
print("[antisym_lgbm] OOF AUC=", anti_auc)

components = []
for name, oof, test in zip(base_names, base_oofs, base_tests):
    components.append((f"base_{name}", oof, test))
components.append(("bt_seed_bag", bt_bag_oof, bt_bag_test))
components.append(("antisym_lgbm", anti_oof, anti_test))

q_path = WORK_DIR / "quality_preds.npz"
if q_path.exists():
    q = np.load(q_path)
    if len(q["oof"]) == len(y) and len(q["test"]) == len(anti_test):
        components.append(("quality", q["oof"].astype(np.float32), q["test"].astype(np.float32)))
        print("added quality component")

def rank01(p):
    p = np.asarray(p, dtype=np.float64)
    return np.argsort(np.argsort(p)).astype(np.float64) / (len(p) - 1)

def stack_matrix(preds):
    ranks = [rank01(p) for p in preds]
    conf = [np.abs(r - 0.5) for r in ranks]
    return np.column_stack(ranks + conf).astype(np.float32)

names = [c[0] for c in components]
oof_list = [c[1] for c in components]
test_list = [c[2] for c in components]

print("components:")
for name, oof, _ in components:
    print(" -", name, "OOF=", roc_auc(y, oof))

from sklearn.linear_model import LogisticRegression

S = stack_matrix(oof_list)
St = stack_matrix(test_list)

meta_oof = np.zeros(len(y), dtype=np.float32)
meta_test = np.zeros(len(St), dtype=np.float32)

for fold, (tr, va) in enumerate(folds):
    clf = LogisticRegression(C=0.25, solver="lbfgs", max_iter=2000)
    clf.fit(S[tr], y[tr])
    meta_oof[va] = clf.predict_proba(S[va])[:, 1]
    meta_test += clf.predict_proba(St)[:, 1] / len(folds)
    print(f"[meta] fold {fold}: AUC={roc_auc(y[va], meta_oof[va]):.5f}")

meta_auc = roc_auc(y, meta_oof)

weights, blend_auc = search_blend_weights(oof_list, y, step=0.05)
blend_test = sum(w * rank01(p) for w, p in zip(weights, test_list))

sub_anti = write_submission(anti_test, cfg.sample_submission, "/kaggle/working/submission_antisym_lgbm.csv", cfg.target_col)
sub_stack = write_submission(meta_test, cfg.sample_submission, "/kaggle/working/submission_fast_stack.csv", cfg.target_col)
sub_oof_blend = write_submission(blend_test, cfg.sample_submission, "/kaggle/working/submission_fast_oof_blend.csv", cfg.target_col)

report = {
    "components": {name: float(roc_auc(y, oof)) for name, oof, _ in components},
    "antisym_fold_auc": anti_aucs,
    "antisym_oof_auc": float(anti_auc),
    "meta_oof_auc": float(meta_auc),
    "oof_blend_auc": float(blend_auc),
    "oof_blend_weights": {name: float(w) for name, w in zip(names, weights)},
    "submissions": {
        "antisym_lgbm": sub_anti,
        "fast_stack": sub_stack,
        "fast_oof_blend": sub_oof_blend,
    },
}
Path("/kaggle/working/report_fast_stack.json").write_text(json.dumps(report, indent=2))
np.savez_compressed(
    WORK_DIR / "fast_stack_preds.npz",
    y=y,
    names=np.array(names),
    oof=np.stack(oof_list),
    test=np.stack(test_list),
    meta_oof=meta_oof,
    meta_test=meta_test,
    anti_oof=anti_oof,
    anti_test=anti_test,
)
archive = shutil.make_archive("/kaggle/working/imgpref_fast_stack_work", "zip", root_dir=WORK_DIR)

print(json.dumps(report, indent=2))
print("archive:", archive)