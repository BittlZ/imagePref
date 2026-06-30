"""High-level orchestration: features -> heads -> CV/blend -> submission.

Typical Kaggle usage::

    from imgpref.config import auto_config
    from imgpref.pipeline import run_all
    cfg = auto_config()
    run_all(cfg)
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np

from .config import Config
from .utils import seed_everything, get_device, timer, roc_auc


def run_features(cfg: Config) -> None:
    from .features import extract_all
    with timer("feature-extraction"):
        extract_all(cfg)


def _load_meta(cfg: Config):
    from .data import read_label_and_meta
    y, meta_tr = read_label_and_meta(cfg.train_parquet, cfg.target_col,
                                     cfg.img1_col, cfg.img2_col, has_target=True)
    _, meta_te = read_label_and_meta(cfg.test_parquet, cfg.target_col,
                                     cfg.img1_col, cfg.img2_col, has_target=False)
    return y, meta_tr, meta_te


def run_train(cfg: Config) -> Dict:
    """Train all heads with CV, blend on OOF, persist OOF/test preds."""
    from .features import load_features
    from .pair_features import build_pair_features
    from .cv import make_folds, search_blend_weights
    from .models.bt_head import train_bt_cv
    from .models.gbdt import train_lgbm_cv

    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    print(f"[train] device={device}")

    y, meta_tr, meta_te = _load_meta(cfg)
    F1, F2, dims = load_features(cfg, "train")
    F1te, F2te, _ = load_features(cfg, "test")
    print(f"[train] features: train {F1.shape}, test {F1te.shape}, dims={dims}")

    folds = make_folds(y, cfg.n_folds, cfg.seed)

    oof_list: List[np.ndarray] = []
    test_list: List[np.ndarray] = []
    names: List[str] = []
    report: Dict = {}

    with timer("bradley-terry"):
        bt_oof, bt_test, bt_aucs = train_bt_cv(F1, F2, y, F1te, F2te, folds, cfg, device)
    oof_list.append(bt_oof); test_list.append(bt_test); names.append("bt")
    report["bt_oof_auc"] = roc_auc(y, bt_oof)

    if cfg.use_gbdt:
        Xtr = build_pair_features(F1, F2, meta_tr, mode=cfg.gbdt_pair_mode)
        Xte = build_pair_features(F1te, F2te, meta_te, mode=cfg.gbdt_pair_mode)
        with timer("lightgbm"):
            g_oof, g_test, g_aucs = train_lgbm_cv(Xtr, y, Xte, folds, cfg)
        oof_list.append(g_oof); test_list.append(g_test); names.append("gbdt")
        report["gbdt_oof_auc"] = roc_auc(y, g_oof)

    if cfg.finetune_enabled:
        from .data import load_column_bytes
        from .models.finetune import finetune_cv
        b1 = load_column_bytes(cfg.train_parquet, cfg.img1_col)
        b2 = load_column_bytes(cfg.train_parquet, cfg.img2_col)
        tb1 = load_column_bytes(cfg.test_parquet, cfg.img1_col)
        tb2 = load_column_bytes(cfg.test_parquet, cfg.img2_col)
        with timer("finetune"):
            f_oof, f_test, f_aucs = finetune_cv(b1, b2, y, tb1, tb2, folds, cfg, device)
        oof_list.append(f_oof); test_list.append(f_test); names.append("finetune")
        report["finetune_oof_auc"] = roc_auc(y, f_oof)

    # blend on the rank scale
    if cfg.blend_search and len(oof_list) > 1:
        weights, blend_auc = search_blend_weights(oof_list, y)
    else:
        weights = [1.0 / len(oof_list)] * len(oof_list)
        from .cv import rankavg
        blend_auc = roc_auc(y, rankavg(oof_list))
    report["blend_weights"] = {n: w for n, w in zip(names, weights)}
    report["blend_oof_auc"] = blend_auc
    print(f"[train] blend weights={report['blend_weights']}  OOF AUC={blend_auc:.5f}")

    # persist everything for run_predict
    np.savez(os.path.join(cfg.work_dir, "preds.npz"),
             oof=np.stack(oof_list), test=np.stack(test_list),
             y=y, names=np.array(names), weights=np.array(weights))
    with open(os.path.join(cfg.work_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[train] report: {json.dumps(report, indent=2)}")
    return report


def run_predict(cfg: Config, out_path: str = None) -> str:
    """Blend cached test predictions and write the submission."""
    from .cv import rankavg
    from .submit import write_submission

    data = np.load(os.path.join(cfg.work_dir, "preds.npz"), allow_pickle=True)
    test = data["test"]
    weights = data["weights"].astype(np.float64)
    ranks = [np.argsort(np.argsort(test[i])).astype(np.float64) / (test.shape[1] - 1)
             for i in range(test.shape[0])]
    blended = sum(w * r for w, r in zip(weights, ranks))
    out_path = out_path or os.path.join(cfg.work_dir, "submission.csv")
    return write_submission(blended, cfg.sample_submission, out_path, cfg.target_col)


def run_all(cfg: Config) -> str:
    run_features(cfg)
    run_train(cfg)
    return run_predict(cfg)
