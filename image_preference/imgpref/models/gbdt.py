"""LightGBM head over engineered pairwise features."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..utils import roc_auc


def train_lgbm_cv(X, y, X_te, folds, cfg) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """Returns (oof_proba, test_proba, per_fold_auc)."""
    import lightgbm as lgb

    oof = np.zeros(len(y), dtype=np.float32)
    test = np.zeros(len(X_te), dtype=np.float32)
    aucs: List[float] = []
    params = dict(cfg.lgbm_params)
    n_estimators = params.pop("n_estimators", 2000)

    for fold, (tr, va) in enumerate(folds):
        dtr = lgb.Dataset(X[tr], label=y[tr])
        dva = lgb.Dataset(X[va], label=y[va])
        model = lgb.train(
            params,
            dtr,
            num_boost_round=n_estimators,
            valid_sets=[dva],
            callbacks=[
                lgb.early_stopping(cfg.gbdt_early_stopping, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        oof[va] = model.predict(X[va], num_iteration=model.best_iteration)
        test += model.predict(X_te, num_iteration=model.best_iteration) / len(folds)
        a = roc_auc(y[va], oof[va])
        aucs.append(a)
        print(f"  [GBDT] fold {fold}: AUC={a:.5f} (best_iter={model.best_iteration})")
    print(f"  [GBDT] OOF AUC={roc_auc(y, oof):.5f}")
    return oof, test, aucs
