"""Central configuration.

A single :class:`Config` dataclass drives the whole pipeline. ``auto_config``
detects whether we run on Kaggle and fills sensible paths.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class BackboneSpec:
    """One frozen feature extractor.

    ``name`` is a ``timm`` model id. ``img_size`` overrides the model's native
    input size when set (some quality cues live in fine detail, so a larger
    input can help). ``weight_path`` lets us load offline weights when Kaggle
    has no internet.
    """
    name: str
    img_size: Optional[int] = None
    weight_path: Optional[str] = None
    batch_size: int = 64


@dataclass
class Config:
    # --- paths ---
    train_parquet: str = "data/train.parquet"
    test_parquet: str = "data/test.parquet"
    sample_submission: str = "data/sample_submission.csv"
    work_dir: str = "work"          # everything we write goes here
    cache_dir: str = "work/cache"   # cached features live here

    # --- data / target ---
    target_col: str = "is_image1_better"
    img1_col: str = "image_1"
    img2_col: str = "image_2"

    # --- feature extraction ---
    backbones: List[BackboneSpec] = field(default_factory=lambda: [
        BackboneSpec("convnextv2_base.fcmae_ft_in22k_in1k_384", img_size=384, batch_size=48),
        BackboneSpec("vit_large_patch14_clip_224.openai", img_size=224, batch_size=64),
    ])
    use_tta_hflip: bool = True       # average features of image and its h-flip
    num_workers: int = 4
    decode_threads: int = 8          # parallel image-decode threads during extraction
    pretrained: bool = True

    # --- Bradley-Terry head ---
    bt_hidden: List[int] = field(default_factory=lambda: [1024, 256])
    bt_dropout: float = 0.3
    bt_lr: float = 1e-3
    bt_weight_decay: float = 1e-2
    bt_epochs: int = 40
    bt_batch_size: int = 256
    bt_patience: int = 6
    bt_swap_aug: bool = True         # randomly swap image_1/image_2 (flip label)

    # --- LightGBM head ---
    use_gbdt: bool = True
    gbdt_pair_mode: str = "compact"  # "compact" -> [diff, prod, cos, meta]; "full" adds f1,f2
    lgbm_params: dict = field(default_factory=lambda: {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "feature_fraction": 0.5,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_child_samples": 40,
        "lambda_l2": 5.0,
        "verbosity": -1,
        "n_estimators": 2000,
    })
    gbdt_early_stopping: int = 100

    # --- cross-validation / ensemble ---
    n_folds: int = 5
    seed: int = 42
    blend_search: bool = True        # grid-search blend weights on OOF

    # --- runtime ---
    device: str = "cuda"
    amp: bool = True

    # --- end-to-end finetune (optional, high-ceiling) ---
    finetune_enabled: bool = False
    finetune_backbone: str = "convnextv2_base.fcmae_ft_in22k_in1k_384"
    finetune_img_size: int = 384
    finetune_epochs: int = 6
    finetune_lr: float = 1e-5
    finetune_head_lr: float = 1e-3
    finetune_batch_size: int = 16
    finetune_freeze_epochs: int = 1

    def ensure_dirs(self) -> None:
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def _find_competition_dir() -> Optional[str]:
    """Locate the Kaggle input dir that holds the parquet files.

    Searches recursively because Kaggle may mount inputs several levels deep
    (e.g. ``/kaggle/input/datasets/<user>/<slug>/...`` or
    ``/kaggle/input/competitions/<comp>/...``).
    """
    import glob
    base = "/kaggle/input"
    if not os.path.isdir(base):
        return None
    hits = glob.glob(os.path.join(base, "**", "train.parquet"), recursive=True)
    return os.path.dirname(hits[0]) if hits else None


def _read_config_file(path: str) -> dict:
    """Read a YAML (preferred) or JSON config file into a plain dict."""
    with open(path, "r") as f:
        text = f.read()
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml
        except Exception as e:  # pragma: no cover
            raise RuntimeError("pyyaml is required to read YAML configs") from e
        return yaml.safe_load(text) or {}
    import json
    return json.loads(text or "{}")


def apply_overrides(cfg: Config, d: dict) -> Config:
    """Apply a dict of overrides onto a Config (``backbones`` handled specially)."""
    for k, v in d.items():
        if k == "backbones" and v is not None:
            cfg.backbones = [
                spec if isinstance(spec, BackboneSpec) else BackboneSpec(**spec)
                for spec in v
            ]
            continue
        if not hasattr(cfg, k):
            raise AttributeError(f"Unknown config field: {k}")
        setattr(cfg, k, v)
    return cfg


_DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               "configs", "default.yaml")


def auto_config(config_path: Optional[str] = _DEFAULT_CONFIG, **overrides) -> Config:
    """Build a :class:`Config` from (defaults -> Kaggle autodetect -> file -> kwargs).

    On Kaggle: inputs are read-only under ``/kaggle/input/...`` and outputs go to
    ``/kaggle/working``. A config file (YAML/JSON) supplies hyperparameters; any
    path it sets wins over autodetection. ``overrides`` (kwargs) win over all.
    Pass ``config_path=None`` to skip the file.
    """
    cfg = Config()

    comp = _find_competition_dir()
    if comp is not None:
        cfg.train_parquet = os.path.join(comp, "train.parquet")
        cfg.test_parquet = os.path.join(comp, "test.parquet")
        cfg.sample_submission = os.path.join(comp, "sample_submission.csv")
        cfg.work_dir = "/kaggle/working/work"
        cfg.cache_dir = "/kaggle/working/work/cache"

    if config_path and os.path.isfile(config_path):
        apply_overrides(cfg, _read_config_file(config_path))

    apply_overrides(cfg, overrides)
    cfg.ensure_dirs()
    return cfg
