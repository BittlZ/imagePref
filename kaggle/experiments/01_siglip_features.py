"""Kaggle experiment exported from notebooks/ivan_german_image_preference_solution.ipynb.

This script is kept for review/reuse; run it inside a Kaggle notebook with the competition data and code dataset attached.
"""

from pathlib import Path
import os
import sys
import zipfile
import shutil
import subprocess
import textwrap
import json

# 1. Locate code and restore previous work/cache
runner_path = next(Path("/kaggle/input").rglob("full_train.py"))
CODE_ROOT = runner_path.parents[1]
PACKAGE_ROOT = CODE_ROOT / "image_preference"
sys.path.insert(0, str(PACKAGE_ROOT))

print("CODE_ROOT:", CODE_ROOT)

work_dir = Path("/kaggle/working/work")
cache_dir = work_dir / "cache"
work_dir.mkdir(parents=True, exist_ok=True)
cache_dir.mkdir(parents=True, exist_ok=True)

work_zip_candidates = list(Path("/kaggle/input").rglob("imgpref_work_final.zip")) + list(Path("/kaggle/input").rglob("imgpref_work.zip"))
print("work zip candidates:", work_zip_candidates)

if work_zip_candidates:
    with zipfile.ZipFile(work_zip_candidates[0]) as zf:
        zf.extractall(work_dir)
    print("restored work from:", work_zip_candidates[0])
else:
    print("WARNING: no work zip found, using current /kaggle/working/work only")

print("cache after restore:")
for p in sorted(cache_dir.glob("*.npy")):
    print(" -", p.name, round(p.stat().st_size / 1024**2, 1), "MB")

# 2. Imports
try:
    import timm
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "timm>=1.0.7"])

try:
    import yaml
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pyyaml>=6.0"])

from imgpref.config import auto_config, BackboneSpec
from imgpref.features import cache_path
from imgpref.pipeline import run_train, run_predict

# 3. Config: old 2 backbones + new SigLIP
siglip_name = "vit_large_patch16_siglip_384.webli"

cfg = auto_config(config_path=str(PACKAGE_ROOT / "configs" / "default.yaml"))
cfg.work_dir = str(work_dir)
cfg.cache_dir = str(cache_dir)
cfg.finetune_enabled = False

# Keep old backbones exactly, append SigLIP.
cfg.backbones = list(cfg.backbones) + [
    BackboneSpec(siglip_name, img_size=384, batch_size=24),
]

cfg.decode_threads = 8
cfg.num_workers = 4
cfg.ensure_dirs()

print("backbones:")
for spec in cfg.backbones:
    print(" -", spec)

# 4. Separate-process feature extractor to hard-free RAM after each file
extract_one = Path("/kaggle/working/extract_one_feature_siglip.py")
extract_one.write_text(textwrap.dedent(r"""
    from __future__ import annotations

    import argparse
    import gc
    import os
    import sys
    from pathlib import Path

    import numpy as np

    CODE_ROOT = Path(os.environ["IMGPREF_CODE_ROOT"])
    PACKAGE_ROOT = CODE_ROOT / "image_preference"
    sys.path.insert(0, str(PACKAGE_ROOT))

    try:
        import timm
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "timm>=1.0.7"])

    from imgpref.config import auto_config, BackboneSpec
    from imgpref.features import build_model_and_transform, cache_path, _extract_stream

    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone-name", required=True)
    parser.add_argument("--img-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--col", choices=["image_1", "image_2"], required=True)
    args = parser.parse_args()

    cfg = auto_config(config_path=str(PACKAGE_ROOT / "configs" / "default.yaml"))
    cfg.work_dir = "/kaggle/working/work"
    cfg.cache_dir = "/kaggle/working/work/cache"
    cfg.finetune_enabled = False
    cfg.decode_threads = int(os.environ.get("IMGPREF_DECODE_THREADS", "8"))
    cfg.num_workers = int(os.environ.get("IMGPREF_NUM_WORKERS", "4"))
    cfg.ensure_dirs()

    spec = BackboneSpec(args.backbone_name, img_size=args.img_size, batch_size=args.batch_size)
    parquet = cfg.train_parquet if args.split == "train" else cfg.test_parquet
    out = cache_path(cfg, spec.name, args.split, args.col)

    if os.path.exists(out):
        print(f"[skip] cached {out}")
        raise SystemExit(0)

    print(f"[one] backbone={spec.name} split={args.split} col={args.col}")
    print(f"[one] batch_size={spec.batch_size} decode_threads={cfg.decode_threads}")

    import torch

    model, transform = build_model_and_transform(spec, cfg.pretrained, cfg.device)
    model.eval()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    feats = _extract_stream(
        model=model,
        parquet=parquet,
        col=args.col,
        transform=transform,
        batch_size=spec.batch_size,
        device=cfg.device,
        amp=cfg.amp,
        tta=cfg.use_tta_hflip,
        desc=f"{spec.name.split('.')[0]} {args.split}/{args.col}",
        num_decode_threads=cfg.decode_threads,
    )

    np.save(out, feats)
    print(f"[saved] {out} shape={feats.shape}")

    del feats
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
"""), encoding="utf-8")

env = os.environ.copy()
env["IMGPREF_CODE_ROOT"] = str(CODE_ROOT)
env["IMGPREF_DECODE_THREADS"] = "8"
env["IMGPREF_NUM_WORKERS"] = "4"

# 5. Extract only SigLIP missing files
siglip_spec = cfg.backbones[-1]
for split, col in [
    ("train", cfg.img1_col),
    ("train", cfg.img2_col),
    ("test", cfg.img1_col),
    ("test", cfg.img2_col),
]:
    out = Path(cache_path(cfg, siglip_spec.name, split, col))
    if out.exists():
        print("[skip existing]", out.name)
        continue

    print("\n=== extracting SigLIP target ===")
    print(siglip_spec.name, split, col)

    subprocess.run(
        [
            sys.executable,
            str(extract_one),
            "--backbone-name", siglip_spec.name,
            "--img-size", str(siglip_spec.img_size),
            "--batch-size", str(siglip_spec.batch_size),
            "--split", split,
            "--col", col,
        ],
        check=True,
        env=env,
    )

    # checkpoint after every completed feature file
    checkpoint = shutil.make_archive(
        "/kaggle/working/imgpref_siglip_checkpoint",
        "zip",
        root_dir=str(work_dir),
    )
    print("checkpoint:", checkpoint)

print("\nAll cache files:")
for p in sorted(cache_dir.glob("*.npy")):
    print(" -", p.name, round(p.stat().st_size / 1024**2, 1), "MB")

# 6. Train and predict with 3 backbones
report = run_train(cfg)
submission = run_predict(cfg, out_path="/kaggle/working/submission_siglip.csv")

with open("/kaggle/working/report_siglip.json", "w") as f:
    json.dump(report, f, indent=2)

archive = shutil.make_archive(
    "/kaggle/working/imgpref_siglip_work",
    "zip",
    root_dir=str(work_dir),
)

print("report:", report)
print("submission:", submission)
print("archive:", archive)