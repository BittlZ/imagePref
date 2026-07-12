"""Kaggle experiment exported from notebooks/ivan_german_image_preference_solution.ipynb.

This script is kept for review/reuse; run it inside a Kaggle notebook with the competition data and code dataset attached.
"""

from pathlib import Path
import sys, os, zipfile, shutil, subprocess, gc, json

runner = next(Path("/kaggle/input").rglob("full_train.py"))
CODE_ROOT = runner.parents[1]
PKG_ROOT = CODE_ROOT / "image_preference"
sys.path.insert(0, str(PKG_ROOT))

WORK_DIR = Path("/kaggle/working/work")
CACHE_DIR = WORK_DIR / "cache"
WORK_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

print("CODE_ROOT:", CODE_ROOT)

# Restore cache only if current working cache is incomplete.
if len(list(CACHE_DIR.glob("feat__*.npy"))) < 12:
    names = [
        "imgpref_dinov2_work.zip",
        "imgpref_dinov2_checkpoint.zip",
        "imgpref_siglip_work.zip",
        "imgpref_siglip_checkpoint.zip",
        "imgpref_work_final.zip",
        "imgpref_work.zip",
    ]
    zips = []
    for name in names:
        zips += list(Path("/kaggle/input").rglob(name))
    if zips:
        print("restoring:", zips[0])
        with zipfile.ZipFile(zips[0]) as zf:
            zf.extractall(WORK_DIR)
    else:
        print("no work zip found; using current cache only")

from imgpref.config import auto_config, BackboneSpec
from imgpref.features import cache_path
from imgpref.pipeline import run_train, run_predict

SIGLIP = "vit_large_patch16_siglip_384.webli"
DINO = "vit_large_patch14_reg4_dinov2.lvd142m"
DINO_IMG_SIZE = 392
DINO_BATCH = 16

cfg = auto_config(config_path=str(PKG_ROOT / "configs" / "default.yaml"))
cfg.work_dir = str(WORK_DIR)
cfg.cache_dir = str(CACHE_DIR)
cfg.finetune_enabled = False
cfg.decode_threads = 8
cfg.num_workers = 2

if SIGLIP not in [s.name for s in cfg.backbones]:
    cfg.backbones.append(BackboneSpec(SIGLIP, img_size=384, batch_size=24))
if DINO not in [s.name for s in cfg.backbones]:
    cfg.backbones.append(BackboneSpec(DINO, img_size=DINO_IMG_SIZE, batch_size=DINO_BATCH))
cfg.ensure_dirs()

helper = Path("/kaggle/working/extract_one_dino.py")
helper.write_text(r'''
from pathlib import Path
import argparse, sys, gc, numpy as np

p = argparse.ArgumentParser()
p.add_argument("--code-root", required=True)
p.add_argument("--name", required=True)
p.add_argument("--img-size", type=int, required=True)
p.add_argument("--batch-size", type=int, required=True)
p.add_argument("--split", required=True)
p.add_argument("--col", required=True)
args = p.parse_args()

code_root = Path(args.code_root)
sys.path.insert(0, str(code_root / "image_preference"))

from imgpref.config import auto_config, BackboneSpec
from imgpref.features import cache_path, build_model_and_transform, _extract_stream

cfg = auto_config(config_path=str(code_root / "image_preference" / "configs" / "default.yaml"))
cfg.work_dir = "/kaggle/working/work"
cfg.cache_dir = "/kaggle/working/work/cache"
cfg.decode_threads = 8
cfg.ensure_dirs()

spec = BackboneSpec(args.name, img_size=args.img_size, batch_size=args.batch_size)
out = Path(cache_path(cfg, spec.name, args.split, args.col))
if out.exists():
    print("[skip existing]", out.name)
    raise SystemExit(0)

parquet = cfg.train_parquet if args.split == "train" else cfg.test_parquet
print(f"[one] backbone={spec.name} split={args.split} col={args.col} img={args.img_size} batch={args.batch_size}")

model, transform = build_model_and_transform(spec, cfg.pretrained, cfg.device)

import torch
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

feats = _extract_stream(
    model, parquet, args.col, transform, spec.batch_size,
    cfg.device, cfg.amp, cfg.use_tta_hflip,
    desc=f"{spec.name.split('.')[0]} {args.split}/{args.col}",
    num_decode_threads=cfg.decode_threads,
)
np.save(out, feats)
print("[saved]", out, "shape=", feats.shape)

del feats, model
gc.collect()
torch.cuda.empty_cache()
''')

for split, col in [("train", "image_1"), ("train", "image_2"), ("test", "image_1"), ("test", "image_2")]:
    out = Path(cache_path(cfg, DINO, split, col))
    if out.exists():
        print("[skip existing]", out.name)
        continue

    cmd = [
        sys.executable, str(helper),
        "--code-root", str(CODE_ROOT),
        "--name", DINO,
        "--img-size", str(DINO_IMG_SIZE),
        "--batch-size", str(DINO_BATCH),
        "--split", split,
        "--col", col,
    ]
    subprocess.run(cmd, check=True)
    gc.collect()

    ckpt = shutil.make_archive("/kaggle/working/imgpref_dinov2_checkpoint", "zip", root_dir=WORK_DIR)
    print("checkpoint:", ckpt)

print("All cache files:")
for p in sorted(CACHE_DIR.glob("feat__*.npy")):
    print(" -", p.name, round(p.stat().st_size / 1024 / 1024, 1), "MB")

report = run_train(cfg)
submission = run_predict(cfg, out_path="/kaggle/working/submission_dinov2.csv")

Path("/kaggle/working/report_dinov2.json").write_text(json.dumps(report, indent=2))
archive = shutil.make_archive("/kaggle/working/imgpref_dinov2_work", "zip", root_dir=WORK_DIR)

print("report:", report)
print("submission:", submission)
print("archive:", archive)