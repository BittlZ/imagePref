"""Data access for the parquet files.

The competition stores raw image bytes (JPEG/PNG/WEBP) in two binary columns
plus an int label. Files are large and single-row-group, so we read column
bytes once into memory (a few GB; Kaggle GPU nodes have ~29 GB RAM) and decode
lazily inside the Dataset for random access during feature extraction.
"""
from __future__ import annotations

import io
from typing import List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


def read_label_and_meta(parquet_path: str, target_col: str = "is_image1_better",
                        img1_col: str = "image_1", img2_col: str = "image_2",
                        has_target: bool = True):
    """Stream the parquet once, returning labels (or None) and metadata.

    Metadata = per-row byte sizes and the container format of each slot, which
    we feed to the GBDT head as a cheap positional/codec prior.
    """
    cols = [img1_col, img2_col] + ([target_col] if has_target else [])
    pf = pq.ParquetFile(parquet_path)
    labels: List[int] = []
    len1: List[int] = []
    len2: List[int] = []
    fmt1: List[str] = []
    fmt2: List[str] = []
    for batch in pf.iter_batches(batch_size=256, columns=cols):
        d = batch.to_pydict()
        b1s, b2s = d[img1_col], d[img2_col]
        ys = d[target_col] if has_target else [None] * len(b1s)
        for b1, b2, y in zip(b1s, b2s, ys):
            len1.append(len(b1)); len2.append(len(b2))
            fmt1.append(Image.open(io.BytesIO(b1)).format or "UNK")
            fmt2.append(Image.open(io.BytesIO(b2)).format or "UNK")
            if has_target:
                labels.append(int(y))
    meta = {
        "len1": np.asarray(len1, dtype=np.float32),
        "len2": np.asarray(len2, dtype=np.float32),
        "fmt1": np.asarray(fmt1),
        "fmt2": np.asarray(fmt2),
    }
    y = np.asarray(labels, dtype=np.int64) if has_target else None
    return y, meta


def load_column_bytes(parquet_path: str, column: str) -> List[bytes]:
    """Load one binary column fully into a Python list (random-access ready)."""
    pf = pq.ParquetFile(parquet_path)
    out: List[bytes] = []
    for batch in pf.iter_batches(batch_size=256, columns=[column]):
        out.extend(batch.to_pydict()[column])
    return out


def num_rows(parquet_path: str) -> int:
    return pq.ParquetFile(parquet_path).metadata.num_rows


def _decode(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")


try:
    import torch
    from torch.utils.data import Dataset

    class BytesImageDataset(Dataset):
        """Decode image bytes on the fly and apply a torchvision-style transform.

        Optionally returns the horizontally flipped image too, so feature
        extraction can do hflip TTA in a single pass.
        """

        def __init__(self, byte_list: List[bytes], transform, tta_hflip: bool = False):
            self.byte_list = byte_list
            self.transform = transform
            self.tta_hflip = tta_hflip

        def __len__(self) -> int:
            return len(self.byte_list)

        def __getitem__(self, idx: int):
            img = _decode(self.byte_list[idx])
            x = self.transform(img)
            if self.tta_hflip:
                xf = self.transform(img.transpose(Image.FLIP_LEFT_RIGHT))
                return x, xf
            return x

    class PairImageDataset(Dataset):
        """For end-to-end finetuning: returns (img1, img2, label).

        ``swap_aug`` randomly swaps the two images and flips the label, which
        enforces the antisymmetry of a Bradley-Terry model.
        """

        def __init__(self, bytes1: List[bytes], bytes2: List[bytes], transform,
                     labels: Optional[np.ndarray] = None, swap_aug: bool = False):
            assert len(bytes1) == len(bytes2)
            self.b1 = bytes1
            self.b2 = bytes2
            self.transform = transform
            self.labels = labels
            self.swap_aug = swap_aug

        def __len__(self) -> int:
            return len(self.b1)

        def __getitem__(self, idx: int):
            x1 = self.transform(_decode(self.b1[idx]))
            x2 = self.transform(_decode(self.b2[idx]))
            y = -1 if self.labels is None else int(self.labels[idx])
            if self.swap_aug and self.labels is not None and torch.rand(1).item() < 0.5:
                x1, x2 = x2, x1
                y = 1 - y
            return x1, x2, y

except Exception:  # torch not available (e.g. local syntax check)
    BytesImageDataset = None  # type: ignore
    PairImageDataset = None   # type: ignore
