"""Patch extraction and balanced sampling for lesion classification.

A full CT volume is 30 million voxels; a classifier sees 32x32 patches.  The
sampling policy matters more than the network: a naive uniform sampler draws
almost exclusively healthy brain because lesions occupy well under 1 % of the
volume, and the model then predicts "no lesion" forever.  This module therefore
samples positives from lesion centroids and negatives from voxels a minimum
distance away, and can re-balance an existing patch list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class Patch:
    """A square image patch with a binary label and its source centre."""

    image: np.ndarray
    label: int
    center: tuple[int, int]
    source_index: int = 0

    def __post_init__(self) -> None:
        image = np.asarray(self.image, dtype=np.float32)
        if image.ndim != 2:
            raise ValueError(f"patch image must be 2-D, got shape {image.shape}")
        if image.shape[0] != image.shape[1]:
            raise ValueError(f"patch must be square, got {image.shape}")
        if self.label not in (0, 1):
            raise ValueError("label must be 0 or 1")
        object.__setattr__(self, "image", image)

    @property
    def size(self) -> int:
        return int(self.image.shape[0])

    @property
    def mean(self) -> float:
        return float(self.image.mean())


def extract_patch(
    image: np.ndarray,
    center: Sequence[int],
    size: int,
    *,
    pad_value: float = 0.0,
) -> np.ndarray:
    """Cut a square patch around ``center``, zero-padding at the border.

    Patches are always exactly ``(size, size)`` so a batch can be stacked
    without a second pass over the list.
    """
    if size <= 0 or size % 2:
        raise ValueError("size must be a positive even integer")
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("image must be 2-D")
    if len(center) != 2:
        raise ValueError("center must be (row, col)")
    half = size // 2
    row, col = int(center[0]), int(center[1])
    out = np.full((size, size), pad_value, dtype=np.float32)
    r0, r1 = row - half, row + half
    c0, c1 = col - half, col + half
    src_r0, src_c0 = max(0, r0), max(0, c0)
    src_r1, src_c1 = min(arr.shape[0], r1), min(arr.shape[1], c1)
    if src_r1 <= src_r0 or src_c1 <= src_c0:
        return out
    out[src_r0 - r0 : src_r1 - r0, src_c0 - c0 : src_c1 - c0] = arr[src_r0:src_r1, src_c0:src_c1]
    return out


def lesion_components(mask: np.ndarray) -> list[dict]:
    """Connected lesion blobs with centroid, bounding box and voxel count."""
    arr = np.asarray(mask).astype(bool)
    if arr.ndim != 2:
        raise ValueError("mask must be 2-D")
    labelled, count = ndimage.label(arr, structure=np.ones((3, 3), dtype=int))
    blobs = []
    for label_id in range(1, count + 1):
        rows, cols = np.nonzero(labelled == label_id)
        blobs.append(
            {
                "label": label_id,
                "centroid": (float(rows.mean()), float(cols.mean())),
                "bbox": (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())),
                "voxels": int(rows.size),
            }
        )
    return blobs
