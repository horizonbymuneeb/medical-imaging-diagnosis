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


def lesion_centers(mask: np.ndarray, min_voxels: int = 1) -> list[tuple[int, int]]:
    """Integer centroids of every lesion blob larger than ``min_voxels``."""
    centers = []
    for blob in lesion_components(mask):
        if blob["voxels"] < min_voxels:
            continue
        row, col = blob["centroid"]
        centers.append((int(round(row)), int(round(col))))
    return centers


def positive_centers(
    mask: np.ndarray,
    rng: np.random.Generator,
    n: int,
    *,
    jitter: int = 0,
) -> list[tuple[int, int]]:
    """Sample ``n`` positive centres from lesion voxels.

    Centres are drawn from the voxel list rather than the blob centroid so a
    large or elongated lesion produces patches from its whole extent.  ``jitter``
    adds uniform noise around each chosen voxel.
    """
    rows, cols = np.nonzero(np.asarray(mask).astype(bool))
    if rows.size == 0:
        raise ValueError("mask has no positive voxels")
    picks = rng.integers(0, rows.size, size=n)
    centers = []
    for pick in picks:
        row, col = int(rows[pick]), int(cols[pick])
        if jitter:
            row += int(rng.integers(-jitter, jitter + 1))
            col += int(rng.integers(-jitter, jitter + 1))
        centers.append((row, col))
    return centers


def negative_centers(
    mask: np.ndarray,
    rng: np.random.Generator,
    n: int,
    *,
    patch_size: int,
    min_distance: int | None = None,
    max_tries: int = 2000,
) -> list[tuple[int, int]]:
    """Sample centres at least ``min_distance`` voxels away from any lesion.

    The default distance is one patch size, which guarantees that a "negative"
    patch cannot contain lesion pixels at its centre.  Sampling is rejection
    based with a bounded number of attempts so a lesion covering the whole
    slice raises instead of looping forever.
    """
    arr = np.asarray(mask).astype(bool)
    if arr.ndim != 2:
        raise ValueError("mask must be 2-D")
    distance = patch_size if min_distance is None else int(min_distance)
    if distance < 0:
        raise ValueError("min_distance must be non-negative")
    height, width = arr.shape
    rows, cols = np.nonzero(arr)
    if rows.size:
        lesion_points = np.stack([rows, cols], axis=1).astype(np.float32)
    else:
        lesion_points = np.empty((0, 2), dtype=np.float32)
    centers: list[tuple[int, int]] = []
    attempts = 0
    while len(centers) < n and attempts < max_tries:
        attempts += 1
        row = int(rng.integers(0, height))
        col = int(rng.integers(0, width))
        if lesion_points.size:
            d = np.sqrt(((lesion_points - np.array([row, col], dtype=np.float32)) ** 2).sum(axis=1))
            if float(d.min()) < distance:
                continue
        centers.append((row, col))
    if len(centers) < n:
        raise RuntimeError(
            f"could only find {len(centers)} negative centres after {max_tries} tries; "
            "the lesion covers too much of the slice"
        )
    return centers


class PatchSampler:
    """Draws balanced patches from one windowed slice.

    ``positive_fraction`` is the target share of positives; ``0.5`` gives the
    classifier a balanced view while ``0.1`` mimics the true prevalence and is
    used when calibrating a prevalence-sensitive threshold.
    """

    def __init__(
        self,
        image: np.ndarray,
        lesion_mask: np.ndarray,
        patch_size: int = 32,
        *,
        seed: int = 0,
        positive_fraction: float = 0.5,
    ) -> None:
        arr = np.asarray(image, dtype=np.float32)
        mask = np.asarray(lesion_mask).astype(bool)
        if arr.ndim != 2 or mask.shape != arr.shape:
            raise ValueError("image and lesion_mask must be 2-D and the same shape")
        if not 0.0 <= positive_fraction <= 1.0:
            raise ValueError("positive_fraction must be in [0, 1]")
        self.image = arr
        self.mask = mask
        self.patch_size = int(patch_size)
        self.positive_fraction = float(positive_fraction)
        self.rng = np.random.default_rng(seed)

    @property
    def has_lesion(self) -> bool:
        return bool(self.mask.any())

    def sample_positive(self, n: int) -> list[Patch]:
        if not self.has_lesion:
            raise ValueError("slice has no lesion voxels")
        centers = positive_centers(self.mask, self.rng, n, jitter=self.patch_size // 8)
        return [
            Patch(extract_patch(self.image, c, self.patch_size), 1, c)
            for c in centers
        ]

    def sample_negative(self, n: int) -> list[Patch]:
        centers = negative_centers(
            self.mask, self.rng, n, patch_size=self.patch_size, min_distance=self.patch_size
        )
        return [
            Patch(extract_patch(self.image, c, self.patch_size), 0, c)
            for c in centers
        ]

    def sample_balanced(self, n: int) -> list[Patch]:
        """``n`` patches split according to ``positive_fraction``."""
        if n <= 0:
            raise ValueError("n must be positive")
        n_pos = int(round(n * self.positive_fraction))
        n_pos = min(n_pos, n) if self.has_lesion else 0
        n_neg = n - n_pos
        patches = self.sample_negative(n_neg)
        if n_pos:
            patches.extend(self.sample_positive(n_pos))
        order = self.rng.permutation(len(patches))
        return [patches[i] for i in order]

    def sample(self, n: int) -> list[Patch]:
        return self.sample_balanced(n)


def balance_patches(
    patches: Sequence[Patch],
    *,
    seed: int = 0,
    ratio: float = 1.0,
) -> list[Patch]:
    """Re-sample (with replacement) so positives are ``ratio`` of the negatives."""
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    positives = [p for p in patches if p.label == 1]
    negatives = [p for p in patches if p.label == 0]
    if not positives or not negatives:
        return list(patches)
    rng = np.random.default_rng(seed)
    target_pos = int(round(len(negatives) * ratio))
    chosen = list(negatives)
    for _ in range(target_pos):
        chosen.append(positives[int(rng.integers(0, len(positives)))])
    order = rng.permutation(len(chosen))
    return [chosen[i] for i in order]


def split_patches(
    patches: Sequence[Patch],
    val_fraction: float = 0.25,
    *,
    seed: int = 0,
) -> tuple[list[Patch], list[Patch]]:
    """Deterministic train/validation split that keeps both classes present."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(patches))
    n_val = max(1, int(round(len(patches) * val_fraction)))
    val_idx = set(int(i) for i in indices[:n_val])
    train = [p for i, p in enumerate(patches) if i not in val_idx]
    val = [p for i, p in enumerate(patches) if i in val_idx]
    return train, val


def patch_array(patches: Iterable[Patch]) -> tuple[np.ndarray, np.ndarray]:
    """Stack patches into ``(N, 1, H, W)`` images and ``(N,)`` labels."""
    images = [p.image[None, :, :] for p in patches]
    if not images:
        raise ValueError("no patches to stack")
    labels = np.asarray([p.label for p in patches], dtype=np.float32)
    return np.stack(images, axis=0), labels


def class_counts(patches: Iterable[Patch]) -> dict[int, int]:
    counts = {0: 0, 1: 0}
    for patch in patches:
        counts[patch.label] += 1
    return counts
