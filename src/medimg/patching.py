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
