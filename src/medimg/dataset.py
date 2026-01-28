"""Torch datasets that build phantom patches on demand.

The dataset is generated, not stored: a spec (an integer seed) deterministically
produces a volume, a lesion slice and a set of patches.  That keeps the repo
free of binary fixtures while still giving the training loop real, varying data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.medimg.augment import AugmentConfig, DEFAULT_AUGMENT, augment_image
from src.medimg.patching import Patch, PatchSampler, patch_array
from src.medimg.phantom import PhantomConfig, brain_phantom, lesion_slice_index
from src.medimg.windowing import apply_preset, normalize_array


@dataclass(frozen=True)
class SyntheticSpec:
    """One dataset item: which phantom to build and how to sample from it."""

    index: int
    seed: int
    positive_fraction: float = 0.5
    axis: int = 2

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("index must be non-negative")
        if not 0.0 <= self.positive_fraction <= 1.0:
            raise ValueError("positive_fraction must be in [0, 1]")
        if self.axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1 or 2")
