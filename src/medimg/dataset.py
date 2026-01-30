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


def build_specs(
    n: int,
    *,
    seed: int = 0,
    positive_fraction: float = 0.5,
) -> list[SyntheticSpec]:
    """Create ``n`` specs whose seeds are derived from ``seed`` deterministically."""
    if n <= 0:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31 - 1, size=n)
    return [
        SyntheticSpec(index=i, seed=int(s), positive_fraction=positive_fraction)
        for i, s in enumerate(seeds)
    ]


def slice_for_spec(
    spec: SyntheticSpec,
    cfg: PhantomConfig,
    *,
    window: str = "brain",
    normalize: str = "minmax",
) -> tuple[np.ndarray, np.ndarray]:
    """Windowed lesion slice and its mask for one spec.

    The slice is taken through the lesion centre, so every generated image has
    a visible lesion; the negative class comes from the sampler, not from
    lesion-free slices.
    """
    phantom_cfg = PhantomConfig(
        shape=cfg.shape,
        spacing=cfg.spacing,
        lesion_radius_mm=cfg.lesion_radius_mm,
        noise_sigma=cfg.noise_sigma,
        seed=spec.seed,
        head_radius_fraction=cfg.head_radius_fraction,
        lesion_intensity=cfg.lesion_intensity,
        noise_model=cfg.noise_model,
    )
    data, lesion_mask = brain_phantom(phantom_cfg)
    index = lesion_slice_index(lesion_mask, axis=spec.axis)
    image = np.take(data, index, axis=spec.axis)
    mask = np.take(lesion_mask, index, axis=spec.axis)
    windowed = apply_preset(image, window)
    normalised = normalize_array(windowed, normalize)
    return normalised.astype(np.float32), mask.astype(bool)
