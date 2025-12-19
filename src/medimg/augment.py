"""Label-safe augmentation for patch classification.

For a patch classifier the label is a property of the patch, so flips, 90-degree
rotations, gamma and intensity scaling are all safe.  Crops and arbitrary
rotations are *not*: they can push the lesion out of the patch and silently
flip the label.  Anything geometric that moves content is therefore applied to
``(image, mask)`` pairs together, never to an image alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class AugmentConfig:
    """Probabilities and ranges for the patch augmentation pipeline."""

    p_flip: float = 0.5
    p_rot90: float = 0.5
    p_gamma: float = 0.3
    gamma_range: tuple[float, float] = (0.8, 1.25)
    p_noise: float = 0.3
    noise_std: float = 0.02
    p_scale: float = 0.3
    scale_range: tuple[float, float] = (0.9, 1.1)
    p_translate: float = 0.0
    max_shift: int = 2

    def __post_init__(self) -> None:
        for name in ("p_flip", "p_rot90", "p_gamma", "p_noise", "p_scale", "p_translate"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.noise_std < 0:
            raise ValueError("noise_std must be non-negative")
        for name in ("gamma_range", "scale_range"):
            low, high = getattr(self, name)
            if low <= 0 or high <= 0 or low > high:
                raise ValueError(f"{name} must be positive and ordered, got {(low, high)}")
        if self.max_shift < 0:
            raise ValueError("max_shift must be non-negative")

    def to_dict(self) -> dict:
        return {
            "p_flip": self.p_flip,
            "p_rot90": self.p_rot90,
            "p_gamma": self.p_gamma,
            "gamma_range": list(self.gamma_range),
            "p_noise": self.p_noise,
            "noise_std": self.noise_std,
            "p_scale": self.p_scale,
            "scale_range": list(self.scale_range),
            "p_translate": self.p_translate,
            "max_shift": self.max_shift,
        }


DEFAULT_AUGMENT = AugmentConfig()


def _as_2d(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D patch, got shape {arr.shape}")
    return arr


def random_flip(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    p: float = 0.5,
    axis: int = 1,
) -> np.ndarray:
    """Flip horizontally (``axis=1``) or vertically (``axis=0``) with prob ``p``."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    arr = _as_2d(image)
    if rng.random() >= p:
        return arr.copy()
    return np.flip(arr, axis=axis).copy()


def random_rot90(image: np.ndarray, rng: np.random.Generator, *, p: float = 0.5) -> np.ndarray:
    """Rotate by a random multiple of 90 degrees (label preserving)."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    arr = _as_2d(image)
    if rng.random() >= p:
        return arr.copy()
    k = int(rng.integers(1, 4))
    return np.ascontiguousarray(np.rot90(arr, k))


def random_intensity_scale(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    p: float = 0.3,
    low: float = 0.9,
    high: float = 1.1,
) -> np.ndarray:
    """Multiply intensities by a factor drawn from ``[low, high]``."""
    if low <= 0 or high <= 0 or low > high:
        raise ValueError("scale bounds must be positive and ordered")
    arr = _as_2d(image)
    if rng.random() >= p:
        return arr.copy()
    factor = float(rng.uniform(low, high))
    return (arr * factor).astype(np.float32)


def random_gamma(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    p: float = 0.3,
    low: float = 0.8,
    high: float = 1.25,
) -> np.ndarray:
    """Gamma-adjust a patch that is already in ``[0, 1]``.

    The patch is clipped to ``[0, 1]`` first because ``x ** gamma`` on negative
    values produces NaN and would poison the whole batch.
    """
    if low <= 0 or high <= 0 or low > high:
        raise ValueError("gamma bounds must be positive and ordered")
    arr = _as_2d(image)
    if rng.random() >= p:
        return arr.copy()
    gamma = float(rng.uniform(low, high))
    clipped = np.clip(arr, 0.0, 1.0)
    return np.power(clipped, gamma).astype(np.float32)
