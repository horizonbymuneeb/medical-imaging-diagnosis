"""Hounsfield-unit windowing and volume normalisation.

CT display is always a window: a centre (level) and a width.  A brain window is
narrow (level 40, width 80) because the brain only spans ~80 HU, while a lung
window is wide and negative (level -600, width 1500).  Feeding raw HU to a CNN
trains on the wrong dynamic range, so windowing is a real preprocessing step
rather than cosmetics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.medimg.volume import Volume


@dataclass(frozen=True)
class WindowPreset:
    """A named (level, width) pair with the tissue it is meant for."""

    name: str
    level: float
    width: float
    description: str

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError(f"window width must be positive, got {self.width}")
        if not self.name:
            raise ValueError("preset needs a name")

    @property
    def low(self) -> float:
        return self.level - self.width / 2.0

    @property
    def high(self) -> float:
        return self.level + self.width / 2.0

    def contains(self, hu: float) -> bool:
        return self.low <= float(hu) <= self.high


WINDOW_PRESETS: dict[str, WindowPreset] = {
    "brain": WindowPreset("brain", level=40.0, width=80.0, description="brain parenchyma"),
    "lung": WindowPreset("lung", level=-600.0, width=1500.0, description="lung fields"),
    "abdomen": WindowPreset("abdomen", level=50.0, width=350.0, description="soft tissue"),
    "bone": WindowPreset("bone", level=400.0, width=1800.0, description="bone detail"),
}


def get_preset(name: str) -> WindowPreset:
    """Look up a preset, with a helpful error listing the valid names."""
    try:
        return WINDOW_PRESETS[name]
    except KeyError as exc:
        raise KeyError(f"unknown window preset {name!r}; choose from {sorted(WINDOW_PRESETS)}") from exc


def apply_window(
    image: np.ndarray,
    level: float,
    width: float,
    *,
    clip: bool = True,
    dtype: type = np.float32,
) -> np.ndarray:
    """Map HU values to ``[0, 1]`` through the window ``(level, width)``.

    Values below ``level - width/2`` become 0 and above ``level + width/2``
    become 1.  ``clip=False`` keeps the out-of-range values instead of
    clamping, which is occasionally useful to inspect what the window hides.
    """
    if width <= 0:
        raise ValueError("width must be positive")
    low = level - width / 2.0
    out = (np.asarray(image, dtype=np.float32) - low) / float(width)
    if clip:
        out = np.clip(out, 0.0, 1.0)
    return out.astype(dtype, copy=False)


def apply_preset(image: np.ndarray, name: str, **kwargs) -> np.ndarray:
    preset = get_preset(name)
    return apply_window(image, preset.level, preset.width, **kwargs)


def window_batch(images: Iterable[np.ndarray], level: float, width: float) -> np.ndarray:
    """Window a sequence of 2-D images into one ``(N, H, W)`` array."""
    stacked = [apply_window(img, level, width) for img in images]
    if not stacked:
        raise ValueError("window_batch needs at least one image")
    shapes = {img.shape for img in stacked}
    if len(shapes) != 1:
        raise ValueError(f"all images must share a shape, got {sorted(shapes)}")
    return np.stack(stacked, axis=0)


def minmax_normalize(image: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    """Scale to ``[0, 1]`` using the observed min and max."""
    arr = np.asarray(image, dtype=np.float32)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < eps:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def zscore_normalize(image: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    """Standardise to zero mean and unit variance."""
    arr = np.asarray(image, dtype=np.float32)
    mean = float(arr.mean())
    std = float(arr.std())
    if std < eps:
        return np.zeros_like(arr)
    return (arr - mean) / std


def percentile_normalize(
    image: np.ndarray,
    low: float = 1.0,
    high: float = 99.0,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """Robust normalisation that ignores the brightest/darkest outliers.

    A single metal-artefact voxel at 3000 HU would otherwise compress the whole
    brain into the bottom 2 % of the range.
    """
    if not 0.0 <= low < high <= 100.0:
        raise ValueError("require 0 <= low < high <= 100")
    arr = np.asarray(image, dtype=np.float32)
    lo = float(np.percentile(arr, low))
    hi = float(np.percentile(arr, high))
    if hi - lo < eps:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
