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
