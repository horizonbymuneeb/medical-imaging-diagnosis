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
