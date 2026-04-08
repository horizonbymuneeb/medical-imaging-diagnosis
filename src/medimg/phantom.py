"""Synthetic CT phantoms.

Everything downstream (windowing, patching, training, calibration) needs
volumes that behave like real CT series: Hounsfield-unit intensities, a skull
that windows to white, brain parenchyma around 35-45 HU, CSF near 0 HU and a
hyperdense lesion that a classifier can actually learn.  Real public datasets
are hundreds of gigabytes, so this module builds deterministic phantoms from a
seed instead.  The generator is intentionally geometric: a human inspecting a
slice should be able to see the head outline, the skull ring, the ventricles
and the lesion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from src.medimg.volume import Volume

# Hounsfield-unit anchors used by the generator and by the tests.
HU_AIR = -1000.0
HU_FAT = -90.0
HU_CSF = 6.0
HU_WHITE_MATTER = 32.0
HU_GRAY_MATTER = 44.0
HU_SOFT_TISSUE = 40.0
HU_BONE = 900.0
HU_LESION = 78.0

DEFAULT_SHAPE: tuple[int, int, int] = (64, 64, 48)
DEFAULT_SPACING: tuple[float, float, float] = (1.5, 1.5, 2.0)


@dataclass(frozen=True)
class PhantomConfig:
    """Geometry and intensity settings for one synthetic head phantom.

    ``shape`` is (rows, cols, slices) and ``spacing`` is millimetres per voxel
    along (x, y, z).  ``lesion_radius_mm`` is converted to voxels using the
    in-plane spacing so the lesion stays the same physical size when the grid
    resolution changes.
    """

    shape: tuple[int, int, int] = DEFAULT_SHAPE
    spacing: tuple[float, float, float] = DEFAULT_SPACING
    lesion_radius_mm: float = 6.0
    noise_sigma: float = 18.0
    seed: int = 7
    head_radius_fraction: float = 0.42
    lesion_intensity: float = HU_LESION
    noise_model: str = "gaussian"  # gaussian | rician

    def __post_init__(self) -> None:
        if len(self.shape) != 3 or any(int(s) <= 0 for s in self.shape):
            raise ValueError(f"shape must be three positive ints, got {self.shape!r}")
        if len(self.spacing) != 3 or any(float(s) <= 0 for s in self.spacing):
            raise ValueError(f"spacing must be three positive floats, got {self.spacing!r}")
        if self.noise_sigma < 0:
            raise ValueError("noise_sigma must be non-negative")
        if self.noise_model not in {"gaussian", "rician"}:
            raise ValueError(f"unknown noise model {self.noise_model!r}")

    @property
    def lesion_radius_voxels(self) -> tuple[float, float, float]:
        """Lesion radius expressed in voxels for each axis."""
        r = float(self.lesion_radius_mm)
        return (r / self.spacing[0], r / self.spacing[1], r / self.spacing[2])

    @property
    def voxel_volume_mm3(self) -> float:
        return float(np.prod(self.spacing))

    def with_shape(self, shape: Iterable[int]) -> "PhantomConfig":
        return PhantomConfig(
            shape=tuple(int(s) for s in shape),
            spacing=self.spacing,
            lesion_radius_mm=self.lesion_radius_mm,
            noise_sigma=self.noise_sigma,
            seed=self.seed,
            head_radius_fraction=self.head_radius_fraction,
            lesion_intensity=self.lesion_intensity,
            noise_model=self.noise_model,
        )
