"""Volume container plus the loading, resampling and slicing helpers.

A ``Volume`` is a NumPy array with the metadata that medical images always
carry: voxel spacing, patient-space origin and modality.  The class is frozen
because a volume is passed between pipeline stages and silent in-place mutation
of the array is the classic way to corrupt a training set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class Volume:
    """A 3-D image with spacing metadata.

    ``data`` must be 3-D; spacing is millimetres per voxel along (x, y, z).
    """

    data: np.ndarray
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    modality: str = "CT"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        data = np.asarray(self.data)
        if data.ndim != 3:
            raise ValueError(f"volume data must be 3-D, got shape {data.shape}")
        if len(self.spacing) != 3 or any(float(s) <= 0 for s in self.spacing):
            raise ValueError(f"spacing must be three positive floats, got {self.spacing!r}")
        if len(self.origin) != 3:
            raise ValueError("origin must have three components")
        object.__setattr__(self, "data", data.astype(np.float32, copy=False))
        object.__setattr__(self, "spacing", tuple(float(s) for s in self.spacing))
        object.__setattr__(self, "origin", tuple(float(o) for o in self.origin))

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(n) for n in self.data.shape)  # type: ignore[return-value]

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def voxel_volume(self) -> float:
        """Physical volume of one voxel in mm^3."""
        return float(np.prod(self.spacing))

    @property
    def extent_mm(self) -> tuple[float, float, float]:
        """Physical size of the volume along each axis in millimetres."""
        return tuple(n * s for n, s in zip(self.shape, self.spacing))  # type: ignore[return-value]

    def with_data(self, data: np.ndarray, **meta_updates) -> "Volume":
        """Return a copy of the volume with new pixel data and merged metadata."""
        meta = dict(self.meta)
        meta.update(meta_updates)
        return Volume(
            data=data,
            spacing=self.spacing,
            origin=self.origin,
            modality=self.modality,
            meta=meta,
        )

    def replace_meta(self, **updates) -> "Volume":
        meta = dict(self.meta)
        meta.update(updates)
        return Volume(
            data=self.data,
            spacing=self.spacing,
            origin=self.origin,
            modality=self.modality,
            meta=meta,
        )
