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


def as_volume(data: np.ndarray, spacing: Sequence[float] | None = None) -> Volume:
    """Wrap an array in a :class:`Volume`, inferring a unit spacing."""
    spacing = tuple(float(s) for s in (spacing or (1.0, 1.0, 1.0)))
    return Volume(data=np.asarray(data), spacing=spacing)


def validate_volume(volume: Volume) -> None:
    """Raise ``ValueError`` for metadata that would break downstream maths."""
    if volume.modality not in {"CT", "MR", "PT", "US", "SYNTH"}:
        raise ValueError(f"unsupported modality {volume.modality!r}")
    if not np.isfinite(volume.data).all():
        raise ValueError("volume contains NaN or infinite voxels")
    if max(volume.spacing) / min(volume.spacing) > 10.0:
        raise ValueError(
            "anisotropy above 10x is not supported by the resampler; resample upstream"
        )


def save_volume(volume: Volume, path: str | Path) -> Path:
    """Save a volume to a ``.npz`` archive with its metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        data=volume.data,
        spacing=np.asarray(volume.spacing, dtype=np.float64),
        origin=np.asarray(volume.origin, dtype=np.float64),
        modality=np.asarray(volume.modality),
        meta=np.asarray(json.dumps(volume.meta)),
    )
    return path


def load_volume(path: str | Path) -> Volume:
    """Load a volume written by :func:`save_volume`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no volume at {path}")
    with np.load(path, allow_pickle=False) as archive:
        missing = {"data", "spacing", "origin", "modality"} - set(archive.files)
        if missing:
            raise ValueError(f"{path} is missing keys: {sorted(missing)}")
        meta = {}
        if "meta" in archive.files:
            meta = json.loads(str(archive["meta"]))
        return Volume(
            data=archive["data"],
            spacing=tuple(float(s) for s in archive["spacing"]),
            origin=tuple(float(o) for o in archive["origin"]),
            modality=str(archive["modality"]),
            meta=meta,
        )


def resample_to_spacing(volume: Volume, target_spacing: Sequence[float], order: int = 1) -> Volume:
    """Resample to isotropic-ish target spacing with a spline zoom.

    ``order=1`` (trilinear) is the default because cubic interpolation can
    overshoot the bone/air boundary and invent HU values above 3000.
    """
    target = tuple(float(s) for s in target_spacing)
    if len(target) != 3 or any(s <= 0 for s in target):
        raise ValueError("target_spacing must be three positive floats")
    if order not in (0, 1, 2, 3):
        raise ValueError("order must be 0..3")
    factors = [cur / tgt for cur, tgt in zip(volume.spacing, target)]
    if all(abs(f - 1.0) < 1e-9 for f in factors):
        return volume
    zoomed = ndimage.zoom(volume.data, factors, order=order, grid_mode=False)
    meta = dict(volume.meta)
    meta["resampled_to"] = target
    meta["source_spacing"] = volume.spacing
    return Volume(
        data=zoomed.astype(np.float32),
        spacing=target,
        origin=volume.origin,
        modality=volume.modality,
        meta=meta,
    )


def crop_or_pad(volume: Volume, shape: Sequence[int], pad_value: float = -1000.0) -> Volume:
    """Centre-crop or centre-pad the volume to ``shape``."""
    target = tuple(int(s) for s in shape)
    if len(target) != 3 or any(s <= 0 for s in target):
        raise ValueError("shape must be three positive ints")
    src = volume.data
    out = np.full(target, pad_value, dtype=np.float32)
    src_slices = []
    dst_slices = []
    for cur, tgt in zip(src.shape, target):
        if cur >= tgt:
            start = (cur - tgt) // 2
            src_slices.append(slice(start, start + tgt))
            dst_slices.append(slice(0, tgt))
        else:
            start = (tgt - cur) // 2
            src_slices.append(slice(0, cur))
            dst_slices.append(slice(start, start + cur))
    out[tuple(dst_slices)] = src[tuple(src_slices)]
    return volume.with_data(out, cropped_to=target)


def take_slice(volume: Volume, axis: int, index: int) -> np.ndarray:
    """Return a 2-D slice, validating the axis and index."""
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1 or 2")
    size = volume.shape[axis]
    if not -size <= index < size:
        raise IndexError(f"slice {index} out of range for axis {axis} (size {size})")
    return np.take(volume.data, index % size, axis=axis)


def iter_slices(volume: Volume, axis: int = 2) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(index, slice)`` pairs along ``axis``."""
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1 or 2")
    for index in range(volume.shape[axis]):
        yield index, take_slice(volume, axis, index)


def volume_stats(volume: Volume) -> dict[str, float]:
    """Intensity statistics used for normalisation diagnostics."""
    data = volume.data
    return {
        "min": float(data.min()),
        "max": float(data.max()),
        "mean": float(data.mean()),
        "std": float(data.std()),
        "p01": float(np.percentile(data, 1)),
        "p99": float(np.percentile(data, 99)),
        "voxel_volume_mm3": volume.voxel_volume,
    }


def nonzero_bounds(volume: Volume, threshold: float = -900.0) -> tuple[tuple[int, int], ...]:
    """Bounding box of voxels above ``threshold`` along every axis.

    Useful to crop the air around a head phantom before patch sampling.
    """
    mask = volume.data > threshold
    if not mask.any():
        raise ValueError("no voxels above threshold")
    bounds = []
    for axis in range(3):
        other = tuple(a for a in range(3) if a != axis)
        projected = mask.any(axis=other)
        idx = np.flatnonzero(projected)
        bounds.append((int(idx[0]), int(idx[-1]) + 1))
    return tuple(bounds)  # type: ignore[return-value]
