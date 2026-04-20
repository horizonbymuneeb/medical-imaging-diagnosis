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


@dataclass(frozen=True)
class Lesion:
    """A spherical lesion in voxel coordinates."""

    center: tuple[float, float, float]
    radius: tuple[float, float, float]
    intensity: float = HU_LESION

    def mask(self, shape: tuple[int, int, int]) -> np.ndarray:
        return ellipsoid_mask(shape, self.center, self.radius)


def voxel_grid(shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return coordinate grids for a volume, in voxel units and ``float32``."""
    if len(shape) != 3:
        raise ValueError("shape must be 3-dimensional")
    axes = [np.arange(int(n), dtype=np.float32) for n in shape]
    return np.meshgrid(*axes, indexing="ij")


def ellipsoid_mask(
    shape: tuple[int, int, int],
    center: Iterable[float],
    radii: Iterable[float],
) -> np.ndarray:
    """Boolean mask of an axis-aligned ellipsoid.

    The mask is ``True`` where ``sum(((x - c) / r) ** 2) <= 1``.  A radius of
    zero raises instead of silently producing an empty mask, because that is
    almost always a units bug (mm passed where voxels were expected).
    """
    center = tuple(float(c) for c in center)
    radii = tuple(float(r) for r in radii)
    if len(center) != 3 or len(radii) != 3:
        raise ValueError("center and radii must have length 3")
    if any(r <= 0 for r in radii):
        raise ValueError(f"radii must be positive, got {radii!r}")
    grids = voxel_grid(tuple(int(s) for s in shape))
    dist = np.zeros(tuple(int(s) for s in shape), dtype=np.float32)
    for grid, c, r in zip(grids, center, radii):
        dist += ((grid - c) / r) ** 2
    return dist <= 1.0


def shell_mask(
    shape: tuple[int, int, int],
    center: Iterable[float],
    radii: Iterable[float],
    inner_scale: float,
) -> np.ndarray:
    """Mask of a shell between ``radii`` and ``radii * inner_scale``."""
    if not 0.0 < inner_scale < 1.0:
        raise ValueError("inner_scale must be in (0, 1)")
    outer = ellipsoid_mask(shape, center, radii)
    inner = ellipsoid_mask(shape, center, [r * inner_scale for r in radii])
    return outer & ~inner


def head_center(shape: tuple[int, int, int]) -> tuple[float, float, float]:
    return tuple((n - 1) / 2.0 for n in shape)


def head_radii(cfg: PhantomConfig) -> tuple[float, float, float]:
    """In-plane radius from the config, with the slice axis stretched to fit."""
    base = cfg.head_radius_fraction * min(cfg.shape[0], cfg.shape[1])
    return (base, base, base * 1.05)


def draw_lesion(
    data: np.ndarray,
    lesion: Lesion,
    *,
    blend: bool = False,
) -> np.ndarray:
    """Write a lesion into ``data`` (a copy) and return it.

    With ``blend=True`` the lesion intensity is added on top of the local
    tissue value instead of replacing it, which produces a slightly brighter
    rim after windowing.
    """
    out = np.array(data, dtype=np.float32, copy=True)
    mask = lesion.mask(tuple(int(s) for s in out.shape))
    if blend:
        out[mask] = out[mask] + lesion.intensity
    else:
        out[mask] = lesion.intensity
    return out


def add_gaussian_noise(
    data: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
    *,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Add zero-mean Gaussian noise, optionally only inside ``mask``."""
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    noise = rng.normal(0.0, sigma, size=data.shape).astype(np.float32)
    out = np.array(data, dtype=np.float32, copy=True)
    if mask is None:
        return out + noise
    out[mask] = out[mask] + noise[mask]
    return out


def add_rician_noise(
    data: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Rician noise, the usual model for magnitude MR images.

    Two independent Gaussian channels are added and the magnitude is taken, so
    the noise is signal-dependent near zero and strictly positive.
    """
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    real = data + rng.normal(0.0, sigma, size=data.shape)
    imag = rng.normal(0.0, sigma, size=data.shape)
    return np.sqrt(real**2 + imag**2).astype(np.float32)


def random_lesion_center(
    cfg: PhantomConfig,
    rng: np.random.Generator,
    *,
    margin: float = 0.35,
) -> tuple[float, float, float]:
    """Pick a lesion centre inside the brain but away from the skull."""
    center = head_center(cfg.shape)
    radii = head_radii(cfg)
    limit = [r * (1.0 - margin) for r in radii]
    offsets = [rng.uniform(-lim, lim) for lim in limit]
    return tuple(c + o for c, o in zip(center, offsets))


def brain_phantom(cfg: PhantomConfig) -> tuple[np.ndarray, np.ndarray]:
    """Build one head phantom.

    Returns ``(data, lesion_mask)`` where ``data`` is a float32 volume in HU
    and ``lesion_mask`` marks the hyperdense lesion voxels.  The image is
    deterministic for a given ``cfg.seed``.
    """
    rng = np.random.default_rng(cfg.seed)
    shape = tuple(int(s) for s in cfg.shape)
    center = head_center(shape)
    radii = head_radii(cfg)

    data = np.full(shape, HU_AIR, dtype=np.float32)
    # Scalp/soft tissue envelope, then skull ring, then brain parenchyma.
    data[ellipsoid_mask(shape, center, radii)] = HU_SOFT_TISSUE
    data[shell_mask(shape, center, radii, inner_scale=0.88)] = HU_BONE
    brain_mask = ellipsoid_mask(shape, center, [r * 0.84 for r in radii])
    data[brain_mask] = HU_WHITE_MATTER

    # Grey-matter ribbon and a pair of ventricles give the slice real texture.
    gm = shell_mask(shape, center, [r * 0.84 for r in radii], inner_scale=0.93)
    data[gm] = HU_GRAY_MATTER
    for sign in (-1.0, 1.0):
        vent_center = (
            center[0] + sign * radii[0] * 0.10,
            center[1] + radii[1] * 0.06,
            center[2],
        )
        vent = ellipsoid_mask(
            shape, vent_center, [radii[0] * 0.10, radii[1] * 0.22, radii[2] * 0.16]
        )
        data[vent & brain_mask] = HU_CSF

    lesion = Lesion(
        center=random_lesion_center(cfg, rng),
        radius=cfg.lesion_radius_voxels,
        intensity=cfg.lesion_intensity,
    )
    lesion_mask = lesion.mask(shape) & brain_mask
    data[lesion_mask] = lesion.intensity

    if cfg.noise_sigma > 0:
        if cfg.noise_model == "rician":
            data = add_rician_noise(data, cfg.noise_sigma, rng)
        else:
            data = add_gaussian_noise(data, cfg.noise_sigma, rng)

    return data.astype(np.float32), lesion_mask
