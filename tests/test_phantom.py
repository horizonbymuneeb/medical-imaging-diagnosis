"""Phantom geometry and intensity tests.

These assert the physical properties a reviewer would check by hand: the
ellipsoid mask encloses the right volume, the skull ring is bone-bright, the
lesion mask matches the painted voxels, and two runs with the same seed are
bit-identical.
"""

import numpy as np
import pytest

from src.medimg.phantom import (
    HU_AIR,
    HU_BONE,
    HU_LESION,
    Lesion,
    PhantomConfig,
    add_gaussian_noise,
    add_rician_noise,
    brain_phantom,
    draw_lesion,
    ellipsoid_mask,
    head_center,
    lesion_slice_index,
    make_phantom_pair,
    phantom_slice,
    phantom_summary,
    random_lesion_center,
    shell_mask,
    voxel_grid,
)


def small_cfg(**kwargs) -> PhantomConfig:
    base = dict(shape=(32, 32, 24), spacing=(1.0, 1.0, 2.0), noise_sigma=0.0, seed=3)
    base.update(kwargs)
    return PhantomConfig(**base)


def test_config_rejects_invalid_geometry():
    with pytest.raises(ValueError):
        PhantomConfig(shape=(32, 32))
    with pytest.raises(ValueError):
        PhantomConfig(spacing=(1.0, 0.0, 1.0))
    with pytest.raises(ValueError):
        PhantomConfig(noise_sigma=-1.0)
    with pytest.raises(ValueError):
        PhantomConfig(noise_model="poisson")


def test_lesion_radius_converts_mm_to_voxels():
    cfg = PhantomConfig(shape=(64, 64, 48), spacing=(1.5, 2.0, 4.0), lesion_radius_mm=6.0)
    assert cfg.lesion_radius_voxels == (4.0, 3.0, 1.5)
    assert cfg.voxel_volume_mm3 == pytest.approx(12.0)
