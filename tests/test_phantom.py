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


def test_ellipsoid_mask_matches_analytic_volume():
    shape = (41, 41, 41)
    radii = (8.0, 8.0, 8.0)
    mask = ellipsoid_mask(shape, (20.0, 20.0, 20.0), radii)
    analytic = 4.0 / 3.0 * np.pi * 8.0**3
    assert mask.dtype == np.bool_
    assert mask[20, 20, 20]
    assert not mask[0, 0, 0]
    assert mask.sum() == pytest.approx(analytic, rel=0.12)


def test_ellipsoid_mask_rejects_zero_radius():
    with pytest.raises(ValueError):
        ellipsoid_mask((8, 8, 8), (4, 4, 4), (2.0, 0.0, 2.0))


def test_shell_mask_is_hollow():
    shape = (41, 41, 41)
    ring = shell_mask(shape, (20, 20, 20), (10.0, 10.0, 10.0), inner_scale=0.5)
    assert ring[20, 20, 20] is np.False_ or not ring[20, 20, 20]
    assert ring[20, 20, 28]  # between r=5 and r=10
    assert not ring[20, 20, 40]


def test_voxel_grid_axis_ordering():
    rows, cols, slices = voxel_grid((3, 4, 5))
    assert rows.shape == (3, 4, 5)
    assert rows[2, 0, 0] == 2.0
    assert cols[0, 3, 0] == 3.0
    assert slices[0, 0, 4] == 4.0


def test_brain_phantom_contains_skull_brain_and_lesion():
    cfg = small_cfg()
    data, lesion = brain_phantom(cfg)
    assert data.shape == cfg.shape
    assert data.dtype == np.float32
    assert data[0, 0, 0] == pytest.approx(HU_AIR)
    assert (data > 800).sum() > 0, "no bone-bright voxels"
    assert (data > 800).max() <= HU_BONE + 1.0
    assert lesion.sum() > 0
    assert data[lesion].min() == pytest.approx(HU_LESION, abs=1e-3)


def test_brain_phantom_is_deterministic_per_seed():
    a, ma = brain_phantom(small_cfg(seed=5))
    b, mb = brain_phantom(small_cfg(seed=5))
    c, mc = brain_phantom(small_cfg(seed=6))
    assert np.array_equal(a, b)
    assert np.array_equal(ma, mb)
    assert not np.array_equal(ma, mc)
