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


def test_noise_changes_values_but_not_geometry():
    clean, mask_clean = brain_phantom(small_cfg(noise_sigma=0.0))
    noisy, mask_noisy = brain_phantom(small_cfg(noise_sigma=10.0))
    assert np.array_equal(mask_clean, mask_noisy)
    assert not np.allclose(clean, noisy)
    assert abs(float(noisy.mean()) - float(clean.mean())) < 3.0


def test_add_gaussian_noise_std_and_mask():
    rng = np.random.default_rng(0)
    base = np.zeros((64, 64), dtype=np.float32)
    noisy = add_gaussian_noise(base, 5.0, rng)
    assert noisy.std() == pytest.approx(5.0, rel=0.08)
    mask = np.zeros((64, 64), dtype=bool)
    mask[:8, :8] = True
    partial = add_gaussian_noise(base, 5.0, rng, mask=mask)
    assert np.allclose(partial[8:, 8:], 0.0)
    assert partial[:8, :8].std() > 1.0


def test_add_rician_noise_is_non_negative():
    rng = np.random.default_rng(1)
    base = np.full((16, 16), 100.0, dtype=np.float32)
    noisy = add_rician_noise(base, 5.0, rng)
    assert noisy.min() >= 0.0
    assert noisy.mean() == pytest.approx(100.0, abs=1.5)
    with pytest.raises(ValueError):
        add_rician_noise(base, -1.0, rng)


def test_lesion_mask_and_draw_lesion_agree():
    shape = (24, 24, 16)
    lesion = Lesion(center=(12, 12, 8), radius=(3.0, 3.0, 2.0))
    mask = lesion.mask(shape)
    data = np.zeros(shape, dtype=np.float32)
    painted = draw_lesion(data, lesion)
    assert np.array_equal(painted > 0, mask)
    assert painted[mask].max() == pytest.approx(HU_LESION)
    blended = draw_lesion(data + 40.0, lesion, blend=True)
    assert blended[mask].max() == pytest.approx(HU_LESION + 40.0)


def test_random_lesion_center_stays_inside_head():
    cfg = small_cfg()
    rng = np.random.default_rng(2)
    center = head_center(cfg.shape)
    for _ in range(10):
        lesion_center = random_lesion_center(cfg, rng)
        distance = np.linalg.norm(np.array(lesion_center) - np.array(center))
        assert distance < min(cfg.shape) / 2.0


def test_lesion_slice_index_returns_densest_slice():
    mask = np.zeros((8, 8, 6), dtype=bool)
    mask[3:5, 3:5, 4] = True
    assert lesion_slice_index(mask, axis=2) == 4
    with pytest.raises(ValueError):
        lesion_slice_index(np.zeros((4, 4), dtype=bool))


def test_phantom_slice_validates_axis_and_index():
    volume, _ = make_phantom_pair(small_cfg())
    sl = phantom_slice(volume, axis=2, index=0)
    assert sl.shape == (32, 32)
    assert phantom_slice(volume, axis=2).shape == (32, 32)
    with pytest.raises(IndexError):
        phantom_slice(volume, axis=2, index=99)
    with pytest.raises(ValueError):
        phantom_slice(volume, axis=7)


def test_phantom_summary_reports_positive_volumes():
    summary = phantom_summary(small_cfg())
    assert summary["voxels"] == 32 * 32 * 24
    assert summary["lesion_voxels"] > 0
    assert summary["lesion_mm3"] == pytest.approx(summary["lesion_voxels"] * 2.0)
    assert summary["hu_min"] <= HU_AIR + 1e-3
