"""Volume metadata, serialisation and geometry tests."""

import numpy as np
import pytest

from src.medimg.volume import (
    Volume,
    as_volume,
    crop_or_pad,
    iter_slices,
    load_volume,
    nonzero_bounds,
    resample_to_spacing,
    save_volume,
    take_slice,
    validate_volume,
    volume_stats,
)


def make_volume(shape=(8, 10, 6), spacing=(1.0, 2.0, 3.0)) -> Volume:
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    return Volume(data=data, spacing=spacing, origin=(1.0, 2.0, 3.0), modality="CT")


def test_volume_rejects_2d_data():
    with pytest.raises(ValueError):
        Volume(data=np.zeros((4, 4)))


def test_volume_rejects_bad_spacing():
    with pytest.raises(ValueError):
        Volume(data=np.zeros((4, 4, 4)), spacing=(1.0, -1.0, 1.0))


def test_voxel_volume_and_extent_are_physical():
    volume = make_volume()
    assert volume.voxel_volume == pytest.approx(6.0)
    assert volume.extent_mm == pytest.approx((8.0, 20.0, 18.0))
    assert volume.shape == (8, 10, 6)


def test_with_data_keeps_metadata_and_merges_meta():
    volume = make_volume()
    updated = volume.with_data(np.zeros_like(volume.data), normalization="minmax")
    assert updated.spacing == volume.spacing
    assert updated.origin == volume.origin
    assert updated.meta["normalization"] == "minmax"
    assert updated.data.max() == 0.0


def test_save_and_load_roundtrip(tmp_path):
    volume = make_volume()
    path = save_volume(volume, tmp_path / "vol.npz")
    loaded = load_volume(path)
    assert loaded.shape == volume.shape
    assert loaded.spacing == pytest.approx(volume.spacing)
    assert loaded.origin == pytest.approx(volume.origin)
    assert loaded.modality == "CT"
    assert np.array_equal(loaded.data, volume.data)


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_volume(tmp_path / "nope.npz")


def test_load_rejects_archive_without_metadata(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, data=np.zeros((4, 4, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        load_volume(path)


def test_resample_to_spacing_changes_shape_proportionally():
    volume = make_volume(shape=(8, 10, 6), spacing=(1.0, 1.0, 2.0))
    resampled = resample_to_spacing(volume, (2.0, 2.0, 2.0))
    assert resampled.spacing == (2.0, 2.0, 2.0)
    assert resampled.shape[0] == pytest.approx(4, abs=1)
    assert resampled.shape[2] == pytest.approx(6, abs=1)
    assert resampled.meta["resampled_to"] == (2.0, 2.0, 2.0)


def test_resample_is_noop_at_same_spacing():
    volume = make_volume()
    assert resample_to_spacing(volume, volume.spacing) is volume
    with pytest.raises(ValueError):
        resample_to_spacing(volume, (1.0, 1.0))


def test_crop_or_pad_both_directions():
    volume = make_volume(shape=(8, 8, 8))
    cropped = crop_or_pad(volume, (4, 4, 4))
    assert cropped.shape == (4, 4, 4)
    padded = crop_or_pad(volume, (10, 10, 10), pad_value=-1000.0)
    assert padded.shape == (10, 10, 10)
    assert padded.data[0, 0, 0] == pytest.approx(-1000.0)
    assert padded.data[1:9, 1:9, 1:9].shape == (8, 8, 8)
    assert np.array_equal(padded.data[1:9, 1:9, 1:9], volume.data)
    with pytest.raises(ValueError):
        crop_or_pad(volume, (0, 4, 4))


def test_take_slice_and_negative_index():
    volume = make_volume(shape=(4, 5, 6))
    assert take_slice(volume, 2, 0).shape == (4, 5)
    assert np.array_equal(take_slice(volume, 2, -1), take_slice(volume, 2, 5))
    with pytest.raises(IndexError):
        take_slice(volume, 2, 6)
    with pytest.raises(ValueError):
        take_slice(volume, 5, 0)


def test_iter_slices_yields_every_index():
    volume = make_volume(shape=(4, 4, 5))
    pairs = list(iter_slices(volume, axis=2))
    assert [index for index, _ in pairs] == [0, 1, 2, 3, 4]
    assert all(slice_.shape == (4, 4) for _, slice_ in pairs)


def test_volume_stats_match_numpy():
    volume = make_volume()
    stats = volume_stats(volume)
    assert stats["min"] == pytest.approx(float(volume.data.min()))
    assert stats["max"] == pytest.approx(float(volume.data.max()))
    assert stats["mean"] == pytest.approx(float(volume.data.mean()))
    assert stats["std"] == pytest.approx(float(volume.data.std()))
    assert stats["p01"] <= stats["p99"]


def test_validate_volume_catches_nan_and_anisotropy():
    ok = make_volume()
    validate_volume(ok)
    with pytest.raises(ValueError):
        validate_volume(Volume(data=np.full((4, 4, 4), np.nan), spacing=(1, 1, 1)))
    with pytest.raises(ValueError):
        validate_volume(Volume(data=np.zeros((4, 4, 4)), spacing=(1, 1, 20)))
    with pytest.raises(ValueError):
        validate_volume(Volume(data=np.zeros((4, 4, 4)), modality="XX"))


def test_nonzero_bounds_returns_tight_box():
    data = np.zeros((10, 10, 10), dtype=np.float32)
    data[2:5, 3:7, 1:9] = 100.0
    bounds = nonzero_bounds(Volume(data=data), threshold=50.0)
    assert bounds == ((2, 5), (3, 7), (1, 9))
    with pytest.raises(ValueError):
        nonzero_bounds(Volume(data=np.zeros((4, 4, 4))), threshold=50.0)


def test_as_volume_wraps_array():
    volume = as_volume(np.zeros((3, 3, 3)))
    assert volume.spacing == (1.0, 1.0, 1.0)
    assert volume.modality == "CT"
