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
