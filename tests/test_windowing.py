"""Windowing and normalisation arithmetic tests."""

import numpy as np
import pytest

from src.medimg.volume import Volume
from src.medimg.windowing import (
    WINDOW_PRESETS,
    WindowPreset,
    apply_preset,
    apply_window,
    get_preset,
    invert_window,
    minmax_normalize,
    normalize_array,
    normalize_volume,
    percentile_normalize,
    window_batch,
    window_statistics,
    zscore_normalize,
)


def test_preset_geometry():
    brain = WINDOW_PRESETS["brain"]
    assert (brain.level, brain.width) == (40.0, 80.0)
    assert brain.low == 0.0
    assert brain.high == 80.0
    assert brain.contains(40.0) and not brain.contains(81.0)
    with pytest.raises(ValueError):
        WindowPreset("bad", level=0.0, width=0.0, description="x")


def test_get_preset_lists_choices_on_error():
    assert get_preset("lung").level == -600.0
    with pytest.raises(KeyError) as excinfo:
        get_preset("nope")
    assert "abdomen" in str(excinfo.value)


def test_apply_window_maps_window_edges_to_zero_and_one():
    level, width = 40.0, 80.0
    values = np.array([-1000.0, 0.0, 40.0, 80.0, 2000.0], dtype=np.float32)
    out = apply_window(values, level, width)
    assert out.tolist() == pytest.approx([0.0, 0.0, 0.5, 1.0, 1.0])
    unclipped = apply_window(values, level, width, clip=False)
    assert unclipped[0] < 0.0 and unclipped[-1] > 1.0


def test_apply_window_rejects_nonpositive_width():
    with pytest.raises(ValueError):
        apply_window(np.zeros((4, 4)), 0.0, 0.0)


def test_apply_preset_matches_manual_window():
    image = np.array([[0.0, 40.0], [80.0, -200.0]], dtype=np.float32)
    assert np.allclose(apply_preset(image, "brain"), apply_window(image, 40.0, 80.0))


def test_window_batch_stacks_and_validates():
    images = [np.full((4, 4), 40.0, dtype=np.float32), np.full((4, 4), 0.0, dtype=np.float32)]
    batch = window_batch(images, 40.0, 80.0)
    assert batch.shape == (2, 4, 4)
    assert batch[0].mean() == pytest.approx(0.5)
    assert batch[1].mean() == pytest.approx(0.0)
    with pytest.raises(ValueError):
        window_batch([], 40.0, 80.0)
    with pytest.raises(ValueError):
        window_batch([np.zeros((4, 4)), np.zeros((4, 5))], 40.0, 80.0)


def test_minmax_normalize_exact_values():
    image = np.array([[10.0, 20.0], [30.0, 50.0]], dtype=np.float32)
    out = minmax_normalize(image)
    assert out.min() == 0.0 and out.max() == 1.0
    assert out[0, 1] == pytest.approx(0.25)
    assert np.allclose(minmax_normalize(np.full((3, 3), 7.0)), 0.0)


def test_zscore_normalize_has_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    image = rng.normal(50.0, 8.0, size=(32, 32))
    out = zscore_normalize(image)
    assert abs(float(out.mean())) < 1e-6
    assert float(out.std()) == pytest.approx(1.0, rel=1e-5)
    assert np.allclose(zscore_normalize(np.full((4, 4), 3.0)), 0.0)


def test_percentile_normalize_clips_outliers():
    image = np.zeros((100, 100), dtype=np.float32)
    image[:99] = 10.0
    image[99, :] = 10_000.0
    out = percentile_normalize(image, 1.0, 95.0)
    assert out.max() <= 1.0
    assert out.min() >= 0.0
    with pytest.raises(ValueError):
        percentile_normalize(image, 90.0, 10.0)


def test_normalize_array_dispatch_and_errors():
    image = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    assert normalize_array(image, "minmax").max() == pytest.approx(1.0)
    assert abs(float(normalize_array(image, "zscore").mean())) < 1e-6
    with pytest.raises(KeyError):
        normalize_array(image, "magic")


def test_normalize_volume_preserves_spacing():
    volume = Volume(data=np.arange(64, dtype=np.float32).reshape(4, 4, 4), spacing=(2.0, 2.0, 3.0))
    out = normalize_volume(volume, "minmax")
    assert out.spacing == volume.spacing
    assert out.meta["normalization"] == "minmax"
    assert out.data.max() == pytest.approx(1.0)


def test_invert_window_flips_and_validates():
    image = np.array([0.0, 0.25, 1.0], dtype=np.float32)
    assert invert_window(image).tolist() == pytest.approx([1.0, 0.75, 0.0])
    with pytest.raises(ValueError):
        invert_window(np.array([-0.5, 0.5], dtype=np.float32))


def test_window_statistics_reports_clipped_fraction():
    image = np.array([-1000.0, -1000.0, 40.0, 40.0], dtype=np.float32)
    stats = window_statistics(image, 40.0, 80.0)
    assert stats["clipped_low"] == pytest.approx(0.5)
    assert stats["clipped_high"] == pytest.approx(0.0)
    assert stats["usable_fraction"] == pytest.approx(0.5)
    assert stats["window_low"] == 0.0 and stats["window_high"] == 80.0
