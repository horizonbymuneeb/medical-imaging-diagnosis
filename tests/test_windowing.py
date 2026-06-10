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
