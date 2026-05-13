"""Patch extraction, lesion centroids and balanced sampling tests."""

import numpy as np
import pytest

from src.medimg.patching import (
    Patch,
    PatchSampler,
    balance_patches,
    class_counts,
    extract_patch,
    lesion_centers,
    lesion_components,
    negative_centers,
    patch_array,
    positive_centers,
    split_patches,
)


def two_blob_mask(shape=(40, 40)) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[5:9, 5:9] = True
    mask[28:34, 28:34] = True
    return mask


def test_extract_patch_takes_centre_window():
    image = np.arange(100, dtype=np.float32).reshape(10, 10)
    patch = extract_patch(image, (5, 5), 4)
    assert patch.shape == (4, 4)
    assert patch.tolist() == image[3:7, 3:7].tolist()


def test_extract_patch_pads_at_border():
    image = np.ones((10, 10), dtype=np.float32)
    patch = extract_patch(image, (0, 0), 4, pad_value=-7.0)
    assert patch.shape == (4, 4)
    assert patch[0, 0] == pytest.approx(-7.0)
    assert patch[3, 3] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        extract_patch(image, (1, 1), 5)
    with pytest.raises(ValueError):
        extract_patch(np.zeros((4, 4, 4)), (1, 1), 4)
