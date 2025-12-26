"""Augmentation tests: probabilities, ranges and label safety."""

import numpy as np
import pytest

from src.medimg.augment import (
    AugmentConfig,
    augmentation_summary,
    augment_batch,
    augment_image,
    augment_patches,
    random_flip,
    random_gamma,
    random_gaussian_noise,
    random_intensity_scale,
    random_rot90,
    translate_pair,
)
from src.medimg.patching import Patch


def checkerboard(size: int = 8) -> np.ndarray:
    image = np.zeros((size, size), dtype=np.float32)
    image[::2, ::2] = 1.0
    return image


def test_config_validation_and_dict():
    cfg = AugmentConfig()
    payload = cfg.to_dict()
    assert payload["p_flip"] == 0.5
    assert payload["gamma_range"] == [0.8, 1.25]
    with pytest.raises(ValueError):
        AugmentConfig(p_flip=1.5)
    with pytest.raises(ValueError):
        AugmentConfig(gamma_range=(2.0, 1.0))
    with pytest.raises(ValueError):
        AugmentConfig(noise_std=-0.1)
    with pytest.raises(ValueError):
        AugmentConfig(max_shift=-1)


def test_flip_with_probability_one_is_exact_flip():
    image = checkerboard()
    rng = np.random.default_rng(0)
    flipped = random_flip(image, rng, p=1.0, axis=1)
    assert np.array_equal(flipped, np.flip(image, axis=1))
    vertical = random_flip(image, rng, p=1.0, axis=0)
    assert np.array_equal(vertical, np.flip(image, axis=0))
