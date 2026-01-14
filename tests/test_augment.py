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


def test_flip_with_probability_zero_is_identity_copy():
    image = checkerboard()
    out = random_flip(image, np.random.default_rng(0), p=0.0)
    assert np.array_equal(out, image)
    assert out is not image


def test_rot90_preserves_shape_and_value_multiset():
    image = checkerboard(7)
    out = random_rot90(image, np.random.default_rng(3), p=1.0)
    assert out.shape == image.shape
    assert sorted(out.ravel().tolist()) == sorted(image.ravel().tolist())
    identity = random_rot90(image, np.random.default_rng(3), p=0.0)
    assert np.array_equal(identity, image)


def test_gamma_with_unity_is_identity():
    image = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
    rng = np.random.default_rng(4)
    out = random_gamma(image, rng, p=1.0, low=1.0, high=1.0)
    assert np.allclose(out, image, atol=1e-6)


def test_gamma_handles_out_of_range_values_without_nan():
    image = np.array([[-2.0, 0.5], [1.5, 0.0]], dtype=np.float32)
    out = random_gamma(image, np.random.default_rng(0), p=1.0, low=2.0, high=2.0)
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_intensity_scale_respects_range():
    image = np.ones((4, 4), dtype=np.float32)
    out = random_intensity_scale(image, np.random.default_rng(5), p=1.0, low=2.0, high=2.0)
    assert np.allclose(out, 2.0)
    unchanged = random_intensity_scale(image, np.random.default_rng(5), p=0.0)
    assert np.allclose(unchanged, 1.0)


def test_gaussian_noise_std_and_clip():
    image = np.full((64, 64), 0.5, dtype=np.float32)
    out = random_gaussian_noise(image, np.random.default_rng(6), p=1.0, std=0.05)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert float(out.std()) == pytest.approx(0.05, rel=0.15)


def test_translate_pair_moves_image_and_mask_together():
    image = np.zeros((10, 10), dtype=np.float32)
    mask = np.zeros((10, 10), dtype=bool)
    image[4:6, 4:6] = 1.0
    mask[4:6, 4:6] = True
    rng = np.random.default_rng(7)
    shifted_image, shifted_mask = translate_pair(
        image, mask, rng, p=1.0, max_shift=2
    )
    assert np.array_equal(shifted_image > 0, shifted_mask)
    assert shifted_mask.sum() == mask.sum()
    assert shifted_image.shape == image.shape


def test_translate_pair_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        translate_pair(np.zeros((4, 4)), np.zeros((4, 5), dtype=bool), np.random.default_rng(0))


def test_augment_image_is_bounded_and_seeded():
    image = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
    cfg = AugmentConfig()
    a = augment_image(image, cfg, seed=11)
    b = augment_image(image, cfg, seed=11)
    c = augment_image(image, cfg, seed=12)
    assert a.shape == image.shape
    assert a.min() >= 0.0 and a.max() <= 1.0
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_augment_batch_handles_3d_and_4d():
    batch3 = np.random.default_rng(8).random((4, 8, 8)).astype(np.float32)
    out3 = augment_batch(batch3, AugmentConfig(), seed=1)
    assert out3.shape == batch3.shape
    batch4 = batch3[:, None, :, :]
    out4 = augment_batch(batch4, AugmentConfig(), seed=1)
    assert out4.shape == batch4.shape
    with pytest.raises(ValueError):
        augment_batch(np.zeros((8, 8)), AugmentConfig())


def test_augment_patches_keeps_labels():
    patches = [
        Patch(np.random.default_rng(i).random((8, 8)).astype(np.float32), i % 2, (0, 0))
        for i in range(6)
    ]
    out = augment_patches(patches, AugmentConfig(), seed=2)
    assert [p.label for p in out] == [p.label for p in patches]
    assert all(p.image.shape == (8, 8) for p in out)


def test_augmentation_summary_probabilities():
    summary = augmentation_summary(AugmentConfig(p_flip=0.5, p_rot90=0.5, p_gamma=0.0, p_noise=0.0, p_scale=0.0))
    assert summary["p_unchanged"] == pytest.approx(0.25)
    assert summary["p_modified"] == pytest.approx(0.75)
    assert summary["expected_transforms"] == pytest.approx(1.0)
