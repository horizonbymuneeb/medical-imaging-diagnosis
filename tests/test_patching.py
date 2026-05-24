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


def test_lesion_components_finds_both_blobs():
    blobs = lesion_components(two_blob_mask())
    assert len(blobs) == 2
    assert [b["voxels"] for b in blobs] == [16, 36]
    first = min(blobs, key=lambda b: b["centroid"][0])
    assert first["centroid"] == pytest.approx((6.5, 6.5))
    assert first["bbox"] == (5, 5, 8, 8)


def test_lesion_centers_filters_small_blobs():
    mask = two_blob_mask()
    assert len(lesion_centers(mask)) == 2
    assert len(lesion_centers(mask, min_voxels=20)) == 1
    with pytest.raises(ValueError):
        lesion_components(np.zeros((4, 4, 4), dtype=bool))


def test_positive_centers_land_on_lesion():
    mask = two_blob_mask()
    rng = np.random.default_rng(0)
    centers = positive_centers(mask, rng, 20)
    assert len(centers) == 20
    assert all(mask[r, c] for r, c in centers)
    with pytest.raises(ValueError):
        positive_centers(np.zeros((8, 8), dtype=bool), rng, 2)


def test_negative_centers_respect_min_distance():
    mask = two_blob_mask((64, 64))
    rng = np.random.default_rng(1)
    centers = negative_centers(mask, rng, 25, patch_size=8, min_distance=10)
    assert len(centers) == 25
    rows, cols = np.nonzero(mask)
    for r, c in centers:
        d = np.sqrt((rows - r) ** 2 + (cols - c) ** 2).min()
        assert d >= 10


def test_negative_centers_raise_when_lesion_covers_everything():
    mask = np.ones((16, 16), dtype=bool)
    rng = np.random.default_rng(2)
    with pytest.raises(RuntimeError):
        negative_centers(mask, rng, 5, patch_size=4, max_tries=50)


def test_patch_validation():
    with pytest.raises(ValueError):
        Patch(image=np.zeros((4, 5), dtype=np.float32), label=0, center=(0, 0))
    with pytest.raises(ValueError):
        Patch(image=np.zeros((4, 4, 4), dtype=np.float32), label=0, center=(0, 0))
    with pytest.raises(ValueError):
        Patch(image=np.zeros((4, 4), dtype=np.float32), label=2, center=(0, 0))


def test_sampler_balance_is_exact():
    mask = two_blob_mask((48, 48))
    image = np.random.default_rng(3).random((48, 48)).astype(np.float32)
    sampler = PatchSampler(image, mask, patch_size=8, seed=4, positive_fraction=0.5)
    patches = sampler.sample(20)
    counts = class_counts(patches)
    assert counts == {0: 10, 1: 10}
    assert all(p.image.shape == (8, 8) for p in patches)


def test_sampler_positive_only_without_lesion_raises():
    image = np.zeros((32, 32), dtype=np.float32)
    mask = np.zeros((32, 32), dtype=bool)
    sampler = PatchSampler(image, mask, patch_size=8, seed=0)
    assert not sampler.has_lesion
    patches = sampler.sample(4)
    assert class_counts(patches) == {0: 4, 1: 0}
    with pytest.raises(ValueError):
        sampler.sample_positive(2)


def test_sampler_rejects_mismatched_shapes_and_bad_fraction():
    with pytest.raises(ValueError):
        PatchSampler(np.zeros((8, 8)), np.zeros((8, 9), dtype=bool), 4)
    with pytest.raises(ValueError):
        PatchSampler(np.zeros((8, 8)), np.zeros((8, 8), dtype=bool), 4, positive_fraction=1.5)


def test_balance_patches_upsamples_positives():
    pos = [Patch(np.ones((4, 4), dtype=np.float32), 1, (0, 0)) for _ in range(2)]
    neg = [Patch(np.zeros((4, 4), dtype=np.float32), 0, (0, 0)) for _ in range(10)]
    balanced = balance_patches(pos + neg, seed=0, ratio=1.0)
    counts = class_counts(balanced)
    assert counts[0] == 10 and counts[1] == 10
    with pytest.raises(ValueError):
        balance_patches(pos + neg, ratio=0.0)


def test_split_patches_is_deterministic_and_disjoint():
    patches = [Patch(np.zeros((4, 4), dtype=np.float32), i % 2, (i, i)) for i in range(20)]
    train_a, val_a = split_patches(patches, 0.25, seed=5)
    train_b, val_b = split_patches(patches, 0.25, seed=5)
    assert len(val_a) == 5 and len(train_a) == 15
    assert [p.center for p in val_a] == [p.center for p in val_b]
    assert not {p.center for p in train_a} & {p.center for p in val_a}
    with pytest.raises(ValueError):
        split_patches(patches, 0.0)


def test_patch_array_shapes_and_labels():
    patches = [Patch(np.full((4, 4), 1.0, dtype=np.float32), 1, (0, 0)) for _ in range(3)]
    images, labels = patch_array(patches)
    assert images.shape == (3, 1, 4, 4)
    assert labels.tolist() == [1.0, 1.0, 1.0]
    with pytest.raises(ValueError):
        patch_array([])
