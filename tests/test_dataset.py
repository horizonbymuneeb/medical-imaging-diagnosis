"""Synthetic dataset tests: shapes, determinism and class balance."""

import numpy as np
import pytest
import torch

from src.medimg.augment import AugmentConfig
from src.medimg.dataset import (
    SyntheticPatchDataset,
    SyntheticSpec,
    arrays_from_dataset,
    build_specs,
    collate_patches,
    dataset_statistics,
    make_dataloaders,
    patch_label_counts,
    slice_for_spec,
)
from src.medimg.phantom import PhantomConfig


def small_cfg() -> PhantomConfig:
    return PhantomConfig(shape=(32, 32, 24), spacing=(1.0, 1.0, 2.0), noise_sigma=5.0, seed=1)


def test_spec_validation():
    with pytest.raises(ValueError):
        SyntheticSpec(index=-1, seed=0)
    with pytest.raises(ValueError):
        SyntheticSpec(index=0, seed=0, positive_fraction=2.0)
    with pytest.raises(ValueError):
        SyntheticSpec(index=0, seed=0, axis=3)


def test_build_specs_is_deterministic():
    a = build_specs(5, seed=42)
    b = build_specs(5, seed=42)
    c = build_specs(5, seed=43)
    assert [s.seed for s in a] == [s.seed for s in b]
    assert [s.seed for s in a] != [s.seed for s in c]
    assert [s.index for s in a] == list(range(5))
    with pytest.raises(ValueError):
        build_specs(0)


def test_slice_for_spec_is_windowed_and_contains_lesion():
    image, mask = slice_for_spec(SyntheticSpec(0, seed=3), small_cfg())
    assert image.shape == (32, 32)
    assert image.dtype == np.float32
    assert 0.0 <= float(image.min()) and float(image.max()) <= 1.0
    assert mask.dtype == np.bool_
    assert mask.sum() > 0


def test_dataset_lengths_and_item_shapes():
    specs = build_specs(4, seed=7)
    ds = SyntheticPatchDataset(
        specs,
        patch_size=16,
        patches_per_spec=2,
        cfg=small_cfg(),
        augment=None,
        seed=0,
    )
    assert len(ds) == 8
    item = ds[0]
    assert item["image"].shape == (1, 16, 16)
    assert item["image"].dtype == torch.float32
    assert item["label"].shape == ()
    assert float(item["label"]) in (0.0, 1.0)
    with pytest.raises(ValueError):
        SyntheticPatchDataset([], patch_size=16)
    with pytest.raises(ValueError):
        SyntheticPatchDataset(specs, patch_size=16, patches_per_spec=0)


def test_dataset_is_deterministic_with_fixed_seed():
    specs = build_specs(3, seed=11)
    kwargs = dict(patch_size=16, cfg=small_cfg(), augment=None, seed=5)
    a = SyntheticPatchDataset(specs, **kwargs)
    b = SyntheticPatchDataset(specs, **kwargs)
    for i in range(len(a)):
        assert torch.equal(a[i]["image"], b[i]["image"])
        assert float(a[i]["label"]) == float(b[i]["label"])
