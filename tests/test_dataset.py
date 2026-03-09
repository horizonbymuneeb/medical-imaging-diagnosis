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
