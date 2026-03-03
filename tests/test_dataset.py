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
