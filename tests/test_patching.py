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
