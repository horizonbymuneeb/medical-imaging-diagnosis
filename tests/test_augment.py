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
