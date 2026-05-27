"""Phantom geometry and intensity tests.

These assert the physical properties a reviewer would check by hand: the
ellipsoid mask encloses the right volume, the skull ring is bone-bright, the
lesion mask matches the painted voxels, and two runs with the same seed are
bit-identical.
"""

import numpy as np
import pytest

from src.medimg.phantom import (
    HU_AIR,
    HU_BONE,
    HU_LESION,
    Lesion,
    PhantomConfig,
    add_gaussian_noise,
    add_rician_noise,
    brain_phantom,
    draw_lesion,
    ellipsoid_mask,
    head_center,
    lesion_slice_index,
    make_phantom_pair,
    phantom_slice,
    phantom_summary,
    random_lesion_center,
    shell_mask,
    voxel_grid,
)
