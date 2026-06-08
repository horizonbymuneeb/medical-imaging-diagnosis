"""Volume metadata, serialisation and geometry tests."""

import numpy as np
import pytest

from src.medimg.volume import (
    Volume,
    as_volume,
    crop_or_pad,
    iter_slices,
    load_volume,
    nonzero_bounds,
    resample_to_spacing,
    save_volume,
    take_slice,
    validate_volume,
    volume_stats,
)
