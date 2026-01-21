"""Calibration tests: reliability bins, ECE, Brier and temperature scaling."""

import numpy as np
import pytest

from src.medimg.calibration import (
    CalibrationBin,
    apply_temperature,
    brier_score,
    calibrate_probabilities,
    calibration_report,
    expected_calibration_error,
    fit_temperature,
    maximum_calibration_error,
    reliability_curve,
)
