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


def test_calibration_bin_gap():
    b = CalibrationBin(lower=0.0, upper=0.5, count=10, mean_confidence=0.2, accuracy=0.3)
    assert b.gap == pytest.approx(0.1)
    assert b.to_dict()["count"] == 10


def test_reliability_curve_bins_and_counts():
    probs = np.array([0.05, 0.15, 0.85, 0.95])
    labels = np.array([0, 0, 1, 1])
    bins = reliability_curve(probs, labels, n_bins=4)
    assert len(bins) == 2  # empty bins are skipped
    assert sum(b.count for b in bins) == 4
    assert bins[0].mean_confidence == pytest.approx(0.10)
    assert bins[0].accuracy == pytest.approx(0.0)
    assert bins[-1].accuracy == pytest.approx(1.0)


def test_reliability_curve_skips_empty_bins():
    probs = np.array([0.01, 0.02])
    labels = np.array([0, 0])
    bins = reliability_curve(probs, labels, n_bins=10)
    assert len(bins) == 1
    assert bins[0].count == 2
