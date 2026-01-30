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


def test_ece_zero_for_calibrated_predictions():
    probs = np.full(100, 0.5)
    labels = np.zeros(100)
    labels[:50] = 1.0
    assert expected_calibration_error(probs, labels, 10) == pytest.approx(0.0, abs=1e-9)


def test_ece_positive_for_overconfident_predictions():
    probs = np.full(100, 0.99)
    labels = np.zeros(100)
    labels[:50] = 1.0
    ece = expected_calibration_error(probs, labels, 10)
    assert ece == pytest.approx(0.49, abs=1e-6)
    assert maximum_calibration_error(probs, labels, 10) == pytest.approx(0.49, abs=1e-6)


def test_brier_score_exact():
    probs = np.array([0.0, 1.0, 0.5])
    labels = np.array([0.0, 1.0, 0.0])
    assert brier_score(probs, labels) == pytest.approx((0.0 + 0.0 + 0.25) / 3)


def test_apply_temperature_validation_and_monotonicity():
    logits = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert apply_temperature(logits, 1.0)[2] == pytest.approx(0.5)
    hot = apply_temperature(logits, 4.0)
    assert np.all(np.diff(hot) > 0)
    assert hot.max() < apply_temperature(logits, 0.5).max()
    with pytest.raises(ValueError):
        apply_temperature(logits, 0.0)


def test_fit_temperature_improves_nll_and_preserves_ranking():
    rng = np.random.default_rng(0)
    labels = (rng.random(200) < 0.4).astype(np.float64)
    logits = rng.normal(0.0, 1.0, size=200) + labels * 2.0
    fit = fit_temperature(logits, labels)
    assert 0.05 <= fit["temperature"] <= 10.0
    assert fit["nll_after"] <= fit["nll_before"] + 1e-9
    # Temperature scaling is monotone, so the ranking (and AUC) is unchanged.
    from src.medimg.metrics import auc_score

    assert auc_score(labels, logits) == pytest.approx(
        auc_score(labels, apply_temperature(logits, fit["temperature"]))
    )


def test_fit_temperature_validation():
    with pytest.raises(ValueError):
        fit_temperature(np.array([0.0, 1.0]), np.array([0.0]))
    with pytest.raises(ValueError):
        fit_temperature(np.array([]), np.array([]))
    with pytest.raises(ValueError):
        fit_temperature(np.array([0.0, 1.0]), np.array([0.0, 1.0]), bounds=(0.0, 2.0))


def test_calibration_report_shape():
    probs = np.array([0.1, 0.4, 0.6, 0.9])
    labels = np.array([0, 0, 1, 1])
    report = calibration_report(probs, labels, n_bins=4)
    assert report["n"] == 4
    assert report["base_rate"] == pytest.approx(0.5)
    assert 0.0 <= report["ece"] <= 1.0
    assert len(report["bins"]) == 4
    assert set(report["bins"][0]) == {"lower", "upper", "count", "mean_confidence", "accuracy", "gap"}


def test_calibrate_probabilities_returns_before_and_after():
    rng = np.random.default_rng(1)
    labels = (rng.random(120) < 0.5).astype(np.float64)
    logits = rng.normal(0.0, 2.0, size=120)
    out = calibrate_probabilities(logits, labels)
    assert out["temperature"] > 0
    assert out["ece_before"] >= 0 and out["ece_after"] >= 0
    assert out["brier_after"] <= out["brier_before"] + 0.05


def test_calibration_input_validation():
    with pytest.raises(ValueError):
        expected_calibration_error(np.array([1.5]), np.array([1]))
    with pytest.raises(ValueError):
        expected_calibration_error(np.array([0.5, 0.5]), np.array([1]))
    with pytest.raises(ValueError):
        reliability_curve(np.array([0.5]), np.array([2]), n_bins=4)
    with pytest.raises(ValueError):
        reliability_curve(np.array([0.5]), np.array([1]), n_bins=0)
