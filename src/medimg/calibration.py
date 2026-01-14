"""Probability calibration: reliability curves, ECE and temperature scaling.

A CNN trained with a balanced patch sampler outputs probabilities that are
confident relative to the true prevalence, so the numbers are not probabilities
yet.  This module measures that (reliability curve, ECE, Brier) and fixes it
with a single temperature parameter fitted on held-out logits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, special


@dataclass(frozen=True)
class CalibrationBin:
    """One bin of a reliability diagram."""

    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float

    @property
    def gap(self) -> float:
        """Signed calibration gap (accuracy minus confidence)."""
        return float(self.accuracy - self.mean_confidence)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "count": self.count,
            "mean_confidence": self.mean_confidence,
            "accuracy": self.accuracy,
            "gap": self.gap,
        }


def _check(probs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probs, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    if p.shape != y.shape:
        raise ValueError("probs and labels must match")
    if p.size == 0:
        raise ValueError("empty input")
    if not np.all((p >= 0.0) & (p <= 1.0)):
        raise ValueError("probs must be in [0, 1]")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("labels must be 0 or 1")
    return p, y


def reliability_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> list[CalibrationBin]:
    """Equal-width reliability diagram over ``[0, 1]``.

    Only non-empty bins are returned; an empty bin has no accuracy to report and
    averaging it in would bias the ECE towards zero.
    """
    p, y = _check(probs, labels)
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # right edge inclusive on the last bin so p == 1.0 lands somewhere.
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    bins: list[CalibrationBin] = []
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        bins.append(
            CalibrationBin(
                lower=float(edges[b]),
                upper=float(edges[b + 1]),
                count=count,
                mean_confidence=float(p[mask].mean()),
                accuracy=float(y[mask].mean()),
            )
        )
    return bins


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Weighted mean absolute gap between confidence and accuracy."""
    p, y = _check(probs, labels)
    bins = reliability_curve(p, y, n_bins)
    total = float(len(p))
    return float(sum(b.count / total * abs(b.gap) for b in bins))


def maximum_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Largest single-bin gap, which ECE can hide."""
    bins = reliability_curve(probs, labels, n_bins)
    if not bins:
        raise ValueError("no bins")
    return float(max(abs(b.gap) for b in bins))


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean squared error between probability and outcome."""
    p, y = _check(probs, labels)
    return float(np.mean((p - y) ** 2))


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Sigmoid probabilities after dividing the logits by ``temperature``."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = np.asarray(logits, dtype=np.float64).ravel()
    return special.expit(logits / float(temperature))


def _nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    probs = apply_temperature(logits, temperature)
    probs = np.clip(probs, 1e-12, 1 - 1e-12)
    return float(-np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs)))


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    bounds: tuple[float, float] = (0.05, 10.0),
) -> dict[str, float]:
    """Fit a single temperature by minimising held-out NLL.

    Temperature scaling has one parameter, so it cannot change the ranking of
    predictions: AUC stays identical and only the probabilities move.  The
    tests assert exactly that.
    """
    logits = np.asarray(logits, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel()
    if logits.shape != labels.shape:
        raise ValueError("logits and labels must match")
    if logits.size == 0:
        raise ValueError("empty input")
    if bounds[0] <= 0 or bounds[1] <= bounds[0]:
        raise ValueError("bounds must be positive and ordered")
    result = optimize.minimize_scalar(
        lambda t: _nll(logits, labels, float(t)),
        bounds=bounds,
        method="bounded",
        options={"xatol": 1e-4},
    )
    temperature = float(result.x)
    return {
        "temperature": temperature,
        "nll_before": _nll(logits, labels, 1.0),
        "nll_after": float(result.fun),
    }


def calibration_report(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Everything a reviewer wants on one screen."""
    p, y = _check(probs, labels)
    bins = reliability_curve(p, y, n_bins)
    return {
        "ece": expected_calibration_error(p, y, n_bins),
        "mce": maximum_calibration_error(p, y, n_bins),
        "brier": brier_score(p, y),
        "n": int(len(p)),
        "mean_confidence": float(p.mean()),
        "base_rate": float(y.mean()),
        "bins": [b.to_dict() for b in bins],
    }


def calibrate_probabilities(
    logits: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Fit a temperature and report the ECE before and after."""
    fit = fit_temperature(logits, labels)
    before = apply_temperature(logits, 1.0)
    after = apply_temperature(logits, fit["temperature"])
    return {
        "temperature": fit["temperature"],
        "ece_before": expected_calibration_error(before, labels, n_bins),
        "ece_after": expected_calibration_error(after, labels, n_bins),
        "brier_before": brier_score(before, labels),
        "brier_after": brier_score(after, labels),
    }
