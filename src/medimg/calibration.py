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
