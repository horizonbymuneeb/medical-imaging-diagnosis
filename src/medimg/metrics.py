"""Classification metrics implemented from the definitions.

scikit-learn is available, but the metrics here are written out for two
reasons: the thresholds and tie handling are part of the project's design
decisions, and a reviewer can check the arithmetic.  Every function validates
its inputs, because silent metric bugs (single-class AUC, mismatched lengths)
are the kind that make a paper wrong.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def _check_pair(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true).astype(np.int64).ravel()
    s = np.asarray(y_score).astype(np.float64).ravel()
    if y.shape != s.shape:
        raise ValueError(f"y_true and y_score must match, got {y.shape} and {s.shape}")
    if y.size == 0:
        raise ValueError("empty input")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y_true must contain only 0 and 1")
    if not np.isfinite(s).all():
        raise ValueError("y_score contains NaN or inf")
    return y, s


def roc_points(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(fpr, tpr, thresholds)`` using every distinct score.

    Ties are handled by grouping equal scores, which is what makes the AUC
    identical to the rank-based Mann-Whitney estimate.
    """
    y, s = _check_pair(y_true, y_score)
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]
    distinct = np.where(np.diff(s_sorted))[0]
    threshold_idx = np.r_[distinct, y_sorted.size - 1]
    tps = np.cumsum(y_sorted)[threshold_idx]
    fps = (threshold_idx + 1) - tps
    positives = int(y.sum())
    negatives = int(y.size - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("ROC needs both classes present")
    tpr = np.r_[0.0, tps / positives]
    fpr = np.r_[0.0, fps / negatives]
    thresholds = np.r_[np.inf, s_sorted[threshold_idx]]
    return fpr, tpr, thresholds


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the ROC curve via the trapezoidal rule on the ROC points."""
    fpr, tpr, _ = roc_points(y_true, y_score)
    # np.trapz wants increasing x; fpr is non-decreasing already.
    return float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
