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
