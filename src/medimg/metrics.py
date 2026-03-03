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


def specificity_at_sensitivity(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_sensitivity: float,
) -> float:
    """Highest specificity reachable at or above a sensitivity target."""
    if not 0.0 < target_sensitivity <= 1.0:
        raise ValueError("target_sensitivity must be in (0, 1]")
    fpr, tpr, _ = roc_points(y_true, y_score)
    valid = tpr >= target_sensitivity - 1e-12
    if not valid.any():
        return 0.0
    return float(1.0 - fpr[valid].min())


def sensitivity_at_specificity(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_specificity: float,
) -> float:
    """Highest sensitivity reachable at or above a specificity target.

    Interpolates linearly between adjacent ROC points instead of snapping to a
    threshold, because with a small validation set the nearest threshold can be
    several specificity points away from the requested one.
    """
    if not 0.0 < target_specificity < 1.0:
        raise ValueError("target_specificity must be in (0, 1)")
    fpr, tpr, _ = roc_points(y_true, y_score)
    spec = 1.0 - fpr
    target_fpr = 1.0 - target_specificity
    if target_fpr <= fpr[0]:
        return float(tpr[0])
    for i in range(1, len(fpr)):
        if fpr[i] >= target_fpr:
            if fpr[i] == fpr[i - 1]:
                return float(tpr[i])
            w = (target_fpr - fpr[i - 1]) / (fpr[i] - fpr[i - 1])
            return float(tpr[i - 1] + w * (tpr[i] - tpr[i - 1]))
    return float(tpr[-1])


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[int] = (0, 1),
) -> np.ndarray:
    """Confusion matrix with ``rows = true`` and ``cols = predicted``."""
    y = np.asarray(y_true).astype(np.int64).ravel()
    p = np.asarray(y_pred).astype(np.int64).ravel()
    if y.shape != p.shape:
        raise ValueError("y_true and y_pred must match")
    labels = tuple(int(l) for l in labels)
    if not set(np.unique(y)) <= set(labels) or not set(np.unique(p)) <= set(labels):
        raise ValueError(f"values outside labels {labels}")
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for t, q in zip(y, p):
        matrix[index[int(t)], index[int(q)]] += 1
    return matrix


def per_class_metrics(
    matrix: np.ndarray,
    labels: Sequence[int] = (0, 1),
    *,
    eps: float = 1e-12,
) -> dict[int, dict[str, float]]:
    """Precision, recall, F1 and support from a confusion matrix."""
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if matrix.shape[0] != len(labels):
        raise ValueError("labels must match the matrix size")
    out: dict[int, dict[str, float]] = {}
    total = float(matrix.sum())
    for i, label in enumerate(labels):
        tp = float(matrix[i, i])
        fn = float(matrix[i, :].sum() - tp)
        fp = float(matrix[:, i].sum() - tp)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        out[int(label)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(matrix[i, :].sum()),
            "prevalence": float(matrix[i, :].sum() / total) if total else 0.0,
        }
    return out


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Binary precision/recall/F1 for the positive class."""
    matrix = confusion_matrix(y_true, y_pred, labels=(0, 1))
    metrics = per_class_metrics(matrix, labels=(0, 1))[1]
    return {
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "support": metrics["support"],
    }


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true).ravel()
    p = np.asarray(y_pred).ravel()
    if y.shape != p.shape:
        raise ValueError("y_true and y_pred must match")
    if y.size == 0:
        raise ValueError("empty input")
    return float((y == p).mean())


def threshold_sweep(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: Iterable[float] | None = None,
) -> list[dict[str, float]]:
    """Metrics at each threshold; defaults to 0.05 .. 0.95 in steps of 0.05."""
    y, s = _check_pair(y_true, y_score)
    if thresholds is None:
        thresholds = np.arange(0.05, 1.0, 0.05)
    rows = []
    for threshold in thresholds:
        pred = (s >= float(threshold)).astype(np.int64)
        row = precision_recall_f1(y, pred)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["f1"],
                "accuracy": accuracy_score(y, pred),
            }
        )
    return rows


def best_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    criterion: str = "youden",
) -> float:
    """Pick a threshold that maximises Youden's J or F1."""
    if criterion not in {"youden", "f1"}:
        raise ValueError("criterion must be 'youden' or 'f1'")
    y, s = _check_pair(y_true, y_score)
    fpr, tpr, thresholds = roc_points(y, s)
    if criterion == "youden":
        scores = tpr - fpr
    else:
        positives = y.sum()
        precision = tpr * positives / np.maximum((tpr * positives) + (fpr * (len(y) - positives)), 1e-12)
        scores = 2 * precision * tpr / np.maximum(precision + tpr, 1e-12)
    best = int(np.argmax(scores))
    return float(thresholds[best])


def classification_report_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[int] = (0, 1),
) -> dict:
    """A JSON-friendly classification report."""
    matrix = confusion_matrix(y_true, y_pred, labels)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "confusion_matrix": matrix.tolist(),
        "per_class": {str(k): v for k, v in per_class_metrics(matrix, labels).items()},
        "support": float(matrix.sum()),
    }
