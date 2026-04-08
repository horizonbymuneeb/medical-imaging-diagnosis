"""Metric arithmetic tests.

Every expected value is computed by hand in the docstring or from a known
property (perfect ranking -> AUC 1.0, reversed ranking -> 0.0), so the tests
fail if the implementation drifts.
"""

import numpy as np
import pytest

from src.medimg.metrics import (
    accuracy_score,
    auc_score,
    best_threshold,
    classification_report_dict,
    confusion_matrix,
    per_class_metrics,
    precision_recall_f1,
    roc_points,
    sensitivity_at_specificity,
    specificity_at_sensitivity,
    threshold_sweep,
)


def test_auc_perfect_reversed_and_tied():
    labels = np.array([0, 0, 1, 1])
    assert auc_score(labels, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert auc_score(labels, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)
    assert auc_score(labels, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)


def test_auc_known_value():
    # 3 of 4 positive-negative pairs are ranked correctly -> AUC 0.75.
    labels = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.9, 0.8, 0.7])
    assert auc_score(labels, scores) == pytest.approx(0.75)


def test_roc_points_monotone_and_anchored():
    labels = np.array([0, 0, 1, 1, 1])
    scores = np.array([0.05, 0.4, 0.35, 0.8, 0.9])
    fpr, tpr, thresholds = roc_points(labels, scores)
    assert fpr[0] == 0.0 and tpr[0] == 0.0
    assert fpr[-1] == pytest.approx(1.0) and tpr[-1] == pytest.approx(1.0)
    assert np.all(np.diff(fpr) >= 0)
    assert np.all(np.diff(tpr) >= 0)
    assert np.isinf(thresholds[0])
