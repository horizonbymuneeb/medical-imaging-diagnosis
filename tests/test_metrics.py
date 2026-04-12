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


def test_auc_requires_both_classes():
    with pytest.raises(ValueError):
        auc_score(np.ones(4, dtype=int), np.linspace(0, 1, 4))
    with pytest.raises(ValueError):
        auc_score(np.array([0, 1]), np.array([0.2, np.nan]))
    with pytest.raises(ValueError):
        auc_score(np.array([0, 1]), np.array([0.2]))
    with pytest.raises(ValueError):
        auc_score(np.array([0, 2]), np.array([0.2, 0.3]))


def test_sensitivity_at_specificity_handles_targets():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert sensitivity_at_specificity(labels, scores, 0.9) == pytest.approx(1.0)
    assert sensitivity_at_specificity(labels, scores, 0.5) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        sensitivity_at_specificity(labels, scores, 1.0)


def test_specificity_at_sensitivity_mirrors():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert specificity_at_sensitivity(labels, scores, 1.0) == pytest.approx(1.0)
    assert specificity_at_sensitivity(labels, scores, 0.5) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        specificity_at_sensitivity(labels, scores, 0.0)


def test_confusion_matrix_layout_is_true_by_predicted():
    y_true = [0, 0, 1, 1, 1]
    y_pred = [0, 1, 1, 1, 0]
    matrix = confusion_matrix(y_true, y_pred)
    assert matrix.tolist() == [[1, 1], [1, 2]]
    assert matrix.sum() == 5
    with pytest.raises(ValueError):
        confusion_matrix([0, 1], [0])
    with pytest.raises(ValueError):
        confusion_matrix([0, 2], [0, 1])


def test_per_class_metrics_exact_values():
    matrix = np.array([[1, 1], [1, 2]])
    metrics = per_class_metrics(matrix)
    assert metrics[0]["precision"] == pytest.approx(0.5)
    assert metrics[0]["recall"] == pytest.approx(0.5)
    assert metrics[0]["support"] == pytest.approx(2.0)
    assert metrics[1]["precision"] == pytest.approx(2 / 3)
    assert metrics[1]["recall"] == pytest.approx(2 / 3)
    assert metrics[1]["f1"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        per_class_metrics(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        per_class_metrics(matrix, labels=(0, 1, 2))


def test_precision_recall_f1_and_accuracy():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    result = precision_recall_f1(y_true, y_pred)
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(0.8)
    assert accuracy_score(y_true, y_pred) == pytest.approx(0.75)
    with pytest.raises(ValueError):
        accuracy_score([0, 1], [1])
