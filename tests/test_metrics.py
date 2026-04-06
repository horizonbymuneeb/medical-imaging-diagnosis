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
