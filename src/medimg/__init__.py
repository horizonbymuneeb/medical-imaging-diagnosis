"""CPU-only CT diagnosis pipeline built on synthetic phantom volumes.

The package is deliberately dependency-light: every stage works on plain NumPy
arrays (plus PyTorch for the classifier) so the whole pipeline can be exercised
without a scanner, a DICOM stack or a GPU.
"""

from src.medimg.calibration import (
    CalibrationBin,
    brier_score,
    expected_calibration_error,
    fit_temperature,
    reliability_curve,
)
from src.medimg.metrics import (
    auc_score,
    confusion_matrix,
    per_class_metrics,
    sensitivity_at_specificity,
)
from src.medimg.phantom import PhantomConfig, make_phantom_pair
from src.medimg.volume import Volume
from src.medimg.windowing import WINDOW_PRESETS, apply_window

__all__ = [
    "CalibrationBin",
    "PhantomConfig",
    "Volume",
    "WINDOW_PRESETS",
    "apply_window",
    "auc_score",
    "brier_score",
    "confusion_matrix",
    "expected_calibration_error",
    "fit_temperature",
    "make_phantom_pair",
    "per_class_metrics",
    "reliability_curve",
    "sensitivity_at_specificity",
]
