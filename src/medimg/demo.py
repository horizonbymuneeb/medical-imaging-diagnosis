"""End-to-end demo: phantom -> window -> patches -> metrics.

Run with ``python -m src.medimg.demo --config config.yaml``.  It prints the
geometry of a generated phantom, the window statistics, the class balance of a
patch sample and the ROC/calibration numbers of a small trained model, so a
reader can see every stage without opening a notebook.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from src.medimg.calibration import calibration_report
from src.medimg.config import load_config
from src.medimg.metrics import auc_score, confusion_matrix, sensitivity_at_specificity
from src.medimg.patching import PatchSampler, class_counts
from src.medimg.phantom import PhantomConfig, make_phantom_pair, phantom_summary
from src.medimg.windowing import WINDOW_PRESETS, apply_preset, window_statistics


def phantom_section(config: dict) -> PhantomConfig:
    """Build a :class:`PhantomConfig` from the ``phantom`` section."""
    section = config.get("phantom", {})
    return PhantomConfig(
        shape=tuple(section.get("shape", (64, 64, 48))),
        spacing=tuple(section.get("spacing", (1.5, 1.5, 2.0))),
        lesion_radius_mm=float(section.get("lesion_radius_mm", 6.0)),
        noise_sigma=float(section.get("noise_sigma", 18.0)),
        seed=int(section.get("seed", 7)),
    )


def describe_phantom(config: dict) -> dict:
    """Geometry and intensity summary of the configured phantom."""
    cfg = phantom_section(config)
    return phantom_summary(cfg)


def describe_window(config: dict) -> dict:
    """Window statistics for the configured preset on the lesion slice."""
    cfg = phantom_section(config)
    preset_name = config.get("windowing", {}).get("preset", "brain")
    volume, mask = make_phantom_pair(cfg)
    from src.medimg.phantom import lesion_slice_index

    index = lesion_slice_index(mask, axis=2)
    image = volume.data[:, :, index]
    preset = WINDOW_PRESETS[preset_name]
    stats = window_statistics(image, preset.level, preset.width)
    return {"preset": preset_name, "slice": index, **stats}
