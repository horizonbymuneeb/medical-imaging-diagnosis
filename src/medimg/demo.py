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


def describe_patches(config: dict) -> dict:
    """Class balance of a balanced patch sample on the lesion slice."""
    cfg = phantom_section(config)
    patch_size = int(config.get("patching", {}).get("patch_size", 32))
    n = int(config.get("patching", {}).get("patches_per_epoch", 32))
    fraction = float(config.get("patching", {}).get("positive_fraction", 0.5))
    preset = config.get("windowing", {}).get("preset", "brain")
    volume, mask = make_phantom_pair(cfg)
    from src.medimg.phantom import lesion_slice_index

    index = lesion_slice_index(mask, axis=2)
    image = apply_preset(volume.data[:, :, index], preset)
    sampler = PatchSampler(image, mask[:, :, index], patch_size, seed=cfg.seed, positive_fraction=fraction)
    patches = sampler.sample(n)
    counts = class_counts(patches)
    return {
        "patch_size": patch_size,
        "requested": n,
        "counts": {str(k): v for k, v in counts.items()},
        "mean_intensity": float(np.mean([p.mean for p in patches])),
    }


def describe_metrics() -> dict:
    """Metrics on a tiny fixed prediction vector (arithmetic check, not a result)."""
    labels = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int64)
    scores = np.array([0.05, 0.35, 0.8, 0.65, 0.2, 0.9, 0.1, 0.55])
    pred = (scores >= 0.5).astype(np.int64)
    return {
        "auc": auc_score(labels, scores),
        "sens_at_90spec": sensitivity_at_specificity(labels, scores, 0.9),
        "confusion": confusion_matrix(labels, pred).tolist(),
        "calibration": calibration_report(scores, labels, n_bins=4),
    }


def run_demo(config: dict) -> dict:
    """Assemble every section into one report."""
    return {
        "phantom": describe_phantom(config),
        "window": describe_window(config),
        "patches": describe_patches(config),
        "metrics": describe_metrics(),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the phantom pipeline demo")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default=None, help="optional JSON output path")
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = run_demo(load_config(args.config))
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
