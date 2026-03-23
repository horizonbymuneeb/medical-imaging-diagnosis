"""Demo script tests: it must run end-to-end from the repository config."""

import json
from pathlib import Path

import pytest

from src.medimg.config import load_config
from src.medimg.demo import (
    describe_metrics,
    describe_patches,
    describe_phantom,
    describe_window,
    main,
    phantom_section,
    run_demo,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def demo_config() -> dict:
    return {
        "phantom": {
            "shape": [32, 32, 24],
            "spacing": [1.0, 1.0, 2.0],
            "lesion_radius_mm": 5.0,
            "noise_sigma": 5.0,
            "seed": 2,
        },
        "windowing": {"preset": "brain", "normalize": "minmax"},
        "patching": {"patch_size": 16, "positive_fraction": 0.5, "patches_per_epoch": 8},
        "augment": {},
        "model": {"in_channels": 1, "base_channels": 8, "dropout": 0.1},
        "train": {"epochs": 1, "batch_size": 4, "lr": 0.001, "patience": 0, "val_fraction": 0.25},
    }


def test_phantom_section_matches_config():
    cfg = phantom_section(demo_config())
    assert cfg.shape == (32, 32, 24)
    assert cfg.lesion_radius_mm == pytest.approx(5.0)
    assert cfg.noise_sigma == pytest.approx(5.0)


def test_describe_phantom_reports_lesion():
    report = describe_phantom(demo_config())
    assert report["voxels"] == 32 * 32 * 24
    assert report["lesion_voxels"] > 0
