"""Config loading/validation and package-level tests."""

from pathlib import Path

import pytest

from src.medimg.config import (
    ConfigError,
    config_fingerprint,
    deep_merge,
    get_path,
    load_config,
    resolve_paths,
    validate_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def valid_config() -> dict:
    return {
        "phantom": {
            "shape": [32, 32, 24],
            "spacing": [1.0, 1.0, 2.0],
            "lesion_radius_mm": 5.0,
            "noise_sigma": 10.0,
            "seed": 1,
        },
        "windowing": {"preset": "brain", "normalize": "minmax"},
        "patching": {"patch_size": 16, "positive_fraction": 0.5, "patches_per_epoch": 8},
        "augment": {"p_flip": 0.5},
        "model": {"in_channels": 1, "base_channels": 8, "dropout": 0.1},
        "train": {
            "epochs": 2,
            "batch_size": 4,
            "lr": 0.001,
            "weight_decay": 0.0,
            "patience": 1,
            "min_delta": 0.0,
            "seed": 1,
            "val_fraction": 0.25,
        },
    }


def test_repo_config_yaml_is_valid():
    config = load_config(REPO_ROOT / "config.yaml")
    assert config["windowing"]["preset"] in {"brain", "lung", "abdomen"}
    assert config["patching"]["patch_size"] == 32


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config(REPO_ROOT / "no-such-config.yaml")


def test_validate_config_lists_all_problems():
    with pytest.raises(ConfigError) as excinfo:
        validate_config({})
    message = str(excinfo.value)
    for section in ("phantom", "windowing", "patching", "train"):
        assert section in message


def test_validate_config_rejects_bad_values():
    config = valid_config()
    config["train"]["epochs"] = 0
    config["patching"]["positive_fraction"] = 2.0
    config["windowing"]["preset"] = "chest"
    with pytest.raises(ConfigError) as excinfo:
        validate_config(config)
    message = str(excinfo.value)
    assert "train.epochs" in message
    assert "patching.positive_fraction" in message
    assert "windowing.preset" in message


def test_validate_config_rejects_non_numeric_and_bad_shape():
    config = valid_config()
    config["train"]["lr"] = "fast"
    config["phantom"]["shape"] = [32, 32]
    config["phantom"]["spacing"] = [1.0, -1.0, 1.0]
    with pytest.raises(ConfigError) as excinfo:
        validate_config(config)
    message = str(excinfo.value)
    assert "train.lr" in message
    assert "phantom.shape" in message
    assert "phantom.spacing" in message
