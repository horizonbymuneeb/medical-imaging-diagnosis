"""Configuration loading and validation.

``config.yaml`` is the only place experiment tunables live; the code reads it
through this module so a typo in a key fails loudly at startup instead of
silently training with defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

REQUIRED_SECTIONS = ("phantom", "windowing", "patching", "augment", "model", "train")

# key -> (type, minimum, maximum)
NUMERIC_BOUNDS: dict[str, tuple[type, float, float]] = {
    "phantom.noise_sigma": (float, 0.0, 5000.0),
    "phantom.lesion_radius_mm": (float, 0.5, 200.0),
    "patching.patch_size": (int, 8, 256),
    "patching.positive_fraction": (float, 0.0, 1.0),
    "train.epochs": (int, 1, 10000),
    "train.batch_size": (int, 1, 4096),
    "train.lr": (float, 1e-8, 10.0),
    "train.patience": (int, 0, 1000),
    "train.val_fraction": (float, 0.01, 0.99),
    "model.base_channels": (int, 1, 1024),
    "model.dropout": (float, 0.0, 0.99),
}


class ConfigError(ValueError):
    """Raised when ``config.yaml`` is missing keys or has out-of-range values."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Read and validate a YAML config file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no config at {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    validate_config(data)
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_path(config: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    """Read ``a.b.c`` from nested mappings, returning ``default`` if absent."""
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return node
