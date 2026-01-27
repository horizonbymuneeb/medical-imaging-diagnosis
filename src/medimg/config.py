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


def _iter_leaf_paths(config: Mapping[str, Any], prefix: str = "") -> Iterable[tuple[str, Any]]:
    for key, value in config.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            yield from _iter_leaf_paths(value, path)
        else:
            yield path, value


def validate_config(config: Mapping[str, Any]) -> None:
    """Raise :class:`ConfigError` listing every problem found at once."""
    problems: list[str] = []
    if not isinstance(config, Mapping):
        raise ConfigError("config must be a mapping")
    for section in REQUIRED_SECTIONS:
        if section not in config:
            problems.append(f"missing section {section!r}")
        elif not isinstance(config[section], Mapping):
            problems.append(f"section {section!r} must be a mapping")
    leaves = dict(_iter_leaf_paths(config))
    for key, (expected, low, high) in NUMERIC_BOUNDS.items():
        if key not in leaves:
            problems.append(f"missing key {key!r}")
            continue
        value = leaves[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            problems.append(f"{key} must be numeric, got {type(value).__name__}")
            continue
        if not low <= float(value) <= high:
            problems.append(f"{key}={value} outside [{low}, {high}]")
        if expected is int and float(value) != int(value):
            problems.append(f"{key} must be an integer, got {value}")
    shape = leaves.get("phantom.shape")
    if shape is not None:
        if not isinstance(shape, (list, tuple)) or len(shape) != 3:
            problems.append("phantom.shape must be a list of three ints")
        elif any(int(v) <= 0 for v in shape):
            problems.append("phantom.shape entries must be positive")
    spacing = leaves.get("phantom.spacing")
    if spacing is not None:
        if not isinstance(spacing, (list, tuple)) or len(spacing) != 3:
            problems.append("phantom.spacing must be a list of three floats")
        elif any(float(v) <= 0 for v in spacing):
            problems.append("phantom.spacing entries must be positive")
    preset = leaves.get("windowing.preset")
    if preset is not None:
        from src.medimg.windowing import WINDOW_PRESETS

        if preset not in WINDOW_PRESETS:
            problems.append(f"windowing.preset={preset!r} not in {sorted(WINDOW_PRESETS)}")
    normalise = leaves.get("windowing.normalize")
    if normalise is not None:
        from src.medimg.windowing import NORMALIZERS

        if normalise not in NORMALIZERS:
            problems.append(f"windowing.normalize={normalise!r} not in {sorted(NORMALIZERS)}")
    if problems:
        raise ConfigError("invalid config:\n  - " + "\n  - ".join(problems))


def resolve_paths(config: Mapping[str, Any], root: str | Path) -> dict[str, Any]:
    """Turn relative ``paths.*`` entries into absolute paths under ``root``."""
    root = Path(root)
    resolved = dict(config)
    paths = dict(resolved.get("paths", {}) or {})
    for key, value in list(paths.items()):
        candidate = Path(str(value))
        paths[key] = str(candidate if candidate.is_absolute() else (root / candidate).resolve())
    if paths:
        resolved["paths"] = paths
    return resolved
