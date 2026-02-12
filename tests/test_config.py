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
