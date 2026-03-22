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
