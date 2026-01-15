"""Configuration loading and validation.

``config.yaml`` is the only place experiment tunables live; the code reads it
through this module so a typo in a key fails loudly at startup instead of
silently training with defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml
