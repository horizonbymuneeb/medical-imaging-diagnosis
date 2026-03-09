"""The classifier: a small CNN sized for 32x32 single-channel patches.

Two million parameters would be a liability on a few hundred synthetic patches,
so the network is deliberately tiny (three conv blocks, global average pooling,
one logit).  It is also fast enough that the test suite can train it for a
couple of epochs on CPU and assert that the loss actually went down.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
