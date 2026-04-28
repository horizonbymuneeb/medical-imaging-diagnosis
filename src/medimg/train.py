"""Training loop with early stopping.

The loop is written out rather than wrapped in a framework helper because the
project needs two things that are easy to get wrong: a validation metric that
is not the training loss, and early stopping that restores the best weights
instead of the last ones.  Both are asserted in the tests.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.medimg.dataset import build_specs, make_dataloaders
from src.medimg.model import SmallCNN, make_model, predict_proba
