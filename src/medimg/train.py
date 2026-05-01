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


@dataclass(frozen=True)
class TrainConfig:
    """Optimiser and schedule settings for :func:`fit`."""

    epochs: int = 6
    batch_size: int = 16
    lr: float = 3e-3
    weight_decay: float = 1e-4
    patience: int = 2
    min_delta: float = 1e-3
    seed: int = 11
    val_fraction: float = 0.25
    pos_weight: float = 1.0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.patience < 0:
            raise ValueError("patience must be non-negative")
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError("val_fraction must be in (0, 1)")
        if self.pos_weight <= 0:
            raise ValueError("pos_weight must be positive")
        if self.device != "cpu":
            raise ValueError("this project only trains on CPU")


class EarlyStopping:
    """Stop when the monitored loss stops improving.

    ``mode='min'`` for losses.  The best state dict is kept in memory so
    ``restore`` can put the model back to the best epoch, which is the whole
    point of early stopping; stopping on the last epoch is not.
    """

    def __init__(self, patience: int = 2, min_delta: float = 1e-3, mode: str = "min") -> None:
        if patience < 0:
            raise ValueError("patience must be non-negative")
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.best = float("inf") if mode == "min" else float("-inf")
        self.best_epoch = -1
        self.counter = 0
        self.best_state: dict | None = None
        self.stopped = False

    def is_improvement(self, value: float) -> bool:
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def step(self, value: float, epoch: int, model: nn.Module | None = None) -> bool:
        """Record ``value``; return ``True`` when training should stop."""
        if self.is_improvement(value):
            self.best = float(value)
            self.best_epoch = int(epoch)
            self.counter = 0
            if model is not None:
                self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter > self.patience:
                self.stopped = True
        return self.stopped

    def restore(self, model: nn.Module) -> bool:
        """Load the best weights back into ``model`` if any were recorded."""
        if self.best_state is None:
            return False
        model.load_state_dict(self.best_state)
        return True

    def state_dict(self) -> dict:
        return {
            "best": self.best,
            "best_epoch": self.best_epoch,
            "counter": self.counter,
            "stopped": self.stopped,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "mode": self.mode,
        }
