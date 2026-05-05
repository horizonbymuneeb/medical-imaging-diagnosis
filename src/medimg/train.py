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


def set_seed(seed: int) -> None:
    """Seed NumPy and Torch so a run is reproducible on CPU."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def _to_device(batch: dict[str, torch.Tensor], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    return batch["image"].to(device), batch["label"].to(device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str = "cpu",
) -> float:
    """One pass over ``loader``; returns the mean batch loss."""
    model.train()
    total = 0.0
    n_batches = 0
    for batch in loader:
        images, labels = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total += float(loss.item())
        n_batches += 1
    if n_batches == 0:
        raise ValueError("loader produced no batches")
    return total / n_batches


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module | None = None,
    device: str = "cpu",
) -> dict[str, np.ndarray | float]:
    """Validation loss plus probabilities and labels for metric computation."""
    model.eval()
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    total = 0.0
    n_batches = 0
    for batch in loader:
        images, target = _to_device(batch, device)
        logits = model(images)
        if criterion is not None:
            total += float(criterion(logits, target).item())
            n_batches += 1
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(target.cpu().numpy())
    return {
        "loss": (total / n_batches) if n_batches else float("nan"),
        "probs": np.concatenate(probs) if probs else np.empty(0),
        "labels": np.concatenate(labels) if labels else np.empty(0),
    }


def fit(
    model: SmallCNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    *,
    verbose: bool = False,
) -> dict:
    """Train with early stopping and return a history dictionary.

    The returned ``history`` contains per-epoch losses, the best epoch, whether
    stopping triggered, and the restored validation probabilities so the caller
    can compute AUC/calibration on the exact weights that were kept.
    """
    set_seed(cfg.seed)
    device = cfg.device
    model.to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([cfg.pos_weight], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    stopper = EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta, mode="min")
    history: dict = {
        "train_loss": [],
        "val_loss": [],
        "best_epoch": -1,
        "stopped_early": False,
        "epochs_run": 0,
        "val_probs": np.empty(0),
        "val_labels": np.empty(0),
    }
    start = time.time()
    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate_epoch(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(float(metrics["loss"]))
        history["epochs_run"] = epoch + 1
        if verbose:
            print(f"epoch {epoch + 1:02d} train={train_loss:.4f} val={metrics['loss']:.4f}")
        stop = stopper.step(float(metrics["loss"]), epoch, model)
        if stop:
            history["stopped_early"] = True
            break
    history["best_epoch"] = stopper.best_epoch
    history["best_val_loss"] = stopper.best
    stopper.restore(model)
    final = evaluate_epoch(model, val_loader, criterion, device)
    history["val_probs"] = final["probs"]
    history["val_labels"] = final["labels"]
    history["seconds"] = time.time() - start
    history["early_stopping"] = stopper.state_dict()
    return history
