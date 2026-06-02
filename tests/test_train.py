"""Training loop, early stopping and reproducibility tests.

The loop is exercised on a separable toy batch so the assertions are about
learning actually happening (loss down, probabilities separated), not about a
number copied from a log.
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.medimg.dataset import build_specs, make_dataloaders
from src.medimg.model import make_model, predict_proba
from src.medimg.phantom import PhantomConfig
from src.medimg.train import (
    EarlyStopping,
    TrainConfig,
    evaluate_epoch,
    fit,
    set_seed,
    train_one_epoch,
)


class ToyDataset(Dataset):
    """Bright 4x4 square = positive, dark corner = negative."""

    def __init__(self, n: int = 32) -> None:
        self.n = n
        self.images = torch.zeros(n, 1, 16, 16)
        self.labels = torch.zeros(n)
        for i in range(n):
            if i % 2 == 0:
                self.images[i, 0, 6:10, 6:10] = 1.0
                self.labels[i] = 1.0
            else:
                self.images[i, 0, :4, :4] = 0.05

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"image": self.images[index], "label": self.labels[index]}
