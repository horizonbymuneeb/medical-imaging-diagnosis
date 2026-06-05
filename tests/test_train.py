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


def toy_loader(n: int = 32, batch_size: int = 8) -> DataLoader:
    return DataLoader(ToyDataset(n), batch_size=batch_size, shuffle=False)


def test_train_config_validation():
    TrainConfig()
    with pytest.raises(ValueError):
        TrainConfig(epochs=0)
    with pytest.raises(ValueError):
        TrainConfig(batch_size=0)
    with pytest.raises(ValueError):
        TrainConfig(lr=0.0)
    with pytest.raises(ValueError):
        TrainConfig(patience=-1)
    with pytest.raises(ValueError):
        TrainConfig(val_fraction=1.0)
    with pytest.raises(ValueError):
        TrainConfig(pos_weight=0.0)
    with pytest.raises(ValueError):
        TrainConfig(device="cuda")


def test_early_stopping_detects_improvement_then_stops():
    stopper = EarlyStopping(patience=1, min_delta=0.01)
    assert stopper.step(1.0, 0) is False
    assert stopper.step(0.5, 1) is False
    assert stopper.counter == 0
    assert stopper.step(0.52, 2) is False
    assert stopper.step(0.53, 3) is True
    assert stopper.best == pytest.approx(0.5)
    assert stopper.best_epoch == 1
    assert stopper.state_dict()["stopped"] is True


def test_early_stopping_restores_best_weights():
    model = make_model(base_channels=2, dropout=0.0, seed=0)
    original = [p.detach().clone() for p in model.parameters()]
    stopper = EarlyStopping(patience=0, min_delta=0.0)
    stopper.step(1.0, 0, model)
    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.0)
    assert stopper.restore(model) is True
    for param, before in zip(model.parameters(), original):
        assert torch.allclose(param, before)


def test_early_stopping_max_mode_and_validation():
    stopper = EarlyStopping(patience=0, min_delta=0.0, mode="max")
    assert stopper.step(0.5, 0) is False
    assert stopper.step(0.4, 1) is True
    assert stopper.best == pytest.approx(0.5)
    with pytest.raises(ValueError):
        EarlyStopping(mode="sideways")
    with pytest.raises(ValueError):
        EarlyStopping(patience=-1)


def test_train_one_epoch_reduces_loss_on_separable_data():
    set_seed(0)
    model = make_model(base_channels=4, dropout=0.0, seed=0)
    loader = toy_loader(32, 8)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    first = train_one_epoch(model, loader, optimizer, criterion)
    for _ in range(10):
        last = train_one_epoch(model, loader, optimizer, criterion)
    assert last < first
    assert np.isfinite(first) and np.isfinite(last)
    with pytest.raises(ValueError):
        train_one_epoch(model, DataLoader([], batch_size=1), optimizer, criterion)
