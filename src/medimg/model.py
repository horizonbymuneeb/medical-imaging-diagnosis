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


@dataclass(frozen=True)
class ModelConfig:
    """Shape of the classifier."""

    in_channels: int = 1
    base_channels: int = 16
    dropout: float = 0.1
    patch_size: int = 32

    def __post_init__(self) -> None:
        if self.in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if self.base_channels <= 0:
            raise ValueError("base_channels must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.patch_size <= 0 or self.patch_size % 2:
            raise ValueError("patch_size must be a positive even integer")


class ConvBlock(nn.Module):
    """conv -> batch norm -> ReLU -> max pool."""

    def __init__(self, in_channels: int, out_channels: int, pool: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.norm(self.conv(x))))
