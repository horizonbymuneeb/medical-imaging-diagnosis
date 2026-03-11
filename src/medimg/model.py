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


class SmallCNN(nn.Module):
    """Three-block CNN with global average pooling and a single logit.

    Global average pooling instead of a flatten head keeps the parameter count
    independent of the patch size, so the same model works for 32 or 64 px
    patches without editing the head.
    """

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()
        c = self.cfg.base_channels
        self.features = nn.Sequential(
            ConvBlock(self.cfg.in_channels, c),
            ConvBlock(c, c * 2),
            ConvBlock(c * 2, c * 4),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(c * 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits with shape ``(N,)``."""
        if x.ndim != 4:
            raise ValueError(f"expected (N, C, H, W) input, got shape {tuple(x.shape)}")
        return self.head(self.features(x)).squeeze(-1)

    def features_for(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate feature map, useful for debugging representations."""
        return self.features(x)


def make_model(
    *,
    in_channels: int = 1,
    base_channels: int = 16,
    dropout: float = 0.1,
    patch_size: int = 32,
    seed: int | None = None,
) -> SmallCNN:
    """Build a model, optionally with a deterministic weight initialisation."""
    cfg = ModelConfig(
        in_channels=in_channels,
        base_channels=base_channels,
        dropout=dropout,
        patch_size=patch_size,
    )
    model = SmallCNN(cfg)
    if seed is not None:
        torch.manual_seed(seed)
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    return model
