"""Model shape, parameter count, checkpoint and inference tests."""

import numpy as np
import pytest
import torch

from src.medimg.model import (
    ConvBlock,
    ModelConfig,
    SmallCNN,
    count_parameters,
    load_checkpoint,
    make_model,
    parameter_breakdown,
    predict_labels,
    predict_proba,
    save_checkpoint,
)
