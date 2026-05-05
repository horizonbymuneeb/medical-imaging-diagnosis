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


def test_model_config_validation():
    with pytest.raises(ValueError):
        ModelConfig(in_channels=0)
    with pytest.raises(ValueError):
        ModelConfig(base_channels=0)
    with pytest.raises(ValueError):
        ModelConfig(dropout=1.0)
    with pytest.raises(ValueError):
        ModelConfig(patch_size=31)


def test_conv_block_halves_spatial_size():
    block = ConvBlock(1, 4)
    out = block(torch.zeros(2, 1, 16, 16))
    assert out.shape == (2, 4, 8, 8)


def test_forward_returns_one_logit_per_sample():
    model = make_model(base_channels=4, seed=0)
    out = model(torch.zeros(3, 1, 32, 32))
    assert out.shape == (3,)
    with pytest.raises(ValueError):
        model(torch.zeros(1, 32, 32))


def test_forward_accepts_other_patch_sizes():
    model = make_model(base_channels=4, seed=0)
    for size in (16, 48):
        assert model(torch.zeros(1, 1, size, size)).shape == (1,)


def test_parameter_count_matches_manual_sum():
    model = make_model(base_channels=8, seed=1)
    manual = sum(p.numel() for p in model.parameters())
    assert count_parameters(model, trainable_only=False) == manual
    assert count_parameters(model) == manual
    breakdown = parameter_breakdown(model)
    assert breakdown["features"] > 0 and breakdown["head"] > 0
    assert sum(breakdown.values()) == manual


def test_seeded_init_is_reproducible():
    a = make_model(base_channels=4, seed=3)
    b = make_model(base_channels=4, seed=3)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


def test_checkpoint_roundtrip_preserves_predictions(tmp_path):
    model = make_model(base_channels=4, dropout=0.0, seed=2)
    x = torch.randn(4, 1, 32, 32)
    before = predict_proba(model, x)
    path = save_checkpoint(model, tmp_path / "ckpt.pt", {"base_channels": 4, "dropout": 0.0, "patch_size": 32})
    loaded, meta = load_checkpoint(path)
    after = predict_proba(loaded, x)
    assert meta["base_channels"] == 4
    assert np.allclose(before, after, atol=1e-6)
    assert path.exists()


def test_load_checkpoint_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nope.pt")


def test_predict_labels_threshold_behaviour():
    model = make_model(base_channels=4, dropout=0.0, seed=4)
    x = torch.randn(5, 1, 32, 32)
    probs = predict_proba(model, x)
    assert probs.shape == (5,)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()
    assert (predict_labels(model, x, threshold=0.0) == 1).all()
    assert (predict_labels(model, x, threshold=1.0) == 0).all()
    with pytest.raises(ValueError):
        predict_labels(model, x, threshold=1.5)
