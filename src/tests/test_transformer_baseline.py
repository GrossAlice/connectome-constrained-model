"""Unit tests for the Transformer baseline package.

Tests cover:
1. Config defaults
2. SlidingWindowDataset shapes and boundary conditions
3. Temporal split correctness
4. Model forward-pass shapes
5. Causal masking (future tokens cannot influence past)
6. Gaussian NLL loss value sanity
7. LOO simulation basic correctness
8. Free-run output shape
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from baseline_transformer.config import TransformerBaselineConfig
from baseline_transformer.dataset import (
    SlidingWindowDataset,
    temporal_train_val_test_split,
    contiguous_cv_folds,
)
from baseline_transformer.model import (
    TemporalTransformerGaussian,
    build_model,
    gaussian_nll_loss,
)


# ---- Config ----

def test_config_defaults():
    cfg = TransformerBaselineConfig()
    assert cfg.d_model == 128
    assert cfg.n_heads == 4
    assert cfg.d_model % cfg.n_heads == 0
    assert 0 < cfg.train_frac < 1
    assert cfg.train_frac + cfg.val_frac < 1


# ---- Dataset ----

def test_sliding_window_shapes():
    T, N = 100, 10
    K = 8
    u = np.random.randn(T, N).astype(np.float32)
    ds = SlidingWindowDataset(u, context_length=K)
    assert len(ds) == T - K  # predict t=K..T-1
    ctx, tgt = ds[0]
    assert ctx.shape == (K, N)
    assert tgt.shape == (N,)


def test_sliding_window_range():
    T, N = 50, 5
    K = 4
    u = np.arange(T * N, dtype=np.float32).reshape(T, N)
    ds = SlidingWindowDataset(u, context_length=K, start=10, end=30)
    # First target is at t=14 (start+K), last target is at t=29
    assert len(ds) == 16
    ctx, tgt = ds[0]
    # Context should be u[10:14], target should be u[14]
    np.testing.assert_array_equal(tgt, u[14])
    np.testing.assert_array_equal(ctx, u[10:14])


def test_sliding_window_empty():
    u = np.random.randn(5, 3).astype(np.float32)
    ds = SlidingWindowDataset(u, context_length=10)  # K > T
    assert len(ds) == 0


def test_temporal_split():
    T = 1000
    split = temporal_train_val_test_split(T, train_frac=0.7, val_frac=0.15)
    tr_s, tr_e = split["train"]
    va_s, va_e = split["val"]
    te_s, te_e = split["test"]

    assert tr_s == 0
    assert tr_e == va_s  # contiguous
    assert va_e == te_s  # contiguous
    assert te_e == T
    assert tr_e > 0
    assert va_e > va_s
    assert te_e > te_s


def test_cv_folds():
    folds = contiguous_cv_folds(100, 5)
    assert len(folds) == 5
    all_val = np.concatenate([f[1] for f in folds])
    assert len(all_val) == 100
    assert len(set(all_val)) == 100  # no duplicates


# ---- Model ----

def test_model_forward_shape():
    N = 20
    K = 8
    B = 4
    cfg = TransformerBaselineConfig(d_model=32, n_heads=2, n_layers=1, d_ff=64, context_length=K)
    model = build_model(N, cfg)

    x = torch.randn(B, K, N)
    mu, sigma = model(x)
    assert mu.shape == (B, N)
    assert sigma.shape == (B, N)
    assert (sigma > 0).all()


def test_model_forward_all_steps():
    N = 15
    K = 6
    B = 3
    cfg = TransformerBaselineConfig(d_model=32, n_heads=2, n_layers=1, d_ff=64, context_length=K)
    model = build_model(N, cfg)

    x = torch.randn(B, K, N)
    mu, sigma = model(x, return_all_steps=True)
    assert mu.shape == (B, K, N)
    assert sigma.shape == (B, K, N)


def test_causal_masking():
    """Verify that changing a future token doesn't affect past predictions."""
    N = 10
    K = 8
    cfg = TransformerBaselineConfig(d_model=32, n_heads=2, n_layers=1, d_ff=64, context_length=K, dropout=0.0)
    model = build_model(N, cfg)
    model.eval()

    x1 = torch.randn(1, K, N)
    x2 = x1.clone()
    x2[0, -1, :] = torch.randn(N)  # change last token

    mu1, _ = model(x1, return_all_steps=True)
    mu2, _ = model(x2, return_all_steps=True)

    # All predictions except the last should be identical
    torch.testing.assert_close(mu1[0, :-1], mu2[0, :-1], atol=1e-5, rtol=1e-5)
    # Last prediction should differ
    assert not torch.allclose(mu1[0, -1], mu2[0, -1], atol=1e-3)


def test_sigma_bounds():
    N = 10
    K = 4
    cfg = TransformerBaselineConfig(
        d_model=32, n_heads=2, n_layers=1, d_ff=64,
        context_length=K, sigma_min=0.01, sigma_max=5.0,
    )
    model = build_model(N, cfg)

    x = torch.randn(8, K, N) * 100  # extreme inputs
    _, sigma = model(x)
    assert (sigma >= 0.01 - 1e-6).all()
    assert (sigma <= 5.0 + 1e-6).all()


# ---- Loss ----

def test_gaussian_nll_basic():
    B, N = 16, 10
    mu = torch.randn(B, N)
    sigma = torch.ones(B, N)
    target = mu.clone()  # perfect prediction

    loss_perfect = gaussian_nll_loss(mu, sigma, target)
    # For σ=1, y=μ:  NLL = 0.5 * log(2π) ≈ 0.919
    assert abs(loss_perfect.item() - 0.5 * np.log(2 * np.pi)) < 0.01

    # Random target should have higher loss
    target_random = torch.randn(B, N) * 5
    loss_random = gaussian_nll_loss(mu, sigma, target_random)
    assert loss_random.item() > loss_perfect.item()


def test_gaussian_nll_gradient():
    mu = torch.randn(4, 5, requires_grad=True)
    sigma = torch.ones(4, 5)
    target = torch.randn(4, 5)
    loss = gaussian_nll_loss(mu, sigma, target)
    loss.backward()
    assert mu.grad is not None
    assert mu.grad.shape == mu.shape


# ---- Free-run & LOO ----

def test_free_run_shape():
    N = 10
    K = 4
    cfg = TransformerBaselineConfig(d_model=32, n_heads=2, n_layers=1, d_ff=64, context_length=K)
    model = build_model(N, cfg)
    model.eval()

    u0 = torch.randn(K, N)
    pred = model.free_run(u0, n_steps=20)
    assert pred.shape == (20, N)


def test_loo_shape():
    N = 10
    K = 4
    T = 30
    cfg = TransformerBaselineConfig(d_model=32, n_heads=2, n_layers=1, d_ff=64, context_length=K)
    model = build_model(N, cfg)
    model.eval()

    u_gt = torch.randn(T, N)
    pred = model.loo_forward_simulate(u_gt, neuron_idx=3)
    assert pred.shape == (T - K, N)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
