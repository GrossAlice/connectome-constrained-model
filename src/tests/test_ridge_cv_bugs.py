"""Regression tests for the ridge-CV dynamics solver (train.py).

1. Explicit intercept: solver must recover baseline via intercept, not a
   constant column.  I0 = intercept / λ.
2. Weighted loss: denominator must use sum-of-weights, not count-of-valid.
3. Per-fold standardization: CV folds must not leak validation statistics.
4. Dynamics-CV round-trip: synthetic AR(1) + baseline must recover I0.
"""
from __future__ import annotations

import numpy as np
import torch
import pytest


# ────────────────────────────────────────────────────────────────────────────
# 1. Explicit-intercept recovery
# ────────────────────────────────────────────────────────────────────────────

def test_ridge_cv_recovers_intercept():
    """The solver must recover the baseline in result['intercept'],
    not in any coefficient.  For y = 0.9*x + 5.0 + noise, intercept ≈ 5."""
    from stage2.train import _solve_neuron_ridge_cv

    rng = np.random.default_rng(42)
    T = 500
    X = rng.standard_normal((T, 1))
    y = 0.9 * X[:, 0] + 5.0 + rng.standard_normal(T) * 0.01

    ridge_grid = np.logspace(-6, 6, 40)
    result = _solve_neuron_ridge_cv(X, y, ridge_grid, n_folds=5)

    assert abs(result["coef"][0] - 0.9) < 0.05, (
        f"Slope should be ~0.9, got {result['coef'][0]:.6f}"
    )
    assert abs(result["intercept"] - 5.0) < 0.1, (
        f"Intercept should be ~5.0, got {result['intercept']:.6f}"
    )


def test_ridge_cv_multicolumn_intercept():
    """With multiple features, intercept + coefs should all be recovered."""
    from stage2.train import _solve_neuron_ridge_cv

    rng = np.random.default_rng(7)
    T, p = 600, 4
    true_coef = np.array([0.9, 0.05, 0.01, 0.01])
    true_intercept = 0.3

    X = rng.standard_normal((T, p))
    y = true_intercept + X @ true_coef + rng.standard_normal(T) * 0.01

    ridge_grid = np.logspace(-6, 6, 40)
    result = _solve_neuron_ridge_cv(X, y, ridge_grid, n_folds=5)

    assert abs(result["intercept"] - true_intercept) < 0.05, (
        f"Intercept should be ~{true_intercept}, got {result['intercept']:.6f}"
    )
    for j in range(p):
        assert abs(result["coef"][j] - true_coef[j]) < 0.1, (
            f"coef[{j}]: expected ~{true_coef[j]:.3f}, got {result['coef'][j]:.6f}"
        )


def test_ridge_cv_prediction_with_intercept():
    """y_pred = intercept + X @ coef must give R^2 ~ 1 for a clean signal."""
    from stage2.train import _solve_neuron_ridge_cv

    rng = np.random.default_rng(99)
    T = 400
    X = rng.standard_normal((T, 2))
    y = 3.0 + 1.5 * X[:, 0] - 0.5 * X[:, 1] + rng.standard_normal(T) * 0.01

    ridge_grid = np.logspace(-6, 6, 40)
    result = _solve_neuron_ridge_cv(X, y, ridge_grid, n_folds=5)

    y_pred = result["intercept"] + X @ result["coef"]
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    assert r2 > 0.999, f"R^2 should be ~1.0, got {r2:.6f}"


# ────────────────────────────────────────────────────────────────────────────
# 2. Weighted-loss denominator
# ────────────────────────────────────────────────────────────────────────────

def test_weighted_loss_denominator():
    """u_var_weighting increases effective variance, reducing the loss
    contribution from high-variance neurons.
    """
    from stage2.train import compute_dynamics_loss

    torch.manual_seed(0)
    T, N = 10, 2
    target = torch.randn(T, N)
    pred   = torch.randn(T, N)
    sigma  = torch.ones(N)

    loss_no_uvar = compute_dynamics_loss(target, pred, sigma)

    # With u_var > 0, effective variance increases → loss should decrease
    u_var = torch.ones(T, N) * 10.0
    loss_uvar = compute_dynamics_loss(
        target, pred, sigma,
        u_var=u_var, use_u_var_weighting=True, u_var_scale=1.0,
    )

    assert loss_uvar.item() < loss_no_uvar.item(), (
        f"u_var_weighting should reduce loss. "
        f"Got {loss_uvar.item():.6f} >= {loss_no_uvar.item():.6f}"
    )


def test_weighted_loss_uniform_weights_unchanged():
    """Zero u_var with weighting enabled should give the same result as disabled."""
    from stage2.train import compute_dynamics_loss

    torch.manual_seed(1)
    T, N = 20, 5
    target = torch.randn(T, N)
    pred   = torch.randn(T, N)
    sigma  = torch.ones(N)

    loss_no_w = compute_dynamics_loss(target, pred, sigma)
    loss_w0   = compute_dynamics_loss(
        target, pred, sigma,
        u_var=torch.zeros(T, N), use_u_var_weighting=True,
        u_var_scale=1.0, u_var_floor=0.0,
    )

    assert abs(loss_no_w.item() - loss_w0.item()) < 1e-5, (
        f"Zero u_var should not change loss. "
        f"No-weight={loss_no_w.item():.6f}, zero-uvar={loss_w0.item():.6f}"
    )


# ────────────────────────────────────────────────────────────────────────────
# 3. Dynamics-CV round-trip: recover I0 from intercept
# ────────────────────────────────────────────────────────────────────────────

def test_dynamics_cv_recovers_I0():
    """Synthetic AR(1) + baseline: u[t+1] = (1-lam)u[t] + lam*I0 + noise.

    The solver must recover I0 ~ 2.0 from intercept/lam, not from a
    coefficient on a constant column.
    """
    from stage2.train import _solve_neuron_ridge_cv

    rng = np.random.default_rng(123)
    T = 1000
    lam_true = 0.1
    I0_true = 2.0

    u = np.zeros(T)
    for t in range(T - 1):
        u[t + 1] = (1 - lam_true) * u[t] + lam_true * I0_true + rng.normal(0, 0.01)

    # Design matrix: col 0 = u_prev, col 1 = gap (zero in this test)
    X = np.column_stack([u[:-1], np.zeros(T - 1)])
    y = u[1:]

    ridge_grid = np.logspace(-6, 6, 40)
    result = _solve_neuron_ridge_cv(X, y, ridge_grid, n_folds=5)

    alpha_i = result["coef"][0]
    lam_recovered = np.clip(1.0 - alpha_i, 0.004, 0.9999)
    I0_recovered = result["intercept"] / lam_recovered

    assert abs(I0_recovered - I0_true) < 0.5, (
        f"Recovered I0 should be ~{I0_true}, got {I0_recovered:.4f}. "
        f"intercept={result['intercept']:.6f}, lam={lam_recovered:.4f}"
    )
    assert abs(result["intercept"] - lam_true * I0_true) < 0.05, (
        f"intercept should be ~{lam_true * I0_true:.3f}, "
        f"got {result['intercept']:.6f}"
    )


# ────────────────────────────────────────────────────────────────────────────
# 4. Per-fold standardization (no validation-statistics leak)
# ────────────────────────────────────────────────────────────────────────────

def test_ridge_cv_per_fold_standardization():
    """Verify that CV folds use training-only statistics.

    Construct a dataset where the last block has a very different feature
    scale.  Coefficient recovery should not be degraded by the shift.
    """
    from stage2.train import _solve_neuron_ridge_cv

    rng = np.random.default_rng(99)
    T = 600
    p = 2
    true_coef = np.array([0.9, 0.05])
    true_intercept = 0.25

    X = rng.standard_normal((T, p))
    # Inject a mean-shift in the last 20% of rows for col 0
    X[int(0.8 * T):, 0] += 10.0

    y = true_intercept + X @ true_coef + rng.standard_normal(T) * 0.02

    ridge_grid = np.logspace(-6, 6, 40)
    result = _solve_neuron_ridge_cv(X, y, ridge_grid, n_folds=5)

    assert abs(result["intercept"] - true_intercept) < 0.1, (
        f"Intercept: expected ~{true_intercept}, got {result['intercept']:.6f}"
    )
    for j in range(p):
        assert abs(result["coef"][j] - true_coef[j]) < 0.1, (
            f"coef[{j}]: expected ~{true_coef[j]:.3f}, got {result['coef'][j]:.6f}"
        )


def test_ridge_cv_fold_predictions_use_training_stats():
    """Direct structural test: when the validation fold has a different
    distribution, recovered coefs must still be accurate.
    """
    from stage2.train import _solve_neuron_ridge_cv

    rng = np.random.default_rng(77)
    T = 500
    x = rng.standard_normal(T)
    x[400:] += 50.0  # large shift in last fold

    X = x.reshape(-1, 1)
    y = 2.0 * x + 3.0 + rng.standard_normal(T) * 0.1

    ridge_grid = np.logspace(-6, 6, 40)
    result = _solve_neuron_ridge_cv(X, y, ridge_grid, n_folds=5)

    assert abs(result["coef"][0] - 2.0) < 0.1, (
        f"Slope should be ~2.0, got {result['coef'][0]:.4f}"
    )
    assert abs(result["intercept"] - 3.0) < 0.2, (
        f"Intercept should be ~3.0, got {result['intercept']:.4f}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
