#!/usr/bin/env python3
"""
Stage 2 Model Constraint Tests

Tests to understand what the Stage 2 model captures about neural dynamics:
1. Self vs Non-Self prediction → intrinsic vs coupled dynamics
2. Ridge vs MLP gap → linear vs nonlinear (chemical synapse) effects
3. Symmetry test → gap junction vs chemical synapse dominance
4. Hub neuron identification → command interneuron importance
5. Lag contribution analysis → fast vs slow dynamics
6. Granger causality matrix → effective connectivity
7. Behavior coupling heterogeneity → motor neuron identification
8. Prediction horizon decay → stability of dynamics
9. Noise correlation structure → unmodeled shared inputs
10. Transformer baseline → how much does architecture matter

Usage:
    python scripts/stage2_constraint_tests.py --device cuda --n_worms 10
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
import h5py

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from baseline_transformer.dataset import load_worm_data
from baseline_transformer.config import TransformerBaselineConfig
from baseline_transformer.model import TemporalTransformerGaussian


# ═══════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════

def build_lagged(x: np.ndarray, n_lags: int) -> np.ndarray:
    """Stack n_lags time-shifted copies → (T, D*n_lags)."""
    T, D = x.shape
    parts = []
    for lag in range(1, n_lags + 1):
        s = np.zeros((T, D), dtype=x.dtype)
        if lag < T:
            s[lag:] = x[:-lag]
        parts.append(s)
    return np.concatenate(parts, axis=1)


def _make_mlp(d_in: int, d_out: int, hidden: int = 128, n_layers: int = 2) -> nn.Sequential:
    layers, d = [], d_in
    for _ in range(n_layers):
        layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(0.1)]
        d = hidden
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


def train_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 1.0):
    return Ridge(alpha=alpha).fit(X, y)


def train_mlp(X: np.ndarray, y: np.ndarray, device: str,
              epochs: int = 150, lr: float = 1e-3, patience: int = 20) -> nn.Sequential:
    nv = max(10, int(X.shape[0] * 0.15))
    Xt = torch.tensor(X[:-nv], dtype=torch.float32, device=device)
    yt = torch.tensor(y[:-nv], dtype=torch.float32, device=device)
    Xv = torch.tensor(X[-nv:], dtype=torch.float32, device=device)
    yv = torch.tensor(y[-nv:], dtype=torch.float32, device=device)

    mlp = _make_mlp(X.shape[1], y.shape[1]).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=1e-4)

    bvl, bs, pat = float("inf"), None, 0
    for _ in range(epochs):
        mlp.train()
        loss = nn.functional.mse_loss(mlp(Xt), yt)
        opt.zero_grad(); loss.backward(); opt.step()

        mlp.eval()
        with torch.no_grad():
            vl = nn.functional.mse_loss(mlp(Xv), yv).item()
        if vl < bvl - 1e-6:
            bvl, bs, pat = vl, {k: v.clone() for k, v in mlp.state_dict().items()}, 0
        else:
            pat += 1
        if pat > patience:
            break

    if bs:
        mlp.load_state_dict(bs)
    mlp.eval().cpu()
    return mlp


def train_transformer(u_train: np.ndarray, K: int, device: str) -> tuple:
    """Train transformer on neural data, return model and predictions."""
    cfg = TransformerBaselineConfig()
    cfg.d_model = 128
    cfg.n_heads = 4
    cfg.n_layers = 2
    cfg.d_ff = 256
    cfg.context_length = K
    cfg.max_epochs = 100
    cfg.patience = 15
    cfg.device = device
    
    T, N = u_train.shape
    
    # Normalize
    mu = u_train[K:].mean(0)
    sig = u_train[K:].std(0) + 1e-8
    u_n = ((u_train - mu) / sig).astype(np.float32)
    
    # Build windows
    tr_idx = np.arange(K, T)
    X_win = np.stack([u_n[t - K:t] for t in tr_idx])
    y_tr = u_n[tr_idx]
    
    # Train/val split
    nv = max(10, int(len(tr_idx) * 0.15))
    Xt = torch.tensor(X_win[:-nv], dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr[:-nv], dtype=torch.float32, device=device)
    Xv = torch.tensor(X_win[-nv:], dtype=torch.float32, device=device)
    yv = torch.tensor(y_tr[-nv:], dtype=torch.float32, device=device)
    
    model = TemporalTransformerGaussian(n_neural=N, n_beh=0, cfg=cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    bvl, bs, pat = float("inf"), None, 0
    for ep in range(cfg.max_epochs):
        model.train()
        mu_out, sig_out, _, _ = model.forward(Xt)
        loss = nn.functional.gaussian_nll_loss(mu_out, yt, sig_out ** 2)
        opt.zero_grad(); loss.backward(); opt.step()
        
        model.eval()
        with torch.no_grad():
            mu_v, sig_v, _, _ = model.forward(Xv)
            vl = nn.functional.gaussian_nll_loss(mu_v, yv, sig_v ** 2).item()
        if vl < bvl - 1e-6:
            bvl, bs, pat = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            pat += 1
        if pat > cfg.patience:
            break
    
    if bs:
        model.load_state_dict(bs)
    model.eval()
    
    # Get predictions
    with torch.no_grad():
        X_all = torch.tensor(X_win, dtype=torch.float32, device=device)
        mu_pred, _, _, _ = model.forward(X_all)
        pred = mu_pred.cpu().numpy() * sig + mu
    
    model.cpu()
    return model, pred, u_train[tr_idx]


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / max(ss_tot, 1e-12)


def compute_r2_per_dim(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    r2s = []
    for d in range(y_true.shape[1]):
        r2s.append(compute_r2(y_true[:, d], y_pred[:, d]))
    return np.array(r2s)


# ═══════════════════════════════════════════════════════════════════════════
# Test 1: Self vs Non-Self Prediction (Optimized)
# ═══════════════════════════════════════════════════════════════════════════

def test_self_vs_nonself(u: np.ndarray, K: int, device: str, n_sample: int = 20) -> dict:
    """Compare prediction from own history vs other neurons' history.
    
    Optimized: sample neurons instead of all, use vectorized Ridge for 'all'.
    """
    T, N = u.shape
    mu = u[K:].mean(0); sig = u[K:].std(0) + 1e-8
    u_n = ((u - mu) / sig).astype(np.float32)
    
    tr_idx = np.arange(K, T)
    
    # Sample neurons for detailed analysis (faster)
    sample_idx = np.random.choice(N, min(n_sample, N), replace=False)
    
    r2_self = np.zeros(len(sample_idx))
    r2_nonself = np.zeros(len(sample_idx))
    r2_all = np.zeros(len(sample_idx))
    
    # Pre-compute X_all once
    X_all = build_lagged(u_n, K)
    
    for ii, i in enumerate(sample_idx):
        # Self only: predict neuron i from its own history
        X_self = build_lagged(u_n[:, i:i+1], K)
        y_i = u_n[tr_idx, i]
        
        ridge_self = train_ridge(X_self[tr_idx], y_i)
        pred_self = ridge_self.predict(X_self[tr_idx])
        r2_self[ii] = compute_r2(y_i, pred_self)
        
        # Non-self: predict from all other neurons
        other_idx = [j for j in range(N) if j != i]
        X_nonself = build_lagged(u_n[:, other_idx], K)
        ridge_nonself = train_ridge(X_nonself[tr_idx], y_i)
        pred_nonself = ridge_nonself.predict(X_nonself[tr_idx])
        r2_nonself[ii] = compute_r2(y_i, pred_nonself)
        
        # All neurons
        ridge_all = train_ridge(X_all[tr_idx], y_i)
        pred_all = ridge_all.predict(X_all[tr_idx])
        r2_all[ii] = compute_r2(y_i, pred_all)
    
    return {
        "r2_self": r2_self,
        "r2_nonself": r2_nonself,
        "r2_all": r2_all,
        "sample_idx": sample_idx,
        "self_contribution": r2_self.mean(),
        "nonself_contribution": r2_nonself.mean(),
        "coupling_ratio": (r2_nonself.mean() / max(r2_self.mean(), 1e-6)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Test 2: Ridge vs MLP Gap (Nonlinearity)
# ═══════════════════════════════════════════════════════════════════════════

def test_ridge_vs_mlp(u: np.ndarray, K: int, device: str) -> dict:
    """Compare Ridge vs MLP to quantify nonlinearity."""
    T, N = u.shape
    mu = u[K:].mean(0); sig = u[K:].std(0) + 1e-8
    u_n = ((u - mu) / sig).astype(np.float32)
    
    X_lag = build_lagged(u_n, K)
    tr_idx = np.arange(K, T)
    y = u_n[tr_idx]
    
    # Ridge
    ridge = train_ridge(X_lag[tr_idx], y)
    pred_ridge = ridge.predict(X_lag[tr_idx])
    r2_ridge = compute_r2_per_dim(y, pred_ridge)
    
    # MLP
    mlp = train_mlp(X_lag[tr_idx], y, device)
    with torch.no_grad():
        pred_mlp = mlp(torch.tensor(X_lag[tr_idx], dtype=torch.float32)).numpy()
    r2_mlp = compute_r2_per_dim(y, pred_mlp)
    
    return {
        "r2_ridge": r2_ridge,
        "r2_mlp": r2_mlp,
        "r2_ridge_mean": r2_ridge.mean(),
        "r2_mlp_mean": r2_mlp.mean(),
        "mlp_gain": (r2_mlp - r2_ridge).mean(),
        "nonlinearity_index": (r2_mlp.mean() - r2_ridge.mean()) / max(r2_ridge.mean(), 1e-6),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Test 2b: Nonlinearity Across Timescales
# ═══════════════════════════════════════════════════════════════════════════

def test_nonlinearity_timescales(u: np.ndarray, device: str, n_sample: int = 20) -> dict:
    """Test nonlinearity (Ridge vs MLP) across different prediction horizons and lags.
    
    This tests whether chemical synapse nonlinearities emerge at longer timescales.
    
    Returns:
        - mlp_gain_by_horizon: MLP improvement over Ridge at each prediction horizon
        - mlp_gain_by_lag: MLP improvement at each lag window
        - per_neuron_nonlinearity: which neurons consistently show nonlinear dynamics
    """
    T, N = u.shape
    mu = u.mean(0); sig = u.std(0) + 1e-8
    u_n = ((u - mu) / sig).astype(np.float32)
    
    # Sample neurons for detailed analysis
    sample_idx = np.random.choice(N, min(n_sample, N), replace=False)
    
    # Test different prediction horizons (1-step to 50-step ahead)
    horizons = [1, 2, 5, 10, 20, 50]
    K_base = 10  # Fixed lag window
    
    mlp_gain_by_horizon = {}
    per_neuron_by_horizon = {}
    
    X_lag = build_lagged(u_n, K_base)
    
    for h in horizons:
        if K_base + h >= T:
            continue
        tr_idx = np.arange(K_base, T - h)
        y = u_n[tr_idx + h]  # Predict h steps ahead
        
        # Ridge
        ridge = train_ridge(X_lag[tr_idx], y)
        pred_ridge = ridge.predict(X_lag[tr_idx])
        r2_ridge = compute_r2_per_dim(y, pred_ridge)
        
        # MLP
        mlp = train_mlp(X_lag[tr_idx], y, device, epochs=100, patience=15)
        with torch.no_grad():
            pred_mlp = mlp(torch.tensor(X_lag[tr_idx], dtype=torch.float32)).numpy()
        r2_mlp = compute_r2_per_dim(y, pred_mlp)
        
        mlp_gain = r2_mlp - r2_ridge
        mlp_gain_by_horizon[h] = float(mlp_gain.mean())
        per_neuron_by_horizon[h] = mlp_gain.tolist()
    
    # Test different lag windows with fixed 1-step prediction
    lags = [1, 3, 5, 10, 15, 20, 30]
    mlp_gain_by_lag = {}
    
    for K in lags:
        if K >= T // 3:
            continue
        X_lag_k = build_lagged(u_n, K)
        tr_idx = np.arange(K, T)
        y = u_n[tr_idx]
        
        ridge = train_ridge(X_lag_k[tr_idx], y)
        pred_ridge = ridge.predict(X_lag_k[tr_idx])
        r2_ridge = compute_r2_per_dim(y, pred_ridge)
        
        mlp = train_mlp(X_lag_k[tr_idx], y, device, epochs=100, patience=15)
        with torch.no_grad():
            pred_mlp = mlp(torch.tensor(X_lag_k[tr_idx], dtype=torch.float32)).numpy()
        r2_mlp = compute_r2_per_dim(y, pred_mlp)
        
        mlp_gain_by_lag[K] = float((r2_mlp - r2_ridge).mean())
    
    # Identify consistently nonlinear neurons
    # A neuron is "nonlinear" if MLP > Ridge across multiple horizons
    per_neuron_gain = np.zeros(N)
    n_horizons_tested = len(per_neuron_by_horizon)
    for h, gains in per_neuron_by_horizon.items():
        per_neuron_gain += np.array(gains)
    per_neuron_gain /= max(n_horizons_tested, 1)
    
    # Neurons with positive mean MLP gain
    nonlinear_neurons = np.where(per_neuron_gain > 0.01)[0]
    
    return {
        "mlp_gain_by_horizon": mlp_gain_by_horizon,
        "mlp_gain_by_lag": mlp_gain_by_lag,
        "per_neuron_gain": per_neuron_gain.tolist(),
        "nonlinear_neurons": nonlinear_neurons.tolist(),
        "n_nonlinear": len(nonlinear_neurons),
        "pct_nonlinear": 100.0 * len(nonlinear_neurons) / N,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Test 3: Symmetry Test (Gap Junction Signature) - Optimized
# ═══════════════════════════════════════════════════════════════════════════

def test_symmetry(u: np.ndarray, K: int, n_pairs: int = 100) -> dict:
    """Test if prediction is symmetric (gap junction) or asymmetric (chemical).
    
    Optimized: sample random pairs instead of full N×N matrix.
    """
    T, N = u.shape
    mu = u[K:].mean(0); sig = u[K:].std(0) + 1e-8
    u_n = ((u - mu) / sig).astype(np.float32)
    
    tr_idx = np.arange(K, T)
    
    # Sample random pairs
    n_pairs = min(n_pairs, N * (N - 1) // 2)
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    np.random.shuffle(all_pairs)
    sampled_pairs = all_pairs[:n_pairs]
    
    r2_ij = []  # i predicts j
    r2_ji = []  # j predicts i
    
    for i, j in sampled_pairs:
        # Predict j from i's history
        X_i = build_lagged(u_n[:, i:i+1], K)
        y_j = u_n[tr_idx, j]
        ridge_ij = train_ridge(X_i[tr_idx], y_j)
        r2_ij.append(compute_r2(y_j, ridge_ij.predict(X_i[tr_idx])))
        
        # Predict i from j's history
        X_j = build_lagged(u_n[:, j:j+1], K)
        y_i = u_n[tr_idx, i]
        ridge_ji = train_ridge(X_j[tr_idx], y_i)
        r2_ji.append(compute_r2(y_i, ridge_ji.predict(X_j[tr_idx])))
    
    r2_ij = np.array(r2_ij)
    r2_ji = np.array(r2_ji)
    
    symmetry_corr = np.corrcoef(r2_ij, r2_ji)[0, 1] if len(r2_ij) > 2 else 0
    
    # Build sparse matrix for visualization (fill sampled pairs only)
    R2_matrix = np.zeros((N, N))
    for idx, (i, j) in enumerate(sampled_pairs):
        R2_matrix[j, i] = r2_ij[idx]  # i predicts j → column i, row j
        R2_matrix[i, j] = r2_ji[idx]  # j predicts i → column j, row i
    
    return {
        "R2_matrix": R2_matrix,
        "symmetry_correlation": symmetry_corr,
        "mean_asymmetry": np.mean(np.abs(r2_ij - r2_ji)),
        "n_pairs_sampled": n_pairs,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Test 4: Hub Neuron Identification (Optimized)
# ═══════════════════════════════════════════════════════════════════════════

def test_hub_neurons(u: np.ndarray, K: int, n_sample: int = 20) -> dict:
    """Identify hub neurons via leave-one-out ablation.
    
    Optimized: sample neurons for LOO instead of all N.
    """
    T, N = u.shape
    mu = u[K:].mean(0); sig = u[K:].std(0) + 1e-8
    u_n = ((u - mu) / sig).astype(np.float32)
    
    tr_idx = np.arange(K, T)
    
    # Baseline: predict all from all
    X_all = build_lagged(u_n, K)
    y = u_n[tr_idx]
    ridge_full = train_ridge(X_all[tr_idx], y)
    pred_full = ridge_full.predict(X_all[tr_idx])
    r2_full = compute_r2(y.ravel(), pred_full.ravel())
    
    # Sample neurons for LOO
    sample_idx = np.random.choice(N, min(n_sample, N), replace=False)
    importance = np.zeros(N)
    
    for i in sample_idx:
        mask = [j for j in range(N) if j != i]
        X_loo = build_lagged(u_n[:, mask], K)
        y_loo = u_n[tr_idx][:, mask]
        ridge_loo = train_ridge(X_loo[tr_idx], y_loo)
        pred_loo = ridge_loo.predict(X_loo[tr_idx])
        r2_loo = compute_r2(y_loo.ravel(), pred_loo.ravel())
        importance[i] = r2_full - r2_loo
    
    return {
        "importance": importance,
        "top_hubs": np.argsort(importance)[::-1][:5],
        "r2_full": r2_full,
        "sample_idx": sample_idx,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Test 5: Lag Contribution Analysis
# ═══════════════════════════════════════════════════════════════════════════

def test_lag_contribution(u: np.ndarray, device: str) -> dict:
    """Test how R² changes with different lag windows."""
    T, N = u.shape
    mu = u.mean(0); sig = u.std(0) + 1e-8
    u_n = ((u - mu) / sig).astype(np.float32)
    
    lags = [1, 3, 5, 10, 15, 20]
    r2_by_lag = {}
    
    for K in lags:
        if K >= T // 3:
            continue
        X_lag = build_lagged(u_n, K)
        tr_idx = np.arange(K, T)
        y = u_n[tr_idx]
        
        ridge = train_ridge(X_lag[tr_idx], y)
        pred = ridge.predict(X_lag[tr_idx])
        r2_by_lag[K] = compute_r2_per_dim(y, pred).mean()
    
    return {
        "r2_by_lag": r2_by_lag,
        "optimal_lag": max(r2_by_lag, key=r2_by_lag.get) if r2_by_lag else 1,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Test 6: Granger Causality Matrix (Optimized)
# ═══════════════════════════════════════════════════════════════════════════

def test_granger_causality(u: np.ndarray, K: int, n_sample: int = 20) -> dict:
    """Compute Granger causality: does j improve prediction of i?
    
    Optimized: sample target neurons instead of full N×N.
    """
    T, N = u.shape
    mu = u[K:].mean(0); sig = u[K:].std(0) + 1e-8
    u_n = ((u - mu) / sig).astype(np.float32)
    
    tr_idx = np.arange(K, T)
    
    # Sample target neurons
    sample_idx = np.random.choice(N, min(n_sample, N), replace=False)
    
    # GC[i,j] = improvement in predicting i when adding j's history
    GC = np.zeros((N, N))
    
    for i in sample_idx:
        y_i = u_n[tr_idx, i]
        
        # Baseline: self-only
        X_self = build_lagged(u_n[:, i:i+1], K)
        ridge_self = train_ridge(X_self[tr_idx], y_i)
        r2_self = compute_r2(y_i, ridge_self.predict(X_self[tr_idx]))
        
        # Test top-k most correlated neurons as sources (faster than all N)
        # Compute correlations for source selection
        corrs = np.abs([np.corrcoef(u_n[:, i], u_n[:, j])[0, 1] for j in range(N)])
        corrs[i] = -1  # exclude self
        top_sources = np.argsort(corrs)[::-1][:min(n_sample, N - 1)]
        
        for j in top_sources:
            # Self + j
            X_ij = np.concatenate([
                build_lagged(u_n[:, i:i+1], K),
                build_lagged(u_n[:, j:j+1], K)
            ], axis=1)
            ridge_ij = train_ridge(X_ij[tr_idx], y_i)
            r2_ij = compute_r2(y_i, ridge_ij.predict(X_ij[tr_idx]))
            
            GC[i, j] = max(0, r2_ij - r2_self)
    
    return {
        "granger_matrix": GC,
        "mean_gc": GC[GC > 0].mean() if (GC > 0).any() else 0,
        "max_gc_pairs": np.unravel_index(np.argsort(GC.ravel())[-5:], GC.shape),
        "sample_idx": sample_idx,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Test 7: Behavior Coupling Heterogeneity (Optimized)
# ═══════════════════════════════════════════════════════════════════════════

def test_behavior_coupling(u: np.ndarray, b: np.ndarray, K: int, n_sample: int = 20) -> dict:
    """Which neurons are most behavior-locked?
    
    Optimized: sample neurons and reuse lagged matrices.
    """
    T, N = u.shape
    Kw = b.shape[1]
    
    mu_u = u[K:].mean(0); sig_u = u[K:].std(0) + 1e-8
    mu_b = b[K:].mean(0); sig_b = b[K:].std(0) + 1e-8
    u_n = ((u - mu_u) / sig_u).astype(np.float32)
    b_n = ((b - mu_b) / sig_b).astype(np.float32)
    
    tr_idx = np.arange(K, T)
    
    # Pre-compute lagged matrices once
    X_neural = build_lagged(u_n, K)
    X_nb = np.concatenate([X_neural, build_lagged(b_n, K)], axis=1)
    
    # Sample neurons
    sample_idx = np.random.choice(N, min(n_sample, N), replace=False)
    beh_coupling = np.zeros(N)
    
    for i in sample_idx:
        y_i = u_n[tr_idx, i]
        
        # Neural only
        ridge_neural = train_ridge(X_neural[tr_idx], y_i)
        r2_neural = compute_r2(y_i, ridge_neural.predict(X_neural[tr_idx]))
        
        # Neural + behavior
        ridge_nb = train_ridge(X_nb[tr_idx], y_i)
        r2_nb = compute_r2(y_i, ridge_nb.predict(X_nb[tr_idx]))
        
        beh_coupling[i] = r2_nb - r2_neural
    
    return {
        "beh_coupling": beh_coupling,
        "top_beh_coupled": np.argsort(beh_coupling)[::-1][:5],
        "mean_beh_gain": beh_coupling[sample_idx].mean(),
        "sample_idx": sample_idx,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Test 8: Prediction Horizon Decay
# ═══════════════════════════════════════════════════════════════════════════

def test_horizon_decay(u: np.ndarray, K: int, device: str) -> dict:
    """How far ahead can model predict?"""
    T, N = u.shape
    mu = u[K:].mean(0); sig = u[K:].std(0) + 1e-8
    u_n = ((u - mu) / sig).astype(np.float32)
    
    horizons = [1, 2, 5, 10, 20, 50]
    r2_by_horizon = {}
    
    X_lag = build_lagged(u_n, K)
    
    for h in horizons:
        if K + h >= T:
            continue
        tr_idx = np.arange(K, T - h)
        y = u_n[tr_idx + h]  # Predict h steps ahead
        
        ridge = train_ridge(X_lag[tr_idx], y)
        pred = ridge.predict(X_lag[tr_idx])
        r2_by_horizon[h] = compute_r2_per_dim(y, pred).mean()
    
    return {
        "r2_by_horizon": r2_by_horizon,
        "decay_rate": None,  # Could fit exponential
    }


# ═══════════════════════════════════════════════════════════════════════════
# Test 9: Noise Correlation Structure
# ═══════════════════════════════════════════════════════════════════════════

def test_noise_correlations(u: np.ndarray, K: int) -> dict:
    """Analyze residual correlations after model prediction."""
    T, N = u.shape
    mu = u[K:].mean(0); sig = u[K:].std(0) + 1e-8
    u_n = ((u - mu) / sig).astype(np.float32)
    
    X_lag = build_lagged(u_n, K)
    tr_idx = np.arange(K, T)
    y = u_n[tr_idx]
    
    ridge = train_ridge(X_lag[tr_idx], y)
    pred = ridge.predict(X_lag[tr_idx])
    residuals = y - pred
    
    # Noise correlation matrix
    noise_corr = np.corrcoef(residuals.T)
    np.fill_diagonal(noise_corr, 0)
    
    return {
        "noise_corr_matrix": noise_corr,
        "mean_noise_corr": np.abs(noise_corr).mean(),
        "max_noise_corr_pairs": np.unravel_index(
            np.argsort(np.abs(noise_corr).ravel())[-5:], noise_corr.shape
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Test 10: Transformer Baseline
# ═══════════════════════════════════════════════════════════════════════════

def test_transformer_baseline(u: np.ndarray, K: int, device: str) -> dict:
    """Compare Transformer to Ridge/MLP."""
    T, N = u.shape
    mu = u[K:].mean(0); sig = u[K:].std(0) + 1e-8
    u_n = ((u - mu) / sig).astype(np.float32)
    
    tr_idx = np.arange(K, T)
    y = u[tr_idx]
    
    # Ridge
    X_lag = build_lagged(u_n, K)
    ridge = train_ridge(X_lag[tr_idx], u_n[tr_idx])
    pred_ridge = ridge.predict(X_lag[tr_idx]) * sig + mu
    r2_ridge = compute_r2_per_dim(y, pred_ridge)
    
    # MLP
    mlp = train_mlp(X_lag[tr_idx], u_n[tr_idx], device)
    with torch.no_grad():
        pred_mlp = mlp(torch.tensor(X_lag[tr_idx], dtype=torch.float32)).numpy() * sig + mu
    r2_mlp = compute_r2_per_dim(y, pred_mlp)
    
    # Transformer
    _, pred_trf, y_trf = train_transformer(u, K, device)
    r2_trf = compute_r2_per_dim(y_trf, pred_trf)
    
    return {
        "r2_ridge": r2_ridge,
        "r2_mlp": r2_mlp,
        "r2_transformer": r2_trf,
        "r2_ridge_mean": r2_ridge.mean(),
        "r2_mlp_mean": r2_mlp.mean(),
        "r2_trf_mean": r2_trf.mean(),
        "trf_gain_over_ridge": r2_trf.mean() - r2_ridge.mean(),
        "trf_gain_over_mlp": r2_trf.mean() - r2_mlp.mean(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_all_tests(results: dict, out_dir: Path, worm_id: str):
    """Generate plots for all tests."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Self vs Non-Self
    fig, ax = plt.subplots(figsize=(8, 6))
    r = results["self_vs_nonself"]
    x = np.arange(len(r["r2_self"]))
    ax.bar(x - 0.2, r["r2_self"], 0.2, label="Self", alpha=0.8)
    ax.bar(x, r["r2_nonself"], 0.2, label="Non-Self", alpha=0.8)
    ax.bar(x + 0.2, r["r2_all"], 0.2, label="All", alpha=0.8)
    ax.set_xlabel("Neuron")
    ax.set_ylabel("R²")
    ax.set_title(f"Self vs Non-Self Prediction\n{worm_id}")
    ax.legend()
    fig.savefig(out_dir / "01_self_vs_nonself.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # 2. Ridge vs MLP
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    r = results["ridge_vs_mlp"]
    axes[0].scatter(r["r2_ridge"], r["r2_mlp"], alpha=0.6)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set_xlabel("R² Ridge")
    axes[0].set_ylabel("R² MLP")
    axes[0].set_title("Ridge vs MLP per Neuron")
    
    axes[1].bar(["Ridge", "MLP"], [r["r2_ridge_mean"], r["r2_mlp_mean"]])
    axes[1].set_ylabel("Mean R²")
    axes[1].set_title(f"MLP Gain: {r['mlp_gain']:.4f}")
    fig.suptitle(f"Nonlinearity Test - {worm_id}")
    fig.savefig(out_dir / "02_ridge_vs_mlp.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # 2b. Nonlinearity Timescales
    if "nonlinearity_timescales" in results:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        r = results["nonlinearity_timescales"]
        
        # Top-left: MLP gain vs prediction horizon
        ax = axes[0, 0]
        horizons = sorted(r["mlp_gain_by_horizon"].keys())
        gains = [r["mlp_gain_by_horizon"][h] for h in horizons]
        ax.plot(horizons, gains, "o-", linewidth=2, markersize=8, color="tab:blue")
        ax.axhline(0, color="k", linestyle="--", alpha=0.5)
        ax.set_xlabel("Prediction Horizon (steps)")
        ax.set_ylabel("MLP Gain over Ridge (ΔR²)")
        ax.set_title("Nonlinearity vs Horizon\n(>0 = MLP helps, <0 = Ridge better)")
        ax.grid(True, alpha=0.3)
        
        # Top-right: MLP gain vs lag window
        ax = axes[0, 1]
        lags = sorted(r["mlp_gain_by_lag"].keys())
        gains = [r["mlp_gain_by_lag"][k] for k in lags]
        ax.plot(lags, gains, "s-", linewidth=2, markersize=8, color="tab:orange")
        ax.axhline(0, color="k", linestyle="--", alpha=0.5)
        ax.set_xlabel("Lag Window (K)")
        ax.set_ylabel("MLP Gain over Ridge (ΔR²)")
        ax.set_title("Nonlinearity vs Lag Window\n(>0 = MLP helps, <0 = Ridge better)")
        ax.grid(True, alpha=0.3)
        
        # Bottom-left: Per-neuron nonlinearity score
        ax = axes[1, 0]
        per_neuron = np.array(r["per_neuron_gain"])
        sorted_idx = np.argsort(per_neuron)[::-1]
        colors = ["red" if per_neuron[i] > 0 else "blue" for i in sorted_idx]
        ax.bar(range(len(sorted_idx)), per_neuron[sorted_idx], color=colors)
        ax.axhline(0, color="k", linestyle="--", alpha=0.5)
        ax.set_xlabel("Neuron (sorted by nonlinearity)")
        ax.set_ylabel("Avg MLP Gain")
        ax.set_title(f"Per-Neuron Nonlinearity\n({np.sum(per_neuron > 0)}/{len(per_neuron)} neurons favor MLP)")
        
        # Bottom-right: Best horizon/lag combos
        ax = axes[1, 1]
        nonlinear_neurons = r.get("nonlinear_neurons", [])
        if len(nonlinear_neurons) > 0:
            text = f"Most Nonlinear Neurons (MLP > Ridge + 0.01):\n"
            # Sort by gain
            gains_sorted = sorted([(idx, per_neuron[idx]) for idx in nonlinear_neurons], 
                                 key=lambda x: -x[1])
            for i, (idx, gain) in enumerate(gains_sorted[:10]):
                text += f"  Neuron {idx}: +{gain:.4f}\n"
            ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=10, 
                   verticalalignment='top', fontfamily='monospace')
        else:
            ax.text(0.5, 0.5, "No neurons consistently favor MLP", 
                   transform=ax.transAxes, ha='center', va='center')
        ax.axis('off')
        ax.set_title("Neurons with Consistent Nonlinear Dynamics")
        
        fig.suptitle(f"Nonlinearity Across Timescales - {worm_id}")
        fig.tight_layout()
        fig.savefig(out_dir / "02b_nonlinearity_timescales.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    
    # 3. Symmetry
    fig, ax = plt.subplots(figsize=(8, 8))
    r = results["symmetry"]
    im = ax.imshow(r["R2_matrix"], cmap="viridis", vmin=0)
    ax.set_xlabel("Predictor Neuron j")
    ax.set_ylabel("Target Neuron i")
    ax.set_title(f"Pairwise R² Matrix\nSymmetry corr: {r['symmetry_correlation']:.3f}")
    plt.colorbar(im, ax=ax, label="R²")
    fig.savefig(out_dir / "03_symmetry_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # 4. Hub Neurons
    fig, ax = plt.subplots(figsize=(10, 5))
    r = results["hub_neurons"]
    # Only plot sampled neurons
    sample_idx = r.get("sample_idx", np.arange(len(r["importance"])))
    importance_sampled = r["importance"][sample_idx]
    sorted_idx = np.argsort(importance_sampled)[::-1]
    ax.bar(range(len(sorted_idx)), importance_sampled[sorted_idx])
    ax.set_xlabel("Neuron (sorted by importance, sampled)")
    ax.set_ylabel("ΔR² when removed")
    ax.set_title(f"Hub Neuron Importance - {worm_id}")
    fig.savefig(out_dir / "04_hub_neurons.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # 5. Lag Analysis
    fig, ax = plt.subplots(figsize=(8, 5))
    r = results["lag_contribution"]
    lags = sorted(r["r2_by_lag"].keys())
    r2s = [r["r2_by_lag"][k] for k in lags]
    ax.plot(lags, r2s, "o-", linewidth=2, markersize=8)
    ax.set_xlabel("Number of Lags (K)")
    ax.set_ylabel("Mean R²")
    ax.set_title(f"Lag Contribution - {worm_id}")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "05_lag_contribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # 6. Granger Causality
    fig, ax = plt.subplots(figsize=(8, 8))
    r = results["granger_causality"]
    im = ax.imshow(r["granger_matrix"], cmap="hot", vmin=0)
    ax.set_xlabel("Source Neuron j")
    ax.set_ylabel("Target Neuron i")
    ax.set_title(f"Granger Causality Matrix - {worm_id}")
    plt.colorbar(im, ax=ax, label="ΔR²")
    fig.savefig(out_dir / "06_granger_causality.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # 7. Behavior Coupling
    fig, ax = plt.subplots(figsize=(10, 5))
    r = results["behavior_coupling"]
    # Only plot sampled neurons
    sample_idx = r.get("sample_idx", np.arange(len(r["beh_coupling"])))
    beh_sampled = r["beh_coupling"][sample_idx]
    sorted_idx = np.argsort(beh_sampled)[::-1]
    colors = ["red" if beh_sampled[i] > 0.01 else "blue" for i in sorted_idx]
    ax.bar(range(len(sorted_idx)), beh_sampled[sorted_idx], color=colors)
    ax.set_xlabel("Neuron (sorted by coupling, sampled)")
    ax.set_ylabel("ΔR² with behavior")
    ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax.set_title(f"Behavior Coupling - {worm_id}")
    fig.savefig(out_dir / "07_behavior_coupling.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # 8. Horizon Decay
    fig, ax = plt.subplots(figsize=(8, 5))
    r = results["horizon_decay"]
    horizons = sorted(r["r2_by_horizon"].keys())
    r2s = [r["r2_by_horizon"][h] for h in horizons]
    ax.semilogy(horizons, np.maximum(r2s, 1e-4), "o-", linewidth=2, markersize=8)
    ax.set_xlabel("Prediction Horizon (steps)")
    ax.set_ylabel("Mean R² (log)")
    ax.set_title(f"Prediction Horizon Decay - {worm_id}")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "08_horizon_decay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # 9. Noise Correlations
    fig, ax = plt.subplots(figsize=(8, 8))
    r = results["noise_correlations"]
    im = ax.imshow(r["noise_corr_matrix"], cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    ax.set_xlabel("Neuron")
    ax.set_ylabel("Neuron")
    ax.set_title(f"Noise Correlation Matrix\nMean |corr|: {r['mean_noise_corr']:.3f}")
    plt.colorbar(im, ax=ax, label="Correlation")
    fig.savefig(out_dir / "09_noise_correlations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # 10. Model Comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    r = results["transformer_baseline"]
    
    # Per-neuron comparison
    N = len(r["r2_ridge"])
    x = np.arange(N)
    width = 0.25
    axes[0].bar(x - width, r["r2_ridge"], width, label="Ridge", alpha=0.8)
    axes[0].bar(x, r["r2_mlp"], width, label="MLP", alpha=0.8)
    axes[0].bar(x + width, r["r2_transformer"], width, label="Transformer", alpha=0.8)
    axes[0].set_xlabel("Neuron")
    axes[0].set_ylabel("R²")
    axes[0].legend()
    axes[0].set_title("Per-Neuron R²")
    
    # Summary bars
    axes[1].bar(["Ridge", "MLP", "Transformer"], 
                [r["r2_ridge_mean"], r["r2_mlp_mean"], r["r2_trf_mean"]])
    axes[1].set_ylabel("Mean R²")
    axes[1].set_title(f"TRF gain over Ridge: {r['trf_gain_over_ridge']:.4f}\n"
                      f"TRF gain over MLP: {r['trf_gain_over_mlp']:.4f}")
    fig.suptitle(f"Model Comparison - {worm_id}")
    fig.savefig(out_dir / "10_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"  Saved plots to {out_dir}")


def plot_summary_across_worms(all_results: list, out_dir: Path):
    """Create summary plots across all worms."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    n_worms = len(all_results)
    
    # Summary figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Self vs Non-Self contribution
    ax = axes[0, 0]
    self_vals = [r["self_vs_nonself"]["self_contribution"] for r in all_results]
    nonself_vals = [r["self_vs_nonself"]["nonself_contribution"] for r in all_results]
    x = np.arange(n_worms)
    ax.bar(x - 0.2, self_vals, 0.4, label="Self", alpha=0.8)
    ax.bar(x + 0.2, nonself_vals, 0.4, label="Non-Self", alpha=0.8)
    ax.set_xlabel("Worm")
    ax.set_ylabel("Mean R²")
    ax.set_title("Self vs Non-Self Prediction")
    ax.legend()
    
    # 2. Ridge vs MLP gap
    ax = axes[0, 1]
    ridge_vals = [r["ridge_vs_mlp"]["r2_ridge_mean"] for r in all_results]
    mlp_vals = [r["ridge_vs_mlp"]["r2_mlp_mean"] for r in all_results]
    ax.bar(x - 0.2, ridge_vals, 0.4, label="Ridge", alpha=0.8)
    ax.bar(x + 0.2, mlp_vals, 0.4, label="MLP", alpha=0.8)
    ax.set_xlabel("Worm")
    ax.set_ylabel("Mean R²")
    ax.set_title("Ridge vs MLP (Nonlinearity)")
    ax.legend()
    
    # 3. Symmetry scores
    ax = axes[0, 2]
    sym_vals = [r["symmetry"]["symmetry_correlation"] for r in all_results]
    ax.bar(range(n_worms), sym_vals)
    ax.axhline(0.5, color="r", linestyle="--", label="Symmetric threshold")
    ax.set_xlabel("Worm")
    ax.set_ylabel("Symmetry Correlation")
    ax.set_title("Gap Junction Signature")
    ax.legend()
    
    # 4. Behavior coupling
    ax = axes[1, 0]
    beh_vals = [r["behavior_coupling"]["mean_beh_gain"] for r in all_results]
    ax.bar(range(n_worms), beh_vals)
    ax.set_xlabel("Worm")
    ax.set_ylabel("Mean ΔR²")
    ax.set_title("Behavior Coupling Gain")
    
    # 5. Model comparison
    ax = axes[1, 1]
    ridge_vals = [r["transformer_baseline"]["r2_ridge_mean"] for r in all_results]
    mlp_vals = [r["transformer_baseline"]["r2_mlp_mean"] for r in all_results]
    trf_vals = [r["transformer_baseline"]["r2_trf_mean"] for r in all_results]
    width = 0.25
    ax.bar(x - width, ridge_vals, width, label="Ridge", alpha=0.8)
    ax.bar(x, mlp_vals, width, label="MLP", alpha=0.8)
    ax.bar(x + width, trf_vals, width, label="Transformer", alpha=0.8)
    ax.set_xlabel("Worm")
    ax.set_ylabel("Mean R²")
    ax.set_title("Model Comparison")
    ax.legend()
    
    # 6. Noise correlations
    ax = axes[1, 2]
    noise_vals = [r["noise_correlations"]["mean_noise_corr"] for r in all_results]
    ax.bar(range(n_worms), noise_vals)
    ax.set_xlabel("Worm")
    ax.set_ylabel("Mean |Noise Corr|")
    ax.set_title("Unmodeled Shared Input")
    
    fig.tight_layout()
    fig.savefig(out_dir / "summary_across_worms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Saved summary to {out_dir / 'summary_across_worms.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def run_all_tests_for_worm(h5_path: str, device: str, out_dir: Path) -> dict:
    """Run all constraint tests for one worm."""
    print(f"\n{'='*60}")
    print(f"Processing: {Path(h5_path).stem}")
    print(f"{'='*60}")
    
    worm_data = load_worm_data(h5_path, n_beh_modes=6)
    u, b, worm_id = worm_data["u"], worm_data["b"], worm_data["worm_id"]
    
    T, N = u.shape
    K = 15
    
    print(f"  T={T}, N={N}, K={K}")
    
    results = {"worm_id": worm_id, "T": T, "N": N, "K": K}
    
    # Run all tests
    print("  [1/11] Self vs Non-Self...")
    results["self_vs_nonself"] = test_self_vs_nonself(u, K, device)
    
    print("  [2/11] Ridge vs MLP...")
    results["ridge_vs_mlp"] = test_ridge_vs_mlp(u, K, device)
    
    print("  [2b/11] Nonlinearity Timescales...")
    results["nonlinearity_timescales"] = test_nonlinearity_timescales(u, device)
    
    print("  [3/11] Symmetry Test...")
    results["symmetry"] = test_symmetry(u, K)
    
    print("  [4/11] Hub Neurons...")
    results["hub_neurons"] = test_hub_neurons(u, K)
    
    print("  [5/11] Lag Contribution...")
    results["lag_contribution"] = test_lag_contribution(u, device)
    
    print("  [6/11] Granger Causality...")
    results["granger_causality"] = test_granger_causality(u, K)
    
    print("  [7/11] Behavior Coupling...")
    results["behavior_coupling"] = test_behavior_coupling(u, b, K)
    
    print("  [8/11] Horizon Decay...")
    results["horizon_decay"] = test_horizon_decay(u, K, device)
    
    print("  [9/11] Noise Correlations...")
    results["noise_correlations"] = test_noise_correlations(u, K)
    
    print("  [10/11] Transformer Baseline...")
    results["transformer_baseline"] = test_transformer_baseline(u, K, device)
    
    # Plot
    worm_out = out_dir / worm_id
    plot_all_tests(results, worm_out, worm_id)
    
    # Save JSON (convert numpy arrays)
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        return obj
    
    with open(worm_out / "results.json", "w") as f:
        json.dump(convert(results), f, indent=2)
    
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_worms", type=int, default=10)
    ap.add_argument("--out_dir", default="output_plots/stage2_constraint_tests")
    args = ap.parse_args()
    
    data_dir = Path("data/used/behaviour+neuronal activity atanas (2023)/2")
    h5_files = sorted(data_dir.glob("*.h5"))[:args.n_worms]
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running Stage 2 Constraint Tests on {len(h5_files)} worms")
    print(f"Device: {args.device}")
    print(f"Output: {out_dir}")
    
    all_results = []
    for h5_path in h5_files:
        try:
            results = run_all_tests_for_worm(str(h5_path), args.device, out_dir)
            all_results.append(results)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    # Summary across worms
    if len(all_results) > 1:
        plot_summary_across_worms(all_results, out_dir)
    
    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Worm':<20} {'Self R²':>8} {'NonSelf':>8} {'Ridge':>8} {'MLP':>8} {'TRF':>8} {'Sym':>6}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['worm_id']:<20} "
              f"{r['self_vs_nonself']['self_contribution']:>8.3f} "
              f"{r['self_vs_nonself']['nonself_contribution']:>8.3f} "
              f"{r['transformer_baseline']['r2_ridge_mean']:>8.3f} "
              f"{r['transformer_baseline']['r2_mlp_mean']:>8.3f} "
              f"{r['transformer_baseline']['r2_trf_mean']:>8.3f} "
              f"{r['symmetry']['symmetry_correlation']:>6.2f}")
    
    print(f"\nDone! Results in {out_dir}")


if __name__ == "__main__":
    main()
