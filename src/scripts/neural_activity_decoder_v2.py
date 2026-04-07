#!/usr/bin/env python3
"""
Neural Activity Decoder v2 — Mirror of behaviour_decoder_models_v8.py

Analogous to behaviour_decoder_models_v8.py but predicts **neural activity**
from other neurons instead of behaviour from neural.

Models:
  • Ridge, MLP, Transformer (B_wide_256h8 - matching behaviour decoder)

Conditions:
  • self_only (AR): predict from own history only
  • cross_only: predict from other neurons only (no self-history)
  • conc+self: concurrent snapshot + own history
  • cross_only_fr: free-run prediction (no GT clamping)

Plots:
  • Lag sweep: R² and Corr vs K (like behaviour decoder)
  • Traces: GT vs Pred for select neurons
  • Summary violin: per-neuron R² across models

Author: Copilot
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from baseline_transformer.config import TransformerBaselineConfig
from baseline_transformer.dataset import load_worm_data
from baseline_transformer.model import TemporalTransformerGaussian

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

K_VALUES = [1, 3, 5, 10, 15, 20]
N_FOLDS = 5
_RIDGE_ALPHAS = np.logspace(-4, 6, 30)


def _make_B_wide_config(context_length=10):
    """Create B_wide_256h8 transformer config (matching behaviour decoder)."""
    cfg = TransformerBaselineConfig()
    cfg.d_model = 256
    cfg.n_heads = 8
    cfg.n_layers = 2
    cfg.d_ff = 512
    cfg.dropout = 0.1
    cfg.context_length = context_length
    cfg.lr = 1e-3
    cfg.weight_decay = 1e-4
    cfg.sigma_min = 1e-4
    cfg.sigma_max = 10.0
    cfg.max_epochs = 200
    cfg.patience = 25
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def _r2(gt, pred):
    ss_res = np.nansum((gt - pred) ** 2)
    ss_tot = np.nansum((gt - np.nanmean(gt)) ** 2) + 1e-12
    return 1 - ss_res / ss_tot


def _corr(gt, pred):
    valid = np.isfinite(gt) & np.isfinite(pred)
    if valid.sum() < 3:
        return np.nan
    return np.corrcoef(gt[valid], pred[valid])[0, 1]


def neural_metrics_heldout(pred, gt):
    """Compute R² and correlation per neuron."""
    N = pred.shape[1]
    r2_per = np.array([_r2(gt[:, i], pred[:, i]) for i in range(N)])
    corr_per = np.array([_corr(gt[:, i], pred[:, i]) for i in range(N)])
    return r2_per, corr_per


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _make_folds(T, warmup, n_folds=5):
    """Create fold boundaries for temporal CV."""
    fold_size = (T - warmup) // n_folds
    folds = []
    for i in range(n_folds):
        ts = warmup + i * fold_size
        te = warmup + (i + 1) * fold_size if i < n_folds - 1 else T
        folds.append((ts, te))
    return folds


def _get_train_indices(T, warmup, ts, te, buffer):
    """Get training indices with buffer zone after test fold."""
    before = np.arange(warmup, ts)
    after_start = min(te + buffer, T)
    after = np.arange(after_start, T)
    return np.concatenate([before, after])


def _build_lagged(x, K):
    """Build lagged feature matrix."""
    T, D = x.shape
    out = np.zeros((T, K * D), dtype=x.dtype)
    for lag in range(1, K + 1):
        out[lag:, (lag-1)*D:lag*D] = x[:-lag]
    return out


def _build_windows(u, indices, K):
    """Build context windows for transformer."""
    windows = []
    for t in indices:
        t = int(t)  # Ensure integer indexing
        if t >= K:
            windows.append(u[t - K:t])
        else:
            # Pad with zeros if not enough history
            pad = np.zeros((K - t, u.shape[1]), dtype=u.dtype)
            if t > 0:
                windows.append(np.vstack([pad, u[:t]]))
            else:
                windows.append(pad)
    return np.stack(windows)


# ══════════════════════════════════════════════════════════════════════════════
# MLP (matching behaviour decoder architecture)
# ══════════════════════════════════════════════════════════════════════════════

def _make_mlp(d_in, d_out, hidden=128, n_layers=2):
    """Create MLP with LayerNorm and Dropout (matching behaviour decoder)."""
    layers, d = [], d_in
    for _ in range(n_layers):
        layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(0.1)]
        d = hidden
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


def _train_mlp(X_tr, y_tr, X_val, y_val, d_out, device, epochs=150, lr=1e-3, wd=1e-3, patience=20):
    """Train MLP with early stopping."""
    d_in = X_tr.shape[1]
    Xt = torch.tensor(X_tr, dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=device)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.float32, device=device)
    
    mlp = _make_mlp(d_in, d_out).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=wd)
    
    bvl, bs, pat = float('inf'), None, 0
    for _ in range(epochs):
        mlp.train()
        loss = F.mse_loss(mlp(Xt), yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        mlp.eval()
        with torch.no_grad():
            vl = F.mse_loss(mlp(Xv), yv).item()
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


# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORMER (B_wide_256h8 matching behaviour decoder)
# ══════════════════════════════════════════════════════════════════════════════

def _train_transformer(X_tr, y_tr, X_val, y_val, N, K, device, epochs=200, patience=25):
    """Train transformer with early stopping."""
    Xt = torch.tensor(X_tr, dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=device)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.float32, device=device)
    
    cfg = _make_B_wide_config(context_length=K)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = TemporalTransformerGaussian(n_neural=N, n_beh=0, cfg=cfg).to(device)
    
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    bvl, bs, pat = float('inf'), None, 0
    for _ in range(epochs):
        model.train()
        pred_mu, _, _, _ = model.forward(Xt, return_all_steps=False)
        loss = F.mse_loss(pred_mu, yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        model.eval()
        with torch.no_grad():
            pred_mu_v, _, _, _ = model.forward(Xv, return_all_steps=False)
            vl = F.mse_loss(pred_mu_v, yv).item()
        
        if vl < bvl - 1e-6:
            bvl, bs, pat = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            pat += 1
        if pat > patience:
            break
    
    if bs:
        model.load_state_dict(bs)
    model.eval().cpu()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# RIDGE REGRESSION
# ══════════════════════════════════════════════════════════════════════════════

from sklearn.linear_model import RidgeCV


def _cv_ridge_self(u, K):
    """Ridge: self-history only → neural prediction (AR model)."""
    T, N = u.shape
    warmup = K
    ho = np.full((T, N), np.nan)
    
    for ts, te in _make_folds(T, warmup):
        tr = _get_train_indices(T, warmup, ts, te, buffer=K)
        
        mu, sig = u[tr].mean(0), u[tr].std(0) + 1e-8
        u_n = ((u - mu) / sig).astype(np.float32)
        
        # For each neuron, use only its own history
        for ni in range(N):
            X_self = _build_lagged(u_n[:, ni:ni+1], K)
            y = u_n[:, ni]
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = RidgeCV(alphas=_RIDGE_ALPHAS).fit(X_self[tr], y[tr])
            
            ho[ts:te, ni] = model.predict(X_self[ts:te]) * sig[ni] + mu[ni]
    
    return ho


def _cv_ridge_cross(u, K):
    """Ridge: cross-neuron only → neural prediction (no self-history)."""
    T, N = u.shape
    warmup = K
    ho = np.full((T, N), np.nan)
    
    for ts, te in _make_folds(T, warmup):
        tr = _get_train_indices(T, warmup, ts, te, buffer=K)
        
        mu, sig = u[tr].mean(0), u[tr].std(0) + 1e-8
        u_n = ((u - mu) / sig).astype(np.float32)
        
        X_all = _build_lagged(u_n, K)
        
        for ni in range(N):
            # Exclude self-history columns
            mask = np.ones(K * N, dtype=bool)
            for lag in range(K):
                mask[lag * N + ni] = False
            X_cross = X_all[:, mask]
            y = u_n[:, ni]
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = RidgeCV(alphas=_RIDGE_ALPHAS).fit(X_cross[tr], y[tr])
            
            ho[ts:te, ni] = model.predict(X_cross[ts:te]) * sig[ni] + mu[ni]
    
    return ho


def _cv_ridge_full(u, K):
    """Ridge: all history (self + cross) → neural prediction."""
    T, N = u.shape
    warmup = K
    ho = np.full((T, N), np.nan)
    
    for ts, te in _make_folds(T, warmup):
        tr = _get_train_indices(T, warmup, ts, te, buffer=K)
        
        mu, sig = u[tr].mean(0), u[tr].std(0) + 1e-8
        u_n = ((u - mu) / sig).astype(np.float32)
        
        X_all = _build_lagged(u_n, K)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = RidgeCV(alphas=_RIDGE_ALPHAS).fit(X_all[tr], u_n[tr])
        
        ho[ts:te] = model.predict(X_all[ts:te]) * sig + mu
    
    return ho


# ══════════════════════════════════════════════════════════════════════════════
# MLP CV
# ══════════════════════════════════════════════════════════════════════════════

def _cv_mlp_self(u, K, device):
    """MLP: self-history only → neural prediction."""
    T, N = u.shape
    warmup = K
    ho = np.full((T, N), np.nan)
    
    for ts, te in _make_folds(T, warmup):
        tr = _get_train_indices(T, warmup, ts, te, buffer=K)
        
        mu, sig = u[tr].mean(0), u[tr].std(0) + 1e-8
        u_n = ((u - mu) / sig).astype(np.float32)
        
        for ni in range(N):
            X_self = _build_lagged(u_n[:, ni:ni+1], K)
            y = u_n[:, ni:ni+1]
            
            # Inner val split
            nv = max(10, int(len(tr) * 0.15))
            tr_inner, val_inner = tr[:-nv], tr[-nv:]
            
            mlp = _train_mlp(X_self[tr_inner], y[tr_inner], 
                           X_self[val_inner], y[val_inner], 1, device)
            
            with torch.no_grad():
                pred = mlp(torch.tensor(X_self[ts:te], dtype=torch.float32)).numpy()
            ho[ts:te, ni] = pred.ravel() * sig[ni] + mu[ni]
    
    return ho


def _cv_mlp_cross(u, K, device):
    """MLP: cross-neuron only → neural prediction."""
    T, N = u.shape
    warmup = K
    ho = np.full((T, N), np.nan)
    
    for ts, te in _make_folds(T, warmup):
        tr = _get_train_indices(T, warmup, ts, te, buffer=K)
        
        mu, sig = u[tr].mean(0), u[tr].std(0) + 1e-8
        u_n = ((u - mu) / sig).astype(np.float32)
        
        X_all = _build_lagged(u_n, K)
        
        for ni in range(N):
            # Exclude self-history
            mask = np.ones(K * N, dtype=bool)
            for lag in range(K):
                mask[lag * N + ni] = False
            X_cross = X_all[:, mask]
            y = u_n[:, ni:ni+1]
            
            nv = max(10, int(len(tr) * 0.15))
            tr_inner, val_inner = tr[:-nv], tr[-nv:]
            
            mlp = _train_mlp(X_cross[tr_inner], y[tr_inner],
                           X_cross[val_inner], y[val_inner], 1, device)
            
            with torch.no_grad():
                pred = mlp(torch.tensor(X_cross[ts:te], dtype=torch.float32)).numpy()
            ho[ts:te, ni] = pred.ravel() * sig[ni] + mu[ni]
    
    return ho


def _cv_mlp_full(u, K, device):
    """MLP: all history → neural prediction."""
    T, N = u.shape
    warmup = K
    ho = np.full((T, N), np.nan)
    
    for ts, te in _make_folds(T, warmup):
        tr = _get_train_indices(T, warmup, ts, te, buffer=K)
        
        mu, sig = u[tr].mean(0), u[tr].std(0) + 1e-8
        u_n = ((u - mu) / sig).astype(np.float32)
        
        X_all = _build_lagged(u_n, K)
        
        nv = max(10, int(len(tr) * 0.15))
        tr_inner, val_inner = tr[:-nv], tr[-nv:]
        
        mlp = _train_mlp(X_all[tr_inner], u_n[tr_inner],
                        X_all[val_inner], u_n[val_inner], N, device)
        
        with torch.no_grad():
            pred = mlp(torch.tensor(X_all[ts:te], dtype=torch.float32)).numpy()
        ho[ts:te] = pred * sig + mu
    
    return ho


# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORMER CV
# ══════════════════════════════════════════════════════════════════════════════

def _cv_trf_full(u, K, device):
    """Transformer: all history → neural prediction."""
    T, N = u.shape
    warmup = K
    ho = np.full((T, N), np.nan)
    
    if K < 2:
        return ho  # Transformer needs K >= 2
    
    for ts, te in _make_folds(T, warmup):
        tr = _get_train_indices(T, warmup, ts, te, buffer=K)
        
        mu, sig = u[tr].mean(0), u[tr].std(0) + 1e-8
        u_n = ((u - mu) / sig).astype(np.float32)
        
        nv = max(10, int(len(tr) * 0.15))
        tr_inner, val_inner = tr[:-nv], tr[-nv:]
        
        # Build windows: for index t, use context [t-K:t] to predict u[t]
        X_tr = _build_windows(u_n, tr_inner, K)
        y_tr = u_n[tr_inner]
        X_val = _build_windows(u_n, val_inner, K)
        y_val = u_n[val_inner]
        
        model = _train_transformer(X_tr, y_tr, X_val, y_val, N, K, device)
        
        X_te = _build_windows(u_n, np.arange(ts, te), K)
        with torch.no_grad():
            pred_mu, _, _, _ = model(torch.tensor(X_te, dtype=torch.float32))
            pred = pred_mu.numpy()
        ho[ts:te] = pred * sig + mu
    
    return ho


def _cv_trf_cross(u, K, device):
    """Transformer: cross-neuron only (zero out self in context)."""
    T, N = u.shape
    warmup = K
    ho = np.full((T, N), np.nan)
    
    if K < 2:
        return ho
    
    for ts, te in _make_folds(T, warmup):
        tr = _get_train_indices(T, warmup, ts, te, buffer=K)
        
        mu, sig = u[tr].mean(0), u[tr].std(0) + 1e-8
        u_n = ((u - mu) / sig).astype(np.float32)
        
        nv = max(10, int(len(tr) * 0.15))
        tr_inner, val_inner = tr[:-nv], tr[-nv:]
        
        # Build windows: for index t, use context [t-K:t] to predict u[t]
        X_tr = _build_windows(u_n, tr_inner, K)
        y_tr = u_n[tr_inner]
        X_val = _build_windows(u_n, val_inner, K)
        y_val = u_n[val_inner]
        
        model = _train_transformer(X_tr, y_tr, X_val, y_val, N, K, device)
        
        X_te_base = _build_windows(u_n, np.arange(ts, te), K)
        
        # For each neuron, zero out its self-history in context
        for ni in range(N):
            X_te = X_te_base.copy()
            X_te[:, :, ni] = 0.0  # Zero out self-history
            with torch.no_grad():
                pred_mu, _, _, _ = model(torch.tensor(X_te, dtype=torch.float32))
                ho[ts:te, ni] = pred_mu[:, ni].numpy() * sig[ni] + mu[ni]
    
    return ho


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def _plot_lag_sweep(results, out, worm_id, N, neuron_type="all"):
    """Line plot with 2 panels (R² and Correlation) - like behaviour decoder."""
    K_vals = sorted(results.keys())
    
    arch_colors = {"trf": "#d62728", "mlp": "#1f77b4", "ridge": "#2ca02c"}
    conditions = [
        ("full", "-", "o", "Full (self+cross)"),
        ("cross", "--", "s", "Cross-only"),
        ("self", ":", "^", "Self-only (AR)"),
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left panel: R²
    ax_r2 = axes[0]
    for arch in ["trf", "mlp", "ridge"]:
        for cond_key, ls, marker, label in conditions:
            key = f"{arch}_{cond_key}"
            ys = []
            xs = []
            for K in K_vals:
                res_k = results[K]
                if f"r2_mean_{key}" in res_k:
                    ys.append(res_k[f"r2_mean_{key}"])
                    xs.append(K)
            if ys:
                ax_r2.plot(xs, ys, ls=ls, marker=marker, ms=6, lw=1.5,
                          color=arch_colors[arch], 
                          label=f"{arch.upper()} {label}")
    
    ax_r2.set_xlabel("Context length K (lags)")
    ax_r2.set_ylabel("Mean R²")
    ax_r2.set_xticks(K_vals)
    ax_r2.set_ylim(-0.5, 1)
    ax_r2.axhline(0, color='k', lw=0.5, ls='--', alpha=0.5)
    ax_r2.grid(alpha=0.3)
    ax_r2.set_title("Neural Prediction R²")
    
    # Right panel: Correlation
    ax_corr = axes[1]
    for arch in ["trf", "mlp", "ridge"]:
        for cond_key, ls, marker, label in conditions:
            key = f"{arch}_{cond_key}"
            ys = []
            xs = []
            for K in K_vals:
                res_k = results[K]
                if f"corr_mean_{key}" in res_k:
                    ys.append(res_k[f"corr_mean_{key}"])
                    xs.append(K)
            if ys:
                ax_corr.plot(xs, ys, ls=ls, marker=marker, ms=6, lw=1.5,
                            color=arch_colors[arch],
                            label=f"{arch.upper()} {label}")
    
    ax_corr.set_xlabel("Context length K (lags)")
    ax_corr.set_ylabel("Mean Correlation")
    ax_corr.set_xticks(K_vals)
    ax_corr.set_ylim(0, 1)
    ax_corr.grid(alpha=0.3)
    ax_corr.set_title("Neural Prediction Correlation")
    ax_corr.legend(fontsize=7, loc='lower right', ncol=3)
    
    fig.suptitle(f"Neural Activity Decoder — {worm_id}  ({neuron_type}, N={N}, {N_FOLDS}-fold CV)",
                 fontsize=12, fontweight="bold")
    
    fname = f"lag_sweep_{worm_id}.png"
    fig.tight_layout()
    fig.savefig(out / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out / fname}")


def _plot_traces(gt, predictions, out, worm_id, K, N, n_show=8, frame_range=None):
    """Plot neural traces: GT vs Pred for select neurons."""
    T = gt.shape[0]
    
    if frame_range is None:
        start = max(K, T // 4)
        end = min(T, start + 300)
        frame_range = (start, end)
    
    fs, fe = frame_range
    t_axis = np.arange(fs, fe)
    
    # Select neurons to show (spread across range)
    show_idx = np.linspace(0, N-1, min(n_show, N), dtype=int)
    
    colors = {
        "trf_full": "#d62728",
        "mlp_full": "#1f77b4",
        "ridge_full": "#2ca02c",
    }
    
    n_models = len(predictions)
    fig, axes = plt.subplots(len(show_idx), n_models, figsize=(4 * n_models, 2 * len(show_idx)), 
                             sharex=True, squeeze=False)
    
    for col, (model_key, pred) in enumerate(predictions.items()):
        mc = colors.get(model_key, "#999")
        label = model_key.replace("_", " ").upper()
        
        for row, ni in enumerate(show_idx):
            ax = axes[row, col]
            ax.plot(t_axis, gt[fs:fe, ni], color="#333", lw=1.0, alpha=0.8, label="GT")
            ax.plot(t_axis, pred[fs:fe, ni], color=mc, lw=0.8, alpha=0.9, ls="--", label="Pred")
            
            r2 = _r2(gt[fs:fe, ni], pred[fs:fe, ni])
            ax.text(0.98, 0.95, f"R²={r2:.2f}", transform=ax.transAxes,
                   ha="right", va="top", fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
            
            if col == 0:
                ax.set_ylabel(f"n{ni}", fontsize=9)
            if row == 0:
                ax.set_title(label, fontsize=10, fontweight="bold")
            if row == len(show_idx) - 1:
                ax.set_xlabel("Time (frames)", fontsize=9)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper left")
    
    fig.suptitle(f"Neural Traces — {worm_id}  K={K}  (frames {fs}-{fe})",
                 fontsize=12, fontweight="bold")
    
    fname = f"traces_K{K}_{worm_id}.png"
    fig.tight_layout()
    fig.savefig(out / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out / fname}")


def _plot_violin_per_neuron(results, out, worm_id, N, labels=None):
    """Violin plot per neuron comparing models (like behaviour decoder)."""
    # Only use K = 1, 3, 5 for violin plots
    ks_all = sorted(results.keys())
    ks = [k for k in ks_all if k in [1, 3, 5]]
    if not ks:
        print("  Skipping violin plots: no K in [1,3,5] available")
        return
    
    models = ["Ridge", "MLP", "TRF"]
    model_keys = {"Ridge": "ridge_full", "MLP": "mlp_full", "TRF": "trf_full"}
    model_colors = {"Ridge": "#2ca02c", "MLP": "#1f77b4", "TRF": "#d62728"}
    
    # Collect per-neuron R² across K values
    data = {m: [] for m in models}  # model -> list over K of array(N,)
    for K in ks:
        res_k = results.get(K, {})
        for m in models:
            mk = model_keys[m]
            field = f"r2_per_neuron_{mk}"
            arr = res_k.get(field)
            if arr is not None:
                data[m].append(np.array(arr))
            else:
                data[m].append(np.full(N, np.nan))
    
    # Figure
    n_models = len(models)
    fig_w = max(14, N * n_models * 0.3 + 2)
    fig, ax = plt.subplots(figsize=(min(fig_w, 40), 6))
    
    group_width = 0.8
    vw = group_width / n_models
    neuron_positions = np.arange(N)
    
    for mi, m in enumerate(models):
        mc = model_colors[m]
        offset = (mi - (n_models - 1) / 2) * vw
        positions = neuron_positions + offset
        
        # Gather values for each neuron
        neuron_vals = []
        for ni in range(N):
            vals = [data[m][ki][ni] for ki in range(len(ks))
                    if not np.isnan(data[m][ki][ni])]
            neuron_vals.append(vals if vals else [np.nan])
        
        # Draw violins
        valid_pos = []
        valid_vals = []
        for ni in range(N):
            if len(neuron_vals[ni]) >= 2 and not any(np.isnan(neuron_vals[ni])):
                valid_pos.append(positions[ni])
                valid_vals.append(neuron_vals[ni])
        
        if valid_vals:
            parts = ax.violinplot(valid_vals, positions=valid_pos,
                                 widths=vw * 0.85, showmeans=False,
                                 showmedians=False, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(mc)
                pc.set_alpha(0.35)
                pc.set_edgecolor(mc)
                pc.set_linewidth(0.8)
        
        # Overlay dots
        for ni in range(N):
            nv = neuron_vals[ni]
            if nv and not any(np.isnan(nv)):
                jitter = np.random.default_rng(ni + mi * 1000).uniform(-vw * 0.18, vw * 0.18, size=len(nv))
                ax.scatter(np.full(len(nv), positions[ni]) + jitter, nv,
                          s=14, color=mc, alpha=0.75, edgecolors="white", linewidths=0.3, zorder=3)
        
        # Median line
        for ni in range(N):
            nv = neuron_vals[ni]
            if nv and not any(np.isnan(nv)):
                med = np.median(nv)
                ax.plot([positions[ni] - vw * 0.25, positions[ni] + vw * 0.25],
                       [med, med], color=mc, lw=1.5, zorder=4)
        
        ax.scatter([], [], s=40, color=mc, label=m)
    
    ax.set_xticks(neuron_positions)
    xlabels = labels if labels and len(labels) == N else [f"n{i}" for i in range(N)]
    ax.set_xticklabels(xlabels, fontsize=max(5, min(7, 180 // N)), rotation=45, ha="right")
    ax.set_xlabel("Neuron", fontsize=11)
    ax.set_ylabel("R² (each dot = one K value)", fontsize=11)
    ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    ax.set_ylim(bottom=-1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    ax.set_title(f"{worm_id} — Full (self+cross)\n"
                f"(N={N}, dots = K∈{{{','.join(str(k) for k in ks)}}}, {N_FOLDS}-fold CV)",
                fontsize=12, fontweight="bold")
    
    fig.tight_layout()
    fname = f"violin_full_{worm_id}.png"
    fig.savefig(out / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out / fname}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation(u, device, out, worm_id, neuron_type="all", labels=None):
    """Run all models for neural activity prediction."""
    T, N = u.shape
    results = {}
    trace_K = 5
    trace_preds = {}
    
    for K in K_VALUES:
        print(f"\n  K={K}")
        print(f"  {'-'*50}")
        res_k = {}
        
        # ── Ridge ──
        print(f"    Ridge self...")
        t0 = time.time()
        ho = _cv_ridge_self(u, K)
        r2, corr = neural_metrics_heldout(ho, u)
        res_k["r2_mean_ridge_self"] = float(np.nanmean(r2))
        res_k["corr_mean_ridge_self"] = float(np.nanmean(corr))
        res_k["r2_per_neuron_ridge_self"] = [float(x) for x in r2]
        res_k["corr_per_neuron_ridge_self"] = [float(x) for x in corr]
        print(f"      R²={np.nanmean(r2):.3f}  corr={np.nanmean(corr):.3f}  ({time.time()-t0:.0f}s)")
        
        print(f"    Ridge cross...")
        t0 = time.time()
        ho = _cv_ridge_cross(u, K)
        r2, corr = neural_metrics_heldout(ho, u)
        res_k["r2_mean_ridge_cross"] = float(np.nanmean(r2))
        res_k["corr_mean_ridge_cross"] = float(np.nanmean(corr))
        res_k["r2_per_neuron_ridge_cross"] = [float(x) for x in r2]
        res_k["corr_per_neuron_ridge_cross"] = [float(x) for x in corr]
        print(f"      R²={np.nanmean(r2):.3f}  corr={np.nanmean(corr):.3f}  ({time.time()-t0:.0f}s)")
        
        print(f"    Ridge full...")
        t0 = time.time()
        ho = _cv_ridge_full(u, K)
        r2, corr = neural_metrics_heldout(ho, u)
        res_k["r2_mean_ridge_full"] = float(np.nanmean(r2))
        res_k["corr_mean_ridge_full"] = float(np.nanmean(corr))
        res_k["r2_per_neuron_ridge_full"] = [float(x) for x in r2]
        res_k["corr_per_neuron_ridge_full"] = [float(x) for x in corr]
        print(f"      R²={np.nanmean(r2):.3f}  corr={np.nanmean(corr):.3f}  ({time.time()-t0:.0f}s)")
        if K == trace_K:
            trace_preds["ridge_full"] = ho.copy()
        
        # ── MLP ──
        print(f"    MLP self...")
        t0 = time.time()
        ho = _cv_mlp_self(u, K, device)
        r2, corr = neural_metrics_heldout(ho, u)
        res_k["r2_mean_mlp_self"] = float(np.nanmean(r2))
        res_k["corr_mean_mlp_self"] = float(np.nanmean(corr))
        res_k["r2_per_neuron_mlp_self"] = [float(x) for x in r2]
        res_k["corr_per_neuron_mlp_self"] = [float(x) for x in corr]
        print(f"      R²={np.nanmean(r2):.3f}  corr={np.nanmean(corr):.3f}  ({time.time()-t0:.0f}s)")
        
        print(f"    MLP cross...")
        t0 = time.time()
        ho = _cv_mlp_cross(u, K, device)
        r2, corr = neural_metrics_heldout(ho, u)
        res_k["r2_mean_mlp_cross"] = float(np.nanmean(r2))
        res_k["corr_mean_mlp_cross"] = float(np.nanmean(corr))
        res_k["r2_per_neuron_mlp_cross"] = [float(x) for x in r2]
        res_k["corr_per_neuron_mlp_cross"] = [float(x) for x in corr]
        print(f"      R²={np.nanmean(r2):.3f}  corr={np.nanmean(corr):.3f}  ({time.time()-t0:.0f}s)")
        
        print(f"    MLP full...")
        t0 = time.time()
        ho = _cv_mlp_full(u, K, device)
        r2, corr = neural_metrics_heldout(ho, u)
        res_k["r2_mean_mlp_full"] = float(np.nanmean(r2))
        res_k["corr_mean_mlp_full"] = float(np.nanmean(corr))
        res_k["r2_per_neuron_mlp_full"] = [float(x) for x in r2]
        res_k["corr_per_neuron_mlp_full"] = [float(x) for x in corr]
        print(f"      R²={np.nanmean(r2):.3f}  corr={np.nanmean(corr):.3f}  ({time.time()-t0:.0f}s)")
        if K == trace_K:
            trace_preds["mlp_full"] = ho.copy()
        
        # ── Transformer ──
        if K >= 2:
            print(f"    TRF full...")
            t0 = time.time()
            ho = _cv_trf_full(u, K, device)
            r2, corr = neural_metrics_heldout(ho, u)
            res_k["r2_mean_trf_full"] = float(np.nanmean(r2))
            res_k["corr_mean_trf_full"] = float(np.nanmean(corr))
            res_k["r2_per_neuron_trf_full"] = [float(x) for x in r2]
            res_k["corr_per_neuron_trf_full"] = [float(x) for x in corr]
            print(f"      R²={np.nanmean(r2):.3f}  corr={np.nanmean(corr):.3f}  ({time.time()-t0:.0f}s)")
            if K == trace_K:
                trace_preds["trf_full"] = ho.copy()
            
            print(f"    TRF cross...")
            t0 = time.time()
            ho = _cv_trf_cross(u, K, device)
            r2, corr = neural_metrics_heldout(ho, u)
            res_k["r2_mean_trf_cross"] = float(np.nanmean(r2))
            res_k["corr_mean_trf_cross"] = float(np.nanmean(corr))
            res_k["r2_per_neuron_trf_cross"] = [float(x) for x in r2]
            res_k["corr_per_neuron_trf_cross"] = [float(x) for x in corr]
            print(f"      R²={np.nanmean(r2):.3f}  corr={np.nanmean(corr):.3f}  ({time.time()-t0:.0f}s)")
        
        results[K] = res_k
        
        # Checkpoint
        with open(out / "results.json", "w") as f:
            json.dump(results, f, indent=2)
    
    # Generate plots
    print("\nGenerating plots...")
    _plot_lag_sweep(results, out, worm_id, N, neuron_type)
    
    if trace_preds:
        _plot_traces(u, trace_preds, out, worm_id, trace_K, N)
    
    _plot_violin_per_neuron(results, out, worm_id, N, labels)
    
    return results


def main():
    ap = argparse.ArgumentParser(description="Neural Activity Decoder v2 (behaviour decoder style)")
    ap.add_argument("--h5", required=True, help="H5 file path")
    ap.add_argument("--out_dir", default="output_plots/neural_activity_decoder_v2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--neurons", default="motor", choices=["motor", "nonmotor", "all"])
    ap.add_argument("--replot", action="store_true", help="Regenerate plots from existing results")
    args = ap.parse_args()
    
    device = args.device
    base_out = Path(args.out_dir)
    
    # Load data
    worm_data = load_worm_data(args.h5, n_beh_modes=6)
    u_all = worm_data["u"]
    worm_id = worm_data["worm_id"]
    motor_idx = worm_data.get("motor_idx")
    labels = worm_data.get("labels", [])
    N_total = u_all.shape[1]
    
    if motor_idx is None or len(motor_idx) == 0:
        motor_idx = []
    
    motor_set = set(motor_idx)
    if args.neurons == "motor":
        sel_idx = sorted(motor_idx)
        subset_tag = "motor"
    elif args.neurons == "nonmotor":
        sel_idx = sorted(i for i in range(N_total) if i not in motor_set)
        subset_tag = "nonmotor"
    else:
        sel_idx = list(range(N_total))
        subset_tag = "all"
    
    if not sel_idx:
        print(f"ERROR: no {args.neurons} neurons found")
        sys.exit(1)
    
    sel_labels = [labels[i] if i < len(labels) else f"n{i}" for i in sel_idx]
    u = u_all[:, sel_idx].astype(np.float32)
    T, N = u.shape
    
    out = base_out / f"{worm_id}_{subset_tag}"
    out.mkdir(parents=True, exist_ok=True)
    
    worm_id_full = f"{worm_id}_{subset_tag}"
    
    print(f"{'='*70}")
    print(f"  WORM: {worm_id}  subset={subset_tag}  T={T}  N={N}")
    print(f"{'='*70}")
    
    results_path = out / "results.json"
    
    if args.replot:
        if not results_path.exists():
            print(f"No results.json at {results_path}")
            sys.exit(1)
        with open(results_path) as f:
            results = json.load(f)
        # Convert keys to int
        results = {int(k): v for k, v in results.items()}
        _plot_lag_sweep(results, out, worm_id_full, N, subset_tag)
        _plot_violin_per_neuron(results, out, worm_id_full, N, sel_labels)
        print("Replot done.")
        return
    
    t_start = time.time()
    results = run_evaluation(u, device, out, worm_id_full, subset_tag, sel_labels)
    total_s = time.time() - t_start
    
    print(f"\nTotal time: {total_s:.0f}s ({total_s/60:.1f}min)")
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
