#!/usr/bin/env python
"""Behaviour decoder comparison v8 — ALL MODELS PREDICT BEHAVIOR ONLY.

MODIFICATIONS:
1. All models predict BEHAVIOR ONLY (not neural+behavior)
2. n+b FR models re-clamp to GT every 10 seconds (~40 frames at 4Hz)
3. Ridge coefficient boundary checking added
4. Added smaller transformer option (d_model=64, n_heads=4, n_layers=1)
5. Run all neurons only

Models:
  • Ridge, MLP, Transformer (small: 64h4, large: 256h8)
  
Conditions:
  • n+b FR: n+b context → beh target, free-run with re-clamping
  • n FR: n context → beh target, direct prediction
  • b 1s: b context → beh target, 1-step teacher-forced

Author: Copilot
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from baseline_transformer.dataset import load_worm_data as _load_worm_data_baseline
from baseline_transformer.config import TransformerBaselineConfig
from baseline_transformer.model import TemporalTransformerGaussian
from stage2.posture_videos import angles_to_xy, _load_eigenvectors


def load_worm_data(h5_path, n_beh_modes=6):
    """Load worm data using baseline transformer loader."""
    data = _load_worm_data_baseline(h5_path, n_beh_modes=n_beh_modes)
    return {
        "u": data["u"],
        "b": data["b"],
        "worm_id": data["worm_id"],
        "motor_idx": data.get("motor_idx"),
    }

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Re-clamp every 10 seconds at 4Hz
RECLAMP_FRAMES = 10


def _make_trf_config_small(context_length=10):
    """Create small transformer config (64h4, ~50K params)."""
    cfg = TransformerBaselineConfig()
    cfg.d_model = 64
    cfg.n_heads = 4
    cfg.n_layers = 1
    cfg.d_ff = 128
    cfg.dropout = 0.1
    cfg.context_length = context_length
    cfg.lr = 1e-3
    cfg.weight_decay = 1e-4
    cfg.sigma_min = 1e-4
    cfg.sigma_max = 10.0
    cfg.max_epochs = 200
    cfg.patience = 25
    return cfg


def _make_trf_config_large(context_length=10):
    """Create large transformer config (256h8, ~1.1M params)."""
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


def _r2(gt, pred):
    ss_res = np.nansum((gt - pred) ** 2)
    ss_tot = np.nansum((gt - np.nanmean(gt)) ** 2) + 1e-12
    return 1 - ss_res / ss_tot


def _corr(gt, pred):
    valid = np.isfinite(gt) & np.isfinite(pred)
    if valid.sum() < 3:
        return np.nan
    return np.corrcoef(gt[valid], pred[valid])[0, 1]


def beh_metrics_heldout(pred, gt):
    """Compute R² and correlation per eigenworm mode."""
    n_modes = pred.shape[1]
    r2_per = np.array([_r2(gt[:, i], pred[:, i]) for i in range(n_modes)])
    corr_per = np.array([_corr(gt[:, i], pred[:, i]) for i in range(n_modes)])
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
    """Get training indices with buffer zone after test fold.
    
    IMPORTANT: Excludes indices in [te, te + buffer) to prevent leakage where
    training samples at time t use lagged features from inside the test fold.
    """
    # Before test fold: [warmup, ts)
    # After test fold + buffer: [te + buffer, T)
    before = np.arange(warmup, ts)
    after_start = min(te + buffer, T)
    after = np.arange(after_start, T)
    return np.concatenate([before, after])


def _build_lagged(x, K, include_current=False):
    """Build lagged feature matrix.
    
    Args:
        x: Input array (T, D)
        K: Number of lags
        include_current: If True, include x[t] as the first feature block
    
    Returns:
        If include_current=False: (T, K*D) with lags 1..K
        If include_current=True: (T, (K+1)*D) with current + lags 1..K
    """
    T, D = x.shape
    n_blocks = K + 1 if include_current else K
    out = np.zeros((T, n_blocks * D), dtype=x.dtype)
    
    if include_current:
        out[:, :D] = x  # Current timepoint
        for lag in range(1, K + 1):
            out[lag:, lag*D:(lag+1)*D] = x[:-lag]
    else:
        for lag in range(1, K + 1):
            out[lag:, (lag-1)*D:lag*D] = x[:-lag]
    return out


# ══════════════════════════════════════════════════════════════════════════════
# RIDGE
# ══════════════════════════════════════════════════════════════════════════════

def _check_ridge_coefficients(model, name="Ridge"):
    """Check if Ridge coefficients hit boundaries (too large)."""
    coef = model.coef_
    max_coef = np.abs(coef).max()
    mean_coef = np.abs(coef).mean()
    n_large = (np.abs(coef) > 10).sum()
    n_total = coef.size
    pct_large = 100 * n_large / n_total
    if pct_large > 1 or max_coef > 100:
        print(f"      ⚠️  {name}: max_coef={max_coef:.1f}, mean={mean_coef:.3f}, {pct_large:.1f}% > 10")
    return {"max_coef": float(max_coef), "mean_coef": float(mean_coef), "pct_large": float(pct_large)}


def _plot_ridge_coefficients(coef_stats_all, out, worm_id):
    """Plot Ridge coefficient statistics across all K values and conditions."""
    if not coef_stats_all:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Organize by condition
    conditions = ["n+b", "n", "b"]
    colors = {"n+b": "#d62728", "n": "#1f77b4", "b": "#2ca02c"}
    
    for ax_idx, metric in enumerate(["max_coef", "mean_coef", "pct_large"]):
        ax = axes[ax_idx]
        
        for cond in conditions:
            K_vals = sorted([k for k in coef_stats_all.keys()])
            vals = []
            for K in K_vals:
                if cond in coef_stats_all[K]:
                    vals.append(coef_stats_all[K][cond].get(metric, np.nan))
                else:
                    vals.append(np.nan)
            
            ax.plot(K_vals, vals, 'o-', label=f"Ridge {cond}", color=colors[cond], lw=2, ms=8)
        
        ax.set_xlabel("Context length K")
        ax.set_xticks(K_vals)
        
        if metric == "max_coef":
            ax.set_ylabel("Max |coefficient|")
            ax.axhline(100, color='red', ls='--', alpha=0.5, label="Warning threshold")
            ax.set_title("Maximum Coefficient Magnitude")
        elif metric == "mean_coef":
            ax.set_ylabel("Mean |coefficient|")
            ax.set_title("Mean Coefficient Magnitude")
        else:
            ax.set_ylabel("% coefficients > 10")
            ax.axhline(1, color='red', ls='--', alpha=0.5, label="Warning threshold")
            ax.set_title("Percentage of Large Coefficients")
        
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    fig.suptitle(f"Ridge Coefficient Analysis — {worm_id}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fname = f"ridge_coef_analysis_{worm_id}.png"
    fig.savefig(out / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out / fname}")


def _cv_ridge_nb_freerun(u, b, K_n, K_b, reclamp_every=RECLAMP_FRAMES):
    """Ridge: n+b context → BEHAVIOR ONLY, free-run with re-clamping."""
    T, N, Kw = u.shape[0], u.shape[1], b.shape[1]
    max_K = max(K_n, K_b)
    warmup = max_K
    ho = np.full((T, Kw), np.nan)
    coef_stats = []

    for ts, te in _make_folds(T, warmup):
        tr = _get_train_indices(T, warmup, ts, te, buffer=max_K)
        
        mu_u, sig_u = u[tr].mean(0), u[tr].std(0) + 1e-8
        u_n = ((u - mu_u) / sig_u).astype(np.float32)
        mu_b, sig_b = b[tr].mean(0), b[tr].std(0) + 1e-8
        b_n = ((b - mu_b) / sig_b).astype(np.float32)

        X_u = _build_lagged(u_n, K_n, include_current=False)  # Lags only — predict b[t+1] from u[t]
        X_b = _build_lagged(b_n, K_b)
        X = np.concatenate([X_u, X_b], axis=1)
        
        # Target is BEHAVIOR ONLY - use RidgeCV for optimal alpha selection
        model = Ridge(alpha=1000.0).fit(X[tr], b_n[tr])
        coef_stats.append(_check_ridge_coefficients(model, "Ridge n+b"))

        # Free-run with re-clamping
        pred_b_fold = np.zeros((te - ts, Kw), dtype=np.float32)
        b_running = b_n.copy()

        for i, t in enumerate(range(ts, te)):
            # Re-clamp to GT every reclamp_every frames
            if reclamp_every is not None and i > 0 and i % reclamp_every == 0:
                b_running[ts:t] = b_n[ts:t]
            
            X_b_t = np.concatenate([b_running[t - lag] for lag in range(1, K_b + 1)])
            # Lags 1..K_n only — no current neural frame
            X_u_t = np.concatenate([u_n[t - lag] for lag in range(1, K_n + 1)])
            X_t = np.concatenate([X_u_t, X_b_t])[None, :]
            pred_beh = model.predict(X_t)[0]
            pred_b_fold[i] = pred_beh
            b_running[t] = pred_beh

        ho[ts:te] = pred_b_fold * sig_b + mu_b

    # Aggregate coef stats across folds
    agg_stats = {
        "max_coef": float(np.max([s["max_coef"] for s in coef_stats])),
        "mean_coef": float(np.mean([s["mean_coef"] for s in coef_stats])),
        "pct_large": float(np.mean([s["pct_large"] for s in coef_stats]))
    }
    return ho, agg_stats


def _cv_ridge_n_freerun(u, b, K_n):
    """Ridge: neural-only context → behavior.
    
    Model uses ONLY neural context (no behavior features).
    This is a direct mapping from neural history to behavior prediction.
    """
    T, N, Kw = u.shape[0], u.shape[1], b.shape[1]
    warmup = K_n
    ho = np.full((T, Kw), np.nan)
    coef_stats = []

    for ts, te in _make_folds(T, warmup):
        tr = _get_train_indices(T, warmup, ts, te, buffer=K_n)
        
        mu_u, sig_u = u[tr].mean(0), u[tr].std(0) + 1e-8
        u_n = ((u - mu_u) / sig_u).astype(np.float32)
        mu_b, sig_b = b[tr].mean(0), b[tr].std(0) + 1e-8
        b_n = ((b - mu_b) / sig_b).astype(np.float32)

        # Neural-only features — lags only, no current frame
        X_u = _build_lagged(u_n, K_n, include_current=False)
        
        # Target is behavior - use RidgeCV for optimal alpha selection
        model = Ridge(alpha=1000.0).fit(X_u[tr], b_n[tr])
        coef_stats.append(_check_ridge_coefficients(model, "Ridge n"))

        # Direct prediction (no behavior feedback)
        ho[ts:te] = model.predict(X_u[ts:te]) * sig_b + mu_b

    # Aggregate coef stats across folds
    agg_stats = {
        "max_coef": float(np.max([s["max_coef"] for s in coef_stats])),
        "mean_coef": float(np.mean([s["mean_coef"] for s in coef_stats])),
        "pct_large": float(np.mean([s["pct_large"] for s in coef_stats]))
    }
    return ho, agg_stats


def _cv_ridge_b_1step(b, K_b):
    """Ridge: beh-only context → beh, 1-step (teacher-forced)."""
    T, Kw = b.shape
    warmup = K_b
    ho = np.full((T, Kw), np.nan)
    coef_stats = []

    for ts, te in _make_folds(T, warmup):
        tr = _get_train_indices(T, warmup, ts, te, buffer=K_b)
        
        mu_b, sig_b = b[tr].mean(0), b[tr].std(0) + 1e-8
        b_n = ((b - mu_b) / sig_b).astype(np.float32)

        X_b = _build_lagged(b_n, K_b)
        # Use RidgeCV for optimal alpha selection
        model = Ridge(alpha=1000.0).fit(X_b[tr], b_n[tr])
        coef_stats.append(_check_ridge_coefficients(model, "Ridge b"))
        
        # 1-step: use GT context (teacher-forced)
        ho[ts:te] = model.predict(X_b[ts:te]) * sig_b + mu_b

    # Aggregate coef stats across folds
    agg_stats = {
        "max_coef": float(np.max([s["max_coef"] for s in coef_stats])),
        "mean_coef": float(np.mean([s["mean_coef"] for s in coef_stats])),
        "pct_large": float(np.mean([s["pct_large"] for s in coef_stats]))
    }
    return ho, agg_stats


# ══════════════════════════════════════════════════════════════════════════════
# MLP (from original behaviour decoder models.py - with LayerNorm, Dropout, rollout training)
# ══════════════════════════════════════════════════════════════════════════════

def _make_mlp(d_in, K_out, hidden=64, n_layers=2):
    """Create MLP with LayerNorm and Dropout (original architecture)."""
    layers, d = [], d_in
    for _ in range(n_layers):
        layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(0.1)]
        d = hidden
    layers.append(nn.Linear(d, K_out))
    return nn.Sequential(*layers)


def _train_mlp_rollout(X_base_norm, b, Kw, device, epochs=150, lr=1e-3, wd=1e-3, patience=20,
                       rollout_prob=0.5, rollout_len=10, n_windows=50, beh_noise_std=0.1, seg_boundary=None):
    """Train MLP with rollout training for free-run robustness (original implementation)."""
    T, nv = b.shape[0], max(10, int(b.shape[0] * 0.15))
    tr_end = T - nv
    min_s, max_s = 1, max(2, tr_end - rollout_len)
    d_in = (X_base_norm.shape[1] if X_base_norm is not None else 0) + Kw
    mlp = _make_mlp(d_in, Kw, hidden=64).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=wd)
    b_t = torch.tensor(b, dtype=torch.float32, device=device)
    mu_b, sig_b = b_t[:tr_end].mean(0), b_t[:tr_end].std(0) + 1e-8
    b_n = (b_t - mu_b) / sig_b
    Xn = torch.tensor(X_base_norm, dtype=torch.float32, device=device) if X_base_norm is not None else None
    Xn_val, b_n_val = (Xn[tr_end:T] if Xn is not None else None), b_n[tr_end:T]
    def _cat(idx, beh): return torch.cat([Xn[idx], beh], dim=1) if Xn is not None else beh
    bvl, bs, pat = float("inf"), None, 0
    for epoch in range(epochs):
        mlp.train()
        starts = np.random.randint(min_s, max_s, size=n_windows)
        if seg_boundary is not None:
            bad = starts == seg_boundary
            while bad.any(): starts[bad] = np.random.randint(min_s, max_s, size=bad.sum()); bad = starts == seg_boundary
        rp, use_ro = rollout_prob * min(1.0, epoch / 30), np.random.random(n_windows)
        tf_s, ro_s = starts[use_ro >= rp], starts[use_ro < rp]
        loss, nt = torch.tensor(0.0, device=device), 0
        if len(tf_s) > 0:
            ta = np.concatenate([np.arange(s, min(s + rollout_len, tr_end) if seg_boundary is None or s >= seg_boundary
                                 else min(s + rollout_len, tr_end, seg_boundary)) for s in tf_s])
            if seg_boundary is not None: ta = ta[ta != seg_boundary]
            if len(ta) > 0:
                bi = b_n[ta - 1]
                if beh_noise_std > 0: bi = bi + beh_noise_std * torch.randn_like(bi)
                loss, nt = loss + nn.functional.mse_loss(mlp(_cat(ta, bi)), b_n[ta]), nt + 1
        if len(ro_s) > 0:
            bh, in_a = b_n[ro_s - 1], (ro_s < seg_boundary) if seg_boundary is not None else None
            preds, tgts = [], []
            for step in range(rollout_len):
                ti, ok = ro_s + step, (ro_s + step) < tr_end
                if seg_boundary is not None: ok = ok & ~(in_a & (ti >= seg_boundary))
                if not ok.any(): break
                bhat = mlp(_cat(ti, bh))
                preds.append(bhat[ok]); tgts.append(b_n[ti[ok]]); bh = bhat.detach()
            if preds: loss, nt = loss + nn.functional.mse_loss(torch.cat(preds), torch.cat(tgts)), nt + 1
        if nt > 0: opt.zero_grad(); (loss / nt).backward(); opt.step()
        mlp.eval()
        with torch.no_grad():
            bh, vp = b_n[tr_end - 1:tr_end], []
            for i in range(nv): xt = torch.cat([Xn_val[i:i+1], bh], dim=1) if Xn is not None else bh; bh = mlp(xt); vp.append(bh)
            vl = nn.functional.mse_loss(torch.cat(vp), b_n_val).item()
        if vl < bvl - 1e-6: bvl, bs, pat = vl, {k: v.clone() for k, v in mlp.state_dict().items()}, 0
        else: pat += 1
        if pat > patience: break
    if bs: mlp.load_state_dict(bs)
    mlp.eval().cpu()
    return mlp, mu_b.cpu(), sig_b.cpu()


def _train_mlp_fast(X, y, K_out, device, epochs=150, lr=1e-3, wd=1e-3, patience=20):
    """Train MLP with simple 1-step supervision (for teacher-forced conditions)."""
    nv = max(10, int(X.shape[0] * 0.15))
    Xt = torch.tensor(X[:-nv], dtype=torch.float32, device=device)
    yt = torch.tensor(y[:-nv], dtype=torch.float32, device=device)
    Xv = torch.tensor(X[-nv:], dtype=torch.float32, device=device)
    yv = torch.tensor(y[-nv:], dtype=torch.float32, device=device)
    mlp = _make_mlp(X.shape[1], K_out).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=wd)
    bvl, bs, pat = float('inf'), None, 0
    for _ in range(epochs):
        mlp.train()
        loss = nn.functional.mse_loss(mlp(Xt), yt)
        opt.zero_grad(); loss.backward(); opt.step()
        mlp.eval()
        with torch.no_grad():
            vl = nn.functional.mse_loss(mlp(Xv), yv).item()
        if vl < bvl - 1e-6: bvl, bs, pat = vl, {k: v.clone() for k, v in mlp.state_dict().items()}, 0
        else: pat += 1
        if pat > patience: break
    if bs: mlp.load_state_dict(bs)
    mlp.eval().cpu()
    return mlp


def _cv_mlp_nb_freerun(u, b, K_n, K_b, device, reclamp_every=RECLAMP_FRAMES):
    """MLP: n+b context → BEHAVIOR ONLY, free-run with re-clamping."""
    T, N, Kw = u.shape[0], u.shape[1], b.shape[1]
    max_K = max(K_b, K_b)
    warmup = max_K
    ho = np.full((T, Kw), np.nan)
    b_t = torch.tensor(b, dtype=torch.float32)

    for ts, te in _make_folds(T, warmup):
        tr = _get_train_indices(T, warmup, ts, te, buffer=max_K)
        
        mu_u, sig_u = u[tr].mean(0), u[tr].std(0) + 1e-8
        u_n = ((u - mu_u) / sig_u).astype(np.float32)
        
        X_u = _build_lagged(u_n, K_n, include_current=False)  # Lags only — predict b[t+1] from u[t]
        
        seg_before = np.arange(warmup, ts)
        seg_after_start = te + max_K
        seg_after = np.arange(seg_after_start, T) if seg_after_start < T else np.array([], dtype=int)
        
        if len(seg_after) > 0:
            X_tr = np.concatenate([X_u[seg_before], X_u[seg_after]])
            b_tr = np.concatenate([b[seg_before], b[seg_after]])
            sb = len(seg_before)
        else:
            X_tr = X_u[seg_before]
            b_tr = b[seg_before]
            sb = None
        
        mlp, mu_b_t, sig_b_t = _train_mlp_rollout(X_tr, b_tr, Kw, device, rollout_len=max_K, seg_boundary=sb)

        # Free-run with re-clamping
        with torch.no_grad():
            Xte = torch.tensor(X_u[ts:te], dtype=torch.float32)
            mu_b, sig_b = b[tr].mean(0), b[tr].std(0) + 1e-8
            b_n = ((b - mu_b) / sig_b).astype(np.float32)
            b_running = b_n.copy()
            
            for i in range(te - ts):
                t = ts + i
                # Re-clamp to GT every reclamp_every frames
                if reclamp_every is not None and i > 0 and i % reclamp_every == 0:
                    b_running[ts:t] = b_n[ts:t]
                
                bh = torch.tensor(b_running[t-1:t], dtype=torch.float32)
                bh = (bh - mu_b_t) / sig_b_t
                xt = torch.cat([Xte[i:i+1], bh], dim=1)
                bh_pred = mlp(xt)
                pred_beh_norm = bh_pred[0].numpy()
                ho[t] = pred_beh_norm * sig_b_t.numpy() + mu_b_t.numpy()
                b_running[t] = pred_beh_norm * sig_b_t.numpy() / sig_b + (mu_b_t.numpy() - mu_b) / sig_b

    return ho


def _cv_mlp_n_freerun(u, b, K_n, device):
    """MLP: neural-only context → behavior (uses fast training, no rollout needed)."""
    T, N, Kw = u.shape[0], u.shape[1], b.shape[1]
    warmup = K_n
    ho = np.full((T, Kw), np.nan)

    for ts, te in _make_folds(T, warmup):
        tr = _get_train_indices(T, warmup, ts, te, buffer=K_n)
        
        mu_u, sig_u = u[tr].mean(0), u[tr].std(0) + 1e-8
        u_n = ((u - mu_u) / sig_u).astype(np.float32)
        mu_b, sig_b = b[tr].mean(0), b[tr].std(0) + 1e-8
        b_n = ((b - mu_b) / sig_b).astype(np.float32)

        # Neural-only features — lags only, no current frame
        X_u = _build_lagged(u_n, K_n, include_current=False)
        
        mlp = _train_mlp_fast(X_u[tr], b_n[tr], Kw, device)

        # Direct prediction (no behavior feedback)
        with torch.no_grad():
            ho[ts:te] = mlp(torch.tensor(X_u[ts:te], dtype=torch.float32)).numpy() * sig_b + mu_b

    return ho


def _cv_mlp_b_1step(b, K_b, device):
    """MLP: beh-only context → beh, 1-step (teacher-forced, uses fast training)."""
    T, Kw = b.shape
    warmup = K_b
    ho = np.full((T, Kw), np.nan)

    for ts, te in _make_folds(T, warmup):
        tr = _get_train_indices(T, warmup, ts, te, buffer=K_b)
        
        mu_b, sig_b = b[tr].mean(0), b[tr].std(0) + 1e-8
        b_n = ((b - mu_b) / sig_b).astype(np.float32)

        X_b = _build_lagged(b_n, K_b)
        mlp = _train_mlp_fast(X_b[tr], b_n[tr], Kw, device)
        
        with torch.no_grad():
            ho[ts:te] = mlp(torch.tensor(X_b[ts:te], dtype=torch.float32)).numpy() * sig_b + mu_b

    return ho


# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORMER
# ══════════════════════════════════════════════════════════════════════════════

def _train_transformer(X_win, y_target, n_neural, n_beh, K, device, use_small=False,
                       epochs=200, lr=1e-3, wd=1e-4, patience=25):
    """Train transformer to predict BEHAVIOR ONLY."""
    nv = max(10, int(len(X_win) * 0.15))
    Xt = torch.tensor(X_win[:-nv], dtype=torch.float32, device=device)
    yt = torch.tensor(y_target[:-nv], dtype=torch.float32, device=device)
    Xv = torch.tensor(X_win[-nv:], dtype=torch.float32, device=device)
    yv = torch.tensor(y_target[-nv:], dtype=torch.float32, device=device)

    cfg = _make_trf_config_small(context_length=K) if use_small else _make_trf_config_large(context_length=K)
    cfg.lr = lr
    cfg.weight_decay = wd
    cfg.max_epochs = epochs
    cfg.patience = patience
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = TemporalTransformerGaussian(n_neural=n_neural, n_beh=n_beh, cfg=cfg).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    bvl, bs, pat = float('inf'), None, 0
    for _ in range(epochs):
        model.train()
        pred_mu, _, pred_b_mu, _ = model.forward(Xt, return_all_steps=False)
        
        # Always predict behavior only
        loss = nn.functional.mse_loss(pred_b_mu, yt)
        
        opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            _, _, pred_b_mu_v, _ = model.forward(Xv, return_all_steps=False)
            vl = nn.functional.mse_loss(pred_b_mu_v, yv).item()
                
        if vl < bvl - 1e-6:
            bvl, bs, pat = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            pat += 1
        if pat > patience: break

    if bs: model.load_state_dict(bs)
    model.eval().cpu()
    model.n_params = n_params
    return model


def _cv_trf_nb_freerun(u, b, K_n, K_b, device, use_small=False, reclamp_every=RECLAMP_FRAMES):
    """Transformer: n+b context → BEHAVIOR ONLY, free-run with re-clamping.
    
    Neural context uses lags only: u[t-K:t] (K timesteps, no current frame).
    Behavior context uses lags only: b[t-K_b:t] (K_b timesteps).
    """
    T, N, Kw = u.shape[0], u.shape[1], b.shape[1]
    K = max(K_n, K_b)
    warmup = K
    ho = np.full((T, Kw), np.nan)

    for ts, te in _make_folds(T, warmup):
        tr = _get_train_indices(T, warmup, ts, te, buffer=K)
        
        mu_u, sig_u = u[tr].mean(0), u[tr].std(0) + 1e-8
        u_n = ((u - mu_u) / sig_u).astype(np.float32)
        mu_b, sig_b = b[tr].mean(0), b[tr].std(0) + 1e-8
        b_n = ((b - mu_b) / sig_b).astype(np.float32)

        X_win = []
        for t in tr:
            # Lags only — no current neural frame: u[t-K:t] -> K timesteps
            ctx_u = u_n[t - K:t]  # Shape: (K, N)
            if K_b < K:
                ctx_b = np.zeros((K, Kw), dtype=np.float32)
                ctx_b[-(K_b):] = b_n[t - K_b:t] if K_b > 0 else np.zeros((0, Kw))
            else:
                ctx_b = np.zeros((K, Kw), dtype=np.float32)
                ctx_b[:K] = b_n[t - K:t]
            ctx = np.concatenate([ctx_u, ctx_b], axis=1)
            X_win.append(ctx)
        X_win = np.stack(X_win)
        
        # Target is BEHAVIOR ONLY
        y_tr_beh = b_n[tr]

        model = _train_transformer(X_win, y_tr_beh, n_neural=N, n_beh=Kw, K=K, device=device, use_small=use_small)

        pred_b_fold = np.zeros((te - ts, Kw), dtype=np.float32)
        b_running = b_n.copy()

        for i, t in enumerate(range(ts, te)):
            # Re-clamp to GT every reclamp_every frames
            if reclamp_every is not None and i > 0 and i % reclamp_every == 0:
                b_running[ts:t] = b_n[ts:t]
            
            # Lags only — no current neural frame
            ctx_u = u_n[t - K:t]  # Shape: (K, N)
            ctx_b = np.zeros((K, Kw), dtype=np.float32)
            if K_b > 0:
                ctx_b[-(K_b):] = b_running[t - K_b:t]
            ctx = np.concatenate([ctx_u, ctx_b], axis=1)[None, ...]
            
            with torch.no_grad():
                ctx_t = torch.tensor(ctx, dtype=torch.float32)
                _, _, pred_b_mu, _ = model.forward(ctx_t, return_all_steps=False)
                pred_beh = pred_b_mu[0].numpy()
            
            pred_b_fold[i] = pred_beh
            b_running[t] = pred_beh

        ho[ts:te] = pred_b_fold * sig_b + mu_b

    return ho


def _cv_trf_n_freerun(u, b, K_n, device, use_small=False):
    """Transformer: neural-only context → behavior (direct prediction).
    
    Neural context uses lags only: u[t-K:t] (K timesteps, no current frame).
    """
    T, N, Kw = u.shape[0], u.shape[1], b.shape[1]
    warmup = K_n
    ho = np.full((T, Kw), np.nan)

    for ts, te in _make_folds(T, warmup):
        tr = _get_train_indices(T, warmup, ts, te, buffer=K_n)
        
        mu_u, sig_u = u[tr].mean(0), u[tr].std(0) + 1e-8
        u_n = ((u - mu_u) / sig_u).astype(np.float32)
        mu_b, sig_b = b[tr].mean(0), b[tr].std(0) + 1e-8
        b_n = ((b - mu_b) / sig_b).astype(np.float32)

        # Lags only — no current neural frame: u[t-K:t] -> K timesteps
        X_win = np.stack([u_n[t - K_n:t] for t in tr])  # Shape: (len(tr), K, N)
        X_win_padded = np.concatenate([X_win, np.zeros((len(tr), K_n, Kw), dtype=np.float32)], axis=-1)
        
        y_tr_beh = b_n[tr]

        model = _train_transformer(X_win_padded, y_tr_beh, n_neural=N, n_beh=Kw, K=K_n, device=device, use_small=use_small)

        # Lags only for test — no current neural frame
        X_test = np.stack([u_n[t - K_n:t] for t in range(ts, te)])
        X_test_padded = np.concatenate([X_test, np.zeros((te - ts, K_n, Kw), dtype=np.float32)], axis=-1)
        
        with torch.no_grad():
            ctx_t = torch.tensor(X_test_padded, dtype=torch.float32)
            _, _, pred_b_mu, _ = model.forward(ctx_t, return_all_steps=False)
            ho[ts:te] = pred_b_mu.numpy() * sig_b + mu_b

    return ho


def _cv_trf_b_1step(b, K_b, device, use_small=False):
    """Transformer: beh-only context → beh, 1-step (teacher-forced)."""
    T, Kw = b.shape
    warmup = K_b
    ho = np.full((T, Kw), np.nan)

    for ts, te in _make_folds(T, warmup):
        tr = _get_train_indices(T, warmup, ts, te, buffer=K_b)
        
        mu_b, sig_b = b[tr].mean(0), b[tr].std(0) + 1e-8
        b_n = ((b - mu_b) / sig_b).astype(np.float32)

        X_win = np.stack([b_n[t - K_b:t] for t in tr])
        y_tr_beh = b_n[tr]

        model = _train_transformer(X_win, y_tr_beh, n_neural=0, n_beh=Kw, K=K_b, device=device, use_small=use_small)

        X_test = np.stack([b_n[t - K_b:t] for t in range(ts, te)])
        with torch.no_grad():
            ctx_t = torch.tensor(X_test, dtype=torch.float32)
            _, _, pred_b_mu, _ = model.forward(ctx_t, return_all_steps=False)
            ho[ts:te] = pred_b_mu.numpy() * sig_b + mu_b

    return ho


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def _plot_lag_sweep(results, out, worm_id, neuron_label, n_neurons, run_name):
    """Line plot with 2 panels (R² and Correlation) - legend only on right."""
    K_VALUES = sorted(results.keys())
    
    # Use trf_large instead of trf for plotting
    arch_colors = {"trf_large": "#d62728", "mlp": "#1f77b4", "ridge": "#2ca02c"}
    conditions = [
        ("nb_fr", "-", "o", "n+b FR"),
        ("n_fr", "-", "s", "n FR"),
        ("b_1s", "--", "^", "b (1-step)"),
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left panel: R²
    ax_r2 = axes[0]
    for arch in ["trf_large", "mlp", "ridge"]:
        for cond_key, ls, marker, label in conditions:
            vals = [results[K].get(f"{arch}_{cond_key}", {}).get("r2_mean", np.nan) for K in K_VALUES]
            ax_r2.plot(K_VALUES, vals, marker=marker, ls=ls, lw=2, ms=6, 
                      color=arch_colors[arch])
    
    ax_r2.set_xlabel("Context length K (lags)")
    ax_r2.set_ylabel("Mean R²")
    ax_r2.set_xticks(K_VALUES)
    ax_r2.set_ylim(0, 1)
    ax_r2.grid(alpha=0.3)
    ax_r2.set_title("Behaviour R²")
    
    # Right panel: Correlation (with legend)
    ax_corr = axes[1]
    for arch in ["trf_large", "mlp", "ridge"]:
        for cond_key, ls, marker, label in conditions:
            vals = [results[K].get(f"{arch}_{cond_key}", {}).get("corr_mean", np.nan) for K in K_VALUES]
            arch_label = "TRF" if arch == "trf_large" else arch.upper()
            full_label = f"{arch_label} {label}"
            ax_corr.plot(K_VALUES, vals, marker=marker, ls=ls, lw=2, ms=6, 
                        color=arch_colors[arch], label=full_label)
    
    ax_corr.set_xlabel("Context length K (lags)")
    ax_corr.set_ylabel("Mean Correlation")
    ax_corr.set_xticks(K_VALUES)
    ax_corr.set_ylim(0, 1)
    ax_corr.grid(alpha=0.3)
    ax_corr.set_title("Behaviour Correlation")
    ax_corr.legend(fontsize=7, loc='lower left', ncol=3)
    
    fig.suptitle(f"Lag Sweep — {worm_id}  ({neuron_label} (N={n_neurons}), 5-fold CV, 6 eigenworms)",
                 fontsize=12, fontweight="bold")
    
    fname = f"lag_sweep_{run_name}_{neuron_label}.png"
    fig.tight_layout()
    fig.savefig(out / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out / fname}")


def _plot_traces(predictions, gt_b, out, worm_id, K, N, neuron_label, frame_range=None):
    """Plot eigenworm traces: GT vs Pred for TRF, MLP, Ridge."""
    T, Kw = gt_b.shape
    
    if frame_range is None:
        # Pick a representative segment (middle portion)
        start = max(K, T // 4)
        end = min(T, start + 200)
        frame_range = (start, end)
    
    fs, fe = frame_range
    t_axis = np.arange(fs, fe)
    
    # Colors matching the attached plot
    colors = {
        "trf_nb_fr": "#d62728",
        "mlp_nb_fr": "#1f77b4", 
        "ridge_n_fr": "#2ca02c",
    }
    labels = {
        "trf_nb_fr": "TRF n+b FR",
        "mlp_nb_fr": "MLP n+b FR",
        "ridge_n_fr": "Ridge n FR",
    }
    
    fig, axes = plt.subplots(Kw, 3, figsize=(15, 2.5 * Kw), sharex=True)
    
    model_order = ["trf_nb_fr", "mlp_nb_fr", "ridge_n_fr"]
    
    for col, model_key in enumerate(model_order):
        pred = predictions.get(model_key)
        if pred is None:
            continue
            
        for row in range(Kw):
            ax = axes[row, col] if Kw > 1 else axes[col]
            
            # Ground truth
            ax.plot(t_axis, gt_b[fs:fe, row], 'k-', lw=1.2, label='GT')
            # Prediction
            ax.plot(t_axis, pred[fs:fe, row], '--', lw=1.2, 
                   color=colors[model_key], label='Pred')
            
            # Compute R² for this segment
            valid = np.isfinite(pred[fs:fe, row])
            if valid.sum() > 3:
                r2 = _r2(gt_b[fs:fe, row][valid], pred[fs:fe, row][valid])
                ax.text(0.98, 0.95, f'R²={r2:.2f}', transform=ax.transAxes,
                       ha='right', va='top', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            if col == 0:
                ax.set_ylabel(f'EW{row+1}')
            if row == 0:
                ax.set_title(labels[model_key])
            if row == 0 and col == 0:
                ax.legend(loc='upper left', fontsize=8)
    
    axes[-1, 1].set_xlabel('Time (s)')
    
    fig.suptitle(f"Eigenworm Traces — {worm_id}  K={K}, N_{neuron_label}={N}  (frames {fs}-{fe})",
                 fontsize=12, fontweight="bold")
    
    fname = f"traces_K{K}_{neuron_label}_{worm_id}.png"
    fig.tight_layout()
    fig.savefig(out / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out / fname}")


def _make_behaviour_video(h5_path, predictions, gt_b, out, worm_id, neuron_label,
                          n_eigenworm_modes=6, frame_range=(400, 600), fps=20, K=1):
    """Generate posture comparison video: GT (left) vs model predictions.
    
    Shows the real worm (reconstructed from eigenworms) on the left,
    and n+b FR predictions from each model class on the right.
    
    Uses same frame range as trace plots for consistency.
    """
    try:
        # Load eigenvectors
        eigvecs = _load_eigenvectors(h5_path=h5_path)
    except FileNotFoundError as e:
        print(f"  WARNING: Cannot create video - {e}")
        return
    
    # Read sample rate from h5
    sr = 4.0  # default
    try:
        with h5py.File(h5_path, "r") as f:
            if "stage1/params" in f and "sample_rate_hz" in f["stage1/params"].attrs:
                sr = float(f["stage1/params"].attrs["sample_rate_hz"])
    except Exception:
        pass
    
    dt = 1.0 / sr
    
    T, Kw = gt_b.shape
    n_modes = min(Kw, n_eigenworm_modes, eigvecs.shape[1])
    E = eigvecs[:, :n_modes]  # (d_recon, n_modes)
    d_recon = E.shape[0]
    
    # Use explicit frame range (same as trace plots)
    start_data, end_data = frame_range
    start_data = max(0, min(start_data, T - 10))
    end_data = min(end_data, T)
    n_frames_data = end_data - start_data
    
    # Video frames: 1:1 with data frames at higher fps for smooth playback
    n_frames_video = n_frames_data
    
    # Build arrays to visualize
    model_order = [("GT", None), ("TRF n+b FR", "trf_nb_fr"), 
                   ("MLP n+b FR", "mlp_nb_fr"), ("Ridge n FR", "ridge_n_fr")]
    
    # Collect valid models
    valid_models = [("GT", gt_b)]
    for name, key in model_order[1:]:
        if key in predictions and predictions[key] is not None:
            valid_models.append((name, predictions[key]))
    
    n_panels = len(valid_models)
    
    # Reconstruct postures for all frames
    xy_data = {}
    for name, arr in valid_models:
        xy = np.zeros((n_frames_data, d_recon, 2), dtype=float)
        for t in range(n_frames_data):
            t_data = start_data + t
            if t_data >= arr.shape[0]:
                continue
            ew = arr[t_data, :n_modes]
            if np.all(np.isfinite(ew)):
                angles = ew @ E.T
                x, y = angles_to_xy(angles)
                xy[t, :, 0], xy[t, :, 1] = x, y
            else:
                xy[t, :, 0] = np.nan
                xy[t, :, 1] = np.nan
        xy_data[name] = xy
    
    # Compute axis limits
    all_xy = np.concatenate([v.reshape(-1, 2) for v in xy_data.values()], axis=0)
    valid_mask = np.all(np.isfinite(all_xy), axis=1)
    if not valid_mask.any():
        print(f"  WARNING: No valid posture data for video")
        return
    xmin, xmax = np.nanmin(all_xy[valid_mask, 0]), np.nanmax(all_xy[valid_mask, 0])
    ymin, ymax = np.nanmin(all_xy[valid_mask, 1]), np.nanmax(all_xy[valid_mask, 1])
    pad = 2.0
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    span = 0.5 * max((xmax - xmin), (ymax - ymin)) + pad
    
    # Create figure
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4), facecolor="white")
    if n_panels == 1:
        axes = [axes]
    
    colors = {"GT": "black", "TRF n+b FR": "#d62728", "MLP n+b FR": "#1f77b4", "Ridge n FR": "#2ca02c"}
    lines = []
    heads = []
    
    for i, (ax, (name, _)) in enumerate(zip(axes, valid_models)):
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlim(cx - span, cx + span)
        ax.set_ylim(cy - span, cy + span)
        ax.set_aspect("equal")
        ax.set_facecolor("#f7f7f7")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for s in ax.spines.values():
            s.set_visible(False)
        color = colors.get(name, "gray")
        line, = ax.plot([], [], "-", lw=2.5, color=color, alpha=0.9)
        head, = ax.plot([], [], "o", color="crimson" if name == "GT" else color, ms=5, alpha=0.9)
        lines.append(line)
        heads.append(head)
    
    time_text = fig.text(0.5, 0.02, "", ha="center", va="bottom", fontsize=11)
    
    def _update(frame_idx):
        # 1:1 mapping - frame_idx is the index into our data arrays (0 to n_frames_data-1)
        t_data = min(frame_idx, n_frames_data - 1)
        
        for i, (name, _) in enumerate(valid_models):
            xy = xy_data[name]
            x, y = xy[t_data, :, 0], xy[t_data, :, 1]
            if np.all(np.isfinite(x)):
                lines[i].set_data(x, y)
                heads[i].set_data([x[0]], [y[0]])
            else:
                lines[i].set_data([], [])
                heads[i].set_data([], [])
        
        time_text.set_text(f"frame {start_data + t_data} | t = {(start_data + t_data) * dt:.1f}s")
        return lines + heads + [time_text]
    
    anim = FuncAnimation(fig, _update, frames=n_frames_video, interval=1000//fps, blit=True)
    
    out_path = out / f"behaviour_video_K{K}_{neuron_label}_{worm_id}.mp4"
    try:
        writer = FFMpegWriter(fps=fps, bitrate=2000)
        anim.save(str(out_path), writer=writer)
        print(f"  Saved: {out_path}")
    except Exception as e:
        print(f"  WARNING: Could not save video: {e}")
    finally:
        plt.close(fig)


def _plot_summary_violin(all_results, out, K_target, run_name):
    """Violin/bar plot summary across all worms like the attached image."""
    # Collect per-mode correlations for each model and neuron type
    archs = ["TRF", "MLP", "Ridge"]
    conditions = ["n+b FR", "n FR", "b 1s"]
    neuron_types = ["motor", "all"]
    n_modes = 6
    
    # Colors
    arch_colors = {"TRF": "#d62728", "MLP": "#1f77b4", "Ridge": "#2ca02c"}
    
    # Organize data: data[arch][cond][neuron_type][mode] = list of corr values
    data = {arch: {cond: {nt: {m: [] for m in range(n_modes)} 
                          for nt in neuron_types} 
                   for cond in conditions} 
            for arch in archs}
    
    for worm_id, worm_results in all_results.items():
        for neuron_type in neuron_types:
            key = f"{neuron_type}_{run_name}"
            if key not in worm_results:
                continue
            res = worm_results[key]
            if K_target not in res:
                continue
            res_k = res[K_target]
            
            # Extract per-mode correlations
            for arch, arch_lower in [("TRF", "trf"), ("MLP", "mlp"), ("Ridge", "ridge")]:
                for cond, cond_key in [("n+b FR", "nb_fr"), ("n FR", "n_fr"), ("b 1s", "b_1s")]:
                    model_key = f"{arch_lower}_{cond_key}"
                    if model_key in res_k and "corr_per_mode" in res_k[model_key]:
                        corrs = res_k[model_key]["corr_per_mode"]
                        for m in range(min(n_modes, len(corrs))):
                            data[arch][cond][neuron_type][m].append(corrs[m])
    
    # Create figure: one subplot per eigenworm mode
    fig, axes = plt.subplots(1, n_modes, figsize=(22, 5), sharey=True)
    
    bar_width = 0.12
    
    for mode_idx in range(n_modes):
        ax = axes[mode_idx]
        x_pos = 0
        x_ticks = []
        x_labels = []
        
        for arch_idx, arch in enumerate(archs):
            for cond_idx, cond in enumerate(conditions):
                # Get data for both neuron types
                motor_vals = data[arch][cond]["motor"][mode_idx]
                all_vals = data[arch][cond]["all"][mode_idx]
                
                if len(motor_vals) > 0:
                    # Motor neurons (darker)
                    mean_m = np.nanmean(motor_vals)
                    ax.bar(x_pos - bar_width/2, mean_m, bar_width * 0.9,
                          color=arch_colors[arch], alpha=0.9, edgecolor='black', linewidth=0.5)
                    # Scatter individual points
                    ax.scatter([x_pos - bar_width/2] * len(motor_vals), motor_vals,
                              c='gray', s=15, alpha=0.7, zorder=3)
                
                if len(all_vals) > 0:
                    # All neurons (lighter)
                    mean_a = np.nanmean(all_vals)
                    ax.bar(x_pos + bar_width/2, mean_a, bar_width * 0.9,
                          color=arch_colors[arch], alpha=0.5, edgecolor='black', linewidth=0.5)
                    ax.scatter([x_pos + bar_width/2] * len(all_vals), all_vals,
                              c='gray', s=15, alpha=0.7, zorder=3)
                
                x_ticks.append(x_pos)
                x_labels.append(f"{arch}\n{cond}")
                x_pos += 1
            
            x_pos += 0.5  # Gap between architectures
        
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=7, rotation=45, ha='right')
        ax.set_title(f"EW{mode_idx + 1}")
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        
        if mode_idx == 0:
            ax.set_ylabel("Pearson Correlation")
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', alpha=0.9, edgecolor='black', label='Motor neurons'),
        Patch(facecolor='gray', alpha=0.5, edgecolor='black', label='All neurons'),
        Patch(facecolor=arch_colors["TRF"], label='Transformer'),
        Patch(facecolor=arch_colors["MLP"], label='MLP'),
        Patch(facecolor=arch_colors["Ridge"], label='Ridge'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=9,
              bbox_to_anchor=(0.5, -0.02))
    
    n_worms = len(all_results)
    fig.suptitle(f"Behaviour Decoding Summary — K = {K_target},  Correlation per Eigenworm  ({n_worms} worms)",
                 fontsize=12, fontweight="bold")
    
    fname = f"summary_violin_K{K_target}_{run_name}.png"
    fig.tight_layout()
    fig.savefig(out / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out / fname}")


# ══════════════════════════════════════════════════════════════════════════════
# RUN EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation(u, b, device, K_VALUES, run_name, label, out, worm_id, h5_path=None, generate_traces=True):
    """Run all models for a given neuron set."""
    T, N, Kw = u.shape[0], u.shape[1], b.shape[1]
    results = {}
    
    print(f"\n  Data: T={T}, N={N}, Kw={Kw}")
    print(f"  Re-clamp every {RECLAMP_FRAMES} frames (~{RECLAMP_FRAMES/4:.0f}s)")
    
    # For trace plots, store predictions at K=1 and K=5
    trace_K_values = [1, 5]  # Generate traces/videos for these K values
    trace_preds_by_K = {k: {} for k in trace_K_values}
    
    # Collect Ridge coefficient stats for plotting
    ridge_coef_stats = {}
    
    for K_n in K_VALUES:
        K_b = 1  # Always Kb=1
            
        print(f"\n  K_n={K_n}, K_b={K_b}")
        print(f"  {'-'*50}")
        res_k = {"K_n": K_n, "K_b": K_b}
        ridge_coef_stats[K_n] = {}

        # ── Ridge ──
        print(f"    Ridge n+b FR (reclamp={RECLAMP_FRAMES})...")
        t0 = time.time()
        ho, coef_stats_nb = _cv_ridge_nb_freerun(u, b, K_n, K_b, reclamp_every=RECLAMP_FRAMES)
        ridge_coef_stats[K_n]["n+b"] = coef_stats_nb
        r2, corr = beh_metrics_heldout(ho, b)
        res_k["ridge_nb_fr"] = {
            "r2_mean": float(np.nanmean(r2)), 
            "corr_mean": float(np.nanmean(corr)),
            "r2_per_mode": [float(x) for x in r2],
            "corr_per_mode": [float(x) for x in corr],
            "coef_stats": coef_stats_nb,
            "time_s": round(time.time() - t0, 2)
        }
        print(f"      R²={np.nanmean(r2):.3f}  corr={np.nanmean(corr):.3f}")
        if K_n in trace_K_values:
            trace_preds_by_K[K_n]["ridge_nb_fr"] = ho.copy()

        print(f"    Ridge n FR (direct)...")
        t0 = time.time()
        ho, coef_stats_n = _cv_ridge_n_freerun(u, b, K_n)
        ridge_coef_stats[K_n]["n"] = coef_stats_n
        r2, corr = beh_metrics_heldout(ho, b)
        res_k["ridge_n_fr"] = {
            "r2_mean": float(np.nanmean(r2)), 
            "corr_mean": float(np.nanmean(corr)),
            "r2_per_mode": [float(x) for x in r2],
            "corr_per_mode": [float(x) for x in corr],
            "coef_stats": coef_stats_n,
            "time_s": round(time.time() - t0, 2)
        }
        print(f"      R²={np.nanmean(r2):.3f}  corr={np.nanmean(corr):.3f}")
        if K_n in trace_K_values:
            trace_preds_by_K[K_n]["ridge_n_fr"] = ho.copy()

        print(f"    Ridge b 1s...")
        t0 = time.time()
        ho, coef_stats_b = _cv_ridge_b_1step(b, K_b)
        ridge_coef_stats[K_n]["b"] = coef_stats_b
        r2, corr = beh_metrics_heldout(ho, b)
        res_k["ridge_b_1s"] = {
            "r2_mean": float(np.nanmean(r2)), 
            "corr_mean": float(np.nanmean(corr)),
            "r2_per_mode": [float(x) for x in r2],
            "corr_per_mode": [float(x) for x in corr],
            "coef_stats": coef_stats_b,
            "corr_per_mode": [float(x) for x in corr],
            "time_s": round(time.time() - t0, 2)
        }
        print(f"      R²={np.nanmean(r2):.3f}  corr={np.nanmean(corr):.3f}")

        # ── MLP ──
        print(f"    MLP n+b FR (reclamp={RECLAMP_FRAMES})...")
        t0 = time.time()
        ho = _cv_mlp_nb_freerun(u, b, K_n, K_b, device, reclamp_every=RECLAMP_FRAMES)
        r2, corr = beh_metrics_heldout(ho, b)
        res_k["mlp_nb_fr"] = {
            "r2_mean": float(np.nanmean(r2)), 
            "corr_mean": float(np.nanmean(corr)),
            "r2_per_mode": [float(x) for x in r2],
            "corr_per_mode": [float(x) for x in corr],
            "time_s": round(time.time() - t0, 2)
        }
        print(f"      R²={np.nanmean(r2):.3f}  corr={np.nanmean(corr):.3f}")
        if K_n in trace_K_values:
            trace_preds_by_K[K_n]["mlp_nb_fr"] = ho.copy()

        print(f"    MLP n FR (direct)...")
        t0 = time.time()
        ho = _cv_mlp_n_freerun(u, b, K_n, device)
        r2, corr = beh_metrics_heldout(ho, b)
        res_k["mlp_n_fr"] = {
            "r2_mean": float(np.nanmean(r2)), 
            "corr_mean": float(np.nanmean(corr)),
            "r2_per_mode": [float(x) for x in r2],
            "corr_per_mode": [float(x) for x in corr],
            "time_s": round(time.time() - t0, 2)
        }
        print(f"      R²={np.nanmean(r2):.3f}  corr={np.nanmean(corr):.3f}")

        print(f"    MLP b 1s...")
        t0 = time.time()
        ho = _cv_mlp_b_1step(b, K_b, device)
        r2, corr = beh_metrics_heldout(ho, b)
        res_k["mlp_b_1s"] = {
            "r2_mean": float(np.nanmean(r2)), 
            "corr_mean": float(np.nanmean(corr)),
            "r2_per_mode": [float(x) for x in r2],
            "corr_per_mode": [float(x) for x in corr],
            "time_s": round(time.time() - t0, 2)
        }
        print(f"      R²={np.nanmean(r2):.3f}  corr={np.nanmean(corr):.3f}")

        # ── Transformer LARGE ONLY ──
        print(f"    TRF n+b FR (reclamp={RECLAMP_FRAMES})...")
        t0 = time.time()
        ho = _cv_trf_nb_freerun(u, b, K_n, K_b, device, use_small=False, reclamp_every=RECLAMP_FRAMES)
        r2, corr = beh_metrics_heldout(ho, b)
        res_k["trf_large_nb_fr"] = {
            "r2_mean": float(np.nanmean(r2)), 
            "corr_mean": float(np.nanmean(corr)),
            "r2_per_mode": [float(x) for x in r2],
            "corr_per_mode": [float(x) for x in corr],
            "time_s": round(time.time() - t0, 2)
        }
        print(f"      R²={np.nanmean(r2):.3f}  corr={np.nanmean(corr):.3f}")
        if K_n in trace_K_values:
            trace_preds_by_K[K_n]["trf_nb_fr"] = ho.copy()

        print(f"    TRF n FR (direct)...")
        t0 = time.time()
        ho = _cv_trf_n_freerun(u, b, K_n, device, use_small=False)
        r2, corr = beh_metrics_heldout(ho, b)
        res_k["trf_large_n_fr"] = {
            "r2_mean": float(np.nanmean(r2)), 
            "corr_mean": float(np.nanmean(corr)),
            "r2_per_mode": [float(x) for x in r2],
            "corr_per_mode": [float(x) for x in corr],
            "time_s": round(time.time() - t0, 2)
        }
        print(f"      R²={np.nanmean(r2):.3f}  corr={np.nanmean(corr):.3f}")

        print(f"    TRF b 1s...")
        t0 = time.time()
        ho = _cv_trf_b_1step(b, K_b, device, use_small=False)
        r2, corr = beh_metrics_heldout(ho, b)
        res_k["trf_large_b_1s"] = {
            "r2_mean": float(np.nanmean(r2)), 
            "corr_mean": float(np.nanmean(corr)),
            "r2_per_mode": [float(x) for x in r2],
            "corr_per_mode": [float(x) for x in corr],
            "time_s": round(time.time() - t0, 2)
        }
        print(f"      R²={np.nanmean(r2):.3f}  corr={np.nanmean(corr):.3f}")

        results[K_n] = res_k
    
    # Generate trace plots and videos at K=1 and K=5
    if generate_traces:
        for trace_K in trace_K_values:
            trace_preds = trace_preds_by_K.get(trace_K, {})
            if trace_preds:
                _plot_traces(trace_preds, b, out, worm_id, trace_K, N, label, frame_range=(400, 600))
                
                # Generate behaviour video
                if h5_path:
                    _make_behaviour_video(h5_path, trace_preds, b, out, worm_id, label,
                                          n_eigenworm_modes=6, frame_range=(400, 600), fps=20, K=trace_K)
    
    # Ridge coefficient plot removed by user request
    
    return results


def process_single_worm(h5_path, device, K_VALUES, base_out, motor_only=False):
    """Process a single worm file, both Kb1 and KbKn."""
    worm_data = load_worm_data(h5_path, n_beh_modes=6)
    u, b, worm_id = worm_data["u"], worm_data["b"], worm_data["worm_id"]
    motor_idx = worm_data.get("motor_idx")

    if motor_only:
        if motor_idx is None or len(motor_idx) == 0:
            print(f"  SKIP {worm_id}: no motor neuron indices available")
            return worm_id, {}
        u = u[:, motor_idx]
        neuron_label = "motor"
    else:
        neuron_label = "all"

    T, N, Kw = u.shape[0], u.shape[1], b.shape[1]

    # Save plots directly to base_out (root directory)
    out = base_out
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  WORM: {worm_id}  T={T}  N_{neuron_label}={N}")
    print(f"  Re-clamp every {RECLAMP_FRAMES} frames (~{RECLAMP_FRAMES/4:.0f}s)")
    print(f"{'='*70}")

    t_worm = time.time()
    all_results = {}

    print(f"\n  --- {neuron_label.upper()} NEURONS (N={N}) ---")
    
    # Run K_b = K_n (full behavior context) ONLY
    print(f"\n  Running K_b = K_n (full context)")
    results_KbKn = run_evaluation(u, b, device, K_VALUES, "run_KbKn", neuron_label, out, worm_id, h5_path=h5_path, generate_traces=True)
    with open(out / f"results_{worm_id}_{neuron_label}_KbKn.json", "w") as f:
        json.dump(results_KbKn, f, indent=2)
    _plot_lag_sweep(results_KbKn, out, worm_id, neuron_label, N, f"{worm_id}_{neuron_label}")
    all_results[f"{neuron_label}_KbKn"] = results_KbKn

    worm_time = time.time() - t_worm
    print(f"\n  Worm {worm_id} completed in {worm_time:.0f}s ({worm_time/60:.1f}m)")
    
    # Print summary table
    for run_name, results in [("KbKn", results_KbKn)]:
        print(f"\n  {'='*70}")
        print(f"  SUMMARY — {worm_id} ({run_name})")
        print(f"  {'='*70}")
        for K in K_VALUES:
            print(f"\n  K={K}:")
            print(f"  {'Model':<25} {'R²':>8} {'Corr':>8}")
            print(f"  {'-'*45}")
            res_k = results.get(K, {})
            for model in sorted(res_k.keys()):
                if model in ["K_n", "K_b"]:
                    continue
                r2 = res_k[model].get("r2_mean", np.nan)
                corr = res_k[model].get("corr_mean", np.nan)
                print(f"  {model:<25} {r2:>8.3f} {corr:>8.3f}")
    
    return worm_id, all_results


def generate_summary_plots(base_out, K_VALUES):
    """Generate summary plots from all processed worm results."""
    print(f"\n{'='*70}")
    print(f"  GENERATING SUMMARY PLOTS")
    print(f"{'='*70}")
    
    # Collect all results
    all_worm_results = {}
    
    for worm_dir in base_out.iterdir():
        if not worm_dir.is_dir():
            continue
        worm_id = worm_dir.name
        worm_results = {}
        
        for json_file in worm_dir.glob("results_*.json"):
            run_key = json_file.stem.replace("results_", "")
            with open(json_file) as f:
                worm_results[run_key] = {int(k): v for k, v in json.load(f).items()}
        
        if worm_results:
            all_worm_results[worm_id] = worm_results
    
    if not all_worm_results:
        print("  No results found to summarize.")
        return
    
    print(f"  Found {len(all_worm_results)} worms")
    
    # Generate summary for K=10 (or closest available)
    K_target = 10
    if K_target not in K_VALUES:
        K_target = min(K_VALUES, key=lambda x: abs(x - 10))
    
    for run_name in ["run1_Kb1", "run2_KbKn"]:
        _plot_summary_violin(all_worm_results, base_out, K_target, run_name)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Behaviour decoder comparison v8")
    ap.add_argument("--h5", help="Single H5 file or directory with H5 files")
    ap.add_argument("--out_dir", default="output_plots/behaviour_decoder/v8")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--summary_only", action="store_true", help="Only generate summary plots from existing results")
    ap.add_argument("--motor_only", action="store_true", help="Use motor neurons only (instead of all neurons)")
    args = ap.parse_args()

    device = args.device
    base_out = Path(args.out_dir)
    base_out.mkdir(parents=True, exist_ok=True)
    
    K_VALUES = [1, 5, 10, 15]  # Just K=10 for this test

    if args.summary_only:
        generate_summary_plots(base_out, K_VALUES)
        return

    if not args.h5:
        print("Error: --h5 required (single file or directory)")
        return

    h5_path = Path(args.h5)
    
    if h5_path.is_file():
        # Single worm
        worm_id, results = process_single_worm(str(h5_path), device, K_VALUES, base_out, motor_only=args.motor_only)
        generate_summary_plots(base_out, K_VALUES)
    elif h5_path.is_dir():
        # Multiple worms
        h5_files = sorted(h5_path.glob("*.h5"))
        print(f"Found {len(h5_files)} H5 files")
        
        t_total = time.time()
        all_results = {}
        
        for i, h5_file in enumerate(h5_files):
            print(f"\n[{i+1}/{len(h5_files)}] Processing {h5_file.name}")
            try:
                worm_id, results = process_single_worm(str(h5_file), device, K_VALUES, base_out, motor_only=args.motor_only)
                all_results[worm_id] = results
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        total_time = time.time() - t_total
        print(f"\n{'='*70}")
        print(f"Total time: {total_time:.0f}s ({total_time/60:.1f}m)")
        
        # Generate summary
        generate_summary_plots(base_out, K_VALUES)
    else:
        print(f"Error: {h5_path} not found")


if __name__ == "__main__":
    main()
