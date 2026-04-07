#!/usr/bin/env python3
"""
Neural Activity Decoder v3 — Full condition sweep.

Predicts neural activity u_i(t) from various input combinations, answering
which information sources (own history, concurrent coupling, lagged cross-
neuron influence) contribute to neural predictability.

CONDITIONS (6):
  self:        u_i(t-1..t-K)                    — pure autoregressive (AR)
  conc:        u_j(t), j≠i                      — concurrent / instantaneous coupling
  causal:      u_j(t-1..t-K), j≠i               — lagged cross-neuron (Granger-type)
  conc_causal: u_j(t) + u_j(t-1..t-K), j≠i      — concurrent + lagged cross-neuron
  conc_self:   u_j(t),j≠i + u_i(t-1..t-K)       — concurrent + own memory
  causal_self: u_j(t-1..t-K), all j              — full causal model

MODELS:
  Ridge — all 6 conditions  (fast, vectorised)
  MLP   — all 6 conditions  (per-neuron for most, multi-output for causal_self)
  TRF   — 3 conditions: self, causal, causal_self
           (causal-masked → cannot access same-timestep features;
            conc, conc_causal, conc_self skipped for TRF)

PLOTS:
  • Lag sweep: 3 rows (arch) × 2 cols (R², Corr)
  • Violin:    one panel per condition, 3 model colours
  • Heatmap:   conditions × models summary at K=5
  • Traces:    GT vs Pred at K=5 for select neurons

Usage:
  # Single worm
  python -m scripts.neural_activity_decoder_v3 \\
      --h5 "data/.../2022-06-14-01.h5" --neurons motor --device cuda

  # Batch: all worms × {motor, nonmotor, all}
  python -m scripts.neural_activity_decoder_v3 \\
      --data_dir "data/used/behaviour+neuronal activity atanas (2023)/2" \\
      --neurons motor nonmotor all --device cuda
"""
from __future__ import annotations

import argparse, glob, json, sys, time, warnings
from collections import OrderedDict
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
from sklearn.linear_model import RidgeCV


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

K_VALUES = [1, 3, 5, 10, 15, 20]
N_FOLDS  = 5
_RIDGE_ALPHAS = np.logspace(-4, 6, 30)

CONDITION_DEFS = OrderedDict([
    ("self",        {"label": "Self (AR)",    "trf": True}),
    ("conc",        {"label": "Concurrent",   "trf": False}),  # TRF can't do concurrent
    ("causal",      {"label": "Causal",       "trf": True}),
    ("conc_causal", {"label": "Conc+Causal",  "trf": False}),  # TRF can't do concurrent part
    ("conc_self",   {"label": "Conc+Self",    "trf": False}),  # TRF can't do concurrent part
    ("causal_self", {"label": "Causal+Self",  "trf": True}),
])


def _make_B_wide_config(context_length: int = 10) -> TransformerBaselineConfig:
    """B_wide_256h8 transformer config (matching behaviour decoder)."""
    cfg = TransformerBaselineConfig()
    cfg.d_model = 256;  cfg.n_heads = 8;  cfg.n_layers = 2;  cfg.d_ff = 512
    cfg.dropout = 0.1;  cfg.context_length = context_length
    cfg.lr = 1e-3;  cfg.weight_decay = 1e-4
    cfg.sigma_min = 1e-4;  cfg.sigma_max = 10.0
    cfg.max_epochs = 200;  cfg.patience = 25
    return cfg


def _make_trf_self_config(context_length: int = 10) -> TransformerBaselineConfig:
    """Smaller TRF for per-neuron self (AR) — 1D input, fast training."""
    cfg = TransformerBaselineConfig()
    cfg.d_model = 64;  cfg.n_heads = 2;  cfg.n_layers = 1;  cfg.d_ff = 128
    cfg.dropout = 0.1;  cfg.context_length = context_length
    cfg.lr = 1e-3;  cfg.weight_decay = 1e-4
    cfg.sigma_min = 1e-4;  cfg.sigma_max = 10.0
    cfg.max_epochs = 100;  cfg.patience = 15
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
    return float(np.corrcoef(gt[valid], pred[valid])[0, 1])


def _per_neuron_metrics(ho, gt):
    N = ho.shape[1]
    r2   = np.array([_r2(gt[:, i], ho[:, i]) for i in range(N)])
    corr = np.array([_corr(gt[:, i], ho[:, i]) for i in range(N)])
    return r2, corr


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _make_folds(T, warmup, n_folds=N_FOLDS):
    fold_size = (T - warmup) // n_folds
    folds = []
    for i in range(n_folds):
        ts = warmup + i * fold_size
        te = warmup + (i + 1) * fold_size if i < n_folds - 1 else T
        folds.append((ts, te))
    return folds


def _get_train_idx(T, warmup, ts, te, buffer):
    before = np.arange(warmup, ts)
    after  = np.arange(min(te + buffer, T), T)
    return np.concatenate([before, after])


def _build_lagged(x, K):
    """(T, D) → (T, K*D) lagged feature matrix [lag-1, lag-2, ..., lag-K]."""
    T, D = x.shape
    out = np.zeros((T, K * D), dtype=x.dtype)
    for lag in range(1, K + 1):
        out[lag:, (lag - 1) * D : lag * D] = x[:-lag]
    return out


def _build_windows(u, indices, K):
    """Build (len(indices), K, D) context windows."""
    windows = []
    for t in indices:
        t = int(t)
        if t >= K:
            windows.append(u[t - K : t])
        else:
            pad = np.zeros((K - t, u.shape[1]), dtype=u.dtype)
            windows.append(np.vstack([pad, u[:t]]) if t > 0 else pad)
    return np.stack(windows)


def _build_features(u_n, K, condition, ni):
    """Build flat feature matrix for neuron *ni* under *condition*.

    Parameters
    ----------
    u_n : (T, N) normalised neural activity
    K   : number of lags (ignored for 'conc')
    condition : one of CONDITION_DEFS keys
    ni  : target neuron index

    Returns (T, d_feat) feature matrix.
    """
    T, N = u_n.shape
    parts = []

    if condition == "self":
        parts.append(_build_lagged(u_n[:, ni : ni + 1], K))       # (T, K)

    elif condition == "conc":
        parts.append(np.delete(u_n, ni, axis=1))                  # (T, N-1)

    elif condition == "causal":
        others = np.delete(u_n, ni, axis=1)                       # (T, N-1)
        parts.append(_build_lagged(others, K))                     # (T, K*(N-1))

    elif condition == "conc_causal":
        others = np.delete(u_n, ni, axis=1)                       # (T, N-1)
        parts.append(others)                                       # concurrent: (T, N-1)
        parts.append(_build_lagged(others, K))                     # causal: (T, K*(N-1))

    elif condition == "conc_self":
        parts.append(np.delete(u_n, ni, axis=1))                  # (T, N-1)
        parts.append(_build_lagged(u_n[:, ni : ni + 1], K))       # (T, K)

    elif condition == "causal_self":
        parts.append(_build_lagged(u_n, K))                        # (T, K*N)

    return np.concatenate(parts, axis=1)


def _warmup_for(condition, K):
    """Minimum valid start index for a given condition and lag K."""
    if condition == "conc":
        return 1            # no lags, but need at least 1 time step
    return K                # all lagged conditions need K steps of history


# ══════════════════════════════════════════════════════════════════════════════
# RIDGE CV
# ══════════════════════════════════════════════════════════════════════════════

def _cv_ridge(u, K, condition):
    T, N = u.shape
    warmup = _warmup_for(condition, K)
    ho = np.full((T, N), np.nan)
    is_multi = (condition == "causal_self")

    for ts, te in _make_folds(T, warmup):
        tr = _get_train_idx(T, warmup, ts, te, buffer=K)
        mu, sig = u[tr].mean(0), u[tr].std(0) + 1e-8
        u_n = ((u - mu) / sig).astype(np.float32)

        if is_multi:
            X = _build_features(u_n, K, condition, ni=0)  # ni irrelevant
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = RidgeCV(alphas=_RIDGE_ALPHAS).fit(X[tr], u_n[tr])
            ho[ts:te] = model.predict(X[ts:te]) * sig + mu
        else:
            for ni in range(N):
                X = _build_features(u_n, K, condition, ni)
                y = u_n[:, ni]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = RidgeCV(alphas=_RIDGE_ALPHAS).fit(X[tr], y[tr])
                ho[ts:te, ni] = model.predict(X[ts:te]) * sig[ni] + mu[ni]

    return ho


# ══════════════════════════════════════════════════════════════════════════════
# MLP
# ══════════════════════════════════════════════════════════════════════════════

def _make_mlp(d_in, d_out, hidden=128, n_layers=2):
    layers, d = [], d_in
    for _ in range(n_layers):
        layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(0.1)]
        d = hidden
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


def _train_mlp(X_tr, y_tr, X_val, y_val, d_out, device,
               epochs=150, lr=1e-3, wd=1e-3, patience=20):
    Xt = torch.tensor(X_tr, dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=device)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.float32, device=device)

    mlp = _make_mlp(X_tr.shape[1], d_out).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=wd)

    bvl, bs, pat = float("inf"), None, 0
    for _ in range(epochs):
        mlp.train()
        loss = F.mse_loss(mlp(Xt), yt)
        opt.zero_grad(); loss.backward(); opt.step()
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


def _cv_mlp(u, K, condition, device):
    T, N = u.shape
    warmup = _warmup_for(condition, K)
    ho = np.full((T, N), np.nan)
    is_multi = (condition == "causal_self")

    for ts, te in _make_folds(T, warmup):
        tr = _get_train_idx(T, warmup, ts, te, buffer=K)
        mu, sig = u[tr].mean(0), u[tr].std(0) + 1e-8
        u_n = ((u - mu) / sig).astype(np.float32)
        nv = max(10, int(len(tr) * 0.15))
        tr_i, val_i = tr[:-nv], tr[-nv:]

        if is_multi:
            X = _build_features(u_n, K, condition, ni=0)
            mlp = _train_mlp(X[tr_i], u_n[tr_i], X[val_i], u_n[val_i], N, device)
            with torch.no_grad():
                pred = mlp(torch.tensor(X[ts:te], dtype=torch.float32)).numpy()
            ho[ts:te] = pred * sig + mu
        else:
            for ni in range(N):
                X = _build_features(u_n, K, condition, ni)
                y = u_n[:, ni : ni + 1]
                mlp = _train_mlp(X[tr_i], y[tr_i], X[val_i], y[val_i], 1, device)
                with torch.no_grad():
                    pred = mlp(torch.tensor(X[ts:te], dtype=torch.float32)).numpy()
                ho[ts:te, ni] = pred.ravel() * sig[ni] + mu[ni]

    return ho


# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORMER
# ══════════════════════════════════════════════════════════════════════════════

def _train_transformer(X_tr, y_tr, X_val, y_val, N_in, N_out, K, device, cfg=None):
    """Train transformer.  N_in = # input features per timestep."""
    Xt = torch.tensor(X_tr, dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=device)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.float32, device=device)

    if cfg is None:
        cfg = _make_B_wide_config(context_length=K)
    else:
        cfg.context_length = K

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = TemporalTransformerGaussian(n_neural=N_out, n_beh=0, cfg=cfg).to(device)
    # Override input projection to match N_in (may differ from N_out)
    if N_in != N_out:
        model.input_proj = nn.Linear(N_in, cfg.d_model).to(device)
        model.n_obs = N_in
        nn.init.xavier_uniform_(model.input_proj.weight)
        nn.init.zeros_(model.input_proj.bias)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    bvl, bs, pat = float("inf"), None, 0
    for _ in range(cfg.max_epochs):
        model.train()
        pred_mu, _, _, _ = model.forward(Xt, return_all_steps=False)
        loss = F.mse_loss(pred_mu, yt)
        opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred_v, _, _, _ = model.forward(Xv, return_all_steps=False)
            vl = F.mse_loss(pred_v, yv).item()
        if vl < bvl - 1e-6:
            bvl, bs, pat = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            pat += 1
        if pat > cfg.patience:
            break
    if bs:
        model.load_state_dict(bs)
    model.eval().cpu()
    return model


def _cv_trf_causal_self(u, K, device):
    """TRF causal_self: full context → predict all neurons."""
    T, N = u.shape
    ho = np.full((T, N), np.nan)

    for ts, te in _make_folds(T, K):
        tr = _get_train_idx(T, K, ts, te, buffer=K)
        mu, sig = u[tr].mean(0), u[tr].std(0) + 1e-8
        u_n = ((u - mu) / sig).astype(np.float32)
        nv = max(10, int(len(tr) * 0.15))
        tr_i, val_i = tr[:-nv], tr[-nv:]

        X_tr  = _build_windows(u_n, tr_i,  K)
        y_tr  = u_n[tr_i]
        X_val = _build_windows(u_n, val_i, K)
        y_val = u_n[val_i]

        model = _train_transformer(X_tr, y_tr, X_val, y_val, N, N, K, device)

        X_te = _build_windows(u_n, np.arange(ts, te), K)
        with torch.no_grad():
            pred, _, _, _ = model(torch.tensor(X_te, dtype=torch.float32))
        ho[ts:te] = pred.numpy() * sig + mu

    return ho


def _cv_trf_causal(u, K, device):
    """TRF causal: train on full context, zero out self at test time."""
    T, N = u.shape
    ho = np.full((T, N), np.nan)

    for ts, te in _make_folds(T, K):
        tr = _get_train_idx(T, K, ts, te, buffer=K)
        mu, sig = u[tr].mean(0), u[tr].std(0) + 1e-8
        u_n = ((u - mu) / sig).astype(np.float32)
        nv = max(10, int(len(tr) * 0.15))
        tr_i, val_i = tr[:-nv], tr[-nv:]

        X_tr  = _build_windows(u_n, tr_i,  K)
        y_tr  = u_n[tr_i]
        X_val = _build_windows(u_n, val_i, K)
        y_val = u_n[val_i]

        model = _train_transformer(X_tr, y_tr, X_val, y_val, N, N, K, device)

        X_te_base = _build_windows(u_n, np.arange(ts, te), K)
        for ni in range(N):
            X_te = X_te_base.copy()
            X_te[:, :, ni] = 0.0        # zero out self-history
            with torch.no_grad():
                pred, _, _, _ = model(torch.tensor(X_te, dtype=torch.float32))
            ho[ts:te, ni] = pred[:, ni].numpy() * sig[ni] + mu[ni]

    return ho


def _cv_trf_self(u, K, device):
    """TRF self: per-neuron 1D transformer (self-history only)."""
    T, N = u.shape
    ho = np.full((T, N), np.nan)
    cfg_base = _make_trf_self_config()      # smaller TRF for speed

    for ts, te in _make_folds(T, K):
        tr = _get_train_idx(T, K, ts, te, buffer=K)
        mu, sig = u[tr].mean(0), u[tr].std(0) + 1e-8
        u_n = ((u - mu) / sig).astype(np.float32)
        nv = max(10, int(len(tr) * 0.15))
        tr_i, val_i = tr[:-nv], tr[-nv:]

        for ni in range(N):
            u_1d  = u_n[:, ni : ni + 1]          # (T, 1)
            X_tr  = _build_windows(u_1d, tr_i,  K)   # (n, K, 1)
            y_tr  = u_n[tr_i, ni : ni + 1]           # (n, 1)
            X_val = _build_windows(u_1d, val_i, K)
            y_val = u_n[val_i, ni : ni + 1]

            cfg = _make_trf_self_config(context_length=K)
            model = _train_transformer(X_tr, y_tr, X_val, y_val, 1, 1, K,
                                       device, cfg=cfg)

            X_te = _build_windows(u_1d, np.arange(ts, te), K)
            with torch.no_grad():
                pred, _, _, _ = model(torch.tensor(X_te, dtype=torch.float32))
            ho[ts:te, ni] = pred[:, 0].numpy() * sig[ni] + mu[ni]

    return ho


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

_COND_STYLE = {
    "self":        ("-",  "o", "#1f77b4"),
    "conc":        ("--", "s", "#ff7f0e"),
    "causal":      ("-",  "^", "#2ca02c"),
    "conc_causal": ("--", "P", "#8c564b"),
    "conc_self":   ("--", "D", "#d62728"),
    "causal_self": ("-",  "v", "#9467bd"),
}


def _plot_lag_sweep(results, out, worm_id, N, neuron_type):
    """3 rows (arch) × 2 cols (R², Corr)."""
    K_vals = sorted(k for k in results if isinstance(k, int))
    archs = [("ridge", "Ridge"), ("mlp", "MLP"), ("trf", "TRF")]

    fig, axes = plt.subplots(3, 2, figsize=(16, 14), sharex=True)

    for row, (arch, arch_label) in enumerate(archs):
        ax_r2, ax_corr = axes[row]

        for cond in CONDITION_DEFS:
            ls, marker, color = _COND_STYLE[cond]
            label = CONDITION_DEFS[cond]["label"]
            xs, ys_r2, ys_co = [], [], []
            for K in K_vals:
                r2_key = f"r2_mean_{arch}_{cond}"
                co_key = f"corr_mean_{arch}_{cond}"
                res_k = results[K]
                if r2_key in res_k:
                    xs.append(K)
                    ys_r2.append(res_k[r2_key])
                    ys_co.append(res_k[co_key])
            if xs:
                ax_r2.plot(xs, ys_r2, ls=ls, marker=marker, ms=6, lw=1.5,
                           color=color, label=label)
                ax_corr.plot(xs, ys_co, ls=ls, marker=marker, ms=6, lw=1.5,
                             color=color, label=label)

        ax_r2.set_ylabel("Mean R²", fontsize=10)
        ax_r2.set_ylim(-0.5, 1)
        ax_r2.axhline(0, color="k", lw=0.5, ls="--", alpha=0.5)
        ax_r2.grid(alpha=0.3)
        ax_r2.set_title(f"{arch_label} — R²", fontsize=11, fontweight="bold")

        ax_corr.set_ylabel("Mean Correlation", fontsize=10)
        ax_corr.set_ylim(0, 1)
        ax_corr.grid(alpha=0.3)
        ax_corr.set_title(f"{arch_label} — Correlation", fontsize=11,
                           fontweight="bold")
        if row == 0:
            ax_corr.legend(fontsize=8, loc="lower right")

    for ax in axes[-1]:
        ax.set_xlabel("Context length K (lags)", fontsize=10)
        ax.set_xticks(K_vals)

    fig.suptitle(
        f"Neural Activity Decoder — {worm_id}  ({neuron_type}, N={N}, "
        f"{N_FOLDS}-fold CV)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fname = f"lag_sweep_{worm_id}.png"
    fig.savefig(out / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out / fname}")


def _plot_traces(gt, predictions, out, worm_id, K, N, labels=None,
                 n_show=8, frame_range=None):
    """GT vs Pred for select neurons."""
    T = gt.shape[0]
    if frame_range is None:
        s = max(K, T // 4)
        frame_range = (s, min(T, s + 300))
    fs, fe = frame_range
    t_ax = np.arange(fs, fe)
    show_idx = np.linspace(0, N - 1, min(n_show, N), dtype=int)

    colors = {
        "trf_causal_self":   "#d62728",
        "mlp_causal_self":   "#1f77b4",
        "ridge_causal_self": "#2ca02c",
    }
    n_models = len(predictions)
    fig, axes = plt.subplots(len(show_idx), n_models,
                             figsize=(4 * n_models, 2 * len(show_idx)),
                             sharex=True, squeeze=False)
    for col, (mk, pred) in enumerate(predictions.items()):
        mc = colors.get(mk, "#999")
        for row, ni in enumerate(show_idx):
            ax = axes[row, col]
            ax.plot(t_ax, gt[fs:fe, ni], color="#333", lw=1.0, alpha=0.8,
                    label="GT")
            ax.plot(t_ax, pred[fs:fe, ni], color=mc, lw=0.8, alpha=0.9,
                    ls="--", label="Pred")
            r2 = _r2(gt[fs:fe, ni], pred[fs:fe, ni])
            ax.text(0.98, 0.95, f"R²={r2:.2f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
            nlab = labels[ni] if labels and ni < len(labels) else f"n{ni}"
            if col == 0:
                ax.set_ylabel(nlab, fontsize=9)
            if row == 0:
                ax.set_title(mk.replace("_", " ").upper(), fontsize=10,
                             fontweight="bold")
            if row == len(show_idx) - 1:
                ax.set_xlabel("Frame", fontsize=9)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(f"Neural Traces — {worm_id}  K={K}", fontsize=12,
                 fontweight="bold")
    fig.tight_layout()
    fname = f"traces_K{K}_{worm_id}.png"
    fig.savefig(out / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out / fname}")


def _plot_violin_all(results, out, worm_id, N, labels=None):
    """One panel per condition, 3 model colours, per-neuron R²."""
    ks = [k for k in sorted(results) if isinstance(k, int) and k in [1, 3, 5]]
    if not ks:
        print("  Skipping violins: no K in {1,3,5}")
        return

    conditions = list(CONDITION_DEFS.keys())
    models = [("ridge", "Ridge", "#2ca02c"),
              ("mlp",   "MLP",   "#1f77b4"),
              ("trf",   "TRF",   "#d62728")]

    n_cond = len(conditions)
    fig, axes = plt.subplots(n_cond, 1,
                             figsize=(max(14, N * 0.4 + 2), 4 * n_cond))
    if n_cond == 1:
        axes = [axes]

    for ci, cond in enumerate(conditions):
        ax = axes[ci]
        n_m = len(models)
        for mi, (mk, ml, mc) in enumerate(models):
            offset = (mi - (n_m - 1) / 2) * 0.25
            vals_per = [[] for _ in range(N)]
            for K in ks:
                arr = results.get(K, {}).get(f"r2_per_neuron_{mk}_{cond}")
                if arr is not None:
                    for ni in range(N):
                        v = arr[ni] if ni < len(arr) else np.nan
                        if np.isfinite(v):
                            vals_per[ni].append(v)

            for ni in range(N):
                nv = vals_per[ni]
                if nv:
                    rng = np.random.default_rng(ni + mi * 1000)
                    jit = rng.uniform(-0.08, 0.08, size=len(nv))
                    ax.scatter(np.full(len(nv), ni + offset) + jit, nv,
                               s=12, color=mc, alpha=0.7, edgecolors="white",
                               linewidths=0.3, zorder=3)
                    med = np.median(nv)
                    ax.plot([ni + offset - 0.1, ni + offset + 0.1],
                            [med, med], color=mc, lw=1.5, zorder=4)
            ax.scatter([], [], s=30, color=mc, label=ml)

        ax.set_ylabel("R²", fontsize=10)
        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylim(bottom=-1)
        ax.set_title(CONDITION_DEFS[cond]["label"], fontsize=11,
                     fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ci == 0:
            ax.legend(fontsize=8, loc="upper right")

        xlabels = labels if labels and len(labels) == N \
            else [f"n{i}" for i in range(N)]
        ax.set_xticks(range(N))
        if ci == n_cond - 1:
            ax.set_xticklabels(xlabels,
                               fontsize=max(5, min(7, 180 // N)),
                               rotation=45, ha="right")
        else:
            ax.set_xticklabels([])

    fig.suptitle(
        f"Per-Neuron R² — {worm_id}\n"
        f"(dots = K∈{{{','.join(str(k) for k in ks)}}}, {N_FOLDS}-fold CV)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fname = f"violin_all_{worm_id}.png"
    fig.savefig(out / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out / fname}")


def _plot_heatmap(results, out, worm_id, K_target=5):
    """Heatmap: conditions × models, cell = mean R²."""
    conditions = list(CONDITION_DEFS.keys())
    models = ["ridge", "mlp", "trf"]
    ml     = ["Ridge", "MLP", "TRF"]

    res_k = results.get(K_target, {})
    data = np.full((len(conditions), len(models)), np.nan)
    for ci, c in enumerate(conditions):
        for mi, m in enumerate(models):
            key = f"r2_mean_{m}_{c}"
            if key in res_k:
                data[ci, mi] = res_k[key]

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(data, cmap="RdYlGn", vmin=-0.5, vmax=1, aspect="auto")
    for ci in range(len(conditions)):
        for mi in range(len(models)):
            v = data[ci, mi]
            if np.isfinite(v):
                ax.text(mi, ci, f"{v:.2f}", ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if v < 0.3 else "black")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(ml, fontsize=11)
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels([CONDITION_DEFS[c]["label"] for c in conditions],
                       fontsize=10)
    plt.colorbar(im, ax=ax, label="Mean R²")
    ax.set_title(f"Neural Decoder Summary — {worm_id}\n(K={K_target})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fname = f"heatmap_K{K_target}_{worm_id}.png"
    fig.savefig(out / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out / fname}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def _store(res_k, arch, cond, r2, corr):
    res_k[f"r2_mean_{arch}_{cond}"]         = float(np.nanmean(r2))
    res_k[f"corr_mean_{arch}_{cond}"]       = float(np.nanmean(corr))
    res_k[f"r2_per_neuron_{arch}_{cond}"]   = [float(x) for x in r2]
    res_k[f"corr_per_neuron_{arch}_{cond}"] = [float(x) for x in corr]


def run_evaluation(u, device, out, worm_id, neuron_type="all", labels=None):
    T, N = u.shape
    results = {}
    trace_K, trace_preds = 5, {}

    # ── 1. Concurrent (K-independent) — Ridge + MLP only ────────────────
    print(f"\n  Computing concurrent (K-independent)...")
    conc_cache = {}

    for arch, cv_fn in [("ridge", lambda: _cv_ridge(u, 1, "conc")),
                         ("mlp",   lambda: _cv_mlp(u, 1, "conc", device))]:
        t0 = time.time()
        ho = cv_fn()
        r2, corr = _per_neuron_metrics(ho, u)
        conc_cache[arch] = (r2, corr)
        print(f"    {arch.upper()} conc:  R²={np.nanmean(r2):.3f}  "
              f"corr={np.nanmean(corr):.3f}  ({time.time()-t0:.0f}s)")

    # ── 2. K-dependent conditions ────────────────────────────────────────
    for K in K_VALUES:
        print(f"\n  K={K}")
        print(f"  {'─'*55}")
        res_k = {}

        # Insert cached concurrent metrics (same for every K)
        for arch, (r2, corr) in conc_cache.items():
            _store(res_k, arch, "conc", r2, corr)

        # ── Ridge (5 K-dependent) ──
        for cond in ["self", "causal", "conc_causal", "conc_self", "causal_self"]:
            t0 = time.time()
            ho = _cv_ridge(u, K, cond)
            r2, corr = _per_neuron_metrics(ho, u)
            _store(res_k, "ridge", cond, r2, corr)
            print(f"    Ridge {cond:<12s}  R²={np.nanmean(r2):.3f}  "
                  f"corr={np.nanmean(corr):.3f}  ({time.time()-t0:.0f}s)")
            if cond == "causal_self" and K == trace_K:
                trace_preds["ridge_causal_self"] = ho.copy()

        # ── MLP (5 K-dependent) ──
        for cond in ["self", "causal", "conc_causal", "conc_self", "causal_self"]:
            t0 = time.time()
            ho = _cv_mlp(u, K, cond, device)
            r2, corr = _per_neuron_metrics(ho, u)
            _store(res_k, "mlp", cond, r2, corr)
            print(f"    MLP   {cond:<12s}  R²={np.nanmean(r2):.3f}  "
                  f"corr={np.nanmean(corr):.3f}  ({time.time()-t0:.0f}s)")
            if cond == "causal_self" and K == trace_K:
                trace_preds["mlp_causal_self"] = ho.copy()

        # ── TRF (causal-masked → no concurrent access) ──
        #   conc, conc_causal, conc_self: skipped (TRF can't do concurrent)

        t0 = time.time()
        ho_cs = _cv_trf_causal_self(u, K, device)
        r2_cs, corr_cs = _per_neuron_metrics(ho_cs, u)
        _store(res_k, "trf", "causal_self", r2_cs, corr_cs)
        print(f"    TRF   causal_self   R²={np.nanmean(r2_cs):.3f}  "
              f"corr={np.nanmean(corr_cs):.3f}  ({time.time()-t0:.0f}s)")
        if K == trace_K:
            trace_preds["trf_causal_self"] = ho_cs.copy()

        t0 = time.time()
        ho_c = _cv_trf_causal(u, K, device)
        r2_c, corr_c = _per_neuron_metrics(ho_c, u)
        _store(res_k, "trf", "causal", r2_c, corr_c)
        print(f"    TRF   causal        R²={np.nanmean(r2_c):.3f}  "
              f"corr={np.nanmean(corr_c):.3f}  ({time.time()-t0:.0f}s)")

        t0 = time.time()
        ho_s = _cv_trf_self(u, K, device)
        r2_s, corr_s = _per_neuron_metrics(ho_s, u)
        _store(res_k, "trf", "self", r2_s, corr_s)
        print(f"    TRF   self           R²={np.nanmean(r2_s):.3f}  "
              f"corr={np.nanmean(corr_s):.3f}  ({time.time()-t0:.0f}s)")

        results[K] = res_k

        # Checkpoint
        with open(out / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    # ── 3. Plots ─────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    _plot_lag_sweep(results, out, worm_id, N, neuron_type)
    if trace_preds:
        _plot_traces(u, trace_preds, out, worm_id, trace_K, N, labels=labels)
    _plot_violin_all(results, out, worm_id, N, labels)
    _plot_heatmap(results, out, worm_id, K_target=5)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _process_one(h5_path, neurons_tag, device, base_out, replot):
    worm_data = load_worm_data(h5_path, n_beh_modes=6)
    u_all = worm_data["u"]
    worm_id = worm_data["worm_id"]
    motor_idx = worm_data.get("motor_idx", [])
    all_labels = worm_data.get("labels", [])
    N_total = u_all.shape[1]

    if motor_idx is None:
        motor_idx = []
    motor_set = set(motor_idx)

    if neurons_tag == "motor":
        sel = sorted(motor_idx)
    elif neurons_tag == "nonmotor":
        sel = sorted(i for i in range(N_total) if i not in motor_set)
    else:
        sel = list(range(N_total))

    if not sel:
        print(f"  SKIP {worm_id}/{neurons_tag}: no neurons")
        return

    sel_labels = [all_labels[i] if i < len(all_labels) else f"n{i}"
                  for i in sel]
    u = u_all[:, sel].astype(np.float32)
    T, N = u.shape

    tag = f"{worm_id}_{neurons_tag}"
    out = base_out / tag
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*70}")
    print(f"  WORM: {worm_id}  subset={neurons_tag}  T={T}  N={N}")
    print(f"{'═'*70}")

    if replot:
        rp = out / "results.json"
        if not rp.exists():
            print(f"  No results.json → skip replot")
            return
        with open(rp) as f:
            results = {int(k): v for k, v in json.load(f).items()}
        _plot_lag_sweep(results, out, tag, N, neurons_tag)
        _plot_violin_all(results, out, tag, N, sel_labels)
        _plot_heatmap(results, out, tag, K_target=5)
        print("  Replot done.")
        return

    t0 = time.time()
    run_evaluation(u, device, out, tag, neurons_tag, sel_labels)
    total = time.time() - t0
    print(f"\n  Total: {total:.0f}s ({total/60:.1f}min)  → {out}")


def main():
    ap = argparse.ArgumentParser(
        description="Neural Activity Decoder v3 — full condition sweep")
    ap.add_argument("--h5", default=None, help="Single H5 file")
    ap.add_argument("--data_dir", default=None,
                    help="Directory of H5 files (batch)")
    ap.add_argument("--out_dir",
                    default="output_plots/neural_activity_decoder_v3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--neurons", nargs="+", default=["motor"],
                    choices=["motor", "nonmotor", "all"])
    ap.add_argument("--replot", action="store_true",
                    help="Regenerate plots from existing results.json")
    args = ap.parse_args()

    base_out = Path(args.out_dir)

    if args.data_dir:
        h5_files = sorted(glob.glob(str(Path(args.data_dir) / "*.h5")))
    elif args.h5:
        h5_files = [args.h5]
    else:
        print("ERROR: provide --h5 or --data_dir")
        sys.exit(1)

    print(f"Files: {len(h5_files)}   Subsets: {args.neurons}   "
          f"Device: {args.device}")

    for h5_path in h5_files:
        for neurons_tag in args.neurons:
            try:
                _process_one(h5_path, neurons_tag, args.device,
                             base_out, args.replot)
            except Exception as e:
                print(f"  ERROR processing {h5_path}/{neurons_tag}: {e}")
                import traceback; traceback.print_exc()
                continue


if __name__ == "__main__":
    main()
