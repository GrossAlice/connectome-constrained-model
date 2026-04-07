#!/usr/bin/env python3
"""
Neural Activity Decoder v4 — Fair retrain-per-condition evaluation.

Approach
--------
Retrain SEPARATE models per condition, one per neuron.
Ridge/MLP use condition-specific feature columns directly.
TRF uses condition-masked windows (dedicated per-neuron models).

6 CONDITIONS (features each model trains on):

    self:        u_i(t-1..t-K)                    — pure autoregressive (AR)
    conc:        u_j(t), j≠i                      — concurrent coupling
    causal:      u_j(t-1..t-K), j≠i               — Granger-type
    conc_causal: u_j(t) + u_j(t-1..t-K), j≠i     — concurrent + lagged
    conc_self:   u_j(t),j≠i + u_i(t-1..t-K)      — concurrent + own memory
    causal_self: u_j(t-1..t-K), all j             — lagged cross + self

    Note: concurrent self (u_i(t)) is NEVER used — it equals the target.

Models
------
    Ridge      — RidgeCV (automatic α selection), per-neuron fits
    Ridge-fast — Ridge(α=10), per-neuron fits (no CV, ~instant)
    MLP        — 256h×2, per-neuron fits (matching Ridge)
    TRF        — 128d4h causal transformer, per-neuron fits on
                 condition-masked K+1-step windows (matching Ridge)

Linearity test
--------------
    Δ(MLP − Ridge) > 0  per neuron  ⟹  nonlinear structure
    Δ(MLP − Ridge) ≈ 0  ⟹  linear relationship suffices

Usage
-----
    # Single worm
    python -m scripts.neural_activity_decoder_v4 \\
        --h5 "data/.../2022-06-14-01.h5" --neurons all --device cuda

    # Batch all worms
    python -m scripts.neural_activity_decoder_v4 \\
        --data_dir "data/used/behaviour+neuronal activity atanas (2023)/2" \\
        --neurons all --device cuda
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
from sklearn.linear_model import Ridge, RidgeCV


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

K_VALUES = [5]           # default: only K=5 (use --K_values 1 3 5 10 for full sweep)
N_FOLDS  = 5

# RidgeCV α candidates — automatically selected per fold/condition
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

# Ridge-fast: single fixed α (no CV overhead)
RIDGE_FAST_ALPHA = 10.0

# MLP architecture for retrain
MLP_HIDDEN = 256
MLP_LAYERS = 2

# Per-neuron training budgets (reduced for N×5cond models)
PN_MLP_EPOCHS   = 100
PN_MLP_PATIENCE = 15
PN_TRF_EPOCHS   = 75
PN_TRF_PATIENCE = 10

# 6 conditions, ordered from most → least features
CONDITION_DEFS = OrderedDict([
    ("causal_self", {"label": "Causal+Self",     "ls": "-", "marker": "v",
                     "color": "#8c564b"}),
    ("conc_self",   {"label": "Conc+Self",       "ls": "-", "marker": "P",
                     "color": "#9467bd"}),
    ("conc_causal", {"label": "Conc+Causal",     "ls": "-", "marker": "D",
                     "color": "#d62728"}),
    ("self",        {"label": "Self (AR)",        "ls": "-", "marker": "o",
                     "color": "#1f77b4"}),
    ("conc",        {"label": "Concurrent",       "ls": "-", "marker": "s",
                     "color": "#ff7f0e"}),
    ("causal",      {"label": "Causal (Granger)", "ls": "-", "marker": "^",
                     "color": "#2ca02c"}),
])


def _make_trf_config(context_length: int = 10) -> TransformerBaselineConfig:
    """128d4h×2L transformer (sweep-validated sweet spot)."""
    cfg = TransformerBaselineConfig()
    cfg.d_model = 128;  cfg.n_heads = 4;  cfg.n_layers = 2;  cfg.d_ff = 256
    cfg.dropout = 0.1;  cfg.context_length = context_length
    cfg.lr = 1e-3;      cfg.weight_decay = 1e-4
    cfg.sigma_min = 1e-4; cfg.sigma_max = 10.0
    cfg.max_epochs = 200;  cfg.patience = 25
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
# HELPERS  —  feature building & column indexing
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


def _build_features(x, K):
    """(T, N) → (T, (K+1)*N).

    Layout:  [concurrent(t), lag1(t-1), lag2(t-2), …, lagK(t-K)]
    Column  k*N + j  =  u_j(t − k)   for k = 0..K.
    k = 0 is the concurrent block.
    """
    T, N = x.shape
    D = (K + 1) * N
    out = np.zeros((T, D), dtype=x.dtype)
    out[:, :N] = x                                  # concurrent (lag 0)
    for k in range(1, K + 1):
        out[k:, k * N : (k + 1) * N] = x[:-k]      # lag k
    return out


def _build_windows(u, indices, K):
    """Build (len(indices), K, N) context windows for the TRF (lags only)."""
    windows = []
    for t in indices:
        t = int(t)
        if t >= K:
            windows.append(u[t - K : t])
        else:
            pad = np.zeros((K - t, u.shape[1]), dtype=u.dtype)
            windows.append(np.vstack([pad, u[:t]]) if t > 0 else pad)
    return np.stack(windows)


def _build_windows_ext(u, indices, K):
    """Build (len(indices), K+1, N) windows: [u(t-K)..u(t-1), u(t)].

    Positions 0..K-1 are lagged steps, position K is concurrent.
    """
    windows = []
    for t in indices:
        t = int(t)
        conc = u[t : t + 1]                            # (1, N)
        if t >= K:
            lags = u[t - K : t]
        else:
            pad = np.zeros((K - t, u.shape[1]), dtype=u.dtype)
            lags = np.vstack([pad, u[:t]]) if t > 0 else pad
        windows.append(np.vstack([lags, conc]))
    return np.stack(windows)


def _trf_cond_mask(ni, N, K, cond):
    """(K+1, N) mask for TRF extended window.

    Positions 0..K-1 = lagged steps, position K = concurrent.
    Concurrent self (col ni at position K) is NEVER included.
    """
    mask = np.zeros((K + 1, N), dtype=np.float32)
    if cond == "self":
        mask[:K, ni] = 1.0                             # own lags only
    elif cond == "causal":
        mask[:K, :] = 1.0;  mask[:K, ni] = 0.0        # other lags
    elif cond == "conc":
        mask[K, :] = 1.0;   mask[K, ni] = 0.0         # concurrent others
    elif cond == "conc_causal":
        mask[:, :] = 1.0;   mask[:, ni] = 0.0         # all others (lag+conc)
    elif cond == "conc_self":
        mask[:K, ni] = 1.0                             # own lags
        mask[K, :] = 1.0;   mask[K, ni] = 0.0         # concurrent others
    elif cond == "causal_self":
        mask[:K, :] = 1.0                              # all lags
    return mask


# ── column-index helpers (constant for a given N, K) ──

def _conc_other_cols(ni, N):
    """Concurrent columns for neurons j≠i."""
    return np.array([j for j in range(N) if j != ni])


def _self_lag_cols(ni, N, K):
    """Lag columns k=1..K for neuron ni."""
    return np.array([k * N + ni for k in range(1, K + 1)])


def _other_lag_cols(ni, N, K):
    """Lag columns k=1..K for neurons j≠i."""
    return np.array([k * N + j
                     for k in range(1, K + 1)
                     for j in range(N) if j != ni])


def _all_lag_cols(N, K):
    """All lag columns k=1..K, all neurons."""
    return np.arange(N, (K + 1) * N)


def _make_condition_mask(ni, N, K, condition):
    """Return ((K+1)*N,) binary mask for predicting neuron *ni* under
    *condition*.  Concurrent self (column ni) is NEVER included."""
    D = (K + 1) * N
    mask = np.zeros(D, dtype=np.float32)

    if condition == "self":
        mask[_self_lag_cols(ni, N, K)] = 1.0
    elif condition == "conc":
        mask[_conc_other_cols(ni, N)] = 1.0
    elif condition == "causal":
        mask[_other_lag_cols(ni, N, K)] = 1.0
    elif condition == "conc_causal":
        mask[_conc_other_cols(ni, N)] = 1.0
        mask[_other_lag_cols(ni, N, K)] = 1.0
    elif condition == "conc_self":
        mask[_conc_other_cols(ni, N)] = 1.0
        mask[_self_lag_cols(ni, N, K)] = 1.0
    elif condition == "causal_self":
        mask[_all_lag_cols(N, K)] = 1.0
    return mask


# ══════════════════════════════════════════════════════════════════════════════
# MLP  (256h×2, condaware for per-neuron conditions)
# ══════════════════════════════════════════════════════════════════════════════

def _make_mlp(d_in, d_out, hidden=256, n_layers=2):
    """MLP with LayerNorm + Dropout."""
    layers, d = [], d_in
    for _ in range(n_layers):
        layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden),
                   nn.ReLU(), nn.Dropout(0.1)]
        d = hidden
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


def _train_mlp_simple(X_tr, y_tr, X_val, y_val, device,
                      hidden=MLP_HIDDEN, n_layers=MLP_LAYERS,
                      epochs=200, lr=1e-3, wd=1e-3, patience=25):
    """Train MLP on pre-selected features (no masking games)."""
    Xt = torch.tensor(X_tr, dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=device)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.float32, device=device)

    d_out = y_tr.shape[1] if y_tr.ndim > 1 else 1
    mlp = _make_mlp(Xt.shape[1], d_out, hidden=hidden,
                    n_layers=n_layers).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    bvl, bs, pat = float("inf"), None, 0
    for _ in range(epochs):
        mlp.train()
        loss = F.mse_loss(mlp(Xt), yt)
        opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        mlp.eval()
        with torch.no_grad():
            vl = F.mse_loss(mlp(Xv), yv).item()
        if vl < bvl - 1e-6:
            bvl = vl
            bs = {k: v.clone() for k, v in mlp.state_dict().items()}
            pat = 0
        else:
            pat += 1
        if pat > patience:
            break

    if bs:
        mlp.load_state_dict(bs)
    mlp.eval().cpu()
    return mlp


def _train_mlp_condaware(X_tr, y_tr, X_val, y_val, N, K, device, cond,
                         hidden=MLP_HIDDEN, n_layers=MLP_LAYERS,
                         epochs=250, patience=30):
    """Train MLP with random per-sample condition masks (for per-neuron conds).

    Each training step applies a random neuron's condition mask to the
    input and computes loss only on that neuron's output.  This teaches
    the model to produce correct predictions under any neuron's mask.
    """
    D = X_tr.shape[1]
    Xt = torch.tensor(X_tr, dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=device)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.float32, device=device)

    masks = torch.stack([
        torch.tensor(_make_condition_mask(ni, N, K, cond),
                     dtype=torch.float32)
        for ni in range(N)
    ]).to(device)  # (N, D)

    mlp = _make_mlp(D, N, hidden=hidden, n_layers=n_layers).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    bvl, bs, pat = float("inf"), None, 0
    for _ in range(epochs):
        mlp.train()
        ni_rand = torch.randint(0, N, (Xt.shape[0],), device=device)
        m = masks[ni_rand]  # (batch, D)
        pred = mlp(Xt * m)
        idx = torch.arange(len(ni_rand), device=device)
        loss = F.mse_loss(pred[idx, ni_rand], yt[idx, ni_rand])
        opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        mlp.eval()
        with torch.no_grad():
            vl = 0.0
            for ni in range(min(5, N)):
                pred_v = mlp(Xv * masks[ni])
                vl += F.mse_loss(pred_v[:, ni], yv[:, ni]).item()
            vl /= min(5, N)
        if vl < bvl - 1e-6:
            bvl = vl
            bs = {k: v.clone() for k, v in mlp.state_dict().items()}
            pat = 0
        else:
            pat += 1
        if pat > patience:
            break

    if bs:
        mlp.load_state_dict(bs)
    mlp.eval().cpu()
    return mlp


# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORMER  (B_wide 256h8 from behaviour_decoder_v8)
# ══════════════════════════════════════════════════════════════════════════════

def _train_trf(X_tr, y_tr, X_val, y_val, N, K, device):
    """Train multi-output causal transformer on lagged windows."""
    Xt = torch.tensor(X_tr, dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=device)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.float32, device=device)

    cfg = _make_trf_config(context_length=K)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = TemporalTransformerGaussian(
            n_neural=N, n_beh=0, cfg=cfg
        ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)

    bvl, bs, pat = float("inf"), None, 0
    for _ in range(cfg.max_epochs):
        model.train()
        pred_mu, _, _, _ = model.forward(Xt, return_all_steps=False)
        loss = F.mse_loss(pred_mu, yt)
        opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv, _, _, _ = model.forward(Xv, return_all_steps=False)
            vl = F.mse_loss(pv, yv).item()
        if vl < bvl - 1e-6:
            bvl = vl
            bs = {k: v.clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
        if pat > cfg.patience:
            break

    if bs:
        model.load_state_dict(bs)
    model.eval().cpu()
    return model


def _train_trf_condaware(W_tr, y_tr, W_val, y_val, N, K, device, cond):
    """Train TRF with per-neuron condition masks on extended windows.

    Windows are (B, ctx, N) where ctx is typically K+1 (including
    concurrent step).  Each training step picks a random neuron nᵢ,
    applies its condition-specific (ctx, N) mask, and computes loss
    on neuron nᵢ's output.
    """
    Xt = torch.tensor(W_tr, dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=device)
    Xv = torch.tensor(W_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.float32, device=device)

    ctx = W_tr.shape[1]  # K+1 typically
    masks = torch.stack([
        torch.tensor(_trf_cond_mask(ni, N, K, cond), dtype=torch.float32)
        for ni in range(N)
    ]).to(device)  # (N, ctx, N)

    cfg = _make_trf_config(context_length=ctx)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = TemporalTransformerGaussian(
            n_neural=N, n_beh=0, cfg=cfg
        ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)

    bvl, bs, pat = float("inf"), None, 0
    for _ in range(cfg.max_epochs):
        model.train()
        ni_rand = torch.randint(0, N, (Xt.shape[0],), device=device)
        m = masks[ni_rand]  # (batch, ctx, N)
        pred_mu, _, _, _ = model(Xt * m)
        idx = torch.arange(Xt.shape[0], device=device)
        loss = F.mse_loss(pred_mu[idx, ni_rand], yt[idx, ni_rand])
        opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            vl, n_check = 0.0, min(5, N)
            for ni in range(n_check):
                pv, _, _, _ = model(Xv * masks[ni].unsqueeze(0))
                vl += F.mse_loss(pv[:, ni], yv[:, ni]).item()
            vl /= n_check

        if vl < bvl - 1e-6:
            bvl = vl
            bs = {k: v.clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
        if pat > cfg.patience:
            break

    if bs:
        model.load_state_dict(bs)
    model.eval().cpu()
    return model


def _train_trf_perneuron(W_tr, y_tr_ni, W_val, y_val_ni,
                          ni, N, K, device, cond,
                          max_epochs=PN_TRF_EPOCHS,
                          patience=PN_TRF_PATIENCE,
                          warm_state=None):
    """Train single-neuron TRF on condition-masked extended windows.

    Same architecture as the multi-output TRF, but loss is computed
    only on neuron *ni*.  Each neuron gets its own fully-dedicated
    model, matching Ridge's per-neuron approach.

    If *warm_state* is provided (state_dict from causal_self TRF),
    weights are warm-started for faster convergence.
    """
    mask_np = _trf_cond_mask(ni, N, K, cond)
    mask_t = torch.tensor(mask_np, dtype=torch.float32,
                          device=device).unsqueeze(0)          # (1, ctx, N)

    Xt = torch.tensor(W_tr, dtype=torch.float32, device=device) * mask_t
    yt = torch.tensor(y_tr_ni, dtype=torch.float32, device=device)
    Xv = torch.tensor(W_val, dtype=torch.float32, device=device) * mask_t
    yv = torch.tensor(y_val_ni, dtype=torch.float32, device=device)

    ctx = W_tr.shape[1]
    cfg = _make_trf_config(context_length=ctx)
    cfg.max_epochs = max_epochs
    cfg.patience = patience

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = TemporalTransformerGaussian(
            n_neural=N, n_beh=0, cfg=cfg
        ).to(device)
    if warm_state is not None:
        model.load_state_dict(warm_state, strict=False)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)

    bvl, bs, pat = float("inf"), None, 0
    for _ in range(max_epochs):
        model.train()
        pred_mu, _, _, _ = model(Xt)
        loss = F.mse_loss(pred_mu[:, ni], yt)
        opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            pv, _, _, _ = model(Xv)
            vl = F.mse_loss(pv[:, ni], yv).item()
        if vl < bvl - 1e-6:
            bvl = vl
            bs = {k: v.clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
        if pat > patience:
            break

    if bs:
        model.load_state_dict(bs)
    model.eval().cpu()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# CORE: FAIR RETRAIN-PER-CONDITION CV
# ══════════════════════════════════════════════════════════════════════════════

def _cv_retrain(u, K, device):
    """Fair retrain-per-condition CV.

    Each condition gets its own separately trained model(s), seeing ONLY
    the features relevant to that condition.  Each per-neuron condition
    trains N separate dedicated models (one per neuron), matching Ridge.
    All models train on the same data split (tr_i for training,
    val_i for early stopping / RidgeCV internal LOO).

    Ridge: RidgeCV (automatic α), retrain on condition features.
    MLP:   causal_self on lag features;  N separate per-neuron models
           on condition-specific features for per-neuron conds.
    TRF:   causal_self on K-step windows;  N separate per-neuron models
           on condition-masked K+1-step windows for per-neuron conds.

    Returns
    -------
    ho : dict[model][condition] → (T, N) held-out predictions
    """
    T, N = u.shape
    warmup = K

    CONDS  = list(CONDITION_DEFS.keys())
    MODELS = ["ridge", "ridge_fast", "mlp", "trf"]
    ho = {m: {c: np.full((T, N), np.nan) for c in CONDS} for m in MODELS}

    # Pre-compute column indices (constant across folds)
    conc_other = [_conc_other_cols(ni, N) for ni in range(N)]
    self_lag   = [_self_lag_cols(ni, N, K) for ni in range(N)]
    other_lag  = [_other_lag_cols(ni, N, K) for ni in range(N)]
    all_lag    = _all_lag_cols(N, K)

    for fi, (ts, te) in enumerate(_make_folds(T, warmup)):
        tr = _get_train_idx(T, warmup, ts, te, buffer=K)
        mu, sig = u[tr].mean(0), u[tr].std(0) + 1e-8
        u_n = ((u - mu) / sig).astype(np.float32)
        nv = max(10, int(len(tr) * 0.15))
        tr_i, val_i = tr[:-nv], tr[-nv:]

        X = _build_features(u_n, K)                # (T, (K+1)*N)
        X_te = X[ts:te]                            # (T_te, (K+1)*N)

        # ═══════ RIDGE — RidgeCV, same train split as MLP/TRF ════════
        for cond in CONDS:
            if cond == "causal_self":
                cols = all_lag
                pred = RidgeCV(alphas=RIDGE_ALPHAS).fit(
                    X[tr_i][:, cols], u_n[tr_i]
                ).predict(X_te[:, cols])
                ho["ridge"][cond][ts:te] = pred * sig + mu
            else:
                for ni in range(N):
                    if cond == "self":
                        cols = self_lag[ni]
                    elif cond == "causal":
                        cols = other_lag[ni]
                    elif cond == "conc":
                        cols = conc_other[ni]
                    elif cond == "conc_causal":
                        cols = np.concatenate(
                            [conc_other[ni], other_lag[ni]])
                    elif cond == "conc_self":
                        cols = np.concatenate(
                            [conc_other[ni], self_lag[ni]])
                    else:
                        continue
                    pred = RidgeCV(alphas=RIDGE_ALPHAS).fit(
                        X[tr_i][:, cols], u_n[tr_i, ni]
                    ).predict(X_te[:, cols])
                    ho["ridge"][cond][ts:te, ni] = \
                        pred * sig[ni] + mu[ni]

        # ═══════ RIDGE-FAST — fixed α, no CV ══════════════════════════
        for cond in CONDS:
            if cond == "causal_self":
                cols = all_lag
                pred = Ridge(alpha=RIDGE_FAST_ALPHA).fit(
                    X[tr_i][:, cols], u_n[tr_i]
                ).predict(X_te[:, cols])
                ho["ridge_fast"][cond][ts:te] = pred * sig + mu
            else:
                for ni in range(N):
                    if cond == "self":
                        cols = self_lag[ni]
                    elif cond == "causal":
                        cols = other_lag[ni]
                    elif cond == "conc":
                        cols = conc_other[ni]
                    elif cond == "conc_causal":
                        cols = np.concatenate(
                            [conc_other[ni], other_lag[ni]])
                    elif cond == "conc_self":
                        cols = np.concatenate(
                            [conc_other[ni], self_lag[ni]])
                    else:
                        continue
                    pred = Ridge(alpha=RIDGE_FAST_ALPHA).fit(
                        X[tr_i][:, cols], u_n[tr_i, ni]
                    ).predict(X_te[:, cols])
                    ho["ridge_fast"][cond][ts:te, ni] = \
                        pred * sig[ni] + mu[ni]

        # ═══════ MLP — retrain per condition ══════════════════════════
        # causal_self: one MLP on lag features
        X_lag = X[:, all_lag]
        mlp = _train_mlp_simple(
            X_lag[tr_i], u_n[tr_i], X_lag[val_i], u_n[val_i], device)
        with torch.no_grad():
            pred = mlp(torch.tensor(
                X_lag[ts:te], dtype=torch.float32)).numpy()
        ho["mlp"]["causal_self"][ts:te] = pred * sig + mu
        del mlp

        # Per-neuron conditions: separate per-neuron MLPs (matching Ridge)
        for cond in ["self", "causal", "conc", "conc_causal", "conc_self"]:
            for ni in range(N):
                if cond == "self":
                    cols = self_lag[ni]
                elif cond == "causal":
                    cols = other_lag[ni]
                elif cond == "conc":
                    cols = conc_other[ni]
                elif cond == "conc_causal":
                    cols = np.concatenate(
                        [conc_other[ni], other_lag[ni]])
                elif cond == "conc_self":
                    cols = np.concatenate(
                        [conc_other[ni], self_lag[ni]])
                else:
                    continue
                mlp_ni = _train_mlp_simple(
                    X[tr_i][:, cols], u_n[tr_i, ni:ni+1],
                    X[val_i][:, cols], u_n[val_i, ni:ni+1],
                    device, epochs=PN_MLP_EPOCHS,
                    patience=PN_MLP_PATIENCE)
                with torch.no_grad():
                    pred = mlp_ni(torch.tensor(
                        X_te[:, cols], dtype=torch.float32)).numpy()
                ho["mlp"][cond][ts:te, ni] = \
                    pred[:, 0] * sig[ni] + mu[ni]
                del mlp_ni
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"      MLP  {cond} done ({N} neurons)", flush=True)

        # ═══════ TRF — retrain per condition (all 6) ═════════════════
        # causal_self: standard TRF on K-step windows
        W_tr  = _build_windows(u_n, tr_i,  K)
        W_val = _build_windows(u_n, val_i, K)
        test_idx = np.arange(ts, te)
        W_te = _build_windows(u_n, test_idx, K)
        trf = _train_trf(W_tr, u_n[tr_i], W_val, u_n[val_i], N, K, device)
        with torch.no_grad():
            pred, _, _, _ = trf(torch.tensor(W_te, dtype=torch.float32))
            ho["trf"]["causal_self"][ts:te] = pred.numpy() * sig + mu
        # Save causal_self weights for warm-starting per-neuron TRFs
        trf_warm = {k: v.clone() for k, v in trf.state_dict().items()}
        del trf

        # Other 5 conditions: per-neuron TRFs on masked K+1 windows
        We_tr  = _build_windows_ext(u_n, tr_i,  K)
        We_val = _build_windows_ext(u_n, val_i, K)
        We_te  = _build_windows_ext(u_n, test_idx, K)
        We_te_t = torch.tensor(We_te, dtype=torch.float32)

        for cond in ["self", "causal", "conc", "conc_causal", "conc_self"]:
            for ni in range(N):
                trf = _train_trf_perneuron(
                    We_tr, u_n[tr_i, ni], We_val, u_n[val_i, ni],
                    ni, N, K, device, cond,
                    warm_state=trf_warm)
                with torch.no_grad():
                    m = torch.tensor(
                        _trf_cond_mask(ni, N, K, cond),
                        dtype=torch.float32).unsqueeze(0)
                    pred, _, _, _ = trf(We_te_t * m)
                    ho["trf"][cond][ts:te, ni] = \
                        pred[:, ni].numpy() * sig[ni] + mu[ni]
                del trf
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"      TRF  {cond} done ({N} neurons)", flush=True)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"    Fold {fi + 1}/{N_FOLDS} done")

    return ho


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS STORAGE
# ══════════════════════════════════════════════════════════════════════════════

def _store(res_k, arch, cond, r2, corr):
    res_k[f"r2_mean_{arch}_{cond}"]         = float(np.nanmean(r2))
    res_k[f"corr_mean_{arch}_{cond}"]       = float(np.nanmean(corr))
    res_k[f"r2_per_neuron_{arch}_{cond}"]   = [float(x) for x in r2]
    res_k[f"corr_per_neuron_{arch}_{cond}"] = [float(x) for x in corr]


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def _plot_lag_sweep(results, out, worm_id, N, neuron_type):
    """4 rows (model) × 2 cols (R², Corr)."""
    K_vals = sorted(k for k in results if isinstance(k, int))
    archs  = [("ridge", "RidgeCV"), ("ridge_fast", "Ridge(α=10)"),
              ("mlp", "MLP"), ("trf", "TRF")]

    fig, axes = plt.subplots(4, 2, figsize=(14, 18), sharex=True)

    for row, (arch, arch_label) in enumerate(archs):
        ax_r2, ax_corr = axes[row]

        for cond, info in CONDITION_DEFS.items():
            xs, ys_r2, ys_co = [], [], []
            for K in K_vals:
                rk = results[K]
                r2k = f"r2_mean_{arch}_{cond}"
                cok = f"corr_mean_{arch}_{cond}"
                if r2k in rk and np.isfinite(rk[r2k]):
                    xs.append(K)
                    ys_r2.append(rk[r2k])
                    ys_co.append(rk[cok])
            if xs:
                ax_r2.plot(xs, ys_r2, ls=info["ls"], marker=info["marker"],
                           ms=5, lw=1.5, color=info["color"],
                           label=info["label"])
                ax_corr.plot(xs, ys_co, ls=info["ls"], marker=info["marker"],
                             ms=5, lw=1.5, color=info["color"],
                             label=info["label"])

        ax_r2.set_ylabel("Mean R²");  ax_r2.set_ylim(-0.5, 1)
        ax_r2.axhline(0, color="k", lw=0.5, ls="--", alpha=0.5)
        ax_r2.grid(alpha=0.3)
        ax_r2.set_title(f"{arch_label} — R²", fontweight="bold")
        ax_corr.set_ylabel("Mean Corr");  ax_corr.set_ylim(0, 1)
        ax_corr.grid(alpha=0.3)
        ax_corr.set_title(f"{arch_label} — Correlation", fontweight="bold")
        if row == 0:
            ax_corr.legend(fontsize=7, loc="lower right")

    for ax in axes[-1]:
        ax.set_xlabel("K (lags)"); ax.set_xticks(K_vals)

    fig.suptitle(
        f"Neural Decoder v4 — {worm_id}  ({neuron_type}, N={N}, "
        f"{N_FOLDS}-fold CV, 6 conditions)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out / f"lag_sweep_{worm_id}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved lag_sweep_{worm_id}.png")


def _plot_heatmap(results, out, worm_id, K_target=5):
    """Heatmap: 6 conditions × 3 models, cell = mean R²."""
    conditions = list(CONDITION_DEFS.keys())
    models = ["ridge", "ridge_fast", "mlp", "trf"]
    ml     = ["RidgeCV", "Ridge(α=10)", "MLP", "TRF"]

    res_k = results.get(K_target, {})
    data = np.full((len(conditions), len(models)), np.nan)
    for ci, c in enumerate(conditions):
        for mi, m in enumerate(models):
            key = f"r2_mean_{m}_{c}"
            if key in res_k:
                data[ci, mi] = res_k[key]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(data, cmap="RdYlGn", vmin=-0.5, vmax=1, aspect="auto")
    for ci in range(len(conditions)):
        for mi in range(len(models)):
            v = data[ci, mi]
            if np.isfinite(v):
                ax.text(mi, ci, f"{v:.2f}", ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if v < 0.3 else "black")
            else:
                ax.text(mi, ci, "—", ha="center", va="center",
                        fontsize=10, color="#999")
    ax.set_xticks(range(len(models)));  ax.set_xticklabels(ml, fontsize=11)
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels([CONDITION_DEFS[c]["label"] for c in conditions],
                       fontsize=10)
    plt.colorbar(im, ax=ax, label="Mean R²")
    ax.set_title(f"Neural Decoder v4 — {worm_id}\n(K={K_target})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / f"heatmap_K{K_target}_{worm_id}.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved heatmap_K{K_target}_{worm_id}.png")


def _plot_linearity(results, out, worm_id, K_target=5):
    """Strip plot: Δ R² (MLP−Ridge) and (TRF−Ridge) per condition."""
    res_k = results.get(K_target, {})
    conditions = list(CONDITION_DEFS.keys())

    fig, axes = plt.subplots(1, len(conditions),
                             figsize=(3.5 * len(conditions), 5),
                             sharey=True)

    for ci, cond in enumerate(conditions):
        ax = axes[ci]
        r2_ridge = res_k.get(f"r2_per_neuron_ridge_{cond}")
        r2_mlp   = res_k.get(f"r2_per_neuron_mlp_{cond}")
        r2_trf   = res_k.get(f"r2_per_neuron_trf_{cond}")

        xticks, xlabels = [], []
        rng = np.random.default_rng(42)

        if r2_ridge is not None and r2_mlp is not None:
            r2_ridge_a = np.array(r2_ridge)
            r2_mlp_a   = np.array(r2_mlp)
            if np.any(np.isfinite(r2_ridge_a)) and \
               np.any(np.isfinite(r2_mlp_a)):
                delta_mlp = r2_mlp_a - r2_ridge_a
                ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
                jit = rng.uniform(-0.15, 0.15, len(delta_mlp))
                ax.scatter(np.zeros(len(delta_mlp)) + jit, delta_mlp,
                           s=15, color="#1f77b4", alpha=0.6,
                           edgecolors="white", linewidths=0.3, zorder=3)
                ax.plot([-0.3, 0.3], [np.nanmean(delta_mlp)] * 2,
                        color="#1f77b4", lw=3, zorder=4)
                xticks.append(0);  xlabels.append("MLP−Ridge")

            if r2_trf is not None:
                r2_trf_a = np.array(r2_trf)
                if np.any(np.isfinite(r2_trf_a)):
                    delta_trf = r2_trf_a - r2_ridge_a
                    jit2 = rng.uniform(-0.15, 0.15, len(delta_trf))
                    ax.scatter(np.ones(len(delta_trf)) + jit2, delta_trf,
                               s=15, color="#d62728", alpha=0.6,
                               edgecolors="white", linewidths=0.3, zorder=3)
                    ax.plot([0.7, 1.3], [np.nanmean(delta_trf)] * 2,
                            color="#d62728", lw=3, zorder=4)
                    xticks.append(1);  xlabels.append("TRF−Ridge")

        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_title(CONDITION_DEFS[cond]["label"], fontsize=10,
                     fontweight="bold")
        ax.grid(axis="y", alpha=0.2)
        if ci == 0:
            ax.set_ylabel("ΔR² per neuron", fontsize=10)

    fig.suptitle(
        f"Linearity Test — {worm_id}  (K={K_target})\n"
        f"Δ > 0  ⟹  nonlinear structure",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out / f"linearity_K{K_target}_{worm_id}.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved linearity_K{K_target}_{worm_id}.png")


def _plot_traces(gt, predictions, out, worm_id, K, N, labels=None,
                 n_show=8, frame_range=None):
    """GT vs Pred for select neurons (causal_self condition)."""
    T = gt.shape[0]
    if frame_range is None:
        s = max(K, T // 4)
        frame_range = (s, min(T, s + 300))
    fs, fe = frame_range
    t_ax = np.arange(fs, fe)
    show_idx = np.linspace(0, N - 1, min(n_show, N), dtype=int)

    colors = {"trf": "#d62728", "mlp": "#1f77b4", "ridge": "#2ca02c"}
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
                ax.set_title(mk.upper(), fontsize=10, fontweight="bold")
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(f"Neural Traces — {worm_id}  K={K}  (causal_self)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / f"traces_K{K}_{worm_id}.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved traces_K{K}_{worm_id}.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation(u, device, out, worm_id, neuron_type="all", labels=None):
    T, N = u.shape
    results = {}
    trace_K, trace_preds = 5, {}

    for K in K_VALUES:
        print(f"\n  K={K}")
        print(f"  {'─' * 60}")
        t0 = time.time()

        ho = _cv_retrain(u, K, device)

        res_k = {}
        for model in ["ridge", "ridge_fast", "mlp", "trf"]:
            for cond in CONDITION_DEFS:
                pred = ho[model][cond]
                if np.all(np.isnan(pred)):
                    continue
                r2, corr = _per_neuron_metrics(pred, u)
                _store(res_k, model, cond, r2, corr)
                print(f"    {model.upper():<12s} {cond:<14s} "
                      f"R²={np.nanmean(r2):.3f}  "
                      f"corr={np.nanmean(corr):.3f}")

            if K == trace_K and model != "ridge_fast":
                trace_preds[model] = ho[model]["causal_self"].copy()

        # Linearity deltas
        print(f"  {'─' * 60}")
        for cond in CONDITION_DEFS:
            r2r = np.array(res_k.get(f"r2_per_neuron_ridge_{cond}", []))
            r2m = np.array(res_k.get(f"r2_per_neuron_mlp_{cond}", []))
            r2t = np.array(res_k.get(f"r2_per_neuron_trf_{cond}", []))
            if len(r2r) and len(r2m):
                dm = np.nanmean(r2m - r2r)
                dt = np.nanmean(r2t - r2r) if len(r2t) else float("nan")
                dt_str = f"{dt:+.4f}" if np.isfinite(dt) else "  n/a "
                print(f"    Δ {cond:<14s}  MLP−Ridge={dm:+.4f}  "
                      f"TRF−Ridge={dt_str}")

        elapsed = time.time() - t0
        print(f"    [{elapsed:.0f}s]")

        results[K] = res_k
        with open(out / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    # ── Plots ──
    print("\nGenerating plots...")
    _plot_lag_sweep(results, out, worm_id, N, neuron_type)
    _plot_heatmap(results, out, worm_id, K_target=5)
    _plot_linearity(results, out, worm_id, K_target=5)
    if trace_preds:
        _plot_traces(u, trace_preds, out, worm_id, trace_K, N, labels)

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

    print(f"\n{'═' * 70}")
    print(f"  WORM: {worm_id}  subset={neurons_tag}  T={T}  N={N}")
    print(f"  RidgeCV  Ridge(α=10)  MLP {MLP_HIDDEN}h×{MLP_LAYERS}  TRF 128d4h  (all per-neuron)")
    print(f"  6 conditions × 4 models  (FAIR retrain per condition, N per-neuron)")
    print(f"{'═' * 70}")

    if replot:
        rp = out / "results.json"
        if not rp.exists():
            print("  No results.json → skip replot")
            return
        with open(rp) as f:
            results = {int(k): v for k, v in json.load(f).items()}
        _plot_lag_sweep(results, out, tag, N, neurons_tag)
        _plot_heatmap(results, out, tag, K_target=5)
        _plot_linearity(results, out, tag, K_target=5)
        print("  Replot done.")
        return

    t0 = time.time()
    run_evaluation(u, device, out, tag, neurons_tag, sel_labels)
    total = time.time() - t0
    print(f"\n  Total: {total:.0f}s ({total / 60:.1f}min)  → {out}")


def main():
    ap = argparse.ArgumentParser(
        description="Neural Activity Decoder v4 — multi-output + masking, "
                    "6 conditions × 3 models")
    ap.add_argument("--h5", default=None, help="Single H5 file")
    ap.add_argument("--data_dir", default=None,
                    help="Directory of H5 files (batch)")
    ap.add_argument("--out_dir",
                    default="output_plots/neural_activity_decoder_v4")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--neurons", nargs="+", default=["all"],
                    choices=["motor", "nonmotor", "all"])
    ap.add_argument("--K_values", nargs="+", type=int, default=None,
                    help="Lag values to evaluate (default: [5])")
    ap.add_argument("--replot", action="store_true",
                    help="Regenerate plots from existing results.json")
    args = ap.parse_args()

    # Override K_VALUES if specified
    global K_VALUES
    if args.K_values:
        K_VALUES = sorted(args.K_values)

    base_out = Path(args.out_dir)

    if args.data_dir:
        h5_files = sorted(glob.glob(str(Path(args.data_dir) / "*.h5")))
    elif args.h5:
        h5_files = [args.h5]
    else:
        print("ERROR: provide --h5 or --data_dir")
        sys.exit(1)

    print(f"Files: {len(h5_files)}   Subsets: {args.neurons}   "
          f"Device: {args.device}   K={K_VALUES}")
    print(f"RidgeCV  Ridge(α=10)  MLP: {MLP_HIDDEN}h×{MLP_LAYERS}  TRF: 128d4h  (all per-neuron)")
    print(f"Approach: FAIR retrain per condition (6 conds × 4 models, N per-neuron)")

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
