#!/usr/bin/env python3
"""
Neural Activity Decoder v5 — Connectome-Constrained Fair Evaluation.

This is a connectome-aware version of v4, where "neighbours" are defined
by the C. elegans connectome rather than all recorded neurons.

═══════════════════════════════════════════════════════════════════════════
CONNECTOME STRUCTURE
═══════════════════════════════════════════════════════════════════════════

We consider two types of connections:

  T_e   — Gap junctions (electrical synapses)
          • SYMMETRIC (bidirectional)
          • Fast, direct electrical coupling
          • Appropriate for CONCURRENT features
          
  T_syn — Chemical synapses (T_sv + T_dcv)
          • ASYMMETRIC (directed: pre → post)
          • Slower, with transmission delay
          • Appropriate for CAUSAL/LAGGED features
          • T_syn[j,i] > 0 means j → i (j is presynaptic to i)

DESIGN FOR STAGE2 MODEL
═══════════════════════════════════════════════════════════════════════════

For stage2 (LOO + free-run), we recommend:

  1. SELF-HISTORY (AR component):
     - Always include own lagged activity u_i(t-1..t-K)
     - This is the dominant predictor (R² ~ 0.84 from v4 analysis)
     
  2. GAP JUNCTION COUPLING (concurrent):
     - Use concurrent u_j(t) from gap-junction partners
     - Gap junctions are fast enough for concurrent effects
     - Keep interaction soft/gated (linear hurts, transformer helps)
     
  3. SYNAPTIC COUPLING (causal):
     - Use lagged u_j(t-1..t-K) from presynaptic partners
     - Respects directionality: only j→i where T_syn[j,i] > 0
     - Consider separate weights for excitatory/inhibitory

  4. WEIGHT INITIALIZATION:
     - Use T_e, T_syn values as weight priors/initializers
     - Or use them as mask for sparse connectivity

═══════════════════════════════════════════════════════════════════════════
CONDITIONS (6 total, matching v4)
═══════════════════════════════════════════════════════════════════════════

  self:        u_i(t-1..t-K)                           — pure AR
  conc:        u_j(t), j∈gap_neighbors(i)              — concurrent gap jxn
  causal:      u_j(t-1..t-K), j∈syn_pre(i)            — causal synaptic
  conc_causal: conc + causal                          — gap + synaptic
  conc_self:   conc + self                            — gap + AR
  causal_self: causal + self                          — synaptic + AR

═══════════════════════════════════════════════════════════════════════════
USAGE
═══════════════════════════════════════════════════════════════════════════

  # Single worm (both connectome types)
  python -m scripts.neural_activity_decoder_v5_connectome \\
      --h5 "data/.../2022-06-14-01.h5" --device cuda

  # Batch all worms
  python -m scripts.neural_activity_decoder_v5_connectome \\
      --data_dir "data/used/behaviour+neuronal activity atanas (2023)/2" \\
      --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
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

K_VALUES = [5]           # default: only K=5
N_FOLDS  = 5

# RidgeCV α candidates
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

# MLP architecture
MLP_HIDDEN = 256
MLP_LAYERS = 2

# Per-neuron training budgets
PN_MLP_EPOCHS   = 100
PN_MLP_PATIENCE = 15
PN_TRF_EPOCHS   = 75
PN_TRF_PATIENCE = 10

# Connectome paths
T_E_PATH   = ROOT / "data/used/masks+motor neurons/T_e.npy"
T_SV_PATH  = ROOT / "data/used/masks+motor neurons/T_sv.npy"
T_DCV_PATH = ROOT / "data/used/masks+motor neurons/T_dcv.npy"
NAMES_PATH = ROOT / "data/used/masks+motor neurons/neuron_names.npy"

# 6 conditions with connectome-aware descriptions
CONDITION_DEFS = OrderedDict([
    ("causal_self", {"label": "Syn+Self",         "ls": "-", "marker": "v",
                     "color": "#8c564b", "desc": "synaptic lags + self-history"}),
    ("conc_self",   {"label": "Gap+Self",         "ls": "-", "marker": "P",
                     "color": "#9467bd", "desc": "gap jxn concurrent + self-history"}),
    ("conc_causal", {"label": "Gap+Syn",          "ls": "-", "marker": "D",
                     "color": "#d62728", "desc": "gap jxn + synaptic (no self)"}),
    ("self",        {"label": "Self (AR)",         "ls": "-", "marker": "o",
                     "color": "#1f77b4", "desc": "pure autoregressive"}),
    ("conc",        {"label": "Gap (conc)",        "ls": "-", "marker": "s",
                     "color": "#ff7f0e", "desc": "gap junction concurrent only"}),
    ("causal",      {"label": "Syn (causal)",      "ls": "-", "marker": "^",
                     "color": "#2ca02c", "desc": "synaptic lagged only"}),
])


# ══════════════════════════════════════════════════════════════════════════════
# CONNECTOME LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_connectome_matrices():
    """Load atlas connectome matrices."""
    T_e = np.load(T_E_PATH)          # 302×302, symmetric (gap junctions)
    T_sv = np.load(T_SV_PATH)        # 302×302, directed (synaptic vesicles)
    T_dcv = np.load(T_DCV_PATH)      # 302×302, directed (dense core vesicles)
    T_syn = T_sv + T_dcv             # combined chemical synapses
    atlas_names = np.load(NAMES_PATH, allow_pickle=True)
    return T_e, T_syn, atlas_names


def map_connectome_to_worm(T_atlas, atlas_names, worm_labels):
    """Map atlas connectome to worm's recorded neurons.
    
    Returns (N_worm, N_worm) submatrix where:
      T_worm[j, i] > 0 means connection from j → i
      
    For gap junctions (symmetric): j↔i
    For synapses (directed): j → i (j presynaptic to i)
    """
    name_to_atlas = {n: idx for idx, n in enumerate(atlas_names)}
    
    # Map worm labels to atlas indices (-1 if not found)
    worm_atlas_idx = []
    for lab in worm_labels:
        if lab in name_to_atlas:
            worm_atlas_idx.append(name_to_atlas[lab])
        else:
            worm_atlas_idx.append(-1)
    
    N = len(worm_labels)
    T_worm = np.zeros((N, N), dtype=np.float32)
    
    for i in range(N):
        for j in range(N):
            ai, aj = worm_atlas_idx[i], worm_atlas_idx[j]
            if ai >= 0 and aj >= 0:
                # T_worm[j,i] = connection strength from j to i
                T_worm[j, i] = T_atlas[aj, ai]
    
    return T_worm


def get_connectome_neighbors(T_gap, T_syn, N):
    """Extract neighbor sets for each neuron.
    
    Returns
    -------
    gap_neighbors : dict[int, list[int]]
        For each neuron i, list of gap junction partners (bidirectional)
    syn_presynaptic : dict[int, list[int]]
        For each neuron i, list of neurons j that synapse TO i (j→i)
    """
    gap_neighbors = {}
    syn_presynaptic = {}
    
    for i in range(N):
        # Gap junctions: symmetric, so either direction counts
        gap_neighbors[i] = [j for j in range(N) if j != i and 
                            (T_gap[j, i] > 0 or T_gap[i, j] > 0)]
        
        # Synapses: directed, j→i means T_syn[j,i] > 0
        syn_presynaptic[i] = [j for j in range(N) if j != i and T_syn[j, i] > 0]
    
    return gap_neighbors, syn_presynaptic


# ══════════════════════════════════════════════════════════════════════════════
# TRF CONFIG
# ══════════════════════════════════════════════════════════════════════════════

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
# HELPERS — feature building & column indexing
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
            windows.append(u[t - K:t])
        else:
            pad = np.zeros((K - t, u.shape[1]), dtype=u.dtype)
            windows.append(np.vstack([pad, u[:t]]))
    return np.stack(windows)


def _build_windows_ext(u, indices, K):
    """Build (len(indices), K+1, N) windows: [u(t-K)..u(t-1), u(t)].

    Positions 0..K-1 are lagged steps, position K is concurrent.
    """
    windows = []
    for t in indices:
        t = int(t)
        conc = u[t : t + 1]
        if t >= K:
            lags = u[t - K:t]
        else:
            pad = np.zeros((K - t, u.shape[1]), dtype=u.dtype)
            lags = np.vstack([pad, u[:t]])
        windows.append(np.vstack([lags, conc]))
    return np.stack(windows)


# ══════════════════════════════════════════════════════════════════════════════
# CONNECTOME-AWARE COLUMN INDEXING
# ══════════════════════════════════════════════════════════════════════════════

def _self_lag_cols(ni, N, K):
    """Lag columns k=1..K for neuron ni (self-history)."""
    return np.array([k * N + ni for k in range(1, K + 1)], dtype=np.int64)


def _gap_conc_cols(ni, N, gap_neighbors):
    """Concurrent columns (k=0) for gap junction neighbors of ni."""
    cols = [j for j in gap_neighbors.get(ni, [])]
    return np.array(cols, dtype=np.int64) if cols else np.array([], dtype=np.int64)


def _syn_lag_cols(ni, N, K, syn_presynaptic):
    """Lag columns k=1..K for synaptic presynaptic partners of ni."""
    partners = syn_presynaptic.get(ni, [])
    cols = [k * N + j for k in range(1, K + 1) for j in partners]
    return np.array(cols, dtype=np.int64) if cols else np.array([], dtype=np.int64)


def _all_lag_cols(N, K):
    """All lag columns k=1..K, all neurons (for causal_self with unconstrained)."""
    return np.arange(N, (K + 1) * N)


def _make_connectome_condition_mask(ni, N, K, condition, gap_neighbors, syn_presynaptic):
    """Return ((K+1)*N,) binary mask for predicting neuron *ni* under
    *condition* using CONNECTOME-defined neighbors.
    
    Concurrent self (column ni) is NEVER included.
    
    Gap junctions → concurrent features (fast coupling)
    Chemical synapses → lagged features (slower, directed)
    """
    D = (K + 1) * N
    mask = np.zeros(D, dtype=np.float32)
    
    self_cols = _self_lag_cols(ni, N, K)
    gap_cols = _gap_conc_cols(ni, N, gap_neighbors)
    syn_cols = _syn_lag_cols(ni, N, K, syn_presynaptic)
    
    if condition == "self":
        # Pure self-history
        if len(self_cols) > 0:
            mask[self_cols] = 1.0
        
    elif condition == "conc":
        # Gap junction concurrent only
        if len(gap_cols) > 0:
            mask[gap_cols] = 1.0
        
    elif condition == "causal":
        # Synaptic presynaptic lagged only
        if len(syn_cols) > 0:
            mask[syn_cols] = 1.0
        
    elif condition == "conc_causal":
        # Gap concurrent + synaptic lagged
        if len(gap_cols) > 0:
            mask[gap_cols] = 1.0
        if len(syn_cols) > 0:
            mask[syn_cols] = 1.0
        
    elif condition == "conc_self":
        # Gap concurrent + self-history
        if len(gap_cols) > 0:
            mask[gap_cols] = 1.0
        if len(self_cols) > 0:
            mask[self_cols] = 1.0
        
    elif condition == "causal_self":
        # Synaptic lagged + self-history
        if len(syn_cols) > 0:
            mask[syn_cols] = 1.0
        if len(self_cols) > 0:
            mask[self_cols] = 1.0
    
    return mask


def _trf_connectome_cond_mask(ni, N, K, cond, gap_neighbors, syn_presynaptic):
    """(K+1, N) mask for TRF extended window with CONNECTOME constraints.

    Positions 0..K-1 = lagged steps, position K = concurrent.
    Concurrent self (col ni at position K) is NEVER included.
    """
    mask = np.zeros((K + 1, N), dtype=np.float32)
    
    gap_set = set(gap_neighbors.get(ni, []))
    syn_set = set(syn_presynaptic.get(ni, []))
    
    if cond == "self":
        mask[:K, ni] = 1.0                                    # own lags only
        
    elif cond == "causal":
        for j in syn_set:
            mask[:K, j] = 1.0                                 # presynaptic lags
            
    elif cond == "conc":
        for j in gap_set:
            mask[K, j] = 1.0                                  # gap concurrent
            
    elif cond == "conc_causal":
        for j in syn_set:
            mask[:K, j] = 1.0                                 # presynaptic lags
        for j in gap_set:
            mask[K, j] = 1.0                                  # gap concurrent
            
    elif cond == "conc_self":
        mask[:K, ni] = 1.0                                    # own lags
        for j in gap_set:
            mask[K, j] = 1.0                                  # gap concurrent
            
    elif cond == "causal_self":
        mask[:K, ni] = 1.0                                    # own lags
        for j in syn_set:
            mask[:K, j] = 1.0                                 # presynaptic lags
    
    return mask


# ══════════════════════════════════════════════════════════════════════════════
# MLP
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
    """Train MLP on pre-selected features."""
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
# TRANSFORMER
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
            pv, _, _, _ = model.forward(Xv)
            vl = F.mse_loss(pv, yv).item()
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


def _train_trf_perneuron(W_tr, y_tr_ni, W_val, y_val_ni,
                          ni, N, K, device, cond,
                          gap_neighbors, syn_presynaptic,
                          max_epochs=PN_TRF_EPOCHS,
                          patience=PN_TRF_PATIENCE,
                          warm_state=None):
    """Train single-neuron TRF on connectome-masked extended windows."""
    mask_np = _trf_connectome_cond_mask(ni, N, K, cond, gap_neighbors, syn_presynaptic)
    mask_t = torch.tensor(mask_np, dtype=torch.float32,
                          device=device).unsqueeze(0)

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
# CORE: CONNECTOME-CONSTRAINED CV
# ══════════════════════════════════════════════════════════════════════════════

def _cv_retrain_connectome(u, K, device, gap_neighbors, syn_presynaptic):
    """Fair retrain-per-condition CV with CONNECTOME constraints.

    Each condition uses connectome-defined neighbors instead of all neurons.
    """
    T, N = u.shape
    warmup = K

    CONDS  = list(CONDITION_DEFS.keys())
    MODELS = ["ridge", "mlp", "trf"]
    ho = {m: {c: np.full((T, N), np.nan) for c in CONDS} for m in MODELS}

    for fi, (ts, te) in enumerate(_make_folds(T, warmup)):
        tr = _get_train_idx(T, warmup, ts, te, buffer=K)
        mu, sig = u[tr].mean(0, keepdims=True), u[tr].std(0, keepdims=True).clip(1e-8)
        u_n = ((u - mu) / sig).astype(np.float32)
        nv = max(10, int(len(tr) * 0.15))
        tr_i, val_i = tr[:-nv], tr[-nv:]

        X = _build_features(u_n, K)
        X_te = X[ts:te]

        # ═══════ RIDGE — per-neuron with connectome masks ════════════
        for cond in CONDS:
            pred_te = np.zeros((te - ts, N), dtype=np.float32)
            for ni in range(N):
                mask = _make_connectome_condition_mask(
                    ni, N, K, cond, gap_neighbors, syn_presynaptic)
                cols = np.where(mask > 0)[0]
                
                if len(cols) == 0:
                    pred_te[:, ni] = 0.0
                    continue
                    
                X_cond = X[:, cols]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ridge = RidgeCV(alphas=RIDGE_ALPHAS)
                    ridge.fit(X_cond[tr_i], u_n[tr_i, ni])
                pred_te[:, ni] = ridge.predict(X_cond[ts:te])
            ho["ridge"][cond][ts:te] = pred_te * sig + mu

        # ═══════ MLP — per-neuron with connectome masks ═══════════════
        for cond in CONDS:
            pred_te = np.zeros((te - ts, N), dtype=np.float32)
            for ni in range(N):
                mask = _make_connectome_condition_mask(
                    ni, N, K, cond, gap_neighbors, syn_presynaptic)
                cols = np.where(mask > 0)[0]
                
                if len(cols) == 0:
                    pred_te[:, ni] = 0.0
                    continue
                
                X_cond = X[:, cols]
                mlp = _train_mlp_simple(
                    X_cond[tr_i], u_n[tr_i, ni:ni+1],
                    X_cond[val_i], u_n[val_i, ni:ni+1],
                    device, epochs=PN_MLP_EPOCHS, patience=PN_MLP_PATIENCE)
                with torch.no_grad():
                    p = mlp(torch.tensor(X_cond[ts:te], dtype=torch.float32))
                    pred_te[:, ni] = p.numpy().ravel()
                del mlp
            ho["mlp"][cond][ts:te] = pred_te * sig + mu

        # ═══════ TRF — per-neuron with connectome masks ═══════════════
        We_tr  = _build_windows_ext(u_n, tr_i,  K)
        We_val = _build_windows_ext(u_n, val_i, K)
        test_idx = np.arange(ts, te)
        We_te  = _build_windows_ext(u_n, test_idx, K)
        We_te_t = torch.tensor(We_te, dtype=torch.float32)

        for cond in CONDS:
            pred_te = np.zeros((te - ts, N), dtype=np.float32)
            for ni in range(N):
                mask = _trf_connectome_cond_mask(
                    ni, N, K, cond, gap_neighbors, syn_presynaptic)
                
                # Check if any features available
                if mask.sum() == 0:
                    pred_te[:, ni] = 0.0
                    continue
                
                trf = _train_trf_perneuron(
                    We_tr, u_n[tr_i, ni], We_val, u_n[val_i, ni],
                    ni, N, K, device, cond, gap_neighbors, syn_presynaptic)
                
                mask_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    pv, _, _, _ = trf(We_te_t * mask_t)
                    pred_te[:, ni] = pv[:, ni].numpy()
                del trf
            ho["trf"][cond][ts:te] = pred_te * sig + mu

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

def _plot_heatmap(results, out, worm_id, K_target=5):
    """Heatmap: 6 conditions × 3 models, cell = mean R²."""
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

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(data, cmap="RdYlGn", vmin=-0.5, vmax=1, aspect="auto")
    for ci in range(len(conditions)):
        for mi in range(len(models)):
            v = data[ci, mi]
            if np.isfinite(v):
                txt = f"{v:.2f}"
                c = "white" if v < 0.25 or v > 0.75 else "black"
                ax.text(mi, ci, txt, ha="center", va="center",
                        fontsize=10, color=c, fontweight="bold")
    ax.set_xticks(range(len(models)));  ax.set_xticklabels(ml, fontsize=11)
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels([f"{CONDITION_DEFS[c]['label']}" for c in conditions],
                       fontsize=10)
    plt.colorbar(im, ax=ax, label="Mean R²")
    ax.set_title(f"Connectome-Constrained Decoder — {worm_id}\n(K={K_target})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / f"heatmap_K{K_target}_{worm_id}.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved heatmap_K{K_target}_{worm_id}.png")


def _plot_connectivity_stats(gap_neighbors, syn_presynaptic, out, worm_id):
    """Plot connectivity statistics."""
    N = len(gap_neighbors)
    n_gap = [len(gap_neighbors[i]) for i in range(N)]
    n_syn = [len(syn_presynaptic[i]) for i in range(N)]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Histogram of gap junction neighbors
    ax = axes[0]
    ax.hist(n_gap, bins=20, alpha=0.7, color="#ff7f0e", edgecolor="black")
    ax.axvline(np.mean(n_gap), color="red", lw=2, ls="--", 
               label=f"mean={np.mean(n_gap):.1f}")
    ax.set_xlabel("# Gap junction partners")
    ax.set_ylabel("# Neurons")
    ax.set_title("Gap Junction Connectivity")
    ax.legend()
    
    # Histogram of synaptic presynaptic partners
    ax = axes[1]
    ax.hist(n_syn, bins=20, alpha=0.7, color="#2ca02c", edgecolor="black")
    ax.axvline(np.mean(n_syn), color="red", lw=2, ls="--",
               label=f"mean={np.mean(n_syn):.1f}")
    ax.set_xlabel("# Presynaptic partners")
    ax.set_ylabel("# Neurons")
    ax.set_title("Synaptic Connectivity (incoming)")
    ax.legend()
    
    # Scatter: gap vs synaptic
    ax = axes[2]
    ax.scatter(n_gap, n_syn, alpha=0.5, s=30, c="#8c564b")
    ax.set_xlabel("# Gap junction partners")
    ax.set_ylabel("# Presynaptic partners")
    ax.set_title("Gap vs Synaptic Connectivity")
    ax.axline((0, 0), slope=1, color="gray", lw=1, ls="--", alpha=0.5)
    
    fig.suptitle(f"Connectome Connectivity — {worm_id} (N={N})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / f"connectivity_stats_{worm_id}.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved connectivity_stats_{worm_id}.png")


def _plot_condition_comparison(results, out, worm_id, K_target=5):
    """Bar plot comparing conditions across models."""
    conditions = list(CONDITION_DEFS.keys())
    models = ["ridge", "mlp", "trf"]
    ml     = ["Ridge", "MLP", "TRF"]
    colors = ["#3498db", "#e74c3c", "#2ecc71"]
    
    res_k = results.get(K_target, {})
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    x = np.arange(len(conditions))
    width = 0.25
    
    for mi, (m, mlab, col) in enumerate(zip(models, ml, colors)):
        vals = [res_k.get(f"r2_mean_{m}_{c}", np.nan) for c in conditions]
        ax.bar(x + mi * width, vals, width, label=mlab, color=col, alpha=0.8)
    
    ax.set_xlabel("Condition")
    ax.set_ylabel("Mean R²")
    ax.set_xticks(x + width)
    ax.set_xticklabels([CONDITION_DEFS[c]["label"] for c in conditions], 
                       rotation=15, ha="right")
    ax.legend()
    ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.5)
    ax.set_ylim(-0.2, 1.0)
    ax.grid(axis="y", alpha=0.3)
    
    ax.set_title(f"Connectome-Constrained Decoder — {worm_id} (K={K_target})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / f"condition_comparison_{worm_id}.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved condition_comparison_{worm_id}.png")


def _plot_r2_vs_connectivity(results, gap_neighbors, syn_presynaptic, 
                              out, worm_id, K_target=5):
    """R² vs number of connectome neighbors."""
    res_k = results.get(K_target, {})
    N = len(gap_neighbors)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    for row, cond in enumerate(["causal_self", "conc_self"]):
        for col, model in enumerate(["ridge", "mlp", "trf"]):
            ax = axes[row, col]
            
            r2_key = f"r2_per_neuron_{model}_{cond}"
            if r2_key not in res_k:
                continue
            r2_vals = np.array(res_k[r2_key])
            
            if cond == "causal_self":
                n_conn = [len(syn_presynaptic[i]) for i in range(N)]
                xlabel = "# Presynaptic partners"
            else:
                n_conn = [len(gap_neighbors[i]) for i in range(N)]
                xlabel = "# Gap junction partners"
            
            valid = np.isfinite(r2_vals)
            ax.scatter(np.array(n_conn)[valid], r2_vals[valid], 
                       alpha=0.5, s=20, c=CONDITION_DEFS[cond]["color"])
            
            # Trend line
            if valid.sum() > 10:
                z = np.polyfit(np.array(n_conn)[valid], r2_vals[valid], 1)
                p = np.poly1d(z)
                x_line = np.linspace(0, max(n_conn), 50)
                ax.plot(x_line, p(x_line), "k--", lw=1.5, alpha=0.7)
            
            ax.set_xlabel(xlabel)
            ax.set_ylabel("R²")
            ax.set_title(f"{model.upper()} — {CONDITION_DEFS[cond]['label']}")
            ax.axhline(0, color="gray", lw=0.5, ls="--", alpha=0.5)
            ax.set_ylim(-0.5, 1.0)
    
    fig.suptitle(f"R² vs Connectivity — {worm_id} (K={K_target})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / f"r2_vs_connectivity_{worm_id}.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved r2_vs_connectivity_{worm_id}.png")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE2 MODEL DESIGN OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def save_stage2_design_info(gap_neighbors, syn_presynaptic, T_gap, T_syn,
                             worm_labels, out, worm_id):
    """Save connectome info in format useful for stage2 model construction.
    
    Outputs:
      - connectome_info.json: neighbor lists and statistics
      - T_gap_worm.npy: gap junction matrix for this worm
      - T_syn_worm.npy: synapse matrix for this worm
    """
    N = len(worm_labels)
    
    info = {
        "worm_id": worm_id,
        "N": N,
        "labels": list(worm_labels),
        
        # Statistics
        "n_gap_total": sum(len(v) for v in gap_neighbors.values()) // 2,  # symmetric
        "n_syn_total": sum(len(v) for v in syn_presynaptic.values()),
        "avg_gap_neighbors": np.mean([len(gap_neighbors[i]) for i in range(N)]),
        "avg_syn_presynaptic": np.mean([len(syn_presynaptic[i]) for i in range(N)]),
        
        # Per-neuron info
        "gap_neighbors": {str(i): gap_neighbors[i] for i in range(N)},
        "syn_presynaptic": {str(i): syn_presynaptic[i] for i in range(N)},
        
        # Design recommendations
        "stage2_design": {
            "gap_junction_usage": "concurrent features u_j(t) from gap partners",
            "synapse_usage": "lagged features u_j(t-k) from presynaptic partners",
            "recommended_model": {
                "self_history": "always include u_i(t-1..t-K)",
                "gap_coupling": "soft/gated concurrent from T_gap neighbors",
                "syn_coupling": "soft/gated lagged from T_syn pre-partners",
                "weight_init": "use T_gap, T_syn values as initialization"
            }
        }
    }
    
    with open(out / f"connectome_info_{worm_id}.json", "w") as f:
        json.dump(info, f, indent=2)
    
    np.save(out / f"T_gap_{worm_id}.npy", T_gap)
    np.save(out / f"T_syn_{worm_id}.npy", T_syn)
    
    print(f"  Saved stage2 design info for {worm_id}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation(u, device, out, worm_id, worm_labels,
                   neuron_type="all"):
    """Run connectome-constrained evaluation."""
    T, N = u.shape
    
    # Load and map connectome
    print("  Loading connectome...")
    T_e_atlas, T_syn_atlas, atlas_names = load_connectome_matrices()
    
    T_gap = map_connectome_to_worm(T_e_atlas, atlas_names, worm_labels)
    T_syn = map_connectome_to_worm(T_syn_atlas, atlas_names, worm_labels)
    
    gap_neighbors, syn_presynaptic = get_connectome_neighbors(T_gap, T_syn, N)
    
    # Stats
    n_gap_total = sum(len(v) for v in gap_neighbors.values())
    n_syn_total = sum(len(v) for v in syn_presynaptic.values())
    print(f"  Connectome mapped: {n_gap_total//2} gap junctions, "
          f"{n_syn_total} synaptic connections")
    print(f"  Avg neighbors: gap={np.mean([len(gap_neighbors[i]) for i in range(N)]):.1f}, "
          f"syn={np.mean([len(syn_presynaptic[i]) for i in range(N)]):.1f}")
    
    # Plot connectivity stats
    _plot_connectivity_stats(gap_neighbors, syn_presynaptic, out, worm_id)
    
    # Save stage2 design info
    save_stage2_design_info(gap_neighbors, syn_presynaptic, T_gap, T_syn,
                            worm_labels, out, worm_id)
    
    results = {}
    
    for K in K_VALUES:
        print(f"\n  K={K}...")
        t0 = time.time()
        
        ho = _cv_retrain_connectome(u, K, device, gap_neighbors, syn_presynaptic)
        
        res_k = {"K": K, "T": T, "N": N}
        
        for model in ["ridge", "mlp", "trf"]:
            for cond in CONDITION_DEFS.keys():
                r2, corr = _per_neuron_metrics(ho[model][cond], u)
                _store(res_k, model, cond, r2, corr)
        
        results[K] = res_k
        print(f"    K={K} done in {time.time() - t0:.0f}s")
    
    # Save results
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate plots
    print("\nGenerating plots...")
    _plot_heatmap(results, out, worm_id, K_target=5)
    _plot_condition_comparison(results, out, worm_id, K_target=5)
    _plot_r2_vs_connectivity(results, gap_neighbors, syn_presynaptic,
                              out, worm_id, K_target=5)
    
    return results


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _process_one(h5_path, neurons_tag, device, base_out):
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
        sel = sorted(motor_set)
    elif neurons_tag == "nonmotor":
        sel = [i for i in range(N_total) if i not in motor_set]
    else:
        sel = list(range(N_total))

    if not sel:
        print(f"  No neurons selected for {worm_id}, skipping")
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
    print(f"  Ridge  MLP {MLP_HIDDEN}h×{MLP_LAYERS}  TRF 128d4h")
    print(f"  6 conditions × 3 models  (CONNECTOME-CONSTRAINED)")
    print(f"{'═' * 70}")

    t0 = time.time()
    run_evaluation(u, device, out, tag, sel_labels, neurons_tag)
    total = time.time() - t0
    print(f"\n  Total: {total:.0f}s ({total / 60:.1f}min)  → {out}")


def main():
    ap = argparse.ArgumentParser(
        description="Neural Activity Decoder v5 — Connectome-Constrained")
    ap.add_argument("--h5", default=None, help="Single H5 file")
    ap.add_argument("--data_dir", default=None,
                    help="Directory of H5 files (batch)")
    ap.add_argument("--out_dir",
                    default="output_plots/neural_activity_decoder_v5_connectome")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--neurons", nargs="+", default=["all"],
                    choices=["motor", "nonmotor", "all"])
    ap.add_argument("--K_values", nargs="+", type=int, default=None,
                    help="Lag values to evaluate (default: [5])")
    ap.add_argument("--worm_ids", nargs="+", default=None,
                    help="Specific worm IDs to process")
    args = ap.parse_args()

    global K_VALUES
    if args.K_values:
        K_VALUES = args.K_values

    base_out = Path(args.out_dir)

    if args.data_dir:
        h5_files = sorted(Path(args.data_dir).glob("*.h5"))
    elif args.h5:
        h5_files = [Path(args.h5)]
    else:
        print("Provide --h5 or --data_dir")
        sys.exit(1)

    # Filter by worm_ids if specified
    if args.worm_ids:
        h5_files = [f for f in h5_files 
                    if any(wid in f.stem for wid in args.worm_ids)]

    print(f"Files: {len(h5_files)}   Subsets: {args.neurons}   "
          f"Device: {args.device}   K={K_VALUES}")
    print(f"Ridge  MLP: {MLP_HIDDEN}h×{MLP_LAYERS}  TRF: 128d4h")
    print(f"Approach: CONNECTOME-CONSTRAINED (Gap junctions + Synapses)")
    print(f"Gap junctions (T_e) → concurrent features")
    print(f"Chemical synapses (T_sv+T_dcv) → causal/lagged features")

    for h5_path in h5_files:
        for neurons_tag in args.neurons:
            _process_one(str(h5_path), neurons_tag, args.device, base_out)


if __name__ == "__main__":
    main()
