#!/usr/bin/env python3
"""
Diagnostic: why does MLP underperform Ridge in neural_activity_decoder_v4?

Hypotheses tested
-----------------
H1  CAPACITY — MLP 64h×2 is too small for (K+1)×N ≈ 576 inputs.
    → Test wider (256h, 512h) and deeper (3-layer) MLPs.

H2  MASK MISMATCH — MLP is trained with neuron-block dropout (drops all
    (K+1) features per neuron), but test-time condition masks zero out
    features organized by lag block, not neuron block.
    → Test (a) no dropout, (b) condition-aware dropout that also drops
      lag blocks, (c) retrain a separate MLP per condition.

H3  VALIDATION GAP — MLP's early stopping uses full (unmasked) features,
    but test uses masked features. Model selection is misaligned.
    → Test validation on masked features.

H4  FULL-FEATURE CONTROL — On unmasked features, does MLP match Ridge?
    → Compare Ridge vs MLP on full input (no condition masking).

Fairness issues documented
--------------------------
• Ridge: exact analytical decomposition. W×mask columns = exact prediction
  for that feature subset. No approximation.
• MLP: zero-masking at test time. A network trained on all features will
  have learned inter-feature correlations (e.g., biases, BN stats) that
  break when features are zeroed. This is NOT equivalent to Ridge's
  decomposition.
• TRF: 256d/8h/2L ≈ 200K params. MLP 64h×2 ≈ 47K params. Ridge has
  N×(K+1)×N ≈ 55K params. The capacity comparison is not balanced.

Usage:
    python -m scripts.diag_mlp_vs_ridge --device cuda
"""
from __future__ import annotations

import argparse, json, sys, time, warnings, glob
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

from baseline_transformer.dataset import load_worm_data
from sklearn.linear_model import Ridge

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

K = 5
N_FOLDS = 5
RIDGE_ALPHA = 1000.0

# Only the 3 conditions shared by all models
CONDS = ["causal_self", "self", "causal"]
COND_LABELS = {"causal_self": "Causal+Self", "self": "Self (AR)",
               "causal": "Causal (Granger)"}

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS  (copied from v4 for self-containment)
# ══════════════════════════════════════════════════════════════════════════════

def _r2(gt, pred):
    ss_res = np.nansum((gt - pred) ** 2)
    ss_tot = np.nansum((gt - np.nanmean(gt)) ** 2) + 1e-12
    return 1 - ss_res / ss_tot

def _per_neuron_r2(ho, gt):
    return np.array([_r2(gt[:, i], ho[:, i]) for i in range(ho.shape[1])])

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
    T, N = x.shape
    D = (K + 1) * N
    out = np.zeros((T, D), dtype=x.dtype)
    out[:, :N] = x
    for k in range(1, K + 1):
        out[k:, k * N : (k + 1) * N] = x[:-k]
    return out

def _conc_other_cols(ni, N):
    return np.array([j for j in range(N) if j != ni])

def _self_lag_cols(ni, N, K):
    return np.array([k * N + ni for k in range(1, K + 1)])

def _other_lag_cols(ni, N, K):
    return np.array([k * N + j for k in range(1, K + 1)
                     for j in range(N) if j != ni])

def _all_lag_cols(N, K):
    return np.arange(N, (K + 1) * N)

def _make_condition_mask(ni, N, K, condition):
    D = (K + 1) * N
    mask = np.zeros(D, dtype=np.float32)
    if condition == "self":
        mask[_self_lag_cols(ni, N, K)] = 1.0
    elif condition == "causal":
        mask[_other_lag_cols(ni, N, K)] = 1.0
    elif condition == "causal_self":
        mask[_all_lag_cols(N, K)] = 1.0
    return mask


# ══════════════════════════════════════════════════════════════════════════════
# MLP VARIANTS
# ══════════════════════════════════════════════════════════════════════════════

def _make_mlp(d_in, d_out, hidden=64, n_layers=2, use_layernorm=True):
    layers, d = [], d_in
    for _ in range(n_layers):
        layers.append(nn.Linear(d, hidden))
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden))
        layers += [nn.ReLU(), nn.Dropout(0.1)]
        d = hidden
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


def _train_mlp(X_tr, y_tr, X_val, y_val, N, K, device,
               hidden=64, n_layers=2, epochs=200, lr=1e-3, wd=1e-3,
               patience=25, neuron_drop_p=0.15,
               lag_block_drop_p=0.0,
               condition_mask=None,
               val_mask=None,
               batch_size=0):
    """
    Extended MLP trainer with multiple dropout modes.

    Parameters
    ----------
    neuron_drop_p : float
        Probability of dropping entire neuron block (K+1 features).
    lag_block_drop_p : float
        Probability of dropping entire lag block (N features at lag k).
    condition_mask : (D,) tensor or None
        If set, apply this mask to all training inputs (train on
        already-masked features = "retrain per condition").
    val_mask : (D,) tensor or None
        Mask for validation inputs. If None, uses full features.
    batch_size : int
        If > 0, use mini-batches. Otherwise full-batch.
    """
    Xt = torch.tensor(X_tr, dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=device)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.float32, device=device)

    if condition_mask is not None:
        cm = condition_mask.to(device)
        Xt = Xt * cm
        Xv = Xv * cm
    if val_mask is not None:
        vm = val_mask.to(device)
    else:
        vm = None

    d_out = y_tr.shape[1]
    mlp = _make_mlp(Xt.shape[1], d_out, hidden=hidden, n_layers=n_layers).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    bvl, bs, pat = float("inf"), None, 0
    n_tr = Xt.shape[0]

    for ep in range(epochs):
        mlp.train()

        # --- Training step (with optional mini-batching) ---
        if batch_size > 0 and n_tr > batch_size:
            perm = torch.randperm(n_tr, device=device)
            epoch_loss = 0.0
            for bi in range(0, n_tr, batch_size):
                idx = perm[bi:bi+batch_size]
                xb, yb = Xt[idx], yt[idx]
                xb = _apply_dropout(xb, N, K, neuron_drop_p, lag_block_drop_p, device)
                loss = F.mse_loss(mlp(xb), yb)
                opt.zero_grad(); loss.backward(); opt.step()
                epoch_loss += loss.item() * len(idx)
        else:
            xb = _apply_dropout(Xt, N, K, neuron_drop_p, lag_block_drop_p, device)
            loss = F.mse_loss(mlp(xb), yt)
            opt.zero_grad(); loss.backward(); opt.step()

        sched.step()

        # --- Validation ---
        mlp.eval()
        with torch.no_grad():
            xv_in = Xv * vm if vm is not None else Xv
            vl = F.mse_loss(mlp(xv_in), yv).item()
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


def _apply_dropout(X, N, K, neuron_drop_p, lag_block_drop_p, device):
    """Apply neuron-block and/or lag-block dropout."""
    if neuron_drop_p > 0:
        keep = (torch.rand(X.shape[0], N, device=device) > neuron_drop_p).float()
        X = X * keep.repeat(1, K + 1)
    if lag_block_drop_p > 0:
        # Drop entire lag blocks (each of N features)
        keep = (torch.rand(X.shape[0], K + 1, device=device) > lag_block_drop_p).float()
        X = X * keep.repeat_interleave(N, dim=1)
    return X


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

def _define_experiments():
    """
    Each experiment defines MLP hyperparams and evaluation strategy.
    Returns dict of {name: config_dict}.
    """
    exps = OrderedDict()

    # ── Baselines ──
    exps["ridge"] = {"type": "ridge"}
    exps["mlp_64h2_v4orig"] = {
        "type": "mlp", "hidden": 64, "n_layers": 2,
        "neuron_drop_p": 0.15, "lag_block_drop_p": 0.0,
        "eval": "mask_test", "note": "Original v4 MLP"
    }

    # ── H1: Capacity ──
    exps["mlp_256h2"] = {
        "type": "mlp", "hidden": 256, "n_layers": 2,
        "neuron_drop_p": 0.15, "lag_block_drop_p": 0.0,
        "eval": "mask_test", "note": "Wider MLP"
    }
    exps["mlp_256h3"] = {
        "type": "mlp", "hidden": 256, "n_layers": 3,
        "neuron_drop_p": 0.15, "lag_block_drop_p": 0.0,
        "eval": "mask_test", "note": "Wider + deeper"
    }

    # ── H2a: No dropout at all (train full, test masked) ──
    exps["mlp_64h2_nodrop"] = {
        "type": "mlp", "hidden": 64, "n_layers": 2,
        "neuron_drop_p": 0.0, "lag_block_drop_p": 0.0,
        "eval": "mask_test", "note": "No input dropout"
    }
    exps["mlp_256h2_nodrop"] = {
        "type": "mlp", "hidden": 256, "n_layers": 2,
        "neuron_drop_p": 0.0, "lag_block_drop_p": 0.0,
        "eval": "mask_test", "note": "Wide + no dropout"
    }

    # ── H2b: Add lag-block dropout alongside neuron-block ──
    exps["mlp_256h2_lagdrop"] = {
        "type": "mlp", "hidden": 256, "n_layers": 2,
        "neuron_drop_p": 0.10, "lag_block_drop_p": 0.15,
        "eval": "mask_test", "note": "Neuron + lag block dropout"
    }

    # ── H2c: Retrain per condition (fairest comparison) ──
    exps["mlp_256h2_retrain"] = {
        "type": "mlp", "hidden": 256, "n_layers": 2,
        "neuron_drop_p": 0.0, "lag_block_drop_p": 0.0,
        "eval": "retrain_per_cond", "note": "Retrain per condition"
    }

    # ── H3: Validate on masked features ──
    exps["mlp_256h2_valmasked"] = {
        "type": "mlp", "hidden": 256, "n_layers": 2,
        "neuron_drop_p": 0.15, "lag_block_drop_p": 0.0,
        "eval": "mask_test_val_masked", "note": "Val on masked features"
    }

    # ── H4: Full-feature control (no masking at test) ──
    exps["mlp_256h2_full"] = {
        "type": "mlp", "hidden": 256, "n_layers": 2,
        "neuron_drop_p": 0.0, "lag_block_drop_p": 0.0,
        "eval": "full_only", "note": "Full features (no masking)"
    }
    exps["ridge_full"] = {"type": "ridge_full", "note": "Ridge full features"}

    return exps


# ══════════════════════════════════════════════════════════════════════════════
# CORE CV LOOP
# ══════════════════════════════════════════════════════════════════════════════

def _run_one_worm(u, device, experiments):
    """Run all experiments on one worm, return results dict."""
    T, N = u.shape
    warmup = K
    D = (K + 1) * N

    results = {}

    for exp_name, cfg in experiments.items():
        print(f"    {exp_name} ...", end=" ", flush=True)
        t0 = time.time()

        if cfg["type"] == "ridge":
            # Ridge with analytical masking (same as v4)
            ho = {c: np.full((T, N), np.nan) for c in CONDS}
            conc_other = [_conc_other_cols(ni, N) for ni in range(N)]
            self_lag   = [_self_lag_cols(ni, N, K) for ni in range(N)]
            other_lag  = [_other_lag_cols(ni, N, K) for ni in range(N)]

            for fi, (ts, te) in enumerate(_make_folds(T, warmup)):
                tr = _get_train_idx(T, warmup, ts, te, buffer=K)
                mu, sig = u[tr].mean(0), u[tr].std(0) + 1e-8
                u_n = ((u - mu) / sig).astype(np.float32)
                X = _build_features(u_n, K)
                X_te = X[ts:te]

                ridge = Ridge(alpha=RIDGE_ALPHA).fit(X[tr], u_n[tr])
                W, br = ridge.coef_, ridge.intercept_

                for ni in range(N):
                    sl = X_te[:, self_lag[ni]]   @ W[ni, self_lag[ni]]
                    ol = X_te[:, other_lag[ni]]  @ W[ni, other_lag[ni]]
                    b  = br[ni]
                    ho["causal_self"][ts:te, ni] = (sl + ol + b) * sig[ni] + mu[ni]
                    ho["self"][ts:te, ni]        = (sl + b)      * sig[ni] + mu[ni]
                    ho["causal"][ts:te, ni]      = (ol + b)      * sig[ni] + mu[ni]

            r2s = {c: _per_neuron_r2(ho[c], u) for c in CONDS}

        elif cfg["type"] == "ridge_full":
            # Ridge on full features (no masking), as upper bound
            ho_full = np.full((T, N), np.nan)
            for fi, (ts, te) in enumerate(_make_folds(T, warmup)):
                tr = _get_train_idx(T, warmup, ts, te, buffer=K)
                mu, sig = u[tr].mean(0), u[tr].std(0) + 1e-8
                u_n = ((u - mu) / sig).astype(np.float32)
                X = _build_features(u_n, K)
                pred = Ridge(alpha=RIDGE_ALPHA).fit(X[tr], u_n[tr]).predict(X[ts:te])
                ho_full[ts:te] = pred * sig + mu
            r2s = {"full": _per_neuron_r2(ho_full, u)}

        elif cfg["type"] == "mlp":
            eval_mode = cfg.get("eval", "mask_test")

            if eval_mode == "retrain_per_cond":
                r2s = _run_mlp_retrain_per_cond(u, N, D, cfg, device)
            elif eval_mode == "full_only":
                r2s = _run_mlp_full_only(u, N, D, cfg, device)
            elif eval_mode == "mask_test_val_masked":
                r2s = _run_mlp_mask_val(u, N, D, cfg, device)
            else:
                r2s = _run_mlp_mask_test(u, N, D, cfg, device)

        elapsed = time.time() - t0
        results[exp_name] = {
            "r2": {c: {"mean": float(np.nanmean(v)),
                        "per_neuron": [float(x) for x in v]}
                   for c, v in r2s.items()},
            "time": elapsed,
            "note": cfg.get("note", ""),
        }
        # Print summary
        summary = "  ".join(f"{c}={np.nanmean(v):.3f}"
                            for c, v in r2s.items())
        print(f"{summary}  [{elapsed:.1f}s]")

    return results


def _run_mlp_mask_test(u, N, D, cfg, device):
    """Standard approach: train on full features, mask at test time."""
    T = u.shape[0]
    warmup = K
    all_lag = _all_lag_cols(N, K)

    ho = {c: np.full((T, N), np.nan) for c in CONDS}

    for fi, (ts, te) in enumerate(_make_folds(T, warmup)):
        tr = _get_train_idx(T, warmup, ts, te, buffer=K)
        mu, sig = u[tr].mean(0), u[tr].std(0) + 1e-8
        u_n = ((u - mu) / sig).astype(np.float32)
        nv = max(10, int(len(tr) * 0.15))
        tr_i, val_i = tr[:-nv], tr[-nv:]
        X = _build_features(u_n, K)

        mlp = _train_mlp(X[tr_i], u_n[tr_i], X[val_i], u_n[val_i],
                         N, K, device,
                         hidden=cfg["hidden"], n_layers=cfg["n_layers"],
                         neuron_drop_p=cfg["neuron_drop_p"],
                         lag_block_drop_p=cfg["lag_block_drop_p"])

        X_te_t = torch.tensor(X[ts:te], dtype=torch.float32)

        with torch.no_grad():
            # causal_self
            mask_cs = torch.zeros(D, dtype=torch.float32)
            mask_cs[all_lag] = 1.0
            pred = mlp(X_te_t * mask_cs).numpy()
            ho["causal_self"][ts:te] = pred * sig + mu

            # self & causal per neuron
            for ni in range(N):
                for cond in ["self", "causal"]:
                    m = torch.tensor(_make_condition_mask(ni, N, K, cond),
                                     dtype=torch.float32)
                    pred = mlp(X_te_t * m).numpy()
                    ho[cond][ts:te, ni] = pred[:, ni] * sig[ni] + mu[ni]
        del mlp

    return {c: _per_neuron_r2(ho[c], u) for c in CONDS}


def _run_mlp_retrain_per_cond(u, N, D, cfg, device):
    """FAIR test: retrain a separate MLP per condition on masked features."""
    T = u.shape[0]
    warmup = K
    all_lag = _all_lag_cols(N, K)

    ho = {c: np.full((T, N), np.nan) for c in CONDS}

    for fi, (ts, te) in enumerate(_make_folds(T, warmup)):
        tr = _get_train_idx(T, warmup, ts, te, buffer=K)
        mu, sig = u[tr].mean(0), u[tr].std(0) + 1e-8
        u_n = ((u - mu) / sig).astype(np.float32)
        nv = max(10, int(len(tr) * 0.15))
        tr_i, val_i = tr[:-nv], tr[-nv:]
        X = _build_features(u_n, K)

        # causal_self: one model (same mask for all neurons)
        mask_cs = np.zeros(D, dtype=np.float32)
        mask_cs[all_lag] = 1.0
        mask_cs_t = torch.tensor(mask_cs, dtype=torch.float32)
        X_masked = X * mask_cs[None, :]
        mlp = _train_mlp(X_masked[tr_i], u_n[tr_i],
                         X_masked[val_i], u_n[val_i],
                         N, K, device,
                         hidden=cfg["hidden"], n_layers=cfg["n_layers"],
                         neuron_drop_p=0, lag_block_drop_p=0)
        with torch.no_grad():
            pred = mlp(torch.tensor(X_masked[ts:te], dtype=torch.float32)).numpy()
            ho["causal_self"][ts:te] = pred * sig + mu
        del mlp

        # self & causal: must handle per-neuron masks
        # Strategy: for each condition, build the union of active columns
        # (they differ per neuron), train on the FULL masked X, predict all
        for cond in ["self", "causal"]:
            # Per-neuron: retrain would need N models — too expensive.
            # Instead: train on features where ALL neurons' masks overlap,
            # plus use the per-neuron mask at test time.
            # Actually, let's train on a "random neuron mask" per sample:
            mlp = _train_mlp_condaware(X[tr_i], u_n[tr_i],
                                        X[val_i], u_n[val_i],
                                        N, K, device, cond,
                                        hidden=cfg["hidden"],
                                        n_layers=cfg["n_layers"])
            X_te_t = torch.tensor(X[ts:te], dtype=torch.float32)
            with torch.no_grad():
                for ni in range(N):
                    m = torch.tensor(_make_condition_mask(ni, N, K, cond),
                                     dtype=torch.float32)
                    pred = mlp(X_te_t * m).numpy()
                    ho[cond][ts:te, ni] = pred[:, ni] * sig[ni] + mu[ni]
            del mlp

    return {c: _per_neuron_r2(ho[c], u) for c in CONDS}


def _train_mlp_condaware(X_tr, y_tr, X_val, y_val, N, K, device, cond,
                          hidden=256, n_layers=2, epochs=200, patience=25):
    """Train MLP where each sample gets a random neuron's condition mask.

    This teaches the network to operate under the condition-specific
    masking pattern, but for random target neurons — so one model handles all.
    """
    D = X_tr.shape[1]
    Xt = torch.tensor(X_tr, dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=device)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.float32, device=device)

    # Pre-compute all N masks
    masks = torch.stack([
        torch.tensor(_make_condition_mask(ni, N, K, cond), dtype=torch.float32)
        for ni in range(N)
    ]).to(device)  # (N, D)

    mlp = _make_mlp(D, N, hidden=hidden, n_layers=n_layers).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    bvl, bs, pat = float("inf"), None, 0
    for ep in range(epochs):
        mlp.train()
        # Random neuron mask per sample
        ni_rand = torch.randint(0, N, (Xt.shape[0],), device=device)
        m = masks[ni_rand]  # (batch, D)
        pred = mlp(Xt * m)
        # Only compute loss on the target neuron for each sample
        loss = F.mse_loss(
            pred[torch.arange(len(ni_rand), device=device), ni_rand],
            yt[torch.arange(len(ni_rand), device=device), ni_rand]
        )
        opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        mlp.eval()
        with torch.no_grad():
            # Validate on a fixed set of neurons (first 5, random masks)
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


def _run_mlp_mask_val(u, N, D, cfg, device):
    """Train with dropout, but validate on causal_self mask (aligned eval)."""
    T = u.shape[0]
    warmup = K
    all_lag = _all_lag_cols(N, K)

    ho = {c: np.full((T, N), np.nan) for c in CONDS}
    val_mask_cs = torch.zeros(D, dtype=torch.float32)
    val_mask_cs[all_lag] = 1.0

    for fi, (ts, te) in enumerate(_make_folds(T, warmup)):
        tr = _get_train_idx(T, warmup, ts, te, buffer=K)
        mu, sig = u[tr].mean(0), u[tr].std(0) + 1e-8
        u_n = ((u - mu) / sig).astype(np.float32)
        nv = max(10, int(len(tr) * 0.15))
        tr_i, val_i = tr[:-nv], tr[-nv:]
        X = _build_features(u_n, K)

        mlp = _train_mlp(X[tr_i], u_n[tr_i], X[val_i], u_n[val_i],
                         N, K, device,
                         hidden=cfg["hidden"], n_layers=cfg["n_layers"],
                         neuron_drop_p=cfg["neuron_drop_p"],
                         lag_block_drop_p=cfg["lag_block_drop_p"],
                         val_mask=val_mask_cs)

        X_te_t = torch.tensor(X[ts:te], dtype=torch.float32)
        mask_cs = torch.zeros(D, dtype=torch.float32)
        mask_cs[all_lag] = 1.0

        with torch.no_grad():
            pred = mlp(X_te_t * mask_cs).numpy()
            ho["causal_self"][ts:te] = pred * sig + mu
            for ni in range(N):
                for cond in ["self", "causal"]:
                    m = torch.tensor(_make_condition_mask(ni, N, K, cond),
                                     dtype=torch.float32)
                    pred = mlp(X_te_t * m).numpy()
                    ho[cond][ts:te, ni] = pred[:, ni] * sig[ni] + mu[ni]
        del mlp

    return {c: _per_neuron_r2(ho[c], u) for c in CONDS}


def _run_mlp_full_only(u, N, D, cfg, device):
    """MLP on full features (no masking), as sanity control."""
    T = u.shape[0]
    warmup = K
    ho_full = np.full((T, N), np.nan)

    for fi, (ts, te) in enumerate(_make_folds(T, warmup)):
        tr = _get_train_idx(T, warmup, ts, te, buffer=K)
        mu, sig = u[tr].mean(0), u[tr].std(0) + 1e-8
        u_n = ((u - mu) / sig).astype(np.float32)
        nv = max(10, int(len(tr) * 0.15))
        tr_i, val_i = tr[:-nv], tr[-nv:]
        X = _build_features(u_n, K)

        mlp = _train_mlp(X[tr_i], u_n[tr_i], X[val_i], u_n[val_i],
                         N, K, device,
                         hidden=cfg["hidden"], n_layers=cfg["n_layers"],
                         neuron_drop_p=0, lag_block_drop_p=0)

        X_te_t = torch.tensor(X[ts:te], dtype=torch.float32)
        with torch.no_grad():
            pred = mlp(X_te_t).numpy()
            ho_full[ts:te] = pred * sig + mu
        del mlp

    return {"full": _per_neuron_r2(ho_full, u)}


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def _plot_results(all_results, out):
    """Bar chart: experiments × conditions, averaged across worms."""
    # Collect mean R² per experiment per condition
    exp_names = list(next(iter(all_results.values())).keys())
    worm_ids = list(all_results.keys())
    n_worms = len(worm_ids)

    # Which conditions does each experiment have?
    all_conds_used = set()
    for wid in worm_ids:
        for ename in exp_names:
            all_conds_used.update(all_results[wid][ename]["r2"].keys())
    conds_order = [c for c in CONDS if c in all_conds_used]
    if "full" in all_conds_used:
        conds_order.append("full")

    cond_labels = {**COND_LABELS, "full": "Full (no mask)"}

    # Compute means ± SEM across worms
    means = {}
    sems = {}
    for ename in exp_names:
        for c in conds_order:
            vals = []
            for wid in worm_ids:
                r2d = all_results[wid][ename]["r2"]
                if c in r2d:
                    vals.append(r2d[c]["mean"])
            if vals:
                means[(ename, c)] = np.mean(vals)
                sems[(ename, c)] = np.std(vals) / max(1, np.sqrt(len(vals)))
            else:
                means[(ename, c)] = np.nan
                sems[(ename, c)] = 0

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(16, 7))
    n_exp = len(exp_names)
    n_cond = len(conds_order)
    bar_w = 0.8 / n_cond
    x = np.arange(n_exp)

    cond_colors = {
        "causal_self": "#8c564b", "self": "#1f77b4", "causal": "#2ca02c",
        "full": "#333333"
    }

    for ci, c in enumerate(conds_order):
        ys = [means.get((e, c), np.nan) for e in exp_names]
        es = [sems.get((e, c), 0) for e in exp_names]
        offset = (ci - (n_cond - 1) / 2) * bar_w
        bars = ax.bar(x + offset, ys, bar_w * 0.9, yerr=es,
                      label=cond_labels.get(c, c),
                      color=cond_colors.get(c, "#999"),
                      capsize=2, edgecolor="white", linewidth=0.5,
                      error_kw=dict(lw=1))
        for bar, y in zip(bars, ys):
            if np.isfinite(y):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{y:.2f}", ha="center", va="bottom", fontsize=6,
                        rotation=45)

    # Get notes for x-labels
    sample_res = all_results[worm_ids[0]]
    xlabels = [f"{e}\n({sample_res[e]['note']})" for e in exp_names]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=7.5, ha="center")
    ax.set_ylabel("Mean R²  (± SEM across worms)", fontsize=11)
    ax.set_ylim(-0.3, 0.8)
    ax.axhline(0, color="k", lw=0.6, ls="--", alpha=0.4)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    ax.set_title(f"MLP Diagnostic — K={K}, {n_worms} worms\n"
                 f"Why does MLP underperform Ridge? Fairness analysis",
                 fontsize=12, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out / "diag_bar.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out / 'diag_bar.png'}")

    # ── Per-condition detail plots ──
    for c in ["causal_self", "self", "causal"]:
        fig, ax = plt.subplots(figsize=(12, 5))
        exp_with_cond = [e for e in exp_names
                         if not np.isnan(means.get((e, c), np.nan))]
        x2 = np.arange(len(exp_with_cond))
        ys = [means[(e, c)] for e in exp_with_cond]
        es = [sems[(e, c)] for e in exp_with_cond]
        colors = []
        for e in exp_with_cond:
            if "ridge" in e:
                colors.append("#2ca02c")
            elif "retrain" in e:
                colors.append("#d62728")
            elif "full" in e and "mlp" in e:
                colors.append("#333")
            else:
                colors.append("#1f77b4")

        bars = ax.bar(x2, ys, 0.6, yerr=es, color=colors, capsize=3,
                      edgecolor="white", linewidth=0.5)
        for bar, y in zip(bars, ys):
            if np.isfinite(y):
                ax.text(bar.get_x() + bar.get_width()/2,
                        max(y, 0) + 0.02,
                        f"{y:.3f}", ha="center", va="bottom",
                        fontsize=8, fontweight="bold")

        xlabels2 = [f"{e}\n({sample_res[e]['note']})" for e in exp_with_cond]
        ax.set_xticks(x2)
        ax.set_xticklabels(xlabels2, fontsize=7, ha="center")
        ax.set_ylabel("Mean R²", fontsize=11)
        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
        ax.grid(axis="y", alpha=0.2)
        ax.set_title(f"Condition: {cond_labels.get(c, c)}  "
                     f"(K={K}, {n_worms} worms)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out / f"diag_{c}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ {out / f'diag_{c}.png'}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",
                    default="data/used/behaviour+neuronal activity atanas (2023)/2")
    ap.add_argument("--n_worms", type=int, default=3,
                    help="Number of worms to test (for speed)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="output_plots/neural_activity_decoder_v4/diag")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(glob.glob(str(Path(args.data_dir) / "*.h5")))
    # Take a spread: first, middle, last
    if len(h5_files) > args.n_worms:
        indices = np.linspace(0, len(h5_files)-1, args.n_worms, dtype=int)
        h5_files = [h5_files[i] for i in indices]

    experiments = _define_experiments()

    print(f"MLP Diagnostic — K={K}, {len(h5_files)} worms, "
          f"{len(experiments)} experiments")
    print(f"Device: {args.device}")
    print(f"Experiments: {list(experiments.keys())}")
    print()

    all_results = {}
    for h5_path in h5_files:
        worm_data = load_worm_data(h5_path, n_beh_modes=6)
        u = worm_data["u"].astype(np.float32)
        worm_id = worm_data["worm_id"]
        T, N = u.shape

        print(f"  ═══ {worm_id}  T={T}  N={N} ═══")
        results = _run_one_worm(u, args.device, experiments)
        all_results[worm_id] = results

        # Save incrementally
        with open(out / "results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # ── Plots ──
    print("\nGenerating plots...")
    _plot_results(all_results, out)
    print("\nDone!")


if __name__ == "__main__":
    main()
