#!/usr/bin/env python3
"""
Parameter sweep for the behaviour decoder (neural → eigenworm).

Sweeps one parameter at a time (all others held at defaults), on a small
set of representative worms, measuring:
  • per-mode R² (coefficient of determination)
  • per-mode Pearson correlation (ρ)
  • per-mode cosine similarity

Generates summary tables, a multi-panel parameter-sensitivity figure,
and a detailed correlation-vs-R² analysis.

Usage
-----
    # Quick run on 1 worm (6–8 min):
    python -m scripts.param_sweep_decoder \
        --worms 2023-01-17-14 --device cuda

    # Full sweep on 3 representative worms (~60 min on GPU):
    python -m scripts.param_sweep_decoder --device cuda

    # Sweep only specific parameters:
    python -m scripts.param_sweep_decoder \
        --params neural_lags hidden weight_decay --device cuda
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stage2.behavior_decoder_eval import (
    _log_ridge_grid,
    _ridge_cv_single_target,
    build_lagged_features_np,
)
from scripts.benchmark_ar_decoder_v2 import (
    load_data, r2_score, E2EOscillatorMLP,
)

# ═══════════════════════════════════════════════════════════════════
#  Default hyperparameters (current "production" values)
# ═══════════════════════════════════════════════════════════════════
DEFAULTS = dict(
    neural_lags=8,
    n_modes=6,
    n_folds=5,
    # MLP
    hidden=128,
    n_layers=2,
    mlp_lr=1e-3,
    mlp_wd=1e-3,
    mlp_dropout=0.1,
    mlp_epochs=500,
    mlp_patience=40,
    # E2E (AR2+MLP)
    e2e_epochs=200,
    e2e_lr=1e-3,
    e2e_wd=1e-3,
    tbptt_chunk=64,
    max_rho=0.98,
    w_phase=0.0,
    diagonal_ar=False,
)

# ═══════════════════════════════════════════════════════════════════
#  Parameter grids (sweep dimension → values)
# ═══════════════════════════════════════════════════════════════════
PARAM_GRIDS = {
    "neural_lags":  [2, 4, 8, 12, 16, 20],
    "n_modes":      [3, 4, 6, 8, 10],
    "hidden":       [32, 64, 128, 256, 512],
    "n_layers":     [1, 2, 3, 4],
    "mlp_wd":       [0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    "mlp_lr":       [1e-4, 3e-4, 1e-3, 3e-3],
    "mlp_dropout":  [0.0, 0.05, 0.1, 0.2, 0.3],
    "e2e_epochs":   [100, 200, 300, 400],
    "e2e_wd":       [0, 1e-4, 1e-3, 1e-2],
    "tbptt_chunk":  [32, 64, 128, 256],
    "max_rho":      [0.90, 0.95, 0.98, 0.995, 1.00],
    "w_phase":      [0.0, 0.1, 0.5, 1.0, 2.0],
    "diagonal_ar":  [False, True],
}

_ROOT = Path(__file__).resolve().parent.parent
_H5_DIR = _ROOT / "data/used/behaviour+neuronal activity atanas (2023)/2"
_OUT_DIR = _ROOT / "output_plots/behaviour_decoder/param_sweep"

# Representative worms: large / medium / small neuron counts
_DEFAULT_WORMS = ["2023-01-17-14", "2023-01-09-28", "2023-01-09-08"]


# ═══════════════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════════════

def pearson_corr(y_true, y_pred):
    """Per-signal Pearson correlation, NaN-safe."""
    yt = y_true - y_true.mean()
    yp = y_pred - y_pred.mean()
    num = (yt * yp).sum()
    den = np.sqrt((yt ** 2).sum() * (yp ** 2).sum())
    if den < 1e-12:
        return 0.0
    return float(num / den)


def compute_metrics(b_true, b_pred):
    """Return dict with per-mode and mean R², Pearson ρ, cosine sim."""
    K = b_true.shape[1]
    r2s, corrs, cossims = [], [], []
    for j in range(K):
        gt, pr = b_true[:, j], b_pred[:, j]
        valid = np.isfinite(gt) & np.isfinite(pr)
        gt, pr = gt[valid], pr[valid]
        if len(gt) < 10:
            r2s.append(np.nan); corrs.append(np.nan); cossims.append(np.nan)
            continue
        r2s.append(r2_score(gt, pr))
        corrs.append(pearson_corr(gt, pr))
        # Cosine similarity
        dot = (gt * pr).sum()
        norms = np.sqrt((gt ** 2).sum()) * np.sqrt((pr ** 2).sum())
        cossims.append(float(dot / max(norms, 1e-12)))
    return {
        "r2_per_mode":   r2s,
        "corr_per_mode": corrs,
        "cos_per_mode":  cossims,
        "r2_mean":       float(np.nanmean(r2s)),
        "corr_mean":     float(np.nanmean(corrs)),
        "cos_mean":      float(np.nanmean(cossims)),
    }


# ═══════════════════════════════════════════════════════════════════
#  Model training helpers  (same as unified_benchmark)
# ═══════════════════════════════════════════════════════════════════

def _make_mlp(d_in, K, hidden=128, n_layers=2, dropout=0.1):
    layers = []
    prev = d_in
    for _ in range(n_layers):
        layers += [nn.Linear(prev, hidden), nn.LayerNorm(hidden),
                   nn.ReLU(), nn.Dropout(dropout)]
        prev = hidden
    layers.append(nn.Linear(prev, K))
    return nn.Sequential(*layers)


def _ridge_fit(X_train, y_train, ridge_grid, n_inner):
    idx = np.arange(X_train.shape[0])
    fit = _ridge_cv_single_target(X_train, y_train, idx, ridge_grid, n_inner)
    return fit["coef"], fit["intercept"], {"alpha": fit["best_lambda"]}


def train_ridge(X_neural, b, K, n_folds, warmup, ridge_grid):
    """Ridge with per-mode α CV. Returns held-out predictions."""
    T = b.shape[0]
    n_inner = n_folds - 1
    valid_len = T - warmup
    fold_size = valid_len // n_folds
    preds = np.full((T, K), np.nan)
    alphas = []

    for fi in range(n_folds):
        s = warmup + fi * fold_size
        e = warmup + (fi + 1) * fold_size if fi < n_folds - 1 else T
        train_mask = np.ones(T, dtype=bool)
        train_mask[:warmup] = False
        train_mask[s:e] = False
        tr_idx = np.where(train_mask)[0]
        X_tr, X_te = X_neural[tr_idx], X_neural[s:e]
        b_tr = b[tr_idx]

        for j in range(K):
            coef, intc, info = _ridge_fit(X_tr, b_tr[:, j], ridge_grid, n_inner)
            preds[s:e, j] = X_te @ coef + intc
            alphas.append(info["alpha"])

    return preds, alphas


def train_mlp(X_neural, b, K, n_folds, warmup, cfg):
    """MLP with inner-CV epoch selection. Returns held-out predictions."""
    T = b.shape[0]
    valid_len = T - warmup
    fold_size = valid_len // n_folds
    preds = np.full((T, K), np.nan)

    for fi in range(n_folds):
        s = warmup + fi * fold_size
        e = warmup + (fi + 1) * fold_size if fi < n_folds - 1 else T
        train_mask = np.ones(T, dtype=bool)
        train_mask[:warmup] = False
        train_mask[s:e] = False
        tr_idx = np.where(train_mask)[0]

        X_tr = X_neural[tr_idx]
        b_tr = b[tr_idx]
        X_te = X_neural[s:e]
        d_in = X_tr.shape[1]

        X_all = torch.tensor(X_tr, dtype=torch.float32)
        y_all = torch.tensor(b_tr, dtype=torch.float32)

        # Phase 1: inner CV to find best epoch
        inner_folds = cfg.get("mlp_cv_folds", 5)
        ifold_size = len(tr_idx) // inner_folds
        val_curves = []

        for ifi in range(inner_folds):
            is_ = ifi * ifold_size
            ie_ = (is_ + ifold_size) if ifi < inner_folds - 1 else len(tr_idx)
            mask = np.ones(len(tr_idx), dtype=bool)
            mask[is_:ie_] = False
            X_t, y_t = X_all[mask], y_all[mask]
            X_v, y_v = X_all[is_:ie_], y_all[is_:ie_]

            mlp_i = _make_mlp(d_in, K, cfg["hidden"], cfg["n_layers"],
                              cfg["mlp_dropout"])
            opt_i = torch.optim.Adam(mlp_i.parameters(), lr=cfg["mlp_lr"],
                                     weight_decay=cfg["mlp_wd"])
            fold_vl = []
            best_vl, pat = float("inf"), 0
            for ep in range(cfg["mlp_epochs"]):
                mlp_i.train()
                loss = nn.functional.mse_loss(mlp_i(X_t), y_t)
                opt_i.zero_grad(); loss.backward(); opt_i.step()
                mlp_i.eval()
                with torch.no_grad():
                    vl = nn.functional.mse_loss(mlp_i(X_v), y_v).item()
                fold_vl.append(vl)
                if vl < best_vl - 1e-6:
                    best_vl, pat = vl, 0
                else:
                    pat += 1
                    if pat > cfg["mlp_patience"]:
                        break
            fold_vl.extend([fold_vl[-1]] * (cfg["mlp_epochs"] - len(fold_vl)))
            val_curves.append(fold_vl)

        mean_curve = np.mean(val_curves, axis=0)
        best_epoch = max(int(np.argmin(mean_curve)) + 1, 10)

        # Phase 2: retrain on all training data
        mlp = _make_mlp(d_in, K, cfg["hidden"], cfg["n_layers"],
                        cfg["mlp_dropout"])
        opt = torch.optim.Adam(mlp.parameters(), lr=cfg["mlp_lr"],
                               weight_decay=cfg["mlp_wd"])
        for ep in range(best_epoch):
            mlp.train()
            loss = nn.functional.mse_loss(mlp(X_all), y_all)
            opt.zero_grad(); loss.backward(); opt.step()

        mlp.eval()
        with torch.no_grad():
            preds[s:e] = mlp(torch.tensor(X_te, dtype=torch.float32)).numpy()

    return preds


def _clamp_full_spectral_radius(M1, M2, K, max_rho=0.98):
    with torch.no_grad():
        C = torch.zeros(2 * K, 2 * K)
        C[:K, :K] = M1.weight.data
        C[:K, K:] = M2.weight.data
        C[K:, :K] = torch.eye(K)
        eigs = torch.linalg.eigvals(C)
        rho = eigs.abs().max().item()
        if rho > max_rho:
            s = max_rho / (rho + 1e-8)
            M1.weight.data.mul_(s)
            M2.weight.data.mul_(s ** 2)


def _clamp_diag_spectral_radius(d1, d2, max_rho=0.98):
    with torch.no_grad():
        d1v, d2v = d1.data, d2.data
        disc = d1v ** 2 + 4.0 * d2v
        sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
        lam_r1 = (d1v + sqrt_disc) / 2.0
        lam_r2 = (d1v - sqrt_disc) / 2.0
        rho_real = torch.max(lam_r1.abs(), lam_r2.abs())
        rho_complex = torch.sqrt(torch.clamp(-d2v, min=0.0))
        rho = torch.where(disc >= 0, rho_real, rho_complex)
        scale = torch.where(rho > max_rho,
                            max_rho / (rho + 1e-8), torch.ones_like(rho))
        d1.data.mul_(scale)
        d2.data.mul_(scale ** 2)


def train_e2e(X_neural, b, K, n_folds, warmup, cfg):
    """E2E VAR(2)+MLP via BPTT. Returns held-out free-run predictions."""
    T = b.shape[0]
    valid_len = T - warmup
    fold_size = valid_len // n_folds
    preds = np.full((T, K), np.nan)

    b_t = torch.tensor(b, dtype=torch.float32)
    X_t = torch.tensor(X_neural, dtype=torch.float32)
    d_in = X_neural.shape[1]
    chunk = cfg["tbptt_chunk"]
    diagonal = cfg.get("diagonal_ar", False)

    for fi in range(n_folds):
        s = warmup + fi * fold_size
        e = warmup + (fi + 1) * fold_size if fi < n_folds - 1 else T
        train_mask = np.ones(T, dtype=bool)
        train_mask[:warmup] = False
        train_mask[s:e] = False
        tr_idx = np.where(train_mask)[0]

        # Build training segments (contiguous blocks within train data)
        segs = []
        i = 0
        while i < len(tr_idx):
            j = i + 1
            while j < len(tr_idx) and tr_idx[j] == tr_idx[j - 1] + 1:
                j += 1
            seg_start = tr_idx[i]
            seg_end = tr_idx[j - 1] + 1
            if seg_end - seg_start >= 4:
                segs.append((seg_start, seg_end))
            i = j

        # Build model
        e2e = E2EOscillatorMLP(d_in, K, hidden=cfg["hidden"],
                               n_layers=cfg["n_layers"])

        if diagonal:
            e2e._d1 = nn.Parameter(torch.full((K,), 0.8))
            e2e._d2 = nn.Parameter(torch.zeros(K))

            class _DiagMul(nn.Module):
                def __init__(self, diag):
                    super().__init__()
                    self.diag = diag
                def forward(self, x):
                    return self.diag * x

            e2e.M1 = _DiagMul(e2e._d1)
            e2e.M2 = _DiagMul(e2e._d2)
        else:
            with torch.no_grad():
                e2e.M1.weight.copy_(0.8 * torch.eye(K))
                e2e.M2.weight.zero_()

        opt = torch.optim.Adam(e2e.parameters(), lr=cfg["e2e_lr"],
                               weight_decay=cfg["e2e_wd"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg["e2e_epochs"])
        best_l, best_state, pat = float("inf"), None, 0

        for ep in range(cfg["e2e_epochs"]):
            e2e.train()
            ep_loss, ep_n = 0.0, 0
            for s0, s1 in segs:
                drv = e2e.mlp(X_t)
                p1e = b_t[s0 - 1].detach()
                p2e = b_t[s0 - 2].detach()
                for cs in range(s0, s1, chunk):
                    ce = min(cs + chunk, s1)
                    cl = torch.tensor(0.0)
                    for tt in range(cs, ce):
                        if diagonal:
                            pt = e2e.M1(p1e) + e2e.M2(p2e) + drv[tt] + e2e.c
                        else:
                            pt = (e2e.M1(p1e.unsqueeze(0)).squeeze(0) +
                                  e2e.M2(p2e.unsqueeze(0)).squeeze(0) +
                                  drv[tt] + e2e.c)
                        cl = cl + nn.functional.mse_loss(pt, b_t[tt])
                        if cfg["w_phase"] > 0 and K >= 2:
                            cos = nn.functional.cosine_similarity(
                                pt[:2].unsqueeze(0),
                                b_t[tt, :2].unsqueeze(0))
                            cl = cl + cfg["w_phase"] * (1.0 - cos.squeeze())
                        p2e, p1e = p1e, pt
                    cl = cl / (ce - cs)
                    opt.zero_grad(); cl.backward()
                    nn.utils.clip_grad_norm_(e2e.parameters(), 1.0)
                    opt.step()
                    if diagonal:
                        _clamp_diag_spectral_radius(
                            e2e._d1, e2e._d2, cfg["max_rho"])
                    else:
                        _clamp_full_spectral_radius(
                            e2e.M1, e2e.M2, K, cfg["max_rho"])
                    ep_loss += cl.item() * (ce - cs)
                    ep_n += ce - cs
                    p1e = p1e.detach(); p2e = p2e.detach()
                    drv = e2e.mlp(X_t)
            sched.step()
            if ep_n > 0:
                avg = ep_loss / ep_n
                if avg < best_l - 1e-6:
                    best_l, pat = avg, 0
                    best_state = {k: v.clone() for k, v
                                  in e2e.state_dict().items()}
                else:
                    pat += 1
                    if pat > 40:
                        break

        # Restore best & free-run evaluate on test fold
        if best_state:
            e2e.load_state_dict(best_state)
        e2e.eval()

        with torch.no_grad():
            drv = e2e.mlp(X_t)
            p1 = b_t[s - 1]
            p2 = b_t[s - 2]
            for tt in range(s, e):
                if diagonal:
                    pt = e2e.M1(p1) + e2e.M2(p2) + drv[tt] + e2e.c
                else:
                    pt = (e2e.M1(p1.unsqueeze(0)).squeeze(0) +
                          e2e.M2(p2.unsqueeze(0)).squeeze(0) +
                          drv[tt] + e2e.c)
                preds[tt] = pt.numpy()
                p2, p1 = p1, pt

    return preds


# ═══════════════════════════════════════════════════════════════════
#  Run one configuration  (Ridge + MLP + AR2+MLP)
# ═══════════════════════════════════════════════════════════════════

def run_one_config(h5_path, cfg, seed=42, skip_e2e=False):
    """Run Ridge, MLP, and optionally AR2+MLP with given config."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    u, b_full, dt = load_data(str(h5_path), all_neurons=False)
    K = min(cfg["n_modes"], b_full.shape[1])
    b = b_full[:, :K]
    T = b.shape[0]
    n_lags = cfg["neural_lags"]
    warmup = max(2, n_lags)
    X_neural = build_lagged_features_np(u, n_lags)

    ridge_grid = _log_ridge_grid(-3.0, 10.0, 50)

    results = {}

    # ── Ridge ──────────────────────────────────────────────────────
    t0 = time.time()
    preds_ridge, alphas = train_ridge(
        X_neural, b, K, cfg["n_folds"], warmup, ridge_grid)
    results["Ridge"] = compute_metrics(b, preds_ridge)
    results["Ridge"]["time_s"] = time.time() - t0
    results["Ridge"]["alphas"] = [float(a) for a in alphas[:K]]

    # ── MLP ────────────────────────────────────────────────────────
    t0 = time.time()
    preds_mlp = train_mlp(X_neural, b, K, cfg["n_folds"], warmup, cfg)
    results["MLP"] = compute_metrics(b, preds_mlp)
    results["MLP"]["time_s"] = time.time() - t0

    # ── AR2+MLP (E2E BPTT) ────────────────────────────────────────
    if not skip_e2e:
        t0 = time.time()
        preds_e2e = train_e2e(X_neural, b, K, cfg["n_folds"], warmup, cfg)
        results["AR2+MLP"] = compute_metrics(b, preds_e2e)
        results["AR2+MLP"]["time_s"] = time.time() - t0

    return results


# ═══════════════════════════════════════════════════════════════════
#  Sweep runner
# ═══════════════════════════════════════════════════════════════════

def sweep_one_param(param_name, grid, h5_paths, out_dir, seed=42,
                    skip_e2e=False):
    """Sweep one parameter, returns list of {value, worm, model, metrics}."""
    rows = []
    for val in grid:
        cfg = dict(DEFAULTS)
        cfg[param_name] = val
        label = f"{param_name}={val}"
        print(f"\n  ── {label} ──")

        for h5 in h5_paths:
            worm = h5.stem
            t0 = time.time()
            try:
                res = run_one_config(h5, cfg, seed=seed,
                                     skip_e2e=skip_e2e)
            except Exception as ex:
                print(f"    [{worm}] FAILED: {ex}")
                continue
            elapsed = time.time() - t0
            for model, metrics in res.items():
                row = {
                    "param": param_name,
                    "value": val if not isinstance(val, bool)
                             else int(val),
                    "worm": worm,
                    "model": model,
                    **metrics,
                }
                rows.append(row)
            print(f"    [{worm}] Ridge R²={res['Ridge']['r2_mean']:.3f} "
                  f"ρ={res['Ridge']['corr_mean']:.3f}  |  "
                  f"MLP R²={res['MLP']['r2_mean']:.3f} "
                  f"ρ={res['MLP']['corr_mean']:.3f}  |  "
                  f"AR2 R²={res['AR2+MLP']['r2_mean']:.3f} "
                  f"ρ={res['AR2+MLP']['corr_mean']:.3f}  "
                  f"({elapsed:.0f}s)")
    return rows


# ═══════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════

def plot_param_sensitivity(all_rows, out_dir, params_swept):
    """Multi-panel: for each swept parameter, R² and Corr vs value."""
    n_params = len(params_swept)
    if n_params == 0:
        return

    fig, axes = plt.subplots(n_params, 2, figsize=(14, 3.5 * n_params),
                             squeeze=False)
    fig.suptitle("Parameter Sensitivity: R² and Pearson ρ", fontsize=14, y=1.01)

    model_colors = {"Ridge": "#3498db", "MLP": "#e74c3c", "AR2+MLP": "#2ecc71"}

    for pi, param in enumerate(params_swept):
        rows = [r for r in all_rows if r["param"] == param]
        if not rows:
            continue

        # Group by (model, value) → mean across worms
        for metric_idx, (metric_key, metric_label) in enumerate([
            ("r2_mean", "R²"), ("corr_mean", "Pearson ρ")
        ]):
            ax = axes[pi, metric_idx]
            for model, color in model_colors.items():
                model_rows = [r for r in rows if r["model"] == model]
                if not model_rows:
                    continue
                vals = sorted(set(r["value"] for r in model_rows))
                means, stds = [], []
                for v in vals:
                    ms = [r[metric_key] for r in model_rows if r["value"] == v]
                    means.append(np.mean(ms))
                    stds.append(np.std(ms) if len(ms) > 1 else 0)
                xs = range(len(vals))
                ax.errorbar(xs, means, yerr=stds, marker="o", label=model,
                            color=color, capsize=3, lw=1.5, markersize=5)
                # Mark default value
                default_val = DEFAULTS.get(param)
                if default_val in vals:
                    di = vals.index(default_val)
                    ax.axvline(di, color="gray", ls="--", alpha=0.4, lw=1)

            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels([str(v) for v in vals], fontsize=8)
            ax.set_xlabel(param)
            ax.set_ylabel(metric_label)
            ax.set_title(f"{metric_label} vs {param}")
            if metric_idx == 0:
                ax.legend(fontsize=8, loc="best")
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "param_sensitivity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved {out_dir / 'param_sensitivity.png'}")


def plot_r2_vs_correlation(all_rows, out_dir):
    """Scatter: R² vs Pearson ρ, colored by model, per-mode and mean."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    model_colors = {"Ridge": "#3498db", "MLP": "#e74c3c", "AR2+MLP": "#2ecc71"}
    model_markers = {"Ridge": "o", "MLP": "s", "AR2+MLP": "^"}

    # Panel 1: Per-mode (all modes from all configs)
    ax = axes[0]
    for model in model_colors:
        r2s_all, corrs_all = [], []
        for r in all_rows:
            if r["model"] != model:
                continue
            r2s_all.extend(r["r2_per_mode"])
            corrs_all.extend(r["corr_per_mode"])
        r2s_all = np.array(r2s_all)
        corrs_all = np.array(corrs_all)
        valid = np.isfinite(r2s_all) & np.isfinite(corrs_all)
        ax.scatter(corrs_all[valid], r2s_all[valid], s=12, alpha=0.3,
                   c=model_colors[model], marker=model_markers[model],
                   label=model)
    # Plot ρ² curve
    rho_range = np.linspace(0, 1, 100)
    ax.plot(rho_range, rho_range ** 2, "k--", lw=1, alpha=0.5,
            label="R² = ρ²")
    ax.set_xlabel("Pearson ρ")
    ax.set_ylabel("R²")
    ax.set_title("Per-mode: R² vs Pearson ρ (all configs)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.1, 1.05)
    ax.set_ylim(-0.5, 1.05)

    # Panel 2: Mean (one dot per worm×config)
    ax = axes[1]
    for model in model_colors:
        r2s = [r["r2_mean"] for r in all_rows if r["model"] == model]
        corrs = [r["corr_mean"] for r in all_rows if r["model"] == model]
        ax.scatter(corrs, r2s, s=30, alpha=0.5, c=model_colors[model],
                   marker=model_markers[model], label=model, edgecolors="k",
                   lw=0.3)
    ax.plot(rho_range, rho_range ** 2, "k--", lw=1, alpha=0.5,
            label="R² = ρ²")
    ax.set_xlabel("Mean Pearson ρ")
    ax.set_ylabel("Mean R²")
    ax.set_title("Per-worm mean: R² vs Pearson ρ")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "r2_vs_correlation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / 'r2_vs_correlation.png'}")


def plot_per_mode_breakdown(all_rows, out_dir):
    """Bar chart: R² and Corr per eigenworm mode, default config only."""
    # Filter to default config
    default_rows = []
    for r in all_rows:
        # Check if this row matches all defaults
        if r["value"] == DEFAULTS.get(r["param"]):
            default_rows.append(r)

    if not default_rows:
        # Just use the first param's default value
        first_param = all_rows[0]["param"] if all_rows else None
        if first_param:
            dv = DEFAULTS.get(first_param)
            default_rows = [r for r in all_rows
                            if r["param"] == first_param and r["value"] == dv]

    if not default_rows:
        return

    models = ["Ridge", "MLP", "AR2+MLP"]
    model_colors = {"Ridge": "#3498db", "MLP": "#e74c3c", "AR2+MLP": "#2ecc71"}

    # Average across worms
    K = len(default_rows[0].get("r2_per_mode", []))
    if K == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    width = 0.25

    for metric_idx, (key, label) in enumerate([
        ("r2_per_mode", "R²"), ("corr_per_mode", "Pearson ρ")
    ]):
        ax = axes[metric_idx]
        for mi, model in enumerate(models):
            m_rows = [r for r in default_rows if r["model"] == model]
            if not m_rows:
                continue
            vals = np.array([r[key] for r in m_rows])
            means = np.nanmean(vals, axis=0)
            stds = np.nanstd(vals, axis=0) if vals.shape[0] > 1 \
                else np.zeros(K)
            xs = np.arange(K) + mi * width
            ax.bar(xs, means, width, yerr=stds, label=model,
                   color=model_colors[model], edgecolor="k", lw=0.4,
                   capsize=2, alpha=0.85)
            for j in range(K):
                ax.text(xs[j], means[j] + stds[j] + 0.01,
                        f"{means[j]:.2f}", ha="center", fontsize=6)
        ax.set_xticks(np.arange(K) + width)
        ax.set_xticklabels([f"a{j+1}" for j in range(K)])
        ax.set_ylabel(label)
        ax.set_title(f"{label} per eigenworm mode (default params)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(0, color="k", lw=0.5)

    fig.tight_layout()
    fig.savefig(out_dir / "per_mode_breakdown.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / 'per_mode_breakdown.png'}")


def plot_corr_analysis(all_rows, out_dir):
    """Detailed correlation analysis:
    - ρ vs R² relationship (is R² ≈ ρ² or is there bias?)
    - Per-mode ρ distribution
    - ρ difference between models
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    model_colors = {"Ridge": "#3498db", "MLP": "#e74c3c", "AR2+MLP": "#2ecc71"}

    # (0,0) Scatter: per-mode ρ² vs actual R²  → shows how much R² < ρ²
    #       If R² = ρ² the model is well-calibrated (zero-mean, correct scale).
    #       R² < ρ² means predictions are biased or mis-scaled.
    ax = axes[0, 0]
    for model, color in model_colors.items():
        r2_all, rho2_all = [], []
        for r in all_rows:
            if r["model"] != model:
                continue
            for j in range(len(r["r2_per_mode"])):
                r2_all.append(r["r2_per_mode"][j])
                rho2_all.append(r["corr_per_mode"][j] ** 2)
        r2_all, rho2_all = np.array(r2_all), np.array(rho2_all)
        valid = np.isfinite(r2_all) & np.isfinite(rho2_all)
        ax.scatter(rho2_all[valid], r2_all[valid], s=10, alpha=0.3,
                   c=color, label=model)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="R² = ρ²")
    ax.set_xlabel("ρ² (squared Pearson correlation)")
    ax.set_ylabel("R² (coefficient of determination)")
    ax.set_title("Calibration gap: R² vs ρ²\n(below diagonal = biased/mis-scaled)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,1) Per-mode correlation histograms
    ax = axes[0, 1]
    default_rows = [r for r in all_rows
                    if r["value"] == DEFAULTS.get(r["param"])]
    if not default_rows:
        p = all_rows[0]["param"]
        default_rows = [r for r in all_rows
                        if r["param"] == p and r["value"] == DEFAULTS.get(p)]
    K = len(default_rows[0]["corr_per_mode"]) if default_rows else 0
    if K > 0:
        for j in range(K):
            corrs_j = [r["corr_per_mode"][j] for r in default_rows
                       if r["model"] == "Ridge"]
            if corrs_j:
                ax.hist(corrs_j, bins=15, alpha=0.5, label=f"a{j+1}",
                        density=True)
        ax.set_xlabel("Pearson ρ")
        ax.set_ylabel("Density")
        ax.set_title("Per-mode ρ distribution (Ridge, default params)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # (1,0) Model comparison: ΔCorr (MLP − Ridge, AR2 − Ridge)
    ax = axes[1, 0]
    default_only = [r for r in all_rows
                    if r["value"] == DEFAULTS.get(r["param"])]
    if not default_only:
        p = all_rows[0]["param"]
        default_only = [r for r in all_rows
                        if r["param"] == p and r["value"] == DEFAULTS.get(p)]
    # Group by (param, value, worm) to compare models
    grouped = defaultdict(dict)
    for r in default_only:
        key = (r["param"], r["value"], r["worm"])
        grouped[key][r["model"]] = r
    delta_mlp, delta_ar2 = [], []
    for key, models in grouped.items():
        if "Ridge" in models and "MLP" in models:
            delta_mlp.append(
                models["MLP"]["corr_mean"] - models["Ridge"]["corr_mean"])
        if "Ridge" in models and "AR2+MLP" in models:
            delta_ar2.append(
                models["AR2+MLP"]["corr_mean"] - models["Ridge"]["corr_mean"])
    if delta_mlp or delta_ar2:
        data_to_plot = []
        labels_to_plot = []
        if delta_mlp:
            data_to_plot.append(delta_mlp)
            labels_to_plot.append(f"MLP−Ridge\n(μ={np.mean(delta_mlp):+.3f})")
        if delta_ar2:
            data_to_plot.append(delta_ar2)
            labels_to_plot.append(
                f"AR2−Ridge\n(μ={np.mean(delta_ar2):+.3f})")
        parts = ax.violinplot(data_to_plot, showmeans=True, showmedians=True)
        ax.set_xticks(range(1, len(labels_to_plot) + 1))
        ax.set_xticklabels(labels_to_plot)
        ax.axhline(0, color="k", ls="--", lw=0.8)
        ax.set_ylabel("Δ Pearson ρ (mean)")
        ax.set_title("Correlation improvement over Ridge")
        ax.grid(True, alpha=0.3, axis="y")

    # (1,1) Summary table
    ax = axes[1, 1]
    ax.axis("off")
    # Build summary table of best config per model
    best_per_model = {}
    for model in ["Ridge", "MLP", "AR2+MLP"]:
        model_rows = [r for r in all_rows if r["model"] == model]
        if not model_rows:
            continue
        # Group by (param, value), average across worms
        cfg_means = defaultdict(list)
        for r in model_rows:
            cfg_means[(r["param"], r["value"])].append(r["r2_mean"])
        best_cfg = max(cfg_means.items(), key=lambda x: np.mean(x[1]))
        best_per_model[model] = {
            "param": best_cfg[0][0],
            "value": best_cfg[0][1],
            "r2_mean": np.mean(best_cfg[1]),
        }
        # Also get corr for that config
        corr_vals = [r["corr_mean"] for r in model_rows
                     if r["param"] == best_cfg[0][0]
                     and r["value"] == best_cfg[0][1]]
        best_per_model[model]["corr_mean"] = np.mean(corr_vals) if corr_vals \
            else np.nan

    text_lines = ["BEST CONFIGURATION PER MODEL", "=" * 40, ""]
    for model, info in best_per_model.items():
        text_lines.append(f"{model}:")
        text_lines.append(f"  Best param: {info['param']}={info['value']}")
        text_lines.append(f"  R² = {info['r2_mean']:.3f}")
        text_lines.append(f"  ρ  = {info['corr_mean']:.3f}")
        text_lines.append("")

    # Add relationship note
    text_lines.extend([
        "─" * 40,
        "KEY RELATIONSHIPS:",
        "• R² = 1 − Var(residual)/Var(signal)",
        "  Sensitive to bias & scale errors",
        "• ρ = Cov(ŷ,y) / (σ_ŷ · σ_y)",
        "  Only measures linear association",
        "• If ρ is high but R² is low →",
        "  predictions have correct shape",
        "  but wrong mean/scale",
        "• Always: R² ≤ ρ² (equality iff",
        "  predictions are unbiased linear",
        "  transform of ground truth)",
    ])
    ax.text(0.05, 0.95, "\n".join(text_lines), transform=ax.transAxes,
            fontsize=9, va="top", family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))

    fig.tight_layout()
    fig.savefig(out_dir / "correlation_analysis.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / 'correlation_analysis.png'}")


def print_summary_table(all_rows, params_swept):
    """Print a text summary table: for each param×value, mean R² and ρ."""
    print("\n" + "═" * 100)
    print("  PARAMETER SWEEP SUMMARY")
    print("═" * 100)

    for param in params_swept:
        rows = [r for r in all_rows if r["param"] == param]
        if not rows:
            continue
        vals = sorted(set(r["value"] for r in rows))
        default = DEFAULTS.get(param)

        print(f"\n  ── {param} (default={default}) ──")
        print(f"  {'Value':>10s}  {'Ridge R²':>9s} {'Ridge ρ':>8s}  "
              f"{'MLP R²':>8s} {'MLP ρ':>7s}  "
              f"{'AR2 R²':>8s} {'AR2 ρ':>7s}")
        print(f"  {'─' * 10}  {'─' * 9} {'─' * 8}  "
              f"{'─' * 8} {'─' * 7}  {'─' * 8} {'─' * 7}")

        for v in vals:
            parts = [f"  {str(v):>10s}"]
            for model in ["Ridge", "MLP", "AR2+MLP"]:
                m_rows = [r for r in rows
                          if r["value"] == v and r["model"] == model]
                if m_rows:
                    r2 = np.mean([r["r2_mean"] for r in m_rows])
                    corr = np.mean([r["corr_mean"] for r in m_rows])
                    marker = " ★" if v == default else "  "
                    parts.append(f"{r2:8.3f}{marker}{corr:7.3f}")
                else:
                    parts.append(f"{'--':>9s} {'--':>8s}")
            print("  ".join(parts))

    # Best overall config per model
    print(f"\n{'═' * 100}")
    print("  BEST CONFIG PER MODEL (highest mean R² across worms)")
    print(f"{'═' * 100}")
    for model in ["Ridge", "MLP", "AR2+MLP"]:
        model_rows = [r for r in all_rows if r["model"] == model]
        if not model_rows:
            continue
        cfg_scores = defaultdict(lambda: {"r2": [], "corr": []})
        for r in model_rows:
            key = (r["param"], r["value"])
            cfg_scores[key]["r2"].append(r["r2_mean"])
            cfg_scores[key]["corr"].append(r["corr_mean"])
        best = max(cfg_scores.items(),
                   key=lambda x: np.mean(x[1]["r2"]))
        r2 = np.mean(best[1]["r2"])
        corr = np.mean(best[1]["corr"])
        print(f"  {model:10s}: {best[0][0]}={best[0][1]:>6}  "
              f"→  R²={r2:.3f}  ρ={corr:.3f}")


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Parameter sweep for behaviour decoder",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worms", nargs="+", default=_DEFAULT_WORMS,
                    help="Worm IDs to test on")
    ap.add_argument("--h5_dir", type=Path, default=_H5_DIR)
    ap.add_argument("--out_dir", type=Path, default=_OUT_DIR)
    ap.add_argument("--params", nargs="+", default=None,
                    help="Parameters to sweep (default: all). "
                    f"Choices: {list(PARAM_GRIDS.keys())}")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu",
                    help="(currently unused, all models run on CPU)")
    ap.add_argument("--quick", action="store_true",
                    help="Quick mode: skip AR2+MLP, fewer grid points")
    ap.add_argument("--skip_e2e", action="store_true",
                    help="Skip AR2+MLP (E2E BPTT) — much faster")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve H5 paths
    h5_paths = []
    for worm in args.worms:
        h5 = args.h5_dir / f"{worm}.h5"
        if h5.exists():
            h5_paths.append(h5)
        else:
            print(f"  [warn] {h5} not found, skipping")
    if not h5_paths:
        print("No worms found!")
        return

    # Select parameters to sweep
    params_to_sweep = args.params or list(PARAM_GRIDS.keys())
    # Filter to valid params
    params_to_sweep = [p for p in params_to_sweep if p in PARAM_GRIDS]

    if args.quick:
        # Reduce grid sizes for quick testing
        quick_grids = {
            "neural_lags": [4, 8, 16],
            "hidden": [64, 128, 256],
            "n_layers": [1, 2, 3],
            "mlp_wd": [1e-4, 1e-3, 1e-2],
            "mlp_lr": [3e-4, 1e-3, 3e-3],
            "mlp_dropout": [0.0, 0.1, 0.2],
        }
        # In quick mode, skip E2E-only params
        params_to_sweep = [p for p in params_to_sweep
                           if p not in ("e2e_epochs", "e2e_wd", "tbptt_chunk",
                                        "max_rho", "w_phase", "diagonal_ar")]
        for p in params_to_sweep:
            if p in quick_grids:
                PARAM_GRIDS[p] = quick_grids[p]

    n_total = sum(len(PARAM_GRIDS[p]) for p in params_to_sweep)
    n_worms = len(h5_paths)
    print(f"\n{'═' * 70}")
    print(f"  PARAMETER SWEEP")
    print(f"  {len(params_to_sweep)} parameters, {n_total} configs × "
          f"{n_worms} worms = {n_total * n_worms} runs")
    print(f"  Worms: {[h.stem for h in h5_paths]}")
    print(f"  Output: {args.out_dir}")
    print(f"{'═' * 70}")

    all_rows = []
    t_start = time.time()

    for pi, param in enumerate(params_to_sweep):
        grid = PARAM_GRIDS[param]
        print(f"\n{'━' * 60}")
        print(f"  [{pi+1}/{len(params_to_sweep)}] Sweeping {param}: {grid}")
        print(f"{'━' * 60}")
        skip = args.skip_e2e or (args.quick and param not in
                                    ('e2e_epochs', 'e2e_wd', 'tbptt_chunk',
                                     'max_rho', 'w_phase', 'diagonal_ar'))
        rows = sweep_one_param(param, grid, h5_paths, args.out_dir,
                               seed=args.seed, skip_e2e=skip)
        all_rows.extend(rows)

        # Save incrementally
        with open(args.out_dir / "sweep_results.json", "w") as f:
            json.dump(all_rows, f, indent=2, default=str)

    total_time = time.time() - t_start
    print(f"\n  Total sweep time: {total_time:.0f}s "
          f"({total_time / 60:.1f} min)")

    # ── Summary ────────────────────────────────────────────────────
    print_summary_table(all_rows, params_to_sweep)

    # ── Plots ──────────────────────────────────────────────────────
    print("\n  Generating plots...")
    plot_param_sensitivity(all_rows, args.out_dir, params_to_sweep)
    plot_r2_vs_correlation(all_rows, args.out_dir)
    plot_per_mode_breakdown(all_rows, args.out_dir)
    plot_corr_analysis(all_rows, args.out_dir)

    print(f"\n  All results saved to {args.out_dir}/")
    print(f"  Done!")


if __name__ == "__main__":
    main()
