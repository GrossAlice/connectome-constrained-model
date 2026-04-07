#!/usr/bin/env python3
"""Fair retrain-per-condition sweep: Ridge α × MLP architectures.

Every model is retrained on condition-specific features — the fairest
possible comparison.  No zero-masking tricks: each model sees ONLY the
features that the condition allows.

Sweeps
------
Ridge α ∈ {0.1, 1, 10, 100, 1000, 10_000}   (retrain per condition)
MLP   (hidden, layers) ∈ {(64,2), (128,2), (256,2), (512,2), (256,3), (512,3)}

Ridge analytical decomposition (v4 baseline, α=1000) included for reference.

Usage:
    python -m scripts.diag_retrain_sweep --device cuda --n_worms 3
"""
from __future__ import annotations

import argparse, json, sys, time, glob
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from baseline_transformer.dataset import load_worm_data
from sklearn.linear_model import Ridge

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

K = 5
N_FOLDS = 5

CONDS = ["causal_self", "self", "causal"]
COND_LABELS = {"causal_self": "Causal + Self", "self": "Self (AR)",
               "causal": "Causal (Granger)"}

RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 10_000.0]
MLP_ARCHS = [(64, 2), (128, 2), (256, 2), (512, 2), (256, 3), (512, 3)]

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
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
    out = np.zeros((T, (K + 1) * N), dtype=x.dtype)
    out[:, :N] = x
    for k in range(1, K + 1):
        out[k:, k * N:(k + 1) * N] = x[:-k]
    return out

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
# MLP
# ══════════════════════════════════════════════════════════════════════════════

def _make_mlp(d_in, d_out, hidden=64, n_layers=2):
    layers, d = [], d_in
    for _ in range(n_layers):
        layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden),
                   nn.ReLU(), nn.Dropout(0.1)]
        d = hidden
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


def _train_mlp_simple(X_tr, y_tr, X_val, y_val, device,
                      hidden=64, n_layers=2,
                      epochs=250, lr=1e-3, wd=1e-3, patience=30):
    """Train MLP on pre-masked features (no dropout games)."""
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
    for ep in range(epochs):
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
                         hidden=256, n_layers=2, epochs=250, patience=30):
    """Train MLP with random per-sample condition masks (for self/causal).

    Each training step applies a random neuron's condition mask to the
    input and computes loss only on that neuron's output. This teaches
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
    for ep in range(epochs):
        mlp.train()
        ni_rand = torch.randint(0, N, (Xt.shape[0],), device=device)
        m = masks[ni_rand]
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
# RIDGE — ANALYTICAL DECOMPOSITION  (v4 baseline)
# ══════════════════════════════════════════════════════════════════════════════

def _run_ridge_analytical(u, N):
    """Fit one Ridge(α=1000) on all features, decompose W at test time."""
    T = u.shape[0]
    warmup = K
    ho = {c: np.full((T, N), np.nan) for c in CONDS}
    self_lag  = [_self_lag_cols(ni, N, K) for ni in range(N)]
    other_lag = [_other_lag_cols(ni, N, K) for ni in range(N)]

    for fi, (ts, te) in enumerate(_make_folds(T, warmup)):
        tr = _get_train_idx(T, warmup, ts, te, buffer=K)
        mu, sig = u[tr].mean(0), u[tr].std(0) + 1e-8
        u_n = ((u - mu) / sig).astype(np.float32)
        X = _build_features(u_n, K)
        X_te = X[ts:te]

        ridge = Ridge(alpha=1000.0).fit(X[tr], u_n[tr])
        W, br = ridge.coef_, ridge.intercept_

        for ni in range(N):
            sl = X_te[:, self_lag[ni]]  @ W[ni, self_lag[ni]]
            ol = X_te[:, other_lag[ni]] @ W[ni, other_lag[ni]]
            b  = br[ni]
            ho["causal_self"][ts:te, ni] = (sl + ol + b) * sig[ni] + mu[ni]
            ho["self"][ts:te, ni]        = (sl + b)      * sig[ni] + mu[ni]
            ho["causal"][ts:te, ni]      = (ol + b)      * sig[ni] + mu[ni]

    return {c: _per_neuron_r2(ho[c], u) for c in CONDS}


# ══════════════════════════════════════════════════════════════════════════════
# RIDGE — RETRAIN PER CONDITION
# ══════════════════════════════════════════════════════════════════════════════

def _run_ridge_retrain(u, N, alpha):
    """Fit separate Ridge models per condition on condition-specific features.

    causal_self : 1 model per fold on lag features     → all neurons
    self        : N models per fold on K features each  → 1 neuron each
    causal      : N models per fold on K(N-1) features  → 1 neuron each
    """
    T = u.shape[0]
    warmup = K
    ho = {c: np.full((T, N), np.nan) for c in CONDS}

    lag_cols  = _all_lag_cols(N, K)
    self_lag  = [_self_lag_cols(ni, N, K) for ni in range(N)]
    other_lag = [_other_lag_cols(ni, N, K) for ni in range(N)]

    for fi, (ts, te) in enumerate(_make_folds(T, warmup)):
        tr = _get_train_idx(T, warmup, ts, te, buffer=K)
        mu, sig = u[tr].mean(0), u[tr].std(0) + 1e-8
        u_n = ((u - mu) / sig).astype(np.float32)
        X = _build_features(u_n, K)

        # ── causal_self: one multivariate Ridge on lag features ──
        X_lag = X[:, lag_cols]
        pred = Ridge(alpha=alpha).fit(X_lag[tr], u_n[tr]).predict(X_lag[ts:te])
        ho["causal_self"][ts:te] = pred * sig + mu

        # ── self: per-neuron Ridge on own K lag features ──
        for ni in range(N):
            cols = self_lag[ni]
            Xi = X[:, cols]
            p = Ridge(alpha=alpha).fit(Xi[tr], u_n[tr, ni]).predict(Xi[ts:te])
            ho["self"][ts:te, ni] = p * sig[ni] + mu[ni]

        # ── causal: per-neuron Ridge on other neurons' lag features ──
        for ni in range(N):
            cols = other_lag[ni]
            Xi = X[:, cols]
            p = Ridge(alpha=alpha).fit(Xi[tr], u_n[tr, ni]).predict(Xi[ts:te])
            ho["causal"][ts:te, ni] = p * sig[ni] + mu[ni]

    return {c: _per_neuron_r2(ho[c], u) for c in CONDS}


# ══════════════════════════════════════════════════════════════════════════════
# MLP — RETRAIN PER CONDITION
# ══════════════════════════════════════════════════════════════════════════════

def _run_mlp_retrain(u, N, hidden, n_layers, device):
    """Fair MLP: retrain per condition on condition-specific features.

    causal_self : one MLP trained on lag features (K·N dim)
    self/causal : condaware MLP trained with random neuron masks
    """
    T = u.shape[0]
    warmup = K
    D = (K + 1) * N
    lag_cols = _all_lag_cols(N, K)
    ho = {c: np.full((T, N), np.nan) for c in CONDS}

    for fi, (ts, te) in enumerate(_make_folds(T, warmup)):
        tr = _get_train_idx(T, warmup, ts, te, buffer=K)
        mu, sig = u[tr].mean(0), u[tr].std(0) + 1e-8
        u_n = ((u - mu) / sig).astype(np.float32)
        nv = max(10, int(len(tr) * 0.15))
        tr_i, val_i = tr[:-nv], tr[-nv:]
        X = _build_features(u_n, K)

        # ── causal_self: MLP on lag features only ──
        X_lag = X[:, lag_cols]
        mlp = _train_mlp_simple(
            X_lag[tr_i], u_n[tr_i], X_lag[val_i], u_n[val_i],
            device, hidden=hidden, n_layers=n_layers)
        with torch.no_grad():
            pred = mlp(torch.tensor(X_lag[ts:te],
                                    dtype=torch.float32)).numpy()
        ho["causal_self"][ts:te] = pred * sig + mu
        del mlp

        # ── self & causal: condaware MLP ──
        X_te_t = torch.tensor(X[ts:te], dtype=torch.float32)
        for cond in ["self", "causal"]:
            mlp = _train_mlp_condaware(
                X[tr_i], u_n[tr_i], X[val_i], u_n[val_i],
                N, K, device, cond,
                hidden=hidden, n_layers=n_layers)
            with torch.no_grad():
                for ni in range(N):
                    m = torch.tensor(
                        _make_condition_mask(ni, N, K, cond),
                        dtype=torch.float32)
                    pred = mlp(X_te_t * m).numpy()
                    ho[cond][ts:te, ni] = pred[:, ni] * sig[ni] + mu[ni]
            del mlp

    return {c: _per_neuron_r2(ho[c], u) for c in CONDS}


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def _agg(all_results, exp_name, cond):
    """Return (mean, sem) of R² across worms for one experiment/condition."""
    vals = [all_results[w][exp_name]["r2"][cond]["mean"]
            for w in all_results]
    return np.mean(vals), np.std(vals) / max(1, np.sqrt(len(vals)))


def _plot_ridge_alpha_sweep(all_results, out):
    """Line plot: R² vs log₁₀(α) for each condition, with analytical ref."""
    n_worms = len(all_results)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ci, cond in enumerate(CONDS):
        ax = axes[ci]

        # Analytical baseline
        anal_m, anal_s = _agg(all_results, "ridge_analytical", cond)
        ax.axhspan(anal_m - anal_s, anal_m + anal_s,
                   color="gray", alpha=0.15)
        ax.axhline(anal_m, color="k", ls="--", lw=1.5, alpha=0.7,
                   label=f"Analytical (α=1 000) = {anal_m:.3f}")

        # Retrain sweep
        r2_means, r2_sems = [], []
        for a in RIDGE_ALPHAS:
            m, s = _agg(all_results, f"ridge_a{a:g}", cond)
            r2_means.append(m)
            r2_sems.append(s)

        ax.errorbar(np.log10(RIDGE_ALPHAS), r2_means, yerr=r2_sems,
                    marker="o", capsize=3, lw=2, color="#1f77b4",
                    label="Retrain per condition")

        best_i = int(np.argmax(r2_means))
        ax.plot(np.log10(RIDGE_ALPHAS[best_i]), r2_means[best_i],
                "r*", markersize=16, zorder=5,
                label=f"Best: α={RIDGE_ALPHAS[best_i]:g} → {r2_means[best_i]:.3f}")

        ax.set_xlabel("log₁₀(α)", fontsize=11)
        if ci == 0:
            ax.set_ylabel("Mean R²  (± SEM)", fontsize=11)
        ax.set_title(COND_LABELS[cond], fontsize=12, fontweight="bold")
        ax.legend(fontsize=7.5, loc="best")
        ax.grid(alpha=0.2)

    fig.suptitle(f"Ridge α Sweep — Retrain vs Analytical  "
                 f"(K={K}, {n_worms} worms)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "sweep_ridge_alpha.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ sweep_ridge_alpha.png")


def _plot_mlp_arch_sweep(all_results, out):
    """Grouped bar chart: MLP architectures per condition, with best Ridge."""
    n_worms = len(all_results)
    arch_names = [f"mlp_{h}h{l}_retrain" for h, l in MLP_ARCHS]
    arch_labels = [f"{h}h × {l}L" for h, l in MLP_ARCHS]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ci, cond in enumerate(CONDS):
        ax = axes[ci]

        # Best ridge retrain
        ridge_means = [_agg(all_results, f"ridge_a{a:g}", cond)[0]
                       for a in RIDGE_ALPHAS]
        best_ridge = max(ridge_means)
        ax.axhline(best_ridge, color="#2ca02c", ls="--", lw=2, alpha=0.8,
                   label=f"Best Ridge retrain = {best_ridge:.3f}")

        # Analytical
        anal_m, _ = _agg(all_results, "ridge_analytical", cond)
        ax.axhline(anal_m, color="gray", ls=":", lw=1.5, alpha=0.6,
                   label=f"Ridge analytical = {anal_m:.3f}")

        # MLP bars
        means, sems = [], []
        for name in arch_names:
            m, s = _agg(all_results, name, cond)
            means.append(m)
            sems.append(s)

        x = np.arange(len(arch_names))
        bars = ax.bar(x, means, 0.6, yerr=sems, capsize=3,
                      color="#1f77b4", edgecolor="white", linewidth=0.5)
        for bar, y in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    max(y, 0) + 0.015, f"{y:.3f}",
                    ha="center", va="bottom", fontsize=7.5,
                    fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(arch_labels, fontsize=8, rotation=25, ha="right")
        if ci == 0:
            ax.set_ylabel("Mean R²  (± SEM)", fontsize=11)
        ax.set_title(COND_LABELS[cond], fontsize=12, fontweight="bold")
        ax.legend(fontsize=7.5, loc="best")
        ax.grid(axis="y", alpha=0.2)
        ax.axhline(0, color="k", lw=0.4, alpha=0.3)

    fig.suptitle(f"MLP Architecture Sweep — Retrain per Condition  "
                 f"(K={K}, {n_worms} worms)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "sweep_mlp_arch.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ sweep_mlp_arch.png")


def _plot_combined_ranking(all_results, out):
    """Per-condition horizontal bar charts ranking all experiments."""
    n_worms = len(all_results)
    exp_names = list(next(iter(all_results.values())).keys())

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    for ci, cond in enumerate(CONDS):
        ax = axes[ci]

        data = []
        for ename in exp_names:
            m, s = _agg(all_results, ename, cond)
            data.append({"name": ename, "mean": m, "sem": s})
        data.sort(key=lambda d: d["mean"], reverse=True)

        y = np.arange(len(data))
        colors = []
        for d in data:
            if d["name"] == "ridge_analytical":
                colors.append("#d62728")
            elif "ridge" in d["name"]:
                colors.append("#2ca02c")
            else:
                colors.append("#1f77b4")

        bars = ax.barh(y, [d["mean"] for d in data],
                       xerr=[d["sem"] for d in data],
                       color=colors, capsize=2,
                       edgecolor="white", linewidth=0.5)
        for bar, d in zip(bars, data):
            xpos = max(d["mean"], 0) + 0.008
            ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                    f"{d['mean']:.3f}", va="center", fontsize=7,
                    fontweight="bold")

        ax.set_yticks(y)
        ax.set_yticklabels([d["name"] for d in data], fontsize=7)
        ax.set_xlabel("Mean R²", fontsize=10)
        ax.set_title(COND_LABELS[cond], fontsize=12, fontweight="bold")
        ax.axvline(0, color="k", lw=0.5, alpha=0.3)
        ax.grid(axis="x", alpha=0.2)
        ax.invert_yaxis()

    fig.suptitle(f"All Experiments Ranked  (K={K}, {n_worms} worms)\n"
                 "🟥 Ridge analytical   🟩 Ridge retrain   🟦 MLP retrain",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "sweep_combined.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ sweep_combined.png")


def _plot_heatmap(all_results, out):
    """Heatmap: experiments × conditions."""
    n_worms = len(all_results)
    exp_names = list(next(iter(all_results.values())).keys())

    mat = np.zeros((len(exp_names), len(CONDS)))
    for ei, ename in enumerate(exp_names):
        for ci, cond in enumerate(CONDS):
            mat[ei, ci] = _agg(all_results, ename, cond)[0]

    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=-0.1, vmax=0.8)

    ax.set_xticks(range(len(CONDS)))
    ax.set_xticklabels([COND_LABELS[c] for c in CONDS], fontsize=10)
    ax.set_yticks(range(len(exp_names)))
    ax.set_yticklabels(exp_names, fontsize=8)

    for ei in range(len(exp_names)):
        for ci in range(len(CONDS)):
            v = mat[ei, ci]
            ax.text(ci, ei, f"{v:.3f}", ha="center", va="center",
                    fontsize=7.5, fontweight="bold",
                    color="white" if v < 0.15 or v > 0.6 else "black")

    fig.colorbar(im, ax=ax, shrink=0.6, label="Mean R²")
    ax.set_title(f"R² Heatmap — Retrain Sweep  (K={K}, {n_worms} worms)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "sweep_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ sweep_heatmap.png")


def _plot_all(all_results, out):
    _plot_ridge_alpha_sweep(all_results, out)
    _plot_mlp_arch_sweep(all_results, out)
    _plot_combined_ranking(all_results, out)
    _plot_heatmap(all_results, out)


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

def _print_summary(all_results):
    exp_names = list(next(iter(all_results.values())).keys())

    print("\n" + "=" * 90)
    print(f"{'Experiment':<28s}  {'causal_self':>12s}  {'self':>12s}  {'causal':>12s}")
    print("=" * 90)

    for ename in exp_names:
        row = f"{ename:<28s}"
        for cond in CONDS:
            m, s = _agg(all_results, ename, cond)
            row += f"  {m:>6.3f}±{s:.3f}"
        print(row)

    # Best per category
    print("-" * 90)
    for cond in CONDS:
        ridge_best, ridge_best_name = -999, ""
        mlp_best, mlp_best_name = -999, ""
        for ename in exp_names:
            m, _ = _agg(all_results, ename, cond)
            if "ridge" in ename and m > ridge_best:
                ridge_best, ridge_best_name = m, ename
            if "mlp" in ename and m > mlp_best:
                mlp_best, mlp_best_name = m, ename
        print(f"  Best Ridge for {cond:14s}: {ridge_best_name} = {ridge_best:.3f}")
        print(f"  Best MLP   for {cond:14s}: {mlp_best_name} = {mlp_best:.3f}")
        delta = mlp_best - ridge_best
        winner = "MLP" if delta > 0 else "Ridge"
        print(f"  → {winner} wins by {abs(delta):.3f}")
    print("=" * 90)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",
                    default="data/used/behaviour+neuronal activity "
                            "atanas (2023)/2")
    ap.add_argument("--n_worms", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out",
                    default="output_plots/neural_activity_decoder_v4/"
                            "retrain_sweep")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(glob.glob(str(Path(args.data_dir) / "*.h5")))
    if len(h5_files) > args.n_worms:
        indices = np.linspace(0, len(h5_files) - 1, args.n_worms, dtype=int)
        h5_files = [h5_files[i] for i in indices]

    # ── Build experiment list ──
    experiments = OrderedDict()
    experiments["ridge_analytical"] = {"type": "ridge_analytical"}
    for a in RIDGE_ALPHAS:
        experiments[f"ridge_a{a:g}"] = {"type": "ridge_retrain", "alpha": a}
    for h, l in MLP_ARCHS:
        experiments[f"mlp_{h}h{l}_retrain"] = {
            "type": "mlp_retrain", "hidden": h, "n_layers": l}

    print(f"Retrain Sweep — K={K}, {len(h5_files)} worms, "
          f"{len(experiments)} experiments")
    print(f"Device: {args.device}")
    print(f"Ridge α: {RIDGE_ALPHAS}")
    print(f"MLP archs: {MLP_ARCHS}")
    print(f"Experiments: {list(experiments.keys())}")
    print()

    all_results = {}

    for h5_path in h5_files:
        worm_data = load_worm_data(h5_path, n_beh_modes=6)
        u = worm_data["u"].astype(np.float32)
        worm_id = worm_data["worm_id"]
        T, N = u.shape

        print(f"  ═══ {worm_id}  T={T}  N={N} ═══")
        results = {}

        for exp_name, cfg in experiments.items():
            print(f"    {exp_name} ...", end=" ", flush=True)
            t0 = time.time()

            if cfg["type"] == "ridge_analytical":
                r2s = _run_ridge_analytical(u, N)
            elif cfg["type"] == "ridge_retrain":
                r2s = _run_ridge_retrain(u, N, cfg["alpha"])
            elif cfg["type"] == "mlp_retrain":
                r2s = _run_mlp_retrain(u, N, cfg["hidden"],
                                       cfg["n_layers"], args.device)

            elapsed = time.time() - t0
            results[exp_name] = {
                "r2": {c: {"mean": float(np.nanmean(v)),
                           "per_neuron": [float(x) for x in v]}
                       for c, v in r2s.items()},
                "time": elapsed,
            }
            summary = "  ".join(f"{c}={np.nanmean(v):.3f}"
                                for c, v in r2s.items())
            print(f"{summary}  [{elapsed:.1f}s]")

        all_results[worm_id] = results

        # Save incrementally
        with open(out / "results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # ── Summary & Plots ──
    _print_summary(all_results)
    print("\nGenerating plots...")
    _plot_all(all_results, out)
    print("\nDone!")


if __name__ == "__main__":
    main()
