#!/usr/bin/env python
"""Compare Ridge, MLP, and AR(2)+MLP decoders across all worms.

Produces:
  1. Per-worm R² comparison plots  (bar, boxplot, heatmap)
  2. Behaviour statistics per model (amplitude distribution, ACF, PSD)
     — averaged across all worms, like the paper's Fig 4 style
  3. Executive summary  (text + JSON)

Usage:
    .venv/bin/python -m scripts.compare_all_worms \
        --h5_dir "data/used/behaviour+neuronal activity atanas (2023)/2" \
        --out_dir output_plots/behaviour_decoder/comparison_3models
"""
from __future__ import annotations

import argparse, glob, json, sys, time, traceback, warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import signal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stage2.behavior_decoder_eval import (
    _log_ridge_grid,
    _ridge_cv_single_target,
    build_lagged_features_np,
)
try:
    from scripts.benchmark_ar_decoder_v2 import (
        load_data, build_ar_features, r2_score,
    )
except ModuleNotFoundError:
    from benchmark_ar_decoder_v2 import (
        load_data, build_ar_features, r2_score,
    )
from scripts.unified_benchmark import _ridge_fit, _train_mlp, _predict_mlp

import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════
#  Single-worm benchmark: Ridge, MLP, AR(2)+MLP
# ═══════════════════════════════════════════════════════════════════════

def _run_one_worm(h5_path: str, *,
                  n_lags: int = 8,
                  n_folds: int = 5,
                  K: int = 6,
                  e2e_epochs: int = 300,
                  tbptt_chunk: int = 64,
                  max_rho: float = 0.98,
                  device: str = "cpu") -> dict:
    """Return dict with R² per model and held-out predictions."""

    u, b_full, dt = load_data(h5_path, all_neurons=False)
    K = min(K, b_full.shape[1])
    b = b_full[:, :K]
    T = b.shape[0]
    warmup = max(2, n_lags)
    M_raw = u.shape[1]
    X_neural = build_lagged_features_np(u, n_lags)
    d_in = X_neural.shape[1]

    ridge_grid = _log_ridge_grid(-3.0, 10.0, 50)
    n_inner = n_folds - 1

    # ── Folds (contiguous temporal blocks) ────────────────────────────
    valid_len = T - warmup
    fold_size = valid_len // n_folds
    folds = []
    for i in range(n_folds):
        s = warmup + i * fold_size
        e = warmup + (i + 1) * fold_size if i < n_folds - 1 else T
        folds.append((s, e))

    RIDGE = "Ridge"
    MLP   = "MLP"
    AR_MLP = "AR(2)+MLP"
    models = [RIDGE, MLP, AR_MLP]
    ho_preds = {m: np.full((T, K), np.nan) for m in models}

    for fi, (ts, te) in enumerate(folds):
        train_mask = np.ones(T, dtype=bool)
        train_mask[:warmup] = False
        train_mask[ts:te] = False
        train_idx = np.where(train_mask)[0]

        X_tr = X_neural[train_idx]
        X_te = X_neural[ts:te]
        b_tr = b[train_idx]

        # Standardise features
        mu, sigma = X_tr.mean(0), X_tr.std(0) + 1e-8
        X_tr_s = (X_tr - mu) / sigma
        X_te_s = (X_neural[ts:te] - mu) / sigma
        X_all_s = (X_neural - mu) / sigma

        # ── Ridge (per-timestep) ─────────────────────────────────────
        for j in range(K):
            coef, intc, _ = _ridge_fit(X_tr_s, b_tr[:, j],
                                       ridge_grid, n_inner)
            ho_preds[RIDGE][ts:te, j] = X_te_s @ coef + intc

        # ── MLP (per-timestep) ───────────────────────────────────────
        mlp = _train_mlp(X_tr_s, b_tr, K, hidden=32, lr=1e-3, epochs=200)
        ho_preds[MLP][ts:te] = _predict_mlp(mlp, X_te_s)

        # ── AR(2)+MLP (2-step, free-run) ─────────────────────────────
        # Step 1: Fit VAR(2) matrices on training data
        X_ar = build_ar_features(b, 2)   # (T, 2K)
        X_ar_tr = X_ar[train_idx]

        M1 = np.zeros((K, K)); M2 = np.zeros((K, K)); c_v = np.zeros(K)
        for j in range(K):
            coef, intc, _ = _ridge_fit(X_ar_tr, b_tr[:, j],
                                       ridge_grid, n_inner)
            M1[j] = coef[:K]
            M2[j] = coef[K:2*K]
            c_v[j] = intc

        # Step 2: Compute residual of VAR(2) fit
        b_tm1 = np.zeros_like(b); b_tm1[1:] = b[:-1]
        b_tm2 = np.zeros_like(b); b_tm2[2:] = b[:-2]
        resid = b - (b_tm1 @ M1.T + b_tm2 @ M2.T + c_v)
        r_tr = resid[train_idx]

        # Step 3: Fit MLP on residual: neural features → residual
        mlp_r = _train_mlp(X_tr_s, r_tr, K, hidden=32, lr=1e-3, epochs=200)
        drive_mlp = _predict_mlp(mlp_r, X_all_s)

        # Step 4: Free-run on test fold
        p1, p2 = b[ts - 1].copy(), b[ts - 2].copy()
        for t in range(ts, te):
            p_new = M1 @ p1 + M2 @ p2 + drive_mlp[t] + c_v
            ho_preds[AR_MLP][t] = p_new
            p2, p1 = p1, p_new

    # ── R² ────────────────────────────────────────────────────────────
    valid = np.arange(warmup, T)
    results = {"T": int(T), "N": int(M_raw), "K": K, "dt": float(dt)}
    for name in models:
        preds = ho_preds[name]
        ok = np.isfinite(preds[valid, 0])
        idx = valid[ok]
        if idx.size < 10:
            results[name] = [float("nan")] * K
        else:
            results[name] = [
                float(r2_score(b[idx, j], preds[idx, j]))
                for j in range(K)]

    return results, ho_preds, b, dt


# ═══════════════════════════════════════════════════════════════════════
#  Behaviour statistics helpers
# ═══════════════════════════════════════════════════════════════════════

def _acf(x, max_lag):
    """Normalised autocorrelation."""
    x = x - x.mean()
    c = np.correlate(x, x, mode="full")
    c = c[len(c) // 2:]
    ml = min(max_lag, len(c))
    return c[:ml] / (c[0] + 1e-15)


def _psd(x, fs, nperseg=256):
    """Welch PSD."""
    nperseg = min(nperseg, len(x) // 2)
    if nperseg < 16:
        return np.array([]), np.array([])
    return signal.welch(x, fs=fs, nperseg=nperseg)


def collect_behaviour_stats(b_gt, preds_dict, dt, K,
                            max_lag_steps=200, nperseg=256):
    """Collect amplitude distribution, ACF, and PSD for GT and each model.

    Returns dict of arrays keyed by model name (and 'Real').
    """
    fs = 1.0 / dt
    max_lag = min(max_lag_steps, b_gt.shape[0] // 2)
    stats = {}

    # Ground truth
    stats["Real"] = {}
    for j in range(K):
        g = b_gt[:, j]
        stats["Real"][j] = {
            "values": g,
            "acf": _acf(g, max_lag),
            "psd": _psd(g, fs, nperseg),
        }

    # Models
    for mname, preds in preds_dict.items():
        stats[mname] = {}
        for j in range(K):
            p = preds[:, j]
            ok = np.isfinite(p)
            if ok.sum() < 50:
                stats[mname][j] = None
                continue
            p_clean = p[ok]
            stats[mname][j] = {
                "values": p_clean,
                "acf": _acf(p_clean, max_lag),
                "psd": _psd(p_clean, fs, nperseg),
            }

    return stats


# ═══════════════════════════════════════════════════════════════════════
#  Plotting functions
# ═══════════════════════════════════════════════════════════════════════

COLORS = {"Ridge": "#1f77b4", "MLP": "#2ca02c", "AR(2)+MLP": "#d62728"}
MODEL_ORDER = ["Ridge", "MLP", "AR(2)+MLP"]


def plot_r2_bar_summary(all_results, out_dir, K=6):
    """Grouped bar: mean ± SEM across worms, per eigenworm mode."""
    worm_ids = sorted(all_results.keys())
    mode_names = [f"EW{j+1}" for j in range(K)]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(K)
    n = len(MODEL_ORDER)
    w = 0.8 / n

    for mi, mname in enumerate(MODEL_ORDER):
        mat = np.array([all_results[wid].get(mname, [np.nan]*K)[:K]
                        for wid in worm_ids])
        means = np.nanmean(mat, axis=0)
        sems  = np.nanstd(mat, axis=0) / np.sqrt(
            np.sum(np.isfinite(mat), axis=0).clip(1))
        ax.bar(x + mi * w, means, w, yerr=sems, capsize=3,
               label=mname, color=COLORS[mname], edgecolor="white",
               linewidth=0.5, alpha=0.85)
        for j in range(K):
            ax.text(x[j] + mi * w, means[j] + sems[j] + 0.02,
                    f"{means[j]:.3f}", ha="center", fontsize=7,
                    color=COLORS[mname])

    ax.set_xticks(x + w * (n - 1) / 2)
    ax.set_xticklabels(mode_names, fontsize=12)
    ax.set_ylabel("R²  (held-out)", fontsize=12)
    ax.set_title(f"Decoder R² across {len(worm_ids)} worms (mean ± SEM)",
                 fontsize=13)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_ylim(-0.15, 1.05)
    ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "01_r2_bar_summary.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def plot_r2_boxplot(all_results, out_dir, K=6):
    """Per-mode boxplots across worms, one subplot per EW."""
    worm_ids = sorted(all_results.keys())
    mode_names = [f"EW{j+1}" for j in range(K)]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
    axes = axes.ravel()

    for j in range(K):
        ax = axes[j]
        data_box = []
        labels_box = []
        for mname in MODEL_ORDER:
            vals = [all_results[w].get(mname, [np.nan]*K)[j]
                    for w in worm_ids]
            vals = np.array(vals)
            data_box.append(vals[np.isfinite(vals)])
            labels_box.append(mname)

        bp = ax.boxplot(data_box, tick_labels=labels_box, patch_artist=True,
                        widths=0.6, showfliers=True, flierprops=dict(
                            marker='o', markersize=3, alpha=0.4))
        for patch, mname in zip(bp['boxes'], MODEL_ORDER):
            patch.set_facecolor(COLORS[mname])
            patch.set_alpha(0.6)

        # Overlay individual worm points
        for mi, mname in enumerate(MODEL_ORDER):
            vals = [all_results[w].get(mname, [np.nan]*K)[j]
                    for w in worm_ids]
            vals = np.array(vals)
            ok = np.isfinite(vals)
            jitter = np.random.uniform(-0.12, 0.12, size=ok.sum())
            ax.scatter(np.full(ok.sum(), mi + 1) + jitter,
                       vals[ok], s=12, alpha=0.5, color=COLORS[mname],
                       edgecolors="white", linewidth=0.3, zorder=3)

        ax.set_title(mode_names[j], fontsize=12)
        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.3)
        ax.set_ylim(-0.5, 1.05)
        if j % 3 == 0:
            ax.set_ylabel("R²", fontsize=11)

    fig.suptitle(f"Per-mode R² distribution across {len(worm_ids)} worms",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "02_r2_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_r2_heatmap(all_results, out_dir, K=6):
    """Side-by-side heatmaps: worms × modes, one per model."""
    worm_ids = sorted(all_results.keys())
    n_worms = len(worm_ids)
    mode_names = [f"EW{j+1}" for j in range(K)]

    fig, axes = plt.subplots(1, 3, figsize=(20, max(8, n_worms * 0.35)),
                             sharey=True)
    for ax, mname in zip(axes, MODEL_ORDER):
        mat = np.array([all_results[w].get(mname, [np.nan]*K)[:K]
                        for w in worm_ids])
        im = ax.imshow(mat, aspect="auto", vmin=-0.2, vmax=1.0,
                       cmap="RdYlGn")
        ax.set_xticks(range(K))
        ax.set_xticklabels(mode_names, fontsize=10)
        ax.set_title(mname, fontsize=13, fontweight="bold")
        for i in range(n_worms):
            for j in range(K):
                v = mat[i, j]
                if np.isfinite(v):
                    color = "white" if v < 0.3 else "black"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=6, color=color)
    axes[0].set_yticks(range(n_worms))
    axes[0].set_yticklabels(worm_ids, fontsize=7)
    axes[0].set_ylabel("Worm", fontsize=11)
    fig.colorbar(im, ax=axes, shrink=0.6, label="R²")
    fig.suptitle("Per-worm decoder R²", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "03_r2_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_behaviour_statistics(all_stats_list, out_dir, K=4, max_lag=200):
    """Behaviour statistics: amplitude distribution, ACF, PSD per EW.

    Plots are averaged across worms, matching the style:
      Row = eigenworm mode (EW1..EW4)
      Col = [Amplitude Distribution | Autocorrelation | Power Spectrum]
    """
    mode_names = [f"EW{j+1}" for j in range(K)]
    n_modes = min(K, 4)  # plot first 4 modes

    for mname in MODEL_ORDER:
        fig, axes = plt.subplots(n_modes, 3, figsize=(16, 3.5 * n_modes))
        fig.suptitle(f"Behaviour Statistics: Real Worm vs {mname}",
                     fontsize=14, fontweight="bold", y=1.01)

        for j in range(n_modes):
            ax_amp = axes[j, 0]
            ax_acf = axes[j, 1]
            ax_psd = axes[j, 2]

            # Collect across worms
            gt_values_all = []
            pred_values_all = []
            gt_acfs = []
            pred_acfs = []
            gt_psds = []
            pred_psds = []

            for stats in all_stats_list:
                if "Real" in stats and j in stats["Real"]:
                    s = stats["Real"][j]
                    gt_values_all.append(s["values"])
                    gt_acfs.append(s["acf"])
                    f_gt, p_gt = s["psd"]
                    if len(f_gt) > 0:
                        gt_psds.append((f_gt, p_gt))

                if mname in stats and j in stats[mname]:
                    s = stats[mname][j]
                    if s is not None:
                        pred_values_all.append(s["values"])
                        pred_acfs.append(s["acf"])
                        f_pr, p_pr = s["psd"]
                        if len(f_pr) > 0:
                            pred_psds.append((f_pr, p_pr))

            # ── Amplitude distribution ────────────────────────────────
            if gt_values_all:
                all_gt = np.concatenate(gt_values_all)
                bins = np.linspace(np.percentile(all_gt, 0.5),
                                   np.percentile(all_gt, 99.5), 50)
                ax_amp.hist(all_gt, bins=bins, density=True, alpha=0.5,
                            color="gray", label="Real", edgecolor="none")
            if pred_values_all:
                all_pr = np.concatenate(pred_values_all)
                ax_amp.hist(all_pr, bins=bins, density=True, alpha=0.5,
                            color=COLORS[mname], label=f"{mname}",
                            edgecolor="none")
            ax_amp.set_title(f"{mode_names[j]}: Amplitude Distribution",
                             fontsize=11)
            ax_amp.set_xlabel("Eigenworm value")
            ax_amp.set_ylabel("Density")
            ax_amp.legend(fontsize=8)

            # ── ACF ───────────────────────────────────────────────────
            if gt_acfs:
                min_len = min(len(a) for a in gt_acfs)
                acf_gt_mean = np.mean(
                    [a[:min_len] for a in gt_acfs], axis=0)
                lags = np.arange(min_len)
                ax_acf.plot(lags, acf_gt_mean, "k-", lw=1.5,
                            label="Real")
            if pred_acfs:
                min_len_p = min(len(a) for a in pred_acfs)
                acf_pr_mean = np.mean(
                    [a[:min_len_p] for a in pred_acfs], axis=0)
                ax_acf.plot(np.arange(min_len_p), acf_pr_mean,
                            color=COLORS[mname], lw=1.5,
                            label=f"{mname}")
            ax_acf.set_title(f"{mode_names[j]}: Autocorrelation",
                             fontsize=11)
            ax_acf.set_xlabel("Lag (steps)")
            ax_acf.set_ylabel("ACF")
            ax_acf.axhline(0, color="gray", lw=0.5, ls="--")
            ax_acf.legend(fontsize=8)

            # ── PSD ───────────────────────────────────────────────────
            if gt_psds:
                # Average PSD in log space
                min_len_f = min(len(p[1]) for p in gt_psds)
                f_common = gt_psds[0][0][:min_len_f]
                psd_gt_arr = np.array(
                    [p[1][:min_len_f] for p in gt_psds])
                psd_gt_mean = np.mean(psd_gt_arr, axis=0)
                ax_psd.semilogy(f_common, psd_gt_mean, "k-", lw=1.5,
                                label="Real")

            if pred_psds:
                min_len_fp = min(len(p[1]) for p in pred_psds)
                f_common_p = pred_psds[0][0][:min_len_fp]
                psd_pr_arr = np.array(
                    [p[1][:min_len_fp] for p in pred_psds])
                psd_pr_mean = np.mean(psd_pr_arr, axis=0)
                ax_psd.semilogy(f_common_p, psd_pr_mean,
                                color=COLORS[mname], lw=1.5,
                                label=f"{mname}")

            ax_psd.set_title(f"{mode_names[j]}: Power Spectrum",
                             fontsize=11)
            ax_psd.set_xlabel("Frequency (cycles/step)")
            ax_psd.set_ylabel("PSD")
            ax_psd.legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(out_dir / f"04_beh_stats_{mname.replace('(','').replace(')','').replace('+','_')}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── Combined: all 3 models on same plot ───────────────────────────
    fig, axes = plt.subplots(n_modes, 3, figsize=(16, 3.5 * n_modes))
    fig.suptitle("Behaviour Statistics: Real Worm vs All Models",
                 fontsize=14, fontweight="bold", y=1.01)

    for j in range(n_modes):
        ax_amp = axes[j, 0]
        ax_acf = axes[j, 1]
        ax_psd = axes[j, 2]

        # GT
        gt_values_all = []
        gt_acfs = []
        gt_psds = []
        for stats in all_stats_list:
            if "Real" in stats and j in stats["Real"]:
                s = stats["Real"][j]
                gt_values_all.append(s["values"])
                gt_acfs.append(s["acf"])
                f_gt, p_gt = s["psd"]
                if len(f_gt) > 0:
                    gt_psds.append((f_gt, p_gt))

        if gt_values_all:
            all_gt = np.concatenate(gt_values_all)
            bins = np.linspace(np.percentile(all_gt, 0.5),
                               np.percentile(all_gt, 99.5), 50)
            ax_amp.hist(all_gt, bins=bins, density=True, alpha=0.4,
                        color="gray", label="Real", edgecolor="none")

        if gt_acfs:
            min_len = min(len(a) for a in gt_acfs)
            acf_gt_mean = np.mean([a[:min_len] for a in gt_acfs], axis=0)
            ax_acf.plot(np.arange(min_len), acf_gt_mean, "k-", lw=2,
                        label="Real")

        if gt_psds:
            min_len_f = min(len(p[1]) for p in gt_psds)
            f_common = gt_psds[0][0][:min_len_f]
            psd_gt_mean = np.mean(
                [p[1][:min_len_f] for p in gt_psds], axis=0)
            ax_psd.semilogy(f_common, psd_gt_mean, "k-", lw=2,
                            label="Real")

        # Models
        for mname in MODEL_ORDER:
            pred_values_all = []
            pred_acfs = []
            pred_psds = []
            for stats in all_stats_list:
                if mname in stats and j in stats[mname]:
                    s = stats[mname][j]
                    if s is not None:
                        pred_values_all.append(s["values"])
                        pred_acfs.append(s["acf"])
                        f_pr, p_pr = s["psd"]
                        if len(f_pr) > 0:
                            pred_psds.append((f_pr, p_pr))

            if pred_values_all and gt_values_all:
                all_pr = np.concatenate(pred_values_all)
                ax_amp.hist(all_pr, bins=bins, density=True, alpha=0.35,
                            color=COLORS[mname], label=mname,
                            edgecolor="none")

            if pred_acfs:
                min_len_p = min(len(a) for a in pred_acfs)
                acf_pr_mean = np.mean(
                    [a[:min_len_p] for a in pred_acfs], axis=0)
                ax_acf.plot(np.arange(min_len_p), acf_pr_mean,
                            color=COLORS[mname], lw=1.5, label=mname)

            if pred_psds:
                min_len_fp = min(len(p[1]) for p in pred_psds)
                f_common_p = pred_psds[0][0][:min_len_fp]
                psd_pr_mean = np.mean(
                    [p[1][:min_len_fp] for p in pred_psds], axis=0)
                ax_psd.semilogy(f_common_p, psd_pr_mean,
                                color=COLORS[mname], lw=1.5,
                                label=mname)

        ax_amp.set_title(f"{mode_names[j]}: Amplitude Distribution",
                         fontsize=11)
        ax_amp.set_xlabel("Eigenworm value")
        ax_amp.set_ylabel("Density")
        ax_amp.legend(fontsize=8)

        ax_acf.set_title(f"{mode_names[j]}: Autocorrelation", fontsize=11)
        ax_acf.set_xlabel("Lag (steps)")
        ax_acf.set_ylabel("ACF")
        ax_acf.axhline(0, color="gray", lw=0.5, ls="--")
        ax_acf.legend(fontsize=8)

        ax_psd.set_title(f"{mode_names[j]}: Power Spectrum", fontsize=11)
        ax_psd.set_xlabel("Frequency (cycles/step)")
        ax_psd.set_ylabel("PSD")
        ax_psd.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "05_beh_stats_combined.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def plot_per_worm_r2_scatter(all_results, out_dir, K=6):
    """Scatter plot: Ridge R² vs AR(2)+MLP R² per worm, per mode."""
    worm_ids = sorted(all_results.keys())
    mode_names = [f"EW{j+1}" for j in range(K)]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()

    for j in range(K):
        ax = axes[j]
        ridge_vals = np.array([all_results[w].get("Ridge", [np.nan]*K)[j]
                               for w in worm_ids])
        ar_vals = np.array([all_results[w].get("AR(2)+MLP", [np.nan]*K)[j]
                            for w in worm_ids])
        mlp_vals = np.array([all_results[w].get("MLP", [np.nan]*K)[j]
                             for w in worm_ids])

        ok = np.isfinite(ridge_vals) & np.isfinite(ar_vals)
        ax.scatter(ridge_vals[ok], ar_vals[ok], s=25, alpha=0.6,
                   color=COLORS["AR(2)+MLP"], label="AR(2)+MLP",
                   edgecolors="white", linewidth=0.3, zorder=3)

        ok2 = np.isfinite(ridge_vals) & np.isfinite(mlp_vals)
        ax.scatter(ridge_vals[ok2], mlp_vals[ok2], s=25, alpha=0.6,
                   color=COLORS["MLP"], label="MLP", marker="^",
                   edgecolors="white", linewidth=0.3, zorder=3)

        # Diagonal
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.3)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Ridge R²"); ax.set_ylabel("Model R²")
        ax.set_title(mode_names[j], fontsize=12)
        ax.set_aspect("equal")
        ax.legend(fontsize=8)

    fig.suptitle(f"Per-worm R² — Ridge vs MLP / AR(2)+MLP  "
                 f"({len(worm_ids)} worms)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "06_r2_scatter_pairwise.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def plot_overall_mean_bar(all_results, out_dir, K=6):
    """Single bar chart: overall mean R² (across all modes and worms)."""
    worm_ids = sorted(all_results.keys())

    fig, ax = plt.subplots(figsize=(6, 4.5))
    means = []
    sems = []
    for mname in MODEL_ORDER:
        mat = np.array([all_results[wid].get(mname, [np.nan]*K)[:K]
                        for wid in worm_ids])
        flat = mat.ravel()
        flat = flat[np.isfinite(flat)]
        means.append(np.mean(flat))
        sems.append(np.std(flat) / np.sqrt(len(flat)))

    bars = ax.bar(MODEL_ORDER, means, yerr=sems, capsize=5,
                  color=[COLORS[m] for m in MODEL_ORDER],
                  edgecolor="white", linewidth=0.5, alpha=0.85)
    for bar, m, s in zip(bars, means, sems):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s + 0.01,
                f"{m:.3f}", ha="center", fontsize=10, fontweight="bold")

    ax.set_ylabel("Overall R² (mean across modes & worms)", fontsize=11)
    ax.set_title(f"Overall decoder performance ({len(worm_ids)} worms)",
                 fontsize=13)
    ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.3)
    ax.set_ylim(-0.1, max(means) + max(sems) + 0.1)
    fig.tight_layout()
    fig.savefig(out_dir / "07_overall_mean_r2.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
#  Executive summary
# ═══════════════════════════════════════════════════════════════════════

def write_executive_summary(all_results, out_dir, K=6, elapsed=0):
    """Write comprehensive text and JSON executive summary."""
    worm_ids = sorted(all_results.keys())
    n_worms = len(worm_ids)
    mode_names = [f"EW{j+1}" for j in range(K)]

    lines = []
    lines.append("=" * 80)
    lines.append("  EXECUTIVE SUMMARY: Behaviour Decoder Comparison")
    lines.append("=" * 80)
    lines.append(f"  Worms: {n_worms}")
    lines.append(f"  Models: {', '.join(MODEL_ORDER)}")
    lines.append(f"  Eigenworm modes: {K}")
    lines.append(f"  CV: 5-fold contiguous temporal blocks")
    lines.append(f"  Ridge & MLP: per-timestep (teacher-forced)")
    lines.append(f"  AR(2)+MLP: free-running (autoregressive)")
    lines.append(f"  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    lines.append("")

    # Per-model summary table
    lines.append("─" * 80)
    header = f"  {'Model':<14s}"
    for mn in mode_names:
        header += f"  {mn:>7s}"
    header += f"  {'Mean':>7s}  {'Median':>7s}"
    lines.append(header)
    lines.append("─" * 80)

    summary_json = {"n_worms": n_worms, "models": {}}

    for mname in MODEL_ORDER:
        mat = np.array([all_results[wid].get(mname, [np.nan]*K)[:K]
                        for wid in worm_ids])
        per_mode_mean = np.nanmean(mat, axis=0)
        per_mode_median = np.nanmedian(mat, axis=0)
        per_mode_std = np.nanstd(mat, axis=0)
        overall_mean = np.nanmean(mat)
        overall_median = np.nanmedian(mat)

        row = f"  {mname:<14s}"
        for v in per_mode_mean:
            row += f"  {v:7.3f}"
        row += f"  {overall_mean:7.3f}  {overall_median:7.3f}"
        lines.append(row)

        # SEM row
        sem = per_mode_std / np.sqrt(np.sum(np.isfinite(mat), axis=0).clip(1))
        row_sem = f"  {'(±SEM)':<14s}"
        for s in sem:
            row_sem += f"  {s:7.3f}"
        lines.append(row_sem)

        summary_json["models"][mname] = {
            "per_mode_mean": per_mode_mean.tolist(),
            "per_mode_median": per_mode_median.tolist(),
            "per_mode_std": per_mode_std.tolist(),
            "overall_mean": float(overall_mean),
            "overall_median": float(overall_median),
        }

    lines.append("─" * 80)
    lines.append("")

    # ── Pairwise comparisons ──────────────────────────────────────────
    lines.append("  PAIRWISE COMPARISONS (mean R² difference, + favours row)")
    lines.append("─" * 80)
    for i, m1 in enumerate(MODEL_ORDER):
        for m2 in MODEL_ORDER[i+1:]:
            mat1 = np.array([all_results[w].get(m1, [np.nan]*K)[:K]
                             for w in worm_ids])
            mat2 = np.array([all_results[w].get(m2, [np.nan]*K)[:K]
                             for w in worm_ids])
            diff = mat1 - mat2  # (n_worms, K)
            ok = np.isfinite(diff)
            mean_diff = np.nanmean(diff)
            # Per-worm mean difference
            per_worm = np.nanmean(diff, axis=1)
            n_win = np.sum(per_worm > 0)
            n_lose = np.sum(per_worm < 0)
            lines.append(
                f"  {m1} vs {m2}: Δ={mean_diff:+.4f}  "
                f"({m1} wins {n_win}/{n_worms}, "
                f"{m2} wins {n_lose}/{n_worms})")

    lines.append("")

    # ── Per-worm details ──────────────────────────────────────────────
    lines.append("  PER-WORM MEAN R² (across all modes)")
    lines.append("─" * 80)
    pw_header = f"  {'Worm':<20s}  {'T':>5s}  {'N':>3s}"
    for mname in MODEL_ORDER:
        pw_header += f"  {mname:>12s}"
    pw_header += f"  {'Best':>12s}"
    lines.append(pw_header)
    lines.append("  " + "─" * 76)

    for wid in worm_ids:
        res = all_results[wid]
        T = res.get("T", "?")
        N = res.get("N", "?")
        row = f"  {wid:<20s}  {T:>5}  {N:>3}"
        best_m = ""
        best_v = -999
        for mname in MODEL_ORDER:
            vals = res.get(mname, [np.nan]*K)
            mean_r2 = np.nanmean(vals[:K])
            row += f"  {mean_r2:12.3f}"
            if mean_r2 > best_v:
                best_v = mean_r2
                best_m = mname
        row += f"  {best_m:>12s}"
        lines.append(row)

    lines.append("")
    lines.append("=" * 80)

    summary_text = "\n".join(lines)

    # Write text
    txt_path = out_dir / "executive_summary.txt"
    with open(txt_path, "w") as f:
        f.write(summary_text)
    print(summary_text)
    print(f"\n  Summary → {txt_path}")

    # Write JSON
    summary_json["per_worm"] = {}
    for wid in worm_ids:
        res = all_results[wid]
        summary_json["per_worm"][wid] = {
            "T": res.get("T"), "N": res.get("N"), "dt": res.get("dt"),
        }
        for mname in MODEL_ORDER:
            summary_json["per_worm"][wid][mname] = res.get(
                mname, [None]*K)

    json_path = out_dir / "executive_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary_json, f, indent=2)
    print(f"  JSON   → {json_path}")

    return summary_text


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5_dir", required=True,
                    help="Directory containing .h5 worm files")
    ap.add_argument("--neural_lags", type=int, default=8)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--n_modes", type=int, default=6)
    ap.add_argument("--e2e_epochs", type=int, default=300)
    ap.add_argument("--tbptt_chunk", type=int, default=64)
    ap.add_argument("--max_rho", type=float, default=0.98)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_dir",
                    default="output_plots/behaviour_decoder/comparison_3models")
    ap.add_argument("--resume", action="store_true",
                    help="Skip worms already in results.json")
    ap.add_argument("--max_worms", type=int, default=0,
                    help="Limit number of worms (0=all)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Discover h5 files ─────────────────────────────────────────────
    h5_files = sorted(glob.glob(str(Path(args.h5_dir) / "*.h5")))
    if not h5_files:
        print(f"  No .h5 files found in {args.h5_dir}")
        sys.exit(1)
    if args.max_worms > 0:
        h5_files = h5_files[:args.max_worms]
    print(f"\n  Found {len(h5_files)} worms in {args.h5_dir}\n")

    # ── Resume support ────────────────────────────────────────────────
    json_path = out_dir / "results.json"
    if args.resume and json_path.exists():
        with open(json_path) as f:
            all_results = json.load(f)
        print(f"  Resuming: {len(all_results)} worms already done\n")
    else:
        all_results = {}

    t0 = time.time()
    K = args.n_modes

    # ── Collect behaviour stats for plotting ──────────────────────────
    all_stats_list = []

    for wi, h5_path in enumerate(h5_files):
        worm_id = Path(h5_path).stem

        if worm_id in all_results and "AR(2)+MLP" in all_results[worm_id]:
            print(f"  [{wi+1}/{len(h5_files)}] {worm_id}  ── SKIP (resume)")
            continue

        print(f"\n  [{wi+1}/{len(h5_files)}] {worm_id}  "
              f"──────────────────────")
        try:
            res, ho_preds, b_gt, dt = _run_one_worm(
                h5_path,
                n_lags=args.neural_lags,
                n_folds=args.n_folds,
                K=K,
                e2e_epochs=args.e2e_epochs,
                tbptt_chunk=args.tbptt_chunk,
                max_rho=args.max_rho,
                device=args.device,
            )
            all_results[worm_id] = res

            # Collect behaviour stats
            warmup = max(2, args.neural_lags)
            b_valid = b_gt[warmup:]
            preds_valid = {m: ho_preds[m][warmup:]
                          for m in MODEL_ORDER}
            stats = collect_behaviour_stats(
                b_valid, preds_valid, dt, K)
            all_stats_list.append(stats)

            # Print summary
            for mname in MODEL_ORDER:
                vals = res.get(mname, [float("nan")]*K)
                mean_r2 = np.nanmean(vals)
                print(f"    {mname:<14s}"
                      f"{' '.join(f'{v:7.3f}' for v in vals)}"
                      f"  mean={mean_r2:.3f}")

        except Exception as exc:
            print(f"    ✗ FAILED: {exc}")
            traceback.print_exc()
            all_results[worm_id] = {"error": str(exc)}

        # Checkpoint after every worm
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)

    elapsed = time.time() - t0

    # ── Filter out errors ─────────────────────────────────────────────
    plot_results = {k: v for k, v in all_results.items()
                    if "error" not in v and "AR(2)+MLP" in v}

    if len(plot_results) < 2:
        print("  Too few successful worms for plotting.")
        return

    # ── If we resumed, we need to regenerate stats for existing worms ─
    if len(all_stats_list) < len(plot_results):
        print(f"\n  Regenerating behaviour stats for {len(plot_results)} "
              f"worms...")
        all_stats_list = []
        for worm_id in sorted(plot_results.keys()):
            h5_path = str(Path(args.h5_dir) / f"{worm_id}.h5")
            if not Path(h5_path).exists():
                continue
            try:
                _, ho_preds, b_gt, dt = _run_one_worm(
                    h5_path, n_lags=args.neural_lags,
                    n_folds=args.n_folds, K=K,
                    e2e_epochs=args.e2e_epochs,
                    tbptt_chunk=args.tbptt_chunk,
                    device=args.device)
                warmup = max(2, args.neural_lags)
                b_valid = b_gt[warmup:]
                preds_valid = {m: ho_preds[m][warmup:]
                              for m in MODEL_ORDER}
                stats = collect_behaviour_stats(
                    b_valid, preds_valid, dt, K)
                all_stats_list.append(stats)
            except Exception:
                pass

    # ── Generate all plots ────────────────────────────────────────────
    print(f"\n  Generating plots for {len(plot_results)} worms...")

    plot_r2_bar_summary(plot_results, out_dir, K)
    print(f"    ✓ 01_r2_bar_summary.png")

    plot_r2_boxplot(plot_results, out_dir, K)
    print(f"    ✓ 02_r2_boxplot.png")

    plot_r2_heatmap(plot_results, out_dir, K)
    print(f"    ✓ 03_r2_heatmap.png")

    if all_stats_list:
        plot_behaviour_statistics(all_stats_list, out_dir, K=min(K, 4))
        print(f"    ✓ 04_beh_stats_*.png (per-model)")
        print(f"    ✓ 05_beh_stats_combined.png")

    plot_per_worm_r2_scatter(plot_results, out_dir, K)
    print(f"    ✓ 06_r2_scatter_pairwise.png")

    plot_overall_mean_bar(plot_results, out_dir, K)
    print(f"    ✓ 07_overall_mean_r2.png")

    # ── Executive summary ─────────────────────────────────────────────
    write_executive_summary(plot_results, out_dir, K, elapsed)

    print(f"\n  All outputs → {out_dir}/")
    print(f"  Total elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print("\n  Done.\n")


if __name__ == "__main__":
    main()
