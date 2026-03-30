#!/usr/bin/env python
"""Batch Ridge + MLP (H=128, wd=1e-3) over all worms, with statistics.

Models (per-timestep, 5-fold temporal CV):
  • Ridge   — per-mode α via inner CV
  • MLP     — 2×128, LayerNorm, ReLU, Dropout 0.1, wd=1e-3

Produces:
  results.json          — per-worm R² dictionary
  heatmap.png           — worms × modes heatmap
  scatter_ridge_mlp.png — paired scatter Ridge vs MLP per mode
  bar_summary.png       — grouped bar  mean ± SEM
  violin.png            — violin plots per mode
  summary.txt           — machine-readable text table

Usage:
    python -m scripts.batch_ridge_mlp \
        --h5_dir "data/used/behaviour+neuronal activity atanas (2023)/2" \
        --resume
"""
from __future__ import annotations

import argparse, glob, json, sys, time, traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stage2.behavior_decoder_eval import (
    _log_ridge_grid, build_lagged_features_np,
)
try:
    from scripts.benchmark_ar_decoder_v2 import load_data, r2_score
except ModuleNotFoundError:
    from benchmark_ar_decoder_v2 import load_data, r2_score
from scripts.unified_benchmark import (
    _ridge_fit, _train_mlp, _predict_mlp,
)

MODELS = ["Ridge", "MLP"]
COLORS = {"Ridge": "#1f77b4", "MLP": "#ff7f0e"}


# ──────────────────────────────────────────────────────────────────────
def _run_one_worm(h5_path, *, n_lags=8, n_folds=5, K=6) -> dict:
    u, b_full, dt = load_data(h5_path, all_neurons=False)
    K = min(K, b_full.shape[1])
    b = b_full[:, :K]
    T = b.shape[0]
    warmup = max(2, n_lags)
    M = u.shape[1]
    X = build_lagged_features_np(u, n_lags)
    d_in = X.shape[1]

    ridge_grid = _log_ridge_grid(-3.0, 10.0, 50)
    n_inner = n_folds - 1

    valid_len = T - warmup
    fold_size = valid_len // n_folds
    folds = []
    for i in range(n_folds):
        s = warmup + i * fold_size
        e = warmup + (i + 1) * fold_size if i < n_folds - 1 else T
        folds.append((s, e))

    ho = {m: np.full((T, K), np.nan) for m in MODELS}

    for fi, (ts, te) in enumerate(folds):
        tr_idx = np.concatenate([np.arange(warmup, ts), np.arange(te, T)])
        X_tr, X_te = X[tr_idx], X[ts:te]
        b_tr = b[tr_idx]
        mu, sig = X_tr.mean(0), X_tr.std(0) + 1e-8
        X_tr_s = (X_tr - mu) / sig
        X_te_s = (X_te - mu) / sig

        # Ridge
        for j in range(K):
            coef, intc, _ = _ridge_fit(X_tr_s, b_tr[:, j],
                                       ridge_grid, n_inner)
            ho["Ridge"][ts:te, j] = X_te_s @ coef + intc

        # MLP (H=128, wd=1e-3 — new defaults in _train_mlp)
        mlp = _train_mlp(X_tr_s, b_tr, K)
        ho["MLP"][ts:te] = _predict_mlp(mlp, X_te_s)

    # R²
    valid = np.arange(warmup, T)
    res = {"T": int(T), "N": int(M), "K": K, "dt": float(dt)}
    for m in MODELS:
        ok = np.isfinite(ho[m][valid, 0])
        idx = valid[ok]
        res[m] = ([float(r2_score(b[idx, j], ho[m][idx, j]))
                   for j in range(K)] if idx.size >= 10
                  else [float("nan")] * K)
    return res


# ──────────────────────────────────────────────────────────────────────
#  Plots
# ──────────────────────────────────────────────────────────────────────
def _gather(results, model, K):
    """Return (n_worms, K) matrix."""
    worms = sorted(results.keys())
    return np.array([results[w].get(model, [np.nan]*K)[:K] for w in worms])


def _plot_heatmap(results, out_dir, K):
    worms = sorted(results.keys())
    nw = len(worms)
    modes = [f"a{j+1}" for j in range(K)]

    fig, axes = plt.subplots(1, 2, figsize=(12, max(8, nw * 0.35)),
                             sharey=True)
    for ax, m in zip(axes, MODELS):
        mat = _gather(results, m, K)
        im = ax.imshow(mat, aspect="auto", vmin=-0.2, vmax=1.0,
                       cmap="RdYlGn")
        ax.set_xticks(range(K))
        ax.set_xticklabels(modes, fontsize=10)
        ax.set_title(m, fontsize=13, fontweight="bold")
        for i in range(nw):
            for j in range(K):
                v = mat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=6, color="white" if v < 0.3 else "black")
    axes[0].set_yticks(range(nw))
    axes[0].set_yticklabels(worms, fontsize=6.5)
    axes[0].set_ylabel("Worm", fontsize=11)
    fig.colorbar(im, ax=axes, shrink=0.5, label="R²")
    fig.suptitle(f"Per-worm R² — {nw} worms, 5-fold CV, per-timestep",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    p = out_dir / "heatmap.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Heatmap → {p}")


def _plot_scatter_paired(results, out_dir, K):
    """Scatter Ridge-R² vs MLP-R² per mode, with identity line."""
    modes = [f"a{j+1}" for j in range(K)]
    ridge = _gather(results, "Ridge", K)
    mlp   = _gather(results, "MLP", K)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for j, ax in enumerate(axes.flat):
        if j >= K:
            ax.set_visible(False); continue
        ok = np.isfinite(ridge[:, j]) & np.isfinite(mlp[:, j])
        rx, mx = ridge[ok, j], mlp[ok, j]
        ax.scatter(rx, mx, s=30, alpha=0.7, edgecolors="white", lw=0.3)
        lo = min(rx.min(), mx.min(), -0.1)
        hi = max(rx.max(), mx.max(), 0.9)
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.4)
        ax.set_xlabel("Ridge R²"); ax.set_ylabel("MLP R²")
        ax.set_title(modes[j], fontsize=12)
        # Stats
        n_mlp_wins = (mx > rx).sum()
        pct = 100 * n_mlp_wins / ok.sum() if ok.sum() > 0 else 0
        t_stat, p_val = sp_stats.ttest_rel(mx, rx) if ok.sum() > 2 else (0, 1)
        ax.text(0.05, 0.95,
                f"MLP > Ridge: {n_mlp_wins}/{ok.sum()} ({pct:.0f}%)\n"
                f"Δ={np.mean(mx-rx):.3f}, p={p_val:.3g}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round", fc="wheat", alpha=0.7))
    fig.suptitle("Ridge vs MLP per-worm R² (each dot = one worm)",
                 fontsize=13)
    fig.tight_layout()
    p = out_dir / "scatter_ridge_mlp.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Scatter → {p}")


def _plot_bar(results, out_dir, K):
    modes = [f"a{j+1}" for j in range(K)]
    x = np.arange(K)
    n = len(MODELS)
    w = 0.8 / n

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for mi, m in enumerate(MODELS):
        mat = _gather(results, m, K)
        means = np.nanmean(mat, axis=0)
        sems  = np.nanstd(mat, axis=0) / np.sqrt(
            np.isfinite(mat).sum(0).clip(1))
        bars = ax.bar(x + mi * w, means, w, yerr=sems, capsize=3,
                      label=m, color=COLORS[m], edgecolor="white",
                      linewidth=0.5, alpha=0.85)
        for j in range(K):
            ax.text(x[j] + mi * w, means[j] + sems[j] + 0.015,
                    f"{means[j]:.3f}", ha="center", fontsize=7.5,
                    color=COLORS[m], fontweight="bold")
    ax.set_xticks(x + w * (n - 1) / 2)
    ax.set_xticklabels(modes, fontsize=12)
    ax.set_ylabel("R²", fontsize=12)
    nw = len(results)
    ax.set_title(f"Mean R² ± SEM across {nw} worms — 5-fold CV",
                 fontsize=13)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_ylim(-0.1, 0.85)
    ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.3)
    fig.tight_layout()
    p = out_dir / "bar_summary.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Bar → {p}")


def _plot_violin(results, out_dir, K):
    modes = [f"a{j+1}" for j in range(K)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for j, ax in enumerate(axes.flat):
        if j >= K:
            ax.set_visible(False); continue
        data = []
        for m in MODELS:
            vals = _gather(results, m, K)[:, j]
            data.append(vals[np.isfinite(vals)])
        vp = ax.violinplot(data, positions=range(len(MODELS)),
                           showmeans=True, showmedians=True)
        for bi, body in enumerate(vp["bodies"]):
            body.set_facecolor(COLORS[MODELS[bi]])
            body.set_alpha(0.5)
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels(MODELS, fontsize=10)
        ax.set_title(modes[j], fontsize=12)
        ax.set_ylabel("R²")
        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.3)
        # Annotate means
        for mi, m in enumerate(MODELS):
            vals = data[mi]
            if len(vals) > 0:
                ax.text(mi, np.mean(vals) + 0.04, f"{np.mean(vals):.3f}",
                        ha="center", fontsize=8, fontweight="bold",
                        color=COLORS[m])
    fig.suptitle("R² distribution across worms", fontsize=13)
    fig.tight_layout()
    p = out_dir / "violin.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Violin → {p}")


def _plot_overall_summary(results, out_dir, K):
    """Single summary figure: overall mean R², paired difference, per-worm ranking."""
    nw = len(results)
    ridge = _gather(results, "Ridge", K)
    mlp   = _gather(results, "MLP", K)

    # Per-worm mean across modes
    ridge_mn = np.nanmean(ridge, axis=1)
    mlp_mn   = np.nanmean(mlp, axis=1)
    diff = mlp_mn - ridge_mn
    ok = np.isfinite(diff)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: histogram of MLP − Ridge (per-worm mean)
    ax = axes[0]
    ax.hist(diff[ok], bins=20, color="#ff7f0e", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="k", lw=1, ls="--")
    ax.axvline(np.nanmean(diff[ok]), color="red", lw=2, ls="-",
               label=f"mean Δ = {np.nanmean(diff[ok]):+.3f}")
    ax.set_xlabel("MLP − Ridge (mean R² per worm)")
    ax.set_ylabel("# worms")
    ax.set_title("Per-worm Δ(MLP − Ridge)")
    ax.legend(fontsize=9)
    t_stat, p_val = sp_stats.ttest_rel(mlp_mn[ok], ridge_mn[ok])
    ax.text(0.05, 0.85, f"paired t={t_stat:.2f}\np={p_val:.3g}\n"
            f"MLP wins {(diff[ok]>0).sum()}/{ok.sum()}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", fc="wheat", alpha=0.7))

    # Panel 2: sorted per-worm mean R² for both models
    ax = axes[1]
    order = np.argsort(ridge_mn)[::-1]
    worms = sorted(results.keys())
    ax.plot(ridge_mn[order], "o-", ms=4, color=COLORS["Ridge"],
            label="Ridge", alpha=0.8)
    ax.plot(mlp_mn[order], "s-", ms=4, color=COLORS["MLP"],
            label="MLP", alpha=0.8)
    ax.set_xlabel("Worm rank (by Ridge)")
    ax.set_ylabel("Mean R² (6 modes)")
    ax.set_title("Per-worm mean R² (ranked)")
    ax.legend(fontsize=9)
    ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.3)

    # Panel 3: per-mode Δ box plot
    ax = axes[2]
    mode_deltas = []
    modes = [f"a{j+1}" for j in range(K)]
    for j in range(K):
        d = mlp[:, j] - ridge[:, j]
        mode_deltas.append(d[np.isfinite(d)])
    bp = ax.boxplot(mode_deltas, labels=modes, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#ff7f0e")
        patch.set_alpha(0.5)
    ax.axhline(0, color="k", lw=1, ls="--")
    ax.set_ylabel("MLP − Ridge  R²")
    ax.set_title("Per-mode improvement (MLP over Ridge)")
    for j in range(K):
        mn = np.nanmean(mode_deltas[j])
        ax.text(j + 1, mn + 0.01, f"{mn:+.3f}", ha="center", fontsize=8,
                color="red", fontweight="bold")

    fig.suptitle(f"MLP vs Ridge — {nw} worms", fontsize=14)
    fig.tight_layout()
    p = out_dir / "overall_summary.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Overall → {p}")


# ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5_dir", required=True)
    ap.add_argument("--neural_lags", type=int, default=8)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--n_modes", type=int, default=6)
    ap.add_argument("--out_dir",
                    default="output_plots/behaviour_decoder/batch_ridge_mlp")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    h5_files = sorted(glob.glob(str(Path(args.h5_dir) / "*.h5")))
    if not h5_files:
        print(f"  No .h5 files in {args.h5_dir}"); sys.exit(1)
    print(f"\n  Found {len(h5_files)} worms in {args.h5_dir}\n")

    json_path = out_dir / "results.json"
    if args.resume and json_path.exists():
        with open(json_path) as f:
            all_results = json.load(f)
        print(f"  Resuming: {len(all_results)} worms already done\n")
    else:
        all_results = {}

    t0 = time.time()
    for wi, h5 in enumerate(h5_files):
        wid = Path(h5).stem
        if wid in all_results:
            print(f"  [{wi+1}/{len(h5_files)}] {wid}  SKIP")
            continue
        print(f"  [{wi+1}/{len(h5_files)}] {wid}")
        try:
            res = _run_one_worm(h5, n_lags=args.neural_lags,
                                n_folds=args.n_folds, K=args.n_modes)
            all_results[wid] = res
            for m in MODELS:
                vals = res[m]
                print(f"    {m:<7s} "
                      + " ".join(f"{v:6.3f}" for v in vals)
                      + f"  mn={np.mean(vals):.3f}")
        except Exception as e:
            print(f"    ✗ {e}")
            traceback.print_exc()
            all_results[wid] = {"error": str(e)}
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.0f}s ({elapsed/60:.1f} min)\n")

    plot_res = {k: v for k, v in all_results.items() if "error" not in v}
    if len(plot_res) < 3:
        print("  Too few worms for plots."); return

    K = args.n_modes
    _plot_heatmap(plot_res, out_dir, K)
    _plot_scatter_paired(plot_res, out_dir, K)
    _plot_bar(plot_res, out_dir, K)
    _plot_violin(plot_res, out_dir, K)
    _plot_overall_summary(plot_res, out_dir, K)

    # Text summary
    worms = sorted(plot_res.keys())
    modes = [f"a{j+1}" for j in range(K)]
    txt = out_dir / "summary.txt"
    with open(txt, "w") as f:
        f.write(f"Batch Ridge vs MLP — {len(worms)} worms, 5-fold CV\n")
        f.write(f"MLP: 2×128, LayerNorm, ReLU, Dropout 0.1, wd=1e-3\n\n")
        f.write(f"{'Worm':<20s}")
        for m in MODELS:
            for mn in modes:
                f.write(f" {m}_{mn:>3s}")
        f.write("\n")
        for w in worms:
            f.write(f"{w:<20s}")
            for m in MODELS:
                for v in plot_res[w][m][:K]:
                    f.write(f" {v:7.3f}")
            f.write("\n")
        f.write(f"\n{'MEAN':<20s}")
        for m in MODELS:
            mat = _gather(plot_res, m, K)
            for v in np.nanmean(mat, 0):
                f.write(f" {v:7.3f}")
        f.write(f"\n{'MEDIAN':<20s}")
        for m in MODELS:
            mat = _gather(plot_res, m, K)
            for v in np.nanmedian(mat, 0):
                f.write(f" {v:7.3f}")
        f.write("\n")
    print(f"  Summary → {txt}")
    print("\n  All done.\n")


if __name__ == "__main__":
    main()
