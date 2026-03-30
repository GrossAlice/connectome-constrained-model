#!/usr/bin/env python
"""Batch Ridge + MLP + AR(2)+MLP(E2E+L2) over worms, with statistics.

Models:
  • Ridge     — per-timestep, per-mode α via inner CV
  • MLP       — per-timestep, 2×128, wd=1e-3, inner CV epoch selection
  • AR2+MLP   — free-run VAR(2)+MLP, E2E BPTT with L2, 200 epochs

The AR(2)+MLP model is trained end-to-end via truncated BPTT:
  b̂_t  =  M₁ b̂_{t-1}  +  M₂ b̂_{t-2}  +  MLP(n_t)  +  c
At test time it's seeded with 2 ground-truth frames, then rolls out
autoregressively using its own predictions.

Usage:
    python -m scripts.batch_ar2_mlp \
        --h5_dir "data/used/behaviour+neuronal activity atanas (2023)/2" \
        --max_worms 10 --resume
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
    _ridge_fit, _train_mlp, _predict_mlp, _train_e2e,
)

MODELS = ["Ridge", "MLP", "AR2+MLP", "AR2d+MLP"]
COLORS = {"Ridge": "#1f77b4", "MLP": "#ff7f0e",
          "AR2+MLP": "#2ca02c", "AR2d+MLP": "#d62728"}


# ──────────────────────────────────────────────────────────────────────
def _run_one_worm(h5_path, *, n_lags=8, n_folds=5, K=6,
                  e2e_epochs=200, tbptt_chunk=64) -> dict:
    """Run all 3 models on one worm with 5-fold temporal CV."""
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

        # ── Ridge (per-timestep) ─────────────────────────────────────
        for j in range(K):
            coef, intc, _ = _ridge_fit(X_tr_s, b_tr[:, j],
                                       ridge_grid, n_inner)
            ho["Ridge"][ts:te, j] = X_te_s @ coef + intc

        # ── MLP (per-timestep, 2×128, inner CV) ─────────────────────
        mlp = _train_mlp(X_tr_s, b_tr, K)
        ho["MLP"][ts:te] = _predict_mlp(mlp, X_te_s)

        # ── AR(2)+MLP  (E2E+L2, free-run, raw inputs) ─────────────
        # Training segments: everything except the test fold
        segs = []
        if ts > warmup + 2:
            segs.append((warmup, ts))
        if te + 2 < T:
            segs.append((te, T))

        M1e, M2e, ce_np, drv_np = _train_e2e(
            d_in, K, segs, b, X,
            e2e_epochs, tbptt_chunk,
            weight_decay=1e-3,
            tag=f"E2E f{fi+1}")

        # Free-run evaluation on test fold
        p1, p2 = b[ts - 1].copy(), b[ts - 2].copy()
        for t in range(ts, te):
            p_new = M1e @ p1 + M2e @ p2 + drv_np[t] + ce_np
            ho["AR2+MLP"][t] = p_new
            p2, p1 = p1, p_new

        # ── AR2d+MLP  (diagonal AR, free-run) ────────────────────
        M1d, M2d, cd_np, drvd_np = _train_e2e(
            d_in, K, segs, b, X,
            e2e_epochs, tbptt_chunk,
            weight_decay=1e-3,
            diagonal_ar=True, max_rho=0.98,
            tag=f"E2Ed f{fi+1}")

        p1, p2 = b[ts - 1].copy(), b[ts - 2].copy()
        for t in range(ts, te):
            p_new = M1d @ p1 + M2d @ p2 + drvd_np[t] + cd_np
            ho["AR2d+MLP"][t] = p_new
            p2, p1 = p1, p_new

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
    nm = len(MODELS)

    fig, axes = plt.subplots(1, nm, figsize=(6*nm, max(8, nw * 0.35)),
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
    fig.colorbar(im, ax=axes.tolist(), shrink=0.5, label="R²")
    fig.suptitle(f"Per-worm R² — {nw} worms, 5-fold CV",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    p = out_dir / "heatmap.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Heatmap → {p}")


def _plot_scatter_paired(results, out_dir, K):
    """Scatter Ridge-R² vs AR2+MLP-R² per mode (and vs MLP)."""
    modes = [f"a{j+1}" for j in range(K)]
    ridge = _gather(results, "Ridge", K)
    mlp   = _gather(results, "MLP", K)
    ar2   = _gather(results, "AR2+MLP", K)

    fig, axes = plt.subplots(2, K, figsize=(5*K, 9))
    # Row 0: Ridge vs AR2+MLP
    for j in range(K):
        ax = axes[0, j]
        ok = np.isfinite(ridge[:, j]) & np.isfinite(ar2[:, j])
        rx, ax2 = ridge[ok, j], ar2[ok, j]
        ax.scatter(rx, ax2, s=30, alpha=0.7, c=COLORS["AR2+MLP"],
                   edgecolors="white", lw=0.3)
        lo = min(rx.min(), ax2.min(), -0.1)
        hi = max(rx.max(), ax2.max(), 0.9)
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.4)
        ax.set_xlabel("Ridge R²"); ax.set_ylabel("AR2+MLP R²")
        ax.set_title(modes[j], fontsize=12)
        n_wins = (ax2 > rx).sum()
        pct = 100 * n_wins / ok.sum() if ok.sum() > 0 else 0
        t_s, p_v = (sp_stats.ttest_rel(ax2, rx) if ok.sum() > 2
                    else (0, 1))
        ax.text(0.05, 0.95,
                f"AR2 > Ridge: {n_wins}/{ok.sum()} ({pct:.0f}%)\n"
                f"Δ={np.mean(ax2-rx):.3f}, p={p_v:.3g}",
                transform=ax.transAxes, fontsize=7, va="top",
                bbox=dict(boxstyle="round", fc="lightgreen", alpha=0.5))

    # Row 1: MLP vs AR2+MLP
    for j in range(K):
        ax = axes[1, j]
        ok = np.isfinite(mlp[:, j]) & np.isfinite(ar2[:, j])
        mx, ax2 = mlp[ok, j], ar2[ok, j]
        ax.scatter(mx, ax2, s=30, alpha=0.7, c=COLORS["AR2+MLP"],
                   edgecolors="white", lw=0.3)
        lo = min(mx.min(), ax2.min(), -0.1)
        hi = max(mx.max(), ax2.max(), 0.9)
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.4)
        ax.set_xlabel("MLP R²"); ax.set_ylabel("AR2+MLP R²")
        ax.set_title(modes[j], fontsize=12)
        n_wins = (ax2 > mx).sum()
        pct = 100 * n_wins / ok.sum() if ok.sum() > 0 else 0
        t_s, p_v = (sp_stats.ttest_rel(ax2, mx) if ok.sum() > 2
                    else (0, 1))
        ax.text(0.05, 0.95,
                f"AR2 > MLP: {n_wins}/{ok.sum()} ({pct:.0f}%)\n"
                f"Δ={np.mean(ax2-mx):.3f}, p={p_v:.3g}",
                transform=ax.transAxes, fontsize=7, va="top",
                bbox=dict(boxstyle="round", fc="lightgreen", alpha=0.5))

    fig.suptitle("Paired scatter (each dot = one worm)", fontsize=13)
    fig.tight_layout()
    p = out_dir / "scatter.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Scatter → {p}")


def _plot_bar(results, out_dir, K):
    modes = [f"a{j+1}" for j in range(K)]
    x = np.arange(K)
    n = len(MODELS)
    w = 0.8 / n

    fig, ax = plt.subplots(figsize=(12, 5.5))
    for mi, m in enumerate(MODELS):
        mat = _gather(results, m, K)
        means = np.nanmean(mat, axis=0)
        sems  = np.nanstd(mat, axis=0) / np.sqrt(
            np.isfinite(mat).sum(0).clip(1))
        ax.bar(x + mi * w, means, w, yerr=sems, capsize=3,
               label=m, color=COLORS[m], edgecolor="white",
               linewidth=0.5, alpha=0.85)
        for j in range(K):
            ax.text(x[j] + mi * w, means[j] + sems[j] + 0.015,
                    f"{means[j]:.3f}", ha="center", fontsize=7,
                    color=COLORS[m], fontweight="bold")
    ax.set_xticks(x + w * (n - 1) / 2)
    ax.set_xticklabels(modes, fontsize=12)
    ax.set_ylabel("R²", fontsize=12)
    nw = len(results)
    ax.set_title(f"Mean R² ± SEM — {nw} worms, 5-fold CV\n"
                 f"(Ridge/MLP = per-timestep, AR2+MLP = free-run)",
                 fontsize=12)
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
        ax.set_xticklabels(MODELS, fontsize=9)
        ax.set_title(modes[j], fontsize=12)
        ax.set_ylabel("R²")
        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.3)
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
    """Summary: per-model comparison across worms."""
    nw = len(results)
    ridge = _gather(results, "Ridge", K)
    mlp   = _gather(results, "MLP", K)
    ar2   = _gather(results, "AR2+MLP", K)

    # Per-worm mean across modes
    ridge_mn = np.nanmean(ridge, axis=1)
    mlp_mn   = np.nanmean(mlp, axis=1)
    ar2_mn   = np.nanmean(ar2, axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    # Panel 1: histogram of AR2+MLP − Ridge (per-worm mean)
    ax = axes[0]
    diff = ar2_mn - ridge_mn
    ok = np.isfinite(diff)
    ax.hist(diff[ok], bins=20, color=COLORS["AR2+MLP"], alpha=0.7,
            edgecolor="white")
    ax.axvline(0, color="k", lw=1, ls="--")
    ax.axvline(np.nanmean(diff[ok]), color="red", lw=2, ls="-",
               label=f"mean Δ = {np.nanmean(diff[ok]):+.3f}")
    ax.set_xlabel("AR2+MLP − Ridge (mean R² per worm)")
    ax.set_ylabel("# worms")
    ax.set_title("Per-worm Δ(AR2+MLP − Ridge)")
    ax.legend(fontsize=9)
    if ok.sum() > 2:
        t_stat, p_val = sp_stats.ttest_rel(ar2_mn[ok], ridge_mn[ok])
        ax.text(0.05, 0.85,
                f"paired t={t_stat:.2f}\np={p_val:.3g}\n"
                f"AR2 wins {(diff[ok]>0).sum()}/{ok.sum()}",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round", fc="lightgreen", alpha=0.7))

    # Panel 2: sorted per-worm mean R² for all models
    ax = axes[1]
    order = np.argsort(ridge_mn)[::-1]
    ax.plot(ridge_mn[order], "o-", ms=4, color=COLORS["Ridge"],
            label="Ridge", alpha=0.8)
    ax.plot(mlp_mn[order], "s-", ms=4, color=COLORS["MLP"],
            label="MLP", alpha=0.8)
    ax.plot(ar2_mn[order], "^-", ms=4, color=COLORS["AR2+MLP"],
            label="AR2+MLP", alpha=0.8)
    ax.set_xlabel("Worm rank (by Ridge)")
    ax.set_ylabel("Mean R² (6 modes)")
    ax.set_title("Per-worm mean R² (ranked)")
    ax.legend(fontsize=9)
    ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.3)

    # Panel 3: per-mode Δ box plot (AR2+MLP − Ridge)
    ax = axes[2]
    mode_deltas = []
    modes = [f"a{j+1}" for j in range(K)]
    for j in range(K):
        d = ar2[:, j] - ridge[:, j]
        mode_deltas.append(d[np.isfinite(d)])
    bp = ax.boxplot(mode_deltas, labels=modes, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor(COLORS["AR2+MLP"])
        patch.set_alpha(0.5)
    ax.axhline(0, color="k", lw=1, ls="--")
    ax.set_ylabel("AR2+MLP − Ridge  R²")
    ax.set_title("Per-mode Δ (AR2+MLP over Ridge)")
    for j in range(K):
        mn = np.nanmean(mode_deltas[j])
        ax.text(j + 1, mn + 0.01, f"{mn:+.3f}", ha="center", fontsize=8,
                color="red", fontweight="bold")

    fig.suptitle(f"AR2+MLP vs Ridge vs MLP — {nw} worms", fontsize=14)
    fig.tight_layout()
    p = out_dir / "overall_summary.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Overall → {p}")


def _print_stats(results, K):
    """Print pairwise statistics table."""
    modes = [f"a{j+1}" for j in range(K)]
    ridge = _gather(results, "Ridge", K)
    mlp   = _gather(results, "MLP", K)
    ar2   = _gather(results, "AR2+MLP", K)
    nw = len(results)

    print(f"\n{'═'*80}")
    print(f"  SUMMARY — {nw} worms, 5-fold CV")
    print(f"{'═'*80}")

    # Per-model mean
    for m in MODELS:
        mat = _gather(results, m, K)
        mn = np.nanmean(mat, 0)
        print(f"  {m:<10s}  " + "  ".join(f"a{j+1}={mn[j]:.3f}"
              for j in range(K))
              + f"  overall={np.nanmean(mn):.3f}")

    # Pairwise comparisons
    pairs = [("AR2+MLP", ar2, "Ridge", ridge),
             ("AR2+MLP", ar2, "MLP", mlp),
             ("MLP", mlp, "Ridge", ridge)]
    print()
    for name_a, mat_a, name_b, mat_b in pairs:
        mn_a = np.nanmean(mat_a, axis=1)
        mn_b = np.nanmean(mat_b, axis=1)
        diff = mn_a - mn_b
        ok = np.isfinite(diff)
        if ok.sum() > 2:
            t_s, p_v = sp_stats.ttest_rel(mn_a[ok], mn_b[ok])
        else:
            t_s, p_v = 0.0, 1.0
        wins = (diff[ok] > 0).sum()
        print(f"  {name_a} vs {name_b}: Δ={np.nanmean(diff[ok]):+.3f}, "
              f"wins {wins}/{ok.sum()}, "
              f"t={t_s:.2f}, p={p_v:.4f}")

    # Per-mode for AR2+MLP vs Ridge
    print(f"\n  Per-mode AR2+MLP vs Ridge:")
    for j in range(K):
        d = ar2[:, j] - ridge[:, j]
        ok = np.isfinite(d)
        if ok.sum() > 2:
            t_s, p_v = sp_stats.ttest_rel(ar2[ok, j], ridge[ok, j])
        else:
            t_s, p_v = 0.0, 1.0
        wins = (d[ok] > 0).sum()
        print(f"    {modes[j]}: Δ={np.nanmean(d[ok]):+.3f}, "
              f"wins {wins}/{ok.sum()}, p={p_v:.4f}")
    print()


# ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5_dir", required=True)
    ap.add_argument("--neural_lags", type=int, default=8)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--n_modes", type=int, default=6)
    ap.add_argument("--e2e_epochs", type=int, default=200)
    ap.add_argument("--tbptt_chunk", type=int, default=64)
    ap.add_argument("--max_worms", type=int, default=0,
                    help="Max worms to process (0 = all)")
    ap.add_argument("--worms", nargs="*", default=None,
                    help="Specific worm IDs (stems) to include")
    ap.add_argument("--out_dir",
                    default="output_plots/behaviour_decoder/batch_ar2_mlp")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    h5_files = sorted(glob.glob(str(Path(args.h5_dir) / "*.h5")))
    if not h5_files:
        print(f"  No .h5 files in {args.h5_dir}"); sys.exit(1)
    if args.worms:
        wanted = set(args.worms)
        h5_files = [f for f in h5_files if Path(f).stem in wanted]
    if args.max_worms > 0:
        h5_files = h5_files[:args.max_worms]
    print(f"\n  Processing {len(h5_files)} worms from {args.h5_dir}")
    print(f"  E2E epochs: {args.e2e_epochs}, TBPTT chunk: {args.tbptt_chunk}\n")

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
        wt0 = time.time()
        try:
            res = _run_one_worm(h5, n_lags=args.neural_lags,
                                n_folds=args.n_folds, K=args.n_modes,
                                e2e_epochs=args.e2e_epochs,
                                tbptt_chunk=args.tbptt_chunk)
            wt = time.time() - wt0
            all_results[wid] = res
            for m in MODELS:
                vals = res[m]
                print(f"    {m:<10s} "
                      + " ".join(f"{v:6.3f}" for v in vals)
                      + f"  mn={np.mean(vals):.3f}")
            print(f"    ({wt:.0f}s)")
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
    _print_stats(plot_res, K)
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
        f.write(f"Batch Ridge + MLP + AR2+MLP + AR2d+MLP — {len(worms)} worms, "
                f"5-fold CV\n")
        f.write(f"AR2+MLP:  full AR + E2E+L2 BPTT, {args.e2e_epochs} epochs, "
                f"wd=1e-3, free-run eval, standardised inputs\n")
        f.write(f"AR2d+MLP: diagonal AR + spectral clamp ≤0.98, same\n")
        f.write(f"MLP: 2×128, LayerNorm, ReLU, Dropout 0.1, wd=1e-3\n\n")
        hdr = f"{'Worm':<20s}"
        for m in MODELS:
            for mn in modes:
                hdr += f" {m[:5]}_{mn:>3s}"
        f.write(hdr + "\n")
        for w in worms:
            line = f"{w:<20s}"
            for m in MODELS:
                for v in plot_res[w][m][:K]:
                    line += f" {v:9.3f}"
            f.write(line + "\n")
        f.write(f"\n{'MEAN':<20s}")
        for m in MODELS:
            mat = _gather(plot_res, m, K)
            for v in np.nanmean(mat, 0):
                f.write(f" {v:9.3f}")
        f.write(f"\n{'MEDIAN':<20s}")
        for m in MODELS:
            mat = _gather(plot_res, m, K)
            for v in np.nanmedian(mat, 0):
                f.write(f" {v:9.3f}")
        f.write("\n")
    print(f"  Summary → {txt}")
    print("\n  All done.\n")


if __name__ == "__main__":
    main()
