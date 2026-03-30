#!/usr/bin/env python
r"""Batch 4-model decoder benchmark on all worms.

Models (5-fold temporal CV, held-out predictions):
  1. Ridge          per-mode α via inner CV
  2. MLP            2×128, wd=1e-3, inner-CV epoch selection
  3. MLP→Ridge      MLP backbone → Ridge-CV readout
  4. AR2+MLP        E2E BPTT, full AR matrix, wd=1e-3

Per worm:
  • predictions.npz  — hold-out predictions (T, K) per model
  • video.mp4        — 5-panel worm animation (GT + 4 models)
  • traces.png       — eigenworm time-series overlay

Summary (across all worms):
  • results.json     — per-worm, per-mode R² and Pearson r
  • heatmap_r2.png   — worms × modes heatmap per model
  • bar_summary.png  — grouped bar chart (mean ± SEM) per mode
  • scatter_r2.png   — strip plot: per-worm mean R² per model
  • summary.txt      — text table

Usage:
    python -m scripts.batch_4model_all_worms \
        --h5_dir "data/used/behaviour+neuronal activity atanas (2023)/2" \
        [--resume] [--no_video] [--max_frames 600]
"""
from __future__ import annotations

import argparse, glob, json, sys, time, traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.unified_benchmark import (
    _ridge_fit, _train_mlp, _predict_mlp, _train_e2e,
    _log_ridge_grid,
)
from scripts.variance_and_mlpridge_comparison import (
    _train_mlp_ridge, _predict_mlp_ridge,
)
from scripts.benchmark_ar_decoder_v2 import (
    load_data, build_lagged_features_np, r2_score,
)
from stage2.posture_videos import angles_to_xy, _load_eigenvectors
from stage1.add_stephens_eigenworms import _preprocess_worm, _ensure_TN
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.stats import pearsonr

# ══════════════════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════════════════
MODELS = ["Ridge", "MLP", "MLP→Ridge", "AR2+MLP"]
COLORS = {
    "Ridge":      "#1b9e77",
    "MLP":        "#d95f02",
    "MLP→Ridge":  "#e7298a",
    "AR2+MLP":    "#7570b3",
}

K = 6
N_LAGS = 8
N_FOLDS = 5
E2E_EPOCHS = 200
TBPTT = 64


# ══════════════════════════════════════════════════════════════════
#  Single-worm: train all 4 models, return hold-out predictions
# ══════════════════════════════════════════════════════════════════
def _run_one_worm(h5_path: str, *, n_lags=N_LAGS, n_folds=N_FOLDS,
                  n_modes=K, e2e_epochs=E2E_EPOCHS, tbptt=TBPTT):
    """Train Ridge, MLP, MLP→Ridge, AR2+MLP with temporal CV.

    Returns (ho_preds, b, dt, u_ncols):
        ho_preds  dict  {model_name: (T, K) np.ndarray}
        b         (T, K) ground truth eigenworm coefficients
        dt        float  sampling interval
        u_ncols   int    number of motor neurons
    """
    u, b_full, dt = load_data(h5_path, all_neurons=False)
    n_modes = min(n_modes, b_full.shape[1])
    b = b_full[:, :n_modes]
    T = b.shape[0]
    warmup = max(2, n_lags)
    X = build_lagged_features_np(u, n_lags)
    d_in = X.shape[1]

    ridge_grid = _log_ridge_grid(-3.0, 10.0, 50)
    n_inner = n_folds - 1

    # folds (contiguous temporal blocks)
    valid_len = T - warmup
    fold_size = valid_len // n_folds
    folds = []
    for i in range(n_folds):
        s = warmup + i * fold_size
        e = warmup + (i + 1) * fold_size if i < n_folds - 1 else T
        folds.append((s, e))

    ho = {m: np.full((T, n_modes), np.nan) for m in MODELS}

    for fi, (ts, te) in enumerate(folds):
        tr_idx = np.concatenate([np.arange(warmup, ts), np.arange(te, T)])
        X_tr, X_te = X[tr_idx], X[ts:te]
        b_tr = b[tr_idx]

        # standardise
        mu, sig = X_tr.mean(0), X_tr.std(0) + 1e-8
        X_tr_s = (X_tr - mu) / sig
        X_te_s = (X_te - mu) / sig

        # ── Ridge ────────────────────────────────────────────────
        for j in range(n_modes):
            coef, intc, _ = _ridge_fit(X_tr_s, b_tr[:, j],
                                       ridge_grid, n_inner)
            ho["Ridge"][ts:te, j] = X_te_s @ coef + intc

        # ── MLP ──────────────────────────────────────────────────
        mlp = _train_mlp(X_tr_s, b_tr, n_modes)
        ho["MLP"][ts:te] = _predict_mlp(mlp, X_te_s)

        # ── MLP→Ridge ────────────────────────────────────────────
        _, backbone, r_coefs, r_intc = _train_mlp_ridge(
            X_tr_s, b_tr, n_modes, ridge_grid, n_inner)
        ho["MLP→Ridge"][ts:te] = _predict_mlp_ridge(
            backbone, r_coefs, r_intc, X_te_s)

        # ── AR2+MLP (E2E BPTT) ──────────────────────────────────
        segs = []
        if ts > warmup + 2:
            segs.append((warmup, ts))
        if te + 2 < T:
            segs.append((te, T))
        M1, M2, c_np, drv_np = _train_e2e(
            d_in, n_modes, segs, b, X,
            e2e_epochs, tbptt,
            weight_decay=1e-3, tag=f"E2E f{fi+1}")
        p1 = b[ts - 1].copy()
        p2 = b[ts - 2].copy()
        for t in range(ts, te):
            p_new = M1 @ p1 + M2 @ p2 + drv_np[t] + c_np
            ho["AR2+MLP"][t] = p_new
            p2, p1 = p1, p_new

        print(f"      fold {fi+1}/{n_folds} done")

    return ho, b, dt, u.shape[1]


# ══════════════════════════════════════════════════════════════════
#  Compute R² and Pearson r per mode
# ══════════════════════════════════════════════════════════════════
def _compute_metrics(ho, b, warmup=N_LAGS):
    """Return dicts  r2[model] = [K floats],  corr[model] = [K floats]."""
    T = b.shape[0]
    K = b.shape[1]
    valid = np.arange(warmup, T)
    r2, corr = {}, {}
    for name in MODELS:
        preds = ho[name]
        ok = np.all(np.isfinite(preds[valid]), axis=1)
        idx = valid[ok]
        if len(idx) < 10:
            r2[name] = [float("nan")] * K
            corr[name] = [float("nan")] * K
        else:
            r2[name] = [float(r2_score(b[idx, j], preds[idx, j]))
                        for j in range(K)]
            corr[name] = [float(pearsonr(b[idx, j], preds[idx, j])[0])
                          for j in range(K)]
    return r2, corr


# ══════════════════════════════════════════════════════════════════
#  Per-worm video (5 panels, no mean angle)
# ══════════════════════════════════════════════════════════════════
def _make_video(h5_path, ho, b, dt, worm_id, out_dir,
                max_frames=600, fps=10):
    """Render 5-panel worm video + traces PNG."""
    K = b.shape[1]
    T = b.shape[0]
    warmup = N_LAGS

    with h5py.File(h5_path, "r") as f:
        _ew_d_target = int(f["behaviour/eigenworms_stephens"].attrs["d_target"])
        _ew_d_w = int(f["behaviour/eigenworms_stephens"].attrs["d_w"])
        _ew_source = f["behaviour/eigenworms_stephens"].attrs["source"]
        ew_raw = np.asarray(f["behaviour/eigenworms_stephens"][:, :K],
                            dtype=float)
        ba_src = _ensure_TN(np.asarray(f[_ew_source][:], dtype=float))

    eigvecs = _load_eigenvectors(h5_path=h5_path, d_target=_ew_d_target)
    E = eigvecs[:, :K]
    d_recon = E.shape[0]

    proc = _preprocess_worm(ba_src, _ew_d_w, _ew_d_target)
    per_frame_mean = proc.mean(axis=1, keepdims=True)

    Tmax = min(T, max_frames)

    # Reconstruct (mean-subtracted)
    recon = {"GT": ew_raw[:Tmax, :K] @ E.T}
    for name in MODELS:
        recon[name] = ho[name][:Tmax, :K] @ E.T
    proc_gt = (proc - per_frame_mean)[:Tmax]

    # R² for labels
    valid = np.arange(warmup, T)
    r2_mean = {}
    for name in MODELS:
        preds = ho[name]
        ok = np.all(np.isfinite(preds[valid]), axis=1)
        idx = valid[ok]
        r2s = [r2_score(b[idx, j], preds[idx, j]) for j in range(K)]
        r2_mean[name] = float(np.mean(r2s))

    # Convert to XY
    panels = ["GT"] + MODELS
    xy = {}
    for p in panels:
        arr = np.zeros((Tmax, d_recon, 2))
        for t in range(Tmax):
            a = recon[p][t]
            if np.all(np.isfinite(a)):
                x, y = angles_to_xy(a)
            else:
                x = np.full(d_recon, np.nan)
                y = np.full(d_recon, np.nan)
            arr[t, :, 0], arr[t, :, 1] = x, y
        xy[p] = arr

    xy_raw = np.zeros((Tmax, d_recon, 2))
    for t in range(Tmax):
        a = proc_gt[t]
        if np.all(np.isfinite(a)):
            x, y = angles_to_xy(a)
        else:
            x = np.full(d_recon, np.nan)
            y = np.full(d_recon, np.nan)
        xy_raw[t, :, 0], xy_raw[t, :, 1] = x, y

    # Axis limits
    all_c = np.concatenate(
        [xy[p].reshape(-1, 2) for p in panels] + [xy_raw.reshape(-1, 2)])
    m_ok = np.all(np.isfinite(all_c), axis=1)
    if m_ok.sum() < 2:
        print(f"    ⚠ no valid frames for video")
        return
    xmin, xmax = all_c[m_ok, 0].min(), all_c[m_ok, 0].max()
    ymin, ymax = all_c[m_ok, 1].min(), all_c[m_ok, 1].max()
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    span = 0.5 * max(xmax - xmin, ymax - ymin) + 2.0

    # Build figure
    n_panels = 1 + len(MODELS)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 4.5),
                             facecolor="white")
    panel_labels = (
        ["Ground Truth"] +
        [f"{n}\n(R²={r2_mean[n]:.3f})" for n in MODELS])
    panel_colors = ["black"] + [COLORS[n] for n in MODELS]

    lines, heads = [], []
    line_raw = head_raw = None

    for i, (ax, ttl, col) in enumerate(zip(axes, panel_labels,
                                            panel_colors)):
        ax.set_title(ttl, fontsize=10, fontweight="bold",
                     color=col if i > 0 else "black")
        ax.set_xlim(cx - span, cx + span)
        ax.set_ylim(cy - span, cy + span)
        ax.set_aspect("equal")
        ax.set_facecolor("#f7f7f7")
        ax.tick_params(left=False, bottom=False,
                       labelleft=False, labelbottom=False)
        for s in ax.spines.values():
            s.set_visible(False)
        lw = 2.5 if i == 0 else 2.0
        c = "black" if i == 0 else col
        ln, = ax.plot([], [], "-", color=c, lw=lw, alpha=0.9)
        hd, = ax.plot([], [], "o", color="crimson", ms=5, alpha=0.9)
        lines.append(ln); heads.append(hd)
        if i == 0:
            line_raw, = ax.plot([], [], "-", color="gray", lw=1.5, alpha=0.4)
            head_raw, = ax.plot([], [], "o", color="gray", ms=3, alpha=0.4)

    time_text = fig.text(0.5, 0.02, "", ha="center", va="bottom",
                         fontsize=11)

    def _update(frame):
        xr, yr = xy_raw[frame, :, 0], xy_raw[frame, :, 1]
        line_raw.set_data(xr, yr)
        if np.isfinite(xr[0]) and np.isfinite(yr[0]):
            head_raw.set_data([xr[0]], [yr[0]])
        else:
            head_raw.set_data([], [])
        for i, key in enumerate(panels):
            x = xy[key][frame, :, 0]
            y = xy[key][frame, :, 1]
            lines[i].set_data(x, y)
            if np.isfinite(x[0]) and np.isfinite(y[0]):
                heads[i].set_data([x[0]], [y[0]])
            else:
                heads[i].set_data([], [])
        time_text.set_text(f"t = {frame * dt:.1f} s  "
                           f"frame {frame + 1}/{Tmax}")
        return lines + heads + [line_raw, head_raw, time_text]

    anim = FuncAnimation(fig, _update, frames=Tmax,
                         interval=max(1, 1000 // max(1, fps)), blit=False)
    out_mp4 = out_dir / f"{worm_id}.mp4"
    writer = FFMpegWriter(fps=fps,
                          metadata={"title": f"{worm_id} 4-model"},
                          bitrate=2400)
    anim.save(str(out_mp4), writer=writer, dpi=100)
    plt.close(fig)
    print(f"    video → {out_mp4}")

    # Trace PNG
    fig2, axes2 = plt.subplots(K, 1, figsize=(14, 2 * K), sharex=True)
    t_axis = np.arange(Tmax) * dt
    for j in range(K):
        ax = axes2[j]
        ax.plot(t_axis, b[:Tmax, j], "k-", lw=1, alpha=0.8, label="GT")
        for n in MODELS:
            ax.plot(t_axis, ho[n][:Tmax, j], "-", color=COLORS[n],
                    lw=0.8, alpha=0.7, label=n)
        ax.set_ylabel(f"a{j+1}", fontsize=10)
        if j == 0:
            ax.legend(fontsize=7, ncol=5, loc="upper right")
    axes2[-1].set_xlabel("Time (s)", fontsize=10)
    fig2.suptitle(f"Eigenworm predictions — {worm_id}", fontsize=12)
    fig2.tight_layout()
    out_png = out_dir / f"{worm_id}_traces.png"
    fig2.savefig(str(out_png), dpi=120, bbox_inches="tight")
    plt.close(fig2)
    print(f"    traces → {out_png}")


# ══════════════════════════════════════════════════════════════════
#  Summary plots (all worms)
# ══════════════════════════════════════════════════════════════════
def _plot_heatmaps(all_res, out_dir, metric="r2"):
    """Heatmap per model: worms × modes."""
    worm_ids = sorted(all_res.keys())
    n_w = len(worm_ids)
    mode_names = [f"a{j+1}" for j in range(K)]
    label = "R²" if metric == "r2" else "Pearson r"

    fig, axes = plt.subplots(1, len(MODELS),
                             figsize=(5.5 * len(MODELS),
                                      max(8, n_w * 0.35)),
                             sharey=True)
    for ax, mname in zip(axes, MODELS):
        key = f"{mname}_{metric}"
        mat = np.array([all_res[w].get(key, [np.nan]*K)[:K]
                        for w in worm_ids])
        vmin = -0.2 if metric == "r2" else 0.0
        im = ax.imshow(mat, aspect="auto", vmin=vmin, vmax=1.0,
                       cmap="RdYlGn")
        ax.set_xticks(range(K))
        ax.set_xticklabels(mode_names, fontsize=10)
        mn = np.nanmean(mat)
        ax.set_title(f"{mname}\nmean={mn:.3f}", fontsize=11,
                     fontweight="bold", color=COLORS[mname])
        for i in range(n_w):
            for j in range(K):
                v = mat[i, j]
                if np.isfinite(v):
                    color = "white" if v < 0.3 else "black"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=6.5, color=color)
    axes[0].set_yticks(range(n_w))
    axes[0].set_yticklabels(worm_ids, fontsize=7)
    axes[0].set_ylabel("Worm", fontsize=11)
    fig.colorbar(im, ax=list(axes), shrink=0.6, label=label)
    fig.suptitle(f"Per-worm decoder {label} — 5-fold temporal CV",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    p = out_dir / f"heatmap_{metric}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap ({metric}) → {p}")


def _plot_bar_summary(all_res, out_dir, metric="r2"):
    """Grouped bar chart: mean ± SEM across worms per mode."""
    worm_ids = sorted(all_res.keys())
    mode_names = [f"a{j+1}" for j in range(K)]
    label = "R²" if metric == "r2" else "Pearson r"

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(K)
    n = len(MODELS)
    w = 0.8 / n
    for mi, mname in enumerate(MODELS):
        key = f"{mname}_{metric}"
        mat = np.array([all_res[wid].get(key, [np.nan]*K)[:K]
                        for wid in worm_ids])
        means = np.nanmean(mat, axis=0)
        n_ok = np.sum(np.isfinite(mat), axis=0).clip(1)
        sems = np.nanstd(mat, axis=0) / np.sqrt(n_ok)
        ax.bar(x + mi * w, means, w, yerr=sems, capsize=3,
               label=mname, color=COLORS[mname], edgecolor="white",
               linewidth=0.5, alpha=0.85)
        for j in range(K):
            ax.text(x[j] + mi * w, means[j] + sems[j] + 0.02,
                    f"{means[j]:.3f}", ha="center", fontsize=6.5,
                    color=COLORS[mname])
    ax.set_xticks(x + w * (n - 1) / 2)
    ax.set_xticklabels(mode_names, fontsize=12)
    ax.set_ylabel(f"{label}  (held-out)", fontsize=12)
    ax.set_title(f"Mean decoder {label} across {len(worm_ids)} worms "
                 f"(± SEM) — 5-fold CV", fontsize=13)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_ylim(-0.15, 1.05)
    ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.3)
    fig.tight_layout()
    p = out_dir / f"bar_{metric}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Bar ({metric}) → {p}")


def _plot_scatter(all_res, out_dir):
    """Strip plot: per-worm mean R² per model."""
    worm_ids = sorted(all_res.keys())

    fig, ax = plt.subplots(figsize=(7, 5))
    for mi, mname in enumerate(MODELS):
        key = f"{mname}_r2"
        vals = np.array([np.nanmean(all_res[w].get(key, [np.nan]*K)[:K])
                         for w in worm_ids])
        ok = np.isfinite(vals)
        jitter = np.random.default_rng(42).uniform(-0.18, 0.18,
                                                    size=ok.sum())
        ax.scatter(np.full(ok.sum(), mi) + jitter, vals[ok],
                   s=28, alpha=0.55, color=COLORS[mname],
                   edgecolors="white", linewidth=0.3)
        if ok.any():
            mn = np.nanmean(vals[ok])
            ax.plot([mi - 0.3, mi + 0.3], [mn, mn],
                    lw=2.5, color=COLORS[mname])
            ax.text(mi, mn + 0.025, f"{mn:.3f}", ha="center",
                    fontsize=10, fontweight="bold", color=COLORS[mname])
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODELS, fontsize=11)
    ax.set_ylabel("Mean R² (across 6 modes)", fontsize=12)
    ax.set_title(f"Per-worm mean R² — {len(worm_ids)} worms", fontsize=13)
    ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.3)
    ax.set_ylim(-0.15, 0.85)
    fig.tight_layout()
    p = out_dir / "scatter_r2.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Scatter → {p}")


def _plot_scatter_corr(all_res, out_dir):
    """Strip plot: per-worm mean Pearson r per model."""
    worm_ids = sorted(all_res.keys())

    fig, ax = plt.subplots(figsize=(7, 5))
    for mi, mname in enumerate(MODELS):
        key = f"{mname}_corr"
        vals = np.array([np.nanmean(all_res[w].get(key, [np.nan]*K)[:K])
                         for w in worm_ids])
        ok = np.isfinite(vals)
        jitter = np.random.default_rng(42).uniform(-0.18, 0.18,
                                                    size=ok.sum())
        ax.scatter(np.full(ok.sum(), mi) + jitter, vals[ok],
                   s=28, alpha=0.55, color=COLORS[mname],
                   edgecolors="white", linewidth=0.3)
        if ok.any():
            mn = np.nanmean(vals[ok])
            ax.plot([mi - 0.3, mi + 0.3], [mn, mn],
                    lw=2.5, color=COLORS[mname])
            ax.text(mi, mn + 0.025, f"{mn:.3f}", ha="center",
                    fontsize=10, fontweight="bold", color=COLORS[mname])
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODELS, fontsize=11)
    ax.set_ylabel("Mean Pearson r (across 6 modes)", fontsize=12)
    ax.set_title(f"Per-worm mean correlation — {len(worm_ids)} worms",
                 fontsize=13)
    ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.3)
    ax.set_ylim(-0.15, 1.05)
    fig.tight_layout()
    p = out_dir / "scatter_corr.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Scatter corr → {p}")


def _write_summary_txt(all_res, out_dir):
    """Text table."""
    worm_ids = sorted(all_res.keys())
    mode_names = [f"a{j+1}" for j in range(K)]
    txt = out_dir / "summary.txt"
    with open(txt, "w") as f:
        f.write(f"4-model decoder benchmark — {len(worm_ids)} worms, "
                f"5-fold temporal CV\n\n")
        # Header
        hdr = f"{'Worm':<20s}"
        for mname in MODELS:
            hdr += f"  {mname:>10s}"
        f.write(hdr + "\n")
        f.write("-" * len(hdr) + "\n")
        for wid in worm_ids:
            line = f"{wid:<20s}"
            for mname in MODELS:
                key = f"{mname}_r2"
                vals = all_res[wid].get(key, [np.nan]*K)
                mn = float(np.nanmean(vals[:K]))
                line += f"  {mn:10.3f}"
            f.write(line + "\n")
        # Mean row
        f.write("-" * len(hdr) + "\n")
        line = f"{'MEAN':<20s}"
        for mname in MODELS:
            key = f"{mname}_r2"
            mat = np.array([all_res[w].get(key, [np.nan]*K)[:K]
                            for w in worm_ids])
            line += f"  {np.nanmean(mat):10.3f}"
        f.write(line + "\n")

        # Correlation section
        f.write(f"\n\n{'Worm':<20s}")
        for mname in MODELS:
            f.write(f"  {mname + '_r':>10s}")
        f.write("\n" + "-" * (20 + 12 * len(MODELS)) + "\n")
        for wid in worm_ids:
            line = f"{wid:<20s}"
            for mname in MODELS:
                key = f"{mname}_corr"
                vals = all_res[wid].get(key, [np.nan]*K)
                mn = float(np.nanmean(vals[:K]))
                line += f"  {mn:10.3f}"
            f.write(line + "\n")
        f.write("-" * (20 + 12 * len(MODELS)) + "\n")
        line = f"{'MEAN':<20s}"
        for mname in MODELS:
            key = f"{mname}_corr"
            mat = np.array([all_res[w].get(key, [np.nan]*K)[:K]
                            for w in worm_ids])
            line += f"  {np.nanmean(mat):10.3f}"
        f.write(line + "\n")
    print(f"  Summary → {txt}")


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5_dir",
                    default="data/used/behaviour+neuronal activity atanas (2023)/2",
                    help="Dir with worm .h5 files")
    ap.add_argument("--out_dir",
                    default="output_plots/behaviour_decoder/batch_4model")
    ap.add_argument("--resume", action="store_true",
                    help="Skip worms already in results.json")
    ap.add_argument("--no_video", action="store_true",
                    help="Skip video generation (much faster)")
    ap.add_argument("--max_frames", type=int, default=600)
    ap.add_argument("--e2e_epochs", type=int, default=200)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vid_dir = out_dir / "videos"
    vid_dir.mkdir(exist_ok=True)

    h5_files = sorted(glob.glob(str(Path(args.h5_dir) / "*.h5")))
    if not h5_files:
        print(f"No .h5 files in {args.h5_dir}")
        sys.exit(1)
    print(f"\nFound {len(h5_files)} worms in {args.h5_dir}")
    print(f"Models: {', '.join(MODELS)}")
    print(f"E2E epochs: {args.e2e_epochs}, video: {not args.no_video}\n")

    # Resume support
    json_path = out_dir / "results.json"
    if args.resume and json_path.exists():
        with open(json_path) as f:
            all_results = json.load(f)
        print(f"Resuming: {len(all_results)} worms already done\n")
    else:
        all_results = {}

    t0 = time.time()

    for wi, h5_path in enumerate(h5_files):
        worm_id = Path(h5_path).stem
        if worm_id in all_results and args.resume:
            print(f"[{wi+1}/{len(h5_files)}] {worm_id}  — SKIP")
            continue

        print(f"[{wi+1}/{len(h5_files)}] {worm_id}")
        try:
            ho, b, dt, n_neurons = _run_one_worm(
                h5_path, e2e_epochs=args.e2e_epochs)

            # Metrics
            r2, corr = _compute_metrics(ho, b)
            res = {"T": int(b.shape[0]), "N": int(n_neurons),
                   "K": K, "dt": float(dt)}
            for mname in MODELS:
                res[f"{mname}_r2"] = r2[mname]
                res[f"{mname}_corr"] = corr[mname]
            all_results[worm_id] = res

            # Print summary line
            for mname in MODELS:
                vals = r2[mname]
                mn = float(np.nanmean(vals))
                print(f"    {mname:<12s} "
                      f"{' '.join(f'{v:.3f}' for v in vals)}"
                      f"  mn={mn:.3f}")

            # Save predictions
            pred_path = out_dir / f"{worm_id}_predictions.npz"
            np.savez(str(pred_path), b_true=b,
                     **{m: ho[m] for m in MODELS})

            # Video
            if not args.no_video:
                try:
                    _make_video(h5_path, ho, b, dt, worm_id, vid_dir,
                                max_frames=args.max_frames)
                except Exception as ve:
                    print(f"    ⚠ video failed: {ve}")

        except Exception as exc:
            print(f"  ✗ FAILED: {exc}")
            traceback.print_exc()
            all_results[worm_id] = {"error": str(exc)}

        # Checkpoint
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nAll worms done in {elapsed:.0f}s ({elapsed/60:.1f} min)\n")

    # ── Filter errors ──────────────────────────────────────────────
    plot_res = {k: v for k, v in all_results.items() if "error" not in v}
    if len(plot_res) < 2:
        print("Too few successful worms for summary plots.")
        return

    # ── Summary plots ──────────────────────────────────────────────
    _plot_heatmaps(plot_res, out_dir, metric="r2")
    _plot_heatmaps(plot_res, out_dir, metric="corr")
    _plot_bar_summary(plot_res, out_dir, metric="r2")
    _plot_bar_summary(plot_res, out_dir, metric="corr")
    _plot_scatter(plot_res, out_dir)
    _plot_scatter_corr(plot_res, out_dir)
    _write_summary_txt(plot_res, out_dir)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
