#!/usr/bin/env python
"""
Run Ridge, MLP, MLP→Ridge, and AR2+MLP on all worms.
For each worm:
  • 5-fold temporal CV  (all frames used for R² / correlation)
  • Save predictions.npz
  • Save per-worm video  (5-panel: GT + 4 models, 600 frames)
  • Save per-worm traces  (PNG, all frames)
After all worms: summary plots (heatmap, bar, scatter, correlation).

Usage
─────
  python -m scripts.batch_4model_benchmark \
      --h5_dir "data/used/behaviour+neuronal activity atanas (2023)/2" \
      --out_dir output_plots/behaviour_decoder/batch_4model \
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
from matplotlib.animation import FuncAnimation, FFMpegWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.unified_benchmark import (
    _ridge_fit, _train_mlp, _predict_mlp, _train_e2e, _log_ridge_grid,
)
from scripts.benchmark_ar_decoder_v2 import (
    load_data, build_lagged_features_np, r2_score,
)
from scripts.variance_and_mlpridge_comparison import (
    _train_mlp_ridge, _predict_mlp_ridge,
)
from stage2.posture_videos import angles_to_xy, _load_eigenvectors
from stage1.add_stephens_eigenworms import _preprocess_worm, _ensure_TN
import h5py
from scipy.stats import pearsonr
from scipy import signal

# ══════════════════════════════════════════════════════════════════
#  Constants
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
VIDEO_FRAMES = 600
VIDEO_FPS = 10


# ══════════════════════════════════════════════════════════════════
#  Single-worm benchmark
# ══════════════════════════════════════════════════════════════════
def _run_one_worm(h5_path: str, worm_dir: Path, *, make_video: bool = True):
    """Train 4 models, compute R² and correlation, optionally save video."""

    worm_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────
    u, b_full, dt = load_data(h5_path, all_neurons=False)
    Kw = min(K, b_full.shape[1])
    b = b_full[:, :Kw]
    T = b.shape[0]
    warmup = max(2, N_LAGS)
    X = build_lagged_features_np(u, N_LAGS)
    d_in = X.shape[1]

    print(f"    T={T}, N_motor={u.shape[1]}, K={Kw}, d_in={d_in}, dt={dt:.3f}s")

    ridge_grid = _log_ridge_grid(-3.0, 10.0, 50)
    n_inner = N_FOLDS - 1

    # ── Folds ─────────────────────────────────────────────────────
    valid_len = T - warmup
    fold_size = valid_len // N_FOLDS
    folds = []
    for i in range(N_FOLDS):
        s = warmup + i * fold_size
        e = warmup + (i + 1) * fold_size if i < N_FOLDS - 1 else T
        folds.append((s, e))

    ho = {m: np.full((T, Kw), np.nan) for m in MODELS}

    for fi, (ts, te) in enumerate(folds):
        tr_idx = np.concatenate([np.arange(warmup, ts), np.arange(te, T)])
        X_tr, X_te = X[tr_idx], X[ts:te]
        b_tr = b[tr_idx]
        mu, sig = X_tr.mean(0), X_tr.std(0) + 1e-8
        X_tr_s = (X_tr - mu) / sig
        X_te_s = (X_te - mu) / sig

        # ── Ridge ─────────────────────────────────────────────────
        for j in range(Kw):
            coef, intc, _ = _ridge_fit(X_tr_s, b_tr[:, j], ridge_grid, n_inner)
            ho["Ridge"][ts:te, j] = X_te_s @ coef + intc

        # ── MLP ───────────────────────────────────────────────────
        mlp = _train_mlp(X_tr_s, b_tr, Kw)
        ho["MLP"][ts:te] = _predict_mlp(mlp, X_te_s)

        # ── MLP→Ridge ─────────────────────────────────────────────
        _, backbone, r_coefs, r_intc = _train_mlp_ridge(
            X_tr_s, b_tr, Kw, ridge_grid, n_inner)
        ho["MLP→Ridge"][ts:te] = _predict_mlp_ridge(
            backbone, r_coefs, r_intc, X_te_s)

        # ── AR2+MLP (E2E, full matrix, wd=1e-3) ──────────────────
        segs = []
        if ts > warmup + 2:
            segs.append((warmup, ts))
        if te + 2 < T:
            segs.append((te, T))
        M1, M2, c_np, drv_np = _train_e2e(
            d_in, Kw, segs, b, X, E2E_EPOCHS, TBPTT,
            weight_decay=1e-3, tag=f"E2E f{fi+1}")
        p1, p2 = b[ts - 1].copy(), b[ts - 2].copy()
        for t in range(ts, te):
            p_new = M1 @ p1 + M2 @ p2 + drv_np[t] + c_np
            ho["AR2+MLP"][t] = p_new
            p2, p1 = p1, p_new

        print(f"      Fold {fi+1}/{N_FOLDS} done")

    # ── Compute R² and Pearson correlation ────────────────────────
    valid = np.arange(warmup, T)
    results = {"T": int(T), "N": int(u.shape[1]), "K": Kw, "dt": float(dt)}

    for name in MODELS:
        preds = ho[name]
        ok = np.all(np.isfinite(preds[valid]), axis=1)
        idx = valid[ok]
        if idx.size < 10:
            results[f"{name}_r2"] = [float("nan")] * Kw
            results[f"{name}_corr"] = [float("nan")] * Kw
            continue
        r2s = [float(r2_score(b[idx, j], preds[idx, j])) for j in range(Kw)]
        corrs = [float(pearsonr(b[idx, j], preds[idx, j])[0]) for j in range(Kw)]
        results[f"{name}_r2"] = r2s
        results[f"{name}_corr"] = corrs

    # Print summary
    print(f"    {'Model':16s}" + "  ".join(f"{'a'+str(j+1):>7s}" for j in range(Kw)) + "    mean")
    for name in MODELS:
        r2s = results.get(f"{name}_r2", [])
        if r2s:
            mn = float(np.nanmean(r2s))
            print(f"    {name:16s}" + "  ".join(f"{v:7.3f}" for v in r2s) + f"    {mn:.3f}")

    # ── Save predictions ──────────────────────────────────────────
    np.savez(worm_dir / "predictions.npz",
             b_true=b, **{m: ho[m] for m in MODELS})

    # ── Behaviour statistics plot ──────────────────────────────────
    try:
        _render_behaviour_stats(ho, b, Kw, dt,
                                Path(h5_path).stem, worm_dir)
    except Exception as exc:
        print(f"    ⚠ Behaviour stats plot failed: {exc}")

    # ── Video + traces ────────────────────────────────────────────
    if make_video:
        try:
            _render_video(h5_path, ho, b, Kw, dt, results, worm_dir)
        except Exception as exc:
            print(f"    ⚠ Video failed: {exc}")

    return results


# ══════════════════════════════════════════════════════════════════
#  Per-worm video rendering
# ══════════════════════════════════════════════════════════════════
def _render_video(h5_path, ho, b, Kw, dt, results, worm_dir):
    """Render 5-panel video (GT + 4 models), mean-subtracted posture."""
    T = b.shape[0]
    warmup = max(2, N_LAGS)

    with h5py.File(h5_path, "r") as f:
        _d_target = int(f["behaviour/eigenworms_stephens"].attrs["d_target"])
        _d_w = int(f["behaviour/eigenworms_stephens"].attrs["d_w"])
        _ew_source = f["behaviour/eigenworms_stephens"].attrs["source"]
        ew_raw = np.asarray(f["behaviour/eigenworms_stephens"][:, :Kw], dtype=float)
        ba_src = _ensure_TN(np.asarray(f[_ew_source][:], dtype=float))

    E = _load_eigenvectors(h5_path=h5_path, d_target=_d_target)[:, :Kw]
    d_recon = E.shape[0]

    # Reconstruct body angles (mean-subtracted)
    Tmax = min(T, VIDEO_FRAMES)
    recon = {"GT": ew_raw[:Tmax, :Kw] @ E.T}
    for name in MODELS:
        recon[name] = ho[name][:Tmax, :Kw] @ E.T

    # GT raw overlay (mean-subtracted)
    proc = _preprocess_worm(ba_src, _d_w, _d_target)
    pfm = proc.mean(axis=1, keepdims=True)
    proc_gt = (proc - pfm)[:Tmax]

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
                x = np.full(d_recon, np.nan); y = x.copy()
            arr[t, :, 0], arr[t, :, 1] = x, y
        xy[p] = arr

    xy_raw = np.zeros((Tmax, d_recon, 2))
    for t in range(Tmax):
        a = proc_gt[t]
        if np.all(np.isfinite(a)):
            x, y = angles_to_xy(a)
        else:
            x = np.full(d_recon, np.nan); y = x.copy()
        xy_raw[t, :, 0], xy_raw[t, :, 1] = x, y

    # Axis limits
    all_c = np.concatenate([xy[p].reshape(-1, 2) for p in panels] + [xy_raw.reshape(-1, 2)])
    ok = np.all(np.isfinite(all_c), axis=1)
    if ok.sum() < 2:
        print("    ⚠ All NaN coordinates, skipping video")
        return
    xmin, xmax = all_c[ok, 0].min(), all_c[ok, 0].max()
    ymin, ymax = all_c[ok, 1].min(), all_c[ok, 1].max()
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    span = 0.5 * max(xmax - xmin, ymax - ymin) + 2.0

    # R² for panel labels
    valid = np.arange(warmup, T)
    r2_mean = {}
    for name in MODELS:
        r2s = results.get(f"{name}_r2", [])
        r2_mean[name] = float(np.nanmean(r2s)) if r2s else float("nan")

    # Figure
    n_panels = 1 + len(MODELS)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 4.5), facecolor="white")
    labels = ["Ground Truth"] + [f"{n}\n(R²={r2_mean[n]:.3f})" for n in MODELS]
    pcols = ["black"] + [COLORS[n] for n in MODELS]

    lines, heads = [], []
    line_raw = head_raw = None
    for i, (ax, ttl, col) in enumerate(zip(axes, labels, pcols)):
        ax.set_title(ttl, fontsize=10, fontweight="bold", color=col if i > 0 else "black")
        ax.set_xlim(cx - span, cx + span); ax.set_ylim(cy - span, cy + span)
        ax.set_aspect("equal"); ax.set_facecolor("#f7f7f7")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for s in ax.spines.values(): s.set_visible(False)
        lw = 2.5 if i == 0 else 2.0
        c = "black" if i == 0 else col
        ln, = ax.plot([], [], "-", color=c, lw=lw, alpha=0.9)
        hd, = ax.plot([], [], "o", color="crimson", ms=5, alpha=0.9)
        lines.append(ln); heads.append(hd)
        if i == 0:
            line_raw, = ax.plot([], [], "-", color="gray", lw=1.5, alpha=0.4)
            head_raw, = ax.plot([], [], "o", color="gray", ms=3, alpha=0.4)

    time_text = fig.text(0.5, 0.02, "", ha="center", va="bottom", fontsize=11)

    def _update(frame):
        xr, yr = xy_raw[frame, :, 0], xy_raw[frame, :, 1]
        line_raw.set_data(xr, yr)
        if np.isfinite(xr[0]):
            head_raw.set_data([xr[0]], [yr[0]])
        else:
            head_raw.set_data([], [])
        for i, key in enumerate(panels):
            x, y = xy[key][frame, :, 0], xy[key][frame, :, 1]
            lines[i].set_data(x, y)
            if np.isfinite(x[0]):
                heads[i].set_data([x[0]], [y[0]])
            else:
                heads[i].set_data([], [])
        time_text.set_text(f"t = {frame * dt:.1f} s    frame {frame+1}/{Tmax}")
        return lines + heads + [line_raw, head_raw, time_text]

    anim = FuncAnimation(fig, _update, frames=Tmax,
                         interval=max(1, 1000 // max(1, VIDEO_FPS)), blit=False)
    mp4 = worm_dir / "comparison.mp4"
    writer = FFMpegWriter(fps=VIDEO_FPS, metadata={"title": "4-model comparison"},
                          bitrate=2400)
    anim.save(str(mp4), writer=writer, dpi=100)
    plt.close(fig)
    print(f"    Video → {mp4}")

    # ── Trace plot (all frames) ───────────────────────────────────
    fig2, axes2 = plt.subplots(Kw, 1, figsize=(16, 2 * Kw), sharex=True)
    if Kw == 1:
        axes2 = [axes2]
    t_ax = np.arange(T) * dt
    for j in range(Kw):
        ax = axes2[j]
        ax.plot(t_ax, b[:, j], "k-", lw=0.8, alpha=0.8, label="GT")
        for n in MODELS:
            ax.plot(t_ax, ho[n][:, j], "-", color=COLORS[n], lw=0.6, alpha=0.7, label=n)
        ax.set_ylabel(f"a{j+1}", fontsize=10)
        if j == 0:
            ax.legend(fontsize=7, ncol=5, loc="upper right")
    axes2[-1].set_xlabel("Time (s)")
    fig2.suptitle(Path(worm_dir).name, fontsize=12)
    fig2.tight_layout()
    fig2.savefig(worm_dir / "traces.png", dpi=120, bbox_inches="tight")
    plt.close(fig2)


# ══════════════════════════════════════════════════════════════════
#  Behaviour statistics helpers
# ══════════════════════════════════════════════════════════════════
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


def _render_behaviour_stats(ho, b, Kw, dt, worm_id, worm_dir):
    """Per-worm behaviour statistics: amplitude distribution, ACF, PSD.

    Generates a K×3 grid matching the style:
      Row = eigenworm mode (EW1..EWK)
      Col = [Amplitude Distribution | Autocorrelation | Power Spectrum]
    All 4 models overlaid on each panel together with GT.
    """
    warmup = max(2, N_LAGS)
    T = b.shape[0]
    fs = 1.0 / dt
    max_lag = min(200, T // 2)
    mode_names = [f"EW{j+1}" for j in range(Kw)]

    fig, axes = plt.subplots(Kw, 3, figsize=(18, 3.2 * Kw))
    if Kw == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(f"Behaviour Statistics: GT vs Decoders — {worm_id}",
                 fontsize=14, fontweight="bold", y=1.01)

    for j in range(Kw):
        ax_amp, ax_acf, ax_psd = axes[j, 0], axes[j, 1], axes[j, 2]

        # ── Ground truth ──────────────────────────────────────────
        gt = b[warmup:, j]
        acf_gt = _acf(gt, max_lag)
        f_gt, p_gt = _psd(gt, fs)

        bins = np.linspace(np.percentile(gt, 0.5),
                           np.percentile(gt, 99.5), 50)
        ax_amp.hist(gt, bins=bins, density=True, alpha=0.45,
                    color="gray", label="GT", edgecolor="none")
        lags_s = np.arange(len(acf_gt)) * dt
        ax_acf.plot(lags_s, acf_gt, "k-", lw=1.5, label="GT")
        if len(f_gt) > 0:
            ax_psd.semilogy(f_gt, p_gt, "k-", lw=1.5, label="GT")

        # ── Models ────────────────────────────────────────────────
        var_ratios = []
        for mname in MODELS:
            preds = ho[mname][warmup:, j]
            ok = np.isfinite(preds)
            if ok.sum() < 50:
                continue
            p_clean = preds[ok]

            # Variance ratio
            vr = np.std(p_clean) / (np.std(gt) + 1e-15)
            var_ratios.append((mname, vr))

            ax_amp.hist(p_clean, bins=bins, density=True, alpha=0.35,
                        color=COLORS[mname], label=mname, edgecolor="none")

            acf_m = _acf(p_clean, max_lag)
            lags_m = np.arange(len(acf_m)) * dt
            ax_acf.plot(lags_m, acf_m, color=COLORS[mname], lw=1.2,
                        label=mname)

            f_m, p_m = _psd(p_clean, fs)
            if len(f_m) > 0:
                ax_psd.semilogy(f_m, p_m, color=COLORS[mname], lw=1.2,
                                label=mname)

        # ── Labels / annotations ──────────────────────────────────
        ax_amp.set_ylabel(f"{mode_names[j]}\nDensity", fontsize=10)
        if var_ratios:
            vr_txt = "σ ratio: " + ", ".join(
                f"{v:.2f}" for _, v in var_ratios)
            ax_amp.text(0.98, 0.95, vr_txt, transform=ax_amp.transAxes,
                        fontsize=7, ha="right", va="top", color="gray")

        ax_acf.set_ylabel("ACF")
        ax_acf.axhline(0, color="gray", lw=0.5, ls="--")

        ax_psd.set_ylabel("PSD")

        if j == 0:
            ax_amp.set_title("Amplitude Distribution", fontsize=12,
                             fontweight="bold")
            ax_acf.set_title("Autocorrelation", fontsize=12,
                             fontweight="bold")
            ax_psd.set_title("Power Spectrum", fontsize=12,
                             fontweight="bold")
            ax_amp.legend(fontsize=7, loc="upper left")
            ax_acf.legend(fontsize=7, loc="upper right")
            ax_psd.legend(fontsize=7, loc="upper right")
        if j == Kw - 1:
            ax_amp.set_xlabel(f"a{j+1} value")
            ax_acf.set_xlabel("Lag (s)")
            ax_psd.set_xlabel("Frequency (Hz)")

    fig.tight_layout()
    out = worm_dir / "behaviour_stats.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Behaviour stats → {out}")


# ══════════════════════════════════════════════════════════════════
#  Aggregate behaviour statistics across all worms
# ══════════════════════════════════════════════════════════════════
def _aggregate_behaviour_stats(worm_ids, out_dir):
    """Load per-worm predictions.npz, compute and plot cross-worm behaviour
    statistics (amplitude distribution, ACF, PSD) averaged over all worms.

    Produces:
      • beh_stats_combined.png  — all models on same axes
      • beh_stats_summary.txt   — variance-ratio table
    """
    n_modes = K
    fs_all = []
    # Collect stats per worm
    per_worm = []  # list of {model -> {j -> {values, acf, psd}}}

    for wid in worm_ids:
        wdir = out_dir / wid
        npz_path = wdir / "predictions.npz"
        if not npz_path.exists():
            continue
        npz = np.load(str(npz_path))
        b_true = npz["b_true"]
        T = b_true.shape[0]
        Kw = min(n_modes, b_true.shape[1])

        # Try to get dt from results.json
        json_path = out_dir / "results.json"
        dt = 0.6  # fallback
        if json_path.exists():
            with open(json_path) as f:
                rj = json.load(f)
            if wid in rj and "dt" in rj[wid]:
                dt = rj[wid]["dt"]
        fs = 1.0 / dt
        fs_all.append(fs)

        warmup = max(2, N_LAGS)
        max_lag = min(200, T // 2)

        worm_stats = {}
        # GT
        worm_stats["GT"] = {}
        for j in range(Kw):
            gt = b_true[warmup:, j]
            worm_stats["GT"][j] = {
                "values": gt,
                "acf": _acf(gt, max_lag),
                "psd": _psd(gt, fs),
            }

        # Models
        for mname in MODELS:
            key = mname
            if key not in npz:
                continue
            preds = npz[key]
            worm_stats[mname] = {}
            for j in range(Kw):
                p = preds[warmup:, j]
                ok = np.isfinite(p)
                if ok.sum() < 50:
                    worm_stats[mname][j] = None
                    continue
                p_clean = p[ok]
                worm_stats[mname][j] = {
                    "values": p_clean,
                    "acf": _acf(p_clean, max_lag),
                    "psd": _psd(p_clean, fs),
                }
        per_worm.append(worm_stats)

    if not per_worm:
        print("  No predictions found for aggregate behaviour stats.")
        return

    n_worms = len(per_worm)
    mode_names = [f"EW{j+1}" for j in range(K)]

    # ── Combined plot: all models on same axes, averaged over worms ──
    fig, axes = plt.subplots(K, 3, figsize=(18, 3.2 * K))
    if K == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(f"Behaviour Statistics: GT vs Decoders — {n_worms} worms averaged",
                 fontsize=14, fontweight="bold", y=1.01)

    var_ratio_table = {}  # model -> [vr per mode]

    for j in range(K):
        ax_amp, ax_acf, ax_psd = axes[j, 0], axes[j, 1], axes[j, 2]

        # ── GT ────────────────────────────────────────────────────
        gt_vals = [s["GT"][j]["values"] for s in per_worm
                   if "GT" in s and j in s["GT"]]
        gt_acfs = [s["GT"][j]["acf"] for s in per_worm
                   if "GT" in s and j in s["GT"]]
        gt_psds = [s["GT"][j]["psd"] for s in per_worm
                   if "GT" in s and j in s["GT"]
                   and len(s["GT"][j]["psd"][0]) > 0]

        if gt_vals:
            all_gt = np.concatenate(gt_vals)
            bins = np.linspace(np.percentile(all_gt, 0.5),
                               np.percentile(all_gt, 99.5), 50)
            ax_amp.hist(all_gt, bins=bins, density=True, alpha=0.45,
                        color="gray", label="GT", edgecolor="none")
        if gt_acfs:
            ml = min(len(a) for a in gt_acfs)
            acf_mean = np.mean([a[:ml] for a in gt_acfs], axis=0)
            med_dt = dt if not fs_all else 1.0 / np.median(fs_all)
            ax_acf.plot(np.arange(ml) * med_dt, acf_mean, "k-", lw=1.5,
                        label="GT")
        if gt_psds:
            ml_f = min(len(p[1]) for p in gt_psds)
            f_com = gt_psds[0][0][:ml_f]
            psd_mean = np.mean([p[1][:ml_f] for p in gt_psds], axis=0)
            ax_psd.semilogy(f_com, psd_mean, "k-", lw=1.5, label="GT")

        # ── Models ────────────────────────────────────────────────
        for mname in MODELS:
            pred_vals = []
            pred_acfs = []
            pred_psds = []
            for s in per_worm:
                if mname in s and j in s[mname] and s[mname][j] is not None:
                    pred_vals.append(s[mname][j]["values"])
                    pred_acfs.append(s[mname][j]["acf"])
                    f_p, p_p = s[mname][j]["psd"]
                    if len(f_p) > 0:
                        pred_psds.append((f_p, p_p))

            if pred_vals and gt_vals:
                all_pr = np.concatenate(pred_vals)
                ax_amp.hist(all_pr, bins=bins, density=True, alpha=0.35,
                            color=COLORS[mname], label=mname,
                            edgecolor="none")
                # Variance ratio
                vr = np.std(all_pr) / (np.std(all_gt) + 1e-15)
                var_ratio_table.setdefault(mname, []).append(vr)

            if pred_acfs:
                ml_p = min(len(a) for a in pred_acfs)
                acf_p_mean = np.mean([a[:ml_p] for a in pred_acfs], axis=0)
                med_dt = dt if not fs_all else 1.0 / np.median(fs_all)
                ax_acf.plot(np.arange(ml_p) * med_dt, acf_p_mean,
                            color=COLORS[mname], lw=1.2, label=mname)

            if pred_psds:
                ml_fp = min(len(p[1]) for p in pred_psds)
                f_p_com = pred_psds[0][0][:ml_fp]
                psd_p_mean = np.mean([p[1][:ml_fp] for p in pred_psds], axis=0)
                ax_psd.semilogy(f_p_com, psd_p_mean,
                                color=COLORS[mname], lw=1.2, label=mname)

        # Labels
        ax_amp.set_ylabel(f"{mode_names[j]}\nDensity", fontsize=10)
        ax_acf.set_ylabel("ACF")
        ax_acf.axhline(0, color="gray", lw=0.5, ls="--")
        ax_psd.set_ylabel("PSD")

        if j == 0:
            ax_amp.set_title("Amplitude Distribution", fontsize=12,
                             fontweight="bold")
            ax_acf.set_title("Autocorrelation", fontsize=12,
                             fontweight="bold")
            ax_psd.set_title("Power Spectrum", fontsize=12,
                             fontweight="bold")
            ax_amp.legend(fontsize=7, loc="upper left")
            ax_acf.legend(fontsize=7, loc="upper right")
            ax_psd.legend(fontsize=7, loc="upper right")
        if j == K - 1:
            ax_amp.set_xlabel(f"a{j+1} value")
            ax_acf.set_xlabel("Lag (s)")
            ax_psd.set_xlabel("Frequency (Hz)")

    fig.tight_layout()
    fig.savefig(out_dir / "beh_stats_combined.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Aggregate behaviour stats → {out_dir / 'beh_stats_combined.png'}")

    # ── Variance-ratio summary table ──────────────────────────────
    txt_path = out_dir / "beh_stats_summary.txt"
    with open(txt_path, "w") as f:
        f.write(f"Variance ratio σ_pred/σ_GT — {n_worms} worms pooled\n\n")
        header = f"{'Model':16s}" + "  ".join(f"{'a'+str(j+1):>7s}" for j in range(K)) + "    mean\n"
        f.write(header)
        f.write("─" * len(header) + "\n")
        for mname in MODELS:
            vrs = var_ratio_table.get(mname, [float("nan")] * K)
            while len(vrs) < K:
                vrs.append(float("nan"))
            mn = float(np.nanmean(vrs))
            f.write(f"{mname:16s}" + "  ".join(f"{v:7.3f}" for v in vrs[:K])
                    + f"    {mn:.3f}\n")
    print(f"  Variance ratio table → {txt_path}")


# ══════════════════════════════════════════════════════════════════
#  Summary plots across all worms
# ══════════════════════════════════════════════════════════════════
def _summary_plots(all_results: dict, out_dir: Path):
    """Generate heatmap, bar, scatter, and correlation summary plots."""
    worm_ids = sorted([k for k, v in all_results.items() if "error" not in v])
    if len(worm_ids) < 2:
        print("  Too few worms for summary plots.")
        return

    # ── Heatmap: one panel per model ──────────────────────────────
    fig, axes = plt.subplots(1, len(MODELS),
                             figsize=(5 * len(MODELS), max(8, len(worm_ids) * 0.35)),
                             sharey=True)
    mode_names = [f"a{j+1}" for j in range(K)]
    for ax, mname in zip(axes, MODELS):
        mat = np.array([all_results[w].get(f"{mname}_r2", [np.nan]*K)[:K]
                        for w in worm_ids])
        im = ax.imshow(mat, aspect="auto", vmin=-0.2, vmax=1.0, cmap="RdYlGn")
        ax.set_xticks(range(K)); ax.set_xticklabels(mode_names, fontsize=10)
        ax.set_title(mname, fontsize=13, fontweight="bold", color=COLORS[mname])
        for i in range(len(worm_ids)):
            for j in range(K):
                v = mat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=6, color="white" if v < 0.3 else "black")
    axes[0].set_yticks(range(len(worm_ids)))
    axes[0].set_yticklabels(worm_ids, fontsize=7)
    axes[0].set_ylabel("Worm")
    fig.colorbar(im, ax=axes, shrink=0.6, label="R²")
    fig.suptitle(f"Per-worm R² — {len(worm_ids)} worms, 5-fold CV", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap_r2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap → {out_dir / 'heatmap_r2.png'}")

    # ── Heatmap: correlations ─────────────────────────────────────
    fig, axes = plt.subplots(1, len(MODELS),
                             figsize=(5 * len(MODELS), max(8, len(worm_ids) * 0.35)),
                             sharey=True)
    for ax, mname in zip(axes, MODELS):
        mat = np.array([all_results[w].get(f"{mname}_corr", [np.nan]*K)[:K]
                        for w in worm_ids])
        im = ax.imshow(mat, aspect="auto", vmin=-0.2, vmax=1.0, cmap="RdYlGn")
        ax.set_xticks(range(K)); ax.set_xticklabels(mode_names, fontsize=10)
        ax.set_title(mname, fontsize=13, fontweight="bold", color=COLORS[mname])
        for i in range(len(worm_ids)):
            for j in range(K):
                v = mat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=6, color="white" if v < 0.3 else "black")
    axes[0].set_yticks(range(len(worm_ids)))
    axes[0].set_yticklabels(worm_ids, fontsize=7)
    axes[0].set_ylabel("Worm")
    fig.colorbar(im, ax=axes, shrink=0.6, label="Pearson r")
    fig.suptitle(f"Per-worm correlation — {len(worm_ids)} worms, 5-fold CV",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap_corr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap (corr) → {out_dir / 'heatmap_corr.png'}")

    # ── Bar chart: mean ± SEM R² across worms ─────────────────────
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(K)
    n = len(MODELS)
    w = 0.8 / n
    for mi, mname in enumerate(MODELS):
        mat = np.array([all_results[wid].get(f"{mname}_r2", [np.nan]*K)[:K]
                        for wid in worm_ids])
        means = np.nanmean(mat, axis=0)
        sems = np.nanstd(mat, axis=0) / np.sqrt(np.sum(np.isfinite(mat), axis=0).clip(1))
        ax.bar(x + mi * w, means, w, yerr=sems, capsize=3,
               label=mname, color=COLORS[mname], edgecolor="white",
               linewidth=0.5, alpha=0.85)
        for j in range(K):
            ax.text(x[j] + mi * w, means[j] + sems[j] + 0.02,
                    f"{means[j]:.3f}", ha="center", fontsize=6.5,
                    color=COLORS[mname])
    ax.set_xticks(x + w * (n - 1) / 2)
    ax.set_xticklabels(mode_names, fontsize=12)
    ax.set_ylabel("R²", fontsize=12)
    ax.set_title(f"Mean R² across {len(worm_ids)} worms (± SEM)", fontsize=13)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_ylim(-0.15, 1.05)
    ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "bar_r2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Bar → {out_dir / 'bar_r2.png'}")

    # ── Bar chart: mean ± SEM correlation across worms ────────────
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for mi, mname in enumerate(MODELS):
        mat = np.array([all_results[wid].get(f"{mname}_corr", [np.nan]*K)[:K]
                        for wid in worm_ids])
        means = np.nanmean(mat, axis=0)
        sems = np.nanstd(mat, axis=0) / np.sqrt(np.sum(np.isfinite(mat), axis=0).clip(1))
        ax.bar(x + mi * w, means, w, yerr=sems, capsize=3,
               label=mname, color=COLORS[mname], edgecolor="white",
               linewidth=0.5, alpha=0.85)
        for j in range(K):
            ax.text(x[j] + mi * w, means[j] + sems[j] + 0.02,
                    f"{means[j]:.3f}", ha="center", fontsize=6.5,
                    color=COLORS[mname])
    ax.set_xticks(x + w * (n - 1) / 2)
    ax.set_xticklabels(mode_names, fontsize=12)
    ax.set_ylabel("Pearson r", fontsize=12)
    ax.set_title(f"Mean correlation across {len(worm_ids)} worms (± SEM)", fontsize=13)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_ylim(-0.15, 1.05)
    ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "bar_corr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Bar (corr) → {out_dir / 'bar_corr.png'}")

    # ── Scatter: per-worm mean R², one dot per worm ───────────────
    fig, axes_sc = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
    axes_sc = axes_sc.ravel()
    for j in range(K):
        ax = axes_sc[j]
        for mi, mname in enumerate(MODELS):
            vals = np.array([all_results[w].get(f"{mname}_r2", [np.nan]*K)[j]
                             for w in worm_ids])
            ok = np.isfinite(vals)
            jitter = np.random.uniform(-0.15, 0.15, size=ok.sum())
            ax.scatter(mi + jitter, vals[ok], s=18, alpha=0.6,
                       color=COLORS[mname], edgecolors="white", linewidth=0.3)
            if ok.any():
                mn = np.nanmean(vals[ok])
                ax.plot([mi - 0.3, mi + 0.3], [mn, mn], lw=2, color=COLORS[mname])
                ax.text(mi, mn + 0.03, f"{mn:.3f}", ha="center", fontsize=8,
                        fontweight="bold", color=COLORS[mname])
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels(MODELS, fontsize=8, rotation=20)
        ax.set_title(mode_names[j], fontsize=12)
        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.3)
        ax.set_ylim(-0.3, 1.05)
        if j % 3 == 0:
            ax.set_ylabel("R²")
    fig.suptitle(f"Per-mode R² across {len(worm_ids)} worms", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_r2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Scatter → {out_dir / 'scatter_r2.png'}")

    # ── Scatter: per-worm mean correlation ────────────────────────
    fig, axes_sc = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
    axes_sc = axes_sc.ravel()
    for j in range(K):
        ax = axes_sc[j]
        for mi, mname in enumerate(MODELS):
            vals = np.array([all_results[w].get(f"{mname}_corr", [np.nan]*K)[j]
                             for w in worm_ids])
            ok = np.isfinite(vals)
            jitter = np.random.uniform(-0.15, 0.15, size=ok.sum())
            ax.scatter(mi + jitter, vals[ok], s=18, alpha=0.6,
                       color=COLORS[mname], edgecolors="white", linewidth=0.3)
            if ok.any():
                mn = np.nanmean(vals[ok])
                ax.plot([mi - 0.3, mi + 0.3], [mn, mn], lw=2, color=COLORS[mname])
                ax.text(mi, mn + 0.03, f"{mn:.3f}", ha="center", fontsize=8,
                        fontweight="bold", color=COLORS[mname])
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels(MODELS, fontsize=8, rotation=20)
        ax.set_title(mode_names[j], fontsize=12)
        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.3)
        ax.set_ylim(-0.3, 1.05)
        if j % 3 == 0:
            ax.set_ylabel("Pearson r")
    fig.suptitle(f"Per-mode correlation across {len(worm_ids)} worms", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_corr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Scatter (corr) → {out_dir / 'scatter_corr.png'}")

    # ── Aggregate behaviour statistics ─────────────────────────────
    _aggregate_behaviour_stats(worm_ids, out_dir)

    # ── Overall summary: mean R² per model ────────────────────────
    print(f"\n  ══ Summary across {len(worm_ids)} worms ══")
    print(f"  {'Model':16s}" + "  ".join(f"{'a'+str(j+1):>7s}" for j in range(K))
          + "    mean_r2   mean_corr")
    for mname in MODELS:
        r2_mat = np.array([all_results[w].get(f"{mname}_r2", [np.nan]*K)[:K]
                           for w in worm_ids])
        corr_mat = np.array([all_results[w].get(f"{mname}_corr", [np.nan]*K)[:K]
                             for w in worm_ids])
        r2_means = np.nanmean(r2_mat, axis=0)
        corr_means = np.nanmean(corr_mat, axis=0)
        print(f"  {mname:16s}" + "  ".join(f"{v:7.3f}" for v in r2_means)
              + f"    {np.mean(r2_means):.3f}     {np.mean(corr_means):.3f}")


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5_dir", required=True)
    ap.add_argument("--out_dir", default="output_plots/behaviour_decoder/batch_4model")
    ap.add_argument("--resume", action="store_true",
                    help="Skip worms already in results.json")
    ap.add_argument("--no_video", action="store_true",
                    help="Skip video rendering (faster)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(glob.glob(str(Path(args.h5_dir) / "*.h5")))
    if not h5_files:
        print(f"No .h5 files in {args.h5_dir}"); sys.exit(1)
    print(f"\n  Found {len(h5_files)} worms in {args.h5_dir}")
    print(f"  Models: {', '.join(MODELS)}")
    print(f"  E2E epochs: {E2E_EPOCHS}, TBPTT chunk: {TBPTT}\n")

    json_path = out_dir / "results.json"
    if args.resume and json_path.exists():
        with open(json_path) as f:
            all_results = json.load(f)
        print(f"  Resuming: {len(all_results)} worms already done\n")
    else:
        all_results = {}

    t0 = time.time()

    for wi, h5_path in enumerate(h5_files):
        worm_id = Path(h5_path).stem
        if worm_id in all_results and args.resume:
            print(f"  [{wi+1}/{len(h5_files)}] {worm_id}  ── SKIP")
            continue

        print(f"  [{wi+1}/{len(h5_files)}] {worm_id}")
        worm_dir = out_dir / worm_id
        try:
            res = _run_one_worm(h5_path, worm_dir,
                                make_video=not args.no_video)
            all_results[worm_id] = res
        except Exception as exc:
            print(f"    ✗ FAILED: {exc}")
            traceback.print_exc()
            all_results[worm_id] = {"error": str(exc)}

        # Checkpoint
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n  All worms done in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # ── Summary plots ─────────────────────────────────────────────
    _summary_plots(all_results, out_dir)

    # ── Save text summary ─────────────────────────────────────────
    worm_ids = sorted([k for k, v in all_results.items() if "error" not in v])
    txt = out_dir / "summary.txt"
    with open(txt, "w") as f:
        f.write(f"Batch 4-model benchmark — {len(worm_ids)} worms, 5-fold CV\n\n")
        f.write(f"{'Worm':<20s}")
        for mname in MODELS:
            for j in range(K):
                f.write(f"  {mname}_a{j+1:>3s}")
        f.write("\n")
        for wid in worm_ids:
            f.write(f"{wid:<20s}")
            for mname in MODELS:
                vals = all_results[wid].get(f"{mname}_r2", [float("nan")]*K)
                for v in vals[:K]:
                    f.write(f"  {v:8.3f}")
            f.write("\n")
    print(f"  Text summary → {txt}")
    print("\n  Done.\n")


if __name__ == "__main__":
    main()
