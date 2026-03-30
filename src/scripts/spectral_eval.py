#!/usr/bin/env python
"""Spectral / distributional evaluation of decoder models.

Standard R² penalises phase errors harshly — a model that generates
perfectly realistic oscillations shifted by Δφ scores poorly.  This
script computes *phase-invariant* metrics that capture whether the
generated behaviour *looks real*:

  1. **PSD match** – power spectral density overlap per mode
  2. **ACF match** – autocorrelation function similarity (captures timescale)
  3. **Amplitude distribution** – KS distance between pred / GT amplitude PDFs
  4. **Phase-velocity distribution** – KS distance on dφ/dt (for a₁–a₂ pair)
  5. **Cross-correlation peak** – max cross-corr and lag at peak (phase offset)
  6. **Standard R²** – for reference

Models evaluated:
  • Ridge (instantaneous, no phase drift)
  • E2E+Diag (free-run AR(2) + MLP, diagonal + spectral clamp)

Usage:
    .venv/bin/python -m scripts.spectral_eval \
        --h5 "data/used/behaviour+neuronal activity atanas (2023)/2/2023-01-17-14.h5"
"""
from __future__ import annotations

import argparse, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import signal, stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stage2.behavior_decoder_eval import (
    _log_ridge_grid,
    _ridge_cv_single_target,
    build_lagged_features_np,
)
try:
    from scripts.benchmark_ar_decoder_v2 import (
        load_data, build_ar_features, r2_score, E2EOscillatorMLP,
    )
except ModuleNotFoundError:
    from benchmark_ar_decoder_v2 import (
        load_data, build_ar_features, r2_score, E2EOscillatorMLP,
    )
from scripts.unified_benchmark import (
    _ridge_fit, _train_e2e, _clamp_diag_spectral_radius,
)

import torch
import torch.nn as nn


# ================================================================== #
#  Spectral / distributional metrics
# ================================================================== #

def psd_match(gt: np.ndarray, pred: np.ndarray, fs: float) -> dict:
    """Power spectral density overlap between GT and predicted signal.

    Returns:
        log_psd_mse: MSE between log10(PSD) — lower is better
        psd_corr: Pearson correlation of log10(PSD)
    """
    nperseg = min(256, len(gt) // 2)
    f_gt, psd_gt = signal.welch(gt, fs=fs, nperseg=nperseg)
    f_pr, psd_pr = signal.welch(pred, fs=fs, nperseg=nperseg)

    # Avoid log(0)
    log_gt = np.log10(np.maximum(psd_gt, 1e-15))
    log_pr = np.log10(np.maximum(psd_pr, 1e-15))

    mse = float(np.mean((log_gt - log_pr) ** 2))
    corr = float(np.corrcoef(log_gt, log_pr)[0, 1]) if len(log_gt) > 2 else 0.0
    return {"log_psd_mse": mse, "psd_corr": corr,
            "freqs": f_gt, "psd_gt": psd_gt, "psd_pred": psd_pr}


def acf_match(gt: np.ndarray, pred: np.ndarray, max_lag: int = 200) -> dict:
    """Autocorrelation function similarity.

    Returns:
        acf_mse: MSE between normalised ACFs
        acf_corr: Pearson correlation of ACFs
    """
    def _acf(x, ml):
        x = x - x.mean()
        c = np.correlate(x, x, mode="full")
        c = c[len(c) // 2:]
        c = c / (c[0] + 1e-15)
        return c[:ml]

    ml = min(max_lag, len(gt) // 2)
    acf_gt = _acf(gt, ml)
    acf_pr = _acf(pred, ml)
    mse = float(np.mean((acf_gt - acf_pr) ** 2))
    corr = float(np.corrcoef(acf_gt, acf_pr)[0, 1]) if ml > 2 else 0.0
    return {"acf_mse": mse, "acf_corr": corr,
            "acf_gt": acf_gt, "acf_pred": acf_pr}


def amplitude_ks(gt: np.ndarray, pred: np.ndarray) -> dict:
    """KS test between amplitude distributions. Lower D = better match."""
    ks_stat, p_val = stats.ks_2samp(gt, pred)
    return {"ks_D": float(ks_stat), "ks_p": float(p_val)}


def xcorr_peak(gt: np.ndarray, pred: np.ndarray, fs: float) -> dict:
    """Cross-correlation: peak value and lag (in seconds).

    Peak xcorr close to 1.0 means waveforms match up to a time shift.
    """
    gt_n = (gt - gt.mean()) / (gt.std() + 1e-12)
    pr_n = (pred - pred.mean()) / (pred.std() + 1e-12)
    cc = np.correlate(gt_n, pr_n, mode="full") / len(gt)
    lags = np.arange(-len(gt) + 1, len(gt)) / fs
    peak_idx = np.argmax(cc)
    return {"xcorr_peak": float(cc[peak_idx]),
            "lag_at_peak_s": float(lags[peak_idx]),
            "xcorr": cc, "lags": lags}


def phase_velocity_ks(a1_gt, a2_gt, a1_pred, a2_pred, dt) -> dict:
    """KS test on dφ/dt distributions for the (a₁, a₂) pair."""
    def _dphi(a1, a2, dt_):
        da1 = np.gradient(a1, dt_)
        da2 = np.gradient(a2, dt_)
        r2 = np.maximum(a1**2 + a2**2, 1e-8)
        return (a1 * da2 - a2 * da1) / r2

    dphi_gt = _dphi(a1_gt, a2_gt, dt)
    dphi_pr = _dphi(a1_pred, a2_pred, dt)
    ks_stat, p_val = stats.ks_2samp(dphi_gt, dphi_pr)
    return {"dphi_ks_D": float(ks_stat), "dphi_ks_p": float(p_val),
            "dphi_gt": dphi_gt, "dphi_pred": dphi_pr}


# ================================================================== #
#  Model fitting  (Ridge + E2E+Diag)
# ================================================================== #

def fit_and_predict(u, b, X_neural, folds, warmup, ridge_grid, n_inner,
                    K, T, d_in, args):
    """Fit Ridge and E2E+Diag, return held-out predictions for both."""
    ho_ridge = np.full((T, K), np.nan)
    ho_e2e   = np.full((T, K), np.nan)

    for fi, (ts, te) in enumerate(folds):
        train_mask = np.ones(T, dtype=bool)
        train_mask[:warmup] = False
        train_mask[ts:te] = False
        train_idx = np.where(train_mask)[0]

        X_tr = X_neural[train_idx]
        X_te = X_neural[ts:te]
        b_tr = b[train_idx]

        print(f"  Fold {fi+1}/{len(folds)}: test=[{ts}:{te})")

        # ── Ridge ────────────────────────────────────────────────────
        for j in range(K):
            coef, intc, _ = _ridge_fit(X_tr, b_tr[:, j], ridge_grid, n_inner)
            ho_ridge[ts:te, j] = X_te @ coef + intc

        # ── E2E+Diag ────────────────────────────────────────────────
        segs = [(max(warmup, 2), ts)]
        if te < T:
            segs.append((te, T))
        # Filter to valid segments
        segs = [(s0, s1) for s0, s1 in segs if s1 - s0 > 10]

        M1e, M2e, ce, drv = _train_e2e(
            d_in, K, segs, b, X_neural,
            epochs=args.e2e_epochs, chunk=args.tbptt_chunk,
            diagonal_ar=True, max_rho=args.max_rho,
            tag=f"E2E+Diag fold{fi+1}")

        # Free-run on test
        drive = drv  # already computed for all T
        p1 = b[ts - 1].copy()
        p2 = b[ts - 2].copy()
        for t in range(ts, te):
            p_new = M1e @ p1 + M2e @ p2 + drive[t] + ce
            ho_e2e[t] = p_new
            p2, p1 = p1, p_new

    return ho_ridge, ho_e2e


# ================================================================== #
#  Plotting
# ================================================================== #

def plot_spectral_comparison(gt, preds_dict, dt, K, out_dir):
    """Comprehensive spectral comparison figure."""
    fs = 1.0 / dt
    mode_names = [f"a{j+1}" for j in range(K)]
    models = list(preds_dict.keys())
    colors = {"Ridge": "#1f77b4", "E2E+Diag": "#d62728"}

    # ── 1. PSD comparison ────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Power Spectral Density: GT vs models", fontsize=14,
                 fontweight="bold")
    for j in range(min(K, 6)):
        ax = axes.ravel()[j]
        nperseg = min(256, len(gt) // 2)
        f_gt, psd_gt = signal.welch(gt[:, j], fs=fs, nperseg=nperseg)
        ax.semilogy(f_gt, psd_gt, "k-", lw=2, label="GT", alpha=0.8)
        for mname in models:
            pred = preds_dict[mname]
            valid = ~np.isnan(pred[:, j])
            if valid.sum() < 50:
                continue
            f_pr, psd_pr = signal.welch(pred[valid, j], fs=fs, nperseg=nperseg)
            ax.semilogy(f_pr, psd_pr, color=colors.get(mname, "gray"),
                        lw=1.5, label=mname, alpha=0.8)
        ax.set_title(mode_names[j], fontsize=12)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD")
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "psd_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 2. ACF comparison ────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Autocorrelation: GT vs models", fontsize=14,
                 fontweight="bold")
    max_lag = min(200, len(gt) // 2)
    lags_s = np.arange(max_lag) * dt
    for j in range(min(K, 6)):
        ax = axes.ravel()[j]

        def _acf(x, ml):
            x = x - x.mean()
            c = np.correlate(x, x, mode="full")
            c = c[len(c) // 2:]
            return c[:ml] / (c[0] + 1e-15)

        acf_gt = _acf(gt[:, j], max_lag)
        ax.plot(lags_s, acf_gt, "k-", lw=2, label="GT", alpha=0.8)
        for mname in models:
            pred = preds_dict[mname]
            valid = ~np.isnan(pred[:, j])
            p = pred[valid, j]
            if len(p) < 50:
                continue
            acf_pr = _acf(p, min(max_lag, len(p) // 2))
            ax.plot(np.arange(len(acf_pr)) * dt, acf_pr,
                    color=colors.get(mname, "gray"),
                    lw=1.5, label=mname, alpha=0.8)
        ax.set_title(mode_names[j])
        ax.set_xlabel("Lag (s)")
        ax.set_ylabel("ACF")
        ax.legend(fontsize=8)
        ax.axhline(0, color="gray", lw=0.5, ls="--")
    plt.tight_layout()
    fig.savefig(out_dir / "acf_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 3. Amplitude distributions ───────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Amplitude distributions: GT vs models", fontsize=14,
                 fontweight="bold")
    for j in range(min(K, 6)):
        ax = axes.ravel()[j]
        ax.hist(gt[:, j], bins=50, density=True, alpha=0.5,
                color="k", label="GT")
        for mname in models:
            pred = preds_dict[mname]
            valid = ~np.isnan(pred[:, j])
            if valid.sum() < 50:
                continue
            ax.hist(pred[valid, j], bins=50, density=True, alpha=0.4,
                    color=colors.get(mname, "gray"), label=mname)
        ax.set_title(mode_names[j])
        ax.set_xlabel("Amplitude")
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "amplitude_dist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 4. Phase plane (a₁ vs a₂) ───────────────────────────────────
    fig, axes = plt.subplots(1, len(models) + 1, figsize=(6 * (len(models) + 1), 5))
    fig.suptitle("Phase plane (a₁ vs a₂)", fontsize=14, fontweight="bold")
    ax = axes[0]
    ax.scatter(gt[:, 0], gt[:, 1], c=np.arange(len(gt)), cmap="viridis",
               s=3, alpha=0.5)
    ax.set_title("GT", fontsize=12)
    ax.set_xlabel("a₁"); ax.set_ylabel("a₂")
    ax.set_aspect("equal")
    for mi, mname in enumerate(models):
        ax = axes[mi + 1]
        pred = preds_dict[mname]
        valid = ~np.isnan(pred[:, 0])
        ax.scatter(pred[valid, 0], pred[valid, 1],
                   c=np.arange(valid.sum()), cmap="viridis", s=3, alpha=0.5)
        ax.set_title(mname, fontsize=12)
        ax.set_xlabel("a₁"); ax.set_ylabel("a₂")
        ax.set_aspect("equal")
        # Match axis limits to GT
        ax.set_xlim(axes[0].get_xlim())
        ax.set_ylim(axes[0].get_ylim())
    plt.tight_layout()
    fig.savefig(out_dir / "phase_plane.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 5. dφ/dt distributions ───────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Phase velocity dφ/dt distribution", fontsize=14,
                 fontweight="bold")
    def _dphi(a1, a2, dt_):
        da1 = np.gradient(a1, dt_)
        da2 = np.gradient(a2, dt_)
        r2_ = np.maximum(a1**2 + a2**2, 1e-8)
        return (a1 * da2 - a2 * da1) / r2_

    dphi_gt = _dphi(gt[:, 0], gt[:, 1], dt)
    # Histogram
    ax = axes[0]
    ax.hist(dphi_gt, bins=80, density=True, alpha=0.5, color="k", label="GT")
    for mname in models:
        pred = preds_dict[mname]
        valid = ~np.isnan(pred[:, 0])
        if valid.sum() < 50:
            continue
        dphi_pr = _dphi(pred[valid, 0], pred[valid, 1], dt)
        ax.hist(dphi_pr, bins=80, density=True, alpha=0.4,
                color=colors.get(mname, "gray"), label=mname)
    ax.set_xlabel("dφ/dt (rad/s)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.set_title("dφ/dt histogram")
    # Time series snippet
    ax = axes[1]
    n_show = min(300, len(gt))
    t_ax = np.arange(n_show) * dt
    ax.plot(t_ax, dphi_gt[:n_show], "k-", lw=1.5, label="GT", alpha=0.7)
    for mname in models:
        pred = preds_dict[mname]
        valid = ~np.isnan(pred[:, 0])
        if valid.sum() < n_show:
            continue
        dphi_pr = _dphi(pred[valid, 0], pred[valid, 1], dt)
        ax.plot(t_ax, dphi_pr[:n_show], color=colors.get(mname, "gray"),
                lw=1, label=mname, alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("dφ/dt (rad/s)")
    ax.set_title("dφ/dt time series (first 300 frames)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / "dphi_dt_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ================================================================== #
#  Main
# ================================================================== #

def main():
    ap = argparse.ArgumentParser(
        description="Spectral / distributional evaluation of decoders")
    ap.add_argument("--h5", required=True)
    ap.add_argument("--n_modes", type=int, default=6)
    ap.add_argument("--neural_lags", type=int, default=3)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--all_neurons", action="store_true")
    ap.add_argument("--e2e_epochs", type=int, default=300)
    ap.add_argument("--tbptt_chunk", type=int, default=50)
    ap.add_argument("--max_rho", type=float, default=0.98)
    ap.add_argument("--out_dir", default="output_plots/spectral_eval")
    args = ap.parse_args()

    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────
    u, b_full, dt = load_data(args.h5, all_neurons=args.all_neurons)
    if args.max_frames > 0:
        u = u[:args.max_frames]
        b_full = b_full[:args.max_frames]
    K = min(args.n_modes, b_full.shape[1])
    b = b_full[:, :K]
    T = b.shape[0]
    n_lags = args.neural_lags
    warmup = max(2, n_lags)
    X_neural = build_lagged_features_np(u, n_lags)
    d_in = X_neural.shape[1]
    fs = 1.0 / dt

    ridge_grid = _log_ridge_grid(-3.0, 10.0, 50)
    n_inner = args.n_folds - 1

    # ── Folds ─────────────────────────────────────────────────────────
    valid_len = T - warmup
    fold_size = valid_len // args.n_folds
    folds = []
    for i in range(args.n_folds):
        s = warmup + i * fold_size
        e = warmup + (i + 1) * fold_size if i < args.n_folds - 1 else T
        folds.append((s, e))

    print(f"\n  Data: T={T}, M={u.shape[1]}, K={K}, dt={dt:.2f}s, fs={fs:.2f}Hz")

    # ── Fit models ────────────────────────────────────────────────────
    ho_ridge, ho_e2e = fit_and_predict(
        u, b, X_neural, folds, warmup, ridge_grid, n_inner,
        K, T, d_in, args)

    preds = {"Ridge": ho_ridge, "E2E+Diag": ho_e2e}

    # ── Compute metrics ───────────────────────────────────────────────
    valid = np.arange(warmup, T)
    print(f"\n{'═' * 90}")
    print("  SPECTRAL / DISTRIBUTIONAL METRICS")
    print(f"{'═' * 90}")

    # Header
    print(f"\n  {'Metric':<25s} {'Model':<12s}", end="")
    for j in range(K):
        print(f" {'a'+str(j+1):>8s}", end="")
    print(f" {'mean':>8s}")
    print("  " + "─" * (25 + 12 + 9 * (K + 1)))

    all_metrics = {}
    for mname, pred in preds.items():
        ok = np.isfinite(pred[valid, 0])
        idx = valid[ok]
        if idx.size < 50:
            print(f"  {mname}: insufficient valid predictions")
            continue

        metrics = {}

        # R² (standard)
        r2s = [r2_score(b[idx, j], pred[idx, j]) for j in range(K)]
        metrics["R²"] = r2s
        print(f"  {'R²':<25s} {mname:<12s}", end="")
        for v in r2s:
            print(f" {v:8.3f}", end="")
        print(f" {np.mean(r2s):8.3f}")

        # PSD correlation
        psd_corrs = []
        for j in range(K):
            res = psd_match(b[idx, j], pred[idx, j], fs)
            psd_corrs.append(res["psd_corr"])
        metrics["PSD corr"] = psd_corrs
        print(f"  {'PSD corr (log)':<25s} {mname:<12s}", end="")
        for v in psd_corrs:
            print(f" {v:8.3f}", end="")
        print(f" {np.mean(psd_corrs):8.3f}")

        # PSD log MSE
        psd_mses = []
        for j in range(K):
            res = psd_match(b[idx, j], pred[idx, j], fs)
            psd_mses.append(res["log_psd_mse"])
        metrics["PSD log-MSE"] = psd_mses
        print(f"  {'PSD log-MSE ↓':<25s} {mname:<12s}", end="")
        for v in psd_mses:
            print(f" {v:8.3f}", end="")
        print(f" {np.mean(psd_mses):8.3f}")

        # ACF correlation
        acf_corrs = []
        for j in range(K):
            res = acf_match(b[idx, j], pred[idx, j])
            acf_corrs.append(res["acf_corr"])
        metrics["ACF corr"] = acf_corrs
        print(f"  {'ACF corr':<25s} {mname:<12s}", end="")
        for v in acf_corrs:
            print(f" {v:8.3f}", end="")
        print(f" {np.mean(acf_corrs):8.3f}")

        # Amplitude KS
        ks_ds = []
        for j in range(K):
            res = amplitude_ks(b[idx, j], pred[idx, j])
            ks_ds.append(res["ks_D"])
        metrics["Amp KS-D ↓"] = ks_ds
        print(f"  {'Amp KS-D ↓':<25s} {mname:<12s}", end="")
        for v in ks_ds:
            print(f" {v:8.3f}", end="")
        print(f" {np.mean(ks_ds):8.3f}")

        # Cross-correlation peak
        xc_peaks = []
        xc_lags = []
        for j in range(K):
            res = xcorr_peak(b[idx, j], pred[idx, j], fs)
            xc_peaks.append(res["xcorr_peak"])
            xc_lags.append(res["lag_at_peak_s"])
        metrics["XCorr peak"] = xc_peaks
        metrics["XCorr lag (s)"] = xc_lags
        print(f"  {'XCorr peak':<25s} {mname:<12s}", end="")
        for v in xc_peaks:
            print(f" {v:8.3f}", end="")
        print(f" {np.mean(xc_peaks):8.3f}")
        print(f"  {'XCorr lag (s)':<25s} {mname:<12s}", end="")
        for v in xc_lags:
            print(f" {v:8.2f}", end="")
        print(f" {np.mean(np.abs(xc_lags)):8.2f}")

        # Phase-velocity KS (a₁–a₂ pair)
        if K >= 2:
            pv = phase_velocity_ks(b[idx, 0], b[idx, 1],
                                   pred[idx, 0], pred[idx, 1], dt)
            metrics["dφ/dt KS-D ↓"] = pv["dphi_ks_D"]
            print(f"  {'dφ/dt KS-D ↓':<25s} {mname:<12s} {pv['dphi_ks_D']:8.3f}")

        all_metrics[mname] = metrics
        print()

    # ── Plots ─────────────────────────────────────────────────────────
    print("  Generating plots...")
    plot_spectral_comparison(b[valid], {m: p[valid] for m, p in preds.items()},
                             dt, K, out_dir)

    # ── Summary bar chart of all metrics ─────────────────────────────
    metric_names = ["R²", "PSD corr", "ACF corr", "XCorr peak"]
    lower_better = ["PSD log-MSE", "Amp KS-D ↓"]

    fig, axes = plt.subplots(1, len(metric_names) + len(lower_better),
                              figsize=(5 * (len(metric_names) + len(lower_better)), 5))
    model_colors = {"Ridge": "#1f77b4", "E2E+Diag": "#d62728"}
    for ai, mname_m in enumerate(metric_names + lower_better):
        ax = axes[ai]
        x = np.arange(K)
        w = 0.35
        for mi, (model_name, metrics) in enumerate(all_metrics.items()):
            key = mname_m.replace(" ↓", "")
            if key in metrics:
                vals = metrics[key]
                if isinstance(vals, (list, np.ndarray)):
                    ax.bar(x + mi * w, vals, w,
                           label=model_name,
                           color=model_colors.get(model_name, "gray"))
        ax.set_xticks(x + w / 2)
        ax.set_xticklabels([f"a{j+1}" for j in range(K)], fontsize=9)
        ax.set_title(mname_m, fontsize=11)
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "metrics_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    elapsed = time.time() - t0
    print(f"\n  All plots saved to {out_dir}/")
    print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)\n")


if __name__ == "__main__":
    main()
