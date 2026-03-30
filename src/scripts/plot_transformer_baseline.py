#!/usr/bin/env python3
"""Comprehensive plots & behaviour statistics for the Transformer baseline.

Generates
---------
1. Neural-trace panels (GT vs one-step vs LOO vs free-run for top neurons)
2. R² histogram/bar charts for all four metrics
3. Free-run rollout comparison (full time-series overlay)
4. Behaviour comparison:  real eigenworms vs simulated (decoded from free-run motor)
5. Behaviour statistics table:  amplitude, autocorrelation, power-spectrum comparison
6. (Optional) Side-by-side with Stage-2 joint-CV run if available

All figures are saved to the model's output directory.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal, stats

# ── project root on path ──
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from baseline_transformer.config import TransformerBaselineConfig
from baseline_transformer.model import build_model
from baseline_transformer.dataset import load_worm_data
from baseline_transformer.evaluate import (
    compute_onestep_r2,
    compute_loo_r2,
    compute_free_run_r2,
)
from stage2._utils import _r2
from stage2.behavior_decoder_eval import (
    build_lagged_features_np,
    _log_ridge_grid,
    _make_contiguous_folds,
    _ridge_cv_single_target,
    valid_lag_mask_np,
)

# ── Paths ────────────────────────────────────────────────────────────────────

WORM_ID = "2022-08-02-01"
H5_DIR = ROOT / "data/used/behaviour+neuronal activity atanas (2023)/the same neurons"
OUT_DIR = ROOT / "output_plots/transformer_baseline/single" / WORM_ID
MODEL_PATH = OUT_DIR / "model.pt"
RESULTS_JSON = OUT_DIR / "eval_results.json"

STAGE2_H5 = ROOT / "output_plots/stage2/joint_cv_1/stage2_results.h5"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 140,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
})

# ═════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════════════


def _load_model_and_data():
    """Load trained model + worm data."""
    h5_path = H5_DIR / f"{WORM_ID}.h5"
    worm = load_worm_data(str(h5_path))
    u = worm["u"]  # (T, N)
    T, N = u.shape
    print(f"Loaded worm {WORM_ID}: T={T}, N={N}")

    # The checkpoint is a plain state_dict (OrderedDict).
    # Use default config (matches what was trained).
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    cfg = TransformerBaselineConfig()
    cfg.device = DEVICE

    model = build_model(n_obs=N, cfg=cfg, device=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    return model, cfg, worm


def _fit_behaviour_decoder(u_motor, b, b_mask, n_lags=8, n_folds=5):
    """Fit ridge-CV decoder motor→eigenworms. Returns held-out predictions."""
    n_modes = b.shape[1]
    ridge_grid = _log_ridge_grid(-3.0, 10.0, 50)
    X = build_lagged_features_np(u_motor, n_lags)
    preds = np.full_like(b, np.nan)

    for j in range(n_modes):
        valid = valid_lag_mask_np(b.shape[0], n_lags, b_mask[:, j] > 0.5)
        idx_v = np.where(valid)[0]
        if len(idx_v) < 10:
            continue
        fit = _ridge_cv_single_target(X, b[:, j], idx_v, ridge_grid, n_folds)
        preds[:, j] = fit["held_out"]
    return preds


# ═════════════════════════════════════════════════════════════════════════════
#  PLOT 1:  R² Bar Charts
# ═════════════════════════════════════════════════════════════════════════════


def plot_r2_summary(eval_res, save_path):
    """Summary bar chart of all four metric means."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

    # 1. One-step R²
    r2_os = np.array(eval_res["onestep"]["r2"])
    ax = axes[0]
    ax.hist(r2_os, bins=25, color="steelblue", edgecolor="k", linewidth=0.3)
    ax.axvline(np.nanmean(r2_os), color="red", ls="--", lw=1.2,
               label=f"mean={np.nanmean(r2_os):.3f}")
    ax.set_title("One-step R² (test)")
    ax.set_xlabel("R²"); ax.set_ylabel("# neurons"); ax.legend(fontsize=7)

    # 2. LOO R²
    r2_loo = np.array(eval_res["loo"]["r2"])
    ax = axes[1]
    finite = r2_loo[np.isfinite(r2_loo)]
    ax.hist(finite, bins=20, color="darkorange", edgecolor="k", linewidth=0.3)
    ax.axvline(np.nanmean(finite), color="red", ls="--", lw=1.2,
               label=f"mean={np.nanmean(finite):.3f}")
    ax.set_title(f"LOO R² ({len(finite)} neurons)")
    ax.set_xlabel("R²"); ax.legend(fontsize=7)

    # 3. Free-run R²
    r2_fr = np.array(eval_res["free_run"]["r2"])
    finite_fr = r2_fr[np.isfinite(r2_fr)]
    ax = axes[2]
    ax.hist(finite_fr, bins=20, color="seagreen", edgecolor="k", linewidth=0.3)
    ax.axvline(np.nanmean(finite_fr), color="red", ls="--", lw=1.2,
               label=f"mean={np.nanmean(finite_fr):.3f}")
    ax.set_title(f"Free-run R² ({eval_res['free_run']['mode']})")
    ax.set_xlabel("R²"); ax.legend(fontsize=7)

    # 4. Behaviour R²
    ax = axes[3]
    beh = eval_res.get("behaviour", {})
    if beh.get("r2_model") is not None:
        modes = np.arange(len(beh["r2_model"]))
        w = 0.35
        ax.bar(modes - w/2, beh["r2_gt"], w, label="GT decoder", color="cornflowerblue")
        ax.bar(modes + w/2, beh["r2_model"], w, label="Model decoder", color="salmon")
        ax.set_title(f"Behaviour R²")
        ax.set_xlabel("Eigenworm mode"); ax.set_ylabel("R²")
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "No behaviour data", ha="center", va="center",
                transform=ax.transAxes)

    fig.suptitle(f"Transformer Baseline — {WORM_ID}", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
#  PLOT 2:  Neural Trace Comparisons
# ═════════════════════════════════════════════════════════════════════════════


def plot_neural_traces(u, onestep_res, loo_res, freerun_res, labels, motor_idx,
                       save_path, n_show=8):
    """Top-N neuron traces: GT vs one-step vs LOO vs free-run."""
    T, N = u.shape
    K = onestep_res.get("K", 16)

    # Pick neurons: top LOO R² + a few motor neurons
    r2_loo = np.array(loo_res["r2"])
    loo_subset = loo_res.get("subset", [])
    finite_idx = [i for i in loo_subset if np.isfinite(r2_loo[i])]
    by_r2 = sorted(finite_idx, key=lambda i: r2_loo[i], reverse=True)

    # Mix of best-LOO and motor neurons
    show = []
    for i in by_r2:
        if len(show) >= n_show:
            break
        if i not in show:
            show.append(i)
    # Pad with motor neurons if needed
    if motor_idx:
        for i in motor_idx:
            if len(show) >= n_show:
                break
            if i not in show and i in finite_idx:
                show.append(i)

    if not show:
        show = list(range(min(n_show, N)))

    fig, axes = plt.subplots(len(show), 1, figsize=(14, 2.2 * len(show)), sharex=True)
    if len(show) == 1:
        axes = [axes]

    t_ax = np.arange(T)

    # Free-run pred
    fr_pred = freerun_res["pred"]  # (T, N)

    for ax_i, neuron_i in enumerate(show):
        ax = axes[ax_i]
        lbl = labels[neuron_i] if neuron_i < len(labels) else f"N{neuron_i}"
        is_motor = motor_idx and neuron_i in motor_idx

        # GT
        ax.plot(t_ax, u[:, neuron_i], color="k", lw=0.8, alpha=0.7, label="GT")

        # One-step (only on eval range)
        if "pred" in onestep_res and onestep_res["pred"].shape[0] > 0:
            s, e = onestep_res["eval_range"]
            ax.plot(np.arange(s, e), onestep_res["pred"][:, neuron_i],
                    color="steelblue", lw=0.7, alpha=0.8, label="1-step")

        # LOO
        if neuron_i in loo_res.get("pred", {}):
            loo_pred = loo_res["pred"][neuron_i]  # (T,) for this neuron
            ax.plot(t_ax, loo_pred, color="darkorange", lw=0.7, alpha=0.8,
                    label=f"LOO (R²={r2_loo[neuron_i]:.3f})")

        # Free-run (only for motor neurons in conditioned mode)
        r2_fr = np.array(freerun_res["r2"])
        if np.isfinite(r2_fr[neuron_i]):
            ax.plot(t_ax, fr_pred[:, neuron_i], color="seagreen", lw=0.7, alpha=0.8,
                    label=f"Free-run (R²={r2_fr[neuron_i]:.3f})")

        tag = " ★motor" if is_motor else ""
        ax.set_ylabel(f"{lbl}{tag}", fontsize=8)
        if ax_i == 0:
            ax.legend(fontsize=6, ncol=4, loc="upper right")

    axes[-1].set_xlabel("Time step")
    fig.suptitle(f"Neural Traces — Transformer Baseline ({WORM_ID})", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
#  PLOT 3:  Behaviour Comparison — Real vs Simulated
# ═════════════════════════════════════════════════════════════════════════════


def plot_behaviour_comparison(u_gt, u_freerun, b, b_mask, motor_idx,
                              labels, save_path, n_lags=8):
    """Compare decoded eigenworms from GT motor vs free-run motor activity."""
    T = u_gt.shape[0]
    n_modes = b.shape[1]
    mode_names = [f"EW{j+1}" for j in range(n_modes)]

    # Decode behaviour from GT motor neurons
    u_motor_gt = u_gt[:, motor_idx]
    beh_pred_gt = _fit_behaviour_decoder(u_motor_gt, b, b_mask, n_lags)

    # Decode behaviour from free-run motor neurons
    u_motor_fr = u_freerun[:, motor_idx]
    beh_pred_fr = _fit_behaviour_decoder(u_motor_fr, b, b_mask, n_lags)

    t_ax = np.arange(T)

    # ── Panel A: Eigenworm traces ──
    fig_traces = plt.figure(figsize=(14, 2.5 * min(n_modes, 4)))
    n_show = min(n_modes, 4)  # show first 4 eigenworms
    for j in range(n_show):
        ax = fig_traces.add_subplot(n_show, 1, j + 1)
        mask_j = b_mask[:, j] > 0.5

        # Real behaviour
        b_plot = b[:, j].copy()
        b_plot[~mask_j] = np.nan
        ax.plot(t_ax, b_plot, color="k", lw=0.8, alpha=0.8, label="Real")

        # Decoded from GT motor
        gt_dec = beh_pred_gt[:, j].copy()
        gt_dec[~mask_j | ~np.isfinite(gt_dec)] = np.nan
        ax.plot(t_ax, gt_dec, color="cornflowerblue", lw=0.7, alpha=0.8, label="GT motor→decoder")

        # Decoded from Free-run motor
        fr_dec = beh_pred_fr[:, j].copy()
        fr_dec[~mask_j | ~np.isfinite(fr_dec)] = np.nan
        ax.plot(t_ax, fr_dec, color="tomato", lw=0.7, alpha=0.8, label="Free-run motor→decoder")

        # R²
        valid = mask_j & np.isfinite(beh_pred_gt[:, j])
        r2_gt_j = _r2(b[valid, j], beh_pred_gt[valid, j]) if valid.sum() > 5 else np.nan
        valid_fr = mask_j & np.isfinite(beh_pred_fr[:, j])
        r2_fr_j = _r2(b[valid_fr, j], beh_pred_fr[valid_fr, j]) if valid_fr.sum() > 5 else np.nan

        ax.set_ylabel(f"{mode_names[j]}")
        ax.set_title(f"{mode_names[j]}:  GT-decoder R²={r2_gt_j:.3f}  |  "
                     f"Free-run-decoder R²={r2_fr_j:.3f}", fontsize=9)
        if j == 0:
            ax.legend(fontsize=7, ncol=3, loc="upper right")

    fig_traces.suptitle("Behaviour: Real vs Simulated Eigenworms", fontsize=11)
    fig_traces.tight_layout()
    trace_path = save_path.parent / "behaviour_traces.png"
    fig_traces.savefig(trace_path, bbox_inches="tight")
    plt.close(fig_traces)
    print(f"  Saved: {trace_path}")

    return beh_pred_gt, beh_pred_fr


def plot_behaviour_statistics(b, b_mask, beh_pred_gt, beh_pred_fr, save_path):
    """Statistical comparison of real vs simulated behaviour.

    Panel A: Amplitude distributions (histograms)
    Panel B: Autocorrelation functions
    Panel C: Power spectral density
    Panel D: Summary statistics table
    """
    n_modes = b.shape[1]
    n_show = min(n_modes, 4)
    mode_names = [f"EW{j+1}" for j in range(n_show)]

    fig = plt.figure(figsize=(16, 4 * n_show))
    gs = gridspec.GridSpec(n_show, 3, figure=fig, hspace=0.4, wspace=0.35)

    stats_table = []

    for j in range(n_show):
        mask_j = b_mask[:, j] > 0.5

        real = b[mask_j, j]
        gt_dec = beh_pred_gt[:, j]
        fr_dec = beh_pred_fr[:, j]
        gt_valid = mask_j & np.isfinite(gt_dec)
        fr_valid = mask_j & np.isfinite(fr_dec)
        gt_dec_v = gt_dec[gt_valid]
        fr_dec_v = fr_dec[fr_valid]

        # ── Column 1: Amplitude distribution ──
        ax_hist = fig.add_subplot(gs[j, 0])
        bins = np.linspace(
            min(np.nanmin(real), np.nanmin(gt_dec_v), np.nanmin(fr_dec_v)) - 0.5,
            max(np.nanmax(real), np.nanmax(gt_dec_v), np.nanmax(fr_dec_v)) + 0.5,
            40,
        )
        ax_hist.hist(real, bins=bins, density=True, alpha=0.5, color="k",
                     label="Real", edgecolor="k", linewidth=0.3)
        ax_hist.hist(fr_dec_v, bins=bins, density=True, alpha=0.5, color="tomato",
                     label="Free-run sim", edgecolor="k", linewidth=0.3)
        ax_hist.set_title(f"{mode_names[j]}: Amplitude Distribution")
        ax_hist.set_xlabel("Eigenworm value"); ax_hist.set_ylabel("Density")
        if j == 0:
            ax_hist.legend(fontsize=7)

        # ── Column 2: Autocorrelation ──
        ax_acf = fig.add_subplot(gs[j, 1])
        max_lag = min(200, len(real) // 2)

        def _acf(x, max_lag):
            x = x - np.nanmean(x)
            var = np.nanvar(x)
            if var < 1e-12:
                return np.zeros(max_lag)
            acf = np.correlate(x, x, mode="full")
            acf = acf[len(x) - 1:len(x) - 1 + max_lag]
            return acf / (var * len(x))

        acf_real = _acf(real, max_lag)
        acf_fr   = _acf(fr_dec_v, max_lag)
        lags = np.arange(max_lag)
        ax_acf.plot(lags, acf_real, color="k", lw=1, label="Real")
        ax_acf.plot(lags, acf_fr, color="tomato", lw=1, label="Free-run sim")
        ax_acf.set_title(f"{mode_names[j]}: Autocorrelation")
        ax_acf.set_xlabel("Lag (steps)"); ax_acf.set_ylabel("ACF")
        ax_acf.axhline(0, color="grey", ls=":", lw=0.5)
        if j == 0:
            ax_acf.legend(fontsize=7)

        # ── Column 3: Power Spectral Density ──
        ax_psd = fig.add_subplot(gs[j, 2])

        def _psd(x, nperseg=256):
            nperseg = min(nperseg, len(x))
            f, Pxx = signal.welch(x - np.mean(x), fs=1.0, nperseg=nperseg,
                                  noverlap=nperseg // 2)
            return f, Pxx

        f_real, psd_real = _psd(real)
        f_fr, psd_fr     = _psd(fr_dec_v)
        ax_psd.semilogy(f_real, psd_real, color="k", lw=1, label="Real")
        ax_psd.semilogy(f_fr, psd_fr, color="tomato", lw=1, label="Free-run sim")
        ax_psd.set_title(f"{mode_names[j]}: Power Spectrum")
        ax_psd.set_xlabel("Frequency (cycles/step)"); ax_psd.set_ylabel("PSD")
        if j == 0:
            ax_psd.legend(fontsize=7)

        # ── Collect statistics ──
        r2_val = _r2(b[fr_valid, j], fr_dec_v) if len(fr_dec_v) > 5 else np.nan
        ks_stat, ks_p = stats.ks_2samp(real, fr_dec_v)
        corr_acf = float(np.corrcoef(acf_real[:min(len(acf_real), len(acf_fr))],
                                      acf_fr[:min(len(acf_real), len(acf_fr))])[0, 1]) \
            if len(acf_real) > 2 and len(acf_fr) > 2 else np.nan

        stats_table.append({
            "mode": mode_names[j],
            "real_mean": float(np.mean(real)),
            "real_std": float(np.std(real)),
            "sim_mean": float(np.mean(fr_dec_v)),
            "sim_std": float(np.std(fr_dec_v)),
            "R2": float(r2_val),
            "KS_stat": float(ks_stat),
            "KS_pval": float(ks_p),
            "ACF_corr": float(corr_acf),
        })

    fig.suptitle("Behaviour Statistics: Real Worm vs Free-run Simulation", fontsize=12, y=1.01)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")

    return stats_table


def print_stats_table(stats_table):
    """Pretty-print the behaviour statistics comparison table."""
    header = (f"{'Mode':<6} {'Real μ':>8} {'Real σ':>8} {'Sim μ':>8} {'Sim σ':>8} "
              f"{'R²':>8} {'KS stat':>8} {'KS p':>8} {'ACF r':>8}")
    sep = "─" * len(header)
    print(f"\n{sep}")
    print("  Behaviour Statistics: Real Worm vs Free-run Simulation")
    print(sep)
    print(header)
    print(sep)
    for row in stats_table:
        print(f"{row['mode']:<6} {row['real_mean']:>8.3f} {row['real_std']:>8.3f} "
              f"{row['sim_mean']:>8.3f} {row['sim_std']:>8.3f} "
              f"{row['R2']:>8.3f} {row['KS_stat']:>8.3f} {row['KS_pval']:>8.3f} "
              f"{row['ACF_corr']:>8.3f}")
    print(sep)
    print("  R²      = variance explained of real behaviour by simulation-decoded signal")
    print("  KS stat = Kolmogorov-Smirnov distance (amplitude distribution mismatch)")
    print("  ACF r   = Pearson correlation of autocorrelation functions (temporal similarity)")
    print(sep)


# ═════════════════════════════════════════════════════════════════════════════
#  PLOT 4:  Stage-2 comparison (if available)
# ═════════════════════════════════════════════════════════════════════════════


def plot_stage2_comparison(u_gt, freerun_pred, motor_idx, labels, save_path):
    """Compare transformer free-run vs stage2 free-run for motor neurons."""
    try:
        import h5py
        with h5py.File(str(STAGE2_H5), "r") as f:
            u_s2 = np.array(f["stage2_pt/u_mean"])  # (T, N)
    except Exception as e:
        print(f"  Stage-2 results not found ({e}), skipping comparison.")
        return

    T, N = u_gt.shape
    n_motor = len(motor_idx) if motor_idx else 0
    if n_motor == 0:
        return

    n_show = min(n_motor, 6)
    show = motor_idx[:n_show]

    fig, axes = plt.subplots(n_show, 1, figsize=(14, 2.2 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]
    t_ax = np.arange(T)

    for ax_i, ni in enumerate(show):
        ax = axes[ax_i]
        lbl = labels[ni] if ni < len(labels) else f"N{ni}"

        ax.plot(t_ax, u_gt[:, ni], "k", lw=0.8, alpha=0.7, label="GT")
        ax.plot(t_ax, freerun_pred[:, ni], color="tomato", lw=0.7, alpha=0.8,
                label=f"Transformer FR (R²={_r2(u_gt[16:,ni], freerun_pred[16:,ni]):.3f})")
        if u_s2.shape == u_gt.shape:
            ax.plot(t_ax, u_s2[:, ni], color="mediumpurple", lw=0.7, alpha=0.8,
                    label=f"Stage-2 (R²={_r2(u_gt[:,ni], u_s2[:,ni]):.3f})")
        ax.set_ylabel(lbl, fontsize=8)
        if ax_i == 0:
            ax.legend(fontsize=7, ncol=3, loc="upper right")

    axes[-1].set_xlabel("Time step")
    fig.suptitle("Motor Neurons: Transformer Free-run vs Stage-2 Joint-CV", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════


def main():
    print("=" * 70)
    print("  Transformer Baseline — Plots & Behaviour Statistics")
    print("=" * 70)

    # ── Load ──
    model, cfg, worm = _load_model_and_data()
    u = worm["u"]
    T, N = u.shape
    K = cfg.context_length
    labels = worm["labels"]
    motor_idx = worm.get("motor_idx") or []
    b = worm.get("b")
    b_mask = worm.get("b_mask")

    with open(RESULTS_JSON) as f:
        eval_res = json.load(f)

    # ── 1. R² Summary ──
    print("\n[1/5] R² summary bar charts...")
    plot_r2_summary(eval_res, OUT_DIR / "r2_summary.png")

    # ── 2. Re-run predictions for traces ──
    print("\n[2/5] Computing neural traces (one-step, LOO, free-run)...")
    print("  One-step...")
    onestep = compute_onestep_r2(model, u)

    print("  LOO (20 neurons)...")
    loo_subset = eval_res["loo"].get("subset", list(range(min(20, N))))
    loo = compute_loo_r2(model, u, subset=loo_subset, verbose=True)

    print("  Free-run...")
    freerun = compute_free_run_r2(model, u, motor_idx=motor_idx if motor_idx else None)

    # ── 3. Neural trace panels ──
    print("\n[3/5] Plotting neural traces...")
    onestep["K"] = K
    plot_neural_traces(u, onestep, loo, freerun, labels, motor_idx,
                       OUT_DIR / "neural_traces.png", n_show=8)

    # ── 4. Behaviour comparison ──
    if b is not None and b_mask is not None and motor_idx:
        print("\n[4/5] Behaviour comparison (real vs simulated)...")
        beh_pred_gt, beh_pred_fr = plot_behaviour_comparison(
            u, freerun["pred"], b, b_mask, motor_idx, labels,
            OUT_DIR / "behaviour_traces.png", n_lags=cfg.behavior_lag_steps,
        )

        print("\n[5/5] Behaviour statistics...")
        stats_table = plot_behaviour_statistics(
            b, b_mask, beh_pred_gt, beh_pred_fr,
            OUT_DIR / "behaviour_statistics.png",
        )
        print_stats_table(stats_table)

        # Save stats as JSON
        stats_path = OUT_DIR / "behaviour_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats_table, f, indent=2)
        print(f"  Saved: {stats_path}")
    else:
        print("\n[4-5/5] Skipping behaviour (no behaviour data or motor neurons)")

    # ── 6. Stage-2 comparison ──
    print("\n[Bonus] Stage-2 comparison...")
    plot_stage2_comparison(u, freerun["pred"], motor_idx, labels,
                           OUT_DIR / "stage2_comparison.png")

    print("\n" + "=" * 70)
    print(f"  All plots saved to: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
