#!/usr/bin/env python
"""
Behaviour statistics for top-1 worm decoder predictions.
3 panels per mode (amplitude distribution, ACF, PSD) × K modes.
Compare GT vs Ridge, MLP, AR2+MLP.
"""
import sys, pathlib, numpy as np
from pathlib import Path

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

# ── Config ────────────────────────────────────────────────────────
OUT_DIR = ROOT / "output_plots/behaviour_decoder/top1_video"
PRED_PATH = OUT_DIR / "predictions.npz"
K = 6
WARMUP = 8
DT = 0.6  # seconds per step
MODELS = ["Ridge", "MLP", "AR2+MLP"]
COLORS = {"GT": "black", "Ridge": "#1b9e77", "MLP": "#d95f02",
          "AR2+MLP": "#7570b3"}

# ── Load ──────────────────────────────────────────────────────────
d = np.load(str(PRED_PATH))
b_true = d["b_true"]
preds = {m: d[m] for m in MODELS}
T = b_true.shape[0]

# Use only valid (non-NaN) region
valid = np.arange(WARMUP, T)
gt = b_true[valid]

# For each model, pick valid timesteps
ho = {}
for m in MODELS:
    p = preds[m][valid]
    ok = np.all(np.isfinite(p), axis=1)
    ho[m] = p  # keep full array, mask NaNs below

# ── Helpers ───────────────────────────────────────────────────────
def acf(x, max_lag=200):
    """Normalised autocorrelation."""
    x = x - x.mean()
    n = len(x)
    ml = min(max_lag, n - 1)
    c = np.correlate(x, x, mode="full")
    c = c[n - 1:]  # positive lags
    return c[:ml + 1] / c[0] if c[0] > 0 else np.zeros(ml + 1)


def compute_psd(x, dt):
    """Welch PSD."""
    fs = 1.0 / dt
    nperseg = min(256, len(x) // 2)
    f, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    return f, Pxx


# ── Figure: 3 columns × K rows ───────────────────────────────────
fig, axes = plt.subplots(K, 3, figsize=(15, 2.8 * K), constrained_layout=True)
fig.suptitle("Behaviour Statistics: GT vs Decoders — 2023-01-17-14",
             fontsize=14, fontweight="bold")

col_titles = ["Amplitude Distribution", "Autocorrelation", "Power Spectrum"]
max_lag = 200

for j in range(K):
    ax_hist, ax_acf, ax_psd = axes[j]

    # GT
    g = gt[:, j]
    ok_g = np.isfinite(g)
    g_clean = g[ok_g]

    # Column titles (top row only)
    if j == 0:
        for ci, title in enumerate(col_titles):
            axes[0, ci].set_title(title, fontsize=11, fontweight="bold")

    # ── Histogram ─────────────────────────────────────────────────
    bins = np.linspace(g_clean.min() - 0.5, g_clean.max() + 0.5, 50)
    ax_hist.hist(g_clean, bins=bins, density=True, alpha=0.5,
                 color="gray", label="GT", edgecolor="none")
    for m in MODELS:
        p = ho[m][:, j]
        ok = np.isfinite(p)
        ax_hist.hist(p[ok], bins=bins, density=True, alpha=0.35,
                     color=COLORS[m], label=m, edgecolor="none")
    ax_hist.set_ylabel("Density", fontsize=9)
    ax_hist.set_xlabel(f"a{j+1} value", fontsize=9)
    ax_hist.text(0.02, 0.95, f"EW{j+1}", transform=ax_hist.transAxes,
                 fontsize=10, fontweight="bold", va="top")
    if j == 0:
        ax_hist.legend(fontsize=7, loc="upper right")

    # Variance ratio annotation
    gt_std = g_clean.std()
    var_text = "σ ratio: " + ", ".join(
        f"{ho[m][np.isfinite(ho[m][:,j]),j].std() / gt_std:.2f}"
        for m in MODELS)
    ax_hist.text(0.98, 0.88, var_text, transform=ax_hist.transAxes,
                 fontsize=6.5, ha="right", va="top", color="gray")

    # ── ACF ───────────────────────────────────────────────────────
    lags = np.arange(max_lag + 1)
    lag_time = lags * DT

    acf_gt = acf(g_clean, max_lag)
    ax_acf.plot(lag_time, acf_gt, "k-", lw=1.5, alpha=0.8, label="GT")
    for m in MODELS:
        p = ho[m][:, j]
        ok = np.isfinite(p)
        acf_m = acf(p[ok], max_lag)
        ax_acf.plot(lag_time, acf_m, "-", color=COLORS[m], lw=1,
                    alpha=0.8, label=m)
    ax_acf.axhline(0, color="gray", lw=0.5, ls="--")
    ax_acf.set_ylabel("ACF", fontsize=9)
    ax_acf.set_xlabel("Lag (s)", fontsize=9)
    ax_acf.set_xlim(0, max_lag * DT)
    ax_acf.text(0.02, 0.95, f"EW{j+1}", transform=ax_acf.transAxes,
                fontsize=10, fontweight="bold", va="top")
    if j == 0:
        ax_acf.legend(fontsize=7, loc="upper right")

    # ── PSD ───────────────────────────────────────────────────────
    f_gt, psd_gt = compute_psd(g_clean, DT)
    ax_psd.semilogy(f_gt, psd_gt, "k-", lw=1.5, alpha=0.8, label="GT")
    for m in MODELS:
        p = ho[m][:, j]
        ok = np.isfinite(p)
        f_m, psd_m = compute_psd(p[ok], DT)
        ax_psd.semilogy(f_m, psd_m, "-", color=COLORS[m], lw=1,
                        alpha=0.8, label=m)
    ax_psd.set_ylabel("PSD", fontsize=9)
    ax_psd.set_xlabel("Frequency (Hz)", fontsize=9)
    ax_psd.text(0.02, 0.95, f"EW{j+1}", transform=ax_psd.transAxes,
                fontsize=10, fontweight="bold", va="top")
    if j == 0:
        ax_psd.legend(fontsize=7, loc="upper right")

out_png = OUT_DIR / "behaviour_stats.png"
fig.savefig(str(out_png), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_png}")

# ── Summary table ─────────────────────────────────────────────────
print("\n── Variance ratio (σ_pred / σ_GT) ──")
print(f"{'Model':14s}", "  ".join(f"  a{j+1}" for j in range(K)), "  mean")
for m in MODELS:
    ratios = []
    for j in range(K):
        p = ho[m][:, j]
        ok = np.isfinite(p)
        ratios.append(p[ok].std() / gt[ok, j].std())
    print(f"{m:14s}", "  ".join(f"{r:.3f}" for r in ratios),
          f"  {np.mean(ratios):.3f}")
