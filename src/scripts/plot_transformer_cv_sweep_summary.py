"""
Summary plots for the transformer CV sweep (4 configs × 3 worms, 5-fold CV).

Reads cv_results.json from the new sweep script that uses
joint neural + behaviour training and evaluation.

Generates:
  1. Ranked bar chart of LOO R² by config
  2. Multi-metric comparison (neural): 1-step, LOO, LOO-w50, free-run
  3. Behaviour metric comparison: direct, ridge-model, ridge-GT, all-neurons
  4. Per-worm heatmap: LOO R² (config × worm)
  5. Per-worm heatmap: behaviour direct R²
  6. Combined neural + behaviour radar / grouped bars

Usage:
    .venv/bin/python -m scripts.plot_transformer_cv_sweep_summary
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
SWEEP_DIR = BASE / "output_plots" / "transformer_baseline" / "cv_sweep"
OUT_DIR = SWEEP_DIR / "summary"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load():
    with open(SWEEP_DIR / "cv_results.json") as f:
        data = json.load(f)
    return data


def _nanmean(vals):
    arr = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.nanmean(arr)) if arr else float("nan")


# ======================================================================
# Plot 1: Ranked bar chart of LOO R² by config
# ======================================================================
def plot_ranked_loo(data):
    configs = data["configs"]
    worms = data["worms"]
    pw = data["per_worm_results"]

    # Per-config mean and per-worm values
    config_vals = {}
    for c in configs:
        cr = [r for r in pw if r["config"] == c]
        vals = [r["loo_r2"] for r in cr]
        config_vals[c] = vals

    # Sort by mean LOO R²
    order = sorted(configs, key=lambda c: -_nanmean(config_vals[c]))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(order))
    means = [_nanmean(config_vals[c]) for c in order]
    lows = [_nanmean(config_vals[c]) - min(config_vals[c]) for c in order]
    highs = [max(config_vals[c]) - _nanmean(config_vals[c]) for c in order]

    colors = plt.cm.viridis(np.linspace(0.8, 0.2, len(order)))
    ax.bar(x, means, color=colors, edgecolor="k", linewidth=0.5)
    ax.errorbar(x, means, yerr=[lows, highs], fmt="none", ecolor="k",
                capsize=4, linewidth=1)

    # Scatter individual worm values
    for i, c in enumerate(order):
        for v in config_vals[c]:
            ax.scatter(i, v, color="red", s=30, zorder=5, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylabel("LOO R² (mean ± range)")
    ax.set_title("5-Fold CV Architecture Sweep — LOO R² Ranking")
    ax.axhline(means[0], ls="--", color="red", alpha=0.3,
               label=f"best = {means[0]:.3f}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ranked_loo_r2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved ranked_loo_r2.png")


# ======================================================================
# Plot 2: Multi-metric comparison (neural)
# ======================================================================
def plot_neural_metrics(data):
    configs = data["configs"]
    pw = data["per_worm_results"]

    metrics = [
        ("onestep_r2", "1-Step R²"),
        ("loo_r2", "LOO R²"),
        ("loo_windowed_r2", "LOO-w50 R²"),
        ("freerun_r2", "Free-Run R²"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5),
                             sharey=False)
    width = 0.18
    x = np.arange(len(configs))
    colors = plt.cm.Set2(np.linspace(0, 0.8, len(configs)))

    for ax, (key, label) in zip(axes, metrics):
        means = []
        stds = []
        for c in configs:
            cr = [r for r in pw if r["config"] == c]
            vals = [r[key] for r in cr]
            means.append(_nanmean(vals))
            stds.append(float(np.nanstd(vals)) if len(vals) > 1 else 0)

        bars = ax.bar(x, means, color=colors, edgecolor="k", linewidth=0.5,
                      yerr=stds, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("5-Fold CV — Neural Metrics (mean ± std across worms)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "neural_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved neural_metrics.png")


# ======================================================================
# Plot 3: Behaviour metric comparison
# ======================================================================
def plot_behaviour_metrics(data):
    configs = data["configs"]
    pw = data["per_worm_results"]

    metrics = [
        ("beh_onestep_r2", "Beh 1-Step R²"),
        ("beh_direct_r2", "Beh Direct R²\n(model output)"),
        ("beh_ridge_model_r2", "Beh Ridge R²\n(model neurons)"),
        ("beh_ridge_gt_r2", "Beh Ridge R²\n(GT neurons)"),
        ("beh_all_neurons_r2", "Beh All-Neuron\n(upper bound)"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(3.5 * len(metrics), 5),
                             sharey=True)
    colors = plt.cm.Set2(np.linspace(0, 0.8, len(configs)))
    x = np.arange(len(configs))

    for ax, (key, label) in zip(axes, metrics):
        means = []
        stds = []
        for c in configs:
            cr = [r for r in pw if r["config"] == c]
            vals = [r.get(key, float("nan")) for r in cr]
            means.append(_nanmean(vals))
            stds.append(float(np.nanstd(vals)) if len(vals) > 1 else 0)

        ax.bar(x, means, color=colors, edgecolor="k", linewidth=0.5,
               yerr=stds, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("R²" if ax == axes[0] else "")
        ax.set_title(label, fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("5-Fold CV — Behaviour Metrics (mean ± std across worms)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "behaviour_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved behaviour_metrics.png")


# ======================================================================
# Plot 4: Per-worm heatmap — LOO R²
# ======================================================================
def plot_heatmap(data, key="loo_r2", title="LOO R²", fname="heatmap_loo_r2.png"):
    configs = data["configs"]
    worms = data["worms"]
    pw = data["per_worm_results"]

    mat = np.full((len(configs), len(worms)), np.nan)
    for i, c in enumerate(configs):
        for j, w in enumerate(worms):
            cr = [r for r in pw if r["config"] == c and r["worm"] == w]
            if cr:
                mat[i, j] = cr[0].get(key, float("nan"))

    fig, ax = plt.subplots(figsize=(max(5, len(worms) * 2), max(3, len(configs) * 0.8)))
    vmin = np.nanmin(mat) if np.any(np.isfinite(mat)) else 0
    vmax = np.nanmax(mat) if np.any(np.isfinite(mat)) else 1
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn",
                   vmin=vmin - 0.02, vmax=vmax + 0.02)
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(configs, fontsize=9)
    ax.set_xticks(range(len(worms)))
    ax.set_xticklabels([w[:10] for w in worms], fontsize=9)
    ax.set_xlabel("Worm")
    ax.set_ylabel("Config")
    ax.set_title(f"5-Fold CV — {title} per Config × Worm")

    for i in range(len(configs)):
        for j in range(len(worms)):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="k" if 0.3 < (v - vmin) / max(vmax - vmin, 1e-6) < 0.7 else "w")

    fig.colorbar(im, ax=ax, shrink=0.8, label=title)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


# ======================================================================
# Plot 5: Combined neural + behaviour grouped bar
# ======================================================================
def plot_combined_overview(data):
    configs = data["configs"]
    pw = data["per_worm_results"]

    metric_keys = [
        ("onestep_r2", "1-Step (N)"),
        ("loo_r2", "LOO (N)"),
        ("freerun_r2", "Free-Run (N)"),
        ("beh_direct_r2", "Beh Direct"),
        ("beh_ridge_model_r2", "Beh Ridge"),
    ]

    n_metrics = len(metric_keys)
    n_configs = len(configs)
    width = 0.15
    x = np.arange(n_metrics)

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.cm.tab10

    for ci, c in enumerate(configs):
        cr = [r for r in pw if r["config"] == c]
        means = []
        for key, _ in metric_keys:
            vals = [r.get(key, float("nan")) for r in cr]
            means.append(_nanmean(vals))
        offset = (ci - n_configs / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width=width,
                      color=cmap(ci / max(n_configs - 1, 1)),
                      edgecolor="k", linewidth=0.5, label=c)

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metric_keys], fontsize=10)
    ax.set_ylabel("R² (mean across worms)")
    ax.set_title("5-Fold CV — Architecture Comparison: Neural + Behaviour")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="k", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "combined_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved combined_overview.png")


# ======================================================================
# Plot 6: Per-worm grouped bars (for each worm, compare configs)
# ======================================================================
def plot_per_worm_detail(data):
    configs = data["configs"]
    worms = data["worms"]
    pw = data["per_worm_results"]

    metrics = [
        ("loo_r2", "LOO R²"),
        ("beh_direct_r2", "Beh Direct R²"),
    ]

    fig, axes = plt.subplots(len(metrics), len(worms),
                             figsize=(5 * len(worms), 4 * len(metrics)),
                             sharey="row")
    if len(worms) == 1:
        axes = axes.reshape(-1, 1)

    colors = plt.cm.Set2(np.linspace(0, 0.8, len(configs)))

    for mi, (key, label) in enumerate(metrics):
        for wi, w in enumerate(worms):
            ax = axes[mi, wi]
            vals = []
            for c in configs:
                cr = [r for r in pw if r["config"] == c and r["worm"] == w]
                v = cr[0].get(key, float("nan")) if cr else float("nan")
                vals.append(v)

            x = np.arange(len(configs))
            ax.bar(x, vals, color=colors, edgecolor="k", linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=8)
            if wi == 0:
                ax.set_ylabel(label)
            ax.set_title(f"{w[:10]} — {label}", fontsize=10)
            ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Per-Worm Detail — Config Comparison", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "per_worm_detail.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved per_worm_detail.png")


# ======================================================================
# main
# ======================================================================
def main():
    data = _load()
    configs = data["configs"]
    worms = data["worms"]
    n_folds = data["n_folds"]

    print(f"Generating CV sweep summary plots")
    print(f"  {len(configs)} configs × {len(worms)} worms × {n_folds} folds")
    print(f"  Behaviour: input={data.get('include_beh_input')}, "
          f"predict={data.get('predict_beh')}")
    print(f"  Output → {OUT_DIR}\n")

    plot_ranked_loo(data)
    plot_neural_metrics(data)
    plot_behaviour_metrics(data)
    plot_heatmap(data, "loo_r2", "LOO R²", "heatmap_loo_r2.png")
    plot_heatmap(data, "beh_direct_r2", "Behaviour Direct R²",
                 "heatmap_beh_direct_r2.png")
    plot_combined_overview(data)
    plot_per_worm_detail(data)

    n_plots = len(list(OUT_DIR.glob("*.png")))
    print(f"\nDone — {n_plots} plots saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
