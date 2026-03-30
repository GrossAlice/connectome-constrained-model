"""
Summary plots for the transformer architecture sweep (20 configs × 3 worms).

Generates:
  1. Ranked bar chart of LOO R² (mean ± per-worm spread)
  2. Metric comparison: LOO vs free-run vs 1-step across configs
  3. Per-worm heatmap of LOO R² (config × worm)
  4. NLL vs LOO R² scatter (colored by #params)
  5. Params vs performance scatter
  6. Per-worm scatter of LOO R² for all configs
"""

from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ── paths ────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
SWEEP_DIR = BASE / "output_plots" / "transformer_baseline" / "arch_sweep"
OUT_DIR = SWEEP_DIR / "summary"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── load data ────────────────────────────────────────────────────────────
with open(SWEEP_DIR / "aggregate.json") as f:
    agg = json.load(f)          # {config_name: {metric: value}}

with open(SWEEP_DIR / "sweep_results.json") as f:
    per_worm = json.load(f)     # list of dicts

with open(SWEEP_DIR / "ranked.json") as f:
    ranked = json.load(f)       # list sorted by rank

# overfit check (top 4 configs)
overfit_path = SWEEP_DIR / "overfit_check" / "overfit_results.json"
overfit = None
if overfit_path.exists():
    with open(overfit_path) as f:
        overfit = json.load(f)

# ── derived structures ───────────────────────────────────────────────────
configs = [r["config"] for r in ranked]   # sorted by rank
worms = sorted(set(r["worm_id"] for r in per_worm))

# per-worm data indexed by (config, worm)
pw = {}
for r in per_worm:
    pw[(r["config_name"], r["worm_id"])] = r

# ── style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})

COLORS = plt.cm.tab20(np.linspace(0, 1, 20))

# ======================================================================
# Plot 1: Ranked bar chart of LOO R² (mean ± range across worms)
# ======================================================================
def plot_ranked_loo():
    fig, ax = plt.subplots(figsize=(14, 5))
    means, lows, highs = [], [], []
    for c in configs:
        vals = [pw[(c, w)]["loo_r2"] for w in worms if (c, w) in pw]
        m = np.mean(vals)
        means.append(m)
        lows.append(m - min(vals))
        highs.append(max(vals) - m)

    x = np.arange(len(configs))
    colors = plt.cm.viridis(np.linspace(0.8, 0.2, len(configs)))
    bars = ax.bar(x, means, color=colors, edgecolor="k", linewidth=0.5)
    ax.errorbar(x, means, yerr=[lows, highs], fmt="none", ecolor="k",
                capsize=3, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("LOO R² (mean ± range)")
    ax.set_title("Transformer Architecture Sweep — LOO R² Ranking (3 worms)")
    ax.axhline(means[0], ls="--", color="red", alpha=0.4, label=f"best = {means[0]:.3f}")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ranked_loo_r2.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved ranked_loo_r2.png")


# ======================================================================
# Plot 2: Multi-metric comparison (LOO, free-run, 1-step, NLL)
# ======================================================================
def plot_metric_comparison():
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    metrics = [
        ("loo_r2_mean", "LOO R²"),
        ("free_run_r2_mean", "Free-Run R²"),
        ("onestep_r2_mean", "1-Step R²"),
        ("val_nll_mean", "Val NLL"),
    ]
    for ax, (key, label) in zip(axes.flat, metrics):
        vals = [agg[c][key] for c in configs]
        # NaN → 0
        vals = [v if not (isinstance(v, float) and math.isnan(v)) else 0 for v in vals]
        colors = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(configs))) if "nll" not in key.lower() else plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(configs)))
        
        x = np.arange(len(configs))
        ax.bar(x, vals, color=colors, edgecolor="k", linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=60, ha="right", fontsize=6)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Transformer Sweep — All Metrics (ranked by LOO R²)", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "metric_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved metric_comparison.png")


# ======================================================================
# Plot 3: Per-worm heatmap (config × worm) for LOO R²
# ======================================================================
def plot_per_worm_heatmap():
    mat = np.full((len(configs), len(worms)), np.nan)
    for i, c in enumerate(configs):
        for j, w in enumerate(worms):
            if (c, w) in pw:
                mat[i, j] = pw[(c, w)]["loo_r2"]

    fig, ax = plt.subplots(figsize=(6, 10))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0.25, vmax=0.55)
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(configs, fontsize=8)
    ax.set_xticks(range(len(worms)))
    ax.set_xticklabels([w[:10] for w in worms], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Worm")
    ax.set_ylabel("Config (ranked)")
    ax.set_title("LOO R² per Config × Worm")

    # annotate cells
    for i in range(len(configs)):
        for j in range(len(worms)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=7, color="k" if 0.3 < v < 0.5 else "w")

    fig.colorbar(im, ax=ax, shrink=0.5, label="LOO R²")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "per_worm_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved per_worm_heatmap.png")


# ======================================================================
# Plot 4: NLL vs LOO R² scatter (point size = #params)
# ======================================================================
def plot_nll_vs_loo():
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, c in enumerate(configs):
        a = agg[c]
        nll = a["val_nll_mean"]
        loo = a["loo_r2_mean"]
        params = a["n_params"]
        size = 30 + 150 * (params / max(agg[cc]["n_params"] for cc in configs))
        ax.scatter(nll, loo, s=size, c=[COLORS[i]], edgecolor="k",
                   linewidth=0.5, zorder=3, label=c)

    ax.set_xlabel("Validation NLL (lower = better)")
    ax.set_ylabel("LOO R² (higher = better)")
    ax.set_title("NLL vs LOO R²  (bubble size = #params)")
    ax.legend(fontsize=6, ncol=2, loc="upper right", bbox_to_anchor=(1.0, 1.0))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "nll_vs_loo_scatter.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved nll_vs_loo_scatter.png")


# ======================================================================
# Plot 5: Params vs performance (LOO R² and free-run R²)
# ======================================================================
def plot_params_vs_perf():
    fig, ax = plt.subplots(figsize=(9, 6))
    
    params_list = [agg[c]["n_params"] for c in configs]
    loo_list = [agg[c]["loo_r2_mean"] for c in configs]
    fr_list = [agg[c]["free_run_r2_mean"] for c in configs]

    ax.scatter(params_list, loo_list, s=80, marker="o", c="tab:blue",
               edgecolor="k", linewidth=0.5, label="LOO R²", zorder=3)
    ax.scatter(params_list, fr_list, s=80, marker="s", c="tab:orange",
               edgecolor="k", linewidth=0.5, label="Free-Run R²", zorder=3)

    for c, p, loo, fr in zip(configs, params_list, loo_list, fr_list):
        ax.annotate(c, (p, loo), fontsize=5.5, ha="center", va="bottom",
                    xytext=(0, 4), textcoords="offset points", alpha=0.8)

    ax.set_xlabel("Number of Parameters")
    ax.set_ylabel("R²")
    ax.set_title("Model Size vs Performance")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "params_vs_perf.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved params_vs_perf.png")


# ======================================================================
# Plot 6: Per-worm LOO R² scatter for all configs
# ======================================================================
def plot_per_worm_scatter():
    fig, axes = plt.subplots(1, len(worms), figsize=(5 * len(worms), 6),
                             sharey=True)
    if len(worms) == 1:
        axes = [axes]

    for ax, w in zip(axes, worms):
        vals = []
        for c in configs:
            if (c, w) in pw:
                vals.append((c, pw[(c, w)]["loo_r2"]))
        names, ys = zip(*vals)
        colors = plt.cm.viridis(np.linspace(0.8, 0.2, len(names)))
        ax.barh(range(len(names)), ys, color=colors, edgecolor="k", linewidth=0.3)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("LOO R²")
        ax.set_title(w, fontsize=10)
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()

    fig.suptitle("LOO R² per Worm (all 20 configs)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "per_worm_scatter.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved per_worm_scatter.png")


# ======================================================================
# Plot 7: Top-5 highlight with optional overfit overlay
# ======================================================================
def plot_top5_highlight():
    top5 = configs[:5]
    metrics = ["loo_r2", "free_run_r2", "onestep_r2"]
    labels = ["LOO R²", "Free-Run R²", "1-Step R²"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    width = 0.15
    x = np.arange(len(worms))

    for ax, metric, lab in zip(axes, metrics, labels):
        for k, c in enumerate(top5):
            vals = [pw[(c, w)].get(metric, 0) for w in worms]
            offset = (k - 2) * width
            ax.bar(x + offset, vals, width=width, label=c, edgecolor="k",
                   linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels([w[:10] for w in worms], fontsize=8)
        ax.set_ylabel(lab)
        ax.set_title(lab)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Top-5 Configs — Per-Worm Breakdown", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "top5_per_worm.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved top5_per_worm.png")


# ======================================================================
# Plot 8: Metric correlation matrix (LOO vs free-run vs NLL vs params)
# ======================================================================
def plot_metric_correlations():
    keys = ["loo_r2_mean", "free_run_r2_mean", "onestep_r2_mean", "val_nll_mean"]
    labels = ["LOO R²", "Free-Run R²", "1-Step R²", "Val NLL"]
    n = len(keys)

    fig, axes = plt.subplots(n, n, figsize=(12, 12))
    for i in range(n):
        for j in range(n):
            ax = axes[i][j]
            xi = [agg[c][keys[j]] for c in configs]
            yi = [agg[c][keys[i]] for c in configs]
            # filter NaN
            valid = [(x, y) for x, y in zip(xi, yi) if not (math.isnan(x) or math.isnan(y))]
            if not valid:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                continue
            vx, vy = zip(*valid)
            if i == j:
                ax.hist(vx, bins=10, color="steelblue", edgecolor="k", linewidth=0.3)
            else:
                ax.scatter(vx, vy, s=25, c="steelblue", edgecolor="k", linewidth=0.3)
                # correlation
                if len(vx) > 2:
                    r = np.corrcoef(vx, vy)[0, 1]
                    ax.set_title(f"r={r:.2f}", fontsize=8, pad=2)
            if i == n - 1:
                ax.set_xlabel(labels[j], fontsize=8)
            if j == 0:
                ax.set_ylabel(labels[i], fontsize=8)
            ax.tick_params(labelsize=6)

    fig.suptitle("Metric Correlation Matrix (20 configs)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "metric_correlations.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved metric_correlations.png")


# ======================================================================
# main
# ======================================================================
if __name__ == "__main__":
    print(f"Generating summary plots for {len(configs)} configs × {len(worms)} worms")
    print(f"Output → {OUT_DIR}\n")

    plot_ranked_loo()
    plot_metric_comparison()
    plot_per_worm_heatmap()
    plot_nll_vs_loo()
    plot_params_vs_perf()
    plot_per_worm_scatter()
    plot_top5_highlight()
    plot_metric_correlations()

    print(f"\nDone — {len(list(OUT_DIR.glob('*.png')))} plots saved to {OUT_DIR}")
