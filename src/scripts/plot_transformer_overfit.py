#!/usr/bin/env python3
"""Plot train-vs-test comparison for top transformer configs.

Reads:
  - overfit_results.json  (per-config × per-worm R² on train/test regions)
  - history.json          (per-epoch train/val NLL curves)

Produces 4 figures:
  1. train_vs_test_bars.png       – grouped bar chart of train/test R² by metric
  2. nll_curves.png               – train/val NLL learning curves per model
  3. metric_gap_heatmap.png       – heatmap of (train − test) gap per metric/config
  4. per_worm_scatter.png         – scatter of train-R² vs test-R² coloured by config
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Colour palette (one per config) ──────────────────────────────────
CONFIG_COLOURS = {
    "B_wide_256h8": "#1f77b4",
    "A_baseline":   "#ff7f0e",
    "E_deep_wide":  "#2ca02c",
    "I_ctx8":       "#d62728",
}
METRIC_LABELS = {
    "onestep": "One-step R²",
    "loo":     "LOO R²",
    "freerun": "Free-run R²",
}


def _load(sweep_dir: Path):
    results = json.load(open(sweep_dir / "overfit_check" / "overfit_results.json"))
    configs = sorted(set(r["config"] for r in results),
                     key=lambda c: -np.mean([r["loo_test"] for r in results if r["config"] == c]))
    worms = sorted(set(r["worm"] for r in results))

    # Load training histories
    histories = {}
    for r in results:
        hpath = sweep_dir / r["config"] / r["worm"] / "history.json"
        if hpath.exists():
            histories[(r["config"], r["worm"])] = json.load(open(hpath))
    return results, configs, worms, histories


# ── Figure 1: Grouped bars ──────────────────────────────────────────

def plot_train_vs_test_bars(results, configs, out_path: Path):
    """Grouped bar chart: for each metric, show train/test R² per config."""
    metrics = [
        ("onestep_train", "onestep_test", "One-step R²"),
        ("loo_train",     "loo_test",     "LOO R²"),
        ("freerun_train", "freerun_test_fresh", "Free-run R²"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    bar_w = 0.18
    x = np.arange(len(configs))

    for ax, (tr_key, te_key, title) in zip(axes, metrics):
        for i, cfg in enumerate(configs):
            recs = [r for r in results if r["config"] == cfg]
            tr_vals = [r[tr_key] for r in recs]
            te_vals = [r[te_key] for r in recs]
            tr_m, tr_s = np.mean(tr_vals), np.std(tr_vals)
            te_m, te_s = np.mean(te_vals), np.std(te_vals)

            c = CONFIG_COLOURS.get(cfg, "gray")
            ax.bar(i - bar_w/2, tr_m, bar_w, yerr=tr_s, capsize=3,
                   color=c, alpha=0.85, label="Train" if i == 0 else "")
            ax.bar(i + bar_w/2, te_m, bar_w, yerr=te_s, capsize=3,
                   color=c, alpha=0.35, hatch="//",
                   label="Test" if i == 0 else "")

            # Annotation: Δ
            gap = tr_m - te_m
            y_top = max(tr_m + tr_s, te_m + te_s) + 0.03
            ax.text(i, y_top, f"Δ={gap:+.2f}", ha="center", fontsize=8, color="k")

        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("_", "\n") for c in configs],
                           fontsize=8, rotation=0)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel("R²")
        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
        ax.set_ylim(min(-0.5, ax.get_ylim()[0]), max(0.85, ax.get_ylim()[1]))
        if ax == axes[0]:
            # Manual legend for train (solid) vs test (hatched)
            from matplotlib.patches import Patch
            ax.legend(handles=[
                Patch(facecolor="gray", alpha=0.85, label="Train"),
                Patch(facecolor="gray", alpha=0.35, hatch="//", label="Test"),
            ], fontsize=9, loc="upper right")

    fig.suptitle("Transformer Train vs Test R²  (mean ± std over 3 worms)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Figure 2: NLL learning curves ───────────────────────────────────

def plot_nll_curves(histories, configs, worms, out_path: Path):
    """Train/val NLL curves per config (one subplot per config, lines per worm)."""
    n_cfg = len(configs)
    fig, axes = plt.subplots(1, n_cfg, figsize=(5 * n_cfg, 4), sharey=True)
    if n_cfg == 1:
        axes = [axes]

    worm_styles = ["-", "--", ":"]

    for ax, cfg in zip(axes, configs):
        for j, worm in enumerate(worms):
            h = histories.get((cfg, worm))
            if h is None:
                continue
            epochs = [e["epoch"] for e in h]
            tr_nll = [e["train_nll"] for e in h]
            va_nll = [e["val_nll"] for e in h]
            ls = worm_styles[j % len(worm_styles)]
            c = CONFIG_COLOURS.get(cfg, "gray")
            ax.plot(epochs, tr_nll, ls=ls, color=c, alpha=0.8, lw=1.2,
                    label=f"{worm[:10]} train")
            ax.plot(epochs, va_nll, ls=ls, color=c, alpha=0.4, lw=1.8,
                    label=f"{worm[:10]} val")

        ax.set_title(cfg.replace("_", " "), fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylim(-0.3, 1.5)
        ax.axhline(0, color="k", lw=0.3, ls="--", alpha=0.3)

        # Add best-epoch marker
        for j, worm in enumerate(worms):
            h = histories.get((cfg, worm))
            if h is None:
                continue
            va_nll = [e["val_nll"] for e in h]
            best_ep = h[int(np.argmin(va_nll))]["epoch"]
            best_val = min(va_nll)
            ax.plot(best_ep, best_val, "v", color="k", ms=6, zorder=5)

    axes[0].set_ylabel("NLL (Gaussian)")

    # Build a compact legend
    from matplotlib.lines import Line2D
    handles = []
    for j, worm in enumerate(worms):
        handles.append(Line2D([0], [0], color="gray", ls=worm_styles[j],
                              lw=1.2, label=worm))
    handles.append(Line2D([0], [0], color="gray", alpha=0.85, lw=1.2,
                          label="Train (opaque)"))
    handles.append(Line2D([0], [0], color="gray", alpha=0.4, lw=1.8,
                          label="Val (faded)"))
    handles.append(Line2D([0], [0], marker="v", color="k", ls="", ms=6,
                          label="Best epoch"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               fontsize=8, bbox_to_anchor=(0.5, 1.08))

    fig.suptitle("Training curves: NLL (train vs val)",
                 fontsize=13, fontweight="bold", y=1.14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Figure 3: Gap heatmap ───────────────────────────────────────────

def plot_gap_heatmap(results, configs, worms, out_path: Path):
    """Heatmap of (train − test) R² gap per config × metric."""
    metric_pairs = [
        ("onestep_train", "onestep_test",         "One-step"),
        ("loo_train",     "loo_test",              "LOO"),
        ("freerun_train", "freerun_test_fresh",    "Free-run"),
        ("train_nll_best", "val_nll_best",         "NLL gap"),
    ]

    # Build matrix: rows=configs, cols=metrics, values=mean gap over worms
    n_metrics = len(metric_pairs)
    n_cfgs = len(configs)
    gap_matrix = np.zeros((n_cfgs, n_metrics))
    gap_per_worm = np.zeros((n_cfgs, n_metrics, len(worms)))

    for i, cfg in enumerate(configs):
        recs = [r for r in results if r["config"] == cfg]
        for j, (tr_k, te_k, _) in enumerate(metric_pairs):
            for w_idx, rec in enumerate(sorted(recs, key=lambda r: r["worm"])):
                gap_per_worm[i, j, w_idx] = rec[tr_k] - rec[te_k]
            gap_matrix[i, j] = np.mean(gap_per_worm[i, j])

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(gap_matrix, cmap="RdYlGn_r", aspect="auto",
                   vmin=0, vmax=max(1.0, gap_matrix.max()))

    ax.set_xticks(range(n_metrics))
    ax.set_xticklabels([m[2] for m in metric_pairs], fontsize=10)
    ax.set_yticks(range(n_cfgs))
    ax.set_yticklabels([c.replace("_", "\n") for c in configs], fontsize=9)

    # Annotate cells
    for i in range(n_cfgs):
        for j in range(n_metrics):
            val = gap_matrix[i, j]
            std = np.std(gap_per_worm[i, j])
            txt = f"{val:.3f}\n±{std:.3f}"
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)

    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("Train − Test (gap)", fontsize=10)
    ax.set_title("Overfitting gap: (Train metric) − (Test metric)\nLarger = more overfit",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Figure 4: Scatter train vs test ─────────────────────────────────

def plot_scatter(results, configs, out_path: Path):
    """Scatter plot: train-R² (x) vs test-R² (y), one point per config×worm."""
    metric_pairs = [
        ("onestep_train", "onestep_test",         "One-step R²"),
        ("loo_train",     "loo_test",              "LOO R²"),
        ("freerun_train", "freerun_test_fresh",    "Free-run R²"),
    ]

    markers = {"2023-01-17-14": "o", "2022-06-14-07": "s", "2023-01-10-07": "D"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (tr_k, te_k, title) in zip(axes, metric_pairs):
        for r in results:
            c = CONFIG_COLOURS.get(r["config"], "gray")
            m = markers.get(r["worm"], "o")
            ax.scatter(r[tr_k], r[te_k], c=c, marker=m, s=80, edgecolors="k",
                       linewidths=0.5, zorder=5)

        # Diagonal y=x line
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0], -0.5)
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1], 0.8)
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.4, label="y = x")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("Train R²", fontsize=11)
        ax.set_ylabel("Test R²", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axhline(0, color="gray", lw=0.3, ls=":")
        ax.axvline(0, color="gray", lw=0.3, ls=":")

    # Combined legend
    from matplotlib.lines import Line2D
    handles = []
    for cfg, c in CONFIG_COLOURS.items():
        if cfg in set(r["config"] for r in results):
            handles.append(Line2D([0], [0], marker="o", color="w",
                                  markerfacecolor=c, markeredgecolor="k",
                                  markersize=8, label=cfg))
    for worm, m in markers.items():
        handles.append(Line2D([0], [0], marker=m, color="w",
                              markerfacecolor="gray", markeredgecolor="k",
                              markersize=8, label=worm))
    handles.append(Line2D([0], [0], ls="--", color="k", alpha=0.4, label="y = x"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               fontsize=8, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Train R² vs Test R²  (points below diagonal = overfit)",
                 fontsize=13, fontweight="bold", y=1.14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Figure 5: Per-worm detailed breakdown ───────────────────────────

def plot_per_worm_detail(results, configs, worms, out_path: Path):
    """Per-worm stacked bar comparison for each metric."""
    metrics = [
        ("onestep_train", "onestep_test", "One-step R²"),
        ("loo_train",     "loo_test",     "LOO R²"),
        ("freerun_train", "freerun_test_fresh", "Free-run R²"),
    ]

    n_worms = len(worms)
    n_cfgs = len(configs)
    fig, axes = plt.subplots(n_worms, 3, figsize=(16, 4 * n_worms),
                             sharey="col")

    bar_w = 0.35
    x = np.arange(n_cfgs)

    for row, worm in enumerate(worms):
        for col, (tr_k, te_k, title) in enumerate(metrics):
            ax = axes[row, col] if n_worms > 1 else axes[col]
            tr_vals, te_vals = [], []
            for cfg in configs:
                rec = next(r for r in results
                           if r["config"] == cfg and r["worm"] == worm)
                tr_vals.append(rec[tr_k])
                te_vals.append(rec[te_k])

            bars_tr = ax.bar(x - bar_w/2, tr_vals, bar_w,
                             color=[CONFIG_COLOURS.get(c, "gray") for c in configs],
                             alpha=0.85, edgecolor="k", linewidth=0.5)
            bars_te = ax.bar(x + bar_w/2, te_vals, bar_w,
                             color=[CONFIG_COLOURS.get(c, "gray") for c in configs],
                             alpha=0.35, hatch="//", edgecolor="k", linewidth=0.5)

            # Value labels on bars
            for b in bars_tr:
                h = b.get_height()
                ax.text(b.get_x() + b.get_width()/2, h + 0.01,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=7)
            for b in bars_te:
                h = b.get_height()
                y = h + 0.01 if h >= 0 else h - 0.04
                ax.text(b.get_x() + b.get_width()/2, y,
                        f"{h:.2f}", ha="center",
                        va="bottom" if h >= 0 else "top", fontsize=7)

            ax.set_xticks(x)
            ax.set_xticklabels([c.replace("_", "\n") for c in configs],
                               fontsize=7)
            ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.3)
            if col == 0:
                ax.set_ylabel(worm, fontsize=10, fontweight="bold")
            if row == 0:
                ax.set_title(title, fontsize=11, fontweight="bold")

    from matplotlib.patches import Patch
    fig.legend(handles=[
        Patch(facecolor="gray", alpha=0.85, edgecolor="k", label="Train"),
        Patch(facecolor="gray", alpha=0.35, hatch="//", edgecolor="k",
              label="Test"),
    ], loc="upper right", fontsize=9)

    fig.suptitle("Per-worm Train vs Test R² breakdown",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── main ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir",
                    default="output_plots/transformer_baseline/arch_sweep")
    args = ap.parse_args()

    sweep = Path(args.sweep_dir)
    out = sweep / "overfit_check"
    out.mkdir(parents=True, exist_ok=True)

    results, configs, worms, histories = _load(sweep)
    print(f"Loaded {len(results)} records: {len(configs)} configs × {len(worms)} worms")

    plot_train_vs_test_bars(results, configs, out / "train_vs_test_bars.png")
    plot_nll_curves(histories, configs, worms, out / "nll_curves.png")
    plot_gap_heatmap(results, configs, worms, out / "metric_gap_heatmap.png")
    plot_scatter(results, configs, out / "per_config_scatter.png")
    plot_per_worm_detail(results, configs, worms, out / "per_worm_detail.png")

    print("\nAll plots saved to", out)


if __name__ == "__main__":
    main()
