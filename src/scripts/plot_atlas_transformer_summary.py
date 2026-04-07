#!/usr/bin/env python3
"""
Summary plots for two atlas-transformer multi-worm runs.

Compares:
  - bg_20260331_allworms      (input_mode = default / concat)
  - bg_20260331_multiply_project  (input_mode = multiply_project)

Produces a multi-page PDF + individual PNGs:
  1. Per-worm bar chart of core neural metrics  (LOO, LOO-windowed, one-step, free-run)
  2. Per-worm bar chart of behaviour metrics    (direct, ridge→GT, MLP→GT)
  3. Box / strip comparison across all worms for each metric
  4. Scatter of N_obs vs LOO R² (do more observed neurons help?)
  5. Per-neuron LOO R² distribution for both runs (pooled across worms)

Usage:
    python -m scripts.plot_atlas_transformer_summary
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = ROOT / "output_plots" / "atlas_transformer"
RUNS = {
    "concat (default)":       RUN_DIR / "bg_20260331_allworms",
    "multiply_project": RUN_DIR / "bg_20260331_multiply_project",
}
SAVE_DIR = RUN_DIR / "summary_comparison"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ── helpers ────────────────────────────────────────────────────────────
def _load_worm(worm_dir: Path) -> dict[str, Any] | None:
    fp = worm_dir / "eval_results.json"
    if not fp.exists():
        return None
    with open(fp) as f:
        return json.load(f)


def load_run(root: Path) -> pd.DataFrame:
    """Return a DataFrame with one row per worm and columns for every metric."""
    rows = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        res = _load_worm(d)
        if res is None:
            continue
        row: dict[str, Any] = {
            "worm_id":   res["worm_id"],
            "N_obs":     res["N_obs"],
            "T":         res["T"],
            "n_folds":   res.get("n_folds", np.nan),
        }
        # neural metrics
        row["onestep_r2"]      = res["onestep"]["r2_mean"]
        row["loo_r2"]          = res["loo"]["r2_mean"]
        row["loo_windowed_r2"] = res["loo_windowed"]["r2_mean"]
        row["free_run_r2"]     = res["free_run"]["r2_mean"]
        row["loo_n_neurons"]   = int(np.sum(~np.isnan(res["loo"]["r2"])))
        # behaviour
        row["beh_direct_r2"]   = res["behaviour_direct"]["r2_mean"]
        row["beh_ridge_r2_gt"] = res["behaviour_ridge"]["r2_gt_mean"]
        row["beh_mlp_r2_gt"]   = res["behaviour_mlp"]["r2_gt_mean"]
        row["beh_ridge_r2_model"] = res["behaviour_ridge"]["r2_model_mean"]
        row["beh_mlp_r2_model"]   = res["behaviour_mlp"]["r2_model_mean"]
        # per-neuron LOO R² (keep raw list for distribution plot)
        row["_loo_per_neuron"] = res["loo"]["r2"]
        row["_loo_w_per_neuron"] = res["loo_windowed"]["r2"]
        rows.append(row)
    return pd.DataFrame(rows)


# ── plotting ───────────────────────────────────────────────────────────
NEURAL_METRICS = [
    ("loo_r2",          "LOO R²"),
    ("loo_windowed_r2", "LOO-windowed R²"),
    ("onestep_r2",      "One-step R²"),
    ("free_run_r2",     "Free-run R²"),
]

BEH_METRICS = [
    ("beh_direct_r2",      "Behaviour direct R²"),
    ("beh_ridge_r2_gt",    "Behaviour ridge→GT R²"),
    ("beh_mlp_r2_gt",      "Behaviour MLP→GT R²"),
]

PALETTE = {
    "concat (default)":       "#1f77b4",
    "multiply_project": "#ff7f0e",
}


def _short_worm(wid: str) -> str:
    """Shorten '2022-08-02-01' → '08-02-01'."""
    parts = wid.split("-")
    if len(parts) >= 4:
        return "-".join(parts[1:])
    return wid


def fig_per_worm_bars(dfs: dict[str, pd.DataFrame], metrics: list, title: str,
                      fname: str):
    """Grouped bar chart: one bar-group per worm, one colour per run."""
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(20, 3.5 * n_metrics),
                             sharex=True)
    if n_metrics == 1:
        axes = [axes]

    # union of worm_ids sorted
    all_worms = sorted(
        set().union(*(set(df["worm_id"]) for df in dfs.values()))
    )
    x = np.arange(len(all_worms))
    width = 0.35
    offsets = {name: (i - 0.5) * width for i, name in enumerate(dfs)}

    for ax, (col, label) in zip(axes, metrics):
        for run_name, df in dfs.items():
            vals = []
            for w in all_worms:
                row = df.loc[df["worm_id"] == w]
                vals.append(row[col].values[0] if len(row) else np.nan)
            vals = np.array(vals, dtype=float)
            ax.bar(x + offsets[run_name], vals, width * 0.9,
                   label=run_name, color=PALETTE[run_name], alpha=0.85)
        ax.set_ylabel(label, fontsize=11)
        ax.axhline(0, color="grey", lw=0.5, ls="--")
        ax.legend(fontsize=9)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([_short_worm(w) for w in all_worms],
                              rotation=70, ha="right", fontsize=8)
    fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(SAVE_DIR / fname, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {SAVE_DIR / fname}")


def fig_box_comparison(dfs: dict[str, pd.DataFrame]):
    """Box + strip side-by-side for each neural metric."""
    all_metrics = NEURAL_METRICS + BEH_METRICS
    n = len(all_metrics)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 5))

    run_names = list(dfs.keys())
    for ax, (col, label) in zip(axes, all_metrics):
        data = []
        positions = []
        colours = []
        for i, name in enumerate(run_names):
            vals = dfs[name][col].dropna().values
            data.append(vals)
            positions.append(i)
            colours.append(PALETTE[name])

        bps = ax.boxplot(data, positions=positions, widths=0.5,
                         patch_artist=True, showfliers=False)
        for patch, c in zip(bps["boxes"], colours):
            patch.set_facecolor(c)
            patch.set_alpha(0.4)
        for i, (name, vals) in enumerate(zip(run_names, data)):
            jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
            ax.scatter(np.full_like(vals, i) + jitter, vals,
                       s=18, alpha=0.7, color=PALETTE[name], zorder=3,
                       edgecolors="white", linewidths=0.3)
            median = np.median(vals)
            ax.text(i, median + 0.02, f"{median:.3f}", ha="center",
                    fontsize=7, fontweight="bold", color=PALETTE[name])
        ax.set_xticks(positions)
        ax.set_xticklabels([n.replace(" (default)", "\n(default)") for n in run_names],
                           fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))

    fig.suptitle("Atlas Transformer — metric distributions across worms",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fname = "box_comparison.png"
    fig.savefig(SAVE_DIR / fname, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {SAVE_DIR / fname}")


def fig_nobs_vs_loo(dfs: dict[str, pd.DataFrame]):
    """Scatter N_obs vs LOO R² coloured by run."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (col, label) in zip(axes, [("loo_r2", "LOO R²"),
                                        ("free_run_r2", "Free-run R²")]):
        for name, df in dfs.items():
            ax.scatter(df["N_obs"], df[col], s=36, alpha=0.75,
                       color=PALETTE[name], label=name,
                       edgecolors="white", linewidths=0.4)
            # fit line
            mask = df[col].notna()
            if mask.sum() > 3:
                z = np.polyfit(df.loc[mask, "N_obs"], df.loc[mask, col], 1)
                xs = np.linspace(df["N_obs"].min(), df["N_obs"].max(), 50)
                ax.plot(xs, np.polyval(z, xs), "--", color=PALETTE[name],
                        lw=1, alpha=0.6)
        ax.set_xlabel("N_obs (observed neurons)", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.legend(fontsize=9)
        ax.set_title(f"{label} vs neuron count", fontsize=11)

    fig.tight_layout()
    fname = "nobs_vs_metrics.png"
    fig.savefig(SAVE_DIR / fname, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {SAVE_DIR / fname}")


def fig_per_neuron_loo_distribution(dfs: dict[str, pd.DataFrame]):
    """Pooled per-neuron LOO R² histograms (all worms concatenated)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, col_key, title in zip(axes,
                                   ["_loo_per_neuron", "_loo_w_per_neuron"],
                                   ["LOO R² (per neuron)", "LOO-windowed R² (per neuron)"]):
        for name, df in dfs.items():
            pooled = []
            for arr in df[col_key]:
                pooled.extend([v for v in arr if v is not None and not np.isnan(v)])
            pooled = np.array(pooled)
            ax.hist(pooled, bins=60, alpha=0.5, color=PALETTE[name],
                    label=f"{name} (n={len(pooled)}, med={np.median(pooled):.3f})",
                    density=True)
            ax.axvline(np.median(pooled), color=PALETTE[name], ls="--", lw=1.5)

        ax.set_xlabel("R²", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8)

    fig.suptitle("Per-neuron LOO R² distributions (pooled across all worms)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fname = "per_neuron_loo_distribution.png"
    fig.savefig(SAVE_DIR / fname, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {SAVE_DIR / fname}")


def fig_paired_delta(dfs: dict[str, pd.DataFrame]):
    """For worms present in BOTH runs, plot the Δ(multiply_project − concat)."""
    names = list(dfs.keys())
    df1 = dfs[names[0]].set_index("worm_id")
    df2 = dfs[names[1]].set_index("worm_id")
    common = sorted(set(df1.index) & set(df2.index))
    if len(common) < 3:
        print("  skipping paired-delta (too few common worms)")
        return

    metrics = NEURAL_METRICS + BEH_METRICS
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 5))
    for ax, (col, label) in zip(axes, metrics):
        deltas = df2.loc[common, col].values - df1.loc[common, col].values
        deltas = deltas.astype(float)
        valid = deltas[~np.isnan(deltas)]
        ax.hist(valid, bins=25, alpha=0.7, color="#2ca02c", edgecolor="white")
        med = np.median(valid)
        ax.axvline(0, color="black", lw=1, ls="-")
        ax.axvline(med, color="red", lw=1.5, ls="--",
                   label=f"median Δ = {med:+.4f}")
        n_pos = np.sum(valid > 0)
        ax.set_title(f"Δ {label}\n({n_pos}/{len(valid)} worms ↑)", fontsize=9)
        ax.set_xlabel("R² change", fontsize=9)
        ax.legend(fontsize=7)

    fig.suptitle(f"Paired Δ  ({names[1]} − {names[0]}),  {len(common)} common worms",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fname = "paired_delta.png"
    fig.savefig(SAVE_DIR / fname, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {SAVE_DIR / fname}")


def print_summary_table(dfs: dict[str, pd.DataFrame]):
    """Print a concise summary table to stdout and save as markdown."""
    lines = []
    header = f"{'metric':<25s}"
    for name in dfs:
        header += f" | {name:>22s} (mean±std, med)"
    lines.append(header)
    lines.append("-" * len(header))

    all_metrics = NEURAL_METRICS + BEH_METRICS
    for col, label in all_metrics:
        row = f"{label:<25s}"
        for name, df in dfs.items():
            vals = df[col].dropna().values
            row += (f" | {vals.mean():.4f}±{vals.std():.3f}  "
                    f"med={np.median(vals):.4f}  n={len(vals)}")
        lines.append(row)

    # extra: LOO neuron counts
    row = f"{'LOO #neurons eval':<25s}"
    for name, df in dfs.items():
        vals = df["loo_n_neurons"].values
        row += f" | mean={vals.mean():.0f}  range=[{vals.min()}, {vals.max()}]        "
    lines.append(row)

    txt = "\n".join(lines)
    print("\n" + txt + "\n")
    with open(SAVE_DIR / "summary_table.txt", "w") as f:
        f.write(txt + "\n")
    print(f"  saved {SAVE_DIR / 'summary_table.txt'}")


# ── main ───────────────────────────────────────────────────────────────
def main():
    dfs: dict[str, pd.DataFrame] = {}
    for name, root in RUNS.items():
        if not root.exists():
            print(f"⚠  {root} not found — skipping")
            continue
        df = load_run(root)
        print(f"Loaded {name}: {len(df)} worms")
        dfs[name] = df

    if not dfs:
        sys.exit("No data found")

    print_summary_table(dfs)

    print("\nGenerating plots …")
    fig_per_worm_bars(dfs, NEURAL_METRICS,
                      "Atlas Transformer — neural prediction per worm",
                      "per_worm_neural.png")
    fig_per_worm_bars(dfs, BEH_METRICS,
                      "Atlas Transformer — behaviour decoding per worm",
                      "per_worm_behaviour.png")
    fig_box_comparison(dfs)
    fig_nobs_vs_loo(dfs)
    fig_per_neuron_loo_distribution(dfs)
    fig_paired_delta(dfs)

    print(f"\nAll outputs in {SAVE_DIR}")


if __name__ == "__main__":
    main()
