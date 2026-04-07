#!/usr/bin/env python3
"""
Summary plots for connectome-constrained MLP comparison.

Aggregates results from all worms and creates summary visualizations:
  1. Overall R² comparison: connectome vs unconstrained (all worms)
  2. Per-worm scatter: R² constrained vs unconstrained
  3. Delta R² distribution across all neurons from all worms

Usage:
    python -m scripts.connectome_mlp.plot_connectome_summary \
        --results_dir output_plots/connectome_mlp_batch
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent


COND_COLOR = {
    "conn+self":   "#e74c3c",
    "conn_only":   "#3498db",
    "all+self":    "#2ecc71",
    "all_only":    "#9b59b6",
}

COND_NAME = {
    "conn+self":   "connectome + self",
    "conn_only":   "connectome only",
    "all+self":    "unconstrained + self",
    "all_only":    "unconstrained only",
}


def load_all_results(results_dir: Path, T_matrix: str = "T_e"):
    """Load results.json from all worm subdirectories."""
    all_results = {}
    
    for worm_dir in sorted(results_dir.iterdir()):
        if not worm_dir.is_dir():
            continue
        if worm_dir.name in ["summary", "batch_log.json"]:
            continue
            
        results_path = worm_dir / T_matrix / "results.json"
        if not results_path.exists():
            print(f"  Missing: {results_path}")
            continue
            
        with open(results_path) as f:
            data = json.load(f)
        
        worm_id = worm_dir.name
        all_results[worm_id] = data
        
    return all_results


def plot_overall_comparison(all_results: dict, out_dir: Path, T_matrix: str,
                            K: int = 5):
    """Bar chart comparing mean R² across all worms for each condition."""
    conditions = ["conn+self", "all+self", "conn_only", "all_only"]
    models = ["ridge", "pca_ridge", "mlp"]
    model_names = ["Ridge", "PCA-Ridge", "MLP"]
    
    # Collect mean R² per condition/model across all worms
    data = {c: {m: [] for m in models} for c in conditions}
    
    for worm_id, results in all_results.items():
        sk = str(K)
        if sk not in results:
            continue
        for cond in conditions:
            cond_data = results[sk].get(cond, {})
            for model in models:
                r2_key = f"r2_mean_{model}"
                if r2_key in cond_data and cond_data[r2_key] is not None:
                    data[cond][model].append(cond_data[r2_key])
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(models))
    width = 0.2
    
    for i, cond in enumerate(conditions):
        means = [np.mean(data[cond][m]) if data[cond][m] else 0 for m in models]
        stds = [np.std(data[cond][m]) if data[cond][m] else 0 for m in models]
        bars = ax.bar(x + i * width, means, width, 
                      label=COND_NAME[cond], color=COND_COLOR[cond],
                      alpha=0.8, yerr=stds, capsize=3)
    
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, fontsize=12)
    ax.set_ylabel("Mean R² (across worms)", fontsize=11)
    ax.set_title(f"Connectome-Constrained vs Unconstrained — {T_matrix}, K={K}\n"
                 f"({len(all_results)} worms)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    fig.tight_layout()
    fname = out_dir / f"overall_comparison_{T_matrix}_K{K}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")


def plot_worm_scatter(all_results: dict, out_dir: Path, T_matrix: str,
                      K: int = 5, model: str = "mlp"):
    """Scatter plot: per-worm mean R² (connectome) vs (unconstrained)."""
    conn_r2 = []
    all_r2 = []
    worm_ids = []
    
    sk = str(K)
    for worm_id, results in sorted(all_results.items()):
        if sk not in results:
            continue
        conn = results[sk].get("conn+self", {}).get(f"r2_mean_{model}")
        unconstrained = results[sk].get("all+self", {}).get(f"r2_mean_{model}")
        if conn is not None and unconstrained is not None:
            conn_r2.append(conn)
            all_r2.append(unconstrained)
            worm_ids.append(worm_id)
    
    if not conn_r2:
        print(f"No data for {model} scatter plot")
        return
    
    conn_r2 = np.array(conn_r2)
    all_r2 = np.array(all_r2)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(all_r2, conn_r2, s=60, alpha=0.7, c=COND_COLOR["conn+self"],
               edgecolors="white", linewidths=0.5)
    
    # Diagonal
    lim = [min(all_r2.min(), conn_r2.min()) - 0.05,
           max(all_r2.max(), conn_r2.max()) + 0.05]
    ax.plot(lim, lim, "k--", lw=1, alpha=0.5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    
    # Stats
    above = (conn_r2 > all_r2).sum()
    below = (conn_r2 < all_r2).sum()
    mean_delta = np.mean(conn_r2 - all_r2)
    
    ax.set_xlabel(f"R² unconstrained + self ({model.upper()})", fontsize=11)
    ax.set_ylabel(f"R² connectome + self ({model.upper()})", fontsize=11)
    ax.set_title(f"Per-Worm Comparison — {T_matrix}, K={K}\n"
                 f"Above diagonal: {above}/{len(conn_r2)}, "
                 f"ΔR² mean={mean_delta:+.4f}",
                 fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    fig.tight_layout()
    fname = out_dir / f"worm_scatter_{model}_{T_matrix}_K{K}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")


def plot_neuron_delta_histogram(all_results: dict, out_dir: Path, T_matrix: str,
                                 K: int = 5, model: str = "pca_ridge"):
    """Histogram of ΔR² (connectome - unconstrained) across all neurons."""
    deltas_self = []
    deltas_cross = []
    
    sk = str(K)
    for worm_id, results in all_results.items():
        if sk not in results:
            continue
        
        # With self
        conn = results[sk].get("conn+self", {}).get(f"r2_per_neuron_{model}")
        unc = results[sk].get("all+self", {}).get(f"r2_per_neuron_{model}")
        if conn and unc and len(conn) == len(unc):
            for c, u in zip(conn, unc):
                if c is not None and u is not None:
                    deltas_self.append(c - u)
        
        # Cross-neuron only
        conn = results[sk].get("conn_only", {}).get(f"r2_per_neuron_{model}")
        unc = results[sk].get("all_only", {}).get(f"r2_per_neuron_{model}")
        if conn and unc and len(conn) == len(unc):
            for c, u in zip(conn, unc):
                if c is not None and u is not None:
                    deltas_cross.append(c - u)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, deltas, title, color in [
        (axes[0], deltas_self, "+self (AR)", COND_COLOR["conn+self"]),
        (axes[1], deltas_cross, "cross-neuron only", COND_COLOR["conn_only"]),
    ]:
        if not deltas:
            continue
        deltas = np.array(deltas)
        
        ax.hist(deltas, bins=50, color=color, alpha=0.7, edgecolor="white")
        ax.axvline(0, color="k", lw=1.5, ls="--")
        ax.axvline(np.median(deltas), color="red", lw=2, ls="-",
                   label=f"median={np.median(deltas):.4f}")
        
        above = (deltas > 0).sum()
        ax.set_xlabel(f"ΔR² (connectome − unconstrained)", fontsize=11)
        ax.set_ylabel("Count (neurons)", fontsize=11)
        ax.set_title(f"{title}\n"
                     f"N={len(deltas)}, above zero: {above} ({100*above/len(deltas):.0f}%)",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    fig.suptitle(f"Per-Neuron ΔR² Distribution — {model.upper()}, {T_matrix}, K={K}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fname = out_dir / f"neuron_delta_hist_{model}_{T_matrix}_K{K}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")


def plot_mlp_vs_ridge_gain(all_results: dict, out_dir: Path, T_matrix: str,
                           K: int = 5):
    """Compare MLP gain over Ridge for connectome vs unconstrained."""
    # MLP - Ridge for each condition
    conditions = ["conn+self", "all+self"]
    
    gains = {c: [] for c in conditions}
    worm_ids = []
    
    sk = str(K)
    for worm_id, results in sorted(all_results.items()):
        if sk not in results:
            continue
        
        valid = True
        for cond in conditions:
            mlp = results[sk].get(cond, {}).get("r2_mean_mlp")
            ridge = results[sk].get(cond, {}).get("r2_mean_pca_ridge")
            if mlp is None or ridge is None:
                valid = False
                break
            gains[cond].append(mlp - ridge)
        
        if valid:
            worm_ids.append(worm_id)
    
    if not worm_ids:
        print("No MLP data for gain plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(worm_ids))
    width = 0.35
    
    for i, cond in enumerate(conditions):
        ax.bar(x + i * width, gains[cond], width, 
               label=COND_NAME[cond], color=COND_COLOR[cond], alpha=0.8)
    
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(worm_ids, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("ΔR² (MLP − PCA-Ridge)", fontsize=11)
    ax.set_title(f"MLP Gain over Ridge — {T_matrix}, K={K}\n"
                 f"Positive = MLP captures nonlinear structure",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    fig.tight_layout()
    fname = out_dir / f"mlp_ridge_gain_{T_matrix}_K{K}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")


def create_summary_json(all_results: dict, out_dir: Path, T_matrix: str):
    """Create a summary JSON with aggregated statistics."""
    summary = {
        "T_matrix": T_matrix,
        "n_worms": len(all_results),
        "worm_ids": sorted(all_results.keys()),
        "by_K": {}
    }
    
    K_values = set()
    for results in all_results.values():
        K_values.update(int(k) for k in results.keys())
    
    for K in sorted(K_values):
        sk = str(K)
        summary["by_K"][K] = {}
        
        for cond in ["conn+self", "all+self", "conn_only", "all_only"]:
            for model in ["ridge", "pca_ridge", "mlp"]:
                key = f"r2_mean_{model}"
                values = []
                for results in all_results.values():
                    v = results.get(sk, {}).get(cond, {}).get(key)
                    if v is not None:
                        values.append(v)
                
                if values:
                    summary["by_K"][K][f"{cond}_{model}"] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "median": float(np.median(values)),
                        "n": len(values)
                    }
    
    fname = out_dir / f"summary_{T_matrix}.json"
    with open(fname, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {fname}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", 
                    default="output_plots/connectome_mlp_batch",
                    help="Directory with per-worm results")
    ap.add_argument("--T_matrices", nargs="+", default=["T_e"],
                    help="Which connectome matrices to summarize")
    ap.add_argument("--K", type=int, default=5,
                    help="Context length to focus on")
    args = ap.parse_args()
    
    results_dir = ROOT / args.results_dir
    summary_dir = results_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    for T_matrix in args.T_matrices:
        print(f"\n{'='*60}")
        print(f"Processing {T_matrix}")
        print(f"{'='*60}")
        
        all_results = load_all_results(results_dir, T_matrix)
        print(f"Loaded {len(all_results)} worms")
        
        if not all_results:
            print(f"No results found for {T_matrix}")
            continue
        
        # Generate all plots
        plot_overall_comparison(all_results, summary_dir, T_matrix, args.K)
        plot_worm_scatter(all_results, summary_dir, T_matrix, args.K, "mlp")
        plot_worm_scatter(all_results, summary_dir, T_matrix, args.K, "pca_ridge")
        plot_neuron_delta_histogram(all_results, summary_dir, T_matrix, args.K, "pca_ridge")
        plot_neuron_delta_histogram(all_results, summary_dir, T_matrix, args.K, "mlp")
        plot_mlp_vs_ridge_gain(all_results, summary_dir, T_matrix, args.K)
        create_summary_json(all_results, summary_dir, T_matrix)
    
    print(f"\nAll summary plots saved to: {summary_dir}")


if __name__ == "__main__":
    main()
