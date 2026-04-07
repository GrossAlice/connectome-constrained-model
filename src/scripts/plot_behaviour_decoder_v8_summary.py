#!/usr/bin/env python
"""
Summary plots for behaviour_decoder_models_v8.py results.

Generates:
1. Violin plots: R² and Correlation side-by-side for all models/conditions
2. Per-eigenworm bar charts: Correlation per EW with motor vs all comparison
3. Heatmap: R² per worm × model/condition
4. Lag sweep: R² vs K for each model/condition
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_all_results(v8_dir: Path) -> dict:
    """Load all v8 results into a structured dict."""
    results = {}
    
    for worm_dir in sorted(v8_dir.iterdir()):
        if not worm_dir.is_dir():
            continue
        worm_id = worm_dir.name
        results[worm_id] = {}
        
        for json_file in worm_dir.glob("results_*.json"):
            # Parse filename: results_{neurons}_{run}_{Kb_config}.json
            parts = json_file.stem.split("_")
            neurons = parts[1]  # motor or all
            run = parts[2]      # run1 or run2
            kb_config = parts[3]  # Kb1 or KbKn
            
            key = f"{neurons}_{kb_config}"
            
            with open(json_file) as f:
                data = json.load(f)
            
            results[worm_id][key] = data
    
    return results


def aggregate_by_K(results: dict, neurons: str, kb_config: str, model: str, metric: str = "r2_mean") -> dict:
    """Aggregate results across worms for a specific configuration."""
    agg = {}
    
    for worm_id, worm_data in results.items():
        key = f"{neurons}_{kb_config}"
        if key not in worm_data:
            continue
        
        data = worm_data[key]
        for K_str, K_data in data.items():
            K = int(K_str)
            if K not in agg:
                agg[K] = []
            
            if model in K_data and metric in K_data[model]:
                agg[K].append(K_data[model][metric])
    
    return agg


# ══════════════════════════════════════════════════════════════════════════════
# Plotting functions - User's preferred style
# ══════════════════════════════════════════════════════════════════════════════

def plot_violin_r2_corr(results: dict, out_dir: Path, neurons: str = "all", kb_config: str = "Kb1", K_target: int = 15):
    """Violin plots with R² and Correlation side-by-side (user's preferred style)."""
    models = ["trf", "mlp", "ridge"]
    conditions = ["nb_fr", "n_fr", "b_1s"]
    model_labels = ["TRF", "MLP", "Ridge"]
    condition_labels = ["n+b FR", "n FR", "b (1-step)"]
    
    # Colors: red for TRF, blue for MLP, green for Ridge
    model_colors = {"trf": "#E57373", "mlp": "#64B5F6", "ridge": "#81C784"}
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for ax_idx, (metric, ylabel) in enumerate([("r2_mean", "R²"), ("corr_mean", "Correlation")]):
        ax = axes[ax_idx]
        
        # Collect all data for violin plots
        all_data = []
        positions = []
        colors = []
        labels = []
        
        pos = 0
        for m_idx, (model, model_label) in enumerate(zip(models, model_labels)):
            for c_idx, (cond, cond_label) in enumerate(zip(conditions, condition_labels)):
                model_key = f"{model}_{cond}"
                
                # Collect values for this model/condition
                vals = []
                for worm_id, worm_data in results.items():
                    key = f"{neurons}_{kb_config}"
                    if key not in worm_data:
                        continue
                    data = worm_data[key]
                    K_str = str(K_target)
                    if K_str not in data:
                        continue
                    if model_key in data[K_str] and metric in data[K_str][model_key]:
                        vals.append(data[K_str][model_key][metric])
                
                if vals:
                    all_data.append(vals)
                    positions.append(pos)
                    colors.append(model_colors[model])
                    labels.append(f"{model_label}\n{cond_label}")
                pos += 1
            pos += 0.5  # Gap between models
        
        # Create violin plots
        if all_data:
            parts = ax.violinplot(all_data, positions=positions, showmeans=True, showmedians=True)
            
            # Color the violins
            for i, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.7)
                pc.set_edgecolor("black")
            
            # Style mean and median lines
            parts["cmeans"].set_color("blue")
            parts["cmeans"].set_linewidth(2)
            parts["cmedians"].set_color("black")
        
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(axis="y", alpha=0.3)
        
        # Add horizontal line at 0 for R²
        if metric == "r2_mean":
            ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    
    n_worms = len(results)
    fig.suptitle(f"Behaviour Decoding at K={K_target} — {n_worms} worms, {neurons} neurons", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / f"violin_r2_corr_{neurons}_{kb_config}_K{K_target}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: violin_r2_corr_{neurons}_{kb_config}_K{K_target}.png")


def plot_per_eigenworm_bars(results: dict, out_dir: Path, kb_config: str = "Kb1", K_target: int = 15):
    """Per-eigenworm bar charts with motor vs all comparison (user's preferred style)."""
    models = ["trf", "mlp", "ridge"]
    conditions = ["nb_fr", "n_fr", "b_1s"]
    model_labels = ["TRF", "MLP", "Ridge"]
    condition_labels = ["n+b FR", "n FR", "b TF"]
    ew_names = ["EW1", "EW2", "EW3", "EW4", "EW5", "EW6"]
    
    # Colors
    model_colors = {"trf": "#E57373", "mlp": "#64B5F6", "ridge": "#81C784"}
    
    fig, axes = plt.subplots(1, 6, figsize=(20, 5), sharey=True)
    
    for ew_idx, (ax, ew_name) in enumerate(zip(axes, ew_names)):
        x_pos = 0
        x_ticks = []
        x_labels = []
        
        for m_idx, (model, model_label) in enumerate(zip(models, model_labels)):
            for c_idx, (cond, cond_label) in enumerate(zip(conditions, condition_labels)):
                model_key = f"{model}_{cond}"
                
                for n_idx, neurons in enumerate(["motor", "all"]):
                    # Collect per-mode correlation for this eigenworm
                    vals = []
                    for worm_id, worm_data in results.items():
                        key = f"{neurons}_{kb_config}"
                        if key not in worm_data:
                            continue
                        data = worm_data[key]
                        K_str = str(K_target)
                        if K_str not in data:
                            continue
                        if model_key in data[K_str] and "corr_per_mode" in data[K_str][model_key]:
                            corr_modes = data[K_str][model_key]["corr_per_mode"]
                            if ew_idx < len(corr_modes):
                                vals.append(corr_modes[ew_idx])
                    
                    if vals:
                        mean_val = np.mean(vals)
                        sem_val = np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0
                        
                        # Bar with hatch for motor vs all
                        hatch = None if neurons == "motor" else "///"
                        alpha = 0.9 if neurons == "motor" else 0.5
                        
                        ax.bar(x_pos, mean_val, width=0.8, color=model_colors[model], 
                               alpha=alpha, hatch=hatch, edgecolor="black", linewidth=0.5)
                        ax.errorbar(x_pos, mean_val, yerr=sem_val, color="black", 
                                   capsize=2, capthick=1, linewidth=1)
                        
                        # Scatter individual points
                        jitter = np.random.uniform(-0.15, 0.15, len(vals))
                        ax.scatter([x_pos] * len(vals) + jitter, vals, 
                                  color="gray", s=10, alpha=0.5, zorder=5)
                    
                    x_pos += 1
                
                x_ticks.append(x_pos - 1)
                x_labels.append(f"{model_label}\n{cond_label}")
            
            x_pos += 0.5  # Gap between models
        
        ax.set_title(ew_name, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        
        if ew_idx == 0:
            ax.set_ylabel("Pearson Correlation", fontsize=11)
    
    # Add legend
    legend_elements = [
        Patch(facecolor=model_colors["trf"], label="Transformer", edgecolor="black"),
        Patch(facecolor=model_colors["mlp"], label="MLP", edgecolor="black"),
        Patch(facecolor=model_colors["ridge"], label="Ridge", edgecolor="black"),
        Patch(facecolor="gray", alpha=0.9, label="Motor neurons", edgecolor="black"),
        Patch(facecolor="gray", alpha=0.5, hatch="///", label="All neurons", edgecolor="black"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02))
    
    n_worms = len(results)
    fig.suptitle(f"Behaviour Decoding Summary — K = {K_target},  Correlation per Eigenworm  ({n_worms} worms)", 
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(out_dir / f"per_eigenworm_bars_{kb_config}_K{K_target}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: per_eigenworm_bars_{kb_config}_K{K_target}.png")


def plot_heatmap_worm_model(results: dict, out_dir: Path, neurons: str = "motor", kb_config: str = "Kb1", K_target: int = 15):
    """Heatmap of R² per worm (rows) × model/condition (columns) (user's preferred style)."""
    models = ["trf", "mlp", "ridge"]
    conditions = ["nb_fr", "n_fr", "b_1s"]
    model_labels = ["TRF", "MLP", "Ridge"]
    condition_labels = ["n+b FR", "n FR", "b (1-step)"]
    
    # Build column labels
    col_labels = []
    for model_label in model_labels:
        for cond_label in condition_labels:
            col_labels.append(f"{model_label}\n{cond_label}")
    
    # Collect data
    worm_ids = sorted(results.keys())
    data_matrix = []
    valid_worms = []
    
    for worm_id in worm_ids:
        worm_data = results[worm_id]
        key = f"{neurons}_{kb_config}"
        if key not in worm_data:
            continue
        
        data = worm_data[key]
        K_str = str(K_target)
        if K_str not in data:
            continue
        
        row = []
        for model in models:
            for cond in conditions:
                model_key = f"{model}_{cond}"
                if model_key in data[K_str] and "r2_mean" in data[K_str][model_key]:
                    row.append(data[K_str][model_key]["r2_mean"])
                else:
                    row.append(np.nan)
        
        if not all(np.isnan(row)):
            data_matrix.append(row)
            valid_worms.append(worm_id)
    
    if not data_matrix:
        print(f"  Skipped heatmap (no data for {neurons}_{kb_config})")
        return
    
    data_matrix = np.array(data_matrix)
    
    # Clip extreme negative values for visualization
    vmin = max(-0.5, np.nanmin(data_matrix))
    vmax = 1.0
    
    fig, ax = plt.subplots(figsize=(10, max(8, len(valid_worms) * 0.25)))
    
    # Use RdYlGn colormap (red = bad, green = good)
    im = ax.imshow(data_matrix, cmap="RdYlGn", aspect="auto", vmin=vmin, vmax=vmax)
    
    # Set ticks
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(valid_worms)))
    ax.set_yticklabels(valid_worms, fontsize=8)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label="R²", shrink=0.8)
    
    n_worms = len(valid_worms)
    ax.set_title(f"R² at K={K_target} — {neurons} neurons ({n_worms} worms)", fontsize=14, pad=10)
    
    fig.tight_layout()
    fig.savefig(out_dir / f"heatmap_{neurons}_{kb_config}_K{K_target}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: heatmap_{neurons}_{kb_config}_K{K_target}.png")


def plot_lag_sweep(results: dict, out_dir: Path, neurons: str = "motor", kb_config: str = "Kb1"):
    """R² vs K for each model and condition."""
    models = ["trf", "mlp", "ridge"]
    conditions = ["nb_fr", "n_fr", "b_1s"]
    condition_labels = ["n+b FR", "n FR", "b 1-step"]
    model_labels = ["Transformer", "MLP", "Ridge"]
    model_colors = {"trf": "#E57373", "mlp": "#64B5F6", "ridge": "#81C784"}
    linestyles = ["-", "--", ":"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    
    for c_idx, (cond, cond_label) in enumerate(zip(conditions, condition_labels)):
        ax = axes[c_idx]
        
        for m_idx, (model, model_label) in enumerate(zip(models, model_labels)):
            model_key = f"{model}_{cond}"
            agg = aggregate_by_K(results, neurons, kb_config, model_key, "r2_mean")
            
            if not agg:
                continue
            
            Ks = sorted(agg.keys())
            means = [np.mean(agg[K]) for K in Ks]
            stds = [np.std(agg[K]) / np.sqrt(len(agg[K])) if len(agg[K]) > 1 else 0 for K in Ks]
            
            ax.errorbar(Ks, means, yerr=stds, label=model_label, 
                       color=model_colors[model], linestyle=linestyles[m_idx],
                       marker="o", markersize=5, capsize=3, linewidth=2)
        
        ax.set_xlabel("K (lags)")
        ax.set_title(cond_label)
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=-0.1)
    
    axes[0].set_ylabel("R² (mean ± SEM)")
    fig.suptitle(f"Lag Sweep — {neurons.upper()} neurons, {kb_config}", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / f"lag_sweep_{neurons}_{kb_config}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: lag_sweep_{neurons}_{kb_config}.png")


def plot_summary_violin_all(results: dict, out_dir: Path, K_target: int = 15):
    """Combined violin plot showing all configurations."""
    models = ["trf", "mlp", "ridge"]
    conditions = ["nb_fr", "n_fr", "b_1s"]
    model_labels = ["TRF", "MLP", "Ridge"]
    condition_labels = ["n+b FR", "n FR", "b (1-step)"]
    model_colors = {"trf": "#E57373", "mlp": "#64B5F6", "ridge": "#81C784"}
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    for row_idx, neurons in enumerate(["motor", "all"]):
        for col_idx, metric in enumerate(["r2_mean", "corr_mean"]):
            ax = axes[row_idx, col_idx]
            
            all_data = []
            positions = []
            colors = []
            labels = []
            
            pos = 0
            for m_idx, (model, model_label) in enumerate(zip(models, model_labels)):
                for c_idx, (cond, cond_label) in enumerate(zip(conditions, condition_labels)):
                    model_key = f"{model}_{cond}"
                    
                    vals = []
                    for worm_id, worm_data in results.items():
                        for kb_config in ["Kb1", "KbKn"]:
                            key = f"{neurons}_{kb_config}"
                            if key not in worm_data:
                                continue
                            data = worm_data[key]
                            K_str = str(K_target)
                            if K_str not in data:
                                continue
                            if model_key in data[K_str] and metric in data[K_str][model_key]:
                                vals.append(data[K_str][model_key][metric])
                    
                    if vals:
                        all_data.append(vals)
                        positions.append(pos)
                        colors.append(model_colors[model])
                        labels.append(f"{model_label}\n{cond_label}")
                    pos += 1
                pos += 0.5
            
            if all_data:
                parts = ax.violinplot(all_data, positions=positions, showmeans=True, showmedians=True)
                for i, pc in enumerate(parts["bodies"]):
                    pc.set_facecolor(colors[i])
                    pc.set_alpha(0.7)
                    pc.set_edgecolor("black")
                parts["cmeans"].set_color("blue")
                parts["cmeans"].set_linewidth(2)
                parts["cmedians"].set_color("black")
            
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ylabel = "R²" if metric == "r2_mean" else "Correlation"
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f"{neurons.upper()} neurons — {ylabel}", fontsize=12)
            ax.grid(axis="y", alpha=0.3)
            if metric == "r2_mean":
                ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    
    n_worms = len(results)
    fig.suptitle(f"Behaviour Decoding Summary at K={K_target} — {n_worms} worms", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / f"summary_violin_all_K{K_target}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: summary_violin_all_K{K_target}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v8_dir", default="output_plots/behaviour_decoder/v8",
                        help="Directory containing v8 results")
    parser.add_argument("--out_dir", default="output_plots/behaviour_decoder/v8_summary",
                        help="Output directory for summary plots")
    parser.add_argument("--K", type=int, default=15, help="K value to use for plots")
    args = parser.parse_args()
    
    v8_dir = Path(args.v8_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from {v8_dir}...")
    results = load_all_results(v8_dir)
    print(f"  Found {len(results)} worms")
    
    print(f"\nGenerating summary plots (K={args.K})...")
    
    # 1. Violin plots - R² and Correlation side-by-side
    for neurons in ["motor", "all"]:
        for kb_config in ["Kb1", "KbKn"]:
            plot_violin_r2_corr(results, out_dir, neurons, kb_config, args.K)
    
    # 2. Per-eigenworm bar charts
    for kb_config in ["Kb1", "KbKn"]:
        plot_per_eigenworm_bars(results, out_dir, kb_config, args.K)
    
    # 3. Heatmaps - worm × model
    for neurons in ["motor", "all"]:
        for kb_config in ["Kb1", "KbKn"]:
            plot_heatmap_worm_model(results, out_dir, neurons, kb_config, args.K)
    
    # 4. Lag sweeps
    for neurons in ["motor", "all"]:
        for kb_config in ["Kb1", "KbKn"]:
            plot_lag_sweep(results, out_dir, neurons, kb_config)
    
    # 5. Summary violin all configs
    plot_summary_violin_all(results, out_dir, args.K)
    
    print(f"\nDone! All plots saved to {out_dir}")


if __name__ == "__main__":
    main()
