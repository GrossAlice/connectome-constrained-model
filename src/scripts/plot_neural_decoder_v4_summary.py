#!/usr/bin/env python3
"""
Summary analysis and plotting for neural_activity_decoder_v4 results.

Compares Ridge, Ridge-fast, MLP, and Transformer models across different
input configurations (causal, concurrent, self).
"""

import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict


def load_all_results(results_dir: Path) -> dict:
    """Load all results.json files from subdirectories."""
    all_results = {}
    for subdir in sorted(results_dir.iterdir()):
        if subdir.is_dir():
            results_file = subdir / "results.json"
            if results_file.exists():
                worm_name = subdir.name.replace("_all", "")
                with open(results_file) as f:
                    all_results[worm_name] = json.load(f)
    return all_results


def extract_metrics(all_results: dict) -> pd.DataFrame:
    """Extract mean R² and correlation metrics into a DataFrame."""
    rows = []
    
    for worm, data in all_results.items():
        for k_val, metrics in data.items():
            for metric_name, value in metrics.items():
                if metric_name.startswith("r2_mean_") or metric_name.startswith("corr_mean_"):
                    # Parse metric name: {r2|corr}_mean_{model}_{inputs}
                    parts = metric_name.split("_")
                    metric_type = parts[0]  # r2 or corr
                    model = parts[2]  # ridge, mlp, trf, ridge_fast
                    
                    # Handle ridge_fast specially
                    if model == "ridge" and len(parts) > 3 and parts[3] == "fast":
                        model = "ridge_fast"
                        inputs = "_".join(parts[4:])
                    else:
                        inputs = "_".join(parts[3:])
                    
                    rows.append({
                        "worm": worm,
                        "K": int(k_val),
                        "model": model,
                        "inputs": inputs,
                        "metric": metric_type,
                        "value": value
                    })
    
    return pd.DataFrame(rows)


def plot_model_comparison_bar(df: pd.DataFrame, save_dir: Path):
    """Bar plot comparing models across input configurations."""
    # Filter to R² metrics
    df_r2 = df[df["metric"] == "r2"].copy()
    
    # Group by model and inputs, compute mean across worms
    summary = df_r2.groupby(["model", "inputs"])["value"].agg(["mean", "std"]).reset_index()
    
    # Define model order and colors (excluding ridge_fast)
    models = ["ridge", "mlp", "trf"]
    model_labels = ["Ridge", "MLP", "Transformer"]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]
    
    # Define input configurations of interest
    input_configs = ["causal_self", "conc_self", "self", "causal", "conc", "conc_causal"]
    input_labels = ["Causal+Self", "Conc+Self", "Self only", "Causal only", "Conc only", "Conc+Causal"]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(input_configs))
    width = 0.25
    n_models = len(models)
    
    for i, (model, label, color) in enumerate(zip(models, model_labels, colors)):
        model_data = summary[summary["model"] == model]
        means = []
        stds = []
        for inp in input_configs:
            row = model_data[model_data["inputs"] == inp]
            if len(row) > 0:
                means.append(row["mean"].values[0])
                stds.append(row["std"].values[0])
            else:
                means.append(0)
                stds.append(0)
        
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, means, width, 
                      label=label, color=color, alpha=0.7, edgecolor="white", linewidth=1)
        
        # Add individual data points
        for j, inp in enumerate(input_configs):
            # Get individual worm values for this model/input combination
            worm_vals = df_r2[(df_r2["model"] == model) & (df_r2["inputs"] == inp)]["value"].values
            if len(worm_vals) > 0:
                # Add jitter to x position
                jitter = np.random.uniform(-width*0.3, width*0.3, len(worm_vals))
                ax.scatter(np.full_like(worm_vals, j + offset) + jitter, worm_vals, 
                          s=25, c=color, edgecolors="black", linewidths=0.5, 
                          alpha=0.8, zorder=5)
    
    ax.set_ylabel("R² (mean ± std across worms)", fontsize=12)
    ax.set_xlabel("Input Configuration", fontsize=12)
    ax.set_title(f"Neural Activity Prediction: Model Comparison (n={len(df['worm'].unique())} worms)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(input_labels, rotation=15, ha="right")
    ax.legend(loc="upper right")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylim(-0.2, 1.0)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / "model_comparison_bar.png", dpi=150)
    plt.savefig(save_dir / "model_comparison_bar.pdf")
    plt.close()
    print(f"Saved: {save_dir / 'model_comparison_bar.png'}")


def plot_per_worm_heatmap(df: pd.DataFrame, save_dir: Path):
    """Heatmap of R² values per worm and model configuration."""
    df_r2 = df[df["metric"] == "r2"].copy()
    
    # Focus on key configurations
    key_configs = [
        ("ridge", "causal_self"),
        ("ridge", "self"),
        ("mlp", "causal_self"),
        ("mlp", "self"),
        ("trf", "causal_self"),
        ("trf", "self"),
    ]
    
    # Build matrix
    worms = sorted(df_r2["worm"].unique())
    config_labels = [f"{m}_{i}" for m, i in key_configs]
    
    matrix = np.zeros((len(worms), len(key_configs)))
    for i, worm in enumerate(worms):
        for j, (model, inputs) in enumerate(key_configs):
            row = df_r2[(df_r2["worm"] == worm) & (df_r2["model"] == model) & (df_r2["inputs"] == inputs)]
            if len(row) > 0:
                matrix[i, j] = row["value"].values[0]
            else:
                matrix[i, j] = np.nan
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(worms) * 0.4)))
    
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    
    ax.set_xticks(range(len(config_labels)))
    ax.set_xticklabels([l.replace("_", "\n") for l in config_labels], fontsize=10)
    ax.set_yticks(range(len(worms)))
    ax.set_yticklabels(worms, fontsize=9)
    
    # Add text annotations
    for i in range(len(worms)):
        for j in range(len(key_configs)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", 
                       color="white" if val < 0.3 or val > 0.7 else "black", fontsize=8)
    
    ax.set_title("R² per Worm and Model Configuration", fontsize=14)
    plt.colorbar(im, ax=ax, label="R²")
    
    plt.tight_layout()
    plt.savefig(save_dir / "per_worm_heatmap.png", dpi=150)
    plt.savefig(save_dir / "per_worm_heatmap.pdf")
    plt.close()
    print(f"Saved: {save_dir / 'per_worm_heatmap.png'}")


def plot_self_vs_causal_scatter(df: pd.DataFrame, save_dir: Path):
    """Scatter plot: self-only prediction vs causal+self prediction."""
    df_r2 = df[df["metric"] == "r2"].copy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    models = ["ridge", "mlp", "trf"]
    model_labels = ["Ridge", "MLP", "Transformer"]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]
    
    for ax, model, label, color in zip(axes, models, model_labels, colors):
        worms = df_r2["worm"].unique()
        self_vals = []
        causal_self_vals = []
        
        for worm in worms:
            self_row = df_r2[(df_r2["worm"] == worm) & (df_r2["model"] == model) & (df_r2["inputs"] == "self")]
            cs_row = df_r2[(df_r2["worm"] == worm) & (df_r2["model"] == model) & (df_r2["inputs"] == "causal_self")]
            
            if len(self_row) > 0 and len(cs_row) > 0:
                self_vals.append(self_row["value"].values[0])
                causal_self_vals.append(cs_row["value"].values[0])
        
        ax.scatter(self_vals, causal_self_vals, s=80, c=color, alpha=0.7, edgecolors="white")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="y=x")
        
        ax.set_xlabel("R² (Self only)", fontsize=11)
        ax.set_ylabel("R² (Causal + Self)", fontsize=11)
        ax.set_title(f"{label}", fontsize=13)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        
        # Add mean annotation
        if self_vals:
            mean_self = np.mean(self_vals)
            mean_cs = np.mean(causal_self_vals)
            ax.axvline(mean_self, color=color, alpha=0.5, linestyle=":")
            ax.axhline(mean_cs, color=color, alpha=0.5, linestyle=":")
            ax.annotate(f"Δ = {mean_cs - mean_self:.3f}", 
                       xy=(0.05, 0.95), xycoords="axes fraction",
                       fontsize=11, fontweight="bold")
    
    plt.suptitle("Self-only vs Causal+Self Prediction (per worm)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_dir / "self_vs_causal_scatter.png", dpi=150, bbox_inches="tight")
    plt.savefig(save_dir / "self_vs_causal_scatter.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_dir / 'self_vs_causal_scatter.png'}")


def plot_ridge_vs_mlp_scatter(df: pd.DataFrame, save_dir: Path):
    """Scatter comparing Ridge vs MLP for each input configuration."""
    df_r2 = df[df["metric"] == "r2"].copy()
    
    input_configs = ["causal_self", "conc_self", "self"]
    input_labels = ["Causal+Self", "Conc+Self", "Self only"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, inp, label in zip(axes, input_configs, input_labels):
        worms = df_r2["worm"].unique()
        ridge_vals = []
        mlp_vals = []
        
        for worm in worms:
            ridge_row = df_r2[(df_r2["worm"] == worm) & (df_r2["model"] == "ridge") & (df_r2["inputs"] == inp)]
            mlp_row = df_r2[(df_r2["worm"] == worm) & (df_r2["model"] == "mlp") & (df_r2["inputs"] == inp)]
            
            if len(ridge_row) > 0 and len(mlp_row) > 0:
                ridge_vals.append(ridge_row["value"].values[0])
                mlp_vals.append(mlp_row["value"].values[0])
        
        ax.scatter(ridge_vals, mlp_vals, s=80, c="#9C27B0", alpha=0.7, edgecolors="white")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="y=x")
        
        ax.set_xlabel("R² (Ridge)", fontsize=11)
        ax.set_ylabel("R² (MLP)", fontsize=11)
        ax.set_title(f"{label}", fontsize=13)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        
        # Count wins
        ridge_wins = sum(1 for r, m in zip(ridge_vals, mlp_vals) if r > m)
        mlp_wins = len(ridge_vals) - ridge_wins
        ax.annotate(f"Ridge wins: {ridge_wins}\nMLP wins: {mlp_wins}", 
                   xy=(0.95, 0.05), xycoords="axes fraction",
                   fontsize=10, ha="right")
    
    plt.suptitle("Ridge vs MLP (per worm)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_dir / "ridge_vs_mlp_scatter.png", dpi=150, bbox_inches="tight")
    plt.savefig(save_dir / "ridge_vs_mlp_scatter.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_dir / 'ridge_vs_mlp_scatter.png'}")


def plot_input_importance(df: pd.DataFrame, save_dir: Path):
    """Show marginal contribution of each input type."""
    df_r2 = df[df["metric"] == "r2"].copy()
    
    # Compute deltas for Ridge model
    summary = df_r2.groupby(["model", "inputs"])["value"].mean().reset_index()
    
    models = ["ridge", "mlp", "trf"]
    model_labels = ["Ridge", "MLP", "Transformer"]
    
    # Contribution analysis
    contributions = {}
    for model in models:
        model_data = summary[summary["model"] == model]
        
        # Get base values
        self_only = model_data[model_data["inputs"] == "self"]["value"].values
        causal_only = model_data[model_data["inputs"] == "causal"]["value"].values
        conc_only = model_data[model_data["inputs"] == "conc"]["value"].values
        causal_self = model_data[model_data["inputs"] == "causal_self"]["value"].values
        conc_self = model_data[model_data["inputs"] == "conc_self"]["value"].values
        
        if len(self_only) > 0 and len(causal_self) > 0:
            contributions[model] = {
                "Self": self_only[0] if len(self_only) > 0 else 0,
                "Causal added to Self": (causal_self[0] - self_only[0]) if len(causal_self) > 0 and len(self_only) > 0 else 0,
                "Conc added to Self": (conc_self[0] - self_only[0]) if len(conc_self) > 0 and len(self_only) > 0 else 0,
            }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models))
    width = 0.25
    
    bars1 = ax.bar(x - width, [contributions.get(m, {}).get("Self", 0) for m in models], 
                   width, label="Self (baseline)", color="#607D8B")
    bars2 = ax.bar(x, [contributions.get(m, {}).get("Causal added to Self", 0) for m in models], 
                   width, label="+ Causal neighbors", color="#2196F3")
    bars3 = ax.bar(x + width, [contributions.get(m, {}).get("Conc added to Self", 0) for m in models], 
                   width, label="+ Concurrent neighbors", color="#4CAF50")
    
    ax.set_ylabel("R² / ΔR²", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_title("Input Contribution Analysis", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_dir / "input_contribution.png", dpi=150)
    plt.savefig(save_dir / "input_contribution.pdf")
    plt.close()
    print(f"Saved: {save_dir / 'input_contribution.png'}")


def create_summary_table(df: pd.DataFrame, save_dir: Path):
    """Create and save a summary statistics table."""
    df_r2 = df[df["metric"] == "r2"].copy()
    
    # Pivot: model x inputs with mean ± std
    summary = df_r2.groupby(["model", "inputs"])["value"].agg(["mean", "std", "count"]).reset_index()
    summary["formatted"] = summary.apply(lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1)
    
    # Create pivot table
    pivot = summary.pivot(index="model", columns="inputs", values="formatted")
    
    # Reorder
    model_order = ["ridge", "ridge_fast", "mlp", "trf"]
    input_order = ["self", "causal", "conc", "causal_self", "conc_self", "conc_causal"]
    
    pivot = pivot.reindex(index=[m for m in model_order if m in pivot.index])
    pivot = pivot.reindex(columns=[c for c in input_order if c in pivot.columns])
    
    # Save as markdown
    md_path = save_dir / "summary_table.md"
    with open(md_path, "w") as f:
        f.write("# Neural Activity Decoder v4 Summary\n\n")
        f.write(f"**Number of worms:** {df_r2['worm'].nunique()}\n\n")
        f.write("## R² by Model and Input Configuration\n\n")
        f.write("Values are mean ± std across worms.\n\n")
        # Write table manually without tabulate
        f.write("| model | " + " | ".join(pivot.columns) + " |\n")
        f.write("|---" + "|---" * len(pivot.columns) + "|\n")
        for idx, row in pivot.iterrows():
            f.write(f"| {idx} | " + " | ".join(str(v) if pd.notna(v) else "" for v in row.values) + " |\n")
        f.write("\n\n## Key Findings\n\n")
        
        # Compute key stats
        for model in model_order:
            if model in summary["model"].values:
                self_val = summary[(summary["model"] == model) & (summary["inputs"] == "self")]["mean"]
                cs_val = summary[(summary["model"] == model) & (summary["inputs"] == "causal_self")]["mean"]
                if len(self_val) > 0 and len(cs_val) > 0:
                    delta = cs_val.values[0] - self_val.values[0]
                    f.write(f"- **{model.upper()}**: Adding causal neighbors to self → ΔR² = {delta:+.4f}\n")
    
    print(f"Saved: {md_path}")
    
    # Also save CSV
    csv_path = save_dir / "summary_stats.csv"
    summary.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze neural_activity_decoder_v4 results")
    parser.add_argument("--results_dir", type=str, 
                       default="output_plots/neural_activity_decoder_v4",
                       help="Directory containing worm result folders")
    parser.add_argument("--save_dir", type=str, default=None,
                       help="Directory to save plots (default: results_dir)")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    save_dir = Path(args.save_dir) if args.save_dir else results_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from: {results_dir}")
    all_results = load_all_results(results_dir)
    print(f"Found {len(all_results)} worms: {list(all_results.keys())}")
    
    if not all_results:
        print("No results found!")
        return
    
    # Extract metrics
    df = extract_metrics(all_results)
    print(f"Extracted {len(df)} metric rows")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_model_comparison_bar(df, save_dir)
    plot_per_neuron_comparison(all_results, save_dir)
    create_summary_table(df, save_dir)
    
    print(f"\n✓ All plots saved to: {save_dir}")


def plot_per_neuron_comparison(all_results: dict, save_dir: Path):
    """Generate one bar plot per worm with each dot being a neuron."""
    
    models = ["ridge", "mlp", "trf"]
    model_labels = ["Ridge", "MLP", "Transformer"]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]
    
    input_configs = ["causal_self", "conc_self", "self", "causal", "conc", "conc_causal"]
    input_labels = ["Causal+Self", "Conc+Self", "Self only", "Causal only", "Conc only", "Conc+Causal"]
    
    # Create per-worm subdirectory for these plots
    per_worm_dir = save_dir / "per_worm_neurons"
    per_worm_dir.mkdir(parents=True, exist_ok=True)
    
    for worm, data in all_results.items():
        # Build data structure for this worm: model -> inputs -> list of per-neuron values
        neuron_data = {m: {inp: [] for inp in input_configs} for m in models}
        
        for k_val, metrics in data.items():
            for metric_name, values in metrics.items():
                if metric_name.startswith("r2_per_neuron_"):
                    # Parse: r2_per_neuron_{model}_{inputs}
                    parts = metric_name.replace("r2_per_neuron_", "").split("_", 1)
                    if len(parts) >= 2:
                        model = parts[0]
                        inputs = parts[1]
                        if model in neuron_data and inputs in neuron_data[model]:
                            neuron_data[model][inputs].extend(values)
        
        # Count neurons for this worm
        n_neurons = 0
        for inp in input_configs:
            if len(neuron_data["ridge"][inp]) > 0:
                n_neurons = len(neuron_data["ridge"][inp])
                break
        
        if n_neurons == 0:
            print(f"  Skipping {worm}: no per-neuron data found")
            continue
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        x = np.arange(len(input_configs))
        width = 0.25
        n_models = len(models)
        
        for i, (model, label, color) in enumerate(zip(models, model_labels, colors)):
            means = []
            for inp in input_configs:
                vals = neuron_data[model][inp]
                if len(vals) > 0:
                    means.append(np.mean(vals))
                else:
                    means.append(0)
            
            offset = (i - (n_models - 1) / 2) * width
            ax.bar(x + offset, means, width, 
                   label=label, color=color, alpha=0.7, edgecolor="white", linewidth=1)
            
            # Add individual neuron points with jitter
            for j, inp in enumerate(input_configs):
                vals = np.array(neuron_data[model][inp])
                if len(vals) > 0:
                    jitter = np.random.uniform(-width*0.35, width*0.35, len(vals))
                    ax.scatter(np.full_like(vals, j + offset) + jitter, vals, 
                              s=25, c=color, alpha=0.6, edgecolors="white", linewidths=0.3, zorder=5)
        
        ax.set_ylabel("R² (per neuron)", fontsize=12)
        ax.set_xlabel("Input Configuration", fontsize=12)
        ax.set_title(f"Worm: {worm} ({n_neurons} neurons)", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(input_labels, rotation=15, ha="right")
        ax.legend(loc="upper right")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylim(-1.0, 1.0)
        ax.grid(axis="y", alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(per_worm_dir / f"{worm}_per_neuron.png", dpi=150)
        plt.close()
    
    print(f"Saved: {per_worm_dir}/ ({len(all_results)} worm plots)")


if __name__ == "__main__":
    main()
