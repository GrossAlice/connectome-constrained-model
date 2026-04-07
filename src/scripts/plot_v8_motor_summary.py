#!/usr/bin/env python
"""
Generate summary plots for v8 per-worm models - motor neurons only.

Creates plots similar to the attached images:
1. R² per worm bar chart
2. Correlation per worm bar chart  
3. R² vs Neuron count scatter plot

For each model: Ridge n, Ridge n+b, MLP n, MLP n+b, TRF n, TRF n+b
"""
from __future__ import annotations

import json
import argparse
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_v8_results(v8_dir: Path, neurons: str = "motor"):
    """Load all v8 KbKn results for motor or all neurons."""
    pattern = f"results_*_{neurons}_KbKn.json"
    
    results = {}
    for json_path in sorted(v8_dir.glob(pattern)):
        # Extract worm_id from filename: results_{worm_id}_{neurons}_KbKn.json
        stem = json_path.stem
        parts = stem.replace("_KbKn", "").split("_")
        # Find where "motor" or "all" is
        for i, p in enumerate(parts):
            if p in ["motor", "all"]:
                worm_id = "_".join(parts[1:i])
                break
        else:
            worm_id = "_".join(parts[1:-1])
        
        with open(json_path) as f:
            data = json.load(f)
        results[worm_id] = data
    
    return results


def load_worm_neuron_counts(h5_dir: Path, motor_only: bool = True):
    """Load neuron counts for each worm."""
    import h5py
    
    counts = {}
    for h5_path in sorted(h5_dir.glob("*.h5")):
        worm_id = h5_path.stem
        try:
            with h5py.File(h5_path, 'r') as f:
                if 'gcamp/trace_array' in f:
                    u = f['gcamp/trace_array'][:]
                else:
                    u = f['gcamp/trace_array_original'][:]
                
                if u.shape[0] < u.shape[1]:
                    u = u.T
                
                total_neurons = u.shape[1]
                
                # Get motor neuron count if available
                motor_idx = None
                if 'neurons/motor_idx' in f:
                    motor_idx = f['neurons/motor_idx'][:]
                elif 'meta/motor_idx' in f:
                    motor_idx = f['meta/motor_idx'][:]
                
                if motor_only and motor_idx is not None:
                    counts[worm_id] = len(motor_idx)
                else:
                    counts[worm_id] = total_neurons
        except Exception as e:
            print(f"Warning: Could not load {worm_id}: {e}")
            continue
    
    return counts


# ══════════════════════════════════════════════════════════════════════════════
# Plotting functions
# ══════════════════════════════════════════════════════════════════════════════

def plot_model_summary(results: dict, out_dir: Path, neurons: str, model_key: str, 
                       model_label: str, K: int = 5, color: str = "#8B4513",
                       neuron_counts: dict = None):
    """Create 3-panel summary plot for a single model."""
    
    worm_ids = sorted(results.keys())
    
    # Extract metrics for the specified K and model
    r2_vals = []
    corr_vals = []
    n_neurons_list = []
    valid_worms = []
    
    K_str = str(K)
    for worm_id in worm_ids:
        if K_str not in results[worm_id]:
            continue
        K_data = results[worm_id][K_str]
        
        if model_key not in K_data:
            continue
        
        metrics = K_data[model_key]
        r2 = metrics.get("r2_mean", np.nan)
        corr = metrics.get("corr_mean", np.nan)
        
        if not np.isnan(r2) and not np.isnan(corr):
            r2_vals.append(r2)
            corr_vals.append(corr)
            valid_worms.append(worm_id)
            
            # Get neuron count
            if neuron_counts and worm_id in neuron_counts:
                n_neurons_list.append(neuron_counts[worm_id])
            else:
                n_neurons_list.append(np.nan)
    
    if not r2_vals:
        print(f"  Skipping {model_label}: no data found")
        return
    
    r2_vals = np.array(r2_vals)
    corr_vals = np.array(corr_vals)
    n_neurons = np.array(n_neurons_list)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    # ── Panel 1: R² per worm ──
    ax = axes[0]
    x = np.arange(len(valid_worms))
    ax.bar(x, r2_vals, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
    mean_r2 = np.mean(r2_vals)
    ax.axhline(mean_r2, color='red', linestyle='--', linewidth=1.5, label=f'Mean={mean_r2:.3f}')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_worms, rotation=90, fontsize=7)
    ax.set_ylabel('R²', fontsize=11)
    ax.set_title('R² per worm', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(min(0, r2_vals.min() - 0.05), max(0.5, r2_vals.max() + 0.05))
    ax.grid(axis='y', alpha=0.3)
    
    # ── Panel 2: Correlation per worm ──
    ax = axes[1]
    ax.bar(x, corr_vals, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
    mean_corr = np.mean(corr_vals)
    ax.axhline(mean_corr, color='red', linestyle='--', linewidth=1.5, label=f'Mean={mean_corr:.3f}')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_worms, rotation=90, fontsize=7)
    ax.set_ylabel('Correlation', fontsize=11)
    ax.set_title('Correlation per worm', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, max(0.7, corr_vals.max() + 0.05))
    ax.grid(axis='y', alpha=0.3)
    
    # ── Panel 3: R² vs Neuron count ──
    ax = axes[2]
    valid_neurons = ~np.isnan(n_neurons)
    if valid_neurons.sum() > 3:
        ax.scatter(n_neurons[valid_neurons], r2_vals[valid_neurons], 
                   color=color, alpha=0.7, s=50, edgecolor='black', linewidth=0.5)
        
        # Fit regression line
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            n_neurons[valid_neurons], r2_vals[valid_neurons])
        x_line = np.linspace(n_neurons[valid_neurons].min(), n_neurons[valid_neurons].max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        ax.text(0.95, 0.95, f'r={r_value:.2f}', transform=ax.transAxes, 
                ha='right', va='top', fontsize=10)
    else:
        ax.scatter(n_neurons[valid_neurons], r2_vals[valid_neurons], 
                   color=color, alpha=0.7, s=50, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Number of neurons', fontsize=11)
    ax.set_ylabel('R²', fontsize=11)
    ax.set_title('R² vs Neuron count', fontsize=12)
    ax.grid(alpha=0.3)
    
    # Main title
    fig.suptitle(f'v8 Per-Worm {model_label} (K={K})', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    safe_name = model_label.replace(' ', '_').replace('+', 'p')
    out_path = out_dir / f"v8_perworm_{neurons}_{safe_name}_K{K}.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path.name}")
    
    return {
        'model': model_label,
        'r2_mean': mean_r2,
        'r2_std': np.std(r2_vals),
        'corr_mean': mean_corr,
        'corr_std': np.std(corr_vals),
        'n_worms': len(valid_worms),
    }


def main():
    parser = argparse.ArgumentParser(description="Plot v8 summary for motor neurons")
    parser.add_argument('--v8_dir', default='output_plots/behaviour_decoder/v8_KbKn',
                        help='Directory with v8 KbKn results')
    parser.add_argument('--h5_dir', default='data/used/behaviour+neuronal activity atanas (2023)/2',
                        help='Directory with H5 files for neuron counts')
    parser.add_argument('--out', default='output_plots/behaviour_decoder/v8_KbKn',
                        help='Output directory')
    parser.add_argument('--neurons', default='motor', choices=['motor', 'all'],
                        help='Neuron subset to plot')
    parser.add_argument('--K', type=int, default=5, help='Context length K')
    args = parser.parse_args()
    
    v8_dir = Path(args.v8_dir)
    h5_dir = Path(args.h5_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading v8 results from: {v8_dir}")
    results = load_v8_results(v8_dir, args.neurons)
    print(f"  Loaded {len(results)} worms")
    
    print(f"Loading neuron counts from: {h5_dir}")
    neuron_counts = load_worm_neuron_counts(h5_dir, motor_only=(args.neurons == 'motor'))
    print(f"  Loaded counts for {len(neuron_counts)} worms")
    
    # Define models to plot
    # Format: (model_key, display_label, color)
    models = [
        ('ridge_n_fr', 'Ridge n FR', '#BC8F8F'),      # Rosy brown
        ('ridge_nb_fr', 'Ridge n+b FR', '#9370DB'),   # Medium purple
        ('mlp_n_fr', 'MLP n FR', '#4682B4'),          # Steel blue
        ('mlp_nb_fr', 'MLP n+b FR', '#3CB371'),       # Medium sea green
        ('trf_large_n_fr', 'TRF n FR', '#FFA500'),    # Orange
        ('trf_large_nb_fr', 'TRF n+b FR', '#CD5C5C'), # Indian red
    ]
    
    print(f"\nGenerating summary plots for K={args.K}, neurons={args.neurons}...")
    
    summaries = []
    for model_key, model_label, color in models:
        summary = plot_model_summary(
            results, out_dir, args.neurons, model_key, model_label, 
            K=args.K, color=color, neuron_counts=neuron_counts
        )
        if summary:
            summaries.append(summary)
    
    # Print summary table
    print(f"\n{'='*70}")
    print(f"SUMMARY — K={args.K}, neurons={args.neurons}")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'R² (mean±std)':<20} {'Corr (mean±std)':<20}")
    print("-" * 70)
    for s in summaries:
        print(f"{s['model']:<20} {s['r2_mean']:.3f}±{s['r2_std']:.3f}        "
              f"{s['corr_mean']:.3f}±{s['corr_std']:.3f}")
    print(f"\nPlots saved to: {out_dir}")


if __name__ == '__main__':
    main()
