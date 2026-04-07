#!/usr/bin/env python3
"""
Generate summary plots for v8_KbKn results:
1. Per-eigenworm correlation bar plots (grouped by model)
2. Violin plots for R² and Correlation across all worms
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import argparse


def load_all_results(results_dir: Path, neuron_type: str = 'motor'):
    """Load all JSON results for given neuron type."""
    results = {}
    pattern = f"results_*_{neuron_type}_KbKn.json"
    
    for f in sorted(results_dir.glob(pattern)):
        worm_id = f.stem.replace(f'results_', '').replace(f'_{neuron_type}_KbKn', '')
        with open(f) as fp:
            results[worm_id] = json.load(fp)
    
    return results


def get_model_display_names():
    """Map internal model names to display names."""
    return {
        'trf_large_nb_fr': 'TRF n+b FR',
        'trf_large_n_fr': 'TRF n FR',
        'trf_large_b_1s': 'TRF b (1-step)',
        'mlp_nb_fr': 'MLP n+b FR',
        'mlp_n_fr': 'MLP n FR',
        'mlp_b_1s': 'MLP b (1-step)',
        'ridge_nb_fr': 'Ridge n+b FR',
        'ridge_n_fr': 'Ridge n FR',
        'ridge_b_1s': 'Ridge b (1-step)',
    }


def get_model_colors():
    """Get colors for each model type."""
    return {
        'TRF': '#E57373',   # red
        'MLP': '#64B5F6',   # blue  
        'Ridge': '#81C784', # green
    }


def plot_per_eigenworm_correlation(results_motor: dict, results_all: dict, 
                                    K: int, out_dir: Path):
    """
    Plot correlation per eigenworm for each model.
    Shows motor neurons (dark) vs all neurons (light) side by side.
    """
    display_names = get_model_display_names()
    colors = get_model_colors()
    
    # Models to plot (in order)
    models = [
        'trf_large_nb_fr', 'trf_large_n_fr', 'trf_large_b_1s',
        'mlp_nb_fr', 'mlp_n_fr', 'mlp_b_1s',
        'ridge_nb_fr', 'ridge_n_fr', 'ridge_b_1s',
    ]
    
    K_str = str(K)
    n_ew = 6  # eigenworms
    n_worms = len(results_motor)
    
    # Collect per-eigenworm correlations for each model
    # Shape: {model: {ew_idx: [correlations across worms]}}
    motor_corrs = {m: {i: [] for i in range(n_ew)} for m in models}
    all_corrs = {m: {i: [] for i in range(n_ew)} for m in models}
    
    for worm_id in results_motor:
        if K_str not in results_motor[worm_id]:
            continue
        motor_data = results_motor[worm_id][K_str]
        all_data = results_all.get(worm_id, {}).get(K_str, {})
        
        for m in models:
            if m in motor_data and 'corr_per_mode' in motor_data[m]:
                for i, c in enumerate(motor_data[m]['corr_per_mode'][:n_ew]):
                    motor_corrs[m][i].append(c)
            if m in all_data and 'corr_per_mode' in all_data[m]:
                for i, c in enumerate(all_data[m]['corr_per_mode'][:n_ew]):
                    all_corrs[m][i].append(c)
    
    # Create figure with subplots for each eigenworm
    fig, axes = plt.subplots(1, n_ew, figsize=(24, 5), sharey=True)
    fig.suptitle(f'Behaviour Decoding Summary — K = {K},  Correlation per Eigenworm  ({n_worms} worms)', 
                 fontsize=14, fontweight='bold')
    
    bar_width = 0.35
    n_models = len(models)
    x = np.arange(n_models)
    
    for ew_idx, ax in enumerate(axes):
        motor_means = []
        motor_stds = []
        all_means = []
        all_stds = []
        
        for m in models:
            if motor_corrs[m][ew_idx]:
                motor_means.append(np.mean(motor_corrs[m][ew_idx]))
                motor_stds.append(np.std(motor_corrs[m][ew_idx]))
            else:
                motor_means.append(0)
                motor_stds.append(0)
                
            if all_corrs[m][ew_idx]:
                all_means.append(np.mean(all_corrs[m][ew_idx]))
                all_stds.append(np.std(all_corrs[m][ew_idx]))
            else:
                all_means.append(0)
                all_stds.append(0)
        
        # Get colors based on model type
        bar_colors_motor = []
        bar_colors_all = []
        for m in models:
            name = display_names[m]
            if 'TRF' in name:
                base_color = colors['TRF']
            elif 'MLP' in name:
                base_color = colors['MLP']
            else:
                base_color = colors['Ridge']
            bar_colors_motor.append(base_color)
            # Lighter version for all neurons
            import matplotlib.colors as mcolors
            rgb = mcolors.to_rgb(base_color)
            lighter = tuple(min(1, c + 0.3) for c in rgb)
            bar_colors_all.append(lighter)
        
        # Plot bars
        bars_motor = ax.bar(x - bar_width/2, motor_means, bar_width, 
                           label='Motor neurons', color=bar_colors_motor, alpha=0.9)
        bars_all = ax.bar(x + bar_width/2, all_means, bar_width,
                         label='All neurons', color=bar_colors_all, alpha=0.7)
        
        # Add individual worm points
        for i, m in enumerate(models):
            if motor_corrs[m][ew_idx]:
                jitter = np.random.uniform(-0.05, 0.05, len(motor_corrs[m][ew_idx]))
                ax.scatter(np.full(len(motor_corrs[m][ew_idx]), i - bar_width/2) + jitter,
                          motor_corrs[m][ew_idx], color='black', s=15, alpha=0.5, zorder=3)
            if all_corrs[m][ew_idx]:
                jitter = np.random.uniform(-0.05, 0.05, len(all_corrs[m][ew_idx]))
                ax.scatter(np.full(len(all_corrs[m][ew_idx]), i + bar_width/2) + jitter,
                          all_corrs[m][ew_idx], color='gray', s=15, alpha=0.5, zorder=3)
        
        ax.set_title(f'EW{ew_idx+1}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([display_names[m].replace(' ', '\n') for m in models], 
                          rotation=45, ha='right', fontsize=8)
        ax.set_ylim(0, 1.05)
        
        if ew_idx == 0:
            ax.set_ylabel('Pearson Correlation', fontsize=11)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', alpha=0.9, label='Motor neurons'),
        Patch(facecolor='gray', alpha=0.4, label='All neurons'),
        Patch(facecolor=colors['TRF'], label='Transformer'),
        Patch(facecolor=colors['MLP'], label='MLP'),
        Patch(facecolor=colors['Ridge'], label='Ridge'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02),
               ncol=5, fontsize=10)
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    out_path = out_dir / f'summary_corr_per_ew_K{K}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_violin_summary(results: dict, K: int, neuron_type: str, out_dir: Path):
    """
    Plot violin plots for R² and Correlation across all worms.
    Similar to the second attached image.
    """
    display_names = get_model_display_names()
    colors = get_model_colors()
    
    # Models to plot (in order matching the image)
    models = [
        'trf_large_nb_fr', 'trf_large_n_fr', 'trf_large_b_1s',
        'mlp_nb_fr', 'mlp_n_fr', 'mlp_b_1s',
        'ridge_nb_fr', 'ridge_n_fr', 'ridge_b_1s',
    ]
    
    K_str = str(K)
    n_worms = len(results)
    
    # Collect R² and correlations for each model
    r2_data = {m: [] for m in models}
    corr_data = {m: [] for m in models}
    
    for worm_id, worm_data in results.items():
        if K_str not in worm_data:
            continue
        data = worm_data[K_str]
        
        for m in models:
            if m in data:
                r2_data[m].append(data[m].get('r2_mean', np.nan))
                corr_data[m].append(data[m].get('corr_mean', np.nan))
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Behaviour Decoding at K={K} — {n_worms} worms, {neuron_type} neurons', 
                 fontsize=14, fontweight='bold')
    
    # Get colors for each model
    model_colors = []
    for m in models:
        name = display_names[m]
        if 'TRF' in name:
            model_colors.append(colors['TRF'])
        elif 'MLP' in name:
            model_colors.append(colors['MLP'])
        else:
            model_colors.append(colors['Ridge'])
    
    # R² violin plot
    positions = range(len(models))
    r2_values = [r2_data[m] for m in models]
    
    parts1 = ax1.violinplot(r2_values, positions=positions, showmeans=True, showmedians=True)
    for i, pc in enumerate(parts1['bodies']):
        pc.set_facecolor(model_colors[i])
        pc.set_alpha(0.6)
    for partname in ('cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes'):
        if partname in parts1:
            parts1[partname].set_color('blue')
            parts1[partname].set_linewidth(1.5)
    
    ax1.set_ylabel('R²', fontsize=12)
    ax1.set_xticks(positions)
    ax1.set_xticklabels([display_names[m] for m in models], rotation=45, ha='right', fontsize=9)
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title('R²', fontsize=12)
    
    # Add mean±std annotation
    for i, m in enumerate(models):
        if r2_data[m]:
            mean_val = np.mean(r2_data[m])
            ax1.scatter(i, mean_val, color='white', s=30, zorder=5, edgecolor='black')
    
    # Correlation violin plot
    corr_values = [corr_data[m] for m in models]
    
    parts2 = ax2.violinplot(corr_values, positions=positions, showmeans=True, showmedians=True)
    for i, pc in enumerate(parts2['bodies']):
        pc.set_facecolor(model_colors[i])
        pc.set_alpha(0.6)
    for partname in ('cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes'):
        if partname in parts2:
            parts2[partname].set_color('blue')
            parts2[partname].set_linewidth(1.5)
    
    ax2.set_ylabel('Correlation', fontsize=12)
    ax2.set_xticks(positions)
    ax2.set_xticklabels([display_names[m] for m in models], rotation=45, ha='right', fontsize=9)
    ax2.set_title('Correlation', fontsize=12)
    
    # Add mean annotation
    for i, m in enumerate(models):
        if corr_data[m]:
            mean_val = np.mean(corr_data[m])
            ax2.scatter(i, mean_val, color='white', s=30, zorder=5, edgecolor='black')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    out_path = out_dir / f'summary_violin_K{K}_{neuron_type}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_per_eigenworm_r2(results_motor: dict, results_all: dict, 
                          K: int, out_dir: Path):
    """
    Plot R² per eigenworm for each model.
    Shows motor neurons (dark) vs all neurons (light) side by side.
    """
    display_names = get_model_display_names()
    colors = get_model_colors()
    
    models = [
        'trf_large_nb_fr', 'trf_large_n_fr', 'trf_large_b_1s',
        'mlp_nb_fr', 'mlp_n_fr', 'mlp_b_1s',
        'ridge_nb_fr', 'ridge_n_fr', 'ridge_b_1s',
    ]
    
    K_str = str(K)
    n_ew = 6
    n_worms = len(results_motor)
    
    motor_r2s = {m: {i: [] for i in range(n_ew)} for m in models}
    all_r2s = {m: {i: [] for i in range(n_ew)} for m in models}
    
    for worm_id in results_motor:
        if K_str not in results_motor[worm_id]:
            continue
        motor_data = results_motor[worm_id][K_str]
        all_data = results_all.get(worm_id, {}).get(K_str, {})
        
        for m in models:
            if m in motor_data and 'r2_per_mode' in motor_data[m]:
                for i, r2 in enumerate(motor_data[m]['r2_per_mode'][:n_ew]):
                    motor_r2s[m][i].append(r2)
            if m in all_data and 'r2_per_mode' in all_data[m]:
                for i, r2 in enumerate(all_data[m]['r2_per_mode'][:n_ew]):
                    all_r2s[m][i].append(r2)
    
    fig, axes = plt.subplots(1, n_ew, figsize=(24, 5), sharey=True)
    fig.suptitle(f'Behaviour Decoding Summary — K = {K},  R² per Eigenworm  ({n_worms} worms)', 
                 fontsize=14, fontweight='bold')
    
    bar_width = 0.35
    n_models = len(models)
    x = np.arange(n_models)
    
    for ew_idx, ax in enumerate(axes):
        motor_means = []
        all_means = []
        
        for m in models:
            if motor_r2s[m][ew_idx]:
                motor_means.append(np.mean(motor_r2s[m][ew_idx]))
            else:
                motor_means.append(0)
                
            if all_r2s[m][ew_idx]:
                all_means.append(np.mean(all_r2s[m][ew_idx]))
            else:
                all_means.append(0)
        
        bar_colors_motor = []
        bar_colors_all = []
        for m in models:
            name = display_names[m]
            if 'TRF' in name:
                base_color = colors['TRF']
            elif 'MLP' in name:
                base_color = colors['MLP']
            else:
                base_color = colors['Ridge']
            bar_colors_motor.append(base_color)
            import matplotlib.colors as mcolors
            rgb = mcolors.to_rgb(base_color)
            lighter = tuple(min(1, c + 0.3) for c in rgb)
            bar_colors_all.append(lighter)
        
        ax.bar(x - bar_width/2, motor_means, bar_width, 
               label='Motor neurons', color=bar_colors_motor, alpha=0.9)
        ax.bar(x + bar_width/2, all_means, bar_width,
               label='All neurons', color=bar_colors_all, alpha=0.7)
        
        # Add individual points
        for i, m in enumerate(models):
            if motor_r2s[m][ew_idx]:
                jitter = np.random.uniform(-0.05, 0.05, len(motor_r2s[m][ew_idx]))
                ax.scatter(np.full(len(motor_r2s[m][ew_idx]), i - bar_width/2) + jitter,
                          motor_r2s[m][ew_idx], color='black', s=15, alpha=0.5, zorder=3)
            if all_r2s[m][ew_idx]:
                jitter = np.random.uniform(-0.05, 0.05, len(all_r2s[m][ew_idx]))
                ax.scatter(np.full(len(all_r2s[m][ew_idx]), i + bar_width/2) + jitter,
                          all_r2s[m][ew_idx], color='gray', s=15, alpha=0.5, zorder=3)
        
        ax.set_title(f'EW{ew_idx+1}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([display_names[m].replace(' ', '\n') for m in models], 
                          rotation=45, ha='right', fontsize=8)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
        
        if ew_idx == 0:
            ax.set_ylabel('R²', fontsize=11)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', alpha=0.9, label='Motor neurons'),
        Patch(facecolor='gray', alpha=0.4, label='All neurons'),
        Patch(facecolor=colors['TRF'], label='Transformer'),
        Patch(facecolor=colors['MLP'], label='MLP'),
        Patch(facecolor=colors['Ridge'], label='Ridge'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02),
               ncol=5, fontsize=10)
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    out_path = out_dir / f'summary_r2_per_ew_K{K}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate v8_KbKn summary plots')
    parser.add_argument('--results_dir', type=str, 
                        default='output_plots/behaviour_decoder/v8_KbKn',
                        help='Directory with JSON results')
    parser.add_argument('--K', type=int, default=15, help='Context length to plot')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    print(f"Loading results from {results_dir}...")
    results_motor = load_all_results(results_dir, 'motor')
    results_all = load_all_results(results_dir, 'all')
    
    print(f"Found {len(results_motor)} motor and {len(results_all)} all neuron results")
    
    K = args.K
    
    # Generate all plots
    print(f"\nGenerating plots for K={K}...")
    
    # 1. Per-eigenworm correlation (motor vs all)
    plot_per_eigenworm_correlation(results_motor, results_all, K, results_dir)
    
    # 2. Per-eigenworm R² (motor vs all)
    plot_per_eigenworm_r2(results_motor, results_all, K, results_dir)
    
    # 3. Violin plots for motor neurons
    plot_violin_summary(results_motor, K, 'motor', results_dir)
    
    # 4. Violin plots for all neurons
    plot_violin_summary(results_all, K, 'all', results_dir)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
