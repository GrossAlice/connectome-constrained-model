#!/usr/bin/env python3
"""
Summarize neural_activity_decoder_v4 results to inform Stage2 model design.

Key insights for LOO + Free-run model:
1. Self-history is dominant (AR structure essential)
2. Neighbors hurt linear models but help Transformer (need soft/gated connectivity)
3. LOO requires causal-only prediction (R²~0.2-0.3 is the ceiling without self)
4. Per-neuron predictability varies widely (heterogeneous dynamics)
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec


def load_per_neuron_data(results_dir: Path):
    """Load all per-neuron R² values."""
    all_data = {'ridge': {}, 'mlp': {}, 'trf': {}}
    input_configs = ['causal_self', 'conc_self', 'self', 'causal', 'conc', 'conc_causal']
    
    for subdir in sorted(results_dir.iterdir()):
        if subdir.is_dir() and subdir.name != 'per_worm_neurons':
            results_file = subdir / 'results.json'
            if results_file.exists():
                with open(results_file) as f:
                    data = json.load(f)
                for k_val, metrics in data.items():
                    for metric_name, value in metrics.items():
                        if metric_name.startswith('r2_per_neuron_'):
                            parts = metric_name.replace('r2_per_neuron_', '').split('_', 1)
                            model, inp = parts[0], parts[1]
                            if model in all_data and inp in input_configs:
                                if inp not in all_data[model]:
                                    all_data[model][inp] = []
                                all_data[model][inp].extend(value)
    
    return all_data


def create_stage2_insights_plot(results_dir: Path, save_dir: Path):
    """Create multi-panel summary plot for stage2 design."""
    
    all_data = load_per_neuron_data(results_dir)
    
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    models = ['ridge', 'mlp', 'trf']
    model_labels = ['Ridge', 'MLP', 'Transformer']
    colors = ['#2196F3', '#4CAF50', '#FF9800']
    
    # =========================================================================
    # Panel A: Self vs Causal+Self (what neighbors add)
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    x_pos = np.arange(len(models))
    width = 0.35
    
    self_means = []
    causal_self_means = []
    for m in models:
        self_means.append(np.mean(all_data[m]['self']))
        causal_self_means.append(np.mean(all_data[m]['causal_self']))
    
    bars1 = ax1.bar(x_pos - width/2, self_means, width, label='Self only', color='#90CAF9', edgecolor='black')
    bars2 = ax1.bar(x_pos + width/2, causal_self_means, width, label='Causal+Self', color='#FFCC80', edgecolor='black')
    
    # Add delta annotations
    for i, (s, cs) in enumerate(zip(self_means, causal_self_means)):
        delta = cs - s
        color = 'green' if delta > 0 else 'red'
        ax1.annotate(f'{delta:+.2f}', xy=(i + width/2, cs + 0.02), ha='center', fontsize=10, color=color, fontweight='bold')
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(model_labels)
    ax1.set_ylabel('R²', fontsize=12)
    ax1.set_title('A. Neighbor Contribution\n(Next-step prediction)', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 1)
    ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(axis='y', alpha=0.3)
    
    # =========================================================================
    # Panel B: Causal-only (LOO scenario - no self-history available)
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    causal_only = [np.array(all_data[m]['causal']) for m in models]
    
    parts = ax2.violinplot(causal_only, positions=x_pos, showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    # Add scatter
    for i, vals in enumerate(causal_only):
        jitter = np.random.uniform(-0.15, 0.15, len(vals))
        ax2.scatter(np.full_like(vals, i) + jitter, vals, s=8, alpha=0.3, c=colors[i])
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(model_labels)
    ax2.set_ylabel('R²', fontsize=12)
    ax2.set_title('B. Causal-Only (LOO Ceiling)\nNo self-history', fontsize=13, fontweight='bold')
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(0.3, color='red', linestyle=':', alpha=0.7, label='Practical ceiling ~0.3')
    ax2.set_ylim(-0.5, 0.8)
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend(loc='upper right', fontsize=9)
    
    # =========================================================================
    # Panel C: Per-neuron predictability histogram
    # =========================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    
    ridge_self = np.array(all_data['ridge']['self'])
    
    bins = np.linspace(-0.2, 1.0, 25)
    ax3.hist(ridge_self, bins=bins, color='#2196F3', alpha=0.7, edgecolor='white')
    
    # Mark thresholds
    ax3.axvline(0.5, color='orange', linestyle='--', label='R²=0.5 (noisy)', linewidth=2)
    ax3.axvline(0.8, color='green', linestyle='--', label='R²=0.8 (good)', linewidth=2)
    ax3.axvline(0.95, color='red', linestyle='--', label='R²=0.95 (excellent)', linewidth=2)
    
    ax3.set_xlabel('R² (Ridge self-only)', fontsize=11)
    ax3.set_ylabel('Count', fontsize=11)
    ax3.set_title('C. Neuron Predictability\nDistribution', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(axis='y', alpha=0.3)
    
    # Add counts
    low = (ridge_self < 0.5).sum()
    med = ((ridge_self >= 0.5) & (ridge_self < 0.8)).sum()
    high = ((ridge_self >= 0.8) & (ridge_self < 0.95)).sum()
    exc = (ridge_self >= 0.95).sum()
    ax3.text(0.95, 0.95, f'<0.5: {low}\n0.5-0.8: {med}\n0.8-0.95: {high}\n>0.95: {exc}',
             transform=ax3.transAxes, ha='right', va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # =========================================================================
    # Panel D: Delta R² per neuron (neighbors help?)
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 0])
    
    ridge_delta = np.array(all_data['ridge']['causal_self']) - np.array(all_data['ridge']['self'])
    trf_delta = np.array(all_data['trf']['causal_self']) - np.array(all_data['trf']['self'])
    
    bins = np.linspace(-0.5, 0.5, 41)
    ax4.hist(ridge_delta, bins=bins, alpha=0.6, label=f'Ridge (mean={ridge_delta.mean():.3f})', color='#2196F3')
    ax4.hist(trf_delta, bins=bins, alpha=0.6, label=f'Transformer (mean={trf_delta.mean():.3f})', color='#FF9800')
    
    ax4.axvline(0, color='black', linewidth=2)
    ax4.set_xlabel('ΔR² (causal+self - self)', fontsize=11)
    ax4.set_ylabel('Count', fontsize=11)
    ax4.set_title('D. Per-Neuron Neighbor Benefit\n(positive = neighbors help)', fontsize=13, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=10)
    ax4.grid(axis='y', alpha=0.3)
    
    # Annotations
    ridge_improved = (ridge_delta > 0).sum()
    trf_improved = (trf_delta > 0).sum()
    n_total = len(ridge_delta)
    ax4.text(0.02, 0.98, f'Ridge: {ridge_improved}/{n_total} improved ({100*ridge_improved/n_total:.0f}%)\nTrf: {trf_improved}/{n_total} improved ({100*trf_improved/n_total:.0f}%)',
             transform=ax4.transAxes, ha='left', va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # =========================================================================
    # Panel E: Self R² vs Causal R² scatter (relationship)
    # =========================================================================
    ax5 = fig.add_subplot(gs[1, 1])
    
    ridge_self = np.array(all_data['ridge']['self'])
    ridge_causal = np.array(all_data['ridge']['causal'])
    
    ax5.scatter(ridge_self, ridge_causal, s=20, alpha=0.5, c='#2196F3', edgecolors='none')
    ax5.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='y=x')
    
    # Fit line
    mask = (ridge_self > 0) & (ridge_causal > -0.5)
    z = np.polyfit(ridge_self[mask], ridge_causal[mask], 1)
    p = np.poly1d(z)
    x_fit = np.linspace(0, 1, 100)
    ax5.plot(x_fit, p(x_fit), 'r-', linewidth=2, label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
    
    ax5.set_xlabel('R² (self-only)', fontsize=11)
    ax5.set_ylabel('R² (causal-only)', fontsize=11)
    ax5.set_title('E. Self vs Causal Predictability\n(Ridge)', fontsize=13, fontweight='bold')
    ax5.set_xlim(-0.1, 1.05)
    ax5.set_ylim(-0.5, 0.8)
    ax5.legend(loc='upper left', fontsize=9)
    ax5.grid(alpha=0.3)
    
    # Correlation
    corr = np.corrcoef(ridge_self, ridge_causal)[0, 1]
    ax5.text(0.95, 0.05, f'r = {corr:.2f}', transform=ax5.transAxes, ha='right', va='bottom', fontsize=11)
    
    # =========================================================================
    # Panel F: Stage2 Design Recommendations (text)
    # =========================================================================
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    recommendations = """
STAGE2 MODEL DESIGN IMPLICATIONS

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ SELF-HISTORY IS DOMINANT
  • Ridge self-only: R² = 0.84
  • Must include strong AR component
  • λ_self >> λ_neighbors in regularization

✓ LINEAR NEIGHBORS HURT
  • Ridge: ΔR² = -0.13 with neighbors
  • Connectome must be soft/gated
  • Don't hard-code adjacency matrix

✓ TRANSFORMER EXTRACTS NEIGHBOR INFO
  • Only model where neighbors help
  • Attention can select relevant inputs
  • Consider attention-weighted connectivity

✓ LOO CEILING IS LOW
  • Causal-only: R² ≈ 0.2-0.3
  • LOO will never be as good as next-step
  • Manage expectations for held-out neurons

✓ NEURONS ARE HETEROGENEOUS
  • 35/686 (5%) hard to predict (R²<0.5)
  • 230/686 (34%) excellent (R²>0.95)
  • Consider per-neuron λ or weighting
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    ax6.text(0.05, 0.95, recommendations, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f5f5f5', edgecolor='gray'))
    
    # Main title
    fig.suptitle('Neural Activity Decoder Analysis → Stage2 Model Design',
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(save_dir / 'stage2_design_insights.png', dpi=150, bbox_inches='tight')
    plt.savefig(save_dir / 'stage2_design_insights.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_dir / 'stage2_design_insights.png'}")


def main():
    results_dir = Path('output_plots/neural_activity_decoder_v4')
    save_dir = results_dir
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return
    
    create_stage2_insights_plot(results_dir, save_dir)
    print("\n✓ Stage2 design insights plot generated")


if __name__ == "__main__":
    main()
