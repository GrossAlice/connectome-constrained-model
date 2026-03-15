#!/usr/bin/env python
"""
build_connectome.py
===================

Build connectivity matrices for the C. elegans nervous system.

This script produces three connectivity matrices:
- T_sv.npy: Synaptic vesicle (fast chemical synapses)
- T_dcv.npy: Dense core vesicle (slow neuromodulation)  
- T_e.npy: Electrical (gap junction) connections

USAGE
-----

Run with default paths (assumes data files in src/data/):

```bash
python build_connectome.py
```

This uses:
  - src/data/chem_adj.pkl
  - src/data/gapjn_symm_adj.pkl (or data/moza (2025)/gapjn_symm_adj.pkl)
  - src/data/NPP_GPCR_networks_long_range_model_2.csv
  - src/data/26012022_num_neuronID.txt

And produces:
  - T_sv.npy (chemical synapses)
  - T_dcv.npy (neuropeptide modulation)
  - T_e.npy (gap junctions)

Custom paths (optional):

```bash
python build_connectome.py \
  --chem_adj /path/to/chem_adj.pkl \
  --gap_adj /path/to/gapjn_symm_adj.pkl \
  --peptide_adj /path/to/NPP_GPCR_networks_long_range_model_2.csv \
  --peptide_map /path/to/26012022_num_neuronID.txt
```

OUTPUTS
-------
- T_sv.npy: (N×N) All chemical synapses from anatomical connectome
- T_dcv.npy: (N×N) Neuropeptide modulation
- T_e.npy: (N×N) Gap junction electrical connections

where N is the total number of neurons.
"""

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving


# -----------------------------------------------------------------------------
# Data loading functions
# -----------------------------------------------------------------------------

def load_chem_adj(path: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Load chemical adjacency from pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_gap_adj(path: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Load gap junction adjacency from pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def build_neuron_index(
    chem_adj: Dict,
    gap_adj: Dict,
    extra_neurons: List[str] = None,
) -> Tuple[List[str], Dict[str, int]]:
    """Create sorted list of neuron names and name→index mapping."""
    neuron_set = set(chem_adj.keys())
    for posts in chem_adj.values():
        neuron_set.update(posts.keys())
    neuron_set.update(gap_adj.keys())
    for posts in gap_adj.values():
        neuron_set.update(posts.keys())
    if extra_neurons:
        neuron_set.update(extra_neurons)
    neurons = sorted(neuron_set)
    return neurons, {n: i for i, n in enumerate(neurons)}


def load_bentley_data(
    monoamine_path: str, neuropeptide_path: str
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]:
    """Load Bentley et al. 2016 monoamine and neuropeptide connectivity.
    
    Converts neuron names to match anatomical convention (AS1 → AS01, etc.)
    
    Returns
    -------
    monoamine_adj : Dict[str, Dict[str, int]]
        Monoamine connections: {pre_neuron: {post_neuron: count}}
    neuropeptide_adj : Dict[str, Dict[str, int]]
        Neuropeptide connections: {pre_neuron: {post_neuron: count}}
    """
    def standardize_neuron_name(name: str) -> str:
        """Convert AS1 → AS01, DA2 → DA02, etc. to match anatomical naming.
        Also handles I03 → I3, M01 → M1 for pharyngeal neurons that don't use padding."""
        import re
        # Match pattern: letters followed by one or two digits
        match = re.match(r'^([A-Z]+)(\d{1,2})$', name)
        if match:
            prefix, digits = match.groups()
            # For single-letter prefixes like I or M, remove leading zero
            if len(prefix) == 1 and len(digits) == 2 and digits.startswith('0'):
                return f"{prefix}{digits[1]}"
            # For multi-letter prefixes, add leading zero for single digits
            elif len(prefix) > 1 and len(digits) == 1:
                return f"{prefix}0{digits}"
        return name
    
    # Load monoamines
    mono_df = pd.read_csv(monoamine_path)
    mono_df.columns = mono_df.columns.str.strip()  # Remove leading spaces
    
    monoamine_adj: Dict[str, Dict[str, int]] = {}
    for _, row in mono_df.iterrows():
        pre = standardize_neuron_name(str(row['#source neuron']).strip())
        post = standardize_neuron_name(str(row['target neuron']).strip())
        
        if pre not in monoamine_adj:
            monoamine_adj[pre] = {}
        monoamine_adj[pre][post] = monoamine_adj[pre].get(post, 0) + 1
    
    # Load neuropeptides
    neuro_df = pd.read_csv(neuropeptide_path)
    neuro_df.columns = neuro_df.columns.str.strip()
    
    neuropeptide_adj: Dict[str, Dict[str, int]] = {}
    for _, row in neuro_df.iterrows():
        pre = standardize_neuron_name(str(row['#source neuron']).strip())
        post = standardize_neuron_name(str(row['target neuron']).strip())
        
        if pre not in neuropeptide_adj:
            neuropeptide_adj[pre] = {}
        neuropeptide_adj[pre][post] = neuropeptide_adj[pre].get(post, 0) + 1
    
    return monoamine_adj, neuropeptide_adj


# -----------------------------------------------------------------------------
# Matrix construction functions
# -----------------------------------------------------------------------------

def construct_t_sv(
    chem_adj: Dict,
    neuron_index: Dict[str, int],
) -> np.ndarray:
    """Construct synaptic vesicle (fast chemical synapse) matrix.
    
    Directly copies all chemical synapse weights from chem_adj.pkl
    without filtering by neurotransmitter type or receptor expression.
    """
    n = len(neuron_index)
    t_sv = np.zeros((n, n), dtype=float)
    
    for pre, posts in chem_adj.items():
        pre_idx = neuron_index.get(pre)
        if pre_idx is None:
            continue
            
        for post, info in posts.items():
            post_idx = neuron_index.get(post)
            if post_idx is None:
                continue
            
            weight = info.get("weight")
            # Skip NaN and zero weights
            if weight is None or (isinstance(weight, float) and np.isnan(weight)) or weight == 0:
                continue
            
            t_sv[post_idx, pre_idx] = float(weight)
    
    return t_sv


def construct_t_dcv_bentley(
    monoamine_adj: Dict[str, Dict[str, int]],
    neuropeptide_adj: Dict[str, Dict[str, int]],
    neuron_index: Dict[str, int],
) -> np.ndarray:
    """Construct DCV matrix from Bentley et al. 2016 monoamine and neuropeptide data.
    
    Combines monoamine and neuropeptide connections by summing their connection counts.
    """
    n = len(neuron_index)
    t_dcv = np.zeros((n, n), dtype=float)
    
    # Add monoamine connections
    for pre, posts in monoamine_adj.items():
        pre_idx = neuron_index.get(pre)
        if pre_idx is None:
            continue
        for post, count in posts.items():
            post_idx = neuron_index.get(post)
            if post_idx is None:
                continue
            t_dcv[post_idx, pre_idx] += float(count)
    
    # Add neuropeptide connections
    for pre, posts in neuropeptide_adj.items():
        pre_idx = neuron_index.get(pre)
        if pre_idx is None:
            continue
        for post, count in posts.items():
            post_idx = neuron_index.get(post)
            if post_idx is None:
                continue
            t_dcv[post_idx, pre_idx] += float(count)
    
    return t_dcv


def construct_t_e(
    gap_adj: Dict,
    neuron_index: Dict[str, int],
) -> np.ndarray:
    """Construct electrical (gap junction) connectivity matrix."""
    n = len(neuron_index)
    t_e = np.zeros((n, n), dtype=float)
    for n1, connections in gap_adj.items():
        n1_idx = neuron_index.get(n1)
        if n1_idx is None:
            continue
        for n2, info in connections.items():
            n2_idx = neuron_index.get(n2)
            if n2_idx is None:
                continue
            weight = info.get("weight", 0)
            if weight > 0:
                # Gap junctions are bidirectional
                t_e[n1_idx, n2_idx] = float(weight)
                t_e[n2_idx, n1_idx] = float(weight)
    return t_e


# -----------------------------------------------------------------------------
# Plotting functions
# -----------------------------------------------------------------------------

def plot_connectivity_matrix(
    matrix: np.ndarray,
    neuron_names: List[str],
    title: str,
    save_path: Path,
    cmap: str = 'viridis',
    log_scale: bool = False,
    max_labels: int = 50,
):
    """Plot connectivity matrix as heatmap with neuron labels.
    
    Parameters
    ----------
    matrix : np.ndarray
        (N, N) connectivity matrix
    neuron_names : List[str]
        List of neuron names (length N)
    title : str
        Plot title
    save_path : Path
        Where to save the plot
    cmap : str
        Colormap name
    log_scale : bool
        Use log scale for colormap
    max_labels : int
        Maximum number of axis labels to show (for readability)
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Apply log scale if requested
    plot_data = matrix.copy()
    if log_scale and np.any(matrix > 0):
        # Use log scale, mask zeros to show as black
        plot_data = np.where(matrix > 0, np.log10(matrix + 1), 0)
        cbar_label = 'log10(weight + 1)'
    else:
        plot_data = matrix.copy()
        cbar_label = 'weight'
    
    # Create masked array where zeros will be black
    plot_data_masked = np.ma.masked_where(plot_data == 0, plot_data)
    
    # Get colormap and set color for masked (zero) values to black
    cmap_obj = matplotlib.colormaps.get_cmap(cmap).copy()
    cmap_obj.set_bad(color='black')
    
    # Plot heatmap
    im = ax.imshow(plot_data_masked, cmap=cmap_obj, aspect='auto', interpolation='nearest')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, rotation=270, labelpad=20)
    
    # Labels - show subset if too many neurons
    n = len(neuron_names)
    if n <= max_labels:
        tick_indices = np.arange(n)
        tick_labels = neuron_names
    else:
        # Show evenly spaced subset
        tick_indices = np.linspace(0, n-1, max_labels, dtype=int)
        tick_labels = [neuron_names[i] for i in tick_indices]
    
    ax.set_xticks(tick_indices)
    ax.set_yticks(tick_indices)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
    ax.set_yticklabels(tick_labels, fontsize=6)
    
    ax.set_xlabel('Presynaptic neuron', fontsize=10)
    ax.set_ylabel('Postsynaptic neuron', fontsize=10)
    ax.set_title(f'{title}\n({n} neurons, {np.count_nonzero(matrix)} non-zero connections)', 
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved plot: {save_path}")


def plot_connectivity_summary(
    t_sv: np.ndarray,
    t_dcv: np.ndarray,
    t_e: np.ndarray,
    neuron_names: List[str],
    save_dir: Path,
):
    """Create summary plots for all three connectivity types."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating connectivity plots...")
    
    # Individual matrix plots
    plot_connectivity_matrix(
        t_sv, neuron_names,
        'T_sv: Fast Chemical Synapses',
        save_dir / 'T_sv_heatmap.png',
        cmap='Reds',
        log_scale=True,
    )
    
    plot_connectivity_matrix(
        t_dcv, neuron_names,
        'T_dcv: Neuropeptide Modulation',
        save_dir / 'T_dcv_heatmap.png',
        cmap='Blues',
        log_scale=True,
    )
    
    plot_connectivity_matrix(
        t_e, neuron_names,
        'T_e: Gap Junctions',
        save_dir / 'T_e_heatmap.png',
        cmap='Greens',
        log_scale=False,
    )
    
    # Combined overview
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, matrix, title, cmap in zip(
        axes,
        [t_sv, t_dcv, t_e],
        ['T_sv (Chemical)', 'T_dcv (Neuropeptide)', 'T_e (Gap Junctions)'],
        ['Reds', 'Blues', 'Greens']
    ):
        # Use log scale for better visualization, mask zeros to show as black
        plot_data = np.where(matrix > 0, np.log10(matrix + 1), 0)
        plot_data_masked = np.ma.masked_where(plot_data == 0, plot_data)
        
        # Set black color for zero connections
        cmap_obj = matplotlib.colormaps.get_cmap(cmap).copy()
        cmap_obj.set_bad(color='black')
        
        im = ax.imshow(plot_data_masked, cmap=cmap_obj, aspect='auto', interpolation='nearest')
        ax.set_title(f'{title}\n{np.count_nonzero(matrix)} connections', fontsize=10)
        ax.set_xlabel('Pre', fontsize=8)
        ax.set_ylabel('Post', fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, label='log10(w+1)')
    
    fig.suptitle(f'C. elegans Connectome Overview ({len(neuron_names)} neurons)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(save_dir / 'connectome_overview.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved plot: {save_dir / 'connectome_overview.png'}")
    
    # Degree distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    for col, (matrix, name, color) in enumerate(zip(
        [t_sv, t_dcv, t_e],
        ['T_sv', 'T_dcv', 'T_e'],
        ['red', 'blue', 'green']
    )):
        # In-degree (sum along rows - postsynaptic)
        in_deg = np.sum(matrix > 0, axis=1)
        axes[0, col].hist(in_deg, bins=30, color=color, alpha=0.7, edgecolor='black')
        axes[0, col].set_xlabel('In-degree', fontsize=9)
        axes[0, col].set_ylabel('Count', fontsize=9)
        axes[0, col].set_title(f'{name}: In-degree\nmean={in_deg.mean():.1f}', fontsize=10)
        axes[0, col].grid(True, alpha=0.3)
        
        # Out-degree (sum along columns - presynaptic)
        out_deg = np.sum(matrix > 0, axis=0)
        axes[1, col].hist(out_deg, bins=30, color=color, alpha=0.7, edgecolor='black')
        axes[1, col].set_xlabel('Out-degree', fontsize=9)
        axes[1, col].set_ylabel('Count', fontsize=9)
        axes[1, col].set_title(f'{name}: Out-degree\nmean={out_deg.mean():.1f}', fontsize=10)
        axes[1, col].grid(True, alpha=0.3)
    
    fig.suptitle('Connectivity Degree Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(save_dir / 'degree_distributions.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved plot: {save_dir / 'degree_distributions.png'}")
    
    print(f"\nAll plots saved to: {save_dir}/")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    
    parser = argparse.ArgumentParser(
        description="Build T_sv, T_dcv, and T_e connectivity matrices for C. elegans."
    )
    parser.add_argument(
        "--chem_adj",
        default=str(data_dir / "raw" / "moza (2025)" / "chem_adj.pkl"),
        help="Path to chemical adjacency pickle",
    )
    parser.add_argument(
        "--gap_adj",
        default=str(data_dir / "raw" / "moza (2025)" / "gapjn_symm_adj.pkl"),
        help="Path to gap junction adjacency pickle",
    )
    parser.add_argument(
        "--monoamine_csv",
        default=str(data_dir / "raw" / "bentley (2016)" / "esconnectome_monoamines_Bentley_2016.csv"),
        help="Path to Bentley 2016 monoamine connectivity CSV",
    )
    parser.add_argument(
        "--neuropeptide_csv",
        default=str(data_dir / "raw" / "bentley (2016)" / "esconnectome_neuropeptides_Bentley_2016.csv"),
        help="Path to Bentley 2016 neuropeptide connectivity CSV",
    )
    parser.add_argument("--out_sv", default="T_sv.npy", help="Output file for T_sv")
    parser.add_argument("--out_dcv", default="T_dcv.npy", help="Output file for T_dcv")
    parser.add_argument("--out_e", default="T_e.npy", help="Output file for T_e")
    parser.add_argument(
        "--plot_dir",
        default="connectome_plots",
        help="Directory to save connectivity plots (default: connectome_plots)",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Skip generating plots",
    )
    args = parser.parse_args()

    print("Loading connectome data...")
    chem_adj = load_chem_adj(args.chem_adj)
    print(f"  ✓ Chemical adjacency: {args.chem_adj}")
    
    gap_adj = load_gap_adj(args.gap_adj)
    print(f"  ✓ Gap junction adjacency: {args.gap_adj}")

    # Load Bentley 2016 monoamine and neuropeptide data
    monoamine_adj = None
    neuropeptide_adj = None
    if Path(args.monoamine_csv).exists() and Path(args.neuropeptide_csv).exists():
        monoamine_adj, neuropeptide_adj = load_bentley_data(args.monoamine_csv, args.neuropeptide_csv)
        print(f"  ✓ Monoamines (Bentley 2016): {args.monoamine_csv}")
        print(f"  ✓ Neuropeptides (Bentley 2016): {args.neuropeptide_csv}")
    else:
        print(f"  ⚠ Bentley 2016 files not found (skipping)")
    
    extra_neurons = []

    # Build neuron index (include neurons from all data sources)
    if monoamine_adj and neuropeptide_adj:
        extra_neurons = list(set(monoamine_adj.keys()) | set(neuropeptide_adj.keys()))
    neurons, neuron_index = build_neuron_index(chem_adj, gap_adj, extra_neurons)
    print(f"\nTotal neurons: {len(neurons)}")

    # Build matrices
    print("\n" + "="*60)
    print("Building connectivity matrices...")
    print("="*60)
    
    print("\n1. T_sv (fast chemical synapses)...")
    t_sv = construct_t_sv(chem_adj, neuron_index)
    print(f"   Shape: {t_sv.shape}, Non-zero: {np.count_nonzero(t_sv)}")
    
    print("\n2. T_dcv (monoamine + neuropeptide modulation)...")
    if monoamine_adj and neuropeptide_adj:
        t_dcv = construct_t_dcv_bentley(monoamine_adj, neuropeptide_adj, neuron_index)
        
        # Calculate separate counts for monoamines and neuropeptides
        t_mono = np.zeros_like(t_dcv)
        for pre, posts in monoamine_adj.items():
            pre_idx = neuron_index.get(pre)
            if pre_idx is None:
                continue
            for post, count in posts.items():
                post_idx = neuron_index.get(post)
                if post_idx is None:
                    continue
                t_mono[post_idx, pre_idx] += count
        
        print(f"   Monoamine connections: {np.count_nonzero(t_mono)}")
        print(f"   Neuropeptide connections: {np.count_nonzero(t_dcv - t_mono)}")
        print(f"   Combined shape: {t_dcv.shape}, Non-zero: {np.count_nonzero(t_dcv)}")
    else:
        print("   ⚠ No Bentley 2016 data, creating empty matrix")
        t_dcv = np.zeros((len(neurons), len(neurons)), dtype=float)
    
    print("\n3. T_e (gap junctions)...")
    t_e = construct_t_e(gap_adj, neuron_index)
    print(f"   Shape: {t_e.shape}, Non-zero: {np.count_nonzero(t_e)}")

    # Save
    print("\n" + "="*60)
    print("Saving matrices...")
    print("="*60)
    np.save(args.out_sv, t_sv)
    print(f"  ✓ {args.out_sv}")
    np.save(args.out_dcv, t_dcv)
    print(f"  ✓ {args.out_dcv}")
    np.save(args.out_e, t_e)
    print(f"  ✓ {args.out_e}")
    
    # Generate plots
    if not args.no_plot:
        plot_connectivity_summary(
            t_sv, t_dcv, t_e, neurons, Path(args.plot_dir)
        )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
