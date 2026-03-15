#!/usr/bin/env python3
"""
Plot neural traces heatmap + eigenworm amplitudes.

Generates plots with:
- Top panel: Heatmap of all neural traces in connectome order (horizontal: time on x-axis)
- Missing neurons shown in black
- Bottom 5 panels: Eigenworm amplitudes a₁ through a₅

Usage:
    python plot_traces_eigenworms.py \
        --h5_dir "data/used/behaviour+neuronal activity atanas (2023)" \
        --connectome_pkl "data/raw/moza (2025)/chem_adj.pkl" \
        --save_dir "neural_trace_plots"
"""

import argparse
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm


def load_eigenworms_from_shapes(shapes_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load shapes.csv and compute eigenworms via PCA.
    
    Returns
    -------
    eigvecs : (100, 100) array
        Eigenvectors (columns are eigenworm modes)
    eigvals : (100,) array
        Eigenvalues (sorted descending)
    """
    data = np.loadtxt(shapes_path, delimiter=",")
    N, D = data.shape
    
    # Center and compute covariance
    x_mean = np.mean(data, axis=0)
    x_cent = data - x_mean
    cov = (x_cent.T @ x_cent) / N
    
    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # Sort descending
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    
    return eigvecs, eigvals


def angle2pos(angles: np.ndarray) -> np.ndarray:
    """Convert consecutive-segment angles (unit length) to endpoints (x,y), centered."""
    angles = np.asarray(angles)
    dx = np.cos(angles)
    dy = np.sin(angles)
    xsum = np.cumsum(dx)
    ysum = np.cumsum(dy)
    pos = np.zeros((angles.size + 1, 2), dtype=float)
    pos[1:, 0] = xsum
    pos[1:, 1] = ysum
    pos -= np.mean(pos, axis=0, keepdims=True)
    return pos


def load_connectome_neuron_order(connectome_path: str) -> list[str]:
    """Load the 302 neuron names in connectome order from chem_adj.pkl."""
    try:
        with open(connectome_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict):
            # chem_adj.pkl is a dict with neuron names as keys
            # Sort alphabetically to get consistent order
            return sorted(data.keys())
        elif isinstance(data, tuple) and len(data) >= 2:
            # Assume (matrix, neuron_names) format
            return list(data[1])
        else:
            print(f"[warning] Unexpected format in {connectome_path}")
            return []
    except Exception as e:
        print(f"[error] Could not load connectome neurons: {e}")
        return []


def plot_neural_traces_and_eigenworms(
    h5_path: Path,
    connectome_neurons: list[str],
    eigenworm_shapes: np.ndarray | None,
    save_path: Path,
) -> None:
    """
    Plot neural traces heatmap (matching connectome neuron order) + eigenworm amplitudes + shapes.
    
    Parameters
    ----------
    h5_path : Path
        Path to HDF5 file with gcamp traces and eigenworms
    connectome_neurons : list[str]
        List of neuron names in the order they appear in connectome (302 neurons)
    eigenworm_shapes : ndarray or None
        Eigenvectors (100, 100) for plotting eigenworm shapes
    save_path : Path
        Where to save the plot
    """
    with h5py.File(h5_path, 'r') as f:
        # Load neuron labels and traces
        if 'gcamp/neuron_labels' not in f or 'gcamp/trace_array_original' not in f:
            print(f"[warning] {h5_path.name} missing required datasets")
            return
        
        labels = f['gcamp/neuron_labels'][:]
        labels = [l.decode() if isinstance(l, bytes) else str(l) for l in labels]
        traces = f['gcamp/trace_array_original'][:]  # (T, N)
        
        # Load eigenworms if available
        eigenworms = None
        if 'behavior/eigenworms' in f:
            eigenworms = f['behavior/eigenworms'][:]  # (T, 5)
        elif all(f'behavior/a_{i}' in f for i in range(1, 6)):
            eigenworms = np.stack([f[f'behavior/a_{i}'][:] for i in range(1, 6)], axis=1)
    
    T = traces.shape[0]
    n_connectome = len(connectome_neurons)
    
    # Create mapping from label to trace index
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    # Build trace array in connectome order (missing neurons = NaN)
    traces_ordered = np.full((T, n_connectome), np.nan, dtype=float)
    matched_count = 0
    for i, neuron in enumerate(connectome_neurons):
        if neuron in label_to_idx:
            traces_ordered[:, i] = traces[:, label_to_idx[neuron]]
            matched_count += 1
    
    # Transpose for horizontal display (neurons on y-axis, time on x-axis)
    traces_heatmap = traces_ordered.T  # (n_connectome, T)
    
    # Create figure with layout for traces, 5 first neurons, eigenworm amplitudes, and shapes
    if eigenworms is not None and eigenworms.shape[1] >= 5 and eigenworm_shapes is not None:
        # Layout: top panel for traces, middle panel for first 5 neurons, 5 rows with amplitude (left) + smaller shape (right)
        fig = plt.figure(figsize=(24, 14))
        gs = fig.add_gridspec(7, 2, height_ratios=[3, 1.5, 1, 1, 1, 1, 1], 
                             width_ratios=[4, 1], hspace=0.3, wspace=0.3)
        ax_traces = fig.add_subplot(gs[0, :])  # Span both columns
        ax_neurons = fig.add_subplot(gs[1, :])  # Span both columns
        ax_eigenworms = [fig.add_subplot(gs[i+2, 0]) for i in range(5)]
        ax_shapes = [fig.add_subplot(gs[i+2, 1]) for i in range(5)]
    elif eigenworms is not None and eigenworms.shape[1] >= 5:
        # Layout: top panel for traces, middle panel for first 5 neurons, bottom 5 panels for eigenworms (no shapes)
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(7, 1, height_ratios=[3, 1.5, 1, 1, 1, 1, 1], hspace=0.3)
        ax_traces = fig.add_subplot(gs[0])
        ax_neurons = fig.add_subplot(gs[1])
        ax_eigenworms = [fig.add_subplot(gs[i+2]) for i in range(5)]
        ax_shapes = []
    else:
        fig, ax_traces = plt.subplots(figsize=(20, 8))
        ax_neurons = None
        ax_eigenworms = []
        ax_shapes = []
    
    # Plot neural traces heatmap
    # Missing neurons (NaN) will appear black with proper masking
    masked_traces = np.ma.masked_invalid(traces_heatmap)
    
    im = ax_traces.imshow(
        masked_traces,
        aspect='auto',
        cmap='viridis',
        interpolation='none',
        extent=[0, T, n_connectome, 0]  # horizontal: time on x-axis
    )
    
    # Set black color for masked (missing) neurons
    im.cmap.set_bad(color='black')
    
    ax_traces.set_xlabel('Time (frames)', fontsize=12)
    ax_traces.set_ylabel('Neuron (connectome order)', fontsize=12)
    ax_traces.set_title(
        f'Neural Traces: {h5_path.name}\n({matched_count}/{n_connectome} neurons present, {T} frames)', 
        fontsize=14, fontweight='bold'
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_traces, fraction=0.02, pad=0.01)
    cbar.set_label('Fluorescence (ΔF/F)', fontsize=10)
    
    # Add tick marks for every 50th neuron
    tick_positions = np.arange(0, n_connectome, 50)
    ax_traces.set_yticks(tick_positions)
    ax_traces.set_yticklabels(
        [connectome_neurons[i] if i < len(connectome_neurons) else '' for i in tick_positions], 
        fontsize=8
    )
    
    # Plot first 5 neuron traces (individual traces from recording)
    if ax_neurons is not None:
        t = np.arange(T)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # Plot first 5 neurons from the actual traces
        n_to_plot = min(5, traces.shape[1])
        for i in range(n_to_plot):
            neuron_label = labels[i] if i < len(labels) else f'N{i}'
            ax_neurons.plot(t, traces[:, i], linewidth=0.8, alpha=0.7, 
                          color=colors[i], label=neuron_label)
        
        ax_neurons.set_xlabel('Time (frames)', fontsize=12)
        ax_neurons.set_ylabel('ΔF/F', fontsize=12)
        ax_neurons.set_title('First 5 Neuron Traces', fontsize=12, fontweight='bold')
        ax_neurons.legend(loc='upper right', fontsize=9, framealpha=0.8)
        ax_neurons.grid(True, alpha=0.3)
        ax_neurons.set_xlim(0, T)
    
    # Plot eigenworm amplitudes
    if eigenworms is not None and len(ax_eigenworms) > 0:
        t = np.arange(T)
        for i, ax in enumerate(ax_eigenworms):
            ax.plot(t, eigenworms[:, i], linewidth=0.8, alpha=0.85, color='C0')
            ax.set_ylabel(f'EW{i+1}', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.axhline(0, linewidth=0.6, linestyle='--', alpha=0.3, color='black')
            ax.set_xlim(0, T)
            
            # Add statistics
            valid_mask = np.isfinite(eigenworms[:, i])
            if valid_mask.any():
                mu = np.mean(eigenworms[valid_mask, i])
                sd = np.std(eigenworms[valid_mask, i])
                ax.text(
                    0.98, 0.95,
                    f'μ={mu:.2f}, σ={sd:.2f}',
                    transform=ax.transAxes,
                    ha='right', va='top',
                    fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
                )
            
            if i == 4:  # Last eigenworm plot
                ax.set_xlabel('Time (frames)', fontsize=12)
            else:
                ax.set_xticklabels([])
    
    # Plot eigenworm shapes (body modes) - smaller now
    if eigenworm_shapes is not None and len(ax_shapes) > 0:
        amp_scale = 5.0  # Amplitude scaling for visualization
        for i, ax in enumerate(ax_shapes):
            # Get eigenvector and convert to body shape
            eigvec = eigenworm_shapes[:, i]
            pos = angle2pos(eigvec * amp_scale)
            
            ax.plot(pos[:, 0], pos[:, 1], '.-', linewidth=1.5, markersize=2, color='C0')
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title(f'Mode {i+1}', fontsize=9, fontweight='bold')
            
            # Set consistent axis limits for all shapes
            max_range = np.max(np.abs(pos))
            ax.set_xlim(-max_range*1.1, max_range*1.1)
            ax.set_ylim(-max_range*1.1, max_range*1.1)
    
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot neural traces heatmap + eigenworm amplitudes"
    )
    
    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--h5_file",
        type=str,
        help="Single HDF5 file to plot"
    )
    group.add_argument(
        "--h5_dir",
        type=str,
        help="Directory containing HDF5 files to plot (batch mode)"
    )
    
    # Connectome reference
    parser.add_argument(
        "--connectome_pkl",
        type=str,
        default="data/raw/moza (2025)/chem_adj.pkl",
        help="Path to connectome pickle file for neuron ordering (default: data/raw/moza (2025)/chem_adj.pkl)"
    )
    
    # Eigenworm shapes
    parser.add_argument(
        "--shapes_csv",
        type=str,
        default="data/raw/white (1986)/shapes.csv",
        help="Path to shapes.csv for eigenworm shape visualization (default: data/raw/white (1986)/shapes.csv)"
    )
    
    # Output options
    parser.add_argument(
        "--save_dir",
        type=str,
        default="neural_trace_plots",
        help="Directory to save plots (default: neural_trace_plots)"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="Output path for single file mode (overrides --save_dir)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent
    
    connectome_path = Path(args.connectome_pkl)
    if not connectome_path.is_absolute():
        connectome_path = script_dir.parent / connectome_path
    
    shapes_path = Path(args.shapes_csv)
    if not shapes_path.is_absolute():
        shapes_path = script_dir.parent / shapes_path
    
    # Load connectome neuron order
    print(f"Loading connectome neurons from: {connectome_path}")
    connectome_neurons = load_connectome_neuron_order(str(connectome_path))
    
    if not connectome_neurons:
        print("[error] Failed to load connectome neurons. Exiting.")
        return 1
    
    print(f"Loaded {len(connectome_neurons)} neurons from connectome")
    
    # Load eigenworm shapes
    eigenworm_shapes = None
    if shapes_path.exists():
        print(f"Loading eigenworm shapes from: {shapes_path}")
        try:
            eigenworm_shapes, _ = load_eigenworms_from_shapes(str(shapes_path))
            print(f"Loaded eigenworm shapes: {eigenworm_shapes.shape}")
        except Exception as e:
            print(f"[warning] Failed to load eigenworm shapes: {e}")
    else:
        print(f"[warning] Shapes file not found: {shapes_path}")
        print("[warning] Eigenworm shape visualization will be disabled")
    
    # Determine files to process
    if args.h5_file:
        h5_files = [Path(args.h5_file)]
        if not h5_files[0].is_absolute():
            h5_files[0] = script_dir.parent / h5_files[0]
    else:
        h5_dir = Path(args.h5_dir)
        if not h5_dir.is_absolute():
            h5_dir = script_dir.parent / h5_dir
        h5_files = sorted(h5_dir.glob("*.h5"))
    
    if not h5_files:
        print("[error] No HDF5 files found")
        return 1
    
    print(f"\nProcessing {len(h5_files)} file(s)...")
    
    # Process files
    save_dir = Path(args.save_dir)
    if not save_dir.is_absolute():
        save_dir = script_dir.parent / save_dir
    
    for h5_path in tqdm(h5_files, desc="Generating plots"):
        try:
            if args.save_path and len(h5_files) == 1:
                save_path = Path(args.save_path)
            else:
                save_path = save_dir / f"{h5_path.stem}_traces_eigenworms.png"
            
            plot_neural_traces_and_eigenworms(
                h5_path=h5_path,
                connectome_neurons=connectome_neurons,
                eigenworm_shapes=eigenworm_shapes,
                save_path=save_path,
            )
            
        except Exception as e:
            print(f"\n[error] Failed to plot {h5_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✓ Saved plots to: {save_dir}/")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
