#!/usr/bin/env python3
"""
Create a heatmap showing which neurons were recorded in each worm.
Each row is a worm, each column is a neuron from the connectome (302 neurons).
"""

import argparse
from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def load_neuron_names(pkl_path: Path) -> list:
    """Load 302 neuron names from connectome pickle."""
    import pickle
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        if isinstance(data, dict):
            return sorted(data.keys())
        else:
            raise ValueError(f"Expected dict, got {type(data)}")


def create_coverage_matrix(h5_dir: Path, connectome_neurons: list) -> tuple:
    """
    Create binary matrix of neuron coverage across all recordings.
    
    Returns
    -------
    matrix : ndarray (n_files, n_neurons)
        Binary matrix: 1 if neuron recorded, 0 otherwise
    file_names : list
        List of file names (worm IDs)
    """
    h5_files = sorted(h5_dir.glob("*.h5"))
    n_files = len(h5_files)
    n_neurons = len(connectome_neurons)
    
    neuron_to_idx = {name: i for i, name in enumerate(connectome_neurons)}
    matrix = np.zeros((n_files, n_neurons), dtype=int)
    file_names = []
    
    print(f"Processing {n_files} files...")
    for i, h5_path in enumerate(h5_files):
        file_names.append(h5_path.stem)
        
        with h5py.File(h5_path, 'r') as f:
            if 'gcamp/neuron_labels' not in f:
                continue
            
            labels = f['gcamp/neuron_labels'][:]
            labels = [l.decode() if isinstance(l, bytes) else str(l) for l in labels]
            
            # Mark which neurons are present
            for label in labels:
                if label in neuron_to_idx:
                    matrix[i, neuron_to_idx[label]] = 1
    
    return matrix, file_names


def plot_coverage_heatmap(
    matrix: np.ndarray,
    file_names: list,
    neuron_names: list,
    save_path: Path
):
    """Plot heatmap of neuron coverage."""
    n_files, n_neurons = matrix.shape
    
    # Find never-recorded neurons
    neuron_counts = np.sum(matrix, axis=0)
    never_recorded = [neuron_names[i] for i in range(n_neurons) if neuron_counts[i] == 0]
    
    # Create figure with extra space on the right for text
    fig = plt.figure(figsize=(28, 12))
    gs = fig.add_gridspec(1, 2, width_ratios=[5, 1], wspace=0.3)
    ax_heatmap = fig.add_subplot(gs[0])
    ax_text = fig.add_subplot(gs[1])
    
    # Create colormap: white for 0, red for 1
    cmap = ListedColormap(['white', 'darkred'])
    
    # Plot heatmap
    im = ax_heatmap.imshow(matrix, cmap=cmap, aspect='auto', interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_heatmap, ticks=[0.25, 0.75])
    cbar.ax.set_yticklabels(['Not Recorded', 'Recorded'])
    cbar.set_label('Recording Status', fontsize=10)
    
    # Set ticks
    ax_heatmap.set_xticks(np.arange(n_neurons))
    ax_heatmap.set_yticks(np.arange(n_files))
    ax_heatmap.set_xticklabels(neuron_names, rotation=90, fontsize=6)
    ax_heatmap.set_yticklabels(file_names, fontsize=8)
    
    # Formatting
    ax_heatmap.set_xlabel('Neuron (302 connectome neurons)', fontsize=12, fontweight='bold')
    ax_heatmap.set_ylabel('Recording (worm)', fontsize=12, fontweight='bold')
    ax_heatmap.set_title(
        f'Neuron Coverage Across {n_files} Recordings\n'
        f'({n_neurons} neurons × {n_files} worms)',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    
    # Add grid
    ax_heatmap.set_xticks(np.arange(n_neurons) - 0.5, minor=True)
    ax_heatmap.set_yticks(np.arange(n_files) - 0.5, minor=True)
    ax_heatmap.grid(which='minor', color='gray', linestyle='-', linewidth=0.1, alpha=0.3)
    
    # Right panel: List of never-recorded neurons
    ax_text.axis('off')
    ax_text.set_xlim(0, 1)
    ax_text.set_ylim(0, 1)
    
    # Title for the list
    ax_text.text(0.5, 0.98, f'Never Recorded\n({len(never_recorded)} neurons)', 
                ha='center', va='top', fontsize=12, fontweight='bold',
                transform=ax_text.transAxes)
    
    # Display neurons in columns
    n_cols = 3
    neurons_per_col = int(np.ceil(len(never_recorded) / n_cols))
    
    x_positions = np.linspace(0.05, 0.95, n_cols)
    y_start = 0.92
    y_spacing = 0.92 / (neurons_per_col + 1)
    
    for col in range(n_cols):
        start_idx = col * neurons_per_col
        end_idx = min(start_idx + neurons_per_col, len(never_recorded))
        
        for i, neuron_idx in enumerate(range(start_idx, end_idx)):
            if neuron_idx < len(never_recorded):
                y_pos = y_start - (i * y_spacing)
                ax_text.text(x_positions[col], y_pos, never_recorded[neuron_idx],
                           ha='center', va='top', fontsize=7,
                           family='monospace',
                           transform=ax_text.transAxes)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap to: {save_path}")


def analyze_coverage(matrix: np.ndarray, neuron_names: list, file_names: list):
    """Print coverage statistics."""
    n_files, n_neurons = matrix.shape
    
    # Neurons never recorded
    neuron_counts = np.sum(matrix, axis=0)
    never_recorded = [neuron_names[i] for i in range(n_neurons) if neuron_counts[i] == 0]
    
    # Per-file statistics
    neurons_per_file = np.sum(matrix, axis=1)
    
    # Per-neuron statistics
    files_per_neuron = np.sum(matrix, axis=0)
    
    print("\n" + "="*70)
    print("NEURON COVERAGE ANALYSIS")
    print("="*70)
    
    print(f"\nTotal recordings: {n_files}")
    print(f"Total neurons in connectome: {n_neurons}")
    
    print(f"\n--- Per-recording statistics ---")
    print(f"Min neurons/recording: {neurons_per_file.min()}")
    print(f"Max neurons/recording: {neurons_per_file.max()}")
    print(f"Mean neurons/recording: {neurons_per_file.mean():.1f}")
    print(f"Median neurons/recording: {np.median(neurons_per_file):.1f}")
    
    print(f"\n--- Per-neuron statistics ---")
    print(f"Neurons recorded in all {n_files} recordings: {np.sum(files_per_neuron == n_files)}")
    print(f"Neurons recorded in >50% of recordings: {np.sum(files_per_neuron > n_files/2)}")
    print(f"Neurons recorded in <10% of recordings: {np.sum(files_per_neuron < n_files*0.1)}")
    print(f"Neurons never recorded: {len(never_recorded)}")
    
    if never_recorded:
        print(f"\n--- Neurons NEVER recorded ({len(never_recorded)} neurons) ---")
        # Print in columns
        never_recorded_sorted = sorted(never_recorded)
        for i in range(0, len(never_recorded_sorted), 10):
            chunk = never_recorded_sorted[i:i+10]
            print(', '.join(chunk))
    
    # Most frequently recorded neurons
    most_frequent_idx = np.argsort(files_per_neuron)[::-1][:20]
    print(f"\n--- Top 20 most frequently recorded neurons ---")
    for idx in most_frequent_idx:
        pct = (files_per_neuron[idx] / n_files) * 100
        print(f"{neuron_names[idx]:10s}: {files_per_neuron[idx]:2d}/{n_files} recordings ({pct:5.1f}%)")
    
    # Least frequently recorded (but not zero)
    recorded_mask = files_per_neuron > 0
    if np.any(recorded_mask):
        recorded_counts = files_per_neuron[recorded_mask]
        recorded_names_idx = np.where(recorded_mask)[0]
        least_frequent_idx = recorded_names_idx[np.argsort(recorded_counts)[:20]]
        
        print(f"\n--- 20 least frequently recorded neurons (excluding never recorded) ---")
        for idx in least_frequent_idx:
            pct = (files_per_neuron[idx] / n_files) * 100
            print(f"{neuron_names[idx]:10s}: {files_per_neuron[idx]:2d}/{n_files} recordings ({pct:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Create neuron coverage heatmap across all recordings"
    )
    parser.add_argument(
        "--h5_dir",
        required=True,
        help="Directory with processed H5 files"
    )
    parser.add_argument(
        "--connectome_pkl",
        default="data/raw/moza (2025)/chem_adj.pkl",
        help="Path to connectome pickle with neuron names"
    )
    parser.add_argument(
        "--save_path",
        default="neuron_coverage_heatmap.png",
        help="Output path for heatmap"
    )
    
    args = parser.parse_args()
    
    # Convert paths
    script_dir = Path(__file__).parent.parent
    h5_dir = Path(args.h5_dir)
    if not h5_dir.is_absolute():
        h5_dir = script_dir / h5_dir
    
    connectome_pkl = Path(args.connectome_pkl)
    if not connectome_pkl.is_absolute():
        connectome_pkl = script_dir / connectome_pkl
    
    save_path = Path(args.save_path)
    if not save_path.is_absolute():
        save_path = script_dir / save_path
    
    # Load connectome neurons
    print(f"Loading connectome neurons from: {connectome_pkl}")
    neuron_names = load_neuron_names(connectome_pkl)
    print(f"Loaded {len(neuron_names)} neurons")
    
    # Create coverage matrix
    matrix, file_names = create_coverage_matrix(h5_dir, neuron_names)
    
    # Plot heatmap
    plot_coverage_heatmap(matrix, file_names, neuron_names, save_path)
    
    # Analyze and print statistics
    analyze_coverage(matrix, neuron_names, file_names)


if __name__ == "__main__":
    main()
