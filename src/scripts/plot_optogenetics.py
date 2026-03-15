#!/usr/bin/env python3
"""
Plot optogenetic pumpprobe data: neural traces with stimulation markers.

Creates a plot showing:
1. Heatmap of neural activity (all neurons)
2. Stimulation events overlay
3. Sample traces of stimulated neurons
"""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def plot_optogenetics_file(h5_path: Path, save_dir: Path):
    """Create visualization for one optogenetic file."""
    
    with h5py.File(h5_path, 'r') as f:
        # Load data
        traces = f['gcamp/trace_array_original'][:]
        labels = f['gcamp/neuron_labels'][:]
        labels = [lbl.decode('utf-8') if isinstance(lbl, bytes) else lbl for lbl in labels]
        
        stim_cells = f['optogenetics/stim_cell_indices'][:]
        stim_frames = f['optogenetics/stim_frame_indices'][:]
        stim_matrix = f['optogenetics/stim_matrix'][:]

        # Optional Stage 1 fitted stimulus weights
        b_all = None
        stim_lags = None
        if 'stage1/params' in f:
            g = f['stage1/params']
            if 'b' in g:
                try:
                    b_all = np.asarray(g['b'][:], dtype=float)
                except Exception:
                    b_all = None
            try:
                stim_lags = int(g.attrs.get('stim_lags', 1))
            except Exception:
                stim_lags = None
        
        T, N = traces.shape
    
    # Create figure with 2 subplots
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 3], hspace=0.3)
    
    # 1. Main heatmap with stimulation overlay
    ax1 = fig.add_subplot(gs[0])
    
    # Plot traces (no scaling - use original values)
    im = ax1.imshow(traces.T, aspect='auto', cmap='viridis', 
                    interpolation='nearest', origin='lower', vmin=0, vmax=400)
    
    # Overlay stimulation events
    for frame in stim_frames:
        ax1.axvline(frame, color='red', alpha=0.3, linewidth=0.5)
    
    # Mark stimulated neurons
    if len(stim_cells) > 0:
        for cell_idx in stim_cells:
            ax1.axhline(cell_idx, color='orange', alpha=0.2, linewidth=0.5, linestyle='--')
    
    ax1.set_xlabel('Time (frames)', fontsize=12)
    ax1.set_ylabel('Neurons', fontsize=12)
    ax1.set_title(f'{h5_path.stem} - Neural Activity with Optogenetic Stimulation', 
                  fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Neural Activity (ΔF/F)', fontsize=10)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.3, label='Stimulation frames'),
        Patch(facecolor='orange', alpha=0.2, label='Stimulated neurons')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # 2. Sample traces of stimulated neurons with stimulation markers
    ax2 = fig.add_subplot(gs[1])
    
    if len(stim_cells) > 0:
        # Filter out neurons with empty labels
        valid_stim_cells = [idx for idx in stim_cells if idx < len(labels) and labels[idx].strip()]
        
        if len(valid_stim_cells) == 0:
            ax2.text(0.5, 0.5, 'No stimulated neurons with valid labels', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=14)
            ax2.set_xticks([])
            ax2.set_yticks([])
        else:
            # Plot up to 10 stimulated neurons
            n_plot = min(10, len(valid_stim_cells))
            plot_indices = valid_stim_cells[:n_plot]
        
        time = np.arange(T)
        colors = plt.cm.tab10(np.linspace(0, 1, n_plot))
        
        for i, cell_idx in enumerate(plot_indices):
            trace = traces[:, cell_idx]
            # Scale from 0 (baseline subtraction)
            trace_min = np.nanmin(trace)
            trace_scaled = trace - trace_min
            # Normalize by std for visibility
            trace_norm = trace_scaled / (np.nanstd(trace_scaled) + 1e-6)
            
            # Add star marker if this neuron was stimulated
            label_text = f"★ {labels[cell_idx]}"
            # If Stage 1 fitted b is available, add a compact summary.
            if b_all is not None and b_all.ndim == 2 and cell_idx < b_all.shape[0]:
                try:
                    bj = np.asarray(b_all[cell_idx], dtype=float).reshape(-1)
                    b_norm = float(np.linalg.norm(bj))
                    label_text += f" | ||b||={b_norm:.2g}"
                    if stim_lags is not None and stim_lags > 1:
                        label_text += f" (lags={stim_lags})"
                except Exception:
                    pass
            
            ax2.plot(time, trace_norm + i * 3, color=colors[i], 
                    linewidth=1.0, label=label_text, alpha=0.9)

            # Add vertical red lines at frames where THIS neuron was stimulated
            try:
                stim_times = np.flatnonzero(np.abs(stim_matrix[:, cell_idx]) > 0)
                for fr in stim_times:
                    ax2.axvline(int(fr), color='red', alpha=0.08, linewidth=0.6)
            except Exception:
                pass
        
        # Mark stars only at frames where these specific neurons were stimulated
        # stim_cells and stim_frames are parallel arrays
        for stim_idx, frame in enumerate(stim_frames):
            if stim_idx < len(stim_cells):
                stim_neuron_idx = stim_cells[stim_idx]
                # Check if this stimulated neuron is in our plot
                if stim_neuron_idx in plot_indices:
                    # Find which plot line this corresponds to
                    plot_line_idx = list(plot_indices).index(stim_neuron_idx)
                    if frame < T:
                        trace = traces[:, stim_neuron_idx]
                        trace_min = np.nanmin(trace)
                        trace_scaled = trace - trace_min
                        trace_norm = trace_scaled / (np.nanstd(trace_scaled) + 1e-6)
                        y_pos = trace_norm[frame] + plot_line_idx * 3
                        ax2.plot(frame, y_pos, marker='*', markersize=12, 
                                color='red', markeredgecolor='darkred', 
                                markeredgewidth=0.5, zorder=10, alpha=1.0)
        
        # Keep stars and faint lines to indicate stim times clearly
        ax2.set_xlabel('Time (frames)', fontsize=12)
        ax2.set_ylabel('Activity (scaled from 0, offset)', fontsize=12)
        ax2.set_title(f'Sample Traces with Stimulation Events (n={len(valid_stim_cells)} stimulated neurons)', 
                     fontsize=12, fontweight='bold')
        ax2.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=9, 
                  ncol=1, framealpha=0.95)
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.set_xlim([0, T])
    else:
        ax2.text(0.5, 0.5, 'No stimulated neurons', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        ax2.set_xticks([])
        ax2.set_yticks([])
    
    # Add info text
    info_text = f'Neurons: {N} | Frames: {T} | Stim cells: {len(stim_cells)} | Stim events: {len(stim_frames)}'
    fig.text(0.5, 0.01, info_text, ha='center', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Save
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{h5_path.stem}_optogenetics.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


def main():
    parser = argparse.ArgumentParser(description='Plot optogenetic pumpprobe data')
    parser.add_argument('--h5_dir', type=str, required=True,
                       help='Directory containing processed H5 files')
    parser.add_argument('--save_dir', type=str, required=True,
                       help='Output directory for plots')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum number of files to plot (for testing)')
    
    args = parser.parse_args()
    
    h5_dir = Path(args.h5_dir)
    save_dir = Path(args.save_dir)
    
    # Find all H5 files
    h5_files = sorted(h5_dir.glob('*.h5'))
    
    if not h5_files:
        print(f"No H5 files found in {h5_dir}")
        return
    
    if args.max_files:
        h5_files = h5_files[:args.max_files]
    
    print(f"Plotting {len(h5_files)} files...")
    print("=" * 70)
    
    for i, h5_path in enumerate(h5_files, 1):
        try:
            save_path = plot_optogenetics_file(h5_path, save_dir)
            print(f"[{i}/{len(h5_files)}] {h5_path.name} → {save_path.name}")
        except Exception as e:
            print(f"[{i}/{len(h5_files)}] ERROR {h5_path.name}: {e}")
    
    print("=" * 70)
    print(f"✓ Plots saved to: {save_dir}")


if __name__ == '__main__':
    main()
