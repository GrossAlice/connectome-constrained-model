"""
Generate demonstration plot for Stage 1 LGSSM slide.
Shows the cascade: stimulus -> latent drive u -> calcium c -> fluorescence y
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import h5py
from pathlib import Path
import argparse

# Set up fonts for presentation (Helvetica-like)
plt.rcParams['font.sans-serif'] = ['Nimbus Sans', 'Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.titlesize'] = 22


def plot_stage1_cascade(
    h5_path,
    neuron_idx=None,
    t_start=0,
    t_end=None,
    save_path="stage1_demo.png",
    dpi=200,
    auto_select_peak=False,
    peak_window_sec=50,
    center_t=None,
):
    """
    Plot the Stage 1 cascade for demonstration.
    
    Parameters
    ----------
    h5_path : str
        Path to Stage 1 output HDF5 file
    neuron_idx : int, optional
        Index of neuron to plot. If None, picks neuron with highest stimulus response.
    center_t : int, optional
        Timepoint index to center the zoom window on (e.g. movement onset).
        Overrides auto_select_peak when provided.
    t_start : int
        Start timepoint (for zooming into interesting region)
    t_end : int, optional
        End timepoint. If None, uses all data.
    save_path : str
        Output figure path
    dpi : int
        Resolution
    auto_select_peak : bool
        If True, center window on the highest value of the latent drive u
    peak_window_sec : float
        Window size in seconds around the peak (default 50 s)
    """
    
    with h5py.File(h5_path, 'r') as f:
        # Load data
        y = f['gcamp/trace_array_original'][()]  # (T, N)
        u_mean = f['stage1/u_mean'][()]  # (T, N)
        c_mean = f['stage1/c_mean'][()]  # (T, N)
        u_var = f['stage1/u_var'][()]  # (T, N)
        
        # Try to load stimulus and b weights
        stim = None
        stim_times = None
        b = None
        if 'stage1/params/b' in f:
            b = f['stage1/params/b'][()]  # (N, d_ell)
        
        if 'optogenetics/stim_matrix' in f:
            stim = f['optogenetics/stim_matrix'][()]  # (T, d_ell) or (d_ell, T)
            if stim.shape[0] < stim.shape[1]:
                stim = stim.T
        elif 'optogenetics/stim_times' in f:
            stim_times = f['optogenetics/stim_times'][()]
        
        # Mapping from stim_matrix column -> original neuron index
        stim_cell_indices = None
        if 'stimulus/stimulated_cell_indices' in f:
            _sci = np.array(
                f['stimulus/stimulated_cell_indices'][()], dtype=int
            ).ravel()
            if len(_sci) > 0:
                stim_cell_indices = _sci
        if stim_cell_indices is None and 'optogenetics/stim_cell_indices' in f:
            _sci = np.array(
                f['optogenetics/stim_cell_indices'][()], dtype=int
            ).ravel()
            if len(_sci) > 0:
                stim_cell_indices = _sci
        
        # Load velocity / behaviour if available
        velocity = None
        if 'behavior/velocity' in f:
            velocity = f['behavior/velocity'][()]

        # Load parameters
        alpha = f['stage1/params/alpha'][()]
        beta = f['stage1/params/beta'][()]
        
        # Get sampling info
        sample_rate = float(f['stage1/params'].attrs.get('sample_rate_hz', 2.0))
        dt = 1.0 / sample_rate
        
        # Get neuron labels if available
        neuron_labels = None
        if 'gcamp/neuron_labels' in f:
            labels_ds = f['gcamp/neuron_labels']
            if hasattr(labels_ds, 'asstr'):
                neuron_labels = labels_ds.asstr()[()]
            else:
                neuron_labels = labels_ds[()]
                if neuron_labels.dtype.kind in ['S', 'O']:
                    neuron_labels = [s.decode('utf-8') if isinstance(s, bytes) else str(s) 
                                   for s in neuron_labels]
    
    T, N = y.shape
    T_u, N_u = u_mean.shape
    if t_end is None:
        t_end = T
    
    # Pick neuron with highest stimulus response if not specified
    if neuron_idx is None:
        # Only consider neurons that were successfully fit (have valid u_mean)
        valid_neurons = ~np.isnan(u_mean).all(axis=0)
        valid_indices = np.where(valid_neurons)[0]
        
        if len(valid_indices) == 0:
            raise ValueError("No neurons with valid Stage 1 fit found")
        
        if b is not None and stim is not None and len(valid_indices) > 0:
            # Find neuron with strongest stimulus drive: max |b_i @ stim(t)|
            # b is (N_fit, d_ell), stim is (T, d_ell)
            if stim.shape[0] != T:
                stim = stim.T
            stim_drive = np.abs(b @ stim.T)  # (N_fit, T)
            total_stim = np.sum(stim_drive, axis=1)  # sum over time -> (N_fit,)
            best_idx = int(np.argmax(total_stim))
            neuron_idx = best_idx  # Index in u_mean space (0 to N_fit-1)
            print(f"Selected most stimulated neuron: {neuron_idx}/{N_u} (total stim drive: {total_stim[neuron_idx]:.2f})")
        else:
            # Fallback: find neuron with highest variance in u_mean
            var_u = np.nanvar(u_mean[:, valid_indices], axis=0)
            best_idx = int(np.argmax(var_u))
            neuron_idx = valid_indices[best_idx]
            print(f"Selected neuron with highest variance: {neuron_idx}/{N_u}")
    
    print(f"Plotting neuron {neuron_idx}/{N_u}")
    if neuron_labels is not None and neuron_idx < len(neuron_labels):
        print(f"  Label: {neuron_labels[neuron_idx]}")
    
    # Note: y has shape (T, N_orig) but u, c have shape (T, N_fit)
    # We need to find which original neuron corresponds to fitted neuron neuron_idx
    # For now, assume neuron_idx is in the fitted space
    
    # Extract data for this neuron from fitted results
    u_i_full = u_mean[:, neuron_idx]
    
    # ── Center on explicit timepoint if provided ──
    if center_t is not None:
        half_win = int(round(peak_window_sec / (2 * dt)))
        window_pts = 2 * half_win
        pre_pts = int(0.2 * window_pts)  # event at 20% from left
        t_start = max(0, int(center_t) - pre_pts)
        t_end = min(len(u_i_full), t_start + window_pts)
        if t_end - t_start < window_pts:
            t_start = max(0, t_end - window_pts)
        print(f"Centered on t={center_t} ({center_t*dt:.1f}s), "
              f"window: {t_start}-{t_end} ({(t_end-t_start)*dt:.0f}s)")
        auto_select_peak = False  # skip auto-select

    # Auto-select time window centered on stimulation event (or peak dc/dt fallback)
    if auto_select_peak:
        half_win = int(round(peak_window_sec / (2 * dt)))
        window_pts = 2 * half_win
        stim_centered = False

        # --- Try to center on this neuron's stimulation event ---
        if stim is not None:
            stim_col = None
            # stim_matrix is (T, N_orig) — column i IS neuron i
            if neuron_idx < stim.shape[1]:
                stim_col = neuron_idx

            if stim_col is not None and stim_col < stim.shape[1]:
                stim_events = np.flatnonzero(stim[:, stim_col] > 0)
                if len(stim_events) > 0:
                    # Center on first stimulation event;
                    # place it at 20 % from the left so the response is visible
                    peak_idx = int(stim_events[0])
                    pre_pts = int(0.2 * window_pts)
                    t_start = max(0, peak_idx - pre_pts)
                    t_end = min(len(u_i_full), t_start + window_pts)
                    if t_end - t_start < window_pts:
                        t_start = max(0, t_end - window_pts)
                    stim_centered = True
                    print(f"Stim event at t={peak_idx} ({peak_idx*dt:.1f}s), "
                          f"window: {t_start}-{t_end} ({(t_end-t_start)*dt:.0f}s)")

        # --- Fallback: center on peak dc/dt ---
        if not stim_centered:
            c_i_full = c_mean[:, neuron_idx]
            dc = np.diff(c_i_full) / dt
            skip = int(round(60.0 / dt))
            dc[:skip] = -np.inf
            finite_mask = np.isfinite(dc)
            if np.any(finite_mask):
                peak_idx = int(np.nanargmax(dc))
            else:
                peak_idx = len(u_i_full) // 2
            t_start = max(0, peak_idx - half_win)
            t_end = min(len(u_i_full), peak_idx + half_win)
            if t_end - t_start < window_pts:
                if t_start == 0:
                    t_end = min(len(u_i_full), window_pts)
                elif t_end == len(u_i_full):
                    t_start = max(0, len(u_i_full) - window_pts)
            print(f"Peak dc/dt at t={peak_idx} ({peak_idx*dt:.1f}s, dc/dt={dc[peak_idx]:.2f}), "
                  f"window: {t_start}-{t_end} ({(t_end-t_start)*dt:.0f}s)")
    
    # Extract data for this neuron
    y_i = y[t_start:t_end, neuron_idx]
    u_i = u_mean[t_start:t_end, neuron_idx]
    c_i = c_mean[t_start:t_end, neuron_idx]
    u_std_i = np.sqrt(u_var[t_start:t_end, neuron_idx])
    
    # Make sure all arrays have the same length
    min_len = min(len(y_i), len(u_i), len(c_i))
    y_i = y_i[:min_len]
    u_i = u_i[:min_len]
    c_i = c_i[:min_len]
    u_std_i = u_std_i[:min_len]
    
    # Time axis in seconds
    t = np.arange(min_len) * dt + t_start * dt
    
    # Compute shared y-axis range for u and c panels only
    y_pred = alpha[neuron_idx] * c_i + beta[neuron_idx]
    uc_vals = np.concatenate([
        u_i - 2*u_std_i, u_i + 2*u_std_i,  # u with uncertainty band
        c_i,
    ])
    uc_vals = uc_vals[np.isfinite(uc_vals)]
    if uc_vals.size > 0:
        uc_lo = float(np.nanmin(uc_vals))
        uc_hi = float(np.nanmax(uc_vals))
    else:
        uc_lo, uc_hi = -1.0, 1.0
    uc_margin = 0.05 * (uc_hi - uc_lo) if uc_hi > uc_lo else 0.1
    uc_ylim = (uc_lo - uc_margin, uc_hi + uc_margin)

    # Compute independent y-axis range for fluorescence panel
    y_vals = np.concatenate([
        y_i[np.isfinite(y_i)],
        y_pred[np.isfinite(y_pred)],
    ])
    if y_vals.size > 0:
        y_lo = float(np.nanmin(y_vals))
        y_hi = float(np.nanmax(y_vals))
    else:
        y_lo, y_hi = -1.0, 1.0
    y_margin = 0.05 * (y_hi - y_lo) if y_hi > y_lo else 0.1
    y_ylim = (y_lo - y_margin, y_hi + y_margin)
    
    # Create figure
    fig = plt.figure(figsize=(5, 10))
    gs = GridSpec(4, 1, height_ratios=[1, 2, 2, 2], hspace=0.3)
    
    # Stimulus panel
    ax_stim = fig.add_subplot(gs[0])
    if stim is not None:
        # Show total stimulus and optionally this neuron's drive
        stim_total = np.sum(stim, axis=1) if stim.ndim > 1 else stim
        stim_plot = stim_total[t_start:t_start+min_len]
        ax_stim.fill_between(t, 0, stim_plot, color='red', alpha=0.3, label='Total stimulus $\\ell(t)$')
        ax_stim.plot(t, stim_plot, color='red', linewidth=2, alpha=0.7)
        
        # If we have b, show this neuron's specific drive
        if b is not None and neuron_idx < len(b):
            # b is (N, d_ell), stim is (T, N) per-neuron stimulus matrix
            # For neuron i: b_i is scalar or vector, stim could be (T, N) or (T, d_ell)
            b_i = b[neuron_idx].flatten()  # Ensure 1D: (d_ell,)
            
            if stim.shape[1] == len(b_i):
                # stim is (T, d_ell): regular case
                neuron_drive = np.abs(stim @ b_i)  # (T,)
            elif stim.shape[1] == b.shape[0]:
                # stim is (T, N): per-neuron stimulus, use this neuron's column
                neuron_drive = np.abs(b_i[0] * stim[:, neuron_idx])  # (T,)
            else:
                # Fallback: just show total stimulus scaled
                neuron_drive = stim_total * np.abs(b_i[0])
                
            neuron_drive_plot = neuron_drive[t_start:t_start+min_len]
            ax2 = ax_stim.twinx()
            ax2.plot(t, neuron_drive_plot, color='darkred', linewidth=2, 
                    linestyle='--', alpha=0.8, label=f'Drive to neuron')
            ax2.set_ylabel('Neuron drive\n$|b_i \\ell_i|$', fontsize=14, color='darkred')
            ax2.tick_params(axis='y', labelcolor='darkred')
            ax2.spines['top'].set_visible(False)
            # ax2.legend(loc='upper left', frameon=False, fontsize=12)
    elif stim_times is not None:
        # Plot as vertical lines
        for st in stim_times:
            if t_start * dt <= st <= t_end * dt:
                ax_stim.axvline(st, color='red', linewidth=2, alpha=0.7)
        ax_stim.set_ylim([0, 1])
        ax_stim.text(0.02, 0.5, 'Stimulus events', transform=ax_stim.transAxes,
                    fontsize=16, color='red', va='center')
    elif velocity is not None:
        # Show speed (|velocity|) instead of stimulus
        speed_full = np.abs(velocity)
        speed_plot = speed_full[t_start:t_start+min_len]
        ax_stim.fill_between(t, 0, speed_plot, color='purple', alpha=0.2)
        ax_stim.plot(t, speed_plot, color='purple', linewidth=1.5, alpha=0.8)
        ax_stim.set_ylabel('Speed\n$|v(t)|$', fontsize=18, fontweight='bold', color='purple')
        ax_stim.tick_params(axis='y', labelcolor='purple')
    else:
        ax_stim.text(0.5, 0.5, 'No stimulus data', transform=ax_stim.transAxes,
                    ha='center', va='center', fontsize=16, color='gray')
        ax_stim.set_ylim([0, 1])
    
    if velocity is not None and stim is None and stim_times is None:
        pass  # ylabel already set above
    else:
        ax_stim.set_ylabel('Stimulus\n$\\ell(t)$', fontsize=18, fontweight='bold')
    ax_stim.set_xlim([t[0], t[-1]])
    ax_stim.spines['top'].set_visible(False)
    ax_stim.spines['right'].set_visible(False)
    ax_stim.set_xticks([])
    # if stim is not None:
    #     ax_stim.legend(loc='upper right', frameon=False)
    
    # Latent drive panel
    ax_u = fig.add_subplot(gs[1])
    ax_u.plot(t, u_i, color='darkblue', linewidth=2.5, label='Latent drive $u_i(t)$')
    ax_u.fill_between(t, u_i - 2*u_std_i, u_i + 2*u_std_i, 
                      color='darkblue', alpha=0.2, label='±2 SD (uncertainty)')
    ax_u.set_ylabel('Latent drive\n$u_i(t)$', fontsize=18, fontweight='bold', color='darkblue')
    ax_u.set_xlim([t[0], t[-1]])
    ax_u.set_ylim(uc_ylim)
    ax_u.tick_params(axis='y', labelcolor='darkblue')
    ax_u.spines['top'].set_visible(False)
    ax_u.spines['right'].set_visible(False)
    ax_u.set_xticks([])
    # ax_u.legend(loc='upper right', frameon=False)
    ax_u.grid(True, alpha=0.3)
    
    # Add annotation about process noise
    # ax_u.text(0.02, 0.95, '$\\tau_u \\frac{du_i}{dt} = -u_i + b_i^\\top \\ell + \\eta_i$',
    #          transform=ax_u.transAxes, fontsize=14, va='top',
    #          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Calcium panel
    ax_c = fig.add_subplot(gs[2])
    ax_c.plot(t, c_i, color='darkorange', linewidth=2.5, label='Calcium $c_i(t)$')
    ax_c.set_ylabel('Calcium\n$c_i(t)$', fontsize=18, fontweight='bold', color='darkorange')
    ax_c.set_xlim([t[0], t[-1]])
    ax_c.set_ylim(uc_ylim)
    ax_c.tick_params(axis='y', labelcolor='darkorange')
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)
    ax_c.set_xticks([])
    # ax_c.legend(loc='upper right', frameon=False)
    ax_c.grid(True, alpha=0.3)
    
    # Add annotation about calcium dynamics
    # ax_c.text(0.02, 0.95, '$\\tau_c \\frac{dc_i}{dt} = -c_i + u_i + \\nu_i$',
    #          transform=ax_c.transAxes, fontsize=14, va='top',
            #  bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # Fluorescence panel
    ax_y = fig.add_subplot(gs[3])
    
    # Raw data (prominent) - green for GCaMP fluorescence
    ax_y.plot(t, y_i, color='green', linewidth=2.5, alpha=0.8, label='Raw data $y_i(t)$', zorder=3)
    
    # Overlay predicted fluorescence (lighter)
    ax_y.plot(t, y_pred, color='black', linewidth=2.0, linestyle='--', alpha=0.7,
             label='Model $\\alpha_i c_i + \\beta_i$', zorder=2)
    
    ax_y.set_ylabel('Fluorescence\n$y_i(t)$', fontsize=18, fontweight='bold')
    ax_y.set_xlabel('Time (seconds)', fontsize=18, fontweight='bold')
    ax_y.set_xlim([t[0], t[-1]])
    ax_y.set_ylim(y_ylim)
    ax_y.spines['top'].set_visible(False)
    ax_y.spines['right'].set_visible(False)
    # ax_y.legend(loc='upper right', frameon=False)
    ax_y.grid(True, alpha=0.3)
    
    # # Add annotation about observation model
    # ax_y.text(0.02, 0.95, '$y_i(t) = \\alpha_i c_i(t) + \\beta_i + \\epsilon_i$ (fixed $\\alpha_i=1$)',
    #          transform=ax_y.transAxes, fontsize=14, va='top',
    #          bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    # Overall title
    title = f'Stage 1 LGSSM: Deconvolving Fluorescence to Neural Drive'
    if neuron_labels is not None and neuron_idx < len(neuron_labels):
        title += f' (Neuron: {neuron_labels[neuron_idx]})'
    fig.suptitle(title, fontsize=22, fontweight='bold', y=0.995)
    
    # Add cascade arrow annotations
    # Add arrows between panels to show cascade
    fig.text(0.01, 0.76, '↓', fontsize=40, ha='center', va='center', color='red', fontweight='bold')
    fig.text(0.01, 0.53, '↓', fontsize=40, ha='center', va='center', color='darkblue', fontweight='bold')
    fig.text(0.01, 0.30, '↓', fontsize=40, ha='center', va='center', color='darkorange', fontweight='bold')
    
    plt.tight_layout(rect=[0.02, 0, 1, 0.99])
    
    # Save zoomed plot
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Saved to {save_path}")
    plt.close()

    # ── Full-trace figure ──
    t_full = np.arange(T_u) * dt
    y_full = y[:T_u, neuron_idx]
    u_full = u_mean[:, neuron_idx]
    c_full = c_mean[:, neuron_idx]
    u_std_full = np.sqrt(np.clip(u_var[:, neuron_idx], 0, np.inf))
    y_pred_full = alpha[neuron_idx] * c_full + beta[neuron_idx]

    # Full-trace y-limits: share between u/c, keep y independent
    uc_full_vals = np.concatenate([
        u_full - 2*u_std_full,
        u_full + 2*u_std_full,
        c_full,
    ])
    uc_full_vals = uc_full_vals[np.isfinite(uc_full_vals)]
    if uc_full_vals.size > 0:
        ucf_lo = float(np.nanmin(uc_full_vals))
        ucf_hi = float(np.nanmax(uc_full_vals))
    else:
        ucf_lo, ucf_hi = -1.0, 1.0
    ucf_margin = 0.05 * (ucf_hi - ucf_lo) if ucf_hi > ucf_lo else 0.1
    uc_full_ylim = (ucf_lo - ucf_margin, ucf_hi + ucf_margin)

    y_full_vals = np.concatenate([
        y_full[np.isfinite(y_full)],
        y_pred_full[np.isfinite(y_pred_full)],
    ])
    if y_full_vals.size > 0:
        yf_lo = float(np.nanmin(y_full_vals))
        yf_hi = float(np.nanmax(y_full_vals))
    else:
        yf_lo, yf_hi = -1.0, 1.0
    yf_margin = 0.05 * (yf_hi - yf_lo) if yf_hi > yf_lo else 0.1
    y_full_ylim = (yf_lo - yf_margin, yf_hi + yf_margin)

    fig2 = plt.figure(figsize=(9, 8))
    gs2 = GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.25)

    # Latent drive u — full
    ax_uf = fig2.add_subplot(gs2[0])
    ax_uf.plot(t_full, u_full, color='darkblue', linewidth=0.8)
    ax_uf.fill_between(t_full, u_full - 2*u_std_full, u_full + 2*u_std_full,
                        color='darkblue', alpha=0.15)
    ax_uf.axvspan(t_start * dt, t_end * dt, color='gold', alpha=0.25, label='Zoom window')
    ax_uf.set_ylabel('$u_i(t)$', fontsize=16, fontweight='bold', color='darkblue')
    ax_uf.tick_params(axis='y', labelcolor='darkblue')
    ax_uf.set_xlim([0, t_full[-1]])
    ax_uf.set_ylim(uc_full_ylim)
    ax_uf.set_xticks([])
    ax_uf.spines['top'].set_visible(False)
    ax_uf.spines['right'].set_visible(False)
    ax_uf.legend(loc='upper right', fontsize=10, frameon=False)
    ax_uf.grid(True, alpha=0.3)

    # Calcium c — full
    ax_cf = fig2.add_subplot(gs2[1])
    ax_cf.plot(t_full, c_full, color='darkorange', linewidth=0.8)
    ax_cf.axvspan(t_start * dt, t_end * dt, color='gold', alpha=0.25)
    ax_cf.set_ylabel('$c_i(t)$', fontsize=16, fontweight='bold', color='darkorange')
    ax_cf.tick_params(axis='y', labelcolor='darkorange')
    ax_cf.set_xlim([0, t_full[-1]])
    ax_cf.set_ylim(uc_full_ylim)
    ax_cf.set_xticks([])
    ax_cf.spines['top'].set_visible(False)
    ax_cf.spines['right'].set_visible(False)
    ax_cf.grid(True, alpha=0.3)

    # Fluorescence y — full
    ax_yf = fig2.add_subplot(gs2[2])
    ax_yf.plot(t_full, y_full, color='green', linewidth=0.8, alpha=0.7, label='Raw $y_i$')
    ax_yf.plot(t_full, y_pred_full, color='black', linewidth=0.8, linestyle='--',
               alpha=0.6, label='Model $\\hat{y}_i$')
    ax_yf.axvspan(t_start * dt, t_end * dt, color='gold', alpha=0.25)
    ax_yf.set_ylabel('$y_i(t)$', fontsize=16, fontweight='bold')
    ax_yf.set_xlabel('Time (seconds)', fontsize=16, fontweight='bold')
    ax_yf.set_xlim([0, t_full[-1]])
    ax_yf.set_ylim(y_full_ylim)
    ax_yf.spines['top'].set_visible(False)
    ax_yf.spines['right'].set_visible(False)
    ax_yf.legend(loc='upper right', fontsize=10, frameon=False)
    ax_yf.grid(True, alpha=0.3)

    label_str = ''
    if neuron_labels is not None and neuron_idx < len(neuron_labels):
        label_str = f' ({neuron_labels[neuron_idx]})'
    fig2.suptitle(f'Full trace — Neuron {neuron_idx}{label_str}', fontsize=18, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    full_path = save_path.parent / (save_path.stem + '_full' + save_path.suffix)
    fig2.savefig(full_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Saved to {full_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate Stage 1 LGSSM demonstration plot"
    )
    parser.add_argument("--h5", type=str, required=True,
                       help="Path to Stage 1 output HDF5 file")
    parser.add_argument("--neuron", type=int, default=None,
                       help="Neuron index to plot (default: auto-select)")
    parser.add_argument("--t_start", type=int, default=0,
                       help="Start timepoint")
    parser.add_argument("--t_end", type=int, default=None,
                       help="End timepoint (default: all)")
    parser.add_argument("--save", type=str, default="slides_figs/stage1_demo.png",
                       help="Output path (single neuron mode)")
    parser.add_argument("--save_dir", type=str, default=None,
                       help="Output directory for --all_neurons mode")
    parser.add_argument("--dpi", type=int, default=200,
                       help="DPI for output")
    parser.add_argument("--auto_peak", action="store_true",
                       help="Center window on highest value of latent drive u")
    parser.add_argument("--peak_window", type=float, default=50,
                       help="Window size in seconds around peak (default: 50)")
    parser.add_argument("--all_neurons", action="store_true",
                       help="Generate one plot per neuron")
    
    args = parser.parse_args()
    
    if args.all_neurons:
        # Determine number of neurons from the h5 file
        with h5py.File(args.h5, 'r') as f:
            N = f['stage1/u_mean'].shape[1]
            neuron_labels = None
            if 'gcamp/neuron_labels' in f:
                labs = f['gcamp/neuron_labels'][()]
                neuron_labels = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in labs]
        
        save_dir = Path(args.save_dir) if args.save_dir else Path(args.save).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(args.h5).stem
        
        for i in range(N):
            label = neuron_labels[i] if neuron_labels and i < len(neuron_labels) else f"n{i}"
            out_path = save_dir / f"{stem}_demo_neuron{i:03d}_{label}.png"
            try:
                plot_stage1_cascade(
                    h5_path=args.h5,
                    neuron_idx=i,
                    auto_select_peak=args.auto_peak,
                    peak_window_sec=args.peak_window,
                    t_start=args.t_start,
                    t_end=args.t_end,
                    save_path=str(out_path),
                    dpi=args.dpi,
                )
            except Exception as e:
                print(f"[SKIP] Neuron {i}: {e}")
        print(f"\nDone — {N} neuron plots saved to {save_dir}/")
    else:
        plot_stage1_cascade(
            h5_path=args.h5,
            neuron_idx=args.neuron,
            auto_select_peak=args.auto_peak,
            peak_window_sec=args.peak_window,
            t_start=args.t_start,
            t_end=args.t_end,
            save_path=args.save,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
