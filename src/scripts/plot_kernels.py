#!/usr/bin/env python3
"""Presentation-quality plots of synaptic kernels as sums of exponentials.

Generates large-font, clean figures suitable for slides showing:
  1. Individual exponential basis functions
  2. Composite SV and DCV kernels
  3. Side-by-side comparison

Uses the default config values or loads fitted parameters from a
Stage 2 results HDF5.

Usage
-----
# Plot default (initial) kernels:
    python -m scripts.plot_kernels --save_dir slides_figs

# Plot fitted kernels from a Stage 2 run:
    python -m scripts.plot_kernels --h5 path/to/file.h5 --save_dir slides_figs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Set Helvetica as the default font (Nimbus Sans is the open-source equivalent)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Nimbus Sans', 'Helvetica', 'Liberation Sans', 'Arial', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Nimbus Sans'
plt.rcParams['mathtext.it'] = 'Nimbus Sans:italic'
plt.rcParams['mathtext.bf'] = 'Nimbus Sans:bold'


# ── colour palette ──────────────────────────────────────────────────
# Colourblind-safe qualitative palette (Wong, 2011)
COLORS_SV = [
    "#0072B2",  # blue
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#CC79A7",  # reddish purple
    "#D55E00",  # vermillion
    "#E69F00",  # orange
    "#000000",  # black
]
COLORS_DCV = [
    "#D55E00",  # vermillion
    "#E69F00",  # orange
    "#CC79A7",  # reddish purple
    "#F0E442",  # yellow
    "#009E73",  # green
    "#56B4E9",  # sky
    "#0072B2",  # blue
    "#000000",  # black
]
COLOR_SV_SUM = "#0072B2"
COLOR_DCV_SUM = "#D55E00"


# ── plotting helpers ────────────────────────────────────────────────
def _style(ax, xlabel="", ylabel="", title=""):
    """Apply clean slide-friendly styling."""
    ax.set_xlabel(xlabel, fontsize=20, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=20, labelpad=8)
    if title:
        ax.set_title(title, fontsize=24, fontweight="bold", pad=14)
    ax.tick_params(labelsize=16, width=1.5, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _kernel_curves(taus, amps, dt, t_max_sec):
    """Compute individual and composite kernel curves.

    Returns
    -------
    t_sec : ndarray, shape (M,)
    components : list of ndarray, each shape (M,)
    composite : ndarray, shape (M,)
    """
    t_sec = np.arange(0, t_max_sec, dt)
    components = []
    composite = np.zeros_like(t_sec)
    for tau, a in zip(taus, amps):
        curve = a * np.exp(-t_sec / tau)
        components.append(curve)
        composite += curve
    return t_sec, components, composite


def plot_kernel_decomposition(
    taus, amps, dt, t_max_sec,
    colors, sum_color,
    title="Synaptic kernel",
    label_prefix="",
    ax=None,
):
    """Plot individual exponentials and composite kernel on one axis."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    t_sec, components, composite = _kernel_curves(taus, amps, dt, t_max_sec)

    # individual components (dashed, thinner)
    for r, (curve, tau, a) in enumerate(zip(components, taus, amps)):
        col = colors[r % len(colors)]
        ax.plot(
            t_sec, curve,
            linestyle="--", linewidth=2, color=col, alpha=0.7,
            label=rf"{label_prefix}$a_{{{r+1}}} e^{{-t/\tau_{{{r+1}}}}}$"
                  rf"  ($\tau$={tau:.1f}s, $a$={a:.3f})",
        )

    # composite (solid, thick)
    ax.plot(
        t_sec, composite,
        linewidth=3.5, color=sum_color,
        label=rf"{label_prefix}$K(t) = \sum_r a_r\, e^{{-t/\tau_r}}$",
    )

    _style(ax, xlabel="Time (s)", ylabel="Amplitude", title=title)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    ax.legend(fontsize=13, frameon=False, loc="upper right")
    return ax


def plot_both_kernels(
    taus_sv, amps_sv, taus_dcv, amps_dcv, dt,
    t_max_sv=15.0, t_max_dcv=120.0,
):
    """Two-panel figure comparing SV and DCV kernels."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(22, 7),
        gridspec_kw={"wspace": 0.30},
    )

    plot_kernel_decomposition(
        taus_sv, amps_sv, dt, t_max_sv,
        COLORS_SV, COLOR_SV_SUM,
        title="SV synapse kernel", label_prefix="",
        ax=ax1,
    )

    plot_kernel_decomposition(
        taus_dcv, amps_dcv, dt, t_max_dcv,
        COLORS_DCV, COLOR_DCV_SUM,
        title="DCV synapse kernel", label_prefix="",
        ax=ax2,
    )

    return fig


def plot_single_composite(
    taus_sv, amps_sv, taus_dcv, amps_dcv, dt,
    t_max=60.0,
):
    """Overlay both composite kernels on a single axis."""
    fig, ax = plt.subplots(figsize=(12, 7))

    t_sv, _, comp_sv = _kernel_curves(taus_sv, amps_sv, dt, min(t_max, 15.0))
    t_dcv, _, comp_dcv = _kernel_curves(taus_dcv, amps_dcv, dt, t_max)

    ax.plot(t_sv, comp_sv, linewidth=3.5, color=COLOR_SV_SUM,
            label="SV kernel")
    ax.plot(t_dcv, comp_dcv, linewidth=3.5, color=COLOR_DCV_SUM,
            label="DCV kernel")

    _style(ax, xlabel="Time (s)", ylabel="Amplitude",
           title="Synaptic kernels: SV vs DCV")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    ax.legend(fontsize=18, frameon=False, loc="upper right")
    return fig


def plot_basis_gallery(taus, dt, t_max, colors, title="Exponential basis set"):
    """Show unit-amplitude basis functions."""
    fig, ax = plt.subplots(figsize=(10, 6))
    t_sec = np.arange(0, t_max, dt)
    for r, tau in enumerate(taus):
        col = colors[r % len(colors)]
        ax.plot(
            t_sec, np.exp(-t_sec / tau),
            linewidth=2.5, color=col,
            label=rf"$e^{{-t/\tau_{{{r+1}}}}}$  ($\tau_{{{r+1}}}$={tau:.1f}s)",
        )
    _style(ax, xlabel="Time (s)", ylabel="Amplitude", title=title)
    ax.legend(fontsize=14, frameon=False, loc="upper right")
    return fig


def plot_edge_kernels(
    edges, edge_type, taus, amps, T_mask, neuron_labels, dt, t_max,
    colors, sum_color,
):
    """Plot edge-specific kernels for a few example edges.
    
    Parameters
    ----------
    edges : list of (i, j) tuples
        Postsynaptic (i) and presynaptic (j) neuron indices.
    edge_type : str
        "SV" or "DCV"
    taus, amps : ndarrays
        Global time constants and amplitudes.
    T_mask : ndarray, shape (N_full, N_full)
        Connectome mask (full 302x302).
    neuron_labels : list
        All 302 neuron names.
    """
    n_edges = len(edges)
    fig, axes = plt.subplots(
        1, n_edges, figsize=(7 * n_edges, 6),
        squeeze=False,
    )
    axes = axes[0]
    
    t_sec = np.arange(0, t_max, dt)
    
    for idx, (i, j) in enumerate(edges):
        ax = axes[idx]
        w_ij = T_mask[i, j]
        # Edge-specific kernel: K_ij(t) = w_ij * sum_r a_r * exp(-t/tau_r)
        composite = np.zeros_like(t_sec)
        for r, (tau, a) in enumerate(zip(taus, amps)):
            curve = w_ij * a * np.exp(-t_sec / tau)
            composite += curve
            col = colors[r % len(colors)]
            ax.plot(
                t_sec, curve,
                linestyle="--", linewidth=1.5, color=col, alpha=0.6,
            )
        
        # Plot composite
        ax.plot(
            t_sec, composite,
            linewidth=3.5, color=sum_color,
            label=rf"$K_{{{neuron_labels[i]} \leftarrow {neuron_labels[j]}}}(t)$",
        )
        
        title = f"{edge_type}: {neuron_labels[i]} ← {neuron_labels[j]}"
        _style(ax, xlabel="Time (s)", ylabel="Amplitude", title=title)
        ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
        ax.legend(fontsize=14, frameon=False, loc="upper right")
    
    return fig


# ── I/O ─────────────────────────────────────────────────────────────
def load_from_h5(h5_path):
    """Load fitted Stage 2 kernel parameters from HDF5."""
    import h5py
    with h5py.File(h5_path, "r") as f:
        g = f["stage2_pt/params"]
        dt = float(g.attrs.get("dt", 1.0)) if "dt" in g.attrs else 1.0
        # Try to read dt from file-level attrs too
        if dt == 1.0 and "dt" in f.attrs:
            dt = float(f.attrs["dt"])

        taus_sv = np.array(g["tau_sv"]) if "tau_sv" in g else np.array([])
        amps_sv = np.array(g["a_sv"]) if "a_sv" in g else np.array([])
        taus_dcv = np.array(g["tau_dcv"]) if "tau_dcv" in g else np.array([])
        amps_dcv = np.array(g["a_dcv"]) if "a_dcv" in g else np.array([])

    return dict(
        taus_sv=taus_sv, amps_sv=amps_sv,
        taus_dcv=taus_dcv, amps_dcv=amps_dcv,
        dt=dt,
    )


def defaults_from_config():
    """Return kernel parameters from Stage2PTConfig defaults."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from stage2_pt.config import Stage2PTConfig
    cfg = Stage2PTConfig(h5_path="")
    return dict(
        taus_sv=np.array(cfg.tau_sv_init),
        amps_sv=np.array(cfg.a_sv_init),
        taus_dcv=np.array(cfg.tau_dcv_init),
        amps_dcv=np.array(cfg.a_dcv_init),
        dt=cfg.dt if cfg.dt else 1.0,
    )


# ── main ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Plot synaptic kernels as sums of exponentials."
    )
    parser.add_argument(
        "--h5", type=str, default=None,
        help="HDF5 with fitted Stage 2 params (stage2_pt/params). "
             "If omitted, default config values are used.",
    )
    parser.add_argument("--save_dir", type=str, default="kernel_plots")
    parser.add_argument("--dt", type=float, default=None,
                        help="Override sampling interval (seconds).")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--fmt", type=str, default="png",
                        choices=["png", "pdf", "svg"])
    args = parser.parse_args()

    # Load parameters
    if args.h5:
        params = load_from_h5(args.h5)
        source = Path(args.h5).stem
    else:
        params = defaults_from_config()
        source = "defaults"

    if args.dt is not None:
        params["dt"] = args.dt

    taus_sv = params["taus_sv"]
    amps_sv = params["amps_sv"]
    taus_dcv = params["taus_dcv"]
    amps_dcv = params["amps_dcv"]
    dt = params["dt"]

    print(f"Source: {source}")
    print(f"  dt = {dt:.4f}s")
    print(f"  SV  τ = {taus_sv},  a = {amps_sv}")
    print(f"  DCV τ = {taus_dcv},  a = {amps_dcv}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Figure 1: SV basis set ──────────────────────────────────────
    if len(taus_sv):
        fig = plot_basis_gallery(
            taus_sv, dt, t_max=max(taus_sv) * 3,
            colors=COLORS_SV, title="SV exponential basis functions",
        )
        p = save_dir / f"sv_basis.{args.fmt}"
        fig.savefig(p, dpi=args.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  → {p}")

    # ── Figure 2: DCV basis set ─────────────────────────────────────
    if len(taus_dcv):
        fig = plot_basis_gallery(
            taus_dcv, dt, t_max=max(taus_dcv) * 3,
            colors=COLORS_DCV, title="DCV exponential basis functions",
        )
        p = save_dir / f"dcv_basis.{args.fmt}"
        fig.savefig(p, dpi=args.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  → {p}")

    # ── Figure 3: SV decomposition ──────────────────────────────────
    if len(taus_sv):
        fig, ax = plt.subplots(figsize=(12, 7))
        plot_kernel_decomposition(
            taus_sv, amps_sv, dt, max(taus_sv) * 3,
            COLORS_SV, COLOR_SV_SUM,
            title="SV kernel decomposition", ax=ax,
        )
        p = save_dir / f"sv_kernel.{args.fmt}"
        fig.savefig(p, dpi=args.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  → {p}")

    # ── Figure 4: DCV decomposition ─────────────────────────────────
    if len(taus_dcv):
        fig, ax = plt.subplots(figsize=(12, 7))
        plot_kernel_decomposition(
            taus_dcv, amps_dcv, dt, max(taus_dcv) * 3,
            COLORS_DCV, COLOR_DCV_SUM,
            title="DCV kernel decomposition", ax=ax,
        )
        p = save_dir / f"dcv_kernel.{args.fmt}"
        fig.savefig(p, dpi=args.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  → {p}")

    # ── Figure 5: side-by-side ──────────────────────────────────────
    if len(taus_sv) and len(taus_dcv):
        fig = plot_both_kernels(
            taus_sv, amps_sv, taus_dcv, amps_dcv, dt,
            t_max_sv=max(taus_sv) * 3,
            t_max_dcv=max(taus_dcv) * 3,
        )
        p = save_dir / f"kernels_side_by_side.{args.fmt}"
        fig.savefig(p, dpi=args.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  → {p}")

    # ── Figure 6: overlay ───────────────────────────────────────────
    if len(taus_sv) and len(taus_dcv):
        fig = plot_single_composite(
            taus_sv, amps_sv, taus_dcv, amps_dcv, dt,
            t_max=max(taus_dcv) * 3,
        )
        p = save_dir / f"kernels_overlay.{args.fmt}"
        fig.savefig(p, dpi=args.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  → {p}")

    # ── Edge-specific kernels (if h5 provided) ──────────────────────
    if args.h5 is not None:
        import h5py
        try:
            # Load connectome masks and full neuron labels
            data_root = Path(__file__).resolve().parent.parent / "data/used/masks+motor neurons"
            T_sv = np.load(data_root / "T_sv.npy")
            T_dcv = np.load(data_root / "T_dcv.npy")
            
            # Load full 302-neuron labels
            neuron_names_path = data_root / "neuron_names.npy"
            if neuron_names_path.exists():
                labels_all = [str(s) for s in np.load(neuron_names_path)]
            else:
                labels_all = [f"N{i}" for i in range(302)]
            
            # Pick a few example edges
            sv_edges_all = np.argwhere(T_sv > 0)
            dcv_edges_all = np.argwhere(T_dcv > 0)
            
            # Top 3 SV edges by connectome weight
            if len(sv_edges_all) and len(taus_sv):
                sv_weights = T_sv[sv_edges_all[:, 0], sv_edges_all[:, 1]]
                top_sv_idx = np.argsort(sv_weights)[::-1][:3]
                example_sv = [tuple(sv_edges_all[i]) for i in top_sv_idx]
                
                fig = plot_edge_kernels(
                    example_sv, "SV", taus_sv, amps_sv, T_sv, labels_all, dt,
                    t_max=max(taus_sv) * 3, colors=COLORS_SV, sum_color=COLOR_SV_SUM,
                )
                p = save_dir / f"sv_edge_examples.{args.fmt}"
                fig.savefig(p, dpi=args.dpi, bbox_inches="tight", facecolor="white")
                plt.close(fig)
                print(f"  → {p}")
            
            # Top 3 DCV edges by connectome weight
            if len(dcv_edges_all) and len(taus_dcv):
                dcv_weights = T_dcv[dcv_edges_all[:, 0], dcv_edges_all[:, 1]]
                top_dcv_idx = np.argsort(dcv_weights)[::-1][:3]
                example_dcv = [tuple(dcv_edges_all[i]) for i in top_dcv_idx]
                
                fig = plot_edge_kernels(
                    example_dcv, "DCV", taus_dcv, amps_dcv, T_dcv, labels_all, dt,
                    t_max=max(taus_dcv) * 3, colors=COLORS_DCV, sum_color=COLOR_DCV_SUM,
                )
                p = save_dir / f"dcv_edge_examples.{args.fmt}"
                fig.savefig(p, dpi=args.dpi, bbox_inches="tight", facecolor="white")
                plt.close(fig)
                print(f"  → {p}")
                
        except Exception as e:
            print(f"\n[warn] Could not generate edge-specific plots: {e}")

    print(f"\n✓ All figures saved to {save_dir}/")


if __name__ == "__main__":
    main()
