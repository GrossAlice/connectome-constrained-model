#!/usr/bin/env python3
"""
Optogenetics analysis: visualise what Stage 2 learned about stimulation.

Plots generated:
  1. Learned causal stimulus kernel h(τ)
  2. Stimulus weights b (bar chart, highlighting stimulated neurons)
  3. Stim-triggered averages for each stimulated neuron (data vs model)
  4. Response heatmap (Δu post – pre) across all neurons
  5. Full-length traces for top stimulated neurons
  6. Data vs model response heatmaps side-by-side
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stage2.config  import make_config
from stage2.io_h5   import load_data_pt
from stage2.model   import Stage2ModelPT
from stage2.init_from_data import init_lambda_u, init_all_from_data
from stage2.train   import snapshot_model_state


# ──────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────

def _load_trained_model(h5_path, results_dir, device="cpu", **cfg_kw):
    """Build the Stage2 model and load trained weights from the results h5."""
    cfg = make_config(h5_path, device=device, **cfg_kw)
    data = load_data_pt(cfg)
    u = data["u_stage1"]
    T, N = u.shape
    d_ell = data.get("d_ell", 0)
    sign_t = data.get("sign_t")
    lambda_u_init = init_lambda_u(u, cfg)
    dev = torch.device(device)

    model = Stage2ModelPT(
        N, data["T_e"], data["T_sv"], data["T_dcv"], data["dt"],
        cfg, dev, d_ell=d_ell, lambda_u_init=lambda_u_init, sign_t=sign_t,
    ).to(dev)
    init_all_from_data(model, u, cfg)

    # Load weights from results h5
    results_h5 = Path(results_dir) / "stage2_results.h5"
    if not results_h5.exists():
        print(f"[warn] {results_h5} not found – using init weights")
        return model, data, cfg

    with h5py.File(results_h5, "r") as f:
        grp = f.get("stage2_pt/params")
        if grp is None:
            print(f"[warn] no stage2_pt/params in {results_h5}")
            return model, data, cfg
        sd = model.state_dict()
        loaded = 0
        # The h5 stores BOTH raw params (e.g. _lambda_u_raw) and their
        # constrained forms (e.g. lambda_u).  Only load keys that match
        # a state-dict entry directly to avoid overwriting raw with
        # constrained values.
        for key in grp.keys():
            if key not in sd:
                continue
            arr = np.array(grp[key])
            t = torch.tensor(arr, dtype=torch.float32, device=dev)
            if t.shape == sd[key].shape:
                sd[key] = t
                loaded += 1
        for key in grp.attrs:
            if key not in sd:
                continue
            val = float(grp.attrs[key])
            if sd[key].numel() == 1:
                sd[key] = torch.tensor(val, dtype=torch.float32, device=dev)
                loaded += 1
        model.load_state_dict(sd)
        print(f"[info] loaded {loaded} params from {results_h5}")

    return model, data, cfg


def _forward_pass(model, u_stage1, stim_conv, gating, device):
    """Run model forward (one-step predictions) and return (T, N) array."""
    dev = torch.device(device)
    u_t = torch.tensor(u_stage1, dtype=torch.float32, device=dev)
    stim_dev = torch.tensor(stim_conv, dtype=torch.float32, device=dev)
    T, N = u_t.shape
    preds = [u_t[0].cpu().numpy()]
    s_sv  = torch.zeros(N, model.r_sv, device=dev)
    s_dcv = torch.zeros(N, model.r_dcv, device=dev)
    with torch.no_grad():
        for t in range(1, T):
            g = gating[t - 1] if gating is not None else torch.ones(N, device=dev)
            s = stim_dev[t - 1]
            u_next, s_sv, s_dcv = model.prior_step(u_t[t - 1], s_sv, s_dcv, g, s)
            preds.append(u_next.cpu().numpy())
    return np.stack(preds)


# ──────────────────────────────────────────────────────────────────────
# plotting functions
# ──────────────────────────────────────────────────────────────────────

def _plot_kernel(kernel, dt, save_dir):
    """Fig 1 – learned causal kernel h(τ)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    k_time = np.arange(len(kernel)) * dt

    axes[0].bar(k_time, kernel, width=dt * 0.8,
                color="steelblue", edgecolor="k", lw=0.5)
    axes[0].set_xlabel("Lag (s)"); axes[0].set_ylabel("Kernel weight")
    axes[0].set_title("Learned stimulus kernel h(τ)")
    axes[0].axhline(0, color="k", lw=0.5)

    axes[1].plot(k_time, np.cumsum(kernel), "o-", color="steelblue", ms=3)
    axes[1].set_xlabel("Lag (s)"); axes[1].set_ylabel("Cumulative weight")
    axes[1].set_title("Cumulative kernel")
    axes[1].axhline(kernel.sum(), color="gray", ls="--", lw=0.8,
                     label=f"sum = {kernel.sum():.2f}")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_dir / "opto_01_kernel.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  → opto_01_kernel.png")


def _plot_b_weights(b, labels, stim_neurons, save_dir):
    """Fig 2 – stimulus weights b sorted by magnitude."""
    N = len(b)
    order = np.argsort(np.abs(b))[::-1]
    colours = ["#d62728" if i in stim_neurons else "#aec7e8" for i in order]

    fig, ax = plt.subplots(figsize=(max(12, N * 0.3), 4))
    ax.bar(range(N), b[order], color=colours, edgecolor="k", lw=0.3)
    ax.set_xticks(range(N))
    ax.set_xticklabels([labels[i] for i in order], rotation=90, fontsize=6)
    ax.set_ylabel("Stimulus weight b")
    ax.set_title("Learned stimulus weights  (red = directly stimulated)")
    ax.axhline(0, color="k", lw=0.5)
    ax.legend(handles=[Patch(color="#d62728", label="stimulated"),
                       Patch(color="#aec7e8", label="not stimulated")],
              fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(save_dir / "opto_02_b_weights.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  → opto_02_b_weights.png")


def _plot_stim_triggered_avg(u_data, u_model, stim_matrix, labels,
                              stim_neurons, dt, save_dir):
    """Fig 3 – stim-triggered average for each stimulated neuron."""
    T, N = u_data.shape
    pre, post = 10, 40          # frames: 5 s before, 20 s after
    window = pre + post
    win_time = (np.arange(window) - pre) * dt

    n_stim = len(stim_neurons)
    n_cols = min(5, n_stim)
    n_rows = max(1, (n_stim + n_cols - 1) // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(3.5 * n_cols, 2.8 * n_rows),
                              sharex=True, squeeze=False)

    for idx, ni in enumerate(stim_neurons):
        ax = axes[idx // n_cols, idx % n_cols]
        events = np.where(stim_matrix[:, ni] > 0)[0]
        events = events[(events >= pre) & (events + post <= T)]
        if len(events) == 0:
            ax.set_title(f"{labels[ni]} (no events)", fontsize=8)
            continue

        for arr, colour, lbl in [(u_data, "C0", "u (stage1)"),
                                  (u_model, "C1", "model")]:
            snippets = np.array([arr[e - pre : e + post, ni] for e in events])
            mean = np.nanmean(snippets, 0)
            sem  = np.nanstd(snippets, 0) / max(np.sqrt(len(events)), 1)
            ax.plot(win_time, mean, color=colour, lw=1.2, label=lbl)
            ax.fill_between(win_time, mean - sem, mean + sem,
                            color=colour, alpha=0.18)

        ax.axvline(0, color="red", ls="--", lw=0.8, label="stim")
        ax.set_title(f"{labels[ni]} (n={len(events)})", fontsize=8)
        if idx % n_cols == 0:
            ax.set_ylabel("u")
        if idx == 0:
            ax.legend(fontsize=6, loc="upper right")

    for ax in axes[-1]:
        ax.set_xlabel("Time from stim (s)")
    for idx in range(n_stim, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    fig.suptitle("Stim-triggered averages  (stimulated neurons)", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(save_dir / "opto_03_stim_triggered_avg.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  → opto_03_stim_triggered_avg.png")


def _response_matrix(u, stim_matrix, stim_neurons, baseline_frames=10,
                      response_frames=15):
    """(n_stim, N) matrix: mean Δu (post – pre) for each stim neuron."""
    T, N = u.shape
    n_stim = len(stim_neurons)
    mat = np.full((n_stim, N), np.nan)
    for si, ni in enumerate(stim_neurons):
        events = np.where(stim_matrix[:, ni] > 0)[0]
        events = events[(events >= baseline_frames) &
                        (events + response_frames <= T)]
        if len(events) == 0:
            continue
        deltas = []
        for e in events:
            pre  = u[e - baseline_frames : e, :].mean(0)
            post = u[e : e + response_frames, :].mean(0)
            deltas.append(post - pre)
        mat[si] = np.nanmean(deltas, 0)
    return mat


def _plot_response_heatmap(u_data, stim_matrix, labels,
                            stim_neurons, stim_names, save_dir):
    """Fig 4 – Δu response heatmap (data only)."""
    mat = _response_matrix(u_data, stim_matrix, stim_neurons)
    vmax = np.nanpercentile(np.abs(mat[np.isfinite(mat)]), 95)
    N = u_data.shape[1]

    fig, ax = plt.subplots(figsize=(max(14, N * 0.35), max(6, len(stim_neurons) * 0.35)))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_yticks(range(len(stim_neurons)))
    ax.set_yticklabels(stim_names, fontsize=7)
    ax.set_xticks(range(N))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("Stimulated neuron"); ax.set_xlabel("Responding neuron")
    ax.set_title("Stim-triggered response: Δu (post – pre baseline)")
    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.set_label("Δu")
    for si, ni in enumerate(stim_neurons):
        ax.plot(ni, si, "k*", ms=6)
    fig.tight_layout()
    fig.savefig(save_dir / "opto_04_response_heatmap.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  → opto_04_response_heatmap.png")
    return mat, vmax


def _plot_traces(u_data, u_model, stim_matrix, stim_conv, b, labels,
                  stim_neurons, dt, save_dir):
    """Fig 5 – full time-series traces for top-stimulated neurons."""
    T, N = u_data.shape
    time = np.arange(T) * dt

    # sort by number of events
    ev_counts = {ni: int((stim_matrix[:, ni] > 0).sum()) for ni in stim_neurons}
    top = sorted(stim_neurons, key=lambda x: -ev_counts[x])[:6]

    fig, axes = plt.subplots(len(top), 1,
                              figsize=(16, 2.5 * len(top)), sharex=True)
    if len(top) == 1:
        axes = [axes]

    for ax, ni in zip(axes, top):
        ax.plot(time, u_data[:, ni], "C0", lw=0.7, alpha=0.8, label="u (stage1)")
        ax.plot(time, u_model[:, ni], "C1", lw=0.7, alpha=0.8, label="model")

        events = np.where(stim_matrix[:, ni] > 0)[0]
        for e in events:
            ax.axvline(e * dt, color="red", alpha=0.5, lw=0.8)

        drive = b[ni] * stim_conv[:, ni]
        if np.abs(drive).max() > 1e-8:
            scale = np.nanstd(u_data[:, ni]) * 0.5 / np.abs(drive).max()
            ax.plot(time, drive * scale + np.nanmean(u_data[:, ni]),
                    "red", lw=0.8, alpha=0.6, label="b·stim (scaled)")

        ax.set_ylabel(f"{labels[ni]}\n(b={b[ni]:.3f})", fontsize=8)
        if ax is axes[0]:
            ax.legend(fontsize=7, loc="upper right", ncol=3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Full traces: stimulated neurons  (red lines = stim events)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(save_dir / "opto_05_traces.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  → opto_05_traces.png")


def _plot_data_vs_model_heatmap(u_data, u_model, stim_matrix, labels,
                                 stim_neurons, stim_names, vmax, save_dir):
    """Fig 6 – side-by-side heatmap: data vs model response."""
    mat_data  = _response_matrix(u_data,  stim_matrix, stim_neurons)
    mat_model = _response_matrix(u_model, stim_matrix, stim_neurons)
    N = u_data.shape[1]

    fig, axes = plt.subplots(1, 2, figsize=(max(14, N * 0.35), max(6, len(stim_neurons) * 0.35)))
    for ax, mat, title in [(axes[0], mat_data, "Data (stage1 u)"),
                           (axes[1], mat_model, "Model predictions")]:
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_yticks(range(len(stim_neurons)))
        ax.set_yticklabels(stim_names, fontsize=7)
        ax.set_xticks(range(N))
        ax.set_xticklabels(labels, rotation=90, fontsize=5)
        ax.set_ylabel("Stimulated"); ax.set_xlabel("Responding")
        ax.set_title(title)
        for si, ni in enumerate(stim_neurons):
            ax.plot(ni, si, "k*", ms=5)

    fig.colorbar(im, ax=axes, shrink=0.7, label="Δu")
    fig.suptitle("Response heatmap: data vs model", fontsize=11)
    fig.savefig(save_dir / "opto_06_data_vs_model_heatmap.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  → opto_06_data_vs_model_heatmap.png")


# ──────────────────────────────────────────────────────────────────────
# main entry point
# ──────────────────────────────────────────────────────────────────────

def plot_all(h5_path: str, results_dir: str,
             save_dir: str | None = None, device: str = "cpu",
             **cfg_kw):
    save_dir = Path(save_dir or results_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[opto_analysis] h5       = {h5_path}")
    print(f"[opto_analysis] results  = {results_dir}")
    print(f"[opto_analysis] save_dir = {save_dir}")

    # ── load raw data ──────────────────────────────────────────────
    with h5py.File(h5_path, "r") as f:
        labels = [l.decode() if isinstance(l, bytes) else str(l)
                  for l in f["gcamp/neuron_labels"][:]]
        stim_matrix   = np.array(f["optogenetics/stim_matrix"], dtype=float)
        stim_cell_idx = np.array(f["optogenetics/stim_cell_indices"])
        u_stage1      = np.array(f["stage1/u_mean"], dtype=float)

    T, N = u_stage1.shape

    stim_neurons = sorted(set(stim_cell_idx.tolist()))
    stim_names = [labels[i] for i in stim_neurons]
    print(f"[opto_analysis] {len(stim_neurons)} stimulated neurons, "
          f"{int((stim_matrix > 0).sum())} events, T={T}, N={N}")

    # ── load trained model ────────────────────────────────────────
    model, data, cfg = _load_trained_model(
        h5_path, results_dir, device, **cfg_kw)
    model.eval()
    dt = float(cfg.dt)

    b = model.b.detach().cpu().numpy().flatten()           # (N,)
    kernel = None
    if model.stim_kernel is not None:
        kernel = F.softplus(model.stim_kernel.detach().cpu()).numpy()  # (K,)

    # ── convolved stimulus ────────────────────────────────────────
    stim_t = torch.tensor(stim_matrix, dtype=torch.float32)
    with torch.no_grad():
        if model.stim_kernel is not None:
            stim_conv = model.convolve_stimulus(
                stim_t.to(next(model.parameters()).device)
            ).cpu().numpy()
        else:
            stim_conv = stim_matrix.copy()

    # ── model forward pass ────────────────────────────────────────
    gating = data.get("gating")
    u_model = _forward_pass(model, u_stage1, stim_conv, gating, device)

    # ── generate plots ────────────────────────────────────────────
    print("[opto_analysis] Generating plots …")

    if kernel is not None:
        _plot_kernel(kernel, cfg.dt, save_dir)

    _plot_b_weights(b, labels, set(stim_neurons), save_dir)

    _plot_stim_triggered_avg(u_stage1, u_model, stim_matrix,
                              labels, stim_neurons, dt, save_dir)

    _, vmax = _plot_response_heatmap(u_stage1, stim_matrix, labels,
                                      stim_neurons, stim_names, save_dir)

    _plot_traces(u_stage1, u_model, stim_matrix, stim_conv,
                  b, labels, stim_neurons, dt, save_dir)

    _plot_data_vs_model_heatmap(u_stage1, u_model, stim_matrix, labels,
                                 stim_neurons, stim_names, vmax, save_dir)

    print(f"[opto_analysis] All 6 figures saved to {save_dir}")


def main():
    p = argparse.ArgumentParser(description="Optogenetics analysis plots")
    p.add_argument("--h5", required=True, help="Path to input HDF5 file")
    p.add_argument("--results_dir", required=True,
                   help="Stage-2 output directory (with stage2_results.h5)")
    p.add_argument("--save_dir", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--set", nargs=2, action="append",
                   metavar=("KEY", "VALUE"), default=[], dest="overrides")
    args = p.parse_args()

    kw = {}
    for key, val in args.overrides:
        for conv in (int, float):
            try: val = conv(val); break
            except ValueError: pass
        else:
            if val.lower() in ("true", "false"):
                val = val.lower() == "true"
        kw[key] = val

    plot_all(args.h5, args.results_dir, args.save_dir, args.device, **kw)


if __name__ == "__main__":
    main()
