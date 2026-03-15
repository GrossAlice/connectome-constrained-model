from __future__ import annotations

import numpy as np
import torch
import h5py
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from .model import Stage2ModelPT
from .evaluate import _pearson, _r2, run_full_evaluation
from . import get_stage2_logger

__all__ = ["generate_eval_loo_plots", "run_full_evaluation"]

_COL_DATA = "#2166ac"
_COL_RAW  = "#d62728"
_COL_CV   = "#2ca02c"
_COL_FR   = "#9467bd"

def setup_plot_style():
    import matplotlib
    import matplotlib.pyplot as plt

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [
        'Nimbus Sans', 'Helvetica', 'Liberation Sans', 'Arial', 'DejaVu Sans',
    ]
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Nimbus Sans'
    plt.rcParams['mathtext.it'] = 'Nimbus Sans:italic'
    plt.rcParams['mathtext.bf'] = 'Nimbus Sans:bold'
    plt.rcParams['mathtext.cal'] = 'Nimbus Sans:italic'

    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 16
    plt.rcParams['figure.titlesize'] = 22

    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['ytick.major.width'] = 1.5
    plt.rcParams['xtick.major.size'] = 6
    plt.rcParams['ytick.major.size'] = 6
    plt.rcParams['lines.linewidth'] = 1.5

    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 200
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.15

    family = plt.rcParams['font.sans-serif'][0]
    return family


def _style(ax, xlabel="", ylabel="", title=""):
    ax.set_xlabel(xlabel, fontsize=20, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=20, labelpad=8)
    if title:
        ax.set_title(title, fontsize=24, fontweight="bold", pad=14)
    ax.tick_params(labelsize=16, width=1.5, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save_png(fig, save_dir: str, filename: str, **kwargs):
    fig.savefig(Path(save_dir) / filename, **kwargs)


def _smooth_1d(x: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    k = int(max(1, window))
    if k <= 1 or arr.size == 0:
        return arr
    kernel = np.ones(k, dtype=float) / float(k)
    return np.convolve(arr, kernel, mode="same")


def _compute_input_decomposition(
    model: Stage2ModelPT,
    data: Dict[str, Any],
) -> Tuple[Dict[str, float], np.ndarray]:
    """Compute RMS of each dynamical term in the update equation.

    Returns
    -------
    summary : dict  label → global RMS
    per_neuron : ndarray (N, 4)  per-neuron RMS of [target_resid, gap, sv, dcv]
    """
    device = next(model.parameters()).device
    u = data["u_stage1"].to(device)
    T, N = u.shape
    gating = data.get("gating")
    stim = data.get("stim")

    with torch.no_grad():
        lam = model.lambda_u
        ar1 = (1.0 - lam) * u[:-1] + lam * model.I0
        target_resid = u[1:] - ar1  # what I_gap + I_sv + I_dcv should explain

        L = model.laplacian()
        gap_all = lam * (L @ u[:-1].T).T

        s_sv = torch.zeros(N, model.r_sv, device=device)
        s_dcv = torch.zeros(N, model.r_dcv, device=device)
        sv_all = torch.zeros(T - 1, N, device=device)
        dcv_all = torch.zeros(T - 1, N, device=device)
        pred_all = torch.zeros(T - 1, N, device=device)

        for t in range(T - 1):
            g = gating[t] if gating is not None else torch.ones(N, device=device)
            s_stim = stim[t] if stim is not None else None
            phi_gated = model.phi(u[t]) * g

            I_sv = I_dcv = torch.zeros(N, device=device)
            if model.r_sv > 0:
                I_sv, s_sv = model._synaptic_current(
                    u[t], phi_gated, s_sv,
                    model.T_sv * model._get_W("W_sv"),
                    model.a_sv, model.tau_sv, model.E_sv)
            if model.r_dcv > 0:
                I_dcv, s_dcv = model._synaptic_current(
                    u[t], phi_gated, s_dcv,
                    model.T_dcv * model._get_W("W_dcv"),
                    model.a_dcv, model.tau_dcv, model.E_dcv)

            sv_all[t] = lam * I_sv
            dcv_all[t] = lam * I_dcv
            pred_all[t] = ar1[t] + gap_all[t] + sv_all[t] + dcv_all[t]

        unexplained = u[1:] - pred_all

        def _rms(x: torch.Tensor) -> float:
            return float(x.pow(2).mean().sqrt().item())

        # per-neuron RMS (N,)
        def _rms_per(x: torch.Tensor) -> np.ndarray:
            return x.pow(2).mean(dim=0).sqrt().cpu().numpy()

        per_neuron = np.column_stack([
            _rms_per(target_resid),
            _rms_per(gap_all),
            _rms_per(sv_all),
            _rms_per(dcv_all),
        ])

    summary = {
        "Target\nresidual": _rms(target_resid),
        "$\\lambda I_{gap}$": _rms(gap_all),
        "$\\lambda I_{sv}$": _rms(sv_all),
        "$\\lambda I_{dcv}$": _rms(dcv_all),
        "Unexplained": _rms(unexplained),
    }
    return summary, per_neuron


def _residual_acf(residuals: np.ndarray, max_lag: int = 20) -> np.ndarray:
    """Mean autocorrelation of one-step residuals across neurons."""
    T, N = residuals.shape
    acf = np.zeros(max_lag)
    for lag in range(1, max_lag + 1):
        r_t = residuals[lag:]
        r_tm = residuals[:T - lag]
        valid = np.isfinite(r_t) & np.isfinite(r_tm)
        if valid.sum() < 10:
            acf[lag - 1] = np.nan
            continue
        a, b = r_t[valid], r_tm[valid]
        a = a - a.mean()
        b = b - b.mean()
        denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
        acf[lag - 1] = float((a * b).sum() / denom) if denom > 0 else 0.0
    return acf

def _get_labels(data: Dict[str, Any]) -> Optional[list]:
    n_expected = int(data.get("u_stage1").shape[1]) if data.get("u_stage1") is not None else None

    def _clean_labels(labels_in: list, n: Optional[int]) -> list:
        out: list[str] = []
        for i, raw in enumerate(labels_in):
            s = "" if raw is None else str(raw).strip()
            if (s == "") or (s.lower() == "nan"):
                s = f"Neuron_{i}"
            out.append(s)
        if n is not None:
            if len(out) < n:
                out.extend([f"Neuron_{j}" for j in range(len(out), n)])
            elif len(out) > n:
                out = out[:n]
        return out

    labels = data.get("neuron_labels")
    if labels is not None:
        return _clean_labels(list(labels), n_expected)
    cfg = data.get("_cfg")
    if cfg is not None:
        try:
            with h5py.File(cfg.h5_path, "r") as f:
                if "gcamp/neuron_labels" in f:
                    raw = f["gcamp/neuron_labels"][:]
                    decoded = [s.decode() if isinstance(s, bytes) else str(s) for s in raw]
                    return _clean_labels(decoded, n_expected)
        except Exception:
            pass
    return None


def _label_for_idx(labels: Optional[list], idx: int) -> str:
    if labels is None:
        return f"Neuron_{idx}"
    if idx < 0 or idx >= len(labels):
        return f"Neuron_{idx}"
    s = "" if labels[idx] is None else str(labels[idx]).strip()
    return s if s else f"Neuron_{idx}"
def _partner_indices(data: Dict[str, Any], idx: int) -> Tuple[np.ndarray, np.ndarray]:
    N = int(data["u_stage1"].shape[1])

    def _to_np(x: Any) -> np.ndarray:
        if x is None:
            return np.zeros((N, N), dtype=float)
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(float)
        arr = np.asarray(x, dtype=float)
        if arr.shape != (N, N):
            return np.zeros((N, N), dtype=float)
        return arr

    te = _to_np(data.get("T_e"))
    tsv = _to_np(data.get("T_sv"))
    tdcv = _to_np(data.get("T_dcv"))
    a = (np.abs(te) > 0) | (np.abs(tsv) > 0) | (np.abs(tdcv) > 0)
    np.fill_diagonal(a, False)

    presyn = np.where(a[:, idx])[0]
    postsyn = np.where(a[idx, :])[0]
    return presyn, postsyn


def _plot_partner_panel(ax, data: Dict[str, Any], labels: Optional[list], idx: int):
    ax.axis("off")

    center_lbl = _label_for_idx(labels, idx)
    presyn, postsyn = _partner_indices(data, idx)

    ax.text(0.5, 0.98, center_lbl, ha="center", va="top",
            fontsize=11, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.90, f"pre={len(presyn)} | post={len(postsyn)}",
            ha="center", va="top", fontsize=9, color="0.35",
            transform=ax.transAxes)

    ax.scatter([0.5], [0.52], s=160, c=_COL_RAW, edgecolors="k",
               linewidths=0.8, transform=ax.transAxes, zorder=5)

    if len(presyn) > 0:
        y_pre = np.linspace(0.15, 0.85, len(presyn))
        ax.scatter(np.full(len(presyn), 0.18), y_pre, s=28, c=_COL_DATA,
                   alpha=0.9, transform=ax.transAxes)
        for y in y_pre:
            ax.annotate("", xy=(0.46, 0.52), xytext=(0.22, float(y)),
                        xycoords=ax.transAxes,
                        arrowprops=dict(arrowstyle="->", lw=0.7,
                                        color=_COL_DATA, alpha=0.45))
    if len(postsyn) > 0:
        y_post = np.linspace(0.15, 0.85, len(postsyn))
        ax.scatter(np.full(len(postsyn), 0.82), y_post, s=28, c=_COL_CV,
                   alpha=0.9, transform=ax.transAxes)
        for y in y_post:
            ax.annotate("", xy=(0.78, float(y)), xytext=(0.54, 0.52),
                        xycoords=ax.transAxes,
                        arrowprops=dict(arrowstyle="->", lw=0.7,
                                        color=_COL_CV, alpha=0.45))

    pre_text = (", ".join(_label_for_idx(labels, int(i)) for i in presyn)
                if len(presyn) else "None")
    post_text = (", ".join(_label_for_idx(labels, int(i)) for i in postsyn)
                 if len(postsyn) else "None")
    ax.text(0.02, 0.03, f"Pre: {pre_text}", ha="left", va="bottom",
            fontsize=7.5, color=_COL_DATA, transform=ax.transAxes, wrap=True)
    ax.text(0.98, 0.03, f"Post: {post_text}", ha="right", va="bottom",
            fontsize=7.5, color=_COL_CV, transform=ax.transAxes, wrap=True)
def plot_summary_slide(
    model: Stage2ModelPT,
    data: Dict[str, Any],
    onestep: Dict[str, Any],
    loo: Dict[str, Any],
    free_run: Dict[str, Any],
    beh_results: Optional[Dict[str, Any]],
    save_dir: str,
    epoch_losses: Optional[list] = None,
    beh_all: Optional[Dict[str, Any]] = None,
    beh_frozen: Optional[Dict[str, Any]] = None,
):
    """Comprehensive evaluation dashboard (plot 00).

    3×4 grid — every panel is the same size.
    Row 0: Training convergence | One-step R² | LOO R² | Free-run R²
    Row 1: Behaviour decoding  | Multistep rollout | α_sv / α_dcv dist.     | Synaptic kernels
    Row 2: Input decomposition | R²: one-step vs LOO | Residual autocorr.   | λ_u vs one-step R²
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    r2_os = onestep["r2"]
    r2_loo = loo["r2"]
    r2_fr = free_run["r2"]
    dt = float(data["dt"])
    u_np = data["u_stage1"].cpu().numpy()
    T, N = u_np.shape

    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(3, 4, hspace=0.42, wspace=0.38)

    ax = fig.add_subplot(gs[0, 0])
    epoch_losses = epoch_losses or []
    if epoch_losses:
        epochs = np.arange(1, len(epoch_losses) + 1)
        ax.plot(epochs, [e["dynamics"] for e in epoch_losses],
                color=_COL_DATA, lw=2.2, label="Dynamics")
        ax.plot(epochs, [e["total"] for e in epoch_losses],
                color=_COL_RAW, lw=1.8, alpha=0.8, label="Total")
        if any(e.get("behaviour_loss") is not None for e in epoch_losses):
            beh_loss = [e.get("behaviour_loss", np.nan) for e in epoch_losses]
            ax.plot(epochs, beh_loss, color=_COL_CV, lw=1.6, alpha=0.8,
                    label="Behaviour")
        ax.set_yscale("log")
        _style(ax, xlabel="Epoch", ylabel="Loss", title="A. Training Convergence")
        ax.legend(frameon=False, fontsize=12, ncol=1)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No training history",
                ha="center", va="center", fontsize=14)

    _hist_kw = dict(alpha=0.75, edgecolor="white")
    for col, (r2, color, label, title) in enumerate([
        (r2_os, _COL_DATA, "One-step R\u00b2", "B. One-step R\u00b2"),
        (r2_loo, _COL_CV,  "LOO R\u00b2",      "C. LOO R\u00b2"),
        (r2_fr,  _COL_FR,  "Free-run R\u00b2",  "D. Free-run R\u00b2"),
    ], start=1):
        ax = fig.add_subplot(gs[0, col])
        valid = r2[np.isfinite(r2)]
        if len(valid) > 0:
            bins = np.linspace(min(valid.min(), -0.1),
                               max(valid.max(), 1.0), 30)
            ax.hist(valid, bins=bins, color=color, **_hist_kw)
            ax.axvline(np.median(valid), color=_COL_RAW, lw=2, ls="--",
                       label=f"med={np.median(valid):.3f}")
            ax.legend(frameon=False, fontsize=11)
        _style(ax, xlabel=label, ylabel="Count", title=title)

    ax = fig.add_subplot(gs[1, 0])
    if beh_results is not None:
        # Red = frozen decoder on GT motor neurons (held-out CV baseline)
        frozen_src = beh_frozen if beh_frozen is not None else beh_results
        r2_motor = frozen_src.get("r2_gt_heldout")
        if r2_motor is None or not np.any(np.isfinite(r2_motor)):
            r2_motor = frozen_src.get("r2_gt", np.full(6, np.nan))

        # Orange = AR(1)-smoothed motor neurons (leak only, no network)
        r2_ar1 = beh_results.get("r2_ar1", np.full(6, np.nan))

        # Green = E2E decoder on model-predicted motor neurons (held-out CV)
        r2_model = beh_results.get("r2_model_heldout")
        if r2_model is None or not np.any(np.isfinite(r2_model)):
            r2_model = beh_results.get("r2_model")
        if r2_model is None:
            r2_model = np.full_like(r2_motor, np.nan)

        n_modes = min(len(r2_model), len(r2_motor), 5)
        x = np.arange(n_modes)
        has_all = beh_all is not None and "r2_all_neurons" in beh_all
        has_ar1 = np.any(np.isfinite(r2_ar1[:n_modes]))

        # Determine number of bar groups and width
        n_bars = 2 + int(has_all) + int(has_ar1)
        w = 0.8 / n_bars
        offsets = np.linspace(-(n_bars - 1) * w / 2, (n_bars - 1) * w / 2, n_bars)
        bi = 0
        if has_all:
            r2_all_n = beh_all["r2_all_neurons"]
            ax.bar(x + offsets[bi], r2_all_n[:n_modes], w, label="All neurons (GT)",
                   color=_COL_DATA, alpha=0.75)
            bi += 1
        ax.bar(x + offsets[bi], r2_motor[:n_modes], w, label="Motor neurons (GT)",
               color=_COL_RAW, alpha=0.75)
        bi += 1
        if has_ar1:
            ax.bar(x + offsets[bi], r2_ar1[:n_modes], w, label="AR(1) baseline",
                   color="orange", alpha=0.75)
            bi += 1
        ax.bar(x + offsets[bi], r2_model[:n_modes], w, label="Model (E2E)",
               color=_COL_CV, alpha=0.75)
        ax.set_xticks(x)
        ax.set_xticklabels([f"EW{i + 1}" for i in range(n_modes)],
                           fontsize=10)
        ax.legend(frameon=False, fontsize=9, loc="upper right")
        _style(ax, xlabel="Eigenworm", ylabel="R\u00b2",
               title="E. Behaviour Decoding")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No behaviour",
                ha="center", va="center", fontsize=14)

    ax = fig.add_subplot(gs[1, 1])
    steps_list = (1, 5, 10, 20)
    device = next(model.parameters()).device
    u = data["u_stage1"].to(device)
    gating = data.get("gating")
    stim = data.get("stim")

    with torch.no_grad():
        s_sv_all = torch.zeros(T, N, model.r_sv, device=device)
        s_dcv_all = torch.zeros(T, N, model.r_dcv, device=device)
        s_sv_w = torch.zeros(N, model.r_sv, device=device)
        s_dcv_w = torch.zeros(N, model.r_dcv, device=device)
        for t in range(T):
            s_sv_all[t], s_dcv_all[t] = s_sv_w, s_dcv_w
            g = (gating[t] if gating is not None
                 else torch.ones(N, device=device))
            s = stim[t] if stim is not None else None
            _, s_sv_w, s_dcv_w = model.prior_step(
                u[t], s_sv_w, s_dcv_w, g, s)

    medians_raw = []
    for K in steps_list:
        stride = max(1, K // 2)
        preds_raw = np.zeros_like(u_np)
        counts = np.zeros(T)
        with torch.no_grad():
            for t0 in range(0, T - K, stride):
                u_t_raw = u[t0]
                sv_r, dcv_r = s_sv_all[t0].clone(), s_dcv_all[t0].clone()
                for k in range(1, K + 1):
                    g = (gating[t0 + k - 1] if gating is not None
                         else torch.ones(N, device=device))
                    s = stim[t0 + k - 1] if stim is not None else None
                    u_t_raw, sv_r, dcv_r = model.prior_step(u_t_raw, sv_r, dcv_r, g, s)
                preds_raw[t0 + K] += u_t_raw.cpu().numpy()
                counts[t0 + K] += 1
        v = counts > 0
        preds_raw[v] /= counts[v, None]
        r2_k_raw = np.array([_r2(u_np[v, i], preds_raw[v, i]) for i in range(N)])
        medians_raw.append(np.nanmedian(r2_k_raw))
    ax.plot(steps_list, medians_raw, color=_COL_CV, marker="o",
            ls="-", lw=2.5, markersize=7, label="Model")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.legend(frameon=False, fontsize=11)
    _style(ax, xlabel="Horizon K (steps)", ylabel="Median R\u00b2",
           title="F. Multistep Rollout")

    ax = fig.add_subplot(gs[1, 2])
    with torch.no_grad():
        a_sv_plot = model.a_sv.detach().cpu().numpy() if model.r_sv > 0 else np.array([])
        a_dcv_plot = model.a_dcv.detach().cpu().numpy() if model.r_dcv > 0 else np.array([])
    a_sv_flat = np.asarray(a_sv_plot, dtype=float).reshape(-1)
    a_dcv_flat = np.asarray(a_dcv_plot, dtype=float).reshape(-1)
    a_sv_flat = a_sv_flat[np.isfinite(a_sv_flat)]
    a_dcv_flat = a_dcv_flat[np.isfinite(a_dcv_flat)]
    if a_sv_flat.size > 0 or a_dcv_flat.size > 0:
        if a_sv_flat.size > 0:
            ax.hist(a_sv_flat, bins=25, color=_COL_DATA, alpha=0.55,
                    edgecolor="white", label=f"SV α (med={np.median(a_sv_flat):.3g})")
            ax.axvline(np.median(a_sv_flat), color=_COL_DATA, lw=2, ls="--")
        if a_dcv_flat.size > 0:
            ax.hist(a_dcv_flat, bins=25, color=_COL_CV, alpha=0.55,
                    edgecolor="white", label=f"DCV α (med={np.median(a_dcv_flat):.3g})")
            ax.axvline(np.median(a_dcv_flat), color=_COL_CV, lw=2, ls="--")
        ax.legend(frameon=False, fontsize=11)
        # Percentile-based xlim so outliers don't compress the bulk
        all_a = np.concatenate([x for x in [a_sv_flat, a_dcv_flat] if x.size > 0])
        if all_a.size > 0:
            hi_clip = np.percentile(all_a, 99) * 1.15
            if hi_clip > 0:
                ax.set_xlim(left=0, right=hi_clip)
    else:
        ax.text(0.5, 0.5, "No alpha parameters",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=14, color="0.5")
    _style(ax, xlabel="Amplitude α", ylabel="Count",
           title="G. Kernel Amplitudes")

    ax = fig.add_subplot(gs[1, 3])
    t_kernel = np.arange(0, 60.0, dt)
    with torch.no_grad():
        a_sv = (model.a_sv.cpu().numpy() if model.r_sv > 0
                else np.array([]))
        tau_sv = (model.tau_sv.cpu().numpy() if model.r_sv > 0
                  else np.array([]))
        a_dcv = (model.a_dcv.cpu().numpy() if model.r_dcv > 0
                 else np.array([]))
        tau_dcv = (model.tau_dcv.cpu().numpy() if model.r_dcv > 0
                   else np.array([]))
    if np.ndim(a_sv) == 2:
        a_sv = np.nanmean(a_sv, axis=0)
    if np.ndim(a_dcv) == 2:
        a_dcv = np.nanmean(a_dcv, axis=0)
    if len(a_sv) > 0:
        kernel_sv = sum(a_sv[r] * np.exp(-t_kernel / (tau_sv[r] + 1e-12))
                        for r in range(len(a_sv)))
        ax.plot(t_kernel, kernel_sv, color=_COL_DATA, lw=2.5, label="SV")
    if len(a_dcv) > 0:
        kernel_dcv = sum(a_dcv[r] * np.exp(-t_kernel / (tau_dcv[r] + 1e-12))
                         for r in range(len(a_dcv)))
        ax.plot(t_kernel, kernel_dcv, color=_COL_CV, lw=2.5, label="DCV")
    ax.legend(frameon=False, fontsize=13)
    _style(ax, xlabel="Lag (s)", ylabel="Amplitude",
           title="H. Synaptic Kernels")

    with torch.no_grad():
        lam = model.lambda_u.cpu().numpy().ravel()
        G_val = model.G.cpu().numpy()

    # --- Row 2: Dynamics diagnostics -----------------------------------

    # I. Input decomposition
    ax = fig.add_subplot(gs[2, 0])
    decomp, per_neuron_rms = _compute_input_decomposition(model, data)
    labels_d = list(decomp.keys())
    vals_d = [decomp[k] for k in labels_d]
    colours_d = [_COL_RAW, _COL_DATA, _COL_CV, _COL_FR, "0.55"]
    y_pos = np.arange(len(labels_d))
    ax.barh(y_pos, vals_d, color=colours_d[:len(vals_d)], alpha=0.8,
            edgecolor="white", height=0.65)
    for i, v in enumerate(vals_d):
        ax.text(v + max(vals_d) * 0.02, i, f"{v:.4g}", va="center",
                fontsize=10)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_d, fontsize=12)
    ax.invert_yaxis()
    _style(ax, xlabel="RMS", ylabel="", title="I. Input Decomposition")

    # J. Parameter trajectories over epochs
    ax = fig.add_subplot(gs[2, 1])
    epoch_losses = epoch_losses or []
    if epoch_losses and "G" in epoch_losses[0]:
        ep_x = np.arange(1, len(epoch_losses) + 1)
        ax.plot(ep_x, [e["G"] for e in epoch_losses],
                color=_COL_DATA, lw=2, label="G")
        ax.plot(ep_x, [e["a_sv_rms"] for e in epoch_losses],
                color=_COL_CV, lw=2, label="\u03b1_sv RMS")
        ax.plot(ep_x, [e["a_dcv_rms"] for e in epoch_losses],
                color=_COL_FR, lw=2, label="\u03b1_dcv RMS")
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.legend(frameon=False, fontsize=9, loc="best")
    else:
        ax.text(0.5, 0.5, "No param history",
                ha="center", va="center", fontsize=14,
                transform=ax.transAxes)
    _style(ax, xlabel="Epoch", ylabel="Value",
           title="J. Parameter Trajectories")

    # K. Residual autocorrelation
    ax = fig.add_subplot(gs[2, 2])
    mu_os = onestep.get("prior_mu")
    if mu_os is not None:
        mu_os_np = mu_os.cpu().numpy() if isinstance(mu_os, torch.Tensor) else mu_os
        resid = u_np - mu_os_np
        acf = _residual_acf(resid, max_lag=20)
        lags = np.arange(1, len(acf) + 1)
        ax.bar(lags, acf, color=_COL_DATA, alpha=0.7, edgecolor="white")
        ax.axhline(0, color="gray", lw=0.8, ls=":")
        # 95% CI for white noise
        ci = 1.96 / np.sqrt(T)
        ax.axhline(ci, color=_COL_RAW, lw=1, ls="--", alpha=0.5)
        ax.axhline(-ci, color=_COL_RAW, lw=1, ls="--", alpha=0.5,
                   label=f"95% CI (n={T})")
        ax.legend(frameon=False, fontsize=10)
    _style(ax, xlabel="Lag (steps)", ylabel="Autocorrelation",
           title="K. Residual Autocorrelation")

    # L. Per-neuron network balance
    ax = fig.add_subplot(gs[2, 3])
    # per_neuron_rms columns: [target_resid, gap, sv, dcv]
    tgt_rms_pn = per_neuron_rms[:, 0]
    net_rms_pn = per_neuron_rms[:, 1] + per_neuron_rms[:, 2] + per_neuron_rms[:, 3]
    ratio = np.where(tgt_rms_pn > 1e-12, net_rms_pn / tgt_rms_pn, 0.0)
    order = np.argsort(ratio)
    colors_ratio = np.where(ratio[order] <= 1.0, _COL_CV, _COL_RAW)
    ax.barh(np.arange(N), ratio[order], color=colors_ratio, alpha=0.6,
            height=1.0, edgecolor="none")
    ax.axvline(1.0, color="0.3", lw=1.5, ls="--", label="balanced")
    n_over = int((ratio > 1.0).sum())
    ax.text(0.97, 0.97,
            f"{n_over}/{N} neurons\nnetwork > target",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round", fc="#f7f7f7", alpha=0.85))
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=10)
    _style(ax, xlabel="Network RMS / Target RMS",
           ylabel="Neurons (sorted)",
           title="L. Per-Neuron Network Balance")

    a_sv_med = np.nanmedian(a_sv_flat) if a_sv_flat.size > 0 else float("nan")
    a_dcv_med = np.nanmedian(a_dcv_flat) if a_dcv_flat.size > 0 else float("nan")

    G_txt = (f"{float(G_val):.6g}" if G_val.ndim == 0
             else f"mean={G_val.mean():.4f}")
    n_loo = int(np.isfinite(r2_loo).sum())
    n_ep = len(epoch_losses)
    fr_med = np.nanmedian(r2_fr[np.isfinite(r2_fr)]) if np.any(np.isfinite(r2_fr)) else float("nan")
    alpha_sv_txt = f"  α_sv med={a_sv_med:.3g}" if np.isfinite(a_sv_med) else ""
    alpha_dcv_txt = f"  α_dcv med={a_dcv_med:.3g}" if np.isfinite(a_dcv_med) else ""
    info = (f"N={N}  T={T}  dt={dt:.3f}s  G={G_txt}  "
            f"LOO={n_loo}  epochs={n_ep}  "
            f"free-run med R\u00b2={fr_med:.3f}{alpha_sv_txt}{alpha_dcv_txt}")
    fig.text(0.99, 0.995, info, ha="right", va="top", fontsize=11,
             bbox=dict(boxstyle="round", fc="#f0f0f0", alpha=0.8))

    fig.suptitle("Stage 2 Evaluation Overview",
                 fontsize=26, fontweight="bold", y=1.01)
    _save_png(fig, save_dir, "00_summary_slide.png", bbox_inches="tight")
    plt.close(fig)


# ======================================================================= #
#  Parameter trajectory plot                                                #
# ======================================================================= #

def plot_parameter_trajectories(
    epoch_losses: list,
    save_dir: str,
    cfg=None,
):
    """Dedicated multi-panel figure showing how every tracked parameter
    evolves across training epochs.  Saved as ``param_trajectories.png``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    if not epoch_losses:
        return

    ep = np.arange(1, len(epoch_losses) + 1)

    def _get(key, default=np.nan):
        vals = []
        for entry in epoch_losses:
            value = entry.get(key, default)
            vals.append(default if value is None else value)
        return np.asarray(vals, dtype=float)

    n_rows, n_cols = 3, 3
    fig = plt.figure(figsize=(18, 13))
    gs = GridSpec(n_rows, n_cols, hspace=0.45, wspace=0.35)

    # --- (0,0) Loss curves ---
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(ep, _get("dynamics"), color=_COL_DATA, lw=2, label="Dynamics")
    ax.plot(ep, _get("total"), color=_COL_RAW, lw=1.6, alpha=0.8, label="Total")
    beh = _get("behaviour_loss")
    if np.any(np.isfinite(beh)):
        ax.plot(ep, beh, color=_COL_CV, lw=1.6, alpha=0.8, label="Behaviour")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=9)
    _style(ax, xlabel="Epoch", ylabel="Loss", title="Loss")

    # --- (0,1) Gap-junction conductance G ---
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(ep, _get("G"), color=_COL_DATA, lw=2)
    ax.axhline(_get("G")[0], color="0.6", lw=1, ls=":", label="init")
    ax.legend(frameon=False, fontsize=9)
    _style(ax, xlabel="Epoch", ylabel="G (mean)", title="Gap-Junction G")

    # --- (0,2) Kernel amplitudes: α_sv and α_dcv RMS ---
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(ep, _get("a_sv_rms"), color=_COL_DATA, lw=2, label="α_sv RMS")
    ax.plot(ep, _get("a_dcv_rms"), color=_COL_CV, lw=2, label="α_dcv RMS")
    ax.set_yscale("symlog", linthresh=1e-5)
    ax.legend(frameon=False, fontsize=9)
    _style(ax, xlabel="Epoch", ylabel="RMS", title="Kernel Amplitudes")

    # --- (1,0) λ_u statistics ---
    ax = fig.add_subplot(gs[1, 0])
    ax.fill_between(ep, _get("lambda_u_min"), _get("lambda_u_max"),
                    color=_COL_DATA, alpha=0.15, label="min–max")
    ax.plot(ep, _get("lambda_u_med"), color=_COL_DATA, lw=2, label="median")
    ax.plot(ep, _get("lambda_u_mean"), color=_COL_CV, lw=1.5, ls="--", label="mean")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False, fontsize=9)
    _style(ax, xlabel="Epoch", ylabel="λ_u", title="Leak Rate λ_u")

    # --- (1,1) Tonic drive I0 ---
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(ep, _get("I0_rms"), color=_COL_DATA, lw=2, label="RMS")
    ax.plot(ep, _get("I0_absmax"), color=_COL_RAW, lw=1.5, ls="--", label="|max|")
    ax.legend(frameon=False, fontsize=9)
    _style(ax, xlabel="Epoch", ylabel=r"$I_0$", title=r"Tonic Drive $I_0$")

    # --- (1,2) Reversal potentials E_sv, E_dcv ---
    ax = fig.add_subplot(gs[1, 2])
    e_sv_mean = _get("E_sv_mean")
    e_sv_min = _get("E_sv_min")
    e_sv_max = _get("E_sv_max")
    ax.fill_between(ep, e_sv_min, e_sv_max, color=_COL_DATA, alpha=0.15,
                    label="E_sv range")
    ax.plot(ep, e_sv_mean, color=_COL_DATA, lw=2, label="E_sv mean")
    ax.plot(ep, _get("E_dcv"), color=_COL_CV, lw=2, label="E_dcv")
    ax.axhline(0, color="0.6", lw=0.8, ls=":")
    ax.legend(frameon=False, fontsize=9)
    _style(ax, xlabel="Epoch", ylabel="Reversal (E)", title="Reversal Potentials")

    # --- (2,0) Time constants τ ---
    ax = fig.add_subplot(gs[2, 0])
    tau_sv_all = [e.get("tau_sv", []) for e in epoch_losses]
    tau_dcv_all = [e.get("tau_dcv", []) for e in epoch_losses]
    if tau_sv_all and len(tau_sv_all[0]) > 0:
        tau_sv_arr = np.array(tau_sv_all)
        for r in range(tau_sv_arr.shape[1]):
            ax.plot(ep, tau_sv_arr[:, r], color=_COL_DATA, lw=1.5,
                    alpha=0.7, label=f"τ_sv[{r}]" if r < 3 else None)
    if tau_dcv_all and len(tau_dcv_all[0]) > 0:
        tau_dcv_arr = np.array(tau_dcv_all)
        for r in range(tau_dcv_arr.shape[1]):
            ax.plot(ep, tau_dcv_arr[:, r], color=_COL_CV, lw=1.5, ls="--",
                    alpha=0.7, label=f"τ_dcv[{r}]" if r < 3 else None)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(frameon=False, fontsize=8, ncol=2, loc="best")
    _style(ax, xlabel="Epoch", ylabel="τ (s)", title="Time Constants")

    # --- (2,1) Stimulus weights b ---
    ax = fig.add_subplot(gs[2, 1])
    b_norm = _get("b_norm")
    if np.any(b_norm > 0):
        ax.plot(ep, b_norm, color=_COL_DATA, lw=2)
        _style(ax, xlabel="Epoch", ylabel="‖b‖", title="Stimulus Weights ‖b‖")
    else:
        ax.text(0.5, 0.5, "No stimulus", ha="center", va="center",
                fontsize=13, transform=ax.transAxes, color="0.5")
        _style(ax, xlabel="", ylabel="", title="Stimulus Weights")

    # --- (2,2) Behaviour R² over epochs (if available) ---
    ax = fig.add_subplot(gs[2, 2])
    beh_r2 = _get("beh_r2_eval")
    if np.any(np.isfinite(beh_r2)):
        valid = np.isfinite(beh_r2)
        ax.plot(ep[valid], beh_r2[valid], color=_COL_CV, lw=2, marker="o",
                markersize=4)
        _style(ax, xlabel="Epoch", ylabel="R²", title="Behaviour R² (eval)")
    else:
        ax.text(0.5, 0.5, "No eval R²", ha="center", va="center",
                fontsize=13, transform=ax.transAxes, color="0.5")
        _style(ax, xlabel="", ylabel="", title="Behaviour R²")

    fig.suptitle("Parameter Trajectories", fontsize=24, fontweight="bold", y=1.01)
    _save_png(fig, save_dir, "param_trajectories.png",
              dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_traces(
    data: Dict[str, Any],
    onestep: Dict[str, Any],
    loo: Dict[str, Any],
    free_run: Dict[str, Any],
    save_dir: str,
):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    u_np = data["u_stage1"].cpu().numpy()
    T, N = u_np.shape
    dt = data["dt"]
    time = np.arange(T) * dt
    labels = _get_labels(data)

    r2_os = onestep["r2"]
    mu_os = onestep["prior_mu"]
    r2_loo = loo["r2"]
    preds_loo = loo["pred"]
    r2_fr = free_run["r2"]
    u_free = free_run["u_free"]

    pick = sorted(preds_loo.keys())
    if len(pick) == 0:
        return

    fig = plt.figure(figsize=(20, 3.5 * len(pick) + 5))
    gs = GridSpec(len(pick) + 1, 2, width_ratios=[1.5, 5],
                  height_ratios=[1] * len(pick) + [1.3],
                  hspace=0.35, wspace=0.20)

    for row, idx in enumerate(pick):
        ax_net = fig.add_subplot(gs[row, 0])
        _plot_partner_panel(ax_net, data, labels, idx)

        ax = fig.add_subplot(gs[row, 1])
        ax.plot(time, u_np[:, idx], color=_COL_DATA,
                lw=2.0, alpha=0.45, label="Data")
        ax.plot(time, mu_os[:, idx], color=_COL_RAW,
                lw=1.5, ls="-", alpha=0.85,
                label=f"One-step (R\u00b2={r2_os[idx]:.3f})")
        if idx in preds_loo:
            ax.plot(time, preds_loo[idx], color=_COL_CV,
                    lw=2.2, ls="--", alpha=0.9,
                    label=f"LOO (R\u00b2={r2_loo[idx]:.3f})")
        if np.isfinite(r2_fr[idx]):
            ax.plot(time, u_free[:, idx], color=_COL_FR,
                    lw=1.8, ls="-.", alpha=0.85,
                    label=f"Free-run (R\u00b2={r2_fr[idx]:.3f})")
        lbl = _label_for_idx(labels, idx)
        _style(ax, ylabel=lbl)
        if row == 0:
            ax.legend(loc="upper right", frameon=False, fontsize=15, ncol=2)
        if row < len(pick) - 1:
            ax.set_xticklabels([])
    _style(ax, xlabel="Time (s)")

    gs_bot = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[-1, :], wspace=0.30)
    for i, (r2, color, label) in enumerate([
        (r2_os, _COL_RAW, "One-step"),
        (r2_loo, _COL_CV, "LOO"),
        (r2_fr, _COL_FR, "Free-run"),
    ]):
        ax = fig.add_subplot(gs_bot[0, i])
        valid = r2[np.isfinite(r2)]
        if len(valid):
            bins = np.linspace(min(valid.min(), -0.2),
                               max(valid.max(), 1.0), 30)
            ax.hist(valid, bins=bins, color=color, alpha=0.7,
                    edgecolor="white")
            ax.axvline(np.median(valid), color="0.2", lw=1.5, ls="--",
                       label=f"med={np.median(valid):.3f}")
            ax.legend(frameon=False, fontsize=14)
        _style(ax, xlabel=f"{label} R\u00b2", ylabel="Count")

    fig.suptitle("Prediction Quality: LOO-Tested Neurons",
                 fontsize=24, fontweight="bold", y=1.01)
    _save_png(fig, save_dir, "02_prediction_traces.png", bbox_inches="tight")
    plt.close(fig)
def plot_free_run_statistics(
    data: Dict[str, Any],
    free_run: Dict[str, Any],
    save_dir: str,
):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    u_np = data["u_stage1"].cpu().numpy()
    u_free = free_run["u_free"]
    r2 = free_run["r2"]
    mode = str(free_run.get("mode", "autonomous"))
    T, N = u_np.shape
    dt = data["dt"]
    time = np.arange(T) * dt
    labels = _get_labels(data)

    finite_mask = np.isfinite(r2)
    if not np.any(finite_mask):
        return
    r2_finite = r2[finite_mask]
    idx_finite = np.where(finite_mask)[0]
    order = np.argsort(r2_finite)[::-1]

    idx_best = idx_finite[order[0]]
    idx_median = idx_finite[order[len(order) // 2]]
    idx_worst = idx_finite[order[-1]]
    picks = [("Best", idx_best), ("Median", idx_median), ("Worst", idx_worst)]

    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 2, hspace=0.38, wspace=0.30)

    ax_traces = fig.add_subplot(gs[0, :])
    _trace_cols = [_COL_RAW, _COL_CV, _COL_DATA]
    for i, (tag, idx) in enumerate(picks):
        lbl = labels[idx] if labels is not None else f"Neuron {idx}"
        offset = i * 6
        ax_traces.plot(time, u_np[:, idx] + offset,
                       color=_COL_DATA, lw=0.8, alpha=0.7)
        ax_traces.plot(time, u_free[:, idx] + offset,
                       color=_trace_cols[i], ls="-", lw=0.8, alpha=0.8,
                       label=f"{tag}: {lbl} (R\u00b2={r2[idx]:.3f})")
    ax_traces.legend(loc="upper right", frameon=False, fontsize=15, ncol=1)
    trace_title = ("A  Motor-neuron traces conditioned on the rest"
                   if mode == "motor_conditioned"
                   else "A  Free-run example traces")
    _style(ax_traces, xlabel="Time (s)", ylabel="Activity (offset)",
           title=trace_title)

    ax_hist = fig.add_subplot(gs[1, 0])
    bins = np.linspace(min(-1, np.nanmin(r2_finite)),
                       max(1, np.nanmax(r2_finite)), 40)
    ax_hist.hist(r2_finite, bins=bins, color=_COL_CV, alpha=0.7,
                 edgecolor="white", label=f"median = {np.nanmedian(r2_finite):.3f}")
    ax_hist.axvline(np.nanmedian(r2_finite), color=_COL_RAW, lw=2.5, ls="--")
    ax_hist.axvline(0, color="gray", lw=1, ls=":")
    ax_hist.legend(frameon=False, fontsize=13)
    _style(ax_hist, xlabel="R\u00b2", ylabel="Count",
           title="B  R\u00b2 distribution")

    ax_rmse = fig.add_subplot(gs[1, 1])
    err2 = (u_np - u_free) ** 2
    rmse_t = np.sqrt(np.nanmean(err2, axis=1))
    win = min(100, T // 10)
    corr_t = np.full(T, np.nan)
    for t in range(win, T):
        seg_true = u_np[t - win:t].ravel()
        seg_free = u_free[t - win:t].ravel()
        m = np.isfinite(seg_true) & np.isfinite(seg_free)
        if m.sum() > 10:
            corr_t[t] = _pearson(seg_true[m], seg_free[m])
    ax_rmse_twin = ax_rmse.twinx()
    ax_rmse.plot(time, rmse_t, color=_COL_RAW, ls="-", lw=1.5, alpha=0.8,
                 label="RMSE")
    ax_rmse_twin.plot(time[win:], corr_t[win:], color=_COL_DATA, ls="-",
                      lw=1.2, alpha=0.6, label="Windowed corr.")
    ax_rmse.set_xlabel("")
    _style(ax_rmse, xlabel="Time (s)", ylabel="RMSE",
           title="C  Error accumulation over time")
    ax_rmse_twin.set_ylabel("Correlation", fontsize=16)
    ax_rmse_twin.tick_params(labelsize=13)
    h1, l1 = ax_rmse.get_legend_handles_labels()
    h2, l2 = ax_rmse_twin.get_legend_handles_labels()
    ax_rmse.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False,
                   fontsize=15)

    ax_scatter = fig.add_subplot(gs[2, 0])
    var_per = np.nanvar(u_np, axis=0)
    ax_scatter.scatter(var_per[finite_mask], r2_finite, s=18, alpha=0.6,
                       c=_COL_CV, edgecolors="none")
    ax_scatter.axhline(0, color="gray", lw=0.8, ls=":")
    ax_scatter.set_xscale("log")
    _style(ax_scatter, xlabel="Neuron variance (log scale)",
           ylabel="Free-run R\u00b2", title="D  R\u00b2 vs signal variance")

    ax_bottom = fig.add_subplot(gs[2, 1])
    from numpy.fft import rfft, rfftfreq
    n_fft = T
    freqs = rfftfreq(n_fft, d=dt)
    ps_true = np.abs(rfft(u_np[:, idx_best]
                           - np.nanmean(u_np[:, idx_best]))) ** 2
    ps_free = np.abs(rfft(u_free[:, idx_best]
                           - np.nanmean(u_free[:, idx_best]))) ** 2
    k_smooth = max(1, len(freqs) // 100)
    ps_true_s = _smooth_1d(ps_true, k_smooth)
    ps_free_s = _smooth_1d(ps_free, k_smooth)
    lbl_best = (labels[idx_best] if labels is not None
                else f"Neuron {idx_best}")
    ax_bottom.loglog(freqs[1:], ps_true_s[1:], color=_COL_DATA, lw=1.5,
                     alpha=0.7, label="Data")
    ax_bottom.loglog(freqs[1:], ps_free_s[1:], color=_COL_CV, lw=1.5,
                     alpha=0.7, label="Free run")
    ax_bottom.legend(frameon=False, fontsize=15)
    _style(ax_bottom, xlabel="Frequency (Hz)", ylabel="Power",
           title=f"E  Power spectrum \u2014 {lbl_best}")

    fig.suptitle("Motor-Conditioned Rollout Diagnostics (CV-reg)"
                 if mode == "motor_conditioned"
                 else "Free-Run Diagnostics (CV-reg)",
                 fontsize=26, fontweight="bold", y=1.01)
    _save_png(fig, save_dir, "11b_free_run_statistics.png",
              bbox_inches="tight", dpi=150)
    plt.close(fig)


# =====================================================================
# Ridge-CV diagnostics (alpha, behaviour decoder, summary)
# =====================================================================

def _run_alpha_cv_for_diagnostics(
    model: Stage2ModelPT,
    data: Dict[str, Any],
    cfg,
) -> Optional[Dict[str, Any]]:
    """Re-run the alpha ridge-CV solver to collect per-neuron diagnostics."""
    if int(getattr(cfg, "alpha_cv_every", 0) or 0) <= 0:
        return None
    try:
        from .train import ridge_cv_solve_alpha
        result = ridge_cv_solve_alpha(model, data, cfg)
        return result
    except Exception as e:
        get_stage2_logger().warning("alpha_cv_diag_failed", error=str(e))
        return None


def plot_ridge_cv_diagnostics(
    model: Stage2ModelPT,
    data: Dict[str, Any],
    onestep: Dict[str, Any],
    beh_results: Optional[Dict[str, Any]],
    save_dir: str,
    cfg=None,
) -> None:
    """3×4 comprehensive ridge-CV diagnostics figure.

    Row 0 — Alpha CV (per-neuron kernel amplitudes):
        A. Sample CV-MSE curves   B. Best-λ histogram   C. Per-neuron fit R²   D. Coefficient heatmap

    Row 1 — Behaviour decoder CV (per eigenworm mode):
        E. GT decoder CV-MSE      F. Model decoder CV-MSE   G. Best-λ comparison   H. Decoder CV-MSE at best λ

    Row 2 — Cross-diagnostics:
        I. α-CV λ vs neuron variance   J. α-CV λ vs one-step R²   K. Boundary hit summary   L. Per-neuron α vs prediction error
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    u_np = data["u_stage1"].cpu().numpy()
    T, N = u_np.shape

    # ---- Run alpha ridge-CV to get diagnostics ----
    alpha_diag = _run_alpha_cv_for_diagnostics(model, data, cfg)

    # ---- Extract behaviour decoder ridge-CV diagnostics ----
    beh_cv_model = None
    beh_cv_gt = None
    if beh_results is not None:
        beh_cv_model = beh_results.get("ridge_cv_model")
        beh_cv_gt = beh_results.get("ridge_cv_gt")

    fig = plt.figure(figsize=(30, 20))
    gs = GridSpec(3, 4, hspace=0.42, wspace=0.38)

    # ================================================================
    # Row 0: Alpha CV (per-neuron kernel amplitudes)
    # ================================================================

    # ---- A. Sample CV-MSE curves for neurons --------------------------
    ax = fig.add_subplot(gs[0, 0])
    if alpha_diag is not None and alpha_diag.get("cv_mse_all") is not None:
        cv_mse_all = alpha_diag["cv_mse_all"]   # (N, n_grid)
        ridge_grid = alpha_diag["ridge_grid"]
        lambdas_a = alpha_diag["lambdas"]

        # Pick ~15 representative neurons: best, worst, and evenly spaced
        valid_neurons = np.where(np.isfinite(lambdas_a))[0]
        if len(valid_neurons) > 0:
            fit_r2_a = alpha_diag["fit_r2"]
            r2_valid = fit_r2_a[valid_neurons]
            order_r2 = valid_neurons[np.argsort(r2_valid)]
            # Sample: 5 worst, 5 middle, 5 best
            n_sample = min(15, len(order_r2))
            if n_sample <= 15:
                idx_sample = np.linspace(0, len(order_r2) - 1, n_sample, dtype=int)
            else:
                idx_sample = np.arange(n_sample)
            sample_neurons = order_r2[idx_sample]

            cmap = plt.cm.viridis(np.linspace(0, 1, len(sample_neurons)))
            for ci, ni in enumerate(sample_neurons):
                curve = cv_mse_all[ni]
                finite = np.isfinite(curve)
                if finite.sum() > 2:
                    ax.plot(ridge_grid[finite], curve[finite],
                            color=cmap[ci], alpha=0.6, lw=1.2)
                    # Mark best λ
                    best_idx = np.nanargmin(curve)
                    ax.plot(ridge_grid[best_idx], curve[best_idx],
                            "o", color=cmap[ci], markersize=4, alpha=0.8)
            ax.set_xscale("log")
            ax.set_yscale("log")
    else:
        ax.text(0.5, 0.5, "No alpha CV data\n(alpha_per_neuron off?)",
                ha="center", va="center", fontsize=13, transform=ax.transAxes)
    _style(ax, xlabel="Ridge λ", ylabel="CV-MSE",
           title="A. Alpha CV: MSE Curves (sample neurons)")

    # ---- B. Best-λ histogram across neurons ----------------------------
    ax = fig.add_subplot(gs[0, 1])
    if alpha_diag is not None:
        lambdas_a = alpha_diag["lambdas"]
        valid_lam = lambdas_a[np.isfinite(lambdas_a)]
        at_upper = alpha_diag.get("at_upper_flags", np.zeros(N, dtype=bool))
        if len(valid_lam) > 0:
            log_lam = np.log10(np.maximum(valid_lam, 1e-12))
            ax.hist(log_lam, bins=30, color=_COL_DATA, alpha=0.7,
                    edgecolor="white")
            ax.axvline(np.median(log_lam), color=_COL_RAW, lw=2, ls="--",
                       label=f"median={10**np.median(log_lam):.2g}")
            n_upper = int(at_upper.sum())
            n_total = int(np.isfinite(lambdas_a).sum())
            ax.text(0.97, 0.97,
                    f"{n_total} neurons solved\n{n_upper} at upper boundary",
                    transform=ax.transAxes, ha="right", va="top", fontsize=11,
                    bbox=dict(boxstyle="round", fc="#f7f7f7", alpha=0.85))
            ax.legend(frameon=False, fontsize=10)
    else:
        ax.text(0.5, 0.5, "No alpha CV", ha="center", va="center",
                fontsize=13, transform=ax.transAxes)
    _style(ax, xlabel=r"$\log_{10}(\lambda)$", ylabel="Count",
           title="B. Alpha CV: Best-\u03bb Distribution")

    # ---- C. Per-neuron fit R² -----------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    if alpha_diag is not None and alpha_diag.get("fit_r2") is not None:
        fit_r2_a = alpha_diag["fit_r2"]
        valid_r2 = np.isfinite(fit_r2_a)
        if valid_r2.sum() > 0:
            r2_vals = fit_r2_a[valid_r2]
            order = np.argsort(r2_vals)
            colors_r2 = np.where(r2_vals[order] >= 0, _COL_CV, _COL_RAW)
            ax.barh(np.arange(len(order)), r2_vals[order],
                    color=colors_r2, alpha=0.6, height=1.0, edgecolor="none")
            ax.axvline(0, color="0.3", lw=1)
            med_r2 = float(np.median(r2_vals))
            ax.axvline(med_r2, color=_COL_DATA, lw=1.5, ls="--")
            ax.text(0.97, 0.97,
                    f"median R²={med_r2:.3f}\n{int((r2_vals>0).sum())}/{len(r2_vals)} > 0",
                    transform=ax.transAxes, ha="right", va="top", fontsize=11,
                    bbox=dict(boxstyle="round", fc="#f7f7f7", alpha=0.85))
            ax.set_yticks([])
    else:
        ax.text(0.5, 0.5, "No alpha CV", ha="center", va="center",
                fontsize=13, transform=ax.transAxes)
    _style(ax, xlabel="R² (ridge fit → synaptic target)",
           ylabel="Neurons (sorted)",
           title="C. Alpha CV: Per-Neuron Fit Quality")

    # ---- D. Coefficient heatmap (neurons × ranks) --------------------
    ax = fig.add_subplot(gs[0, 3])
    if alpha_diag is not None:
        r_sv = model.r_sv
        r_dcv = model.r_dcv
        with torch.no_grad():
            a_sv_np = model.a_sv.cpu().numpy()
            a_dcv_np = model.a_dcv.cpu().numpy()
        if a_sv_np.ndim == 1:
            a_sv_np = np.tile(a_sv_np, (N, 1))
        if a_dcv_np.ndim == 1:
            a_dcv_np = np.tile(a_dcv_np, (N, 1))
        # Combine SV and DCV side by side
        combined = np.concatenate([a_sv_np, a_dcv_np], axis=1)  # (N, r_sv+r_dcv)
        # Sort neurons by total amplitude
        neuron_order = np.argsort(combined.sum(axis=1))[::-1]
        combined_sorted = combined[neuron_order]

        vmax = np.percentile(combined[combined > 0], 95) if (combined > 0).any() else 1.0
        im = ax.imshow(combined_sorted, aspect="auto", cmap="YlOrRd",
                       vmin=0, vmax=vmax, interpolation="nearest")
        ax.axvline(r_sv - 0.5, color="white", lw=2, ls="--")
        ax.set_xlabel("Rank", fontsize=16)
        ax.set_ylabel(f"Neurons (sorted, N={N})", fontsize=16)
        # Label ranks
        rank_labels = [f"SV{i}" for i in range(r_sv)] + [f"DCV{i}" for i in range(r_dcv)]
        ax.set_xticks(np.arange(len(rank_labels)))
        ax.set_xticklabels(rank_labels, fontsize=9, rotation=45)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="α")
    else:
        ax.text(0.5, 0.5, "No alpha data", ha="center", va="center",
                fontsize=13, transform=ax.transAxes)
    _style(ax, xlabel="", ylabel="", title="D. Alpha Coefficients (sorted)")

    # ================================================================
    # Row 1: Behaviour decoder CV
    # ================================================================

    def _plot_beh_cv_curves(ax, cv_dict, title_str):
        """Plot CV-MSE curves for each eigenworm mode."""
        if cv_dict is None:
            ax.text(0.5, 0.5, "No decoder data", ha="center", va="center",
                    fontsize=13, transform=ax.transAxes)
            _style(ax, title=title_str)
            return
        ridge_grid = cv_dict.get("ridge_grid")
        cv_mse = cv_dict.get("cv_mse_curves")
        best_lam = cv_dict.get("best_lambdas")
        if ridge_grid is None or cv_mse is None:
            ax.text(0.5, 0.5, "CV curves not stored", ha="center", va="center",
                    fontsize=13, transform=ax.transAxes)
            _style(ax, title=title_str)
            return
        L_b = cv_mse.shape[0]
        mode_colors = [_COL_DATA, _COL_RAW, _COL_CV, _COL_FR, "#ff7f0e",
                       "#8c564b", "#e377c2", "#7f7f7f"]
        for j in range(L_b):
            c = mode_colors[j % len(mode_colors)]
            finite = np.isfinite(cv_mse[j])
            if finite.sum() > 2:
                ax.plot(ridge_grid[finite], cv_mse[j, finite],
                        color=c, lw=2, alpha=0.8, label=f"EW{j+1}")
                if best_lam is not None:
                    best_idx = np.nanargmin(cv_mse[j])
                    ax.plot(ridge_grid[best_idx], cv_mse[j, best_idx],
                            "o", color=c, markersize=8, zorder=5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=9, ncol=2)
        _style(ax, xlabel="Ridge λ", ylabel="CV-MSE", title=title_str)

    # ---- E. GT decoder CV-MSE -----------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    _plot_beh_cv_curves(ax, beh_cv_gt, "E. GT Decoder CV-MSE")

    # ---- F. Model decoder CV-MSE --------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    _plot_beh_cv_curves(ax, beh_cv_model, "F. Model Decoder CV-MSE")

    # ---- G. Best-λ comparison: GT vs Model ----------------------------
    ax = fig.add_subplot(gs[1, 2])
    has_gt = beh_cv_gt is not None and beh_cv_gt.get("best_lambdas") is not None
    has_model = beh_cv_model is not None and beh_cv_model.get("best_lambdas") is not None
    if has_gt and has_model:
        lam_gt = np.asarray(beh_cv_gt["best_lambdas"])
        lam_mod = np.asarray(beh_cv_model["best_lambdas"])
        L_b = min(len(lam_gt), len(lam_mod))
        x = np.arange(L_b)
        w = 0.35
        ax.bar(x - w / 2, np.log10(np.maximum(lam_gt[:L_b], 1e-12)),
               w, label="GT", color=_COL_DATA, alpha=0.75)
        ax.bar(x + w / 2, np.log10(np.maximum(lam_mod[:L_b], 1e-12)),
               w, label="Model", color=_COL_CV, alpha=0.75)
        ax.set_xticks(x)
        ax.set_xticklabels([f"EW{i+1}" for i in range(L_b)], fontsize=11)
        ax.legend(frameon=False, fontsize=11)
        # Annotate boundary flags
        at_upper_gt = beh_cv_gt.get("at_upper")
        at_upper_mod = beh_cv_model.get("at_upper")
        n_up = 0
        if at_upper_gt is not None:
            n_up += int(np.sum(at_upper_gt[:L_b]))
        if at_upper_mod is not None:
            n_up += int(np.sum(at_upper_mod[:L_b]))
        if n_up > 0:
            ax.text(0.97, 0.03, f"{n_up} at upper bound",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=10, color=_COL_RAW)
    elif has_model:
        lam_mod = np.asarray(beh_cv_model["best_lambdas"])
        L_b = len(lam_mod)
        x = np.arange(L_b)
        ax.bar(x, np.log10(np.maximum(lam_mod, 1e-12)),
               0.5, label="Model", color=_COL_CV, alpha=0.75)
        ax.set_xticks(x)
        ax.set_xticklabels([f"EW{i+1}" for i in range(L_b)], fontsize=11)
        ax.legend(frameon=False, fontsize=11)
    else:
        ax.text(0.5, 0.5, "No decoder data", ha="center", va="center",
                fontsize=13, transform=ax.transAxes)
    _style(ax, xlabel="Eigenworm mode", ylabel=r"$\log_{10}$(Best \u03bb)",
           title="G. Behaviour Decoder: Best-\u03bb")

    # ---- H. Decoder CV-MSE at Best λ (per mode) ---------------------
    ax = fig.add_subplot(gs[1, 3])
    has_gt_mse = beh_cv_gt is not None and beh_cv_gt.get("cv_mse_curves") is not None
    has_mod_mse = beh_cv_model is not None and beh_cv_model.get("cv_mse_curves") is not None
    if has_gt_mse or has_mod_mse:
        gt_curves = beh_cv_gt["cv_mse_curves"] if has_gt_mse else None
        mod_curves = beh_cv_model["cv_mse_curves"] if has_mod_mse else None
        n_modes = 0
        if gt_curves is not None:
            n_modes = max(n_modes, gt_curves.shape[0])
        if mod_curves is not None:
            n_modes = max(n_modes, mod_curves.shape[0])
        n_modes = min(n_modes, 6)
        x = np.arange(n_modes)
        w = 0.35
        if has_gt_mse:
            min_mse_gt = np.array([np.nanmin(gt_curves[j]) for j in range(n_modes)])
            ax.bar(x - w / 2, min_mse_gt, w, label="GT decoder",
                   color=_COL_DATA, alpha=0.75)
        if has_mod_mse:
            min_mse_mod = np.array([np.nanmin(mod_curves[j]) for j in range(n_modes)])
            offset = w / 2 if has_gt_mse else 0
            ax.bar(x + offset, min_mse_mod, w, label="Model decoder",
                   color=_COL_CV, alpha=0.75)
        ax.set_xticks(x)
        ax.set_xticklabels([f"EW{i+1}" for i in range(n_modes)], fontsize=11)
        ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=10)
    else:
        ax.text(0.5, 0.5, "No decoder CV data", ha="center", va="center",
                fontsize=13, transform=ax.transAxes)
    _style(ax, xlabel="Eigenworm mode", ylabel="Min CV-MSE",
           title="H. Decoder CV-MSE at Best \u03bb")

    # ================================================================
    # Row 2: Cross-diagnostics
    # ================================================================

    # ---- I. Alpha-CV λ vs neuron signal variance ----------------------
    ax = fig.add_subplot(gs[2, 0])
    if alpha_diag is not None:
        lambdas_a = alpha_diag["lambdas"]
        valid_lam = np.isfinite(lambdas_a)
        var_per = np.nanvar(u_np, axis=0)
        if valid_lam.sum() > 0 and (var_per[valid_lam] > 0).any():
            ax.scatter(var_per[valid_lam],
                       lambdas_a[valid_lam],
                       s=20, alpha=0.5, c=_COL_DATA, edgecolors="none")
            ax.set_xscale("log")
            ax.set_yscale("log")
            # Spearman correlation
            from scipy.stats import spearmanr
            rho, pval = spearmanr(var_per[valid_lam], lambdas_a[valid_lam])
            ax.text(0.03, 0.97, f"ρ={rho:.2f}, p={pval:.2g}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=11,
                    bbox=dict(boxstyle="round", fc="#f7f7f7", alpha=0.85))
    else:
        ax.text(0.5, 0.5, "No alpha CV", ha="center", va="center",
                fontsize=13, transform=ax.transAxes)
    _style(ax, xlabel="Neuron variance", ylabel="Best λ (alpha CV)",
           title="I. Neuron Variance vs Alpha-CV λ")

    # ---- J. Alpha-CV λ vs one-step R² --------------------------------
    ax = fig.add_subplot(gs[2, 1])
    r2_os = onestep["r2"]
    if alpha_diag is not None:
        lambdas_a = alpha_diag["lambdas"]
        valid_both = np.isfinite(lambdas_a) & np.isfinite(r2_os)
        if valid_both.sum() > 0:
            ax.scatter(lambdas_a[valid_both], r2_os[valid_both],
                       s=20, alpha=0.5, c=_COL_CV, edgecolors="none")
            ax.set_xscale("log")
            ax.axhline(0, color="gray", lw=0.6, ls=":")
            from scipy.stats import spearmanr
            rho, pval = spearmanr(lambdas_a[valid_both], r2_os[valid_both])
            ax.text(0.03, 0.97, f"ρ={rho:.2f}, p={pval:.2g}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=11,
                    bbox=dict(boxstyle="round", fc="#f7f7f7", alpha=0.85))
    else:
        ax.text(0.5, 0.5, "No alpha CV", ha="center", va="center",
                fontsize=13, transform=ax.transAxes)
    _style(ax, xlabel="Best λ (alpha CV)", ylabel="One-step R²",
           title="J. Alpha-CV λ vs Model Fit")

    # ---- K. Boundary hit summary (stacked bar) -----------------------
    ax = fig.add_subplot(gs[2, 2])
    cv_types = []
    n_interior_list, n_upper_list, n_zero_list = [], [], []

    if alpha_diag is not None:
        lam_a = alpha_diag["lambdas"]
        at_up_a = alpha_diag.get("at_upper_flags", np.zeros(N, dtype=bool))
        n_solved = int(np.isfinite(lam_a).sum())
        n_up = int(at_up_a.sum())
        # Check at lower boundary (λ = smallest grid value)
        rg = alpha_diag.get("ridge_grid")
        n_lo = 0
        if rg is not None and n_solved > 0:
            lo_thresh = float(rg[0]) * 1.01
            n_lo = int((lam_a[np.isfinite(lam_a)] <= lo_thresh).sum())
        cv_types.append("Alpha\n(per-neuron)")
        n_zero_list.append(n_lo)
        n_upper_list.append(n_up)
        n_interior_list.append(n_solved - n_up - n_lo)

    if beh_cv_gt is not None and beh_cv_gt.get("at_zero") is not None:
        az = np.asarray(beh_cv_gt["at_zero"])
        au = np.asarray(beh_cv_gt["at_upper"])
        n_t = len(az)
        cv_types.append("GT decoder\n(per-mode)")
        n_zero_list.append(int(az.sum()))
        n_upper_list.append(int(au.sum()))
        n_interior_list.append(n_t - int(az.sum()) - int(au.sum()))

    if beh_cv_model is not None and beh_cv_model.get("at_zero") is not None:
        az = np.asarray(beh_cv_model["at_zero"])
        au = np.asarray(beh_cv_model["at_upper"])
        n_t = len(az)
        cv_types.append("Model decoder\n(per-mode)")
        n_zero_list.append(int(az.sum()))
        n_upper_list.append(int(au.sum()))
        n_interior_list.append(n_t - int(az.sum()) - int(au.sum()))

    if cv_types:
        x = np.arange(len(cv_types))
        w = 0.5
        ax.bar(x, n_interior_list, w, label="Interior", color=_COL_CV, alpha=0.7)
        ax.bar(x, n_upper_list, w, bottom=n_interior_list,
               label="At upper", color=_COL_RAW, alpha=0.7)
        bottoms = [a + b for a, b in zip(n_interior_list, n_upper_list)]
        ax.bar(x, n_zero_list, w, bottom=bottoms,
               label="At lower/zero", color="0.65", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(cv_types, fontsize=10)
        ax.legend(frameon=False, fontsize=10)
    else:
        ax.text(0.5, 0.5, "No CV data", ha="center", va="center",
                fontsize=13, transform=ax.transAxes)
    _style(ax, xlabel="", ylabel="Count",
           title="K. Boundary Hits (all CV types)")

    # ---- L. Per-neuron α sum vs one-step RMSE -----------------------
    ax = fig.add_subplot(gs[2, 3])
    if alpha_diag is not None:
        with torch.no_grad():
            a_sv_pn = model.a_sv.cpu().numpy()
            a_dcv_pn = model.a_dcv.cpu().numpy()
        if a_sv_pn.ndim == 1:
            a_sv_pn = np.tile(a_sv_pn, (N, 1))
        if a_dcv_pn.ndim == 1:
            a_dcv_pn = np.tile(a_dcv_pn, (N, 1))
        alpha_total = a_sv_pn.sum(axis=1) + a_dcv_pn.sum(axis=1)
        rmse_os = onestep.get("rmse", np.full(N, np.nan))
        valid_both = (np.isfinite(alpha_total) & np.isfinite(rmse_os)
                      & (alpha_total > 0))
        if valid_both.sum() > 2:
            ax.scatter(alpha_total[valid_both], rmse_os[valid_both],
                       s=20, alpha=0.5, c=_COL_FR, edgecolors="none")
            ax.set_xscale("log")
            ax.set_yscale("log")
            from scipy.stats import spearmanr
            rho, pval = spearmanr(alpha_total[valid_both],
                                  rmse_os[valid_both])
            ax.text(0.03, 0.97, f"\u03c1={rho:.2f}, p={pval:.2g}",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=11,
                    bbox=dict(boxstyle="round", fc="#f7f7f7", alpha=0.85))
        else:
            ax.text(0.5, 0.5, "Insufficient data", ha="center",
                    va="center", fontsize=13, transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "No alpha CV", ha="center", va="center",
                fontsize=13, transform=ax.transAxes)
    _style(ax, xlabel="Total \u03b1 (SV + DCV)", ylabel="One-step RMSE",
           title="L. Per-Neuron \u03b1 vs Prediction Error")

    fig.suptitle("Ridge-CV Diagnostics", fontsize=28,
                 fontweight="bold", y=1.01)
    _save_png(fig, save_dir, "ridge_cv_diagnostics.png",
              dpi=150, bbox_inches="tight")
    plt.close(fig)
def generate_eval_loo_plots(
    model,
    data: Dict[str, Any],
    cfg,
    epoch_losses: list,
    save_dir: str,
    show: bool = False,
    decoder=None,
    e2e_decoder=None,
    beh_all_baseline=None,
    skip_beh_all: bool = False,
) -> Dict[str, Any]:
    """Run full evaluation, then render the three diagnostic figures."""
    import matplotlib

    if not show:
        matplotlib.use("Agg")

    setup_plot_style()
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    results = run_full_evaluation(model, data, cfg, decoder=decoder,
                                  e2e_decoder=e2e_decoder,
                                  beh_all_baseline=beh_all_baseline,
                                  skip_beh_all=skip_beh_all)

    onestep  = results["onestep"]
    loo      = results["loo"]
    free_run = results["free_run"]
    beh_frozen = results.get("beh")          # frozen motor baseline (has ridge-CV data)
    beh_e2e    = results.get("beh_e2e")      # E2E decoder (if active)
    beh        = beh_e2e or beh_frozen        # primary for model-prediction panels
    beh_all    = results.get("beh_all")

    plot_summary_slide(model, data, onestep, loo, free_run, beh,
                       save_dir, epoch_losses=epoch_losses,
                       beh_all=beh_all, beh_frozen=beh_frozen)
    plot_parameter_trajectories(epoch_losses, save_dir, cfg=cfg)
    plot_prediction_traces(data, onestep, loo, free_run, save_dir)
    plot_free_run_statistics(data, free_run, save_dir)

    # Ridge-CV diagnostics — use frozen decoder for ridge-CV curves
    plot_ridge_cv_diagnostics(model, data, onestep, beh_frozen, save_dir, cfg=cfg)

    return {
        "beh": beh,
        "beh_r2_model": results.get("beh_r2_model"),
    }
