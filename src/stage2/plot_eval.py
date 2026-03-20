from __future__ import annotations

import numpy as np
import torch
import h5py
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import matplotlib.colors as mcolors

from .model import Stage2ModelPT
from .evaluate import _pearson, _r2, run_full_evaluation
from .worm_state import WormState
from . import get_stage2_logger

__all__ = [
    "generate_eval_loo_plots",
    "run_full_evaluation",
    "generate_multi_worm_plots",
    "plot_r2_three_metrics",
]

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


def _compute_power_spectrum(signal: np.ndarray, dt: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    arr = np.asarray(signal, dtype=float).reshape(-1)
    if arr.size < 4:
        return None, None
    finite = np.isfinite(arr)
    if finite.sum() < 4:
        return None, None
    if not finite.all():
        idx = np.arange(arr.size)
        arr = arr.copy()
        arr[~finite] = np.interp(idx[~finite], idx[finite], arr[finite])
    arr = arr - np.mean(arr)
    from numpy.fft import rfft, rfftfreq
    freqs = rfftfreq(arr.size, d=dt)
    power = np.abs(rfft(arr)) ** 2
    k_smooth = max(1, len(freqs) // 100)
    return freqs, _smooth_1d(power, k_smooth)


def _compute_input_decomposition(
    model: Stage2ModelPT,
    data: Dict[str, Any],
) -> Tuple[Dict[str, float], np.ndarray]:
    device = next(model.parameters()).device
    u = data["u_stage1"].to(device)
    T, N = u.shape
    gating = data.get("gating")
    stim = data.get("stim")

    with torch.no_grad():
        lam = model.lambda_u
        ar1 = (1.0 - lam) * u[:-1] + lam * model.I0
        target_resid = u[1:] - ar1  # what I_gap + I_sv + I_dcv + I_stim should explain

        L = model.laplacian()
        gap_all = lam * (L @ u[:-1].T).T

        g_default = torch.ones(N, device=device) if gating is None else None
        I_zero = torch.zeros(N, device=device)
        has_stim = model.d_ell > 0 and stim is not None

        s_sv = torch.zeros(N, model.r_sv, device=device)
        s_dcv = torch.zeros(N, model.r_dcv, device=device)
        sv_all = torch.zeros(T - 1, N, device=device)
        dcv_all = torch.zeros(T - 1, N, device=device)
        stim_all = torch.zeros(T - 1, N, device=device)
        pred_all = torch.zeros(T - 1, N, device=device)

        for t in range(T - 1):
            g = gating[t] if gating is not None else g_default
            phi_gated = model.phi(u[t]) * g

            I_sv = I_dcv = I_zero
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

            I_stim = I_zero
            if has_stim:
                I_stim = (model.b * stim[t].view(N)) if model.stim_diagonal_only else (model.b @ stim[t])

            sv_all[t] = lam * I_sv
            dcv_all[t] = lam * I_dcv
            stim_all[t] = lam * I_stim
            pred_all[t] = ar1[t] + gap_all[t] + sv_all[t] + dcv_all[t] + stim_all[t]

        unexplained = u[1:] - pred_all

        def _rms(x: torch.Tensor) -> float:
            return float(x.pow(2).mean().sqrt())

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
    }
    stim_rms = _rms(stim_all)
    if stim_rms > 1e-12:
        summary["$\\lambda I_{stim}$"] = stim_rms
    summary["Unexplained"] = _rms(unexplained)
    return summary, per_neuron


def _residual_acf(residuals: np.ndarray, max_lag: int = 20) -> np.ndarray:
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
):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    r2_os = onestep["r2"]
    r2_loo = loo["r2"]
    r2_fr = free_run["r2"]
    dt = float(data["dt"])
    u_np = data["u_stage1"].cpu().numpy()
    T, N = u_np.shape

    fig = plt.figure(figsize=(24, 14))
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
                       label=f"{np.median(valid):.3f}")
            ax.legend(frameon=False, fontsize=11)
        _style(ax, xlabel=label, ylabel="Count", title=title)

    ax = fig.add_subplot(gs[1, 0])
    if beh_results is not None:
        # All R² values are held-out / cross-validated (out-of-fold).

        # Red = ridge decoder on GT motor neurons (held-out CV)
        r2_motor = beh_results.get("r2_gt", np.full(6, np.nan))

        # Orange = AR(1) baseline on motor neurons (held-out CV)
        r2_ar1 = beh_results.get("r2_ar1", np.full(6, np.nan))

        # Green = ridge decoder on model-predicted motor neurons (held-out CV)
        r2_model = beh_results.get("r2_model", np.full(6, np.nan))

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
        ax.bar(x + offsets[bi], r2_model[:n_modes], w, label="Model (CV)",
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

    g_default = torch.ones(N, device=device) if gating is None else None
    with torch.no_grad():
        s_sv_all = torch.zeros(T, N, model.r_sv, device=device)
        s_dcv_all = torch.zeros(T, N, model.r_dcv, device=device)
        s_sv_w = torch.zeros(N, model.r_sv, device=device)
        s_dcv_w = torch.zeros(N, model.r_dcv, device=device)
        for t in range(T):
            s_sv_all[t], s_dcv_all[t] = s_sv_w, s_dcv_w
            g = gating[t] if gating is not None else g_default
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
                         else g_default)
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

    # G-H. Kernel amplitudes – one panel per synapse type (SV / DCV)
    with torch.no_grad():
        a_sv_arr = model.a_sv.detach().cpu().numpy() if model.r_sv > 0 else np.empty((0, 0))
        a_dcv_arr = model.a_dcv.detach().cpu().numpy() if model.r_dcv > 0 else np.empty((0, 0))
        tau_sv_arr = model.tau_sv.detach().cpu().numpy() if model.r_sv > 0 else np.array([])
        tau_dcv_arr = model.tau_dcv.detach().cpu().numpy() if model.r_dcv > 0 else np.array([])
    if a_sv_arr.ndim == 1:
        a_sv_arr = a_sv_arr[np.newaxis, :]
    if a_dcv_arr.ndim == 1:
        a_dcv_arr = a_dcv_arr[np.newaxis, :]

    gs_gh = gs[1, 2:].subgridspec(2, 1, hspace=0.50)
    for panel_i, (a_arr, tau_arr, syn_label, syn_col) in enumerate([
        (a_sv_arr, tau_sv_arr, "SV", _COL_DATA),
        (a_dcv_arr, tau_dcv_arr, "DCV", _COL_CV),
    ]):
        ax = fig.add_subplot(gs_gh[panel_i, 0])
        r = a_arr.shape[1] if a_arr.size else 0
        if r == 0:
            ax.text(0.5, 0.5, f"No {syn_label} ranks",
                    ha="center", va="center", transform=ax.transAxes, fontsize=12)
            ax.set_axis_off()
            continue
        box_data, tick_labels = [], []
        for ri in range(r):
            col = a_arr[:, ri]
            col = col[np.isfinite(col)]
            box_data.append(col if col.size > 0 else np.array([0.0]))
            tau_r = float(tau_arr[ri]) if ri < len(tau_arr) else 0
            med = float(np.median(col)) if col.size > 0 else 0
            tick_labels.append(f"τ={tau_r:.1f}s\n{med:.2e}")
        for ri, col in enumerate(box_data, start=1):
            if col.size == 0:
                continue
            jitter = np.linspace(-0.16, 0.16, col.size) if col.size > 1 else np.array([0.0])
            ax.scatter(
                ri + jitter,
                col,
                s=14,
                color=syn_col,
                alpha=0.28,
                linewidths=0,
                zorder=2,
            )
            q25, med, q75 = np.percentile(col, [25, 50, 75])
            ax.vlines(ri, q25, q75, color="0.15", lw=4.0, zorder=3)

        finite_positive = np.concatenate([col[col > 0] for col in box_data if np.any(col > 0)]) if box_data else np.array([])
        if finite_positive.size > 0:
            y_lo = float(np.min(finite_positive))
            y_hi = float(np.max(finite_positive))
            if y_hi > y_lo:
                ax.set_yscale("log")
                ax.set_ylim(y_lo / 1.35, y_hi * 1.35)
            else:
                pad = y_hi * 0.25 if y_hi > 0 else 1e-6
                ax.set_ylim(max(y_lo - pad, 1e-12), y_hi + pad)
        ax.grid(axis="y", alpha=0.18, lw=0.8)
        ax.set_xticks(range(1, r + 1))
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.tick_params(axis="y", labelsize=10)
        ylabel = "α (log)" if finite_positive.size > 0 and np.max(finite_positive) > np.min(finite_positive) else "α"
        title_str = "G. Kernel Amplitudes" if panel_i == 0 else ""
        _style(ax, xlabel="", ylabel=f"{syn_label}  {ylabel}", title=title_str)

    with torch.no_grad():
        lam = model.lambda_u.cpu().numpy().ravel()
        G_raw = model.G.cpu().numpy()
        G_val = G_raw[model.T_e.cpu().numpy() > 0].mean() if G_raw.ndim > 0 and G_raw.size > 1 else float(G_raw)

    # --- Row 2: Dynamics diagnostics -----------------------------------

    # I. Input decomposition
    ax = fig.add_subplot(gs[2, 0])
    decomp, per_neuron_rms = _compute_input_decomposition(model, data)
    labels_d = list(decomp.keys())
    vals_d = [decomp[k] for k in labels_d]
    colours_d = [_COL_RAW, _COL_DATA, _COL_CV, _COL_FR, "#ff7f0e", "0.55"]
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

    # J. Network strength trajectories over epochs
    ax = fig.add_subplot(gs[2, 1])
    if epoch_losses and "G" in epoch_losses[0]:
        ep_x = np.arange(1, len(epoch_losses) + 1)
        G_hist = np.asarray([e.get("G", np.nan) for e in epoch_losses], dtype=float)
        a_sv_hist = np.asarray([e.get("a_sv_rms", np.nan) for e in epoch_losses], dtype=float)
        a_dcv_hist = np.asarray([e.get("a_dcv_rms", np.nan) for e in epoch_losses], dtype=float)

        if np.any(np.isfinite(G_hist) & (G_hist > 0)):
            ax.plot(ep_x, G_hist, color=_COL_RAW, lw=2.4, label="G")
        if np.any(np.isfinite(a_sv_hist) & (a_sv_hist > 0)):
            ax.plot(ep_x, a_sv_hist, color=_COL_DATA, lw=2.2, label="α_sv RMS")
        if np.any(np.isfinite(a_dcv_hist) & (a_dcv_hist > 0)):
            ax.plot(ep_x, a_dcv_hist, color=_COL_CV, lw=2.2, label="α_dcv RMS")

        all_vals = np.concatenate([G_hist, a_sv_hist, a_dcv_hist])
        positive_vals = all_vals[np.isfinite(all_vals) & (all_vals > 0)]
        if positive_vals.size > 0:
            ax.set_yscale("log")
            ax.set_ylim(positive_vals.min() / 1.5, positive_vals.max() * 1.5)
        ax.grid(axis="y", alpha=0.18, lw=0.8)
        ax.legend(frameon=False, fontsize=9, loc="best")
    else:
        ax.text(0.5, 0.5, "No network history",
                ha="center", va="center", fontsize=14,
                transform=ax.transAxes)
    _style(ax, xlabel="Epoch", ylabel="Strength (log)",
           title="J. Network Strength Trajectories")

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

    a_sv_flat = a_sv_arr.ravel(); a_sv_flat = a_sv_flat[np.isfinite(a_sv_flat)]
    a_dcv_flat = a_dcv_arr.ravel(); a_dcv_flat = a_dcv_flat[np.isfinite(a_dcv_flat)]
    a_sv_med = np.nanmedian(a_sv_flat) if a_sv_flat.size > 0 else float("nan")
    a_dcv_med = np.nanmedian(a_dcv_flat) if a_dcv_flat.size > 0 else float("nan")

    G_txt = (f"{float(G_val):.6g}" if np.ndim(G_val) == 0
             else f"mean={np.mean(G_val):.4f}")
    n_loo = int(np.isfinite(r2_loo).sum())
    n_ep = len(epoch_losses)
    fr_med = np.nanmedian(r2_fr[np.isfinite(r2_fr)]) if np.any(np.isfinite(r2_fr)) else float("nan")
    alpha_sv_txt = f"  α_sv={a_sv_med:.3g}" if np.isfinite(a_sv_med) else ""
    alpha_dcv_txt = f"  α_dcv={a_dcv_med:.3g}" if np.isfinite(a_dcv_med) else ""
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

    n_rows, n_cols = 2, 4
    fig = plt.figure(figsize=(24, 9.5))
    gs = GridSpec(n_rows, n_cols, hspace=0.45, wspace=0.35)

    # --- (0,0) Gap-junction conductance G ---
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(ep, _get("G"), color=_COL_DATA, lw=2)
    ax.axhline(_get("G")[0], color="0.6", lw=1, ls=":", label="init")
    ax.legend(frameon=False, fontsize=9)
    _style(ax, xlabel="Epoch", ylabel="G (mean)", title="Gap-Junction G")

    # --- (0,1) Synaptic weights W ---
    ax = fig.add_subplot(gs[0, 1])
    w_sv_min = _get("W_sv_min")
    w_sv_max = _get("W_sv_max")
    w_sv_med = _get("W_sv_med")
    w_dcv_min = _get("W_dcv_min")
    w_dcv_max = _get("W_dcv_max")
    w_dcv_med = _get("W_dcv_med")
    if np.any(np.isfinite(w_sv_med)):
        ax.fill_between(ep, w_sv_min, w_sv_max, color=_COL_DATA, alpha=0.15,
                        label=r"$W_{sv}$ min–max")
        ax.plot(ep, w_sv_med, color=_COL_DATA, lw=2, label=r"$W_{sv}$ median")
    if np.any(np.isfinite(w_dcv_med)):
        ax.fill_between(ep, w_dcv_min, w_dcv_max, color=_COL_CV, alpha=0.15,
                        label=r"$W_{dcv}$ min–max")
        ax.plot(ep, w_dcv_med, color=_COL_CV, lw=2, label=r"$W_{dcv}$ median")
    ax.legend(frameon=False, fontsize=9)
    _style(ax, xlabel="Epoch", ylabel="Weight", title="Synaptic Weights W")

    # --- (0,2) λ_u statistics ---
    ax = fig.add_subplot(gs[0, 2])
    ax.fill_between(ep, _get("lambda_u_min"), _get("lambda_u_max"),
                    color=_COL_DATA, alpha=0.15, label="min–max")
    ax.plot(ep, _get("lambda_u_med"), color=_COL_DATA, lw=2, label="median")
    ax.plot(ep, _get("lambda_u_mean"), color=_COL_CV, lw=1.5, ls="--", label="mean")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False, fontsize=9)
    _style(ax, xlabel="Epoch", ylabel="λ_u", title="Leak Rate λ_u")

    # --- (0,3) Tonic drive I0 ---
    ax = fig.add_subplot(gs[0, 3])
    ax.fill_between(ep, _get("I0_min"), _get("I0_max"),
                    color=_COL_DATA, alpha=0.15, label="min–max")
    ax.plot(ep, _get("I0_med"), color=_COL_DATA, lw=2, label="median")
    ax.plot(ep, _get("I0_mean"), color=_COL_CV, lw=1.5, ls="--", label="mean")
    ax.legend(frameon=False, fontsize=9)
    _style(ax, xlabel="Epoch", ylabel=r"$I_0$", title=r"Tonic Drive $I_0$")

    # --- (1,0) Reversal potentials E_sv, E_dcv ---
    ax = fig.add_subplot(gs[1, 0])
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

    # --- (1,1) Time constants τ ---
    ax = fig.add_subplot(gs[1, 1])
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

    # --- (1,2) Stimulus weights b ---
    ax = fig.add_subplot(gs[1, 2])
    b_norm = _get("b_norm")
    if np.any(b_norm > 0):
        ax.plot(ep, b_norm, color=_COL_DATA, lw=2)
        _style(ax, xlabel="Epoch", ylabel="‖b‖", title="Stimulus Weights ‖b‖")
    else:
        ax.text(0.5, 0.5, "No stimulus", ha="center", va="center",
                fontsize=13, transform=ax.transAxes, color="0.5")
        _style(ax, xlabel="", ylabel="", title="Stimulus Weights")

    # --- (1,3) Behaviour R² over epochs (if available) ---
    ax = fig.add_subplot(gs[1, 3])
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
    from matplotlib.gridspec import GridSpec

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

    fig = plt.figure(figsize=(25, 3.6 * len(pick) + 1.5))
    gs = GridSpec(len(pick), 3, width_ratios=[1.35, 5.2, 2.15],
                  hspace=0.35, wspace=0.20)

    for row, idx in enumerate(pick):
        ax_net = fig.add_subplot(gs[row, 0])
        _plot_partner_panel(ax_net, data, labels, idx)

        ax = fig.add_subplot(gs[row, 1])
        ax.plot(time, u_np[:, idx], color=_COL_DATA,
                lw=2.0, alpha=0.45, label="Data", zorder=2)
        ax.plot(time, mu_os[:, idx], color=_COL_RAW,
                lw=1.5, ls="-", alpha=0.85,
                label=f"One-step (R\u00b2={r2_os[idx]:.3f})", zorder=3)
        if idx in preds_loo:
            ax.plot(time, preds_loo[idx], color=_COL_CV,
                    lw=2.2, ls="--", alpha=0.9,
                    label=f"LOO (R\u00b2={r2_loo[idx]:.3f})", zorder=4)
        if np.isfinite(r2_fr[idx]):
            ax.plot(time, u_free[:, idx], color=_COL_FR,
                    lw=1.8, ls="-.", alpha=0.85,
                    label=f"Free-run (R\u00b2={r2_fr[idx]:.3f})", zorder=5)
        lbl = _label_for_idx(labels, idx)
        _style(ax, ylabel=lbl)
        ax.legend(loc="upper right", frameon=False, fontsize=13, ncol=2)
        if row < len(pick) - 1:
            ax.set_xticklabels([])
        ax_ps = fig.add_subplot(gs[row, 2])
        spectra = [
            (u_np[:, idx], _COL_DATA, "Data"),
            (mu_os[:, idx], _COL_RAW, "One-step"),
        ]
        if idx in preds_loo:
            spectra.append((preds_loo[idx], _COL_CV, "LOO"))
        if np.isfinite(r2_fr[idx]):
            spectra.append((u_free[:, idx], _COL_FR, "Free-run"))
        any_ps = False
        for series, color, label in spectra:
            freqs, power = _compute_power_spectrum(series, dt)
            if freqs is None or power is None or len(freqs) <= 1:
                continue
            ax_ps.loglog(freqs[1:], np.maximum(power[1:], 1e-12),
                         color=color, lw=1.5, alpha=0.85, label=label)
            any_ps = True
        if any_ps:
            if row == 0:
                ax_ps.legend(loc="upper right", frameon=False, fontsize=11)
            _style(ax_ps, ylabel="Power")
        else:
            ax_ps.text(0.5, 0.5, "No spectrum", ha="center", va="center",
                       fontsize=12, transform=ax_ps.transAxes, color="0.5")
            _style(ax_ps, ylabel="Power")
        if row < len(pick) - 1:
            ax_ps.set_xticklabels([])
    _style(ax, xlabel="Time (s)")
    _style(ax_ps, xlabel="Frequency (Hz)")

    fig.suptitle("Prediction Quality: LOO-Tested Neurons",
                 fontsize=24, fontweight="bold", y=1.01)
    _save_png(fig, save_dir, "02_prediction_traces.png", bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Ridge-CV diagnostics (alpha, behaviour decoder, summary)
# =====================================================================

def _run_alpha_cv_for_diagnostics(
    model: Stage2ModelPT,
    data: Dict[str, Any],
    cfg,
) -> Optional[Dict[str, Any]]:
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
    free_run: Dict[str, Any],
    beh_results: Optional[Dict[str, Any]],
    save_dir: str,
    cfg=None,
) -> None:
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

    def _plot_alpha_best_regularization(ax) -> None:
        if alpha_diag is not None:
            lambdas_a = alpha_diag["lambdas"]
            valid_lam = lambdas_a[np.isfinite(lambdas_a)]
            at_upper = alpha_diag.get("at_upper_flags", np.zeros(N, dtype=bool))
            if len(valid_lam) > 0:
                log_lam = np.log10(np.maximum(valid_lam, 1e-12))
                ax.hist(log_lam, bins=30, color=_COL_DATA, alpha=0.7,
                        edgecolor="white")
                med_log = float(np.median(log_lam))
                ax.axvline(med_log, color=_COL_RAW, lw=2, ls="--",
                           label=f"median={10**med_log:.2g}")
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
        _style(ax, xlabel=r"$\log_{10}(\alpha)$", ylabel="Count",
               title="A. Alpha CV: Best Ridge α")

    def _plot_behavior_best_regularization(ax) -> None:
        has_gt = beh_cv_gt is not None and beh_cv_gt.get("best_lambdas") is not None
        has_model = beh_cv_model is not None and beh_cv_model.get("best_lambdas") is not None
        if has_gt and has_model:
            lam_gt = np.asarray(beh_cv_gt["best_lambdas"])
            lam_mod = np.asarray(beh_cv_model["best_lambdas"])
            n_modes = min(len(lam_gt), len(lam_mod))
            x = np.arange(n_modes)
            ax.plot(x - 0.1, np.maximum(lam_gt[:n_modes], 1e-12), marker="s", ls="none",
                    label="GT", color=_COL_DATA, alpha=0.9)
            ax.plot(x + 0.1, np.maximum(lam_mod[:n_modes], 1e-12), marker="o", ls="none",
                    label="Model", color=_COL_CV, alpha=0.9)
            ax.set_yscale("log")
            ax.set_xticks(x)
            ax.set_xticklabels([f"EW{i+1}" for i in range(n_modes)], fontsize=11)
            ax.legend(frameon=False, fontsize=11)
            at_upper_gt = beh_cv_gt.get("at_upper")
            at_upper_mod = beh_cv_model.get("at_upper")
            n_up = 0
            if at_upper_gt is not None:
                n_up += int(np.sum(at_upper_gt[:n_modes]))
            if at_upper_mod is not None:
                n_up += int(np.sum(at_upper_mod[:n_modes]))
            if n_up > 0:
                ax.text(0.97, 0.03, f"{n_up} at upper bound",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=10, color=_COL_RAW)
        elif has_model:
            lam_mod = np.asarray(beh_cv_model["best_lambdas"])
            n_modes = len(lam_mod)
            x = np.arange(n_modes)
            ax.plot(x, np.maximum(lam_mod, 1e-12), marker="o", ls="none",
                    label="Model", color=_COL_CV, alpha=0.9)
            ax.set_yscale("log")
            ax.set_xticks(x)
            ax.set_xticklabels([f"EW{i+1}" for i in range(n_modes)], fontsize=11)
            ax.legend(frameon=False, fontsize=11)
        else:
            ax.text(0.5, 0.5, "No decoder data", ha="center", va="center",
                    fontsize=13, transform=ax.transAxes)
        _style(ax, xlabel="Eigenworm mode", ylabel="Best α",
               title="B. Behaviour Decoder: Best Ridge α")

    def _plot_alpha_cv_curves(ax) -> None:
        if alpha_diag is not None and alpha_diag.get("cv_mse_all") is not None:
            cv_mse_all = alpha_diag["cv_mse_all"]
            ridge_grid = alpha_diag["ridge_grid"]
            lambdas_a = alpha_diag["lambdas"]
            valid_neurons = np.where(np.isfinite(lambdas_a))[0]
            if len(valid_neurons) > 0:
                fit_r2_a = alpha_diag["fit_r2"]
                r2_valid = fit_r2_a[valid_neurons]
                order_r2 = valid_neurons[np.argsort(r2_valid)]
                n_sample = min(15, len(order_r2))
                idx_sample = np.linspace(0, len(order_r2) - 1, n_sample, dtype=int)
                sample_neurons = order_r2[idx_sample]
                cmap = plt.cm.viridis(np.linspace(0, 1, len(sample_neurons)))
                for ci, ni in enumerate(sample_neurons):
                    curve = cv_mse_all[ni]
                    finite = np.isfinite(curve)
                    if finite.sum() > 2:
                        ax.plot(ridge_grid[finite], curve[finite],
                                color=cmap[ci], alpha=0.6, lw=1.2)
                        best_idx = np.nanargmin(curve)
                        ax.plot(ridge_grid[best_idx], curve[best_idx],
                                "o", color=cmap[ci], markersize=4, alpha=0.8)
                ax.set_xscale("log")
                ax.set_yscale("log")
        else:
            ax.text(0.5, 0.5, "No alpha CV data\n(alpha_per_neuron off?)",
                    ha="center", va="center", fontsize=13, transform=ax.transAxes)
        _style(ax, xlabel="Ridge α", ylabel="CV-MSE",
               title="H. Alpha CV: MSE Curves (sample neurons)")

    # ================================================================
    # Row 0: Best ridge-α summaries
    # ================================================================

    # ---- A. Best-α histogram across neurons --------------------------
    ax = fig.add_subplot(gs[0, 0])
    _plot_alpha_best_regularization(ax)

    # ---- B. Best-α comparison: behaviour decoder ----------------------
    ax = fig.add_subplot(gs[0, 1])
    _plot_behavior_best_regularization(ax)

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

    # ---- G. Decoder CV-MSE at Best α (per mode) ----------------------
    ax = fig.add_subplot(gs[1, 2])
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
           title="G. Decoder CV-MSE at Best α")

    # ---- H. Sample alpha CV-MSE curves -------------------------------
    ax = fig.add_subplot(gs[1, 3])
    _plot_alpha_cv_curves(ax)

    # ================================================================
    # Row 2: Free Run Diagnostics
    # ================================================================

    r2_fr = free_run["r2"]
    r2_os = onestep["r2"]
    var_per = np.nanvar(u_np, axis=0)

    from scipy.stats import spearmanr

    # ---- I. Neuron Variance vs Free-run R² ----------------------
    ax = fig.add_subplot(gs[2, 0])
    valid = np.isfinite(var_per) & np.isfinite(r2_fr) & (var_per > 0)
    if valid.sum() > 2:
        ax.scatter(var_per[valid], r2_fr[valid], s=20, alpha=0.5, c=_COL_DATA, edgecolors="none")
        ax.set_xscale("log")
        ax.axhline(0, color="gray", lw=0.6, ls=":")
        rho, pval = spearmanr(var_per[valid], r2_fr[valid])
        ax.text(0.03, 0.97, f"ρ={rho:.2f}, p={pval:.2g}",
                transform=ax.transAxes, ha="left", va="top", fontsize=11,
                bbox=dict(boxstyle="round", fc="#f7f7f7", alpha=0.85))
    _style(ax, xlabel="Neuron Variance (data)", ylabel="Free-run R²",
           title="I. Variance vs Free-run R²")

    # ---- J. One-step R² vs Free-run R² --------------------------------
    ax = fig.add_subplot(gs[2, 1])
    valid = np.isfinite(r2_os) & np.isfinite(r2_fr)
    if valid.sum() > 2:
        ax.scatter(r2_os[valid], r2_fr[valid], s=20, alpha=0.5, c=_COL_CV, edgecolors="none")
        ax.axhline(0, color="gray", lw=0.6, ls=":")
        ax.axvline(0, color="gray", lw=0.6, ls=":")
        mn = min(np.nanmin(r2_os[valid]), np.nanmin(r2_fr[valid]))
        mx = max(np.nanmax(r2_os[valid]), np.nanmax(r2_fr[valid]))
        ax.plot([mn, mx], [mn, mx], "k--", alpha=0.3, zorder=-1)
        rho, pval = spearmanr(r2_os[valid], r2_fr[valid])
        ax.text(0.03, 0.97, f"ρ={rho:.2f}, p={pval:.2g}",
                transform=ax.transAxes, ha="left", va="top", fontsize=11,
                bbox=dict(boxstyle="round", fc="#f7f7f7", alpha=0.85))
    _style(ax, xlabel="One-step R²", ylabel="Free-run R²",
           title="J. One-step R² vs Free-run R²")

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
        n_interior = n_solved - n_up - n_lo
        cv_types.append("Alpha\n(per-neuron)")
        n_interior_list.append(max(n_interior, 0))
        n_upper_list.append(n_up)
        n_zero_list.append(n_lo)

    if cv_types:
        x_k = np.arange(len(cv_types))
        w_k = 0.5
        ax.bar(x_k, n_interior_list, w_k, label="Interior", color=_COL_CV, alpha=0.75)
        ax.bar(x_k, n_upper_list, w_k, bottom=n_interior_list, label="At upper",
               color=_COL_RAW, alpha=0.75)
        bottoms = [a + b for a, b in zip(n_interior_list, n_upper_list)]
        ax.bar(x_k, n_zero_list, w_k, bottom=bottoms, label="At lower",
               color="0.6", alpha=0.75)
        ax.set_xticks(x_k)
        ax.set_xticklabels(cv_types, fontsize=10)
        ax.legend(frameon=False, fontsize=9)
    else:
        ax.text(0.5, 0.5, "No boundary data", ha="center", va="center",
                fontsize=13, transform=ax.transAxes)
    _style(ax, xlabel="", ylabel="Neurons",
           title="K. Boundary Hit Summary")

    # ---- L. Alpha λ vs one-step prediction error -----------------------
    ax = fig.add_subplot(gs[2, 3])
    if alpha_diag is not None:
        lam_a = alpha_diag["lambdas"]
        valid_l = np.isfinite(lam_a) & np.isfinite(r2_os)
        if valid_l.sum() > 2:
            ax.scatter(np.log10(np.maximum(lam_a[valid_l], 1e-12)),
                       r2_os[valid_l], s=20, alpha=0.5, c=_COL_DATA,
                       edgecolors="none")
            rho, pval = spearmanr(lam_a[valid_l], r2_os[valid_l])
            ax.text(0.03, 0.97, f"ρ={rho:.2f}, p={pval:.2g}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=11,
                    bbox=dict(boxstyle="round", fc="#f7f7f7", alpha=0.85))
    else:
        ax.text(0.5, 0.5, "No alpha data", ha="center", va="center",
                fontsize=13, transform=ax.transAxes)
    _style(ax, xlabel=r"$\log_{10}(\alpha)$", ylabel="One-step R²",
           title="L. Ridge α vs One-step R²")


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
    beh_all_baseline=None,
    include_ridge_diagnostics: bool = True,
) -> Dict[str, Any]:
    """Run full evaluation, then render the three diagnostic figures."""
    import matplotlib

    if not show:
        matplotlib.use("Agg")

    setup_plot_style()
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    results = run_full_evaluation(model, data, cfg, decoder=decoder,
                                  beh_all_baseline=beh_all_baseline)

    onestep  = results["onestep"]
    loo      = results["loo"]
    free_run = results["free_run"]
    beh      = results.get("beh")
    beh_all  = results.get("beh_all")

    plot_summary_slide(model, data, onestep, loo, free_run, beh,
                       save_dir, epoch_losses=epoch_losses,
                       beh_all=beh_all)
    plot_parameter_trajectories(epoch_losses, save_dir, cfg=cfg)
    plot_prediction_traces(data, onestep, loo, free_run, save_dir)

    if include_ridge_diagnostics:
        plot_ridge_cv_diagnostics(model, data, onestep, free_run, beh, save_dir, cfg=cfg)

    return {
        "beh": beh,
        "beh_r2_model": results.get("beh_r2_model"),
    }


# ======================================================================= #
#  Multi-worm summary plots  (merged from plot_eval_multi.py)             #
# ======================================================================= #

_COL_OS    = "#2166ac"   # one-step
_COL_LOO   = "#d6604d"   # LOO
_COL_OBS   = "#1a9641"   # observed
_COL_UNOBS = "#d4d4d4"   # unobserved
_COL_WORM  = "#756bb1"   # per-worm accent


def _plot_per_worm_r2(
    eval_results: Dict[str, Any],
    fig,
    axes: list,
) -> None:
    """Axes[0]: sorted one-step R² bars.  Axes[1]: sorted LOO R² bars.
    Axes[2]: overlaid histograms for both metrics."""
    per_worm   = eval_results["per_worm"]
    worm_ids   = list(per_worm.keys())

    r2_os  = np.array([per_worm[w]["r2_onestep_median"] for w in worm_ids])
    r2_loo = np.array([per_worm[w]["r2_loo_median"]     for w in worm_ids])
    n_obs  = np.array([per_worm[w]["N_obs"]             for w in worm_ids])

    order = np.argsort(r2_os)[::-1]
    worm_ids_s = [worm_ids[i] for i in order]
    r2_os_s    = r2_os[order]
    r2_loo_s   = r2_loo[order]
    n_obs_s    = n_obs[order]

    x = np.arange(len(worm_ids_s))

    ax0 = axes[0]
    ax0.bar(x, r2_os_s, color=_COL_OS, alpha=0.85, width=0.7)
    ax0.axhline(float(np.nanmedian(r2_os)), ls="--", lw=1.2, color="k",
                label=f"median={np.nanmedian(r2_os):.3f}")
    ax0.set_xticks(x)
    ax0.set_xticklabels(worm_ids_s, rotation=90, fontsize=6)
    ax0.set_ylim(min(0.0, float(np.nanmin(r2_os_s)) - 0.05),
                 min(1.0, float(np.nanmax(r2_os_s)) + 0.05))
    _style(ax0, "Worm", "Median R²", "One-step R² (val)")
    ax0.legend(fontsize=7)

    for xi, ni in zip(x, n_obs_s):
        ax0.text(xi, ax0.get_ylim()[0] + 0.01, str(ni),
                 ha="center", va="bottom", fontsize=5, color="grey")

    ax1 = axes[1]
    ax1.bar(x, r2_loo_s, color=_COL_LOO, alpha=0.85, width=0.7)
    ax1.axhline(float(np.nanmedian(r2_loo)), ls="--", lw=1.2, color="k",
                label=f"median={np.nanmedian(r2_loo):.3f}")
    ax1.set_xticks(x)
    ax1.set_xticklabels(worm_ids_s, rotation=90, fontsize=6)
    ax1.set_ylim(min(0.0, float(np.nanmin(r2_loo_s)) - 0.05),
                 min(1.0, float(np.nanmax(r2_loo_s)) + 0.05))
    _style(ax1, "Worm", "Median R²", "LOO R² (val)")
    ax1.legend(fontsize=7)

    ax2 = axes[2]
    bins  = np.linspace(min(np.nanmin(r2_os), np.nanmin(r2_loo)) - 0.05,
                        min(1.01, max(np.nanmax(r2_os), np.nanmax(r2_loo)) + 0.05),
                        25)
    ax2.hist(r2_os[np.isfinite(r2_os)],   bins=bins, color=_COL_OS,  alpha=0.6,
             label="one-step")
    ax2.hist(r2_loo[np.isfinite(r2_loo)], bins=bins, color=_COL_LOO, alpha=0.6,
             label="LOO")
    _style(ax2, "Median R² per worm", "Count", "R² distribution")
    ax2.legend(fontsize=7)


def _plot_coverage_heatmap(
    eval_results:  Dict[str, Any],
    worm_states:   list,
    atlas_labels:  Optional[list],
    ax,
    max_neurons:   int = 60,
) -> None:
    """Binary heatmap: rows = neurons (top-covered), cols = worms."""
    summary  = eval_results["summary"]
    cov_mat  = summary["coverage_matrix"]
    obs_cnt  = summary["neuron_obs_count"]
    worm_ids = summary["worm_ids"]

    order_n  = np.argsort(obs_cnt)[::-1][:max_neurons]
    cov_sub  = cov_mat[order_n, :]
    cnt_sub  = obs_cnt[order_n]

    ylabels = (
        [atlas_labels[i] for i in order_n]
        if atlas_labels is not None
        else [str(i) for i in order_n]
    )
    xlabels = worm_ids

    cmap = mcolors.ListedColormap([_COL_UNOBS, _COL_OBS])
    ax.imshow(cov_sub.astype(float), aspect="auto", cmap=cmap,
              vmin=0, vmax=1, interpolation="none")

    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=5)
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=90, fontsize=5)
    ax.set_xlabel("Worm", fontsize=8)
    ax.set_ylabel("Neuron (top by coverage)", fontsize=8)
    ax.set_title(
        f"Coverage matrix  (top {len(order_n)} neurons × {len(worm_ids)} worms)",
        fontsize=8,
    )

    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(np.arange(len(cnt_sub)))
    ax2.set_yticklabels([str(c) for c in cnt_sub], fontsize=4)
    ax2.set_ylabel("# worms observed", fontsize=6)


def _plot_per_neuron_r2_vs_coverage(
    eval_results: Dict[str, Any],
    atlas_labels: Optional[list],
    ax,
) -> None:
    summary = eval_results["summary"]
    obs_cnt = summary["neuron_obs_count"]
    r2_os   = summary["neuron_r2_onestep"]
    r2_loo  = summary["neuron_r2_loo"]

    valid_os  = np.isfinite(r2_os)
    valid_loo = np.isfinite(r2_loo)

    ax.scatter(obs_cnt[valid_os], r2_os[valid_os], s=15, alpha=0.7,
               color=_COL_OS, label="one-step", zorder=3)
    ax.scatter(obs_cnt[valid_loo], r2_loo[valid_loo], s=15, alpha=0.5,
               color=_COL_LOO, label="LOO", zorder=2)

    if atlas_labels is not None:
        top_k = np.argsort(obs_cnt)[::-1][:10]
        for i in top_k:
            if valid_os[i]:
                ax.annotate(atlas_labels[i], (obs_cnt[i], r2_os[i]),
                            fontsize=4, ha="left", va="bottom")

    ax.axhline(0, ls="--", lw=0.8, color="k", alpha=0.3)
    _style(ax, "# worms observing neuron", "Mean R² (val)", "Per-neuron R² vs coverage")
    ax.legend(fontsize=7)


def _plot_lambda_u_distributions(
    eval_results: Dict[str, Any],
    worm_states:  list,
    ax,
) -> None:
    """Boxplot of per-neuron λ_u values, one box per worm."""
    per_worm = eval_results["per_worm"]
    worm_ids = list(per_worm.keys())

    data_boxes = []
    labels_b   = []
    for wid in worm_ids:
        lam_w = per_worm[wid]["lambda_u"]
        obs   = per_worm[wid]["onestep"]["obs_idx"]
        data_boxes.append(lam_w[obs])
        labels_b.append(wid)

    ax.boxplot(
        data_boxes,
        labels=labels_b,
        patch_artist=True,
        medianprops=dict(color="k", lw=1.5),
        boxprops=dict(facecolor=_COL_WORM, alpha=0.6),
        flierprops=dict(marker=".", markersize=2, alpha=0.4),
        whiskerprops=dict(lw=0.8),
        capprops=dict(lw=0.8),
    )
    import matplotlib.pyplot as plt
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=6)
    _style(ax, "Worm", "λ_u (observed neurons)", "Per-worm λ_u distributions")


def _plot_per_worm_Gvalues(
    eval_results: Dict[str, Any],
    ax,
) -> None:
    """Bar chart of per-worm G scalar (only if per_worm_G is active)."""
    per_worm = eval_results["per_worm"]
    worm_ids = list(per_worm.keys())
    G_vals   = [per_worm[w]["G_worm"] for w in worm_ids]

    if all(g is None for g in G_vals):
        ax.text(0.5, 0.5, "per-worm G not enabled",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        _style(ax, "", "", "Per-worm G")
        return

    G_arr = np.array([g if g is not None else np.nan for g in G_vals])
    x     = np.arange(len(worm_ids))
    ax.bar(x, G_arr, color=_COL_WORM, alpha=0.8, width=0.7)
    ax.axhline(1.0, ls="--", lw=1, color="k", alpha=0.4, label="G=1 (shared)")
    ax.set_xticks(x)
    ax.set_xticklabels(worm_ids, rotation=90, fontsize=6)
    _style(ax, "Worm", "G scalar", "Per-worm G")
    ax.legend(fontsize=7)


def plot_multi_worm_r2(
    eval_results: Dict[str, Any],
    worm_states:  list,
    atlas_labels: Optional[list],
    save_dir:     str,
    show:         bool = False,
) -> None:
    """Save ``01_multi_worm_r2.png``."""
    import matplotlib.pyplot as plt

    setup_plot_style()
    n_worms = len(eval_results["per_worm"])

    fig_w  = max(10, n_worms * 0.4 + 3)
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, 4),
                             gridspec_kw={"width_ratios": [2, 2, 1]})
    fig.suptitle(f"Multi-worm R²  ({n_worms} worms)", fontsize=10, fontweight="bold")

    _plot_per_worm_r2(eval_results, fig, list(axes))
    fig.tight_layout()
    _save_png(fig, save_dir, "01_multi_worm_r2.png")
    if show:
        plt.show()
    plt.close(fig)


def plot_coverage_heatmap(
    eval_results: Dict[str, Any],
    worm_states:  list,
    atlas_labels: Optional[list],
    save_dir:     str,
    show:         bool = False,
    max_neurons:  int  = 60,
) -> None:
    """Save ``02_coverage_heatmap.png``."""
    import matplotlib.pyplot as plt

    setup_plot_style()
    n_worms  = len(worm_states)
    fig_w    = max(8, n_worms * 0.35 + 3)
    fig_h    = max(8, min(max_neurons * 0.18, 20))
    fig, ax  = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    fig.suptitle("Neuron coverage across worms", fontsize=10, fontweight="bold")

    _plot_coverage_heatmap(eval_results, worm_states, atlas_labels, ax,
                           max_neurons=max_neurons)
    fig.tight_layout()
    _save_png(fig, save_dir, "02_coverage_heatmap.png")
    if show:
        plt.show()
    plt.close(fig)


def plot_per_neuron_analysis(
    eval_results: Dict[str, Any],
    worm_states:  list,
    atlas_labels: Optional[list],
    save_dir:     str,
    show:         bool = False,
) -> None:
    """Save ``03_per_neuron_analysis.png``."""
    import matplotlib.pyplot as plt

    setup_plot_style()
    n_worms = len(worm_states)
    fig_w   = max(12, n_worms * 0.4 + 4)
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, 4))
    fig.suptitle("Per-neuron analysis", fontsize=10, fontweight="bold")

    _plot_per_neuron_r2_vs_coverage(eval_results, atlas_labels, axes[0])
    _plot_lambda_u_distributions(eval_results, worm_states, axes[1])
    _plot_per_worm_Gvalues(eval_results, axes[2])

    fig.tight_layout()
    _save_png(fig, save_dir, "03_per_neuron_analysis.png")
    if show:
        plt.show()
    plt.close(fig)


def plot_r2_three_metrics(
    eval_results: Dict[str, Any],
    save_dir: Optional[str] = None,
    show: bool = False,
) -> None:
    """Summary figure: per-worm per-neuron R² distributions for one-step, LOO,
    and free-run side-by-side as boxplots + median markers."""
    import matplotlib.pyplot as plt

    setup_plot_style()

    per_worm  = eval_results["per_worm"]
    worm_ids  = list(per_worm.keys())
    n_worms   = len(worm_ids)

    metrics = [
        ("onestep",  "One-step R²",  _COL_OS),
        ("loo",      "LOO R²",        _COL_LOO),
        ("freerun",  "Free-run R²",   "#4dac26"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(max(10, 3.5 * n_worms), 6),
                             sharey=False)
    fig.suptitle(f"R² across metrics  ({n_worms} worms, val set)",
                 fontsize=18, fontweight="bold", y=1.02)

    for ax, (key, title, color) in zip(axes, metrics):
        x_pos = np.arange(n_worms)
        for xi, wid in enumerate(worm_ids):
            obs_arr = per_worm[wid][key]["obs_idx"]
            r2_vals = per_worm[wid][key]["r2"][obs_arr]
            r2_vals = r2_vals[np.isfinite(r2_vals)]
            if r2_vals.size == 0:
                continue
            jitter = np.random.default_rng(42).uniform(-0.18, 0.18, r2_vals.size)
            ax.scatter(xi + jitter, r2_vals,
                       s=12, color=color, alpha=0.25, linewidths=0, zorder=2)
            q25, med, q75 = np.percentile(r2_vals, [25, 50, 75])
            ax.vlines(xi, q25, q75, color="0.15", lw=5, zorder=3)
            ax.scatter([xi], [med], s=60, color="white",
                       edgecolors="0.15", linewidths=1.5, zorder=4)
            ax.text(xi, q25 - 0.03, f"{med:.2f}",
                    ha="center", va="top", fontsize=9, color="0.3")

        ax.axhline(0, color="gray", lw=0.8, ls=":")
        short_ids = [w.replace("-", "\u2011") for w in worm_ids]
        ax.set_xticks(x_pos)
        ax.set_xticklabels(short_ids, rotation=45, ha="right", fontsize=9)
        ax.set_xlim(-0.6, n_worms - 0.4)
        _style(ax, ylabel="R² (val)", title=title)

    plt.tight_layout()
    if save_dir is not None:
        _save_png(fig, save_dir, "00_r2_summary.png", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def generate_multi_worm_plots(
    model:        Stage2ModelPT,
    worm_states:  list,
    eval_results: Dict[str, Any],
    cfg           = None,
    atlas_labels: Optional[list] = None,
    save_dir:     Optional[str] = None,
    show:         bool = False,
) -> None:
    """Generate and save all multi-worm diagnostic plots."""
    if not eval_results:
        print("[MultiWorm Plots] eval_results is empty; skipping.")
        return

    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    if atlas_labels is None:
        atlas_labels = eval_results.get("atlas_labels", None)

    n_worms = len(worm_states)
    print(f"[MultiWorm Plots] Generating plots for {n_worms} worms …")

    try:
        plot_multi_worm_r2(eval_results, worm_states, atlas_labels,
                           save_dir=save_dir, show=show)
        print("[MultiWorm Plots] 01_multi_worm_r2.png  ✓")
    except Exception as exc:
        print(f"[MultiWorm Plots] 01 FAILED: {exc}")

    try:
        plot_coverage_heatmap(eval_results, worm_states, atlas_labels,
                              save_dir=save_dir, show=show)
        print("[MultiWorm Plots] 02_coverage_heatmap.png  ✓")
    except Exception as exc:
        print(f"[MultiWorm Plots] 02 FAILED: {exc}")

    try:
        plot_per_neuron_analysis(eval_results, worm_states, atlas_labels,
                                 save_dir=save_dir, show=show)
        print("[MultiWorm Plots] 03_per_neuron_analysis.png  ✓")
    except Exception as exc:
        print(f"[MultiWorm Plots] 03 FAILED: {exc}")

    if save_dir is not None:
        print(f"[MultiWorm Plots] All plots saved to {save_dir}")
