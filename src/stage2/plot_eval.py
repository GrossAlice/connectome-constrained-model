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
    "plot_merged_summary_slide",
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

    rms_gap = _rms(gap_all)
    rms_sv = _rms(sv_all)
    rms_dcv = _rms(dcv_all)
    rms_net = float(np.sqrt(rms_gap**2 + rms_sv**2 + rms_dcv**2))

    summary = {
        "AR(1) residual\n(network should explain)": _rms(target_resid),
        "$\\lambda I_{gap}$": rms_gap,
        "$\\lambda I_{sv}$": rms_sv,
        "$\\lambda I_{dcv}$": rms_dcv,
    }
    stim_rms = _rms(stim_all)
    if stim_rms > 1e-12:
        summary["$\\lambda I_{stim}$"] = stim_rms
    summary["Network total"] = rms_net
    summary["Model error\n(predicted \u2212 actual)"] = _rms(unexplained)
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
    hist_panels = [
        (r2_os, _COL_DATA, "One-step R\u00b2", "B. One-step R\u00b2"),
        (r2_loo, _COL_CV,  "LOO R\u00b2",      "C. LOO R\u00b2"),
        (r2_fr,  _COL_FR,  "Free-run R\u00b2",  "D. Free-run R\u00b2"),
    ]
    for col, (r2, color, label, title) in enumerate(hist_panels, start=1):
        ax = fig.add_subplot(gs[0, min(col, 3)])
        if col > 3:
            # Overlay CV-reg on top of one-step histogram (panel B)
            ax = fig.add_subplot(gs[0, 1])
            valid = r2[np.isfinite(r2)]
            if len(valid) > 0:
                bins = np.linspace(min(valid.min(), -0.1),
                                   max(valid.max(), 1.0), 30)
                ax.hist(valid, bins=bins, color=color, alpha=0.35,
                        edgecolor="none", label=f"CV-reg {np.median(valid):.3f}")
                ax.legend(frameon=False, fontsize=10)
            continue
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
    colours_d = [_COL_RAW, _COL_DATA, _COL_CV, _COL_FR, "#ff7f0e", "#333333", "0.55"]
    y_pos = np.arange(len(labels_d))
    ax.barh(y_pos, vals_d, color=colours_d[:len(vals_d)], alpha=0.8,
            edgecolor="white", height=0.65)
    for i, v in enumerate(vals_d):
        ax.text(v + max(vals_d) * 0.02, i, f"{v:.4g}", va="center",
                fontsize=10)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_d, fontsize=11)
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
    # Each neuron's "target" is its AR(1) residual RMS — the signal the
    # network must reproduce.  "Network" is the combined RMS of gap +
    # SV + DCV currents that the model actually produces.  Ratio < 1
    # means the network under-drives that neuron (amplitude too low);
    # ratio > 1 means it over-drives (potential instability).
    ax = fig.add_subplot(gs[2, 3])
    # per_neuron_rms columns: [target_resid, gap, sv, dcv]
    tgt_rms_pn = per_neuron_rms[:, 0]
    net_rms_pn = per_neuron_rms[:, 1] + per_neuron_rms[:, 2] + per_neuron_rms[:, 3]
    ratio = np.where(tgt_rms_pn > 1e-12, net_rms_pn / tgt_rms_pn, 0.0)
    order = np.argsort(ratio)
    colors_ratio = np.where(ratio[order] <= 1.0, _COL_CV, _COL_RAW)
    ax.barh(np.arange(N), ratio[order], color=colors_ratio, alpha=0.6,
            height=1.0, edgecolor="none")
    ax.axvline(1.0, color="0.3", lw=1.5, ls="--", label="ratio = 1 (balanced)")
    med_ratio = float(np.median(ratio))
    n_over = int((ratio > 1.0).sum())
    n_under = int((ratio < 0.1).sum())
    ax.text(0.97, 0.97,
            f"median ratio = {med_ratio:.2f}\n"
            f"{n_over}/{N} over-driven (>1)\n"
            f"{n_under}/{N} near-silent (<0.1)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="#f7f7f7", alpha=0.85))
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    _style(ax, xlabel="Network RMS / AR(1) residual RMS",
           ylabel="Neurons (sorted)",
           title="L. Per-Neuron Network Gain")

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
    freerun_stoch: Optional[Dict[str, Any]] = None,
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

    # Stochastic trajectory samples (may or may not be present)
    loo_samples = loo.get("samples")  # dict idx -> (n_samples, T)

    pick = sorted(preds_loo.keys())
    if len(pick) == 0:
        return

    fig = plt.figure(figsize=(25, 3.6 * len(pick) + 1.5))
    gs = GridSpec(len(pick), 3, width_ratios=[1.35, 5.2, 2.15],
                  hspace=0.35, wspace=0.20)

    _COL_SAMPLE = "#9467bd"   # muted purple for stochastic trajectories

    for row, idx in enumerate(pick):
        ax_net = fig.add_subplot(gs[row, 0])
        _plot_partner_panel(ax_net, data, labels, idx)

        ax = fig.add_subplot(gs[row, 1])

        # --- Stochastic samples: confidence band + thin lines ---
        if loo_samples is not None and idx in loo_samples:
            samp = loo_samples[idx]                    # (n_samples, T)
            n_samp = samp.shape[0]
            samp_lo = np.percentile(samp, 2.5, axis=0)
            samp_hi = np.percentile(samp, 97.5, axis=0)
            _ci_label = "95% CI (samples)"
            ax.fill_between(
                time, samp_lo, samp_hi,
                color=_COL_SAMPLE, alpha=0.12, zorder=1,
                label=_ci_label,
            )
            n_show = min(n_samp, 5)
            for k in range(n_show):
                ax.plot(
                    time, samp[k], color=_COL_SAMPLE,
                    lw=0.7, alpha=0.35, zorder=1,
                    label="Sample" if k == 0 else None,
                )

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
        # --- Full-brain stochastic free-run: CI band + thin samples ---
        _COL_FRSTOCH = "#2ca02c"  # green, matching free-run hue
        if freerun_stoch is not None:
            fr_ci_lo = freerun_stoch["ci_lo"][:, idx]
            fr_ci_hi = freerun_stoch["ci_hi"][:, idx]
            ax.fill_between(
                time, fr_ci_lo, fr_ci_hi,
                color=_COL_FRSTOCH, alpha=0.10, zorder=1,
                label="Free-run 95% CI",
            )
            fr_samp = freerun_stoch["samples"]  # (K, T, N)
            n_show = min(fr_samp.shape[0], 5)
            for kk in range(n_show):
                ax.plot(
                    time, fr_samp[kk, :, idx],
                    color=_COL_FRSTOCH, lw=0.6, alpha=0.30, zorder=1,
                    label="FR sample" if kk == 0 else None,
                )
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


def generate_eval_loo_plots(
    model,
    data: Dict[str, Any],
    cfg,
    epoch_losses: list,
    save_dir: str,
    show: bool = False,
    decoder=None,
    beh_all_baseline=None,
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
    plot_prediction_traces(data, onestep, loo, free_run, save_dir,
                           freerun_stoch=results.get("freerun_stoch"))

    return {
        "beh": beh,
        "beh_r2_model": results.get("beh_r2_model"),
        "onestep": onestep,
        "loo": loo,
        "free_run": free_run,
        "freerun_stoch": results.get("freerun_stoch"),
        "beh_all": beh_all,
    }


# ======================================================================= #
#  Merged (all-worms) summary slide                                       #
# ======================================================================= #

def plot_merged_summary_slide(
    model: Stage2ModelPT,
    worm_entries: list,
    epoch_losses: list,
    save_dir: str,
    cfg=None,
    include_rollout: bool = True,
) -> None:
    """All-worms-merged 12-panel summary.

    Pools per-neuron R² across worms into joint histograms, averages
    behaviour decoding and input-decomposition, and overlays per-worm
    multistep rollout curves.  Model-level panels (training convergence,
    kernel amplitudes, network trajectories) are unchanged.

    Parameters
    ----------
    worm_entries : list of (worm_id, worm_data, eval_results)
        Each *eval_results* is the dict returned by
        ``generate_eval_loo_plots`` (must contain ``onestep``, ``loo``,
        ``free_run``, and optionally ``beh`` / ``beh_all``).
    include_rollout : bool
        If *True* (default), compute and overlay per-worm multistep
        rollout curves (panel F).  Set to *False* to skip the
        moderately-expensive rollout computation.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    if not worm_entries:
        return

    setup_plot_style()
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    n_worms = len(worm_entries)
    worm_ids = [e[0] for e in worm_entries]
    worm_cols = plt.cm.tab10(np.linspace(0, 0.9, max(n_worms, 2)))[:n_worms]

    # Resolve per-worm R² -----------------------------------------------
    _metric_keys = ("onestep", "loo", "free_run")
    per_worm_r2 = {k: [] for k in _metric_keys}          # list-of-arrays
    for _, _, res in worm_entries:
        for k in _metric_keys:
            per_worm_r2[k].append(res[k]["r2"])
    pooled_r2 = {k: np.concatenate(per_worm_r2[k]) for k in _metric_keys}

    # First worm's dt (should be identical across worms)
    dt = float(worm_entries[0][1].get("dt", 0.6))

    # -------------------------------------------------------------------
    fig = plt.figure(figsize=(24, 14))
    gs = GridSpec(3, 4, hspace=0.42, wspace=0.38)

    # ── A. Training Convergence ────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    _epoch_losses = epoch_losses or []
    if _epoch_losses:
        epochs = np.arange(1, len(_epoch_losses) + 1)
        ax.plot(epochs, [e["dynamics"] for e in _epoch_losses],
                color=_COL_DATA, lw=2.2, label="Dynamics")
        ax.plot(epochs, [e["total"] for e in _epoch_losses],
                color=_COL_RAW, lw=1.8, alpha=0.8, label="Total")
        if any(e.get("behaviour_loss") is not None for e in _epoch_losses):
            beh_loss = [e.get("behaviour_loss", np.nan) for e in _epoch_losses]
            ax.plot(epochs, beh_loss, color=_COL_CV, lw=1.6, alpha=0.8,
                    label="Behaviour")
        ax.set_yscale("log")
        _style(ax, xlabel="Epoch", ylabel="Loss",
               title="A. Training Convergence")
        ax.legend(frameon=False, fontsize=12, ncol=1)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No training history",
                ha="center", va="center", fontsize=14)

    # ── B / C / D.  Pooled R² histograms ──────────────────────────────
    _hist_meta = [
        ("onestep",  _COL_DATA, "One-step R\u00b2", "B. One-step R\u00b2 (all worms)"),
        ("loo",      _COL_CV,   "LOO R\u00b2",      "C. LOO R\u00b2 (all worms)"),
        ("free_run", _COL_FR,   "Free-run R\u00b2",  "D. Free-run R\u00b2 (all worms)"),
    ]
    for col, (key, color, xlabel_txt, title_txt) in enumerate(_hist_meta, start=1):
        ax = fig.add_subplot(gs[0, col])
        all_valid = pooled_r2[key][np.isfinite(pooled_r2[key])]
        if all_valid.size > 0:
            bins = np.linspace(min(all_valid.min(), -0.1),
                               max(all_valid.max(), 1.0), 30)
            per_worm_vals = [
                r2[np.isfinite(r2)] for r2 in per_worm_r2[key]
            ]
            ax.hist(per_worm_vals, bins=bins, stacked=True,
                    color=worm_cols[:n_worms], alpha=0.75,
                    edgecolor="white", label=worm_ids)
            overall_med = float(np.nanmedian(all_valid))
            ax.axvline(overall_med, color="k", lw=2, ls="--",
                       label=f"median={overall_med:.3f}")
            for wi, r2_w in enumerate(per_worm_vals):
                if r2_w.size:
                    ax.axvline(float(np.median(r2_w)),
                               color=worm_cols[wi], lw=1.2, ls=":",
                               alpha=0.8)
            ncol = 1 if n_worms <= 4 else 2
            ax.legend(frameon=False, fontsize=8, loc="upper left",
                      ncol=ncol)
        _style(ax, xlabel=xlabel_txt, ylabel="Count", title=title_txt)

    # ── E. Behaviour Decoding (mean across worms) ─────────────────────
    ax = fig.add_subplot(gs[1, 0])
    beh_list = [(e[0], e[2].get("beh"), e[2].get("beh_all"))
                for e in worm_entries]
    has_any_beh = any(b is not None for _, b, _ in beh_list)
    if has_any_beh:
        n_modes = 5
        all_r2_motor, all_r2_model, all_r2_ar1, all_r2_all = [], [], [], []
        for _, beh, beh_all in beh_list:
            if beh is None:
                continue
            all_r2_motor.append(beh.get("r2_gt",    np.full(6, np.nan))[:n_modes])
            all_r2_model.append(beh.get("r2_model", np.full(6, np.nan))[:n_modes])
            all_r2_ar1.append(beh.get("r2_ar1",   np.full(6, np.nan))[:n_modes])
            if beh_all is not None and "r2_all_neurons" in beh_all:
                all_r2_all.append(beh_all["r2_all_neurons"][:n_modes])
        x = np.arange(n_modes)
        has_all = len(all_r2_all) > 0
        has_ar1 = (len(all_r2_ar1) > 0
                   and any(np.any(np.isfinite(a)) for a in all_r2_ar1))
        n_bars = 2 + int(has_all) + int(has_ar1)
        w = 0.8 / n_bars
        offsets = np.linspace(-(n_bars - 1) * w / 2,
                              (n_bars - 1) * w / 2, n_bars)

        def _ms(arrs):
            return (np.nanmean(np.stack(arrs), axis=0) if arrs
                    else np.full(n_modes, np.nan))

        bi = 0
        if has_all:
            ax.bar(x + offsets[bi], _ms(all_r2_all), w,
                   label="All neurons (GT)", color=_COL_DATA, alpha=0.75)
            bi += 1
        ax.bar(x + offsets[bi], _ms(all_r2_motor), w,
               label="Motor neurons (GT)", color=_COL_RAW, alpha=0.75)
        bi += 1
        if has_ar1:
            ax.bar(x + offsets[bi], _ms(all_r2_ar1), w,
                   label="AR(1) baseline", color="orange", alpha=0.75)
            bi += 1
        ax.bar(x + offsets[bi], _ms(all_r2_model), w,
               label=f"Model (mean, n={len(all_r2_model)})",
               color=_COL_CV, alpha=0.75)
        ax.set_xticks(x)
        ax.set_xticklabels([f"EW{i+1}" for i in range(n_modes)],
                           fontsize=10)
        ax.legend(frameon=False, fontsize=9, loc="upper right")
        _style(ax, xlabel="Eigenworm", ylabel="R\u00b2",
               title="E. Behaviour Decoding (mean)")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No behaviour",
                ha="center", va="center", fontsize=14)

    # ── F. Multistep Rollout (per-worm + average) ─────────────────────
    ax = fig.add_subplot(gs[1, 1])
    steps_list = (1, 5, 10, 20)
    device = next(model.parameters()).device
    N = model.N

    if include_rollout:
        worm_rollout_meds: list[list[float]] = []
        for wi, (wid, wdata, _) in enumerate(worm_entries):
            u = wdata["u_stage1"].to(device)
            T_w = u.shape[0]
            gating = wdata.get("gating")
            stim_w = wdata.get("stim")
            g_def = (torch.ones(N, device=device)
                     if gating is None else None)
            u_np = u.cpu().numpy()

            with torch.no_grad():
                s_sv_a = torch.zeros(T_w, N, model.r_sv, device=device)
                s_dcv_a = torch.zeros(T_w, N, model.r_dcv, device=device)
                s_sv_t = torch.zeros(N, model.r_sv, device=device)
                s_dcv_t = torch.zeros(N, model.r_dcv, device=device)
                for t in range(T_w):
                    s_sv_a[t], s_dcv_a[t] = s_sv_t, s_dcv_t
                    g = gating[t] if gating is not None else g_def
                    s = stim_w[t] if stim_w is not None else None
                    _, s_sv_t, s_dcv_t = model.prior_step(
                        u[t], s_sv_t, s_dcv_t, g, s)

            meds_w: list[float] = []
            for K in steps_list:
                stride = max(1, K // 2)
                preds = np.zeros_like(u_np)
                counts = np.zeros(T_w)
                with torch.no_grad():
                    for t0 in range(0, T_w - K, stride):
                        u_t = u[t0]
                        sv_r = s_sv_a[t0].clone()
                        dcv_r = s_dcv_a[t0].clone()
                        for k in range(1, K + 1):
                            g = (gating[t0 + k - 1]
                                 if gating is not None else g_def)
                            s = (stim_w[t0 + k - 1]
                                 if stim_w is not None else None)
                            u_t, sv_r, dcv_r = model.prior_step(
                                u_t, sv_r, dcv_r, g, s)
                        preds[t0 + K] += u_t.cpu().numpy()
                        counts[t0 + K] += 1
                v = counts > 0
                preds[v] /= counts[v, None]
                r2_k = np.array([_r2(u_np[v, i], preds[v, i])
                                 for i in range(N)])
                meds_w.append(float(np.nanmedian(r2_k)))
            worm_rollout_meds.append(meds_w)
            ax.plot(steps_list, meds_w, color=worm_cols[wi], marker="o",
                    ls="--", lw=1.2, markersize=4, alpha=0.65,
                    label=wid)

        avg_roll = np.nanmean(worm_rollout_meds, axis=0)
        ax.plot(steps_list, avg_roll, color="k", marker="s", ls="-",
                lw=2.5, markersize=7, label="Mean")
    else:
        ax.text(0.5, 0.5, "Rollout skipped",
                ha="center", va="center", fontsize=14,
                transform=ax.transAxes)
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.legend(frameon=False, fontsize=9)
    _style(ax, xlabel="Horizon K (steps)", ylabel="Median R\u00b2",
           title="F. Multistep Rollout (all worms)")

    # ── G / H. Kernel Amplitudes (model-level) ────────────────────────
    with torch.no_grad():
        a_sv_arr = (model.a_sv.detach().cpu().numpy()
                    if model.r_sv > 0 else np.empty((0, 0)))
        a_dcv_arr = (model.a_dcv.detach().cpu().numpy()
                     if model.r_dcv > 0 else np.empty((0, 0)))
        tau_sv_arr = (model.tau_sv.detach().cpu().numpy()
                      if model.r_sv > 0 else np.array([]))
        tau_dcv_arr = (model.tau_dcv.detach().cpu().numpy()
                       if model.r_dcv > 0 else np.array([]))
    if a_sv_arr.ndim == 1:
        a_sv_arr = a_sv_arr[np.newaxis, :]
    if a_dcv_arr.ndim == 1:
        a_dcv_arr = a_dcv_arr[np.newaxis, :]

    gs_gh = gs[1, 2:].subgridspec(2, 1, hspace=0.50)
    for panel_i, (a_arr, tau_arr, syn_lbl, syn_col) in enumerate([
        (a_sv_arr, tau_sv_arr, "SV", _COL_DATA),
        (a_dcv_arr, tau_dcv_arr, "DCV", _COL_CV),
    ]):
        ax = fig.add_subplot(gs_gh[panel_i, 0])
        r = a_arr.shape[1] if a_arr.size else 0
        if r == 0:
            ax.text(0.5, 0.5, f"No {syn_lbl} ranks",
                    ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
            ax.set_axis_off()
            continue
        for ri in range(r):
            col_data = a_arr[:, ri]
            col_data = col_data[np.isfinite(col_data)]
            if col_data.size == 0:
                continue
            jit = (np.linspace(-0.16, 0.16, col_data.size)
                   if col_data.size > 1 else np.array([0.0]))
            ax.scatter(ri + 1 + jit, col_data, s=14, color=syn_col,
                       alpha=0.28, linewidths=0, zorder=2)
            q25, med, q75 = np.percentile(col_data, [25, 50, 75])
            ax.vlines(ri + 1, q25, q75, color="0.15", lw=4.0, zorder=3)
        fp = np.concatenate([
            a_arr[:, ri][a_arr[:, ri] > 0]
            for ri in range(r) if np.any(a_arr[:, ri] > 0)
        ]) if r > 0 else np.array([])
        if fp.size > 0 and fp.max() > fp.min():
            ax.set_yscale("log")
            ax.set_ylim(fp.min() / 1.35, fp.max() * 1.35)
        ax.grid(axis="y", alpha=0.18, lw=0.8)
        tick_labels = []
        for ri in range(r):
            tau_r = float(tau_arr[ri]) if ri < len(tau_arr) else 0
            col_data = a_arr[:, ri]
            col_data = col_data[np.isfinite(col_data)]
            med_r = float(np.median(col_data)) if col_data.size else 0
            tick_labels.append(f"\u03c4={tau_r:.1f}s\n{med_r:.2e}")
        ax.set_xticks(range(1, r + 1))
        ax.set_xticklabels(tick_labels, fontsize=9)
        ylabel = ("\u03b1 (log)" if fp.size > 0 and fp.max() > fp.min()
                  else "\u03b1")
        title_str = "G. Kernel Amplitudes" if panel_i == 0 else ""
        _style(ax, xlabel="", ylabel=f"{syn_lbl}  {ylabel}",
               title=title_str)

    # ── Row 2: per-worm averaged diagnostics ───────────────────────────

    # Compute input-decomposition and residual data per worm
    decomp_per_worm: list[Dict[str, float]] = []
    pn_rms_per_worm: list[np.ndarray] = []
    resid_per_worm: list[np.ndarray] = []
    T_max = 0
    for _, wdata, res in worm_entries:
        decomp, pn_rms = _compute_input_decomposition(model, wdata)
        decomp_per_worm.append(decomp)
        pn_rms_per_worm.append(pn_rms)
        mu_os = res["onestep"].get("prior_mu")
        u_np = wdata["u_stage1"].cpu().numpy()
        if mu_os is not None:
            mu_np = (mu_os.cpu().numpy() if isinstance(mu_os, torch.Tensor)
                     else mu_os)
            resid_per_worm.append(u_np - mu_np)
        T_max = max(T_max, u_np.shape[0])

    # I. Input Decomposition (averaged RMS across worms)
    ax = fig.add_subplot(gs[2, 0])
    all_keys = list(decomp_per_worm[0].keys())
    avg_vals = []
    for key in all_keys:
        vals = [d.get(key, 0.0) for d in decomp_per_worm]
        avg_vals.append(float(np.mean(vals)))
    colours_d = [_COL_RAW, _COL_DATA, _COL_CV, _COL_FR, "#ff7f0e", "#333333", "0.55"]
    y_pos = np.arange(len(all_keys))
    ax.barh(y_pos, avg_vals,
            color=colours_d[:len(avg_vals)], alpha=0.8,
            edgecolor="white", height=0.65)
    for i, v in enumerate(avg_vals):
        ax.text(v + max(avg_vals) * 0.02, i, f"{v:.4g}",
                va="center", fontsize=10)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_keys, fontsize=11)
    ax.invert_yaxis()
    _style(ax, xlabel="RMS (mean across worms)", ylabel="",
           title="I. Input Decomposition")

    # J. Network Strength Trajectories
    ax = fig.add_subplot(gs[2, 1])
    if _epoch_losses and "G" in _epoch_losses[0]:
        ep_x = np.arange(1, len(_epoch_losses) + 1)
        G_hist = np.asarray([e.get("G", np.nan) for e in _epoch_losses],
                            dtype=float)
        a_sv_h = np.asarray([e.get("a_sv_rms", np.nan)
                             for e in _epoch_losses], dtype=float)
        a_dcv_h = np.asarray([e.get("a_dcv_rms", np.nan)
                              for e in _epoch_losses], dtype=float)
        if np.any(np.isfinite(G_hist) & (G_hist > 0)):
            ax.plot(ep_x, G_hist, color=_COL_RAW, lw=2.4, label="G")
        if np.any(np.isfinite(a_sv_h) & (a_sv_h > 0)):
            ax.plot(ep_x, a_sv_h, color=_COL_DATA, lw=2.2,
                    label="\u03b1_sv RMS")
        if np.any(np.isfinite(a_dcv_h) & (a_dcv_h > 0)):
            ax.plot(ep_x, a_dcv_h, color=_COL_CV, lw=2.2,
                    label="\u03b1_dcv RMS")
        all_v = np.concatenate([G_hist, a_sv_h, a_dcv_h])
        pos_v = all_v[np.isfinite(all_v) & (all_v > 0)]
        if pos_v.size > 0:
            ax.set_yscale("log")
            ax.set_ylim(pos_v.min() / 1.5, pos_v.max() * 1.5)
        ax.grid(axis="y", alpha=0.18, lw=0.8)
        ax.legend(frameon=False, fontsize=9, loc="best")
    else:
        ax.text(0.5, 0.5, "No network history",
                ha="center", va="center", fontsize=14,
                transform=ax.transAxes)
    _style(ax, xlabel="Epoch", ylabel="Strength (log)",
           title="J. Network Strength Trajectories")

    # K. Residual Autocorrelation (averaged across worms)
    ax = fig.add_subplot(gs[2, 2])
    if resid_per_worm:
        acfs = [_residual_acf(r, max_lag=20) for r in resid_per_worm]
        avg_acf = np.nanmean(acfs, axis=0)
        lags = np.arange(1, len(avg_acf) + 1)
        ax.bar(lags, avg_acf, color=_COL_DATA, alpha=0.7,
               edgecolor="white")
        ax.axhline(0, color="gray", lw=0.8, ls=":")
        ci = 1.96 / np.sqrt(T_max)
        ax.axhline(ci, color=_COL_RAW, lw=1, ls="--", alpha=0.5)
        ax.axhline(-ci, color=_COL_RAW, lw=1, ls="--", alpha=0.5,
                   label=f"95% CI")
        ax.legend(frameon=False, fontsize=10)
    _style(ax, xlabel="Lag (steps)", ylabel="Autocorrelation",
           title="K. Residual ACF (mean)")

    # L. Per-Neuron Network Gain (pooled across worms)
    # Ratio < 1 = network under-drives neuron; > 1 = over-drives.
    ax = fig.add_subplot(gs[2, 3])
    pn_all = np.concatenate(pn_rms_per_worm, axis=0)
    tgt_rms = pn_all[:, 0]
    net_rms = pn_all[:, 1] + pn_all[:, 2] + pn_all[:, 3]
    ratio = np.where(tgt_rms > 1e-12, net_rms / tgt_rms, 0.0)
    order = np.argsort(ratio)
    colors_ratio = np.where(ratio[order] <= 1.0, _COL_CV, _COL_RAW)
    n_total = len(ratio)
    ax.barh(np.arange(n_total), ratio[order], color=colors_ratio,
            alpha=0.6, height=1.0, edgecolor="none")
    ax.axvline(1.0, color="0.3", lw=1.5, ls="--", label="ratio = 1 (balanced)")
    med_ratio = float(np.median(ratio))
    n_over = int((ratio > 1.0).sum())
    n_under = int((ratio < 0.1).sum())
    ax.text(0.97, 0.97,
            f"median ratio = {med_ratio:.2f}\n"
            f"{n_over}/{n_total} over-driven (>1)\n"
            f"{n_under}/{n_total} near-silent (<0.1)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="#f7f7f7", alpha=0.85))
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    _style(ax, xlabel="Network RMS / AR(1) residual RMS",
           ylabel="Neurons (sorted, pooled)",
           title="L. Per-Neuron Network Gain")

    # ── Suptitle & info ribbon ─────────────────────────────────────────
    with torch.no_grad():
        G_raw = model.G.cpu().numpy()
        G_val = (G_raw[model.T_e.cpu().numpy() > 0].mean()
                 if G_raw.ndim > 0 and G_raw.size > 1
                 else float(G_raw))
    G_txt = (f"{float(G_val):.6g}" if np.ndim(G_val) == 0
             else f"mean={np.mean(G_val):.4f}")
    n_ep = len(_epoch_losses)
    med_os  = float(np.nanmedian(
        pooled_r2["onestep"][np.isfinite(pooled_r2["onestep"])]))
    med_loo = float(np.nanmedian(
        pooled_r2["loo"][np.isfinite(pooled_r2["loo"])]))
    med_fr  = float(np.nanmedian(
        pooled_r2["free_run"][np.isfinite(pooled_r2["free_run"])]))

    info = (f"{n_worms} worms  N={N}  G={G_txt}  epochs={n_ep}  "
            f"one-step={med_os:.3f}  LOO={med_loo:.3f}  "
            f"free-run={med_fr:.3f}")
    fig.text(0.99, 0.995, info, ha="right", va="top", fontsize=11,
             bbox=dict(boxstyle="round", fc="#f0f0f0", alpha=0.8))
    fig.suptitle("Stage 2 Merged Evaluation (all worms)",
                 fontsize=26, fontweight="bold", y=1.01)

    _save_png(fig, save_dir, "00_merged_summary.png", bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"[MultiWorm] 00_merged_summary.png  \u2713")


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
