from __future__ import annotations
import math
import numpy as np
import torch
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from .model import Stage2ModelPT


def _ols_lambda_u(u: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    x, y = u[:-1], u[1:]
    mx, my = x.mean(0), y.mean(0)
    var_x = ((x - mx) ** 2).mean(0)
    cov_xy = ((x - mx) * (y - my)).mean(0)
    a = torch.where(var_x > 1e-30, cov_xy / var_x, torch.ones_like(var_x))
    return (1.0 - a).clamp(lo, hi)


def init_lambda_u(u: torch.Tensor, cfg=None) -> torch.Tensor:
    # Slightly tighter than the physical bounds to avoid degenerate
    # starting points (λ=0 ⇒ no leak, λ≈1 ⇒ memoryless).
    lo = float(getattr(cfg, "lambda_u_lo", 0.0)) if cfg else 0.0
    hi = float(getattr(cfg, "lambda_u_hi", 0.9999)) if cfg else 0.9999
    init_lo = max(lo + 1e-3, 1e-3)
    init_hi = min(hi - 5e-4, 1.0 - 5e-4)
    lam = _ols_lambda_u(u, init_lo, init_hi)
    print(f"[init] lambda_u (OLS): mean={lam.mean():.4f}, "
          f"min={lam.min():.4f}, max={lam.max():.4f}")
    return lam

def init_I0(model: "Stage2ModelPT", u: torch.Tensor) -> torch.Tensor:
    lam = model.lambda_u.detach()
    x, y = u[:-1].to(model.I0.device), u[1:].to(model.I0.device)
    b = y.mean(0) - (1 - lam) * x.mean(0)
    I0_ols = b / lam.clamp(min=0.01)
    with torch.no_grad():
        model.I0.data.copy_(I0_ols)
    print(f"[init] I0 (OLS): mean={I0_ols.mean():.4f}, "
          f"min={I0_ols.min():.4f}, max={I0_ols.max():.4f}")
    return I0_ols

def init_reversals(model: "Stage2ModelPT", u: torch.Tensor, baseline: torch.Tensor, cfg=None):
    q_lo, q_hi = 0.01, 0.99
    finite = u.detach().cpu().numpy().ravel()
    finite = finite[np.isfinite(finite)]
    if finite.size < 10:
        return
    rest = float(torch.nanmedian(baseline.to(u.device)))
    e_inh, e_exc = float(np.quantile(finite, q_lo)), float(np.quantile(finite, q_hi))
    if not np.isfinite(rest):
        rest = float(np.median(finite))
    pad = max(1e-3, 0.1 * max(abs(rest), 1.0))
    if not np.isfinite(e_inh) or e_inh >= rest:
        e_inh = rest - pad
    if not np.isfinite(e_exc) or e_exc <= rest:
        e_exc = rest + pad
    with torch.no_grad():
        if model.E_sv.numel() > 1:
            sign = torch.sign(model.E_sv)
            model.E_sv.data.copy_(torch.where(sign > 0, e_exc, torch.where(sign < 0, e_inh, 0.0)))
        else:
            model.E_sv.data.fill_(e_exc)
        model.E_dcv.data.fill_(rest)
    print(f"[init] reversals: E_exc={e_exc:.4f}, E_inh={e_inh:.4f}, E_dcv={rest:.4f}")

def _collect_current_patterns(
    model: "Stage2ModelPT", u: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor], list[str]]:
    """Forward-simulate gap / sv / dcv currents at current parameter values.

    Returns
    -------
    resid : (T-1, N)  AR(1) residual
    patterns : list of (T-1, N) tensors, one per channel
    names : list of channel names ("G", "a_sv", "a_dcv")
    """
    dev, lam = model.I0.device, model.lambda_u
    u = u.to(dev)
    T, N = u.shape

    resid = u[1:] - ((1 - lam) * u[:-1] + lam * model.I0)

    L = model.laplacian()
    g = torch.ones(N, device=dev)
    names = ["G"]
    c_gap = torch.zeros(T - 1, N, device=dev)
    s_sv  = torch.zeros(N, model.r_sv,  device=dev)
    s_dcv = torch.zeros(N, model.r_dcv, device=dev)
    c_sv_all  = torch.zeros(T - 1, N, device=dev) if model.r_sv  > 0 else None
    c_dcv_all = torch.zeros(T - 1, N, device=dev) if model.r_dcv > 0 else None

    for t in range(T - 1):
        c_gap[t] = lam * (L @ u[t])
        phi_g = model.phi(u[t]) * g
        if model.r_sv > 0:
            I, s_sv = model._synaptic_current(
                u[t], phi_g, s_sv,
                model.T_sv * model._get_W("W_sv"),
                model.a_sv, model.tau_sv, model.E_sv)
            c_sv_all[t] = lam * I
        if model.r_dcv > 0:
            I, s_dcv = model._synaptic_current(
                u[t], phi_g, s_dcv,
                model.T_dcv * model._get_W("W_dcv"),
                model.a_dcv, model.tau_dcv, model.E_dcv)
            c_dcv_all[t] = lam * I

    patterns = [c_gap]
    if model.r_sv  > 0: names.append("a_sv");  patterns.append(c_sv_all)
    if model.r_dcv > 0: names.append("a_dcv"); patterns.append(c_dcv_all)
    return resid, patterns, names


def _apply_scales(
    model: "Stage2ModelPT",
    names: list[str],
    betas: list[float],
    rms_list: list[float],
    rms_resid: float,
    label: str,
) -> None:
    """Scale model parameters by *betas* and log diagnostics.

    For synaptic channels (a_sv, a_dcv) the scale is absorbed into W_sv / W_dcv
    rather than into the per-rank amplitudes, so the user-specified rank
    structure in a_sv_init / a_dcv_init is preserved and doesn't saturate the
    sigmoid reparameterisation.
    """
    total_rms = math.sqrt(sum(r ** 2 for r in rms_list))
    print(f"[init] network_scale ({label}): AR1_rms={rms_resid:.4f}"
          f"  net_rms={total_rms:.4f} ({total_rms / rms_resid:.0%})")
    # Map synaptic amplitude name → weight matrix name
    _AMP_TO_W = {"a_sv": "W_sv", "a_dcv": "W_dcv"}
    for i, name in enumerate(names):
        b = betas[i]
        rms_after = rms_list[i]
        frac = rms_after / rms_resid
        if name == "G" and model.edge_specific_G:
            cur = getattr(model, name)
            if cur.numel() == 0:
                continue
            from .model import _reparam_inv
            model._G_raw.data.copy_(_reparam_inv(cur * b, model._G_lo, model._G_hi))
            print(f"[init]   G: \u03b2={b:.4f}, RMS={rms_after:.4f} ({frac:.0%} of AR1)")
        elif name in _AMP_TO_W:
            # Absorb the scale into W rather than collapsing a_sv/a_dcv
            w_name = _AMP_TO_W[name]
            cur_w = getattr(model, w_name)
            if cur_w.numel() == 0:
                continue
            model.set_param_constrained(w_name, cur_w * b)
            print(f"[init]   {name}: \u03b2={b:.4f} \u2192 applied to {w_name}, "
                  f"RMS={rms_after:.4f} ({frac:.0%} of AR1)")
        else:
            cur = getattr(model, name)
            if cur.numel() == 0:
                continue
            model.set_param_constrained(name, cur * b)
            print(f"[init]   {name}: \u03b2={b:.4f}, RMS={rms_after:.4f} ({frac:.0%} of AR1)")


# ── Global-OLS init (original) ──────────────────────────────────────────

def _init_network_scale_ols(
    model: "Stage2ModelPT", u: torch.Tensor,
    min_network_frac: float = 0.25,
) -> None:
    with torch.no_grad():
        resid, patterns, names = _collect_current_patterns(model, u)
        rms_resid = resid.pow(2).mean().sqrt().item()
        if rms_resid < 1e-12:
            return

        dev = model.I0.device
        K = len(patterns)
        r_flat = resid.reshape(-1)
        C = torch.stack([p.reshape(-1) for p in patterns], dim=1)
        CTC = C.T @ C + 1e-8 * torch.eye(K, device=dev)
        CTr = C.T @ r_flat
        beta = torch.linalg.solve(CTC, CTr).clamp(min=0)

        betas = [max(beta[i].item(), 1e-6) for i in range(K)]
        rms_list = [patterns[i].pow(2).mean().sqrt().item() * betas[i]
                    for i in range(K)]
        total_rms = math.sqrt(sum(r ** 2 for r in rms_list))

        target_rms = min_network_frac * rms_resid
        if total_rms < target_rms and total_rms > 1e-12:
            base_boost = target_rms / total_rms
        else:
            base_boost = 1.0

        scales = [base_boost for _ in range(K)]
        channel_floor_frac = {
            "G": 0.10,
            "a_sv": 0.20,
            "a_dcv": 0.10,
        }
        for i, name in enumerate(names):
            floor_frac = channel_floor_frac.get(name, 0.0)
            if floor_frac <= 0:
                continue
            rms_after_base = rms_list[i] * base_boost
            target_channel_rms = floor_frac * rms_resid
            if rms_after_base < target_channel_rms and rms_list[i] > 1e-12:
                scales[i] = max(scales[i], target_channel_rms / rms_list[i])

        final_betas = [betas[i] * scales[i] for i in range(K)]
        final_rms   = [rms_list[i] * scales[i] for i in range(K)]

        print(f"  boost={base_boost:.2f}")
        _apply_scales(model, names, final_betas, final_rms, rms_resid, "OLS")


# ── Per-neuron ridge init ───────────────────────────────────────────────

def _init_network_scale_per_neuron_ridge(
    model: "Stage2ModelPT", u: torch.Tensor,
    ridge_alpha: float = 1.0,
    min_beta: float = 1e-6,
) -> None:
    """Initialise G / a_sv / a_dcv via per-neuron ridge regression.

    For each neuron *n* independently, solve the ridge problem::

        r_{t,n} ≈ β_{G,n} · c_gap_{t,n} + β_{sv,n} · c_sv_{t,n}
                   + β_{dcv,n} · c_dcv_{t,n}

    then take the **median** across neurons for each channel's global
    scale factor.  This is more robust than the pooled OLS because it
    avoids cancellation across neurons and doesn't need hard-coded
    floor fractions.
    """
    with torch.no_grad():
        resid, patterns, names = _collect_current_patterns(model, u)
        rms_resid = resid.pow(2).mean().sqrt().item()
        if rms_resid < 1e-12:
            return

        dev = model.I0.device
        K = len(patterns)
        T_1, N = resid.shape

        # (T-1, N, K) feature tensor
        X = torch.stack(patterns, dim=2)          # (T-1, N, K)
        # Per-neuron ridge: β_n = (X_n'X_n + αI)^{-1} X_n' r_n
        # Batch over neurons using (N, K, T) x (N, T, 1) → (N, K, 1)
        X_t = X.permute(1, 2, 0)                  # (N, K, T-1)
        r_t = resid.permute(1, 0).unsqueeze(2)    # (N, T-1, 1)
        XtX = torch.bmm(X_t, X_t.permute(0, 2, 1))  # (N, K, K)
        reg = ridge_alpha * torch.eye(K, device=dev).unsqueeze(0).expand(N, -1, -1)
        Xtr = torch.bmm(X_t, r_t)                    # (N, K, 1)
        beta_all = torch.linalg.solve(XtX + reg, Xtr).squeeze(2)  # (N, K)
        # Clamp to non-negative (physical: these are conductances/amplitudes)
        beta_all = beta_all.clamp(min=0)

        # Robust summary: median across neurons
        beta_median = beta_all.median(dim=0).values   # (K,)
        beta_mean   = beta_all.mean(dim=0)             # (K,)
        beta_q25    = beta_all.quantile(0.25, dim=0)
        beta_q75    = beta_all.quantile(0.75, dim=0)

        betas = [max(beta_median[i].item(), min_beta) for i in range(K)]
        raw_rms = [patterns[i].pow(2).mean().sqrt().item() for i in range(K)]
        rms_list = [raw_rms[i] * betas[i] for i in range(K)]

        # Log per-neuron distribution
        print(f"[init] network_scale (per_neuron_ridge, α={ridge_alpha:.1g}):")
        for i, name in enumerate(names):
            print(f"[init]   {name} β distribution: "
                  f"median={beta_median[i]:.4f}  mean={beta_mean[i]:.4f}  "
                  f"IQR=[{beta_q25[i]:.4f}, {beta_q75[i]:.4f}]")

        _apply_scales(model, names, betas, rms_list, rms_resid,
                      "per_neuron_ridge")


# ── Dispatch ────────────────────────────────────────────────────────────

def init_network_scale(model: "Stage2ModelPT", u: torch.Tensor,
                       cfg=None) -> None:
    mode = str(getattr(cfg, "network_init_mode", "ols") or "ols").lower()
    # support old names for backward compat
    if mode in ("per_neuron", "per_neuron_ridge"):
        _init_network_scale_per_neuron_ridge(model, u)
    elif mode in ("global", "ols"):
        _init_network_scale_ols(model, u)
    else:
        raise ValueError(f"Unknown network_init_mode={mode!r}; "
                         f"expected 'global'/'ols' or 'per_neuron'/'per_neuron_ridge'")


def init_all_from_data(model: "Stage2ModelPT", u: torch.Tensor, cfg=None):
    baseline = init_I0(model, u)
    init_reversals(model, u, baseline, cfg)
    init_network_scale(model, u, cfg)
