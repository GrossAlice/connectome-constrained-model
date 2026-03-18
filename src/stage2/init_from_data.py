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
    I0_ols = b / lam
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

def init_network_scale(model: "Stage2ModelPT", u: torch.Tensor,
                       min_network_frac: float = 0.25):
    with torch.no_grad():
        dev, lam = model.I0.device, model.lambda_u
        u = u.to(dev)
        T, N = u.shape

        resid = u[1:] - ((1 - lam) * u[:-1] + lam * model.I0)
        rms_resid = resid.pow(2).mean().sqrt().item()
        if rms_resid < 1e-12:
            return

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

        r_flat = resid.reshape(-1)
        C = torch.stack([p.reshape(-1) for p in patterns], dim=1)
        K = C.shape[1]
        CTC = C.T @ C + 1e-8 * torch.eye(K, device=dev)
        CTr = C.T @ r_flat
        beta = torch.linalg.solve(CTC, CTr).clamp(min=0)

        # Per-channel RMS after OLS scaling
        betas = [max(beta[i].item(), 1e-6) for i in range(K)]
        rms_list = [patterns[i].pow(2).mean().sqrt().item() * betas[i]
                    for i in range(K)]
        total_rms = math.sqrt(sum(r ** 2 for r in rms_list))

        # If network contribution is too small, boost uniformly
        target_rms = min_network_frac * rms_resid
        if total_rms < target_rms and total_rms > 1e-12:
            base_boost = target_rms / total_rms
        else:
            base_boost = 1.0

        scales = [base_boost for _ in range(K)]
        channel_floor_frac = {
            "G": 0.10,
            "a_sv": 0.06,
            "a_dcv": 0.03,
        }
        for i, name in enumerate(names):
            floor_frac = channel_floor_frac.get(name, 0.0)
            if floor_frac <= 0:
                continue
            rms_after_base = rms_list[i] * base_boost
            target_channel_rms = floor_frac * rms_resid
            if rms_after_base < target_channel_rms and rms_list[i] > 1e-12:
                scales[i] = max(scales[i], target_channel_rms / rms_list[i])

        final_rms_list = [rms_list[i] * scales[i] for i in range(K)]
        final_total_rms = math.sqrt(sum(r ** 2 for r in final_rms_list))

        print(f"[init] network_scale (OLS): AR1_rms={rms_resid:.4f}"
              f"  net_rms={final_total_rms:.4f} ({final_total_rms / rms_resid:.0%})"
              f"  boost={base_boost:.2f}")
        for i, name in enumerate(names):
            b = betas[i] * scales[i]
            cur = getattr(model, name)
            if cur.numel() == 0:
                continue
            rms_after = final_rms_list[i]
            frac = rms_after / rms_resid
            if name == "G" and model.edge_specific_G:
                from .model import _reparam_inv
                model._G_raw.data.copy_(_reparam_inv(cur * b, model._G_lo, model._G_hi))
            else:
                model.set_param_constrained(name, cur * b)
            print(f"[init]   {name}: \u03b2={b:.4f}, RMS={rms_after:.4f} ({frac:.0%} of AR1)")

def init_all_from_data(model: "Stage2ModelPT", u: torch.Tensor, cfg=None):
    baseline = init_I0(model, u)
    init_reversals(model, u, baseline, cfg)
    init_network_scale(model, u)
