from __future__ import annotations
import numpy as np
import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from .model import Stage2ModelPT

@dataclass
class InitConfig:
    lambda_u_fallback: float = 0.3
    lambda_u_min: float = 0.001
    lambda_u_max: float = 0.999
    auto_reversals: bool = True
    reversal_q_lo: float = 0.01
    reversal_q_hi: float = 0.99

DEFAULT_INIT = InitConfig()

def estimate_mode_baseline(u: torch.Tensor) -> torch.Tensor:
    u_np = u.detach().cpu().numpy()
    modes = np.zeros(u_np.shape[1], dtype=np.float32)
    for i in range(u_np.shape[1]):
        col = u_np[:, i]; finite = col[np.isfinite(col)]
        if finite.size < 10: continue
        iqr = np.subtract(*np.percentile(finite, [75, 25]))
        bw = 2.0 * iqr / finite.size ** (1/3) if iqr > 0 else 0.1
        counts, edges = np.histogram(finite, bins=max(int(np.ptp(finite) / bw), 20))
        modes[i] = 0.5 * (edges[counts.argmax()] + edges[counts.argmax() + 1])
    return torch.tensor(modes, dtype=torch.float32, device=u.device)

def init_lambda_u(rho: Optional[torch.Tensor], N: int, cfg: InitConfig = DEFAULT_INIT) -> torch.Tensor:
    if rho is not None:
        lam = (1.0 - rho).clamp(cfg.lambda_u_min, cfg.lambda_u_max)
        print(f"[init] lambda_u: mean={lam.mean():.4f}, min={lam.min():.4f}, max={lam.max():.4f}")
    else:
        lam = torch.full((N,), cfg.lambda_u_fallback, dtype=torch.float32).clamp(cfg.lambda_u_min, cfg.lambda_u_max)
        print(f"[init] lambda_u fallback: {cfg.lambda_u_fallback:.4f}")
    return lam

def init_I0(model: "Stage2ModelPT", u: torch.Tensor) -> torch.Tensor:
    mode = estimate_mode_baseline(u).to(model.I0.device)
    with torch.no_grad(): model.I0.data.copy_(mode)
    print(f"[init] I0: mean={mode.mean():.4f}, min={mode.min():.4f}, max={mode.max():.4f}")
    return mode

def init_reversals(model: "Stage2ModelPT", u: torch.Tensor, baseline: torch.Tensor, cfg: InitConfig = DEFAULT_INIT):
    if not cfg.auto_reversals: return
    finite = u.detach().cpu().numpy().ravel(); finite = finite[np.isfinite(finite)]
    if finite.size < 10: return
    rest = float(torch.nanmedian(baseline.to(u.device)))
    e_inh, e_exc = float(np.quantile(finite, cfg.reversal_q_lo)), float(np.quantile(finite, cfg.reversal_q_hi))
    if not np.isfinite(rest): rest = float(np.median(finite))
    pad = max(1e-3, 0.1 * max(abs(rest), 1.0))
    if not np.isfinite(e_inh) or e_inh >= rest: e_inh = rest - pad
    if not np.isfinite(e_exc) or e_exc <= rest: e_exc = rest + pad
    with torch.no_grad():
        if model.E_sv.numel() > 1:
            sign = torch.sign(model.E_sv)
            model.E_sv.data.copy_(torch.where(sign > 0, e_exc, torch.where(sign < 0, e_inh, 0.0)))
        else:
            model.E_sv.data.fill_(e_exc)
        model.E_dcv.data.fill_(rest)
    print(f"[init] reversals: E_exc={e_exc:.4f}, E_inh={e_inh:.4f}, E_dcv={rest:.4f}")

def init_network_scale(model: "Stage2ModelPT", u: torch.Tensor, network_frac: float = 0.3):
    if network_frac <= 0: return
    with torch.no_grad():
        dev, lam = model.I0.device, model.lambda_u
        u = u.to(dev); T, N = u.shape
        rms_resid = (u[1:] - ((1 - lam) * u[:-1] + lam * model.I0)).pow(2).mean().sqrt().item()
        if rms_resid < 1e-12: return
        rms_target = network_frac * rms_resid
        s_sv, s_dcv = torch.zeros(N, model.r_sv, device=dev), torch.zeros(N, model.r_dcv, device=dev)
        g, L = torch.ones(N, device=dev), model.laplacian()
        sums = {"G": 0.0, "a_sv": 0.0, "a_dcv": 0.0}
        for t in range(T - 1):
            phi_g = model.phi(u[t]) * g
            sums["G"] += (lam * (L @ u[t])).pow(2).sum().item()
            if model.r_sv > 0:
                I, s_sv = model._synaptic_current(u[t], phi_g, s_sv, model.T_sv * model._get_W("W_sv"), model.a_sv, model.tau_sv, model.E_sv)
                sums["a_sv"] += (lam * I).pow(2).sum().item()
            if model.r_dcv > 0:
                I, s_dcv = model._synaptic_current(u[t], phi_g, s_dcv, model.T_dcv * model._get_W("W_dcv"), model.a_dcv, model.tau_dcv, model.E_dcv)
                sums["a_dcv"] += (lam * I).pow(2).sum().item()
        n_el = (T - 1) * N
        print(f"[init] network_scale: AR1_rms={rms_resid:.4f}, target={rms_target:.4f}")
        for name, ss in sums.items():
            cur = getattr(model, name)
            if cur.numel() == 0: continue
            rms = (ss / max(n_el, 1)) ** 0.5
            if rms > 1e-12:
                model.set_param_constrained(name, cur * (rms_target / rms))
                print(f"[init]   {name}: scale={rms_target/rms:.4f}")

def init_all_from_data(model: "Stage2ModelPT", u: torch.Tensor, cfg: InitConfig = DEFAULT_INIT, network_frac: float = 0.3):
    baseline = init_I0(model, u)
    init_reversals(model, u, baseline, cfg)
    init_network_scale(model, u, network_frac)
