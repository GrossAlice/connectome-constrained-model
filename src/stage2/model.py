"""Conductance-based neural dynamics model for C. elegans."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ── Reparameterization ────────────────────────────────────────────────

def _softplus_inv(y: torch.Tensor) -> torch.Tensor:
    return torch.where(y > 20.0, y, y + torch.log(-torch.expm1(-y)))

def _reparam_fwd(raw: torch.Tensor, lo: float, hi: Optional[float]) -> torch.Tensor:
    if hi is not None:
        return torch.sigmoid(raw) * (hi - lo) + lo
    return F.softplus(raw) + lo

def _reparam_inv(val: torch.Tensor, lo: float, hi: Optional[float]) -> torch.Tensor:
    if hi is not None:
        t = ((val - lo) / max(hi - lo, 1e-12)).clamp(1e-6, 1 - 1e-6)
        return torch.logit(t)
    return _softplus_inv((val - lo).clamp(min=1e-6))

def _match_rank(x: torch.Tensor, r: int) -> torch.Tensor:
    x = x.flatten()
    if r <= 0:
        return x[:0]
    if x.numel() == 1:
        return x.repeat(r)
    return x.repeat((r + x.numel() - 1) // x.numel())[:r]

def _build_per_source_reversals(T_sv, sign_t, e_exc, e_inh):
    mask = T_sv > 0
    total = mask.sum(1).float().clamp(min=1)
    E = (((sign_t > 0) & mask).sum(1).float() / total * e_exc
         + ((sign_t < 0) & mask).sum(1).float() / total * e_inh)
    return torch.where(total <= 1, torch.zeros_like(E), E)


class Stage2ModelPT(nn.Module):
    """Conductance-based neural dynamics: gap junctions + SV/DCV synapses."""

    def __init__(self, N: int, T_e: torch.Tensor, T_sv: torch.Tensor,
                 T_dcv: torch.Tensor, dt: float, cfg, device: torch.device,
                 d_ell: int = 0, lambda_u_init: Optional[torch.Tensor] = None,
                 sign_t: Optional[torch.Tensor] = None, logger=None) -> None:
        super().__init__()
        self.logger, self.N, self.dt, self.d_ell = logger, N, float(dt), d_ell
        self.r_sv, self.r_dcv = int(cfg.r_sv), int(cfg.r_dcv)
        self.u_clip = (getattr(cfg, "u_next_clip_min", -35.0),
                       getattr(cfg, "u_next_clip_max", 35.0))
        self.alpha_per_neuron = bool(getattr(cfg, "alpha_per_neuron", False))

        # Connectivity matrices
        self.register_buffer("T_e", T_e.to(device).float())
        self.register_buffer("T_sv", T_sv.to(device).float().abs())
        self.register_buffer("T_dcv", T_dcv.to(device).float().abs())

        # Synaptic weight matrices (learnable or fixed)
        for attr, T_mat, cfg_name in [("W_sv", self.T_sv, "W_sv_init"),
                                       ("W_dcv", self.T_dcv, "W_dcv_init")]:
            val = float(getattr(cfg, cfg_name, 1.0))
            W_raw = torch.full_like(T_mat, float(torch.log(torch.exp(torch.tensor(val)) - 1)))
            W_raw = W_raw * (T_mat > 0).float()
            learn = bool(getattr(cfg, f"learn_{attr}", False))
            if learn:
                setattr(self, f"_{attr}_raw", nn.Parameter(W_raw))
            else:
                self.register_buffer(f"_{attr}_raw", W_raw, persistent=False)

        # Lambda (leak/decay)
        lu = (lambda_u_init.to(device).float() if lambda_u_init is not None
              else torch.tensor(float(getattr(cfg, "lambda_u_fallback", 0.3)), device=device))
        self._lambda_u_lo, self._lambda_u_hi = 0.0, float(getattr(cfg, "lambda_u_max", 0.9999))
        self._lambda_u_raw = nn.Parameter(
            _reparam_inv(lu, self._lambda_u_lo, self._lambda_u_hi),
            requires_grad=bool(getattr(cfg, "learn_lambda_u", False)))

        # Gap-junction conductance
        self._G_lo = float(getattr(cfg, "G_min", 0.0))
        self._G_hi = float(cfg.G_max) if getattr(cfg, "G_max", None) else None
        G_init = torch.tensor(cfg.G_init, dtype=torch.float32, device=device)
        self._G_raw = nn.Parameter(_reparam_inv(G_init, self._G_lo, self._G_hi))

        # Tonic drive
        self.I0 = nn.Parameter(torch.zeros(N, device=device),
                               requires_grad=bool(getattr(cfg, "learn_I0", False)))

        # Stimulus weights
        self.stim_diagonal_only = d_ell > 0 and bool(getattr(cfg, "stim_diagonal_only", False)) and d_ell == N
        if d_ell > 0:
            shape = (N,) if self.stim_diagonal_only else (N, d_ell)
            self.b = nn.Parameter(torch.zeros(shape, device=device))
        else:
            self.register_buffer("b", torch.zeros((N, 1), device=device), persistent=False)

        # Synaptic amplitudes and time constants
        self._init_synaptic_params(cfg, device, N)
        # Reversal potentials
        self._init_reversals(cfg, sign_t, device)

    def _init_synaptic_params(self, cfg, device, N):
        for prefix, r in [("sv", self.r_sv), ("dcv", self.r_dcv)]:
            a_lo = float(getattr(cfg, f"a_{prefix}_min", 0.0))
            a_hi = float(v) if (v := getattr(cfg, f"a_{prefix}_max", None)) else None
            tau_lo = float(getattr(cfg, f"tau_{prefix}_min", 1e-4))
            tau_hi = float(v) if (v := getattr(cfg, f"tau_{prefix}_max", None)) else None
            setattr(self, f"_a_{prefix}_lo", a_lo); setattr(self, f"_a_{prefix}_hi", a_hi)
            setattr(self, f"_tau_{prefix}_lo", tau_lo); setattr(self, f"_tau_{prefix}_hi", tau_hi)

            if r > 0:
                a_init = _match_rank(torch.tensor(getattr(cfg, f"a_{prefix}_init")), r).to(device)
                if self.alpha_per_neuron:
                    a_init = a_init.unsqueeze(0).expand(N, -1).contiguous()
                setattr(self, f"_a_{prefix}_raw", nn.Parameter(_reparam_inv(a_init, a_lo, a_hi)))
                tau_init = _match_rank(torch.tensor(getattr(cfg, f"tau_{prefix}_init")), r).to(device)
                setattr(self, f"_tau_{prefix}_raw", nn.Parameter(
                    _reparam_inv(tau_init, tau_lo, tau_hi),
                    requires_grad=not bool(getattr(cfg, f"fix_tau_{prefix}", False))))
            else:
                z = torch.zeros(0, device=device)
                setattr(self, f"_a_{prefix}_raw", nn.Parameter(z.clone(), requires_grad=False))
                setattr(self, f"_tau_{prefix}_raw", nn.Parameter(z.clone(), requires_grad=False))

    def _init_reversals(self, cfg, sign_t, device):
        e_exc = float(cfg.E_sv_exc_init)
        e_inh = float(getattr(cfg, "E_sv_inh_init", -e_exc))
        if bool(getattr(cfg, "per_neuron_reversals", True)) and sign_t is not None:
            E_sv = _build_per_source_reversals(self.T_sv, sign_t.to(device).float(), e_exc, e_inh)
        else:
            E_sv = torch.tensor(e_exc, device=device)
        E_dcv = torch.tensor(float(getattr(cfg, "E_dcv_init", 0.0)), device=device)
        if bool(getattr(cfg, "learn_reversals", False)):
            self.E_sv, self.E_dcv = nn.Parameter(E_sv), nn.Parameter(E_dcv)
        else:
            self.register_buffer("E_sv", E_sv); self.register_buffer("E_dcv", E_dcv)

    # ── Properties (constrained parameters) ────────────────────────────

    def _get_W(self, name: str) -> torch.Tensor:
        return F.softplus(getattr(self, f"_{name}_raw"))

    @property
    def W_sv(self): return self._get_W("W_sv")
    @property
    def W_dcv(self): return self._get_W("W_dcv")
    @property
    def lambda_u(self): return _reparam_fwd(self._lambda_u_raw, self._lambda_u_lo, self._lambda_u_hi)
    @property
    def G(self): return _reparam_fwd(self._G_raw, self._G_lo, self._G_hi)
    @property
    def a_sv(self): return _reparam_fwd(self._a_sv_raw, self._a_sv_lo, self._a_sv_hi)
    @property
    def a_dcv(self): return _reparam_fwd(self._a_dcv_raw, self._a_dcv_lo, self._a_dcv_hi)
    @property
    def tau_sv(self): return _reparam_fwd(self._tau_sv_raw, self._tau_sv_lo, self._tau_sv_hi)
    @property
    def tau_dcv(self): return _reparam_fwd(self._tau_dcv_raw, self._tau_dcv_lo, self._tau_dcv_hi)

    def set_param_constrained(self, name: str, value: torch.Tensor) -> None:
        raw = getattr(self, f"_{name}_raw")
        lo, hi = getattr(self, f"_{name}_lo"), getattr(self, f"_{name}_hi")
        raw.data.copy_(_reparam_inv(value.to(raw.device), lo, hi))

    # ── Core dynamics ──────────────────────────────────────────────────

    def phi(self, u: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(u)

    def laplacian(self) -> torch.Tensor:
        W = self.T_e * self.G; W = (W + W.t()) * 0.5
        return W - torch.diag(W.sum(0))

    def _synaptic_current(self, u_prev, phi_gated, s, T_eff, a, tau, E):
        gamma = torch.exp(-self.dt / (tau + 1e-12))
        s_next = gamma.view(1, -1) * s + phi_gated.unsqueeze(1)
        pool = T_eff.t() @ s_next
        a_post = a.unsqueeze(0) if a.dim() == 1 else a
        g = (pool * a_post).sum(1)
        if E.dim() == 0:
            I = g * (E - u_prev)
        else:
            I = (T_eff.t() @ (s_next * E.view(-1, 1)) * a_post).sum(1) - g * u_prev
        return I, s_next

    def prior_step(self, u_prev: torch.Tensor, s_sv: torch.Tensor, s_dcv: torch.Tensor,
                   gating: torch.Tensor, stim: Optional[torch.Tensor] = None):
        """One-step dynamics: u(t) → u(t+1)."""
        N, device = self.N, u_prev.device
        phi_gated = self.phi(u_prev) * gating.view(N)

        I_sv = I_dcv = torch.zeros(N, device=device)
        if self.r_sv > 0:
            I_sv, s_sv = self._synaptic_current(u_prev, phi_gated, s_sv,
                self.T_sv * self._get_W("W_sv"), self.a_sv, self.tau_sv, self.E_sv)
        if self.r_dcv > 0:
            I_dcv, s_dcv = self._synaptic_current(u_prev, phi_gated, s_dcv,
                self.T_dcv * self._get_W("W_dcv"), self.a_dcv, self.tau_dcv, self.E_dcv)

        I_gap = self.laplacian() @ u_prev
        I_stim = torch.zeros(N, device=device)
        if self.d_ell > 0 and stim is not None:
            I_stim = (self.b * stim.view(N)) if self.stim_diagonal_only else (self.b @ stim)

        u_next = (1.0 - self.lambda_u) * u_prev + self.lambda_u * (self.I0 + I_gap + I_sv + I_dcv + I_stim)
        lo, hi = self.u_clip
        if lo is not None or hi is not None:
            u_next = u_next.clamp(min=lo, max=hi)
        return u_next, s_sv, s_dcv

