from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

_G_INIT = 0.01
_G_MAX  = 2.0

#    E overwritten by init_reversals, W starts at neutral 1.0) ───────────────────
_A_SV_MAX  = 2.0
_A_DCV_MAX = 2.0
_W_SV_INIT = 1.0
_W_DCV_INIT = 1.0
_E_SV_EXC_INIT = 0.1
_E_SV_INH_INIT = -0.1
_E_DCV_INIT = 0.0

_LAMBDA_U_LO = 0.0
_LAMBDA_U_HI = 0.9999
_G_LO        = 0.0
_A_LO        = 0.0
_TAU_LO      = 1e-4
_W_LO        = 0.0


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


def _build_edge_specific_reversals(T_sv, sign_t, e_exc, e_inh):
    mask = T_sv > 0
    E = torch.where(
        sign_t > 0,
        torch.full_like(sign_t, float(e_exc)),
        torch.where(sign_t < 0, torch.full_like(sign_t, float(e_inh)), torch.zeros_like(sign_t)),
    )
    return E * mask.float()


class Stage2ModelPT(nn.Module):
    def __init__(self, N: int, T_e: torch.Tensor, T_sv: torch.Tensor,
                 T_dcv: torch.Tensor, dt: float, cfg, device: torch.device,
                 d_ell: int = 0, lambda_u_init: Optional[torch.Tensor] = None,
                 sign_t: Optional[torch.Tensor] = None, logger=None) -> None:
        super().__init__()
        self.logger, self.N, self.dt, self.d_ell = logger, N, float(dt), d_ell
        self.r_sv  = len(cfg.tau_sv_init)  if hasattr(cfg, 'tau_sv_init')  else int(getattr(cfg, 'r_sv', 0))
        self.r_dcv = len(cfg.tau_dcv_init) if hasattr(cfg, 'tau_dcv_init') else int(getattr(cfg, 'r_dcv', 0))
        self.u_clip = (getattr(cfg, "u_next_clip_min", -35.0),
                       getattr(cfg, "u_next_clip_max", 35.0))
        self.alpha_per_neuron = bool(getattr(cfg, "alpha_per_neuron", False))

        # Connectivity matrices
        self.register_buffer("T_e", T_e.to(device).float())
        self.register_buffer("T_sv", T_sv.to(device).float().abs())
        self.register_buffer("T_dcv", T_dcv.to(device).float().abs())

        # Synaptic weight matrices (learnable or fixed)
        W_sv_init_mode = str(getattr(cfg, "W_sv_init_mode", "uniform"))
        W_sv_normalize = bool(getattr(cfg, "W_sv_normalize", False))
        for attr, T_mat, init_val, mode, do_norm in [
                ("W_sv",  self.T_sv,  _W_SV_INIT,  W_sv_init_mode, W_sv_normalize),
                ("W_dcv", self.T_dcv, _W_DCV_INIT, "uniform",       False)]:
            val = init_val
            W_lo, W_hi = _W_LO, None
            setattr(self, f"_{attr}_lo", W_lo)
            setattr(self, f"_{attr}_hi", W_hi)
            mask = (T_mat > 0).float()
            safe_c = T_mat.clamp(min=1.0)          # avoid div-by-zero; only used where mask=1
            if mode == "log_counts":
                # T*W = log2(1+c)  →  W = log2(1+c)/c
                W_init = torch.where(mask.bool(),
                                     torch.log2(1.0 + T_mat) / safe_c,
                                     torch.ones_like(T_mat))  # safe value for non-edges
            elif mode == "sqrt_counts":
                # T*W = sqrt(c)  →  W = 1/sqrt(c)
                W_init = torch.where(mask.bool(),
                                     1.0 / safe_c.sqrt(),
                                     torch.ones_like(T_mat))  # safe value for non-edges
            else:   # "uniform" (default)
                W_init = torch.full_like(T_mat, val)
            if do_norm:
                # Row-normalise: make (T_mat * W_init).sum(dim=1) == 1 per post-syn neuron
                eff = T_mat * W_init
                row_sum = eff.sum(dim=1, keepdim=True).clamp(min=1e-8)
                W_init = W_init / row_sum
            W_raw = _reparam_inv(W_init, W_lo, W_hi)
            W_raw = W_raw * mask                   # zero raw weights for non-edges
            learn = bool(getattr(cfg, f"learn_{attr}", False))
            if learn:
                setattr(self, f"_{attr}_raw", nn.Parameter(W_raw))
            else:
                self.register_buffer(f"_{attr}_raw", W_raw, persistent=False)

        # Lambda (leak/decay)
        if lambda_u_init is None:
            raise ValueError("lambda_u_init is required (from OLS on smoothed data)")
        lu = lambda_u_init.to(device).float()
        self._lambda_u_lo = _LAMBDA_U_LO
        self._lambda_u_hi = _LAMBDA_U_HI
        self._lambda_u_raw = nn.Parameter(
            _reparam_inv(lu, self._lambda_u_lo, self._lambda_u_hi),
            requires_grad=bool(getattr(cfg, "learn_lambda_u", False)))

        # Gap-junction conductance
        self._G_lo = _G_LO
        self._G_hi = _G_MAX
        self.edge_specific_G = bool(getattr(cfg, "edge_specific_G", False))
        if self.edge_specific_G:
            mask_e  = (self.T_e > 0).float()
            safe_te = self.T_e.clamp(min=1.0)
            G_init_mode = str(getattr(cfg, "G_init_mode", "uniform"))
            if G_init_mode == "log_counts":
                # T_e * G = log2(1+c)  →  G = log2(1+c)/c
                G_mat = torch.where(mask_e.bool(),
                                    torch.log2(1.0 + self.T_e) / safe_te,
                                    torch.full_like(self.T_e, _G_INIT))
            elif G_init_mode == "sqrt_counts":
                # T_e * G = sqrt(c)  →  G = 1/sqrt(c)
                G_mat = torch.where(mask_e.bool(),
                                    1.0 / safe_te.sqrt(),
                                    torch.full_like(self.T_e, _G_INIT))
            else:   # "uniform"
                G_mat = torch.full_like(self.T_e, _G_INIT)
            G_mat = G_mat * mask_e
            self._G_raw = nn.Parameter(_reparam_inv(G_mat, self._G_lo, self._G_hi))
        else:
            G_init = torch.tensor(_G_INIT, dtype=torch.float32, device=device)
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
            a_lo = _A_LO
            a_hi = _A_SV_MAX if prefix == "sv" else _A_DCV_MAX
            tau_lo = _TAU_LO
            tau_hi = None
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
        e_exc = _E_SV_EXC_INIT
        e_inh = _E_SV_INH_INIT
        mode = str(getattr(cfg, "reversal_mode", "per_neuron"))
        if mode == "per_edge" and sign_t is not None:
            E_sv = _build_edge_specific_reversals(self.T_sv, sign_t.to(device).float(), e_exc, e_inh)
        elif mode == "per_neuron" and sign_t is not None:
            E_sv = _build_per_source_reversals(self.T_sv, sign_t.to(device).float(), e_exc, e_inh)
        else:
            E_sv = torch.tensor(e_exc, device=device)
        E_dcv = torch.tensor(_E_DCV_INIT, device=device)
        if bool(getattr(cfg, "learn_reversals", False)):
            self.E_sv, self.E_dcv = nn.Parameter(E_sv), nn.Parameter(E_dcv)
        else:
            self.register_buffer("E_sv", E_sv); self.register_buffer("E_dcv", E_dcv)


    def _get_W(self, name: str) -> torch.Tensor:
        lo = getattr(self, f"_{name}_lo")
        hi = getattr(self, f"_{name}_hi")
        return _reparam_fwd(getattr(self, f"_{name}_raw"), lo, hi)

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


    def phi(self, u: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(u)

    def laplacian_with_G(self, G: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Gap-junction Laplacian, optionally overriding the shared conductance G.

        Parameters
        ----------
        G : Tensor, optional
            Per-call conductance override (scalar or edge-specific matrix).
            When ``None``, falls back to ``self.G``.
        """
        G_eff = self.G if G is None else G.to(device=self.T_e.device, dtype=self.T_e.dtype)
        if self.edge_specific_G:
            W = G_eff * (self.T_e > 0).float()
        else:
            W = self.T_e * G_eff
        W = (W + W.t()) * 0.5
        return W - torch.diag(W.sum(0))

    def laplacian(self) -> torch.Tensor:
        return self.laplacian_with_G()

    def _synaptic_current(self, u_prev, phi_gated, s, T_eff, a, tau, E):
        gamma = torch.exp(-self.dt / (tau + 1e-12))
        s_next = gamma.view(1, -1) * s + phi_gated.unsqueeze(1)
        pool = T_eff.t() @ s_next
        a_post = a.unsqueeze(0) if a.dim() == 1 else a
        g = (pool * a_post).sum(1)
        if E.dim() == 0:
            I = g * (E - u_prev)
        elif E.dim() == 1:
            I = (T_eff.t() @ (s_next * E.view(-1, 1)) * a_post).sum(1) - g * u_prev
        else:
            I = (((T_eff * E).t() @ (s_next * a_post)).sum(1) - g * u_prev)
        return I, s_next

    def prior_step(
        self,
        u_prev: torch.Tensor,
        s_sv: torch.Tensor,
        s_dcv: torch.Tensor,
        gating: torch.Tensor,
        stim: Optional[torch.Tensor] = None,
        lambda_u: Optional[torch.Tensor] = None,
        I0: Optional[torch.Tensor] = None,
        G: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None,
    ):
        N, device = self.N, u_prev.device
        phi_gated = self.phi(u_prev) * gating.view(N)
        lambda_u_eff = self.lambda_u if lambda_u is None else lambda_u.to(device=device, dtype=u_prev.dtype)
        I0_eff = self.I0 if I0 is None else I0.to(device=device, dtype=u_prev.dtype)

        I_sv = I_dcv = torch.zeros(N, device=device)
        if self.r_sv > 0:
            I_sv, s_sv = self._synaptic_current(u_prev, phi_gated, s_sv,
                self.T_sv * self._get_W("W_sv"), self.a_sv, self.tau_sv, self.E_sv)
        if self.r_dcv > 0:
            I_dcv, s_dcv = self._synaptic_current(u_prev, phi_gated, s_dcv,
                self.T_dcv * self._get_W("W_dcv"), self.a_dcv, self.tau_dcv, self.E_dcv)

        I_gap = self.laplacian_with_G(G) @ u_prev
        I_stim = torch.zeros(N, device=device)
        if b is not None and stim is not None:
            # Multi-worm path: per-worm diagonal stimulus weights
            I_stim = b.to(device=device, dtype=u_prev.dtype) * stim.view(N)
        elif self.d_ell > 0 and stim is not None:
            # Single-worm path: model's own b
            I_stim = (self.b * stim.view(N)) if self.stim_diagonal_only else (self.b @ stim)

        u_next = (1.0 - lambda_u_eff) * u_prev + lambda_u_eff * (I0_eff + I_gap + I_sv + I_dcv + I_stim)
        lo, hi = self.u_clip
        if lo is not None or hi is not None:
            u_next = u_next.clamp(min=lo, max=hi)
        return u_next, s_sv, s_dcv

