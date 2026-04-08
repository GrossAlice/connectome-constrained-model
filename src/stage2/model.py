from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

_G_INIT = 0.01
_G_MAX  = 2.0

_A_SV_MAX  = 10.0
_A_DCV_MAX = 10.0
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


class Stage2ModelPT(nn.Module):
    def __init__(self, N: int, T_e: torch.Tensor, T_sv: torch.Tensor,
                 T_dcv: torch.Tensor, dt: float, cfg, device: torch.device,
                 d_ell: int = 0, lambda_u_init: Optional[torch.Tensor] = None,
                 sign_t: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.N, self.dt, self.d_ell = N, float(dt), d_ell
        self.r_sv  = len(cfg.tau_sv_init)  if hasattr(cfg, 'tau_sv_init')  else int(getattr(cfg, 'r_sv', 0))
        self.r_dcv = len(cfg.tau_dcv_init) if hasattr(cfg, 'tau_dcv_init') else int(getattr(cfg, 'r_dcv', 0))
        self.u_clip = (getattr(cfg, "u_next_clip_min", -35.0),
                       getattr(cfg, "u_next_clip_max", 35.0))
        self.alpha_per_neuron = bool(
            getattr(cfg, "per_neuron_amplitudes",
                    getattr(cfg, "alpha_per_neuron", False)))
        self._chem_act = str(
            getattr(cfg, "chemical_synapse_activation", "sigmoid"))
        if self._chem_act == "shifted_sigmoid":
            self._phi_alpha = nn.Parameter(torch.tensor(1.0, device=device))
            self._phi_beta  = nn.Parameter(torch.zeros(N, device=device))

        # FIR chemical synapse mode
        self._chem_mode = str(getattr(cfg, "chemical_synapse_mode", "iir"))
        self._fir_kernel_len = int(getattr(cfg, "fir_kernel_len", 5))
        self._fir_act = str(getattr(cfg, "fir_activation", "identity"))
        self._fir_include_reversal = bool(getattr(cfg, "fir_include_reversal", False))

        self.register_buffer("T_e", T_e.to(device).float())
        self.register_buffer("T_sv", T_sv.to(device).float().abs())
        self.register_buffer("T_dcv", T_dcv.to(device).float().abs())

        for attr, T_mat, init_val in [
                ("W_sv",  self.T_sv,  _W_SV_INIT),
                ("W_dcv", self.T_dcv, _W_DCV_INIT)]:
            W_lo, W_hi = _W_LO, None
            setattr(self, f"_{attr}_lo", W_lo)
            setattr(self, f"_{attr}_hi", W_hi)
            mask = (T_mat > 0).float()
            self.register_buffer(f"_{attr}_mask", mask, persistent=False)
            W_init = torch.full_like(T_mat, init_val)
            W_raw = _reparam_inv(W_init, W_lo, W_hi)
            W_raw = W_raw * mask
            learn = bool(getattr(cfg, f"learn_{attr}", False))
            if learn:
                param = nn.Parameter(W_raw)
                setattr(self, f"_{attr}_raw", param)
                param.register_hook(lambda g, m=mask: g * m)
            else:
                self.register_buffer(f"_{attr}_raw", W_raw, persistent=False)

        if lambda_u_init is None:
            lambda_u_init = torch.full((N,), 0.1)
        lu = lambda_u_init.to(device).float()
        self._lambda_u_lo = float(getattr(cfg, "lambda_u_lo", _LAMBDA_U_LO))
        self._lambda_u_hi = float(getattr(cfg, "lambda_u_hi", _LAMBDA_U_HI))
        self._lambda_u_raw = nn.Parameter(
            _reparam_inv(lu, self._lambda_u_lo, self._lambda_u_hi),
            requires_grad=bool(getattr(cfg, "learn_lambda_u", False)))

        self._G_lo = _G_LO
        self._G_hi = _G_MAX
        self.edge_specific_G = bool(getattr(cfg, "edge_specific_G", False))
        if self.edge_specific_G:
            mask_g = (self.T_e > 0).float()
            self.register_buffer("_G_mask", mask_g, persistent=False)
            G_mat = torch.full_like(self.T_e, _G_INIT) * mask_g
            G_raw = _reparam_inv(G_mat, self._G_lo, self._G_hi) * mask_g
            self._G_raw = nn.Parameter(G_raw)
            self._G_raw.register_hook(lambda g, m=mask_g: g * m)
        else:
            G_init = torch.tensor(_G_INIT, dtype=torch.float32, device=device)
            self._G_raw = nn.Parameter(_reparam_inv(G_init, self._G_lo, self._G_hi))

        self.I0 = nn.Parameter(torch.zeros(N, device=device),
                               requires_grad=bool(getattr(cfg, "learn_I0", False)))

        self.stim_diagonal_only = d_ell > 0 and bool(getattr(cfg, "stim_diagonal_only", False)) and d_ell == N
        if d_ell > 0:
            shape = (N,) if self.stim_diagonal_only else (N, d_ell)
            self.b = nn.Parameter(torch.zeros(shape, device=device))
        else:
            self.register_buffer("b", torch.zeros((N, 1), device=device), persistent=False)

        self.stim_kernel_len = int(getattr(cfg, "stim_kernel_len", 0))
        if self.stim_kernel_len > 0 and d_ell > 0:
            tau_init = float(getattr(cfg, "stim_kernel_tau_init", 2.0))
            k = torch.arange(self.stim_kernel_len, dtype=torch.float32, device=device)
            h_init = torch.exp(-k * self.dt / max(tau_init, 1e-6))
            h_init = h_init / h_init.sum().clamp(min=1e-8)
            self.stim_kernel = nn.Parameter(_softplus_inv(h_init.clamp(min=1e-6)))
        else:
            self.stim_kernel = None

        self._init_synaptic_params(cfg, device, N)
        self._init_fir_params(cfg, device, N)
        self._init_reversals(cfg, sign_t, device)

        self._learn_noise = bool(getattr(cfg, "learn_noise", False))
        self._noise_floor = float(getattr(cfg, "noise_floor", 1e-3))
        self._noise_mode = str(getattr(cfg, "noise_mode", "homoscedastic"))
        if self._learn_noise and self._noise_mode == "heteroscedastic":
            self._sigma_w = nn.Parameter(torch.zeros(N, device=device))
            self._sigma_b = nn.Parameter(
                _softplus_inv(torch.full((N,), 0.1, device=device)))
            self.register_buffer(
                "_log_sigma_u",
                torch.zeros(N, device=device),
                persistent=False,
            )
        elif self._learn_noise:
            self._log_sigma_u = nn.Parameter(
                _softplus_inv(torch.full((N,), 0.1, device=device)))
        else:
            self.register_buffer(
                "_log_sigma_u",
                torch.zeros(N, device=device),
                persistent=False,
            )

        self.noise_corr_rank = int(getattr(cfg, 'noise_corr_rank', 0))
        if self.noise_corr_rank > 0:
            R = self.noise_corr_rank
            self._noise_V = nn.Parameter(
                torch.randn(N, R, device=device) * 0.01)

        self.lowrank_rank = int(getattr(cfg, 'lowrank_rank', 0))
        if self.lowrank_rank > 0:
            K = self.lowrank_rank
            self.U_lowrank = nn.Parameter(
                torch.randn(K, N, device=device) * (2.0 / (K + N)) ** 0.5)
            self.V_lowrank = nn.Parameter(
                torch.randn(N, K, device=device) * (2.0 / (K + N)) ** 0.5)

        self.graph_poly_order = int(getattr(cfg, 'graph_poly_order', 1))
        if self.graph_poly_order > 1:
            self.graph_poly_alpha = nn.Parameter(
                torch.zeros(self.graph_poly_order - 1, device=device))

        self._coupling_dropout = float(getattr(cfg, 'coupling_dropout', 0.0))

        self._lag_order = int(getattr(cfg, 'lag_order', 0))
        self._lag_neighbor = bool(getattr(cfg, 'lag_neighbor', False))
        self._lag_conn_mask = str(getattr(cfg, 'lag_connectome_mask', 'T_e'))
        self._lag_nbr_act = str(getattr(cfg, 'lag_neighbor_activation', 'none'))
        self._lag_nbr_per_type = bool(getattr(cfg, 'lag_neighbor_per_type', False))
        if self._lag_order > 0:
            K_lag = self._lag_order
            self._lag_alpha = nn.Parameter(
                torch.zeros(K_lag, N, device=device))
            self.register_buffer(
                '_lag_history_buf',
                torch.zeros(K_lag, N, device=device))
            if self._lag_neighbor:
                if self._lag_nbr_per_type:
                    # Separate per-type neighbor lag weights
                    self._lag_nbr_types = []  # list of prefixes with >0 edges
                    for prefix, T_mat in [('e', self.T_e), ('sv', self.T_sv), ('dcv', self.T_dcv)]:
                        mask = (T_mat.abs() > 0).float()
                        if mask.sum() == 0:
                            continue
                        self._lag_nbr_types.append(prefix)
                        self.register_buffer(f'_lag_nbr_mask_{prefix}', mask)
                        param = nn.Parameter(torch.zeros(K_lag, N, N, device=device))
                        setattr(self, f'_lag_G_{prefix}', param)
                        param.register_hook(
                            lambda g, m=mask: g * m.unsqueeze(0).expand_as(g))
                else:
                    # Single union mask (original behaviour)
                    if self._lag_conn_mask == 'all':
                        mask = ((self.T_e.abs() + self.T_sv + self.T_dcv) > 0).float()
                    else:
                        mask = (self.T_e > 0).float()
                    self.register_buffer('_lag_nbr_mask', mask)
                    self._lag_G = nn.Parameter(
                        torch.zeros(K_lag, N, N, device=device))
                    self._lag_G.register_hook(
                        lambda g, m=mask: g * m.unsqueeze(0).expand_as(g))

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
                fix_a = bool(getattr(cfg, f"fix_a_{prefix}", False))
                setattr(self, f"_a_{prefix}_raw", nn.Parameter(
                    _reparam_inv(a_init, a_lo, a_hi), requires_grad=not fix_a))
                tau_init = _match_rank(torch.tensor(getattr(cfg, f"tau_{prefix}_init")), r).to(device)
                setattr(self, f"_tau_{prefix}_raw", nn.Parameter(
                    _reparam_inv(tau_init, tau_lo, tau_hi),
                    requires_grad=not bool(getattr(cfg, f"fix_tau_{prefix}", False))))
            else:
                z = torch.zeros(0, device=device)
                setattr(self, f"_a_{prefix}_raw", nn.Parameter(z.clone(), requires_grad=False))
                setattr(self, f"_tau_{prefix}_raw", nn.Parameter(z.clone(), requires_grad=False))

    def _init_fir_params(self, cfg, device, N):
        """Initialise per-edge FIR kernel parameters for chemical synapses.

        Creates W_fir_sv (K, N, N) and W_fir_dcv (K, N, N), masked by
        the respective connectome adjacency.  Only allocated when
        chemical_synapse_mode == 'fir'.
        """
        if self._chem_mode != 'fir':
            return
        K = self._fir_kernel_len
        for prefix, T_mat in [('sv', self.T_sv), ('dcv', self.T_dcv)]:
            mask = (T_mat > 0).float()                     # (N, N)
            n_edges = mask.sum().item()
            if n_edges == 0:
                self.register_buffer(f'_W_fir_{prefix}',
                                     torch.zeros(K, N, N, device=device))
                continue
            # Xavier-like init scaled by 1/sqrt(avg_fan_in)
            fan = max(mask.sum(0).mean().item(), 1.0)
            scale = (1.0 / (fan * K)) ** 0.5
            W_init = torch.randn(K, N, N, device=device) * scale
            W_init = W_init * mask.unsqueeze(0)              # sparse
            param = nn.Parameter(W_init)
            setattr(self, f'_W_fir_{prefix}', param)
            param.register_hook(
                lambda g, m=mask: g * m.unsqueeze(0).expand_as(g))
            self.register_buffer(f'_W_fir_{prefix}_mask', mask,
                                 persistent=False)

    def _init_reversals(self, cfg, sign_t, device):
        if sign_t is not None:
            self.register_buffer("sign_t", sign_t.to(device).float(),
                                 persistent=False)
        else:
            self.register_buffer("sign_t", None)
        mode = str(getattr(cfg, "reversal_mode", "per_neuron"))
        if mode == "per_edge" and sign_t is not None:
            E_sv = torch.zeros(self.N, self.N, device=device)
        elif mode == "per_neuron" and sign_t is not None:
            E_sv = torch.zeros(self.N, device=device)
        else:
            E_sv = torch.tensor(0.0, device=device)
        E_dcv = torch.tensor(0.0, device=device)
        if bool(getattr(cfg, "learn_reversals", False)):
            self.E_sv, self.E_dcv = nn.Parameter(E_sv), nn.Parameter(E_dcv)
            if E_sv.dim() == 2:
                mask_e = (self.T_sv > 0).float()
                self.register_buffer("_E_sv_mask", mask_e, persistent=False)
                self.E_sv.register_hook(lambda g, m=mask_e: g * m)
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

    def sigma_at(self, u: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self._learn_noise and self._noise_mode == "heteroscedastic":
            if u is None:
                return F.softplus(self._sigma_b) + self._noise_floor
            return F.softplus(self._sigma_w * u + self._sigma_b) + self._noise_floor
        if self._learn_noise:
            return F.softplus(self._log_sigma_u) + self._noise_floor
        return torch.full((self.N,), self._noise_floor,
                          device=self._log_sigma_u.device)

    @property
    def sigma_process(self) -> torch.Tensor:
        return self.sigma_at(None)

    def sample_correlated_noise(self, sigma_diag: torch.Tensor) -> torch.Tensor:
        device = sigma_diag.device
        eps_diag = torch.randn_like(sigma_diag) * sigma_diag
        if self.noise_corr_rank > 0:
            R = self.noise_corr_rank
            z = torch.randn(*sigma_diag.shape[:-1], R, device=device)
            eps_corr = z @ self._noise_V.t()
            return eps_diag + eps_corr
        return eps_diag

    def set_param_constrained(self, name: str, value: torch.Tensor) -> None:
        raw = getattr(self, f"_{name}_raw")
        lo, hi = getattr(self, f"_{name}_lo"), getattr(self, f"_{name}_hi")
        new_raw = _reparam_inv(value.to(raw.device), lo, hi)
        mask_buf = f"_{name}_mask"
        if hasattr(self, mask_buf):
            new_raw = new_raw * getattr(self, mask_buf)
        raw.data.copy_(new_raw)


    def precompute_params(self) -> dict:
        p: dict = {}
        p["lambda_u"] = self.lambda_u
        p["I0"] = self.I0
        p["L"] = self.laplacian()
        if self._chem_mode == 'fir':
            if hasattr(self, '_W_fir_sv'):
                p["W_fir_sv"] = self._W_fir_sv
                p["E_sv"] = self.E_sv
            if hasattr(self, '_W_fir_dcv'):
                p["W_fir_dcv"] = self._W_fir_dcv
                p["E_dcv"] = self.E_dcv
        else:
            if self.r_sv > 0:
                p["T_sv_eff"] = self.T_sv * self._get_W("W_sv")
                p["a_sv"] = self.a_sv
                p["tau_sv"] = self.tau_sv
                p["E_sv"] = self.E_sv
            if self.r_dcv > 0:
                p["T_dcv_eff"] = self.T_dcv * self._get_W("W_dcv")
                p["a_dcv"] = self.a_dcv
                p["tau_dcv"] = self.tau_dcv
                p["E_dcv"] = self.E_dcv
        return p

    def forward_sequence(
        self,
        u: torch.Tensor,
        gating_data: Optional[torch.Tensor] = None,
        stim_data: Optional[torch.Tensor] = None,
        params: Optional[dict] = None,
    ) -> torch.Tensor:
        if self.graph_poly_order > 1 or self.lowrank_rank > 0:
            return self._forward_sequence_fallback(u, gating_data, stim_data)

        T, N = u.shape
        device = u.device
        if params is None:
            params = self.precompute_params()

        lambda_u = params["lambda_u"]
        I0 = params["I0"]
        L = params["L"]
        lo, hi = self.u_clip

        use_fir = (self._chem_mode == 'fir')
        has_sv = self.r_sv > 0 if not use_fir else hasattr(self, '_W_fir_sv')
        has_dcv = self.r_dcv > 0 if not use_fir else hasattr(self, '_W_fir_dcv')

        # IIR state vectors (only used in IIR mode)
        s_sv = torch.zeros(N, self.r_sv, device=device) if not use_fir else None
        s_dcv = torch.zeros(N, self.r_dcv, device=device) if not use_fir else None

        preds = torch.zeros_like(u)
        preds[0] = u[0]

        # Pre-compute activations for IIR or FIR
        if use_fir:
            phi_fir_all = self.phi_fir(u)              # (T, N)
            if gating_data is not None:
                phi_fir_all = phi_fir_all * gating_data
            # Pre-compute FIR buffers: pad front with zeros for causality
            K_fir = self._fir_kernel_len
            phi_pad = F.pad(phi_fir_all, (0, 0, K_fir, 0))  # (K_fir+T, N)
            # fir_bufs[t] = phi for lags 0..K_fir-1 before time t
            # At time t, the synaptic input uses phi(u[t-1]), phi(u[t-2]), ...
            # phi_pad is indexed: phi_pad[K_fir + t_orig] = phi_fir_all[t_orig]
            # For lag k at time t: phi(u[t-1-k]) = phi_pad[K_fir + t - 1 - k]
        else:
            phi_all = self.phi(u)
            if gating_data is not None:
                phi_gated_all = phi_all * gating_data
            else:
                phi_gated_all = phi_all

        I_gap_all = (L @ u.T).T

        I_stim_all = None
        if self.d_ell > 0 and stim_data is not None:
            if self.stim_diagonal_only:
                I_stim_all = self.b * stim_data
            else:
                I_stim_all = stim_data @ self.b.T

        has_lag = self._lag_order > 0
        lag_self_all = None
        lag_nbr_all = None
        if has_lag:
            K_lag = self._lag_order
            u_lag_pad = F.pad(u, (0, 0, K_lag, 0))
            lag_buf = torch.stack(
                [u_lag_pad[K_lag - 1 - k : K_lag - 1 - k + T] for k in range(K_lag)],
                dim=1,
            )
            lag_self_all = (self._lag_alpha.unsqueeze(0) * lag_buf).sum(1)
            if self._lag_neighbor:
                if self._lag_nbr_per_type:
                    # Per-type: separate weights, activation only on chemical
                    lag_nbr_all = torch.zeros(T, N, device=device)
                    nbr_buf_act = self._apply_lag_nbr_act(lag_buf)  # for sv/dcv
                    for prefix in getattr(self, '_lag_nbr_types', []):
                        G_p = getattr(self, f'_lag_G_{prefix}')
                        mask_p = getattr(self, f'_lag_nbr_mask_{prefix}')
                        G_masked = G_p * mask_p.unsqueeze(0)
                        # Gap junctions: linear; chemical (sv/dcv): activation
                        buf_p = lag_buf if prefix == 'e' else nbr_buf_act
                        lag_nbr_all = lag_nbr_all + torch.bmm(
                            buf_p.permute(1, 0, 2),
                            G_masked.transpose(1, 2),
                        ).sum(0)
                elif hasattr(self, '_lag_G'):
                    G_masked = self._lag_G * self._lag_nbr_mask.unsqueeze(0)
                    nbr_buf = self._apply_lag_nbr_act(lag_buf)
                    lag_nbr_all = torch.bmm(
                        nbr_buf.permute(1, 0, 2),
                        G_masked.transpose(1, 2),
                    ).sum(0)

        zero_N = torch.zeros(N, device=device)

        for t in range(1, T):
            u_prev = u[t - 1]

            I_sv = I_dcv = zero_N

            if use_fir:
                # Build phi_buf (K_fir, N): phi(u[t-1]), phi(u[t-2]), ...
                # phi_pad offset: phi_pad[K_fir + t - 1 - k] for k=0..K-1
                idx_start = t - 1          # maps to phi_pad[K_fir + t - 1]
                phi_buf = torch.stack(
                    [phi_pad[K_fir + idx_start - k] for k in range(K_fir)]
                )  # (K_fir, N)
                if has_sv:
                    I_sv = self._synaptic_current_fir(
                        u_prev, phi_buf, params["W_fir_sv"], params["E_sv"])
                if has_dcv:
                    I_dcv = self._synaptic_current_fir(
                        u_prev, phi_buf, params["W_fir_dcv"], params["E_dcv"])
            else:
                phi_gated = phi_gated_all[t - 1]
                if has_sv:
                    I_sv, s_sv = self._synaptic_current(
                        u_prev, phi_gated, s_sv,
                        params["T_sv_eff"], params["a_sv"],
                        params["tau_sv"], params["E_sv"])
                if has_dcv:
                    I_dcv, s_dcv = self._synaptic_current(
                        u_prev, phi_gated, s_dcv,
                        params["T_dcv_eff"], params["a_dcv"],
                        params["tau_dcv"], params["E_dcv"])

            I_gap = I_gap_all[t - 1]

            I_stim = I_stim_all[t - 1] if I_stim_all is not None else zero_N

            I_coupling = I_gap + I_sv + I_dcv
            if self._coupling_dropout > 0 and self.training:
                I_coupling = F.dropout(I_coupling, p=self._coupling_dropout, training=True)

            I_lag = zero_N
            if has_lag:
                I_lag = lag_self_all[t]
                if lag_nbr_all is not None:
                    I_lag = I_lag + lag_nbr_all[t]

            u_next = (1.0 - lambda_u) * u_prev + lambda_u * (I0 + I_coupling + I_stim + I_lag)
            if lo is not None or hi is not None:
                u_next = u_next.clamp(min=lo, max=hi)
            preds[t] = u_next

        return preds

    def _forward_sequence_fallback(
        self,
        u: torch.Tensor,
        gating_data: Optional[torch.Tensor] = None,
        stim_data: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        T, N = u.shape
        device = u.device
        ones = torch.ones(N, device=device)
        s_sv = torch.zeros(N, self.r_sv, device=device)
        s_dcv = torch.zeros(N, self.r_dcv, device=device)
        self.reset_lag_history()
        self.reset_fir_history()
        preds = torch.zeros_like(u)
        preds[0] = u[0]
        for t in range(1, T):
            g = gating_data[t - 1] if gating_data is not None else ones
            s = stim_data[t - 1] if stim_data is not None else None
            preds[t], s_sv, s_dcv = self.prior_step(u[t - 1], s_sv, s_dcv, g, s)
        return preds


    def phi(self, u: torch.Tensor) -> torch.Tensor:
        act = self._chem_act
        if act == "identity":
            return u
        if act == "sigmoid":
            return torch.sigmoid(u)
        if act == "tanh":
            return torch.tanh(u)
        if act == "softplus":
            return F.softplus(u)
        if act == "relu":
            return F.relu(u)
        if act == "elu":
            return F.elu(u)
        if act == "swish":
            return u * torch.sigmoid(u)
        if act == "shifted_sigmoid":
            return torch.sigmoid(self._phi_alpha * (u - self._phi_beta))
        return torch.sigmoid(u)

    def phi_fir(self, u: torch.Tensor) -> torch.Tensor:
        """Activation applied before FIR synaptic filter."""
        act = self._fir_act
        if act == "identity":
            return u
        if act == "sigmoid":
            return torch.sigmoid(u)
        if act == "softplus":
            return F.softplus(u)
        if act == "tanh":
            return torch.tanh(u)
        if act == "relu":
            return F.relu(u)
        return u  # default: identity

    def _synaptic_current_fir(self, u_prev, phi_buf, W_fir, E):
        """FIR chemical synapse: convolve per-edge kernel with activation buffer.

        Parameters
        ----------
        u_prev : (N,) current membrane potential (for driving-force term)
        phi_buf : (K, N) ring-buffer of phi_fir(u) for the last K steps
        W_fir : (K, N, N) per-edge, per-lag weight tensor (masked sparse)
        E : reversal potential (scalar, (N,), or (N,N))

        Returns
        -------
        I : (N,) post-synaptic current
        """
        # I_i = sum_k sum_j W_fir[k, j, i] * phi_buf[k, j]
        # Efficiently: bmm  (K, N_post, N_pre) @ (K, N_pre, 1) -> (K, N_post, 1)
        I = torch.bmm(
            W_fir.transpose(1, 2),       # (K, N, N)
            phi_buf.unsqueeze(2),         # (K, N, 1)
        ).squeeze(2).sum(0)               # (N,)

        if self._fir_include_reversal:
            if E.dim() == 0:
                I = I * (E - u_prev)
            elif E.dim() == 1:
                I = I * (E - u_prev)
            # per-edge reversal not supported in FIR mode
        return I

    def reset_lag_history(self):
        if self._lag_order > 0 and hasattr(self, '_lag_history_buf'):
            self._lag_history_buf.zero_()
        self._lag_history_live = None

    def init_lag_history(self, u_seq: torch.Tensor, t: int):
        if self._lag_order <= 0 or not hasattr(self, '_lag_history_buf'):
            return
        K = self._lag_order
        buf = self._lag_history_buf
        for k in range(K):
            idx = t - 1 - k
            if 0 <= idx < u_seq.shape[0]:
                buf.data[k].copy_(u_seq[idx].detach())
            else:
                buf.data[k].zero_()
        self._lag_history_live = None

    def _apply_lag_nbr_act(self, buf: torch.Tensor) -> torch.Tensor:
        """Apply optional nonlinearity to neighbor lag buffer (K, N) or (T, K, N)."""
        act = self._lag_nbr_act
        if act in ('none', 'identity', ''):
            return buf
        if act == 'sigmoid':
            return torch.sigmoid(buf)
        if act == 'softplus':
            return F.softplus(buf)
        if act == 'tanh':
            return torch.tanh(buf)
        if act == 'relu':
            return F.relu(buf)
        return buf

    def _lag_push_and_compute(self, u_prev: torch.Tensor) -> torch.Tensor:
        if self._lag_order <= 0:
            return torch.zeros(self.N, device=u_prev.device)
        K = self._lag_order

        # Read from differentiable live buffer when available
        live = getattr(self, '_lag_history_live', None)
        if live is None:
            live = self._lag_history_buf

        old = live[:-1]
        new_buf = torch.cat([u_prev.unsqueeze(0), old], dim=0)

        # Keep differentiable reference for the next call (rollout)
        self._lag_history_live = new_buf
        # Sync the registered buffer (detached)
        self._lag_history_buf.data.copy_(new_buf.data.detach())

        I_lag = (self._lag_alpha * new_buf).sum(0)
        if self._lag_neighbor:
            if self._lag_nbr_per_type:
                nbr_buf_act = self._apply_lag_nbr_act(new_buf)
                for prefix in getattr(self, '_lag_nbr_types', []):
                    G_p = getattr(self, f'_lag_G_{prefix}')
                    mask_p = getattr(self, f'_lag_nbr_mask_{prefix}')
                    buf_p = new_buf if prefix == 'e' else nbr_buf_act
                    for k in range(K):
                        I_lag = I_lag + (G_p[k] * mask_p) @ buf_p[k]
            elif hasattr(self, '_lag_G'):
                mask = self._lag_nbr_mask
                nbr_buf = self._apply_lag_nbr_act(new_buf)
                for k in range(K):
                    G_k = self._lag_G[k] * mask
                    I_lag = I_lag + G_k @ nbr_buf[k]
        return I_lag

    def convolve_stimulus(self, stim_raw: torch.Tensor) -> torch.Tensor:
        if self.stim_kernel is None or self.stim_kernel_len <= 1:
            return stim_raw
        h = F.softplus(self.stim_kernel)
        T, N = stim_raw.shape
        x = stim_raw.t().unsqueeze(1)
        w = h.flip(0).view(1, 1, -1)
        x_pad = F.pad(x, (self.stim_kernel_len - 1, 0))
        out = F.conv1d(x_pad, w)
        return out.squeeze(1).t()

    def laplacian_with_G(self, G: Optional[torch.Tensor] = None) -> torch.Tensor:
        G_eff = self.G if G is None else G.to(device=self.T_e.device, dtype=self.T_e.dtype)
        if self.edge_specific_G:
            W = G_eff * self._G_mask
        else:
            W = self.T_e * G_eff
        W = (W + W.t()) * 0.5
        return W - torch.diag(W.sum(0))

    def laplacian(self) -> torch.Tensor:
        return self.laplacian_with_G()

    def _synaptic_current(self, u_prev, phi_gated, s, T_eff, a, tau, E):
        gamma = torch.exp(-self.dt / (tau + 1e-12))
        gamma = gamma.view(1, -1)
        s_next = gamma * s + phi_gated.unsqueeze(1)
        pool = T_eff.t() @ s_next
        a_post = a.unsqueeze(0) if a.dim() == 1 else a
        g = (pool * a_post).sum(1)
        if E.dim() == 0:
            I = g * (E - u_prev)
        elif E.dim() == 1:
            I = (T_eff.t() @ (s_next * E.view(-1, 1)) * a_post).sum(1) - g * u_prev
        else:
            I = (((T_eff * E).t() @ s_next) * a_post).sum(1) - g * u_prev
        return I, s_next

    def reset_fir_history(self):
        if hasattr(self, '_fir_history_buf'):
            self._fir_history_buf.zero_()
        self._fir_history_live = None

    def init_fir_history(self, u_seq: torch.Tensor, t: int,
                         gating: Optional[torch.Tensor] = None):
        """Seed the FIR history buffer from the teacher-forced trajectory at time *t*.

        After calling this the live buffer is cleared so the first
        `_fir_push_and_current` reads from the (detached) seeded buffer.
        """
        if self._chem_mode != 'fir':
            return
        K = self._fir_kernel_len
        N = self.N
        device = u_seq.device
        if not hasattr(self, '_fir_history_buf'):
            self.register_buffer('_fir_history_buf',
                                torch.zeros(K, N, device=device))
        buf = self._fir_history_buf
        for k in range(K):
            idx = t - 1 - k
            if 0 <= idx < u_seq.shape[0]:
                phi_val = self.phi_fir(u_seq[idx])
                if gating is not None and idx < gating.shape[0]:
                    phi_val = phi_val * gating[idx]
                buf.data[k].copy_(phi_val.detach())
            else:
                buf.data[k].zero_()
        self._fir_history_live = None

    def _fir_push_and_current(self, u_prev, gating):
        """Push phi_fir(u_prev) into ring buffer and compute FIR synaptic currents.

        Maintains a differentiable 'live' buffer (`_fir_history_live`) so that
        multi-step rollout gradients flow through the FIR temporal history.
        The registered buffer is still kept in sync (detached) for device
        tracking and serialisation.
        """
        N, device = self.N, u_prev.device
        phi_val = self.phi_fir(u_prev) * gating.view(N)

        K = self._fir_kernel_len
        if not hasattr(self, '_fir_history_buf'):
            self.register_buffer('_fir_history_buf',
                                torch.zeros(K, N, device=device))

        # Read from the differentiable live buffer when available;
        # fall back to the registered buffer (e.g. after reset / init).
        live = getattr(self, '_fir_history_live', None)
        if live is None:
            live = self._fir_history_buf

        # Shift buffer: newest at index 0
        old = live[:-1]
        new_buf = torch.cat([phi_val.unsqueeze(0), old], dim=0)

        # Keep a differentiable reference for the *next* call (rollout)
        self._fir_history_live = new_buf
        # Sync the registered buffer (detached) for serialisation / device ops
        self._fir_history_buf.data.copy_(new_buf.data.detach())

        I_sv = I_dcv = torch.zeros(N, device=device)
        if hasattr(self, '_W_fir_sv'):
            I_sv = self._synaptic_current_fir(
                u_prev, new_buf, self._W_fir_sv, self.E_sv)
        if hasattr(self, '_W_fir_dcv'):
            I_dcv = self._synaptic_current_fir(
                u_prev, new_buf, self._W_fir_dcv, self.E_dcv)
        return I_sv, I_dcv

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
        lambda_u_eff = self.lambda_u if lambda_u is None else lambda_u.to(device=device, dtype=u_prev.dtype)
        I0_eff = self.I0 if I0 is None else I0.to(device=device, dtype=u_prev.dtype)

        I_sv = I_dcv = torch.zeros(N, device=device)
        if self._chem_mode == 'fir':
            I_sv, I_dcv = self._fir_push_and_current(u_prev, gating)
        else:
            phi_gated = self.phi(u_prev) * gating.view(N)
            if self.r_sv > 0:
                I_sv, s_sv = self._synaptic_current(u_prev, phi_gated, s_sv,
                    self.T_sv * self._get_W("W_sv"), self.a_sv, self.tau_sv, self.E_sv)
            if self.r_dcv > 0:
                I_dcv, s_dcv = self._synaptic_current(u_prev, phi_gated, s_dcv,
                    self.T_dcv * self._get_W("W_dcv"), self.a_dcv, self.tau_dcv, self.E_dcv)

        L = self.laplacian_with_G(G)
        I_gap = L @ u_prev

        if self.graph_poly_order > 1:
            alpha = torch.tanh(self.graph_poly_alpha)
            L_pow_u = I_gap
            for p in range(self.graph_poly_order - 1):
                L_pow_u = L @ L_pow_u
                I_gap = I_gap + alpha[p] * L_pow_u

        I_lr = torch.zeros(N, device=device)
        if self.lowrank_rank > 0:
            I_lr = self.V_lowrank @ torch.tanh(self.U_lowrank @ u_prev)

        I_stim = torch.zeros(N, device=device)
        if b is not None and stim is not None:
            I_stim = b.to(device=device, dtype=u_prev.dtype) * stim.view(N)
        elif self.d_ell > 0 and stim is not None:
            I_stim = (self.b * stim.view(N)) if self.stim_diagonal_only else (self.b @ stim)

        I_coupling = I_gap + I_sv + I_dcv + I_lr
        if self._coupling_dropout > 0 and self.training:
            I_coupling = F.dropout(I_coupling, p=self._coupling_dropout, training=True)

        I_lag = self._lag_push_and_compute(u_prev)

        u_next = (1.0 - lambda_u_eff) * u_prev + lambda_u_eff * (I0_eff + I_coupling + I_stim + I_lag)
        lo, hi = self.u_clip
        if lo is not None or hi is not None:
            u_next = u_next.clamp(min=lo, max=hi)
        return u_next, s_sv, s_dcv
