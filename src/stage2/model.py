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
        # Constraint-test option: linearize chemical synapse activation
        self.linear_chemical_synapses = bool(
            getattr(cfg, "linear_chemical_synapses", False))
        # Named activation function for chemical synapses
        self._chem_act = str(
            getattr(cfg, "chemical_synapse_activation", "sigmoid"))
        if self.linear_chemical_synapses:
            self._chem_act = "identity"
        # Shifted sigmoid: σ(α(u−β)) with learned slope and threshold
        if self._chem_act == "shifted_sigmoid":
            self._phi_alpha = nn.Parameter(torch.tensor(1.0, device=device))
            self._phi_beta  = nn.Parameter(torch.zeros(N, device=device))

        # Connectivity matrices
        self.register_buffer("T_e", T_e.to(device).float())
        self.register_buffer("T_sv", T_sv.to(device).float().abs())
        self.register_buffer("T_dcv", T_dcv.to(device).float().abs())

        # Synaptic weight matrices (learnable or fixed)
        # Non-uniform modes (log_counts, sqrt_counts, normalize) are applied
        # by init_W_from_config() in init_from_data after construction.
        for attr, T_mat, init_val in [
                ("W_sv",  self.T_sv,  _W_SV_INIT),
                ("W_dcv", self.T_dcv, _W_DCV_INIT)]:
            W_lo, W_hi = _W_LO, None
            setattr(self, f"_{attr}_lo", W_lo)
            setattr(self, f"_{attr}_hi", W_hi)
            mask = (T_mat > 0).float()
            W_init = torch.full_like(T_mat, init_val)
            W_raw = _reparam_inv(W_init, W_lo, W_hi)
            W_raw = W_raw * mask                   # zero raw weights for non-edges
            learn = bool(getattr(cfg, f"learn_{attr}", False))
            if learn:
                setattr(self, f"_{attr}_raw", nn.Parameter(W_raw))
            else:
                self.register_buffer(f"_{attr}_raw", W_raw, persistent=False)

        # Lambda (leak/decay)
        # lambda_u_init is normally computed by init_lambda_u() from stage-1
        # data; allow None for eval-only paths that immediately load_state_dict.
        if lambda_u_init is None:
            lambda_u_init = torch.full((N,), 0.1)
        lu = lambda_u_init.to(device).float()
        self._lambda_u_lo = float(getattr(cfg, "lambda_u_lo", _LAMBDA_U_LO))
        self._lambda_u_hi = float(getattr(cfg, "lambda_u_hi", _LAMBDA_U_HI))
        self._lambda_u_raw = nn.Parameter(
            _reparam_inv(lu, self._lambda_u_lo, self._lambda_u_hi),
            requires_grad=bool(getattr(cfg, "learn_lambda_u", False)))

        # Gap-junction conductance
        # Non-uniform modes (log_counts, sqrt_counts) are applied by
        # init_G_from_config() in init_from_data after construction.
        self._G_lo = _G_LO
        self._G_hi = _G_MAX
        self.edge_specific_G = bool(getattr(cfg, "edge_specific_G", False))
        if self.edge_specific_G:
            G_mat = torch.full_like(self.T_e, _G_INIT) * (self.T_e > 0).float()
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

        # Stimulus kernel: learnable causal impulse response
        self.stim_kernel_len = int(getattr(cfg, "stim_kernel_len", 0))
        if self.stim_kernel_len > 0 and d_ell > 0:
            # Initialise as decaying exponential  h(k) = exp(-k*dt / tau)
            tau_init = float(getattr(cfg, "stim_kernel_tau_init", 2.0))
            k = torch.arange(self.stim_kernel_len, dtype=torch.float32, device=device)
            h_init = torch.exp(-k * self.dt / max(tau_init, 1e-6))
            h_init = h_init / h_init.sum().clamp(min=1e-8)  # normalise to sum=1
            self.stim_kernel = nn.Parameter(h_init)          # (K,)
        else:
            self.stim_kernel = None

        # Synaptic amplitudes and time constants
        self._init_synaptic_params(cfg, device, N)
        # Reversal potentials
        self._init_reversals(cfg, sign_t, device)

        # Per-neuron process noise: learnable sigma_u(i)
        self._learn_noise = bool(getattr(cfg, "learn_noise", False))
        self._noise_floor = float(getattr(cfg, "noise_floor", 1e-3))
        self._noise_mode = str(getattr(cfg, "noise_mode", "homoscedastic"))
        if self._learn_noise and self._noise_mode == "heteroscedastic":
            # Diagonal linear map:  sigma_{t,i} = softplus(w_i * u_{t,i} + b_i) + floor
            # Initialise w ≈ 0 (starts near homoscedastic), b so softplus(b) ≈ 0.1
            self._sigma_w = nn.Parameter(torch.zeros(N, device=device))
            self._sigma_b = nn.Parameter(
                _softplus_inv(torch.full((N,), 0.1, device=device)))
            # Keep _log_sigma_u as non-learnable buffer for backward compat
            self.register_buffer(
                "_log_sigma_u",
                torch.zeros(N, device=device),
                persistent=False,
            )
        elif self._learn_noise:
            # Homoscedastic: sigma_i = softplus(b_i) + floor
            self._log_sigma_u = nn.Parameter(
                _softplus_inv(torch.full((N,), 0.1, device=device)))
        else:
            self.register_buffer(
                "_log_sigma_u",
                torch.zeros(N, device=device),
                persistent=False,
            )

        # -- Low-rank correlated noise (captures shared unmodeled inputs) ----
        # Σ = diag(σ²) + V V^T  where V ∈ R^{N×R}
        # Constraint test 9: mean|noise_corr| ≈ 0.05-0.10 → small R suffices.
        self.noise_corr_rank = int(getattr(cfg, 'noise_corr_rank', 0))
        if self.noise_corr_rank > 0:
            R = self.noise_corr_rank
            self._noise_V = nn.Parameter(
                torch.randn(N, R, device=device) * 0.01)

        # -- Low-rank dense coupling (captures non-connectome interactions) --
        # I_lr = V @ tanh(U @ u)  where U ∈ R^{K×N}, V ∈ R^{N×K}
        self.lowrank_rank = int(getattr(cfg, 'lowrank_rank', 0))
        if self.lowrank_rank > 0:
            K = self.lowrank_rank
            # Small Xavier init so it starts near zero (doesn't disrupt AR1+connectome init)
            self.U_lowrank = nn.Parameter(
                torch.randn(K, N, device=device) * (2.0 / (K + N)) ** 0.5)
            self.V_lowrank = nn.Parameter(
                torch.randn(N, K, device=device) * (2.0 / (K + N)) ** 0.5)

        # -- Graph polynomial for multi-hop gap-junction propagation ---------
        # I_gap = α₁ L u + α₂ L² u + α₃ L³ u  (α₁=1 fixed, α₂,α₃ learnable)
        self.graph_poly_order = int(getattr(cfg, 'graph_poly_order', 1))
        if self.graph_poly_order > 1:
            self.graph_poly_alpha = nn.Parameter(
                torch.zeros(self.graph_poly_order - 1, device=device))

        # -- Residual MLP correction (captures non-connectome interactions) ---
        # I_mlp = MLP(u) added to the dynamics; regularised to be small.
        self._residual_mlp_hidden = int(getattr(cfg, 'residual_mlp_hidden', 0))
        self._mlp_context_K = int(getattr(cfg, 'residual_mlp_context', 1))
        if self._residual_mlp_hidden > 0:
            n_layers = int(getattr(cfg, 'residual_mlp_layers', 2))
            dropout = float(getattr(cfg, 'residual_mlp_dropout', 0.1))
            layers = []
            d_in = N * self._mlp_context_K   # K-step context: concatenate K frames
            for i in range(n_layers - 1):
                layers.append(nn.Linear(d_in, self._residual_mlp_hidden))
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                d_in = self._residual_mlp_hidden
            layers.append(nn.Linear(d_in, N))
            # Init output layer near zero so MLP starts as no-op
            nn.init.zeros_(layers[-1].weight)
            nn.init.zeros_(layers[-1].bias)
            self.residual_mlp = nn.Sequential(*layers).to(device)
            # Ring buffer for prior_step history (K, N) — newest at [0]
            if self._mlp_context_K > 1:
                self.register_buffer(
                    '_mlp_history_buf',
                    torch.zeros(self._mlp_context_K, N, device=device))
        else:
            self.residual_mlp = None

        # -- Per-neuron coupling gate (decoder analysis: neighbors hurt for linear) --
        # g_i = σ(raw_i) ∈ [0,1] multiplies total coupling for postsynaptic neuron i.
        self.has_coupling_gate = bool(getattr(cfg, 'coupling_gate', False))
        if self.has_coupling_gate:
            _cg_init = float(getattr(cfg, 'coupling_gate_init', 0.0))
            self._coupling_gate_raw = nn.Parameter(
                torch.full((N,), _cg_init, device=device))

        # -- Coupling dropout (regularise the connectome path) ---------------
        # Randomly zeros I_coupling for each neuron during training.
        self._coupling_dropout = float(getattr(cfg, 'coupling_dropout', 0.0))

        # -- Linear lag terms: AR(K) self-lags + connectome-sparse neighbor lags
        # Self-lags: I_self_lag_i = sum_k alpha_{k,i} * u_{i,t-k}
        # Neighbor-lags: I_nbr_lag_i = sum_k sum_{j in N(i)} G^(k)_{ij} * u_{j,t-k}
        self._lag_order = int(getattr(cfg, 'lag_order', 0))
        self._lag_neighbor = bool(getattr(cfg, 'lag_neighbor', False))
        if self._lag_order > 0:
            K_lag = self._lag_order
            # Per-neuron self-lag coefficients, init near zero
            self._lag_alpha = nn.Parameter(
                torch.zeros(K_lag, N, device=device))
            # Ring buffer for prior_step (K, N) — index 0 = most recent lag (u_{t-1})
            self.register_buffer(
                '_lag_history_buf',
                torch.zeros(K_lag, N, device=device))
            # Connectome-sparse neighbor-lag weights
            if self._lag_neighbor:
                # Use the gap-junction mask as the sparsity pattern
                mask = (self.T_e > 0).float()  # (N, N)
                self.register_buffer('_lag_nbr_mask', mask)
                # One weight per edge per lag (init near zero)
                self._lag_G = nn.Parameter(
                    torch.zeros(K_lag, N, N, device=device))

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

    def _init_reversals(self, cfg, sign_t, device):
        # Store sign_t for init_from_data.init_reversals to use.
        if sign_t is not None:
            self.register_buffer("sign_t", sign_t.to(device).float(),
                                 persistent=False)
        else:
            self.register_buffer("sign_t", None)
        # Create E_sv / E_dcv with correct *shape* (values are set by
        # init_from_data.init_reversals or load_state_dict).
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

    @property
    def coupling_gate_values(self) -> Optional[torch.Tensor]:
        """Per-neuron coupling gate σ(raw) ∈ [0,1], shape (N,)."""
        if self.has_coupling_gate:
            return torch.sigmoid(self._coupling_gate_raw)
        return None

    def sigma_at(self, u: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Per-neuron noise std, optionally state-dependent.

        Parameters
        ----------
        u : Tensor, shape ``(..., N)`` or ``None``
            Current neural state.  Required for heteroscedastic mode;
            ignored (but accepted) for homoscedastic.

        Returns
        -------
        Tensor, shape ``(..., N)``
            ``softplus(raw) + floor``, strictly positive.

        In **homoscedastic** mode the result is broadcast-friendly ``(N,)``
        regardless of *u*.  In **heteroscedastic** mode the result has the
        same leading dimensions as *u*.
        """
        if self._learn_noise and self._noise_mode == "heteroscedastic":
            if u is None:
                # Fallback: return sigma at u=0 (just the bias)
                return F.softplus(self._sigma_b) + self._noise_floor
            return F.softplus(self._sigma_w * u + self._sigma_b) + self._noise_floor
        if self._learn_noise:
            return F.softplus(self._log_sigma_u) + self._noise_floor
        return torch.full((self.N,), self._noise_floor,
                          device=self._log_sigma_u.device)

    @property
    def sigma_process(self) -> torch.Tensor:
        """Per-neuron process noise std (N,) — homoscedastic / baseline.

        For heteroscedastic models this returns sigma at u=0 (the bias term).
        Use :meth:`sigma_at` for state-dependent noise.
        """
        return self.sigma_at(None)

    def sample_correlated_noise(self, sigma_diag: torch.Tensor) -> torch.Tensor:
        """Sample ε ~ N(0, diag(σ²) + V V^T) via reparametrisation.

        Constraint test 9 showed mean|noise_corr| ≈ 0.05-0.10 in residuals.
        The low-rank factor V captures shared unmodeled inputs
        (neuromodulation, unobserved neurons, extrasynaptic signaling).

        Returns shape ``(..., N)`` matching *sigma_diag*.
        """
        device = sigma_diag.device
        eps_diag = torch.randn_like(sigma_diag) * sigma_diag
        if self.noise_corr_rank > 0:
            R = self.noise_corr_rank
            z = torch.randn(*sigma_diag.shape[:-1], R, device=device)
            eps_corr = z @ self._noise_V.t()  # (..., N)
            return eps_diag + eps_corr
        return eps_diag

    def set_param_constrained(self, name: str, value: torch.Tensor) -> None:
        raw = getattr(self, f"_{name}_raw")
        lo, hi = getattr(self, f"_{name}_lo"), getattr(self, f"_{name}_hi")
        raw.data.copy_(_reparam_inv(value.to(raw.device), lo, hi))

    # ------------------------------------------------------------------ #
    #  Fast forward: pre-compute params once, loop with cached values     #
    # ------------------------------------------------------------------ #

    def precompute_params(self) -> dict:
        """Compute all reparameterized parameters once (call per-epoch).

        Returns a dict of detached-for-eval or graph-attached tensors that
        :meth:`prior_step_fast` / :meth:`forward_sequence` consume,
        avoiding thousands of redundant sigmoid/softplus calls per epoch.
        """
        p: dict = {}
        p["lambda_u"] = self.lambda_u
        p["I0"] = self.I0
        p["L"] = self.laplacian()
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
        if self.has_coupling_gate:
            p["coupling_gate"] = self.coupling_gate_values
        return p

    def forward_sequence(
        self,
        u: torch.Tensor,
        gating_data: Optional[torch.Tensor] = None,
        stim_data: Optional[torch.Tensor] = None,
        params: Optional[dict] = None,
    ) -> torch.Tensor:
        """Teacher-forced forward pass over a full (T, N) sequence.

        Uses pre-computed parameters from :meth:`precompute_params` to
        avoid redundant reparameterization every step.  Returns ``(T, N)``
        one-step predictions (first row = ``u[0]`` identity).

        Falls back to ``prior_step`` loop when graph_poly_order > 1 or
        lowrank_rank > 0 (these need extra state not cached in params).
        """
        # Fallback for complex model variants
        if self.graph_poly_order > 1 or self.lowrank_rank > 0:
            return self._forward_sequence_fallback(u, gating_data, stim_data)

        T, N = u.shape
        device = u.device
        if params is None:
            params = self.precompute_params()

        lambda_u = params["lambda_u"]
        I0 = params["I0"]
        L = params["L"]
        has_sv = self.r_sv > 0
        has_dcv = self.r_dcv > 0
        has_gate = self.has_coupling_gate
        lo, hi = self.u_clip

        s_sv = torch.zeros(N, self.r_sv, device=device)
        s_dcv = torch.zeros(N, self.r_dcv, device=device)
        ones = torch.ones(N, device=device)
        preds = torch.zeros_like(u)
        preds[0] = u[0]

        # Pre-compute MLP inputs with K-step context (avoids ring buffer in hot loop)
        has_mlp = self.residual_mlp is not None
        mlp_inputs = None  # (T, N*K) or None
        if has_mlp and self._mlp_context_K > 1:
            K_ctx = self._mlp_context_K
            u_pad = F.pad(u, (0, 0, K_ctx - 1, 0))  # (T+K-1, N) zero-padded
            mlp_inputs = torch.cat(
                [u_pad[K_ctx - 1 - k : K_ctx - 1 - k + T] for k in range(K_ctx)],
                dim=1,
            )  # (T, N*K)  — row t = [u_t, u_{t-1}, ..., u_{t-K+1}]

        # Pre-compute lag inputs: (T, K_lag, N) — lag_inputs[t, k] = u[t-1-k]
        has_lag = self._lag_order > 0
        lag_self_all = None   # (T, N) pre-computed I_self_lag for each t
        lag_nbr_all = None    # (T, N) pre-computed I_nbr_lag for each t
        if has_lag:
            K_lag = self._lag_order
            # u_lag_pad: zero-padded so index -1..-K map to zeros
            u_lag_pad = F.pad(u, (0, 0, K_lag, 0))  # (T+K, N)
            # lag_buf[t, k] = u[t-1-k] = u_lag_pad[K_lag + t - 1 - k]
            lag_buf = torch.stack(
                [u_lag_pad[K_lag - 1 - k : K_lag - 1 - k + T] for k in range(K_lag)],
                dim=1,
            )  # (T, K_lag, N)
            # Self-lags: sum_k alpha_{k,i} * lag_buf[t, k, i]
            lag_self_all = (self._lag_alpha.unsqueeze(0) * lag_buf).sum(1)  # (T, N)
            # Neighbor-lags
            if self._lag_neighbor and hasattr(self, '_lag_G'):
                mask = self._lag_nbr_mask  # (N, N)
                lag_nbr_all = torch.zeros(T, N, device=device)
                for k in range(K_lag):
                    G_k = self._lag_G[k] * mask  # (N, N)
                    lag_nbr_all = lag_nbr_all + (lag_buf[:, k, :] @ G_k.T)  # (T, N)

        for t in range(1, T):
            u_prev = u[t - 1]
            g = gating_data[t - 1] if gating_data is not None else ones
            phi_gated = self.phi(u_prev) * g

            I_sv = I_dcv = torch.zeros(N, device=device)
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

            I_gap = L @ u_prev

            I_stim = torch.zeros(N, device=device)
            if self.d_ell > 0 and stim_data is not None:
                s = stim_data[t - 1]
                I_stim = (self.b * s.view(N)) if self.stim_diagonal_only else (self.b @ s)

            I_coupling = I_gap + I_sv + I_dcv
            if has_gate:
                I_coupling = params["coupling_gate"] * I_coupling
            if self._coupling_dropout > 0 and self.training:
                I_coupling = F.dropout(I_coupling, p=self._coupling_dropout, training=True)

            # Residual MLP correction (with K-step context)
            I_mlp = torch.zeros(N, device=device)
            if has_mlp:
                if mlp_inputs is not None:
                    I_mlp = self.residual_mlp(mlp_inputs[t - 1])
                else:
                    I_mlp = self.residual_mlp(u_prev)

            # Linear lag terms
            I_lag = torch.zeros(N, device=device)
            if has_lag:
                I_lag = lag_self_all[t]  # pre-computed self-lags at step t
                if lag_nbr_all is not None:
                    I_lag = I_lag + lag_nbr_all[t]

            u_next = (1.0 - lambda_u) * u_prev + lambda_u * (I0 + I_coupling + I_stim + I_mlp + I_lag)
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
        """Fallback using prior_step for complex model variants."""
        T, N = u.shape
        device = u.device
        ones = torch.ones(N, device=device)
        s_sv = torch.zeros(N, self.r_sv, device=device)
        s_dcv = torch.zeros(N, self.r_dcv, device=device)
        self.reset_mlp_history()
        self.reset_lag_history()
        preds = torch.zeros_like(u)
        preds[0] = u[0]
        for t in range(1, T):
            g = gating_data[t - 1] if gating_data is not None else ones
            s = stim_data[t - 1] if stim_data is not None else None
            preds[t], s_sv, s_dcv = self.prior_step(u[t - 1], s_sv, s_dcv, g, s)
        return preds


    def phi(self, u: torch.Tensor) -> torch.Tensor:
        """Chemical synapse activation φ(u).

        Supported: sigmoid (default), tanh, softplus, relu, elu, swish,
        shifted_sigmoid (σ(α(u−β)) with learned slope+threshold), identity.
        """
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
        # Fallback
        return torch.sigmoid(u)

    # -- MLP context history helpers ----------------------------------------
    def reset_mlp_history(self):
        """Zero the MLP context ring buffer (call at sequence start)."""
        if (self._mlp_context_K > 1 and self.residual_mlp is not None
                and hasattr(self, '_mlp_history_buf')):
            self._mlp_history_buf.zero_()

    def init_mlp_history(self, u_seq: torch.Tensor, t: int):
        """Fill MLP history from a (T, N) sequence ending at time *t*.

        Uses ``.data.copy_()`` to avoid inplace mutation of the autograd
        graph (safe because the source slices are detached ground-truth).
        """
        if (self._mlp_context_K <= 1 or self.residual_mlp is None
                or not hasattr(self, '_mlp_history_buf')):
            return
        K = self._mlp_context_K
        buf = self._mlp_history_buf
        for k in range(K):
            idx = t - k
            if 0 <= idx < u_seq.shape[0]:
                buf.data[k].copy_(u_seq[idx].detach())
            else:
                buf.data[k].zero_()

    def _mlp_push_and_forward(self, u_prev: torch.Tensor) -> torch.Tensor:
        """Push *u_prev* into the history buffer and compute I_mlp.

        Uses functional (non-inplace) operations so the computation graph
        is not corrupted when ``u_prev`` requires grad (rollout training).
        """
        if self.residual_mlp is None:
            return torch.zeros(self.N, device=u_prev.device)
        if self._mlp_context_K <= 1:
            return self.residual_mlp(u_prev)
        # Build new buffer: [u_prev, old[0], old[1], ..., old[K-2]]
        # Purely functional — no inplace mutation of the registered buffer.
        old = self._mlp_history_buf[:-1]          # (K-1, N), detached view
        new_buf = torch.cat([u_prev.unsqueeze(0), old], dim=0)  # (K, N)
        # Persist for next call (detached so the buffer itself stays leaf)
        self._mlp_history_buf.data.copy_(new_buf.detach())
        return self.residual_mlp(new_buf.reshape(-1))

    # -- Lag history helpers -------------------------------------------------
    def reset_lag_history(self):
        """Zero the lag ring buffer (call at sequence start)."""
        if self._lag_order > 0 and hasattr(self, '_lag_history_buf'):
            self._lag_history_buf.zero_()

    def init_lag_history(self, u_seq: torch.Tensor, t: int):
        """Fill lag history from a (T, N) sequence ending at time *t*.

        After this call, _lag_history_buf[k] = u_seq[t - 1 - k] for k in [0, K).
        """
        if self._lag_order <= 0 or not hasattr(self, '_lag_history_buf'):
            return
        K = self._lag_order
        buf = self._lag_history_buf
        for k in range(K):
            idx = t - 1 - k  # lag k=0 → u_{t-1}
            if 0 <= idx < u_seq.shape[0]:
                buf.data[k].copy_(u_seq[idx].detach())
            else:
                buf.data[k].zero_()

    def _lag_push_and_compute(self, u_prev: torch.Tensor) -> torch.Tensor:
        """Push u_prev into the lag buffer and compute I_lag.

        Uses functional ops to keep the autograd graph intact.
        Returns (N,) tensor.
        """
        if self._lag_order <= 0:
            return torch.zeros(self.N, device=u_prev.device)
        K = self._lag_order
        # Build new buffer: [u_prev, old[0], ..., old[K-2]]
        old = self._lag_history_buf[:-1]   # (K-1, N)
        new_buf = torch.cat([u_prev.unsqueeze(0), old], dim=0)  # (K, N)
        # Persist detached copy for next call
        self._lag_history_buf.data.copy_(new_buf.detach())
        # Self-lags: sum_k alpha_{k,i} * u_{i,t-1-k}
        # new_buf[k] = u_{t-1-k}, _lag_alpha[k] = alpha_k  →  (K,N)*(K,N) summed
        I_lag = (self._lag_alpha * new_buf).sum(0)  # (N,)
        # Neighbor-lags: sum_k G^(k) * mask * u_{t-1-k}
        if self._lag_neighbor and hasattr(self, '_lag_G'):
            mask = self._lag_nbr_mask  # (N, N)
            for k in range(K):
                # G_k is (N, N) sparse along connectome edges
                G_k = self._lag_G[k] * mask  # (N, N)
                I_lag = I_lag + G_k @ new_buf[k]  # (N,)
        return I_lag

    def convolve_stimulus(self, stim_raw: torch.Tensor) -> torch.Tensor:
        if self.stim_kernel is None or self.stim_kernel_len <= 1:
            return stim_raw
        # Ensure non-negative kernel (softplus) and normalise
        h = F.softplus(self.stim_kernel)  # (K,)
        # Causal 1-D conv: kernel flipped for conv1d convention
        # stim_raw is (T, N) → treat as batch of N channels
        T, N = stim_raw.shape
        # conv1d expects (batch, channels, length) → (N, 1, T)
        x = stim_raw.t().unsqueeze(1)     # (N, 1, T)
        w = h.flip(0).view(1, 1, -1)      # (1, 1, K)
        # left-pad for causal conv (output length == T)
        x_pad = F.pad(x, (self.stim_kernel_len - 1, 0))
        out = F.conv1d(x_pad, w)           # (N, 1, T)
        return out.squeeze(1).t()          # (T, N)

    def laplacian_with_G(self, G: Optional[torch.Tensor] = None) -> torch.Tensor:
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

        # -- Graph polynomial: multi-hop gap-junction propagation ----------
        # Test 3/5: gap junctions dominate; multi-hop adds L²u, L³u terms.
        if self.graph_poly_order > 1:
            L = self.laplacian_with_G(G)
            alpha = torch.tanh(self.graph_poly_alpha)  # bounded (-1, 1)
            L_pow_u = L @ u_prev  # already computed as I_gap
            for p in range(self.graph_poly_order - 1):
                L_pow_u = L @ L_pow_u
                I_gap = I_gap + alpha[p] * L_pow_u

        # -- Low-rank dense coupling (non-connectome interactions) ---------
        I_lr = torch.zeros(N, device=device)
        if self.lowrank_rank > 0:
            I_lr = self.V_lowrank @ torch.tanh(self.U_lowrank @ u_prev)

        I_stim = torch.zeros(N, device=device)
        if b is not None and stim is not None:
            # Multi-worm path: per-worm diagonal stimulus weights
            I_stim = b.to(device=device, dtype=u_prev.dtype) * stim.view(N)
        elif self.d_ell > 0 and stim is not None:
            # Single-worm path: model's own b
            I_stim = (self.b * stim.view(N)) if self.stim_diagonal_only else (self.b @ stim)

        # -- Coupling gate: scale all network coupling (I_gap+I_sv+I_dcv+I_lr)
        # by a per-neuron gate σ(g_i) ∈ [0,1].  Decoder analysis showed that
        # linear neighbors hurt LOO; the gate lets each neuron learn whether
        # to accept or reject coupling input.
        I_coupling = I_gap + I_sv + I_dcv + I_lr
        if self.has_coupling_gate:
            I_coupling = self.coupling_gate_values * I_coupling
        if self._coupling_dropout > 0 and self.training:
            I_coupling = F.dropout(I_coupling, p=self._coupling_dropout, training=True)

        # -- Residual MLP correction: I_mlp = MLP([u_t, ..., u_{t-K+1}])
        I_mlp = self._mlp_push_and_forward(u_prev)

        # -- Linear lag terms: I_lag = sum_k alpha_k * u_{t-1-k} + neighbor lags
        I_lag = self._lag_push_and_compute(u_prev)

        u_next = (1.0 - lambda_u_eff) * u_prev + lambda_u_eff * (I0_eff + I_coupling + I_stim + I_mlp + I_lag)
        lo, hi = self.u_clip
        if lo is not None or hi is not None:
            u_next = u_next.clamp(min=lo, max=hi)
        return u_next, s_sv, s_dcv

