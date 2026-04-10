"""Connectome-informed initialization and regularization for unobserved neurons.

Four improvements over the zeros-init baseline:

1. **Connectome-informed init** — use the gap-junction graph (T_e) to initialise
   u_unobs as a weighted average of observed neurons, so the dynamics optimiser
   starts from a reasonable trajectory rather than zeros.

1b. **SV/DCV-informed init** — extend #1 to also use chemical synapse matrices
    (T_sv, T_dcv).  Observed traces are convolved with the IIR synaptic kernel
    before mixing, so the initialised unobserved trajectories reflect the
    temporal filtering that the dynamics model will apply.

2. **Low-rank mixing matrix** — optionally parameterise
       u_unobs(t) = u_obs(t) @ C        C: (N_obs, N_unobs)
   where C is initialised from the connectome. This reduces the free parameters
   from T × N_unobs → N_obs × N_unobs and acts as a structural regulariser.

3. **Warmup pre-training** — before the main training loop, run a short phase
   that only optimises u_unobs (model frozen) using dynamics + smoothness loss,
   giving the latent trajectories a head start.
"""
from __future__ import annotations

import math
import sys
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

if TYPE_CHECKING:
    from .config import Stage2PTConfig
    from .model import Stage2ModelPT
    from .worm_state import WormState

__all__ = [
    "connectome_init_u_unobs",
    "connectome_init_u_unobs_svdcv",
    "warmup_u_unobs",
    "LowRankUnobs",
]


# ──────────────────────────────────────────────────────────────────────────────
# 1. Connectome-informed initialisation
# ──────────────────────────────────────────────────────────────────────────────

def _connectome_mixing_matrix(
    T_e: torch.Tensor,          # (N_atlas, N_atlas) gap-junction counts
    obs_idx: torch.Tensor,      # (N_obs,)  long
    unobs_idx: torch.Tensor,    # (N_unobs,) long
    alpha: float = 1.0,         # ridge regularisation
) -> torch.Tensor:
    """Compute a mixing matrix C: (N_obs, N_unobs) from gap-junction topology.

    For each unobserved neuron j we solve a small ridge regression:
        ĉ_j = argmin_c || T_e[obs, j] - T_e[obs, obs] @ c ||² + α ||c||²

    i.e. we predict the connectivity profile of neuron j from the observed
    neurons' connectivity profiles.  Then at runtime:
        u_unobs[:, j] ≈ u_obs @ c_j

    This produces a (N_obs, N_unobs) matrix that can be used either for
    one-shot initialisation or as a persistent low-rank parameterisation.
    """
    T_oo = T_e[obs_idx][:, obs_idx].float()      # (N_obs, N_obs)
    T_ou = T_e[obs_idx][:, unobs_idx].float()     # (N_obs, N_unobs)

    N_obs = len(obs_idx)
    # Ridge solve: C = (T_oo^T T_oo + αI)^{-1} T_oo^T T_ou
    A = T_oo.t() @ T_oo + alpha * torch.eye(N_obs, device=T_e.device)
    B = T_oo.t() @ T_ou
    C = torch.linalg.solve(A, B)                  # (N_obs, N_unobs)

    return C


def connectome_init_u_unobs(
    worm_states: List[WormState],
    T_e: torch.Tensor,
    alpha: float = 1.0,
    temporal_smooth_sigma: float = 0.0,
) -> None:
    """Initialise u_unobs for every worm from observed data + connectome.

    Steps:
      1. Build per-worm mixing matrix C from gap-junction topology.
      2. Compute u_unobs_init = u_obs[:, obs_idx] @ C.
      3. Optionally temporal-smooth the result (Gaussian kernel).
      4. Write into worm.u_unobs.data.

    Parameters
    ----------
    worm_states : list of WormState
    T_e : (N_atlas, N_atlas) gap-junction count matrix
    alpha : ridge regularisation for the mixing matrix
    temporal_smooth_sigma : if > 0, apply causal Gaussian smoothing (in frames)
    """
    for worm in worm_states:
        if worm.u_unobs is None or worm.N_unobs == 0:
            continue

        C = _connectome_mixing_matrix(
            T_e, worm.obs_idx, worm.unobs_idx, alpha=alpha,
        )  # (N_obs, N_unobs)

        # u_obs stores full (T, N_atlas) with NaN/0 for unobserved
        # extract only the observed columns
        u_observed = worm.u_obs[:, worm.obs_idx]   # (T, N_obs)
        u_init = u_observed @ C                     # (T, N_unobs)

        # Optional temporal smoothing
        if temporal_smooth_sigma > 0:
            u_init = _temporal_smooth(u_init, temporal_smooth_sigma)

        # Scale to match observed activity magnitude
        obs_std = u_observed.std()
        init_std = u_init.std()
        if init_std > 1e-8:
            u_init = u_init * (obs_std / init_std) * 0.5  # conservative scale

        worm.u_unobs.data.copy_(u_init.detach())

    n_inited = sum(1 for ws in worm_states if ws.u_unobs is not None and ws.N_unobs > 0)
    print(f"[UnobsInit] Connectome-init u_unobs for {n_inited} worms (α={alpha})")


# ──────────────────────────────────────────────────────────────────────────────
# 1b. SV/DCV-informed initialisation
# ──────────────────────────────────────────────────────────────────────────────

def _iir_filter_traces(
    u: torch.Tensor,            # (T, N)
    tau: float,                 # IIR time constant (seconds)
    dt: float,                  # timestep (seconds)
    activation: str = "sigmoid",
) -> torch.Tensor:
    """Apply IIR synaptic filter: s(t+1) = gamma * s(t) + phi(u(t)).

    Returns the filtered (postsynaptic) signal s: (T, N).
    This is the causal convolution that the dynamics model applies to
    presynaptic activity before it enters the synaptic current.
    """
    gamma = math.exp(-dt / max(tau, 1e-6))
    T, N = u.shape

    # Apply activation
    if activation == "sigmoid":
        phi_u = torch.sigmoid(u)
    elif activation == "softplus":
        phi_u = F.softplus(u)
    elif activation == "relu":
        phi_u = F.relu(u)
    else:
        phi_u = u  # identity — not recommended

    # Causal IIR: s[0] = phi(u[0]), s[t] = gamma*s[t-1] + phi(u[t])
    s = torch.zeros_like(phi_u)
    s[0] = phi_u[0]
    for t in range(1, T):
        s[t] = gamma * s[t - 1] + phi_u[t]

    return s


def _combined_connectivity_matrix(
    T_e: torch.Tensor,             # (N, N) gap junctions (symmetric)
    T_sv: torch.Tensor,            # (N, N) short-range chemical [post, pre]
    T_dcv: torch.Tensor,           # (N, N) long-range chemical [post, pre]
    obs_idx: torch.Tensor,
    unobs_idx: torch.Tensor,
    weight_e: float = 1.0,
    weight_sv: float = 1.0,
    weight_dcv: float = 0.5,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Build a mixing matrix C from combined gap + chemical connectivity.

    The combined adjacency W = w_e * T_e + w_sv * T_sv + w_dcv * T_dcv
    is used in place of T_e alone in the ridge regression.

    T_sv and T_dcv are [post, pre] → W[i, j] means j→i for all types.
    T_e is symmetric so direction doesn't matter.

    Returns C: (N_obs, N_unobs).
    """
    # Normalise each matrix to unit Frobenius norm before combining
    def _norm(T: torch.Tensor) -> torch.Tensor:
        fn = T.float().norm()
        return T.float() / fn.clamp(min=1e-8)

    W = (weight_e * _norm(T_e) + weight_sv * _norm(T_sv)
         + weight_dcv * _norm(T_dcv))

    W_oo = W[obs_idx][:, obs_idx]        # (N_obs, N_obs)
    W_ou = W[obs_idx][:, unobs_idx]      # (N_obs, N_unobs)

    N_obs = len(obs_idx)
    A = W_oo.t() @ W_oo + alpha * torch.eye(N_obs, device=T_e.device)
    B = W_oo.t() @ W_ou
    C = torch.linalg.solve(A, B)         # (N_obs, N_unobs)
    return C


def connectome_init_u_unobs_svdcv(
    worm_states: "List[WormState]",
    T_e: torch.Tensor,
    T_sv: torch.Tensor,
    T_dcv: torch.Tensor,
    dt: float = 0.6,
    tau_sv: float = 1.0,
    tau_dcv: float = 4.0,
    activation: str = "sigmoid",
    alpha: float = 1.0,
    weight_e: float = 1.0,
    weight_sv: float = 1.0,
    weight_dcv: float = 0.5,
    temporal_smooth_sigma: float = 0.0,
) -> None:
    """Initialise u_unobs using gap junctions + IIR-filtered chemical synapses.

    For each worm:
      1. Build a combined mixing matrix C from T_e, T_sv, T_dcv.
      2. Compute instantaneous contribution: u_obs @ C_gap
      3. Compute IIR-filtered contribution:
           s_sv(t)  = IIR(u_obs, tau_sv)  @ C_sv
           s_dcv(t) = IIR(u_obs, tau_dcv) @ C_dcv
      4. Blend: u_init = (gap_part + sv_part + dcv_part) / 3
      5. Scale to match observed activity magnitude.

    This gives unobserved neurons a head start that reflects not just
    who they're connected to (topology) but the temporal filtering the
    dynamics model will apply (IIR kernels).

    Parameters
    ----------
    worm_states : list of WormState
    T_e, T_sv, T_dcv : (N_atlas, N_atlas) connectivity matrices
    dt : timestep in seconds
    tau_sv, tau_dcv : IIR time constants (used for filtering observed traces)
    activation : nonlinearity applied before IIR filtering
    alpha : ridge regularisation
    weight_e, weight_sv, weight_dcv : relative weights for combined matrix
    temporal_smooth_sigma : temporal Gaussian smoothing (0 = off)
    """
    for worm in worm_states:
        if worm.u_unobs is None or worm.N_unobs == 0:
            continue

        obs_idx = worm.obs_idx
        unobs_idx = worm.unobs_idx
        u_observed = worm.u_obs[:, obs_idx]   # (T, N_obs)

        # --- Gap-junction channel: instantaneous mixing ---
        C_gap = _connectome_mixing_matrix(T_e, obs_idx, unobs_idx, alpha=alpha)
        u_gap = u_observed @ C_gap            # (T, N_unobs)

        # --- SV channel: IIR-filtered mixing ---
        # Filter observed traces with SV kernel, then mix via SV connectivity
        C_sv = _combined_connectivity_matrix(
            torch.zeros_like(T_e), T_sv, torch.zeros_like(T_dcv),
            obs_idx, unobs_idx,
            weight_e=0.0, weight_sv=1.0, weight_dcv=0.0,
            alpha=alpha,
        )
        s_sv = _iir_filter_traces(u_observed, tau_sv, dt, activation)
        u_sv = s_sv @ C_sv                    # (T, N_unobs)

        # --- DCV channel: IIR-filtered mixing ---
        C_dcv = _combined_connectivity_matrix(
            torch.zeros_like(T_e), torch.zeros_like(T_sv), T_dcv,
            obs_idx, unobs_idx,
            weight_e=0.0, weight_sv=0.0, weight_dcv=1.0,
            alpha=alpha,
        )
        s_dcv = _iir_filter_traces(u_observed, tau_dcv, dt, activation)
        u_dcv = s_dcv @ C_dcv                 # (T, N_unobs)

        # --- Blend all channels ---
        # Normalise each channel to unit std, then weight-average
        parts = []
        weights = []
        for name, part, w in [("gap", u_gap, weight_e),
                               ("sv", u_sv, weight_sv),
                               ("dcv", u_dcv, weight_dcv)]:
            pstd = part.std()
            if pstd > 1e-8 and w > 0:
                parts.append(part / pstd)
                weights.append(w)

        if not parts:
            continue

        w_sum = sum(weights)
        u_init = sum(w * p for w, p in zip(weights, parts)) / w_sum

        # Optional temporal smoothing
        if temporal_smooth_sigma > 0:
            u_init = _temporal_smooth(u_init, temporal_smooth_sigma)

        # Scale to match observed activity magnitude (conservative)
        obs_std = u_observed.std()
        init_std = u_init.std()
        if init_std > 1e-8:
            u_init = u_init * (obs_std / init_std) * 0.5

        worm.u_unobs.data.copy_(u_init.detach())

    n_inited = sum(1 for ws in worm_states if ws.u_unobs is not None and ws.N_unobs > 0)
    print(
        f"[UnobsInit] SV/DCV-informed init u_unobs for {n_inited} worms"
        f" (α={alpha}, w_e={weight_e}, w_sv={weight_sv}, w_dcv={weight_dcv},"
        f" τ_sv={tau_sv:.1f}, τ_dcv={tau_dcv:.1f})"
    )


def _temporal_smooth(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply causal Gaussian smoothing along dim=0."""
    if sigma <= 0:
        return x
    T, N = x.shape
    k_size = max(int(3 * sigma), 1) * 2 + 1  # odd kernel
    t = torch.arange(k_size, device=x.device, dtype=x.dtype) - k_size // 2
    kernel = torch.exp(-0.5 * (t / sigma) ** 2)
    kernel[t < 0] = 0  # causal: only past + present
    kernel = kernel / kernel.sum().clamp(min=1e-8)
    kernel = kernel.view(1, 1, -1)  # (1, 1, K)

    # conv1d: (N, 1, T)
    xp = x.t().unsqueeze(1)        # (N, 1, T)
    xp = F.pad(xp, (k_size - 1, 0))
    out = F.conv1d(xp, kernel)     # (N, 1, T)
    return out.squeeze(1).t()      # (T, N)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Low-rank parameterisation
# ──────────────────────────────────────────────────────────────────────────────

class LowRankUnobs(nn.Module):
    """Parameterise u_unobs(t) = u_obs(t) @ C  with learnable C.

    Instead of storing T × N_unobs free parameters, this stores an
    N_obs × N_unobs mixing matrix C, initialised from the connectome.
    The forward map computes u_unobs on-the-fly from the (fixed) observed
    data, and C is optimised via the dynamics loss.

    This acts as a *structural regulariser*: unobserved trajectories are
    forced to be linear combinations of observed ones, with the connectome
    providing a strong prior on the mixing weights.

    Parameter count: N_obs × N_unobs  vs  T × N_unobs
      typical: 70 × 30 = 2,100  vs  1600 × 30 = 48,000  (23× reduction)
    """

    def __init__(
        self,
        N_obs: int,
        N_unobs: int,
        C_init: Optional[torch.Tensor] = None,     # (N_obs, N_unobs)
        residual_weight: float = 0.1,               # blend with free residual
    ):
        super().__init__()
        if C_init is not None:
            self.C = nn.Parameter(C_init.clone())
        else:
            self.C = nn.Parameter(torch.randn(N_obs, N_unobs) * 0.01)

        self._residual_weight = residual_weight

    def forward(self, u_obs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        u_obs : (T, N_obs) observed neuron trajectories

        Returns
        -------
        u_unobs : (T, N_unobs) predicted unobserved trajectories
        """
        return u_obs @ self.C   # (T, N_unobs)

    @staticmethod
    def from_connectome(
        T_e: torch.Tensor,
        obs_idx: torch.Tensor,
        unobs_idx: torch.Tensor,
        alpha: float = 1.0,
        residual_weight: float = 0.1,
    ) -> "LowRankUnobs":
        """Factory: build from gap-junction connectivity."""
        C = _connectome_mixing_matrix(T_e, obs_idx, unobs_idx, alpha=alpha)
        return LowRankUnobs(
            N_obs=len(obs_idx),
            N_unobs=len(unobs_idx),
            C_init=C,
            residual_weight=residual_weight,
        )


# ──────────────────────────────────────────────────────────────────────────────
# 3. Warmup pre-training
# ──────────────────────────────────────────────────────────────────────────────

def warmup_u_unobs(
    model: Stage2ModelPT,
    worm_states: List[WormState],
    cfg: Stage2PTConfig,
    num_warmup_epochs: int = 20,
    warmup_lr: Optional[float] = None,
    smoothness_w: Optional[float] = None,
    grad_clip: float = 1.0,
) -> float:
    """Pre-train u_unobs trajectories with the dynamics model frozen.

    This runs a short optimisation loop updating *only* u_unobs parameters,
    using the dynamics loss + smoothness prior.  The shared model parameters
    (W, λ, G, I0, …) are held fixed, so this is essentially solving for
    the maximum-a-posteriori unobserved trajectories given the current model.

    Returns
    -------
    final_loss : float
        Last-epoch average loss across all worms.
    """
    mc = cfg.multi
    device = next(model.parameters()).device
    if warmup_lr is None:
        warmup_lr = float(mc.u_unobs_lr)
    if smoothness_w is None:
        smoothness_w = float(mc.u_unobs_smoothness)

    # Collect u_unobs parameters (or low-rank C)
    u_params = []
    for ws in worm_states:
        _, u_list = ws.param_groups()
        u_params.extend(u_list)

    if not u_params:
        print("[UnobsWarmup] No unobserved parameters to warm up.")
        return 0.0

    # Freeze model
    model_grad_state = {}
    for name, p in model.named_parameters():
        model_grad_state[name] = p.requires_grad
        p.requires_grad_(False)

    # Freeze per-worm ψ params (G, b, sigma)
    psi_grad_state = []
    for ws in worm_states:
        psi, _ = ws.param_groups()
        for p in psi:
            psi_grad_state.append((p, p.requires_grad))
            p.requires_grad_(False)

    opt = optim.Adam(u_params, lr=warmup_lr)

    # Import forward pass helper
    from .train_multi import _forward_pass_worm
    from .train import compute_dynamics_loss

    print(f"[UnobsWarmup] {num_warmup_epochs} epochs, lr={warmup_lr}, "
          f"smoothness={smoothness_w}")

    final_loss = 0.0
    for epoch in range(num_warmup_epochs):
        opt.zero_grad()
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        for worm in worm_states:
            if not worm._infer_unobserved:
                continue
            u_full = worm.assemble()
            pm = _forward_pass_worm(model, worm, u_full)
            tm = worm.train_mask()

            dyn_loss = compute_dynamics_loss(
                u_full[tm], pm[tm], worm.sigma_u,
            )
            smooth_loss = worm.smoothness_loss(smoothness_w)
            total_loss = total_loss + worm.weight * (dyn_loss + smooth_loss)

        total_loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(u_params, grad_clip)
        opt.step()
        final_loss = float(total_loss.item())

        if (epoch + 1) % max(1, num_warmup_epochs // 5) == 0 or epoch == 0:
            print(f"  warmup epoch {epoch + 1:3d}/{num_warmup_epochs}: "
                  f"loss={final_loss:.4f}")

    # Restore grad states
    for name, p in model.named_parameters():
        p.requires_grad_(model_grad_state[name])
    for p, grad in psi_grad_state:
        p.requires_grad_(grad)

    print(f"[UnobsWarmup] Done. Final loss = {final_loss:.4f}")
    return final_loss
