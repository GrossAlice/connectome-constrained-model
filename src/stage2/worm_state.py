from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .config import Stage2PTConfig
from .model import _reparam_fwd, _reparam_inv, _G_LO

__all__ = ["WormState", "build_worm_states"]


class WormState(nn.Module):
    def __init__(
        self,
        worm_dict: Dict[str, Any],
        cfg: Stage2PTConfig,
        infer_unobserved: bool = True,
    ) -> None:
        super().__init__()

        self.worm_id: str = worm_dict["worm_id"]
        self.dataset_type: str = worm_dict["dataset_type"]
        self.has_stage1: bool = bool(worm_dict["has_stage1"])
        self.weight: float = float(worm_dict["weight"])

        T: int = int(worm_dict["T"])
        N: int = int(worm_dict["u"].shape[1])  # N_atlas
        self.T = T
        self.N = N
        self.N_obs: int = int(worm_dict["N_obs"])
        self.N_unobs: int = int(worm_dict["N_unobs"])

        self._infer_unobserved: bool = infer_unobserved and (self.N_unobs > 0)

        device = worm_dict["u"].device

        self.register_buffer("u_obs",     worm_dict["u"])           # (T, N)
        self.register_buffer("obs_mask",  worm_dict["obs_mask"])    # (N,) bool
        self.register_buffer("obs_idx",   worm_dict["obs_idx"])     # (n_obs,) long
        self.register_buffer("unobs_idx", worm_dict["unobs_idx"])   # (n_unobs,) long
        self.register_buffer("val_mask",  worm_dict["val_mask"])    # (T,) bool
        self.register_buffer("gating",    worm_dict["gating"])      # (T, N)

        self._learn_sigma_u: bool = bool(getattr(cfg, "learn_sigma_u", False))
        _sigma_u_init = worm_dict["sigma_u"]
        if self._learn_sigma_u:
            self._log_sigma_u = nn.Parameter(
                torch.log(_sigma_u_init.clamp(min=1e-6))
            )
        else:
            self._log_sigma_u = None
            self.register_buffer("_sigma_u_buf", _sigma_u_init)     # (N,)

        if worm_dict.get("u_var") is not None:
            self.register_buffer("u_var", worm_dict["u_var"])       # (T, N)
        else:
            self.u_var: Optional[torch.Tensor] = None

        if worm_dict.get("behaviour") is not None:
            self.register_buffer("behaviour", worm_dict["behaviour"])
        else:
            self.behaviour: Optional[torch.Tensor] = None

        if worm_dict.get("stim") is not None:
            self.register_buffer("stim", worm_dict["stim"])         # (T, N)
        else:
            self.stim: Optional[torch.Tensor] = None

        # Per-worm G (lambda_u and I0 are shared on the model)
        self._per_worm_G: bool = bool(getattr(cfg, "per_worm_G", False))
        self._G_lo: float = _G_LO
        from .model import _G_MAX, _G_INIT
        self._G_hi: Optional[float] = _G_MAX

        if self._per_worm_G:
            G_init = torch.tensor(
                _G_INIT, dtype=torch.float32, device=device
            )
            self._G_raw: Optional[nn.Parameter] = nn.Parameter(
                _reparam_inv(G_init, self._G_lo, self._G_hi)
            )
        else:
            self._G_raw = None

        # Stimulus weight (Randi only)
        if self.stim is not None:
            self.b: Optional[nn.Parameter] = nn.Parameter(
                torch.zeros(N, device=device)
            )
        else:
            self.b = None

        # (T, n_unobs) — refined in Step 1 inner loop
        if self._infer_unobserved:
            self.u_unobs: Optional[nn.Parameter] = nn.Parameter(
                torch.zeros(T, self.N_unobs, device=device)
            )
        else:
            self.u_unobs = None


    @property
    def G(self) -> Optional[torch.Tensor]:
        """Per-worm gap-junction conductance (scalar), or None if shared G is used."""
        if self._G_raw is None:
            return None
        return _reparam_fwd(self._G_raw, self._G_lo, self._G_hi)

    @property
    def sigma_u(self) -> torch.Tensor:
        """Per-neuron process-noise std (N,). Differentiable when learnable."""
        if self._log_sigma_u is not None:
            return torch.exp(self._log_sigma_u)
        return self._sigma_u_buf


    def assemble(self) -> torch.Tensor:
        """Full (T, N_atlas) trajectory, differentiable through u_unobs (Step 1)."""
        if not self._infer_unobserved:
            return self.u_obs

        T, N = self.T, self.N
        zeros = torch.zeros(T, N, device=self.u_unobs.device, dtype=torch.float32)
        u_unobs_full = zeros.index_copy(1, self.unobs_idx, self.u_unobs)
        return self.u_obs + u_unobs_full

    def assemble_detached(self) -> torch.Tensor:
        """Full (T, N_atlas) trajectory with u_unobs detached (Step 2)."""
        if not self._infer_unobserved:
            return self.u_obs

        T, N = self.T, self.N
        zeros = torch.zeros(T, N, device=self.u_unobs.device, dtype=torch.float32)
        u_unobs_full = zeros.index_copy(1, self.unobs_idx, self.u_unobs.detach())
        return self.u_obs + u_unobs_full


    def smoothness_loss(self, weight: float = 1.0) -> torch.Tensor:
        """L2 temporal smoothness prior on u_unobs."""
        if not self._infer_unobserved or weight <= 0.0 or self.u_unobs is None:
            return torch.tensor(0.0, device=self.u_obs.device)
        diff = self.u_unobs[1:] - self.u_unobs[:-1]   # (T-1, n_unobs)
        return weight * (diff ** 2).mean()


    def param_groups(self) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        """Return (psi_params, u_params) for optimizer setup."""
        psi: List[nn.Parameter] = []
        if self._G_raw is not None:
            psi.append(self._G_raw)
        if self.b is not None:
            psi.append(self.b)
        if self._log_sigma_u is not None:
            psi.append(self._log_sigma_u)

        u: List[nn.Parameter] = []
        if self._infer_unobserved and self.u_unobs is not None:
            u.append(self.u_unobs)

        return psi, u


    def train_mask(self) -> torch.Tensor:
        """Bool (T,) — True for training time points (not held out)."""
        return ~self.val_mask

    def __repr__(self) -> str:
        return (
            f"WormState(id={self.worm_id!r}, type={self.dataset_type!r}, "
            f"T={self.T}, N_obs={self.N_obs}, N_unobs={self.N_unobs}, "
            f"stage1={self.has_stage1}, infer_unobs={self._infer_unobserved})"
        )



def build_worm_states(
    data: Dict[str, Any],
    cfg: Stage2PTConfig,
) -> List[WormState]:
    """Build WormState objects from load_multi_worm_data output."""
    infer = bool(getattr(cfg, "infer_unobserved", True))
    states: List[WormState] = []
    for worm_dict in data["worms"]:
        ws = WormState(worm_dict, cfg, infer_unobserved=infer)
        states.append(ws)
    return states
