"""Atlas-indexed causal Transformer with Gaussian output + masked loss.

Architecture
------------
Input  (B, K, D) where D = 2*N_atlas + n_beh
  = concat(u_atlas_302, obs_mask_302, beh_n_beh)
  → Linear(D, d_model)
  → add sinusoidal positional encoding
  → TransformerEncoder (n_layers, causal mask, pre-norm)
  → take last timestep embedding
  → neural_head  → Linear(d_model, 2*N_atlas)   → μ_u, σ_u
  → beh_head     → Linear(d_model, 2*n_beh)     → μ_b, σ_b  (if predict_beh)

Loss: masked_gaussian_nll(neural, obs_mask) + w_beh * masked_gaussian_nll(beh, beh_mask)

Only observed neurons contribute to the neural loss.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AtlasTransformerConfig

__all__ = [
    "AtlasTransformerGaussian",
    "masked_gaussian_nll_loss",
    "joint_loss",
    "build_atlas_model",
]


# ── Positional encoding ─────────────────────────────────────────────────────


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.0):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ── Main model ───────────────────────────────────────────────────────────────


class AtlasTransformerGaussian(nn.Module):
    """Causal Transformer predicting next-step Gaussian (μ, σ) per atlas neuron
    and (optionally) behaviour eigenworm amplitudes.

    Input : (B, K, D) where D = 2*N_atlas + n_beh
    Output: μ_u (B, N_atlas), σ_u (B, N_atlas)
            μ_b (B, n_beh),   σ_b (B, n_beh)   [if predict_beh]
    """

    def __init__(self, cfg: AtlasTransformerConfig, n_beh: int = 0):
        super().__init__()
        self.cfg = cfg
        self.n_atlas = cfg.n_atlas
        self.n_beh = n_beh
        self.input_mode = getattr(cfg, "input_mode", "concat")

        # Input encoding
        if self.input_mode == "multiply_project":
            # Separate projections — avoids zero-padding contamination.
            # Unobserved neurons get a *learned* baseline instead of silence.
            self.neural_proj = nn.Linear(cfg.n_atlas, cfg.d_model)
            self.mask_proj = nn.Linear(cfg.n_atlas, cfg.d_model)
            self.missing_val = nn.Parameter(torch.zeros(cfg.n_atlas))
            self.beh_proj = nn.Linear(n_beh, cfg.d_model) if n_beh > 0 else None
            self.input_proj = None  # unused
        else:
            # Legacy: single linear over concatenated [u, mask, beh]
            input_dim = cfg.input_dim  # 2*N_atlas + n_beh
            self.input_proj = nn.Linear(input_dim, cfg.d_model)
            self.neural_proj = None
            self.mask_proj = None
            self.missing_val = None
            self.beh_proj = None

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(
            d_model=cfg.d_model,
            max_len=max(cfg.context_length, 2048),
            dropout=cfg.dropout,
        )

        # Transformer encoder (causal, pre-norm)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.n_layers,
            enable_nested_tensor=False,
        )

        # Neural output head: (B, d_model) → (B, 2 * N_atlas)
        self.neural_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, 2 * cfg.n_atlas),
        )

        # Behaviour output head (optional)
        if n_beh > 0:
            self.beh_head = nn.Sequential(
                nn.LayerNorm(cfg.d_model),
                nn.Linear(cfg.d_model, 2 * n_beh),
            )
        else:
            self.beh_head = None

        self.sigma_min = cfg.sigma_min
        self.sigma_max = cfg.sigma_max

        self._register_causal_mask(cfg.context_length)
        self._init_weights()

    # ── backward compatibility ──
    @property
    def output_head(self):
        return self.neural_head

    def _register_causal_mask(self, max_len: int) -> None:
        mask = torch.triu(
            torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1
        )
        self.register_buffer("_causal_mask", mask, persistent=False)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _split_mu_sigma(
        self, raw: torch.Tensor, n: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = raw[..., :n]
        log_sigma = raw[..., n:]
        sigma = F.softplus(log_sigma).clamp(self.sigma_min, self.sigma_max)
        return mu, sigma

    def forward(
        self,
        context: torch.Tensor,
        obs_mask: Optional[torch.Tensor] = None,
        return_all_steps: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor,
               Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass.

        Parameters
        ----------
        context : (B, K, D)
            Full input = [u_atlas, obs_mask_expanded, beh].
            When called from legacy code that packs input manually, pass the
            packed tensor directly.  ``obs_mask`` is then unused.
        obs_mask : (B, N_atlas) or (N_atlas,) or None
            Observation mask. If *context* is already packed (D == input_dim),
            this is ignored.  Kept for backward compatibility.
        return_all_steps : bool

        Returns
        -------
        mu_u, sigma_u : (B, N_atlas) or (B, K, N_atlas)
        mu_b, sigma_b : (B, n_beh)  or (B, K, n_beh)  — None if n_beh == 0
        """
        B, K, D = context.shape

        if self.input_mode == "multiply_project":
            u    = context[..., :self.n_atlas]                # (B, K, 302)
            mask = context[..., self.n_atlas:2*self.n_atlas]  # (B, K, 302)
            # Observed neurons: real activity.  Unobserved: learned default.
            u_gated = u * mask + self.missing_val * (1.0 - mask)
            x = self.neural_proj(u_gated) + self.mask_proj(mask)
            if self.beh_proj is not None and D > 2 * self.n_atlas:
                x = x + self.beh_proj(context[..., 2*self.n_atlas:])
        else:
            x = self.input_proj(context)             # (B, K, d_model)

        x = self.pos_enc(x)                     # (B, K, d_model)

        causal = self._causal_mask[:K, :K]
        x = self.transformer(x, mask=causal)    # (B, K, d_model)

        if return_all_steps:
            h = x
        else:
            h = x[:, -1, :]

        # Neural head
        mu_u, sigma_u = self._split_mu_sigma(
            self.neural_head(h), self.n_atlas)

        # Behaviour head
        if self.beh_head is not None:
            mu_b, sigma_b = self._split_mu_sigma(
                self.beh_head(h), self.n_beh)
        else:
            mu_b, sigma_b = None, None

        return mu_u, sigma_u, mu_b, sigma_b

    def predict_mean(
        self,
        context: torch.Tensor,
        obs_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return concatenated mean prediction [mu_u, mu_b].

        Returns (B, N_atlas + n_beh).
        """
        mu_u, _, mu_b, _ = self.forward(context, obs_mask, return_all_steps=False)
        if mu_b is not None:
            return torch.cat([mu_u, mu_b], dim=-1)
        return mu_u

    def predict_mean_split(
        self,
        context: torch.Tensor,
        obs_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return (mu_u, mu_b) separately."""
        mu_u, _, mu_b, _ = self.forward(context, obs_mask, return_all_steps=False)
        return mu_u, mu_b

    @torch.no_grad()
    def free_run(
        self,
        x0: torch.Tensor,
        obs_mask: Optional[torch.Tensor] = None,
        n_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive free-run from initial context.

        Parameters
        ----------
        x0 : (K, D) or (1, K, D)
        obs_mask : (N_atlas,) — unused in joint-state mode, kept for compat
        n_steps  : steps to predict

        Returns
        -------
        pred : (n_steps, D)
        """
        if x0.dim() == 2:
            x0 = x0.unsqueeze(0)

        K = self.cfg.context_length
        buf = x0.clone()
        preds = []

        for _ in range(n_steps):
            ctx = buf[:, -K:]
            mu = self.predict_mean(ctx)  # (1, D)
            preds.append(mu.squeeze(0))
            buf = torch.cat([buf, mu.unsqueeze(1)], dim=1)

        return torch.stack(preds, dim=0)  # (n_steps, D)

    @torch.no_grad()
    def loo_forward_simulate(
        self,
        x_gt: torch.Tensor,
        neuron_idx: int,
    ) -> torch.Tensor:
        """Leave-one-out forward simulation for a single neuron.

        At each step t, the context window uses ground truth for all
        channels except neuron_idx, which uses the model's own prediction
        from step t-1.

        Parameters
        ----------
        x_gt : (T, D) float tensor
        neuron_idx : int  (must be < n_atlas)

        Returns
        -------
        pred : (T - K, D)
        """
        T, D = x_gt.shape
        K = self.cfg.context_length
        assert 0 <= neuron_idx < self.n_atlas

        device = next(self.parameters()).device
        u = x_gt.clone().to(device)

        preds = []
        loo_buf = u.clone()

        for t in range(K, T):
            ctx = loo_buf[t - K : t].unsqueeze(0)
            mu = self.predict_mean(ctx)
            preds.append(mu.squeeze(0))
            if t < T:
                loo_buf[t, neuron_idx] = mu[0, neuron_idx].item()

        return torch.stack(preds, dim=0)


# ── Masked Gaussian NLL loss ────────────────────────────────────────────────


def masked_gaussian_nll_loss(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
    obs_mask: torch.Tensor,
) -> torch.Tensor:
    """Gaussian NLL loss masked to observed elements only.

    Parameters
    ----------
    mu, sigma, target : same shape (B, N) or (B, K, N)
    obs_mask : same shape, or broadcastable — 1.0 observed, 0.0 unobs

    Returns scalar loss (mean over observed elements).
    """
    var = sigma ** 2
    nll = 0.5 * (math.log(2 * math.pi) + torch.log(var) + (target - mu) ** 2 / var)

    if obs_mask.dim() < nll.dim():
        obs_mask = obs_mask.unsqueeze(0).expand_as(nll)

    masked_nll = nll * obs_mask.float()
    n_obs = obs_mask.float().sum().clamp(min=1.0)
    return masked_nll.sum() / n_obs


def gaussian_nll_loss(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Gaussian NLL (no mask, all elements)."""
    var = sigma ** 2
    nll = 0.5 * (math.log(2 * math.pi) + torch.log(var) + (target - mu) ** 2 / var)
    return nll.mean()


def joint_loss(
    mu_u: torch.Tensor,
    sigma_u: torch.Tensor,
    target_u: torch.Tensor,
    obs_mask: torch.Tensor,
    mu_b: Optional[torch.Tensor],
    sigma_b: Optional[torch.Tensor],
    target_b: Optional[torch.Tensor],
    mask_b: Optional[torch.Tensor],
    w_beh: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combined masked-neural + behaviour NLL loss.

    Returns (total, neural_nll, beh_nll).
    """
    neural_nll = masked_gaussian_nll_loss(mu_u, sigma_u, target_u, obs_mask)
    if mu_b is not None and target_b is not None and mask_b is not None:
        beh_nll = masked_gaussian_nll_loss(mu_b, sigma_b, target_b, mask_b)
        total = neural_nll + w_beh * beh_nll
    else:
        beh_nll = torch.tensor(0.0, device=mu_u.device)
        total = neural_nll
    return total, neural_nll, beh_nll


# ── Factory ──────────────────────────────────────────────────────────────────


def build_atlas_model(
    cfg: Optional[AtlasTransformerConfig] = None,
    device: str = "cpu",
    n_beh: int = 0,
) -> AtlasTransformerGaussian:
    """Convenience factory."""
    if cfg is None:
        cfg = AtlasTransformerConfig()
    model = AtlasTransformerGaussian(cfg, n_beh=n_beh)
    return model.to(device)
