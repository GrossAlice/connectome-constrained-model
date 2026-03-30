"""Temporal causal Transformer with Gaussian output heads.

Architecture
------------
Input  (B, K, N_obs) →  Linear(N_obs, d_model)
  → add sinusoidal positional encoding
  → TransformerEncoder (n_layers, causal mask, pre-norm)
  → take last timestep embedding  → Linear(d_model, 2*N_obs)
  → split into μ (N_obs) and log_σ (N_obs)
  → σ = softplus(log_σ).clamp(sigma_min, sigma_max)

Loss: Gaussian NLL  −log N(y | μ, σ²)
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TransformerBaselineConfig

__all__ = [
    "TemporalTransformerGaussian",
    "gaussian_nll_loss",
    "build_model",
]


# ── Positional encoding ─────────────────────────────────────────────────────


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (no learnable params).

    Adds position-dependent signal to an input of shape (B, T, d_model).
    Registered as a buffer so it moves with .to(device).
    """

    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.0):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ── Main model ───────────────────────────────────────────────────────────────


class TemporalTransformerGaussian(nn.Module):
    """Causal Transformer predicting next-step Gaussian (μ, σ) per neuron.

    Parameters
    ----------
    n_obs : int
        Number of observed neurons for this worm.
    cfg : TransformerBaselineConfig
        Hyperparameters.
    """

    def __init__(self, n_obs: int, cfg: TransformerBaselineConfig):
        super().__init__()
        self.n_obs = n_obs
        self.cfg = cfg

        # Input projection: (B, K, N_obs) → (B, K, d_model)
        self.input_proj = nn.Linear(n_obs, cfg.d_model)

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
            norm_first=True,  # Pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.n_layers,
            enable_nested_tensor=False,  # Avoid issues with masks
        )

        # Output head: (B, d_model) → (B, 2 * N_obs)
        self.output_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, 2 * n_obs),
        )

        # Bounds for σ
        self.sigma_min = cfg.sigma_min
        self.sigma_max = cfg.sigma_max

        # Causal mask — register as buffer so it moves with device
        # Shape: (K, K) — True means "mask this position"
        self._register_causal_mask(cfg.context_length)

        self._init_weights()

    def _register_causal_mask(self, max_len: int) -> None:
        """Upper-triangular causal mask for nn.TransformerEncoder."""
        mask = torch.triu(
            torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1
        )
        self.register_buffer("_causal_mask", mask, persistent=False)

    def _init_weights(self) -> None:
        """Xavier uniform for linear layers; zeros for biases."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        context: torch.Tensor,
        return_all_steps: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        context : (B, K, N_obs) float
            Context window of neural activity.
        return_all_steps : bool
            If True, return predictions for *every* timestep in the context
            window (useful for free-run / sampling).

        Returns
        -------
        mu : (B, N_obs) or (B, K, N_obs)
        sigma : (B, N_obs) or (B, K, N_obs)
        """
        B, K, N = context.shape
        assert N == self.n_obs, f"Expected N={self.n_obs}, got {N}"

        x = self.input_proj(context)            # (B, K, d_model)
        x = self.pos_enc(x)                     # (B, K, d_model)

        # Causal mask for this sequence length
        causal = self._causal_mask[:K, :K]      # (K, K)
        x = self.transformer(x, mask=causal)    # (B, K, d_model)

        if return_all_steps:
            raw = self.output_head(x)                       # (B, K, 2N)
            mu = raw[..., :N]                               # (B, K, N)
            log_sigma = raw[..., N:]                        # (B, K, N)
        else:
            last = x[:, -1, :]                               # (B, d_model)
            raw = self.output_head(last)                     # (B, 2N)
            mu = raw[:, :N]                                  # (B, N)
            log_sigma = raw[:, N:]                           # (B, N)

        sigma = F.softplus(log_sigma).clamp(self.sigma_min, self.sigma_max)
        return mu, sigma

    def predict_mean(self, context: torch.Tensor) -> torch.Tensor:
        """Return only the mean prediction (for evaluation)."""
        mu, _ = self.forward(context, return_all_steps=False)
        return mu

    def one_step(self, context: torch.Tensor) -> torch.Tensor:
        """Convenience alias: teacher-forced one-step prediction.

        Parameters
        ----------
        context : (B, K, N_obs)

        Returns
        -------
        mu : (B, N_obs)
        """
        return self.predict_mean(context)

    @torch.no_grad()
    def free_run(
        self,
        u0: torch.Tensor,
        n_steps: int,
    ) -> torch.Tensor:
        """Autoregressive free-run from initial context window.

        Parameters
        ----------
        u0 : (K, N_obs) or (1, K, N_obs)
            Initial context window.
        n_steps : int
            Number of steps to predict after the context window.

        Returns
        -------
        pred : (n_steps, N_obs)
        """
        if u0.dim() == 2:
            u0 = u0.unsqueeze(0)  # (1, K, N)
        assert u0.dim() == 3 and u0.size(0) == 1

        K = self.cfg.context_length
        buf = u0.clone()  # (1, K, N)
        preds = []
        for _ in range(n_steps):
            mu, _ = self.forward(buf[:, -K:], return_all_steps=False)  # (1, N)
            preds.append(mu.squeeze(0))  # (N,)
            buf = torch.cat([buf, mu.unsqueeze(1)], dim=1)  # grow buffer

        return torch.stack(preds, dim=0)  # (n_steps, N)

    @torch.no_grad()
    def loo_forward_simulate(
        self,
        u_gt: torch.Tensor,
        neuron_idx: int,
    ) -> torch.Tensor:
        """Leave-one-out forward simulation for a single neuron.

        At each step t, the context window uses ground truth for all
        neurons except neuron_idx, which uses the model's own prediction
        from step t-1.  This matches the LOO semantics in stage2.evaluate.

        Parameters
        ----------
        u_gt : (T, N_obs) float tensor
            Full ground-truth activity.
        neuron_idx : int
            Index of the held-out neuron.

        Returns
        -------
        pred : (T - K, N_obs)  float tensor
            One-step predictions for ALL neurons.  The LOO R² is computed
            from column ``neuron_idx`` only.
        """
        T, N = u_gt.shape
        K = self.cfg.context_length
        assert 0 <= neuron_idx < N

        device = next(self.parameters()).device
        u = u_gt.clone().to(device)

        preds = []
        # For the held-out neuron, replace GT with model predictions
        # starting from the first prediction step.
        loo_buf = u.clone()  # We'll modify neuron_idx in this buffer

        for t in range(K, T):
            ctx = loo_buf[t - K : t].unsqueeze(0)  # (1, K, N)
            mu = self.one_step(ctx)                 # (1, N)
            preds.append(mu.squeeze(0))             # (N,)
            # Replace neuron_idx's value at time t with prediction
            if t < T:
                loo_buf[t, neuron_idx] = mu[0, neuron_idx].item()

        return torch.stack(preds, dim=0)  # (T-K, N)


# ── Gaussian NLL loss ────────────────────────────────────────────────────────


def gaussian_nll_loss(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Gaussian negative log-likelihood loss (per-sample mean).

    −log N(y | μ, σ²) = 0.5 * [log(2π) + 2·log(σ) + ((y − μ)/σ)²]

    Parameters
    ----------
    mu, sigma, target : (B, N) or (B, K, N) — same shape.

    Returns
    -------
    Scalar loss (mean over all elements).
    """
    var = sigma ** 2
    nll = 0.5 * (math.log(2 * math.pi) + torch.log(var) + (target - mu) ** 2 / var)
    return nll.mean()


# ── Factory ──────────────────────────────────────────────────────────────────


def build_model(
    n_obs: int,
    cfg: Optional[TransformerBaselineConfig] = None,
    device: str = "cpu",
) -> TemporalTransformerGaussian:
    """Convenience factory."""
    if cfg is None:
        cfg = TransformerBaselineConfig()
    model = TemporalTransformerGaussian(n_obs=n_obs, cfg=cfg)
    return model.to(device)
