"""Temporal causal Transformer with Gaussian output heads.

Architecture
------------
Input  (B, K, D)  where D = n_neural + n_beh
  → Linear(D, d_model)
  → add sinusoidal positional encoding
  → TransformerEncoder (n_layers, causal mask, pre-norm)
  → take last timestep embedding
  → neural_head → Linear(d_model, 2*n_neural)   → μ_u, σ_u
  → beh_head    → Linear(d_model, 2*n_beh)      → μ_b, σ_b  (if predict_beh)

Loss: gaussian_nll(neural) + w_beh * masked_gaussian_nll(beh)
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
    "masked_gaussian_nll_loss",
    "joint_loss",
    "build_model",
    "DiffusionSchedule",
    "DiffusionDenoiserHead",
    "diffusion_denoising_loss",
    "diffusion_sample",
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


# ── Diffusion schedule & denoiser ────────────────────────────────────────────


class DiffusionSchedule:
    """Noise schedule for denoising diffusion.

    Provides a set of noise levels σ_i for i = 0 … S-1,
    geometrically spaced between sigma_min and sigma_max.
    """

    def __init__(
        self,
        n_steps: int = 50,
        sigma_min: float = 0.01,
        sigma_max: float = 1.0,
        schedule: str = "log_linear",
    ):
        self.n_steps = n_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        if schedule == "log_linear":
            self.sigmas = torch.exp(
                torch.linspace(math.log(sigma_max), math.log(sigma_min), n_steps)
            )  # decreasing: noisy → clean
        elif schedule == "cosine":
            t = torch.linspace(0, 1, n_steps)
            self.sigmas = sigma_max * torch.cos(t * math.pi / 2).clamp(min=sigma_min)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

    def sample_sigma(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample random noise levels for training (uniform over schedule)."""
        idx = torch.randint(0, self.n_steps, (batch_size,), device=device)
        return self.sigmas.to(device)[idx]  # (B,)


class DiffusionDenoiserHead(nn.Module):
    """Denoiser with FiLM conditioning and Karras skip connection.

    Key design:
    - **FiLM conditioning**: h (transformer embedding) produces per-layer
      γ and β that modulate intermediate features.  This makes it
      structurally impossible to ignore the context.
    - **Karras skip connection**: ``x̂₀ = c_skip(σ)·x_noisy + c_out(σ)·F_θ(…)``
      so at low σ the output ≈ x_noisy (trivially correct) and at high σ
      the network (forced to use h) dominates.

    Parameters
    ----------
    d_model : int   dimension of transformer embedding
    n_out   : int   output dimension (n_neural or n_beh)
    hidden  : int   hidden width of trunk MLP
    n_layers: int   number of trunk layers
    sigma_data: float  approximate data std (for Karras c_skip/c_out)
    """

    def __init__(
        self, d_model: int, n_out: int, hidden: int = 256,
        n_layers: int = 3, sigma_data: float = 1.0,
    ):
        super().__init__()
        self.n_out = n_out
        self.sigma_data = sigma_data
        self.hidden = hidden
        self.n_layers = n_layers

        # Sinusoidal embedding for noise level (scalar → vector)
        self.sigma_embed_dim = 64

        # ── σ encoder ──
        self.sigma_enc = nn.Sequential(
            nn.Linear(self.sigma_embed_dim, hidden),
            nn.SiLU(),
        )

        # ── x encoder ──
        self.x_enc = nn.Linear(n_out, hidden)

        # ── h → FiLM parameters (γ, β) for each trunk layer ──
        self.h_to_film = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden * 2 * n_layers),  # (γ, β) per layer
        )

        # ── Trunk MLP (modulated by h via FiLM) ──
        trunk_layers: list[nn.Module] = []
        for _ in range(n_layers):
            trunk_layers.append(nn.LayerNorm(hidden))
            trunk_layers.append(nn.Linear(hidden, hidden))
            trunk_layers.append(nn.SiLU())
        # Store as flat list but we'll index manually
        self.trunk_norms = nn.ModuleList(
            [nn.LayerNorm(hidden) for _ in range(n_layers)]
        )
        self.trunk_lins = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(n_layers)]
        )
        self.trunk_acts = nn.ModuleList(
            [nn.SiLU() for _ in range(n_layers)]
        )

        # ── Output projection ──
        self.out_proj = nn.Linear(hidden, n_out)

    def _sigma_embedding(self, sigma: torch.Tensor) -> torch.Tensor:
        """Sinusoidal embedding for noise level σ.  σ: (B,) → (B, 64)."""
        half = self.sigma_embed_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, dtype=sigma.dtype, device=sigma.device)
            / half
        )
        args = sigma.unsqueeze(-1) * freqs.unsqueeze(0)  # (B, half)
        return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, 64)

    def _karras_coeffs(
        self, sigma: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Karras preconditioning coefficients.

        c_skip(σ) = σ_d² / (σ² + σ_d²)   → 1 when σ→0, 0 when σ→∞
        c_out(σ)  = σ·σ_d / √(σ² + σ_d²) → 0 when σ→0, σ_d when σ→∞
        """
        sd2 = self.sigma_data ** 2
        s2 = sigma ** 2
        c_skip = sd2 / (s2 + sd2)        # (B,)
        c_out = sigma * self.sigma_data / (s2 + sd2).sqrt()  # (B,)
        return c_skip.unsqueeze(-1), c_out.unsqueeze(-1)  # (B, 1)

    def forward(
        self, h: torch.Tensor, x_noisy: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """Predict clean x₀ with FiLM conditioning + Karras skip.

        Parameters
        ----------
        h       : (B, d_model)  transformer context embedding
        x_noisy : (B, n_out)    noised target
        sigma   : (B,)          noise level per sample

        Returns
        -------
        x_hat : (B, n_out)  predicted clean target
        """
        B = h.shape[0]

        # Encode inputs
        sigma_emb = self._sigma_embedding(sigma)       # (B, 64)
        sigma_feat = self.sigma_enc(sigma_emb)          # (B, hidden)
        x_feat = self.x_enc(x_noisy)                    # (B, hidden)

        # FiLM parameters from h
        film_raw = self.h_to_film(h)  # (B, hidden*2*n_layers)
        film_raw = film_raw.view(B, self.n_layers, 2, self.hidden)

        # Trunk: start from x_feat + sigma_feat, modulate by h
        feat = x_feat + sigma_feat
        for i in range(self.n_layers):
            gamma = film_raw[:, i, 0, :]   # (B, hidden)
            beta = film_raw[:, i, 1, :]    # (B, hidden)
            feat = self.trunk_norms[i](feat)
            feat = self.trunk_lins[i](feat)
            feat = feat * (1.0 + gamma) + beta   # FiLM modulation
            feat = self.trunk_acts[i](feat)

        f_theta = self.out_proj(feat)  # (B, n_out) — raw network output

        # Karras skip connection
        c_skip, c_out = self._karras_coeffs(sigma)
        x_hat = c_skip * x_noisy + c_out * f_theta

        return x_hat


def diffusion_denoising_loss(
    denoiser: DiffusionDenoiserHead,
    h: torch.Tensor,
    x_clean: torch.Tensor,
    schedule: DiffusionSchedule,
    sigma_data: float = 1.0,
) -> torch.Tensor:
    """Denoising score matching loss (uniform weighting).

    For each sample, draw a random σ, noise the target, and train the
    denoiser to recover the clean target.  Uses **uniform MSE** since the
    Karras skip connection in the architecture already handles
    preconditioning (c_skip · x_noisy + c_out · F_θ).

    Parameters
    ----------
    denoiser : DiffusionDenoiserHead
    h        : (B, d_model)  context embedding from transformer
    x_clean  : (B, n_out)    clean target
    schedule : DiffusionSchedule
    sigma_data : float       (unused, kept for API compatibility)

    Returns
    -------
    Scalar loss.
    """
    B = x_clean.shape[0]
    sigma = schedule.sample_sigma(B, x_clean.device)  # (B,)
    eps = torch.randn_like(x_clean)                    # (B, n_out)
    x_noisy = x_clean + sigma.unsqueeze(-1) * eps      # (B, n_out)
    x_hat = denoiser(h, x_noisy, sigma)                # (B, n_out)
    loss = F.mse_loss(x_hat, x_clean)
    return loss


def masked_diffusion_denoising_loss(
    denoiser: DiffusionDenoiserHead,
    h: torch.Tensor,
    x_clean: torch.Tensor,
    schedule: DiffusionSchedule,
    mask: torch.Tensor,
    sigma_data: float = 1.0,
) -> torch.Tensor:
    """Denoising loss with validity mask (for behaviour channels)."""
    B = x_clean.shape[0]
    sigma = schedule.sample_sigma(B, x_clean.device)
    eps = torch.randn_like(x_clean)
    x_noisy = x_clean + sigma.unsqueeze(-1) * eps
    x_hat = denoiser(h, x_noisy, sigma)
    per_elem = (x_hat - x_clean) ** 2
    valid = mask > 0.5
    if valid.sum() == 0:
        return torch.tensor(0.0, device=x_clean.device, requires_grad=True)
    return per_elem[valid].mean()


@torch.no_grad()
def diffusion_sample(
    denoiser: DiffusionDenoiserHead,
    h: torch.Tensor,
    schedule: DiffusionSchedule,
    n_out: int,
    n_sampling_steps: Optional[int] = None,
    stochastic: bool = False,
) -> torch.Tensor:
    """Generate samples via iterative denoising.

    Default is **deterministic** (DDIM-style): start from x ~ N(0, σ_max²),
    denoise through the schedule without adding noise.  This gives the
    best single-sample "mean" estimate.

    Set ``stochastic=True`` for ancestral (DDPM-style) sampling with
    noise injection — useful for diversity / uncertainty estimation.

    Parameters
    ----------
    denoiser : DiffusionDenoiserHead
    h        : (B, d_model)  conditioning vector
    schedule : DiffusionSchedule
    n_out    : int  output dimension
    n_sampling_steps : int or None  (if None, uses schedule.n_steps)
    stochastic : bool  if True, add noise at each step (ancestral)

    Returns
    -------
    x : (B, n_out)  generated samples
    """
    B = h.shape[0]
    device = h.device
    n_steps = n_sampling_steps or schedule.n_steps

    # Sub-select sigmas if n_sampling_steps < schedule.n_steps
    if n_steps < schedule.n_steps:
        indices = torch.linspace(0, schedule.n_steps - 1, n_steps).long()
        sigmas = schedule.sigmas[indices].to(device)
    else:
        sigmas = schedule.sigmas.to(device)

    # Start from noise at the largest sigma
    x = sigmas[0] * torch.randn(B, n_out, device=device)

    for i in range(len(sigmas)):
        sigma_cur = sigmas[i]
        sigma_next = sigmas[i + 1] if i + 1 < len(sigmas) else torch.tensor(0.0, device=device)
        sigma_batch = sigma_cur.expand(B)  # (B,)

        # Predict clean x₀
        x0_hat = denoiser(h, x, sigma_batch)

        if sigma_next > 0:
            # Deterministic (DDIM) update: interpolate toward x0_hat
            # so that the result has noise level σ_next
            coeff = sigma_next / sigma_cur
            x = x0_hat + coeff * (x - x0_hat)

            if stochastic:
                # Ancestral noise injection: add fresh noise and
                # compensate by moving closer to x0_hat.
                # noise_level² = σ_next² − (σ_next²/σ_cur²)·σ_cur² ... = 0
                # Instead, use the "churn" approach: inject small noise η
                eta = 0.5  # noise amount (0 = deterministic, 1 = full)
                noise_level = eta * sigma_next * math.sqrt(1 - coeff ** 2)
                x = x + noise_level * torch.randn_like(x)
        else:
            x = x0_hat  # final step: just use the prediction

    return x


# ── Main model ───────────────────────────────────────────────────────────────


class TemporalTransformerGaussian(nn.Module):
    """Causal Transformer predicting next-step Gaussian (μ, σ) per channel.

    The model accepts a joint state ``x = [u, b]`` where ``u`` is neural
    activity (n_neural columns, always valid) and ``b`` is behaviour
    eigenworm amplitudes (n_beh columns, possibly masked).

    Parameters
    ----------
    n_neural : int
        Number of neural columns.
    n_beh : int
        Number of behaviour columns (0 if not predicting behaviour).
    cfg : TransformerBaselineConfig
        Hyperparameters.
    """

    def __init__(self, n_neural: int, n_beh: int, cfg: TransformerBaselineConfig):
        super().__init__()
        self.n_neural = n_neural
        self.n_beh = n_beh
        self.n_obs = n_neural + n_beh   # total input dimension
        self.cfg = cfg

        D = self.n_obs  # input dimension

        # Input projection: (B, K, D) → (B, K, d_model)
        self.input_proj = nn.Linear(D, cfg.d_model)

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

        # Output heads — separate for neural and behaviour
        self.neural_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, 2 * n_neural),
        )
        if n_beh > 0:
            self.beh_head = nn.Sequential(
                nn.LayerNorm(cfg.d_model),
                nn.Linear(cfg.d_model, 2 * n_beh),
            )
        else:
            self.beh_head = None

        # Bounds for σ (Gaussian mode)
        self.sigma_min = cfg.sigma_min
        self.sigma_max = cfg.sigma_max

        # ── Diffusion heads (alternative to Gaussian) ──
        self.use_diffusion = getattr(cfg, "use_diffusion", False)
        if self.use_diffusion:
            self.diff_schedule = DiffusionSchedule(
                n_steps=getattr(cfg, "diffusion_steps", 100),
                sigma_min=getattr(cfg, "diffusion_sigma_min", 0.05),
                sigma_max=getattr(cfg, "diffusion_sigma_max", 10.0),
                schedule=getattr(cfg, "diffusion_schedule", "log_linear"),
            )
            self.diff_sampling_steps = getattr(cfg, "diffusion_sampling_steps", 50)
            # Context projection: take transformer embedding for conditioning
            self.diff_ctx_proj = nn.Sequential(
                nn.LayerNorm(cfg.d_model),
                nn.Linear(cfg.d_model, cfg.d_model),
                nn.SiLU(),
            )
            # Neural denoiser (sigma_data ≈ 1.0 for z-scored neural data)
            self.neural_denoiser = DiffusionDenoiserHead(
                d_model=cfg.d_model, n_out=n_neural,
                hidden=max(256, cfg.d_model), n_layers=3,
                sigma_data=1.0,
            )
            # Behaviour denoiser (sigma_data ≈ 5.0 for behaviour modes)
            if n_beh > 0:
                self.beh_denoiser = DiffusionDenoiserHead(
                    d_model=cfg.d_model, n_out=n_beh,
                    hidden=max(256, cfg.d_model), n_layers=3,
                    sigma_data=5.0,
                )
            else:
                self.beh_denoiser = None

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

    def _split_mu_sigma(
        self, raw: torch.Tensor, n: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split a raw (…, 2*n) tensor into μ and σ."""
        mu = raw[..., :n]
        log_sigma = raw[..., n:]
        sigma = F.softplus(log_sigma).clamp(self.sigma_min, self.sigma_max)
        return mu, sigma

    def forward(
        self,
        context: torch.Tensor,
        return_all_steps: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass.

        Parameters
        ----------
        context : (B, K, D) float
            Context window of joint state [u, b].
        return_all_steps : bool
            If True, return predictions for *every* timestep in the context
            window (useful for free-run / sampling).

        Returns
        -------
        mu_u   : (B, N_neural) or (B, K, N_neural)
        sigma_u: (B, N_neural) or (B, K, N_neural)
        mu_b   : (B, N_beh) or (B, K, N_beh)  — None if n_beh == 0
        sigma_b: (B, N_beh) or (B, K, N_beh)  — None if n_beh == 0
        """
        B, K, D = context.shape
        assert D == self.n_obs, f"Expected D={self.n_obs}, got {D}"

        x = self.input_proj(context)            # (B, K, d_model)
        x = self.pos_enc(x)                     # (B, K, d_model)

        # Causal mask for this sequence length
        causal = self._causal_mask[:K, :K]      # (K, K)
        x = self.transformer(x, mask=causal)    # (B, K, d_model)

        if return_all_steps:
            h = x                                           # (B, K, d_model)
        else:
            h = x[:, -1, :]                                 # (B, d_model)

        # Neural head
        mu_u, sigma_u = self._split_mu_sigma(self.neural_head(h), self.n_neural)

        # Behaviour head
        if self.beh_head is not None:
            mu_b, sigma_b = self._split_mu_sigma(self.beh_head(h), self.n_beh)
        else:
            mu_b, sigma_b = None, None

        return mu_u, sigma_u, mu_b, sigma_b

    def _get_context_embedding(self, context: torch.Tensor) -> torch.Tensor:
        """Run transformer backbone, return last-step embedding (B, d_model)."""
        B, K, D = context.shape
        x = self.input_proj(context)
        x = self.pos_enc(x)
        causal = self._causal_mask[:K, :K]
        x = self.transformer(x, mask=causal)
        return x[:, -1, :]  # (B, d_model)

    def diffusion_loss(
        self, context: torch.Tensor, target: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
        w_beh: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute denoising diffusion loss (use instead of forward + NLL).

        Parameters
        ----------
        context     : (B, K, D)
        target      : (B, D)  clean next-step target
        target_mask : (B, D)  validity mask (optional)
        w_beh       : float   weight on behaviour loss

        Returns
        -------
        total, neural_loss, beh_loss : scalars
        """
        h = self._get_context_embedding(context)       # (B, d_model)
        h = self.diff_ctx_proj(h)                       # (B, d_model)

        target_u = target[:, :self.n_neural]
        # Estimate sigma_data from batch statistics (or use ~1.0 for z-scored neural)
        sd_u = target_u.std().clamp(min=0.1).item()
        neural_loss = diffusion_denoising_loss(
            self.neural_denoiser, h, target_u, self.diff_schedule,
            sigma_data=sd_u,
        )

        if self.beh_denoiser is not None and self.n_beh > 0:
            target_b = target[:, self.n_neural:]
            sd_b = target_b.std().clamp(min=0.1).item()
            if target_mask is not None:
                mask_b = target_mask[:, self.n_neural:]
                beh_loss = masked_diffusion_denoising_loss(
                    self.beh_denoiser, h, target_b, self.diff_schedule, mask_b,
                    sigma_data=sd_b,
                )
            else:
                beh_loss = diffusion_denoising_loss(
                    self.beh_denoiser, h, target_b, self.diff_schedule,
                    sigma_data=sd_b,
                )
            total = neural_loss + w_beh * beh_loss
        else:
            beh_loss = torch.tensor(0.0, device=target.device)
            total = neural_loss

        return total, neural_loss, beh_loss

    @torch.no_grad()
    def diffusion_sample(
        self, context: torch.Tensor, n_samples: int = 1,
    ) -> torch.Tensor:
        """Sample next-step predictions via iterative denoising.

        Parameters
        ----------
        context   : (B, K, D)
        n_samples : int  number of independent samples per input

        Returns
        -------
        If n_samples == 1: (B, D)
        If n_samples  > 1: (n_samples, B, D)
        """
        h = self._get_context_embedding(context)
        h = self.diff_ctx_proj(h)  # (B, d_model)

        results = []
        for _ in range(n_samples):
            u_sample = diffusion_sample(
                self.neural_denoiser, h, self.diff_schedule,
                self.n_neural, self.diff_sampling_steps,
            )
            if self.beh_denoiser is not None:
                b_sample = diffusion_sample(
                    self.beh_denoiser, h, self.diff_schedule,
                    self.n_beh, self.diff_sampling_steps,
                )
                results.append(torch.cat([u_sample, b_sample], dim=-1))
            else:
                results.append(u_sample)

        if n_samples == 1:
            return results[0]
        return torch.stack(results, dim=0)

    def predict_mean(self, context: torch.Tensor) -> torch.Tensor:
        """Return concatenated mean prediction [mu_u, mu_b] (for evaluation).

        Returns shape (B, D) where D = n_neural + n_beh.
        In diffusion mode, averages N deterministic denoised samples to
        get a stable "mean" estimate (single stochastic samples are too noisy).
        """
        if self.use_diffusion:
            n_avg = 1  # deterministic DDIM with Karras skip is consistent
            samples = self.diffusion_sample(context, n_samples=n_avg)
            return samples if n_avg == 1 else samples.mean(dim=0)
        mu_u, _, mu_b, _ = self.forward(context, return_all_steps=False)
        if mu_b is not None:
            return torch.cat([mu_u, mu_b], dim=-1)
        return mu_u

    def predict_mean_split(
        self, context: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return (mu_u, mu_b) separately."""
        if self.use_diffusion:
            n_avg = 1
            samples = self.diffusion_sample(context, n_samples=n_avg)
            full = samples if n_avg == 1 else samples.mean(dim=0)
            mu_u = full[:, :self.n_neural]
            mu_b = full[:, self.n_neural:] if self.n_beh > 0 else None
            return mu_u, mu_b
        mu_u, _, mu_b, _ = self.forward(context, return_all_steps=False)
        return mu_u, mu_b

    def one_step(self, context: torch.Tensor) -> torch.Tensor:
        """Convenience alias: teacher-forced one-step prediction.

        Parameters
        ----------
        context : (B, K, D)

        Returns
        -------
        mu : (B, D)  — concatenated [mu_u, mu_b]
        """
        return self.predict_mean(context)

    @torch.no_grad()
    def free_run(
        self,
        x0: torch.Tensor,
        n_steps: int,
    ) -> torch.Tensor:
        """Autoregressive free-run from initial context window.

        Parameters
        ----------
        x0 : (K, D) or (1, K, D)
            Initial context window (joint state).
        n_steps : int
            Number of steps to predict after the context window.

        Returns
        -------
        pred : (n_steps, D)
        """
        if x0.dim() == 2:
            x0 = x0.unsqueeze(0)  # (1, K, D)
        assert x0.dim() == 3 and x0.size(0) == 1

        K = self.cfg.context_length
        buf = x0.clone()  # (1, K, D)
        preds = []
        for _ in range(n_steps):
            mu = self.predict_mean(buf[:, -K:])  # (1, D)
            preds.append(mu.squeeze(0))  # (D,)
            buf = torch.cat([buf, mu.unsqueeze(1)], dim=1)  # grow buffer

        return torch.stack(preds, dim=0)  # (n_steps, D)

    @torch.no_grad()
    def stochastic_free_run(
        self,
        x0: torch.Tensor,
        n_steps: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Stochastic autoregressive free-run: sample x_{t+1} ~ N(μ, (T·σ)²).

        The Gaussian head learns (μ_θ, σ_θ) = (E[x|h], √Var[x|h]).
        Sampling with the learned σ injects the right amount of noise
        to counterbalance the contraction of iterating the mean,
        maintaining realistic variance / PSD / autocorrelation.

        Parameters
        ----------
        x0 : (K, D) or (1, K, D)
            Initial context window (joint state).
        n_steps : int
            Number of steps to generate.
        temperature : float
            Scale factor on σ.  T=0 → deterministic (= free_run),
            T=1 → learned noise, T>1 → over-dispersed exploration.

        Returns
        -------
        pred : (n_steps, D)
        """
        if x0.dim() == 2:
            x0 = x0.unsqueeze(0)  # (1, K, D)
        assert x0.dim() == 3 and x0.size(0) == 1

        K = self.cfg.context_length
        buf = x0.clone()  # (1, K, D)
        preds = []

        for _ in range(n_steps):
            ctx = buf[:, -K:]  # (1, K, D)
            mu_u, sigma_u, mu_b, sigma_b = self.forward(ctx)

            # Concatenate neural + behaviour
            mu = torch.cat([mu_u, mu_b], dim=-1) if mu_b is not None else mu_u
            sigma = torch.cat([sigma_u, sigma_b], dim=-1) if sigma_b is not None else sigma_u

            # Sample: x ~ N(μ, (temperature · σ)²)
            eps = torch.randn_like(mu)
            sample = mu + temperature * sigma * eps

            preds.append(sample.squeeze(0))  # (D,)
            buf = torch.cat([buf, sample.unsqueeze(1)], dim=1)

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
        from step t-1.  This matches the LOO semantics in stage2.evaluate.

        Parameters
        ----------
        x_gt : (T, D) float tensor
            Full ground-truth joint state.
        neuron_idx : int
            Index of the held-out neuron (must be < n_neural).

        Returns
        -------
        pred : (T - K, D) float tensor
            One-step predictions for ALL channels.  The LOO R² is computed
            from column ``neuron_idx`` only.
        """
        T, D = x_gt.shape
        K = self.cfg.context_length
        assert 0 <= neuron_idx < self.n_neural

        device = next(self.parameters()).device
        u = x_gt.clone().to(device)

        preds = []
        loo_buf = u.clone()

        for t in range(K, T):
            ctx = loo_buf[t - K : t].unsqueeze(0)  # (1, K, D)
            mu = self.one_step(ctx)                 # (1, D)
            preds.append(mu.squeeze(0))             # (D,)
            # Replace neuron_idx's value at time t with prediction
            if t < T:
                loo_buf[t, neuron_idx] = mu[0, neuron_idx].item()

        return torch.stack(preds, dim=0)  # (T-K, D)


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


def masked_gaussian_nll_loss(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Gaussian NLL loss with a validity mask.

    Only elements where ``mask > 0.5`` contribute to the loss.

    Parameters
    ----------
    mu, sigma, target, mask : same shape (B, N) or (B, K, N).

    Returns
    -------
    Scalar loss (mean over valid elements), or 0 if no valid elements.
    """
    var = sigma ** 2
    nll = 0.5 * (math.log(2 * math.pi) + torch.log(var) + (target - mu) ** 2 / var)
    valid = mask > 0.5
    if valid.sum() == 0:
        return torch.tensor(0.0, device=mu.device, requires_grad=True)
    return nll[valid].mean()


def joint_loss(
    mu_u: torch.Tensor,
    sigma_u: torch.Tensor,
    target_u: torch.Tensor,
    mu_b: Optional[torch.Tensor],
    sigma_b: Optional[torch.Tensor],
    target_b: Optional[torch.Tensor],
    mask_b: Optional[torch.Tensor],
    w_beh: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combined neural + behaviour NLL loss.

    Returns
    -------
    total  : scalar   neural_nll + w_beh * beh_nll
    neural_nll : scalar
    beh_nll    : scalar  (0 if no behaviour)
    """
    neural_nll = gaussian_nll_loss(mu_u, sigma_u, target_u)
    if mu_b is not None and target_b is not None and mask_b is not None:
        beh_nll = masked_gaussian_nll_loss(mu_b, sigma_b, target_b, mask_b)
        total = neural_nll + w_beh * beh_nll
    else:
        beh_nll = torch.tensor(0.0, device=mu_u.device)
        total = neural_nll
    return total, neural_nll, beh_nll


# ── Factory ──────────────────────────────────────────────────────────────────


def build_model(
    n_neural: int,
    cfg: Optional[TransformerBaselineConfig] = None,
    device: str = "cpu",
    n_beh: int = 0,
) -> TemporalTransformerGaussian:
    """Convenience factory.

    Parameters
    ----------
    n_neural : int
        Number of neural channels.
    cfg : TransformerBaselineConfig
    device : str
    n_beh : int
        Number of behaviour channels (0 = neural only).
    """
    if cfg is None:
        cfg = TransformerBaselineConfig()
    model = TemporalTransformerGaussian(n_neural=n_neural, n_beh=n_beh, cfg=cfg)
    return model.to(device)
