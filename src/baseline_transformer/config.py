"""Configuration dataclass for the Transformer baseline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class TransformerBaselineConfig:
    # ── Architecture ──
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1
    context_length: int = 16          # K past frames as context

    # ── Training ──
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 200
    patience: int = 20                # early stopping on val NLL
    grad_clip: float = 1.0
    device: str = "cuda"

    # ── Scheduled sampling ──
    ss_start_epoch: int = 10          # epoch to begin annealing
    ss_end_epoch: int = 100           # epoch at which p_teacher = ss_p_min
    ss_p_min: float = 0.5            # minimum teacher-forcing probability

    # ── Data ──
    train_frac: float = 0.7
    val_frac: float = 0.15
    # test_frac = 1 - train_frac - val_frac
    behavior_lag_steps: int = 8       # lags for behaviour decoder

    # ── Cross-validation ──
    n_cv_folds: int = 5               # number of contiguous temporal CV folds
    val_frac_inner: float = 0.15      # fraction of training folds used as
                                      # inner validation for early stopping

    # ── Behaviour joint prediction ──
    n_beh_modes: int = 6              # number of eigenworm modes to use
    w_beh: float = 1.0               # weight on behaviour NLL loss
    include_beh_input: bool = True    # feed behaviour amplitudes as input
    predict_beh: bool = True          # predict behaviour as an output

    # ── Output ──
    sigma_min: float = 1e-4           # floor for predicted std
    sigma_max: float = 10.0           # ceiling for predicted std

    # ── Diffusion output head (alternative to Gaussian) ──
    use_diffusion: bool = False       # if True, use denoising diffusion head
    diffusion_steps: int = 100        # number of noise levels for training
    diffusion_sampling_steps: int = 20  # DDPM steps at inference (can be < diffusion_steps)
    diffusion_sigma_min: float = 0.05 # smallest noise level
    diffusion_sigma_max: float = 10.0 # largest noise level (should be ≥ data std)
    diffusion_schedule: str = "log_linear"  # "log_linear" or "cosine"

    # ── Hyperparameter sweep grid ──
    sweep_d_model: Tuple[int, ...] = (64, 128)
    sweep_n_layers: Tuple[int, ...] = (2, 3)
    sweep_context_length: Tuple[int, ...] = (8, 16, 32)
    sweep_dropout: Tuple[float, ...] = (0.1, 0.2)
    sweep_n_folds: int = 3            # temporal CV folds for sweep
