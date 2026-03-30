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
    lr: float = 3e-4
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

    # ── Output ──
    sigma_min: float = 1e-4           # floor for predicted std
    sigma_max: float = 10.0           # ceiling for predicted std

    # ── Hyperparameter sweep grid ──
    sweep_d_model: Tuple[int, ...] = (64, 128)
    sweep_n_layers: Tuple[int, ...] = (2, 3)
    sweep_context_length: Tuple[int, ...] = (8, 16, 32)
    sweep_dropout: Tuple[float, ...] = (0.1, 0.2)
    sweep_n_folds: int = 3            # temporal CV folds for sweep
