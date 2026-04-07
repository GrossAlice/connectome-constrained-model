"""Configuration for the Atlas-indexed Transformer.

Aligned with ``baseline_transformer.config.TransformerBaselineConfig`` so
that the two are directly comparable.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AtlasTransformerConfig:
    """Configuration for the Atlas-indexed Transformer.

    Defaults match the baseline ``TransformerBaselineConfig`` (B_wide_256h8)
    adapted for the 302-neuron atlas input.
    """

    # ── Atlas ──
    n_atlas: int = 302               # canonical C. elegans atlas size

    # ── Architecture ──
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 2
    d_ff: int = 512
    dropout: float = 0.1
    context_length: int = 16          # K past frames as context

    # ── Training ──
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 300
    patience: int = 30                # early stopping on val NLL
    grad_clip: float = 1.0
    device: str = "cuda"

    # ── Scheduled sampling ──
    ss_start_epoch: int = 10
    ss_end_epoch: int = 150
    ss_p_min: float = 0.5

    # ── Data ──
    train_frac: float = 0.7
    val_frac: float = 0.15
    behavior_lag_steps: int = 8

    # ── Cross-validation ──
    n_cv_folds: int = 5               # contiguous temporal CV folds
    val_frac_inner: float = 0.15      # inner val for early stopping

    # ── Behaviour joint prediction ──
    n_beh_modes: int = 6              # eigenworm modes
    w_beh: float = 1.0               # weight on behaviour NLL loss
    include_beh_input: bool = True    # feed behaviour as input
    predict_beh: bool = True          # predict behaviour as output

    # ── Input encoding ──
    # "concat"           : legacy — concat [u, mask, beh] → single Linear
    # "multiply_project" : separate projections + learned missing-neuron value
    input_mode: str = "multiply_project"

    # ── Output bounds ──
    sigma_min: float = 1e-4
    sigma_max: float = 10.0

    @property
    def input_dim(self) -> int:
        """Input dimension per frame.

        Layout: [u_atlas (N_atlas), obs_mask (N_atlas), beh (n_beh)]
        where n_beh = n_beh_modes if include_beh_input else 0.
        """
        base = 2 * self.n_atlas  # activity + observation mask
        if self.include_beh_input and self.predict_beh:
            base += self.n_beh_modes
        return base
