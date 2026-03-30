"""Transformer baseline for C. elegans neural dynamics prediction.

This package provides a temporal causal Transformer that predicts
next-step neural activity from a context window of past observations.
It serves as a connectome-free baseline for comparison against the
connectome-constrained Stage-2 model.

Modules
-------
config   : TransformerBaselineConfig dataclass
dataset  : SlidingWindowDataset, temporal splits, data loading
model    : TemporalTransformerGaussian with Gaussian output heads
train    : Training loop with scheduled sampling and early stopping
evaluate : One-step, LOO, free-run, and behaviour R² evaluation
run_baseline : CLI entry point
"""

from .config import TransformerBaselineConfig
from .model import TemporalTransformerGaussian, build_model, gaussian_nll_loss
from .dataset import SlidingWindowDataset, load_worm_data, discover_worm_files
from .train import train_single_worm
from .evaluate import run_full_evaluation
