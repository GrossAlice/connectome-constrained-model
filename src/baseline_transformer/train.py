"""Training loop with early stopping, scheduled sampling, and sweep.

Trains one TemporalTransformerGaussian per worm.  Supports:
* AdamW with gradient clipping
* Cosine LR scheduling
* Scheduled-sampling (teacher-forcing annealing)
* Early stopping on validation NLL
* JSON checkpoint logging
"""
from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import TransformerBaselineConfig
from .dataset import SlidingWindowDataset, temporal_train_val_test_split
from .model import (
    TemporalTransformerGaussian,
    build_model,
    gaussian_nll_loss,
)

__all__ = [
    "train_single_worm",
    "scheduled_sampling_prob",
]


# ── Scheduled sampling ──────────────────────────────────────────────────────


def scheduled_sampling_prob(epoch: int, cfg: TransformerBaselineConfig) -> float:
    """Probability of using *ground truth* (teacher forcing) at this epoch.

    Linearly anneals from 1.0 at ``ss_start_epoch`` to ``ss_p_min`` at
    ``ss_end_epoch``, then stays constant.
    """
    if epoch < cfg.ss_start_epoch:
        return 1.0
    if epoch >= cfg.ss_end_epoch:
        return cfg.ss_p_min
    frac = (epoch - cfg.ss_start_epoch) / max(1, cfg.ss_end_epoch - cfg.ss_start_epoch)
    return 1.0 - frac * (1.0 - cfg.ss_p_min)


# ── One training epoch ──────────────────────────────────────────────────────


def _train_epoch(
    model: TemporalTransformerGaussian,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
    ss_p_gt: float = 1.0,
) -> float:
    """Run one epoch of training, return mean NLL loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for context, target in loader:
        context = context.to(device)   # (B, K, N)
        target = target.to(device)     # (B, N)

        # Scheduled sampling: with probability (1 - ss_p_gt), corrupt the
        # last frame of the context with the model's own prediction from
        # the context[:-1] window.
        if ss_p_gt < 1.0 and torch.rand(1).item() > ss_p_gt:
            with torch.no_grad():
                if context.size(1) > 1:
                    mu_prev, _ = model(context[:, :-1], return_all_steps=False)
                    context = context.clone()
                    context[:, -1] = mu_prev

        mu, sigma = model(context, return_all_steps=False)
        loss = gaussian_nll_loss(mu, sigma, target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def _eval_epoch(
    model: TemporalTransformerGaussian,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate on validation set, return mean NLL loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for context, target in loader:
        context = context.to(device)
        target = target.to(device)

        mu, sigma = model(context, return_all_steps=False)
        loss = gaussian_nll_loss(mu, sigma, target)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ── Main single-worm training ───────────────────────────────────────────────


def train_single_worm(
    u: np.ndarray,
    cfg: TransformerBaselineConfig,
    device: str = "cpu",
    verbose: bool = True,
    save_dir: Optional[str] = None,
    worm_id: str = "worm",
) -> Dict[str, Any]:
    """Train a Transformer baseline for a single worm.

    Parameters
    ----------
    u : (T, N_obs) float32
        Deconvolved neural activity.
    cfg : TransformerBaselineConfig
    device : str
    verbose : bool
    save_dir : Optional path to save model + logs.
    worm_id : str for logging.

    Returns
    -------
    dict with:
        model     : trained TemporalTransformerGaussian (on cpu)
        best_val  : best validation NLL
        history   : list of dicts per epoch
        split     : train/val/test boundaries
        cfg       : config used
    """
    dev = torch.device(device)
    T, N_obs = u.shape

    # ---- Temporal split ----
    split = temporal_train_val_test_split(T, cfg.train_frac, cfg.val_frac)
    tr_s, tr_e = split["train"]
    va_s, va_e = split["val"]

    if verbose:
        print(f"[{worm_id}] T={T}, N={N_obs}, "
              f"train=[{tr_s},{tr_e}), val=[{va_s},{va_e}), "
              f"test=[{split['test'][0]},{split['test'][1]})")

    # ---- Datasets ----
    train_ds = SlidingWindowDataset(u, cfg.context_length, start=tr_s, end=tr_e)
    val_ds = SlidingWindowDataset(u, cfg.context_length, start=va_s, end=va_e)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(f"[{worm_id}] Empty dataset: train={len(train_ds)}, val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=(device != "cpu"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=(device != "cpu"),
    )

    # ---- Model & optimizer ----
    model = build_model(N_obs, cfg, device=device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"[{worm_id}] Model: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.max_epochs,
        eta_min=cfg.lr * 0.01,
    )

    # ---- Training loop ----
    best_val = float("inf")
    best_state = None
    patience_counter = 0
    history: List[Dict[str, Any]] = []

    t0 = time.time()

    for epoch in range(1, cfg.max_epochs + 1):
        ss_p = scheduled_sampling_prob(epoch, cfg)

        train_loss = _train_epoch(
            model, train_loader, optimizer, dev,
            grad_clip=cfg.grad_clip,
            ss_p_gt=ss_p,
        )
        val_loss = _eval_epoch(model, val_loader, dev)
        scheduler.step()

        record = {
            "epoch": epoch,
            "train_nll": round(train_loss, 6),
            "val_nll": round(val_loss, 6),
            "lr": round(optimizer.param_groups[0]["lr"], 8),
            "ss_p_gt": round(ss_p, 4),
        }
        history.append(record)

        improved = val_loss < best_val - 1e-6
        if improved:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch <= 5 or epoch % 10 == 0 or improved):
            tag = " *" if improved else ""
            print(f"  epoch {epoch:4d}  train={train_loss:.4f}  "
                  f"val={val_loss:.4f}  ss={ss_p:.2f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}{tag}")

        if patience_counter >= cfg.patience:
            if verbose:
                print(f"  Early stop at epoch {epoch} (patience={cfg.patience})")
            break

    elapsed = time.time() - t0
    if verbose:
        print(f"[{worm_id}] Training done in {elapsed:.1f}s, best val NLL={best_val:.4f}")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.cpu().eval()

    result = {
        "model": model,
        "best_val_nll": float(best_val),
        "history": history,
        "split": split,
        "cfg": cfg,
        "n_params": n_params,
        "worm_id": worm_id,
        "elapsed_s": round(elapsed, 1),
    }

    # ---- Save ----
    if save_dir is not None:
        out = Path(save_dir) / worm_id
        out.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out / "model.pt")
        meta = {
            "worm_id": worm_id,
            "n_obs": N_obs,
            "T": T,
            "n_params": n_params,
            "best_val_nll": float(best_val),
            "elapsed_s": round(elapsed, 1),
            "split": {k: list(v) for k, v in split.items()},
        }
        (out / "train_meta.json").write_text(json.dumps(meta, indent=2))
        (out / "history.json").write_text(json.dumps(history, indent=2))

    return result
