"""Training loop with early stopping, scheduled sampling, and 5-fold CV.

Trains one TemporalTransformerGaussian per worm.  Supports:
* 5-fold contiguous temporal cross-validation
* Inner val split for early stopping within each fold
* Joint neural + behaviour loss with masked behaviour NLL
* AdamW with gradient clipping
* Cosine LR scheduling
* Scheduled-sampling (teacher-forcing annealing)
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
from torch.utils.data import DataLoader, ConcatDataset

from .config import TransformerBaselineConfig
from .dataset import (
    SlidingWindowDataset,
    make_cv_folds,
    build_joint_state,
)
from .model import (
    TemporalTransformerGaussian,
    build_model,
    gaussian_nll_loss,
    masked_gaussian_nll_loss,
    joint_loss,
)

__all__ = [
    "train_single_worm",
    "train_single_worm_cv",
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
    w_beh: float = 1.0,
) -> Dict[str, float]:
    """Run one epoch of training, return mean losses."""
    model.train()
    total_loss = 0.0
    total_neural = 0.0
    total_beh = 0.0
    n_batches = 0
    n_neural = model.n_neural
    use_diff = getattr(model, "use_diffusion", False)

    for context, target, target_mask in loader:
        context = context.to(device)        # (B, K, D)
        target = target.to(device)          # (B, D)
        target_mask = target_mask.to(device) # (B, D)

        # Scheduled sampling: with probability (1 - ss_p_gt), corrupt the
        # last frame of the context with the model's own prediction from
        # the context[:-1] window.
        # NOTE: skip in diffusion mode — predict_mean is stochastic there
        # and corrupts the context signal.
        if not use_diff and ss_p_gt < 1.0 and torch.rand(1).item() > ss_p_gt:
            with torch.no_grad():
                if context.size(1) > 1:
                    mu_prev = model.predict_mean(context[:, :-1])  # (B, D)
                    context = context.clone()
                    context[:, -1] = mu_prev

        if use_diff:
            # Diffusion denoising loss
            loss, neural_nll, beh_nll = model.diffusion_loss(
                context, target, target_mask, w_beh=w_beh,
            )
        else:
            mu_u, sigma_u, mu_b, sigma_b = model(context, return_all_steps=False)

            # Split target and mask
            target_u = target[:, :n_neural]
            target_b = target[:, n_neural:] if model.n_beh > 0 else None
            mask_b = target_mask[:, n_neural:] if model.n_beh > 0 else None

            loss, neural_nll, beh_nll = joint_loss(
                mu_u, sigma_u, target_u,
                mu_b, sigma_b, target_b, mask_b,
                w_beh=w_beh,
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_neural += neural_nll.item()
        total_beh += beh_nll.item()
        n_batches += 1

    denom = max(n_batches, 1)
    return {
        "loss": total_loss / denom,
        "neural_nll": total_neural / denom,
        "beh_nll": total_beh / denom,
    }


@torch.no_grad()
def _eval_epoch(
    model: TemporalTransformerGaussian,
    loader: DataLoader,
    device: torch.device,
    w_beh: float = 1.0,
) -> Dict[str, float]:
    """Evaluate on validation set, return mean losses."""
    model.eval()
    total_loss = 0.0
    total_neural = 0.0
    total_beh = 0.0
    n_batches = 0
    n_neural = model.n_neural
    use_diff = getattr(model, "use_diffusion", False)

    for context, target, target_mask in loader:
        context = context.to(device)
        target = target.to(device)
        target_mask = target_mask.to(device)

        if use_diff:
            loss, neural_nll, beh_nll = model.diffusion_loss(
                context, target, target_mask, w_beh=w_beh,
            )
        else:
            mu_u, sigma_u, mu_b, sigma_b = model(context, return_all_steps=False)

            target_u = target[:, :n_neural]
            target_b = target[:, n_neural:] if model.n_beh > 0 else None
            mask_b = target_mask[:, n_neural:] if model.n_beh > 0 else None

            loss, neural_nll, beh_nll = joint_loss(
                mu_u, sigma_u, target_u,
                mu_b, sigma_b, target_b, mask_b,
                w_beh=w_beh,
            )

        total_loss += loss.item()
        total_neural += neural_nll.item()
        total_beh += beh_nll.item()
        n_batches += 1

    denom = max(n_batches, 1)
    return {
        "loss": total_loss / denom,
        "neural_nll": total_neural / denom,
        "beh_nll": total_beh / denom,
    }


# ── Train one fold ───────────────────────────────────────────────────────────


def _train_one_fold(
    x: np.ndarray,
    n_neural: int,
    n_beh: int,
    b_mask: Optional[np.ndarray],
    fold_spec: Dict[str, Any],
    fold_idx: int,
    cfg: TransformerBaselineConfig,
    device: str = "cpu",
    verbose: bool = True,
    worm_id: str = "worm",
) -> Dict[str, Any]:
    """Train a model on one CV fold (train segments + inner val)."""
    dev = torch.device(device)

    # Build training dataset from (possibly disjoint) segments
    train_datasets = []
    for seg_s, seg_e in fold_spec["train"]:
        ds = SlidingWindowDataset(
            x, n_neural=n_neural, context_length=cfg.context_length,
            start=seg_s, end=seg_e, b_mask=b_mask,
        )
        if len(ds) > 0:
            train_datasets.append(ds)

    if not train_datasets:
        raise ValueError(f"Fold {fold_idx}: empty training set")
    train_ds = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]

    # Inner validation dataset
    val_s, val_e = fold_spec["val"]
    val_ds = SlidingWindowDataset(
        x, n_neural=n_neural, context_length=cfg.context_length,
        start=val_s, end=val_e, b_mask=b_mask,
    )

    if verbose:
        tr_total = sum(len(d) for d in train_datasets)
        print(f"  [fold {fold_idx}] train={tr_total} samples, val={len(val_ds)} samples")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        drop_last=False, pin_memory=(device != "cpu"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        drop_last=False, pin_memory=(device != "cpu"),
    )

    # Model & optimizer (fresh per fold)
    model = build_model(n_neural, cfg, device=device, n_beh=n_beh)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.max_epochs, eta_min=cfg.lr * 0.01,
    )

    # Training loop
    best_val = float("inf")
    best_state = None
    patience_counter = 0
    history: List[Dict[str, Any]] = []
    t0 = time.time()

    for epoch in range(1, cfg.max_epochs + 1):
        ss_p = scheduled_sampling_prob(epoch, cfg)

        train_metrics = _train_epoch(
            model, train_loader, optimizer, dev,
            grad_clip=cfg.grad_clip, ss_p_gt=ss_p, w_beh=cfg.w_beh,
        )
        val_metrics = _eval_epoch(model, val_loader, dev, w_beh=cfg.w_beh)
        scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 6),
            "train_neural_nll": round(train_metrics["neural_nll"], 6),
            "train_beh_nll": round(train_metrics["beh_nll"], 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_neural_nll": round(val_metrics["neural_nll"], 6),
            "val_beh_nll": round(val_metrics["beh_nll"], 6),
            "lr": round(optimizer.param_groups[0]["lr"], 8),
            "ss_p_gt": round(ss_p, 4),
        }
        history.append(record)

        val_loss = val_metrics["loss"]
        improved = val_loss < best_val - 1e-6
        if improved:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch <= 3 or epoch % 20 == 0 or improved):
            tag = " *" if improved else ""
            print(f"    ep {epoch:4d}  tr={train_metrics['loss']:.4f} "
                  f"(u={train_metrics['neural_nll']:.4f} b={train_metrics['beh_nll']:.4f})  "
                  f"val={val_loss:.4f}  ss={ss_p:.2f}{tag}")

        if patience_counter >= cfg.patience:
            if verbose:
                print(f"    Early stop at epoch {epoch}")
            break

    elapsed = time.time() - t0

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.cpu().eval()

    return {
        "model": model,
        "best_val_loss": float(best_val),
        "history": history,
        "n_params": n_params,
        "elapsed_s": round(elapsed, 1),
    }


# ── Collect held-out predictions for one fold ────────────────────────────────


@torch.no_grad()
def _collect_fold_predictions(
    model: TemporalTransformerGaussian,
    x: np.ndarray,
    test_start: int,
    test_end: int,
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    """One-step teacher-forced predictions on [test_start, test_end).

    Returns
    -------
    dict with:
        pred_u : (T_test, n_neural)
        pred_b : (T_test, n_beh) or None
        gt_u   : (T_test, n_neural)
        gt_b   : (T_test, n_beh) or None
    """
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()
    K = model.cfg.context_length
    n_neural = model.n_neural
    n_beh = model.n_beh

    x_t = torch.tensor(x, dtype=torch.float32, device=dev)

    first = max(K, test_start)
    if first >= test_end:
        empty_u = np.zeros((0, n_neural), dtype=np.float32)
        empty_b = np.zeros((0, n_beh), dtype=np.float32) if n_beh > 0 else None
        return {"pred_u": empty_u, "pred_b": empty_b, "gt_u": empty_u, "gt_b": empty_b}

    preds_u, preds_b = [], []
    for t in range(first, test_end):
        ctx = x_t[t - K : t].unsqueeze(0)
        mu_u, mu_b = model.predict_mean_split(ctx)
        preds_u.append(mu_u.squeeze(0).cpu().numpy())
        if mu_b is not None:
            preds_b.append(mu_b.squeeze(0).cpu().numpy())

    pred_u = np.stack(preds_u, axis=0)
    gt_u = x[first:test_end, :n_neural]
    pred_b = np.stack(preds_b, axis=0) if preds_b else None
    gt_b = x[first:test_end, n_neural:] if n_beh > 0 else None

    model = model.cpu()
    return {"pred_u": pred_u, "pred_b": pred_b, "gt_u": gt_u, "gt_b": gt_b}


# ── Main 5-fold CV training ─────────────────────────────────────────────────


def train_single_worm_cv(
    u: np.ndarray,
    cfg: TransformerBaselineConfig,
    device: str = "cpu",
    verbose: bool = True,
    save_dir: Optional[str] = None,
    worm_id: str = "worm",
    b: Optional[np.ndarray] = None,
    b_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Train a Transformer baseline for a single worm with K-fold CV.

    Parameters
    ----------
    u : (T, N_obs) float32
        Deconvolved neural activity.
    cfg : TransformerBaselineConfig
    device, verbose, save_dir, worm_id : standard
    b : (T, n_beh) float32 or None
        Behaviour eigenworm amplitudes.
    b_mask : (T, n_beh) float32 or None
        Validity mask for behaviour.

    Returns
    -------
    dict with per-fold models, aggregated held-out predictions, etc.
    """
    T, N_obs = u.shape

    # Build joint state
    x, x_mask, n_neural, n_beh = build_joint_state(
        u, b, b_mask, include_beh=cfg.include_beh_input and cfg.predict_beh,
    )
    D = x.shape[1]

    if verbose:
        print(f"[{worm_id}] T={T}, N_neural={n_neural}, N_beh={n_beh}, "
              f"D={D}, n_folds={cfg.n_cv_folds}")

    # Create CV folds
    folds = make_cv_folds(T, cfg.n_cv_folds, cfg.context_length, cfg.val_frac_inner)

    # Storage for stitched held-out predictions
    K = cfg.context_length
    pred_u_full = np.full((T, n_neural), np.nan, dtype=np.float32)
    pred_b_full = np.full((T, n_beh), np.nan, dtype=np.float32) if n_beh > 0 else None

    fold_results = []

    for fi, fold_spec in enumerate(folds):
        te_s, te_e = fold_spec["test"]
        if verbose:
            print(f"\n[{worm_id}] Fold {fi+1}/{cfg.n_cv_folds}  "
                  f"test=[{te_s},{te_e})  "
                  f"train={fold_spec['train']}  "
                  f"val={fold_spec['val']}")

        # Train
        fold_res = _train_one_fold(
            x, n_neural, n_beh, x_mask,
            fold_spec, fi, cfg, device, verbose, worm_id,
        )

        # Collect held-out predictions on test region
        preds = _collect_fold_predictions(
            fold_res["model"], x, te_s, te_e, device=device,
        )

        # Stitch into full arrays
        first = max(K, te_s)
        length = preds["pred_u"].shape[0]
        pred_u_full[first : first + length] = preds["pred_u"]
        if pred_b_full is not None and preds["pred_b"] is not None:
            pred_b_full[first : first + length] = preds["pred_b"]

        fold_res["test_range"] = (te_s, te_e)
        fold_res["fold_idx"] = fi
        fold_results.append(fold_res)

        if verbose:
            print(f"  [fold {fi}] best val loss = {fold_res['best_val_loss']:.4f}, "
                  f"time = {fold_res['elapsed_s']:.1f}s")

    # Keep only the best-fold model for LOO / free-run eval
    best_fold_idx = int(np.argmin([fr["best_val_loss"] for fr in fold_results]))
    best_model = fold_results[best_fold_idx]["model"]

    result = {
        "models": [fr["model"] for fr in fold_results],
        "best_model": best_model,
        "best_fold_idx": best_fold_idx,
        "fold_results": fold_results,
        "folds": folds,
        "pred_u_full": pred_u_full,
        "pred_b_full": pred_b_full,
        "n_neural": n_neural,
        "n_beh": n_beh,
        "x": x,
        "x_mask": x_mask,
        "cfg": cfg,
        "worm_id": worm_id,
    }

    # Save
    if save_dir is not None:
        out = Path(save_dir) / worm_id
        out.mkdir(parents=True, exist_ok=True)
        for fi, fr in enumerate(fold_results):
            torch.save(fr["model"].state_dict(), out / f"model_fold{fi}.pt")
        meta = {
            "worm_id": worm_id,
            "n_neural": n_neural,
            "n_beh": n_beh,
            "T": T,
            "n_params": fold_results[0]["n_params"],
            "n_folds": cfg.n_cv_folds,
            "best_fold_idx": best_fold_idx,
            "folds": [
                {
                    "test": list(f["test"]),
                    "train": [list(s) for s in f["train"]],
                    "val": list(f["val"]),
                }
                for f in folds
            ],
            "fold_val_losses": [fr["best_val_loss"] for fr in fold_results],
            "fold_elapsed": [fr["elapsed_s"] for fr in fold_results],
        }
        (out / "train_meta.json").write_text(json.dumps(meta, indent=2))
        for fi, fr in enumerate(fold_results):
            (out / f"history_fold{fi}.json").write_text(
                json.dumps(fr["history"], indent=2)
            )

    return result


# ── Legacy single-split training (kept for backward compat) ─────────────────


def train_single_worm(
    u: np.ndarray,
    cfg: TransformerBaselineConfig,
    device: str = "cpu",
    verbose: bool = True,
    save_dir: Optional[str] = None,
    worm_id: str = "worm",
    b: Optional[np.ndarray] = None,
    b_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Train a Transformer baseline for a single worm (single split).

    This is the legacy interface. For cross-validated training, use
    ``train_single_worm_cv`` instead.

    Parameters
    ----------
    u : (T, N_obs) float32
        Deconvolved neural activity.
    cfg : TransformerBaselineConfig
    device, verbose, save_dir, worm_id : standard
    b : (T, n_beh) or None
    b_mask : (T, n_beh) or None

    Returns
    -------
    dict with model, best_val, history, split, cfg
    """
    from .dataset import temporal_train_val_test_split

    dev = torch.device(device)
    T, N_obs = u.shape

    # Build joint state
    x, x_mask, n_neural, n_beh = build_joint_state(
        u, b, b_mask, include_beh=cfg.include_beh_input and cfg.predict_beh,
    )

    # Temporal split
    split = temporal_train_val_test_split(T, cfg.train_frac, cfg.val_frac)
    tr_s, tr_e = split["train"]
    va_s, va_e = split["val"]

    if verbose:
        print(f"[{worm_id}] T={T}, N_neural={n_neural}, N_beh={n_beh}, "
              f"train=[{tr_s},{tr_e}), val=[{va_s},{va_e}), "
              f"test=[{split['test'][0]},{split['test'][1]})")

    train_ds = SlidingWindowDataset(
        x, n_neural=n_neural, context_length=cfg.context_length,
        start=tr_s, end=tr_e, b_mask=x_mask,
    )
    val_ds = SlidingWindowDataset(
        x, n_neural=n_neural, context_length=cfg.context_length,
        start=va_s, end=va_e, b_mask=x_mask,
    )

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(f"[{worm_id}] Empty dataset")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        drop_last=False, pin_memory=(device != "cpu"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        drop_last=False, pin_memory=(device != "cpu"),
    )

    model = build_model(n_neural, cfg, device=device, n_beh=n_beh)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"[{worm_id}] Model: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.max_epochs, eta_min=cfg.lr * 0.01,
    )

    best_val = float("inf")
    best_state = None
    patience_counter = 0
    history: List[Dict[str, Any]] = []
    t0 = time.time()

    for epoch in range(1, cfg.max_epochs + 1):
        ss_p = scheduled_sampling_prob(epoch, cfg)

        train_metrics = _train_epoch(
            model, train_loader, optimizer, dev,
            grad_clip=cfg.grad_clip, ss_p_gt=ss_p, w_beh=cfg.w_beh,
        )
        val_metrics = _eval_epoch(model, val_loader, dev, w_beh=cfg.w_beh)
        scheduler.step()

        record = {
            "epoch": epoch,
            "train_nll": round(train_metrics["loss"], 6),
            "val_nll": round(val_metrics["loss"], 6),
            "lr": round(optimizer.param_groups[0]["lr"], 8),
            "ss_p_gt": round(ss_p, 4),
        }
        history.append(record)

        val_loss = val_metrics["loss"]
        improved = val_loss < best_val - 1e-6
        if improved:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch <= 5 or epoch % 10 == 0 or improved):
            tag = " *" if improved else ""
            print(f"  epoch {epoch:4d}  train={train_metrics['loss']:.4f}  "
                  f"val={val_loss:.4f}  ss={ss_p:.2f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}{tag}")

        if patience_counter >= cfg.patience:
            if verbose:
                print(f"  Early stop at epoch {epoch} (patience={cfg.patience})")
            break

    elapsed = time.time() - t0
    if verbose:
        print(f"[{worm_id}] Training done in {elapsed:.1f}s, best val NLL={best_val:.4f}")

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
        "n_neural": n_neural,
        "n_beh": n_beh,
        "worm_id": worm_id,
        "elapsed_s": round(elapsed, 1),
    }

    if save_dir is not None:
        out = Path(save_dir) / worm_id
        out.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out / "model.pt")
        meta = {
            "worm_id": worm_id,
            "n_neural": n_neural,
            "n_beh": n_beh,
            "T": T,
            "n_params": n_params,
            "best_val_nll": float(best_val),
            "elapsed_s": round(elapsed, 1),
            "split": {k: list(v) for k, v in split.items()},
        }
        (out / "train_meta.json").write_text(json.dumps(meta, indent=2))
        (out / "history.json").write_text(json.dumps(history, indent=2))

    return result
