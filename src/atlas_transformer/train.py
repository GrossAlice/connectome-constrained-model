"""Training loop for the Atlas Transformer.

Trains a single unified model on concatenated data from all worms.
Each worm's neurons are embedded in the 302-neuron atlas, with masked
Gaussian NLL loss so only observed neurons contribute.

Supports joint neural + behaviour prediction matching the baseline
transformer.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

from .config import AtlasTransformerConfig
from .model import (
    AtlasTransformerGaussian,
    build_atlas_model,
    masked_gaussian_nll_loss,
    joint_loss,
)

__all__ = ["train_atlas_model", "train_atlas_model_cv"]


# ── Scheduled sampling ──────────────────────────────────────────────────────


def _scheduled_sampling_prob(epoch: int, cfg: AtlasTransformerConfig) -> float:
    """Probability of using ground truth (teacher-forcing) at this epoch."""
    if epoch < cfg.ss_start_epoch:
        return 1.0
    if epoch >= cfg.ss_end_epoch:
        return cfg.ss_p_min
    frac = (epoch - cfg.ss_start_epoch) / max(cfg.ss_end_epoch - cfg.ss_start_epoch, 1)
    return 1.0 - frac * (1.0 - cfg.ss_p_min)


# ── Train / eval epochs ─────────────────────────────────────────────────────


def _train_epoch(
    model: AtlasTransformerGaussian,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
    ss_p_gt: float = 1.0,
    w_beh: float = 1.0,
) -> Dict[str, float]:
    """One training epoch. Returns mean losses."""
    model.train()
    total_loss = 0.0
    total_neural = 0.0
    total_beh = 0.0
    n_batches = 0
    n_atlas = model.n_atlas

    for context, target, target_mask in loader:
        context = context.to(device)          # (B, K, D)
        target = target.to(device)            # (B, D)
        target_mask = target_mask.to(device)  # (B, D)

        # Scheduled sampling: corrupt last frame
        if ss_p_gt < 1.0 and torch.rand(1).item() > ss_p_gt:
            with torch.no_grad():
                if context.size(1) > 1:
                    mu_u_ss, mu_b_ss = model.predict_mean_split(context[:, :-1])
                    context = context.clone()
                    # Joint-state layout: [u_atlas, obs_mask, beh]
                    # Replace neural activity with predicted mean, keep obs_mask
                    context[:, -1, :n_atlas] = mu_u_ss
                    if mu_b_ss is not None:
                        context[:, -1, 2*n_atlas:] = mu_b_ss

        mu_u, sigma_u, mu_b, sigma_b = model(context)

        # Split target and mask
        target_u = target[:, :n_atlas]
        obs_mask = target_mask[:, :n_atlas]

        target_b = target[:, 2*n_atlas:] if model.n_beh > 0 else None
        mask_b = target_mask[:, 2*n_atlas:] if model.n_beh > 0 else None

        loss, neural_nll, beh_nll = joint_loss(
            mu_u, sigma_u, target_u, obs_mask,
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
    model: AtlasTransformerGaussian,
    loader: DataLoader,
    device: torch.device,
    w_beh: float = 1.0,
) -> Dict[str, float]:
    """One validation epoch. Returns mean losses."""
    model.eval()
    total_loss = 0.0
    total_neural = 0.0
    total_beh = 0.0
    n_batches = 0
    n_atlas = model.n_atlas

    for context, target, target_mask in loader:
        context = context.to(device)
        target = target.to(device)
        target_mask = target_mask.to(device)

        mu_u, sigma_u, mu_b, sigma_b = model(context)

        target_u = target[:, :n_atlas]
        obs_mask = target_mask[:, :n_atlas]
        target_b = target[:, 2*n_atlas:] if model.n_beh > 0 else None
        mask_b = target_mask[:, 2*n_atlas:] if model.n_beh > 0 else None

        loss, neural_nll, beh_nll = joint_loss(
            mu_u, sigma_u, target_u, obs_mask,
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


# ── Main training function ──────────────────────────────────────────────────


def train_atlas_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: AtlasTransformerConfig,
    device: str = "cuda",
    save_dir: Optional[str] = None,
    verbose: bool = True,
    n_beh: int = 0,
) -> Dict[str, Any]:
    """Train the unified Atlas Transformer.

    Parameters
    ----------
    train_loader : multi-worm concatenated training loader
    val_loader   : multi-worm concatenated validation loader
    cfg          : AtlasTransformerConfig
    device       : "cuda" or "cpu"
    save_dir     : directory to save model + logs
    verbose      : print progress
    n_beh        : number of behaviour output channels

    Returns
    -------
    dict with model, best_val_nll, history, cfg
    """
    dev = torch.device(device)

    model = build_atlas_model(cfg, device=device, n_beh=n_beh)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"[Atlas] Model: {n_params:,} parameters")
        print(f"[Atlas] Input dim: {cfg.input_dim} (= 2×{cfg.n_atlas}"
              f"{f' + {n_beh} beh' if n_beh > 0 else ''})")
        print(f"[Atlas] Architecture: d_model={cfg.d_model}, n_heads={cfg.n_heads}, "
              f"n_layers={cfg.n_layers}, d_ff={cfg.d_ff}, K={cfg.context_length}")

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

    best_val = float("inf")
    best_state = None
    patience_counter = 0
    history: List[Dict[str, Any]] = []

    t0 = time.time()

    for epoch in range(1, cfg.max_epochs + 1):
        ss_p = _scheduled_sampling_prob(epoch, cfg)

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

        if verbose and (epoch <= 5 or epoch % 10 == 0 or improved):
            tag = " *" if improved else ""
            print(
                f"  epoch {epoch:4d}  tr={train_metrics['loss']:.4f} "
                f"(u={train_metrics['neural_nll']:.4f} b={train_metrics['beh_nll']:.4f})  "
                f"val={val_loss:.4f}  ss={ss_p:.2f}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}{tag}"
            )

        if patience_counter >= cfg.patience:
            if verbose:
                print(f"  Early stop at epoch {epoch} (patience={cfg.patience})")
            break

    elapsed = time.time() - t0
    if verbose:
        print(f"[Atlas] Training done in {elapsed:.1f}s, best val NLL={best_val:.4f}")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.cpu().eval()

    result = {
        "model": model,
        "best_val_nll": float(best_val),
        "history": history,
        "cfg": cfg,
        "n_params": n_params,
        "n_beh": n_beh,
        "elapsed_s": round(elapsed, 1),
    }

    # Save
    if save_dir is not None:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out / "atlas_model.pt")
        meta = {
            "n_atlas": cfg.n_atlas,
            "n_beh": n_beh,
            "n_params": n_params,
            "best_val_nll": float(best_val),
            "elapsed_s": round(elapsed, 1),
            "d_model": cfg.d_model,
            "n_heads": cfg.n_heads,
            "n_layers": cfg.n_layers,
            "d_ff": cfg.d_ff,
            "context_length": cfg.context_length,
        }
        (out / "train_meta.json").write_text(json.dumps(meta, indent=2))
        (out / "history.json").write_text(json.dumps(history, indent=2))

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  5-fold CV training (per-worm, atlas-space)
# ══════════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def _collect_fold_predictions_atlas(
    model: AtlasTransformerGaussian,
    x: np.ndarray,
    test_start: int,
    test_end: int,
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    """One-step teacher-forced predictions on [test_start, test_end).

    Works with packed joint state x = [u_atlas, obs_mask, beh].

    Returns
    -------
    dict with pred_u (T_test, n_atlas), pred_b (T_test, n_beh) or None,
    gt_u, gt_b
    """
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()
    K = model.cfg.context_length
    n_atlas = model.n_atlas
    n_beh = model.n_beh

    x_t = torch.tensor(x, dtype=torch.float32, device=dev)

    first = max(K, test_start)
    if first >= test_end:
        empty_u = np.zeros((0, n_atlas), dtype=np.float32)
        empty_b = np.zeros((0, n_beh), dtype=np.float32) if n_beh > 0 else None
        return {"pred_u": empty_u, "pred_b": empty_b,
                "gt_u": empty_u, "gt_b": empty_b}

    preds_u, preds_b = [], []
    for t in range(first, test_end):
        ctx = x_t[t - K : t].unsqueeze(0)
        mu_u, mu_b = model.predict_mean_split(ctx)
        preds_u.append(mu_u.squeeze(0).cpu().numpy())
        if mu_b is not None:
            preds_b.append(mu_b.squeeze(0).cpu().numpy())

    pred_u = np.stack(preds_u, axis=0)
    gt_u = x[first:test_end, :n_atlas]
    pred_b = np.stack(preds_b, axis=0) if preds_b else None
    gt_b = x[first:test_end, 2 * n_atlas:] if n_beh > 0 else None

    model = model.cpu()
    return {"pred_u": pred_u, "pred_b": pred_b,
            "gt_u": gt_u, "gt_b": gt_b}


def _train_one_fold_atlas(
    x: np.ndarray,
    obs_mask: np.ndarray,
    b_mask_out: Optional[np.ndarray],
    fold_spec: Dict[str, Any],
    fold_idx: int,
    cfg: AtlasTransformerConfig,
    device: str = "cpu",
    verbose: bool = True,
    n_beh: int = 0,
) -> Dict[str, Any]:
    """Train one atlas model on one CV fold (disjoint train segments + inner val)."""
    from .dataset import SlidingWindowAtlasDataset

    dev = torch.device(device)

    # Build training dataset from (possibly disjoint) segments
    train_datasets = []
    for seg_s, seg_e in fold_spec["train"]:
        ds = SlidingWindowAtlasDataset(
            x, n_atlas=cfg.n_atlas, context_length=cfg.context_length,
            obs_mask=obs_mask, b_mask=b_mask_out,
            start=seg_s, end=seg_e,
        )
        if len(ds) > 0:
            train_datasets.append(ds)

    if not train_datasets:
        raise ValueError(f"Fold {fold_idx}: empty training set")
    train_ds = (ConcatDataset(train_datasets)
                if len(train_datasets) > 1 else train_datasets[0])

    # Inner validation dataset
    val_s, val_e = fold_spec["val"]
    val_ds = SlidingWindowAtlasDataset(
        x, n_atlas=cfg.n_atlas, context_length=cfg.context_length,
        obs_mask=obs_mask, b_mask=b_mask_out,
        start=val_s, end=val_e,
    )

    if verbose:
        tr_total = sum(len(d) for d in train_datasets)
        print(f"  [fold {fold_idx}] train={tr_total} samples, "
              f"val={len(val_ds)} samples")

    use_pin = device != "cpu"
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        drop_last=False, pin_memory=use_pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        drop_last=False, pin_memory=use_pin,
    )

    # Fresh model per fold
    model = build_atlas_model(cfg, device=device, n_beh=n_beh)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

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
        ss_p = _scheduled_sampling_prob(epoch, cfg)

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
                  f"(u={train_metrics['neural_nll']:.4f} "
                  f"b={train_metrics['beh_nll']:.4f})  "
                  f"val={val_loss:.4f}  ss={ss_p:.2f}{tag}")

        if patience_counter >= cfg.patience:
            if verbose:
                print(f"    Early stop at epoch {epoch}")
            break

    elapsed = time.time() - t0
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


def train_atlas_model_cv(
    worm: Dict[str, Any],
    cfg: AtlasTransformerConfig,
    device: str = "cuda",
    verbose: bool = True,
    save_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """5-fold temporal CV for a single worm using the Atlas Transformer.

    Trains one fresh model per fold, collects held-out one-step predictions,
    and stitches them into a full-length array for out-of-sample one-step R².

    Parameters
    ----------
    worm      : dict from load_atlas_worm_data
    cfg       : AtlasTransformerConfig
    device    : "cuda" or "cpu"
    verbose   : print progress
    save_dir  : optional save directory

    Returns
    -------
    dict with models, pred_u_full, pred_b_full, best_model, folds, etc.
    """
    from .dataset import build_joint_state_atlas, make_cv_folds

    u_atlas = worm["u_atlas"]          # (T, N_atlas)
    obs_mask = worm["obs_mask"]        # (N_atlas,) bool
    b = worm.get("b")                 # (T, n_beh) or None
    b_mask = worm.get("b_mask")       # (T, n_beh) or None
    worm_id = worm["worm_id"]
    T = worm["T"]
    n_atlas = cfg.n_atlas

    include_beh = cfg.predict_beh and cfg.include_beh_input
    n_beh = cfg.n_beh_modes if include_beh else 0

    # Build packed joint state
    x, b_mask_out, _n_beh = build_joint_state_atlas(
        u_atlas, obs_mask, b if include_beh else None,
        b_mask if include_beh else None,
        include_beh=include_beh,
    )

    K = cfg.context_length
    n_folds = cfg.n_cv_folds

    if verbose:
        print(f"\n[Atlas-CV] {worm_id}: T={T}, N_obs={worm['N_obs']}, "
              f"n_beh={n_beh}, n_folds={n_folds}")

    # Create CV folds (same logic as baseline transformer)
    folds = make_cv_folds(T, n_folds, K, cfg.val_frac_inner)

    # Storage for stitched held-out predictions
    pred_u_full = np.full((T, n_atlas), np.nan, dtype=np.float32)
    pred_b_full = (np.full((T, n_beh), np.nan, dtype=np.float32)
                   if n_beh > 0 else None)

    fold_results = []

    for fi, fold_spec in enumerate(folds):
        te_s, te_e = fold_spec["test"]
        if verbose:
            print(f"\n[Atlas-CV] {worm_id} — Fold {fi+1}/{n_folds}  "
                  f"test=[{te_s},{te_e})  "
                  f"train={fold_spec['train']}  "
                  f"val={fold_spec['val']}")

        # Train this fold
        fold_res = _train_one_fold_atlas(
            x, obs_mask, b_mask_out, fold_spec, fi,
            cfg, device=device, verbose=verbose, n_beh=n_beh,
        )

        # Collect held-out predictions on test region
        preds = _collect_fold_predictions_atlas(
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
            print(f"  [fold {fi}] best val loss = "
                  f"{fold_res['best_val_loss']:.4f}, "
                  f"time = {fold_res['elapsed_s']:.1f}s")

    # Keep the best-fold model for LOO / free-run
    best_fold_idx = int(np.argmin(
        [fr["best_val_loss"] for fr in fold_results]
    ))
    best_model = fold_results[best_fold_idx]["model"]

    result = {
        "models": [fr["model"] for fr in fold_results],
        "best_model": best_model,
        "best_fold_idx": best_fold_idx,
        "fold_results": fold_results,
        "folds": folds,
        "pred_u_full": pred_u_full,
        "pred_b_full": pred_b_full,
        "x": x,
        "b_mask_out": b_mask_out,
        "n_beh": n_beh,
        "obs_mask": obs_mask,
        "worm_id": worm_id,
        "cfg": cfg,
    }

    # Save per-fold models + metadata
    if save_dir is not None:
        out = Path(save_dir) / worm_id
        out.mkdir(parents=True, exist_ok=True)
        for fi, fr in enumerate(fold_results):
            torch.save(fr["model"].state_dict(),
                       out / f"atlas_model_fold{fi}.pt")
        meta = {
            "worm_id": worm_id,
            "n_atlas": n_atlas,
            "n_beh": n_beh,
            "T": T,
            "N_obs": worm["N_obs"],
            "n_params": fold_results[0]["n_params"],
            "n_folds": n_folds,
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
        (out / "train_cv_meta.json").write_text(json.dumps(meta, indent=2))
        for fi, fr in enumerate(fold_results):
            (out / f"history_fold{fi}.json").write_text(
                json.dumps(fr["history"], indent=2)
            )

    return result
