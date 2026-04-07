"""Per-worm evaluation for the Atlas Transformer.

Evaluates the unified model on each worm individually.  Now works with
the packed joint-state representation [u_atlas, obs_mask, beh] matching
the baseline transformer's evaluation pipeline.

Metrics match the per-worm baseline:
  - One-step R² (teacher-forced) — neural + behaviour
  - LOO R² (leave-one-out forward simulation)
  - Free-run R² (autoregressive)
  - Behaviour R² (ridge-CV and MLP decoders from motor neurons → eigenworms)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .config import AtlasTransformerConfig
from .model import AtlasTransformerGaussian
from .dataset import build_joint_state_atlas

# Reuse stage2 / baseline_transformer helpers
import sys
_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC))

from stage2.evaluate import _r2
from baseline_transformer.evaluate import (
    build_lagged_features_np,
    valid_lag_mask_np,
    _log_ridge_grid,
    _ridge_cv_single_target,
    compute_behaviour_r2_mlp,
)

__all__ = [
    "compute_onestep_r2_atlas",
    "compute_onestep_r2_from_preds",
    "compute_loo_r2_atlas",
    "compute_free_run_r2_atlas",
    "compute_behaviour_r2_atlas",
    "run_per_worm_evaluation",
    "run_per_worm_evaluation_cv",
]


def _per_neuron_r2_masked(
    gt: np.ndarray, pred: np.ndarray, obs_mask: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Per observed-neuron R²; returns (r2_arr[N_obs], mean)."""
    obs_idx = np.where(obs_mask)[0]
    r2_arr = np.full(len(obs_idx), np.nan)
    for i, idx in enumerate(obs_idx):
        r2_arr[i] = _r2(gt[:, idx], pred[:, idx])
    return r2_arr, float(np.nanmean(r2_arr)) if len(r2_arr) > 0 else float("nan")


# ── 1. One-step R² ──────────────────────────────────────────────────────────


@torch.no_grad()
def compute_onestep_r2_atlas(
    model: AtlasTransformerGaussian,
    x: np.ndarray,
    obs_mask: np.ndarray,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> Dict[str, Any]:
    """Teacher-forced one-step R² in atlas space, reported for observed neurons.

    Parameters
    ----------
    model    : trained AtlasTransformerGaussian
    x        : (T, D) packed joint state [u_atlas, obs_mask, beh]
    obs_mask : (N_atlas,) bool
    start, end : time range for eval (default: full)

    Returns
    -------
    dict with r2 (per observed neuron), r2_mean, pred_u, pred_b
    """
    device = next(model.parameters()).device
    model.eval()
    n_atlas = model.n_atlas
    n_beh = model.n_beh

    T, D = x.shape
    K = model.cfg.context_length
    if start is None:
        start = 0
    if end is None:
        end = T

    obs_idx = np.where(obs_mask)[0]
    N_obs = len(obs_idx)

    first_t = max(K, start + K)

    x_t = torch.tensor(x, dtype=torch.float32, device=device)
    preds_u = []
    preds_b = []

    for t in range(first_t, end):
        ctx = x_t[t - K : t].unsqueeze(0)  # (1, K, D)
        mu_u, mu_b = model.predict_mean_split(ctx)
        preds_u.append(mu_u.squeeze(0).cpu().numpy())
        if mu_b is not None:
            preds_b.append(mu_b.squeeze(0).cpu().numpy())

    if not preds_u:
        return {"r2": np.full(N_obs, np.nan), "r2_mean": float("nan"),
                "pred_u": np.array([]), "eval_range": (first_t, end)}

    pred_u = np.stack(preds_u, axis=0)  # (T_eval, N_atlas)
    gt_u = x[first_t:end, :n_atlas]

    # R² only for observed neurons
    r2_arr, r2_mean = _per_neuron_r2_masked(gt_u, pred_u, obs_mask)

    result: Dict[str, Any] = {
        "r2": r2_arr,
        "r2_mean": r2_mean,
        "pred_u": pred_u,
        "eval_range": (first_t, end),
        "obs_idx": obs_idx,
    }

    if preds_b:
        result["pred_b"] = np.stack(preds_b, axis=0)
        result["gt_b"] = x[first_t:end, 2*n_atlas:]

    return result


def compute_onestep_r2_from_preds(
    pred_u: np.ndarray,
    gt_u: np.ndarray,
    obs_mask: np.ndarray,
    pred_b: Optional[np.ndarray] = None,
    gt_b: Optional[np.ndarray] = None,
    b_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute one-step R² from pre-computed predictions (matches baseline).

    Parameters
    ----------
    pred_u, gt_u : (T, N_atlas)
    obs_mask     : (N_atlas,) bool
    pred_b, gt_b : (T, n_beh) or None
    b_mask       : (T, n_beh) or None
    """
    # Filter out NaN rows (unfilled from any stitching)
    valid = np.all(np.isfinite(pred_u), axis=1)
    pu = pred_u[valid]
    gu = gt_u[valid]

    obs_idx = np.where(obs_mask)[0]
    if pu.shape[0] == 0:
        return {"r2": np.full(len(obs_idx), np.nan), "r2_mean": float("nan")}

    r2_arr = np.full(len(obs_idx), np.nan)
    for i, idx in enumerate(obs_idx):
        r2_arr[i] = _r2(gu[:, idx], pu[:, idx])

    result: Dict[str, Any] = {
        "r2": r2_arr,
        "r2_mean": float(np.nanmean(r2_arr)),
        "n_valid_samples": int(valid.sum()),
    }

    # Behaviour R²
    if pred_b is not None and gt_b is not None:
        pb = pred_b[valid]
        gb = gt_b[valid]
        n_beh = gb.shape[1]
        beh_r2 = np.full(n_beh, np.nan)
        bm = b_mask[valid] if b_mask is not None else np.ones_like(gb)
        for j in range(n_beh):
            mask_j = bm[:, j] > 0.5
            if mask_j.sum() > 2:
                beh_r2[j] = _r2(gb[mask_j, j], pb[mask_j, j])
        result["beh_r2"] = beh_r2
        result["beh_r2_mean"] = float(np.nanmean(beh_r2))

    return result


# ── 2. LOO R² ───────────────────────────────────────────────────────────────


@torch.no_grad()
def _loo_single_neuron_atlas(
    model: AtlasTransformerGaussian,
    x: np.ndarray,
    neuron_atlas_idx: int,
) -> np.ndarray:
    """LOO forward simulation for a single atlas neuron.

    Works with packed joint state x = [u_atlas, obs_mask, beh].
    Only neuron_atlas_idx column is replaced with model predictions.
    """
    device = next(model.parameters()).device
    T, D = x.shape
    K = model.cfg.context_length

    buf = torch.tensor(x, dtype=torch.float32, device=device).clone()
    pred_all = np.zeros((T, D), dtype=np.float32)
    pred_all[:K] = x[:K]

    for t in range(K, T):
        ctx = buf[t - K : t].unsqueeze(0)
        mu = model.predict_mean(ctx).squeeze(0)  # (D_out,) = n_atlas + n_beh
        mu_np = mu.cpu().numpy()
        pred_all[t, :len(mu_np)] = mu_np
        # Replace held-out neuron in buffer (column 0..n_atlas-1)
        buf[t, neuron_atlas_idx] = mu[neuron_atlas_idx]

    return pred_all


@torch.no_grad()
def _loo_single_neuron_atlas_windowed(
    model: AtlasTransformerGaussian,
    x: np.ndarray,
    neuron_atlas_idx: int,
    window_size: int = 50,
) -> np.ndarray:
    """LOO forward simulation with periodic re-seeding."""
    device = next(model.parameters()).device
    T, D = x.shape
    K = model.cfg.context_length
    i = neuron_atlas_idx

    x_gt = torch.tensor(x, dtype=torch.float32, device=device)
    buf = x_gt.clone()
    pred_all = np.zeros((T, D), dtype=np.float32)
    pred_all[:K] = x[:K]

    for t in range(K, T):
        if (t - K) % window_size == 0:
            start = max(K, t - K)
            for s in range(start, t):
                buf[s, i] = x_gt[s, i]

        ctx = buf[t - K : t].unsqueeze(0)
        mu = model.predict_mean(ctx).squeeze(0)
        mu_np = mu.cpu().numpy()
        pred_all[t, :len(mu_np)] = mu_np
        buf[t, i] = mu[i]

    return pred_all


def compute_loo_r2_atlas(
    model: AtlasTransformerGaussian,
    x: np.ndarray,
    obs_mask: np.ndarray,
    subset_size: int = 20,
    verbose: bool = False,
    window_size: int = 0,
) -> Dict[str, Any]:
    """LOO R² for observed neurons.

    Parameters
    ----------
    model       : trained model
    x           : (T, D) packed joint state
    obs_mask    : (N_atlas,) bool
    subset_size : number of neurons to evaluate (0=all observed)
    verbose     : print progress
    window_size : if > 0, use windowed LOO with periodic re-seeding
    """
    n_atlas = model.n_atlas
    K = model.cfg.context_length
    obs_idx = np.where(obs_mask)[0]

    if 0 < subset_size < len(obs_idx):
        # Top-variance observed neurons
        variances = np.nanvar(x[:, obs_idx], axis=0)
        top = np.argsort(variances)[::-1][:subset_size]
        subset = obs_idx[top]
    else:
        subset = obs_idx

    r2_arr = np.full(len(obs_idx), np.nan)
    obs_idx_list = list(obs_idx)
    use_windowed = window_size > 0
    tag = f"LOO-w{window_size}" if use_windowed else "LOO"

    for cnt, atlas_i in enumerate(subset):
        if use_windowed:
            pred_i = _loo_single_neuron_atlas_windowed(
                model, x, atlas_i, window_size)
        else:
            pred_i = _loo_single_neuron_atlas(model, x, atlas_i)
        r2_i = _r2(x[K:, atlas_i], pred_i[K:, atlas_i])

        obs_pos = obs_idx_list.index(atlas_i)
        r2_arr[obs_pos] = r2_i

        if verbose and (cnt == 0 or (cnt + 1) % 10 == 0 or cnt == len(subset) - 1):
            print(f"  {tag} {cnt+1}/{len(subset)}  atlas_idx={atlas_i}  R²={r2_i:.4f}")

    finite = np.isfinite(r2_arr)
    r2_mean = float(np.nanmean(r2_arr)) if finite.any() else float("nan")

    return {
        "r2": r2_arr,
        "r2_mean": r2_mean,
        "subset_atlas": list(subset),
        "window_size": window_size,
    }


# ── 3. Free-run R² ──────────────────────────────────────────────────────────


@torch.no_grad()
def compute_free_run_r2_atlas(
    model: AtlasTransformerGaussian,
    x: np.ndarray,
    obs_mask: np.ndarray,
    motor_idx_atlas: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Autoregressive free-run using packed joint state."""
    device = next(model.parameters()).device
    model.eval()
    n_atlas = model.n_atlas

    T, D = x.shape
    K = model.cfg.context_length
    obs_idx = np.where(obs_mask)[0]

    conditioned = motor_idx_atlas is not None and 0 < len(motor_idx_atlas) < len(obs_idx)
    motor_mask = np.zeros(n_atlas, dtype=bool)
    if motor_idx_atlas:
        motor_mask[np.array(motor_idx_atlas)] = True

    x_t = torch.tensor(x, dtype=torch.float32, device=device)

    pred_u = np.zeros((T, n_atlas), dtype=np.float32)
    pred_u[:K] = x[:K, :n_atlas]
    # Build buffer with full D columns so context stays valid
    buf = x_t[:K].clone()

    for t in range(K, T):
        ctx = buf[-K:].unsqueeze(0)
        mu_u, mu_b = model.predict_mean_split(ctx)
        mu_u_np = mu_u.squeeze(0).cpu().numpy()

        if conditioned:
            step_u = x[t, :n_atlas].copy()
            step_u[motor_mask] = mu_u_np[motor_mask]
        else:
            step_u = mu_u_np
            step_u[~obs_mask] = 0.0

        pred_u[t] = step_u

        # Build next buffer row from GT x row but replace u_atlas columns
        new_row = x_t[t].clone() if t < T else torch.zeros(D, device=device)
        new_row[:n_atlas] = torch.tensor(step_u, dtype=torch.float32, device=device)
        buf = torch.cat([buf, new_row.unsqueeze(0)], dim=0)

    # Evaluate on observed neurons
    eval_idx = list(np.where(motor_mask & obs_mask)[0]) if conditioned else list(obs_idx)
    r2_arr = np.full(len(obs_idx), np.nan)
    obs_idx_list = list(obs_idx)
    for atlas_i in eval_idx:
        if atlas_i in obs_idx_list:
            pos = obs_idx_list.index(atlas_i)
            r2_arr[pos] = _r2(x[K:, atlas_i], pred_u[K:, atlas_i])

    return {
        "pred_u": pred_u,
        "r2": r2_arr,
        "r2_mean": float(np.nanmean(r2_arr[np.isfinite(r2_arr)])) if np.any(np.isfinite(r2_arr)) else float("nan"),
        "mode": "motor_conditioned" if conditioned else "autonomous",
    }


# ── 4. Behaviour R² (ridge) ─────────────────────────────────────────────────


def compute_behaviour_r2_atlas(
    u_pred_atlas: np.ndarray,
    u_gt_atlas: np.ndarray,
    b: np.ndarray,
    b_mask: np.ndarray,
    motor_idx_atlas: List[int],
    n_lags: int = 8,
    n_folds: int = 5,
) -> Dict[str, Any]:
    """Behaviour R² via ridge-CV decoder on motor neurons → eigenworms."""
    n_modes = b.shape[1]
    ridge_grid = _log_ridge_grid(-3.0, 10.0, 50)

    X_gt = build_lagged_features_np(u_gt_atlas[:, motor_idx_atlas], n_lags)
    X_pred = build_lagged_features_np(u_pred_atlas[:, motor_idx_atlas], n_lags)

    r2_gt = np.full(n_modes, np.nan)
    r2_model = np.full(n_modes, np.nan)

    for j in range(n_modes):
        valid = valid_lag_mask_np(b.shape[0], n_lags, b_mask[:, j] > 0.5)
        idx_v = np.where(valid)[0]
        if len(idx_v) < 10:
            continue

        fit_gt = _ridge_cv_single_target(X_gt, b[:, j], idx_v, ridge_grid, n_folds)
        mask_gt = np.isfinite(fit_gt["held_out"]) & (b_mask[:, j] > 0.5)
        if mask_gt.sum() > 2:
            r2_gt[j] = _r2(b[mask_gt, j], fit_gt["held_out"][mask_gt])

        fit_model = _ridge_cv_single_target(X_pred, b[:, j], idx_v, ridge_grid, n_folds)
        mask_model = np.isfinite(fit_model["held_out"]) & (b_mask[:, j] > 0.5)
        if mask_model.sum() > 2:
            r2_model[j] = _r2(b[mask_model, j], fit_model["held_out"][mask_model])

    return {
        "r2_gt": r2_gt,
        "r2_model": r2_model,
        "r2_gt_mean": float(np.nanmean(r2_gt)),
        "r2_model_mean": float(np.nanmean(r2_model)),
        "n_motor": len(motor_idx_atlas),
        "n_lags": n_lags,
    }


# ── Full per-worm evaluation ────────────────────────────────────────────────


def run_per_worm_evaluation(
    model: AtlasTransformerGaussian,
    worm: Dict[str, Any],
    cfg: AtlasTransformerConfig,
    loo_subset_size: int = 20,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run all evaluation metrics for a single worm using the atlas model.

    Uses packed joint state matching baseline_transformer's evaluation.
    Output JSON format matches ``baseline_transformer.evaluate.run_full_evaluation_cv``.

    Parameters
    ----------
    model  : trained AtlasTransformerGaussian (on CPU or device)
    worm   : dict from load_atlas_worm_data
    cfg    : config
    loo_subset_size : LOO neurons (0=all)
    verbose : print progress
    """
    model.eval()
    u_atlas = worm["u_atlas"]       # (T, 302)
    obs_mask = worm["obs_mask"]     # (302,) bool
    b = worm.get("b")              # (T, n_beh) or None
    b_mask = worm.get("b_mask")    # (T, n_beh) or None
    motor_idx = worm.get("motor_idx_atlas")
    T = worm["T"]
    K = cfg.context_length
    worm_id = worm["worm_id"]
    n_atlas = model.n_atlas
    n_beh = model.n_beh

    # Build packed joint state for this worm
    include_beh = n_beh > 0 and b is not None
    x = build_joint_state_atlas(u_atlas, obs_mask, b if include_beh else None)

    if verbose:
        print(f"\n  [{worm_id}] Evaluating (N_obs={worm['N_obs']}, T={T}, "
              f"n_beh={n_beh})")

    # ---- 1. One-step R² (full recording) ----
    if verbose:
        print(f"    One-step R² (full)...")
    onestep_full = compute_onestep_r2_atlas(model, x, obs_mask)
    if verbose:
        print(f"    One-step R² = {onestep_full['r2_mean']:.4f}")

    # One-step behaviour R² (direct from model)
    beh_direct_result = None
    if "pred_b" in onestep_full and n_beh > 0 and b is not None and b_mask is not None:
        er = onestep_full["eval_range"]
        pred_b = onestep_full["pred_b"]
        gt_b = b[er[0]:er[1]]
        bm = b_mask[er[0]:er[1]]
        n_beh_modes = gt_b.shape[1]
        beh_r2 = np.full(n_beh_modes, np.nan)
        for j in range(n_beh_modes):
            valid_j = bm[:, j] > 0.5
            if valid_j.sum() > 2:
                beh_r2[j] = _r2(gt_b[valid_j, j], pred_b[valid_j, j])
        beh_direct_result = {
            "r2_direct": beh_r2,
            "r2_direct_mean": float(np.nanmean(beh_r2)),
        }
        if verbose:
            print(f"    One-step beh R² (direct) = {beh_direct_result['r2_direct_mean']:.4f}")

    # ---- 2. LOO R² ----
    if verbose:
        print(f"    LOO R² (full-series)...")
    loo = compute_loo_r2_atlas(
        model, x, obs_mask,
        subset_size=loo_subset_size, verbose=verbose,
    )
    if verbose:
        print(f"    LOO R² = {loo['r2_mean']:.4f}")

    loo_window_size = 50
    if verbose:
        print(f"    LOO R² (windowed, w={loo_window_size})...")
    loo_windowed = compute_loo_r2_atlas(
        model, x, obs_mask,
        subset_size=loo_subset_size, verbose=verbose,
        window_size=loo_window_size,
    )
    if verbose:
        print(f"    LOO-windowed R² = {loo_windowed['r2_mean']:.4f}")

    # ---- 3. Free-run R² ----
    if verbose:
        print(f"    Free-run R²...")
    free_run = compute_free_run_r2_atlas(model, x, obs_mask, motor_idx)
    if verbose:
        print(f"    Free-run R² = {free_run['r2_mean']:.4f} ({free_run['mode']})")

    # ---- 4. Behaviour R² (ridge + MLP decoders) ----
    beh_ridge_result = None
    beh_mlp_result = None

    if b is not None and b_mask is not None and motor_idx is not None:
        er = onestep_full["eval_range"]
        pred_u = onestep_full["pred_u"]
        gt_u = u_atlas[er[0]:er[1]]
        b_slice = b[er[0]:er[1]]
        bm_slice = b_mask[er[0]:er[1]]

        if verbose:
            print(f"    Behaviour R² (ridge decoder)...")
        beh_ridge_result = compute_behaviour_r2_atlas(
            u_pred_atlas=pred_u,
            u_gt_atlas=gt_u,
            b=b_slice,
            b_mask=bm_slice,
            motor_idx_atlas=motor_idx,
            n_lags=cfg.behavior_lag_steps,
        )
        if verbose:
            print(f"    Behaviour R² (ridge, model): {beh_ridge_result['r2_model_mean']:.4f}")

        if verbose:
            print(f"    Behaviour R² (MLP decoder)...")
        beh_mlp_result = compute_behaviour_r2_mlp(
            u_pred=pred_u,
            u_gt=gt_u,
            b=b_slice,
            b_mask=bm_slice,
            motor_idx=motor_idx,
            n_lags=cfg.behavior_lag_steps,
        )
        if verbose:
            print(f"    Behaviour R² (MLP, model): {beh_mlp_result['r2_model_mean']:.4f}")
    elif verbose:
        print(f"    Skipping behaviour (no data or motor neurons)")

    # ---- Build output matching baseline format ----
    return {
        "worm_id": worm_id,
        "N_obs": worm["N_obs"],
        "n_neural": n_atlas,
        "n_beh": n_beh,
        "T": T,
        "onestep": {
            "r2": onestep_full["r2"].tolist(),
            "r2_mean": onestep_full["r2_mean"],
        },
        "onestep_beh": {
            "r2": beh_direct_result["r2_direct"].tolist() if beh_direct_result else None,
            "r2_mean": beh_direct_result["r2_direct_mean"] if beh_direct_result else None,
        },
        "loo": {
            "r2": loo["r2"].tolist(),
            "r2_mean": loo["r2_mean"],
            "subset_atlas": loo["subset_atlas"],
        },
        "loo_windowed": {
            "r2": loo_windowed["r2"].tolist(),
            "r2_mean": loo_windowed["r2_mean"],
            "subset_atlas": loo_windowed["subset_atlas"],
            "window_size": loo_windowed["window_size"],
        },
        "free_run": {
            "r2": free_run["r2"].tolist(),
            "r2_mean": free_run["r2_mean"],
            "mode": free_run["mode"],
        },
        "behaviour_direct": {
            "r2": beh_direct_result["r2_direct"].tolist() if beh_direct_result else None,
            "r2_mean": beh_direct_result["r2_direct_mean"] if beh_direct_result else None,
        },
        "behaviour_ridge": {
            "r2_model": beh_ridge_result["r2_model"].tolist() if beh_ridge_result else None,
            "r2_model_mean": beh_ridge_result["r2_model_mean"] if beh_ridge_result else None,
            "r2_gt": beh_ridge_result["r2_gt"].tolist() if beh_ridge_result else None,
            "r2_gt_mean": beh_ridge_result["r2_gt_mean"] if beh_ridge_result else None,
        },
        "behaviour_mlp": {
            "r2_model": beh_mlp_result["r2_model"].tolist() if beh_mlp_result else None,
            "r2_model_mean": beh_mlp_result["r2_model_mean"] if beh_mlp_result else None,
            "r2_gt": beh_mlp_result["r2_gt"].tolist() if beh_mlp_result else None,
            "r2_gt_mean": beh_mlp_result["r2_gt_mean"] if beh_mlp_result else None,
        },
    }


def run_per_worm_evaluation_cv(
    train_result: Dict[str, Any],
    worm: Dict[str, Any],
    cfg: "AtlasTransformerConfig",
    loo_subset_size: int = 20,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run all evaluation metrics after 5-fold CV training.

    Uses stitched held-out predictions for one-step R² (out-of-sample),
    and the best-fold model for LOO / free-run / behaviour.

    Parameters
    ----------
    train_result : dict from train_atlas_model_cv
    worm         : dict from load_atlas_worm_data
    cfg          : AtlasTransformerConfig
    loo_subset_size : number of neurons for LOO (0=all)
    verbose      : print progress
    """
    from .dataset import build_joint_state_atlas

    u_atlas = worm["u_atlas"]
    obs_mask = worm["obs_mask"]
    b = worm.get("b")
    b_mask = worm.get("b_mask")
    motor_idx = worm.get("motor_idx_atlas")
    worm_id = worm["worm_id"]
    T = worm["T"]
    n_atlas = cfg.n_atlas

    pred_u_full = train_result["pred_u_full"]
    pred_b_full = train_result.get("pred_b_full")
    n_beh = train_result["n_beh"]
    best_model = train_result["best_model"]

    include_beh = n_beh > 0 and b is not None
    x = train_result["x"]
    K = cfg.context_length

    if verbose:
        print(f"\n  [{worm_id}] CV Evaluation (N_obs={worm['N_obs']}, "
              f"T={T}, n_beh={n_beh})")

    # ---- 1. One-step R² from stitched CV predictions (out-of-sample) ----
    if verbose:
        print(f"    One-step R² (CV held-out)...")

    gt_b_for_eval = b if (b is not None and n_beh > 0) else None
    bm_for_eval = b_mask if (b_mask is not None and n_beh > 0) else None

    onestep = compute_onestep_r2_from_preds(
        pred_u=pred_u_full,
        gt_u=u_atlas,
        obs_mask=obs_mask,
        pred_b=pred_b_full,
        gt_b=gt_b_for_eval,
        b_mask=bm_for_eval,
    )
    if verbose:
        print(f"    One-step R² (neural, CV) = {onestep['r2_mean']:.4f}")
        if "beh_r2_mean" in onestep:
            print(f"    One-step R² (beh, CV) = {onestep['beh_r2_mean']:.4f}")

    # Behaviour R² from stitched predictions (direct)
    beh_direct_result = None
    if pred_b_full is not None and n_beh > 0 and b is not None and b_mask is not None:
        valid_rows = np.all(np.isfinite(pred_b_full), axis=1)
        if valid_rows.sum() > 10:
            pred_b_v = pred_b_full[valid_rows]
            gt_b_v = b[valid_rows]
            bm_v = b_mask[valid_rows]
            n_modes = gt_b_v.shape[1]
            beh_r2 = np.full(n_modes, np.nan)
            for j in range(n_modes):
                mask_j = bm_v[:, j] > 0.5
                if mask_j.sum() > 2:
                    beh_r2[j] = _r2(gt_b_v[mask_j, j], pred_b_v[mask_j, j])
            beh_direct_result = {
                "r2_direct": beh_r2,
                "r2_direct_mean": float(np.nanmean(beh_r2)),
            }
            if verbose:
                print(f"    One-step beh R² (direct, CV) = "
                      f"{beh_direct_result['r2_direct_mean']:.4f}")

    # ---- 2. LOO R² (best-fold model, full recording) ----
    best_model = best_model.eval()

    if verbose:
        print(f"    LOO R² (full-series, best-fold model)...")
    loo = compute_loo_r2_atlas(
        best_model, x, obs_mask,
        subset_size=loo_subset_size, verbose=verbose,
    )
    if verbose:
        print(f"    LOO R² = {loo['r2_mean']:.4f}")

    loo_window_size = 50
    if verbose:
        print(f"    LOO R² (windowed, w={loo_window_size})...")
    loo_windowed = compute_loo_r2_atlas(
        best_model, x, obs_mask,
        subset_size=loo_subset_size, verbose=verbose,
        window_size=loo_window_size,
    )
    if verbose:
        print(f"    LOO-windowed R² = {loo_windowed['r2_mean']:.4f}")

    # ---- 3. Free-run R² (best-fold model) ----
    if verbose:
        print(f"    Free-run R²...")
    free_run = compute_free_run_r2_atlas(best_model, x, obs_mask, motor_idx)
    if verbose:
        print(f"    Free-run R² = {free_run['r2_mean']:.4f} ({free_run['mode']})")

    # ---- 4. Behaviour R² (ridge + MLP from best-fold one-step preds) ----
    beh_ridge_result = None
    beh_mlp_result = None

    if b is not None and b_mask is not None and motor_idx is not None:
        # Get best-fold one-step preds for full recording (for decoder eval)
        onestep_full_model = compute_onestep_r2_atlas(best_model, x, obs_mask)
        er = onestep_full_model["eval_range"]
        pred_u_model = onestep_full_model["pred_u"]
        gt_u_model = u_atlas[er[0]:er[1]]
        b_slice = b[er[0]:er[1]]
        bm_slice = b_mask[er[0]:er[1]]

        if verbose:
            print(f"    Behaviour R² (ridge decoder)...")
        beh_ridge_result = compute_behaviour_r2_atlas(
            u_pred_atlas=pred_u_model,
            u_gt_atlas=gt_u_model,
            b=b_slice,
            b_mask=bm_slice,
            motor_idx_atlas=motor_idx,
            n_lags=cfg.behavior_lag_steps,
        )
        if verbose:
            print(f"    Behaviour R² (ridge, model): "
                  f"{beh_ridge_result['r2_model_mean']:.4f}")

        if verbose:
            print(f"    Behaviour R² (MLP decoder)...")
        beh_mlp_result = compute_behaviour_r2_mlp(
            u_pred=pred_u_model,
            u_gt=gt_u_model,
            b=b_slice,
            b_mask=bm_slice,
            motor_idx=motor_idx,
            n_lags=cfg.behavior_lag_steps,
        )
        if verbose:
            print(f"    Behaviour R² (MLP, model): "
                  f"{beh_mlp_result['r2_model_mean']:.4f}")
    elif verbose:
        print(f"    Skipping behaviour (no data or motor neurons)")

    # ---- Build output ----
    return {
        "worm_id": worm_id,
        "N_obs": worm["N_obs"],
        "n_neural": n_atlas,
        "n_beh": n_beh,
        "n_folds": cfg.n_cv_folds,
        "T": T,
        "onestep": {
            "r2": onestep["r2"].tolist(),
            "r2_mean": onestep["r2_mean"],
            "n_valid_samples": onestep.get("n_valid_samples", T),
        },
        "onestep_beh": {
            "r2": beh_direct_result["r2_direct"].tolist() if beh_direct_result else None,
            "r2_mean": beh_direct_result["r2_direct_mean"] if beh_direct_result else None,
        },
        "loo": {
            "r2": loo["r2"].tolist(),
            "r2_mean": loo["r2_mean"],
            "subset_atlas": loo["subset_atlas"],
        },
        "loo_windowed": {
            "r2": loo_windowed["r2"].tolist(),
            "r2_mean": loo_windowed["r2_mean"],
            "subset_atlas": loo_windowed["subset_atlas"],
            "window_size": loo_windowed["window_size"],
        },
        "free_run": {
            "r2": free_run["r2"].tolist(),
            "r2_mean": free_run["r2_mean"],
            "mode": free_run["mode"],
        },
        "behaviour_direct": {
            "r2": beh_direct_result["r2_direct"].tolist() if beh_direct_result else None,
            "r2_mean": beh_direct_result["r2_direct_mean"] if beh_direct_result else None,
        },
        "behaviour_ridge": {
            "r2_model": beh_ridge_result["r2_model"].tolist() if beh_ridge_result else None,
            "r2_model_mean": beh_ridge_result["r2_model_mean"] if beh_ridge_result else None,
            "r2_gt": beh_ridge_result["r2_gt"].tolist() if beh_ridge_result else None,
            "r2_gt_mean": beh_ridge_result["r2_gt_mean"] if beh_ridge_result else None,
        },
        "behaviour_mlp": {
            "r2_model": beh_mlp_result["r2_model"].tolist() if beh_mlp_result else None,
            "r2_model_mean": beh_mlp_result["r2_model_mean"] if beh_mlp_result else None,
            "r2_gt": beh_mlp_result["r2_gt"].tolist() if beh_mlp_result else None,
            "r2_gt_mean": beh_mlp_result["r2_gt_mean"] if beh_mlp_result else None,
        },
    }


def save_per_worm_evaluation(
    results: Dict[str, Any],
    save_dir: str,
    worm_id: str,
) -> None:
    """Save per-worm evaluation results."""
    out = Path(save_dir) / worm_id
    out.mkdir(parents=True, exist_ok=True)

    def _sanitize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize(v) for v in obj]
        return obj

    clean = _sanitize(results)
    (out / "eval_results.json").write_text(json.dumps(clean, indent=2))
