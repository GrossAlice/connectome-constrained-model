from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from . import get_stage2_logger

__all__ = [
	"build_lagged_features_np",
	"build_lagged_features_torch",
	"valid_lag_mask_np",
	"MLPBehaviourDecoder",
	"fit_linear_behaviour_decoder_for_training",
	"init_behaviour_decoder",
	"compute_behaviour_loss",
	"evaluate_training_decoder",
	"behaviour_all_neurons_prediction",
	# Ridge utilities
	"_log_ridge_grid",
	"_make_contiguous_folds",
	"_fit_ridge_regression",
	"_predict_linear_model",
	"_predict_linear_model_cv",
	"_ridge_cv_single_target",
]


# --------------------------------------------------------------------------- #
#  Ridge regression utilities
# --------------------------------------------------------------------------- #

def _log_ridge_grid(log_min: float, log_max: float, n_grid: int) -> np.ndarray:
	return np.concatenate([[0.0], np.logspace(float(log_min), float(log_max), max(int(n_grid), 2))])


def _make_contiguous_folds(indices: np.ndarray, n_folds: int) -> List[np.ndarray]:
	idx = np.asarray(indices, dtype=int)
	if idx.size == 0:
		return []
	n_folds = max(1, min(int(n_folds), idx.size))
	sizes = np.full(n_folds, idx.size // n_folds, dtype=int)
	sizes[: idx.size % n_folds] += 1
	folds: List[np.ndarray] = []
	start = 0
	for size in sizes:
		folds.append(idx[start : start + size])
		start += size
	return [fold for fold in folds if fold.size > 0]


def _fit_ridge_regression(
	X: np.ndarray,
	y: np.ndarray,
	ridge_lambda: float,
) -> Optional[Tuple[float, np.ndarray]]:
	X = np.asarray(X, dtype=float)
	y = np.asarray(y, dtype=float)
	if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0] or X.shape[0] < 3:
		return None

	x_mean = X.mean(axis=0)
	x_std = X.std(axis=0)
	x_std = np.where(x_std > 1e-12, x_std, 1.0)
	Xs = (X - x_mean) / x_std
	y_mean = float(y.mean())
	yc = y - y_mean

	gram = Xs.T @ Xs
	if ridge_lambda > 0:
		gram += float(ridge_lambda) * np.eye(Xs.shape[1])
	try:
		coef_std = np.linalg.solve(gram, Xs.T @ yc)
	except np.linalg.LinAlgError:
		coef_std, *_ = np.linalg.lstsq(gram, Xs.T @ yc, rcond=None)

	coef = coef_std / x_std
	intercept = y_mean - float(x_mean @ coef)
	return intercept, coef


def _predict_linear_model(X: np.ndarray, intercept: float, coef: np.ndarray) -> np.ndarray:
	return float(intercept) + np.asarray(X, dtype=float) @ np.asarray(coef, dtype=float)


def _predict_linear_model_cv(
	X: np.ndarray,
	cv_models: List[Dict[str, Any]],
) -> np.ndarray:
	pred = np.full(X.shape[0], np.nan)
	for fold_model in cv_models:
		idx = np.asarray(fold_model["fold_idx"], dtype=int)
		if idx.size > 0:
			pred[idx] = _predict_linear_model(
				X[idx],
				float(fold_model["intercept"]),
				np.asarray(fold_model["coef"]),
			)
	return pred


def _ridge_cv_single_target(
	X_full: np.ndarray,
	y_full: np.ndarray,
	eval_idx: np.ndarray,
	ridge_grid: np.ndarray,
	n_folds: int,
) -> Dict[str, Any]:
	folds = _make_contiguous_folds(eval_idx, n_folds)
	train_folds = [eval_idx[~np.isin(eval_idx, fold_idx)] for fold_idx in folds]
	cv_mse = np.full(len(ridge_grid), np.inf)
	# Cache per-fold fits for every λ so we can reuse at best_idx
	fold_fits: list[list[Optional[Tuple[float, np.ndarray]]]] = []

	for lam_idx, ridge_lambda in enumerate(ridge_grid):
		fold_errors = []
		fits_this_lam: list[Optional[Tuple[float, np.ndarray]]] = []
		for fold_idx, train_idx in zip(folds, train_folds):
			if train_idx.size < 3 or fold_idx.size == 0:
				fits_this_lam.append(None)
				continue
			fit = _fit_ridge_regression(X_full[train_idx], y_full[train_idx], float(ridge_lambda))
			fits_this_lam.append(fit)
			if fit is None:
				continue
			mse = np.mean((y_full[fold_idx] - _predict_linear_model(X_full[fold_idx], fit[0], fit[1])) ** 2)
			if np.isfinite(mse):
				fold_errors.append(float(mse))
		fold_fits.append(fits_this_lam)
		if fold_errors:
			cv_mse[lam_idx] = float(np.mean(fold_errors))

	if not np.any(np.isfinite(cv_mse)):
		best_idx = 0  # all folds failed; fall back to first (unregularized) λ
	else:
		best_idx = int(np.nanargmin(np.where(np.isfinite(cv_mse), cv_mse, np.inf)))
	best_lambda = float(ridge_grid[best_idx])
	full_fit = _fit_ridge_regression(X_full[eval_idx], y_full[eval_idx], best_lambda)
	if full_fit is None:
		intercept = float("nan")
		coef = np.full(X_full.shape[1], np.nan)
		pred_full = np.full(X_full.shape[0], np.nan)
	else:
		intercept, coef = full_fit
		pred_full = _predict_linear_model(X_full, intercept, coef)

	# Reuse cached fold fits at best λ for held-out predictions
	held_out = np.full(X_full.shape[0], np.nan)
	best_fold_models: list[dict[str, Any]] = []
	for fold_idx, fit in zip(folds, fold_fits[best_idx]):
		if fit is not None:
			held_out[fold_idx] = _predict_linear_model(X_full[fold_idx], fit[0], fit[1])
			best_fold_models.append({
				"fold_idx": np.asarray(fold_idx, dtype=int),
				"intercept": float(fit[0]),
				"coef": np.asarray(fit[1], dtype=float),
			})

	return {
		"best_lambda": best_lambda,
		"best_idx": best_idx,
		"intercept": float(intercept),
		"coef": coef,
		"pred_full": pred_full,
		"held_out": held_out,
		"folds": folds,
		"best_fold_models": best_fold_models,
		"cv_mse": cv_mse,
		"at_zero": bool(best_lambda == 0.0),
		"at_upper_boundary": bool(best_idx == len(ridge_grid) - 1),
	}


# --------------------------------------------------------------------------- #
#  Lagged features                                                              #
# --------------------------------------------------------------------------- #

def _lagged_slices(padded, n_lags: int, length: int):
	return [padded[n_lags - lag : n_lags - lag + length] for lag in range(n_lags + 1)]


def build_lagged_features_np(u: np.ndarray, n_lags: int) -> np.ndarray:
	u = np.asarray(u)
	if u.ndim != 2:
		raise ValueError(f"Expected 2D array, got shape {u.shape!r}")
	if n_lags <= 0:
		return u
	length, n_features = u.shape
	padded = np.concatenate([np.zeros((n_lags, n_features), dtype=u.dtype), u], axis=0)
	return np.concatenate(_lagged_slices(padded, n_lags, length), axis=1)


def build_lagged_features_torch(u: torch.Tensor, n_lags: int) -> torch.Tensor:
	if u.ndim != 2:
		raise ValueError(f"Expected 2D tensor, got shape {tuple(u.shape)!r}")
	if n_lags <= 0:
		return u
	length, n_features = u.shape
	padded = torch.cat(
		[torch.zeros((n_lags, n_features), device=u.device, dtype=u.dtype), u],
		dim=0,
	)
	return torch.cat(_lagged_slices(padded, n_lags, length), dim=1)


def valid_lag_mask_np(T: int, n_lags: int, base_mask: np.ndarray | None = None) -> np.ndarray:
	if base_mask is not None:
		mask = np.asarray(base_mask, dtype=bool).copy()
	else:
		mask = np.ones(T, dtype=bool)
	if n_lags > 0:
		mask[:n_lags] = False
	return mask


def fit_linear_behaviour_decoder_for_training(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
	cfg = data.get("_cfg")
	if cfg is None or cfg.motor_neurons is None:
		return None
	b_seq = data.get("b")
	b_mask = data.get("b_mask")
	if b_seq is None:
		return None

	b_np = b_seq.cpu().numpy() if isinstance(b_seq, torch.Tensor) else b_seq
	bm_np = b_mask.cpu().numpy() if isinstance(b_mask, torch.Tensor) else b_mask
	u_np = data["u_stage1"].cpu().numpy()
	motor_idx = list(cfg.motor_neurons)
	n_lags = int(getattr(cfg, "behavior_lag_steps", 0) or 0)
	n_folds = int(getattr(cfg, "train_behavior_ridge_folds", 5) or 5)
	log_lambda_min = float(getattr(cfg, "train_behavior_ridge_log_lambda_min", -3.0))
	log_lambda_max = float(getattr(cfg, "train_behavior_ridge_log_lambda_max", 10.0))
	n_grid = int(getattr(cfg, "train_behavior_ridge_n_grid", 50) or 50)
	ridge_grid = _log_ridge_grid(log_lambda_min, log_lambda_max, n_grid)

	X_gt = build_lagged_features_np(u_np[:, motor_idx], n_lags)
	valid = np.all(bm_np > 0.5, axis=1) & np.all(np.isfinite(X_gt), axis=1)
	if valid.sum() < 10:
		return None

	idx_valid = np.where(valid)[0]
	n_modes = b_np.shape[1]
	W = np.zeros((X_gt.shape[1] + 1, n_modes), dtype=float)
	best_lambda = np.zeros(n_modes, dtype=float)
	boundary_zero = np.zeros(n_modes, dtype=bool)
	boundary_upper = np.zeros(n_modes, dtype=bool)
	cv_mse_curves = np.full((n_modes, len(ridge_grid)), np.inf)
	cv_models: list[list[dict[str, Any]]] = []
	for j in range(n_modes):
		fit_j = _ridge_cv_single_target(X_gt, b_np[:, j], idx_valid, ridge_grid, n_folds)
		best_lambda[j] = fit_j["best_lambda"]
		boundary_zero[j] = fit_j["at_zero"]
		boundary_upper[j] = fit_j["at_upper_boundary"]
		W[:-1, j] = fit_j["coef"]
		W[-1, j] = fit_j["intercept"]
		cv_mse_curves[j] = fit_j["cv_mse"]
		cv_models.append(fit_j["best_fold_models"])

	device = data["u_stage1"].device
	print(
		f"  [behaviour-decoder/train] ridge-CV linear model fitted: {W.shape[0]} features → {b_np.shape[1]} modes "
		f"({int(valid.sum())} valid frames, median λ={np.median(best_lambda):.3f})"
	)

	return {
		"type": "linear",
		"W": torch.tensor(W, dtype=torch.float32, device=device),
		"motor_idx": motor_idx,
		"n_lags": n_lags,
		"valid": torch.tensor(valid, dtype=torch.bool, device=device),
		"b_actual": b_seq if isinstance(b_seq, torch.Tensor) else torch.tensor(b_np, dtype=torch.float32, device=device),
		"ridge_lambdas": torch.tensor(best_lambda, dtype=torch.float32, device=device),
		"ridge_boundary_zero": torch.tensor(boundary_zero, dtype=torch.bool, device=device),
		"ridge_boundary_upper": torch.tensor(boundary_upper, dtype=torch.bool, device=device),
		"ridge_grid": ridge_grid,
		"cv_mse_curves": cv_mse_curves,
		"cv_models": cv_models,
	}


class MLPBehaviourDecoder(nn.Module):
	"""MLP decoder: N × (Linear → LayerNorm → ReLU → Dropout) → Linear.

	Default architecture (2×128) matches the benchmark winner from
	``unified_benchmark.py``.
	"""

	def __init__(self, in_dim: int, out_dim: int,
				 hidden: int = 128, n_layers: int = 2,
				 dropout: float = 0.1):
		super().__init__()
		layers: list[nn.Module] = []
		in_d = in_dim
		for _ in range(n_layers):
			layers.append(nn.Linear(in_d, hidden))
			layers.append(nn.LayerNorm(hidden))
			layers.append(nn.ReLU())
			layers.append(nn.Dropout(dropout) if dropout > 0 else nn.Identity())
			in_d = hidden
		layers.append(nn.Linear(in_d, out_dim))
		self.net = nn.Sequential(*layers)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)


def init_behaviour_decoder(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
	"""Initialise a learnable behaviour decoder for the training loss.

	Supports ``"linear"`` (warm-started from GT ridge-CV) and ``"mlp"``
	(Linear→LayerNorm→ReLU→Dropout→Linear).  The decoder type is read from
	``cfg.behavior_decoder_type`` (default ``"mlp"``).
	"""
	# Fit a frozen ridge decoder on GT traces first (used for both types)
	gt_fit = fit_linear_behaviour_decoder_for_training(data)
	if gt_fit is None:
		return None

	cfg = data.get("_cfg")
	decoder_type = str(getattr(cfg, "behavior_decoder_type", "mlp") or "mlp").strip().lower()
	device = gt_fit["W"].device
	n_features, n_modes = gt_fit["W"].shape
	n_valid = int(gt_fit["valid"].sum().item())

	base = {
		"motor_idx": gt_fit["motor_idx"],
		"n_lags": gt_fit["n_lags"],
		"valid": gt_fit["valid"],
		"b_actual": gt_fit["b_actual"],
		"ridge_lambdas": gt_fit["ridge_lambdas"],
		"ridge_boundary_zero": gt_fit["ridge_boundary_zero"],
		"ridge_boundary_upper": gt_fit["ridge_boundary_upper"],
		"ridge_grid": gt_fit["ridge_grid"],
		"cv_mse_curves": gt_fit["cv_mse_curves"],
		"cv_models": gt_fit["cv_models"],
	}

	if decoder_type == "mlp":
		hidden = int(getattr(cfg, "behavior_decoder_hidden", 128) or 128)
		n_layers = int(getattr(cfg, "behavior_decoder_n_layers", 2) or 2)
		dropout = float(getattr(cfg, "behavior_decoder_dropout", 0.1) or 0.0)
		model = MLPBehaviourDecoder(n_features, n_modes, hidden, n_layers, dropout).to(device)
		n_params = sum(p.numel() for p in model.parameters())
		hidden_desc = "→".join([str(hidden)] * n_layers)
		print(
			f"  [behaviour-decoder/init] MLP decoder ({n_features}→{hidden_desc}→{n_modes}, "
			f"{n_params:,} params, {n_valid} valid frames)"
		)
		base["type"] = "mlp"
		base["model"] = model
		return base

	# Linear: warm-start from ridge-CV
	W = gt_fit["W"].clone().detach().requires_grad_(True)
	print(
		f"  [behaviour-decoder/init] learnable linear decoder (warm-started from GT ridge): "
		f"{n_features} features \u2192 {n_modes} modes "
		f"({n_valid} valid frames)"
	)
	base["type"] = "linear"
	base["W"] = W
	return base


def compute_behaviour_loss(prior_mu: torch.Tensor, decoder: Dict[str, Any]) -> torch.Tensor:
	motor_idx = decoder["motor_idx"]
	n_lags = decoder["n_lags"]
	valid = decoder["valid"]
	b_actual = decoder["b_actual"]
	device = prior_mu.device

	# Ensure 1D mask so MSE averages over all (time, mode) pairs equally
	if valid.ndim == 2:
		valid = valid.all(dim=1)

	u_motor = prior_mu[:, motor_idx]
	X = build_lagged_features_torch(u_motor, n_lags)

	if decoder["type"] == "mlp":
		X_aug = torch.cat([X, torch.ones(X.shape[0], 1, device=device)], dim=1)
		b_pred = decoder["model"](X_aug)
	else:
		W = decoder["W"]
		X_aug = torch.cat([X, torch.ones(X.shape[0], 1, device=device)], dim=1)
		b_pred = X_aug @ W

	b_pred_v = b_pred[valid]
	b_true_v = b_actual[valid].to(device)
	return nn.functional.mse_loss(b_pred_v, b_true_v)

from ._utils import _r2, _cfg_val


def _extract_beh_data(data: Dict[str, Any]):
	b_seq = data.get("b")
	if b_seq is None:
		return None
	b_np = b_seq.cpu().numpy() if isinstance(b_seq, torch.Tensor) else b_seq
	bm = data["b_mask"]
	bm_np = bm.cpu().numpy() if isinstance(bm, torch.Tensor) else bm
	u_np = data["u_stage1"].cpu().numpy()
	return b_np, bm_np, u_np


def _fit_ridge_cv_decoder(
	X: np.ndarray, b_np: np.ndarray, bm_np: np.ndarray,
	ridge_grid: np.ndarray, n_folds: int,
) -> Dict[str, Any]:
	n_modes = b_np.shape[1]
	r2_full = np.full(n_modes, np.nan)
	r2_ho = np.full(n_modes, np.nan)
	b_pred_full = np.full_like(b_np, np.nan)
	b_pred_ho = np.full_like(b_np, np.nan)
	cv_models: list[list[dict[str, Any]]] = []

	best_lambdas = np.zeros(n_modes)
	cv_mse_curves = np.full((n_modes, len(ridge_grid)), np.inf)
	coefs = np.zeros((X.shape[1], n_modes))
	intercepts = np.zeros(n_modes)
	at_zero = np.zeros(n_modes, dtype=bool)
	at_upper = np.zeros(n_modes, dtype=bool)
	valid_masks = [bm_np[:, j] > 0.5 for j in range(n_modes)]
	valid_indices = [np.where(valid_masks[j])[0] for j in range(n_modes)]

	for j in range(n_modes):
		valid = valid_masks[j]
		idx_v = valid_indices[j]
		if idx_v.size < 10:
			cv_models.append([])
			continue
		fit_j = _ridge_cv_single_target(X, b_np[:, j], idx_v, ridge_grid, n_folds)
		b_pred_full[:, j] = fit_j["pred_full"]
		if valid.sum() > 2:
			r2_full[j] = _r2(b_np[valid, j], b_pred_full[valid, j])

		b_pred_ho[:, j] = fit_j["held_out"]
		mask = valid & np.isfinite(fit_j["held_out"])
		if mask.sum() > 2:
			r2_ho[j] = _r2(b_np[mask, j], fit_j["held_out"][mask])

		best_lambdas[j] = fit_j["best_lambda"]
		cv_mse_curves[j] = fit_j["cv_mse"]
		coefs[:, j] = fit_j["coef"]
		intercepts[j] = fit_j["intercept"]
		at_zero[j] = fit_j["at_zero"]
		at_upper[j] = fit_j["at_upper_boundary"]
		cv_models.append(fit_j["best_fold_models"])

	return {
		"r2_full": r2_full,
		"r2_ho": r2_ho,
		"b_pred_full": b_pred_full,
		"b_pred_ho": b_pred_ho,
		"cv_models": cv_models,
		"ridge_grid": ridge_grid,
		"best_lambdas": best_lambdas,
		"cv_mse_curves": cv_mse_curves,
		"coefs": coefs,
		"intercepts": intercepts,
		"at_zero": at_zero,
		"at_upper": at_upper,
	}


def _eval_decoder_common(
	coef: np.ndarray, intercept: np.ndarray,
	motor_idx: list, n_lags: int,
	data: Dict[str, Any], onestep: Dict[str, Any],
	cv_models=None,
	decoder_type: str = "linear_frozen",
) -> Optional[Dict[str, Any]]:
	beh = _extract_beh_data(data)
	if beh is None:
		return None
	b_np, bm_np, u_np = beh
	mu_np = onestep["prior_mu"]
	n_modes = b_np.shape[1]
	cfg = data.get("_cfg")
	valid_masks = [valid_lag_mask_np(b_np.shape[0], n_lags, bm_np[:, j] > 0.5) for j in range(n_modes)]

	X_gt = build_lagged_features_np(u_np[:, motor_idx], n_lags)
	X_pred = build_lagged_features_np(mu_np[:, motor_idx], n_lags)

	r2_gt = np.full(n_modes, np.nan)
	b_pred_gt = np.zeros_like(b_np)
	for j in range(n_modes):
		valid = valid_masks[j]
		b_pred_gt[:, j] = _predict_linear_model(X_gt, intercept[j], coef[:, j])
		if valid.sum() > 2:
			r2_gt[j] = _r2(b_np[valid, j], b_pred_gt[valid, j])

	r2_gt_ho = np.full(n_modes, np.nan)
	b_gt_ho = np.full_like(b_np, np.nan)
	if cv_models is not None:
		for j in range(min(n_modes, len(cv_models))):
			valid = valid_masks[j]
			b_gt_ho[:, j] = _predict_linear_model_cv(X_gt, cv_models[j])
			mask = valid & np.isfinite(b_gt_ho[:, j])
			if mask.sum() > 2:
				r2_gt_ho[j] = _r2(b_np[mask, j], b_gt_ho[mask, j])

	r2_transfer = np.full(n_modes, np.nan)
	b_pred_transfer = np.zeros_like(b_np)
	for j in range(n_modes):
		valid = valid_masks[j]
		b_pred_transfer[:, j] = _predict_linear_model(X_pred, intercept[j], coef[:, j])
		if valid.sum() > 2:
			r2_transfer[j] = _r2(b_np[valid, j], b_pred_transfer[valid, j])

	n_folds = int(_cfg_val(cfg, "train_behavior_ridge_folds", 5, int))
	log_lam_min = _cfg_val(cfg, "train_behavior_ridge_log_lambda_min", -3.0)
	log_lam_max = _cfg_val(cfg, "train_behavior_ridge_log_lambda_max", 10.0)
	n_grid = int(_cfg_val(cfg, "train_behavior_ridge_n_grid", 50, int))
	ridge_grid = _log_ridge_grid(log_lam_min, log_lam_max, n_grid)

	bm_nopad = valid_lag_mask_np(bm_np.shape[0], n_lags, bm_np).astype(float)
	model_fit = _fit_ridge_cv_decoder(X_pred, b_np, bm_nopad, ridge_grid, n_folds)

	ar1_np = onestep.get("ar1_mu")
	r2_ar1 = np.full(n_modes, np.nan)
	if ar1_np is not None:
		X_ar1 = build_lagged_features_np(ar1_np[:, motor_idx], n_lags)
		ar1_fit = _fit_ridge_cv_decoder(X_ar1, b_np, bm_nopad, ridge_grid, n_folds)
		r2_ar1 = ar1_fit["r2_ho"]

	result: Dict[str, Any] = {
		"b_actual": b_np,
		"b_pred_gt": b_pred_gt,
		"b_pred_transfer": b_pred_transfer,
		"r2_transfer": r2_transfer,
		"b_pred_model": model_fit["b_pred_full"],
		"r2_model": model_fit["r2_ho"],
		"r2_model_insample": model_fit["r2_full"],
		"r2_gt": r2_gt_ho,
		"r2_gt_insample": r2_gt,
		"r2_ar1": r2_ar1,
		"b_mask": bm_np,
		"decoder_type": decoder_type,
		"b_pred_gt_heldout": b_gt_ho,
		"b_pred_model_heldout": model_fit["b_pred_ho"],
		"r2_model_heldout": model_fit["r2_ho"],
		"r2_gt_heldout": r2_gt_ho,
		"ridge_cv_model": {
			"ridge_grid": model_fit["ridge_grid"],
			"best_lambdas": model_fit["best_lambdas"],
			"cv_mse_curves": model_fit["cv_mse_curves"],
			"coefs": model_fit["coefs"],
			"intercepts": model_fit["intercepts"],
			"at_zero": model_fit["at_zero"],
			"at_upper": model_fit["at_upper"],
		},
	}

	return result


@torch.no_grad()
def _evaluate_mlp_decoder(
	decoder: Dict[str, Any], data: Dict[str, Any], onestep: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
	"""Evaluate an MLP behaviour decoder on GT and model traces."""
	beh = _extract_beh_data(data)
	if beh is None:
		return None
	b_np, bm_np, u_np = beh
	mu_np = onestep["prior_mu"]
	n_modes = b_np.shape[1]
	cfg = data.get("_cfg")
	motor_idx = decoder["motor_idx"]
	n_lags = decoder["n_lags"]
	model = decoder["model"]
	model.eval()
	device = next(model.parameters()).device

	valid_masks = [valid_lag_mask_np(b_np.shape[0], n_lags, bm_np[:, j] > 0.5) for j in range(n_modes)]

	# MLP predictions on GT and model traces
	X_gt = build_lagged_features_np(u_np[:, motor_idx], n_lags)
	X_pred = build_lagged_features_np(mu_np[:, motor_idx], n_lags)

	def _predict_mlp(X: np.ndarray) -> np.ndarray:
		X_aug = np.concatenate([X, np.ones((X.shape[0], 1), dtype=X.dtype)], axis=1)
		t = torch.tensor(X_aug, dtype=torch.float32, device=device)
		return model(t).cpu().numpy()

	b_pred_gt = _predict_mlp(X_gt)
	b_pred_transfer = _predict_mlp(X_pred)

	r2_gt = np.full(n_modes, np.nan)
	r2_transfer = np.full(n_modes, np.nan)
	for j in range(n_modes):
		valid = valid_masks[j]
		if valid.sum() > 2:
			r2_gt[j] = _r2(b_np[valid, j], b_pred_gt[valid, j])
			r2_transfer[j] = _r2(b_np[valid, j], b_pred_transfer[valid, j])

	# Refit ridge-CV on model outputs (decoder-independent upper bound)
	n_folds = int(_cfg_val(cfg, "train_behavior_ridge_folds", 5, int))
	log_lam_min = _cfg_val(cfg, "train_behavior_ridge_log_lambda_min", -3.0)
	log_lam_max = _cfg_val(cfg, "train_behavior_ridge_log_lambda_max", 10.0)
	n_grid = int(_cfg_val(cfg, "train_behavior_ridge_n_grid", 50, int))
	ridge_grid = _log_ridge_grid(log_lam_min, log_lam_max, n_grid)
	bm_nopad = valid_lag_mask_np(bm_np.shape[0], n_lags, bm_np).astype(float)
	model_fit = _fit_ridge_cv_decoder(X_pred, b_np, bm_nopad, ridge_grid, n_folds)
	gt_fit = _fit_ridge_cv_decoder(X_gt, b_np, bm_nopad, ridge_grid, n_folds)

	ar1_np = onestep.get("ar1_mu")
	r2_ar1 = np.full(n_modes, np.nan)
	if ar1_np is not None:
		X_ar1 = build_lagged_features_np(ar1_np[:, motor_idx], n_lags)
		ar1_fit = _fit_ridge_cv_decoder(X_ar1, b_np, bm_nopad, ridge_grid, n_folds)
		r2_ar1 = ar1_fit["r2_ho"]

	result: Dict[str, Any] = {
		"b_actual": b_np,
		"b_pred_gt": b_pred_gt,
		"b_pred_transfer": b_pred_transfer,
		"r2_transfer": r2_transfer,
		"b_pred_model": model_fit["b_pred_full"],
		"r2_model": model_fit["r2_ho"],
		"r2_model_insample": model_fit["r2_full"],
		"r2_gt": gt_fit["r2_ho"],
		"r2_gt_insample": r2_gt,
		"r2_ar1": r2_ar1,
		"b_mask": bm_np,
		"decoder_type": "mlp",
		"b_pred_gt_heldout": gt_fit["b_pred_ho"],
		"b_pred_model_heldout": model_fit["b_pred_ho"],
		"r2_model_heldout": model_fit["r2_ho"],
		"r2_gt_heldout": gt_fit["r2_ho"],
		"ridge_cv_model": {
			"ridge_grid": model_fit["ridge_grid"],
			"best_lambdas": model_fit["best_lambdas"],
			"cv_mse_curves": model_fit["cv_mse_curves"],
			"coefs": model_fit["coefs"],
			"intercepts": model_fit["intercepts"],
			"at_zero": model_fit["at_zero"],
			"at_upper": model_fit["at_upper"],
		},
	}

	n_show = min(n_modes, 6)
	logger = get_stage2_logger()
	logger.info(
		"behaviour_eval_training",
		r2_gt_ho=gt_fit["r2_ho"][:n_show].tolist(),
		r2_ar1_ho=r2_ar1[:n_show].tolist(),
		r2_model_ho=model_fit["r2_ho"][:n_show].tolist(),
		decoder_type="mlp",
	)
	return result


def evaluate_training_decoder(
	decoder: Dict[str, Any], data: Dict[str, Any], onestep: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
	if decoder is None or data.get("b") is None:
		return None

	if decoder["type"] == "mlp":
		return _evaluate_mlp_decoder(decoder, data, onestep)

	W_np = decoder["W"].detach().cpu().numpy()
	dtype = "linear_learned" if decoder["W"].requires_grad else "linear_frozen"
	result = _eval_decoder_common(
		coef=W_np[:-1],
		intercept=W_np[-1],
		motor_idx=decoder["motor_idx"],
		n_lags=decoder["n_lags"],
		data=data,
		onestep=onestep,
		cv_models=decoder.get("cv_models"),
		decoder_type=dtype,
	)
	if not result:
		return None

	if "ridge_lambdas" in decoder:
		result["ridge_cv_gt"] = {
			"coefs": W_np[:-1],
			"intercepts": W_np[-1],
			"best_lambdas": decoder["ridge_lambdas"].cpu().numpy(),
			"at_zero": decoder["ridge_boundary_zero"].cpu().numpy(),
			"at_upper": decoder["ridge_boundary_upper"].cpu().numpy(),
			"ridge_grid": decoder.get("ridge_grid"),
			"cv_mse_curves": decoder.get("cv_mse_curves"),
		}
	else:
		result["ridge_cv_gt"] = {
			"coefs": W_np[:-1],
			"intercepts": W_np[-1],
		}

	n_show = min(result["r2_gt"].size, 6)
	logger = get_stage2_logger()
	logger.info(
		"behaviour_eval_training",
		r2_gt_ho=result["r2_gt"][:n_show].tolist(),
		r2_ar1_ho=result["r2_ar1"][:n_show].tolist(),
		r2_transfer=result["r2_transfer"][:n_show].tolist(),
		r2_model_ho=result["r2_model"][:n_show].tolist(),
	)
	return result


def behaviour_all_neurons_prediction(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
	beh = _extract_beh_data(data)
	if beh is None:
		return None
	b_np, bm_np, u_np = beh
	cfg = data.get("_cfg")
	n_modes = b_np.shape[1]
	n_lags = _cfg_val(cfg, "behavior_lag_steps", 0, int)

	X_all = build_lagged_features_np(u_np, n_lags)
	log_lam_min = _cfg_val(cfg, "train_behavior_ridge_log_lambda_min", -3.0)
	log_lam_max = _cfg_val(cfg, "train_behavior_ridge_log_lambda_max", 10.0)
	n_grid = int(_cfg_val(cfg, "train_behavior_ridge_n_grid", 50, int))
	n_folds = int(_cfg_val(cfg, "train_behavior_ridge_folds", 5, int))
	ridge_grid = _log_ridge_grid(log_lam_min, log_lam_max, n_grid)

	r2_all = np.full(n_modes, np.nan)
	valid_indices = [
		np.where(valid_lag_mask_np(b_np.shape[0], n_lags, bm_np[:, j] > 0.5))[0]
		for j in range(n_modes)
	]
	for j in range(n_modes):
		idx_v = valid_indices[j]
		if len(idx_v) < 10:
			continue
		fit_j = _ridge_cv_single_target(X_all, b_np[:, j], idx_v, ridge_grid, n_folds)
		mask = np.isfinite(fit_j["held_out"]) & (bm_np[:, j] > 0.5)
		if mask.sum() > 2:
			r2_all[j] = _r2(b_np[mask, j], fit_j["held_out"][mask])

	logger = get_stage2_logger()
	logger.info("behaviour_all_neurons", r2=r2_all[:min(n_modes, 6)].tolist())
	return {"r2_all_neurons": r2_all}
