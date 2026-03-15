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
	"valid_lag_mask_torch",
	"normalize_behavior_decoder_mode",
	"EndToEndDecoder",
	"fit_linear_behaviour_decoder",
	"fit_linear_behaviour_decoder_for_training",
	"compute_behaviour_loss",
	"evaluate_training_decoder",
	"evaluate_e2e_decoder",
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
#  Ridge regression utilities (merged from ridge_utils.py)                      #
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
	cv_mse = np.full(len(ridge_grid), np.inf)

	for lam_idx, ridge_lambda in enumerate(ridge_grid):
		fold_errors = []
		for fold_idx in folds:
			train_idx = eval_idx[~np.isin(eval_idx, fold_idx)]
			if train_idx.size < 3 or fold_idx.size == 0:
				continue
			fit = _fit_ridge_regression(X_full[train_idx], y_full[train_idx], float(ridge_lambda))
			if fit is None:
				continue
			mse = np.mean((y_full[fold_idx] - _predict_linear_model(X_full[fold_idx], fit[0], fit[1])) ** 2)
			if np.isfinite(mse):
				fold_errors.append(float(mse))
		if fold_errors:
			cv_mse[lam_idx] = float(np.mean(fold_errors))

	best_idx = int(np.nanargmin(cv_mse))
	best_lambda = float(ridge_grid[best_idx])
	full_fit = _fit_ridge_regression(X_full[eval_idx], y_full[eval_idx], best_lambda)
	if full_fit is None:
		intercept = float("nan")
		coef = np.full(X_full.shape[1], np.nan)
		pred_full = np.full(X_full.shape[0], np.nan)
	else:
		intercept, coef = full_fit
		pred_full = _predict_linear_model(X_full, intercept, coef)

	held_out = np.full(X_full.shape[0], np.nan)
	for fold_idx in folds:
		train_idx = eval_idx[~np.isin(eval_idx, fold_idx)]
		if train_idx.size < 3 or fold_idx.size == 0:
			continue
		fit = _fit_ridge_regression(X_full[train_idx], y_full[train_idx], best_lambda)
		if fit is not None:
			held_out[fold_idx] = _predict_linear_model(X_full[fold_idx], fit[0], fit[1])

	return {
		"best_lambda": best_lambda,
		"best_idx": best_idx,
		"intercept": float(intercept),
		"coef": coef,
		"pred_full": pred_full,
		"held_out": held_out,
		"folds": folds,
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


def valid_lag_mask_torch(
	T: int, n_lags: int, base_mask: torch.Tensor | None = None,
	device: torch.device | None = None,
) -> torch.Tensor:
	if base_mask is not None:
		mask = base_mask.clone().bool()
		if device is not None:
			mask = mask.to(device)
	else:
		mask = torch.ones(T, dtype=torch.bool, device=device)
	if n_lags > 0:
		mask[:n_lags] = False
	return mask


_VALID_DECODER_MODES = {"none", "frozen", "e2e"}


def normalize_behavior_decoder_mode(value: object) -> str:
	mode = str(value or "frozen").strip().lower()
	if mode == "off":
		mode = "none"
	if mode not in _VALID_DECODER_MODES:
		raise ValueError(
			f"Invalid behavior decoder mode {value!r}; expected one of {sorted(_VALID_DECODER_MODES)}"
		)
	return mode


def fit_linear_behaviour_decoder(
	data: Dict[str, Any],
	logger=None,
) -> Optional[Dict[str, Any]]:
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
	ridge_disable = bool(getattr(cfg, "train_behavior_ridge_disable", False))
	if ridge_disable:
		ridge_grid = np.array([0.0])
	else:
		log_lambda_min = float(getattr(cfg, "train_behavior_ridge_log_lambda_min", -3.0))
		log_lambda_max = float(getattr(cfg, "train_behavior_ridge_log_lambda_max", 10.0))
		n_grid = int(getattr(cfg, "train_behavior_ridge_n_grid", 50) or 50)
		ridge_grid = _log_ridge_grid(log_lambda_min, log_lambda_max, n_grid)

	X_gt = build_lagged_features_np(u_np[:, motor_idx], n_lags)
	feat_valid = valid_lag_mask_np(X_gt.shape[0], n_lags, np.all(np.isfinite(X_gt), axis=1))
	mode_valid = (bm_np > 0.5) & np.isfinite(b_np)
	valid_2d = feat_valid[:, None] & mode_valid
	if valid_2d.any(axis=1).sum() < 10:
		return None

	n_modes = b_np.shape[1]
	W = np.zeros((X_gt.shape[1] + 1, n_modes), dtype=float)
	best_lambda = np.zeros(n_modes, dtype=float)
	boundary_zero = np.zeros(n_modes, dtype=bool)
	boundary_upper = np.zeros(n_modes, dtype=bool)
	cv_mse_curves = np.full((n_modes, len(ridge_grid)), np.inf)
	cv_models: list[list[dict[str, Any]]] = []
	for j in range(n_modes):
		idx_valid_j = np.where(valid_2d[:, j])[0]
		if idx_valid_j.size < 10:
			cv_models.append([])
			continue
		fit_j = _ridge_cv_single_target(X_gt, b_np[:, j], idx_valid_j, ridge_grid, n_folds)
		best_lambda[j] = fit_j["best_lambda"]
		boundary_zero[j] = fit_j["at_zero"]
		boundary_upper[j] = fit_j["at_upper_boundary"]
		cv_mse_curves[j] = fit_j["cv_mse"]
		W[:-1, j] = fit_j["coef"]
		W[-1, j] = fit_j["intercept"]

		fold_models_j: list[dict[str, Any]] = []
		for fold_idx in fit_j["folds"]:
			train_idx = idx_valid_j[~np.isin(idx_valid_j, fold_idx)]
			if train_idx.size < 3 or fold_idx.size == 0:
				continue
			fold_fit = _fit_ridge_regression(
				X_gt[train_idx],
				b_np[train_idx, j],
				float(fit_j["best_lambda"]),
			)
			if fold_fit is None:
				continue
			fold_models_j.append({
				"fold_idx": np.asarray(fold_idx, dtype=int),
				"intercept": float(fold_fit[0]),
				"coef": np.asarray(fold_fit[1], dtype=float),
			})
		cv_models.append(fold_models_j)

	device = data["u_stage1"].device
	valid_per_mode = [int(valid_2d[:, j].sum()) for j in range(n_modes)]
	if logger is not None:
		logger.info(
			"behaviour_decoder_fit",
			features=int(W.shape[0]),
			modes=int(n_modes),
			median_lambda=float(np.median(best_lambda)) if best_lambda.size else 0.0,
			valid_per_mode=valid_per_mode,
		)

	return {
		"type": "linear",
		"W": torch.tensor(W, dtype=torch.float32, device=device),
		"motor_idx": motor_idx,
		"n_lags": n_lags,
		"valid": torch.tensor(valid_2d, dtype=torch.bool, device=device),
		"b_actual": b_seq if isinstance(b_seq, torch.Tensor) else torch.tensor(b_np, dtype=torch.float32, device=device),
		"ridge_lambdas": torch.tensor(best_lambda, dtype=torch.float32, device=device),
		"ridge_boundary_zero": torch.tensor(boundary_zero, dtype=torch.bool, device=device),
		"ridge_boundary_upper": torch.tensor(boundary_upper, dtype=torch.bool, device=device),
		"ridge_grid": ridge_grid,
		"cv_mse_curves": cv_mse_curves,
		"cv_models": cv_models,
	}


class EndToEndDecoder(nn.Module):
	def __init__(
		self,
		n_features: int,
		n_modes: int,
		motor_idx: list[int],
		n_lags: int,
		valid: torch.Tensor,
		b_actual: torch.Tensor,
	):
		super().__init__()
		self.linear = nn.Linear(n_features, n_modes)
		self.motor_idx = motor_idx
		self.n_lags = n_lags
		valid = valid_lag_mask_torch(valid.shape[0], n_lags, valid)
		self.register_buffer("valid", valid)
		self.register_buffer("b_actual", b_actual)

	@classmethod
	def from_ridge_decoder(cls, ridge_decoder: Dict[str, Any]) -> "EndToEndDecoder":
		W = ridge_decoder["W"]
		n_features = W.shape[0] - 1
		n_modes = W.shape[1]
		dec = cls(
			n_features=n_features,
			n_modes=n_modes,
			motor_idx=ridge_decoder["motor_idx"],
			n_lags=ridge_decoder["n_lags"],
			valid=ridge_decoder["valid"],
			b_actual=ridge_decoder["b_actual"],
		)
		with torch.no_grad():
			dec.linear.weight.copy_(W[:-1].T)
			dec.linear.bias.copy_(W[-1])
		return dec

	def forward(self, prior_mu: torch.Tensor) -> torch.Tensor:
		u_motor = prior_mu[:, self.motor_idx]
		X = build_lagged_features_torch(u_motor, self.n_lags)
		return self.linear(X)

	def loss(self, prior_mu: torch.Tensor, l2: float = 0.0) -> torch.Tensor:
		b_pred = self.forward(prior_mu)
		sq_err = (b_pred - self.b_actual).pow(2)
		mse = sq_err[self.valid].mean() if self.valid.any() else sq_err.mean()
		if l2 > 0.0:
			mse = mse + l2 * self.linear.weight.pow(2).mean()
		return mse


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
	log_lambda_max = float(getattr(cfg, "train_behavior_ridge_log_lambda_max", 3.0))
	n_grid = int(getattr(cfg, "train_behavior_ridge_n_grid", 60) or 60)
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
	for j in range(n_modes):
		fit_j = _ridge_cv_single_target(X_gt, b_np[:, j], idx_valid, ridge_grid, n_folds)
		best_lambda[j] = fit_j["best_lambda"]
		boundary_zero[j] = fit_j["at_zero"]
		boundary_upper[j] = fit_j["at_upper_boundary"]
		W[:-1, j] = fit_j["coef"]
		W[-1, j] = fit_j["intercept"]

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
	}


def compute_behaviour_loss(prior_mu: torch.Tensor, decoder: Dict[str, Any]) -> torch.Tensor:
	motor_idx = decoder["motor_idx"]
	n_lags = decoder["n_lags"]
	valid = decoder["valid"]
	b_actual = decoder["b_actual"]
	device = prior_mu.device

	u_motor = prior_mu[:, motor_idx]
	X = build_lagged_features_torch(u_motor, n_lags)

	W = decoder["W"]
	X_aug = torch.cat([X, torch.ones(X.shape[0], 1, device=device)], dim=1)
	b_pred = X_aug @ W

	b_pred_v = b_pred[valid]
	b_true_v = b_actual[valid].to(device)
	return nn.functional.mse_loss(b_pred_v, b_true_v)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	mask = np.isfinite(y_true) & np.isfinite(y_pred)
	if mask.sum() < 3:
		return float("nan")
	yt, yp = y_true[mask].astype(np.float64), y_pred[mask].astype(np.float64)
	ss_tot = np.sum((yt - yt.mean()) ** 2)
	if ss_tot < 1e-12:
		return float("nan")
	return float(1.0 - np.sum((yt - yp) ** 2) / ss_tot)


def _cfg_val(cfg, attr: str, default, cast=float):
	return cast(getattr(cfg, attr, default) or default) if cfg is not None else cast(default)


def _extract_beh_data(data: Dict[str, Any]):
	b_seq = data.get("b")
	if b_seq is None:
		return None
	b_np = b_seq.cpu().numpy() if isinstance(b_seq, torch.Tensor) else b_seq
	bm = data["b_mask"]
	bm_np = bm.cpu().numpy() if isinstance(bm, torch.Tensor) else bm
	u_np = data["u_stage1"].cpu().numpy()
	return b_np, bm_np, u_np


def _calibrate_predictions(
	b_np: np.ndarray, bm_np: np.ndarray, *pred_arrays: np.ndarray,
) -> None:
	for j in range(b_np.shape[1]):
		valid_j = bm_np[:, j] > 0.5
		for arr in pred_arrays:
			mask = valid_j & np.isfinite(arr[:, j])
			if mask.sum() < 3:
				continue
			mu_y, mu_p = np.mean(b_np[mask, j]), np.mean(arr[mask, j])
			sd_y, sd_p = np.std(b_np[mask, j]), np.std(arr[mask, j])
			if sd_p < 1e-12:
				continue
			arr[:, j] = mu_y + (arr[:, j] - mu_p) * (sd_y / sd_p)


def _fit_ridge_cv_decoder(
	X: np.ndarray, b_np: np.ndarray, bm_np: np.ndarray,
	ridge_grid: np.ndarray, n_folds: int,
) -> Dict[str, Any]:
	n_modes = b_np.shape[1]
	r2_full = np.zeros(n_modes)
	r2_ho = np.full(n_modes, np.nan)
	b_pred_full = np.zeros_like(b_np)
	b_pred_ho = np.full_like(b_np, np.nan)
	cv_models: list[list[dict[str, Any]]] = []

	best_lambdas = np.zeros(n_modes)
	cv_mse_curves = np.full((n_modes, len(ridge_grid)), np.inf)
	coefs = np.zeros((X.shape[1], n_modes))
	intercepts = np.zeros(n_modes)
	at_zero = np.zeros(n_modes, dtype=bool)
	at_upper = np.zeros(n_modes, dtype=bool)

	for j in range(n_modes):
		valid = bm_np[:, j] > 0.5
		idx_v = np.where(valid)[0]
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

		fold_models_j: list[dict[str, Any]] = []
		for fold_idx in fit_j["folds"]:
			train_idx = idx_v[~np.isin(idx_v, fold_idx)]
			if train_idx.size < 3 or fold_idx.size == 0:
				continue
			fold_fit = _fit_ridge_regression(X[train_idx], b_np[train_idx, j], fit_j["best_lambda"])
			if fold_fit is None:
				continue
			fold_models_j.append({
				"fold_idx": np.asarray(fold_idx, dtype=int),
				"intercept": float(fold_fit[0]),
				"coef": np.asarray(fold_fit[1], dtype=float),
			})
		cv_models.append(fold_models_j)

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
	calibrate: bool, cv_models=None,
	decoder_type: str = "linear_frozen",
) -> Optional[Dict[str, Any]]:
	beh = _extract_beh_data(data)
	if beh is None:
		return None
	b_np, bm_np, u_np = beh
	mu_np = onestep["prior_mu"]
	n_modes = b_np.shape[1]
	cfg = data.get("_cfg")

	X_gt = build_lagged_features_np(u_np[:, motor_idx], n_lags)
	X_pred = build_lagged_features_np(mu_np[:, motor_idx], n_lags)

	r2_gt = np.zeros(n_modes)
	b_pred_gt = np.zeros_like(b_np)
	for j in range(n_modes):
		valid = valid_lag_mask_np(b_np.shape[0], n_lags, bm_np[:, j] > 0.5)
		b_pred_gt[:, j] = _predict_linear_model(X_gt, intercept[j], coef[:, j])
		if valid.sum() > 2:
			r2_gt[j] = _r2(b_np[valid, j], b_pred_gt[valid, j])

	r2_gt_ho = np.full(n_modes, np.nan)
	b_gt_ho = np.full_like(b_np, np.nan)
	if cv_models is not None:
		for j in range(min(n_modes, len(cv_models))):
			valid = valid_lag_mask_np(b_np.shape[0], n_lags, bm_np[:, j] > 0.5)
			b_gt_ho[:, j] = _predict_linear_model_cv(X_gt, cv_models[j])
			mask = valid & np.isfinite(b_gt_ho[:, j])
			if mask.sum() > 2:
				r2_gt_ho[j] = _r2(b_np[mask, j], b_gt_ho[mask, j])

	n_folds = int(_cfg_val(cfg, "train_behavior_ridge_folds", 5, int))
	log_lam_min = _cfg_val(cfg, "train_behavior_ridge_log_lambda_min", -3.0)
	log_lam_max = _cfg_val(cfg, "train_behavior_ridge_log_lambda_max", 10.0)
	n_grid = int(_cfg_val(cfg, "train_behavior_ridge_n_grid", 80, int))
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
		"b_pred_model": model_fit["b_pred_full"],
		"r2_model": model_fit["r2_full"],
		"r2_gt": r2_gt,
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

	if calibrate:
		_calibrate_predictions(b_np, bm_np, b_pred_gt, result["b_pred_model"])

	return result


def evaluate_training_decoder(
	decoder: Dict[str, Any], data: Dict[str, Any], onestep: Dict[str, Any],
	*, calibrate: bool = True,
) -> Optional[Dict[str, Any]]:
	if decoder is None or data.get("b") is None:
		return None
	W_np = decoder["W"].cpu().numpy()
	result = _eval_decoder_common(
		coef=W_np[:-1],
		intercept=W_np[-1],
		motor_idx=decoder["motor_idx"],
		n_lags=decoder["n_lags"],
		data=data,
		onestep=onestep,
		calibrate=calibrate,
		cv_models=decoder.get("cv_models"),
		decoder_type="linear_frozen",
	)
	if not result:
		return None

	W_np_full = decoder["W"].cpu().numpy()
	result["ridge_cv_gt"] = {
		"coefs": W_np_full[:-1],
		"intercepts": W_np_full[-1],
		"best_lambdas": decoder["ridge_lambdas"].cpu().numpy(),
		"at_zero": decoder["ridge_boundary_zero"].cpu().numpy(),
		"at_upper": decoder["ridge_boundary_upper"].cpu().numpy(),
		"ridge_grid": decoder.get("ridge_grid"),
		"cv_mse_curves": decoder.get("cv_mse_curves"),
	}

	n_show = min(result["r2_gt"].size, 6)
	logger = get_stage2_logger()
	logger.info(
		"behaviour_eval_training",
		r2_gt_full=result["r2_gt"][:n_show].tolist(),
		r2_ar1_ho=result["r2_ar1"][:n_show].tolist(),
		r2_model_full=result["r2_model"][:n_show].tolist(),
	)
	if np.any(np.isfinite(result.get("r2_gt_heldout", []))):
		logger.info("behaviour_eval_training_ho", r2_gt_ho=result["r2_gt_heldout"][:n_show].tolist())
	if np.any(np.isfinite(result.get("r2_model_heldout", []))):
		logger.info("behaviour_eval_training_ho", r2_model_ho=result["r2_model_heldout"][:n_show].tolist())
	return result


def evaluate_e2e_decoder(
	e2e_decoder, data: Dict[str, Any], onestep: Dict[str, Any],
	*, calibrate: bool = True,
) -> Optional[Dict[str, Any]]:
	if e2e_decoder is None or data.get("b") is None:
		return None
	beh = _extract_beh_data(data)
	if beh is None:
		return None
	b_np, bm_np, u_np = beh
	mu_np = onestep["prior_mu"]
	n_modes = b_np.shape[1]

	with torch.no_grad():
		coef = e2e_decoder.linear.weight.cpu().numpy().T
		intercept = e2e_decoder.linear.bias.cpu().numpy()
	motor_idx = e2e_decoder.motor_idx
	n_lags = e2e_decoder.n_lags

	X_gt = build_lagged_features_np(u_np[:, motor_idx], n_lags)
	X_pred = build_lagged_features_np(mu_np[:, motor_idx], n_lags)

	r2_gt = np.zeros(n_modes)
	r2_model = np.zeros(n_modes)
	b_pred_gt = np.zeros_like(b_np)
	b_pred_model = np.zeros_like(b_np)

	ar1_np = onestep.get("ar1_mu")
	r2_ar1 = np.full(n_modes, np.nan)
	b_pred_ar1 = np.zeros_like(b_np)
	if ar1_np is not None:
		X_ar1 = build_lagged_features_np(ar1_np[:, motor_idx], n_lags)

	for j in range(n_modes):
		valid = valid_lag_mask_np(b_np.shape[0], n_lags, bm_np[:, j] > 0.5)
		b_pred_gt[:, j] = _predict_linear_model(X_gt, intercept[j], coef[:, j])
		b_pred_model[:, j] = _predict_linear_model(X_pred, intercept[j], coef[:, j])
		if ar1_np is not None:
			b_pred_ar1[:, j] = _predict_linear_model(X_ar1, intercept[j], coef[:, j])
		if valid.sum() > 2:
			r2_gt[j] = _r2(b_np[valid, j], b_pred_gt[valid, j])
			r2_model[j] = _r2(b_np[valid, j], b_pred_model[valid, j])
			if ar1_np is not None:
				r2_ar1[j] = _r2(b_np[valid, j], b_pred_ar1[valid, j])

	if calibrate:
		_calibrate_predictions(b_np, bm_np, b_pred_gt, b_pred_model)

	result: Dict[str, Any] = {
		"b_actual": b_np,
		"b_pred_gt": b_pred_gt,
		"b_pred_model": b_pred_model,
		"r2_model": r2_model,
		"r2_gt": r2_gt,
		"r2_ar1": r2_ar1,
		"b_mask": bm_np,
		"decoder_type": "linear_e2e",
	}
	n_show = min(n_modes, 6)
	logger = get_stage2_logger()
	logger.info(
		"behaviour_eval_e2e",
		r2_gt=r2_gt[:n_show].tolist(),
		r2_ar1=r2_ar1[:n_show].tolist(),
		r2_model=r2_model[:n_show].tolist(),
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
	ridge_grid = _log_ridge_grid(-3.0, 10.0, 80)

	r2_all = np.zeros(n_modes)
	for j in range(n_modes):
		idx_v = np.where(valid_lag_mask_np(b_np.shape[0], n_lags, bm_np[:, j] > 0.5))[0]
		if len(idx_v) < 10:
			continue
		fit_j = _ridge_cv_single_target(X_all, b_np[:, j], idx_v, ridge_grid, 5)
		mask = np.isfinite(fit_j["held_out"]) & (bm_np[:, j] > 0.5)
		if mask.sum() > 2:
			r2_all[j] = _r2(b_np[mask, j], fit_j["held_out"][mask])

	logger = get_stage2_logger()
	logger.info("behaviour_all_neurons", r2=r2_all[:min(n_modes, 6)].tolist())
	return {"r2_all_neurons": r2_all}
