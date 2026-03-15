#!/usr/bin/env python
"""Motor-neuron decoder analysis and posture videos.

Consolidated module for four motor-decoder analysis tools:

* **ridge** – Ridge-CV motor→eigenworm decoder with diagnostics plot + optional video.
* **decoder_video** – Fit decoder via ``stage2.behavior`` and render posture comparison.
* **traintest** – Train-fold vs test-fold posture video with fold-colour indicators.
* **lag_comparison** – Motor-only vs all-neuron decoders across lag steps.

CLI
---
    python -m scripts.motor_decoder_analysis ridge          --h5 ...
    python -m scripts.motor_decoder_analysis decoder_video  --h5 ...
    python -m scripts.motor_decoder_analysis traintest      --h5 ...
    python -m scripts.motor_decoder_analysis lag_comparison --h5 ...
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np

from stage2.config import Stage2PTConfig
from stage2.io_h5 import load_data_pt
from stage2.behavior import build_lagged_features_np
from stage2.evaluate import _r2
from stage2.ridge_utils import (
    _fit_ridge_regression,
    _log_ridge_grid,
    _make_contiguous_folds,
    _predict_linear_model,
    _ridge_cv_single_target,
)

# Re-use geometry helpers already consolidated in posture_videos
from scripts.posture_videos import (
    angles_to_xy as _angles_to_xy,
    _find_eigenvectors,
    make_posture_compare_video,
)


# ====================================================================
# Shared helpers
# ====================================================================

def _boundary_flag(res: dict, n_grid: int) -> str:
    if res["at_zero"]:
        return "  ⚠ AT LOWER BOUNDARY (λ=0, no regularisation)"
    if res["at_upper_boundary"]:
        return "  ⚠ AT UPPER BOUNDARY"
    return ""


def _calibrate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Rescale predictions to match observed mean and variance per mode."""
    out = y_pred.copy()
    L = y_true.shape[1]
    for j in range(L):
        m = valid_mask[:, j] & np.isfinite(y_pred[:, j])
        if m.sum() < 3:
            continue
        mu_y = np.mean(y_true[m, j])
        mu_p = np.mean(y_pred[m, j])
        sd_y = np.std(y_true[m, j])
        sd_p = np.std(y_pred[m, j])
        if sd_p < 1e-12:
            continue
        scale = sd_y / sd_p
        out[:, j] = mu_y + (y_pred[:, j] - mu_p) * scale
    return out


# ====================================================================
# Video renderers
# ====================================================================

def _render_video_ridge(
    h5_path: str,
    out_path: str,
    ew_gt: np.ndarray,
    ew_full: np.ndarray,
    ew_held: np.ndarray,
    r2_full: np.ndarray,
    r2_held: np.ndarray,
    *,
    n_modes: int = 6,
    fps: int = 15,
    dpi: int = 120,
    max_frames: int = 0,
    sample_rate: float = 1.0,
):
    """3-panel posture video: ground truth | full-fit | held-out."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    eigvecs_all = _find_eigenvectors(h5_path)
    n_modes = min(n_modes, ew_gt.shape[1], ew_full.shape[1],
                  ew_held.shape[1], eigvecs_all.shape[1])
    E = eigvecs_all[:, :n_modes]
    d_seg = E.shape[0]

    T = min(ew_gt.shape[0], ew_full.shape[0], ew_held.shape[0])
    if max_frames > 0:
        T = min(T, max_frames)
    dt = 1.0 / sample_rate

    ew_gt = ew_gt[:T, :n_modes]
    ew_full = ew_full[:T, :n_modes]
    ew_held = ew_held[:T, :n_modes]

    recon_gt = ew_gt @ E.T
    recon_full = ew_full @ E.T
    recon_held = ew_held @ E.T

    body_angles_raw = None
    with h5py.File(h5_path, "r") as f:
        for ba_key in ("behaviour/body_angle_dtarget",
                       "behavior/body_angle_all",
                       "behaviour/body_angle_all"):
            if ba_key in f:
                body_angles_raw = np.asarray(f[ba_key][:T], dtype=float)
                break

    datasets = [recon_gt, recon_full, recon_held]
    xy_all = []
    for recon in datasets:
        xy = np.zeros((T, d_seg, 2))
        for t in range(T):
            ang = recon[t]
            if np.all(np.isfinite(ang)):
                x, y = _angles_to_xy(ang)
            else:
                x = np.full(d_seg, np.nan)
                y = np.full(d_seg, np.nan)
            xy[t, :, 0], xy[t, :, 1] = x, y
        xy_all.append(xy)
    xy_gt, xy_full, xy_held = xy_all

    xy_raw = None
    if body_angles_raw is not None:
        ba = body_angles_raw
        xy_raw = np.zeros((T, ba.shape[1], 2))
        for t in range(T):
            shape = ba[t] - np.nanmean(ba[t])
            if np.all(np.isfinite(shape)):
                xr, yr = _angles_to_xy(shape)
            else:
                xr = np.full(ba.shape[1], np.nan)
                yr = np.full(ba.shape[1], np.nan)
            xy_raw[t, :, 0], xy_raw[t, :, 1] = xr, yr

    limit_arrays = xy_all + ([xy_raw] if xy_raw is not None else [])
    all_coords = np.concatenate([a.reshape(-1, 2) for a in limit_arrays], axis=0)
    m = np.all(np.isfinite(all_coords), axis=1)
    xmin, xmax = all_coords[m, 0].min(), all_coords[m, 0].max()
    ymin, ymax = all_coords[m, 1].min(), all_coords[m, 1].max()
    pad = 2.0
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    span = 0.5 * max(xmax - xmin, ymax - ymin) + pad

    r2f_str = ", ".join(f"{v:.2f}" for v in r2_full if np.isfinite(v))
    r2h_str = ", ".join(f"{v:.2f}" for v in r2_held if np.isfinite(v))
    has_raw = xy_raw is not None
    titles = [
        f"Ground truth  (raw + {n_modes}-mode recon)" if has_raw else "Ground truth",
        f"Full-fit  (R²: {r2f_str})",
        f"Held-out  (R²: {r2h_str})",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor="white")
    lines, heads = [], []
    line_raw, head_raw = None, None
    line_recon, head_recon = None, None

    for i, (ax, ttl) in enumerate(zip(axes, titles)):
        ax.set_title(ttl, fontsize=10, fontweight="bold")
        ax.set_xlim(cx - span, cx + span)
        ax.set_ylim(cy - span, cy + span)
        ax.set_aspect("equal")
        ax.set_facecolor("#f7f7f7")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for s in ax.spines.values():
            s.set_visible(False)
        line, = ax.plot([], [], "-", lw=2.5, alpha=0.9, color="k")
        head, = ax.plot([], [], "o", color="crimson", ms=5, alpha=0.9)
        lines.append(line)
        heads.append(head)
        if i == 0 and has_raw:
            line.set_label("Raw body angles")
            line_raw, head_raw = line, head
            line_recon, = ax.plot([], [], "-", lw=2.0, alpha=0.8,
                                  color="darkorange",
                                  label=f"EW recon ({n_modes} modes)")
            head_recon, = ax.plot([], [], "o", color="darkorange", ms=4, alpha=0.8)
            ax.legend(loc="upper right", frameon=False, fontsize=7)

    time_text = fig.text(0.5, 0.02, "", ha="center", va="bottom", fontsize=10)

    def _set(line, head, xy, t):
        x, y = xy[t, :, 0], xy[t, :, 1]
        line.set_data(x, y)
        if np.isfinite(x[0]) and np.isfinite(y[0]):
            head.set_data([x[0]], [y[0]])
        else:
            head.set_data([], [])

    def _update(frame):
        if has_raw:
            _set(line_raw, head_raw, xy_raw, frame)
            _set(line_recon, head_recon, xy_gt, frame)
        else:
            _set(lines[0], heads[0], xy_gt, frame)

        is_valid_full = np.isfinite(ew_full[frame]).all()
        lines[1].set_color("C0" if is_valid_full else "#cccccc")
        lines[1].set_alpha(0.9 if is_valid_full else 0.3)
        _set(lines[1], heads[1], xy_full, frame)

        is_valid_held = np.isfinite(ew_held[frame]).all()
        lines[2].set_color("C1" if is_valid_held else "#cccccc")
        lines[2].set_alpha(0.9 if is_valid_held else 0.3)
        _set(lines[2], heads[2], xy_held, frame)

        time_text.set_text(f"t = {frame * dt:.2f} s    frame {frame + 1}/{T}")
        extra = [line_recon, head_recon] if line_recon is not None else []
        return lines + heads + extra + [time_text]

    anim = FuncAnimation(fig, _update, frames=T,
                         interval=max(1, 1000 // max(1, fps)), blit=False)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, metadata={"title": "Motor ridge → eigenworm posture"},
                          bitrate=2400)
    print(f"[video] Rendering {T} frames at {fps} fps → {out_path}")
    anim.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"[video] Done: {out_path}")


def _render_video_traintest(
    h5_path: str,
    out_path: str,
    ew_gt: np.ndarray,
    ew_middle: np.ndarray,
    ew_test: np.ndarray,
    fold_ids: np.ndarray,
    r2_middle: np.ndarray,
    r2_test: np.ndarray,
    body_angles_raw: np.ndarray | None = None,
    *,
    middle_title: str = "Train prediction",
    n_modes: int = 6,
    fps: int = 15,
    dpi: int = 120,
    max_frames: int = 0,
    sample_rate: float = 1.0,
):
    """3-panel posture video with fold-colour indicator bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    eigvecs_all = _find_eigenvectors(h5_path)
    n_modes = min(n_modes, ew_gt.shape[1], ew_middle.shape[1],
                  ew_test.shape[1], eigvecs_all.shape[1])
    E = eigvecs_all[:, :n_modes]
    d_seg = E.shape[0]

    T = min(ew_gt.shape[0], ew_middle.shape[0], ew_test.shape[0])
    if max_frames > 0:
        T = min(T, max_frames)
    dt = 1.0 / sample_rate

    ew_gt = ew_gt[:T, :n_modes]
    ew_middle = ew_middle[:T, :n_modes]
    ew_test = ew_test[:T, :n_modes]
    fold_ids = fold_ids[:T]

    recon_gt = ew_gt @ E.T
    recon_middle = ew_middle @ E.T
    recon_test = ew_test @ E.T

    datasets = [recon_gt, recon_middle, recon_test]
    xy_all = []
    for recon in datasets:
        xy = np.zeros((T, d_seg, 2))
        for t in range(T):
            ang = recon[t]
            if np.all(np.isfinite(ang)):
                x, y = _angles_to_xy(ang)
            else:
                x = np.full(d_seg, np.nan)
                y = np.full(d_seg, np.nan)
            xy[t, :, 0], xy[t, :, 1] = x, y
        xy_all.append(xy)
    xy_gt, xy_middle, xy_test = xy_all

    xy_raw = None
    if body_angles_raw is not None:
        ba = body_angles_raw[:T]
        xy_raw = np.zeros((T, ba.shape[1], 2))
        for t in range(T):
            shape = ba[t] - np.nanmean(ba[t])
            if np.all(np.isfinite(shape)):
                xr, yr = _angles_to_xy(shape)
            else:
                xr = np.full(ba.shape[1], np.nan)
                yr = np.full(ba.shape[1], np.nan)
            xy_raw[t, :, 0], xy_raw[t, :, 1] = xr, yr

    limit_arrays = xy_all + ([xy_raw] if xy_raw is not None else [])
    all_coords = np.concatenate([a.reshape(-1, 2) for a in limit_arrays], axis=0)
    m = np.all(np.isfinite(all_coords), axis=1)
    xmin, xmax = all_coords[m, 0].min(), all_coords[m, 0].max()
    ymin, ymax = all_coords[m, 1].min(), all_coords[m, 1].max()
    pad = 2.0
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    span = 0.5 * max(xmax - xmin, ymax - ymin) + pad

    n_folds = int(fold_ids.max()) + 1
    fold_cmap = plt.get_cmap("Set2", max(n_folds, 3))

    fig = plt.figure(figsize=(14, 5.5), facecolor="white")
    gs = fig.add_gridspec(2, 3, height_ratios=[0.04, 1],
                          hspace=0.05, wspace=0.25,
                          left=0.04, right=0.96, top=0.90, bottom=0.06)
    ax_fb = [fig.add_subplot(gs[0, i]) for i in range(3)]
    axes = [fig.add_subplot(gs[1, i]) for i in range(3)]

    r2mid_str = ", ".join(f"{v:.2f}" for v in r2_middle if np.isfinite(v))
    r2te_str = ", ".join(f"{v:.2f}" for v in r2_test if np.isfinite(v))
    has_raw = xy_raw is not None
    titles = [
        f"Ground truth  (raw + {n_modes}-mode recon)" if has_raw else "Ground truth",
        f"{middle_title}  (R²: {r2mid_str})",
        f"Test prediction  (R²: {r2te_str})",
    ]

    lines, heads = [], []
    line_raw, head_raw = None, None
    line_recon, head_recon = None, None
    for i, (ax, ttl) in enumerate(zip(axes, titles)):
        ax.set_title(ttl, fontsize=10, fontweight="bold")
        ax.set_xlim(cx - span, cx + span)
        ax.set_ylim(cy - span, cy + span)
        ax.set_aspect("equal")
        ax.set_facecolor("#f7f7f7")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for s in ax.spines.values():
            s.set_visible(False)
        line, = ax.plot([], [], "-", lw=2.5, alpha=0.9, color="k")
        head, = ax.plot([], [], "o", color="crimson", ms=5, alpha=0.9)
        lines.append(line)
        heads.append(head)
        if i == 0 and has_raw:
            line.set_label("Raw body angles")
            line_raw, head_raw = line, head
            line_recon, = ax.plot([], [], "-", lw=2.0, alpha=0.8,
                                  color="darkorange",
                                  label=f"EW recon ({n_modes} modes)")
            head_recon, = ax.plot([], [], "o", color="darkorange", ms=4, alpha=0.8)
            ax.legend(loc="upper right", frameon=False, fontsize=7)

    for i, ax in enumerate(ax_fb):
        ax.set_xlim(0, T)
        ax.set_ylim(0, 1)
        ax.axis("off")
        for fi in range(n_folds):
            mask_fi = np.where(fold_ids == fi)[0]
            if mask_fi.size == 0:
                continue
            starts = [mask_fi[0]]
            for k in range(1, mask_fi.size):
                if mask_fi[k] != mask_fi[k - 1] + 1:
                    starts.append(mask_fi[k])
            for s in starts:
                e = s
                while e + 1 < T and fold_ids[e + 1] == fi:
                    e += 1
                ax.axvspan(s, e + 1, color=fold_cmap(fi), alpha=0.5)

    fold_indicators = []
    for ax in ax_fb:
        vl = ax.axvline(0, color="red", lw=1.2)
        fold_indicators.append(vl)

    time_text = fig.text(0.5, 0.01, "", ha="center", va="bottom", fontsize=10)
    fold_label = fig.text(0.5, 0.94, "", ha="center", va="bottom", fontsize=10,
                          fontstyle="italic", color="#555")

    def _set(line, head, xy, t):
        x, y = xy[t, :, 0], xy[t, :, 1]
        line.set_data(x, y)
        if np.isfinite(x[0]) and np.isfinite(y[0]):
            head.set_data([x[0]], [y[0]])
        else:
            head.set_data([], [])

    def _update(frame):
        if has_raw:
            _set(line_raw, head_raw, xy_raw, frame)
            _set(line_recon, head_recon, xy_gt, frame)
        else:
            _set(lines[0], heads[0], xy_gt, frame)

        is_valid_middle = np.isfinite(ew_middle[frame]).all()
        lines[1].set_color("C0" if is_valid_middle else "#cccccc")
        lines[1].set_alpha(0.9 if is_valid_middle else 0.3)
        _set(lines[1], heads[1], xy_middle, frame)

        is_valid_test = np.isfinite(ew_test[frame]).all()
        lines[2].set_color("C1" if is_valid_test else "#cccccc")
        lines[2].set_alpha(0.9 if is_valid_test else 0.3)
        _set(lines[2], heads[2], xy_test, frame)

        for vl in fold_indicators:
            vl.set_xdata([frame, frame])

        fi = fold_ids[frame]
        fold_label.set_text(f"Fold {fi}" if fi >= 0 else "")
        time_text.set_text(f"t = {frame * dt:.2f} s    frame {frame + 1}/{T}")
        extra = [line_recon, head_recon] if line_recon is not None else []
        return lines + heads + extra + fold_indicators + [time_text, fold_label]

    anim = FuncAnimation(fig, _update, frames=T,
                         interval=max(1, 1000 // max(1, fps)), blit=False)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, metadata={"title": "Posture train vs test"},
                          bitrate=2400)
    print(f"[video] Rendering {T} frames at {fps} fps → {out_path}")
    anim.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"[video] Done: {out_path}")


# ====================================================================
# Ridge-CV train/test helpers  (used by "traintest" subcommand)
# ====================================================================

def _ridge_cv_traintest(
    X: np.ndarray,
    Y: np.ndarray,
    valid_mask: np.ndarray,
    ridge_grid: np.ndarray,
    n_folds: int = 5,
):
    """Return full-length train-pred and test-pred arrays plus fold labels."""
    T, L_b = Y.shape
    n_lam = len(ridge_grid)

    train_preds = np.full((T, L_b), np.nan)
    test_preds = np.full((T, L_b), np.nan)
    r2_train = np.full(L_b, np.nan)
    r2_test = np.full(L_b, np.nan)
    best_lam = np.full(L_b, np.nan)

    all_valid = np.where(np.all(valid_mask, axis=1) & np.all(np.isfinite(X), axis=1))[0]
    folds = _make_contiguous_folds(all_valid, n_folds)
    fold_ids = np.full(T, -1, dtype=int)
    for fi, fold_idx in enumerate(folds):
        fold_ids[fold_idx] = fi

    for j in range(L_b):
        valid_j = valid_mask[:, j] & np.all(np.isfinite(X), axis=1)
        idx = np.where(valid_j)[0]
        if idx.size < 10:
            continue
        folds_j = _make_contiguous_folds(idx, n_folds)

        cv_mse = np.full(n_lam, np.inf)
        for li, lam in enumerate(ridge_grid):
            fold_mse = []
            for fold_idx in folds_j:
                train_idx = idx[~np.isin(idx, fold_idx)]
                if train_idx.size < 3 or fold_idx.size == 0:
                    continue
                fit = _fit_ridge_regression(X[train_idx], Y[train_idx, j], float(lam))
                if fit is None:
                    continue
                pred = _predict_linear_model(X[fold_idx], fit[0], fit[1])
                mse = float(np.mean((Y[fold_idx, j] - pred) ** 2))
                if np.isfinite(mse):
                    fold_mse.append(mse)
            if fold_mse:
                cv_mse[li] = np.mean(fold_mse)

        bi = int(np.nanargmin(cv_mse))
        best_lam[j] = ridge_grid[bi]

        for fold_idx in folds_j:
            train_idx = idx[~np.isin(idx, fold_idx)]
            if train_idx.size < 3:
                continue
            fit = _fit_ridge_regression(X[train_idx], Y[train_idx, j], float(best_lam[j]))
            if fit is None:
                continue
            intercept, coef = fit
            train_preds[train_idx, j] = _predict_linear_model(X[train_idx], intercept, coef)
            test_preds[fold_idx, j] = _predict_linear_model(X[fold_idx], intercept, coef)

        m_tr = np.isfinite(train_preds[:, j]) & valid_j
        m_te = np.isfinite(test_preds[:, j]) & valid_j
        if m_tr.sum() > 2:
            r2_train[j] = _r2(Y[m_tr, j], train_preds[m_tr, j])
        if m_te.sum() > 2:
            r2_test[j] = _r2(Y[m_te, j], test_preds[m_te, j])

    return {
        "train_preds": train_preds,
        "test_preds": test_preds,
        "fold_ids": fold_ids,
        "r2_train": r2_train,
        "r2_test": r2_test,
        "best_lambda": best_lam,
    }


def _ridge_cv_insample(
    X: np.ndarray,
    Y: np.ndarray,
    valid_mask: np.ndarray,
    ridge_grid: np.ndarray,
    n_folds: int = 5,
):
    """Return full-data in-sample predictions using λ chosen by CV per mode."""
    T, L_b = Y.shape
    pred_full = np.full((T, L_b), np.nan)
    r2_full = np.full(L_b, np.nan)
    best_lam = np.full(L_b, np.nan)

    for j in range(L_b):
        valid_j = valid_mask[:, j] & np.all(np.isfinite(X), axis=1)
        idx = np.where(valid_j)[0]
        if idx.size < 10:
            continue
        fit_j = _ridge_cv_single_target(X, Y[:, j], idx, ridge_grid, n_folds)
        pred_full[:, j] = fit_j["pred_full"]
        best_lam[j] = fit_j["best_lambda"]
        m = np.isfinite(pred_full[:, j]) & valid_j
        if m.sum() > 2:
            r2_full[j] = _r2(Y[m, j], pred_full[m, j])

    return {
        "pred_full": pred_full,
        "r2_full": r2_full,
        "best_lambda": best_lam,
    }


# ====================================================================
# Lag-comparison helper  (used by "lag_comparison" subcommand)
# ====================================================================

def _fit_r2_per_mode(
    u_np: np.ndarray,
    b_np: np.ndarray,
    bm_np: np.ndarray,
    neuron_idx: list[int],
    n_lags: int,
    ridge_grid: np.ndarray,
    n_folds: int,
) -> np.ndarray:
    """Return held-out CV R² per behaviour mode for one feature set and lag."""
    X = build_lagged_features_np(u_np[:, neuron_idx], n_lags)
    valid_X = np.all(np.isfinite(X), axis=1)

    L_b = b_np.shape[1]
    r2_modes = np.full(L_b, np.nan, dtype=float)

    for j in range(L_b):
        valid = (bm_np[:, j] > 0.5) & valid_X
        idx_v = np.where(valid)[0]
        if idx_v.size < 10:
            continue
        fit_j = _ridge_cv_single_target(X, b_np[:, j], idx_v, ridge_grid, n_folds)
        held = fit_j["held_out"]
        m = np.isfinite(held) & valid
        if m.sum() > 2:
            r2_modes[j] = _r2(b_np[m, j], held[m])

    return r2_modes


# ====================================================================
# Subcommand: ridge
# ====================================================================

def _cmd_ridge(args) -> None:
    """Motor-neuron → eigenworm ridge-CV decoder with diagnostics."""
    cfg = Stage2PTConfig(h5_path=args.h5, device=args.device)
    data = load_data_pt(cfg)

    u_np = data["u_stage1"].cpu().numpy()
    b_np = data["b"].cpu().numpy()
    bm_np = data["b_mask"].cpu().numpy()

    T, N = u_np.shape
    L_b = b_np.shape[1]
    motor_idx = list(cfg.motor_neurons) if cfg.motor_neurons else []
    M = len(motor_idx)
    if M == 0:
        raise RuntimeError("No motor neurons mapped for this recording.")

    X = build_lagged_features_np(u_np[:, motor_idx], args.lags)
    P = X.shape[1]

    if args.ols:
        ridge_grid = np.array([0.0])
    else:
        ridge_grid = _log_ridge_grid(args.log_lam_min, args.log_lam_max, args.n_grid)
    n_lam = len(ridge_grid)

    print(f"Recording : {Path(args.h5).stem}")
    print(f"T={T}  N={N}  motor={M}  eigenworm modes={L_b}")
    print(f"Lags={args.lags}  →  features P={P}   (T/P={T/P:.1f})")
    if args.ols:
        print("Regularisation: DISABLED (pure OLS, λ=0)")
    else:
        print(f"Lambda grid: {n_lam} values  "
              f"[0, 10^{args.log_lam_min} … 10^{args.log_lam_max}]")
    print(f"CV folds: {args.n_folds} (contiguous)")
    print()

    results: list[dict] = []
    t0 = time.time()

    for j in range(L_b):
        valid_j = (bm_np[:, j] > 0.5) & np.all(np.isfinite(X), axis=1)
        eval_idx = np.where(valid_j)[0]

        if eval_idx.size < 10:
            print(f"  EW{j+1}: skipped (only {eval_idx.size} valid frames)")
            results.append(None)
            continue

        res = _ridge_cv_single_target(
            X_full=X, y_full=b_np[:, j], eval_idx=eval_idx,
            ridge_grid=ridge_grid, n_folds=args.n_folds,
        )
        results.append(res)

        ho = res["held_out"]
        m_ho = np.isfinite(ho) & valid_j
        r2_ho = _r2(b_np[m_ho, j], ho[m_ho]) if m_ho.sum() > 2 else float("nan")

        pf = res["pred_full"]
        m_pf = np.isfinite(pf) & valid_j
        r2_full = _r2(b_np[m_pf, j], pf[m_pf]) if m_pf.sum() > 2 else float("nan")

        lam = res["best_lambda"]
        log_lam = np.log10(max(lam, 1e-50))
        flag = _boundary_flag(res, n_lam)

        print(f"  EW{j+1}:  R²(full)={r2_full:.4f}   R²(held-out)={r2_ho:.4f}   "
              f"best λ={lam:.2e}  log₁₀(λ)={log_lam:.1f}{flag}")

    elapsed = time.time() - t0
    print(f"\nTotal fit time: {elapsed:.1f}s")

    # ── summary table ────────────────────────────────────────────────
    r2_ho_arr = np.full(L_b, np.nan)
    r2_full_arr = np.full(L_b, np.nan)
    best_lam_arr = np.full(L_b, np.nan)
    at_lower = np.zeros(L_b, dtype=bool)
    at_upper = np.zeros(L_b, dtype=bool)

    for j, res in enumerate(results):
        if res is None:
            continue
        valid_j = (bm_np[:, j] > 0.5) & np.all(np.isfinite(X), axis=1)
        ho = res["held_out"]
        m = np.isfinite(ho) & valid_j
        if m.sum() > 2:
            r2_ho_arr[j] = _r2(b_np[m, j], ho[m])
        pf = res["pred_full"]
        m2 = np.isfinite(pf) & valid_j
        if m2.sum() > 2:
            r2_full_arr[j] = _r2(b_np[m2, j], pf[m2])
        best_lam_arr[j] = res["best_lambda"]
        at_lower[j] = res["at_zero"]
        at_upper[j] = res["at_upper_boundary"]

    print()
    print("=" * 68)
    print(f"{'Mode':>6}  {'R²(full)':>10}  {'R²(held-out)':>13}  "
          f"{'log₁₀(λ)':>10}  {'Boundary':>10}")
    print("-" * 68)
    for j in range(L_b):
        bnd = ""
        if at_lower[j]:
            bnd = "LOW ⚠"
        elif at_upper[j]:
            bnd = "HIGH ⚠"
        print(f"  EW{j+1:>2}  {r2_full_arr[j]:>10.4f}  {r2_ho_arr[j]:>13.4f}  "
              f"{np.log10(max(best_lam_arr[j], 1e-50)):>10.1f}  {bnd:>10}")
    print("-" * 68)
    print(f"  median {np.nanmedian(r2_full_arr):>10.4f}  "
          f"{np.nanmedian(r2_ho_arr):>13.4f}")
    n_lo = at_lower.sum()
    n_hi = at_upper.sum()
    if n_lo or n_hi:
        print(f"\n  ⚠  Boundary hits: {n_lo} at lower, {n_hi} at upper  — "
              "consider widening the λ grid.")
    else:
        print("\n  ✓  No boundary hits — λ grid is adequate.")
    print("=" * 68)

    # ── diagnostic plot ──────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    worm = Path(args.h5).stem
    out_path = args.out or f"output_plots/behaviour_plots/motor_ridge_eigenworms_{worm}.png"

    fig, axes_plot = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    ew_names = [f"EW{j+1}" for j in range(L_b)]

    ax = axes_plot[0]
    x = np.arange(L_b)
    w = 0.35
    ax.bar(x - w / 2, r2_full_arr, w, color="C0", alpha=0.8, label="Full-fit R²")
    ax.bar(x + w / 2, r2_ho_arr, w, color="C3", alpha=0.8, label="Held-out R²")
    ax.set_xticks(x)
    ax.set_xticklabels(ew_names)
    ax.set_ylabel("R²")
    ax.set_title("A.  Per-mode R²  (motor → eigenworms)")
    ax.axhline(0, color="grey", lw=0.6, ls=":")
    ax.legend(frameon=False, fontsize=9)

    ax = axes_plot[1]
    log_lam_plot = np.log10(np.maximum(ridge_grid, 1e-50))
    log_lam_plot[0] = args.log_lam_min - 1.5
    cmap = plt.get_cmap("tab10")
    for j, res in enumerate(results):
        if res is None:
            continue
        mse = res["cv_mse"]
        finite = np.isfinite(mse)
        if finite.sum() < 2:
            continue
        ax.plot(log_lam_plot[finite], mse[finite], lw=1.4, color=cmap(j),
                label=ew_names[j], alpha=0.85)
        bi = res["best_idx"]
        if bi >= 0:
            ax.axvline(log_lam_plot[bi], color=cmap(j), ls=":", lw=0.7, alpha=0.5)
    ax.set_xlabel("log₁₀(λ)")
    ax.set_ylabel("CV MSE")
    ax.set_title(f"B.  CV-MSE vs λ  (lag={args.lags})")
    ax.legend(frameon=False, fontsize=8, ncol=2)

    ax = axes_plot[2]
    log_best = np.log10(np.maximum(best_lam_arr, 1e-50))
    ax.bar(x, log_best, color="C2", alpha=0.8)
    for j in range(L_b):
        if at_lower[j]:
            ax.plot(x[j], log_best[j], "v", color="red", ms=9, zorder=5)
        if at_upper[j]:
            ax.plot(x[j], log_best[j], "^", color="red", ms=9, zorder=5)
    ax.axhline(args.log_lam_min, color="red", ls="--", lw=0.8,
               label=f"grid min={args.log_lam_min}")
    ax.axhline(args.log_lam_max, color="red", ls="--", lw=0.8,
               label=f"grid max={args.log_lam_max}")
    ax.set_xticks(x)
    ax.set_xticklabels(ew_names)
    ax.set_ylabel("log₁₀(best λ)")
    ax.set_title("C.  Selected λ  (▼▲ = boundary hit)")
    ax.legend(frameon=False, fontsize=8)

    if not args.no_plot:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        print(f"\nPlot saved: {out_path}")
    plt.close(fig)

    # ── optional video ───────────────────────────────────────────────
    if args.video and not args.no_plot:
        pred_full = np.full((T, L_b), np.nan)
        pred_held = np.full((T, L_b), np.nan)
        for j, res in enumerate(results):
            if res is None:
                continue
            pred_full[:, j] = res["pred_full"]
            pred_held[:, j] = res["held_out"]

        valid = bm_np > 0.5
        print("\n[video] Variance ratio  σ(pred)/σ(GT):")
        for j in range(L_b):
            m = valid[:, j] & np.isfinite(pred_full[:, j])
            if m.sum() < 3:
                continue
            ratio = np.std(pred_full[m, j]) / max(np.std(b_np[m, j]), 1e-12)
            print(f"  EW{j+1}: {ratio:.3f}")

        if not args.no_calibrate:
            pred_full = _calibrate_predictions(b_np, pred_full, valid)
            pred_held = _calibrate_predictions(b_np, pred_held, valid)
            print("[video] Predictions variance-calibrated.")
        else:
            print("[video] Calibration disabled — using raw ridge output.")

        sr = 1.0
        with h5py.File(args.h5, "r") as f:
            if "stage1/params" in f and "sample_rate_hz" in f["stage1/params"].attrs:
                sr = float(f["stage1/params"].attrs["sample_rate_hz"])
            elif "sample_rate_hz" in f.attrs:
                sr = float(f.attrs["sample_rate_hz"])

        vid_path = str(Path(out_path).with_suffix(".mp4"))
        _render_video_ridge(
            h5_path=args.h5, out_path=vid_path,
            ew_gt=b_np, ew_full=pred_full, ew_held=pred_held,
            r2_full=r2_full_arr, r2_held=r2_ho_arr,
            n_modes=L_b, fps=args.fps, dpi=args.dpi,
            max_frames=args.max_frames, sample_rate=sr,
        )


# ====================================================================
# Subcommand: decoder_video
# ====================================================================

def _cmd_decoder_video(args) -> None:
    """Fit motor→behaviour decoder via stage2.behavior and render posture video."""
    from stage2.behavior import (
        evaluate_training_decoder,
        fit_linear_behaviour_decoder_for_training,
    )

    t0 = time.time()
    cfg = Stage2PTConfig(h5_path=args.h5, device=args.device)
    data = load_data_pt(cfg)
    data["_cfg"] = cfg

    T, N = data["u_stage1"].shape
    n_motor = len(cfg.motor_neurons) if cfg.motor_neurons else 0
    L_b = data["b"].shape[1] if data.get("b") is not None else 0
    print(f"Loaded: T={T}  N={N}  motor={n_motor}  modes={L_b}")

    decoder = fit_linear_behaviour_decoder_for_training(data)
    if decoder is None:
        raise RuntimeError("Decoder fitting failed (no motor neurons / behaviour?)")

    onestep = {"prior_mu": data["u_stage1"].cpu().numpy()}
    res = evaluate_training_decoder(decoder, data, onestep)

    r2 = np.asarray(res["r2_model"], dtype=float)
    print("\nDecoder R² per eigenworm mode:")
    for i, v in enumerate(r2, start=1):
        print(f"  EW{i}: {v:.4f}")
    print(f"  median={np.nanmedian(r2):.4f}  mean={np.nanmean(r2):.4f}")

    worm_name = Path(args.h5).stem
    out_dir = args.out_dir or f"output_plots/stage1_quality_check/{worm_name}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(out_dir) / "posture_motor_decoder.mp4")

    make_posture_compare_video(
        h5_path=args.h5, out_path=out_path,
        ew_raw=res["b_actual"],
        ew_stage1=res["b_pred_gt"],
        ew_model_cv=res["b_pred_model"],
        fps=args.fps, dpi=args.dpi, max_frames=args.max_frames,
    )
    print(f"\nTotal wall time: {time.time() - t0:.1f}s")


# ====================================================================
# Subcommand: traintest
# ====================================================================

def _cmd_traintest(args) -> None:
    """Train-fold vs test-fold posture video with fold-colour indicators."""
    cfg = Stage2PTConfig(h5_path=args.h5, device=args.device)
    data = load_data_pt(cfg)

    u_np = data["u_stage1"].cpu().numpy()
    b_np = data["b"].cpu().numpy()
    bm_np = data["b_mask"].cpu().numpy()
    T, N = u_np.shape
    L_b = b_np.shape[1]
    motor_idx = list(cfg.motor_neurons) if cfg.motor_neurons else list(range(N))
    M = len(motor_idx)

    print(f"T={T}  N={N}  motor={M}  modes={L_b}  lag={args.lag}")
    print(f"Features per mode: {M * (1 + args.lag)}  (T/p = {T / (M * (1 + args.lag)):.1f})")

    ridge_grid = _log_ridge_grid(args.log_lam_min, args.log_lam_max, args.n_grid)
    X = build_lagged_features_np(u_np[:, motor_idx], args.lag)

    res = _ridge_cv_traintest(X, b_np, bm_np > 0.5, ridge_grid, args.n_folds)
    insample = _ridge_cv_insample(X, b_np, bm_np > 0.5, ridge_grid, args.n_folds)

    print(f"\nPer-mode results (lag={args.lag}):")
    for j in range(L_b):
        print(f"  EW{j+1}: train R²={res['r2_train'][j]:.4f}  "
              f"test R²={res['r2_test'][j]:.4f}  "
              f"best_λ={res['best_lambda'][j]:.2e}")
    print(f"  Median train R² = {np.nanmedian(res['r2_train']):.4f}")
    print(f"  Median test  R² = {np.nanmedian(res['r2_test']):.4f}")
    print(f"  Median in-sample R² = {np.nanmedian(insample['r2_full']):.4f}")

    if args.middle_panel == "insample":
        ew_middle = insample["pred_full"]
        r2_middle = insample["r2_full"]
        middle_title = "In-sample prediction"
    else:
        ew_middle = res["train_preds"]
        r2_middle = res["r2_train"]
        middle_title = "Train prediction"

    sr = 1.0
    body_angles_raw = None
    with h5py.File(args.h5, "r") as f:
        if "stage1/params" in f and "sample_rate_hz" in f["stage1/params"].attrs:
            sr = float(f["stage1/params"].attrs["sample_rate_hz"])
        elif "sample_rate_hz" in f.attrs:
            sr = float(f.attrs["sample_rate_hz"])
        for ba_key in ("behaviour/body_angle_dtarget",
                       "behavior/body_angle_all",
                       "behaviour/body_angle_all"):
            if ba_key in f:
                body_angles_raw = np.asarray(f[ba_key][:], dtype=float)
                print(f"Loaded raw body angles from '{ba_key}': {body_angles_raw.shape}")
                break

    out = args.out
    if out is None:
        stem = Path(args.h5).stem
        out = f"output_plots/behaviour_plots/{stem}_posture_traintest_lag{args.lag}.mp4"

    _render_video_traintest(
        args.h5, out,
        ew_gt=b_np, ew_middle=ew_middle,
        ew_test=res["test_preds"],
        fold_ids=res["fold_ids"],
        r2_middle=r2_middle, r2_test=res["r2_test"],
        body_angles_raw=body_angles_raw,
        middle_title=middle_title,
        n_modes=L_b, fps=args.fps, dpi=args.dpi,
        max_frames=args.max_frames, sample_rate=sr,
    )


# ====================================================================
# Subcommand: lag_comparison
# ====================================================================

def _cmd_lag_comparison(args) -> None:
    """Compare motor-only vs all-neuron decoders across lag steps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = Stage2PTConfig(h5_path=args.h5, device=args.device)
    data = load_data_pt(cfg)

    u_np = data["u_stage1"].cpu().numpy()
    b_np = data["b"].cpu().numpy()
    bm_np = data["b_mask"].cpu().numpy()
    T, N = u_np.shape
    L_b = b_np.shape[1]

    motor_idx = list(cfg.motor_neurons) if cfg.motor_neurons is not None else []
    if len(motor_idx) == 0:
        raise RuntimeError("No motor neurons mapped; cannot run motor-only comparison")
    all_idx = list(range(N))

    n_folds = int(getattr(cfg, "train_behavior_ridge_folds", 5) or 5)
    log_lambda_min = float(getattr(cfg, "train_behavior_ridge_log_lambda_min", -3.0))
    log_lambda_max = float(getattr(cfg, "train_behavior_ridge_log_lambda_max", 3.0))
    n_grid = int(getattr(cfg, "train_behavior_ridge_n_grid", 60) or 60)
    ridge_grid = _log_ridge_grid(log_lambda_min, log_lambda_max, n_grid)

    lags = sorted(set(int(x) for x in args.lags))

    r2_motor = np.full((len(lags), L_b), np.nan, dtype=float)
    r2_all = np.full((len(lags), L_b), np.nan, dtype=float)

    print(f"Data: T={T}, N={N}, behaviour modes={L_b}, motor neurons={len(motor_idx)}")
    print(f"Lags: {lags}")

    for i, lag in enumerate(lags):
        print(f"[lag={lag}] fitting motor-only...")
        r2_motor[i] = _fit_r2_per_mode(
            u_np, b_np, bm_np, motor_idx, lag, ridge_grid, n_folds,
        )
        print(f"[lag={lag}] fitting all-neuron...")
        r2_all[i] = _fit_r2_per_mode(
            u_np, b_np, bm_np, all_idx, lag, ridge_grid, n_folds,
        )

    med_motor = np.nanmedian(r2_motor, axis=1)
    med_all = np.nanmedian(r2_all, axis=1)

    fig, axes_plot = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    ax = axes_plot[0]
    ax.plot(lags, med_motor, marker="o", lw=2.2, label="Motor neurons only")
    ax.plot(lags, med_all, marker="s", lw=2.2, label="All neurons")
    ax.axhline(0, color="gray", lw=1, ls=":")
    ax.set_xlabel("Lag steps")
    ax.set_ylabel("Median held-out CV R² across eigenworm modes")
    ax.set_title("A. Decoder dynamics vs lag")
    ax.legend(frameon=False)

    ax = axes_plot[1]
    cmap = plt.get_cmap("tab10")
    for j in range(L_b):
        c = cmap(j % 10)
        ax.plot(lags, r2_motor[:, j], color=c, lw=1.8, ls="-", alpha=0.95)
        ax.plot(lags, r2_all[:, j], color=c, lw=1.8, ls="--", alpha=0.95)
    ax.axhline(0, color="gray", lw=1, ls=":")
    ax.set_xlabel("Lag steps")
    ax.set_ylabel("Held-out CV R²")
    ax.set_title("B. Per-mode dynamics (solid=motor, dashed=all)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    np.savetxt(out_path.with_suffix(".motor_r2.csv"), r2_motor, delimiter=",", fmt="%.6f")
    np.savetxt(out_path.with_suffix(".all_r2.csv"), r2_all, delimiter=",", fmt="%.6f")
    np.savetxt(out_path.with_suffix(".lags.csv"), np.array(lags), delimiter=",", fmt="%d")

    print(f"Saved plot: {out_path}")
    print(f"Saved CSVs: {out_path.with_suffix('.motor_r2.csv')}, "
          f"{out_path.with_suffix('.all_r2.csv')}")


# ====================================================================
# CLI
# ====================================================================

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Motor-neuron decoder analysis and posture videos.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── ridge ────────────────────────────────────────────────────────
    p = sub.add_parser("ridge",
                       help="Ridge-CV: motor-neuron → eigenworm decoder with diagnostics.")
    p.add_argument("--h5", required=True)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--lags", type=int, default=10)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--log_lam_min", type=float, default=-4.0)
    p.add_argument("--log_lam_max", type=float, default=6.0)
    p.add_argument("--n_grid", type=int, default=80)
    p.add_argument("--out", default=None)
    p.add_argument("--video", action="store_true")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--max_frames", type=int, default=0)
    p.add_argument("--no-calibrate", action="store_true")
    p.add_argument("--ols", action="store_true")
    p.add_argument("--no_plot", action="store_true")
    p.set_defaults(func=_cmd_ridge)

    # ── decoder_video ────────────────────────────────────────────────
    p = sub.add_parser("decoder_video",
                       help="Fit decoder via stage2.behavior and render comparison video.")
    p.add_argument("--h5", required=True)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--out_dir", default=None)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--max_frames", type=int, default=0)
    p.set_defaults(func=_cmd_decoder_video)

    # ── traintest ────────────────────────────────────────────────────
    p = sub.add_parser("traintest",
                       help="Train-fold vs test-fold posture video with fold indicators.")
    p.add_argument("--h5", required=True)
    p.add_argument("--lag", type=int, default=8)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--log_lam_min", type=float, default=-4.0)
    p.add_argument("--log_lam_max", type=float, default=6.0)
    p.add_argument("--n_grid", type=int, default=80)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--middle-panel", choices=("train", "insample"), default="train")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--max_frames", type=int, default=0)
    p.set_defaults(func=_cmd_traintest)

    # ── lag_comparison ───────────────────────────────────────────────
    p = sub.add_parser("lag_comparison",
                       help="Motor-only vs all-neuron decoders across lag steps.")
    p.add_argument("--h5", required=True)
    p.add_argument("--out", default="output_plots/behaviour_plots/motor_vs_all_lags.png")
    p.add_argument("--lags", nargs="+", type=int, default=[3, 8, 20, 60])
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.set_defaults(func=_cmd_lag_comparison)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
