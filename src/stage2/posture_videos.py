#!/usr/bin/env python
"""Worm posture video generation.

Consolidated module for three posture-video tools:

* **make_video** – Animate raw body-angle posture with curvature colouring.
* **make_eigenworm_video** – Reconstruct posture from eigenworm amplitudes.
* **make_posture_compare_video** – 3-panel comparison
  (ground truth | motor raw decoded | motor model decoded).

CLI
---
    python -m scripts.posture_videos raw       --h5 ... --out ...
    python -m scripts.posture_videos eigenworm --h5 ... --out ...
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.interpolate import interp1d

from stage1.preprocess import normalize_body_angle_fixed_length

__all__ = ["make_posture_compare_video"]


# ====================================================================
# Shared geometry helpers
# ====================================================================

def angles_to_xy(angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert tangent angles to centred (x, y) centerline coordinates."""
    x = np.cumsum(np.cos(angles))
    y = np.cumsum(np.sin(angles))
    x -= np.nanmean(x)
    y -= np.nanmean(y)
    return x, y


# Default Stephens eigenvectors path (relative to the file or project root).
_STEPHENS_EIGVEC_NAME = "eigenvectors_stephens.npy"


def _load_eigenvectors(eigvec_npy: Optional[str] = None,
                       h5_path: Optional[str] = None) -> np.ndarray:
    """Load Stephens eigenvectors from *eigvec_npy*.

    Search order (first match wins):
    1. Explicit *eigvec_npy* argument.
    2. ``eigenvectors_stephens.npy`` in the same folder as the H5 file.
    3. ``eigenvectors_stephens.npy`` in the parent / grandparent folders of the H5.
    """
    candidates: list[Path] = []
    if eigvec_npy:
        candidates.append(Path(eigvec_npy))
    if h5_path:
        h5_dir = Path(h5_path).resolve().parent
        for d in (h5_dir, h5_dir.parent, h5_dir.parent.parent):
            candidates.append(d / _STEPHENS_EIGVEC_NAME)
            # Also check sibling ``stephens (2016)`` folder.
            candidates.append(d / "stephens (2016)" / _STEPHENS_EIGVEC_NAME)
    for p in candidates:
        if p.exists():
            return np.load(p)
    raise FileNotFoundError(
        f"Cannot find {_STEPHENS_EIGVEC_NAME}. "
        "Run scripts/add_stephens_eigenworms.py first, or pass --eigvec_npy."
    )


def _make_colored_line(x, y, curvature, cmap, norm):
    """Return a :class:`LineCollection` coloured by local curvature."""
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    c = 0.5 * (curvature[:-1] + curvature[1:])
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=4,
                        capstyle="round")
    lc.set_array(c)
    return lc


# ── helpers unique to *make_video* ───────────────────────────────────

def _interpolate_angles(angles: np.ndarray) -> np.ndarray:
    """Fill NaN gaps in a 1-D angle array by linear interpolation."""
    finite = np.isfinite(angles)
    if not np.any(finite):
        return np.zeros_like(angles)
    if np.all(finite):
        return angles
    idx = np.arange(len(angles))
    f = interp1d(idx[finite], angles[finite], kind="linear",
                 fill_value="extrapolate")
    return f(idx)


def _longest_finite_run(mask: np.ndarray) -> tuple[int, int] | None:
    """Return [start, end) of the longest contiguous True run in *mask*."""
    best = None
    best_len = 0
    start = None
    for i, ok in enumerate(mask):
        if ok and start is None:
            start = i
        elif (not ok) and start is not None:
            length = i - start
            if length > best_len:
                best_len = length
                best = (start, i)
            start = None
    if start is not None:
        length = len(mask) - start
        if length > best_len:
            best = (start, len(mask))
            best_len = length
    if best is None or best_len < 2:
        return None
    return best


def original_trace_xy(angles_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute visible original trace from raw angles (longest finite run)."""
    finite = np.isfinite(angles_raw)
    run = _longest_finite_run(finite)
    if run is None:
        return np.array([]), np.array([])
    a, b = run
    return angles_to_xy(angles_raw[a:b])


# ====================================================================
# 1) Raw body-angle video  (was make_worm_video.py)
# ====================================================================

def make_video(
    h5_path: str,
    out_path: str,
    fps: int = 15,
    dpi: int = 120,
    normalize_length: bool = False,
) -> None:
    """Render the worm posture video from raw body angles."""

    # ── load data ────────────────────────────────────────────────────
    with h5py.File(h5_path, "r") as f:
        body_angle = f["behavior/body_angle_all"][:]       # (T, n_seg)
        eigenworms = f["behavior/eigenworms"][:]           # (T, n_modes)
        sr = 1.0
        if "stage1/params" in f and "sample_rate_hz" in f["stage1/params"].attrs:
            sr = float(f["stage1/params"].attrs["sample_rate_hz"])
        elif "sample_rate_hz" in f.attrs:
            sr = float(f.attrs["sample_rate_hz"])
        velocity = None
        if "behavior/velocity" in f:
            velocity = f["behavior/velocity"][:]

    T, n_seg = body_angle.shape
    n_modes = eigenworms.shape[1]
    dt = 1.0 / sr

    body_angle_for_recon = body_angle
    if normalize_length:
        body_angle_for_recon, valid_norm = normalize_body_angle_fixed_length(
            body_angle, n_target_segments=n_seg, min_valid_segments=10,
        )
        for t in range(T):
            if not valid_norm[t]:
                body_angle_for_recon[t] = body_angle[t]

    # ── preprocess: interpolate NaNs per frame ───────────────────────
    angles_clean = np.zeros_like(body_angle)
    valid_mask = np.zeros(T, dtype=bool)
    for t in range(T):
        n_fin = np.sum(np.isfinite(body_angle_for_recon[t]))
        if n_fin >= 10:
            angles_clean[t] = _interpolate_angles(body_angle_for_recon[t])
            valid_mask[t] = True
        else:
            if t > 0:
                angles_clean[t] = angles_clean[t - 1]
            valid_mask[t] = False

    # ── precompute all (x,y) and curvatures ──────────────────────────
    all_xy = np.zeros((T, n_seg, 2))
    all_curv = np.zeros((T, n_seg))
    all_xy_orig = []
    for t in range(T):
        x, y = angles_to_xy(angles_clean[t])
        all_xy[t, :, 0] = x
        all_xy[t, :, 1] = y
        all_curv[t, 1:] = np.diff(angles_clean[t])
        all_curv[t, 0] = all_curv[t, 1]
        x0, y0 = original_trace_xy(body_angle[t])
        all_xy_orig.append((x0, y0))

    # global axis limits
    pad = 5
    xmin, xmax = all_xy[:, :, 0].min() - pad, all_xy[:, :, 0].max() + pad
    ymin, ymax = all_xy[:, :, 1].min() - pad, all_xy[:, :, 1].max() + pad
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    span = max(xmax - xmin, ymax - ymin) / 2
    xmin, xmax = cx - span, cx + span
    ymin, ymax = cy - span, cy + span

    curv_abs_max = np.percentile(np.abs(all_curv), 99)
    norm = mcolors.Normalize(vmin=-curv_abs_max, vmax=curv_abs_max)
    cmap = plt.get_cmap("coolwarm")

    ew_max = np.nanpercentile(np.abs(eigenworms), 99)

    # ── set up figure ────────────────────────────────────────────────
    fig = plt.figure(figsize=(7, 7), facecolor="white")
    gs = fig.add_gridspec(5, 1, height_ratios=[4, 0.15, 0.6, 0.05, 0.3],
                          hspace=0.3)

    ax_worm = fig.add_subplot(gs[0])
    ax_worm.set_xlim(xmin, xmax)
    ax_worm.set_ylim(ymin, ymax)
    ax_worm.set_aspect("equal")
    ax_worm.set_facecolor("#f7f7f7")
    ax_worm.tick_params(left=False, bottom=False,
                        labelleft=False, labelbottom=False)
    for spine in ax_worm.spines.values():
        spine.set_visible(False)

    title_text = ax_worm.set_title("", fontsize=13, fontweight="bold")
    head_dot, = ax_worm.plot([], [], "o", color="black", markersize=7,
                             zorder=10)
    raw_line, = ax_worm.plot([], [], color="black", linewidth=1.2,
                             alpha=0.85, zorder=12)

    # colorbar
    ax_cb = fig.add_subplot(gs[1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax_cb, orientation="horizontal")
    cb.set_label("Local curvature", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    # eigenworm bar axes
    ax_ew = fig.add_subplot(gs[2])
    n_show = min(n_modes, 5)
    ew_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"][:n_show]
    bars = ax_ew.barh(range(n_show), np.zeros(n_show), color=ew_colors,
                      height=0.7)
    ax_ew.set_xlim(-ew_max, ew_max)
    ax_ew.set_yticks(range(n_show))
    ax_ew.set_yticklabels([f"EW{i+1}" for i in range(n_show)], fontsize=9)
    ax_ew.axvline(0, color="grey", linewidth=0.5)
    ax_ew.set_xlabel("Amplitude", fontsize=9)
    ax_ew.tick_params(labelsize=8)
    ax_ew.invert_yaxis()

    # velocity text
    ax_vel = fig.add_subplot(gs[4])
    ax_vel.axis("off")
    vel_text = ax_vel.text(0.5, 0.5, "", ha="center", va="center",
                           fontsize=11, transform=ax_vel.transAxes)

    lc_holder = [None]

    def _update(frame):
        t = frame
        if lc_holder[0] is not None:
            lc_holder[0].remove()

        x = all_xy[t, :, 0]
        y = all_xy[t, :, 1]
        curv = all_curv[t]

        lc = _make_colored_line(x, y, curv, cmap, norm)
        ax_worm.add_collection(lc)
        lc_holder[0] = lc

        x0, y0 = all_xy_orig[t]
        raw_line.set_data(x0, y0)
        head_dot.set_data([x[0]], [y[0]])

        time_s = t * dt
        title_text.set_text(
            f"t = {time_s:.1f} s  (frame {t}/{T})"
            + ("  [len-normalized]" if normalize_length else "")
            + ("" if valid_mask[t] else "  [interpolated]")
        )

        for i in range(n_show):
            bars[i].set_width(eigenworms[t, i])

        if velocity is not None and np.isfinite(velocity[t]):
            vel_text.set_text(f"Velocity: {velocity[t]:.2f}")
        else:
            vel_text.set_text("Velocity: N/A")

        return [lc, head_dot, raw_line, title_text] + list(bars) + [vel_text]

    anim = FuncAnimation(fig, _update, frames=T, interval=1000 // fps,
                         blit=False)
    writer = FFMpegWriter(fps=fps, metadata={"title": "Worm posture"},
                          bitrate=2000)
    print(f"Rendering {T} frames at {fps} fps → {out_path} ...")
    anim.save(out_path, writer=writer, dpi=dpi)
    print(f"Done. Video saved to {out_path}")
    plt.close(fig)


# ====================================================================
# 2) Eigenworm reconstruction video  (was eigenworm_video.py)
# ====================================================================

def make_eigenworm_video(
    h5_path: str,
    out_path: str,
    fps: int = 15,
    dpi: int = 120,
    show_original: bool = True,
    n_modes: int = 6,
    eigvec_npy: Optional[str] = None,
) -> None:
    """Render worm posture video from eigenworm amplitudes."""

    eigvecs_all = _load_eigenvectors(eigvec_npy=eigvec_npy, h5_path=h5_path)
    E = eigvecs_all[:, :n_modes]
    print(f"Loaded eigenvectors: {eigvecs_all.shape[1]} total, using {n_modes}")

    # ── load data ────────────────────────────────────────────────────
    with h5py.File(h5_path, "r") as f:
        for ba_key in ("behaviour/body_angle_dtarget",
                       "behavior/body_angle_all",
                       "behaviour/body_angle_all"):
            if ba_key in f:
                break
        body_angle = f[ba_key][:]

        for ew_key in ("behaviour/eigenworms_stephens",
                       "behavior/eigenworms"):
            if ew_key in f:
                break
        eigenworms = f[ew_key][:]

        sr = 1.0
        if "stage1/params" in f and "sample_rate_hz" in f["stage1/params"].attrs:
            sr = float(f["stage1/params"].attrs["sample_rate_hz"])
        elif "sample_rate_hz" in f.attrs:
            sr = float(f.attrs["sample_rate_hz"])
        velocity = f["behavior/velocity"][:] if "behavior/velocity" in f else None

    T, n_seg = body_angle.shape
    n_modes = min(n_modes, eigenworms.shape[1], E.shape[1])
    E = E[:, :n_modes]
    eigenworms = eigenworms[:, :n_modes]
    dt = 1.0 / sr
    n_seg_recon = E.shape[0]   # eigenvector dimension (e.g. 100 for Stephens basis)

    # ── reconstruct: amps @ E.T ──────────────────────────────────────
    print(f"Reconstructing from {n_modes} eigenworms → ({n_seg_recon},) body angles")
    all_xy_recon = np.zeros((T, n_seg_recon, 2))
    all_curv_recon = np.zeros((T, n_seg_recon))
    for t in range(T):
        recon = eigenworms[t] @ E.T
        x, y = angles_to_xy(recon)
        all_xy_recon[t, :, 0] = x
        all_xy_recon[t, :, 1] = y
        all_curv_recon[t, 1:] = np.diff(recon)
        all_curv_recon[t, 0] = all_curv_recon[t, 1]

    # ── precompute original body angles ──────────────────────────────
    # Resample raw angles to n_seg_recon so arc-length matches the reconstruction.
    all_xy_orig = []
    if show_original:
        xp = np.linspace(0, 1, n_seg)
        xq = np.linspace(0, 1, n_seg_recon)
        for t in range(T):
            row = body_angle[t]
            shape = row - np.nanmean(row)
            if np.all(np.isfinite(shape)):
                shape_r = np.interp(xq, xp, shape)
                x, y = angles_to_xy(shape_r)
                all_xy_orig.append((x, y))
            else:
                all_xy_orig.append((np.array([]), np.array([])))
    else:
        all_xy_orig = [(np.array([]), np.array([]))] * T

    # ── axis limits ──────────────────────────────────────────────────
    pad = 3
    xmin = all_xy_recon[:, :, 0].min() - pad
    xmax = all_xy_recon[:, :, 0].max() + pad
    ymin = all_xy_recon[:, :, 1].min() - pad
    ymax = all_xy_recon[:, :, 1].max() + pad
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    span = max(xmax - xmin, ymax - ymin) / 2
    xmin, xmax = cx - span, cx + span
    ymin, ymax = cy - span, cy + span

    curv_abs_max = np.percentile(np.abs(all_curv_recon), 99)
    norm = mcolors.Normalize(vmin=-curv_abs_max, vmax=curv_abs_max)
    cmap = plt.get_cmap("coolwarm")

    ew_max = np.nanpercentile(np.abs(eigenworms), 99)

    # ── figure ───────────────────────────────────────────────────────
    fig = plt.figure(figsize=(7, 7), facecolor="white")
    gs = fig.add_gridspec(4, 1, height_ratios=[4, 0.15, 0.6, 0.3],
                          hspace=0.30)

    ax_worm = fig.add_subplot(gs[0])
    ax_worm.set_xlim(xmin, xmax)
    ax_worm.set_ylim(ymin, ymax)
    ax_worm.set_aspect("equal")
    ax_worm.set_facecolor("#f7f7f7")
    ax_worm.tick_params(left=False, bottom=False,
                        labelleft=False, labelbottom=False)
    for spine in ax_worm.spines.values():
        spine.set_visible(False)

    title_text = ax_worm.set_title("", fontsize=13, fontweight="bold")
    head_dot, = ax_worm.plot([], [], "o", color="black", markersize=7,
                             zorder=10)
    raw_line, = ax_worm.plot([], [], color="black", linewidth=1.0,
                             alpha=0.4, zorder=5, label="Original")

    ax_cb = fig.add_subplot(gs[1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax_cb, orientation="horizontal")
    cb.set_label("Local curvature", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    ax_ew = fig.add_subplot(gs[2])
    n_show = min(n_modes, 5)
    ew_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"][:n_show]
    bars = ax_ew.barh(range(n_show), np.zeros(n_show), color=ew_colors,
                      height=0.7)
    ax_ew.set_xlim(-ew_max, ew_max)
    ax_ew.set_yticks(range(n_show))
    ax_ew.set_yticklabels([f"EW{i+1}" for i in range(n_show)], fontsize=9)
    ax_ew.axvline(0, color="grey", linewidth=0.5)
    ax_ew.set_xlabel("Amplitude", fontsize=9)
    ax_ew.tick_params(labelsize=8)
    ax_ew.invert_yaxis()

    ax_info = fig.add_subplot(gs[3])
    ax_info.axis("off")
    info_text = ax_info.text(0.5, 0.5, "", ha="center", va="center",
                             fontsize=11, transform=ax_info.transAxes)

    lc_holder = [None]

    def _update(frame):
        t = frame
        if lc_holder[0] is not None:
            lc_holder[0].remove()

        x = all_xy_recon[t, :, 0]
        y = all_xy_recon[t, :, 1]
        curv = all_curv_recon[t]

        lc = _make_colored_line(x, y, curv, cmap, norm)
        ax_worm.add_collection(lc)
        lc_holder[0] = lc

        x0, y0 = all_xy_orig[t]
        raw_line.set_data(x0, y0)
        head_dot.set_data([x[0]], [y[0]])

        time_s = t * dt
        title_text.set_text(
            f"Eigenworm reconstruction  |  t = {time_s:.1f} s  (frame {t}/{T})")

        for i in range(n_show):
            bars[i].set_width(eigenworms[t, i])

        parts = []
        if velocity is not None and np.isfinite(velocity[t]):
            parts.append(f"Velocity: {velocity[t]:.2f}")
        info_text.set_text("    ".join(parts))

        return [lc, head_dot, raw_line, title_text] + list(bars) + [info_text]

    anim = FuncAnimation(fig, _update, frames=T, interval=1000 // fps,
                         blit=False)
    writer = FFMpegWriter(fps=fps, metadata={"title": "Worm eigenworm posture"},
                          bitrate=2000)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"Rendering {T} frames at {fps} fps → {out_path} …")
    anim.save(out_path, writer=writer, dpi=dpi)
    print(f"Done. Video saved to {out_path}")
    plt.close(fig)


# ====================================================================
# 3) Posture comparison video  (was posture_compare_video.py)
# ====================================================================

def make_posture_compare_video(
    h5_path: str,
    out_path: str,
    *,
    ew_raw: Optional[np.ndarray] = None,
    ew_stage1: Optional[np.ndarray] = None,
    ew_model_cv: Optional[np.ndarray] = None,
    # Legacy kwargs (accepted for backward compat, unused)
    u_stage1: Optional[np.ndarray] = None,
    u_model_pred: Optional[np.ndarray] = None,
    motor_neuron_indices: Optional[Sequence[int]] = None,
    n_eigenworm_modes: int = 6,
    n_decoder_lags: int = 3,
    ridge_alpha: float = 10.0,
    smooth_window: int = 5,
    fps: int = 15,
    dpi: int = 120,
    max_frames: int = 0,
    eigvec_npy: Optional[str] = None,
) -> None:
    """Render side-by-side posture video.

    Panels: ground truth | motor raw decoded | motor model decoded.

    Parameters
    ----------
    h5_path : str
        Path to the recording HDF5 file.
    out_path : str
        Output MP4 path.
    ew_raw : (T, n_modes) array
        Ground-truth eigenworm amplitudes.
    ew_stage1 : (T, n_modes) array
        Eigenworms decoded from raw motor-neuron activity.
    ew_model_cv : (T, n_modes) array
        Eigenworms decoded from model-predicted motor-neuron activity.
    """
    if ew_stage1 is None or ew_model_cv is None:
        raise ValueError(
            "make_posture_compare_video requires ew_stage1 (motor raw) "
            "and ew_model_cv (motor model) eigenworm predictions."
        )

    eigvecs_all = _load_eigenvectors(eigvec_npy=eigvec_npy, h5_path=h5_path)

    with h5py.File(h5_path, "r") as f:
        for ba_key in ("behaviour/body_angle_dtarget",
                       "behavior/body_angle_all",
                       "behaviour/body_angle_all"):
            if ba_key in f:
                break
        body_angle = np.asarray(f[ba_key][:], dtype=float)

        if ew_raw is None:
            for ew_key in ("behaviour/eigenworms_stephens",):
                if ew_key in f:
                    break
            ew_raw = np.asarray(f[ew_key][:], dtype=float)

        sr = 1.0
        if "stage1/params" in f and "sample_rate_hz" in f["stage1/params"].attrs:
            sr = float(f["stage1/params"].attrs["sample_rate_hz"])
        elif "sample_rate_hz" in f.attrs:
            sr = float(f.attrs["sample_rate_hz"])

    dt = 1.0 / sr

    n_modes = min(
        ew_raw.shape[1], ew_stage1.shape[1], ew_model_cv.shape[1],
        eigvecs_all.shape[1], n_eigenworm_modes,
    )
    E = eigvecs_all[:, :n_modes]

    T = min(
        body_angle.shape[0], ew_raw.shape[0],
        ew_stage1.shape[0], ew_model_cv.shape[0],
    )
    if max_frames > 0:
        T = min(T, int(max_frames))

    body_angle = body_angle[:T]
    ew_raw_t = ew_raw[:T, :n_modes]
    ew_s1 = ew_stage1[:T, :n_modes]
    ew_mod = ew_model_cv[:T, :n_modes]
    d_seg = body_angle.shape[1]
    d_recon = E.shape[0]  # eigenvector dimension (e.g. 100 for Stephens basis)

    print(f"[posture_video] {n_modes} eigenworm modes, {d_seg} body segments, "
          f"{d_recon} recon segments, {T} frames")

    recon_raw_ew = ew_raw_t @ E.T   # (T, d_recon)
    recon_s1 = ew_s1 @ E.T
    recon_mod = ew_mod @ E.T

    # ── compute R² for decoded eigenworms ────────────────────────────
    for label, ew_dec in [("Motor raw", ew_s1), ("Motor model", ew_mod)]:
        r2_modes = []
        for j in range(n_modes):
            v = np.isfinite(ew_raw_t[:, j]) & np.isfinite(ew_dec[:, j])
            if v.sum() > 10:
                ss_res = np.sum((ew_raw_t[v, j] - ew_dec[v, j]) ** 2)
                ss_tot = np.sum((ew_raw_t[v, j] - np.mean(ew_raw_t[v, j])) ** 2)
                r2_modes.append(1 - ss_res / max(ss_tot, 1e-12))
        if r2_modes:
            print(f"[posture_video] Eigenworm R² ({label}): " +
                  ", ".join(f"m{i}={r:.3f}" for i, r in enumerate(r2_modes)))

    # ── convert to XY trajectories ───────────────────────────────────
    # Raw body angles use d_seg; eigenworm reconstructions use d_recon.
    xy_raw = np.zeros((T, d_seg, 2), dtype=float)
    xy_raw_ew = np.zeros((T, d_recon, 2), dtype=float)
    xy_s1 = np.zeros((T, d_recon, 2), dtype=float)
    xy_mod = np.zeros((T, d_recon, 2), dtype=float)

    for t in range(T):
        shape = body_angle[t] - np.mean(body_angle[t])
        if np.all(np.isfinite(shape)):
            x, y = angles_to_xy(shape)
        else:
            x = np.full(d_seg, np.nan)
            y = np.full(d_seg, np.nan)
        xy_raw[t, :, 0], xy_raw[t, :, 1] = x, y

        xr, yr = angles_to_xy(recon_raw_ew[t])
        xy_raw_ew[t, :, 0], xy_raw_ew[t, :, 1] = xr, yr

        x1, y1 = angles_to_xy(recon_s1[t])
        xy_s1[t, :, 0], xy_s1[t, :, 1] = x1, y1

        x2, y2 = angles_to_xy(recon_mod[t])
        xy_mod[t, :, 0], xy_mod[t, :, 1] = x2, y2

    # ── shared axis limits ───────────────────────────────────────────
    all_x = np.concatenate([
        xy_raw[:, :, 0].ravel(), xy_raw_ew[:, :, 0].ravel(),
        xy_s1[:, :, 0].ravel(), xy_mod[:, :, 0].ravel(),
    ])
    all_y = np.concatenate([
        xy_raw[:, :, 1].ravel(), xy_raw_ew[:, :, 1].ravel(),
        xy_s1[:, :, 1].ravel(), xy_mod[:, :, 1].ravel(),
    ])
    m = np.isfinite(all_x) & np.isfinite(all_y)
    if not np.any(m):
        raise ValueError("No finite posture points to render")
    xmin, xmax = np.nanmin(all_x[m]), np.nanmax(all_x[m])
    ymin, ymax = np.nanmin(all_y[m]), np.nanmax(all_y[m])
    pad = 2.0
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    span = 0.5 * max((xmax - xmin), (ymax - ymin)) + pad

    # ── figure setup ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), facecolor="white")
    titles = ["Ground truth", "Motor raw decoded", "Motor model decoded"]
    lines = []
    heads = []
    line_raw_ew = None
    head_raw_ew = None

    for i, (ax, ttl) in enumerate(zip(axes, titles)):
        ax.set_title(ttl, fontsize=12, fontweight="bold")
        ax.set_xlim(cx - span, cx + span)
        ax.set_ylim(cy - span, cy + span)
        ax.set_aspect("equal")
        ax.set_facecolor("#f7f7f7")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for s in ax.spines.values():
            s.set_visible(False)
        line, = ax.plot([], [], "k-", lw=2.5, alpha=0.9)
        head, = ax.plot([], [], "o", color="crimson", ms=5, alpha=0.9)
        lines.append(line)
        heads.append(head)
        if i == 0:
            line_raw_ew, = ax.plot([], [], "-", color="darkorange", lw=1.8,
                                   alpha=0.75,
                                   label=f"EW reconstruction ({n_modes} modes)")
            head_raw_ew, = ax.plot([], [], "o", color="darkorange", ms=4,
                                   alpha=0.75)
            ax.legend(handles=[line_raw_ew], loc="upper right", frameon=False,
                      fontsize=8)

    time_text = fig.text(0.5, 0.02, "", ha="center", va="bottom", fontsize=11)

    def _set_trace(line, head, arr_xy, t):
        x = arr_xy[t, :, 0]
        y = arr_xy[t, :, 1]
        line.set_data(x, y)
        if np.isfinite(x[0]) and np.isfinite(y[0]):
            head.set_data([x[0]], [y[0]])
        else:
            head.set_data([], [])

    def _update(frame):
        _set_trace(lines[0], heads[0], xy_raw, frame)
        xo = xy_raw_ew[frame, :, 0]
        yo = xy_raw_ew[frame, :, 1]
        line_raw_ew.set_data(xo, yo)
        if np.isfinite(xo[0]) and np.isfinite(yo[0]):
            head_raw_ew.set_data([xo[0]], [yo[0]])
        else:
            head_raw_ew.set_data([], [])
        _set_trace(lines[1], heads[1], xy_s1, frame)
        _set_trace(lines[2], heads[2], xy_mod, frame)
        time_text.set_text(f"t = {frame * dt:.2f} s    frame {frame + 1}/{T}")
        return lines + heads + [line_raw_ew, head_raw_ew, time_text]

    anim = FuncAnimation(fig, _update, frames=T,
                         interval=max(1, 1000 // max(1, fps)), blit=False)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, metadata={"title": "Worm posture comparison"},
                          bitrate=2400)
    print(f"[posture_video] Rendering {T} frames at {fps} fps → {out_path}")
    anim.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"[posture_video] Done: {out_path}")


# ====================================================================
# CLI
# ====================================================================

def _cli_raw(args):
    out = args.out or (Path(args.h5).stem + "_worm.mp4")
    make_video(args.h5, out, fps=args.fps, dpi=args.dpi,
               normalize_length=args.normalize_length)


def _cli_eigenworm(args):
    out = args.out or (Path(args.h5).stem + "_eigenworm.mp4")
    make_eigenworm_video(args.h5, out, fps=args.fps, dpi=args.dpi,
                         show_original=not args.no_original,
                         n_modes=args.n_modes,
                         eigvec_npy=getattr(args, "eigvec_npy", None))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Worm posture video generation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── raw subcommand ───────────────────────────────────────────────
    p_raw = sub.add_parser("raw",
                           help="Animate raw body-angle posture (curvature coloured).")
    p_raw.add_argument("--h5", required=True, help="Path to HDF5 file")
    p_raw.add_argument("--out", default=None, help="Output MP4 path")
    p_raw.add_argument("--fps", type=int, default=15)
    p_raw.add_argument("--dpi", type=int, default=120)
    p_raw.add_argument("--normalize_length", action="store_true",
                       help="Normalize each frame to fixed worm length")
    p_raw.set_defaults(func=_cli_raw)

    # ── eigenworm subcommand ─────────────────────────────────────────
    p_ew = sub.add_parser("eigenworm",
                          help="Reconstruct posture from eigenworm amplitudes.")
    p_ew.add_argument("--h5", required=True, help="Path to HDF5 file")
    p_ew.add_argument("--out", default=None, help="Output MP4 path")
    p_ew.add_argument("--fps", type=int, default=15)
    p_ew.add_argument("--dpi", type=int, default=120)
    p_ew.add_argument("--n_modes", type=int, default=6,
                      help="Number of eigenworm modes (default: 6)")
    p_ew.add_argument("--no_original", action="store_true",
                      help="Don't overlay the original body-angle trace")
    p_ew.add_argument("--eigvec_npy", default=None,
                      help="Path to eigenvectors_stephens.npy (auto-detected if omitted)")
    p_ew.set_defaults(func=_cli_eigenworm)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
