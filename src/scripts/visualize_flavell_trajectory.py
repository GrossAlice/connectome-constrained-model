#!/usr/bin/env python
"""
Visualize a Flavell worm: animate body posture along the (x,y) trajectory.

Usage:
    python -m scripts.visualize_flavell_trajectory \
        --h5 /home/agross/Downloads/flavell/2022-08-02-01-data.h5 \
        --out output_plots/flavell_trajectory.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.ndimage import uniform_filter1d


def load_data(h5_path: str):
    with h5py.File(h5_path, "r") as f:
        body_angle = f["behavior/body_angle"][:]        # (T, 30) tangent angles per segment (rad)
        x = f["behavior/zeroed_x_confocal"][:]          # (T,)
        y = f["behavior/zeroed_y_confocal"][:]           # (T,)
        vel = f["behavior/velocity"][:]                  # (T,)
        head_angle = f["behavior/head_angle"][:]         # (T,)
        ts = f["timing/timestamp_confocal"][:]           # (T,)
    return body_angle, x, y, vel, head_angle, ts


def eigenworm_filter(body_angle: np.ndarray, n_modes: int = 6) -> np.ndarray:
    """Project body_angle onto top-k PCA modes (eigenworms) and reconstruct."""
    # body_angle: (T, n_seg)
    mean_ba = body_angle.mean(axis=0)
    centered = body_angle - mean_ba
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    # Keep top n_modes
    recon = U[:, :n_modes] @ np.diag(S[:n_modes]) @ Vt[:n_modes, :]
    var_explained = np.sum(S[:n_modes]**2) / np.sum(S**2)
    print(f"  Eigenworm filter: {n_modes} modes explain {var_explained:.1%} of variance")
    return recon + mean_ba


def compute_heading(x: np.ndarray, y: np.ndarray, smooth: int = 5) -> np.ndarray:
    """Compute smoothed heading angle from the trajectory."""
    xs = uniform_filter1d(x, smooth, mode="nearest")
    ys = uniform_filter1d(y, smooth, mode="nearest")
    dx = np.gradient(xs)
    dy = np.gradient(ys)
    heading = np.arctan2(dy, dx)
    return heading


def reconstruct_worm(body_angle_t: np.ndarray, heading_t: float,
                     cx: float, cy: float, seg_len: float):
    """
    Reconstruct worm body coordinates for one frame.

    body_angle_t contains the tangent angle of each segment relative to
    the head direction.  We add the trajectory heading to obtain absolute
    angles and lay out segments from the head position.

    Parameters
    ----------
    body_angle_t : (n_seg,) tangent angle of each segment relative to head (rad)
    heading_t : absolute heading of the head segment (from trajectory)
    cx, cy : head position in stage coordinates
    seg_len : length of each segment in stage-coordinate units

    Returns
    -------
    wx, wy : (n_seg+1,) arrays of worm body (x,y) from head to tail
    """
    n_seg = len(body_angle_t)
    # body_angle values are tangent angles at each segment, relative to head.
    # Absolute angle of each segment = heading + body_angle[i]   (NO cumsum)
    abs_angles = heading_t + body_angle_t

    # Build skeleton: head at (cx, cy), extending toward tail
    wx = np.empty(n_seg + 1)
    wy = np.empty(n_seg + 1)
    wx[0] = cx
    wy[0] = cy
    for i in range(n_seg):
        wx[i + 1] = wx[i] + seg_len * np.cos(abs_angles[i])
        wy[i + 1] = wy[i] + seg_len * np.sin(abs_angles[i])
    return wx, wy


def make_video(h5_path: str, out_path: str, fps: int = 15,
               max_frames: int = 0, dpi: int = 100, skip: int = 1,
               n_modes: int = 6):
    body_angle, x, y, vel, head_angle, ts = load_data(h5_path)
    T, n_seg = body_angle.shape

    # Eigenworm PCA filter
    if n_modes > 0:
        body_angle = eigenworm_filter(body_angle, n_modes=n_modes)

    heading = compute_heading(x, y, smooth=7)

    if max_frames > 0:
        T = min(T, max_frames)

    # Scale segment length so worm body is visible on the trajectory
    traj_span = max(x.max() - x.min(), y.max() - y.min())
    seg_len = traj_span / (20 * n_seg)  # body ≈ 1/20 of trajectory span

    frame_indices = list(range(0, T, skip))

    # --- Set up figure ---
    fig, (ax_traj, ax_worm) = plt.subplots(1, 2, figsize=(14, 6))

    # Trajectory panel
    pad = traj_span * 0.05
    ax_traj.set_xlim(x[:T].min() - pad, x[:T].max() + pad)
    ax_traj.set_ylim(y[:T].min() - pad, y[:T].max() + pad)
    ax_traj.set_aspect("equal")
    ax_traj.set_xlabel("x (stage coords)")
    ax_traj.set_ylabel("y (stage coords)")
    ax_traj.set_title("Trajectory + worm posture")

    # Plot faint full trajectory
    ax_traj.plot(x[:T], y[:T], color="lightgray", lw=0.5, zorder=0)

    # Animated elements on trajectory panel
    trail_line, = ax_traj.plot([], [], color="steelblue", lw=1, alpha=0.6)
    worm_line, = ax_traj.plot([], [], color="orangered", lw=2.5, solid_capstyle="round")
    head_dot, = ax_traj.plot([], [], "o", color="red", ms=4, zorder=5)
    time_text = ax_traj.text(0.02, 0.97, "", transform=ax_traj.transAxes,
                             fontsize=10, va="top",
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # Zoomed worm panel (local body shape)
    zoom_radius = seg_len * (n_seg + 1) * 0.7
    ax_worm.set_aspect("equal")
    ax_worm.set_title("Worm body (zoom)")
    worm_zoom_line, = ax_worm.plot([], [], color="orangered", lw=3, solid_capstyle="round")
    head_zoom_dot, = ax_worm.plot([], [], "o", color="red", ms=6, zorder=5)
    tail_zoom_dot, = ax_worm.plot([], [], "s", color="blue", ms=5, zorder=5)

    # Velocity color bar idea: color trail by velocity
    vel_text = ax_worm.text(0.02, 0.97, "", transform=ax_worm.transAxes,
                            fontsize=10, va="top",
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    worm_name = Path(h5_path).stem
    fig.suptitle(f"Flavell worm: {worm_name}", fontsize=12, fontweight="bold")

    def init():
        trail_line.set_data([], [])
        worm_line.set_data([], [])
        head_dot.set_data([], [])
        worm_zoom_line.set_data([], [])
        head_zoom_dot.set_data([], [])
        tail_zoom_dot.set_data([], [])
        time_text.set_text("")
        vel_text.set_text("")
        return (trail_line, worm_line, head_dot, worm_zoom_line,
                head_zoom_dot, tail_zoom_dot, time_text, vel_text)

    def update(frame_idx):
        t = frame_indices[frame_idx]

        # Reconstruct worm
        wx, wy = reconstruct_worm(body_angle[t], heading[t], x[t], y[t], seg_len)

        # Trail (past trajectory)
        trail_start = max(0, t - 200)
        trail_line.set_data(x[trail_start:t+1], y[trail_start:t+1])

        # Worm on trajectory
        worm_line.set_data(wx, wy)
        head_dot.set_data([x[t]], [y[t]])

        # Time label
        elapsed = ts[t] - ts[0]
        time_text.set_text(f"t = {elapsed:.1f} s  (frame {t})")

        # Zoomed worm panel
        mx, my = wx.mean(), wy.mean()
        ax_worm.set_xlim(mx - zoom_radius, mx + zoom_radius)
        ax_worm.set_ylim(my - zoom_radius, my + zoom_radius)
        worm_zoom_line.set_data(wx, wy)
        head_zoom_dot.set_data([wx[0]], [wy[0]])
        tail_zoom_dot.set_data([wx[-1]], [wy[-1]])

        vel_text.set_text(f"velocity = {vel[t]:.4f}\nhead angle = {head_angle[t]:.2f} rad")

        return (trail_line, worm_line, head_dot, worm_zoom_line,
                head_zoom_dot, tail_zoom_dot, time_text, vel_text)

    n_frames = len(frame_indices)
    print(f"Rendering {n_frames} frames at {fps} fps → {n_frames/fps:.1f}s video ...")
    anim = animation.FuncAnimation(fig, update, init_func=init,
                                   frames=n_frames, interval=1000 // fps, blit=False)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    writer = animation.FFMpegWriter(fps=fps, bitrate=2000,
                                    extra_args=["-pix_fmt", "yuv420p"])
    anim.save(str(out), writer=writer, dpi=dpi)
    print(f"Saved → {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Animate Flavell worm trajectory + posture")
    parser.add_argument("--h5", required=True, help="Path to Flavell .h5 file")
    parser.add_argument("--out", default="output_plots/flavell_trajectory.mp4",
                        help="Output video path")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Limit number of frames (0 = all)")
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--skip", type=int, default=1,
                        help="Render every N-th frame (1=all, 2=half, ...)")
    parser.add_argument("--modes", type=int, default=6,
                        help="Number of eigenworm PCA modes (0=raw)")
    args = parser.parse_args()
    make_video(args.h5, args.out, fps=args.fps, max_frames=args.max_frames,
               dpi=args.dpi, skip=args.skip, n_modes=args.modes)


if __name__ == "__main__":
    main()
