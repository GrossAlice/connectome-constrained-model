#!/usr/bin/env python3
"""
Generate a comprehensive comparison video showing all decoder models side by side.

This script creates a single video with:
- Ground truth posture (eigenworm reconstruction)
- All decoder models: TRF Joint, MLP Joint, Ridge Joint, etc.

Usage:
    python -m scripts.free_run.generate_all_models_video \
        --h5 "data/used/behaviour+neuronal activity atanas (2023)/2/2022-06-14-07.h5" \
        --results_dir output_plots/free_run/2022-06-14-07/motor \
        --out_dir output_plots/free_run/comparison_videos \
        --duration 20 --fps 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


# ═══════════════════════════════════════════════════════════════════════════════
# Geometry helpers
# ═══════════════════════════════════════════════════════════════════════════════

def angles_to_xy(angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert tangent angles to centered (x, y) centerline coordinates."""
    x = np.cumsum(np.cos(angles))
    y = np.cumsum(np.sin(angles))
    x -= np.mean(x)
    y -= np.mean(y)
    return x, y


def load_eigenvectors(n_segments: int = 100, n_modes: int = 6) -> np.ndarray:
    """
    Load or create eigenvectors for posture reconstruction.
    Returns eigenvectors of shape (n_segments, n_modes).
    """
    # Try to load from standard locations
    candidates = [
        ROOT / "data" / "used" / "stephens (2016)" / "eigenvectors_stephens.npy",
        ROOT / "data" / "raw" / "stephens (2016)" / "eigenvectors_stephens.npy",
    ]
    
    for p in candidates:
        if p.exists():
            eigvecs = np.load(p)
            if eigvecs.shape[0] != n_segments:
                # Interpolate to match target segments
                old_x = np.linspace(0, 1, eigvecs.shape[0])
                new_x = np.linspace(0, 1, n_segments)
                eigvecs_new = np.zeros((n_segments, eigvecs.shape[1]))
                for i in range(eigvecs.shape[1]):
                    eigvecs_new[:, i] = np.interp(new_x, old_x, eigvecs[:, i])
                eigvecs = eigvecs_new
            return eigvecs[:, :n_modes]
    
    # Generate sinusoidal eigenvectors as fallback (Stephens approximation)
    print("[video] Using sinusoidal eigenvector approximation")
    s = np.linspace(0, 1, n_segments)
    eigvecs = np.zeros((n_segments, n_modes))
    for k in range(n_modes):
        # Stephens modes are approximately sinusoidal
        eigvecs[:, k] = np.sin((k + 1) * np.pi * s) * np.sqrt(2 / n_segments)
    return eigvecs


def eigenworms_to_posture(eigenworms: np.ndarray, eigvecs: np.ndarray) -> np.ndarray:
    """
    Convert eigenworm amplitudes to body posture (x, y) coordinates.
    
    Args:
        eigenworms: (T, Kw) eigenworm amplitudes
        eigvecs: (n_segments, Kw) eigenvector basis
    
    Returns:
        xy: (T, n_segments, 2) posture coordinates
    """
    T, Kw = eigenworms.shape
    n_seg = eigvecs.shape[0]
    
    # Reconstruct angles from eigenworms
    angles = eigenworms @ eigvecs[:, :Kw].T  # (T, n_seg)
    
    # Convert to XY coordinates
    xy = np.zeros((T, n_seg, 2))
    for t in range(T):
        x, y = angles_to_xy(angles[t])
        xy[t, :, 0] = x
        xy[t, :, 1] = y
    
    return xy


# ═══════════════════════════════════════════════════════════════════════════════
# Model running functions (simplified for video generation)
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_models_for_video(h5_path: str, n_frames: int, K: int = 15,
                              K_beh: int = 30, neurons: str = "motor",
                              device: str = "cpu", temperature: float = 1.0):
    """
    Run all decoder models and return generated behavior for video.
    
    Returns:
        dict mapping model_name -> (T, Kw) generated eigenworms
    """
    import torch
    from sklearn.linear_model import Ridge
    
    # Import training functions
    from scripts.free_run.utils import build_lagged, estimate_residual_std
    from baseline_transformer.config import TransformerBaselineConfig
    from baseline_transformer.dataset import load_worm_data
    from baseline_transformer.train import train_single_worm_cv
    
    print(f"[video] Loading data from {h5_path}")
    
    # Load data using the standard loader
    worm_data = load_worm_data(h5_path, n_beh_modes=6)
    u_raw = worm_data["u"].astype(np.float32)
    beh_raw = worm_data["b"].astype(np.float32)
    worm_id = worm_data["worm_id"]
    motor_idx = worm_data.get("motor_idx")
    sr = worm_data.get("sr", 4.0)
    
    # Apply motor mask if requested
    if neurons == "motor" and motor_idx is not None:
        u_raw = u_raw[:, motor_idx]
        print(f"[video] Using {u_raw.shape[1]} motor neurons (from {len(motor_idx)} matched)")
    else:
        print(f"[video] Using all {u_raw.shape[1]} neurons")
    
    T_total, N = u_raw.shape
    Kw = beh_raw.shape[1]
    
    # Use first half for training, generate from second half seed
    train_end = T_total // 2
    test_start = train_end
    
    u_train = u_raw[:train_end]
    beh_train = beh_raw[:train_end]
    
    # Normalize
    u_mu, u_sig = u_train.mean(0), u_train.std(0) + 1e-8
    beh_mu, beh_sig = beh_train.mean(0), beh_train.std(0) + 1e-8
    
    u_train_n = (u_train - u_mu) / u_sig
    beh_train_n = (beh_train - beh_mu) / beh_sig
    u_test_n = (u_raw[test_start:] - u_mu) / u_sig
    beh_test_n = (beh_raw[test_start:] - beh_mu) / beh_sig
    
    # Ground truth (test portion)
    gt_beh = beh_raw[test_start:test_start + n_frames]
    
    results = {"Ground Truth": gt_beh}
    
    # Seed for generation
    seed_start = test_start - K
    joint_seed = np.concatenate([u_raw[seed_start:test_start], beh_raw[seed_start:test_start]], axis=1)
    joint_seed_n = np.concatenate([(u_raw[seed_start:test_start] - u_mu) / u_sig,
                                    (beh_raw[seed_start:test_start] - beh_mu) / beh_sig], axis=1)
    
    # ─── Train and run models ───────────────────────────────────────────────────
    
    # 1. Ridge Joint (fast and stable for motor neurons)
    print("[video] Training Ridge Joint...")
    joint_train_n = np.concatenate([u_train_n, beh_train_n], axis=1)
    X_joint = build_lagged(joint_train_n, K)
    Y_joint = joint_train_n[K:]  # Target is next time step
    X_joint = X_joint[K:]  # Align with Y
    ridge_joint = Ridge(alpha=1.0).fit(X_joint, Y_joint)
    res_std_joint = estimate_residual_std(ridge_joint, X_joint, Y_joint, is_torch=False)
    
    # Run Ridge Joint
    D = N + Kw
    hist = [joint_seed_n[K - 1 - lag].copy() for lag in range(K)]
    pred_ridge = np.zeros((n_frames, D), dtype=np.float32)
    for t in range(n_frames):
        x_lags = np.concatenate(hist)
        mu = ridge_joint.predict(x_lags.reshape(1, -1))[0]
        noise = temperature * res_std_joint * np.random.randn(D).astype(np.float32)
        sample = mu + noise
        pred_ridge[t] = sample
        hist = [sample.copy()] + hist[:-1]
    
    pred_ridge_beh = pred_ridge[:, N:] * beh_sig + beh_mu
    results["Ridge Joint"] = pred_ridge_beh
    print(f"  → Ridge Joint done")
    
    # 2. MLP Joint
    print("[video] Training MLP Joint...")
    import torch.nn as nn
    
    class SimpleMLP(nn.Module):
        def __init__(self, in_dim, out_dim, hidden=256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, out_dim),
            )
        def forward(self, x):
            return self.net(x)
    
    mlp_joint = SimpleMLP(K * D, D)
    mlp_joint = mlp_joint.to(device)
    
    X_mlp = build_lagged(joint_train_n, K)[K:]
    Y_mlp = joint_train_n[K:]
    X_t = torch.tensor(X_mlp, dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y_mlp, dtype=torch.float32, device=device)
    
    opt = torch.optim.AdamW(mlp_joint.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(100):
        mlp_joint.train()
        opt.zero_grad()
        pred = mlp_joint(X_t)
        loss = ((pred - Y_t) ** 2).mean()
        loss.backward()
        opt.step()
    
    mlp_joint.eval()
    with torch.no_grad():
        res = (mlp_joint(X_t) - Y_t).cpu().numpy()
    res_std_mlp = res.std(axis=0).astype(np.float32)
    
    # Run MLP Joint
    hist = [joint_seed_n[K - 1 - lag].copy() for lag in range(K)]
    pred_mlp = np.zeros((n_frames, D), dtype=np.float32)
    mlp_joint.eval()
    with torch.no_grad():
        for t in range(n_frames):
            x_lags = np.concatenate(hist)
            x_t = torch.tensor(x_lags, dtype=torch.float32, device=device).unsqueeze(0)
            mu = mlp_joint(x_t)[0].cpu().numpy()
            noise = temperature * res_std_mlp * np.random.randn(D).astype(np.float32)
            sample = mu + noise
            pred_mlp[t] = sample
            hist = [sample.copy()] + hist[:-1]
    
    pred_mlp_beh = pred_mlp[:, N:] * beh_sig + beh_mu
    results["MLP Joint"] = pred_mlp_beh
    print(f"  → MLP Joint done")
    
    # 3. TRF Joint (Transformer) - Use MLP with more layers as approximation
    print("[video] Training TRF-like model (MLP with attention-like capacity)...")
    try:
        class TransformerLikeMLP(nn.Module):
            """MLP with more capacity to approximate transformer behavior."""
            def __init__(self, in_dim, out_dim, hidden=512):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, hidden),
                    nn.LayerNorm(hidden),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden, hidden),
                    nn.LayerNorm(hidden),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden, hidden // 2),
                    nn.LayerNorm(hidden // 2),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden // 2, out_dim),
                )
            def forward(self, x):
                return self.net(x)
        
        trf_like = TransformerLikeMLP(K * D, D, hidden=512)
        trf_like = trf_like.to(device)
        
        X_trf = build_lagged(joint_train_n, K)[K:]
        Y_trf = joint_train_n[K:]
        X_trf_t = torch.tensor(X_trf, dtype=torch.float32, device=device)
        Y_trf_t = torch.tensor(Y_trf, dtype=torch.float32, device=device)
        
        opt_trf = torch.optim.AdamW(trf_like.parameters(), lr=1e-3, weight_decay=1e-4)
        for epoch in range(150):
            trf_like.train()
            opt_trf.zero_grad()
            pred = trf_like(X_trf_t)
            loss = ((pred - Y_trf_t) ** 2).mean()
            loss.backward()
            opt_trf.step()
        
        trf_like.eval()
        with torch.no_grad():
            res_trf = (trf_like(X_trf_t) - Y_trf_t).cpu().numpy()
        res_std_trf = res_trf.std(axis=0).astype(np.float32)
        
        # Run TRF-like model
        hist = [joint_seed_n[K - 1 - lag].copy() for lag in range(K)]
        pred_trf = np.zeros((n_frames, D), dtype=np.float32)
        trf_like.eval()
        with torch.no_grad():
            for t in range(n_frames):
                x_lags = np.concatenate(hist)
                x_t = torch.tensor(x_lags, dtype=torch.float32, device=device).unsqueeze(0)
                mu = trf_like(x_t)[0].cpu().numpy()
                noise = temperature * res_std_trf * np.random.randn(D).astype(np.float32)
                sample = mu + noise
                pred_trf[t] = sample
                hist = [sample.copy()] + hist[:-1]
        
        pred_trf_beh = pred_trf[:, N:] * beh_sig + beh_mu
        results["TRF-like"] = pred_trf_beh
        trf_like.cpu()
        print(f"  → TRF-like done")
    except Exception as e:
        print(f"  ✗ TRF Joint failed: {e}")
    
    # 4. Ridge Cascaded (Neural AR + Beh from neural)
    print("[video] Training Ridge Cascaded...")
    
    # Neural AR
    X_u = build_lagged(u_train_n, K)[K:]
    Y_u = u_train_n[K:]
    ridge_neural = Ridge(alpha=1.0).fit(X_u, Y_u)
    res_std_u = estimate_residual_std(ridge_neural, X_u, Y_u, is_torch=False)
    
    # Beh decoder from neural
    X_dec = build_lagged(u_train_n, K)[K:]
    Y_dec = beh_train_n[K:]
    ridge_dec = Ridge(alpha=1.0).fit(X_dec, Y_dec)
    res_std_dec = estimate_residual_std(ridge_dec, X_dec, Y_dec, is_torch=False)
    
    # Run cascaded
    u_seed_n = (u_raw[seed_start:test_start] - u_mu) / u_sig
    neural_hist = [u_seed_n[K - 1 - lag].copy() for lag in range(K)]
    pred_neural = np.zeros((n_frames, N), dtype=np.float32)
    pred_beh_casc = np.zeros((n_frames, Kw), dtype=np.float32)
    
    for t in range(n_frames):
        u_lags = np.concatenate(neural_hist)
        mu_u = ridge_neural.predict(u_lags.reshape(1, -1))[0]
        u_sample = mu_u + temperature * res_std_u * np.random.randn(N).astype(np.float32)
        pred_neural[t] = u_sample
        
        # Decode behavior
        u_dec_lags = np.concatenate([u_sample] + neural_hist[:-1])
        mu_b = ridge_dec.predict(u_dec_lags.reshape(1, -1))[0]
        b_sample = mu_b + temperature * res_std_dec * np.random.randn(Kw).astype(np.float32)
        pred_beh_casc[t] = b_sample
        
        neural_hist = [u_sample.copy()] + neural_hist[:-1]
    
    pred_casc_beh = pred_beh_casc * beh_sig + beh_mu
    results["Ridge Cascaded"] = pred_casc_beh
    print(f"  → Ridge Cascaded done")
    
    return results, sr


# ═══════════════════════════════════════════════════════════════════════════════
# Video generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_comparison_video(
    model_results: dict,
    out_path: str,
    eigvecs: np.ndarray,
    fps: int = 5,
    dpi: int = 100,
    dt: float = 0.25,
):
    """
    Generate a comparison video showing all models side by side.
    
    Args:
        model_results: dict mapping model_name -> (T, Kw) eigenworms
        out_path: output video path
        eigvecs: (n_segments, Kw) eigenvector basis
        fps: frames per second (lower = slower)
        dpi: output resolution
        dt: time step between frames
    """
    model_names = list(model_results.keys())
    n_models = len(model_names)
    
    # Determine grid layout
    if n_models <= 3:
        n_cols = n_models
        n_rows = 1
    elif n_models <= 6:
        n_cols = 3
        n_rows = 2
    else:
        n_cols = 4
        n_rows = (n_models + 3) // 4
    
    # Get number of frames (minimum across all models)
    T = min(arr.shape[0] for arr in model_results.values())
    Kw = next(iter(model_results.values())).shape[1]
    
    print(f"[video] Generating {T} frames, {n_models} models, {n_cols}x{n_rows} grid")
    
    # Convert all eigenworms to XY postures
    xy_data = {}
    for name, ew in model_results.items():
        xy_data[name] = eigenworms_to_posture(ew[:T], eigvecs)
    
    # Compute axis limits from all data
    all_xy = np.concatenate([xy[:, :, :] for xy in xy_data.values()], axis=0)
    xmin, xmax = np.nanmin(all_xy[:, :, 0]), np.nanmax(all_xy[:, :, 0])
    ymin, ymax = np.nanmin(all_xy[:, :, 1]), np.nanmax(all_xy[:, :, 1])
    pad = 2.0
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    span = 0.5 * max(xmax - xmin, ymax - ymin) + pad
    
    # Create figure
    fig_width = 4 * n_cols
    fig_height = 4 * n_rows + 0.5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), 
                             facecolor="white", squeeze=False)
    fig.subplots_adjust(hspace=0.15, wspace=0.1, top=0.92, bottom=0.08)
    
    # Setup each axis
    lines = []
    heads = []
    titles = []
    
    for i, name in enumerate(model_names):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Style based on model type
        if name == "Ground Truth":
            color = "#1a1a1a"
            title_color = "#1a1a1a"
        elif "TRF" in name:
            color = "#e41a1c"  # red
            title_color = "#e41a1c"
        elif "MLP" in name:
            color = "#377eb8"  # blue
            title_color = "#377eb8"
        elif "Ridge" in name:
            color = "#4daf4a"  # green
            title_color = "#4daf4a"
        else:
            color = "#984ea3"  # purple
            title_color = "#984ea3"
        
        title = ax.set_title(name, fontsize=11, fontweight="bold", color=title_color)
        titles.append(title)
        
        ax.set_xlim(cx - span, cx + span)
        ax.set_ylim(cy - span, cy + span)
        ax.set_aspect("equal")
        ax.set_facecolor("#f9f9f9")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        line, = ax.plot([], [], "-", color=color, lw=3.0, alpha=0.9, solid_capstyle="round")
        head, = ax.plot([], [], "o", color="crimson", ms=6, zorder=10)
        lines.append(line)
        heads.append(head)
    
    # Hide unused axes
    for i in range(n_models, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    # Time text
    time_text = fig.text(0.5, 0.02, "", ha="center", va="bottom", fontsize=12)
    
    # Suptitle
    fig.suptitle("Behavior Decoder Comparison", fontsize=14, fontweight="bold", y=0.97)
    
    def _update(frame):
        for i, name in enumerate(model_names):
            xy = xy_data[name]
            x = xy[frame, :, 0]
            y = xy[frame, :, 1]
            lines[i].set_data(x, y)
            if np.isfinite(x[0]) and np.isfinite(y[0]):
                heads[i].set_data([x[0]], [y[0]])
            else:
                heads[i].set_data([], [])
        
        time_s = frame * dt
        time_text.set_text(f"t = {time_s:.1f} s   (frame {frame + 1}/{T})")
        return lines + heads + [time_text]
    
    # Create animation
    anim = FuncAnimation(fig, _update, frames=T, interval=1000 // fps, blit=False)
    
    # Save video
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, metadata={"title": "Behavior Decoder Comparison"}, bitrate=3000)
    print(f"[video] Rendering {T} frames at {fps} fps → {out_path}")
    anim.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"[video] Done: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate all-models comparison video")
    parser.add_argument("--h5", required=True, help="Path to HDF5 worm data file")
    parser.add_argument("--out_dir", default="output_plots/free_run/comparison_videos",
                        help="Output directory")
    parser.add_argument("--duration", type=float, default=20.0,
                        help="Video duration in seconds (default: 20)")
    parser.add_argument("--fps", type=int, default=5,
                        help="Frames per second (lower = slower, default: 5)")
    parser.add_argument("--neurons", choices=["motor", "all"], default="motor",
                        help="Which neurons to use (default: motor)")
    parser.add_argument("--K", type=int, default=15, help="Neural context length")
    parser.add_argument("--K_beh", type=int, default=30, help="Behavior context length")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--dpi", type=int, default=100, help="Video DPI")
    
    args = parser.parse_args()
    
    # Get worm ID
    h5_path = Path(args.h5)
    worm_id = h5_path.stem
    
    # Calculate number of frames
    # Sample rate is typically 4 Hz
    sr = 4.0  # Will be updated from h5 file
    n_frames = int(args.duration * sr)
    
    print(f"\n{'='*60}")
    print(f"Generating comparison video for {worm_id}")
    print(f"Duration: {args.duration}s, FPS: {args.fps}, Neurons: {args.neurons}")
    print(f"{'='*60}\n")
    
    # Run all models
    model_results, sr = run_all_models_for_video(
        args.h5, 
        n_frames=int(args.duration * sr),
        K=args.K,
        K_beh=args.K_beh,
        neurons=args.neurons,
        device=args.device,
        temperature=args.temperature
    )
    
    # Load eigenvectors
    Kw = next(iter(model_results.values())).shape[1]
    eigvecs = load_eigenvectors(n_segments=100, n_modes=Kw)
    
    # Generate video
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{worm_id}_{args.neurons}_comparison.mp4"
    
    generate_comparison_video(
        model_results,
        str(out_path),
        eigvecs,
        fps=args.fps,
        dpi=args.dpi,
        dt=1.0 / sr,
    )
    
    print(f"\n✓ Video saved to: {out_path}")


if __name__ == "__main__":
    main()
