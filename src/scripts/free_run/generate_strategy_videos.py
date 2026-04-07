"""
Generate videos and comparison plots for all multi-worm behavior strategies.

Strategies:
1. Shared AR Prior - train_shared_beh_prior.py
2. Prior + Fine-tuning - beh_prior_finetune.py  
3. Behavior VAE - train_beh_vae.py
4. Atlas Transformer - atlas_transformer

This script generates posture videos and comparison plots for each strategy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np
import torch
import torch.nn as nn


def load_behavior_from_h5(h5_path: str) -> np.ndarray:
    """Load behavior eigenworms from h5 file."""
    with h5py.File(h5_path, "r") as f:
        beh = f["behaviour"]["eigenworms_calc_6"][:]
    return beh


def generate_ar_samples(
    model_path: str,
    n_samples: int,
    T: int,
    K_beh: int,
    seed_beh: np.ndarray,
    device: str = "cuda",
) -> np.ndarray:
    """Generate samples from the shared AR prior."""
    # Load model
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    
    # Reconstruct model with correct architecture (3-layer with LayerNorm, GELU)
    class BehaviorARModel(nn.Module):
        def __init__(self, Kw: int, K_beh: int, hidden: int = 256, n_layers: int = 3):
            super().__init__()
            self.Kw = Kw
            self.K_beh = K_beh
            
            d_in = K_beh * Kw
            layers, d = [], d_in
            for _ in range(n_layers):
                layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.1)]
                d = hidden
            layers.append(nn.Linear(d, Kw * 2))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            out = self.net(x)
            mu, log_std = out.split(self.Kw, dim=-1)
            log_std = torch.clamp(log_std, -5, 2)
            return mu, log_std

        def sample(self, x, temperature=1.0):
            mu, log_std = self.forward(x)
            std = torch.exp(log_std) * temperature
            return mu + std * torch.randn_like(mu)

    Kw = ckpt["Kw"]
    model = BehaviorARModel(Kw=Kw, K_beh=K_beh, hidden=256, n_layers=3).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    # Get normalization params
    mu_beh = ckpt["mu"]
    sig_beh = ckpt["sig"]
    
    # Normalize seed
    seed_norm = (seed_beh - mu_beh) / sig_beh

    # Generate samples
    samples = np.zeros((n_samples, T, Kw))
    
    with torch.no_grad():
        for i in range(n_samples):
            # Initialize trajectory with normalized seed
            traj = [seed_norm[t] for t in range(K_beh)]  # List of (Kw,) arrays
            
            for t in range(T - K_beh):
                # Get context (last K_beh frames flattened)
                context = np.concatenate(traj[-K_beh:]).flatten()
                context_t = torch.tensor(context, dtype=torch.float32, device=device).unsqueeze(0)
                next_frame = model.sample(context_t, temperature=1.0)  # (1, Kw)
                traj.append(next_frame.cpu().numpy()[0])
            
            # Stack and denormalize
            full_traj = np.stack(traj[:T], axis=0)  # (T, Kw)
            samples[i] = full_traj * sig_beh + mu_beh
    
    return samples


def generate_vae_samples(
    model_path: str,
    n_samples: int,
    T: int,
    K_beh: int,
    seed_beh: np.ndarray,
    device: str = "cuda",
) -> np.ndarray:
    """Generate samples from the behavior VAE (autoregressive generation)."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    
    # Reconstruct VAE model with correct architecture
    class TemporalBehaviorVAE(nn.Module):
        def __init__(self, Kw: int, K_beh: int, latent_dim: int = 32, hidden: int = 256):
            super().__init__()
            self.Kw = Kw
            self.K_beh = K_beh
            self.latent_dim = latent_dim
            d_in = K_beh * Kw
            
            # Encoder
            self.encoder = nn.Sequential(
                nn.Linear(d_in, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
            )
            self.fc_mu = nn.Linear(hidden, latent_dim)
            self.fc_logvar = nn.Linear(hidden, latent_dim)
            
            # Decoder: latent + sequence → next frame
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim + d_in, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Linear(hidden, Kw),
            )

        def encode(self, x):
            h = self.encoder(x)
            return self.fc_mu(h), self.fc_logvar(h)

        def sample(self, x, temperature=1.0):
            mu, logvar = self.encode(x)
            std = torch.exp(0.5 * logvar) * temperature
            z = mu + std * torch.randn_like(std)
            return self.decoder(torch.cat([z, x], dim=-1))

    Kw = ckpt.get("Kw", 6)
    latent_dim = ckpt.get("latent_dim", 32)
    model = TemporalBehaviorVAE(Kw=Kw, K_beh=K_beh, latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Get normalization params
    mu_beh = ckpt["mu"]
    sig_beh = ckpt["sig"]
    
    # Normalize seed
    seed_norm = (seed_beh - mu_beh) / sig_beh

    # Generate samples autoregressively
    samples = np.zeros((n_samples, T, Kw))
    
    with torch.no_grad():
        for i in range(n_samples):
            # Initialize with normalized seed
            trajectory = list(seed_norm[:K_beh])
            
            for t in range(T - K_beh):
                # Get context (last K_beh frames)
                context = np.array(trajectory[-K_beh:]).flatten()
                context_t = torch.tensor(context, dtype=torch.float32, device=device).unsqueeze(0)
                next_frame = model.sample(context_t, temperature=1.0)
                trajectory.append(next_frame.cpu().numpy()[0])
            
            # Denormalize
            traj = np.array(trajectory)[:T]
            samples[i] = traj * sig_beh + mu_beh
    
    return samples


def plot_strategy_comparison(
    results: Dict[str, Dict],
    ground_truth: np.ndarray,
    out_dir: Path,
    worm_id: str,
):
    """Create comparison plots for all strategies."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    # Plot 1: Time series comparison (first eigenworm mode)
    ax = axes[0, 0]
    t = np.arange(min(500, len(ground_truth)))
    ax.plot(t, ground_truth[:len(t), 0], 'k-', alpha=0.8, label='Ground Truth', linewidth=1.5)
    colors = ['C0', 'C1', 'C2', 'C3']
    for i, (name, data) in enumerate(results.items()):
        if 'samples' in data and data['samples'] is not None:
            ax.plot(t, data['samples'][0, :len(t), 0], colors[i], alpha=0.7, label=name, linewidth=1)
    ax.set_xlabel('Time (frames)')
    ax.set_ylabel('Eigenworm 1')
    ax.set_title('Time Series Comparison')
    ax.legend(fontsize=8)
    
    # Plot 2: Variance comparison
    ax = axes[0, 1]
    gt_var = np.var(ground_truth, axis=0)
    x = np.arange(len(gt_var))
    width = 0.15
    ax.bar(x - 2*width, gt_var, width, label='Ground Truth', color='black', alpha=0.7)
    for i, (name, data) in enumerate(results.items()):
        if 'samples' in data and data['samples'] is not None:
            var = np.var(data['samples'], axis=(0, 1))
            ax.bar(x + (i-1)*width, var, width, label=name, color=colors[i], alpha=0.7)
    ax.set_xlabel('Eigenworm Mode')
    ax.set_ylabel('Variance')
    ax.set_title('Variance per Mode')
    ax.legend(fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'EW{i+1}' for i in range(len(gt_var))])
    
    # Plot 3: PSD comparison
    ax = axes[1, 0]
    from scipy import signal
    fs = 15  # ~15 Hz sampling
    f_gt, psd_gt = signal.welch(ground_truth[:, 0], fs=fs, nperseg=256)
    ax.semilogy(f_gt, psd_gt, 'k-', linewidth=2, label='Ground Truth')
    for i, (name, data) in enumerate(results.items()):
        if 'samples' in data and data['samples'] is not None:
            f, psd = signal.welch(data['samples'][0, :, 0], fs=fs, nperseg=min(256, len(data['samples'][0])))
            ax.semilogy(f, psd, colors[i], alpha=0.8, label=name)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD')
    ax.set_title('Power Spectral Density (Mode 1)')
    ax.legend(fontsize=8)
    ax.set_xlim([0, 3])
    
    # Plot 4: Autocorrelation
    ax = axes[1, 1]
    max_lag = 100
    gt_acf = np.correlate(ground_truth[:500, 0] - ground_truth[:500, 0].mean(), 
                          ground_truth[:500, 0] - ground_truth[:500, 0].mean(), mode='full')
    gt_acf = gt_acf[len(gt_acf)//2:len(gt_acf)//2 + max_lag]
    gt_acf /= gt_acf[0]
    ax.plot(np.arange(max_lag), gt_acf, 'k-', linewidth=2, label='Ground Truth')
    for i, (name, data) in enumerate(results.items()):
        if 'samples' in data and data['samples'] is not None:
            sig = data['samples'][0, :500, 0]
            acf = np.correlate(sig - sig.mean(), sig - sig.mean(), mode='full')
            acf = acf[len(acf)//2:len(acf)//2 + max_lag]
            if acf[0] > 0:
                acf /= acf[0]
            ax.plot(np.arange(max_lag), acf, colors[i], alpha=0.8, label=name)
    ax.set_xlabel('Lag (frames)')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Autocorrelation (Mode 1)')
    ax.legend(fontsize=8)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Plot 5: Metrics bar chart
    ax = axes[2, 0]
    metrics_names = ['PSD Distance', 'Variance Ratio']
    x = np.arange(len(metrics_names))
    width = 0.2
    for i, (name, data) in enumerate(results.items()):
        if 'metrics' in data:
            vals = [data['metrics'].get('psd_distance', 0), 
                    data['metrics'].get('variance_ratio', 0)]
            ax.bar(x + i*width, vals, width, label=name, color=colors[i], alpha=0.8)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metrics_names)
    ax.set_ylabel('Value')
    ax.set_title('Quantitative Metrics')
    ax.legend(fontsize=8)
    ax.axhline(1.0, color='green', linestyle='--', alpha=0.5, label='Ideal (variance)')
    
    # Plot 6: Summary table
    ax = axes[2, 1]
    ax.axis('off')
    table_data = [['Strategy', 'PSD Dist', 'Var Ratio', 'Status']]
    for name, data in results.items():
        if 'metrics' in data:
            psd = f"{data['metrics'].get('psd_distance', 'N/A'):.2f}" if isinstance(data['metrics'].get('psd_distance'), float) else 'N/A'
            var = f"{data['metrics'].get('variance_ratio', 'N/A'):.2f}" if isinstance(data['metrics'].get('variance_ratio'), float) else 'N/A'
            status = '✓' if data.get('samples') is not None else '✗'
            table_data.append([name, psd, var, status])
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.35, 0.2, 0.2, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax.set_title('Strategy Summary', pad=20)
    
    plt.suptitle(f'Multi-Worm Strategy Comparison - {worm_id}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / f'strategy_comparison_{worm_id}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved comparison plot: {out_dir / f'strategy_comparison_{worm_id}.png'}")


def angles_to_xy(angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert body angles to x,y coordinates."""
    cumang = np.cumsum(angles)
    x = np.cumsum(np.cos(cumang))
    y = np.cumsum(np.sin(cumang))
    # Center
    x = x - x.mean()
    y = y - y.mean()
    return x, y


def generate_posture_video(
    samples: np.ndarray,
    out_path: Path,
    eigvecs: np.ndarray,
    fps: int = 15,
    n_frames: int = 300,
):
    """Generate posture video from eigenworm samples."""
    # Take first sample, first n_frames
    beh = samples[0, :n_frames, :]
    n_modes = min(beh.shape[1], eigvecs.shape[1])
    
    try:
        # Reconstruct body angles from eigenworms
        angles_recon = beh[:, :n_modes] @ eigvecs[:, :n_modes].T
        
        # Convert to xy coordinates
        all_xy = np.zeros((n_frames, eigvecs.shape[0], 2))
        for t in range(n_frames):
            x, y = angles_to_xy(angles_recon[t])
            all_xy[t, :, 0] = x
            all_xy[t, :, 1] = y
        
        # Create video
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect('equal')
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_title('Generated Posture')
        
        writer = FFMpegWriter(fps=fps)
        with writer.saving(fig, str(out_path), dpi=100):
            for t in range(n_frames):
                ax.clear()
                ax.plot(all_xy[t, :, 0], all_xy[t, :, 1], 'b-', linewidth=2)
                ax.plot(all_xy[t, 0, 0], all_xy[t, 0, 1], 'ro', markersize=8)  # Head
                ax.set_xlim(-50, 50)
                ax.set_ylim(-50, 50)
                ax.set_aspect('equal')
                ax.set_title(f'Frame {t+1}/{n_frames}')
                writer.grab_frame()
        plt.close(fig)
        print(f"  Saved video: {out_path}")
    except Exception as e:
        print(f"  Warning: Could not generate video: {e}")


def load_eigenvectors(h5_path: str) -> np.ndarray:
    """Load eigenvectors from h5 file or default Stephens basis."""
    # Try to load from h5
    try:
        with h5py.File(h5_path, "r") as f:
            if "behaviour/eigenvectors" in f:
                return f["behaviour/eigenvectors"][:]
    except:
        pass
    
    # Use Stephens eigenvectors (100 segments, 6 modes)
    # These are the standard C. elegans eigenworm basis vectors
    # Create approximate basis (sinusoidal approximation)
    n_seg = 100
    n_modes = 6
    eigvecs = np.zeros((n_seg, n_modes))
    x = np.linspace(0, 1, n_seg)
    for i in range(n_modes):
        eigvecs[:, i] = np.sin((i + 1) * np.pi * x)
    # Normalize
    eigvecs /= np.linalg.norm(eigvecs, axis=0, keepdims=True)
    return eigvecs


def main():
    parser = argparse.ArgumentParser(description="Generate videos and plots for all strategies")
    parser.add_argument("--h5", type=str, 
                        default="data/used/behaviour+neuronal activity atanas (2023)/2/2022-06-14-07.h5",
                        help="H5 file for ground truth behavior")
    parser.add_argument("--out_dir", type=str, default="output_plots/free_run/strategy_comparison")
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--n_frames", type=int, default=300)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    worm_id = Path(args.h5).stem
    print(f"Generating comparison for {worm_id}")
    
    # Load ground truth and eigenvectors
    print("Loading ground truth...")
    gt_beh = load_behavior_from_h5(args.h5)
    eigvecs = load_eigenvectors(args.h5)
    T, Kw = gt_beh.shape
    print(f"  Ground truth: T={T}, Kw={Kw}")
    print(f"  Eigenvectors: {eigvecs.shape}")
    
    results = {}
    
    # Strategy 1: Shared AR Prior
    print("\n--- Strategy 1: Shared AR Prior ---")
    prior_path = Path("output_plots/free_run/shared_prior/shared_beh_prior.pt")
    if prior_path.exists():
        try:
            samples = generate_ar_samples(
                str(prior_path), args.n_samples, T, K_beh=30,
                seed_beh=gt_beh[:30], device=args.device
            )
            # Load metrics
            with open(prior_path.parent / "results.json") as f:
                metrics_all = json.load(f)
            metrics = metrics_all.get("metrics", {}).get(worm_id, {})
            results["Shared AR Prior"] = {
                "samples": samples,
                "metrics": {
                    "psd_distance": metrics.get("psd_distance"),
                    "variance_ratio": metrics.get("variance_ratio"),
                }
            }
            print(f"  Generated {args.n_samples} samples")
            # Generate video
            generate_posture_video(samples, out_dir / f"shared_prior_{worm_id}.mp4", eigvecs, n_frames=args.n_frames)
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            results["Shared AR Prior"] = {"samples": None, "metrics": {}}
    else:
        print("  Model not found")
        results["Shared AR Prior"] = {"samples": None, "metrics": {}}
    
    # Strategy 2: Prior + Fine-tuning
    print("\n--- Strategy 2: Prior + Fine-tuning ---")
    finetune_dir = Path("output_plots/free_run/prior_finetune")
    finetune_results = finetune_dir / f"results_{worm_id}.json"
    if finetune_results.exists():
        try:
            with open(finetune_results) as f:
                ft_data = json.load(f)
            
            # Note: Can't generate samples without neural data
            # The decoder needs both neural and behavior context
            results["Prior+Finetune"] = {
                "samples": None,
                "metrics": {
                    "psd_distance": ft_data.get("psd_distance"),
                    "variance_ratio": ft_data.get("variance_ratio"),
                }
            }
            print(f"  Metrics: PSD={ft_data.get('psd_distance'):.2f}, Var={ft_data.get('variance_ratio'):.2f}")
            print(f"  (Video not generated - decoder needs neural input)")
        except Exception as e:
            print(f"  Error: {e}")
            results["Prior+Finetune"] = {"samples": None, "metrics": {}}
    else:
        print("  Results not found (may still be running)")
        results["Prior+Finetune"] = {"samples": None, "metrics": {}}
    
    # Strategy 3: Behavior VAE
    print("\n--- Strategy 3: Behavior VAE ---")
    vae_path = Path("output_plots/free_run/beh_vae/beh_vae.pt")
    if vae_path.exists():
        try:
            samples = generate_vae_samples(
                str(vae_path), args.n_samples, T, K_beh=30,
                seed_beh=gt_beh[:30], device=args.device
            )
            # Load metrics
            with open(vae_path.parent / "results.json") as f:
                metrics_all = json.load(f)
            metrics = metrics_all.get("metrics", {}).get(worm_id, {})
            results["Behavior VAE"] = {
                "samples": samples,
                "metrics": {
                    "psd_distance": metrics.get("psd_distance"),
                    "variance_ratio": metrics.get("variance_ratio"),
                }
            }
            print(f"  Generated {args.n_samples} samples")
            generate_posture_video(samples, out_dir / f"beh_vae_{worm_id}.mp4", eigvecs, n_frames=args.n_frames)
        except Exception as e:
            print(f"  Error: {e}")
            results["Behavior VAE"] = {"samples": None, "metrics": {}}
    else:
        print("  Model not found")
        results["Behavior VAE"] = {"samples": None, "metrics": {}}
    
    # Strategy 4: Atlas Transformer (behavior decoder output)
    print("\n--- Strategy 4: Atlas Transformer ---")
    atlas_dir = Path("output_plots/atlas_transformer/multi_worm_beh") / worm_id
    if atlas_dir.exists():
        try:
            eval_results = atlas_dir / "eval_results.json"
            if eval_results.exists():
                with open(eval_results) as f:
                    atlas_data = json.load(f)
                results["Atlas Transformer"] = {
                    "samples": None,  # Would need to load model and generate
                    "metrics": {
                        "beh_r2_ridge": atlas_data.get("BehRdg", {}).get("mean"),
                        "beh_r2_mlp": atlas_data.get("BehMLP", {}).get("mean"),
                    }
                }
                print(f"  Loaded evaluation results")
        except Exception as e:
            print(f"  Error: {e}")
            results["Atlas Transformer"] = {"samples": None, "metrics": {}}
    else:
        print("  Results not found")
        results["Atlas Transformer"] = {"samples": None, "metrics": {}}
    
    # Generate ground truth video
    print("\n--- Ground Truth ---")
    generate_posture_video(gt_beh[np.newaxis, ...], out_dir / f"ground_truth_{worm_id}.mp4", eigvecs, n_frames=args.n_frames)
    
    # Generate comparison plot
    print("\nGenerating comparison plot...")
    plot_strategy_comparison(results, gt_beh, out_dir, worm_id)
    
    # Save summary
    summary = {
        "worm_id": worm_id,
        "strategies": {
            name: {
                "metrics": data["metrics"],
                "has_samples": data["samples"] is not None
            }
            for name, data in results.items()
        }
    }
    with open(out_dir / f"summary_{worm_id}.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
