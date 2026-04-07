"""
Train a shared behavior dynamics prior on ALL worms.

The key insight: eigenworm modes are standardized across worms (Stephens et al.),
so behavior dynamics should be similar. By pooling data from all worms, we get:
1. More training data for oscillation patterns
2. Better generalization of locomotion dynamics
3. A "behavior prior" that can regularize per-worm generation

Usage:
    python -m scripts.free_run.train_shared_beh_prior --data_dir "data/used/behaviour+neuronal activity atanas (2023)/2" --out_dir output_plots/free_run/shared_prior
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from scipy import signal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.free_run.utils import build_lagged, _make_mlp, compute_psd, NumpyEncoder


def load_all_behavior(data_dir: str, max_worms: int = None) -> tuple[np.ndarray, list[str]]:
    """Load and concatenate behavior from all worms."""
    data_path = Path(data_dir)
    h5_files = sorted(data_path.glob("*.h5"))
    
    if max_worms:
        h5_files = h5_files[:max_worms]
    
    all_beh = []
    worm_ids = []
    
    for h5_path in h5_files:
        try:
            with h5py.File(h5_path, "r") as f:
                # Try different possible paths for behavior eigenworms
                beh = None
                # First try nested paths
                if "behaviour" in f and "eigenworms_calc_6" in f["behaviour"]:
                    beh = f["behaviour"]["eigenworms_calc_6"][:]
                elif "behaviour" in f and "eigenworms_stephens" in f["behaviour"]:
                    beh = f["behaviour"]["eigenworms_stephens"][:]
                elif "beh" in f:
                    beh = f["beh"][:]
                
                if beh is None:
                    print(f"  Skipping {h5_path.name}: no behavior data found")
                    continue
                
                # Ensure 2D
                if beh.ndim == 1:
                    beh = beh.reshape(-1, 1)
                
                # Skip if too short
                if len(beh) < 100:
                    print(f"  Skipping {h5_path.name}: too short ({len(beh)} frames)")
                    continue
                
                all_beh.append(beh.astype(np.float32))
                worm_ids.append(h5_path.stem)
                print(f"  Loaded {h5_path.name}: T={len(beh)}, Kw={beh.shape[1]}")
        except Exception as e:
            print(f"  Error loading {h5_path.name}: {e}")
    
    return all_beh, worm_ids


def normalize_behavior(beh_list: list[np.ndarray]) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Normalize behavior across all worms using global statistics."""
    # Compute global mean/std
    all_concat = np.concatenate(beh_list, axis=0)
    mu = all_concat.mean(axis=0)
    sig = all_concat.std(axis=0) + 1e-8
    
    # Normalize each worm
    normed = [(b - mu) / sig for b in beh_list]
    return normed, mu, sig


class BehaviorARModel(nn.Module):
    """Autoregressive behavior model with multi-step rollout capability."""
    
    def __init__(self, Kw: int, K_beh: int, hidden: int = 256, n_layers: int = 3):
        super().__init__()
        self.Kw = Kw
        self.K_beh = K_beh
        
        d_in = K_beh * Kw
        layers, d = [], d_in
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.1)]
            d = hidden
        # Output mean and log_std
        layers.append(nn.Linear(d, Kw * 2))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        """x: (B, K_beh * Kw) -> (mu, log_std) each (B, Kw)"""
        out = self.net(x)
        mu, log_std = out.split(self.Kw, dim=-1)
        # Clamp log_std for stability
        log_std = torch.clamp(log_std, -5, 2)
        return mu, log_std
    
    def sample(self, x, temperature=1.0):
        """Sample from predicted distribution."""
        mu, log_std = self.forward(x)
        std = torch.exp(log_std) * temperature
        return mu + std * torch.randn_like(mu)


def train_shared_ar_rollout(beh_list: list[np.ndarray], K_beh: int, device: str,
                            rollout_steps: int = 30, epochs: int = 300,
                            lr: float = 1e-3, wd: float = 1e-4,
                            patience: int = 40, batch_size: int = 64) -> BehaviorARModel:
    """Train shared AR model with multi-step rollout on all worms."""
    Kw = beh_list[0].shape[1]
    
    # Build training segments from all worms
    # Each segment: (K_beh + rollout_steps) contiguous frames
    seg_len = K_beh + rollout_steps
    segments = []
    
    for beh in beh_list:
        T = len(beh)
        for start in range(0, T - seg_len, rollout_steps // 2):  # 50% overlap
            segments.append(beh[start:start + seg_len])
    
    segments = np.array(segments, dtype=np.float32)  # (N_seg, seg_len, Kw)
    np.random.shuffle(segments)
    
    # Train/val split
    n_val = max(50, int(len(segments) * 0.15))
    train_segs = torch.tensor(segments[:-n_val], device=device)
    val_segs = torch.tensor(segments[-n_val:], device=device)
    
    print(f"  Training segments: {len(train_segs)}, Validation: {len(val_segs)}")
    
    model = BehaviorARModel(Kw, K_beh).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=1e-5)
    
    best_val_loss, best_state, patience_counter = float("inf"), None, 0
    
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(train_segs))
        total_loss = 0.0
        n_batches = 0
        
        for i in range(0, len(train_segs), batch_size):
            batch_idx = perm[i:i + batch_size]
            batch = train_segs[batch_idx]  # (B, seg_len, Kw)
            
            # Rollout training
            loss = 0.0
            beh_history = batch[:, :K_beh].clone()  # (B, K_beh, Kw)
            
            for r in range(rollout_steps):
                inp = beh_history.reshape(batch.shape[0], -1)  # (B, K_beh * Kw)
                mu, log_std = model(inp)
                target = batch[:, K_beh + r]  # (B, Kw)
                
                # Gaussian NLL loss
                std = torch.exp(log_std)
                nll = 0.5 * (((target - mu) / std) ** 2 + 2 * log_std + np.log(2 * np.pi))
                loss = loss + nll.mean()
                
                # Update history with prediction (teacher forcing with some probability)
                if np.random.random() < 0.3:  # 30% teacher forcing
                    next_beh = target
                else:
                    next_beh = model.sample(inp, temperature=0.8)
                beh_history = torch.cat([beh_history[:, 1:], next_beh.unsqueeze(1)], dim=1)
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        
        # Validation (1-step NLL)
        model.eval()
        with torch.no_grad():
            val_inp = val_segs[:, :K_beh].reshape(len(val_segs), -1)
            val_target = val_segs[:, K_beh]
            mu, log_std = model(val_inp)
            std = torch.exp(log_std)
            val_nll = 0.5 * (((val_target - mu) / std) ** 2 + 2 * log_std + np.log(2 * np.pi))
            val_loss = val_nll.mean().item()
        
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (ep + 1) % 20 == 0:
            print(f"  Epoch {ep+1}: train_loss={total_loss/n_batches:.4f}, val_nll={val_loss:.4f}")
        
        if patience_counter > patience:
            print(f"  Early stopping at epoch {ep+1}")
            break
    
    if best_state:
        model.load_state_dict(best_state)
    model.eval().cpu()
    return model


def generate_samples(model: BehaviorARModel, seed: np.ndarray, n_steps: int,
                     n_samples: int = 20, temperature: float = 1.0,
                     device: str = "cpu") -> np.ndarray:
    """Generate behavior samples from the AR model."""
    model = model.to(device).eval()
    K_beh = model.K_beh
    Kw = model.Kw
    
    samples = np.zeros((n_samples, n_steps, Kw), dtype=np.float32)
    
    with torch.no_grad():
        for s in range(n_samples):
            history = torch.tensor(seed[-K_beh:].copy(), dtype=torch.float32, device=device)
            
            for t in range(n_steps):
                inp = history.reshape(1, -1)
                sample = model.sample(inp, temperature=temperature)
                samples[s, t] = sample[0].cpu().numpy()
                history = torch.cat([history[1:], sample], dim=0)
    
    model.cpu()
    return samples


def plot_samples(gt: np.ndarray, samples: np.ndarray, out_path: Path, title: str = ""):
    """Plot ground truth vs generated samples."""
    n_steps = min(500, samples.shape[1], len(gt))
    Kw = min(4, gt.shape[1])  # Plot first 4 modes
    
    fig, axes = plt.subplots(Kw, 1, figsize=(14, 3 * Kw), sharex=True)
    if Kw == 1:
        axes = [axes]
    
    t = np.arange(n_steps)
    
    for d, ax in enumerate(axes):
        ax.plot(t, gt[:n_steps, d], 'k-', lw=1.5, label='Ground Truth', alpha=0.8)
        
        # Plot sample ensemble
        median = np.median(samples[:, :n_steps, d], axis=0)
        q25 = np.percentile(samples[:, :n_steps, d], 25, axis=0)
        q75 = np.percentile(samples[:, :n_steps, d], 75, axis=0)
        
        ax.fill_between(t, q25, q75, alpha=0.3, color='C0', label='IQR')
        ax.plot(t, median, 'C0-', lw=1, label='Median')
        ax.plot(t, samples[0, :n_steps, d], 'C0--', lw=0.5, alpha=0.5, label='Sample')
        
        ax.set_ylabel(f'Mode {d+1}')
        if d == 0:
            ax.legend(loc='upper right')
    
    axes[-1].set_xlabel('Time (frames)')
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_psd_comparison(gt: np.ndarray, samples: np.ndarray, out_path: Path):
    """Compare PSD of ground truth vs generated."""
    Kw = min(4, gt.shape[1])
    
    fig, axes = plt.subplots(1, Kw, figsize=(4 * Kw, 3))
    if Kw == 1:
        axes = [axes]
    
    fs = 1 / 0.6  # ~1.67 Hz sampling
    
    for d, ax in enumerate(axes):
        # GT PSD
        f_gt, psd_gt = signal.welch(gt[:, d], fs=fs, nperseg=min(256, len(gt) // 2))
        ax.semilogy(f_gt, psd_gt, 'k-', lw=2, label='GT')
        
        # Sample PSDs
        for s in range(min(5, len(samples))):
            f_s, psd_s = signal.welch(samples[s, :, d], fs=fs, nperseg=min(256, samples.shape[1] // 2))
            ax.semilogy(f_s, psd_s, 'C0-', alpha=0.3, lw=0.5)
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD')
        ax.set_title(f'Mode {d+1}')
        if d == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Directory with worm h5 files")
    ap.add_argument("--out_dir", default="output_plots/free_run/shared_prior")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--K_beh", type=int, default=30, help="Behavior context length")
    ap.add_argument("--rollout", type=int, default=30, help="Rollout steps for training")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--max_worms", type=int, default=None, help="Max worms to use")
    ap.add_argument("--n_samples", type=int, default=20)
    args = ap.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    
    print("=" * 60)
    print("TRAINING SHARED BEHAVIOR PRIOR")
    print("=" * 60)
    
    # Load all worm behavior
    print(f"\nLoading behavior from {args.data_dir}...")
    beh_list, worm_ids = load_all_behavior(args.data_dir, args.max_worms)
    print(f"Loaded {len(beh_list)} worms, total frames: {sum(len(b) for b in beh_list)}")
    
    if len(beh_list) == 0:
        print("No data loaded!")
        return
    
    Kw = beh_list[0].shape[1]
    print(f"Behavior dimensions: Kw={Kw}")
    
    # Normalize
    print("\nNormalizing behavior...")
    beh_norm, mu, sig = normalize_behavior(beh_list)
    
    # Train shared AR model
    print(f"\nTraining shared AR model (K_beh={args.K_beh}, rollout={args.rollout})...")
    t0 = time.time()
    model = train_shared_ar_rollout(
        beh_norm, args.K_beh, device,
        rollout_steps=args.rollout, epochs=args.epochs
    )
    print(f"Training done in {time.time() - t0:.1f}s")
    
    # Save model
    model_path = out_dir / "shared_beh_prior.pt"
    torch.save({
        "model_state": model.state_dict(),
        "Kw": Kw,
        "K_beh": args.K_beh,
        "mu": mu,
        "sig": sig,
        "worm_ids": worm_ids,
    }, model_path)
    print(f"Model saved to {model_path}")
    
    # Generate samples for each worm and evaluate
    print("\nGenerating samples and evaluating...")
    
    all_metrics = {}
    for i, (beh, worm_id) in enumerate(zip(beh_norm, worm_ids)):
        if i >= 5:  # Limit evaluation to first 5 worms
            break
        
        # Use first half as context, generate second half
        T = len(beh)
        train_end = T // 2
        gt_test = beh[train_end:]
        seed = beh[train_end - args.K_beh:train_end]
        n_test = len(gt_test)
        
        # Generate samples
        samples = generate_samples(model, seed, n_test, args.n_samples, temperature=1.0, device=device)
        
        # Compute metrics
        gt_psd = compute_psd(gt_test)[1]
        sample_psds = [compute_psd(samples[s])[1] for s in range(len(samples))]
        
        # PSD distance (mean over samples and dimensions)
        psd_dist = np.mean([np.sqrt(np.mean((sp - gt_psd) ** 2)) for sp in sample_psds])
        
        # Variance ratio
        var_gt = gt_test.var(axis=0).mean()
        var_gen = np.mean([samples[s].var(axis=0).mean() for s in range(len(samples))])
        var_ratio = var_gen / (var_gt + 1e-8)
        
        all_metrics[worm_id] = {
            "psd_distance": float(psd_dist),
            "variance_ratio": float(var_ratio),
        }
        
        print(f"  {worm_id}: PSD_dist={psd_dist:.4f}, Var_ratio={var_ratio:.3f}")
        
        # Plot
        plot_samples(gt_test, samples, out_dir / f"samples_{worm_id}.png", f"Worm: {worm_id}")
        plot_psd_comparison(gt_test, samples, out_dir / f"psd_{worm_id}.png")
    
    # Save results
    results = {
        "K_beh": args.K_beh,
        "rollout": args.rollout,
        "n_worms": len(worm_ids),
        "worm_ids": worm_ids,
        "metrics": all_metrics,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
