"""
Strategy 3: Behavior VAE for Realistic Generation

This approach:
1. Trains a VAE on pooled behavior data from all worms
2. The VAE learns a latent manifold of realistic locomotion
3. During neural→behavior decoding, we project predictions onto this manifold

The VAE decoder acts as a "realism filter" that ensures generated
behavior lies within the distribution of real movements.

Usage:
    python -m scripts.free_run.train_beh_vae \
        --data_dir "data/used/behaviour+neuronal activity atanas (2023)/2" \
        --out_dir output_plots/free_run/beh_vae
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
import torch.nn.functional as F
from scipy import signal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.free_run.utils import NumpyEncoder, compute_psd


def load_all_behavior(data_dir: str, max_worms: int = None):
    """Load behavior from all worms."""
    from pathlib import Path
    import h5py
    
    data_path = Path(data_dir)
    h5_files = sorted(data_path.glob("*.h5"))
    
    if max_worms:
        h5_files = h5_files[:max_worms]
    
    all_beh = []
    worm_ids = []
    
    for h5_path in h5_files:
        try:
            with h5py.File(h5_path, "r") as f:
                beh = None
                # Try nested paths
                if "behaviour" in f and "eigenworms_calc_6" in f["behaviour"]:
                    beh = f["behaviour"]["eigenworms_calc_6"][:]
                elif "behaviour" in f and "eigenworms_stephens" in f["behaviour"]:
                    beh = f["behaviour"]["eigenworms_stephens"][:]
                elif "beh" in f:
                    beh = f["beh"][:]
                
                if beh is None:
                    continue
                
                if beh.ndim == 1:
                    beh = beh.reshape(-1, 1)
                
                if len(beh) < 100:
                    continue
                
                all_beh.append(beh.astype(np.float32))
                worm_ids.append(h5_path.stem)
                print(f"  Loaded {h5_path.name}: T={len(beh)}, Kw={beh.shape[1]}")
        except Exception as e:
            print(f"  Error loading {h5_path.name}: {e}")
    
    return all_beh, worm_ids


def normalize_behavior(beh_list):
    """Normalize behavior across all worms."""
    all_concat = np.concatenate(beh_list, axis=0)
    mu = all_concat.mean(axis=0)
    sig = all_concat.std(axis=0) + 1e-8
    normed = [(b - mu) / sig for b in beh_list]
    return normed, mu, sig


class TemporalBehaviorVAE(nn.Module):
    """VAE for behavior sequences with temporal structure.
    
    Encodes a sequence of K_beh frames into a latent z,
    then decodes to predict the next frame.
    """
    
    def __init__(self, Kw: int, K_beh: int, latent_dim: int = 32, hidden: int = 256):
        super().__init__()
        self.Kw = Kw
        self.K_beh = K_beh
        self.latent_dim = latent_dim
        
        d_in = K_beh * Kw
        
        # Encoder: sequence → latent
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
        """x: (B, K_beh * Kw) → mu, logvar"""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, x):
        """z: (B, latent_dim), x: (B, K_beh * Kw) → (B, Kw)"""
        return self.decoder(torch.cat([z, x], dim=-1))
    
    def forward(self, x):
        """x: (B, K_beh * Kw) → recon, mu, logvar"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, x)
        return recon, mu, logvar
    
    def sample(self, x, temperature=1.0):
        """Generate next frame from sequence."""
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar) * temperature
        z = mu + std * torch.randn_like(std)
        return self.decode(z, x)


def vae_loss(recon, target, mu, logvar, beta=0.1):
    """VAE loss = reconstruction + β * KL divergence."""
    recon_loss = F.mse_loss(recon, target)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def train_behavior_vae(
    beh_list: list[np.ndarray], K_beh: int, device: str,
    latent_dim: int = 32, beta: float = 0.1,
    epochs: int = 300, lr: float = 1e-3, patience: int = 40,
    batch_size: int = 128
) -> TemporalBehaviorVAE:
    """Train VAE on pooled behavior data."""
    Kw = beh_list[0].shape[1]
    
    # Build training sequences
    sequences = []
    targets = []
    for beh in beh_list:
        T = len(beh)
        for t in range(K_beh, T):
            sequences.append(beh[t-K_beh:t].flatten())
            targets.append(beh[t])
    
    sequences = np.array(sequences, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)
    
    # Shuffle
    perm = np.random.permutation(len(sequences))
    sequences, targets = sequences[perm], targets[perm]
    
    # Train/val split
    n_val = max(100, int(len(sequences) * 0.15))
    train_seq = torch.tensor(sequences[:-n_val], device=device)
    train_tgt = torch.tensor(targets[:-n_val], device=device)
    val_seq = torch.tensor(sequences[-n_val:], device=device)
    val_tgt = torch.tensor(targets[-n_val:], device=device)
    
    print(f"  Training samples: {len(train_seq)}, Validation: {len(val_seq)}")
    
    model = TemporalBehaviorVAE(Kw, K_beh, latent_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=1e-5)
    
    best_val, best_state, pat = float("inf"), None, 0
    
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(train_seq))
        total_loss, total_recon, total_kl = 0.0, 0.0, 0.0
        n_batches = 0
        
        for i in range(0, len(train_seq), batch_size):
            idx = perm[i:i+batch_size]
            x, y = train_seq[idx], train_tgt[idx]
            
            recon, mu, logvar = model(x)
            loss, recon_l, kl_l = vae_loss(recon, y, mu, logvar, beta)
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            total_loss += loss.item()
            total_recon += recon_l.item()
            total_kl += kl_l.item()
            n_batches += 1
        
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            recon, mu, logvar = model(val_seq)
            val_loss, _, _ = vae_loss(recon, val_tgt, mu, logvar, beta)
            val_loss = val_loss.item()
        
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
        
        if (ep + 1) % 20 == 0:
            print(f"  Epoch {ep+1}: loss={total_loss/n_batches:.4f} "
                  f"(recon={total_recon/n_batches:.4f}, kl={total_kl/n_batches:.4f}), "
                  f"val={val_loss:.4f}")
        
        if pat > patience:
            print(f"  Early stopping at epoch {ep+1}")
            break
    
    if best_state:
        model.load_state_dict(best_state)
    model.eval().cpu()
    return model


def generate_samples(model: TemporalBehaviorVAE, seed: np.ndarray, n_steps: int,
                     n_samples: int = 20, temperature: float = 1.0,
                     device: str = "cpu") -> np.ndarray:
    """Generate behavior samples from VAE."""
    model = model.to(device).eval()
    K_beh = model.K_beh
    Kw = model.Kw
    
    samples = np.zeros((n_samples, n_steps, Kw), dtype=np.float32)
    
    with torch.no_grad():
        for s in range(n_samples):
            history = seed[-K_beh:].copy()
            
            for t in range(n_steps):
                x = torch.tensor(history.flatten(), dtype=torch.float32, device=device).unsqueeze(0)
                sample = model.sample(x, temperature=temperature)
                samples[s, t] = sample[0].cpu().numpy()
                history = np.vstack([history[1:], samples[s, t]])
    
    model.cpu()
    return samples


def plot_latent_space(model: TemporalBehaviorVAE, beh_list: list[np.ndarray],
                      out_path: Path, device: str = "cpu"):
    """Visualize latent space."""
    model = model.to(device).eval()
    K_beh = model.K_beh
    
    all_z = []
    for beh in beh_list[:5]:  # First 5 worms
        T = len(beh)
        sequences = np.array([beh[t-K_beh:t].flatten() for t in range(K_beh, T)], dtype=np.float32)
        with torch.no_grad():
            x = torch.tensor(sequences, device=device)
            mu, _ = model.encode(x)
            all_z.append(mu.cpu().numpy())
    
    model.cpu()
    
    # Plot first 2 dimensions
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_z)))
    for i, z in enumerate(all_z):
        ax.scatter(z[:, 0], z[:, 1], c=[colors[i]], alpha=0.3, s=5, label=f"Worm {i+1}")
    ax.set_xlabel("z₁")
    ax.set_ylabel("z₂")
    ax.set_title("Behavior VAE Latent Space")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default="output_plots/free_run/beh_vae")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--K_beh", type=int, default=30)
    ap.add_argument("--latent_dim", type=int, default=32)
    ap.add_argument("--beta", type=float, default=0.1, help="KL weight")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--n_samples", type=int, default=20)
    args = ap.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    
    print("=" * 60)
    print("STRATEGY 3: Behavior VAE")
    print("=" * 60)
    
    # Load all worm behavior
    print(f"\nLoading behavior from {args.data_dir}...")
    beh_list, worm_ids = load_all_behavior(args.data_dir)
    print(f"Loaded {len(beh_list)} worms")
    
    # Normalize
    beh_norm, mu, sig = normalize_behavior(beh_list)
    Kw = beh_norm[0].shape[1]
    
    # Train VAE
    print(f"\nTraining VAE (K_beh={args.K_beh}, latent_dim={args.latent_dim}, β={args.beta})...")
    t0 = time.time()
    model = train_behavior_vae(
        beh_norm, args.K_beh, device,
        latent_dim=args.latent_dim, beta=args.beta, epochs=args.epochs
    )
    print(f"Training done in {time.time() - t0:.1f}s")
    
    # Save model
    torch.save({
        "model_state": model.state_dict(),
        "Kw": Kw,
        "K_beh": args.K_beh,
        "latent_dim": args.latent_dim,
        "mu": mu,
        "sig": sig,
        "worm_ids": worm_ids,
    }, out_dir / "beh_vae.pt")
    
    # Plot latent space
    plot_latent_space(model, beh_norm, out_dir / "latent_space.png", device)
    
    # Generate and evaluate samples for each worm
    print("\nGenerating samples and evaluating...")
    all_metrics = {}
    
    for i, (beh, worm_id) in enumerate(zip(beh_norm, worm_ids)):
        if i >= 5:
            break
        
        T = len(beh)
        train_end = T // 2
        gt_test = beh[train_end:]
        seed = beh[train_end - args.K_beh:train_end]
        
        samples = generate_samples(model, seed, len(gt_test), args.n_samples,
                                   temperature=1.0, device=device)
        
        # Metrics
        gt_psd = compute_psd(gt_test)[1]
        sample_psds = [compute_psd(samples[s])[1] for s in range(len(samples))]
        psd_dist = np.mean([np.sqrt(np.mean((sp - gt_psd) ** 2)) for sp in sample_psds])
        var_ratio = samples.var(axis=(0,1)).mean() / (gt_test.var(axis=0).mean() + 1e-8)
        
        all_metrics[worm_id] = {
            "psd_distance": float(psd_dist),
            "variance_ratio": float(var_ratio),
        }
        print(f"  {worm_id}: PSD_dist={psd_dist:.4f}, Var_ratio={var_ratio:.3f}")
    
    # Save results
    results = {
        "K_beh": args.K_beh,
        "latent_dim": args.latent_dim,
        "beta": args.beta,
        "n_worms": len(worm_ids),
        "metrics": all_metrics,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
