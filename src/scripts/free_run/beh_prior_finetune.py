"""
Strategy 2: Pre-trained Behavior Prior + Per-Worm Fine-tuning

This approach:
1. Loads a pre-trained shared behavior prior (from train_shared_beh_prior.py)
2. For each worm, trains a neural→behavior decoder regularized by the prior
3. The prior provides a "behavior manifold" constraint

The decoder loss combines:
- MSE(predicted_beh, true_beh)  -- reconstruction
- KL(predicted_beh | prior_beh) -- regularization toward realistic dynamics

Usage:
    python -m scripts.free_run.beh_prior_finetune \
        --prior_path output_plots/free_run/shared_prior/shared_beh_prior.pt \
        --h5 "data/used/behaviour+neuronal activity atanas (2023)/2/2022-06-14-07.h5" \
        --out_dir output_plots/free_run/prior_finetune
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
from scripts.free_run.train_shared_beh_prior import BehaviorARModel, generate_samples, plot_psd_comparison


class NeuralToBehDecoder(nn.Module):
    """Neural→Behavior decoder with prior regularization."""
    
    def __init__(self, N: int, Kw: int, K_neural: int, K_beh: int,
                 hidden: int = 256, n_layers: int = 3):
        super().__init__()
        self.N = N
        self.Kw = Kw
        self.K_neural = K_neural
        self.K_beh = K_beh
        
        # Input: neural_lags (K_neural * N) + beh_lags (K_beh * Kw)
        d_in = K_neural * N + K_beh * Kw
        
        layers, d = [], d_in
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.1)]
            d = hidden
        # Output: mean and log_std
        layers.append(nn.Linear(d, Kw * 2))
        self.net = nn.Sequential(*layers)
    
    def forward(self, neural_lags, beh_lags):
        """
        neural_lags: (B, K_neural * N)
        beh_lags: (B, K_beh * Kw)
        Returns: mu (B, Kw), log_std (B, Kw)
        """
        x = torch.cat([neural_lags, beh_lags], dim=-1)
        out = self.net(x)
        mu, log_std = out.split(self.Kw, dim=-1)
        log_std = torch.clamp(log_std, -5, 2)
        return mu, log_std
    
    def sample(self, neural_lags, beh_lags, temperature=1.0):
        mu, log_std = self.forward(neural_lags, beh_lags)
        std = torch.exp(log_std) * temperature
        return mu + std * torch.randn_like(mu)


def kl_divergence_gaussian(mu1, log_std1, mu2, log_std2):
    """KL(N(mu1, std1^2) || N(mu2, std2^2))"""
    std1, std2 = torch.exp(log_std1), torch.exp(log_std2)
    var1, var2 = std1 ** 2, std2 ** 2
    kl = log_std2 - log_std1 + (var1 + (mu1 - mu2) ** 2) / (2 * var2) - 0.5
    return kl.mean()


def train_decoder_with_prior(
    u_train: np.ndarray, b_train: np.ndarray,
    prior: BehaviorARModel, K_neural: int, K_beh: int,
    device: str, kl_weight: float = 0.1,
    epochs: int = 200, lr: float = 1e-3, patience: int = 30
) -> NeuralToBehDecoder:
    """Train decoder with prior regularization."""
    T, N = u_train.shape
    Kw = b_train.shape[1]
    
    # Build data
    K = max(K_neural, K_beh)
    valid_idx = np.arange(K, T)
    
    # Pre-compute lagged features
    u_lags_all = build_lagged(u_train, K_neural)
    b_lags_all = build_lagged(b_train, K_beh)
    
    # Tensors
    u_lags = torch.tensor(u_lags_all[valid_idx], dtype=torch.float32, device=device)
    b_lags = torch.tensor(b_lags_all[valid_idx], dtype=torch.float32, device=device)
    b_target = torch.tensor(b_train[valid_idx], dtype=torch.float32, device=device)
    
    # Train/val split
    n_val = max(50, int(len(valid_idx) * 0.15))
    train_idx = slice(0, -n_val)
    val_idx = slice(-n_val, None)
    
    # Model
    decoder = NeuralToBehDecoder(N, Kw, K_neural, K_beh).to(device)
    prior = prior.to(device).eval()
    
    opt = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=1e-4)
    
    best_val, best_state, pat = float("inf"), None, 0
    
    for ep in range(epochs):
        decoder.train()
        
        # Forward
        mu_dec, log_std_dec = decoder(u_lags[train_idx], b_lags[train_idx])
        
        # Reconstruction loss (Gaussian NLL)
        std_dec = torch.exp(log_std_dec)
        nll = 0.5 * (((b_target[train_idx] - mu_dec) / std_dec) ** 2 + 2 * log_std_dec)
        recon_loss = nll.mean()
        
        # Prior regularization: KL(decoder || prior)
        with torch.no_grad():
            mu_prior, log_std_prior = prior(b_lags[train_idx])
        kl_loss = kl_divergence_gaussian(mu_dec, log_std_dec, mu_prior, log_std_prior)
        
        loss = recon_loss + kl_weight * kl_loss
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        opt.step()
        
        # Validation
        decoder.eval()
        with torch.no_grad():
            mu_val, log_std_val = decoder(u_lags[val_idx], b_lags[val_idx])
            std_val = torch.exp(log_std_val)
            val_nll = 0.5 * (((b_target[val_idx] - mu_val) / std_val) ** 2 + 2 * log_std_val)
            val_loss = val_nll.mean().item()
        
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in decoder.state_dict().items()}
            pat = 0
        else:
            pat += 1
        
        if (ep + 1) % 20 == 0:
            print(f"  Epoch {ep+1}: recon={recon_loss:.4f}, kl={kl_loss:.4f}, val={val_loss:.4f}")
        
        if pat > patience:
            print(f"  Early stopping at epoch {ep+1}")
            break
    
    if best_state:
        decoder.load_state_dict(best_state)
    decoder.eval().cpu()
    prior.cpu()
    return decoder


def generate_with_decoder(
    decoder: NeuralToBehDecoder, u_test: np.ndarray, b_seed: np.ndarray,
    n_samples: int, temperature: float, device: str
) -> np.ndarray:
    """Generate behavior autoregressively using the decoder."""
    decoder = decoder.to(device).eval()
    T_test = len(u_test)
    Kw = decoder.Kw
    K_neural = decoder.K_neural
    K_beh = decoder.K_beh
    
    samples = np.zeros((n_samples, T_test, Kw), dtype=np.float32)
    
    with torch.no_grad():
        for s in range(n_samples):
            beh_history = torch.tensor(b_seed[-K_beh:].copy(), dtype=torch.float32, device=device)
            
            for t in range(T_test):
                # Neural lags
                if t < K_neural:
                    u_lag = np.zeros(K_neural * decoder.N, dtype=np.float32)
                    u_lag[-(t+1)*decoder.N:] = u_test[:t+1].flatten()
                else:
                    u_lag = u_test[t-K_neural+1:t+1].flatten()
                u_lag_t = torch.tensor(u_lag, dtype=torch.float32, device=device).unsqueeze(0)
                
                # Behavior lags
                b_lag_t = beh_history.reshape(1, -1)
                
                # Sample
                sample = decoder.sample(u_lag_t, b_lag_t, temperature=temperature)
                samples[s, t] = sample[0].cpu().numpy()
                
                # Update history
                beh_history = torch.cat([beh_history[1:], sample], dim=0)
    
    decoder.cpu()
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior_path", required=True, help="Path to shared_beh_prior.pt")
    ap.add_argument("--h5", required=True, help="Path to worm h5 file")
    ap.add_argument("--out_dir", default="output_plots/free_run/prior_finetune")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--K_neural", type=int, default=15)
    ap.add_argument("--K_beh", type=int, default=30)
    ap.add_argument("--kl_weight", type=float, default=0.1)
    ap.add_argument("--n_samples", type=int, default=20)
    ap.add_argument("--train_frac", type=float, default=0.5)
    args = ap.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    
    print("=" * 60)
    print("STRATEGY 2: Behavior Prior + Fine-tuning")
    print("=" * 60)
    
    # Load prior
    print(f"\nLoading prior from {args.prior_path}...")
    prior_data = torch.load(args.prior_path, map_location="cpu", weights_only=False)
    prior = BehaviorARModel(prior_data["Kw"], prior_data["K_beh"])
    prior.load_state_dict(prior_data["model_state"])
    prior.eval()
    mu_beh, sig_beh = prior_data["mu"], prior_data["sig"]
    print(f"  Prior K_beh={prior_data['K_beh']}, Kw={prior_data['Kw']}")
    
    # Load worm data
    print(f"\nLoading worm data from {args.h5}...")
    with h5py.File(args.h5, "r") as f:
        # Handle nested h5 structure
        if "stage1" in f and "u_mean" in f["stage1"]:
            u = f["stage1"]["u_mean"][:]
        elif "u_mean" in f:
            u = f["u_mean"][:]
        else:
            u = f["u"][:]
        
        if "behaviour" in f and "eigenworms_calc_6" in f["behaviour"]:
            b = f["behaviour"]["eigenworms_calc_6"][:]
        else:
            b = f["beh"][:]
    
    worm_id = Path(args.h5).stem
    T, N = u.shape
    Kw = b.shape[1]
    train_end = int(T * args.train_frac)
    
    print(f"  Worm: {worm_id}, T={T}, N={N}, Kw={Kw}")
    print(f"  Train: [0, {train_end}), Test: [{train_end}, {T})")
    
    # Normalize
    u_train, u_test = u[:train_end], u[train_end:]
    b_train, b_test = b[:train_end], b[train_end:]
    
    mu_u, sig_u = u_train.mean(0), u_train.std(0) + 1e-8
    u_train_norm = (u_train - mu_u) / sig_u
    u_test_norm = (u_test - mu_u) / sig_u
    
    # Use prior's normalization for behavior
    b_train_norm = (b_train - mu_beh) / sig_beh
    b_test_norm = (b_test - mu_beh) / sig_beh
    
    # Train decoder with prior
    print(f"\nTraining decoder with prior (kl_weight={args.kl_weight})...")
    t0 = time.time()
    decoder = train_decoder_with_prior(
        u_train_norm, b_train_norm, prior,
        args.K_neural, args.K_beh, device,
        kl_weight=args.kl_weight
    )
    print(f"Training done in {time.time() - t0:.1f}s")
    
    # Generate samples
    print(f"\nGenerating {args.n_samples} samples...")
    b_seed = b_train_norm[-args.K_beh:]
    samples_norm = generate_with_decoder(
        decoder, u_test_norm, b_seed,
        args.n_samples, temperature=1.0, device=device
    )
    
    # Denormalize
    samples = samples_norm * sig_beh + mu_beh
    
    # Metrics
    gt_psd = compute_psd(b_test)[1]
    sample_psds = [compute_psd(samples[s])[1] for s in range(len(samples))]
    psd_dist = np.mean([np.sqrt(np.mean((sp - gt_psd) ** 2)) for sp in sample_psds])
    var_ratio = samples.var(axis=(0,1)).mean() / (b_test.var(axis=0).mean() + 1e-8)
    
    print(f"\nMetrics: PSD_dist={psd_dist:.4f}, Var_ratio={var_ratio:.3f}")
    
    # Save
    torch.save({
        "decoder_state": decoder.state_dict(),
        "K_neural": args.K_neural,
        "K_beh": args.K_beh,
        "N": N, "Kw": Kw,
        "mu_u": mu_u, "sig_u": sig_u,
    }, out_dir / f"decoder_{worm_id}.pt")
    
    results = {
        "worm_id": worm_id,
        "kl_weight": args.kl_weight,
        "psd_distance": float(psd_dist),
        "variance_ratio": float(var_ratio),
    }
    with open(out_dir / f"results_{worm_id}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Plot
    plot_psd_comparison(b_test, samples, out_dir / f"psd_{worm_id}.png")
    
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
