#!/usr/bin/env python
"""Diagnostic script: inspect free-run stability of a trained Stage-2 model."""

from __future__ import annotations

import sys, os
import torch
import numpy as np
import h5py

# ── paths ──────────────────────────────────────────────────────────────────
h5_path   = "data/used/behaviour+neuronal activity atanas (2023)/2/2023-01-17-14.h5"
ckpt_path = "output_plots/closed_loop/run1/stage2/stage2_results.h5"

# ── imports ────────────────────────────────────────────────────────────────
from stage2.config import make_config
from stage2.io_h5 import load_data_pt
from stage2.model import Stage2ModelPT
from stage2.init_from_data import init_lambda_u, init_all_from_data
from stage2.evaluate import compute_free_run

device = torch.device("cuda")

# ── build model ────────────────────────────────────────────────────────────
cfg  = make_config(h5_path, device="cuda")
data = load_data_pt(cfg)
u    = data["u_stage1"]
T, N = u.shape
dt   = float(data["dt"])

print(f"Data: T={T}, N={N}, dt={dt:.3f}s")

lambda_u_init = init_lambda_u(u, cfg)
model = Stage2ModelPT(
    N, data["T_e"], data["T_sv"], data["T_dcv"], dt, cfg, device,
    d_ell=data.get("d_ell", 0),
    lambda_u_init=lambda_u_init,
    sign_t=data.get("sign_t"),
).to(device)
init_all_from_data(model, u.to(device), cfg)

# ── load checkpoint ────────────────────────────────────────────────────────
with h5py.File(ckpt_path, "r") as f:
    grp = f["stage2_pt/params"]
    sd  = model.state_dict()
    loaded_keys = []
    for key in sd:
        if key in grp:
            sd[key] = torch.tensor(np.array(grp[key]), dtype=sd[key].dtype)
            loaded_keys.append(key)
    model.load_state_dict(sd, strict=False)

print(f"Loaded {len(loaded_keys)}/{len(sd)} keys from checkpoint")
model.eval()

# ── 1. free-run ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FREE-RUN EVALUATION")
print("=" * 60)

fr = compute_free_run(model, data)
r2   = fr["r2"]
mode = fr["mode"]
u_free = fr["u_free"]          # (T, N)
u_true = data["u_stage1"].cpu().numpy()

valid = np.isfinite(r2)
r2v   = r2[valid]
print(f"Mode: {mode}")
print(f"Eval neurons: {valid.sum()} / {N}")
print(f"Per-neuron free-run R²:")
print(f"  mean   = {np.mean(r2v):.4f}")
print(f"  median = {np.median(r2v):.4f}")
print(f"  min    = {np.min(r2v):.4f}")
print(f"  max    = {np.max(r2v):.4f}")
print(f"  frac>0 = {(r2v > 0).mean():.3f}")
print(f"  frac>0.3 = {(r2v > 0.3).mean():.3f}")
print(f"  frac>0.5 = {(r2v > 0.5).mean():.3f}")

# ── 2. neural activity std: start vs end ──────────────────────────────────
print("\n" + "=" * 60)
print("NEURAL ACTIVITY STD: START vs END (explosion check)")
print("=" * 60)

window = min(50, T // 4)
std_start_true = np.std(u_true[:window], axis=0)
std_end_true   = np.std(u_true[-window:], axis=0)
std_start_free = np.std(u_free[:window], axis=0)
std_end_free   = np.std(u_free[-window:], axis=0)

print(f"Window = first/last {window} frames")
print(f"TRUE data:  mean-std start={np.mean(std_start_true):.4f}  end={np.mean(std_end_true):.4f}")
print(f"FREE-RUN:   mean-std start={np.mean(std_start_free):.4f}  end={np.mean(std_end_free):.4f}")
print(f"FREE-RUN:   max-std  start={np.max(std_start_free):.4f}  end={np.max(std_end_free):.4f}")

# Also check for NaN / Inf
n_nan = np.isnan(u_free).sum()
n_inf = np.isinf(u_free).sum()
print(f"\nNaN count in free-run: {n_nan}")
print(f"Inf count in free-run: {n_inf}")

# Global range
print(f"Free-run value range: [{np.nanmin(u_free):.3f}, {np.nanmax(u_free):.3f}]")
print(f"True data value range: [{np.nanmin(u_true):.3f}, {np.nanmax(u_true):.3f}]")

# ── 3. sigma_at (process noise) ───────────────────────────────────────────
print("\n" + "=" * 60)
print("PROCESS NOISE: sigma_at(u)")
print("=" * 60)

try:
    sigma = model.sigma_at().detach().cpu().numpy()
    print(f"sigma shape: {sigma.shape}")
    print(f"  mean  = {np.mean(sigma):.6f}")
    print(f"  min   = {np.min(sigma):.6f}")
    print(f"  max   = {np.max(sigma):.6f}")
    print(f"  median= {np.median(sigma):.6f}")
except Exception as e:
    print(f"sigma_at() failed: {e}")

# ── 4. u_clip bounds ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("u_clip BOUNDS")
print("=" * 60)

u_clip = getattr(model, "u_clip", None)
print(f"u_clip = {u_clip}")

# ── 5. lambda_u stats ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("lambda_u (leak / decay) STATS")
print("=" * 60)

with torch.no_grad():
    lu = model.lambda_u.detach().cpu().numpy()
print(f"lambda_u shape: {lu.shape}")
print(f"  mean   = {np.mean(lu):.6f}")
print(f"  median = {np.median(lu):.6f}")
print(f"  min    = {np.min(lu):.6f}")
print(f"  max    = {np.max(lu):.6f}")
print(f"  frac >= 0.99 = {(lu >= 0.99).mean():.3f}")
print(f"  frac >= 0.95 = {(lu >= 0.95).mean():.3f}")
print(f"  frac < 0.5   = {(lu < 0.5).mean():.3f}")

# Worst 5 neurons by lambda_u (closest to 1 = slowest decay = most likely to drift)
top5 = np.argsort(lu.ravel())[-5:][::-1]
print(f"\nTop-5 lambda_u neurons (slowest decay):")
for idx in top5:
    print(f"  neuron {idx:3d}: lambda_u={lu.ravel()[idx]:.6f}  free-run R²={r2[idx]:.4f}")

# Worst 5 neurons by R²
bot5 = np.argsort(r2v)[:5]
print(f"\nBottom-5 R² neurons:")
for idx in bot5:
    real_idx = np.where(valid)[0][idx]
    print(f"  neuron {real_idx:3d}: R²={r2[real_idx]:.4f}  lambda_u={lu.ravel()[real_idx]:.6f}")

print("\nDone.")
