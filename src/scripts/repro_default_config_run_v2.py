#!/usr/bin/env python
"""Reproduce the old default_config_run_v2 result (LOO_w ≈ 0.17).

The key difference vs the current default config:
  - seed=0  (no seeding → non-deterministic training)
  - make_posture_video=False

All other old config fields that are no longer in the dataclass
(W_sv_init_mode, W_sv_normalize, noise_sigma_source, per_neuron_tau_scale,
 interaction_l2, ridge_W_sv, ridge_W_dcv, G_reg) were all no-ops.

New config fields not in the old config (lag_order,
coupling_dropout, chemical_synapse_activation, etc.) all default to their
off/no-op values, so they don't affect the run.

Since training is non-deterministic with seed=0, we run N_RUNS times and
report the distribution of LOO_w to see if 0.1712 is within the range.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np

from stage2.config import make_config
from stage2.train import train_stage2

H5 = "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5"
SAVE_ROOT = "output_plots/stage2/repro_v2"

N_RUNS = 3  # run multiple times to see the variance from non-determinism

results = []

for run_i in range(N_RUNS):
    save_dir = f"{SAVE_ROOT}_run{run_i}"
    print(f"\n{'#'*70}")
    print(f"# REPRO RUN {run_i+1}/{N_RUNS} → {save_dir}")
    print(f"{'#'*70}\n")

    cfg = make_config(
        H5,
        # Match old config exactly:
        seed=0,                      # NO seeding (non-deterministic)
        make_posture_video=False,    # old default
        # All other params: current defaults match the old run
    )

    eval_result = train_stage2(cfg, save_dir=save_dir, show=False)

    # Extract metrics
    cv_r2_mean = eval_result.get("cv_onestep_r2_mean", float("nan"))
    cv_loo_mean = eval_result.get("cv_loo_r2_mean", float("nan"))
    cv_loo_w_mean = eval_result.get("cv_loo_r2_windowed_mean", float("nan"))

    results.append({
        "run": run_i,
        "cv_r2_mean": cv_r2_mean,
        "cv_loo_mean": cv_loo_mean,
        "cv_loo_w_mean": cv_loo_w_mean,
    })
    print(f"\n>>> RUN {run_i}: CV R²={cv_r2_mean:.4f}  "
          f"LOO={cv_loo_mean:.4f}  LOO_w={cv_loo_w_mean:.4f}")

# Summary
print(f"\n{'='*70}")
print(f"REPRO SUMMARY ({N_RUNS} runs, seed=0 non-deterministic)")
print(f"{'='*70}")
print(f"{'Run':<5} {'CV R²':<10} {'LOO':<10} {'LOO_w':<10}")
for r in results:
    print(f"{r['run']:<5} {r['cv_r2_mean']:<10.4f} "
          f"{r['cv_loo_mean']:<10.4f} {r['cv_loo_w_mean']:<10.4f}")

cv_vals = [r["cv_r2_mean"] for r in results]
loo_vals = [r["cv_loo_mean"] for r in results]
loo_w_vals = [r["cv_loo_w_mean"] for r in results]

print(f"\nCV R²:  mean={np.mean(cv_vals):.4f}  "
      f"range=[{np.min(cv_vals):.4f}, {np.max(cv_vals):.4f}]")
print(f"LOO:    mean={np.mean(loo_vals):.4f}  "
      f"range=[{np.min(loo_vals):.4f}, {np.max(loo_vals):.4f}]")
print(f"LOO_w:  mean={np.mean(loo_w_vals):.4f}  "
      f"range=[{np.min(loo_w_vals):.4f}, {np.max(loo_w_vals):.4f}]")
print(f"\nTarget: LOO_w ≈ 0.1712 (default_config_run_v2)")
print(f"Previous repro: LOO_w = 0.1443 (seed=42)")
