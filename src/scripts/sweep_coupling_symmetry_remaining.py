#!/usr/bin/env python
"""
Coupling symmetry sweep — remaining conditions with 5-fold CV
==============================================================
Conditions already done (2-fold): T00–T04
This script runs the remaining 6 conditions with 5-fold CV:
  T05  edge_G + no_gate          (was incomplete)
  T06  pn_amp + edge_G
  T08  ks + pn_amp + edge_G
  T09  pn_amp 100ep
  T10  ks + pn_amp 100ep
  T11  full combo 100ep
"""
import sys, os, time, shutil, json
sys.path.insert(0, "/home/agross/Downloads/connectome-constrained model/src")

import torch
import numpy as np
from stage2.config import make_config
from stage2.train import train_stage2_cv

H5 = "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5"
SAVE_ROOT = "output_plots/stage2/coupling_symmetry_sweep"

# Kitchen-sink recipe
_KS = dict(
    chemical_synapse_activation="softplus",
    graph_poly_order=3,
    input_noise_sigma=0.05,
    coupling_dropout=0.1,
    rollout_weight=0.3,
    rollout_steps=15,
    rollout_starts=4,
)

# ---------- remaining condition definitions ----------
CONDITIONS = {
    "T05_edge_G+no_gate": dict(
        chemical_synapse_activation="softplus",
        edge_specific_G=True,
        coupling_gate=False,
        _epochs=60,
    ),

    "T06_pn_amp+edge_G": dict(
        chemical_synapse_activation="softplus",
        per_neuron_amplitudes=True,
        edge_specific_G=True,
        _epochs=60,
    ),

    "T08_ks+pn_amp+edge_G": dict(
        **_KS,
        per_neuron_amplitudes=True,
        edge_specific_G=True,
        _epochs=60,
    ),

    "T09_pn_amp_100ep": dict(
        chemical_synapse_activation="softplus",
        per_neuron_amplitudes=True,
        _epochs=100,
    ),

    "T10_ks+pn_amp_100ep": dict(
        **_KS,
        per_neuron_amplitudes=True,
        _epochs=100,
    ),

    "T11_full_100ep": dict(
        chemical_synapse_activation="softplus",
        per_neuron_amplitudes=True,
        edge_specific_G=True,
        _epochs=100,
    ),
}

os.makedirs(SAVE_ROOT, exist_ok=True)
summary_rows = []

for cond_name, overrides in CONDITIONS.items():
    save_dir = os.path.join(SAVE_ROOT, cond_name)

    # Skip if already completed
    if os.path.exists(os.path.join(save_dir, "summary.json")):
        print(f"\n{'='*60}\n[SKIP] {cond_name} — already done\n{'='*60}")
        with open(os.path.join(save_dir, "summary.json")) as f:
            row = json.load(f)
        summary_rows.append(row)
        continue

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    # Extract epochs
    epochs = overrides.pop("_epochs", 60)

    print(f"\n{'='*60}\n[RUN] {cond_name}  epochs={epochs}  cv_folds=5  overrides={overrides}\n{'='*60}",
          flush=True)

    # Build config — 5-fold CV
    cfg = make_config(H5, cv_folds=5, **overrides)
    cfg.seed = 42
    cfg.num_epochs = epochs
    cfg.network_strength_floor = 1.0
    cfg.network_strength_target = 0.8

    t0 = time.time()
    results = train_stage2_cv(cfg, save_dir=save_dir)
    elapsed = time.time() - t0

    # Collect results
    row = {"condition": cond_name, "elapsed_sec": round(elapsed, 1),
           "epochs": epochs, "cv_folds": 5,
           "overrides": {k: str(v) for k, v in overrides.items()}}
    if results is not None:
        for k, v in results.items():
            if isinstance(v, (float, int)):
                row[k] = round(v, 6) if isinstance(v, float) else v
    summary_rows.append(row)

    # Save per-condition summary
    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(row, f, indent=2)

    # Print running table
    print(f"\n[DONE] {cond_name}  time={elapsed:.0f}s")
    for k in ["cv_onestep_r2_mean", "cv_onestep_r2_median",
              "cv_loo_r2_mean", "cv_loo_r2_windowed_mean"]:
        if k in row:
            print(f"  {k}: {row[k]:.4f}")

# ---------- Final summary table ----------
print("\n\n" + "=" * 80)
print("COUPLING SYMMETRY SWEEP (REMAINING) — 5-FOLD CV SUMMARY")
print("=" * 80)
header = f"{'Condition':<30s} {'1step_R²':>9s} {'1step_med':>9s} {'LOO_w_mean':>10s} {'LOO_w_med':>9s} {'LOO_mean':>9s} {'Time(s)':>8s}"
print(header)
print("-" * len(header))
for row in summary_rows:
    name = row.get("condition", "?")
    r2 = row.get("cv_onestep_r2_mean", float('nan'))
    r2m = row.get("cv_onestep_r2_median", float('nan'))
    loo_w = row.get("cv_loo_r2_windowed_mean", float('nan'))
    loo_wm = row.get("cv_loo_r2_windowed_median", float('nan'))
    loo = row.get("cv_loo_r2_mean", float('nan'))
    t = row.get("elapsed_sec", 0)
    print(f"{name:<30s} {r2:9.4f} {r2m:9.4f} {loo_w:10.4f} {loo_wm:9.4f} {loo:9.4f} {t:8.0f}")

# Save combined summary
with open(os.path.join(SAVE_ROOT, "sweep_remaining_summary.json"), "w") as f:
    json.dump(summary_rows, f, indent=2)
print(f"\nSaved to {SAVE_ROOT}/sweep_remaining_summary.json")
