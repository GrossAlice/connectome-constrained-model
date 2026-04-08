#!/usr/bin/env python
"""
Coupling symmetry sweep — 3-fold CV, 30 epochs
================================================
Systematic test of every modelling feature.

Block A – Activation baselines:
  T00  softplus baseline
  T01  sigmoid baseline

Block B – Individual features (softplus base):
  T02  per_neuron_amplitudes
  T03  edge_specific_G
  T04  dynamics_l2=0.01
  T05  lag_neighbor (lag_order=2)
  T06  per_edge reversals

Block C – Kitchen-sink combos:
  T07  ks + pn_amp
  T08  ks + pn_amp + edge_G

Block D – Sigmoid combos:
  T09  sigmoid + pn_amp
  T10  sigmoid + ks + pn_amp

Block E – Feature combinations with pn_amp:
  T11  pn_amp + edge_G
  T12  pn_amp + dynamics_l2
  T13  pn_amp + lag_neighbor
  T14  pn_amp + per_edge_rev
"""
import sys, os, time, shutil, json
sys.path.insert(0, "/home/agross/Downloads/connectome-constrained model/src")

import torch
import numpy as np
from stage2.config import make_config
from stage2.train import train_stage2_cv

H5 = "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5"
SAVE_ROOT = "output_plots/stage2/coupling_symmetry_sweep_3fold"

CV_FOLDS = 3
EPOCHS   = 30

# Kitchen-sink recipe (best-ever from followup_sweep)
_KS = dict(
    graph_poly_order=3,
    input_noise_sigma=0.05,
    coupling_dropout=0.1,
    rollout_weight=0.3,
    rollout_steps=15,
    rollout_starts=4,
)

# ---------- condition definitions ----------
CONDITIONS = {
    # --- Block A: activation baselines ---
    "T00_softplus_baseline": dict(
        chemical_synapse_activation="softplus",
    ),
    "T01_sigmoid_baseline": dict(
        chemical_synapse_activation="sigmoid",
    ),

    # --- Block B: individual features (softplus) ---
    "T02_pn_amp": dict(
        chemical_synapse_activation="softplus",
        per_neuron_amplitudes=True,
    ),
    "T03_edge_G": dict(
        chemical_synapse_activation="softplus",
        edge_specific_G=True,
    ),
    "T04_dynamics_l2": dict(
        chemical_synapse_activation="softplus",
        _dynamics_l2=0.01,
    ),
    "T05_lag_neighbor": dict(
        chemical_synapse_activation="softplus",
        lag_order=2,
        lag_neighbor=True,
    ),
    "T06_per_edge_rev": dict(
        chemical_synapse_activation="softplus",
        learn_reversals=True,
        reversal_mode="per_edge",
    ),

    # --- Block C: kitchen-sink combos ---
    "T07_ks+pn_amp": dict(
        chemical_synapse_activation="softplus",
        **_KS,
        per_neuron_amplitudes=True,
    ),
    "T08_ks+pn_amp+edge_G": dict(
        chemical_synapse_activation="softplus",
        **_KS,
        per_neuron_amplitudes=True,
        edge_specific_G=True,
    ),

    # --- Block D: sigmoid combos ---
    "T09_sigmoid+pn_amp": dict(
        chemical_synapse_activation="sigmoid",
        per_neuron_amplitudes=True,
    ),
    "T10_sigmoid+ks+pn_amp": dict(
        chemical_synapse_activation="sigmoid",
        **_KS,
        per_neuron_amplitudes=True,
    ),

    # --- Block E: feature combos with pn_amp ---
    "T11_pn_amp+edge_G": dict(
        chemical_synapse_activation="softplus",
        per_neuron_amplitudes=True,
        edge_specific_G=True,
    ),
    "T12_pn_amp+dyn_l2": dict(
        chemical_synapse_activation="softplus",
        per_neuron_amplitudes=True,
        _dynamics_l2=0.01,
    ),
    "T13_pn_amp+lag_nbr": dict(
        chemical_synapse_activation="softplus",
        per_neuron_amplitudes=True,
        lag_order=2,
        lag_neighbor=True,
    ),
    "T14_pn_amp+per_edge_rev": dict(
        chemical_synapse_activation="softplus",
        per_neuron_amplitudes=True,
        learn_reversals=True,
        reversal_mode="per_edge",
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

    # Separate train-level overrides (prefixed with _) from dynamics-level
    train_overrides = {}
    dyn_overrides = {}
    for k, v in overrides.items():
        if k.startswith("_"):
            train_overrides[k[1:]] = v   # strip leading _
        else:
            dyn_overrides[k] = v

    print(f"\n{'='*60}\n[RUN] {cond_name}  epochs={EPOCHS}  cv_folds={CV_FOLDS}"
          f"\n      overrides={overrides}\n{'='*60}", flush=True)

    # Build config — 3-fold CV
    cfg = make_config(H5, cv_folds=CV_FOLDS, **dyn_overrides)
    cfg.seed = 42
    cfg.num_epochs = EPOCHS
    cfg.network_strength_floor = 1.0
    cfg.network_strength_target = 0.8
    # Apply train-level overrides
    for k, v in train_overrides.items():
        setattr(cfg, k, v)

    t0 = time.time()
    results = train_stage2_cv(cfg, save_dir=save_dir)
    elapsed = time.time() - t0

    # Collect results
    row = {"condition": cond_name, "elapsed_sec": round(elapsed, 1),
           "epochs": EPOCHS, "cv_folds": CV_FOLDS,
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
              "cv_loo_r2_mean", "cv_loo_r2_median"]:
        if k in row:
            print(f"  {k}: {row[k]:.4f}")

# ---------- Final summary table ----------
print("\n\n" + "=" * 80)
print(f"COUPLING SYMMETRY SWEEP — {CV_FOLDS}-FOLD CV, {EPOCHS} EPOCHS")
print("=" * 80)
header = (f"{'Condition':<30s} {'1step_R²':>9s} {'1step_med':>9s} "
          f"{'LOO_mean':>9s} {'LOO_med':>9s} {'Time(s)':>8s}")
print(header)
print("-" * len(header))
for row in summary_rows:
    name = row.get("condition", "?")
    r2 = row.get("cv_onestep_r2_mean", float('nan'))
    r2m = row.get("cv_onestep_r2_median", float('nan'))
    loo = row.get("cv_loo_r2_mean", float('nan'))
    loom = row.get("cv_loo_r2_median", float('nan'))
    t = row.get("elapsed_sec", 0)
    print(f"{name:<30s} {r2:9.4f} {r2m:9.4f} {loo:9.4f} {loom:9.4f} {t:8.0f}")

# Save combined summary
with open(os.path.join(SAVE_ROOT, "sweep_summary.json"), "w") as f:
    json.dump(summary_rows, f, indent=2)
print(f"\nSaved to {SAVE_ROOT}/sweep_summary.json")
