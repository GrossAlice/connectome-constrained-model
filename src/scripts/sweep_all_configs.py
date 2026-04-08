#!/usr/bin/env python
"""
Comprehensive config sweep — 3-fold CV, 50 epochs
===================================================
Tests every config option that can improve coupling expressivity / LOO.

Block A – Baselines (sigmoid default):
  T00  sigmoid baseline (pure defaults)
  T01  softplus baseline

Block B – Temporal memory (closes AR(1) → AR(K) gap):
  T02  lag_order=3  (self-lags only)
  T03  lag_order=5  (self-lags only)
  T04  lag_order=5 + lag_neighbor  (connectome-sparse neighbor lags)
  T05  lag_order=3 + lag_neighbor

Block C – Coupling expressivity:
  T06  per_neuron_amplitudes  (a_sv/a_dcv become (N, r))
  T07  learn_reversals (per_neuron)
  T08  edge_specific_G
  T09  pn_amp + learn_reversals
  T10  pn_amp + edge_G

Block D – Non-connectome coupling / regularisation:
  T11  noise_corr_rank=5  (low-rank correlated noise)
  T12  graph_poly_order=3  (multi-hop gap junctions)
  T13  lowrank_rank=10  (low-rank dense coupling)
  T14  (removed — residual MLP deleted)
  T15  coupling_dropout=0.1

Block E – Best combos (lag + coupling expressivity):
  T16  lag5 + pn_amp
  T17  lag5 + pn_amp + learn_rev
  T18  lag5_nbr + pn_amp
  T19  lag5_nbr + pn_amp + learn_rev
  T20  lag5_nbr + pn_amp + noise_corr5

Block F – Regularised combos:
  T21  lag5_nbr + pn_amp + coupling_dropout=0.1
  T22  lag5_nbr + pn_amp + dynamics_l2=0.01
  T23  lag5_nbr + pn_amp + input_noise=0.05
"""
import sys, os, time, shutil, json
sys.path.insert(0, "/home/agross/Downloads/connectome-constrained model/src")

# ── disable torch.compile (triton not installed → BackendCompilerFailed spam) ──
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
import numpy as np
from stage2.config import make_config
from stage2.train import train_stage2_cv

H5 = "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5"
SAVE_ROOT = "output_plots/stage2/config_sweep_all"

CV_FOLDS = 3
EPOCHS   = 50

# ---------- condition definitions ----------
CONDITIONS = {
    # ═══ Block A: Baselines ═══
    "T00_sigmoid_baseline": dict(),
    "T01_softplus_baseline": dict(
        chemical_synapse_activation="softplus",
    ),

    # ═══ Block B: Temporal memory ═══
    "T02_lag3_self": dict(
        lag_order=3,
    ),
    "T03_lag5_self": dict(
        lag_order=5,
    ),
    "T04_lag5_nbr": dict(
        lag_order=5,
        lag_neighbor=True,
    ),
    "T05_lag3_nbr": dict(
        lag_order=3,
        lag_neighbor=True,
    ),

    # ═══ Block C: Coupling expressivity ═══
    "T06_pn_amp": dict(
        per_neuron_amplitudes=True,
    ),
    "T07_learn_rev": dict(
        learn_reversals=True,
    ),
    "T08_edge_G": dict(
        edge_specific_G=True,
    ),
    "T09_pn_amp+learn_rev": dict(
        per_neuron_amplitudes=True,
        learn_reversals=True,
    ),
    "T10_pn_amp+edge_G": dict(
        per_neuron_amplitudes=True,
        edge_specific_G=True,
    ),

    # ═══ Block D: Non-connectome coupling / regularisation ═══
    "T11_noise_corr5": dict(
        noise_corr_rank=5,
    ),
    "T12_graph_poly3": dict(
        graph_poly_order=3,
    ),
    "T13_lowrank10": dict(
        lowrank_rank=10,
    ),
    "T15_coupling_drop": dict(
        _coupling_dropout=0.1,
    ),

    # ═══ Block E: Best combos (lag + coupling) ═══
    "T16_lag5+pn_amp": dict(
        lag_order=5,
        per_neuron_amplitudes=True,
    ),
    "T17_lag5+pn_amp+rev": dict(
        lag_order=5,
        per_neuron_amplitudes=True,
        learn_reversals=True,
    ),
    "T18_lag5nbr+pn_amp": dict(
        lag_order=5,
        lag_neighbor=True,
        per_neuron_amplitudes=True,
    ),
    "T19_lag5nbr+pn_amp+rev": dict(
        lag_order=5,
        lag_neighbor=True,
        per_neuron_amplitudes=True,
        learn_reversals=True,
    ),
    "T20_lag5nbr+pn_amp+ncorr": dict(
        lag_order=5,
        lag_neighbor=True,
        per_neuron_amplitudes=True,
        noise_corr_rank=5,
    ),

    # ═══ Block F: Regularised combos ═══
    "T21_lag5nbr+pn_amp+cdrop": dict(
        lag_order=5,
        lag_neighbor=True,
        per_neuron_amplitudes=True,
        _coupling_dropout=0.1,
    ),
    "T22_lag5nbr+pn_amp+l2": dict(
        lag_order=5,
        lag_neighbor=True,
        per_neuron_amplitudes=True,
        _dynamics_l2=0.01,
    ),
    "T23_lag5nbr+pn_amp+inoise": dict(
        lag_order=5,
        lag_neighbor=True,
        per_neuron_amplitudes=True,
        _input_noise_sigma=0.05,
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

    # Build config — sigmoid defaults, 3-fold CV, 50 epochs
    cfg = make_config(H5, cv_folds=CV_FOLDS, **dyn_overrides)
    cfg.seed = 42
    cfg.num_epochs = EPOCHS
    cfg.network_strength_floor = 1.0
    cfg.network_strength_target = 0.8
    cfg.skip_free_run = True   # only need LOO R²; skip free-run + decomposition
    cfg.skip_final_eval = True # plots need decomposition data; skip them
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
              "cv_freerun_r2_mean", "cv_loo_r2_mean", "cv_loo_r2_median"]:
        if k in row:
            print(f"  {k}: {row[k]:.4f}")

# ---------- Final summary table ----------
print("\n\n" + "=" * 90)
print(f"CONFIG SWEEP — {CV_FOLDS}-FOLD CV, {EPOCHS} EPOCHS — ALL OPTIONS")
print("=" * 90)
header = (f"{'Condition':<35s} {'1step_R²':>9s} {'1step_med':>9s} "
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
    print(f"{name:<35s} {r2:9.4f} {r2m:9.4f} {loo:9.4f} {loom:9.4f} {t:8.0f}")

# Save combined summary
with open(os.path.join(SAVE_ROOT, "sweep_summary.json"), "w") as f:
    json.dump(summary_rows, f, indent=2)
print(f"\nSaved to {SAVE_ROOT}/sweep_summary.json")
