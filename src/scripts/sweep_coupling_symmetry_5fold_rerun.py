#!/usr/bin/env python
"""
Rerun T00 (baseline) and T03 (ks+pn_amp) with 5-fold CV.
Saves to separate directories so original 2-fold results are preserved.
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

CONDITIONS = {
    "T00_softplus_baseline_5fold": dict(
        chemical_synapse_activation="softplus",
        _epochs=60,
    ),
    "T03_ks+pn_amp_5fold": dict(
        **_KS,
        per_neuron_amplitudes=True,
        _epochs=60,
    ),
}

os.makedirs(SAVE_ROOT, exist_ok=True)
summary_rows = []

for cond_name, overrides in CONDITIONS.items():
    save_dir = os.path.join(SAVE_ROOT, cond_name)

    if os.path.exists(os.path.join(save_dir, "summary.json")):
        print(f"\n{'='*60}\n[SKIP] {cond_name} — already done\n{'='*60}")
        with open(os.path.join(save_dir, "summary.json")) as f:
            row = json.load(f)
        summary_rows.append(row)
        continue

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    epochs = overrides.pop("_epochs", 60)

    print(f"\n{'='*60}\n[RUN] {cond_name}  epochs={epochs}  cv_folds=5  overrides={overrides}\n{'='*60}",
          flush=True)

    cfg = make_config(H5, cv_folds=5, **overrides)
    cfg.seed = 42
    cfg.num_epochs = epochs
    cfg.network_strength_floor = 1.0
    cfg.network_strength_target = 0.8

    t0 = time.time()
    results = train_stage2_cv(cfg, save_dir=save_dir)
    elapsed = time.time() - t0

    row = {"condition": cond_name, "elapsed_sec": round(elapsed, 1),
           "epochs": epochs, "cv_folds": 5,
           "overrides": {k: str(v) for k, v in overrides.items()}}
    if results is not None:
        for k, v in results.items():
            if isinstance(v, (float, int)):
                row[k] = round(v, 6) if isinstance(v, float) else v
    summary_rows.append(row)

    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(row, f, indent=2)

    print(f"\n[DONE] {cond_name}  time={elapsed:.0f}s")
    for k in ["cv_onestep_r2_mean", "cv_onestep_r2_median",
              "cv_loo_r2_mean", "cv_loo_r2_windowed_mean"]:
        if k in row:
            print(f"  {k}: {row[k]:.4f}")

print("\n\n" + "=" * 80)
print("5-FOLD RERUN SUMMARY")
print("=" * 80)
header = f"{'Condition':<35s} {'1step_R²':>9s} {'1step_med':>9s} {'LOO_w_mean':>10s} {'LOO_w_med':>9s} {'LOO_mean':>9s} {'Time(s)':>8s}"
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
    print(f"{name:<35s} {r2:9.4f} {r2m:9.4f} {loo_w:10.4f} {loo_wm:9.4f} {loo:9.4f} {t:8.0f}")

with open(os.path.join(SAVE_ROOT, "rerun_5fold_summary.json"), "w") as f:
    json.dump(summary_rows, f, indent=2)
print(f"\nSaved to {SAVE_ROOT}/rerun_5fold_summary.json")
