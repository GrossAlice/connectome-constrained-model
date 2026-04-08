#!/usr/bin/env python
"""
Sweep: break coupling symmetry  (evidence-informed v2)
=======================================================
Problem: all 302 neurons share 9 coupling params (G, a_sv×2, tau_sv×2,
a_dcv×2, tau_dcv×2).  Every synapse carries the same effective weight.

Evidence base (13 sweeps, ~130+ conditions):
────────────────────────────────────────────
 ✓ softplus best activation       (nonlinearity_sweep: LOO_w 0.145 vs sigmoid 0.140)
 ✓ More epochs help               (30→60→100ep steadily improves)
 ✓ Kitchen-sink 100ep best-ever   (followup_sweep: LOO_w = 0.1519)
 ✓ scalar_G improved LOO          (loo_v2: scalar_G LOO_w=0.1727 vs per-neuron-G 0.1336)
 ✗ learn_reversals: NO-OP         (LOO unchanged in loo_v3_run3 + nonlinearity)
 ✗ lowrank_rank: NO-OP            (r=5,10,20 all identical; nonlinearity + constraint_feat)
 ✗ noise_corr_rank: NO-OP         (rank 5,10 identical; nonlinearity + constraint_feat)
 ✗ input_noise alone: NO-OP       (σ=0.02-0.10; followup + nonlinearity)
 ✗ coupling_dropout alone: NO-OP  (combined_sweep)
 ✗ Free/per-neuron tau: NO-OP     (tau_kernel_sweep flat; loo_v2 1tau=2tau=baseline)
 ✗ MLP: removed                       (marginal; hurts generalisation)
 ✗ Rollout: marginal for LOO      (helps 1-step only; combined_sweep)
 ✗ dynamics_l2: NO-OP             (loo_v3: int_l2_01 identical to baseline)
 ✗ loo_aux_weight: marginal       (loo_v2: hi_loo_aux LOO_w 0.137 vs 0.134; 20ep: same)
 ✗ learnW (at low floor): NO-OP   (lamfloor_sweep: learnW identical to base)
 ✗ synLR multiplier: NO-OP        (20ep + lamfloor: 1.0/5.0/10.0 identical)
 ✗ Linear activation: DESTROYS    (loo_v2: LOO_w 0.066 vs 0.134; nonlinearity: tanh 0.037)
 ✗ lambda_floor>0.1: hurts        (lamfloor: 0.1 optimal, higher crushes 1step)
 ✗ graph_poly alone: marginal     (followup_sweep: poly3 LOO_w≈0.140)

NEVER PROPERLY TESTED (directly addresses 9-param symmetry):
  ▸ per_neuron_amplitudes: a_sv becomes (N, r) not (r,) → +604 params per syn type
    · constraint_feature_sweep had it as BASELINE but ALL conditions killed
    · Was tested ONCE with old code (forward_seq bug) → LOO dropped (invalid)
  ▸ edge_specific_G: per-edge gap conductance → +2195 params
    · CAUTION: loo_v2 showed scalar_G > per-neuron-G (fewer params → better LOO)
    · But current code is very different (softplus, growth enc, prior_step)
    · Still worth testing with proper regularisation

Conditions:
───────────
  BLOCK A: per_neuron_amplitudes (the primary untested variable)
   T00: softplus baseline (60ep)      — reference with best activation
   T01: per_neuron_amplitudes          — break amplitude symmetry
   T03: ks + pn_amp                   — best recipe + amplitude break

  BLOCK B: edge_specific_G (secondary, risk of overfitting per loo_v2)
   T04: edge_specific_G               — break gap-junction symmetry

  BLOCK C: combinations
   T06: pn_amp + edge_G               — both symmetry breakers
   T08: ks + pn_amp + edge_G          — best recipe + both

  BLOCK D: longer training with best topology
   T09: pn_amp at 100ep               — more epochs for per-neuron amps
   T10: ks + pn_amp 100ep             — kitchen_sink + per-neuron amps, long
   T11: full combo at 100ep           — pn_amp+edge_G, 100ep
"""
import sys, os, time, shutil, json
sys.path.insert(0, "/home/agross/Downloads/connectome-constrained model/src")

import torch
import numpy as np
from stage2.config import make_config
from stage2.train import train_stage2_cv

H5 = "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5"
SAVE_ROOT = "output_plots/stage2/coupling_symmetry_sweep"

# Kitchen-sink recipe (best-ever from followup_sweep: LOO_w=0.1519)
_KS = dict(
    chemical_synapse_activation="softplus",
    graph_poly_order=3,
    input_noise_sigma=0.05,
    coupling_dropout=0.1,
    rollout_weight=0.3,
    rollout_steps=15,
    rollout_starts=4,
)

# ---------- condition definitions ----------
CONDITIONS = {
    # --- BLOCK A: per_neuron_amplitudes (primary) ---
    "T00_softplus_baseline": dict(
        chemical_synapse_activation="softplus",
        _epochs=60,
    ),

    "T01_pn_amp": dict(
        chemical_synapse_activation="softplus",
        per_neuron_amplitudes=True,
        _epochs=60,
    ),

    "T03_ks+pn_amp": dict(
        **_KS,
        per_neuron_amplitudes=True,
        _epochs=60,
    ),

    # --- BLOCK B: edge_specific_G (risk of overfitting) ---
    "T04_edge_G": dict(
        chemical_synapse_activation="softplus",
        edge_specific_G=True,
        _epochs=60,
    ),

    # --- BLOCK C: combinations ---
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

    # --- BLOCK D: longer training with best topology ---
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

    print(f"\n{'='*60}\n[RUN] {cond_name}  epochs={epochs}  overrides={overrides}\n{'='*60}",
          flush=True)

    # Build config
    cfg = make_config(H5, **overrides)
    cfg.seed = 42
    cfg.num_epochs = epochs
    cfg.network_strength_floor = 1.0
    cfg.network_strength_target = 0.8

    t0 = time.time()
    results = train_stage2_cv(cfg, save_dir=save_dir)
    elapsed = time.time() - t0

    # Collect results
    row = {"condition": cond_name, "elapsed_sec": round(elapsed, 1),
           "epochs": epochs, "overrides": {k: str(v) for k, v in overrides.items()}}
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
    for k in ["cv_r2", "cv_mse", "loo_r2_w", "loo_r2"]:
        if k in row:
            print(f"  {k}: {row[k]:.4f}")

# ---------- Final summary table ----------
print("\n\n" + "=" * 80)
print("COUPLING SYMMETRY SWEEP — FINAL SUMMARY")
print("=" * 80)
header = f"{'Condition':<35s} {'1step_R²':>9s} {'LOO_w':>9s} {'LOO':>9s} {'MSE':>9s} {'Time(s)':>8s}"
print(header)
print("-" * len(header))
for row in summary_rows:
    name = row.get("condition", "?")
    r2 = row.get("cv_r2", float('nan'))
    loo_w = row.get("loo_r2_w", float('nan'))
    loo = row.get("loo_r2", float('nan'))
    mse = row.get("cv_mse", float('nan'))
    t = row.get("elapsed_sec", 0)
    print(f"{name:<35s} {r2:9.4f} {loo_w:9.4f} {loo:9.4f} {mse:9.4f} {t:8.0f}")

# Save combined summary
with open(os.path.join(SAVE_ROOT, "sweep_summary.json"), "w") as f:
    json.dump(summary_rows, f, indent=2)
print(f"\nSaved to {SAVE_ROOT}/sweep_summary.json")
