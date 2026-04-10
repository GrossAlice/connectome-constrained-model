#!/usr/bin/env python
"""
Stage2 ODE — no biophysics + gain ablation.

All biophysical currents OFF (I_gap=I_sv=I_dcv=0), only lag terms remain.
Tests gain g_i and per-neuron L2 in isolation — sole surviving difference
vs Ridge is optimisation (SGD vs closed-form).

  N0_lagonly          — biophys off, baseline
  N1_gain             — + learn_input_gain
  N2_gain_pnl2        — + per_neuron_l2
  N3_gain_pnl2_200    — 200 epochs

Reference baselines:
  S2_equiv_lag5  (60ep):  LOO = 0.3996
  U3_full_lag5   (Ridge): LOO = 0.4802
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))
os.chdir(SRC)

from stage2.config import make_config
from stage2.train import train_stage2_cv

H5 = "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5"
SAVE_ROOT = SRC / "output_plots" / "stage2" / "no_biophys_gain"

# ── Shared base kwargs ───────────────────────────────────────────
BASE = dict(
    # I_lag: exact match to U3 Ridge
    lag_order              = 5,
    lag_neighbor           = True,
    lag_connectome_mask    = "all",
    lag_neighbor_per_type  = False,
    lag_neighbor_activation = "none",
    # Biophysical channels ALL OFF
    use_gap_junctions      = False,
    use_sv_synapses        = False,
    use_dcv_synapses       = False,
    chemical_synapse_mode  = "iir",
    chemical_synapse_activation = "sigmoid",
    edge_specific_G        = True,
    per_neuron_amplitudes  = True,
    G_init_mode            = "corr_weighted",
    W_init_mode            = "corr_weighted",
    graph_poly_order       = 1,
    learn_reversals        = False,
    synapse_lag_taps       = 0,
    # Training
    num_epochs             = 60,
    cv_folds               = 3,
    dynamics_l2            = 1e-3,
    seed                   = 42,
    synaptic_lr_multiplier = 5.0,
    # No rollout
    rollout_steps          = 0,
    rollout_weight         = 0.0,
    rollout_starts         = 0,
    skip_free_run          = True,
)

CONDITIONS = {
    "N0_lagonly": dict(),

    "N1_gain": dict(
        learn_input_gain=True,
    ),

    "N2_gain_pnl2": dict(
        learn_input_gain=True,
        per_neuron_l2=True,
    ),

    "N3_gain_pnl2_200": dict(
        learn_input_gain=True,
        per_neuron_l2=True,
        num_epochs=200,
    ),
}

results_all = {}
for cond_name, overrides in CONDITIONS.items():
    print(f"\n{'='*60}")
    print(f"  CONDITION: {cond_name}")
    print(f"  overrides: {overrides}")
    print(f"{'='*60}\n")

    save_dir = SAVE_ROOT / cond_name
    save_dir.mkdir(parents=True, exist_ok=True)

    kw = {**BASE, **overrides}
    cfg = make_config(H5, **kw)

    t0 = time.time()
    results = train_stage2_cv(cfg, save_dir=str(save_dir))
    elapsed = (time.time() - t0) / 60

    summary = {
        "condition": cond_name,
        "overrides": overrides,
        "elapsed_min": round(elapsed, 1),
    }
    for k in ["cv_onestep_r2_mean", "cv_onestep_r2_median",
              "cv_loo_r2_mean", "cv_loo_r2_median", "best_fold_idx"]:
        if k in results:
            summary[k] = results[k]

    (save_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    results_all[cond_name] = summary
    print(f"\n  → {cond_name}: 1step={summary.get('cv_onestep_r2_mean','?'):.4f}  "
          f"LOO={summary.get('cv_loo_r2_mean','?'):.4f}  "
          f"med_LOO={summary.get('cv_loo_r2_median','?'):.4f}  "
          f"({elapsed:.1f} min)\n")

# ── Final summary ────────────────────────────────────────────────
print(f"\n{'='*60}")
print("FINAL SUMMARY  (no biophysics + gain + per-neuron L2)")
print(f"{'='*60}")
print(f"{'Condition':<25} {'1step':>8} {'LOO':>8} {'med_LOO':>8} {'time':>6}")
print("-" * 65)
for name, s in results_all.items():
    print(f"{name:<25} {s.get('cv_onestep_r2_mean',0):.4f}   "
          f"{s.get('cv_loo_r2_mean',0):.4f}   "
          f"{s.get('cv_loo_r2_median',0):.4f}   "
          f"{s.get('elapsed_min',0):5.1f}m")
print("-" * 65)
print("Reference baselines:")
print("  S2_equiv_lag5 (60ep):         LOO = 0.3996")
print("  S2_200ep (200ep, l2=1e-3):    LOO = 0.4266")
print("  U3_full_lag5 (Ridge):         LOO = 0.4802")
print(f"{'='*60}")

(SAVE_ROOT / "all_summaries.json").write_text(
    json.dumps(results_all, indent=2, default=str))
