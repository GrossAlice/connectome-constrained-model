#!/usr/bin/env python
"""
Ridge warm-start experiment.

Tests whether initialising S2 lag params from Ridge closed-form
closes the optimisation gap (S2 ≈ 0.40 vs Ridge ≈ 0.48).

Conditions
----------
  W0  baseline          Cold start, biophys ON, 60 ep      (= S2_equiv_lag5, expect ~0.40)
  W1  warmstart_biophys Warm start, biophys ON, 60 ep      (biophys can improve on Ridge init)
  W2  warmstart_nosgd   Warm start, biophys OFF, 0 ep      (pure Ridge → S2 mapping, expect ~0.48)
  W3  warmstart_sgd60   Warm start, biophys OFF, 60 ep     (can Adam maintain Ridge-level perf?)
  W4  warmstart_gain    Warm start, biophys ON, gain, 60ep (g_i lets λ·g=1 naturally)

Reference baselines:
  S2_equiv_lag5 (60ep, biophys ON):   LOO = 0.3996
  U3_full_lag5  (Ridge):              LOO = 0.4802
  N0_lagonly    (60ep, biophys OFF):  LOO = 0.3273
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
SAVE_ROOT = SRC / "output_plots" / "stage2" / "warmstart_test"

# ── Shared base kwargs ───────────────────────────────────────────
BASE = dict(
    # Lag: match Ridge U3_full_lag5
    lag_order              = 5,
    lag_neighbor           = True,
    lag_connectome_mask    = "all",
    lag_neighbor_per_type  = False,     # single union mask (like Ridge)
    lag_neighbor_activation = "none",
    # Biophysics ON by default
    use_gap_junctions      = True,
    use_sv_synapses        = True,
    use_dcv_synapses       = True,
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
    "W0_baseline": dict(
        # Cold start, biophys ON (= S2_equiv_lag5 reference)
        ridge_warmstart=False,
    ),

    "W1_warmstart_biophys": dict(
        # Warm start from Ridge, biophys ON, 60 ep SGD
        ridge_warmstart=True,
        learn_input_gain=True,
    ),

    "W2_warmstart_nosgd": dict(
        # Warm start, biophys OFF, 0 epochs (pure Ridge→S2 mapping test)
        ridge_warmstart=True,
        learn_input_gain=True,
        use_gap_junctions=False,
        use_sv_synapses=False,
        use_dcv_synapses=False,
        num_epochs=0,
    ),

    "W3_warmstart_sgd60": dict(
        # Warm start, biophys OFF, 60 ep (can Adam maintain Ridge perf?)
        ridge_warmstart=True,
        learn_input_gain=True,
        use_gap_junctions=False,
        use_sv_synapses=False,
        use_dcv_synapses=False,
    ),

    "W4_warmstart_gain": dict(
        # Warm start, biophys ON, gain, 60 ep
        ridge_warmstart=True,
        learn_input_gain=True,
        per_neuron_l2=True,
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
print("FINAL SUMMARY  (Ridge warm-start experiment)")
print(f"{'='*60}")
print(f"{'Condition':<30} {'1step':>8} {'LOO':>8} {'med_LOO':>8} {'time':>6}")
print("-" * 70)
for name, s in results_all.items():
    print(f"{name:<30} {s.get('cv_onestep_r2_mean',0):.4f}   "
          f"{s.get('cv_loo_r2_mean',0):.4f}   "
          f"{s.get('cv_loo_r2_median',0):.4f}   "
          f"{s.get('elapsed_min',0):5.1f}m")
print("-" * 70)
print("Reference baselines:")
print("  S2_equiv_lag5 (60ep, biophys ON):  LOO = 0.3996")
print("  N0_lagonly    (60ep, biophys OFF):  LOO = 0.3273")
print("  U3_full_lag5  (Ridge):             LOO = 0.4802")
print("  U3_full_lag5_ols (OLS):            LOO = 0.4370")
print(f"{'='*60}")

(SAVE_ROOT / "all_summaries.json").write_text(
    json.dumps(results_all, indent=2, default=str))
