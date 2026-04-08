#!/usr/bin/env python
"""
Sweep: linear stage2 variants to match conn-ridge performance
=============================================================
Conn-ridge LOO_w ≈ 0.369, 1-step R² median ≈ 0.614.

Two families:

FAMILY A — "Linear conn-ridge analog"
  edge_specific_G=True, chemical_synapse_activation="identity", lag_order=5
  Per-neuron λ_i, I0_i already learned by default.
  This gives per-edge gap weights + linear chemical synapses + AR(5) self-lags.
  Vary regularisation strength to find the sweet spot.

FAMILY B — "Gap-junction only"
  edge_specific_G=True, NO chemical synapses (tau_sv_init=(), tau_dcv_init=())
  Just gap junctions + leak + bias.  Simplest possible connectome model.
  With lag_order for temporal memory.

Both use skip_free_run=True, skip_cv_loo=False for faster turnaround.
"""
import sys, os, time, shutil, json
os.environ["TORCHDYNAMO_DISABLE"] = "1"
sys.path.insert(0, "/home/agross/Downloads/connectome-constrained model/src")

import torch
import numpy as np
from stage2.config import make_config
from stage2.train import train_stage2_cv

H5 = "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5"
SAVE_ROOT = "output_plots/stage2/linear_ridge_match"

# ─── Common settings ───────────────────────────────────────────────────
_COMMON = dict(
    seed=42,
    cv_folds=3,
    num_epochs=60,
    network_strength_floor=1.0,
    network_strength_target=0.8,
    skip_free_run=True,
)

# ─── FAMILY A: Linear conn-ridge analog ────────────────────────────────
# edge_specific_G + identity φ + lag_order=5
_LINEAR_BASE = dict(
    edge_specific_G=True,
    chemical_synapse_activation="identity",
    lag_order=5,
    per_neuron_amplitudes=True,  # per-neuron a_sv, a_dcv
)

# ─── FAMILY B: Gap-junction only ──────────────────────────────────────
# No chemical synapses at all
_GAP_ONLY_BASE = dict(
    edge_specific_G=True,
    tau_sv_init=(),      # r_sv = 0 → no SV synapses
    tau_dcv_init=(),     # r_dcv = 0 → no DCV synapses
    a_sv_init=(),
    a_dcv_init=(),
)

CONDITIONS = {
    # ── FAMILY A: Linear stage2 matching conn-ridge ──────────────────

    # A0: baseline — minimal regularisation
    "A0_linear_base": dict(
        **_LINEAR_BASE,
        dynamics_l2=0.0,
    ),

    # A1: light L2 on dynamics
    "A1_linear_l2_1e-3": dict(
        **_LINEAR_BASE,
        dynamics_l2=1e-3,
    ),

    # A2: stronger L2
    "A2_linear_l2_1e-2": dict(
        **_LINEAR_BASE,
        dynamics_l2=1e-2,
    ),

    # A3: heavy L2 (match ridge α)
    "A3_linear_l2_1e-1": dict(
        **_LINEAR_BASE,
        dynamics_l2=0.1,
    ),

    # A4: no lag (just edge_G + linear synapses)
    "A4_linear_no_lag": dict(
        edge_specific_G=True,
        chemical_synapse_activation="identity",
        per_neuron_amplitudes=True,
        dynamics_l2=1e-3,
    ),

    # A5: with neighbor lags (full conn-ridge equivalent)
    "A5_linear_lag_neighbor": dict(
        **_LINEAR_BASE,
        lag_neighbor=True,
        dynamics_l2=1e-3,
    ),

    # A6: smaller LR (avoid overfitting per-edge G)
    "A6_linear_lr_3e-4": dict(
        **_LINEAR_BASE,
        learning_rate=3e-4,
        dynamics_l2=1e-3,
    ),

    # A7: 100 epochs + moderate L2
    "A7_linear_100ep": dict(
        **_LINEAR_BASE,
        dynamics_l2=1e-3,
        _epochs=100,
    ),

    # ── FAMILY B: Gap-junction only ─────────────────────────────────

    # B0: pure gap junctions (no chem synapses, no lag)
    "B0_gap_only": dict(
        **_GAP_ONLY_BASE,
        dynamics_l2=0.0,
    ),

    # B1: gap + light L2
    "B1_gap_l2_1e-3": dict(
        **_GAP_ONLY_BASE,
        dynamics_l2=1e-3,
    ),

    # B2: gap + stronger L2
    "B2_gap_l2_1e-2": dict(
        **_GAP_ONLY_BASE,
        dynamics_l2=1e-2,
    ),

    # B3: gap + lag_order=5 (self-lags compensate for missing chem syn)
    "B3_gap_lag5": dict(
        **_GAP_ONLY_BASE,
        lag_order=5,
        dynamics_l2=1e-3,
    ),

    # B4: gap + lag_order=5 + neighbor lags
    "B4_gap_lag5_neighbor": dict(
        **_GAP_ONLY_BASE,
        lag_order=5,
        lag_neighbor=True,
        dynamics_l2=1e-3,
    ),

    # B5: gap + scalar G (no per-edge — simplest possible)
    "B5_gap_scalar_G": dict(
        tau_sv_init=(),
        tau_dcv_init=(),
        a_sv_init=(),
        a_dcv_init=(),
        edge_specific_G=False,
        dynamics_l2=0.0,
    ),

    # B6: gap scalar + lag5
    "B6_gap_scalar_lag5": dict(
        tau_sv_init=(),
        tau_dcv_init=(),
        a_sv_init=(),
        a_dcv_init=(),
        edge_specific_G=False,
        lag_order=5,
        dynamics_l2=1e-3,
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

    # Extract epochs override
    epochs = overrides.pop("_epochs", _COMMON.get("num_epochs", 60))

    print(f"\n{'='*60}\n[RUN] {cond_name}  epochs={epochs}\n  overrides={overrides}\n{'='*60}",
          flush=True)

    # Build config: common + condition overrides
    all_kw = {**_COMMON, **overrides}
    all_kw["num_epochs"] = epochs
    cfg = make_config(H5, **all_kw)

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

# ─── Final summary ────────────────────────────────────────────────────
print("\n\n" + "=" * 80)
print("LINEAR RIDGE-MATCH SWEEP — FINAL SUMMARY")
print("=" * 80)
header = f"{'Condition':<30s} {'1step_R²':>9s} {'LOO_w':>9s} {'LOO':>9s} {'MSE':>9s} {'Time(s)':>8s}"
print(header)
print("-" * len(header))
for row in summary_rows:
    name = row.get("condition", "?")
    r2 = row.get("cv_r2", float('nan'))
    loo_w = row.get("loo_r2_w", float('nan'))
    loo = row.get("loo_r2", float('nan'))
    mse = row.get("cv_mse", float('nan'))
    t = row.get("elapsed_sec", 0)
    print(f"{name:<30s} {r2:9.4f} {loo_w:9.4f} {loo:9.4f} {mse:9.4f} {t:8.0f}")

with open(os.path.join(SAVE_ROOT, "sweep_summary.json"), "w") as f:
    json.dump(summary_rows, f, indent=2)
print(f"\nSaved to {SAVE_ROOT}/sweep_summary.json")
