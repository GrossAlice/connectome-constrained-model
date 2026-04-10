#!/usr/bin/env python
"""
Sweep: lag-neighbor routing strategies.
───────────────────────────────────────
Test all 8 ways of injecting I_lag,nbr into the biophysical currents:
  additive           — baseline (original sum inside λ·g gate)
  gap_laplacian      — lag drives L @ u(t-k) added to I_gap
  iir_augmented      — lag injects phi(u(t-k)) into s_sv / s_dcv states
  per_current        — separate learned lag kernels per gap / sv / dcv
  conductance_modulated — lag modulates total I_coupling multiplicatively
  gap_only           — lag neighbor via gap-junction topology only
  chem_only          — lag neighbor via chemical-synapse topology only
  weighted_topology  — learnable convex combination of all three topologies

Each routing mode is tested with the best known base config.
"""

import json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stage2.config import make_config
from stage2.train import train_stage2_cv

H5 = "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5"
OUT_ROOT = Path("output_plots/stage2/sweep_lag_nbr_routing")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ── Best-known base config ──
BASE = dict(
    lag_order=10,
    lag_neighbor=True,
    lag_connectome_mask="all",
    lag_neighbor_per_type=True,
    dynamics_l2=1e-3,
    rollout_steps=10,
    rollout_weight=0.3,
    warmstart_rollout=True,
    num_epochs=100,
    lr_schedule="cosine",
    input_noise_sigma=0.05,
    seed=42,
    skip_free_run=True,
    eval_loo_subset_size=30,
    eval_loo_subset_mode="variance",
)

# For routing modes that don't use per-type, disable that flag
# (per_type is only relevant for additive routing)
ROUTING_MODES = [
    "additive",
    "gap_laplacian",
    "iir_augmented",
    "per_current",
    "conductance_modulated",
    "gap_only",
    "chem_only",
    "weighted_topology",
]

CONDITIONS = {}
for mode in ROUTING_MODES:
    tag = f"R_{mode}"
    overrides = dict(lag_nbr_routing=mode)
    # gap_laplacian & gap_only don't need per_type since they only use gap topology
    if mode in ("gap_laplacian", "gap_only"):
        overrides["lag_neighbor_per_type"] = False
    # chem_only doesn't use gap at all
    if mode == "chem_only":
        overrides["lag_neighbor_per_type"] = False
    CONDITIONS[tag] = overrides

# Also test routing modes with larger lag order
for mode in ["per_current", "gap_laplacian", "weighted_topology"]:
    tag = f"R_{mode}_lag12"
    CONDITIONS[tag] = dict(lag_nbr_routing=mode, lag_order=12)

# Seed variations on the most promising modes
for mode in ["per_current", "weighted_topology"]:
    for seed in [123, 7]:
        tag = f"R_{mode}_s{seed}"
        CONDITIONS[tag] = dict(lag_nbr_routing=mode, seed=seed)

all_results = {}

for tag, overrides in CONDITIONS.items():
    out_dir = OUT_ROOT / tag
    summary_path = out_dir / "summary.json"

    if summary_path.exists():
        print(f"\n{'='*60}\n  SKIP {tag} (already done)\n{'='*60}")
        try:
            all_results[tag] = json.loads(summary_path.read_text())
        except Exception:
            pass
        continue

    merged = {**BASE, **overrides}
    print(f"\n{'='*60}\n  {tag}\n  overrides: {overrides}\n{'='*60}")

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = make_config(H5, **merged)
    t0 = time.time()

    try:
        result = train_stage2_cv(cfg, save_dir=str(out_dir))
    except Exception as e:
        import traceback
        print(f"  ERROR in {tag}: {e}")
        (out_dir / "ERROR.txt").write_text(traceback.format_exc())
        continue

    elapsed = (time.time() - t0) / 60.0

    summary = dict(
        condition=tag,
        overrides=overrides,
        elapsed_min=round(elapsed, 1),
        cv_onestep_r2_mean=result.get("cv_onestep_r2_mean"),
        cv_onestep_r2_median=result.get("cv_onestep_r2_median"),
        cv_fr_r2_mean=result.get("cv_fr_r2_mean"),
        cv_fr_r2_median=result.get("cv_fr_r2_median"),
        best_fold_idx=result.get("best_fold_idx"),
        cv_loo_r2_mean=result.get("cv_loo_r2_mean"),
        cv_loo_r2_median=result.get("cv_loo_r2_median"),
    )
    summary_path.write_text(json.dumps(summary, indent=2))
    all_results[tag] = summary

    loo = summary.get("cv_loo_r2_mean", float("nan"))
    os1 = summary.get("cv_onestep_r2_mean", float("nan"))
    print(f"\n  >>> {tag}: 1step={os1:.4f}  LOO={loo:.4f}  ({elapsed:.1f} min)")
    if loo > 0.4802:
        print(f"  *** BEATS RIDGE (0.4802)! ***")

# ── Final summary ──
print("\n" + "="*80)
print("SWEEP COMPLETE — sorted by LOO R²")
print("="*80)
ranked = sorted(all_results.items(), key=lambda kv: -(kv[1].get("cv_loo_r2_mean") or -99))
for tag, s in ranked:
    loo = s.get("cv_loo_r2_mean", float("nan"))
    os1 = s.get("cv_onestep_r2_mean", float("nan"))
    flag = " ***" if loo > 0.4802 else ""
    print(f"  {tag:35s}  1step={os1:.4f}  LOO={loo:.4f}{flag}")

print(f"\nRidge target: 0.4802")
(OUT_ROOT / "all_results.json").write_text(
    json.dumps(all_results, indent=2, default=str))
