#!/usr/bin/env python
"""
Sweep: typed lag routing v2 — all 10 modes (A–J).
══════════════════════════════════════════════════
Each mode controls how the pooled neighbour-lag signal I_lag,nbr
is decomposed and routed into the biophysically typed coupling
currents I_gap, I_sv, I_dcv.

Modes
-----
  A  pooled_baseline          — same as legacy additive (sanity check)
  B  typed_add                — per-type G matrices added to matching currents
  C  typed_gain_scalar        — B + learnable scalar β_e, β_sv, β_dcv
  D  typed_gain_per_neuron    — B + per-neuron gains β(i) per type
  E  typed_residual           — B + small η correction from self-lag
  F  gap_laplacian_typed_chem — Laplacian Σ_k α_k·L@u(t-k) for gap,
                                per-edge G for chemical
  G  chem_inside_channel      — inject lag into chemical pathway via T_eff @ φ(u_delayed)
  H  typed_synapse_lag        — linear FIR through each topology matrix
  I  delayed_source           — soft-delay (softmax attention) per pathway
  J  routing_matrix           — per-type G + learnable 3×3 routing matrix R

Usage
-----
  .venv/bin/python -u -m scripts.sweep_lag_route_modes
"""

import json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stage2.config import make_config
from stage2.train import train_stage2_cv

H5 = "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5"
OUT_ROOT = Path("output_plots/stage2/sweep_lag_route_modes")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ── Best known base config ─────────────────────────────────────────
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

# ── All 10 modes ───────────────────────────────────────────────────
MODES = [
    "pooled_baseline",            # A
    "typed_add",                  # B
    "typed_gain_scalar",          # C
    "typed_gain_per_neuron",      # D
    "typed_residual",             # E
    "gap_laplacian_typed_chem",   # F
    "chem_inside_channel",        # G
    "typed_synapse_lag",          # H
    "delayed_source",             # I
    "routing_matrix",             # J
]

LABELS = dict(zip(MODES, "ABCDEFGHIJ"))

CONDITIONS = {}

# 1) Each mode with default lag_order=10
for mode in MODES:
    tag = f"LR_{LABELS[mode]}_{mode}"
    CONDITIONS[tag] = dict(lag_route_mode=mode)

# 2) Top candidates with lag_order=12
for mode in ["typed_add", "typed_gain_scalar", "gap_laplacian_typed_chem",
             "routing_matrix", "delayed_source"]:
    tag = f"LR_{LABELS[mode]}_{mode}_lag12"
    CONDITIONS[tag] = dict(lag_route_mode=mode, lag_order=12)

# 3) Seed variations on most promising structural modes
for mode in ["typed_add", "typed_gain_scalar", "routing_matrix"]:
    for seed in [123, 7]:
        tag = f"LR_{LABELS[mode]}_{mode}_s{seed}"
        CONDITIONS[tag] = dict(lag_route_mode=mode, seed=seed)

# 4) Legacy additive baseline for comparison
CONDITIONS["LR_legacy_additive"] = dict(lag_route_mode="off", lag_nbr_routing="additive")

# ── Run sweep ──────────────────────────────────────────────────────
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
    if loo and loo > 0.4802:
        print(f"  *** BEATS RIDGE (0.4802)! ***")

# ── Final summary ──────────────────────────────────────────────────
print("\n" + "="*80)
print("SWEEP COMPLETE — sorted by LOO R²")
print("="*80)
ranked = sorted(
    all_results.items(),
    key=lambda kv: -(kv[1].get("cv_loo_r2_mean") or -99),
)
for tag, s in ranked:
    loo = s.get("cv_loo_r2_mean", float("nan"))
    os1 = s.get("cv_onestep_r2_mean", float("nan"))
    flag = " ***" if loo and loo > 0.4802 else ""
    print(f"  {tag:45s}  1step={os1:.4f}  LOO={loo:.4f}{flag}")

print(f"\nRidge target: 0.4802")
(OUT_ROOT / "all_results.json").write_text(
    json.dumps(all_results, indent=2, default=str))
