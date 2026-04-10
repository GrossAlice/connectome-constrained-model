#!/usr/bin/env python
"""Smoke-test all 10 typed lag routing modes (A–J) — 1 epoch each."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stage2.config import make_config
from stage2.train import train_stage2_cv

H5 = "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5"

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

BASE = dict(
    lag_order=5,
    lag_neighbor=True,
    lag_connectome_mask="all",
    lag_neighbor_per_type=True,
    num_epochs=1,
    skip_free_run=True,
    eval_loo_subset_size=3,
    eval_loo_subset_mode="variance",
    num_folds=2,
    seed=42,
)

results = []
for mode in MODES:
    label = LABELS[mode]
    tag = f"{label}:{mode}"
    cfg = make_config(H5, lag_route_mode=mode, **BASE)
    t0 = time.time()
    try:
        res = train_stage2_cv(cfg, save_dir=f"/tmp/smoke_lag_route/{mode}")
        dt = time.time() - t0
        os1 = res.get("cv_onestep_r2_mean", float("nan"))
        loo = res.get("cv_loo_r2_mean", float("nan"))
        print(f"  OK  {tag:40s}  1step={os1:.4f}  LOO={loo:.4f}  ({dt:.1f}s)")
        results.append((tag, "OK", os1, loo, dt))
    except Exception as e:
        dt = time.time() - t0
        print(f"  FAIL {tag:40s}  {e}  ({dt:.1f}s)")
        import traceback; traceback.print_exc()
        results.append((tag, f"FAIL: {e}", None, None, dt))

print("\n" + "="*70)
print("SMOKE TEST SUMMARY")
print("="*70)
for tag, status, os1, loo, dt in results:
    if os1 is not None:
        print(f"  {tag:40s}  {status:4s}  1step={os1:.4f}  LOO={loo:.4f}  ({dt:.1f}s)")
    else:
        print(f"  {tag:40s}  {status}")

n_pass = sum(1 for _, s, _, _, _ in results if s == "OK")
n_total = len(results)
print(f"\n{n_pass}/{n_total} passed")
