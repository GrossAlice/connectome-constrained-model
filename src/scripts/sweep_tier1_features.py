#!/usr/bin/env python
"""Sweep Tier-1 features (already implemented, config-only) on top of
corr_weighted init (the current best from the corr_weighted_sweep).

Conditions (all share the C1 base: edge_specific_G, per_neuron_amplitudes,
corr_weighted G+W init):

  T0  base          – C1_corr_init reproduction
  T1  +poly2        – graph_poly_order=2 (2-hop gap-junction diffusion)
  T2  +lowrank5     – lowrank_rank=5 (non-connectome dense coupling)
  T3  +lag5         – lag_order=5, lag_neighbor=True (AR-5 + neighbor lags)
  T4  +noise_lr3    – noise_corr_rank=3 (low-rank correlated noise)
  T5  +rollout      – rollout_weight=0.2, rollout_steps=20
  T6  +kitchen_sink – T1+T2+T3+T4+T5 combined
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from stage2.config import make_config
from stage2.train import train_stage2

H5 = "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5"
OUT_ROOT = Path("output_plots/stage2/tier1_feature_sweep")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ── Base config: C1_corr_init (our best so far) ──────────────────────
BASE = dict(
    num_epochs=50,
    cv_folds=3,
    seed=42,
    edge_specific_G=True,
    per_neuron_amplitudes=True,
    G_init_mode="corr_weighted",
    W_init_mode="corr_weighted",
    network_strength_floor=1.0,
    network_strength_target=0.8,
)

# ── Per-condition overrides (on top of BASE) ─────────────────────────
CONDITIONS = {
    "T0_base": {},

    "T1_poly2": dict(
        graph_poly_order=2,
    ),

    "T2_lowrank5": dict(
        lowrank_rank=5,
    ),

    "T3_lag5": dict(
        lag_order=5,
        lag_neighbor=True,
    ),

    "T4_noise_lr3": dict(
        noise_corr_rank=3,
    ),

    "T5_rollout": dict(
        rollout_weight=0.2,
        rollout_steps=20,
        warmstart_rollout=True,
    ),

    "T6_kitchen_sink": dict(
        graph_poly_order=2,
        lowrank_rank=5,
        lag_order=5,
        lag_neighbor=True,
        noise_corr_rank=3,
        rollout_weight=0.2,
        rollout_steps=20,
        warmstart_rollout=True,
    ),
}

log = OUT_ROOT / "run.log"


def tee(msg):
    print(msg, flush=True)
    with open(log, "a") as f:
        f.write(msg + "\n")


tee(f"=== Tier-1 feature sweep ({len(CONDITIONS)} conditions) ===")
tee(f"Base: epochs={BASE['num_epochs']}, folds={BASE['cv_folds']}, "
    f"edge_specific_G=True, pn_amp=True, corr_weighted G+W init")

for name, overrides in CONDITIONS.items():
    save_dir = OUT_ROOT / name
    summary_path = save_dir / "summary.json"

    if summary_path.exists():
        tee(f"\n>>> SKIP {name} (summary.json exists)")
        continue

    tee(f"\n{'='*60}")
    tee(f">>> {name}")
    tee(f"    overrides: {overrides}")
    tee(f"{'='*60}")

    kw = {**BASE, **overrides}
    cfg = make_config(H5, **kw)
    t0 = time.time()

    try:
        result = train_stage2(cfg, save_dir=str(save_dir), show=False)
        elapsed = time.time() - t0
        tee(f">>> {name} DONE in {elapsed/60:.1f} min")

        # Save summary
        summary = {"condition": name, "overrides": overrides,
                   "elapsed_min": round(elapsed / 60, 1)}
        if result and isinstance(result, dict):
            for k, v in result.items():
                if isinstance(v, (int, float, str, bool)):
                    summary[k] = v
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        tee(f">>> Saved {summary_path}")

    except Exception as e:
        elapsed = time.time() - t0
        tee(f">>> {name} FAILED after {elapsed/60:.1f} min: {e}")
        import traceback
        tee(traceback.format_exc())

tee(f"\n=== All {len(CONDITIONS)} conditions complete ===")
