#!/usr/bin/env python
"""Sweep: tau initialisation and learning.

Current defaults: tau_sv=(0.5, 2.5), tau_dcv=(3.0, 5.0), both FIXED (rank=2).
gamma = exp(-dt/tau), with dt=0.6 s.

Physical interpretation:
  SV (synaptic vesicles) → fast neurotransmitters (glutamate, GABA)
  DCV (dense-core vesicles) → slow neuropeptides

Current gamma values:
  tau=0.5 → gamma=0.30  (very fast decay, ~1 step memory)
  tau=2.5 → gamma=0.79  (moderate, ~4 step memory)
  tau=3.0 → gamma=0.82  (moderate-slow)
  tau=5.0 → gamma=0.89  (slow, ~8 step memory)

This sweep tests:
  Block A – Learn taus (fix=False) with different regularisation
    TAU0  baseline (fixed, default init)       [control / reproduction]
    TAU1  learn taus, no reg                   [free optimisation]
    TAU2  learn taus, tau_reg=0.01             [soft anchor]
    TAU3  learn taus, tau_reg=0.1              [strong anchor]

  Block B – Different fixed initialisations (rank=2)
    TAU4  faster SV  τ_sv=(0.3, 1.0)          [sub-second dynamics]
    TAU5  slower SV  τ_sv=(1.0, 5.0)          [longer SV memory]
    TAU6  wider DCV  τ_dcv=(2.0, 12.0)        [broader neuropeptide range]
    TAU7  all slower  τ_sv=(1.5, 5.0) τ_dcv=(5.0, 15.0) [very long memory]

  Block C – Different ranks
    TAU8   rank=1  τ_sv=(1.0,)  τ_dcv=(4.0,)   [simpler, fewer params]
    TAU9   rank=3  τ_sv=(0.3,1.5,5.0)  τ_dcv=(2.0,5.0,12.0) [more expressive]
    TAU10  rank=4  τ_sv=(0.3,1.0,3.0,8.0) τ_dcv=(1.5,4.0,8.0,20.0) [rich]

  Block D – Learn + different ranks
    TAU11  learn + rank=3  τ_sv=(0.3,1.5,5.0)  τ_dcv=(2.0,5.0,12.0)
    TAU12  learn + rank=1  τ_sv=(1.0,)  τ_dcv=(4.0,)
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from stage2.config import make_config
from stage2.train import train_stage2

H5 = "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5"
OUT_ROOT = Path("output_plots/stage2/sweep_tau_learning")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ── Base: C1_corr_init (our best LOO=0.216) ─────────────────────────
BASE = dict(
    num_epochs=30,
    cv_folds=3,
    seed=42,
    edge_specific_G=True,
    per_neuron_amplitudes=True,
    G_init_mode="corr_weighted",
    W_init_mode="corr_weighted",
    network_strength_floor=1.0,
    network_strength_target=0.8,
)

CONDITIONS = {
    # ═══ Block A: Learn taus (rank=2, default inits) ═══════════════
    "TAU0_baseline_fixed": {},   # control: fixed taus, default init

    "TAU1_learn_no_reg": dict(
        fix_tau_sv=False,
        fix_tau_dcv=False,
    ),

    "TAU2_learn_reg001": dict(
        fix_tau_sv=False,
        fix_tau_dcv=False,
        tau_reg=0.01,
    ),

    "TAU3_learn_reg01": dict(
        fix_tau_sv=False,
        fix_tau_dcv=False,
        tau_reg=0.1,
    ),

    # ═══ Block B: Different fixed initialisations (rank=2) ═════════
    "TAU4_fast_sv": dict(
        tau_sv_init=(0.3, 1.0),
        a_sv_init=(2.0, 0.8),
    ),

    "TAU5_slow_sv": dict(
        tau_sv_init=(1.0, 5.0),
        a_sv_init=(2.0, 0.8),
    ),

    "TAU6_wide_dcv": dict(
        tau_dcv_init=(2.0, 12.0),
        a_dcv_init=(0.8, 0.6),
    ),

    "TAU7_all_slow": dict(
        tau_sv_init=(1.5, 5.0),
        a_sv_init=(2.0, 0.8),
        tau_dcv_init=(5.0, 15.0),
        a_dcv_init=(0.8, 0.6),
    ),

    # ═══ Block C: Different ranks (fixed taus) ═════════════════════
    "TAU8_rank1": dict(
        tau_sv_init=(1.0,),
        a_sv_init=(1.5,),
        tau_dcv_init=(4.0,),
        a_dcv_init=(0.7,),
    ),

    "TAU9_rank3": dict(
        tau_sv_init=(0.3, 1.5, 5.0),
        a_sv_init=(2.0, 1.2, 0.5),
        tau_dcv_init=(2.0, 5.0, 12.0),
        a_dcv_init=(0.8, 0.6, 0.3),
    ),

    "TAU10_rank4": dict(
        tau_sv_init=(0.3, 1.0, 3.0, 8.0),
        a_sv_init=(2.0, 1.5, 0.8, 0.3),
        tau_dcv_init=(1.5, 4.0, 8.0, 20.0),
        a_dcv_init=(0.8, 0.6, 0.4, 0.2),
    ),

    # ═══ Block D: Learn + different ranks ══════════════════════════
    "TAU11_learn_rank3": dict(
        fix_tau_sv=False,
        fix_tau_dcv=False,
        tau_reg=0.01,
        tau_sv_init=(0.3, 1.5, 5.0),
        a_sv_init=(2.0, 1.2, 0.5),
        tau_dcv_init=(2.0, 5.0, 12.0),
        a_dcv_init=(0.8, 0.6, 0.3),
    ),

    "TAU12_learn_rank1": dict(
        fix_tau_sv=False,
        fix_tau_dcv=False,
        tau_reg=0.01,
        tau_sv_init=(1.0,),
        a_sv_init=(1.5,),
        tau_dcv_init=(4.0,),
        a_dcv_init=(0.7,),
    ),
}

log = OUT_ROOT / "run.log"


def tee(msg):
    print(msg, flush=True)
    with open(log, "a") as f:
        f.write(msg + "\n")


tee(f"=== Tau learning sweep ({len(CONDITIONS)} conditions) ===")
tee(f"Base: C1_corr_init, {BASE['num_epochs']} epochs, {BASE['cv_folds']}-fold CV")
tee(f"Testing: learned vs fixed taus, different inits, different ranks (1-4)")

for name, overrides in CONDITIONS.items():
    save_dir = OUT_ROOT / name
    summary_path = save_dir / "summary.json"

    if summary_path.exists():
        tee(f"\n>>> SKIP {name} (summary.json exists)")
        continue

    # ── Print tau config ────────────────────────────────────────────
    kw = {**BASE, **overrides}
    tau_sv = kw.get("tau_sv_init", (0.5, 2.5))
    tau_dcv = kw.get("tau_dcv_init", (3.0, 5.0))
    fix_sv = kw.get("fix_tau_sv", True)
    fix_dcv = kw.get("fix_tau_dcv", True)

    tee(f"\n{'='*60}")
    tee(f">>> {name}")
    tee(f"    tau_sv_init={tau_sv}  fix={fix_sv}  (rank={len(tau_sv)})")
    tee(f"    tau_dcv_init={tau_dcv}  fix={fix_dcv}  (rank={len(tau_dcv)})")
    tee(f"    overrides: {overrides}")
    tee(f"{'='*60}")

    cfg = make_config(H5, **kw)
    t0 = time.time()

    try:
        result = train_stage2(cfg, save_dir=str(save_dir), show=False)
        elapsed = time.time() - t0
        tee(f">>> {name} DONE in {elapsed/60:.1f} min")

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
