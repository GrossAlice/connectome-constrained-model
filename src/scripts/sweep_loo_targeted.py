#!/usr/bin/env python
"""Sweep: LOO-targeted training strategies.

Key finding: C1_corr_init at 30 epochs → LOO=0.216. The *same* config at 50
epochs → LOO=0.183.  More teacher-forced training HURTS LOO because the model
shifts explanatory power from coupling into self-dynamics (λ, I0).

This sweep tests interventions that force the model to rely on network coupling:

  L0  baseline_30ep      – C1 reproduction (30 epochs)
  L1  loo_aux             – LOO auxiliary loss during training
  L2  rollout             – multi-step rollout loss
  L3  input_noise         – Gaussian noise on teacher-forced inputs
  L4  rollout+noise       – combine L2+L3
  L5  loo_aux+noise       – combine L1+L3
  L6  loo_aux+rollout     – combine L1+L2
  L7  kitchen_sink        – L1+L2+L3 combined
  L8  fewer_epochs_20     – even fewer epochs (test early-stopping hypothesis)
  L9  curriculum_rollout  – ramp rollout horizon over epochs
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from stage2.config import make_config
from stage2.train import train_stage2

H5 = "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5"
OUT_ROOT = Path("output_plots/stage2/loo_targeted_sweep")
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
    # ── Controls ─────────────────────────────────────────────────────
    "L0_baseline_30ep": {},

    "L8_fewer_20ep": dict(
        num_epochs=20,
    ),

    # ── LOO auxiliary loss (directly trains the LOO objective) ───────
    # Randomly holds out neurons during training and penalises
    # multi-step prediction error on the held-out neuron.
    "L1_loo_aux": dict(
        loo_aux_weight=0.5,
        loo_aux_steps=30,
        loo_aux_neurons=8,
        loo_aux_starts=4,
    ),

    # ── Rollout loss (multi-step stability) ─────────────────────────
    "L2_rollout": dict(
        rollout_weight=0.3,
        rollout_steps=20,
        warmstart_rollout=True,
    ),

    # ── Input noise (corrupts self-prediction → forces coupling) ────
    "L3_input_noise": dict(
        input_noise_sigma=0.15,
    ),

    # ── Combos ──────────────────────────────────────────────────────
    "L4_rollout+noise": dict(
        rollout_weight=0.3,
        rollout_steps=20,
        warmstart_rollout=True,
        input_noise_sigma=0.15,
    ),

    "L5_loo_aux+noise": dict(
        loo_aux_weight=0.5,
        loo_aux_steps=30,
        loo_aux_neurons=8,
        loo_aux_starts=4,
        input_noise_sigma=0.15,
    ),

    "L6_loo_aux+rollout": dict(
        loo_aux_weight=0.5,
        loo_aux_steps=30,
        loo_aux_neurons=8,
        loo_aux_starts=4,
        rollout_weight=0.3,
        rollout_steps=20,
        warmstart_rollout=True,
    ),

    "L7_kitchen_sink": dict(
        loo_aux_weight=0.5,
        loo_aux_steps=30,
        loo_aux_neurons=8,
        loo_aux_starts=4,
        rollout_weight=0.3,
        rollout_steps=20,
        warmstart_rollout=True,
        input_noise_sigma=0.15,
    ),

    # ── Curriculum: ramp rollout horizon from 5→30 over 30 epochs ──
    "L9_curriculum": dict(
        rollout_weight=0.3,
        rollout_steps=30,
        warmstart_rollout=True,
        rollout_curriculum=True,
        rollout_K_start=5,
        rollout_K_end=30,
        rollout_curriculum_start_epoch=0,
        rollout_curriculum_end_epoch=30,
    ),
}

log = OUT_ROOT / "run.log"


def tee(msg):
    print(msg, flush=True)
    with open(log, "a") as f:
        f.write(msg + "\n")


tee(f"=== LOO-targeted sweep ({len(CONDITIONS)} conditions) ===")
tee(f"Base: C1_corr_init, {BASE['num_epochs']} epochs, {BASE['cv_folds']}-fold CV")
tee(f"Key hypothesis: teacher-forced training overfits self-dynamics at expense of coupling")

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
