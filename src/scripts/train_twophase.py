#!/usr/bin/env python
"""Two-phase training: Phase 1 (one-step) → Phase 2 (rollout + noise + curriculum).

Phase 1: Standard one-step teacher-forced training to get good initial dynamics.
Phase 2: Continue from Phase 1 weights with rollout loss, noise injection, and
          curriculum scheduling to improve multi-step stability and LOO R².

Usage
-----
    python -m scripts.train_twophase \
        --h5 data/used/behaviour+neuronal\ activity\ atanas\ \(2023\)/2/2022-08-02-01.h5 \
        --save_dir output_plots/stage2/twophase_run \
        --device cuda

    # Customize phase durations:
    python -m scripts.train_twophase \
        --h5 ... --save_dir ... \
        --phase1_epochs 100 --phase2_epochs 100 \
        --rollout_weight 0.3 --noise_sigma 0.05 \
        --K_start 5 --K_end 30
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── project imports ──────────────────────────────────────────────────
from stage2.config import make_config, Stage2PTConfig
from stage2.train import train_stage2_cv


def run_twophase(
    h5_path: str,
    save_dir: str,
    device: str = "cuda",
    # Phase 1
    phase1_epochs: int = 100,
    # Phase 2
    phase2_epochs: int = 100,
    rollout_weight: float = 0.3,
    noise_sigma: float = 0.05,
    K_start: int = 5,
    K_end: int = 30,
    rollout_starts: int = 8,
    # Also try LOO auxiliary loss
    loo_aux_weight: float = 0.0,
    loo_aux_steps: int = 20,
    loo_aux_neurons: int = 4,
    # Common
    cv_folds: int = 2,
    grad_clip: float = 1.0,
    lr: float = 1e-3,
    lr_phase2: float | None = None,  # lower LR for fine-tuning (default: lr/3)
) -> dict:
    """Run two-phase training and return evaluation results."""

    save = Path(save_dir)
    save.mkdir(parents=True, exist_ok=True)
    lr2 = lr_phase2 if lr_phase2 is not None else lr / 3.0

    # ═══════════════════════════════════════════════════════════════════
    #  Phase 1: one-step teacher-forced training
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'═'*70}")
    print(f"  PHASE 1: One-step training ({phase1_epochs} epochs)")
    print(f"{'═'*70}\n")
    t0 = time.time()

    cfg1 = make_config(
        h5_path,
        device=device,
        num_epochs=phase1_epochs,
        learning_rate=lr,
        cv_folds=cv_folds,
        grad_clip_norm=grad_clip,
        # No rollout/noise in phase 1
        rollout_weight=0.0,
        input_noise_sigma=0.0,
        loo_aux_weight=0.0,
    )

    p1_dir = str(save / "phase1")
    result1 = train_stage2_cv(cfg1, save_dir=p1_dir, show=False)

    t_p1 = time.time() - t0
    p1_onestep = result1.get("cv_onestep_r2_mean", float("nan"))
    p1_loo = result1.get("cv_loo_r2_windowed_mean", float("nan"))
    print(f"\n  Phase 1 done in {t_p1:.0f}s")
    print(f"  one-step R²={p1_onestep:.4f}  LOO_w R²={p1_loo:.4f}")

    # ═══════════════════════════════════════════════════════════════════
    #  Phase 2: rollout + noise + curriculum (warm-start from phase 1)
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'═'*70}")
    print(f"  PHASE 2: Rollout + noise training ({phase2_epochs} epochs)")
    print(f"  rollout_weight={rollout_weight}  noise_σ={noise_sigma}")
    print(f"  curriculum K={K_start}→{K_end}  lr={lr2:.1e}")
    if loo_aux_weight > 0:
        print(f"  loo_aux_weight={loo_aux_weight}  loo_aux_steps={loo_aux_steps}")
    print(f"{'═'*70}\n")
    t0 = time.time()

    cfg2 = make_config(
        h5_path,
        device=device,
        num_epochs=phase2_epochs,
        learning_rate=lr2,
        cv_folds=cv_folds,
        grad_clip_norm=grad_clip,
        # Rollout training ON
        rollout_weight=rollout_weight,
        rollout_steps=K_end,        # will be overridden by curriculum
        rollout_starts=rollout_starts,
        warmstart_rollout=True,
        # Noise injection ON
        input_noise_sigma=noise_sigma,
        # Curriculum scheduling ON
        rollout_curriculum=True,
        rollout_K_start=K_start,
        rollout_K_end=K_end,
        rollout_curriculum_start_epoch=0,
        rollout_curriculum_end_epoch=phase2_epochs,
        # LOO auxiliary loss (optional)
        loo_aux_weight=loo_aux_weight,
        loo_aux_steps=loo_aux_steps,
        loo_aux_neurons=loo_aux_neurons,
        loo_aux_starts=rollout_starts,
    )

    p2_dir = str(save / "phase2")
    result2 = train_stage2_cv(cfg2, save_dir=p2_dir, show=False,
                              warm_start_dir=p1_dir)

    t_p2 = time.time() - t0
    p2_onestep = result2.get("cv_onestep_r2_mean", float("nan"))
    p2_loo = result2.get("cv_loo_r2_windowed_mean", float("nan"))
    print(f"\n  Phase 2 done in {t_p2:.0f}s")
    print(f"  one-step R²={p2_onestep:.4f}  LOO_w R²={p2_loo:.4f}")

    # ═══════════════════════════════════════════════════════════════════
    #  Summary
    # ═══════════════════════════════════════════════════════════════════
    summary = {
        "h5_path": h5_path,
        "phase1": {
            "epochs": phase1_epochs,
            "onestep_r2": p1_onestep,
            "loo_w_r2": p1_loo,
            "time_s": t_p1,
        },
        "phase2": {
            "epochs": phase2_epochs,
            "rollout_weight": rollout_weight,
            "noise_sigma": noise_sigma,
            "K_start": K_start,
            "K_end": K_end,
            "loo_aux_weight": loo_aux_weight,
            "onestep_r2": p2_onestep,
            "loo_w_r2": p2_loo,
            "time_s": t_p2,
        },
        "improvement": {
            "onestep_delta": p2_onestep - p1_onestep,
            "loo_w_delta": p2_loo - p1_loo,
        }
    }

    with open(save / "twophase_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'═'*70}")
    print(f"  TWO-PHASE SUMMARY")
    print(f"  Phase 1 → Phase 2:")
    print(f"    one-step R²: {p1_onestep:.4f} → {p2_onestep:.4f} "
          f"(Δ={p2_onestep - p1_onestep:+.4f})")
    print(f"    LOO_w R²:    {p1_loo:.4f} → {p2_loo:.4f} "
          f"(Δ={p2_loo - p1_loo:+.4f})")
    print(f"  Total time: {t_p1 + t_p2:.0f}s ({(t_p1+t_p2)/60:.1f}min)")
    print(f"{'═'*70}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Two-phase Stage2 training: one-step → rollout+noise")
    parser.add_argument("--h5", required=True, help="Path to HDF5 data file")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--device", default="cuda")
    # Phase 1
    parser.add_argument("--phase1_epochs", type=int, default=100)
    # Phase 2
    parser.add_argument("--phase2_epochs", type=int, default=100)
    parser.add_argument("--rollout_weight", type=float, default=0.3)
    parser.add_argument("--noise_sigma", type=float, default=0.05)
    parser.add_argument("--K_start", type=int, default=5)
    parser.add_argument("--K_end", type=int, default=30)
    parser.add_argument("--rollout_starts", type=int, default=8)
    # LOO aux
    parser.add_argument("--loo_aux_weight", type=float, default=0.0)
    parser.add_argument("--loo_aux_steps", type=int, default=20)
    parser.add_argument("--loo_aux_neurons", type=int, default=4)
    # Common
    parser.add_argument("--cv_folds", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_phase2", type=float, default=None)
    args = parser.parse_args()

    run_twophase(
        h5_path=args.h5,
        save_dir=args.save_dir,
        device=args.device,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        rollout_weight=args.rollout_weight,
        noise_sigma=args.noise_sigma,
        K_start=args.K_start,
        K_end=args.K_end,
        rollout_starts=args.rollout_starts,
        loo_aux_weight=args.loo_aux_weight,
        loo_aux_steps=args.loo_aux_steps,
        loo_aux_neurons=args.loo_aux_neurons,
        cv_folds=args.cv_folds,
        lr=args.lr,
        lr_phase2=args.lr_phase2,
    )


if __name__ == "__main__":
    main()
