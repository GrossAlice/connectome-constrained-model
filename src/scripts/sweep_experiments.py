#!/usr/bin/env python3
"""Minimal experiment sweep runner for stage2 hyperparameter tuning.

Usage:
    .venv/bin/python -u scripts/sweep_experiments.py --plan stage_c --device cuda
    .venv/bin/python -u scripts/sweep_experiments.py --plan baseline --device cuda
    .venv/bin/python -u scripts/sweep_experiments.py --plan focused --device cuda

Logs a CSV summary after each run for easy comparison.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure project root (src/) is on sys.path so `stage2` is importable
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

# --------------------------------------------------------------------------- #
#  Experiment definitions                                                       #
# --------------------------------------------------------------------------- #

# Default baseline config (matches recent run_config.json)
BASELINE = dict(
    num_epochs=50,
    learning_rate=0.001,
    synaptic_lr_multiplier=5.0,
    rollout_steps=5,
    rollout_weight=0.3,
    rollout_starts=8,
    warmstart_rollout=False,
    alpha_cv_every=10,
    alpha_cv_blend=0.5,
    alpha_cv_max_ratio=0.0,
    lambda_u_reg=0.0,
    I0_reg=0.0,
    G_reg=0.0,
    tau_reg=0.0,
    interaction_l2=0.0,
    dynamics_l2=0.0,
    behavior_weight=0.1,
    # Expanded evaluation — essential for reliable comparisons
    eval_loo_subset_mode="variance",
    eval_loo_subset_size=20,
    plot_every=0,  # disable intermediate plots for speed
)

H5_DEFAULT = "data/used/behaviour+neuronal activity atanas (2023)/the same neurons/2022-08-02-01.h5"


def _merge(base: dict, overrides: dict) -> dict:
    out = dict(base)
    out.update(overrides)
    return out


def make_plan_baseline(seeds=(42, 43)):
    """Stage A: 2 seeds, pure default config."""
    runs = []
    for s in seeds:
        runs.append(("baseline_s{s}", _merge(BASELINE, {}), s))
    return runs


def make_plan_stage_c():
    """Stage C: coarse screening of high-impact knobs (1 seed each)."""
    seed = 42
    runs = [
        ("C00_baseline",    BASELINE, seed),
        ("C01_ep100",       _merge(BASELINE, dict(num_epochs=100)), seed),
        ("C02_ep200",       _merge(BASELINE, dict(num_epochs=200)), seed),
        ("C03_reg_light",   _merge(BASELINE, dict(lambda_u_reg=0.01, I0_reg=0.01)), seed),
        ("C04_reg_medium",  _merge(BASELINE, dict(lambda_u_reg=0.1, I0_reg=0.1)), seed),
        ("C05_acv_aggr",    _merge(BASELINE, dict(alpha_cv_every=5, alpha_cv_blend=1.0)), seed),
        ("C06_acv_conserv", _merge(BASELINE, dict(alpha_cv_every=5, alpha_cv_blend=0.3,
                                                   alpha_cv_max_ratio=3.0)), seed),
        ("C07_rollout_strong", _merge(BASELINE, dict(rollout_steps=10, rollout_weight=0.5)), seed),
        ("C08_rollout_long",   _merge(BASELINE, dict(rollout_steps=20, rollout_weight=0.3,
                                                      warmstart_rollout=True)), seed),
        ("C09_G_reg",       _merge(BASELINE, dict(G_reg=0.01)), seed),
        ("C10_interact_l2", _merge(BASELINE, dict(interaction_l2=0.01)), seed),
        ("C11_low_lr",      _merge(BASELINE, dict(learning_rate=0.0003, num_epochs=150)), seed),
        ("C12_syn_lr1",     _merge(BASELINE, dict(synaptic_lr_multiplier=1.0)), seed),
    ]
    return runs


def make_plan_focused():
    """Stage D: test untested high-impact knobs."""
    seed = 42
    # Shared base: 100 epochs, moderate rollout, no alpha/backbone CV
    # so we isolate the effect of each knob.
    BASE_D = _merge(BASELINE, dict(
        num_epochs=100,
        rollout_steps=20,
        rollout_weight=1.0,
        warmstart_rollout=True,
        alpha_cv_every=0,
        backbone_cv_every=0,
    ))
    runs = [
        # ---------- Control ----------
        ("D00_control", BASE_D, seed),

        # ---------- 3. LOO aux training (most direct LOO fix) ----------
        ("D01_loo_aux_light", _merge(BASE_D, dict(
            loo_aux_weight=0.1, loo_aux_steps=20,
            loo_aux_neurons=4, loo_aux_starts=2)), seed),
        ("D02_loo_aux_medium", _merge(BASE_D, dict(
            loo_aux_weight=0.5, loo_aux_steps=20,
            loo_aux_neurons=8, loo_aux_starts=2)), seed),
        ("D03_loo_aux_heavy", _merge(BASE_D, dict(
            loo_aux_weight=1.0, loo_aux_steps=40,
            loo_aux_neurons=8, loo_aux_starts=4)), seed),

        # ---------- 5. Learn W (breaks N²-param bottleneck) ----------
        ("D04_learn_W_sv", _merge(BASE_D, dict(
            learn_W_sv=True, ridge_W_sv=0.01)), seed),
        ("D05_learn_W_both", _merge(BASE_D, dict(
            learn_W_sv=True, learn_W_dcv=True,
            ridge_W_sv=0.01, ridge_W_dcv=0.01)), seed),
        ("D06_learn_W_both_heavy_reg", _merge(BASE_D, dict(
            learn_W_sv=True, learn_W_dcv=True,
            ridge_W_sv=0.1, ridge_W_dcv=0.1)), seed),

        # ---------- 6. Trainable taus ----------
        ("D07_free_tau_sv", _merge(BASE_D, dict(
            fix_tau_sv=False)), seed),
        ("D08_free_tau_both", _merge(BASE_D, dict(
            fix_tau_sv=False, fix_tau_dcv=False)), seed),
        ("D09_free_tau_both_reg", _merge(BASE_D, dict(
            fix_tau_sv=False, fix_tau_dcv=False,
            tau_reg=0.1)), seed),

        # ---------- Combos (best of each) ----------
        ("D10_loo_aux+free_tau", _merge(BASE_D, dict(
            loo_aux_weight=0.5, loo_aux_steps=20,
            loo_aux_neurons=8, loo_aux_starts=2,
            fix_tau_sv=False, fix_tau_dcv=False)), seed),
        ("D11_loo_aux+learn_W", _merge(BASE_D, dict(
            loo_aux_weight=0.5, loo_aux_steps=20,
            loo_aux_neurons=8, loo_aux_starts=2,
            learn_W_sv=True, learn_W_dcv=True,
            ridge_W_sv=0.01, ridge_W_dcv=0.01)), seed),
        ("D12_all_three", _merge(BASE_D, dict(
            loo_aux_weight=0.5, loo_aux_steps=20,
            loo_aux_neurons=8, loo_aux_starts=2,
            learn_W_sv=True, learn_W_dcv=True,
            ridge_W_sv=0.01, ridge_W_dcv=0.01,
            fix_tau_sv=False, fix_tau_dcv=False,
            tau_reg=0.1)), seed),
    ]
    return runs


PLANS = {
    "baseline": make_plan_baseline,
    "stage_c": make_plan_stage_c,
    "focused": make_plan_focused,
}


# --------------------------------------------------------------------------- #
#  Runner                                                                       #
# --------------------------------------------------------------------------- #

def run_single_experiment(
    name: str,
    overrides: dict,
    seed: int,
    h5_path: str,
    save_root: str,
    device: str,
) -> dict:
    """Run a single stage2 training experiment and return summary metrics."""
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    from stage2.config import make_config
    from stage2.train import train_stage2

    save_dir = str(Path(save_root) / name)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Build config with overrides
    cfg_kwargs = dict(overrides)
    cfg_kwargs["device"] = device
    cfg = make_config(h5_path, **cfg_kwargs)

    t0 = time.time()
    eval_result = train_stage2(cfg, save_dir=save_dir, show=False)
    elapsed = time.time() - t0

    # Extract metrics
    metrics = {"name": name, "seed": seed, "elapsed_s": round(elapsed, 1)}

    if eval_result is not None:
        loo = eval_result.get("loo", {})
        loo_r2 = loo.get("r2")
        if loo_r2 is not None:
            import numpy as np
            finite = loo_r2[np.isfinite(loo_r2)]
            metrics["loo_r2_mean"] = round(float(np.mean(finite)), 4) if len(finite) > 0 else None
            metrics["loo_r2_median"] = round(float(np.median(finite)), 4) if len(finite) > 0 else None
            metrics["loo_r2_min"] = round(float(np.min(finite)), 4) if len(finite) > 0 else None
            metrics["loo_r2_max"] = round(float(np.max(finite)), 4) if len(finite) > 0 else None
            metrics["loo_n_negative"] = int((finite < 0).sum())
            metrics["loo_n_evaluated"] = len(finite)

        onestep = eval_result.get("onestep", {})
        os_r2 = onestep.get("r2")
        if os_r2 is not None:
            import numpy as np
            finite = os_r2[np.isfinite(os_r2)]
            metrics["onestep_r2_mean"] = round(float(np.mean(finite)), 4) if len(finite) > 0 else None

        beh_r2 = eval_result.get("beh_r2_model")
        if beh_r2 is not None:
            metrics["beh_r2_model"] = round(float(beh_r2), 4)

    # Also extract from run.log for extra detail
    log_path = Path(save_dir) / "run.log"
    if log_path.exists():
        log_text = log_path.read_text()
        # Final dynamics loss
        dyn_matches = re.findall(r"dynamics=([\d.]+)", log_text)
        if dyn_matches:
            metrics["final_dynamics_loss"] = round(float(dyn_matches[-1]), 5)
        # Alpha-CV upper boundary count
        upper_matches = re.findall(r"(\d+) at upper boundary", log_text)
        if upper_matches:
            metrics["alpha_cv_at_upper"] = int(upper_matches[-1])
        # I0 drift
        i0_matches = re.findall(r"I0_absmax=([\d.]+)", log_text)
        if i0_matches:
            metrics["I0_absmax_final"] = round(float(i0_matches[-1]), 4)

    # Save per-run config for reproducibility
    with open(Path(save_dir) / "experiment_config.json", "w") as f:
        json.dump({"name": name, "seed": seed, "overrides": overrides}, f, indent=2)

    return metrics


def append_csv(csv_path: str, row: dict):
    """Append a dict row to CSV, creating header if needed."""
    p = Path(csv_path)
    write_header = not p.exists()
    with open(p, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Stage2 experiment sweep")
    parser.add_argument("--plan", required=True, choices=list(PLANS.keys()),
                        help="Which experiment plan to run")
    parser.add_argument("--h5", default=H5_DEFAULT,
                        help="Input HDF5 path")
    parser.add_argument("--save_root", default="output_plots/stage2/sweep",
                        help="Root directory for sweep outputs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--start_from", type=int, default=0,
                        help="Skip the first N runs (for resuming)")
    args = parser.parse_args()

    plan_fn = PLANS[args.plan]
    runs = plan_fn() if not callable(plan_fn) else plan_fn()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_root = f"{args.save_root}/{args.plan}_{timestamp}"
    csv_path = f"{save_root}/summary.csv"
    Path(save_root).mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"Sweep: {args.plan}  ({len(runs)} runs)")
    print(f"Output: {save_root}")
    print(f"CSV:    {csv_path}")
    print(f"{'='*60}\n")

    for idx, (name, overrides, seed) in enumerate(runs):
        if idx < args.start_from:
            print(f"[{idx+1}/{len(runs)}] SKIP {name}")
            continue

        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(runs)}] {name}  (seed={seed})")
        print(f"  Overrides: {json.dumps({k: v for k, v in overrides.items() if v != BASELINE.get(k)})}")
        print(f"{'='*60}\n")

        try:
            metrics = run_single_experiment(
                name, overrides, seed, args.h5, save_root, args.device,
            )
            metrics["timestamp"] = datetime.now().isoformat()
            append_csv(csv_path, metrics)
            print(f"\n✓ {name}: LOO R² mean={metrics.get('loo_r2_mean', '?')}  "
                  f"dynamics={metrics.get('final_dynamics_loss', '?')}  "
                  f"time={metrics.get('elapsed_s', '?')}s")
        except Exception as e:
            print(f"\n✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            append_csv(csv_path, {"name": name, "seed": seed, "error": str(e)})

    print(f"\n{'='*60}")
    print(f"Sweep complete. Results in {csv_path}")


if __name__ == "__main__":
    main()
