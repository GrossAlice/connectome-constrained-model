#!/usr/bin/env python
"""
Sweep: Force stronger coupling by raising λ floor + complementary strategies
============================================================================
Problem: λ ≈ 0.138 (mean) → 86% AR(1) self-persistence, only 14% coupling.
         The optimizer pushes λ toward lambda_u_lo because AR(1) is the
         easiest way to minimise one-step loss.

Strategies tested:
  A – Raise lambda_u_lo floor: directly forces more coupling contribution
  B – Rollout training: multi-step loss where AR(1) decays as (1-λ)^K
  C – Input noise: degrades teacher-forced AR(1) strategy
  D – λ floor + rollout (combined)
  E – λ floor + coupling expressivity (pn_amp, edge_G, corr_weighted)

Conditions (20 total):

  Block A: Lambda floor sweep
    L00  baseline (lo=0.1)            — reproduction baseline for comparison
    L01  lo=0.2                       — mild increase
    L02  lo=0.3                       — moderate
    L03  lo=0.5                       — strong (50% must come from coupling)
    L04  lo=0.7                       — very strong

  Block B: Rollout training (forces multi-step accuracy)
    L05  rollout K=10 w=0.5           — short horizon, moderate weight
    L06  rollout K=20 w=0.5           — medium horizon
    L07  rollout K=30 w=1.0           — long horizon, heavy weight

  Block C: Input noise (degrades AR(1) strategy)
    L08  input_noise=0.05             — mild noise
    L09  input_noise=0.10             — moderate noise
    L10  input_noise=0.20             — strong noise

  Block D: Lambda floor + rollout (combined attack)
    L11  lo=0.3 + rollout K=20 w=0.5
    L12  lo=0.5 + rollout K=20 w=0.5
    L13  lo=0.3 + rollout K=30 w=1.0

  Block E: Lambda floor + input noise
    L14  lo=0.3 + input_noise=0.05
    L15  lo=0.5 + input_noise=0.10

  Block F: Lambda floor + expressivity features
    L16  lo=0.3 + edge_G + pn_amp + corr_weighted
    L17  lo=0.5 + edge_G + pn_amp + corr_weighted
    L18  lo=0.3 + edge_G + pn_amp + corr_weighted + rollout K=20
    L19  lo=0.5 + edge_G + pn_amp + corr_weighted + rollout K=20
"""
from __future__ import annotations
import json, os, sys, time, shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
from stage2.config import make_config
from stage2.train import train_stage2_cv

H5 = "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5"
SAVE_ROOT = Path("output_plots/stage2/sweep_lambda_coupling")
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

CV_FOLDS = 3
EPOCHS   = 50

# ---------- condition definitions ----------
# Keys starting with _ are train-level overrides (strip _ when applying)
CONDITIONS = {
    # ═══ Block A: Lambda floor sweep ═══
    "L00_baseline_lo10": dict(
        lambda_u_lo=0.1,
    ),
    "L01_lo20": dict(
        lambda_u_lo=0.2,
    ),
    "L02_lo30": dict(
        lambda_u_lo=0.3,
    ),
    "L03_lo50": dict(
        lambda_u_lo=0.5,
    ),
    "L04_lo70": dict(
        lambda_u_lo=0.7,
    ),

    # ═══ Block B: Rollout training ═══
    "L05_rollout_K10": dict(
        _rollout_weight=0.5,
        _rollout_steps=10,
        _rollout_starts=8,
        _warmstart_rollout=True,
    ),
    "L06_rollout_K20": dict(
        _rollout_weight=0.5,
        _rollout_steps=20,
        _rollout_starts=8,
        _warmstart_rollout=True,
    ),
    "L07_rollout_K30": dict(
        _rollout_weight=1.0,
        _rollout_steps=30,
        _rollout_starts=8,
        _warmstart_rollout=True,
    ),

    # ═══ Block C: Input noise ═══
    "L08_inoise05": dict(
        _input_noise_sigma=0.05,
    ),
    "L09_inoise10": dict(
        _input_noise_sigma=0.10,
    ),
    "L10_inoise20": dict(
        _input_noise_sigma=0.20,
    ),

    # ═══ Block D: Lambda floor + rollout ═══
    "L11_lo30+roll20": dict(
        lambda_u_lo=0.3,
        _rollout_weight=0.5,
        _rollout_steps=20,
        _rollout_starts=8,
        _warmstart_rollout=True,
    ),
    "L12_lo50+roll20": dict(
        lambda_u_lo=0.5,
        _rollout_weight=0.5,
        _rollout_steps=20,
        _rollout_starts=8,
        _warmstart_rollout=True,
    ),
    "L13_lo30+roll30": dict(
        lambda_u_lo=0.3,
        _rollout_weight=1.0,
        _rollout_steps=30,
        _rollout_starts=8,
        _warmstart_rollout=True,
    ),

    # ═══ Block E: Lambda floor + input noise ═══
    "L14_lo30+inoise05": dict(
        lambda_u_lo=0.3,
        _input_noise_sigma=0.05,
    ),
    "L15_lo50+inoise10": dict(
        lambda_u_lo=0.5,
        _input_noise_sigma=0.10,
    ),

    # ═══ Block F: Lambda floor + expressivity ═══
    "L16_lo30+expr": dict(
        lambda_u_lo=0.3,
        edge_specific_G=True,
        per_neuron_amplitudes=True,
        G_init_mode="corr_weighted",
        W_init_mode="corr_weighted",
    ),
    "L17_lo50+expr": dict(
        lambda_u_lo=0.5,
        edge_specific_G=True,
        per_neuron_amplitudes=True,
        G_init_mode="corr_weighted",
        W_init_mode="corr_weighted",
    ),
    "L18_lo30+expr+roll20": dict(
        lambda_u_lo=0.3,
        edge_specific_G=True,
        per_neuron_amplitudes=True,
        G_init_mode="corr_weighted",
        W_init_mode="corr_weighted",
        _rollout_weight=0.5,
        _rollout_steps=20,
        _rollout_starts=8,
        _warmstart_rollout=True,
    ),
    "L19_lo50+expr+roll20": dict(
        lambda_u_lo=0.5,
        edge_specific_G=True,
        per_neuron_amplitudes=True,
        G_init_mode="corr_weighted",
        W_init_mode="corr_weighted",
        _rollout_weight=0.5,
        _rollout_steps=20,
        _rollout_starts=8,
        _warmstart_rollout=True,
    ),
}

# ---------- Logging ----------
log_path = SAVE_ROOT / "run.log"

def tee(msg):
    print(msg, flush=True)
    with open(log_path, "a") as f:
        f.write(msg + "\n")

# ---------- Main loop ----------
tee(f"\n{'='*70}")
tee(f"LAMBDA-COUPLING SWEEP — {len(CONDITIONS)} conditions")
tee(f"  {CV_FOLDS}-fold CV, {EPOCHS} epochs each")
tee(f"  Strategy: force stronger coupling via λ floor / rollout / input noise")
tee(f"{'='*70}\n")

summary_rows = []

for cond_name, overrides in CONDITIONS.items():
    save_dir = SAVE_ROOT / cond_name

    # Resume: skip completed conditions
    summary_path = save_dir / "summary.json"
    if summary_path.exists():
        tee(f"\n[SKIP] {cond_name} — already done")
        with open(summary_path) as f:
            row = json.load(f)
        summary_rows.append(row)
        continue

    # Clean partial runs
    if save_dir.exists():
        shutil.rmtree(save_dir)

    # Separate train-level overrides (prefixed with _) from dynamics-level
    train_overrides = {}
    dyn_overrides = {}
    for k, v in overrides.items():
        if k.startswith("_"):
            train_overrides[k[1:]] = v
        else:
            dyn_overrides[k] = v

    tee(f"\n{'='*60}")
    tee(f"[RUN] {cond_name}  epochs={EPOCHS}  cv_folds={CV_FOLDS}")
    tee(f"      dyn:   {dyn_overrides}")
    tee(f"      train: {train_overrides}")
    tee(f"{'='*60}")

    # Build config
    cfg = make_config(H5, cv_folds=CV_FOLDS, **dyn_overrides)
    cfg.seed = 42
    cfg.num_epochs = EPOCHS
    cfg.network_strength_floor = 1.0
    cfg.network_strength_target = 0.8
    cfg.skip_free_run = True
    cfg.skip_final_eval = True

    # Apply train-level overrides
    for k, v in train_overrides.items():
        setattr(cfg, k, v)

    t0 = time.time()
    try:
        results = train_stage2_cv(cfg, save_dir=str(save_dir))
    except Exception as e:
        elapsed = time.time() - t0
        tee(f"[FAIL] {cond_name} after {elapsed:.0f}s: {e}")
        import traceback
        tee(traceback.format_exc())
        row = {"condition": cond_name, "status": "FAILED", "error": str(e),
               "elapsed_sec": round(elapsed, 1)}
        summary_rows.append(row)
        with open(summary_path.parent / "error.txt", "w") as f:
            f.write(traceback.format_exc())
        continue

    elapsed = time.time() - t0

    # Collect results
    row = {"condition": cond_name, "status": "OK",
           "elapsed_sec": round(elapsed, 1),
           "epochs": EPOCHS, "cv_folds": CV_FOLDS,
           "overrides": {k: str(v) for k, v in overrides.items()}}
    if results is not None:
        for k, v in results.items():
            if isinstance(v, (float, int)):
                row[k] = round(v, 6) if isinstance(v, float) else v
    summary_rows.append(row)

    # Extract lambda_u from saved model to verify floor is working
    try:
        st = torch.load(str(save_dir / "fold_0_state.pt"),
                        map_location="cpu", weights_only=False)
        raw = st.get("_lambda_u_raw")
        if raw is not None:
            lo = dyn_overrides.get("lambda_u_lo", 0.1)
            hi = 0.9999
            lu = torch.sigmoid(raw) * (hi - lo) + lo
            row["lambda_u_mean"] = round(lu.mean().item(), 4)
            row["lambda_u_min"] = round(lu.min().item(), 4)
            row["lambda_u_max"] = round(lu.max().item(), 4)
            row["ar1_fraction"] = round((1 - lu).mean().item(), 4)
            tee(f"  lambda_u: mean={lu.mean():.4f} min={lu.min():.4f} "
                f"max={lu.max():.4f}  AR(1)={(1-lu).mean():.1%}")
    except Exception:
        pass

    # Save per-condition summary
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(row, f, indent=2)

    # Print running table
    tee(f"\n[DONE] {cond_name}  time={elapsed:.0f}s")
    for k in ["cv_onestep_r2_mean", "cv_onestep_r2_median",
              "cv_loo_r2_mean", "cv_loo_r2_median"]:
        if k in row:
            tee(f"  {k}: {row[k]:.4f}")

# ---------- Final summary ----------
tee(f"\n\n{'='*100}")
tee(f"LAMBDA-COUPLING SWEEP — FINAL SUMMARY — {CV_FOLDS}-fold CV, {EPOCHS} epochs")
tee(f"{'='*100}")
header = (f"{'Condition':<30s} {'λ_mean':>7s} {'AR(1)%':>7s} "
          f"{'1step_R²':>9s} {'LOO_mean':>9s} {'LOO_med':>9s} {'Time(s)':>8s}")
tee(header)
tee("-" * len(header))
for row in summary_rows:
    name = row.get("condition", "?")
    lam = row.get("lambda_u_mean", float("nan"))
    ar1 = row.get("ar1_fraction", float("nan"))
    r2 = row.get("cv_onestep_r2_mean", float("nan"))
    loo = row.get("cv_loo_r2_mean", float("nan"))
    loom = row.get("cv_loo_r2_median", float("nan"))
    t = row.get("elapsed_sec", 0)
    tee(f"{name:<30s} {lam:7.4f} {ar1:7.1%} {r2:9.4f} {loo:9.4f} {loom:9.4f} {t:8.0f}")

# Save combined summary
with open(SAVE_ROOT / "sweep_summary.json", "w") as f:
    json.dump(summary_rows, f, indent=2, default=str)
tee(f"\nSaved to {SAVE_ROOT}/sweep_summary.json")
