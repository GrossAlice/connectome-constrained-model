#!/usr/bin/env python
"""
Sweep: beat the connectome-Ridge LOO R² baseline (0.4802).
──────────────────────────────────────────────────────────
Best S2 result so far: A0_baseline (lag_10, l2=1e-3, 60ep) → LOO=0.4705.
Gap to Ridge = 0.0097.

Strategy: start from the best-known config and systematically test:
  • More epochs (100 → 200)  — A0 only trained 60ep
  • Larger lag order (12, 15)  — lag_5→10 gave +0.04 LOO
  • Input noise  — σ=0.1 gave +0.02 LOO in regularisation sweep
  • Cosine LR schedule  — standard late-training improvement
  • Combos of the above
  • Seed variation  — confidence intervals
"""

import json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stage2.config import make_config
from stage2.train import train_stage2_cv

H5 = "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5"
OUT_ROOT = Path("output_plots/stage2/sweep_beat_ridge")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ── Best-known base config (= A0_baseline from sweep_lag_ablation) ──
BASE = dict(
    lag_order=10,
    lag_neighbor=True,
    lag_connectome_mask="all",
    lag_neighbor_per_type=True,
    dynamics_l2=1e-3,
    rollout_steps=10,
    rollout_weight=0.3,
    warmstart_rollout=True,
    num_epochs=60,
    seed=42,
    skip_free_run=True,
    eval_loo_subset_size=30,
    eval_loo_subset_mode="variance",
)

CONDITIONS = {
    # ── Axis 1: more epochs ──────────────────────────────────────────
    "B00_100ep":       dict(num_epochs=100),
    "B01_150ep":       dict(num_epochs=150),
    "B02_200ep":       dict(num_epochs=200),

    # ── Axis 2: larger lag order ─────────────────────────────────────
    "B03_lag12_100ep":  dict(lag_order=12, num_epochs=100),
    "B04_lag15_100ep":  dict(lag_order=15, num_epochs=100),

    # ── Axis 3: input noise ──────────────────────────────────────────
    "B05_noise005_100ep": dict(input_noise_sigma=0.05, num_epochs=100),
    "B06_noise01_100ep":  dict(input_noise_sigma=0.1,  num_epochs=100),

    # ── Axis 4: cosine LR schedule ──────────────────────────────────
    "B07_cosine_100ep":   dict(lr_schedule="cosine", num_epochs=100),
    "B08_cosine_200ep":   dict(lr_schedule="cosine", num_epochs=200),

    # ── Axis 5: weight decay (AdamW) ────────────────────────────────
    "B09_wd1e-4_100ep":   dict(weight_decay=1e-4, num_epochs=100),

    # ── Combos ───────────────────────────────────────────────────────
    "B10_lag12_noise_100ep":  dict(lag_order=12, input_noise_sigma=0.05, num_epochs=100),
    "B11_lag12_cos_150ep":    dict(lag_order=12, lr_schedule="cosine", num_epochs=150),
    "B12_combo_150ep":        dict(lag_order=12, input_noise_sigma=0.05,
                                   lr_schedule="cosine", num_epochs=150),
    "B13_combo_200ep":        dict(lag_order=12, input_noise_sigma=0.05,
                                   lr_schedule="cosine", num_epochs=200),

    # ── Seed variation on best single-axis (100ep) ───────────────────
    "B14_seed123_100ep":  dict(seed=123, num_epochs=100),
    "B15_seed7_100ep":    dict(seed=7,   num_epochs=100),
}

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
        print(f"  ERROR in {tag}: {e}")
        (out_dir / "ERROR.txt").write_text(str(e))
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
