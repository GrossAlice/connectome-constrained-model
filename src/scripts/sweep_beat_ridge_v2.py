#!/usr/bin/env python
"""
Sweep: Concrete Next Experiment — combine all best-known improvements.
──────────────────────────────────────────────────────────────────────
Best single results so far:
  • lag10 + per-type: LOO ≈ 0.471
  • Ridge warmstart:  LOO ≈ 0.453  (only 60ep, lag5)
  • noise σ=0.1:      LOO ≈ 0.447  (+0.02 over no-noise)
  • More epochs:      LOO ≈ 0.456  (100ep, lag10)

Strategy: combine all known-good axes:
  1. lag10 + Ridge warmstart + 100ep
  2. + cosine LR
  3. + input noise (0.05, 0.1)
  4. + weight decay
  5. lag12 variants
  6. Bigger rollout
"""

import json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stage2.config import make_config
from stage2.train import train_stage2_cv

H5 = "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5"
OUT_ROOT = Path("output_plots/stage2/sweep_beat_ridge_v2")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ── Base config (all known-good defaults) ──
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
    seed=42,
    skip_free_run=True,
    eval_loo_subset_size=30,
    eval_loo_subset_mode="variance",
)

CONDITIONS = {
    # ── Baseline with Ridge warmstart ──
    "C00_warmstart_100ep":         dict(ridge_warmstart=True),
    "C01_warmstart_cos_100ep":     dict(ridge_warmstart=True, lr_schedule="cosine"),
    "C02_warmstart_noise005":      dict(ridge_warmstart=True, input_noise_sigma=0.05),
    "C03_warmstart_noise01":       dict(ridge_warmstart=True, input_noise_sigma=0.1),

    # ── Combos ──
    "C04_warmstart_cos_noise005":  dict(ridge_warmstart=True, lr_schedule="cosine",
                                        input_noise_sigma=0.05),
    "C05_warmstart_cos_noise01":   dict(ridge_warmstart=True, lr_schedule="cosine",
                                        input_noise_sigma=0.1),
    "C06_warmstart_cos_noise_wd":  dict(ridge_warmstart=True, lr_schedule="cosine",
                                        input_noise_sigma=0.05, weight_decay=1e-4),

    # ── More epochs ──
    "C07_warmstart_cos_150ep":     dict(ridge_warmstart=True, lr_schedule="cosine",
                                        num_epochs=150),
    "C08_warmstart_combo_150ep":   dict(ridge_warmstart=True, lr_schedule="cosine",
                                        input_noise_sigma=0.05, num_epochs=150),
    "C09_warmstart_combo_200ep":   dict(ridge_warmstart=True, lr_schedule="cosine",
                                        input_noise_sigma=0.05, num_epochs=200),

    # ── Larger lag ──
    "C10_warmstart_lag12_cos":     dict(ridge_warmstart=True, lag_order=12,
                                        lr_schedule="cosine", input_noise_sigma=0.05),
    "C11_warmstart_lag15_cos":     dict(ridge_warmstart=True, lag_order=15,
                                        lr_schedule="cosine", input_noise_sigma=0.05),

    # ── Bigger rollout ──
    "C12_warmstart_roll15":        dict(ridge_warmstart=True, lr_schedule="cosine",
                                        input_noise_sigma=0.05, rollout_steps=15,
                                        rollout_weight=0.4),
    "C13_warmstart_roll20":        dict(ridge_warmstart=True, lr_schedule="cosine",
                                        input_noise_sigma=0.05, rollout_steps=20,
                                        rollout_weight=0.5),

    # ── Seed variation on the kitchen-sink combo ──
    "C14_combo_seed123":           dict(ridge_warmstart=True, lr_schedule="cosine",
                                        input_noise_sigma=0.05, seed=123),
    "C15_combo_seed7":             dict(ridge_warmstart=True, lr_schedule="cosine",
                                        input_noise_sigma=0.05, seed=7),
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
