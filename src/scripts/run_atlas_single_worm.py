#!/usr/bin/env python3
"""
Atlas Transformer — Single-worm training + LOO/free-run evaluation.

Trains the atlas transformer on ONE worm with 5-fold temporal CV,
then evaluates:
  1. LOO R² (per-neuron): mask one neuron, predict from the rest (autoregressive)
  2. Free-run R²: replace ALL neurons with model predictions
  3. One-step R² (teacher-forced, for reference)

Results are saved alongside a comparison table against neural_activity_decoder_v4
so you can directly compare:
  - Atlas TRF LOO  vs  v4 TRF causal_self  (both = "predict neuron i from the rest")
  - Atlas TRF free-run  (fully autonomous)

Usage:
    python -u -m scripts.run_atlas_single_worm \
        --worm 2022-06-14-01 --device cuda

    # All 8 v4 worms:
    python -u -m scripts.run_atlas_single_worm \
        --worm 2022-06-14-01 2022-06-14-07 2022-07-15-06 2022-07-15-12 \
               2022-12-21-06 2023-01-05-01 2023-01-06-15 2023-01-09-08 \
        --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from atlas_transformer.config import AtlasTransformerConfig
from atlas_transformer.dataset import (
    load_atlas_worm_data,
    build_joint_state_atlas,
    make_cv_folds,
)
from atlas_transformer.train import train_atlas_model_cv
from atlas_transformer.evaluate import (
    compute_onestep_r2_from_preds,
    compute_loo_r2_atlas,
    compute_free_run_r2_atlas,
)

from stage2.io_multi import (
    _load_full_atlas,
    _atlas_embedding_indices,
)


# ── Defaults ────────────────────────────────────────────────────────────────

H5_DIR = "data/used/behaviour+neuronal activity atanas (2023)/2"
V4_DIR = "output_plots/neural_activity_decoder_v4"
OUT_DIR = "output_plots/atlas_single_worm_vs_v4"


# ── Load v4 results for comparison ──────────────────────────────────────────

def _load_v4_results(worm_id: str) -> dict | None:
    """Load v4 results.json for a given worm (if available)."""
    path = ROOT / V4_DIR / f"{worm_id}_all" / "results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _v4_summary(v4: dict, K: int = 5) -> dict:
    """Extract per-condition mean R² from v4 results at given K."""
    res_k = v4.get(str(K), {})
    out = {}
    for model in ("ridge", "mlp", "trf"):
        for cond in ("self", "causal", "conc", "causal_self", "conc_self", "conc_causal"):
            key = f"r2_mean_{model}_{cond}"
            if key in res_k:
                out[f"{model}_{cond}"] = res_k[key]
    return out


# ── Run one worm ────────────────────────────────────────────────────────────

def run_single_worm(
    worm_id: str,
    *,
    device: str = "cuda",
    out_dir: Path,
    loo_subset: int = 0,
    no_beh: bool = False,
    context_length: int = 16,
    max_epochs: int = 300,
    patience: int = 30,
) -> dict:
    """Train atlas TRF on one worm, evaluate LOO + free-run."""

    h5_path = str(ROOT / H5_DIR / f"{worm_id}.h5")
    worm_out = out_dir / worm_id
    worm_out.mkdir(parents=True, exist_ok=True)

    # ── Config ──
    cfg = AtlasTransformerConfig(device=device)
    cfg.context_length = context_length
    cfg.max_epochs = max_epochs
    cfg.patience = patience
    if no_beh:
        cfg.predict_beh = False
        cfg.include_beh_input = False

    # ── Load worm into atlas ──
    full_labels = _load_full_atlas()
    n_atlas = len(full_labels)
    atlas_idx = np.arange(n_atlas, dtype=np.int64)

    print(f"\n{'='*70}")
    print(f"  Atlas TRF — Single Worm: {worm_id}")
    print(f"{'='*70}")
    print(f"  device={device}, K={cfg.context_length}, "
          f"d_model={cfg.d_model}, n_heads={cfg.n_heads}, "
          f"n_layers={cfg.n_layers}")

    worm = load_atlas_worm_data(
        h5_path, full_labels, atlas_idx, n_atlas,
        n_beh_modes=cfg.n_beh_modes,
    )
    if worm is None:
        print(f"  FAILED to load {worm_id}")
        return {"worm_id": worm_id, "error": "load_failed"}

    print(f"  N_obs={worm['N_obs']}, T={worm['T']}")

    # ── Train with 5-fold CV ──
    t0 = time.time()
    train_result = train_atlas_model_cv(
        worm=worm,
        cfg=cfg,
        device=device,
        verbose=True,
        save_dir=str(worm_out),
    )
    train_time = time.time() - t0
    print(f"\n  Training done in {train_time:.1f}s")

    # ── Evaluate ──
    include_beh = cfg.predict_beh and cfg.include_beh_input
    n_beh = cfg.n_beh_modes if include_beh else 0
    x = train_result["x"]
    obs_mask = worm["obs_mask"]
    best_model = train_result["best_model"]
    pred_u_full = train_result["pred_u_full"]
    pred_b_full = train_result.get("pred_b_full")
    K = cfg.context_length

    # 1. One-step R² (stitched from CV — out-of-sample)
    print(f"\n  One-step R² (CV held-out)...")
    onestep = compute_onestep_r2_from_preds(
        pred_u=pred_u_full,
        gt_u=worm["u_atlas"],
        obs_mask=obs_mask,
        pred_b=pred_b_full,
        gt_b=worm.get("b") if n_beh > 0 else None,
        b_mask=worm.get("b_mask") if n_beh > 0 else None,
    )
    print(f"  One-step R² = {onestep['r2_mean']:.4f}")

    # 2. LOO R² (per-neuron, autoregressive — best-fold model)
    print(f"\n  LOO R² (autoregressive, all observed neurons)...")
    loo = compute_loo_r2_atlas(
        best_model, x, obs_mask,
        subset_size=loo_subset,
        verbose=True,
    )
    print(f"  LOO R² mean = {loo['r2_mean']:.4f}")

    # Windowed LOO
    print(f"\n  LOO R² (windowed, w=50)...")
    loo_w = compute_loo_r2_atlas(
        best_model, x, obs_mask,
        subset_size=loo_subset,
        verbose=True,
        window_size=50,
    )
    print(f"  LOO-windowed R² mean = {loo_w['r2_mean']:.4f}")

    # 3. Free-run R² (fully autoregressive)
    print(f"\n  Free-run R² (fully autoregressive)...")
    free_run = compute_free_run_r2_atlas(
        best_model, x, obs_mask,
        motor_idx_atlas=worm.get("motor_idx_atlas"),
    )
    print(f"  Free-run R² = {free_run['r2_mean']:.4f} ({free_run['mode']})")

    # ── Comparison with v4 ──
    v4 = _load_v4_results(worm_id)
    v4_summary = _v4_summary(v4) if v4 else {}

    # ── Results dict ──
    results = {
        "worm_id": worm_id,
        "N_obs": worm["N_obs"],
        "T": worm["T"],
        "n_atlas": n_atlas,
        "context_length": cfg.context_length,
        "d_model": cfg.d_model,
        "n_heads": cfg.n_heads,
        "n_layers": cfg.n_layers,
        "n_folds": cfg.n_cv_folds,
        "train_time_s": round(train_time, 1),
        "atlas_trf": {
            "onestep_r2_mean": onestep["r2_mean"],
            "onestep_r2_per_neuron": onestep["r2"].tolist(),
            "loo_r2_mean": loo["r2_mean"],
            "loo_r2_per_neuron": loo["r2"].tolist(),
            "loo_windowed_r2_mean": loo_w["r2_mean"],
            "loo_windowed_r2_per_neuron": loo_w["r2"].tolist(),
            "free_run_r2_mean": free_run["r2_mean"],
            "free_run_r2_per_neuron": free_run["r2"].tolist(),
            "free_run_mode": free_run["mode"],
        },
        "v4_comparison": v4_summary,
    }

    # ── Print comparison table ──
    print(f"\n{'='*70}")
    print(f"  COMPARISON: Atlas TRF vs v4 — {worm_id}")
    print(f"{'='*70}")
    print(f"\n  Atlas TRF (single-worm, K={cfg.context_length}, 5-fold CV):")
    print(f"    One-step R²      = {onestep['r2_mean']:.4f}")
    print(f"    LOO R²           = {loo['r2_mean']:.4f}")
    print(f"    LOO-windowed R²  = {loo_w['r2_mean']:.4f}")
    print(f"    Free-run R²      = {free_run['r2_mean']:.4f}")

    if v4_summary:
        print(f"\n  v4 (per-neuron retrained, K=5, 5-fold CV, 1-step):")
        for key in ("trf_causal_self", "trf_conc_self", "trf_self",
                     "ridge_causal_self", "ridge_conc_self", "ridge_self"):
            if key in v4_summary:
                model, cond = key.split("_", 1)
                print(f"    {model:>8s} {cond:<15s} R² = {v4_summary[key]:.4f}")

        print(f"\n  Key comparison (predict neuron i from the rest):")
        print(f"    v4 TRF causal_self (1-step) = "
              f"{v4_summary.get('trf_causal_self', float('nan')):.4f}")
        print(f"    v4 Ridge conc_self (1-step)  = "
              f"{v4_summary.get('ridge_conc_self', float('nan')):.4f}")
        print(f"    Atlas TRF LOO (free-run)     = {loo['r2_mean']:.4f}")
        print(f"    Atlas TRF free-run (all)     = {free_run['r2_mean']:.4f}")

    # ── Save ──
    def _sanitize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize(v) for v in obj]
        return obj

    (worm_out / "results.json").write_text(
        json.dumps(_sanitize(results), indent=2)
    )
    print(f"\n  Saved to {worm_out}/")

    return results


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Atlas TRF single-worm training + LOO/free-run eval vs v4"
    )
    parser.add_argument(
        "--worm", nargs="+",
        default=["2022-06-14-01"],
        help="Worm ID(s) to process",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=OUT_DIR)
    parser.add_argument("--loo_subset", type=int, default=0,
                        help="LOO subset size (0=all observed neurons)")
    parser.add_argument("--no_beh", action="store_true",
                        help="Disable behaviour (neural-only)")
    parser.add_argument("--context_length", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for worm_id in args.worm:
        res = run_single_worm(
            worm_id,
            device=args.device,
            out_dir=out_dir,
            loo_subset=args.loo_subset,
            no_beh=args.no_beh,
            context_length=args.context_length,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )
        all_results.append(res)

    # ── Aggregate summary ──
    if len(all_results) > 1:
        print(f"\n\n{'='*70}")
        print(f"  AGGREGATE SUMMARY ({len(all_results)} worms)")
        print(f"{'='*70}")

        header = (f"{'Worm':<18s} {'N_obs':>5s} {'1-step':>7s} "
                  f"{'LOO':>7s} {'LOO-w':>7s} {'Free':>7s} "
                  f"{'v4-TRF-cs':>10s} {'v4-Rdg-cs':>10s}")
        print(f"\n  {header}")
        print(f"  {'-'*len(header)}")

        for r in all_results:
            if "error" in r:
                print(f"  {r['worm_id']:<18s}  ERROR")
                continue
            a = r["atlas_trf"]
            v4 = r.get("v4_comparison", {})
            v4_trf = v4.get("trf_causal_self", float("nan"))
            v4_rdg = v4.get("ridge_conc_self", float("nan"))
            print(
                f"  {r['worm_id']:<18s} {r['N_obs']:>5d} "
                f"{a['onestep_r2_mean']:>7.4f} "
                f"{a['loo_r2_mean']:>7.4f} "
                f"{a['loo_windowed_r2_mean']:>7.4f} "
                f"{a['free_run_r2_mean']:>7.4f} "
                f"{v4_trf:>10.4f} "
                f"{v4_rdg:>10.4f}"
            )

        def _mean(vals):
            v = [x for x in vals if np.isfinite(x)]
            return np.mean(v) if v else float("nan")

        valid = [r for r in all_results if "error" not in r]
        print(
            f"\n  {'MEAN':<18s} {'':>5s} "
            f"{_mean([r['atlas_trf']['onestep_r2_mean'] for r in valid]):>7.4f} "
            f"{_mean([r['atlas_trf']['loo_r2_mean'] for r in valid]):>7.4f} "
            f"{_mean([r['atlas_trf']['loo_windowed_r2_mean'] for r in valid]):>7.4f} "
            f"{_mean([r['atlas_trf']['free_run_r2_mean'] for r in valid]):>7.4f} "
            f"{_mean([r.get('v4_comparison', {}).get('trf_causal_self', float('nan')) for r in valid]):>10.4f} "
            f"{_mean([r.get('v4_comparison', {}).get('ridge_conc_self', float('nan')) for r in valid]):>10.4f}"
        )

    # Save aggregate
    def _sanitize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize(v) for v in obj]
        return obj

    (out_dir / "all_results.json").write_text(
        json.dumps(_sanitize(all_results), indent=2)
    )
    print(f"\n  All results saved to {out_dir}/all_results.json")


if __name__ == "__main__":
    main()
