"""CLI entry point for the Atlas-indexed Transformer.

Usage
-----
# Train + evaluate on all worms (default: joint neural+behaviour):
python -m atlas_transformer \
    --h5_dir "data/used/behaviour+neuronal activity atanas (2023)/2" \
    --save_dir output_plots/atlas_transformer/run1 \
    --device cuda

# No behaviour prediction:
python -m atlas_transformer \
    --h5_dir "data/used/..." --save_dir ... --no_beh

# Eval-only (load existing model):
python -m atlas_transformer \
    --h5_dir "data/used/..." --save_dir ... --eval_only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .config import AtlasTransformerConfig
from .dataset import (
    load_all_worms_atlas,
    build_dataloaders,
    temporal_train_val_test_split,
    build_joint_state_atlas,
)
from .model import build_atlas_model
from .train import train_atlas_model, train_atlas_model_cv
from .evaluate import (
    run_per_worm_evaluation,
    run_per_worm_evaluation_cv,
    save_per_worm_evaluation,
)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Atlas-indexed Transformer for multi-worm neural dynamics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--h5_dir", type=str, required=True,
        help="Directory containing worm .h5 files",
    )
    p.add_argument("--save_dir", type=str, required=True, help="Output directory")
    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])
    p.add_argument("--eval_only", action="store_true", help="Only evaluate existing model")
    p.add_argument("--loo_subset", type=int, default=20, help="LOO subset size (0=all)")
    p.add_argument("--no_cv", action="store_true",
                   help="Disable 5-fold CV (use legacy single-split training)")
    p.add_argument("--n_cv_folds", type=int, default=None,
                   help="Number of CV folds (default: 5)")

    # Behaviour flags
    p.add_argument("--no_beh", action="store_true",
                   help="Disable behaviour prediction (neural only)")
    p.add_argument("--n_beh_modes", type=int, default=None,
                   help="Number of eigenworm modes (default: 6)")
    p.add_argument("--w_beh", type=float, default=None,
                   help="Behaviour loss weight (default: 1.0)")

    # Architecture overrides
    p.add_argument("--d_model", type=int, default=None)
    p.add_argument("--n_heads", type=int, default=None)
    p.add_argument("--n_layers", type=int, default=None)
    p.add_argument("--d_ff", type=int, default=None)
    p.add_argument("--context_length", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--max_epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)

    return p.parse_args(argv)


def _apply_overrides(
    cfg: AtlasTransformerConfig, args: argparse.Namespace
) -> AtlasTransformerConfig:
    for field in ["d_model", "n_heads", "n_layers", "d_ff", "context_length",
                  "dropout", "lr", "batch_size", "max_epochs", "patience",
                  "n_beh_modes", "w_beh"]:
        val = getattr(args, field, None)
        if val is not None:
            setattr(cfg, field, val)

    # --no_beh disables behaviour input/prediction
    if args.no_beh:
        cfg.predict_beh = False
        cfg.include_beh_input = False

    if getattr(args, 'n_cv_folds', None) is not None:
        cfg.n_cv_folds = args.n_cv_folds

    return cfg


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    cfg = AtlasTransformerConfig(device=args.device)
    cfg = _apply_overrides(cfg, args)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    use_cv = not args.no_cv
    include_beh = cfg.predict_beh and cfg.include_beh_input
    n_beh = cfg.n_beh_modes if include_beh else 0

    print("=" * 70)
    print("  Atlas-Indexed Transformer — Multi-Worm Training")
    print("=" * 70)
    print(f"  h5_dir:   {args.h5_dir}")
    print(f"  save_dir: {args.save_dir}")
    print(f"  device:   {args.device}")
    print(f"  mode:     {'5-fold CV per-worm' if use_cv else 'single-split unified'}")
    print(f"  behaviour: {'ON (n_beh=' + str(n_beh) + ', w_beh=' + str(cfg.w_beh) + ')' if include_beh else 'OFF'}")
    print(f"  config:   d_model={cfg.d_model}, n_heads={cfg.n_heads}, "
          f"n_layers={cfg.n_layers}, d_ff={cfg.d_ff}, K={cfg.context_length}")
    print()

    # ── Load all worms ──
    data = load_all_worms_atlas(args.h5_dir, cfg)
    worms = data["worms"]
    n_atlas = data["n_atlas"]
    assert n_atlas == cfg.n_atlas, f"Atlas mismatch: {n_atlas} vs {cfg.n_atlas}"

    # Save config + data summary
    summary = {
        "n_worms": len(worms),
        "n_atlas": n_atlas,
        "n_beh": n_beh,
        "include_beh": include_beh,
        "use_cv": use_cv,
        "n_cv_folds": cfg.n_cv_folds if use_cv else 0,
        "worms": [
            {"worm_id": w["worm_id"], "N_obs": w["N_obs"], "T": w["T"]}
            for w in worms
        ],
        "config": asdict(cfg),
        "coverage_observed": int((data["coverage"] > 0).sum()),
        "coverage_never": int((data["coverage"] == 0).sum()),
    }
    (save_dir / "data_summary.json").write_text(json.dumps(summary, indent=2))

    all_results = []

    if use_cv:
        # ── Per-worm 5-fold CV ──
        print(f"\n[Atlas] Per-worm {cfg.n_cv_folds}-fold CV training + evaluation")
        print("=" * 70)

        for wi, worm in enumerate(worms):
            worm_id = worm["worm_id"]
            print(f"\n{'─'*70}")
            print(f"  Worm {wi+1}/{len(worms)}: {worm_id}")
            print(f"{'─'*70}")

            # Train with CV
            train_result = train_atlas_model_cv(
                worm=worm,
                cfg=cfg,
                device=args.device,
                verbose=True,
                save_dir=str(save_dir),
            )

            # Evaluate with CV-stitched predictions
            eval_result = run_per_worm_evaluation_cv(
                train_result=train_result,
                worm=worm,
                cfg=cfg,
                loo_subset_size=args.loo_subset,
                verbose=True,
            )
            save_per_worm_evaluation(eval_result, str(save_dir), worm_id)
            all_results.append(eval_result)

    else:
        # ── Legacy: single-split unified model ──
        train_loader, val_loader, splits = build_dataloaders(
            worms, cfg, args.device)

        model_path = save_dir / "atlas_model.pt"

        if args.eval_only and model_path.exists():
            print(f"\n[Atlas] Loading existing model from {model_path}")
            model = build_atlas_model(cfg, device=args.device, n_beh=n_beh)
            state = torch.load(model_path, map_location=args.device,
                               weights_only=True)
            model.load_state_dict(state)
            model.eval()
        else:
            print("\n[Atlas] Training unified model...")
            train_result = train_atlas_model(
                train_loader=train_loader,
                val_loader=val_loader,
                cfg=cfg,
                device=args.device,
                save_dir=str(save_dir),
                verbose=True,
                n_beh=n_beh,
            )
            model = train_result["model"]

        # ── Per-worm evaluation (legacy, in-sample one-step) ──
        print("\n" + "=" * 70)
        print("  Per-Worm Evaluation (single-split, in-sample one-step)")
        print("=" * 70)

        device = args.device
        model = model.to(device)
        model.eval()

        for worm in worms:
            worm_id = worm["worm_id"]
            eval_result = run_per_worm_evaluation(
                model=model,
                worm=worm,
                cfg=cfg,
                loo_subset_size=args.loo_subset,
                verbose=True,
            )
            save_per_worm_evaluation(eval_result, str(save_dir), worm_id)
            all_results.append(eval_result)

    # ── Aggregate summary ──
    print("\n" + "=" * 70)
    print(f"  Summary ({'5-fold CV' if use_cv else 'single-split'})")
    print("=" * 70)

    loo_r2s = [r["loo"]["r2_mean"] for r in all_results if np.isfinite(r["loo"]["r2_mean"])]
    fr_r2s = [r["free_run"]["r2_mean"] for r in all_results if np.isfinite(r["free_run"]["r2_mean"])]
    os_r2s = [r["onestep"]["r2_mean"] for r in all_results if np.isfinite(r["onestep"]["r2_mean"])]
    beh_mlp_r2s = [
        r["behaviour_mlp"]["r2_model_mean"]
        for r in all_results
        if r.get("behaviour_mlp") and r["behaviour_mlp"].get("r2_model_mean") is not None
        and np.isfinite(r["behaviour_mlp"]["r2_model_mean"])
    ]
    beh_ridge_r2s = [
        r["behaviour_ridge"]["r2_model_mean"]
        for r in all_results
        if r.get("behaviour_ridge") and r["behaviour_ridge"].get("r2_model_mean") is not None
        and np.isfinite(r["behaviour_ridge"]["r2_model_mean"])
    ]
    beh_direct_r2s = [
        r["onestep_beh"]["r2_mean"]
        for r in all_results
        if r.get("onestep_beh") and r["onestep_beh"].get("r2_mean") is not None
        and np.isfinite(r["onestep_beh"]["r2_mean"])
    ]

    print(f"\n  {'Worm':<20s} {'N_obs':>5s} {'1-step':>7s} {'LOO':>7s} {'Free':>7s} "
          f"{'BehDir':>7s} {'BehRdg':>7s} {'BehMLP':>7s}")
    print(f"  {'-'*72}")
    for r in all_results:
        def _fmt(section, key):
            v = r.get(section, {})
            if v and v.get(key) is not None and np.isfinite(v[key]):
                return f"{v[key]:.4f}"
            return "  N/A"
        print(
            f"  {r['worm_id']:<20s} {r['N_obs']:>5d} "
            f"{r['onestep']['r2_mean']:>7.4f} "
            f"{r['loo']['r2_mean']:>7.4f} "
            f"{r['free_run']['r2_mean']:>7.4f} "
            f"{_fmt('onestep_beh', 'r2_mean'):>7s} "
            f"{_fmt('behaviour_ridge', 'r2_model_mean'):>7s} "
            f"{_fmt('behaviour_mlp', 'r2_model_mean'):>7s}"
        )

    def _safe_mean(lst):
        return float(np.mean(lst)) if lst else float("nan")

    print(f"\n  {'MEAN':<20s} {'':>5s} "
          f"{_safe_mean(os_r2s):>7.4f} "
          f"{_safe_mean(loo_r2s):>7.4f} "
          f"{_safe_mean(fr_r2s):>7.4f} "
          f"{_safe_mean(beh_direct_r2s):>7.4f} "
          f"{_safe_mean(beh_ridge_r2s):>7.4f} "
          f"{_safe_mean(beh_mlp_r2s):>7.4f}")

    # Save aggregate
    agg = {
        "n_worms": len(all_results),
        "n_beh": n_beh,
        "use_cv": use_cv,
        "n_cv_folds": cfg.n_cv_folds if use_cv else 0,
        "onestep_r2_mean": _safe_mean(os_r2s),
        "loo_r2_mean": _safe_mean(loo_r2s),
        "free_run_r2_mean": _safe_mean(fr_r2s),
        "beh_direct_r2_mean": _safe_mean(beh_direct_r2s),
        "beh_ridge_r2_mean": _safe_mean(beh_ridge_r2s),
        "beh_mlp_r2_mean": _safe_mean(beh_mlp_r2s),
        "per_worm": all_results,
    }

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

    (save_dir / "aggregate_results.json").write_text(
        json.dumps(_sanitize(agg), indent=2)
    )
    print(f"\n  Results saved to {save_dir}/")


if __name__ == "__main__":
    main()
