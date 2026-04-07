"""CLI entry point for the Transformer baseline.

Usage
-----
# Train + evaluate a single worm (5-fold CV, joint neural+beh):
python -m baseline_transformer.run_baseline \
    --h5 data/used/behaviour+neuronal\ activity\ atanas\ \(2023\)/2/worm0.h5 \
    --save_dir output_plots/transformer_baseline/single

# Train + evaluate all worms in a directory:
python -m baseline_transformer.run_baseline \
    --h5_dir "data/used/behaviour+neuronal activity atanas (2023)/2" \
    --save_dir output_plots/transformer_baseline/batch \
    --device cuda

# Disable behaviour prediction (neural-only):
python -m baseline_transformer.run_baseline \
    --h5 worm0.h5 --save_dir out --no_beh

# Legacy single-split mode (no CV):
python -m baseline_transformer.run_baseline \
    --h5 worm0.h5 --save_dir out --no_cv
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

from .config import TransformerBaselineConfig
from .dataset import discover_worm_files, load_worm_data
from .evaluate import (
    run_full_evaluation,
    run_full_evaluation_cv,
    save_evaluation,
)
from .train import train_single_worm, train_single_worm_cv


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Transformer baseline for C. elegans neural dynamics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--h5", type=str, help="Path to a single worm .h5 file")
    g.add_argument("--h5_dir", type=str, help="Directory containing worm .h5 files")

    p.add_argument("--save_dir", type=str, required=True, help="Output directory")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--sweep", action="store_true", help="Run hyperparameter sweep")
    p.add_argument("--eval_only", action="store_true", help="Only evaluate (load existing model)")
    p.add_argument("--loo_subset", type=int, default=20, help="LOO subset size (0=all)")

    # CV and behaviour flags
    p.add_argument("--no_cv", action="store_true",
                   help="Disable cross-validation (use legacy 70/15/15 split)")
    p.add_argument("--n_cv_folds", type=int, default=None, help="Number of CV folds")
    p.add_argument("--no_beh", action="store_true",
                   help="Disable behaviour input/prediction (neural only)")
    p.add_argument("--n_beh_modes", type=int, default=None, help="Number of eigenworm modes")
    p.add_argument("--w_beh", type=float, default=None, help="Weight on behaviour loss")

    # Override config fields
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

    # Diffusion head
    p.add_argument("--use_diffusion", action="store_true",
                   help="Use diffusion denoising head instead of Gaussian NLL")

    return p.parse_args(argv)


def _apply_overrides(cfg: TransformerBaselineConfig, args: argparse.Namespace) -> TransformerBaselineConfig:
    """Apply CLI overrides to config."""
    for field in ["d_model", "n_heads", "n_layers", "d_ff", "context_length",
                  "dropout", "lr", "batch_size", "max_epochs", "patience",
                  "n_cv_folds", "n_beh_modes", "w_beh"]:
        val = getattr(args, field, None)
        if val is not None:
            setattr(cfg, field, val)

    if args.no_beh:
        cfg.include_beh_input = False
        cfg.predict_beh = False

    if getattr(args, "use_diffusion", False):
        cfg.use_diffusion = True

    return cfg


def _run_one_worm(
    h5_path: str,
    cfg: TransformerBaselineConfig,
    save_dir: str,
    device: str,
    loo_subset: int,
    eval_only: bool = False,
    use_cv: bool = True,
) -> Dict[str, Any]:
    """Train (or load) + evaluate one worm."""
    worm_data = load_worm_data(h5_path, n_beh_modes=cfg.n_beh_modes)
    worm_id = worm_data["worm_id"]
    u = worm_data["u"]
    b = worm_data.get("b")
    b_mask = worm_data.get("b_mask")

    print(f"\n{'='*60}")
    print(f"Worm: {worm_id}  (T={worm_data['T']}, N={worm_data['N_obs']})")
    if b is not None:
        print(f"  Behaviour: {b.shape[1]} modes, "
              f"valid={np.mean(b_mask > 0.5):.1%}" if b_mask is not None else "")
    print(f"  Mode: {'CV' if use_cv else 'single-split'}, "
          f"predict_beh={cfg.predict_beh}, include_beh={cfg.include_beh_input}")
    print(f"{'='*60}")

    import torch

    if use_cv:
        # ---- 5-fold CV training ----
        train_result = train_single_worm_cv(
            u=u, cfg=cfg, device=device, verbose=True,
            save_dir=save_dir, worm_id=worm_id,
            b=b, b_mask=b_mask,
        )

        # Evaluate
        eval_results = run_full_evaluation_cv(
            train_result=train_result,
            worm_data=worm_data,
            cfg=cfg,
            loo_subset_size=loo_subset,
            verbose=True,
        )
    else:
        # ---- Legacy single-split ----
        model_path = Path(save_dir) / worm_id / "model.pt"

        if eval_only and model_path.exists():
            from .model import build_model
            from .dataset import build_joint_state, temporal_train_val_test_split

            x, x_mask, n_neural, n_beh = build_joint_state(
                u, b, b_mask, include_beh=cfg.include_beh_input and cfg.predict_beh,
            )
            model = build_model(n_neural, cfg, device=device, n_beh=n_beh)
            state = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(state)
            model.eval()

            meta_path = Path(save_dir) / worm_id / "train_meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                split = {k: tuple(v) for k, v in meta["split"].items()}
            else:
                split = temporal_train_val_test_split(worm_data["T"], cfg.train_frac, cfg.val_frac)

            train_result = {"model": model, "split": split, "cfg": cfg}
        else:
            train_result = train_single_worm(
                u=u, cfg=cfg, device=device, verbose=True,
                save_dir=save_dir, worm_id=worm_id,
                b=b, b_mask=b_mask,
            )

        model = train_result["model"]
        model = model.to(device)
        split = train_result.get("split")
        if split is None:
            from .dataset import temporal_train_val_test_split
            split = temporal_train_val_test_split(worm_data["T"], cfg.train_frac, cfg.val_frac)

        eval_results = run_full_evaluation(
            model=model,
            worm_data=worm_data,
            split=split,
            cfg=cfg,
            loo_subset_size=loo_subset,
            verbose=True,
        )

    save_evaluation(eval_results, save_dir, worm_id)
    return eval_results


def _run_sweep(
    h5_files: List[str],
    save_dir: str,
    device: str,
    loo_subset: int,
    use_cv: bool = True,
) -> None:
    """Run hyperparameter sweep over the grid defined in config."""
    base_cfg = TransformerBaselineConfig()
    grid = base_cfg.sweep_grid

    import itertools
    keys = list(grid.keys())
    values = list(grid.values())
    combos = list(itertools.product(*values))

    print(f"\n{'='*60}")
    print(f"SWEEP: {len(combos)} configurations × {len(h5_files)} worms")
    print(f"{'='*60}")

    sweep_files = h5_files[:min(5, len(h5_files))]
    print(f"  Using {len(sweep_files)} worms for sweep")

    all_results = []

    for ci, combo in enumerate(combos):
        cfg = TransformerBaselineConfig()
        for k, v in zip(keys, combo):
            setattr(cfg, k, v)

        combo_name = "_".join(f"{k}={v}" for k, v in zip(keys, combo))
        combo_dir = str(Path(save_dir) / combo_name)
        print(f"\n--- Sweep {ci+1}/{len(combos)}: {combo_name} ---")

        worm_r2s = []
        for h5_path in sweep_files:
            try:
                result = _run_one_worm(h5_path, cfg, combo_dir, device,
                                       loo_subset, use_cv=use_cv)
                worm_r2s.append(result["onestep"]["r2_mean"])
            except Exception as e:
                print(f"  ERROR: {e}")
                worm_r2s.append(float("nan"))

        mean_r2 = float(np.nanmean(worm_r2s))
        summary = {
            "config": combo_name,
            "params": dict(zip(keys, combo)),
            "worm_r2s": worm_r2s,
            "mean_r2": mean_r2,
        }
        all_results.append(summary)
        print(f"  → Mean one-step R² = {mean_r2:.4f}")

    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sweep_summary.json").write_text(
        json.dumps(all_results, indent=2, default=str)
    )

    best = max(all_results, key=lambda x: x["mean_r2"] if np.isfinite(x["mean_r2"]) else -np.inf)
    print(f"\n{'='*60}")
    print(f"BEST CONFIG: {best['config']}")
    print(f"  Mean R² = {best['mean_r2']:.4f}")
    print(f"{'='*60}")


def _aggregate_results(
    all_results: List[Dict[str, Any]],
    save_dir: str,
) -> None:
    """Print and save aggregate table across worms."""
    print(f"\n{'='*60}")
    print(f"AGGREGATE RESULTS ({len(all_results)} worms)")
    print(f"{'='*60}")

    # Detect whether we have CV results (behaviour_direct) or legacy
    has_direct_beh = any(
        r.get("behaviour_direct", {}).get("r2_mean") is not None
        for r in all_results
    )

    if has_direct_beh:
        header = (f"{'Worm':<20} {'1-step':>8} {'BehDir':>8} "
                  f"{'LOO':>8} {'LOO-W':>8} {'FreeRun':>8} "
                  f"{'BehRdg':>8}")
    else:
        header = (f"{'Worm':<20} {'1-step':>8} {'LOO':>8} "
                  f"{'LOO-W':>8} {'FreeRun':>8} {'Beh(M)':>8} {'Beh(GT)':>8}")

    print(header)
    print("-" * len(header))

    rows = []
    for r in all_results:
        wid = r.get("worm_id", "?")
        os_r2 = r.get("onestep", {}).get("r2_mean", float("nan"))
        loo_r2 = r.get("loo", {}).get("r2_mean", float("nan"))
        loo_w_r2 = r.get("loo_windowed", {}).get("r2_mean", float("nan"))
        fr_r2 = r.get("free_run", {}).get("r2_mean", float("nan"))

        # Behaviour: prefer direct, fall back to ridge
        beh_direct = r.get("onestep_beh", {}).get("r2_mean") or \
                     r.get("behaviour_direct", {}).get("r2_mean") or float("nan")
        beh_ridge = r.get("behaviour_ridge", {}).get("r2_model_mean") or \
                    r.get("behaviour", {}).get("r2_model_mean") or float("nan")
        beh_gt = r.get("behaviour_ridge", {}).get("r2_gt_mean") or \
                 r.get("behaviour", {}).get("r2_gt_mean") or float("nan")

        if has_direct_beh:
            print(f"{wid:<20} {os_r2:>8.4f} {beh_direct:>8.4f} "
                  f"{loo_r2:>8.4f} {loo_w_r2:>8.4f} {fr_r2:>8.4f} "
                  f"{beh_ridge:>8.4f}")
        else:
            print(f"{wid:<20} {os_r2:>8.4f} {loo_r2:>8.4f} {loo_w_r2:>8.4f} "
                  f"{fr_r2:>8.4f} {beh_ridge:>8.4f} {beh_gt:>8.4f}")

        rows.append({
            "worm_id": wid,
            "onestep_r2": os_r2,
            "beh_direct_r2": beh_direct,
            "loo_r2": loo_r2,
            "loo_windowed_r2": loo_w_r2,
            "free_run_r2": fr_r2,
            "beh_ridge_r2": beh_ridge,
            "beh_gt_r2": beh_gt,
        })

    def _safe_mean(key):
        vals = [r[key] for r in rows if np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    print("-" * len(header))
    print(f"{'MEAN':<20} {_safe_mean('onestep_r2'):>8.4f} "
          f"{_safe_mean('beh_direct_r2'):>8.4f} "
          f"{_safe_mean('loo_r2'):>8.4f} "
          f"{_safe_mean('loo_windowed_r2'):>8.4f} "
          f"{_safe_mean('free_run_r2'):>8.4f} "
          f"{_safe_mean('beh_ridge_r2'):>8.4f}")

    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "aggregate_results.json").write_text(json.dumps(rows, indent=2, default=str))

    with open(out / "aggregate_results.tsv", "w") as f:
        f.write("worm_id\tonestep_r2\tbeh_direct_r2\tloo_r2\tloo_windowed_r2\t"
                "free_run_r2\tbeh_ridge_r2\tbeh_gt_r2\n")
        for r in rows:
            f.write(f"{r['worm_id']}\t{r['onestep_r2']:.6f}\t"
                    f"{r['beh_direct_r2']:.6f}\t{r['loo_r2']:.6f}\t"
                    f"{r['loo_windowed_r2']:.6f}\t"
                    f"{r['free_run_r2']:.6f}\t{r['beh_ridge_r2']:.6f}\t"
                    f"{r['beh_gt_r2']:.6f}\n")


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    cfg = _apply_overrides(TransformerBaselineConfig(), args)
    use_cv = not args.no_cv

    # Save config
    out = Path(args.save_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))

    # Discover files
    if args.h5 is not None:
        h5_files = [args.h5]
    else:
        h5_files = discover_worm_files(args.h5_dir)
        if not h5_files:
            print(f"No .h5 files found in {args.h5_dir}", file=sys.stderr)
            sys.exit(1)

    print(f"Found {len(h5_files)} worm file(s)")
    print(f"Config: d_model={cfg.d_model}, n_layers={cfg.n_layers}, "
          f"n_heads={cfg.n_heads}, K={cfg.context_length}")
    print(f"CV: {'off' if not use_cv else f'{cfg.n_cv_folds}-fold'}, "
          f"predict_beh={cfg.predict_beh}, n_beh_modes={cfg.n_beh_modes}")
    print(f"Device: {args.device}")
    print(f"Save dir: {args.save_dir}")

    if args.sweep:
        _run_sweep(h5_files, args.save_dir, args.device, args.loo_subset,
                   use_cv=use_cv)
        return

    # Single run: train + evaluate all worms
    t0 = time.time()
    all_results = []

    for i, h5_path in enumerate(h5_files):
        print(f"\n[{i+1}/{len(h5_files)}] {Path(h5_path).stem}")
        try:
            result = _run_one_worm(
                h5_path, cfg, args.save_dir, args.device, args.loo_subset,
                eval_only=args.eval_only,
                use_cv=use_cv,
            )
            all_results.append(result)
        except Exception as e:
            print(f"  ERROR processing {h5_path}: {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    if all_results:
        _aggregate_results(all_results, args.save_dir)


if __name__ == "__main__":
    main()
