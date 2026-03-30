#!/usr/bin/env python
"""Transformer architecture sweep on the best worms.

Sweeps over multiple architecture configurations:
  - d_model / n_heads / d_ff
  - n_layers
  - context_length
  - dropout
  - learning rate / weight decay
  - scheduled-sampling parameters

Runs each config on the top-N worms (by behaviour-decoder R²),
collects one-step R², LOO R², free-run R², and behaviour R²,
then prints a ranked summary table and saves results JSON.

Usage
-----
python -m scripts.transformer_arch_sweep --device cuda
python -m scripts.transformer_arch_sweep --device cuda --fast   # quick smoke test
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline_transformer.config import TransformerBaselineConfig
from baseline_transformer.dataset import load_worm_data, temporal_train_val_test_split
from baseline_transformer.evaluate import run_full_evaluation
from baseline_transformer.train import train_single_worm


# ── Default worms (top-5 by behaviour-decoder Ridge R²) ─────────────────────

BEST_WORMS = [
    "2023-01-17-14",  # R²=0.486, T=1615, N=24
    "2022-06-14-07",  # R²=0.449, T=1600, N=25
    "2023-01-10-07",  # R²=0.374, T=1615, N=27
    "2022-06-14-01",  # R²=0.371, T=1600, N=27
    "2023-01-09-28",  # R²=0.369, T=1615, N=24
]

H5_DIR = "data/used/behaviour+neuronal activity atanas (2023)/2"


# ── Architecture configurations to sweep ─────────────────────────────────────

def _build_configs() -> List[Tuple[str, TransformerBaselineConfig]]:
    """Return a list of (name, config) pairs to evaluate."""
    configs = []

    # ── A) Baseline (current defaults) ──
    cfg = TransformerBaselineConfig()
    configs.append(("A_baseline", cfg))

    # ── B) Wider model ──
    cfg = TransformerBaselineConfig(d_model=256, n_heads=8, d_ff=512)
    configs.append(("B_wide_256h8", cfg))

    # ── C) Narrower model ──
    cfg = TransformerBaselineConfig(d_model=64, n_heads=4, d_ff=128)
    configs.append(("C_narrow_64h4", cfg))

    # ── D) Deeper (3 layers) ──
    cfg = TransformerBaselineConfig(n_layers=3)
    configs.append(("D_3layers", cfg))

    # ── E) Deeper + wider (3 layers, 256) ──
    cfg = TransformerBaselineConfig(d_model=256, n_heads=8, d_ff=512, n_layers=3)
    configs.append(("E_deep_wide", cfg))

    # ── F) 4 layers ──
    cfg = TransformerBaselineConfig(n_layers=4, d_model=128, n_heads=4, d_ff=256)
    configs.append(("F_4layers", cfg))

    # ── G) Longer context (32 frames) ──
    cfg = TransformerBaselineConfig(context_length=32)
    configs.append(("G_ctx32", cfg))

    # ── H) Longer context (64 frames) ──
    cfg = TransformerBaselineConfig(context_length=64)
    configs.append(("H_ctx64", cfg))

    # ── I) Short context (8 frames) ──
    cfg = TransformerBaselineConfig(context_length=8)
    configs.append(("I_ctx8", cfg))

    # ── J) Higher dropout ──
    cfg = TransformerBaselineConfig(dropout=0.2)
    configs.append(("J_drop0.2", cfg))

    # ── K) Lower dropout ──
    cfg = TransformerBaselineConfig(dropout=0.05)
    configs.append(("K_drop0.05", cfg))

    # ── L) Higher learning rate ──
    cfg = TransformerBaselineConfig(lr=1e-3)
    configs.append(("L_lr1e-3", cfg))

    # ── M) Lower learning rate ──
    cfg = TransformerBaselineConfig(lr=1e-4)
    configs.append(("M_lr1e-4", cfg))

    # ── N) More weight decay ──
    cfg = TransformerBaselineConfig(weight_decay=1e-3)
    configs.append(("N_wd1e-3", cfg))

    # ── O) Combined: wider + longer context + more dropout ──
    cfg = TransformerBaselineConfig(
        d_model=256, n_heads=8, d_ff=512,
        context_length=32, dropout=0.2,
    )
    configs.append(("O_wide_ctx32_drop", cfg))

    # ── P) Combined: 3 layers + context 32 + lower lr ──
    cfg = TransformerBaselineConfig(
        n_layers=3, context_length=32, lr=1e-4,
    )
    configs.append(("P_3L_ctx32_lr1e-4", cfg))

    # ── Q) Aggressive scheduled sampling (more free-run-like) ──
    cfg = TransformerBaselineConfig(
        ss_start_epoch=5, ss_end_epoch=50, ss_p_min=0.3,
    )
    configs.append(("Q_aggressiveSS", cfg))

    # ── R) Big model: 256d, 4 layers, ctx=32 ──
    cfg = TransformerBaselineConfig(
        d_model=256, n_heads=8, d_ff=512,
        n_layers=4, context_length=32,
        dropout=0.15, lr=2e-4,
    )
    configs.append(("R_big_4L_ctx32", cfg))

    # ── S) Small efficient: 64d, 2 layers, ctx=32 ──
    cfg = TransformerBaselineConfig(
        d_model=64, n_heads=4, d_ff=128,
        n_layers=2, context_length=32,
        dropout=0.1, lr=3e-4,
    )
    configs.append(("S_small_ctx32", cfg))

    # ── T) Balanced: 128d, 3 layers, ctx=32, drop=0.15 ──
    cfg = TransformerBaselineConfig(
        d_model=128, n_heads=4, d_ff=256,
        n_layers=3, context_length=32,
        dropout=0.15, lr=2e-4,
    )
    configs.append(("T_balanced", cfg))

    return configs


def _build_fast_configs() -> List[Tuple[str, TransformerBaselineConfig]]:
    """Subset of configs for a quick smoke test."""
    configs = []

    cfg = TransformerBaselineConfig(max_epochs=50, patience=10)
    configs.append(("A_baseline", cfg))

    cfg = TransformerBaselineConfig(d_model=256, n_heads=8, d_ff=512,
                                   max_epochs=50, patience=10)
    configs.append(("B_wide_256h8", cfg))

    cfg = TransformerBaselineConfig(d_model=64, n_heads=4, d_ff=128,
                                   max_epochs=50, patience=10)
    configs.append(("C_narrow_64h4", cfg))

    cfg = TransformerBaselineConfig(n_layers=3, max_epochs=50, patience=10)
    configs.append(("D_3layers", cfg))

    cfg = TransformerBaselineConfig(context_length=32, max_epochs=50, patience=10)
    configs.append(("G_ctx32", cfg))

    cfg = TransformerBaselineConfig(context_length=64, max_epochs=50, patience=10)
    configs.append(("H_ctx64", cfg))

    return configs


# ── Run one config on one worm ───────────────────────────────────────────────

def run_one(
    h5_path: str,
    cfg: TransformerBaselineConfig,
    config_name: str,
    save_dir: str,
    device: str,
    loo_subset: int = 10,
    skip_beh: bool = False,
) -> Dict[str, Any]:
    """Train + evaluate one config on one worm. Returns metrics dict."""
    worm_data = load_worm_data(h5_path)
    worm_id = worm_data["worm_id"]
    u = worm_data["u"]

    cfg = copy.deepcopy(cfg)
    cfg.device = device

    combo_dir = str(Path(save_dir) / config_name)

    t0 = time.time()

    # Train
    train_result = train_single_worm(
        u=u,
        cfg=cfg,
        device=device,
        verbose=True,
        save_dir=combo_dir,
        worm_id=worm_id,
    )

    model = train_result["model"]
    import torch
    model = model.to(device)
    split = train_result["split"]

    # Evaluate
    if skip_beh:
        # Fast evaluation: skip behaviour R² (which is very slow)
        from baseline_transformer.evaluate import (
            compute_onestep_r2, compute_loo_r2, compute_free_run_r2,
        )
        print(f"\n[{worm_id}] Fast evaluation (skip_beh)")
        te_s, te_e = split["test"]

        # 1. One-step R²
        print(f"  Computing one-step R²...")
        onestep = compute_onestep_r2(model, worm_data["u"], start=te_s, end=te_e)
        onestep_full = compute_onestep_r2(model, worm_data["u"])
        print(f"  One-step R² mean = {onestep['r2_mean']:.4f}")

        # 2. LOO R²
        print(f"  Computing LOO R² (subset={loo_subset})...")
        u = worm_data["u"]
        N = u.shape[1]
        variances = np.nanvar(u, axis=0)
        subset = list(np.argsort(variances)[::-1][:min(loo_subset, N)])
        loo = compute_loo_r2(model, u, subset=subset, verbose=True)
        print(f"  LOO R² mean = {loo['r2_mean']:.4f}")

        # 3. Free-run R²
        print(f"  Computing free-run R²...")
        motor_idx = worm_data.get("motor_idx")
        free_run = compute_free_run_r2(model, u, motor_idx=motor_idx)
        print(f"  Free-run R² mean = {free_run['r2_mean']:.4f}  ({free_run['mode']})")

        eval_results = {
            "worm_id": worm_id,
            "onestep": {"r2": onestep["r2"].tolist(), "r2_mean": onestep["r2_mean"],
                        "eval_range": onestep["eval_range"]},
            "onestep_full": {"r2": onestep_full["r2"].tolist(), "r2_mean": onestep_full["r2_mean"]},
            "loo": {"r2": loo["r2"].tolist(), "r2_mean": loo["r2_mean"], "subset": loo["subset"]},
            "free_run": {"r2": free_run["r2"].tolist(), "r2_mean": free_run["r2_mean"],
                         "mode": free_run["mode"]},
            "behaviour": {"r2_model_mean": None, "r2_gt_mean": None},
            "behaviour_all_neurons": {"r2": None, "r2_mean": None},
        }
    else:
        eval_results = run_full_evaluation(
            model=model,
            worm_data=worm_data,
            split=split,
            cfg=cfg,
            loo_subset_size=loo_subset,
            verbose=True,
        )

    elapsed = time.time() - t0

    # Extract key metrics
    metrics = {
        "worm_id": worm_id,
        "config_name": config_name,
        "onestep_r2": eval_results["onestep"]["r2_mean"],
        "onestep_full_r2": eval_results["onestep_full"]["r2_mean"],
        "loo_r2": eval_results["loo"]["r2_mean"],
        "free_run_r2": eval_results["free_run"]["r2_mean"],
        "beh_model_r2": eval_results["behaviour"].get("r2_model_mean"),
        "beh_gt_r2": eval_results["behaviour"].get("r2_gt_mean"),
        "best_val_nll": train_result["best_val_nll"],
        "n_params": train_result["n_params"],
        "elapsed_s": round(elapsed, 1),
    }

    # Save full eval results
    from baseline_transformer.evaluate import save_evaluation
    save_evaluation(eval_results, combo_dir, worm_id)

    return metrics


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Transformer architecture sweep")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--fast", action="store_true", help="Quick test (fewer configs, less training)")
    parser.add_argument("--n_worms", type=int, default=5, help="Number of top worms to use")
    parser.add_argument("--save_dir", type=str,
                        default="output_plots/transformer_baseline/arch_sweep")
    parser.add_argument("--loo_subset", type=int, default=10)
    parser.add_argument("--skip_beh", action="store_true",
                        help="Skip slow behaviour R² evaluation for speed")
    parser.add_argument("--worms", type=str, nargs="*", default=None,
                        help="Override worm IDs (space-separated)")
    parser.add_argument("--configs", type=str, nargs="*", default=None,
                        help="Only run specific config names (space-separated)")
    args = parser.parse_args()

    # Select worms
    if args.worms:
        worm_ids = args.worms
    else:
        worm_ids = BEST_WORMS[:args.n_worms]

    # Resolve H5 paths
    h5_dir = Path(H5_DIR)
    h5_paths = []
    for wid in worm_ids:
        p = h5_dir / f"{wid}.h5"
        if p.exists():
            h5_paths.append(str(p))
        else:
            print(f"  WARNING: {p} not found, skipping")

    if not h5_paths:
        print("ERROR: No valid worm files found", file=sys.stderr)
        sys.exit(1)

    # Select configs
    if args.fast:
        all_configs = _build_fast_configs()
    else:
        all_configs = _build_configs()

    if args.configs:
        selected = set(args.configs)
        all_configs = [(n, c) for n, c in all_configs if n in selected]
        if not all_configs:
            print(f"ERROR: No matching configs for {args.configs}", file=sys.stderr)
            sys.exit(1)

    n_total = len(all_configs) * len(h5_paths)

    print("=" * 80)
    print(f"  TRANSFORMER ARCHITECTURE SWEEP")
    print(f"  {len(all_configs)} configs × {len(h5_paths)} worms = {n_total} runs")
    print(f"  Device: {args.device}")
    print(f"  Save dir: {args.save_dir}")
    print("=" * 80)
    print()

    for name, cfg in all_configs:
        n_params_est = 2 * cfg.n_layers * (4 * cfg.d_model**2 + cfg.d_model * cfg.d_ff * 2)
        print(f"  {name:25s}  d={cfg.d_model} L={cfg.n_layers} h={cfg.n_heads} "
              f"ff={cfg.d_ff} ctx={cfg.context_length} drop={cfg.dropout} "
              f"lr={cfg.lr:.0e} wd={cfg.weight_decay:.0e} ~{n_params_est:,d} params")
    print()

    # ── Run all combinations ──
    all_results: List[Dict[str, Any]] = []
    run_idx = 0
    t0_global = time.time()

    for config_name, cfg in all_configs:
        for h5_path in h5_paths:
            run_idx += 1
            worm_id = Path(h5_path).stem
            print(f"\n{'─'*80}")
            print(f"  [{run_idx}/{n_total}] Config={config_name}  Worm={worm_id}")
            print(f"{'─'*80}")

            try:
                metrics = run_one(
                    h5_path=h5_path,
                    cfg=cfg,
                    config_name=config_name,
                    save_dir=args.save_dir,
                    device=args.device,
                    loo_subset=args.loo_subset,
                    skip_beh=args.skip_beh,
                )
                all_results.append(metrics)

                # Print quick summary  (handle None values from skip_beh)
                def _fmt(v, fmt=".4f"):
                    return f"{v:{fmt}}" if v is not None else "n/a"

                print(f"\n  → 1-step={_fmt(metrics['onestep_r2'])}  "
                      f"LOO={_fmt(metrics['loo_r2'])}  "
                      f"FreeRun={_fmt(metrics['free_run_r2'])}  "
                      f"Beh={_fmt(metrics.get('beh_model_r2'))}  "
                      f"({metrics['elapsed_s']:.0f}s)")

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                # Only append error entry if metrics weren't already appended
                if not all_results or all_results[-1].get("config_name") != config_name \
                        or all_results[-1].get("worm_id") != worm_id:
                    all_results.append({
                        "worm_id": worm_id,
                        "config_name": config_name,
                        "error": str(e),
                    })

    total_elapsed = time.time() - t0_global

    # ── Save full results ──
    out = Path(args.save_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sweep_results.json").write_text(
        json.dumps(all_results, indent=2, default=str)
    )

    # ── Aggregate and rank ──
    print(f"\n\n{'='*100}")
    print(f"  SWEEP RESULTS SUMMARY  ({total_elapsed/60:.1f} min total)")
    print(f"{'='*100}")

    # Group by config
    config_metrics: Dict[str, Dict[str, List[float]]] = {}
    for r in all_results:
        if "error" in r:
            continue
        cn = r["config_name"]
        if cn not in config_metrics:
            config_metrics[cn] = {
                "onestep_r2": [], "loo_r2": [], "free_run_r2": [],
                "beh_model_r2": [], "val_nll": [], "n_params": [],
            }
        config_metrics[cn]["onestep_r2"].append(r.get("onestep_r2", float("nan")))
        config_metrics[cn]["loo_r2"].append(r.get("loo_r2", float("nan")))
        config_metrics[cn]["free_run_r2"].append(r.get("free_run_r2", float("nan")))
        beh = r.get("beh_model_r2")
        config_metrics[cn]["beh_model_r2"].append(beh if beh is not None else float("nan"))
        config_metrics[cn]["val_nll"].append(r.get("best_val_nll", float("nan")))
        config_metrics[cn]["n_params"].append(r.get("n_params", 0))

    # Rank by LOO R² (most meaningful metric)
    ranked = []
    for cn, m in config_metrics.items():
        ranked.append((
            cn,
            float(np.nanmean(m["onestep_r2"])),
            float(np.nanmean(m["loo_r2"])),
            float(np.nanmean(m["free_run_r2"])),
            float(np.nanmean(m["beh_model_r2"])),
            float(np.nanmean(m["val_nll"])),
            int(np.mean(m["n_params"])) if m["n_params"] else 0,
            len(m["loo_r2"]),
        ))

    ranked.sort(key=lambda x: x[2], reverse=True)  # Sort by LOO R²

    header = (f"{'Rank':>4} {'Config':<25s} {'1-step':>8} {'LOO':>8} "
              f"{'FreeRun':>8} {'Beh':>8} {'ValNLL':>8} {'Params':>10} {'N':>3}")
    print(header)
    print("─" * len(header))

    for rank, (cn, os_r2, loo_r2, fr_r2, beh_r2, vnll, npar, nw) in enumerate(ranked, 1):
        print(f"{rank:>4d} {cn:<25s} {os_r2:>8.4f} {loo_r2:>8.4f} "
              f"{fr_r2:>8.4f} {beh_r2:>8.4f} {vnll:>8.4f} {npar:>10,d} {nw:>3d}")

    # ── Per-worm breakdown for top 3 configs ──
    print(f"\n\n{'─'*80}")
    print(f"  PER-WORM BREAKDOWN (top 3 configs)")
    print(f"{'─'*80}")

    for cn, _, _, _, _, _, _, _ in ranked[:3]:
        print(f"\n  ── {cn} ──")
        worm_results = [r for r in all_results if r.get("config_name") == cn and "error" not in r]
        for wr in worm_results:
            _b = wr.get('beh_model_r2')
            beh_str = f"{_b:.4f}" if _b is not None else "n/a"
            print(f"    {wr['worm_id']:>15s}  1-step={wr['onestep_r2']:.4f}  "
                  f"LOO={wr['loo_r2']:.4f}  FreeRun={wr['free_run_r2']:.4f}  "
                  f"Beh={beh_str}")

    # ── Save aggregate ──
    agg = {cn: {
        "onestep_r2_mean": float(np.nanmean(m["onestep_r2"])),
        "loo_r2_mean": float(np.nanmean(m["loo_r2"])),
        "free_run_r2_mean": float(np.nanmean(m["free_run_r2"])),
        "beh_model_r2_mean": float(np.nanmean(m["beh_model_r2"])),
        "val_nll_mean": float(np.nanmean(m["val_nll"])),
        "n_params": int(np.mean(m["n_params"])) if m["n_params"] else 0,
        "n_worms": len(m["loo_r2"]),
    } for cn, m in config_metrics.items()}

    (out / "aggregate.json").write_text(json.dumps(agg, indent=2, default=str))
    (out / "ranked.json").write_text(json.dumps(
        [{"rank": i+1, "config": cn, "loo_r2": loo, "onestep_r2": os,
          "free_run_r2": fr, "beh_r2": beh, "n_params": np_}
         for i, (cn, os, loo, fr, beh, _, np_, _) in enumerate(ranked)],
        indent=2
    ))

    best = ranked[0]
    print(f"\n{'='*80}")
    print(f"  BEST CONFIG: {best[0]}  (LOO R²={best[2]:.4f})")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
