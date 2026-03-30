#!/usr/bin/env python
"""Re-evaluate top transformer configs with train/test split diagnostics.

For each saved model, computes:
  1. One-step R² on TRAIN vs TEST regions
  2. LOO R² on TRAIN vs TEST regions
  3. Free-run R² starting from TEST context (not from t=0)
  4. Train-val NLL learning curves summary

This reveals whether the sweep rankings are inflated by train-region leakage.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline_transformer.config import TransformerBaselineConfig
from baseline_transformer.dataset import load_worm_data, temporal_train_val_test_split
from baseline_transformer.model import TemporalTransformerGaussian, build_model

# Reuse stage2 utilities
from stage2._utils import _r2


# ── Helpers ──────────────────────────────────────────────────────────────────

def _per_neuron_r2(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    N = gt.shape[1]
    return np.array([_r2(gt[:, i], pred[:, i]) for i in range(N)])


# ── 1. One-step R² on a specific region ─────────────────────────────────────

@torch.no_grad()
def onestep_r2_region(
    model: TemporalTransformerGaussian,
    u: np.ndarray,
    start: int,
    end: int,
) -> Dict[str, Any]:
    """Teacher-forced one-step R² on [start, end)."""
    device = next(model.parameters()).device
    T, N = u.shape
    K = model.cfg.context_length
    first = max(K, start + K)
    if first >= end:
        return {"r2": np.full(N, np.nan), "r2_mean": float("nan"), "n_steps": 0}

    u_t = torch.tensor(u, dtype=torch.float32, device=device)
    preds = []
    for t in range(first, end):
        ctx = u_t[t - K : t].unsqueeze(0)
        mu = model.predict_mean(ctx)
        preds.append(mu.squeeze(0).cpu().numpy())

    pred = np.stack(preds)
    gt = u[first:end]
    r2 = _per_neuron_r2(gt, pred)
    return {"r2": r2, "r2_mean": float(np.nanmean(r2)), "n_steps": len(preds)}


# ── 2. LOO R² on a specific region ──────────────────────────────────────────

@torch.no_grad()
def loo_r2_region(
    model: TemporalTransformerGaussian,
    u: np.ndarray,
    neuron_indices: List[int],
    eval_start: int,
    eval_end: int,
    sim_start: int = 0,
) -> Dict[str, Any]:
    """LOO forward-sim starting from sim_start, R² evaluated on [eval_start, eval_end).

    The simulation always runs from sim_start through eval_end to be faithful
    to the autoregressive nature — but R² is only scored on [eval_start, eval_end).
    """
    device = next(model.parameters()).device
    T, N = u.shape
    K = model.cfg.context_length

    r2_arr = np.full(N, np.nan)

    for cnt, i in enumerate(neuron_indices):
        # Run LOO sim from sim_start
        buf = u.copy().astype(np.float32)
        u_t = torch.tensor(buf, dtype=torch.float32, device=device)
        pred_neuron = np.zeros(T, dtype=np.float32)
        pred_neuron[:max(K, sim_start + K)] = buf[:max(K, sim_start + K), i]

        first_step = max(K, sim_start + K)
        for t in range(first_step, eval_end):
            ctx = u_t[t - K : t].unsqueeze(0)
            mu = model.predict_mean(ctx)
            pred_neuron[t] = mu[0, i].item()
            # Replace neuron i with model's prediction for next step
            u_t[t, i] = mu[0, i]

        # Score only on eval region
        s = max(eval_start, first_step)
        if s < eval_end:
            r2_arr[i] = _r2(u[s:eval_end, i], pred_neuron[s:eval_end])

        if cnt == 0 or (cnt + 1) % 5 == 0 or cnt == len(neuron_indices) - 1:
            print(f"    LOO {cnt+1}/{len(neuron_indices)}  neuron={i}  R²={r2_arr[i]:.4f}")

    valid = np.isfinite(r2_arr)
    return {
        "r2": r2_arr,
        "r2_mean": float(np.nanmean(r2_arr)) if valid.any() else float("nan"),
        "subset": neuron_indices,
        "eval_range": (eval_start, eval_end),
    }


# ── 3. Free-run R² on a specific region ─────────────────────────────────────

@torch.no_grad()
def freerun_r2_region(
    model: TemporalTransformerGaussian,
    u: np.ndarray,
    motor_idx: Optional[List[int]],
    run_start: int,
    eval_start: int,
    eval_end: int,
) -> Dict[str, Any]:
    """Free-run starting from u[run_start : run_start+K], score on [eval_start, eval_end).

    Motor-conditioned mode if motor_idx is provided.
    """
    device = next(model.parameters()).device
    T, N = u.shape
    K = model.cfg.context_length

    conditioned = motor_idx is not None and 0 < len(motor_idx) < N
    motor_mask = np.zeros(N, dtype=bool)
    if motor_idx:
        motor_mask[np.array(motor_idx)] = True

    # Seed with GT context starting at run_start
    seed_end = run_start + K
    assert seed_end <= T, f"run_start={run_start} + K={K} exceeds T={T}"

    buf = torch.tensor(u[run_start:seed_end], dtype=torch.float32, device=device)
    pred = np.zeros((eval_end - run_start, N), dtype=np.float32)
    pred[:K] = u[run_start:seed_end]

    for step in range(K, eval_end - run_start):
        t_abs = run_start + step
        ctx = buf[-K:].unsqueeze(0)
        mu = model.predict_mean(ctx).squeeze(0)
        mu_np = mu.cpu().numpy()

        if conditioned:
            out = u[t_abs].copy()
            out[motor_mask] = mu_np[motor_mask]
        else:
            out = mu_np

        pred[step] = out
        buf = torch.cat([buf, torch.tensor(out, dtype=torch.float32, device=device).unsqueeze(0)])

    # Score on [eval_start, eval_end)
    es = eval_start - run_start
    ee = eval_end - run_start
    u_region = u[eval_start:eval_end]
    p_region = pred[es:ee]

    eval_idx = list(np.where(motor_mask)[0]) if conditioned else list(range(N))
    r2_arr = np.full(N, np.nan)
    for i in eval_idx:
        r2_arr[i] = _r2(u_region[:, i], p_region[:, i])

    return {
        "r2": r2_arr,
        "r2_mean": float(np.nanmean(r2_arr[eval_idx])) if eval_idx else float("nan"),
        "mode": "motor_conditioned" if conditioned else "autonomous",
        "run_start": run_start,
        "eval_range": (eval_start, eval_end),
    }


# ── Load saved model ────────────────────────────────────────────────────────

def load_saved_model(
    sweep_dir: str,
    config_name: str,
    worm_id: str,
    cfg: TransformerBaselineConfig,
    n_obs: int,
    device: str,
) -> TemporalTransformerGaussian:
    """Load a model saved during the sweep."""
    model_path = Path(sweep_dir) / config_name / worm_id / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = build_model(n_obs, cfg, device=device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


# ── Config reconstruction ───────────────────────────────────────────────────

def get_config(name: str) -> TransformerBaselineConfig:
    """Reconstruct config from name (matching transformer_arch_sweep.py)."""
    if name == "A_baseline":
        return TransformerBaselineConfig()
    elif name == "B_wide_256h8":
        return TransformerBaselineConfig(d_model=256, n_heads=8, d_ff=512)
    elif name == "E_deep_wide":
        return TransformerBaselineConfig(d_model=256, n_heads=8, d_ff=512, n_layers=3)
    elif name == "I_ctx8":
        return TransformerBaselineConfig(context_length=8)
    else:
        raise ValueError(f"Unknown config: {name}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sweep_dir", default="output_plots/transformer_baseline/arch_sweep")
    parser.add_argument("--loo_n", type=int, default=10, help="Number of LOO neurons")
    args = parser.parse_args()

    device = args.device
    sweep_dir = args.sweep_dir

    TOP_CONFIGS = ["B_wide_256h8", "A_baseline", "E_deep_wide", "I_ctx8"]
    WORMS = ["2023-01-17-14", "2022-06-14-07", "2023-01-10-07"]
    H5_DIR = Path("data/used/behaviour+neuronal activity atanas (2023)/2")

    all_results = []

    print("=" * 90)
    print("  TRANSFORMER OVERFITTING CHECK")
    print(f"  {len(TOP_CONFIGS)} configs × {len(WORMS)} worms")
    print(f"  Metrics: one-step, LOO ({args.loo_n} neurons), free-run")
    print(f"  Comparing TRAIN region vs TEST region vs FULL recording")
    print("=" * 90)

    for ci, config_name in enumerate(TOP_CONFIGS):
        cfg = get_config(config_name)
        K = cfg.context_length

        for wi, worm_id in enumerate(WORMS):
            run_label = f"[{ci*len(WORMS)+wi+1}/{len(TOP_CONFIGS)*len(WORMS)}]"
            print(f"\n{'─'*90}")
            print(f"  {run_label} {config_name} / {worm_id}")
            print(f"{'─'*90}")

            h5_path = str(H5_DIR / f"{worm_id}.h5")
            worm_data = load_worm_data(h5_path)
            u = worm_data["u"]
            T, N = u.shape
            motor_idx = worm_data.get("motor_idx")

            # Reconstruct split
            split = temporal_train_val_test_split(T, cfg.train_frac, cfg.val_frac)
            tr_s, tr_e = split["train"]
            va_s, va_e = split["val"]
            te_s, te_e = split["test"]
            print(f"  T={T}, N={N}, K={K}")
            print(f"  Train=[{tr_s},{tr_e}), Val=[{va_s},{va_e}), Test=[{te_s},{te_e})")

            # Load model
            t0 = time.time()
            model = load_saved_model(sweep_dir, config_name, worm_id, cfg, N, device)
            print(f"  Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

            # Select LOO neurons (top variance)
            variances = np.nanvar(u, axis=0)
            loo_neurons = list(np.argsort(variances)[::-1][:min(args.loo_n, N)])

            # ── 1. One-step R² ──
            print(f"\n  [One-step R²]")
            os_train = onestep_r2_region(model, u, tr_s, tr_e)
            os_val   = onestep_r2_region(model, u, va_s, va_e)
            os_test  = onestep_r2_region(model, u, te_s, te_e)
            os_full  = onestep_r2_region(model, u, 0, T)
            print(f"    TRAIN: {os_train['r2_mean']:.4f}  ({os_train['n_steps']} steps)")
            print(f"    VAL:   {os_val['r2_mean']:.4f}  ({os_val['n_steps']} steps)")
            print(f"    TEST:  {os_test['r2_mean']:.4f}  ({os_test['n_steps']} steps)")
            print(f"    FULL:  {os_full['r2_mean']:.4f}  ({os_full['n_steps']} steps)")

            # ── 2. LOO R² ──
            print(f"\n  [LOO R² — {len(loo_neurons)} neurons]")

            # LOO on TRAIN region: sim from 0, score on [K, tr_e)
            print(f"  → TRAIN region [{K}, {tr_e}):")
            loo_train = loo_r2_region(model, u, loo_neurons, K, tr_e, sim_start=0)
            print(f"    LOO TRAIN: {loo_train['r2_mean']:.4f}")

            # LOO on TEST region: sim from 0, score on [te_s, te_e)
            print(f"  → TEST region [{te_s}, {te_e}):")
            loo_test = loo_r2_region(model, u, loo_neurons, te_s, te_e, sim_start=0)
            print(f"    LOO TEST:  {loo_test['r2_mean']:.4f}")

            # LOO on FULL: sim from 0, score on [K, T) — matches original sweep
            print(f"  → FULL [{K}, {T}):")
            loo_full = loo_r2_region(model, u, loo_neurons, K, T, sim_start=0)
            print(f"    LOO FULL:  {loo_full['r2_mean']:.4f}")

            # ── 3. Free-run R² ──
            print(f"\n  [Free-run R²]")

            # Free-run from t=0 (original), score on TRAIN
            fr_train = freerun_r2_region(model, u, motor_idx, 0, K, tr_e)
            print(f"    TRAIN (start=0):     {fr_train['r2_mean']:.4f}  [{fr_train['mode']}]")

            # Free-run from t=0, score on TEST
            fr_test_from0 = freerun_r2_region(model, u, motor_idx, 0, te_s, te_e)
            print(f"    TEST  (start=0):     {fr_test_from0['r2_mean']:.4f}  [{fr_test_from0['mode']}]")

            # Free-run from test context (te_s - K), score on TEST
            test_ctx_start = max(0, te_s - K)
            fr_test_fresh = freerun_r2_region(model, u, motor_idx,
                                              test_ctx_start, te_s, te_e)
            print(f"    TEST  (start={test_ctx_start}): {fr_test_fresh['r2_mean']:.4f}  [{fr_test_fresh['mode']}]")

            # Free-run from t=0, score on FULL
            fr_full = freerun_r2_region(model, u, motor_idx, 0, K, T)
            print(f"    FULL  (start=0):     {fr_full['r2_mean']:.4f}  [{fr_full['mode']}]")

            elapsed = time.time() - t0

            # ── Training history gap ──
            hist_path = Path(sweep_dir) / config_name / worm_id / "history.json"
            train_nll_best, val_nll_best, gap_best = np.nan, np.nan, np.nan
            best_ep = 0
            if hist_path.exists():
                hist = json.loads(hist_path.read_text())
                best_idx = min(range(len(hist)), key=lambda i: hist[i]["val_nll"])
                best = hist[best_idx]
                train_nll_best = best["train_nll"]
                val_nll_best = best["val_nll"]
                gap_best = val_nll_best - train_nll_best
                best_ep = best["epoch"]

            result = {
                "config": config_name,
                "worm": worm_id,
                "T": T, "N": N, "K": K,
                "split": {k: list(v) for k, v in split.items()},
                "onestep_train": os_train["r2_mean"],
                "onestep_val": os_val["r2_mean"],
                "onestep_test": os_test["r2_mean"],
                "onestep_full": os_full["r2_mean"],
                "loo_train": loo_train["r2_mean"],
                "loo_test": loo_test["r2_mean"],
                "loo_full": loo_full["r2_mean"],
                "freerun_train": fr_train["r2_mean"],
                "freerun_test_from0": fr_test_from0["r2_mean"],
                "freerun_test_fresh": fr_test_fresh["r2_mean"],
                "freerun_full": fr_full["r2_mean"],
                "train_nll_best": train_nll_best,
                "val_nll_best": val_nll_best,
                "nll_gap": gap_best,
                "best_epoch": best_ep,
                "elapsed_s": round(elapsed, 1),
            }
            all_results.append(result)

            print(f"\n  Summary ({elapsed:.0f}s):")
            print(f"    NLL gap @best_ep={best_ep}: {gap_best:.4f}")
            print(f"    One-step  TRAIN={os_train['r2_mean']:.4f}  TEST={os_test['r2_mean']:.4f}  Δ={os_train['r2_mean']-os_test['r2_mean']:.4f}")
            print(f"    LOO       TRAIN={loo_train['r2_mean']:.4f}  TEST={loo_test['r2_mean']:.4f}  Δ={loo_train['r2_mean']-loo_test['r2_mean']:.4f}")
            print(f"    Free-run  TRAIN={fr_train['r2_mean']:.4f}  TEST={fr_test_fresh['r2_mean']:.4f}  Δ={fr_train['r2_mean']-fr_test_fresh['r2_mean']:.4f}")

    # ── Aggregate summary ──
    print(f"\n\n{'='*110}")
    print(f"  OVERFITTING SUMMARY TABLE")
    print(f"{'='*110}")

    header = (f"{'Config':<20s} {'Worm':>15s}  "
              f"{'1s_TR':>6s} {'1s_TE':>6s} {'Δ1s':>6s}  "
              f"{'LOO_TR':>7s} {'LOO_TE':>7s} {'ΔLOO':>7s}  "
              f"{'FR_TR':>6s} {'FR_TE':>6s} {'ΔFR':>6s}  "
              f"{'NLLgap':>7s}")
    print(header)
    print("─" * 110)

    for r in all_results:
        d_os = r["onestep_train"] - r["onestep_test"]
        d_loo = r["loo_train"] - r["loo_test"]
        d_fr = r["freerun_train"] - r["freerun_test_fresh"]
        print(f"{r['config']:<20s} {r['worm']:>15s}  "
              f"{r['onestep_train']:>6.3f} {r['onestep_test']:>6.3f} {d_os:>+6.3f}  "
              f"{r['loo_train']:>7.4f} {r['loo_test']:>7.4f} {d_loo:>+7.4f}  "
              f"{r['freerun_train']:>6.3f} {r['freerun_test_fresh']:>6.3f} {d_fr:>+6.3f}  "
              f"{r['nll_gap']:>7.4f}")

    # ── Per-config averages ──
    print(f"\n{'─'*110}")
    print(f"  PER-CONFIG AVERAGES (across {len(WORMS)} worms)")
    print(f"{'─'*110}")

    header2 = (f"{'Config':<20s}  "
               f"{'1s_TRAIN':>9s} {'1s_TEST':>8s} {'Δ':>7s}  "
               f"{'LOO_TRAIN':>10s} {'LOO_TEST':>9s} {'Δ':>8s}  "
               f"{'FR_TRAIN':>9s} {'FR_TEST':>8s} {'Δ':>7s}  "
               f"{'LOO_FULL':>9s} {'FR_FULL':>8s}")
    print(header2)
    print("─" * 110)

    for cfg_name in TOP_CONFIGS:
        cfg_results = [r for r in all_results if r["config"] == cfg_name]
        n = len(cfg_results)
        if n == 0:
            continue

        avg = lambda key: np.mean([r[key] for r in cfg_results])
        d_os = avg("onestep_train") - avg("onestep_test")
        d_loo = avg("loo_train") - avg("loo_test")
        d_fr = avg("freerun_train") - avg("freerun_test_fresh")
        print(f"{cfg_name:<20s}  "
              f"{avg('onestep_train'):>9.4f} {avg('onestep_test'):>8.4f} {d_os:>+7.4f}  "
              f"{avg('loo_train'):>10.4f} {avg('loo_test'):>9.4f} {d_loo:>+8.4f}  "
              f"{avg('freerun_train'):>9.4f} {avg('freerun_test_fresh'):>8.4f} {d_fr:>+7.4f}  "
              f"{avg('loo_full'):>9.4f} {avg('freerun_full'):>8.4f}")

    # ── Revised ranking by TEST-only LOO ──
    print(f"\n{'─'*110}")
    print(f"  REVISED RANKING (by TEST-only LOO R²)")
    print(f"{'─'*110}")

    cfg_test_loo = {}
    for cfg_name in TOP_CONFIGS:
        cfg_results = [r for r in all_results if r["config"] == cfg_name]
        cfg_test_loo[cfg_name] = np.mean([r["loo_test"] for r in cfg_results])

    for rank, (cfg_name, test_loo) in enumerate(
        sorted(cfg_test_loo.items(), key=lambda x: -x[1]), 1
    ):
        cfg_results = [r for r in all_results if r["config"] == cfg_name]
        full_loo = np.mean([r["loo_full"] for r in cfg_results])
        inflation = full_loo - test_loo
        print(f"  #{rank}  {cfg_name:<20s}  LOO_TEST={test_loo:.4f}  "
              f"LOO_FULL={full_loo:.4f}  inflation={inflation:+.4f}")

    # ── Save ──
    out_dir = Path(sweep_dir) / "overfit_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "overfit_results.json").write_text(
        json.dumps(all_results, indent=2, default=str)
    )
    print(f"\n  Results saved to {out_dir / 'overfit_results.json'}")


if __name__ == "__main__":
    main()
