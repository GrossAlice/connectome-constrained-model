#!/usr/bin/env python3
"""
Evaluate Atlas Transformer with v4-style one-step teacher-forced R².

Computes per-neuron one-step R² under three masking conditions that
match the v4 neural_activity_decoder conditions:

  causal_self : full context (all neurons' K lags → predict neuron i)
                ≡ atlas TRF standard one-step R²
  causal      : mask neuron i in context (other neurons' lags only)
  self        : mask all OTHER neurons, keep only neuron i's lags

These are one-step TEACHER-FORCED predictions (no autoregressive),
making them directly comparable to v4 Ridge/MLP/TRF results.

Usage:
    python -u -m scripts.eval_atlas_onestep_conditions \
        --worm 2022-06-14-01 --device cuda
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
from atlas_transformer.model import build_atlas_model
from atlas_transformer.dataset import load_atlas_worm_data
from stage2._utils import _r2
from stage2.io_multi import _load_full_atlas


# ── Defaults ─────────────────────────────────────────────────────────────────

H5_DIR = "data/used/behaviour+neuronal activity atanas (2023)/2"
ATLAS_MODEL_DIR = "output_plots/atlas_single_worm_vs_v4"
V4_DIR = "output_plots/neural_activity_decoder_v4"


# ── Load saved atlas model ──────────────────────────────────────────────────

def load_atlas_model(worm_id: str, device: str = "cpu") -> tuple:
    """Load trained atlas transformer (best fold) and worm data."""
    model_dir = ROOT / ATLAS_MODEL_DIR / worm_id / worm_id
    meta_path = model_dir / "train_cv_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No meta at {meta_path}")

    meta = json.loads(meta_path.read_text())
    best_fold = meta["best_fold_idx"]

    # Reconstruct config
    cfg = AtlasTransformerConfig(device=device)
    n_beh = meta["n_beh"]

    # Build model and load weights
    model = build_atlas_model(cfg, device=device, n_beh=n_beh)
    state = torch.load(
        model_dir / f"atlas_model_fold{best_fold}.pt",
        map_location=device, weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()

    return model, meta


def load_worm_data(worm_id: str, cfg: AtlasTransformerConfig):
    """Load worm data into atlas format."""
    h5_path = str(ROOT / H5_DIR / f"{worm_id}.h5")
    full_labels = _load_full_atlas()
    n_atlas = len(full_labels)
    atlas_idx = np.arange(n_atlas, dtype=np.int64)

    worm = load_atlas_worm_data(
        h5_path, full_labels, atlas_idx, n_atlas,
        n_beh_modes=cfg.n_beh_modes,
    )
    return worm


# ── One-step teacher-forced evaluation with masking ─────────────────────────

@torch.no_grad()
def compute_onestep_r2_conditions(
    model,
    x: np.ndarray,
    obs_mask: np.ndarray,
    folds: list[dict],
    conditions: list[str] = ("causal_self", "causal", "self"),
    verbose: bool = True,
) -> dict:
    """One-step teacher-forced R² per neuron under different masking conditions.

    For each time step t in the test fold, we:
      1. Take the ground-truth context window x[t-K : t]  (teacher-forced)
      2. Apply the condition mask to the neural columns
      3. Predict one step ahead
      4. Record prediction for each observed neuron

    Conditions:
      causal_self : full context (all neurons in [t-K, t))
      causal      : zero out neuron i in context → predict i from others
      self        : zero out all OTHER neurons → predict i from own history
    """
    device = next(model.parameters()).device
    n_atlas = model.n_atlas
    K = model.cfg.context_length
    T, D = x.shape
    obs_idx = np.where(obs_mask)[0]
    N_obs = len(obs_idx)

    x_t = torch.tensor(x, dtype=torch.float32, device=device)

    # Build test masks from CV folds — use held-out time ranges
    test_mask = np.zeros(T, dtype=bool)
    for fold in folds:
        ts, te = fold["test"]
        # Only include timesteps where we have K context
        for t in range(max(ts, K), te):
            test_mask[t] = True
    test_times = np.where(test_mask)[0]

    if verbose:
        print(f"  Evaluating {len(test_times)} test time steps, "
              f"{N_obs} observed neurons, {len(conditions)} conditions")

    results = {}

    for cond in conditions:
        t0 = time.time()

        if cond == "causal_self":
            # Full context — predict all neurons at once (fast batch)
            preds = np.full((T, n_atlas), np.nan, dtype=np.float32)
            for t in test_times:
                ctx = x_t[t - K : t].unsqueeze(0)  # (1, K, D)
                mu_u, _ = model.predict_mean_split(ctx)
                preds[t] = mu_u.squeeze(0).cpu().numpy()

            # Per-neuron R² on test times
            r2_arr = np.full(N_obs, np.nan)
            for i, idx in enumerate(obs_idx):
                gt = x[test_times, idx]
                pr = preds[test_times, idx]
                r2_arr[i] = _r2(gt, pr)

        elif cond == "causal":
            # For each neuron i: mask neuron i in context, predict i
            r2_arr = np.full(N_obs, np.nan)
            for i, idx in enumerate(obs_idx):
                preds_i = []
                gts_i = []
                for t in test_times:
                    ctx = x_t[t - K : t].clone().unsqueeze(0)  # (1, K, D)
                    # Zero out neuron i in neural columns AND obs_mask column
                    ctx[0, :, idx] = 0.0              # neural activity
                    ctx[0, :, n_atlas + idx] = 0.0    # obs_mask column
                    mu_u, _ = model.predict_mean_split(ctx)
                    preds_i.append(mu_u[0, idx].item())
                    gts_i.append(x[t, idx])

                preds_i = np.array(preds_i)
                gts_i = np.array(gts_i)
                r2_arr[i] = _r2(gts_i, preds_i)

                if verbose and (i == 0 or (i + 1) % 20 == 0 or i == N_obs - 1):
                    print(f"    {cond} neuron {i+1}/{N_obs}  "
                          f"atlas_idx={idx}  R²={r2_arr[i]:.4f}")

        elif cond == "self":
            # For each neuron i: keep ONLY neuron i, mask all others
            r2_arr = np.full(N_obs, np.nan)
            for i, idx in enumerate(obs_idx):
                preds_i = []
                gts_i = []
                for t in test_times:
                    ctx = x_t[t - K : t].clone().unsqueeze(0)  # (1, K, D)
                    # Zero out ALL other neurons in neural + obs_mask columns
                    for j in range(n_atlas):
                        if j != idx:
                            ctx[0, :, j] = 0.0              # neural
                            ctx[0, :, n_atlas + j] = 0.0    # obs_mask
                    mu_u, _ = model.predict_mean_split(ctx)
                    preds_i.append(mu_u[0, idx].item())
                    gts_i.append(x[t, idx])

                preds_i = np.array(preds_i)
                gts_i = np.array(gts_i)
                r2_arr[i] = _r2(gts_i, preds_i)

                if verbose and (i == 0 or (i + 1) % 20 == 0 or i == N_obs - 1):
                    print(f"    {cond} neuron {i+1}/{N_obs}  "
                          f"atlas_idx={idx}  R²={r2_arr[i]:.4f}")

        else:
            raise ValueError(f"Unknown condition: {cond}")

        r2_mean = float(np.nanmean(r2_arr))
        elapsed = time.time() - t0

        results[cond] = {
            "r2_mean": r2_mean,
            "r2_per_neuron": r2_arr.tolist(),
            "n_test_times": len(test_times),
        }

        if verbose:
            print(f"  {cond:>15s}  R² = {r2_mean:.4f}  ({elapsed:.1f}s)")

    return results


# ── Load v4 results ─────────────────────────────────────────────────────────

def load_v4_summary(worm_id: str, K: int = 5) -> dict:
    """Load v4 mean R² per model×condition."""
    path = ROOT / V4_DIR / f"{worm_id}_all" / "results.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    res_k = data.get(str(K), {})
    out = {}
    for model in ("ridge", "mlp", "trf"):
        for cond in ("self", "causal", "conc", "causal_self", "conc_self", "conc_causal"):
            key = f"r2_mean_{model}_{cond}"
            if key in res_k:
                out[f"{model}_{cond}"] = res_k[key]
    return out


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Atlas TRF one-step teacher-forced R² (v4-matched evaluation)"
    )
    parser.add_argument("--worm", nargs="+", default=["2022-06-14-01"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=ATLAS_MODEL_DIR)
    parser.add_argument("--conditions", nargs="+",
                        default=["causal_self", "causal", "self"])
    args = parser.parse_args()

    out_dir = Path(args.out)

    for worm_id in args.worm:
        print(f"\n{'='*70}")
        print(f"  Atlas TRF — v4-matched one-step eval: {worm_id}")
        print(f"{'='*70}")

        # Load model + data
        model, meta = load_atlas_model(worm_id, device=args.device)
        cfg = model.cfg
        print(f"  Model: d={cfg.d_model}, h={cfg.n_heads}, L={cfg.n_layers}, "
              f"K={cfg.context_length}")
        print(f"  Best fold: {meta['best_fold_idx']}")

        worm = load_worm_data(worm_id, cfg)
        obs_mask = worm["obs_mask"]
        N_obs = int(obs_mask.sum())
        print(f"  N_obs={N_obs}, T={worm['T']}")

        # Build packed joint state
        from atlas_transformer.dataset import build_joint_state_atlas
        include_beh = cfg.predict_beh and cfg.include_beh_input
        x, _, _ = build_joint_state_atlas(
            worm["u_atlas"], obs_mask,
            b=worm.get("b"),
            b_mask=worm.get("b_mask"),
            include_beh=include_beh,
        )

        # Get folds from meta
        folds = meta["folds"]

        # Run evaluation
        results = compute_onestep_r2_conditions(
            model, x, obs_mask, folds,
            conditions=args.conditions,
            verbose=True,
        )

        # Load v4 for comparison
        v4 = load_v4_summary(worm_id)

        # Print comparison table
        print(f"\n{'='*70}")
        print(f"  COMPARISON TABLE — {worm_id}")
        print(f"{'='*70}")
        print(f"\n  {'Condition':<15s} {'Atlas TRF':>10s} {'v4-Ridge':>10s} "
              f"{'v4-MLP':>10s} {'v4-TRF':>10s}")
        print(f"  {'-'*55}")

        for cond in args.conditions:
            atlas_r2 = results[cond]["r2_mean"]
            v4_ridge = v4.get(f"ridge_{cond}", float("nan"))
            v4_mlp = v4.get(f"mlp_{cond}", float("nan"))
            v4_trf = v4.get(f"trf_{cond}", float("nan"))
            print(f"  {cond:<15s} {atlas_r2:>10.4f} {v4_ridge:>10.4f} "
                  f"{v4_mlp:>10.4f} {v4_trf:>10.4f}")

        # Also show concurrent conditions from v4 (atlas can't do these)
        for cond in ("conc", "conc_self", "conc_causal"):
            v4_ridge = v4.get(f"ridge_{cond}", float("nan"))
            v4_mlp = v4.get(f"mlp_{cond}", float("nan"))
            v4_trf = v4.get(f"trf_{cond}", float("nan"))
            print(f"  {cond:<15s} {'n/a':>10s} {v4_ridge:>10.4f} "
                  f"{v4_mlp:>10.4f} {v4_trf:>10.4f}")

        print(f"\n  Note: Atlas TRF uses K={cfg.context_length} (v4 uses K=5)")
        print(f"  Note: 'conc' conditions use concurrent u(t) features "
              f"— not available in the atlas TRF architecture")

        # Save results
        save_path = out_dir / worm_id / "onestep_conditions.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_data = {
            "worm_id": worm_id,
            "K": cfg.context_length,
            "N_obs": N_obs,
            "best_fold": meta["best_fold_idx"],
            "atlas_trf": results,
            "v4_comparison": v4,
        }
        save_path.write_text(json.dumps(save_data, indent=2))
        print(f"\n  Saved to {save_path}")


if __name__ == "__main__":
    main()
