#!/usr/bin/env python3
"""Stage2 connectome model hyper-parameter sweep for free-run quality.

Tests multiple configurations (epochs, rollout loss, temperature, noise rank,
low-rank coupling) and reports distributional metrics so the best config can
be plugged into the full 11-model comparison.

Usage:
    python -u -m scripts.free_run.sweep_stage2 \
        --h5 data/used/behaviour+neuronal\ activity\ atanas\ \(2023\)/2/2022-06-14-07.h5 \
        --device cuda
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# ── Stage2 imports ──────────────────────────────────────────────────────
from stage2.config import make_config as _s2_make_config
from stage2.io_h5 import load_data_pt as _s2_load_data
from stage2.init_from_data import (
    init_lambda_u as _s2_init_lambda_u,
    init_all_from_data as _s2_init_all,
)
from stage2.model import Stage2ModelPT
from stage2.train import (
    compute_dynamics_loss as _s2_dynamics_loss,
    compute_rollout_loss as _s2_rollout_loss,
)

# ── Free-run utilities ─────────────────────────────────────────────────
from baseline_transformer.dataset import load_worm_data
from scripts.free_run.utils import (
    build_lagged,
    train_ridge,
    compute_distributional_metrics,
    ensemble_median_metrics,
)


# ═══════════════════════════════════════════════════════════════════════
# Sweep configuration
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SweepConfig:
    """One point in the hyper-parameter sweep."""
    name: str
    num_epochs: int = 50
    learning_rate: float = 0.001
    # Rollout loss
    rollout_weight: float = 0.0
    rollout_steps: int = 15
    rollout_starts: int = 4
    rollout_nll: bool = True        # NLL calibrates noise on multi-step residuals
    # Architecture
    noise_corr_rank: int = 0
    lowrank_rank: int = 0
    graph_poly_order: int = 2
    linear_chemical_synapses: bool = False
    coupling_gate: bool = True
    # Generation
    temperature: float = 1.0        # scale noise at inference time
    # Regularisation
    input_noise_sigma: float = 0.0  # noise injection during training


SWEEP_CONFIGS = [
    # ── Baselines ───────────────────────────────────────────────────────
    SweepConfig(
        name="A_baseline_20ep",
        num_epochs=20,
    ),
    SweepConfig(
        name="B_50ep",
        num_epochs=50,
    ),
    # ── Rollout loss (key for VarR calibration) ─────────────────────────
    SweepConfig(
        name="C_50ep_rollout",
        num_epochs=50,
        rollout_weight=0.3,
        rollout_steps=15,
        rollout_starts=4,
    ),
    SweepConfig(
        name="D_100ep_rollout",
        num_epochs=100,
        rollout_weight=0.3,
        rollout_steps=15,
        rollout_starts=6,
    ),
    # ── Temperature boost ───────────────────────────────────────────────
    SweepConfig(
        name="E_50ep_rollout_T1.5",
        num_epochs=50,
        rollout_weight=0.3,
        rollout_steps=15,
        rollout_starts=4,
        temperature=1.5,
    ),
    SweepConfig(
        name="F_50ep_rollout_T2.0",
        num_epochs=50,
        rollout_weight=0.3,
        rollout_steps=15,
        rollout_starts=4,
        temperature=2.0,
    ),
    # ── Temperature without rollout ─────────────────────────────────────
    SweepConfig(
        name="G_50ep_T2.0",
        num_epochs=50,
        temperature=2.0,
    ),
    # ── Correlated noise ────────────────────────────────────────────────
    SweepConfig(
        name="H_50ep_rollout_corrR3",
        num_epochs=50,
        rollout_weight=0.3,
        rollout_steps=15,
        rollout_starts=4,
        noise_corr_rank=3,
    ),
    # ── Low-rank coupling ───────────────────────────────────────────────
    SweepConfig(
        name="I_50ep_rollout_lr5",
        num_epochs=50,
        rollout_weight=0.3,
        rollout_steps=15,
        rollout_starts=4,
        lowrank_rank=5,
    ),
    # ── More rollout ────────────────────────────────────────────────────
    SweepConfig(
        name="J_50ep_heavyRollout",
        num_epochs=50,
        rollout_weight=1.0,
        rollout_steps=30,
        rollout_starts=8,
    ),
    # ── Best guess: rollout + corr noise + temperature ──────────────────
    SweepConfig(
        name="K_100ep_rollout_corrR3_T1.3",
        num_epochs=100,
        rollout_weight=0.3,
        rollout_steps=15,
        rollout_starts=6,
        noise_corr_rank=3,
        temperature=1.3,
    ),
]


# ═══════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════

def _train_stage2(
    h5_path: str,
    train_end: int,
    device: str,
    sc: SweepConfig,
) -> tuple:
    """Train Stage2 model with the given sweep config."""
    cfg = _s2_make_config(
        h5_path,
        device=device,
        num_epochs=sc.num_epochs,
        learning_rate=sc.learning_rate,
        learn_noise=True,
        noise_mode="heteroscedastic",
        coupling_gate=sc.coupling_gate,
        graph_poly_order=sc.graph_poly_order,
        behavior_weight=0.0,
        rollout_weight=sc.rollout_weight,
        rollout_steps=sc.rollout_steps,
        rollout_starts=sc.rollout_starts,
        loo_aux_weight=0.0,
        grad_clip_norm=1.0,
        noise_corr_rank=sc.noise_corr_rank,
        lowrank_rank=sc.lowrank_rank,
        linear_chemical_synapses=sc.linear_chemical_synapses,
        input_noise_sigma=sc.input_noise_sigma,
    )
    data = _s2_load_data(cfg)
    u_full = data["u_stage1"]
    T_full, N_all = u_full.shape
    dev = torch.device(device)

    u_train = u_full[:train_end]
    sigma_u = data["sigma_u"]
    g_train = data["gating"][:train_end] if data.get("gating") is not None else None
    s_train = data["stim"][:train_end] if data.get("stim") is not None else None

    lambda_u_init = _s2_init_lambda_u(u_full, cfg)
    model = Stage2ModelPT(
        N_all, data["T_e"], data["T_sv"], data["T_dcv"], data["dt"],
        cfg, dev, d_ell=data.get("d_ell", 0),
        lambda_u_init=lambda_u_init,
        sign_t=data.get("sign_t"),
    ).to(dev)
    _s2_init_all(model, u_full, cfg)

    params_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params_list, lr=sc.learning_rate)

    use_rollout = (sc.rollout_weight > 0 and sc.rollout_steps > 0)

    for epoch in range(sc.num_epochs):
        model.train()
        optimizer.zero_grad()

        # Add input noise during training if requested
        if sc.input_noise_sigma > 0:
            u_noisy = u_train + sc.input_noise_sigma * torch.randn_like(u_train)
        else:
            u_noisy = u_train

        _p = model.precompute_params()
        prior_mu = model.forward_sequence(
            u_noisy, gating_data=g_train, stim_data=s_train, params=_p)

        if model._noise_mode == "heteroscedastic":
            model_sigma = model.sigma_at(u_train[:-1])
        else:
            model_sigma = model.sigma_at()

        loss = _s2_dynamics_loss(
            u_train[1:], prior_mu[1:], sigma_u,
            model_sigma=model_sigma,
        )

        if use_rollout:
            r_loss = _s2_rollout_loss(
                model, u_train, sigma_u,
                rollout_steps=sc.rollout_steps,
                rollout_starts=sc.rollout_starts,
                gating_data=g_train,
                stim_data=s_train,
                use_nll=sc.rollout_nll,
            )
            loss = loss + sc.rollout_weight * r_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params_list, max_norm=1.0)
        optimizer.step()

        if epoch == 0 or (epoch + 1) % 10 == 0:
            rl_str = f"  roll={r_loss.item():.4f}" if use_rollout else ""
            print(f"      ep {epoch+1:3d}/{sc.num_epochs}: "
                  f"loss={loss.item():.4f}{rl_str}")

    model.eval()
    return model, data


# ═══════════════════════════════════════════════════════════════════════
# Generation
# ═══════════════════════════════════════════════════════════════════════

def _stage2_generate(
    model: Stage2ModelPT,
    data: dict,
    train_end: int,
    n_steps: int,
    n_samples: int,
    motor_idx: list[int],
    temperature: float = 1.0,
) -> list[np.ndarray]:
    """Stochastic autonomous free-run with optional temperature scaling."""
    u0 = data["u_stage1"]
    T_full, N_all = u0.shape
    device = next(model.parameters()).device
    gating = data.get("gating")
    stim = data.get("stim")
    lo, hi = model.u_clip
    ones = torch.ones(N_all, device=device)
    is_hetero = (model._noise_mode == "heteroscedastic")

    if not is_hetero:
        sigma_const = model.sigma_at().detach()

    motor_np = np.array(motor_idx, dtype=int)
    results: list[np.ndarray] = []

    with torch.no_grad():
        for k in range(n_samples):
            # Phase 1: teacher-forced burn-in
            s_sv = torch.zeros(N_all, model.r_sv, device=device)
            s_dcv = torch.zeros(N_all, model.r_dcv, device=device)
            for t in range(train_end):
                g = gating[t] if gating is not None else ones
                s = stim[t] if stim is not None else None
                _, s_sv, s_dcv = model.prior_step(u0[t], s_sv, s_dcv, g, s)

            # Phase 2: autonomous generation
            buf = np.zeros((n_steps, len(motor_idx)), dtype=np.float32)
            u_cur = u0[train_end].clone()
            buf[0] = u_cur.cpu().numpy()[motor_np]

            for t in range(n_steps - 1):
                t_global = train_end + t
                g = gating[t_global] if gating is not None and t_global < T_full else ones
                s = stim[t_global] if stim is not None and t_global < T_full else None
                mu_next, s_sv, s_dcv = model.prior_step(u_cur, s_sv, s_dcv, g, s)

                # Process noise with temperature scaling
                if is_hetero:
                    sigma_t = model.sigma_at(u_cur).detach() * temperature
                else:
                    sigma_t = sigma_const * temperature

                if getattr(model, 'noise_corr_rank', 0) > 0:
                    # Scale correlated noise by temperature
                    eps_diag = sigma_t * torch.randn(N_all, device=device)
                    R = model.noise_corr_rank
                    z = torch.randn(R, device=device)
                    eps_corr = (z @ model._noise_V.t()) * temperature
                    u_next = mu_next + eps_diag + eps_corr
                else:
                    u_next = mu_next + sigma_t * torch.randn(N_all, device=device)

                u_next = u_next.clamp(min=lo, max=hi)
                u_cur = u_next
                buf[t + 1] = u_cur.cpu().numpy()[motor_np]

            results.append(buf)
    return results


# ═══════════════════════════════════════════════════════════════════════
# Evaluation pipeline
# ═══════════════════════════════════════════════════════════════════════

def run_sweep(args):
    h5_path = args.h5
    device = args.device
    n_steps = args.n_steps
    n_samples = args.n_samples
    K = 3  # neural context lag

    # ── Load worm data (same as run_all_models_quick) ────────────────
    worm_data = load_worm_data(h5_path, n_beh_modes=6)
    u, b, worm_id = worm_data["u"], worm_data["b"], worm_data["worm_id"]
    motor_idx = worm_data.get("motor_idx")
    all_labels = worm_data.get("labels", [])

    if motor_idx is not None:
        u_sel = u[:, motor_idx]
    else:
        u_sel = u

    T, N, Kw = u_sel.shape[0], u_sel.shape[1], b.shape[1]
    train_end = T // 2

    print(f"Worm: {worm_id}  T={T}  N_motor={N}  Kw={Kw}")
    print(f"SPLIT: train=[0,{train_end})  test=[{train_end},{T})")
    print(f"N_STEPS={n_steps}  N_SAMPLES={n_samples}  K={K}")
    print()

    # ── Train-set normalisation ──────────────────────────────────────
    u_train = u_sel[:train_end]
    b_train = b[:train_end]
    mu_u = u_train[K:].mean(0).astype(np.float32)
    sig_u = u_train[K:].std(0).astype(np.float32)
    sig_u[sig_u < 1e-8] = 1.0
    mu_b = b_train.mean(0).astype(np.float32)
    sig_b = b_train.std(0).astype(np.float32)
    sig_b[sig_b < 1e-8] = 1.0

    u_norm = ((u_sel - mu_u) / sig_u).astype(np.float32)
    b_norm = ((b - mu_b) / sig_b).astype(np.float32)

    # ── Ridge behaviour decoder (shared across configs) ──────────────
    X_u_lag = build_lagged(u_norm[:train_end], K)
    y_beh = b_norm[K:train_end]
    tr_idx = slice(0, len(y_beh))
    ridge_beh_dec = train_ridge(X_u_lag[tr_idx], y_beh)

    # ── GT test data ─────────────────────────────────────────────────
    gt_beh_test = b[train_end:]

    # ── Run sweep ────────────────────────────────────────────────────
    results_table = []

    for i, sc in enumerate(SWEEP_CONFIGS):
        print(f"{'='*60}")
        print(f"  [{i+1}/{len(SWEEP_CONFIGS)}]  {sc.name}")
        print(f"    epochs={sc.num_epochs}  rollout_w={sc.rollout_weight}  "
              f"rollout_K={sc.rollout_steps}  T={sc.temperature}")
        print(f"    noise_corr_rank={sc.noise_corr_rank}  "
              f"lowrank_rank={sc.lowrank_rank}  "
              f"graph_poly={sc.graph_poly_order}")
        print(f"{'='*60}")

        t0 = time.time()
        try:
            # Train
            model, data = _train_stage2(h5_path, train_end, device, sc)
            train_time = time.time() - t0

            # Generate
            t1 = time.time()
            neural_samples = _stage2_generate(
                model, data, train_end, n_steps, n_samples,
                motor_idx, temperature=sc.temperature)
            gen_time = time.time() - t1

            # Decode behaviour
            u_norm_seed = u_norm[train_end - K: train_end].copy()
            beh_samples = []
            for ks in range(n_samples):
                pn_raw = neural_samples[ks]
                pn_norm = ((pn_raw - mu_u) / sig_u).astype(np.float32)
                pn_ctx = np.concatenate([u_norm_seed, pn_norm], axis=0)
                X_lag = build_lagged(pn_ctx, K)
                pb_norm = ridge_beh_dec.predict(X_lag[K:])
                pb_raw = (pb_norm * sig_b + mu_b).astype(np.float32)
                beh_samples.append(pb_raw)

            # Metrics
            met_list = [compute_distributional_metrics(
                gt_beh_test[:n_steps], s)[0] for s in beh_samples]
            mets = ensemble_median_metrics(met_list)

            total_time = time.time() - t0
            results_table.append({
                "name": sc.name,
                "PSD": mets["psd_log_distance"],
                "ACF": mets["autocorr_rmse"],
                "W1": mets["wasserstein_1"],
                "KS": mets["ks_statistic"],
                "VarR": mets["variance_ratio_mean"],
                "train_s": train_time,
                "total_s": total_time,
            })

            print(f"    → PSD={mets['psd_log_distance']:.3f}  ACF={mets['autocorr_rmse']:.3f}  "
                  f"W1={mets['wasserstein_1']:.3f}  KS={mets['ks_statistic']:.3f}  "
                  f"VarR={mets['variance_ratio_mean']:.3f}  "
                  f"({total_time:.1f}s)")

            # Free GPU memory
            del model, data
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"    ✗ FAILED: {e}")
            import traceback; traceback.print_exc()
            results_table.append({
                "name": sc.name,
                "PSD": float("nan"),
                "ACF": float("nan"),
                "W1": float("nan"),
                "KS": float("nan"),
                "VarR": float("nan"),
                "train_s": time.time() - t0,
                "total_s": time.time() - t0,
            })

    # ── Summary table ────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"SWEEP SUMMARY — {worm_id}  ({n_steps} steps, {n_samples} samples)")
    print(f"{'='*80}")
    print(f"{'Config':<35s}  {'PSD':>6s}  {'ACF':>6s}  {'W1':>6s}  "
          f"{'KS':>6s}  {'VarR':>6s}  {'Time':>6s}")
    print("-" * 80)

    # Composite score: lower is better.
    # PSD, ACF, W1, KS want to be low; VarR wants to be 1.0
    best_score = float("inf")
    best_name = ""

    for r in results_table:
        if np.isnan(r["PSD"]):
            score = float("inf")
        else:
            score = (r["PSD"] + r["ACF"] + 0.3 * r["W1"]
                     + 0.3 * r["KS"] + abs(r["VarR"] - 1.0))

        marker = ""
        if score < best_score:
            best_score = score
            best_name = r["name"]

        print(f"  {r['name']:<33s}  {r['PSD']:6.3f}  {r['ACF']:6.3f}  "
              f"{r['W1']:6.3f}  {r['KS']:6.3f}  {r['VarR']:6.3f}  "
              f"{r['total_s']:5.1f}s")

    # Mark the best
    print("-" * 80)
    print(f"  ★ BEST: {best_name}  (composite={best_score:.3f})")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Stage2 hyper-parameter sweep for free-run quality")
    parser.add_argument("--h5", required=True, help="Path to worm HDF5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_steps", type=int, default=50)
    parser.add_argument("--n_samples", type=int, default=3)
    args = parser.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
