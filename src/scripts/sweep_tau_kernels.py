#!/usr/bin/env python
"""
Sweep over synaptic-kernel / tau configurations to diagnose whether
temporal information is properly transmitted through the Stage2 model.

Background
──────────
With dt ≈ 0.6 s (Atanas calcium imaging) and the current defaults:
  • tau_sv  = (0.5, 2.5) → γ = (0.30, 0.79)
  • tau_dcv = (3.0, 5.0) → γ = (0.82, 0.89)
  • fix_tau_sv = True, fix_tau_dcv = True   ← FIXED, not learned

The fast SV kernel (τ=0.5 s) has γ=0.30, meaning it loses 70 % of its
state *every single timestep*.  Its half-life is 0.35 s — shorter than
the sampling interval!  It carries essentially no temporal memory.

This sweep tests whether:
  A) Learning taus (instead of fixing them) helps
  B) Longer tau init values help (shift range upward)
  C) More ranks (3-4 time constants) help
  D) Per-neuron tau scaling helps (different neurons → different timescales)
  E) Combinations of the above

Evaluation:  standard LOO windowed R² (2-fold CV, 30 neurons) — fast
and well-correlated with retrain-LOO.

Usage
─────
    python -u -m scripts.sweep_tau_kernels \
        --h5 "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5" \
        --out output_plots/stage2/tau_kernel_sweep \
        --epochs 30 --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ═══════════════════════════════════════════════════════════════════════
#  Sweep configurations
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TauSweepConfig:
    """A single tau/kernel configuration to test."""
    name: str
    description: str
    # Tau init values (determines number of ranks by length)
    tau_sv_init: Tuple[float, ...] = (0.5, 2.5)
    tau_dcv_init: Tuple[float, ...] = (3.0, 5.0)
    # Amplitude init (matched to tau ranks by length)
    a_sv_init: Tuple[float, ...] = (2.0, 0.8)
    a_dcv_init: Tuple[float, ...] = (0.2, 0.08)
    # Whether to fix or learn taus
    fix_tau_sv: bool = True
    fix_tau_dcv: bool = True
    # Per-neuron tau scaling
    per_neuron_tau_scale: bool = False
    # Other dynamics settings (keep defaults unless noted)
    per_neuron_amplitudes: bool = True
    coupling_gate: bool = True
    graph_poly_order: int = 2
    # Epochs
    num_epochs: int = 30


def build_sweep_configs(epochs: int = 30) -> List[TauSweepConfig]:
    """Build all sweep configurations."""
    cfgs = []

    # ── A: Baseline (current defaults) ──
    cfgs.append(TauSweepConfig(
        name="A_baseline",
        description="Current defaults: fixed τ_sv=(0.5,2.5), τ_dcv=(3,5)",
        num_epochs=epochs,
    ))

    # ── B: Learn taus (same init, but unfix them) ──
    cfgs.append(TauSweepConfig(
        name="B_learn_taus",
        description="Same init but taus are LEARNED (fix=False)",
        fix_tau_sv=False,
        fix_tau_dcv=False,
        num_epochs=epochs,
    ))

    # ── C: Longer fixed taus (shift range up since dt=0.6s) ──
    cfgs.append(TauSweepConfig(
        name="C_longer_fixed",
        description="Longer fixed τ_sv=(2,8), τ_dcv=(5,15) — more memory",
        tau_sv_init=(2.0, 8.0),
        tau_dcv_init=(5.0, 15.0),
        a_sv_init=(1.5, 0.5),
        a_dcv_init=(0.15, 0.05),
        num_epochs=epochs,
    ))

    # ── D: Much longer taus (integrator regime) ──
    cfgs.append(TauSweepConfig(
        name="D_very_long",
        description="Very long τ_sv=(5,20), τ_dcv=(10,40) — integrator regime",
        tau_sv_init=(5.0, 20.0),
        tau_dcv_init=(10.0, 40.0),
        a_sv_init=(1.0, 0.3),
        a_dcv_init=(0.1, 0.03),
        num_epochs=epochs,
    ))

    # ── E: Learn + longer init (best of B+C) ──
    cfgs.append(TauSweepConfig(
        name="E_learn_longer",
        description="Learn taus starting from longer init (2,8)/(5,15)",
        tau_sv_init=(2.0, 8.0),
        tau_dcv_init=(5.0, 15.0),
        a_sv_init=(1.5, 0.5),
        a_dcv_init=(0.15, 0.05),
        fix_tau_sv=False,
        fix_tau_dcv=False,
        num_epochs=epochs,
    ))

    # ── F: 3 ranks (wider temporal coverage) ──
    cfgs.append(TauSweepConfig(
        name="F_3ranks",
        description="3 ranks: τ_sv=(1,5,15), τ_dcv=(2,10,30)",
        tau_sv_init=(1.0, 5.0, 15.0),
        tau_dcv_init=(2.0, 10.0, 30.0),
        a_sv_init=(2.0, 0.8, 0.3),
        a_dcv_init=(0.2, 0.08, 0.03),
        num_epochs=epochs,
    ))

    # ── G: 4 ranks (full temporal spectrum) ──
    cfgs.append(TauSweepConfig(
        name="G_4ranks",
        description="4 ranks: τ_sv=(0.8,3,10,30), τ_dcv=(1,5,15,50)",
        tau_sv_init=(0.8, 3.0, 10.0, 30.0),
        tau_dcv_init=(1.0, 5.0, 15.0, 50.0),
        a_sv_init=(2.0, 1.0, 0.4, 0.15),
        a_dcv_init=(0.2, 0.1, 0.04, 0.015),
        num_epochs=epochs,
    ))

    # ── H: 3 ranks + learn taus ──
    cfgs.append(TauSweepConfig(
        name="H_3ranks_learn",
        description="3 ranks + learn taus: maximum temporal adaptability",
        tau_sv_init=(1.0, 5.0, 15.0),
        tau_dcv_init=(2.0, 10.0, 30.0),
        a_sv_init=(2.0, 0.8, 0.3),
        a_dcv_init=(0.2, 0.08, 0.03),
        fix_tau_sv=False,
        fix_tau_dcv=False,
        num_epochs=epochs,
    ))

    # ── I: Per-neuron tau scale (current tau init) ──
    cfgs.append(TauSweepConfig(
        name="I_perneuron_tau",
        description="Per-neuron tau scale exp(s_i) with current defaults",
        per_neuron_tau_scale=True,
        num_epochs=epochs,
    ))

    # ── J: Per-neuron tau + learn taus ──
    cfgs.append(TauSweepConfig(
        name="J_perneuron_learn",
        description="Per-neuron tau scale + learn taus + longer init",
        tau_sv_init=(2.0, 8.0),
        tau_dcv_init=(5.0, 15.0),
        a_sv_init=(1.5, 0.5),
        a_dcv_init=(0.15, 0.05),
        fix_tau_sv=False,
        fix_tau_dcv=False,
        per_neuron_tau_scale=True,
        num_epochs=epochs,
    ))

    # ── K: 3 ranks + per-neuron tau + learn ──
    cfgs.append(TauSweepConfig(
        name="K_3ranks_perneuron_learn",
        description="3 ranks + per-neuron tau + learn: full flexibility",
        tau_sv_init=(1.0, 5.0, 15.0),
        tau_dcv_init=(2.0, 10.0, 30.0),
        a_sv_init=(2.0, 0.8, 0.3),
        a_dcv_init=(0.2, 0.08, 0.03),
        fix_tau_sv=False,
        fix_tau_dcv=False,
        per_neuron_tau_scale=True,
        num_epochs=epochs,
    ))

    # ── L: No coupling gate (test if gate is masking kernel effects) ──
    cfgs.append(TauSweepConfig(
        name="L_no_gate_learn",
        description="Learn taus + longer init, NO coupling gate",
        tau_sv_init=(2.0, 8.0),
        tau_dcv_init=(5.0, 15.0),
        a_sv_init=(1.5, 0.5),
        a_dcv_init=(0.15, 0.05),
        fix_tau_sv=False,
        fix_tau_dcv=False,
        coupling_gate=False,
        num_epochs=epochs,
    ))

    # ── M: Wide tau span fixed (test range hypothesis) ──
    cfgs.append(TauSweepConfig(
        name="M_wide_span",
        description="Wide fixed span: τ_sv=(1,30), τ_dcv=(2,60)",
        tau_sv_init=(1.0, 30.0),
        tau_dcv_init=(2.0, 60.0),
        a_sv_init=(1.5, 0.3),
        a_dcv_init=(0.15, 0.03),
        num_epochs=epochs,
    ))

    return cfgs


# ═══════════════════════════════════════════════════════════════════════
#  Run one config through train_stage2_cv
# ═══════════════════════════════════════════════════════════════════════

def run_one_config(
    h5_path: str,
    sc: TauSweepConfig,
    device: str = "cuda",
    cv_folds: int = 2,
    loo_subset: int = 30,
    save_dir: Optional[str] = None,
) -> Dict:
    """Train Stage2 with the given tau config, return metrics."""
    import torch
    from stage2.config import make_config
    from stage2.train import train_stage2_cv

    cfg = make_config(
        h5_path,
        num_epochs=sc.num_epochs,
        device=device,
        cv_folds=cv_folds,
        # Tau / kernel settings from sweep config
        tau_sv_init=sc.tau_sv_init,
        tau_dcv_init=sc.tau_dcv_init,
        a_sv_init=sc.a_sv_init,
        a_dcv_init=sc.a_dcv_init,
        fix_tau_sv=sc.fix_tau_sv,
        fix_tau_dcv=sc.fix_tau_dcv,
        per_neuron_tau_scale=sc.per_neuron_tau_scale,
        per_neuron_amplitudes=sc.per_neuron_amplitudes,
        coupling_gate=sc.coupling_gate,
        graph_poly_order=sc.graph_poly_order,
        # Standard dynamics settings
        learn_lambda_u=True,
        learn_I0=True,
        edge_specific_G=True,
        learn_W_sv=True,
        learn_W_dcv=True,
        learn_noise=True,
        noise_mode="heteroscedastic",
        # LOO evaluation
        eval_loo_subset_size=loo_subset,
        eval_loo_subset_mode="variance",
        eval_loo_window_size=50,
        eval_loo_warmup_steps=40,
        # Don't waste time on final eval plots
        skip_final_eval=True,
    )

    t0 = time.time()
    result = train_stage2_cv(cfg, save_dir=save_dir, show=False)
    elapsed = time.time() - t0

    # Extract metrics from saved npz
    metrics = {"time_s": elapsed, "name": sc.name, "description": sc.description}

    if save_dir is not None:
        cv_path = Path(save_dir) / "cv_onestep.npz"
        if cv_path.exists():
            d = np.load(cv_path, allow_pickle=True)
            cv_r2 = d["cv_r2"]
            metrics["onestep_r2_mean"] = float(np.nanmean(cv_r2))
            metrics["onestep_r2_median"] = float(np.nanmedian(cv_r2))
            metrics["onestep_r2_std"] = float(np.nanstd(cv_r2))
            metrics["n_negative_onestep"] = int((cv_r2 < 0).sum())

            if "cv_loo_r2" in d:
                loo_r2 = d["cv_loo_r2"]
                valid = loo_r2[np.isfinite(loo_r2)]
                metrics["loo_r2_mean"] = float(np.nanmean(valid))
                metrics["loo_r2_median"] = float(np.nanmedian(valid))
                metrics["n_negative_loo"] = int((valid < 0).sum())
                metrics["n_loo"] = len(valid)

            if "cv_loo_r2_windowed" in d:
                loo_w = d["cv_loo_r2_windowed"]
                valid_w = loo_w[np.isfinite(loo_w)]
                metrics["loo_w_r2_mean"] = float(np.nanmean(valid_w))
                metrics["loo_w_r2_median"] = float(np.nanmedian(valid_w))
                metrics["n_negative_loo_w"] = int((valid_w < 0).sum())

    # Extract learned tau values if available
    if save_dir is not None:
        for fold_i in range(cv_folds):
            pt_path = Path(save_dir) / f"fold_{fold_i}_state.pt"
            if pt_path.exists():
                import torch as _torch
                state = _torch.load(pt_path, map_location="cpu", weights_only=True)
                # Extract tau values (reparameterised)
                for prefix in ("sv", "dcv"):
                    raw_key = f"_tau_{prefix}_raw"
                    lo_key = f"_tau_{prefix}_lo"
                    if raw_key in state:
                        from stage2.model import _reparam_fwd, _TAU_LO
                        raw = state[raw_key]
                        tau_val = _reparam_fwd(raw, _TAU_LO, None)
                        metrics[f"fold{fold_i}_tau_{prefix}"] = tau_val.tolist()
                # Extract coupling gate values
                if "_coupling_gate_raw" in state:
                    gate = torch.sigmoid(state["_coupling_gate_raw"])
                    metrics[f"fold{fold_i}_gate_mean"] = float(gate.mean())
                    metrics[f"fold{fold_i}_gate_min"] = float(gate.min())
                    metrics[f"fold{fold_i}_gate_max"] = float(gate.max())
                # Extract tau scale if present
                if "_tau_scale_log" in state:
                    ts = torch.exp(state["_tau_scale_log"])
                    metrics[f"fold{fold_i}_tau_scale_mean"] = float(ts.mean())
                    metrics[f"fold{fold_i}_tau_scale_min"] = float(ts.min())
                    metrics[f"fold{fold_i}_tau_scale_max"] = float(ts.max())
                break  # Only need one fold for diagnostics

    return metrics


# ═══════════════════════════════════════════════════════════════════════
#  Summary + plotting
# ═══════════════════════════════════════════════════════════════════════

def print_summary(all_results: List[Dict], dt: float):
    """Print formatted summary table."""
    print(f"\n{'='*100}")
    print(f"  TAU / KERNEL SWEEP SUMMARY   (dt = {dt:.3f}s)")
    print(f"{'='*100}")

    # Header
    hdr = (f"{'Config':<28s} {'OS_mean':>7s} {'OS_med':>7s} "
           f"{'LOO_mean':>8s} {'LOO_med':>8s} "
           f"{'LOOw_mean':>9s} {'LOOw_med':>9s} "
           f"{'#neg':>4s} {'Time':>6s}  Description")
    print(hdr)
    print("-" * 100)

    # Sort by LOO windowed R² (descending)
    sorted_results = sorted(
        all_results,
        key=lambda r: r.get("loo_w_r2_mean", float("-inf")),
        reverse=True,
    )

    for r in sorted_results:
        nm = r.get("name", "?")[:27]
        os_mean = r.get("onestep_r2_mean", float("nan"))
        os_med = r.get("onestep_r2_median", float("nan"))
        loo_mean = r.get("loo_r2_mean", float("nan"))
        loo_med = r.get("loo_r2_median", float("nan"))
        loo_w_mean = r.get("loo_w_r2_mean", float("nan"))
        loo_w_med = r.get("loo_w_r2_median", float("nan"))
        n_neg = r.get("n_negative_loo_w", -1)
        t_s = r.get("time_s", 0)
        desc = r.get("description", "")[:40]

        print(f"{nm:<28s} {os_mean:>7.4f} {os_med:>7.4f} "
              f"{loo_mean:>8.4f} {loo_med:>8.4f} "
              f"{loo_w_mean:>9.4f} {loo_w_med:>9.4f} "
              f"{n_neg:>4d} {t_s:>5.0f}s  {desc}")

    # Print learned tau values for configs that learned them
    print(f"\n{'='*70}")
    print("  LEARNED TAU VALUES (fold 0)")
    print(f"{'='*70}")
    for r in sorted_results:
        if "fold0_tau_sv" in r:
            nm = r["name"]
            tau_sv = r["fold0_tau_sv"]
            tau_dcv = r.get("fold0_tau_dcv", [])
            # Compute gammas
            gamma_sv = [np.exp(-dt / t) for t in tau_sv]
            gamma_dcv = [np.exp(-dt / t) for t in tau_dcv]
            print(f"\n  {nm}:")
            print(f"    τ_sv  = [{', '.join(f'{t:.2f}' for t in tau_sv)}]  "
                  f"γ = [{', '.join(f'{g:.3f}' for g in gamma_sv)}]")
            print(f"    τ_dcv = [{', '.join(f'{t:.2f}' for t in tau_dcv)}]  "
                  f"γ = [{', '.join(f'{g:.3f}' for g in gamma_dcv)}]")
            # Half-lives
            hl_sv = [-np.log(2) / np.log(g) * dt if g > 0 else 0 for g in gamma_sv]
            hl_dcv = [-np.log(2) / np.log(g) * dt if g > 0 else 0 for g in gamma_dcv]
            print(f"    half-life_sv  = [{', '.join(f'{h:.1f}s' for h in hl_sv)}]")
            print(f"    half-life_dcv = [{', '.join(f'{h:.1f}s' for h in hl_dcv)}]")

    # Print tau scale stats
    for r in sorted_results:
        if "fold0_tau_scale_mean" in r:
            nm = r["name"]
            print(f"\n  {nm}: per-neuron tau scale "
                  f"mean={r['fold0_tau_scale_mean']:.3f}  "
                  f"range=[{r['fold0_tau_scale_min']:.3f}, "
                  f"{r['fold0_tau_scale_max']:.3f}]")

    # Print gate stats
    for r in sorted_results:
        if "fold0_gate_mean" in r:
            nm = r["name"]
            print(f"  {nm}: coupling gate "
                  f"mean={r['fold0_gate_mean']:.3f}  "
                  f"range=[{r['fold0_gate_min']:.3f}, "
                  f"{r['fold0_gate_max']:.3f}]")

    print()
    best = sorted_results[0]
    print(f"  ★ BEST: {best['name']}  LOO_w_mean={best.get('loo_w_r2_mean', float('nan')):.4f}")
    print(f"    {best.get('description', '')}")
    print()


def plot_sweep_results(all_results: List[Dict], save_dir: Path, dt: float):
    """Create comparison bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sorted_results = sorted(
        all_results,
        key=lambda r: r.get("loo_w_r2_mean", float("-inf")),
        reverse=True,
    )

    names = [r["name"] for r in sorted_results]
    loo_w_means = [r.get("loo_w_r2_mean", 0) for r in sorted_results]
    os_means = [r.get("onestep_r2_mean", 0) for r in sorted_results]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Panel 1: LOO windowed R²
    ax = axes[0]
    colors = ["#2ca02c" if v == max(loo_w_means) else "#4682b4" for v in loo_w_means]
    bars = ax.barh(range(len(names)), loo_w_means, color=colors, alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("LOO R² (windowed, mean)", fontsize=11)
    ax.set_title("LOO Windowed R² by Tau Configuration", fontsize=13, fontweight="bold")
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    ax.invert_yaxis()
    for i, v in enumerate(loo_w_means):
        ax.text(max(v + 0.005, 0.005), i, f"{v:.4f}", va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.3)

    # Panel 2: One-step R²
    ax = axes[1]
    colors2 = ["#d62728" if v == max(os_means) else "#ff7f0e" for v in os_means]
    ax.barh(range(len(names)), os_means, color=colors2, alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("One-step R² (mean)", fontsize=11)
    ax.set_title("One-step R² by Tau Configuration", fontsize=13, fontweight="bold")
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    ax.invert_yaxis()
    for i, v in enumerate(os_means):
        ax.text(max(v + 0.005, 0.005), i, f"{v:.4f}", va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.3)

    fig.suptitle(f"Tau / Kernel Sweep (dt = {dt:.3f}s)", fontsize=14,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(save_dir / "tau_sweep_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_dir / 'tau_sweep_comparison.png'}")

    # Panel 3: Gamma decay plot (for configs with learned taus)
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    plotted = False
    for r in sorted_results:
        if "fold0_tau_sv" in r:
            tau_sv = r["fold0_tau_sv"]
            tau_dcv = r.get("fold0_tau_dcv", [])
            nm = r["name"]
            x_sv = list(range(len(tau_sv)))
            gamma_sv = [np.exp(-dt / t) for t in tau_sv]
            ax2.scatter(tau_sv, gamma_sv, s=80, zorder=5, label=f"{nm} SV")
            ax2.scatter(tau_dcv, [np.exp(-dt / t) for t in tau_dcv],
                       s=80, marker="^", zorder=5, label=f"{nm} DCV")
            plotted = True

    if plotted:
        taus = np.linspace(0.1, 60, 200)
        gammas = np.exp(-dt / taus)
        ax2.plot(taus, gammas, "k-", lw=1, alpha=0.4, label=f"γ=exp(-{dt:.2f}/τ)")
        ax2.axhline(0.5, color="red", lw=0.8, ls="--", alpha=0.5, label="γ=0.5 (half-life = 1 step)")
        ax2.axhline(0.9, color="orange", lw=0.8, ls="--", alpha=0.5, label="γ=0.9")
        ax2.set_xlabel("τ (seconds)", fontsize=11)
        ax2.set_ylabel("γ = exp(-dt/τ)", fontsize=11)
        ax2.set_title(f"Learned Kernel Decay Rates (dt={dt:.3f}s)", fontsize=13, fontweight="bold")
        ax2.legend(fontsize=8, ncol=2)
        ax2.grid(alpha=0.3)
        ax2.set_xlim(0, 65)
        ax2.set_ylim(0, 1.05)
        fig2.tight_layout()
        fig2.savefig(save_dir / "tau_gamma_learned.png", dpi=200, bbox_inches="tight")
        plt.close(fig2)
        print(f"  Saved {save_dir / 'tau_gamma_learned.png'}")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Tau/kernel sweep for Stage2")
    ap.add_argument("--h5", required=True, help="Path to worm .h5 file")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cv_folds", type=int, default=2)
    ap.add_argument("--loo_subset", type=int, default=30)
    ap.add_argument("--configs", nargs="*", default=None,
                    help="Run only specific configs (e.g. A_baseline B_learn_taus)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    worm = Path(args.h5).stem

    # Resolve dt
    import h5py
    with h5py.File(args.h5, "r") as f:
        ts = f["timing/timestamp_confocal"][:]
        dt = float(np.median(np.diff(ts)))
    print(f"  Worm: {worm}  dt={dt:.4f}s")
    print(f"  T={len(ts)} timesteps,  duration={ts[-1]-ts[0]:.0f}s")

    # Print gamma analysis for current defaults
    print(f"\n{'='*60}")
    print(f"  CURRENT KERNEL DECAY ANALYSIS (dt={dt:.3f}s)")
    print(f"{'='*60}")
    for name, taus in [("SV", (0.5, 2.5)), ("DCV", (3.0, 5.0))]:
        for tau in taus:
            gamma = np.exp(-dt / tau)
            hl = -np.log(2) / np.log(gamma) * dt if gamma > 0 else 0
            print(f"  τ_{name}={tau:.1f}s  γ={gamma:.4f}  "
                  f"half-life={hl:.2f}s ({hl/dt:.1f} steps)")
    print()

    # Build configs
    all_configs = build_sweep_configs(epochs=args.epochs)
    if args.configs:
        all_configs = [c for c in all_configs if c.name in args.configs]
        if not all_configs:
            print(f"ERROR: No configs matched {args.configs}")
            sys.exit(1)

    print(f"  Running {len(all_configs)} configurations:")
    for c in all_configs:
        r_sv = len(c.tau_sv_init)
        r_dcv = len(c.tau_dcv_init)
        fix_str = "fixed" if c.fix_tau_sv else "LEARN"
        pnt = " +per-neuron-τ" if c.per_neuron_tau_scale else ""
        gate = " -gate" if not c.coupling_gate else ""
        print(f"    {c.name}: r_sv={r_sv} r_dcv={r_dcv} τ={fix_str}{pnt}{gate}")
    print()

    # Run sweep
    all_results = []
    t_total = time.time()

    for i, sc in enumerate(all_configs):
        print(f"\n{'═'*70}")
        print(f"  [{i+1}/{len(all_configs)}] {sc.name}: {sc.description}")
        print(f"{'═'*70}")
        cfg_dir = str(out / sc.name)

        try:
            metrics = run_one_config(
                args.h5, sc,
                device=args.device,
                cv_folds=args.cv_folds,
                loo_subset=args.loo_subset,
                save_dir=cfg_dir,
            )
            all_results.append(metrics)

            # Print key metric
            loo_w = metrics.get("loo_w_r2_mean", float("nan"))
            os_m = metrics.get("onestep_r2_mean", float("nan"))
            print(f"\n  → {sc.name}: OS_mean={os_m:.4f}  LOO_w_mean={loo_w:.4f}  "
                  f"({metrics['time_s']:.0f}s)")

            # Incremental save
            with open(out / "sweep_results_partial.json", "w") as f:
                json.dump(all_results, f, indent=2, default=str)

        except Exception as e:
            print(f"\n  ✗ {sc.name} FAILED: {e}")
            import traceback; traceback.print_exc()
            all_results.append({"name": sc.name, "error": str(e)})

    total_time = time.time() - t_total

    # Save final results
    with open(out / "sweep_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary
    valid_results = [r for r in all_results if "error" not in r]
    if valid_results:
        print_summary(valid_results, dt)
        plot_sweep_results(valid_results, out, dt)

    print(f"\n  Total sweep time: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"  Results saved to {out}")


if __name__ == "__main__":
    main()
