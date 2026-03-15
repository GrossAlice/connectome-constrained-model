#!/usr/bin/env python
"""Attractor diagnostics for Stage 2 connectome-constrained model.

Computes:
  1. Spectral radius ρ(J) of the full-state Jacobian at the time-mean state.
  2. ρ(J) at every time step along the observed trajectory → histogram + time series.
  3. Maximal Lyapunov exponent (MLE) from the product of Jacobians along the trajectory.
  4. AR(1) contraction rate (1 − λ_u) for comparison.
  5. Eigenvalue spectrum of the u–u block and the full Jacobian.

Usage:
    python -m scripts.attractor_diagnostics \\
        --h5 "data/used/behaviour+neuronal activity atanas (2023)/2022-07-20-01.h5" \\
        --device cuda --save_dir output_plots/attractor_diagnostics
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Allow running from src/ as working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stage2.config import Stage2PTConfig
from stage2.io_h5 import load_data_pt
from stage2.model import Stage2ModelPT
from stage2.spectral import compute_jacobian


# --------------------------------------------------------------------------- #
#  Jacobian analysis helpers                                                    #
# --------------------------------------------------------------------------- #

def spectral_radius(J: torch.Tensor) -> float:
    """Spectral radius ρ(J) = max |eigenvalue|."""
    eigs = torch.linalg.eigvals(J.float())
    return float(eigs.abs().max().item())


def eigenvalues(J: torch.Tensor) -> np.ndarray:
    """Complex eigenvalues of J, sorted by descending modulus."""
    eigs = torch.linalg.eigvals(J.float()).cpu().numpy()
    idx = np.argsort(-np.abs(eigs))
    return eigs[idx]


def lyapunov_exponent_qr(
    model,
    data: dict,
    *,
    n_steps: int | None = None,
    warmup: int = 50,
    reorth_every: int = 1,
) -> dict:
    """Estimate the maximal Lyapunov exponent (MLE) via QR decomposition.

    Propagates Q through the Jacobian product J_T · J_{T-1} · ... · J_1
    using QR re-orthonormalisation to avoid numerical overflow.  Returns
    the full Lyapunov spectrum (one exponent per state dimension).

    Parameters
    ----------
    model : Stage2ModelPT
    data : dict from load_data_pt
    n_steps : number of trajectory steps to use (default: all)
    warmup : transient steps to discard before accumulating exponents
    reorth_every : QR reorthonormalise every this many steps (1 = every step)

    Returns
    -------
    dict with:
        lyap_spectrum : (D,) array — all Lyapunov exponents, descending
        mle : float — maximal Lyapunov exponent (lyap_spectrum[0])
        rho_trajectory : (T-1,) array — ρ(J_t) at each time step
    """
    device = next(model.parameters()).device
    u = data["u_stage1"].to(device)
    T, N = u.shape
    gating = data.get("gating")
    stim = data.get("stim")

    r_sv, r_dcv = model.r_sv, model.r_dcv
    D = N + N * r_sv + N * r_dcv

    if n_steps is None:
        n_steps = T - 1
    n_steps = min(n_steps, T - 1)

    # --- initialise synaptic states by running forward ---
    s_sv = torch.zeros(N, r_sv, device=device)
    s_dcv = torch.zeros(N, r_dcv, device=device)

    # Q matrix for QR tracking (identity at start)
    # Only track top-k exponents to save memory if D is large
    k = min(D, 50)  # top-50 exponents is plenty for diagnostics
    Q = torch.eye(D, k, device=device, dtype=torch.float64)

    lyap_sums = np.zeros(k)
    n_accumulated = 0
    rho_traj = np.zeros(n_steps)

    print(f"[attractor] Lyapunov analysis: D={D}, T={n_steps}, warmup={warmup}, k={k}")
    t0 = time.time()

    for t in range(n_steps):
        g = gating[t] if gating is not None else torch.ones(N, device=device)
        s = stim[t] if stim is not None else None

        # Compute Jacobian at this point
        J = compute_jacobian(model, u[t], s_sv, s_dcv, g, s)  # (D, D)

        # Track spectral radius
        rho_traj[t] = spectral_radius(J)

        # Advance synaptic states (need them for next step)
        with torch.no_grad():
            _, s_sv, s_dcv = model.prior_step(u[t], s_sv, s_dcv, g, s)

        # Propagate perturbation basis: Q' = J @ Q
        JQ = J.double() @ Q  # (D, k)

        # QR decomposition for numerical stability
        if (t + 1) % reorth_every == 0 or t == n_steps - 1:
            Q_new, R = torch.linalg.qr(JQ)
            Q = Q_new

            if t >= warmup:
                # Accumulate log of diagonal of R
                diag_R = R.diag().abs().clamp(min=1e-300)
                lyap_sums += torch.log(diag_R).cpu().numpy()
                n_accumulated += 1
        else:
            Q = JQ

        if (t + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (t + 1) / elapsed
            print(f"  step {t+1}/{n_steps}  ρ(J)={rho_traj[t]:.4f}  ({rate:.1f} steps/s)")

    if n_accumulated > 0:
        lyap_spectrum = lyap_sums / n_accumulated
    else:
        lyap_spectrum = np.zeros(k)

    # Sort descending
    idx = np.argsort(-lyap_spectrum)
    lyap_spectrum = lyap_spectrum[idx]

    elapsed = time.time() - t0
    print(f"[attractor] Lyapunov done in {elapsed:.1f}s")
    print(f"  MLE = {lyap_spectrum[0]:.4f}")
    print(f"  Top-5 exponents: {lyap_spectrum[:5]}")
    print(f"  ρ(J) mean={rho_traj.mean():.4f} max={rho_traj.max():.4f}")

    return {
        "lyap_spectrum": lyap_spectrum,
        "mle": float(lyap_spectrum[0]),
        "rho_trajectory": rho_traj,
    }


def jacobian_at_mean(model, data: dict) -> dict:
    """Jacobian analysis at the temporal-mean state."""
    device = next(model.parameters()).device
    u = data["u_stage1"].to(device)
    T, N = u.shape

    u_mean = u.mean(dim=0)
    s_sv = torch.zeros(N, model.r_sv, device=device)
    s_dcv = torch.zeros(N, model.r_dcv, device=device)
    g = torch.ones(N, device=device)

    # Run a few steps from the mean state to get typical s
    with torch.no_grad():
        for _ in range(50):
            _, s_sv, s_dcv = model.prior_step(u_mean, s_sv, s_dcv, g)

    J = compute_jacobian(model, u_mean, s_sv, s_dcv, g)
    eigs = eigenvalues(J)
    rho = float(np.abs(eigs[0]))

    # u-u block (top-left N×N)
    J_uu = J[:N, :N]
    eigs_uu = eigenvalues(J_uu)
    rho_uu = float(np.abs(eigs_uu[0]))

    print(f"\n[attractor] Jacobian at time-mean state:")
    print(f"  Full Jacobian: ρ = {rho:.4f}   (D = {J.shape[0]})")
    print(f"  u-u block:     ρ = {rho_uu:.4f}  (N = {N})")
    print(f"  Top-5 |eig| (full): {np.abs(eigs[:5])}")
    print(f"  Top-5 |eig| (u-u):  {np.abs(eigs_uu[:5])}")

    return {
        "J": J.cpu().numpy(),
        "eigs": eigs,
        "rho": rho,
        "J_uu": J_uu.cpu().numpy(),
        "eigs_uu": eigs_uu,
        "rho_uu": rho_uu,
    }


# --------------------------------------------------------------------------- #
#  Plotting                                                                     #
# --------------------------------------------------------------------------- #

def plot_eigenvalue_spectrum(eigs: np.ndarray, eigs_uu: np.ndarray,
                            save_path: str | None = None) -> None:
    """Plot eigenvalues in the complex plane with unit circle."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, ev, title in [
        (axes[0], eigs, "Full Jacobian spectrum"),
        (axes[1], eigs_uu, "$u$-$u$ block spectrum"),
    ]:
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.3, lw=1, label="|λ|=1")
        ax.scatter(ev.real, ev.imag, s=8, alpha=0.6, c=np.abs(ev),
                   cmap="plasma", edgecolors="none")
        rho = np.abs(ev).max()
        ax.set_title(f"{title}\nρ = {rho:.4f}", fontsize=11)
        ax.set_xlabel("Re(λ)")
        ax.set_ylabel("Im(λ)")
        ax.set_aspect("equal")
        ax.axhline(0, color="k", alpha=0.1, lw=0.5)
        ax.axvline(0, color="k", alpha=0.1, lw=0.5)
        ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_rho_trajectory(rho_traj: np.ndarray, lambda_u_mean: float,
                        save_path: str | None = None) -> None:
    """Plot spectral radius over time with AR(1) baseline."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), height_ratios=[2, 1])

    # Time series
    ax1.plot(rho_traj, lw=0.5, alpha=0.7, color="steelblue", label="ρ(J_t)")
    ax1.axhline(1.0, color="red", ls="--", lw=1.5, label="stability boundary (ρ=1)")
    ar1_rate = 1.0 - lambda_u_mean
    ax1.axhline(ar1_rate, color="orange", ls=":", lw=1.5,
                label=f"AR(1) rate = 1−λ_u = {ar1_rate:.3f}")
    ax1.set_ylabel("ρ(J)")
    ax1.set_xlabel("time step")
    ax1.set_title(f"Spectral radius along trajectory — mean={rho_traj.mean():.4f}, max={rho_traj.max():.4f}")
    ax1.legend(fontsize=9)

    # Histogram
    ax2.hist(rho_traj, bins=60, density=True, color="steelblue", alpha=0.7, edgecolor="none")
    ax2.axvline(1.0, color="red", ls="--", lw=1.5)
    ax2.axvline(ar1_rate, color="orange", ls=":", lw=1.5)
    ax2.axvline(rho_traj.mean(), color="steelblue", ls="-", lw=1.5, alpha=0.8)
    ax2.set_xlabel("ρ(J)")
    ax2.set_ylabel("density")
    ax2.set_title("Distribution of ρ(J)")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_lyapunov_spectrum(lyap: np.ndarray, save_path: str | None = None) -> None:
    """Bar chart of the top Lyapunov exponents."""
    n_show = min(30, len(lyap))
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["red" if l > 0 else "steelblue" for l in lyap[:n_show]]
    ax.bar(range(n_show), lyap[:n_show], color=colors, edgecolor="none", alpha=0.8)
    ax.axhline(0, color="k", ls="-", lw=0.5)
    ax.set_xlabel("exponent index")
    ax.set_ylabel("Lyapunov exponent")
    ax.set_title(f"Lyapunov spectrum — MLE = {lyap[0]:.4f}")
    n_pos = (lyap > 0).sum()
    ax.annotate(f"{n_pos} positive exponents", xy=(0.98, 0.95), xycoords="axes fraction",
                ha="right", va="top", fontsize=10,
                color="red" if n_pos > 0 else "steelblue")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_summary(jac_mean: dict, lyap_result: dict, lambda_u: np.ndarray,
                 save_path: str | None = None) -> None:
    """Compact summary panel."""
    rho = jac_mean["rho"]
    rho_uu = jac_mean["rho_uu"]
    mle = lyap_result["mle"]
    rho_traj = lyap_result["rho_trajectory"]
    ar1 = float(1.0 - lambda_u.mean())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")

    lines = [
        ("Metric", "Value", "Interpretation"),
        ("─" * 30, "─" * 12, "─" * 40),
        ("ρ(J) at mean state", f"{rho:.4f}", "< 1 → contracting" if rho < 1 else "≥ 1 → unstable"),
        ("ρ(J_uu) at mean state", f"{rho_uu:.4f}", "u-u block only"),
        ("ρ(J) trajectory mean", f"{rho_traj.mean():.4f}", ""),
        ("ρ(J) trajectory max", f"{rho_traj.max():.4f}", "> 1 at any step?" if rho_traj.max() >= 1 else "always < 1"),
        ("Max Lyapunov exponent", f"{mle:.4f}", "< 0 → attracting" if mle < 0 else "≥ 0 → edge of chaos / chaotic"),
        ("AR(1) contraction 1−λ̄_u", f"{ar1:.4f}", "baseline decay rate"),
        ("λ̄_u (mean)", f"{lambda_u.mean():.4f}", f"range [{lambda_u.min():.3f}, {lambda_u.max():.3f}]"),
        ("State dimension D", f"{jac_mean['J'].shape[0]}", ""),
    ]

    y = 0.95
    for header, value, interp in lines:
        ax.text(0.02, y, header, transform=ax.transAxes, fontsize=10,
                family="monospace", fontweight="bold" if y > 0.9 else "normal")
        ax.text(0.45, y, value, transform=ax.transAxes, fontsize=10, family="monospace")
        ax.text(0.62, y, interp, transform=ax.transAxes, fontsize=9,
                family="monospace", color="gray")
        y -= 0.08

    regime = "strongly contractive" if rho < 0.8 else "weakly contractive" if rho < 1 else "unstable"
    ax.text(0.5, 0.02, f"Dynamical regime: {regime}", transform=ax.transAxes,
            fontsize=13, ha="center", fontweight="bold",
            color="steelblue" if rho < 1 else "red",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"))

    fig.suptitle("Attractor Diagnostics Summary", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Main                                                                         #
# --------------------------------------------------------------------------- #

def run_diagnostics(cfg: Stage2PTConfig, save_dir: str, device: str = "cuda") -> dict:
    """Run full attractor diagnostics pipeline."""
    cfg.device = device
    data = load_data_pt(cfg)
    data["_cfg"] = cfg

    u = data["u_stage1"]
    T, N = u.shape
    device_t = torch.device(device)

    # Build model (same as train_stage2)
    lo = max(float(cfg.lambda_u_min), 1e-3)
    hi = min(float(cfg.lambda_u_max), 0.999)
    if data.get("rho_stage1") is not None:
        lambda_u_init = (1.0 - data["rho_stage1"]).clamp(lo, hi)
    else:
        lambda_u_init = torch.full((N,), float(cfg.lambda_u_fallback)).clamp(lo, hi)

    sign_t = data.get("sign_t")

    model = Stage2ModelPT(
        N, data["T_e"], data["T_sv"], data["T_dcv"], data["dt"],
        cfg, device_t, d_ell=data.get("d_ell", 0),
        lambda_u_init=lambda_u_init,
        sign_t=sign_t,
    ).to(device_t)

    # Load trained parameters if available
    import h5py
    h5_path = cfg.h5_path
    try:
        with h5py.File(h5_path, "r") as f:
            if "stage2_pt/params" in f:
                grp = f["stage2_pt/params"]
                state = model.state_dict()
                loaded = []
                for key in grp:
                    if key in state:
                        arr = grp[key][()]
                        t = torch.tensor(arr, dtype=torch.float32, device=device_t)
                        if t.shape == state[key].shape:
                            state[key] = t
                            loaded.append(key)
                model.load_state_dict(state, strict=False)
                print(f"[attractor] Loaded {len(loaded)} params from {h5_path}")
            else:
                print(f"[attractor][warn] No stage2_pt/params in {h5_path} — using init params")
    except Exception as e:
        print(f"[attractor][warn] Could not load params: {e}")

    model.eval()
    lambda_u_np = model.lambda_u.detach().cpu().numpy()

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # 1. Jacobian at time-mean state
    print("\n" + "=" * 60)
    print("  ATTRACTOR DIAGNOSTICS")
    print("=" * 60)
    jac_mean = jacobian_at_mean(model, data)

    # 2. Lyapunov exponent + ρ(J) along trajectory
    lyap = lyapunov_exponent_qr(model, data)

    # 3. Plots
    print("\n[attractor] Generating plots...")
    plot_eigenvalue_spectrum(
        jac_mean["eigs"], jac_mean["eigs_uu"],
        save_path=str(save_path / "eigenvalue_spectrum.png"),
    )
    plot_rho_trajectory(
        lyap["rho_trajectory"], float(lambda_u_np.mean()),
        save_path=str(save_path / "rho_trajectory.png"),
    )
    plot_lyapunov_spectrum(
        lyap["lyap_spectrum"],
        save_path=str(save_path / "lyapunov_spectrum.png"),
    )
    plot_summary(
        jac_mean, lyap, lambda_u_np,
        save_path=str(save_path / "attractor_summary.png"),
    )

    # 4. Save numerical results
    np.savez(
        str(save_path / "attractor_results.npz"),
        eigs_full=jac_mean["eigs"],
        eigs_uu=jac_mean["eigs_uu"],
        rho_mean=jac_mean["rho"],
        rho_uu_mean=jac_mean["rho_uu"],
        lyap_spectrum=lyap["lyap_spectrum"],
        mle=lyap["mle"],
        rho_trajectory=lyap["rho_trajectory"],
        lambda_u=lambda_u_np,
    )
    print(f"[attractor] Results saved to {save_path}/")

    return {"jac_mean": jac_mean, "lyap": lyap}


# --------------------------------------------------------------------------- #
#  CLI                                                                          #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Attractor diagnostics for Stage 2 model")
    parser.add_argument("--h5", "--h5_path", dest="h5_path", required=True,
                        help="H5 file with stage1 results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_dir", default="output_plots/attractor_diagnostics")
    args = parser.parse_args()

    cfg = Stage2PTConfig(h5_path=args.h5_path)
    run_diagnostics(cfg, save_dir=args.save_dir, device=args.device)


if __name__ == "__main__":
    main()
