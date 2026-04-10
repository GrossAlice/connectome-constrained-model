"""Run Stage 1 EM deconvolution — all parameters come from Stage1Config."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import h5py

from .config import Stage1Config
from .io_h5 import load_traces_and_regressor, write_stage1_outputs
from .em import fit_stage1_all_neurons


# ── Helpers ─────────────────────────────────────────────────────────────────

def _as_array(v, n: int) -> np.ndarray:
    """Broadcast scalar-or-array to length *n*."""
    a = np.asarray(v, dtype=float)
    return np.broadcast_to(a, n).copy() if a.ndim == 0 else a[:n]


def _load_neuron_labels(cfg: Stage1Config, n_neurons: int) -> Dict[int, str]:
    """Try label CSV first, then fall back to H5-embedded labels."""
    labels: Dict[int, str] = {}

    if cfg.label_csv and cfg.label_csv.strip():
        try:
            from stage1.preprocess import (
                load_label_table, build_label_lookup, get_neuron_labels_for_h5,
            )
            lookup = build_label_lookup(load_label_table(cfg.label_csv))
            labels = get_neuron_labels_for_h5(cfg.h5_path, lookup, n_neurons=n_neurons)
            if not labels:
                print(f"[warn] no matching labels in {cfg.label_csv}")
        except Exception as e:
            print(f"[warn] label CSV load failed: {e}")

    if labels:
        return labels

    try:
        with h5py.File(cfg.h5_path, "r") as f:
            if "gcamp/neuron_labels" in f:
                raw = f["gcamp/neuron_labels"][()]
                labs = [b.decode() if isinstance(b, bytes) else str(b) for b in raw]
                if cfg.max_neurons is not None:
                    labs = labs[:cfg.max_neurons]
                labels = dict(enumerate(labs))
    except Exception:
        pass
    return labels


def _apply_neuron_mask(
    cfg: Stage1Config,
    X: np.ndarray,
    labels: Dict[int, str],
) -> tuple[np.ndarray, Dict[int, str]]:
    """Keep only neurons whose labels appear in the mask file."""
    if not cfg.neuron_mask:
        return X, labels
    if not labels:
        print("[warn] neuron_mask provided but no labels available; skipping")
        return X, labels

    try:
        allowed = {str(x).strip() for x in np.load(cfg.neuron_mask, allow_pickle=True)}
    except Exception as e:
        print(f"[warn] could not load neuron_mask: {e}")
        return X, labels

    alt_re = re.compile(r"-alt\d*$")
    used: set[str] = set()

    def _match(label: str) -> Optional[str]:
        s = label.strip()
        if not s:
            return None
        base = alt_re.sub("", s)
        for cand in (s, base, base + "L", base + "R"):
            if cand in allowed and cand not in used:
                return cand
        return None

    keep: List[int] = []
    for j in range(X.shape[1]):
        tgt = _match(labels.get(j, ""))
        if tgt is not None:
            keep.append(j)
            labels[j] = tgt
            used.add(tgt)

    if not keep:
        print("[warn] neuron_mask kept 0 neurons; skipping")
        return X, labels

    n_before = X.shape[1]
    X = X[:, keep]
    labels = {i: labels[j] for i, j in enumerate(keep)}
    print(f"[mask] Kept {len(keep)}/{n_before} neurons")
    return X, labels


# ── Printing ────────────────────────────────────────────────────────────────

def _print_config(cfg: Stage1Config) -> None:
    dt = 1.0 / cfg.sample_rate_hz
    rho_init = np.exp(-dt / cfg.tau_u_init_sec)
    lam_init = 1.0 - np.exp(-dt / cfg.tau_c_init_sec)
    lines = [
        "\nSTAGE 1 LGSSM — TRAINING CONFIG",
        f"  h5_path           : {cfg.h5_path}",
        f"  trace_dataset     : {cfg.trace_dataset}",
        f"  center_traces     : {cfg.center_traces}",
        f"  use_dff           : {cfg.use_dff}",
    ]
    if cfg.use_dff:
        lines += [
            f"  f0_method         : {cfg.f0_method}",
            f"  f0_quantile       : {cfg.f0_quantile}",
            f"  f0_window_sec     : {cfg.f0_window_sec}",
            f"  f0_eps            : {cfg.f0_eps}",
        ]
    lines += [
        f"  sample_rate_hz    : {cfg.sample_rate_hz}  (dt = {dt:.4f} s)",
        f"  tau_u_init_sec    : {cfg.tau_u_init_sec}  (rho_init = {rho_init:.4f})",
        f"  tau_c_init_sec    : {cfg.tau_c_init_sec}  (lambda_init = {lam_init:.4f})",
        f"  rho_clip          : [{cfg.rho_clip[0]:.4f}, {cfg.rho_clip[1]:.4f}]",
        f"  lambda_clip       : [{cfg.lambda_clip[0]:.4f}, {cfg.lambda_clip[1]:.4f}]",
        f"  share_rho         : {cfg.share_rho}",
        f"  share_lambda_c    : {cfg.share_lambda_c}",
        f"  share_sigma_c     : {cfg.share_sigma_c}",
        f"  fix_alpha         : {cfg.fix_alpha}  (alpha_value = {cfg.alpha_value})",
        f"  fix_tau_c         : {getattr(cfg, 'fix_tau_c', False)}  (tau_c = {cfg.tau_c_init_sec} s)",
        f"  em_max_iters      : {cfg.em_max_iters}",
        f"  em_tol_rel_ll     : {cfg.em_tol_rel_ll}",
    ]
    if cfg.max_neurons is not None:
        lines.append(f"  max_neurons       : {cfg.max_neurons}")
    if cfg.neuron_mask:
        lines.append(f"  neuron_mask       : {cfg.neuron_mask}")
    print("\n".join(lines) + "\n")


def _print_fit_summary(cfg: Stage1Config, out: dict,
                       labels: Dict[int, str], n_fit: int) -> None:
    dt = 1.0 / cfg.sample_rate_hz
    rho    = _as_array(out.get("rho", 0.99), n_fit)
    alpha  = _as_array(out.get("alpha", 1.0), n_fit)
    beta   = _as_array(out.get("beta", 0.0), n_fit)
    sig_u  = _as_array(out.get("sigma_u", 1.0), n_fit)
    sig_y  = _as_array(out.get("sigma_y", 1.0), n_fit)
    lam_c  = _as_array(out.get("lambda_c", 0.5), n_fit)
    sig_c  = _as_array(out.get("sigma_c", 0.01), n_fit)

    tau_u = np.where((rho > 0) & (rho < 1), -dt / np.log(rho), np.inf)
    tau_c = np.where((lam_c > 0) & (lam_c < 1), -dt / np.log(1.0 - lam_c), np.inf)

    print("\nSTAGE 1 LGSSM — FITTED PARAMETERS")
    if np.ptp(lam_c) < 1e-12:
        print(f"  lambda_c (shared) : {lam_c[0]:.6f}  =>  tau_c = {tau_c[0]:.3f} s")
    else:
        print(f"  lambda_c (per-neuron): median={np.nanmedian(lam_c):.4f}")
    if np.ptp(sig_c) < 1e-12:
        print(f"  sigma_c (shared)  : {sig_c[0]:.6f}")
    else:
        print(f"  sigma_c (per-neuron): median={np.nanmedian(sig_c):.6f}  "
              f"range=[{np.nanmin(sig_c):.6f}, {np.nanmax(sig_c):.6f}]")

    hdr = f"  {'Neuron':<8} {'Label':<10} {'rho':>8} {'tau_u(s)':>9} {'sigma_u':>9} {'sigma_y':>9} {'alpha':>8} {'beta':>9}"
    print(f"\n{hdr}\n  {'-' * len(hdr.strip())}")
    for j in range(n_fit):
        lab = labels.get(j, "")
        print(f"  {j:<8d} {lab:<10s} {rho[j]:8.4f} {tau_u[j]:9.2f} "
              f"{sig_u[j]:9.4f} {sig_y[j]:9.4f} {alpha[j]:8.4f} {beta[j]:9.4f}")

    print("\n  Summary:")
    for name, arr in [("rho", rho), ("tau_u", tau_u), ("sigma_c", sig_c),
                      ("sigma_u", sig_u), ("sigma_y", sig_y)]:
        v = arr[np.isfinite(arr)]
        if v.size:
            print(f"    {name:8s}: median={np.nanmedian(v):.4f}  "
                  f"range=[{np.nanmin(v):.4f}, {np.nanmax(v):.4f}]")
    print()


# ── Plotting ────────────────────────────────────────────────────────────────

def _plot_deconvolved_examples(
    cfg: Stage1Config,
    labels: Dict[int, str],
    out: dict,
    X: np.ndarray,
    n_examples: int = 15,
    per_page: int = 5,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    u_mean = np.asarray(out.get("u_mean"), dtype=float)
    c_mean = np.asarray(out.get("c_mean"), dtype=float)
    if u_mean.ndim != 2 or c_mean.ndim != 2:
        return

    valid = np.flatnonzero(~np.isnan(u_mean).all(axis=0))
    if valid.size == 0:
        return

    ranked = valid[np.argsort(np.nanvar(u_mean[:, valid], axis=0))[::-1]]
    chosen = ranked[:min(n_examples, ranked.size)]
    dt = 1.0 / float(cfg.sample_rate_hz)
    time = np.arange(X.shape[0]) * dt
    stem = Path(cfg.h5_path).stem
    n_pages = (len(chosen) + per_page - 1) // per_page

    for page, start in enumerate(range(0, len(chosen), per_page)):
        batch = chosen[start:start + per_page]
        fig, axes = plt.subplots(len(batch), 1,
                                 figsize=(12, 2.8 * len(batch)), sharex=True)
        if len(batch) == 1:
            axes = [axes]

        twin_axes = []
        for ax, idx in zip(axes, batch):
            lab = labels.get(int(idx), f"n{int(idx)}")
            ax.plot(time, X[:, idx], color="0.7", lw=1.0, label="trace")
            ax.plot(time, c_mean[:, idx], color="tab:blue", lw=1.2, label="c_mean")
            ax.set_ylabel(lab)
            ax.grid(alpha=0.2, lw=0.5)
            ax2 = ax.twinx()
            ax2.plot(time, u_mean[:, idx], color="tab:orange", lw=1.0, label="u_mean")
            ax2.set_ylabel("u", color="tab:orange")
            ax2.tick_params(axis="y", colors="tab:orange")
            twin_axes.append(ax2)

        h1, l1 = axes[0].get_legend_handles_labels()
        h2, l2 = twin_axes[0].get_legend_handles_labels()
        axes[0].legend(h1 + h2, l1 + l2, loc="upper right", ncol=3, fontsize=8)
        axes[-1].set_xlabel("time (s)")
        fig.suptitle(f"{stem}: deconvolved examples ({page + 1}/{n_pages})", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.98])

        suffix = f"_{page + 1:02d}" if n_pages > 1 else ""
        fig.savefig(os.path.join(cfg.save_dir, f"{stem}_deconvolved_examples{suffix}.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)
    print(f"[plot] Saved deconvolved examples -> {cfg.save_dir}")


def _plot_parameter_distributions(cfg: Stage1Config, out: dict, n_fit: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    dt = 1.0 / float(cfg.sample_rate_hz)
    rho   = _as_array(out.get("rho", 0.99), n_fit)
    beta  = _as_array(out.get("beta", 0.0), n_fit)
    sig_u = _as_array(out.get("sigma_u", 1.0), n_fit)
    sig_y = _as_array(out.get("sigma_y", 1.0), n_fit)
    lam_c = _as_array(out.get("lambda_c", 0.5), n_fit)
    sig_c = _as_array(out.get("sigma_c", 0.01), n_fit)

    tau_u = np.where((rho > 0) & (rho < 1), -dt / np.log(rho), np.nan)
    tau_c = np.where((lam_c > 0) & (lam_c < 1), -dt / np.log(1.0 - lam_c), np.nan)

    panels = [
        (tau_u, "tau_u (s)"), (tau_c, "tau_c (s)"),
        (sig_u, "sigma_u"),   (sig_y, "sigma_y"),
        (sig_c, "sigma_c"),   (beta,  "beta"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, (vals, title) in zip(axes.ravel(), panels):
        v = vals[np.isfinite(vals)]
        if v.size == 0:
            ax.set_title(title)
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            continue
        if np.ptp(v) < 1e-12:
            ax.axvline(v[0], color="tab:blue", lw=2)
            ax.set_yticks([])
        else:
            ax.hist(v, bins=min(30, max(10, int(np.sqrt(v.size)))), color="tab:blue", alpha=0.8)
        ax.set_title(title)
        ax.grid(alpha=0.2, lw=0.5)

    stem = Path(cfg.h5_path).stem
    fig.suptitle(f"{stem}: learned parameter distributions", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(cfg.save_dir, f"{stem}_parameter_distributions.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved parameter distributions -> {out_path}")


def _plot_cascade(cfg: Stage1Config, labels: Dict[int, str],
                  out: dict, n_fit: int) -> None:
    """Plot cascade demos for stimulated or movement-onset neurons."""
    try:
        from stage1.plot import plot_stage1_cascade
    except ImportError:
        print("[plot] plot_stage1_demo not available; skipping cascade")
        return

    stem = Path(cfg.h5_path).stem

    # Try stimulated neurons first
    stim_idx = _find_stim_neurons(cfg.h5_path, n_fit)
    if stim_idx:
        print(f"\n[stim] {len(stim_idx)} stimulated neurons detected")
        for j in stim_idx:
            lab = labels.get(j, f"n{j}")
            out_png = os.path.join(cfg.save_dir, f"{stem}_neuron{j:03d}_{lab}.png")
            try:
                plot_stage1_cascade(h5_path=cfg.h5_path, neuron_idx=j,
                                    auto_select_peak=True, peak_window_sec=50,
                                    save_path=out_png, dpi=200)
            except Exception as e:
                print(f"  [skip] neuron {j} ({lab}): {e}")
        return

    # Fall back to movement onset
    best_onset = _find_movement_onset(cfg)
    if best_onset is None:
        print("[plot] No stimulated neurons or velocity data; skipping cascade")
        return

    u_mean = out.get("u_mean")
    if u_mean is not None:
        valid = np.flatnonzero(~np.isnan(u_mean).all(axis=0))
        best_neuron = int(valid[np.argmax(np.nanvar(u_mean[:, valid], axis=0))]) if valid.size else 0
    else:
        best_neuron = 0

    lab = labels.get(best_neuron, f"n{best_neuron}")
    out_png = os.path.join(cfg.save_dir, f"{stem}_movement_onset_{lab}.png")
    try:
        plot_stage1_cascade(h5_path=cfg.h5_path, neuron_idx=best_neuron,
                            center_t=best_onset, peak_window_sec=50,
                            save_path=out_png, dpi=200)
    except Exception as e:
        print(f"  [skip] movement-onset plot: {e}")


def _find_stim_neurons(h5_path: str, n_fit: int) -> List[int]:
    """Return sorted indices of stimulated neurons from H5 metadata."""
    try:
        with h5py.File(h5_path, "r") as f:
            for key in ("stimulus/stimulated_cell_indices",
                        "optogenetics/stim_cell_indices"):
                if key in f:
                    sci = np.array(f[key][()], dtype=int).ravel()
                    return sorted(int(s) for s in set(sci) if 0 <= s < n_fit)
            if "optogenetics/stim_matrix" in f:
                sm = f["optogenetics/stim_matrix"][()]
                if sm.shape[0] < sm.shape[1]:
                    sm = sm.T
                return sorted(c for c in range(min(sm.shape[1], n_fit))
                              if np.any(sm[:, c] > 0))
    except Exception:
        pass
    return []


def _find_movement_onset(cfg: Stage1Config) -> Optional[int]:
    """Find best quiescence->movement onset from velocity data."""
    try:
        with h5py.File(cfg.h5_path, "r") as f:
            if "behavior/velocity" not in f:
                return None
            velocity = np.array(f["behavior/velocity"][()], dtype=float)
    except Exception:
        return None

    if velocity.size == 0:
        return None

    speed = np.abs(velocity)
    finite = speed[np.isfinite(speed)]
    if finite.size == 0:
        return None

    thresh = max(float(np.percentile(finite, 25)), 0.005)
    moving = speed > thresh
    min_quiet = max(3, int(2.0 * cfg.sample_rate_hz))

    onsets: List[int] = []
    quiet = 0
    for t in range(1, len(moving)):
        quiet = quiet + 1 if not moving[t - 1] else 0
        if moving[t] and quiet >= min_quiet:
            onsets.append(t)
            quiet = 0

    if not onsets:
        return None

    burst = min(5, len(velocity) - max(onsets))
    return max(onsets, key=lambda o: float(np.nanmean(speed[o:o + burst]))
               if o + burst <= len(speed) else 0.0)


# ── Main entry point ────────────────────────────────────────────────────────

def run_stage1(cfg: Stage1Config) -> dict:
    """Run the full Stage 1 pipeline: load -> fit -> write -> validate -> plot."""
    _print_config(cfg)

    # Load & mask
    X = load_traces_and_regressor(cfg)
    if cfg.max_neurons is not None and X.shape[1] > cfg.max_neurons:
        X = X[:, :cfg.max_neurons]

    labels = _load_neuron_labels(cfg, n_neurons=X.shape[1])
    X, labels = _apply_neuron_mask(cfg, X, labels)
    n_fit = X.shape[1]

    # Fit
    out = fit_stage1_all_neurons(X, cfg)
    out["fit_mask"] = np.any(np.isfinite(np.asarray(out["u_mean"])), axis=0).astype(np.uint8)
    _print_fit_summary(cfg, out, labels, n_fit)

    # Write
    write_stage1_outputs(cfg, out, overwrite=cfg.overwrite)

    # Validate
    print("[validate] Running checks...")
    try:
        from stage1.preprocess import validate_results
        validate_results(X, out, cfg,
                         sample_neurons=list(range(min(5, n_fit))),
                         verbose=True)
    except ImportError:
        print("[warn] scripts.preprocess not available; skipping validation")

    # Plot
    os.makedirs(cfg.save_dir, exist_ok=True)
    _plot_deconvolved_examples(cfg, labels, out, X)
    _plot_parameter_distributions(cfg, out, n_fit)
    _plot_cascade(cfg, labels, out, n_fit)

    return out


def run_batch(cfgs: List[Stage1Config]) -> None:
    """Run Stage 1 on multiple H5 files sequentially."""
    for i, cfg in enumerate(cfgs, 1):
        print(f"\n{'=' * 60}\n[Stage1] File {i}/{len(cfgs)}: {cfg.h5_path}\n{'=' * 60}")
        run_stage1(cfg)


def main(argv=None) -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Run Stage 1 EM deconvolution on one or more HDF5 files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--h5", dest="h5_paths", nargs="+", required=True, metavar="PATH",
        help="One or more HDF5 worm files, or a directory of .h5 files",
    )
    parser.add_argument(
        "--save_dir", default="output_plots/stage1",
        help="Root output directory; a sub-folder per worm is created automatically",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cpu",
                        help="'cpu' or 'cuda' (passed to Stage1Config if supported)")

    args = parser.parse_args(argv)

    # Expand any directories to .h5 files inside them
    resolved: List[Path] = []
    for p in args.h5_paths:
        pp = Path(p)
        if pp.is_dir():
            resolved.extend(sorted(pp.glob("*.h5")))
        else:
            resolved.append(pp)

    if not resolved:
        parser.error("No .h5 files found")

    save_root = Path(args.save_dir)
    cfgs = [
        Stage1Config(
            h5_path=str(p),
            save_dir=str(save_root / p.stem),
            overwrite=args.overwrite,
        )
        for p in resolved
    ]

    print(f"[Stage1] Running on {len(cfgs)} file(s) → {save_root}")
    run_batch(cfgs)


if __name__ == "__main__":
    main()
