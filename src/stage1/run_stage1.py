from __future__ import annotations

import argparse
import importlib
import os
import re
import sys
from pathlib import Path
from typing import List

import numpy as np
import h5py

from .config import Stage1Config
from .io_h5 import load_traces_and_regressor, write_stage1_outputs
from .em import fit_stage1_all_neurons

from scripts.preprocess import (
    validate_results,
    load_label_table,
    build_label_lookup,
    get_neuron_labels_for_h5,
)



def _cfg_default(name: str):
    return Stage1Config.__dataclass_fields__[name].default


def _build_config(args: argparse.Namespace, h5_path: str) -> Stage1Config:
    share_lambda_c = _cfg_default("share_lambda_c")
    if args.per_neuron_lambda_c:
        share_lambda_c = False
    if args.share_lambda_c:
        share_lambda_c = True

    share_sigma_c = _cfg_default("share_sigma_c")
    if args.per_neuron_sigma_c:
        share_sigma_c = False
    if args.share_sigma_c:
        share_sigma_c = True

    share_rho = _cfg_default("share_rho")
    if args.share_rho:
        share_rho = True
    if args.per_neuron_rho:
        share_rho = False

    center_traces = _cfg_default("center_traces")
    if args.no_center_traces:
        center_traces = False

    use_dff = _cfg_default("use_dff")
    if getattr(args, "use_dff", None) is not None:
        use_dff = bool(args.use_dff)

    fix_alpha = _cfg_default("fix_alpha")
    if getattr(args, "fix_alpha", None) is not None:
        fix_alpha = bool(args.fix_alpha)

    alpha_value = _cfg_default("alpha_value")
    if getattr(args, "alpha_value", None) is not None:
        alpha_value = float(args.alpha_value)

    cfg_kwargs = dict(
        h5_path=h5_path,
        trace_dataset=args.trace,
        center_traces=center_traces,
        use_dff=use_dff,
        share_rho=share_rho,
        share_lambda_c=share_lambda_c,
        share_sigma_c=share_sigma_c,
        fix_alpha=fix_alpha,
        alpha_value=alpha_value,
    )

    if args.f0_method is not None:
        cfg_kwargs["f0_method"] = str(args.f0_method)
    if args.f0_quantile is not None:
        cfg_kwargs["f0_quantile"] = float(args.f0_quantile)
    if args.f0_window_sec is not None:
        cfg_kwargs["f0_window_sec"] = float(args.f0_window_sec)
    if args.f0_eps is not None:
        cfg_kwargs["f0_eps"] = float(args.f0_eps)

    if args.em_max_iters is not None:
        cfg_kwargs["em_max_iters"] = int(args.em_max_iters)
    if args.em_tol_rel_ll is not None:
        cfg_kwargs["em_tol_rel_ll"] = float(args.em_tol_rel_ll)

    if args.sample_rate_hz is not None:
        cfg_kwargs["sample_rate_hz"] = float(args.sample_rate_hz)
    if args.tau_c_init_sec is not None:
        cfg_kwargs["tau_c_init_sec"] = float(args.tau_c_init_sec)
    if args.tau_u_init_sec is not None:
        cfg_kwargs["tau_u_init_sec"] = float(args.tau_u_init_sec)
    if args.sigma_y_floor is not None:
        cfg_kwargs["sigma_y_floor"] = float(args.sigma_y_floor)
    if args.sigma_y_floor_frac is not None:
        cfg_kwargs["sigma_y_floor_frac"] = float(args.sigma_y_floor_frac)

    if args.lambda_clip_min is not None or args.lambda_clip_max is not None:
        default_clip = _cfg_default("lambda_clip")
        lo = float(args.lambda_clip_min) if args.lambda_clip_min is not None else float(default_clip[0])
        hi = float(args.lambda_clip_max) if args.lambda_clip_max is not None else float(default_clip[1])
        cfg_kwargs["lambda_clip"] = (lo, hi)
    if args.rho_clip_min is not None or args.rho_clip_max is not None:
        default_clip = _cfg_default("rho_clip")
        lo = float(args.rho_clip_min) if args.rho_clip_min is not None else float(default_clip[0])
        hi = float(args.rho_clip_max) if args.rho_clip_max is not None else float(default_clip[1])
        cfg_kwargs["rho_clip"] = (lo, hi)

    return Stage1Config(**cfg_kwargs)


def _print_training_config(cfg: Stage1Config, args: argparse.Namespace) -> None:
    print("\nSTAGE 1 LGSSM — TRAINING CONFIG")
    dt = 1.0 / cfg.sample_rate_hz
    rho_init = np.exp(-dt / cfg.tau_u_init_sec)
    lam_init = 1.0 - np.exp(-dt / cfg.tau_c_init_sec)
    print(f"  h5_path           : {cfg.h5_path}")
    print(f"  trace_dataset     : {cfg.trace_dataset}")
    print(f"  center_traces     : {cfg.center_traces}")
    print(f"  use_dff           : {cfg.use_dff}")
    if cfg.use_dff:
        print(f"  f0_method         : {cfg.f0_method}")
        print(f"  f0_quantile       : {cfg.f0_quantile}")
        print(f"  f0_window_sec     : {cfg.f0_window_sec}")
        print(f"  f0_eps            : {cfg.f0_eps}")
    print(f"  sample_rate_hz    : {cfg.sample_rate_hz}  (dt = {dt:.4f} s)")
    print(f"  tau_u_init_sec    : {cfg.tau_u_init_sec}  (rho_init = {rho_init:.4f})")
    print(f"  tau_c_init_sec    : {cfg.tau_c_init_sec}  (lambda_init = {lam_init:.4f})")
    print(f"  rho_clip          : [{cfg.rho_clip[0]:.4f}, {cfg.rho_clip[1]:.4f}]  "
          f"(tau_u range: [{-dt/np.log(cfg.rho_clip[1]):.2f}, {-dt/np.log(cfg.rho_clip[0]):.2f}] s)")
    print(f"  lambda_clip       : [{cfg.lambda_clip[0]:.4f}, {cfg.lambda_clip[1]:.4f}]  "
          f"(tau_c range: [{-dt/np.log(1.0-cfg.lambda_clip[0]):.2f}, {-dt/np.log(1.0-cfg.lambda_clip[1]):.2f}] s)")
    print(f"  share_rho         : {cfg.share_rho}")
    print(f"  share_lambda_c    : {cfg.share_lambda_c}")
    print(f"  share_sigma_c     : {cfg.share_sigma_c}")
    print(f"  fix_alpha         : {cfg.fix_alpha}  (alpha_value = {cfg.alpha_value})")
    print(f"  sigma_u_scale_init: {cfg.sigma_u_scale_init}")
    print(f"  sigma_c_init      : {cfg.sigma_c_init}")
    print(f"  sigma_y_floor     : {cfg.sigma_y_floor}")
    print(f"  sigma_y_floor_frac: {cfg.sigma_y_floor_frac}")
    print(f"  em_max_iters      : {cfg.em_max_iters}")
    print(f"  em_tol_rel_ll     : {cfg.em_tol_rel_ll}")
    if getattr(args, "neuron_mask", None):
        print(f"  neuron_mask       : {args.neuron_mask}")
    if args.max_neurons is not None:
        print(f"  max_neurons       : {args.max_neurons}")
    print()


def _load_neuron_labels(args: argparse.Namespace, h5_path: str, n_neurons: int) -> dict[int, str]:
    neuron_labels: dict[int, str] = {}
    if args.label_csv and args.label_csv.strip():
        try:
            df_labels = load_label_table(args.label_csv)
            label_lookup = build_label_lookup(df_labels)
            neuron_labels = get_neuron_labels_for_h5(
                h5_path,
                label_lookup,
                n_neurons=n_neurons,
            )
            if not neuron_labels:
                print(f"[warning] no matching labels found for H5 {h5_path}; check worm ID in CSV")
        except Exception as e:
            print(f"[warning] could not load labels from {args.label_csv}: {e}")

    if neuron_labels:
        return neuron_labels

    try:
        with h5py.File(h5_path, "r") as f:
            if "gcamp/neuron_labels" in f:
                labs = f["gcamp/neuron_labels"][()]
                labels_list = []
                for lab in labs:
                    if isinstance(lab, bytes):
                        labels_list.append(lab.decode("utf-8"))
                    else:
                        labels_list.append(str(lab))
                if args.max_neurons is not None:
                    labels_list = labels_list[: args.max_neurons]
                neuron_labels = {i: labels_list[i] for i in range(len(labels_list))}
    except Exception:
        pass
    return neuron_labels


def _apply_neuron_mask(
    args: argparse.Namespace,
    X: np.ndarray,
    neuron_labels: dict[int, str],
) -> tuple[np.ndarray, dict[int, str]]:
    if not getattr(args, "neuron_mask", None):
        return X, neuron_labels
    try:
        mask_arr = np.load(str(Path(args.neuron_mask).expanduser()), allow_pickle=True)
        allowed = {str(x).strip() for x in mask_arr}
        n_before = X.shape[1]

        if not neuron_labels:
            print("[warning] neuron_mask provided but no neuron labels available; skipping mask filter")
            return X, neuron_labels

        used_targets = set()
        alt_suffix_re = re.compile(r"-alt\d*$")

        def _map_to_allowed(label: str) -> str | None:
            s = str(label).strip()
            if not s:
                return None
            if s in allowed and s not in used_targets:
                return s
            base = alt_suffix_re.sub("", s)
            if base in allowed and base not in used_targets:
                return base
            if not base.endswith("L") and not base.endswith("R"):
                cand_l = base + "L"
                if cand_l in allowed and cand_l not in used_targets:
                    return cand_l
                cand_r = base + "R"
                if cand_r in allowed and cand_r not in used_targets:
                    return cand_r
            return None

        keep_idx = []
        remapped = 0
        for j in range(X.shape[1]):
            src_label = neuron_labels.get(j, "")
            tgt_label = _map_to_allowed(src_label)
            if tgt_label is not None:
                keep_idx.append(j)
                if tgt_label != src_label:
                    remapped += 1
                neuron_labels[j] = tgt_label
                used_targets.add(tgt_label)

        if len(keep_idx) == 0:
            print("[warning] neuron_mask kept 0 neurons; skipping mask filter")
            return X, neuron_labels
        if len(keep_idx) < X.shape[1]:
            X = X[:, keep_idx]
            neuron_labels = {new_j: neuron_labels[old_j] for new_j, old_j in enumerate(keep_idx)}
            print(f"[mask] Kept {len(keep_idx)}/{n_before} neurons using neuron_mask")
            if remapped > 0:
                print(f"[mask] Recovered/remapped {remapped} labels to allowed atlas names")
        return X, neuron_labels
    except Exception as e:
        print(f"[warning] could not apply neuron_mask {args.neuron_mask}: {e}")
        return X, neuron_labels


def _print_fit_summary(cfg: Stage1Config, out: dict, neuron_labels: dict[int, str], n_fit: int) -> None:
    print("\nSTAGE 1 LGSSM — FITTED PARAMETERS")
    dt = 1.0 / cfg.sample_rate_hz
    rho = np.asarray(out.get("rho", []), dtype=float)
    lam_c = out.get("lambda_c", None)
    sigma_c = out.get("sigma_c", None)
    alpha = np.asarray(out.get("alpha", []), dtype=float)
    beta = np.asarray(out.get("beta", []), dtype=float)
    sigma_u = np.asarray(out.get("sigma_u", []), dtype=float)
    sigma_y = np.asarray(out.get("sigma_y", []), dtype=float)

    if lam_c is not None:
        lam_c_val = float(lam_c) if np.ndim(lam_c) == 0 else lam_c
        if np.ndim(lam_c_val) == 0:
            tau_c = -dt / np.log(1.0 - float(lam_c_val))
            print(f"  lambda_c (shared) : {float(lam_c_val):.6f}  =>  tau_c = {tau_c:.3f} s")
        else:
            print(f"  lambda_c (per-neuron): median={np.nanmedian(lam_c_val):.4f}")
    if sigma_c is not None:
        if np.ndim(sigma_c) == 0:
            print(f"  sigma_c (shared)  : {float(sigma_c):.6f}")
        else:
            sigma_c_arr = np.asarray(sigma_c, dtype=float)[:n_fit]
            print(
                f"  sigma_c (per-neuron): median={np.nanmedian(sigma_c_arr):.6f}  "
                f"range=[{np.nanmin(sigma_c_arr):.6f}, {np.nanmax(sigma_c_arr):.6f}]"
            )

    if rho.size > 0:
        print(f"\n  {'Neuron':<8} {'Label':<10} {'rho':>8} {'tau_u(s)':>9} {'sigma_u':>9} {'sigma_y':>9} {'alpha':>8} {'beta':>9}")
        print(f"  {'-'*8} {'-'*10} {'-'*8} {'-'*9} {'-'*9} {'-'*9} {'-'*8} {'-'*9}")
        for j in range(n_fit):
            label = neuron_labels.get(j, "")
            rho_j = float(rho[j]) if j < len(rho) else float(rho)
            tau_u_j = -dt / np.log(rho_j) if 0 < rho_j < 1 else float('inf')
            su_j = float(sigma_u[j]) if j < len(sigma_u) else float('nan')
            sy_j = float(sigma_y[j]) if j < len(sigma_y) else float('nan')
            a_j = float(alpha[j]) if j < len(alpha) else float('nan')
            b_j = float(beta[j]) if j < len(beta) else float('nan')
            print(f"  {j:<8d} {label:<10s} {rho_j:8.4f} {tau_u_j:9.2f} {su_j:9.4f} {sy_j:9.4f} {a_j:8.4f} {b_j:9.4f}")

        tau_u_all = np.array(
            [-dt / np.log(float(rho[j])) if 0 < float(rho[j]) < 1 else np.nan
             for j in range(min(len(rho), n_fit))]
        )
        print("\n  Summary:")
        print(f"    rho     : median={np.nanmedian(rho[:n_fit]):.4f}  "
              f"range=[{np.nanmin(rho[:n_fit]):.4f}, {np.nanmax(rho[:n_fit]):.4f}]")
        print(f"    tau_u   : median={np.nanmedian(tau_u_all):.2f}s  "
              f"range=[{np.nanmin(tau_u_all):.2f}, {np.nanmax(tau_u_all):.2f}]s")
        if sigma_c is not None and np.ndim(sigma_c) != 0:
            sigma_c_arr = np.asarray(sigma_c, dtype=float)[:n_fit]
            print(f"    sigma_c : median={np.nanmedian(sigma_c_arr):.4f}  "
                  f"range=[{np.nanmin(sigma_c_arr):.4f}, {np.nanmax(sigma_c_arr):.4f}]")
        print(f"    sigma_u : median={np.nanmedian(sigma_u[:n_fit]):.4f}  "
              f"range=[{np.nanmin(sigma_u[:n_fit]):.4f}, {np.nanmax(sigma_u[:n_fit]):.4f}]")
        print(f"    sigma_y : median={np.nanmedian(sigma_y[:n_fit]):.4f}  "
              f"range=[{np.nanmin(sigma_y[:n_fit]):.4f}, {np.nanmax(sigma_y[:n_fit]):.4f}]")
    print()


def _compute_fit_mask(out: dict) -> None:
    """Mark which neurons have valid (finite) fitted outputs."""
    out["fit_mask"] = np.any(np.isfinite(np.asarray(out["u_mean"])), axis=0).astype(np.uint8)


def _plot_deconvolved_examples(
    cfg: Stage1Config,
    save_dir: str,
    neuron_labels: dict[int, str],
    out: dict,
    X: np.ndarray,
    n_examples: int = 15,
    examples_per_figure: int = 5,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[Plot] matplotlib unavailable for deconvolved examples: {exc}")
        return

    u_mean = np.asarray(out.get("u_mean"), dtype=float)
    c_mean = np.asarray(out.get("c_mean"), dtype=float)
    if u_mean.ndim != 2 or c_mean.ndim != 2 or X.ndim != 2:
        print("[Plot] Missing deconvolved outputs — skipping deconvolved examples")
        return

    valid = np.flatnonzero(~np.isnan(u_mean).all(axis=0))
    if valid.size == 0:
        print("[Plot] No fitted neurons available — skipping deconvolved examples")
        return

    u_var = np.nanvar(u_mean[:, valid], axis=0)
    ranked = valid[np.argsort(np.nan_to_num(u_var, nan=-np.inf))[::-1]]
    chosen = ranked[: min(n_examples, ranked.size)]
    dt = 1.0 / float(cfg.sample_rate_hz)
    time = np.arange(X.shape[0], dtype=float) * dt
    saved_paths: list[str] = []
    n_per_fig = max(1, int(examples_per_figure))
    n_pages = (len(chosen) + n_per_fig - 1) // n_per_fig

    for page_idx in range(n_pages):
        page_chosen = chosen[page_idx * n_per_fig: (page_idx + 1) * n_per_fig]
        fig, axes = plt.subplots(len(page_chosen), 1, figsize=(12, 2.8 * len(page_chosen)), sharex=True)
        if len(page_chosen) == 1:
            axes = [axes]

        right_axes = []
        for ax, neuron_idx in zip(axes, page_chosen):
            label = neuron_labels.get(int(neuron_idx), f"n{int(neuron_idx)}")
            trace = np.asarray(X[:, neuron_idx], dtype=float)
            calcium = np.asarray(c_mean[:, neuron_idx], dtype=float)
            drive = np.asarray(u_mean[:, neuron_idx], dtype=float)

            ax.plot(time, trace, color="0.7", linewidth=1.0, label="trace")
            ax.plot(time, calcium, color="tab:blue", linewidth=1.2, label="c_mean")
            ax.set_ylabel(label)
            ax.grid(alpha=0.2, linewidth=0.5)

            ax_right = ax.twinx()
            ax_right.plot(time, drive, color="tab:orange", linewidth=1.0, label="u_mean deconv")
            ax_right.set_ylabel("u", color="tab:orange")
            ax_right.tick_params(axis="y", colors="tab:orange")
            right_axes.append(ax_right)

        handles_left, labels_left = axes[0].get_legend_handles_labels()
        handles_right, labels_right = right_axes[0].get_legend_handles_labels()
        axes[0].legend(handles_left + handles_right, labels_left + labels_right, loc="upper right", ncol=3, fontsize=8)
        axes[-1].set_xlabel("time (s)")
        fig.suptitle(
            f"{Path(cfg.h5_path).stem}: deconvolved signal examples ({page_idx + 1}/{n_pages})",
            fontsize=14,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.98])

        if n_pages == 1:
            out_name = f"{Path(cfg.h5_path).stem}_deconvolved_examples.png"
        else:
            out_name = f"{Path(cfg.h5_path).stem}_deconvolved_examples_{page_idx + 1:02d}.png"
        out_path = os.path.join(save_dir, out_name)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(out_path)

    if len(saved_paths) == 1:
        print(f"[Plot] Saved deconvolved examples -> {saved_paths[0]}")
    else:
        print(f"[Plot] Saved deconvolved examples -> {len(saved_paths)} files in {save_dir}")


def _plot_parameter_distributions(cfg: Stage1Config, save_dir: str, out: dict, n_fit: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[Plot] matplotlib unavailable for parameter distributions: {exc}")
        return

    dt = 1.0 / float(cfg.sample_rate_hz)
    rho = np.asarray(out.get("rho", []), dtype=float)
    beta = np.asarray(out.get("beta", []), dtype=float)[:n_fit]
    sigma_u = np.asarray(out.get("sigma_u", []), dtype=float)[:n_fit]
    sigma_y = np.asarray(out.get("sigma_y", []), dtype=float)[:n_fit]
    lambda_c = out.get("lambda_c")
    sigma_c = out.get("sigma_c")

    if rho.ndim == 0:
        rho_vals = np.full(n_fit, float(rho), dtype=float)
    else:
        rho_vals = rho[:n_fit]
    tau_u = np.array([
        -dt / np.log(v) if np.isfinite(v) and 0.0 < v < 1.0 else np.nan
        for v in rho_vals
    ])

    if np.ndim(lambda_c) == 0:
        lambda_vals = np.full(n_fit, float(lambda_c), dtype=float)
    else:
        lambda_vals = np.asarray(lambda_c, dtype=float)[:n_fit]
    tau_c = np.array([
        -dt / np.log(1.0 - v) if np.isfinite(v) and 0.0 < v < 1.0 else np.nan
        for v in lambda_vals
    ])

    if np.ndim(sigma_c) == 0:
        sigma_c_vals = np.full(n_fit, float(sigma_c), dtype=float)
    else:
        sigma_c_vals = np.asarray(sigma_c, dtype=float)[:n_fit]

    panels = [
        (tau_u, "tau_u (s)"),
        (tau_c, "tau_c (s)"),
        (sigma_u, "sigma_u"),
        (sigma_y, "sigma_y"),
        (sigma_c_vals, "sigma_c"),
        (beta, "beta"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.ravel()
    for ax, (values, title) in zip(axes, panels):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            ax.set_title(title)
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        if np.nanmax(values) - np.nanmin(values) < 1e-12:
            ax.axvline(values[0], color="tab:blue", linewidth=2)
            ax.set_xlim(values[0] - 0.5, values[0] + 0.5)
            ax.set_yticks([])
        else:
            ax.hist(values, bins=min(30, max(10, int(np.sqrt(values.size)))), color="tab:blue", alpha=0.8)
        ax.set_title(title)
        ax.grid(alpha=0.2, linewidth=0.5)

    for ax in axes[len(panels):]:
        ax.set_axis_off()

    fig.suptitle(f"{Path(cfg.h5_path).stem}: learned parameter distributions", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(save_dir, f"{Path(cfg.h5_path).stem}_parameter_distributions.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved parameter distributions -> {out_path}")


def _make_stage1_plots(
    cfg: Stage1Config,
    save_dir: str,
    neuron_labels: dict[int, str],
    out: dict,
    n_fit: int,
    X: np.ndarray,
) -> None:
    try:
        os.makedirs(save_dir, exist_ok=True)
    except OSError as e:
        print(f"[ERROR] Could not create output directory '{save_dir}': {e}")
        return

    plot_stage1_cascade = importlib.import_module("plot_stage1_demo").plot_stage1_cascade

    _plot_deconvolved_examples(cfg, save_dir, neuron_labels, out, X, n_examples=15, examples_per_figure=5)
    _plot_parameter_distributions(cfg, save_dir, out, n_fit)

    stim_neuron_indices: list[int] = []
    try:
        with h5py.File(cfg.h5_path, "r") as f:
            if "stimulus/stimulated_cell_indices" in f:
                sci = np.array(f["stimulus/stimulated_cell_indices"][()], dtype=int).ravel()
                stim_neuron_indices = sorted(set(int(s) for s in sci if 0 <= s < n_fit))
            if not stim_neuron_indices and "optogenetics/stim_cell_indices" in f:
                sci = np.array(f["optogenetics/stim_cell_indices"][()], dtype=int).ravel()
                stim_neuron_indices = sorted(set(int(s) for s in sci if 0 <= s < n_fit))
            if not stim_neuron_indices and "optogenetics/stim_matrix" in f:
                sm = f["optogenetics/stim_matrix"][()]
                if sm.shape[0] < sm.shape[1]:
                    sm = sm.T
                stim_neuron_indices = sorted(
                    int(c) for c in range(min(sm.shape[1], n_fit))
                    if np.any(sm[:, c] > 0)
                )
    except Exception:
        pass

    if stim_neuron_indices:
        stem = Path(cfg.h5_path).stem
        print(f"\n[Stimulated neurons] {len(stim_neuron_indices)} detected:")
        for j in stim_neuron_indices:
            jlabel = neuron_labels.get(j, f"n{j}")
            print(f"  neuron {j:3d} : {jlabel}")
        print(f"\n[Plot] Generating cascade plots -> {save_dir}/")
        for j in stim_neuron_indices:
            jlabel = neuron_labels.get(j, f"n{j}")
            out_png = os.path.join(save_dir, f"{stem}_neuron{j:03d}_{jlabel}.png")
            try:
                plot_stage1_cascade(
                    h5_path=cfg.h5_path,
                    neuron_idx=j,
                    auto_select_peak=True,
                    peak_window_sec=50,
                    save_path=out_png,
                    dpi=200,
                )
            except Exception as pe:
                print(f"  [SKIP] Neuron {j} ({jlabel}): {pe}")
        print(f"[Plot] Done — {len(stim_neuron_indices)} plots saved to {save_dir}/")
        return

    velocity = None
    try:
        with h5py.File(cfg.h5_path, "r") as f:
            if "behavior/velocity" in f:
                velocity = np.array(f["behavior/velocity"][()], dtype=float)
    except Exception:
        pass

    if velocity is None or len(velocity) == 0:
        print("[Plot] No stimulated neurons or velocity data found — skipping cascade plots")
        return

    speed = np.abs(velocity)
    speed_thresh = float(np.nanpercentile(speed[np.isfinite(speed)], 25))
    speed_thresh = max(speed_thresh, 0.005)
    moving = speed > speed_thresh
    min_quiet = max(3, int(2.0 * cfg.sample_rate_hz))
    onsets: list[int] = []
    quiet_run = 0
    for ti in range(1, len(moving)):
        if not moving[ti - 1]:
            quiet_run += 1
        else:
            quiet_run = 0
        if moving[ti] and quiet_run >= min_quiet:
            onsets.append(ti)
            quiet_run = 0

    if not onsets:
        print("[Plot] No quiescence→movement onsets detected — skipping plots")
        return

    burst_len = min(5, len(velocity) - max(onsets))
    best_onset = max(
        onsets,
        key=lambda o: float(np.nanmean(speed[o:o + burst_len])) if o + burst_len <= len(speed) else 0.0,
    )
    dt_sec = 1.0 / cfg.sample_rate_hz
    print(f"\n[Movement onset] {len(onsets)} quiescence→movement transitions detected (speed_thresh={speed_thresh:.4f})")
    print(f"  Best onset at t={best_onset} ({best_onset * dt_sec:.1f}s)")

    u_out = out.get("u_mean")
    if u_out is not None:
        valid = ~np.isnan(u_out).all(axis=0)
        valid_idx = np.flatnonzero(valid)
        if len(valid_idx) > 0:
            var_u = np.nanvar(u_out[:, valid_idx], axis=0)
            best_neuron = int(valid_idx[np.argmax(var_u)])
        else:
            best_neuron = 0
    else:
        best_neuron = 0

    jlabel = neuron_labels.get(best_neuron, f"n{best_neuron}")
    stem = Path(cfg.h5_path).stem
    out_png = os.path.join(save_dir, f"{stem}_movement_onset_{jlabel}.png")
    print(f"[Plot] Generating movement-onset plot -> {out_png}")
    try:
        plot_stage1_cascade(
            h5_path=cfg.h5_path,
            neuron_idx=best_neuron,
            center_t=best_onset,
            peak_window_sec=50,
            save_path=out_png,
            dpi=200,
        )
    except Exception as pe:
        print(f"  [SKIP] Movement-onset plot: {pe}")


def _run_stage1_for_file(args: argparse.Namespace, h5_path: str, save_dir: str) -> None:
    cfg = _build_config(args, h5_path)
    _print_training_config(cfg, args)

    X = load_traces_and_regressor(cfg)
    if args.max_neurons is not None and X.shape[1] > args.max_neurons:
        X = X[:, : args.max_neurons]

    neuron_labels = _load_neuron_labels(args, h5_path, n_neurons=X.shape[1])
    X, neuron_labels = _apply_neuron_mask(args, X, neuron_labels)

    out = fit_stage1_all_neurons(X, cfg)
    _print_fit_summary(cfg, out, neuron_labels, n_fit=int(X.shape[1]))
    _compute_fit_mask(out)
    write_stage1_outputs(cfg, out, overwrite=args.overwrite)

    print("\n[Validation] Running comprehensive checks...")
    validate_results(X, out, cfg,
                     sample_neurons=list(range(min(5, int(X.shape[1])))),
                     verbose=True)
    _make_stage1_plots(cfg, save_dir, neuron_labels, out, n_fit=int(X.shape[1]), X=X)


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Learn Stage 1 LGSSM parameters and write outputs to HDF5.\n"
            "\n"
            "This script operates exclusively on HDF5 files prepared via ``prepare_h5_data.py``."
            " It no longer supports loading raw Creamer‑format files."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--h5", help="Path to the HDF5 file containing traces")
    input_group.add_argument("--h5_dir", help="Folder containing HDF5 recordings to process")
    parser.add_argument("--pattern", default="*.h5", help="Glob pattern for H5 files when using --h5_dir")
    parser.add_argument("--recursive", action="store_true", help="Recursively search subfolders when using --h5_dir")
    parser.add_argument(
        "--trace",
        default=Stage1Config.__dataclass_fields__["trace_dataset"].default,
        help="Dataset key for the fluorescence traces",
    )

    parser.add_argument(
        "--no_center_traces",
        action="store_true",
        help="Disable per-neuron mean-centering of fluorescence before fitting.",
    )
    dff_group = parser.add_mutually_exclusive_group()
    dff_group.add_argument(
        "--use_dff",
        action="store_true",
        default=None,
        help="Enable ΔF/F0 preprocessing before fitting.",
    )
    dff_group.add_argument(
        "--no_use_dff",
        action="store_false",
        dest="use_dff",
        help="Disable ΔF/F0 preprocessing before fitting.",
    )
    parser.add_argument(
        "--f0_method",
        choices=["quantile", "rolling_quantile", "pre_stim"],
        default=None,
        help="Baseline method for ΔF/F0 when --use_dff is enabled.",
    )
    parser.add_argument(
        "--f0_quantile",
        type=float,
        default=None,
        help="Quantile q in (0,1) used to estimate F0 for ΔF/F0.",
    )
    parser.add_argument(
        "--f0_window_sec",
        type=float,
        default=None,
        help="Window length in seconds for --f0_method rolling_quantile.",
    )
    parser.add_argument(
        "--f0_eps",
        type=float,
        default=None,
        help="Small positive denominator floor for ΔF/F0 computation.",
    )

    parser.add_argument(
        "--sample_rate_hz",
        type=float,
        default=None,
        help="Sampling rate in Hz (controls dt and tau->(rho,lambda_c) init). Defaults to Stage1Config.sample_rate_hz.",
    )
    parser.add_argument(
        "--em_max_iters",
        type=int,
        default=None,
        help="Maximum number of EM iterations (overrides Stage1Config.em_max_iters).",
    )
    parser.add_argument(
        "--em_tol_rel_ll",
        type=float,
        default=None,
        help="Relative LL improvement tolerance for early stopping (overrides Stage1Config.em_tol_rel_ll).",
    )
    parser.add_argument(
        "--tau_c_init_sec",
        type=float,
        default=None,
        help="Initial calcium time constant in seconds. Defaults to Stage1Config.tau_c_init_sec.",
    )
    parser.add_argument(
        "--tau_u_init_sec",
        type=float,
        default=None,
        help="Initial drive time constant in seconds. Defaults to Stage1Config.tau_u_init_sec.",
    )
    parser.add_argument(
        "--sigma_y_floor",
        type=float,
        default=None,
        help="Global minimum floor on observation noise std (in fluorescence units).",
    )
    parser.add_argument(
        "--sigma_y_floor_frac",
        type=float,
        default=None,
        help="Fraction of initial robust sigma_y to use as per-neuron floor. "
             "Effective floor_i = max(sigma_y_floor, frac * robust_sigma_y_init_i). "
             "Set to 0 to disable adaptive flooring. Default 0.8.",
    )
    parser.add_argument(
        "--rho_clip_min",
        type=float,
        default=None,
        help="Lower bound for rho clipping during EM. Defaults to Stage1Config.rho_clip[0].",
    )
    parser.add_argument(
        "--rho_clip_max",
        type=float,
        default=None,
        help="Upper bound for rho clipping during EM. Defaults to Stage1Config.rho_clip[1].",
    )
    parser.add_argument(
        "--lambda_clip_min",
        type=float,
        default=None,
        help="Lower bound for lambda_c clipping during EM. Defaults to Stage1Config.lambda_clip[0].",
    )
    parser.add_argument(
        "--lambda_clip_max",
        type=float,
        default=None,
        help="Upper bound for lambda_c clipping during EM. Lowering this enforces slower calcium decay.",
    )
    lambda_group = parser.add_mutually_exclusive_group()
    lambda_group.add_argument(
        "--share_lambda_c",
        action="store_true",
        help="Use shared lambda_c across all neurons (default).",
    )
    lambda_group.add_argument(
        "--per_neuron_lambda_c",
        action="store_true",
        help="Learn per-neuron lambda_c values.",
    )
    sigma_group = parser.add_mutually_exclusive_group()
    sigma_group.add_argument(
        "--share_sigma_c",
        action="store_true",
        help="Use shared sigma_c across all neurons (default).",
    )
    sigma_group.add_argument(
        "--per_neuron_sigma_c",
        action="store_true",
        help="Learn per-neuron sigma_c values.",
    )

    rho_group = parser.add_mutually_exclusive_group()
    rho_group.add_argument(
        "--share_rho",
        action="store_true",
        help="Learn a single shared rho across all neurons.",
    )
    rho_group.add_argument(
        "--per_neuron_rho",
        action="store_true",
        help="Learn per-neuron rho_i values (default).",
    )
    alpha_group = parser.add_mutually_exclusive_group()
    alpha_group.add_argument(
        "--fix_alpha",
        action="store_true",
        default=None,
        help="Fix alpha to a constant value (use --alpha_value to set it).",
    )
    alpha_group.add_argument(
        "--learn_alpha",
        action="store_false",
        dest="fix_alpha",
        help="Learn alpha during EM updates.",
    )
    parser.add_argument(
        "--alpha_value",
        type=float,
        default=None,
        help="Fixed alpha value used when alpha is fixed.",
    )
    parser.add_argument(
        "--max_neurons",
        type=int,
        default=None,
        help="Limit fitting to the first N neurons (useful for quick tests).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output datasets")
    parser.add_argument("--save_dir", default="./stage1_plots", help="Directory in which to save plots")
    parser.add_argument(
        "--label_csv",
        default=None,
        help="Path to CSV mapping worm/date and ROI indices to neuron labels (optional)",
    )
    parser.add_argument(
        "--neuron_mask",
        default=None,
        help="Path to .npy list of allowed neuron labels; keep only matching neurons",
    )

    args = parser.parse_args(argv)

    if args.h5_dir:
        h5_dir = Path(args.h5_dir).expanduser().resolve()
        if not h5_dir.exists():
            print(f"[ERROR] h5_dir does not exist: {h5_dir}")
            return
        files = sorted(h5_dir.rglob(args.pattern) if args.recursive else h5_dir.glob(args.pattern))
        if not files:
            print(f"[WARN] No files matched {args.pattern} in {h5_dir}")
            return
        base_save = Path(args.save_dir).expanduser().resolve()
        base_save.mkdir(parents=True, exist_ok=True)
        for f in files:
            save_dir = base_save / f.stem
            save_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n[Stage1] {f.name} -> {save_dir}")
            _run_stage1_for_file(args, str(f), str(save_dir))
    else:
        _run_stage1_for_file(args, args.h5, args.save_dir)


if __name__ == "__main__":
    main()
