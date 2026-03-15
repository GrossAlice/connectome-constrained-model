#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from collections import Counter
from typing import Optional

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


def _ensure_TN(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")
    if arr.shape[0] < 400 and arr.shape[1] >= 400 and arr.shape[1] / max(arr.shape[0], 1) > 2:
        return arr.T
    return arr


def _find_body_angle_key(f: h5py.File) -> Optional[str]:
    for key in (
        "behavior/body_angle_all",
        "behavior/body_angle",
        "behaviour/body_angle_all",
        "behaviour/body_angle",
    ):
        if key in f:
            return key
    return None


def _resample_1d(row: np.ndarray, d_target: int) -> np.ndarray:
    if row.size == d_target:
        return row.astype(float)
    x_old = np.linspace(0.0, 1.0, row.size)
    x_new = np.linspace(0.0, 1.0, d_target)
    return np.interp(x_new, x_old, row)


def _fill_frame_no_drop(row: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Fill one frame without dropping it.

    Strategy:
    - >=2 finite values: linear interpolation over index.
    - 1 finite value: constant line at that value.
    - 0 finite values: use fallback (previous frame if available, else zeros).
    """
    out = np.asarray(row, dtype=float)
    finite = np.isfinite(out)
    n = int(finite.sum())

    if n >= 2:
        idx = np.arange(out.size)
        valid_idx = idx[finite]
        valid_val = out[finite]

        # Fill interior gaps first.
        out = np.interp(idx, valid_idx, valid_val)

        # Improve edge fills to avoid artificially flat tails/heads.
        first_i = int(valid_idx[0])
        last_i = int(valid_idx[-1])

        # Leading extrapolation using first local slope.
        if first_i > 0:
            if valid_idx.size >= 2:
                i0, i1 = int(valid_idx[0]), int(valid_idx[1])
                denom = max(1, i1 - i0)
                slope_head = (valid_val[1] - valid_val[0]) / float(denom)
            else:
                slope_head = 0.0
            for j in range(first_i - 1, -1, -1):
                out[j] = out[j + 1] - slope_head

        # Trailing extrapolation using last local slope.
        if last_i < out.size - 1:
            if valid_idx.size >= 2:
                i0, i1 = int(valid_idx[-2]), int(valid_idx[-1])
                denom = max(1, i1 - i0)
                slope_tail = (valid_val[-1] - valid_val[-2]) / float(denom)
            else:
                slope_tail = 0.0
            for j in range(last_i + 1, out.size):
                out[j] = out[j - 1] + slope_tail

        return out

    if n == 1:
        # One observed point: keep temporal continuity from fallback, shifted
        # to match the observed value at that index.
        idx_obs = int(np.where(finite)[0][0])
        v = float(out[idx_obs])
        out_fb = fallback.copy()
        out_fb = out_fb + (v - float(out_fb[idx_obs]))
        return out_fb

    return fallback.copy()


def _prefix_len_from_head(row: np.ndarray) -> int:
    """Length of contiguous finite segment starting from head index 0."""
    m = np.isfinite(row)
    if not m.any() or not bool(m[0]):
        return 0
    bad = np.where(~m)[0]
    return int(bad[0]) if bad.size else int(len(row))


def _angles_to_xy(angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.cumsum(np.cos(angles))
    y = np.cumsum(np.sin(angles))
    x = x - np.mean(x)
    y = y - np.mean(y)
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build shapes.csv from all frames (no frame dropping) and render one reconstructed-worm video."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="data/used/behaviour+neuronal activity atanas (2023)",
        help="Directory containing H5 worms.",
    )
    parser.add_argument(
        "--example_worm",
        type=str,
        default="2022-07-20-01.h5",
        help="Example worm filename (inside dataset_dir) for reconstruction video.",
    )
    parser.add_argument(
        "--d_target",
        type=int,
        default=0,
        help="Target segment count. If 0, uses mode across worms.",
    )
    parser.add_argument(
        "--n_modes",
        type=int,
        default=8,
        help="Ignored. Script always uses 5 eigenworms for reconstruction.",
    )
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument(
        "--make_video",
        action="store_true",
        help="If set, render reconstruction video for --example_worm. Default: do not render video.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="output_plots/eigenworms",
        help="Output directory for plots/video.",
    )
    parser.add_argument(
        "--write_h5_resampled",
        action="store_true",
        help="Write per-worm preprocessed arrays to each H5 under /behaviour.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    worms = sorted(dataset_dir.glob("*.h5"))
    if not worms:
        raise SystemExit(f"No .h5 files found in {dataset_dir}")

    # Pass 1: determine per-worm mode length from contiguous head-visible prefix.
    worm_info: list[tuple[Path, str, int, int, int]] = []
    d_counts: Counter[int] = Counter()
    for fp in worms:
        with h5py.File(fp, "r") as f:
            key = _find_body_angle_key(f)
            if key is None:
                continue
            arr = _ensure_TN(np.asarray(f[key], dtype=float))
        T, D = arr.shape
        pref = np.array([_prefix_len_from_head(arr[t]) for t in range(T)], dtype=int)
        pref_valid = pref[pref >= 2]
        if pref_valid.size == 0:
            d_w = max(2, D)
        else:
            c = Counter(pref_valid.tolist())
            d_w = sorted(c.items(), key=lambda x: (-x[1], x[0]))[0][0]
        worm_info.append((fp, key, T, D, int(d_w)))
        d_counts[int(d_w)] += 1

    if not worm_info:
        raise SystemExit("No usable body-angle datasets found")

    if args.d_target and args.d_target > 0:
        d_target = int(args.d_target)
    else:
        d_target = sorted(d_counts.items(), key=lambda x: (-x[1], x[0]))[0][0]

    # Pass 2: build shapes matrix with no frame dropping.
    # Per user request, apply per-worm mode rule per frame:
    #   if observed length > d_w: cut
    #   if observed length < d_w: resample observed prefix up to d_w
    # Then map all worms to global d_target.
    rows: list[np.ndarray] = []
    for fp, key, T, D, d_w in worm_info:
        with h5py.File(fp, "r") as f:
            arr = _ensure_TN(np.asarray(f[key], dtype=float))

        fallback_w = np.zeros(d_w, dtype=float)
        worm_mode_preproc = np.zeros((T, d_w), dtype=float)
        worm_dtarget = np.zeros((T, d_target), dtype=float)
        for t in range(T):
            row = np.asarray(arr[t], dtype=float)
            l_obs = _prefix_len_from_head(row)

            if l_obs >= 2:
                obs = row[:l_obs]
                obs = _fill_frame_no_drop(obs, np.zeros(max(2, l_obs), dtype=float))
                if l_obs > d_w:
                    proc_w = obs[:d_w]
                elif l_obs < d_w:
                    proc_w = _resample_1d(obs, d_w)
                else:
                    proc_w = obs.copy()
                fallback_w = proc_w
            else:
                proc_w = fallback_w.copy()

            proc_dtarget = _resample_1d(proc_w, d_target)
            worm_mode_preproc[t] = proc_w
            worm_dtarget[t] = proc_dtarget
            rows.append(proc_dtarget)

        if args.write_h5_resampled:
            with h5py.File(fp, "a") as f:
                grp = f.require_group("behaviour")
                ds_mode = "body_angle_mode_preproc"
                ds_dt = "body_angle_dtarget"
                for ds_name in (ds_mode, ds_dt):
                    if ds_name in grp:
                        del grp[ds_name]
                dsm = grp.create_dataset(ds_mode, data=worm_mode_preproc, compression="gzip", compression_opts=4)
                dsd = grp.create_dataset(ds_dt, data=worm_dtarget, compression="gzip", compression_opts=4)
                dsm.attrs["source_dataset"] = key
                dsm.attrs["d_w"] = int(d_w)
                dsd.attrs["source_dataset"] = key
                dsd.attrs["d_w"] = int(d_w)
                dsd.attrs["d_target"] = int(d_target)

    X = np.asarray(rows, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2:
        raise SystemExit("Could not build shapes matrix")

    shapes_csv = dataset_dir / "shapes.csv"
    np.savetxt(shapes_csv, X, delimiter=",")

    # PCA/eigenvectors.
    Xc = X - X.mean(axis=0, keepdims=True)
    C = (Xc.T @ Xc) / Xc.shape[0]
    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    eigvecs_path = dataset_dir / "eigenvectors.npy"
    eigvals_path = dataset_dir / "eigenvalues.npy"
    np.save(eigvecs_path, eigvecs)
    np.save(eigvals_path, eigvals)

    # Explained-variance diagnostics.
    total = float(np.sum(eigvals))
    evr = eigvals / total if total > 0 else np.zeros_like(eigvals)
    cev = np.cumsum(evr)

    def _n_for(thr: float) -> int:
        return int(np.searchsorted(cev, thr) + 1)

    n90 = _n_for(0.90)
    n95 = _n_for(0.95)
    n99 = _n_for(0.99)

    ev_csv = out_dir / "eigen_explained_variance.csv"
    with open(ev_csv, "w") as f:
        f.write("component,eigenvalue,explained_variance_ratio,cumulative_explained_variance\n")
        for i in range(len(eigvals)):
            f.write(f"{i+1},{eigvals[i]:.10g},{evr[i]:.10g},{cev[i]:.10g}\n")

    fig_ev, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    k = np.arange(1, len(eigvals) + 1)
    ax1.plot(k, evr, "o-", lw=1.4, ms=3.5, color="#1f77b4")
    ax1.set_title("Explained variance ratio")
    ax1.set_xlabel("Component")
    ax1.set_ylabel("EVR")

    ax2.plot(k, cev, "o-", lw=1.4, ms=3.5, color="#2ca02c")
    ax2.axhline(0.90, color="0.5", ls="--", lw=1)
    ax2.axhline(0.95, color="0.5", ls="--", lw=1)
    ax2.axhline(0.99, color="0.5", ls="--", lw=1)
    ax2.axvline(n90, color="#d62728", ls=":", lw=1.2)
    ax2.axvline(n95, color="#d62728", ls=":", lw=1.2)
    ax2.axvline(n99, color="#d62728", ls=":", lw=1.2)
    ax2.set_ylim(0.0, 1.01)
    ax2.set_title("Cumulative explained variance")
    ax2.set_xlabel("Component")
    ax2.set_ylabel("Cumulative")
    fig_ev.tight_layout()
    ev_plot = out_dir / "eigen_explained_variance.png"
    fig_ev.savefig(ev_plot, dpi=160)
    plt.close(fig_ev)

    # Plot all eigenworms (all found components).
    fig_all, ax_all = plt.subplots(1, 1, figsize=(12, 6))
    im = ax_all.imshow(
        eigvecs.T,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
    )
    ax_all.set_title("All eigenworms (component loadings)")
    ax_all.set_xlabel("Segment index")
    ax_all.set_ylabel("Eigenworm component")
    cbar = fig_all.colorbar(im, ax=ax_all, fraction=0.025, pad=0.02)
    cbar.set_label("Loading")
    all_eig_plot = out_dir / "eigenworms_all_components.png"
    fig_all.tight_layout()
    fig_all.savefig(all_eig_plot, dpi=170)
    plt.close(fig_all)

    # Plot first 6 eigenworms as curves across segment coordinate.
    n_show = min(6, eigvecs.shape[1])
    seg_x = np.arange(eigvecs.shape[0])
    fig6, axes6 = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    axes6 = axes6.ravel()
    for i in range(6):
        ax = axes6[i]
        if i < n_show:
            vec = eigvecs[:, i]
            ax.plot(seg_x, vec, color="#1f77b4", lw=1.8)
            ax.axhline(0.0, color="0.5", lw=0.8, ls="--")
            ax.set_title(f"EW{i+1}  (EVR={evr[i]:.3f})", fontsize=10)
        else:
            ax.axis("off")
    for ax in axes6[3:]:
        ax.set_xlabel("Segment index")
    axes6[0].set_ylabel("Loading")
    axes6[3].set_ylabel("Loading")
    fig6.suptitle("First 6 eigenworms", fontsize=13)
    fig6.tight_layout()
    first6_plot = out_dir / "eigenworms_first6.png"
    fig6.savefig(first6_plot, dpi=170)
    plt.close(fig6)

    # Save calculated amplitudes to each H5 using the same preprocessing.
    # - behaviour/eigenworms_calc_5   : first 5 component amplitudes
    # - behaviour/eigenworms_calc_6   : first 6 component amplitudes
    # - behaviour/eigenworms_calc_all : all component amplitudes
    for fp, key, T, D, d_w in worm_info:
        with h5py.File(fp, "r") as f:
            arr = _ensure_TN(np.asarray(f[key], dtype=float))

        proc_dtarget = np.zeros((T, d_target), dtype=float)
        fallback_w = np.zeros(d_w, dtype=float)
        for t in range(T):
            row = np.asarray(arr[t], dtype=float)
            l_obs = _prefix_len_from_head(row)
            if l_obs >= 2:
                obs = row[:l_obs]
                obs = _fill_frame_no_drop(obs, np.zeros(max(2, l_obs), dtype=float))
                if l_obs > d_w:
                    proc_w = obs[:d_w]
                elif l_obs < d_w:
                    proc_w = _resample_1d(obs, d_w)
                else:
                    proc_w = obs.copy()
                fallback_w = proc_w
            else:
                proc_w = fallback_w.copy()
            proc_dtarget[t] = _resample_1d(proc_w, d_target)

        proc_centered = proc_dtarget - proc_dtarget.mean(axis=1, keepdims=True)
        amps_all = proc_centered @ eigvecs
        amps_5 = amps_all[:, : min(5, amps_all.shape[1])]
        amps_6 = amps_all[:, : min(6, amps_all.shape[1])]

        with h5py.File(fp, "a") as f:
            grp = f.require_group("behaviour")
            datasets_to_write = {
                "eigenworms_calc_5": amps_5,
                "eigenworms_calc_6": amps_6,
                "eigenworms_calc_all": amps_all,
            }
            if args.write_h5_resampled:
                datasets_to_write["body_angle_dtarget"] = proc_dtarget

            for ds_name, data_arr in datasets_to_write.items():
                if ds_name in grp:
                    del grp[ds_name]
                ds = grp.create_dataset(ds_name, data=data_arr, compression="gzip", compression_opts=4)
                ds.attrs["source_dataset"] = key
                ds.attrs["d_w"] = int(d_w)
                ds.attrs["d_target"] = int(d_target)
                if ds_name == "eigenworms_calc_5":
                    ds.attrs["n_modes"] = int(amps_5.shape[1])
                if ds_name == "eigenworms_calc_6":
                    ds.attrs["n_modes"] = int(amps_6.shape[1])
                if ds_name == "eigenworms_calc_all":
                    ds.attrs["n_modes"] = int(amps_all.shape[1])

    video_path = None
    ev_global = None
    n_modes = min(5, eigvecs.shape[1])
    if args.make_video:
        # Build video for one example worm with exactly same preprocessing as shapes.
        worm_path = dataset_dir / args.example_worm
        if not worm_path.exists():
            raise SystemExit(f"Example worm not found: {worm_path}")

        with h5py.File(worm_path, "r") as f:
            key = _find_body_angle_key(f)
            if key is None:
                raise SystemExit("No body-angle dataset in example worm")
            arr = _ensure_TN(np.asarray(f[key], dtype=float))

            # Prefer exactly the saved d_target preprocessing if available.
            arr_dtarget_saved = None
            if "behaviour/body_angle_dtarget" in f:
                arr_dtarget_saved = np.asarray(f["behaviour/body_angle_dtarget"], dtype=float)

            sample_rate_hz = 1.0
            if "stage1/params" in f and "sample_rate_hz" in f["stage1/params"].attrs:
                sample_rate_hz = float(f["stage1/params"].attrs["sample_rate_hz"])
            elif "sample_rate_hz" in f.attrs:
                sample_rate_hz = float(f.attrs["sample_rate_hz"])

        T, D_native = arr.shape
        E = eigvecs[:, :n_modes]

        target_angles = np.full((T, d_target), np.nan, dtype=float)
        recon_angles = np.full((T, d_target), np.nan, dtype=float)
        coeffs = np.full((T, n_modes), np.nan, dtype=float)

        if arr_dtarget_saved is not None and arr_dtarget_saved.shape[1] == d_target:
            proc_all = arr_dtarget_saved
            for t in range(T):
                proc_shape = proc_all[t] - np.mean(proc_all[t])
                c = proc_shape @ E
                r = c @ E.T
                target_angles[t] = proc_shape
                recon_angles[t] = r
                coeffs[t] = c
        else:
            # Fallback: recompute if saved d_target array is unavailable.
            pref = np.array([_prefix_len_from_head(arr[t]) for t in range(T)], dtype=int)
            pref_valid = pref[pref >= 2]
            if pref_valid.size == 0:
                d_w_video = D_native
            else:
                c = Counter(pref_valid.tolist())
                d_w_video = sorted(c.items(), key=lambda x: (-x[1], x[0]))[0][0]

            fallback_w = np.zeros(d_w_video, dtype=float)
            for t in range(T):
                row = np.asarray(arr[t], dtype=float)
                l_obs = _prefix_len_from_head(row)
                if l_obs >= 2:
                    obs = row[:l_obs]
                    obs = _fill_frame_no_drop(obs, np.zeros(max(2, l_obs), dtype=float))
                    if l_obs > d_w_video:
                        proc_w = obs[:d_w_video]
                    elif l_obs < d_w_video:
                        proc_w = _resample_1d(obs, d_w_video)
                    else:
                        proc_w = obs.copy()
                    fallback_w = proc_w
                else:
                    proc_w = fallback_w.copy()

                proc = _resample_1d(proc_w, d_target)
                proc_shape = proc - np.mean(proc)
                c = proc_shape @ E
                r = c @ E.T
                target_angles[t] = proc_shape
                recon_angles[t] = r
                coeffs[t] = c

        xy_target = np.zeros((T, d_target, 2), dtype=float)
        xy_recon = np.zeros((T, d_target, 2), dtype=float)
        for t in range(T):
            x0, y0 = _angles_to_xy(target_angles[t])
            x1, y1 = _angles_to_xy(recon_angles[t])
            xy_target[t, :, 0], xy_target[t, :, 1] = x0, y0
            xy_recon[t, :, 0], xy_recon[t, :, 1] = x1, y1

        xmin = min(np.min(xy_target[:, :, 0]), np.min(xy_recon[:, :, 0])) - 2
        xmax = max(np.max(xy_target[:, :, 0]), np.max(xy_recon[:, :, 0])) + 2
        ymin = min(np.min(xy_target[:, :, 1]), np.min(xy_recon[:, :, 1])) - 2
        ymax = max(np.max(xy_target[:, :, 1]), np.max(xy_recon[:, :, 1])) + 2

        fig = plt.figure(figsize=(8.5, 8.5), facecolor="white")
        gs = fig.add_gridspec(3, 1, height_ratios=[4.0, 1.2, 0.45], hspace=0.35)

        ax = fig.add_subplot(gs[0])
        ax.set_aspect("equal")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.axis("off")

        line_recon, = ax.plot([], [], color="#1f77b4", lw=2.5, label=f"Recon ({n_modes} EW)")
        line_target, = ax.plot([], [], color="black", lw=1.0, alpha=0.45, label="Processed target")
        head_dot, = ax.plot([], [], "o", color="red", ms=4)
        title = ax.set_title("")
        ax.legend(frameon=False, fontsize=9, loc="upper right")

        axb = fig.add_subplot(gs[1])
        colors = plt.cm.tab10(np.linspace(0.0, 1.0, n_modes))
        bars = axb.barh(np.arange(n_modes), np.zeros(n_modes), color=colors, height=0.72)
        cmax = np.percentile(np.abs(coeffs), 99)
        if not np.isfinite(cmax) or cmax <= 0:
            cmax = 1.0
        axb.set_xlim(-cmax, cmax)
        axb.set_yticks(np.arange(n_modes), [f"EW{i+1}" for i in range(n_modes)])
        axb.axvline(0, color="0.4", lw=0.8)
        axb.invert_yaxis()
        axb.set_xlabel("Projection coefficient")

        ax_info = fig.add_subplot(gs[2])
        ax_info.axis("off")
        info_txt = ax_info.text(0.5, 0.5, "", ha="center", va="center", fontsize=10)

        dt = 1.0 / max(sample_rate_hz, 1e-12)

        def _update(t: int):
            xr, yr = xy_recon[t, :, 0], xy_recon[t, :, 1]
            xt, yt = xy_target[t, :, 0], xy_target[t, :, 1]

            line_recon.set_data(xr, yr)
            line_target.set_data(xt, yt)
            head_dot.set_data([xr[0]], [yr[0]])

            ct = coeffs[t]
            for i, b in enumerate(bars):
                b.set_width(float(ct[i]))

            den = float(np.sum(target_angles[t] ** 2))
            ev = 1.0 - float(np.sum((target_angles[t] - recon_angles[t]) ** 2)) / den if den > 0 else np.nan

            title.set_text(
                f"{worm_path.stem} | no-drop preprocessing | {n_modes} eigenworms | t={t*dt:.1f}s"
            )
            info_txt.set_text(
                f"Per-frame EV: {ev:.3f}   D_target={d_target}   total frames={T}"
            )

            return [line_recon, line_target, head_dot, title, info_txt, *bars]

        anim = FuncAnimation(fig, _update, frames=T, interval=1000 // max(1, int(args.fps)), blit=False)

        video_path = out_dir / f"{worm_path.stem}_reconstruction_{n_modes}eigenworms_nodrop.mp4"
        writer = FFMpegWriter(
            fps=int(args.fps),
            bitrate=2600,
            metadata={"title": f"{n_modes}-eigenworm reconstruction (no-drop)"},
        )
        anim.save(str(video_path), writer=writer, dpi=int(args.dpi))
        plt.close(fig)

        ev_global = 1.0 - np.sum((target_angles - recon_angles) ** 2) / max(np.sum(target_angles ** 2), 1e-12)

    print("DONE")
    print(f"worms={len(worm_info)}  d_target={d_target}  n_modes={n_modes}")
    print(f"shapes.csv: {shapes_csv}  shape={X.shape}")
    print(f"eigenvectors: {eigvecs_path}  shape={eigvecs.shape}")
    if video_path is not None and ev_global is not None:
        print(f"video: {video_path}")
        print(f"global EV ({n_modes} modes): {ev_global:.4f}")
    else:
        print("video: skipped (use --make_video to render)")
    print(f"components for 90/95/99% EV: {n90}/{n95}/{n99}")
    print(f"explained variance CSV: {ev_csv}")
    print(f"explained variance plot: {ev_plot}")
    print(f"all-eigenworms plot: {all_eig_plot}")
    if args.write_h5_resampled:
        print("saved per-worm arrays to each H5:")
        print("  /behaviour/body_angle_mode_preproc")
        print("  /behaviour/body_angle_dtarget")
    print("saved per-worm amplitudes to each H5:")
    print("  /behaviour/eigenworms_calc_5")
    print("  /behaviour/eigenworms_calc_6")
    print("  /behaviour/eigenworms_calc_all")


if __name__ == "__main__":
    main()
