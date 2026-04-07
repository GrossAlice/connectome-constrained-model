#!/usr/bin/env python3
"""
Test whether synaptic-vesicle (sv) connectivity alone is sufficient to predict
neuronal activity and posture, or whether sparsity limits what the model can do.

Conditions (all strict-causal, 1-second neuronal lag):
─────────────────────────────────────────────────────────────────────

  Neuronal prediction (per-neuron LOO, predict u_i(t) from other neurons):
    1. full          — all other observed neurons (N−1) as features
    2. sv_only       — only sv-connected presynaptic partners
    3. sv+dcv        — sv + dcv connected partners
    4. random_sparse — random subset matching sv in-degree (control)

  Behaviour prediction (predict 6 eigenworm amplitudes from neurons):
    5. full_beh      — all neurons + 1 beh snapshot (0.6 s)
    6. sv_motor_beh  — only sv-connected-to-motor neurons + 1 beh snapshot

Models: Ridge + MLP for all conditions.
CV: 5-fold contiguous temporal.

Usage
-----
  python -m scripts.masked_neuron.connectivity_test \
      --h5 "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5" \
      --device cuda
"""
from __future__ import annotations

import argparse, json, sys, time, csv
from pathlib import Path

import numpy as np
import h5py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent.parent
_ALPHAS = np.logspace(-4, 6, 30)

sys.path.insert(0, str(_ROOT))

from scripts.masked_neuron.masked_neuron_prediction import (
    _make_folds, _inner_split, _zscore, _make_mlp, _train_mlp, _r2,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Data & connectivity
# ═══════════════════════════════════════════════════════════════════════════════

def _load(h5_path: str, beh_modes: int = 6):
    """Load deconvolved activity, behaviour, labels."""
    with h5py.File(h5_path, "r") as f:
        u = np.array(f["stage1/u_mean"], dtype=np.float32)
        labels_raw = f["gcamp/neuron_labels"][:]
        labels = [x.decode() if isinstance(x, bytes) else str(x)
                  for x in labels_raw]
        beh = np.array(f["behaviour/eigenworms_stephens"][:, :beh_modes],
                       dtype=np.float32)
        dt = 0.6
        if "timing/timestamp_confocal" in f:
            ts = f["timing/timestamp_confocal"][:]
            if len(ts) > 1:
                dt = float(np.median(np.diff(ts)))
    return u, beh, labels, dt


def _load_connectivity():
    """Load atlas-space connectivity matrices and neuron name list."""
    base = _ROOT / "data" / "used" / "masks+motor neurons"
    atlas = list(np.load(base / "neuron_names.npy", allow_pickle=True))
    T_sv = np.load(base / "T_sv.npy")
    T_dcv = np.load(base / "T_dcv.npy")
    T_e = np.load(base / "T_e.npy")
    motor_file = base / "motor_neurons_with_control.txt"
    if motor_file.exists():
        motor_set = {l.strip() for l in motor_file.read_text().splitlines()
                     if l.strip()}
    else:
        motor_set = set()
    return atlas, T_sv, T_dcv, T_e, motor_set


def _obs_connectivity(labels, atlas, T_sv, T_dcv):
    """Map observed neurons to atlas indices, extract observed-space masks.

    Returns:
        sv_mask:  (N, N) bool — sv_mask[i,j] means j→i sv synapse exists
        dcv_mask: (N, N) bool
    """
    N = len(labels)
    idx = []
    for lb in labels:
        if lb in atlas:
            idx.append(atlas.index(lb))
        else:
            idx.append(-1)

    sv_mask = np.zeros((N, N), dtype=bool)
    dcv_mask = np.zeros((N, N), dtype=bool)
    for i in range(N):
        for j in range(N):
            if idx[i] < 0 or idx[j] < 0:
                continue
            if T_sv[idx[i], idx[j]] > 0:
                sv_mask[i, j] = True
            if T_dcv[idx[i], idx[j]] > 0:
                dcv_mask[i, j] = True
    return sv_mask, dcv_mask


# ═══════════════════════════════════════════════════════════════════════════════
#  Feature builders (strict causal)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_lagged(u, lag_frames, feat_mask):
    """Build strictly-causal feature matrix.

    Parameters
    ----------
    u : (T, N)
    lag_frames : int  (number of past frames to include)
    feat_mask : (N,) bool  — which neurons to include as features

    Returns (X, y_start_idx) where X is (T, lag_frames * sum(feat_mask)).
    Rows 0..lag_frames−1 have incomplete history and should be excluded.
    """
    T, N = u.shape
    N_feat = int(feat_mask.sum())
    blocks = []
    for lag in range(1, lag_frames + 1):
        col = np.zeros((T, N_feat), dtype=np.float32)
        if lag < T:
            col[lag:] = u[:-lag][:, feat_mask]
        blocks.append(col)
    return np.concatenate(blocks, axis=1), lag_frames


def _build_beh_snapshot(beh, lag_fr):
    """Single behaviour snapshot at lag_fr frames in the past."""
    T, K = beh.shape
    s = np.zeros_like(beh)
    if 0 < lag_fr < T:
        s[lag_fr:] = beh[:-lag_fr]
    return s


# ═══════════════════════════════════════════════════════════════════════════════
#  Evaluation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _ridge_cv(X_tr, y_tr, X_te):
    from sklearn.linear_model import RidgeCV
    ridge = RidgeCV(alphas=_ALPHAS, fit_intercept=True)
    ridge.fit(X_tr, y_tr)
    return ridge.predict(X_te)


def _mlp_cv(X_tr, y_tr, X_te, *, out_dim=1, device="cpu",
            hidden=128, dropout=0.1, epochs=200, patience=20,
            lr=1e-3, batch_size=64, weight_decay=1e-4, seed=0):
    import torch
    dev = torch.device(device)
    to_t = lambda a: torch.from_numpy(a).to(dev)

    # inner split for early stopping
    n_va = max(1, int(len(X_tr) * 0.2))
    Xtr_raw, Xva_raw = X_tr[:-n_va], X_tr[-n_va:]
    Ytr_raw, Yva_raw = y_tr[:-n_va], y_tr[-n_va:]

    mu_x, std_x = _zscore(Xtr_raw)
    mu_y = Ytr_raw.mean(0)
    std_y = np.maximum(Ytr_raw.std(0), 1e-8)

    Xtr_z = to_t(((Xtr_raw - mu_x) / std_x).astype(np.float32))
    Ytr_z = to_t(((Ytr_raw - mu_y) / std_y).astype(np.float32))
    Xva_z = to_t(((Xva_raw - mu_x) / std_x).astype(np.float32))
    Yva_z = to_t(((Yva_raw - mu_y) / std_y).astype(np.float32))

    in_dim = Xtr_z.shape[1]
    torch.manual_seed(seed)
    net = _make_mlp(in_dim, hidden, out_dim, dropout).to(dev)
    _train_mlp(
        net, Xtr_z, Ytr_z, Xva_z, Yva_z,
        epochs=epochs, lr=lr, batch_size=batch_size,
        weight_decay=weight_decay, patience=patience, device=dev,
    )

    Xte_z = to_t(((X_te - mu_x) / std_x).astype(np.float32))
    net.eval()
    with torch.no_grad():
        pred_z = net(Xte_z).cpu().numpy()
    return (pred_z * std_y + mu_y).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-neuron prediction with different connectivity masks
# ═══════════════════════════════════════════════════════════════════════════════

def _run_neuron_condition(u, lag_frames, warmup, feat_mask_per_neuron,
                          n_folds, device, condition_name, labels):
    """Run ridge + MLP for each neuron under a given connectivity mask.

    Parameters
    ----------
    feat_mask_per_neuron : dict[int, np.ndarray(bool)]
        For each target neuron index → bool mask of which neurons to use.
    """
    T, N = u.shape
    folds = _make_folds(T - warmup, n_folds)
    # Shift folds to account for warmup
    folds = [(tr + warmup, te + warmup) for tr, te in folds]

    ridge_r2 = []
    mlp_r2 = []

    for ni, idx in enumerate(sorted(feat_mask_per_neuron.keys())):
        fm = feat_mask_per_neuron[idx]
        n_feat = int(fm.sum())
        if n_feat == 0:
            ridge_r2.append(float("nan"))
            mlp_r2.append(float("nan"))
            continue

        X, _ = _build_lagged(u, lag_frames, fm)
        y = u[:, idx]

        pred_ridge = np.zeros(T, dtype=np.float32)
        pred_mlp = np.zeros(T, dtype=np.float32)

        for tr_idx, te_idx in folds:
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            pred_ridge[te_idx] = _ridge_cv(X_tr, y_tr, X_te).ravel()
            pred_mlp[te_idx] = _mlp_cv(
                X_tr, y_tr.reshape(-1, 1), X_te,
                out_dim=1, device=device, seed=ni,
            ).ravel()

        valid = np.concatenate([te for _, te in folds])
        ridge_r2.append(_r2(y[valid], pred_ridge[valid]))
        mlp_r2.append(_r2(y[valid], pred_mlp[valid]))

        if (ni + 1) % 20 == 0 or ni == 0 or (ni + 1) == len(feat_mask_per_neuron):
            print(f"      {ni+1}/{len(feat_mask_per_neuron)} "
                  f"{labels[idx]:10s} nfeat={n_feat:4d}  "
                  f"ridge={ridge_r2[-1]:.3f}  mlp={mlp_r2[-1]:.3f}")

    return ridge_r2, mlp_r2


def _run_behaviour_condition(u, beh, lag_frames, beh_lag_frames, warmup,
                             feat_mask, n_folds, device, condition_name):
    """Predict behaviour from a subset of neurons + 1 beh snapshot."""
    T, K = beh.shape
    folds = _make_folds(T - warmup, n_folds)
    folds = [(tr + warmup, te + warmup) for tr, te in folds]

    X_neuro, _ = _build_lagged(u, lag_frames, feat_mask)
    X_beh = _build_beh_snapshot(beh, beh_lag_frames)
    X = np.concatenate([X_neuro, X_beh], axis=1)

    pred_ridge = np.full((T, K), np.nan, dtype=np.float32)
    pred_mlp = np.full((T, K), np.nan, dtype=np.float32)

    for tr_idx, te_idx in folds:
        X_tr, X_te = X[tr_idx], X[te_idx]
        b_tr = beh[tr_idx]

        # Ridge per mode
        for j in range(K):
            pred_ridge[te_idx, j] = _ridge_cv(X_tr, b_tr[:, j], X_te).ravel()

        # MLP all modes at once
        pred_mlp[te_idx] = _mlp_cv(
            X_tr, b_tr, X_te, out_dim=K, device=device,
        )

    valid = np.concatenate([te for _, te in folds])
    ridge_r2 = [_r2(beh[valid, j], pred_ridge[valid, j]) for j in range(K)]
    mlp_r2 = [_r2(beh[valid, j], pred_mlp[valid, j]) for j in range(K)]
    return ridge_r2, mlp_r2


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--out_dir",
                    default="output_plots/masked_neuron_prediction/connectivity_test")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lag_sec", type=float, default=1.0,
                    help="Neuronal lag in seconds (strict causal)")
    ap.add_argument("--beh_lag_sec", type=float, default=0.6,
                    help="Behaviour snapshot lag in seconds")
    ap.add_argument("--beh_modes", type=int, default=6)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)

    # ── Load data ────────────────────────────────────────────────
    u, beh, labels, dt = _load(args.h5, args.beh_modes)
    T, N = u.shape
    K = beh.shape[1]
    lag_frames = max(1, round(args.lag_sec / dt))
    beh_lag_frames = max(1, round(args.beh_lag_sec / dt))
    warmup = max(lag_frames, beh_lag_frames)

    print(f"\n  Worm: T={T}, N={N}, K={K}, dt={dt:.3f}s")
    print(f"  Neuronal lag: {args.lag_sec}s → {lag_frames} frames")
    print(f"  Beh lag:      {args.beh_lag_sec}s → {beh_lag_frames} frames")
    print(f"  Warmup:       {warmup} frames ({warmup*dt:.1f}s)")

    # ── Load connectivity ────────────────────────────────────────
    atlas, T_sv, T_dcv, T_e, motor_set = _load_connectivity()
    sv_mask, dcv_mask = _obs_connectivity(labels, atlas, T_sv, T_dcv)
    combined_mask = sv_mask | dcv_mask

    sv_in = sv_mask.sum(axis=1)   # in-degree per neuron
    dcv_in = dcv_mask.sum(axis=1)
    comb_in = combined_mask.sum(axis=1)

    print(f"\n  Connectivity (observed {N}×{N}):")
    print(f"    sv:      nnz={sv_mask.sum()}, density={sv_mask.sum()/(N*N):.3f}, "
          f"mean in-deg={sv_in.mean():.1f}")
    print(f"    dcv:     nnz={dcv_mask.sum()}, density={dcv_mask.sum()/(N*N):.3f}, "
          f"mean in-deg={dcv_in.mean():.1f}")
    print(f"    sv+dcv:  nnz={combined_mask.sum()}, "
          f"mean in-deg={comb_in.mean():.1f}")

    # ── Motor neuron indices (observed) ──────────────────────────
    motor_idx = [i for i, lb in enumerate(labels) if lb in motor_set]
    print(f"  Motor neurons observed: {len(motor_idx)}/{N}")

    records = []

    # ══════════════════════════════════════════════════════════════
    #  PART A: Per-neuron prediction with different masks
    # ══════════════════════════════════════════════════════════════

    conditions = {}

    # 1. Full connectivity (all N−1 others)
    full_masks = {}
    for i in range(N):
        m = np.ones(N, dtype=bool)
        m[i] = False
        full_masks[i] = m
    conditions["full"] = full_masks

    # 2. sv-only
    sv_masks = {}
    for i in range(N):
        m = sv_mask[i].copy()
        m[i] = False  # exclude self
        sv_masks[i] = m
    conditions["sv_only"] = sv_masks

    # 3. sv+dcv
    svdcv_masks = {}
    for i in range(N):
        m = combined_mask[i].copy()
        m[i] = False
        svdcv_masks[i] = m
    conditions["sv_dcv"] = svdcv_masks

    # 4. Random sparse (match sv in-degree per neuron)
    rng = np.random.RandomState(args.seed)
    rand_masks = {}
    for i in range(N):
        deg = int(sv_in[i])
        if deg == 0:
            rand_masks[i] = np.zeros(N, dtype=bool)
            continue
        others = [j for j in range(N) if j != i]
        chosen = rng.choice(others, size=min(deg, len(others)), replace=False)
        m = np.zeros(N, dtype=bool)
        m[chosen] = True
        rand_masks[i] = m
    conditions["random_sparse"] = rand_masks

    for cond_name, masks in conditions.items():
        n_feat_list = [int(m.sum()) for m in masks.values()]
        print(f"\n{'='*60}")
        print(f"  NEURON PREDICTION: {cond_name}")
        print(f"  Features/neuron: mean={np.mean(n_feat_list):.1f}, "
              f"median={np.median(n_feat_list):.0f}, "
              f"range=[{min(n_feat_list)}, {max(n_feat_list)}]")
        print(f"  Total features per neuron = n_presynaptic × {lag_frames} lags")
        print(f"{'='*60}")

        t0 = time.time()
        r2_ridge, r2_mlp = _run_neuron_condition(
            u, lag_frames, warmup, masks,
            args.n_folds, args.device, cond_name, labels,
        )
        elapsed = time.time() - t0

        r_arr = np.array(r2_ridge)
        m_arr = np.array(r2_mlp)
        valid_r = r_arr[np.isfinite(r_arr)]
        valid_m = m_arr[np.isfinite(m_arr)]

        print(f"\n  {cond_name} summary ({elapsed:.0f}s):")
        if len(valid_r) > 0:
            print(f"    Ridge: mean={valid_r.mean():.4f}, "
                  f"median={np.median(valid_r):.4f}, "
                  f">0: {(valid_r>0).sum()}/{len(valid_r)}")
        if len(valid_m) > 0:
            print(f"    MLP:   mean={valid_m.mean():.4f}, "
                  f"median={np.median(valid_m):.4f}, "
                  f">0: {(valid_m>0).sum()}/{len(valid_m)}")

        for ni, idx in enumerate(sorted(masks.keys())):
            records.append({
                "test": "neuron",
                "condition": cond_name,
                "neuron": labels[idx],
                "neuron_idx": idx,
                "n_features": int(masks[idx].sum()) * lag_frames,
                "model": "ridge",
                "r2": r2_ridge[ni],
            })
            records.append({
                "test": "neuron",
                "condition": cond_name,
                "neuron": labels[idx],
                "neuron_idx": idx,
                "n_features": int(masks[idx].sum()) * lag_frames,
                "model": "mlp",
                "r2": r2_mlp[ni],
            })

    # ══════════════════════════════════════════════════════════════
    #  PART B: Behaviour prediction with different neuron subsets
    # ══════════════════════════════════════════════════════════════

    beh_conditions = {}

    # 5. Full neurons + beh snapshot
    full_beh_mask = np.ones(N, dtype=bool)
    beh_conditions["full_beh"] = full_beh_mask

    # 6. Only motor neurons + beh
    motor_mask = np.zeros(N, dtype=bool)
    for mi in motor_idx:
        motor_mask[mi] = True
    beh_conditions["motor_only_beh"] = motor_mask

    # 7. Neurons that are sv-connected TO any motor neuron + beh
    sv_to_motor = np.zeros(N, dtype=bool)
    for mi in motor_idx:
        sv_to_motor |= sv_mask[mi]  # neurons projecting to motor
    sv_to_motor_mask = sv_to_motor | motor_mask
    beh_conditions["sv_motor_beh"] = sv_to_motor_mask

    for cond_name, mask in beh_conditions.items():
        n_neur = int(mask.sum())
        n_total = n_neur * lag_frames + K  # neuro feats + beh snapshot
        print(f"\n{'='*60}")
        print(f"  BEHAVIOUR PREDICTION: {cond_name}")
        print(f"  Using {n_neur}/{N} neurons × {lag_frames} lags "
              f"+ {K} beh snapshot = {n_total} features")
        print(f"{'='*60}")

        t0 = time.time()
        r2_ridge, r2_mlp = _run_behaviour_condition(
            u, beh, lag_frames, beh_lag_frames, warmup,
            mask, args.n_folds, args.device, cond_name,
        )
        elapsed = time.time() - t0

        r_arr = np.array(r2_ridge)
        m_arr = np.array(r2_mlp)
        print(f"\n  {cond_name} ({elapsed:.0f}s):")
        print(f"    Ridge per mode: {['%.3f' % x for x in r_arr]}")
        print(f"    MLP   per mode: {['%.3f' % x for x in m_arr]}")
        print(f"    Ridge mean: {r_arr.mean():.4f}")
        print(f"    MLP   mean: {m_arr.mean():.4f}")

        for j in range(K):
            records.append({
                "test": "behaviour",
                "condition": cond_name,
                "beh_mode": j,
                "n_neurons": n_neur,
                "n_features": n_total,
                "model": "ridge",
                "r2": r2_ridge[j],
            })
            records.append({
                "test": "behaviour",
                "condition": cond_name,
                "beh_mode": j,
                "n_neurons": n_neur,
                "n_features": n_total,
                "model": "mlp",
                "r2": r2_mlp[j],
            })

    # ── Save CSV ─────────────────────────────────────────────────
    csv_path = out / "connectivity_test_results.csv"
    if records:
        keys = list(records[0].keys())
        all_keys = set()
        for r in records:
            all_keys |= r.keys()
        keys = sorted(all_keys)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(records)
        print(f"\nSaved {len(records)} records → {csv_path}")

    # ── Save config ──────────────────────────────────────────────
    cfg = {
        "h5": args.h5,
        "lag_sec": args.lag_sec,
        "beh_lag_sec": args.beh_lag_sec,
        "beh_modes": args.beh_modes,
        "lag_frames": lag_frames,
        "beh_lag_frames": beh_lag_frames,
        "T": T, "N": N, "K": K, "dt": dt,
        "n_folds": args.n_folds,
        "seed": args.seed,
    }
    with open(out / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # ══════════════════════════════════════════════════════════════
    #  Plots
    # ══════════════════════════════════════════════════════════════
    _make_plots(records, labels, out)
    print(f"\nAll outputs → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def _make_plots(records, labels, out):
    plt.rcParams.update({
        "figure.dpi": 140, "font.size": 10, "axes.titlesize": 12,
        "axes.labelsize": 11, "legend.fontsize": 9,
        "figure.facecolor": "white",
    })

    import pandas as pd
    df = pd.DataFrame(records)

    # ── Plot 1: Neuron prediction — box/violin by condition ──────
    neuron_df = df[df["test"] == "neuron"].copy()
    if not neuron_df.empty:
        cond_order = ["full", "sv_dcv", "sv_only", "random_sparse"]
        cond_order = [c for c in cond_order if c in neuron_df["condition"].unique()]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        for ax, model in zip(axes, ["ridge", "mlp"]):
            sub = neuron_df[neuron_df["model"] == model]
            data = [sub[sub["condition"] == c]["r2"].dropna().values
                    for c in cond_order]
            medians = [np.median(d) if len(d) else 0 for d in data]

            bp = ax.boxplot(data, labels=cond_order, patch_artist=True,
                            showfliers=False, widths=0.6)
            colors = ["#3498db", "#2ecc71", "#e67e22", "#95a5a6"]
            for patch, color in zip(bp["boxes"], colors[:len(data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            for i, d in enumerate(data):
                jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(d))
                ax.scatter(np.full(len(d), i + 1) + jitter, d,
                           alpha=0.3, s=10, color="k", zorder=3)

            ax.axhline(0, color="gray", ls="--", lw=0.8)
            ax.set_ylabel("R²")
            ax.set_title(f"{model.upper()} — per-neuron prediction")
            for i, med in enumerate(medians):
                ax.text(i + 1, ax.get_ylim()[1] * 0.95, f"med={med:.3f}",
                        ha="center", va="top", fontsize=8, color=colors[i])

        fig.suptitle("Neuronal prediction: connectivity mask comparison\n"
                     f"(1s lag, strict causal, 5-fold CV)", y=1.02)
        fig.tight_layout()
        fig.savefig(out / "neuron_box.png", bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out / 'neuron_box.png'}")

    # ── Plot 2: Neuron prediction — scatter sv_only vs full ──────
    if not neuron_df.empty and "sv_only" in neuron_df["condition"].unique():
        for model in ["ridge", "mlp"]:
            sub_full = neuron_df[(neuron_df["model"] == model) &
                                (neuron_df["condition"] == "full")]
            sub_sv = neuron_df[(neuron_df["model"] == model) &
                               (neuron_df["condition"] == "sv_only")]
            if sub_full.empty or sub_sv.empty:
                continue

            merged = sub_full[["neuron_idx", "r2"]].merge(
                sub_sv[["neuron_idx", "r2"]],
                on="neuron_idx", suffixes=("_full", "_sv"),
            )

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(merged["r2_full"], merged["r2_sv"], alpha=0.6, s=20)
            lims = [min(merged["r2_full"].min(), merged["r2_sv"].min()) - 0.05,
                    max(merged["r2_full"].max(), merged["r2_sv"].max()) + 0.05]
            ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5)
            ax.set_xlabel(f"R² (full, N−1 neurons)")
            ax.set_ylabel(f"R² (sv-only neighbours)")
            ax.set_title(f"{model.upper()}: sv-only vs full connectivity")
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            fig.tight_layout()
            fig.savefig(out / f"scatter_sv_vs_full_{model}.png",
                        bbox_inches="tight")
            plt.close(fig)
            print(f"  saved {out / f'scatter_sv_vs_full_{model}.png'}")

    # ── Plot 3: Behaviour prediction — grouped bar chart ─────────
    beh_df = df[df["test"] == "behaviour"].copy()
    if not beh_df.empty:
        cond_order = ["full_beh", "sv_motor_beh", "motor_only_beh"]
        cond_order = [c for c in cond_order if c in beh_df["condition"].unique()]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        for ax, model in zip(axes, ["ridge", "mlp"]):
            sub = beh_df[beh_df["model"] == model]
            n_modes = sub["beh_mode"].nunique()
            x = np.arange(n_modes)
            width = 0.8 / len(cond_order)
            colors = ["#3498db", "#e67e22", "#e74c3c"]

            for ci, cond in enumerate(cond_order):
                vals = sub[sub["condition"] == cond].sort_values("beh_mode")["r2"].values
                ax.bar(x + ci * width, vals, width, label=cond,
                       color=colors[ci % len(colors)], alpha=0.8)

            ax.set_xticks(x + width * (len(cond_order) - 1) / 2)
            ax.set_xticklabels([f"mode {j}" for j in range(n_modes)])
            ax.set_ylabel("R²")
            ax.set_title(f"{model.upper()} — behaviour prediction")
            ax.legend(fontsize=8)
            ax.axhline(0, color="gray", ls="--", lw=0.8)

        fig.suptitle("Behaviour decoding: neuron subset comparison\n"
                     f"(1s neuro lag + 0.6s beh snapshot, strict causal)",
                     y=1.02)
        fig.tight_layout()
        fig.savefig(out / "behaviour_bar.png", bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out / 'behaviour_bar.png'}")

    # ── Plot 4: R² vs sv in-degree ───────────────────────────────
    if not neuron_df.empty and "sv_only" in neuron_df["condition"].unique():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, model in zip(axes, ["ridge", "mlp"]):
            sub = neuron_df[(neuron_df["model"] == model) &
                            (neuron_df["condition"] == "sv_only")]
            if sub.empty:
                continue
            # n_features = n_presynaptic × lag_frames
            in_deg = sub["n_features"].values / max(1, lag_frames)
            ax.scatter(in_deg, sub["r2"].values, alpha=0.6, s=20)
            ax.set_xlabel("sv in-degree")
            ax.set_ylabel("R² (sv-only)")
            ax.set_title(f"{model.upper()}: R² vs sv in-degree")
            ax.axhline(0, color="gray", ls="--", lw=0.8)
            # Trend line
            valid = np.isfinite(sub["r2"].values) & np.isfinite(in_deg)
            if valid.sum() > 5:
                z = np.polyfit(in_deg[valid], sub["r2"].values[valid], 1)
                p = np.poly1d(z)
                xs = np.linspace(in_deg[valid].min(), in_deg[valid].max(), 50)
                ax.plot(xs, p(xs), "r--", lw=1, alpha=0.7,
                        label=f"slope={z[0]:.4f}")
                ax.legend(fontsize=8)

        fig.suptitle("Does sv in-degree predict predictability?", y=1.02)
        fig.tight_layout()
        fig.savefig(out / "r2_vs_indegree.png", bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out / 'r2_vs_indegree.png'}")


if __name__ == "__main__":
    main()
