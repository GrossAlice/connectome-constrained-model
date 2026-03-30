#!/usr/bin/env python
r"""GRU behaviour decoder benchmark with free-run rollout evaluation.

Model:
    h_t = GRU(h_{t-1}, n_t)          # hidden state carries phase memory
    b̂_t = W_out · h_t + c            # linear readout

Training:
    Free-run rollout loss via truncated BPTT.  During training the GRU
    sees GT neural input at each step but NEVER GT posture (except for
    the first 'warmup' seed frames).  This forces the hidden state to
    learn to carry the body-wave phase.

Evaluation:
    5-fold temporal CV.  Each test fold is a contiguous block of ~320
    frames (~190s).  The model is trained on the other 4 folds, then
    free-run evaluated on the held-out fold.

Baselines included for comparison:
    - Ridge linear:  b̂ = W·n_t + c   (no recurrence)

Usage:
    python -m scripts.benchmark_gru_decoder \
        --h5 "data/used/behaviour+neuronal activity atanas (2023)/2/2022-06-14-01.h5"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import h5py
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stage2.behavior_decoder_eval import (
    _log_ridge_grid,
    _ridge_cv_single_target,
    build_lagged_features_np,
)


# --------------------------------------------------------------------------- #
#  Data loading  (reuses pattern from benchmark_ar_decoder_v2)
# --------------------------------------------------------------------------- #

def load_data(h5_path: str):
    motor_file = (Path(__file__).resolve().parent.parent /
                  "data/used/masks+motor neurons/motor_neurons_with_control.txt")
    motor_names = [l.strip() for l in motor_file.read_text().splitlines()
                   if l.strip() and not l.startswith("#")]

    with h5py.File(h5_path, "r") as f:
        u = f["stage1/u_mean"][:]
        for key in ("gcamp/neuron_labels", "gcamp/neuron_names", "neuron_names"):
            if key in f:
                neuron_names = [n.decode() if isinstance(n, bytes) else n
                                for n in f[key][:]]
                break
        else:
            raise KeyError("Cannot find neuron names in h5 file")
        b = f["behaviour/eigenworms_stephens"][:]
        dt = float(f.attrs.get("dt", 0.6))

    name2idx = {n: i for i, n in enumerate(neuron_names)}
    motor_idx = [name2idx[n] for n in motor_names if n in name2idx]
    print(f"  Loaded {h5_path}")
    print(f"    T={u.shape[0]}, N_total={u.shape[1]}, "
          f"M_motor={len(motor_idx)}, K={b.shape[1]}, dt={dt:.3f}s")
    return u[:, motor_idx], b, dt


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / max(ss_tot, 1e-30)


# --------------------------------------------------------------------------- #
#  GRU decoder model
# --------------------------------------------------------------------------- #

class GRUDecoder(nn.Module):
    """GRU that maps neural input sequence → eigenworm amplitudes.

    Forward pass can operate in two modes:
      - teacher_forcing=True:  h_t = GRU(h_{t-1}, [n_t, b_{t-1}^GT])
      - teacher_forcing=False: h_t = GRU(h_{t-1}, [n_t, b̂_{t-1}])
        (free-run: feeds own predictions back)
    """

    def __init__(self, d_neural: int, K: int, hidden: int = 64,
                 n_layers: int = 1, dropout: float = 0.0,
                 use_beh_input: bool = True):
        super().__init__()
        self.K = K
        self.hidden = hidden
        self.use_beh_input = use_beh_input

        d_in = d_neural + (K if use_beh_input else 0)
        self.gru = nn.GRU(d_in, hidden, num_layers=n_layers,
                          dropout=dropout if n_layers > 1 else 0.0,
                          batch_first=True)
        self.readout = nn.Linear(hidden, K)

    def forward(self, neural: torch.Tensor, b_gt: torch.Tensor = None,
                teacher_forcing: bool = False,
                warmup: int = 2) -> torch.Tensor:
        """
        Args:
            neural: (T, d_neural)  GT neural features for every frame
            b_gt:   (T, K)        GT behaviour (used for warmup + teacher forcing)
            teacher_forcing: if True, feed GT b_{t-1} as input
            warmup: frames to seed with GT before free-running

        Returns:
            preds: (T, K)  predicted eigenworm amplitudes
        """
        T = neural.shape[0]
        device = neural.device
        h = torch.zeros(self.gru.num_layers, 1, self.hidden, device=device)
        preds = torch.zeros(T, self.K, device=device)

        # Initial behaviour (for input at t=0)
        b_prev = b_gt[0] if b_gt is not None else torch.zeros(self.K, device=device)

        for t in range(T):
            # Build input
            if self.use_beh_input:
                inp = torch.cat([neural[t], b_prev]).unsqueeze(0).unsqueeze(0)  # (1,1,d)
            else:
                inp = neural[t].unsqueeze(0).unsqueeze(0)  # (1,1,d_neural)

            out, h = self.gru(inp, h)
            b_hat = self.readout(out.squeeze(0).squeeze(0))  # (K,)
            preds[t] = b_hat

            # Decide what b_{t} to feed as input at t+1
            if t < warmup and b_gt is not None:
                b_prev = b_gt[t]
            elif teacher_forcing and b_gt is not None:
                b_prev = b_gt[t]
            else:
                b_prev = b_hat.detach()  # free-run: use own prediction

        return preds


# --------------------------------------------------------------------------- #
#  Training loop with scheduled sampling
# --------------------------------------------------------------------------- #

def train_gru(model: GRUDecoder, neural: np.ndarray, b_gt: np.ndarray,
              train_segments: list[tuple[int, int]],
              warmup: int = 2,
              epochs: int = 300, lr: float = 1e-3,
              tbptt_chunk: int = 64,
              ss_start: float = 1.0, ss_end: float = 0.0,
              device: str = "cpu"):
    """Train GRU with scheduled sampling via truncated BPTT.

    Scheduled sampling: probability of using GT b_{t-1} starts at ss_start
    and linearly anneals to ss_end over training.

    Args:
        train_segments: list of (start, end) contiguous training regions
        warmup: seed frames at the start of each segment
    """
    model = model.to(device)
    model.train()

    n_t = torch.tensor(neural, dtype=torch.float32, device=device)
    b_t = torch.tensor(b_gt, dtype=torch.float32, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_loss = float("inf")
    best_state = None
    patience = 0
    max_patience = 60

    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        ep_count = 0

        # Scheduled sampling probability
        ss_p = ss_start + (ss_end - ss_start) * min(ep / max(epochs * 0.7, 1), 1.0)

        for seg_s, seg_e in train_segments:
            seg_len = seg_e - seg_s
            if seg_len < warmup + 2:
                continue

            # Process segment in chunks for truncated BPTT
            h = torch.zeros(model.gru.num_layers, 1, model.hidden, device=device)
            b_prev = b_t[seg_s]  # GT seed

            for cs in range(seg_s, seg_e, tbptt_chunk):
                ce = min(cs + tbptt_chunk, seg_e)
                chunk_loss = torch.tensor(0.0, device=device)
                n_frames = 0

                for t in range(cs, ce):
                    # Build input
                    if model.use_beh_input:
                        inp = torch.cat([n_t[t], b_prev]).unsqueeze(0).unsqueeze(0)
                    else:
                        inp = n_t[t].unsqueeze(0).unsqueeze(0)

                    out, h = model.gru(inp, h)
                    b_hat = model.readout(out.squeeze(0).squeeze(0))

                    # Only count loss after warmup
                    if t >= seg_s + warmup:
                        chunk_loss = chunk_loss + nn.functional.mse_loss(b_hat, b_t[t])
                        n_frames += 1

                    # Scheduled sampling: GT with prob ss_p, own pred otherwise
                    if t < seg_s + warmup:
                        b_prev = b_t[t]  # always GT during warmup
                    elif np.random.random() < ss_p:
                        b_prev = b_t[t]  # teacher forcing
                    else:
                        b_prev = b_hat.detach()  # free-run

                if n_frames > 0:
                    chunk_loss = chunk_loss / n_frames
                    opt.zero_grad()
                    chunk_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    ep_loss += chunk_loss.item() * n_frames
                    ep_count += n_frames

                # Detach hidden state for truncated BPTT
                h = h.detach()
                b_prev = b_prev.detach()

        sched.step()

        if ep_count > 0:
            avg_loss = ep_loss / ep_count
            if avg_loss < best_loss - 1e-6:
                best_loss = avg_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1

            if ep % 50 == 0 or ep == epochs - 1:
                print(f"    ep {ep:3d}: loss={avg_loss:.5f}  "
                      f"ss_p={ss_p:.2f}  best={best_loss:.5f}  pat={patience}")

            if patience > max_patience:
                print(f"    early stop at ep {ep}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
#  Free-run evaluation
# --------------------------------------------------------------------------- #

def free_run_eval(model: GRUDecoder, neural: np.ndarray, b_gt: np.ndarray,
                  start: int, end: int, warmup: int = 2,
                  device: str = "cpu") -> np.ndarray:
    """Free-run the GRU on a test segment [start, end).

    Seed with GT for 'warmup' frames before start, then free-run.
    Returns predictions for [start, end).
    """
    model.eval()
    model = model.to(device)

    # We need to run the GRU from (start - warmup) to end
    # to build up hidden state during warmup
    run_start = max(0, start - warmup)

    n_t = torch.tensor(neural, dtype=torch.float32, device=device)
    b_t = torch.tensor(b_gt, dtype=torch.float32, device=device)

    with torch.no_grad():
        h = torch.zeros(model.gru.num_layers, 1, model.hidden, device=device)
        b_prev = b_t[run_start]

        preds = []
        for t in range(run_start, end):
            if model.use_beh_input:
                inp = torch.cat([n_t[t], b_prev]).unsqueeze(0).unsqueeze(0)
            else:
                inp = n_t[t].unsqueeze(0).unsqueeze(0)

            out, h = model.gru(inp, h)
            b_hat = model.readout(out.squeeze(0).squeeze(0))

            if t < start:
                # Warmup: use GT
                b_prev = b_t[t]
            else:
                # Free-run: use own prediction
                b_prev = b_hat
                preds.append(b_hat.cpu().numpy())

    return np.array(preds)  # (end - start, K)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_beh_input", action="store_true",
                        help="GRU without behaviour feedback (neural only)")
    args = parser.parse_args()

    u_motor, b, dt = load_data(args.h5)
    T, M = u_motor.shape
    K = min(6, b.shape[1])
    b = b[:, :K]

    n_folds = args.n_folds
    warmup = 8  # frames for AR-lag baseline
    gru_warmup = 10  # frames for GRU hidden-state buildup

    # Ridge baseline features
    ridge_grid = _log_ridge_grid(-3.0, 10.0, 50)
    L = 8
    X_neural = build_lagged_features_np(u_motor, L)

    # 5-fold temporal CV
    valid_len = T - warmup
    fold_size = valid_len // n_folds
    folds = []
    for i in range(n_folds):
        start = warmup + i * fold_size
        end = warmup + (i + 1) * fold_size if i < n_folds - 1 else T
        folds.append((start, end))

    # Accumulators
    ho_ridge = np.full((T, K), np.nan)
    ho_gru = np.full((T, K), np.nan)
    ho_gru_no_beh = np.full((T, K), np.nan)

    print(f"\n  Config: hidden={args.hidden}, epochs={args.epochs}, "
          f"lr={args.lr}, device={args.device}")
    print(f"  Folds: {n_folds}, fold_size≈{fold_size} frames ({fold_size*dt:.0f}s)")
    print()

    for fold_i, (ts, te) in enumerate(folds):
        test_len = te - ts
        train_mask = np.ones(T, dtype=bool)
        train_mask[:warmup] = False
        train_mask[ts:te] = False
        train_idx = np.where(train_mask)[0]

        print(f"  ═══ Fold {fold_i+1}/{n_folds}: test=[{ts}:{te}) "
              f"({test_len} frames, {test_len*dt:.0f}s) ═══")

        # ── Ridge baseline ──
        print("    Ridge: b̂=Wn+c ...")
        for j in range(K):
            fit = _ridge_cv_single_target(
                X_neural, b[:, j], train_idx, ridge_grid, n_folds)
            ho_ridge[ts:te, j] = X_neural[ts:te] @ fit["coef"] + fit["intercept"]

        # ── Build training segments (contiguous blocks outside test fold) ──
        segments = []
        if ts > warmup + gru_warmup + 2:
            segments.append((warmup, ts))
        if te + gru_warmup + 2 < T:
            segments.append((te, T))
        # If the test fold is at the start, we only have one segment after
        if not segments:
            segments.append((warmup, ts) if ts > warmup else (te, T))

        # ── GRU with behaviour feedback ──
        print(f"    GRU (h={args.hidden}, beh_input=True) ...")
        gru = GRUDecoder(M, K, hidden=args.hidden, use_beh_input=True)
        gru = train_gru(gru, u_motor, b, segments,
                        warmup=gru_warmup, epochs=args.epochs,
                        lr=args.lr, device=args.device)
        preds = free_run_eval(gru, u_motor, b, ts, te,
                              warmup=gru_warmup, device=args.device)
        ho_gru[ts:te] = preds

        # ── GRU without behaviour feedback (ablation) ──
        print(f"    GRU (h={args.hidden}, beh_input=False) ...")
        gru_nb = GRUDecoder(M, K, hidden=args.hidden, use_beh_input=False)
        gru_nb = train_gru(gru_nb, u_motor, b, segments,
                           warmup=gru_warmup, epochs=args.epochs,
                           lr=args.lr, device=args.device)
        preds_nb = free_run_eval(gru_nb, u_motor, b, ts, te,
                                 warmup=gru_warmup, device=args.device)
        ho_gru_no_beh[ts:te] = preds_nb
        print()

    # ================================================================ #
    #  Compute R²
    # ================================================================ #
    valid = np.arange(warmup, T)
    mode_names = [f"a{j+1}" for j in range(K)]

    models = {
        "b̂=Wn+c  (Ridge)": ho_ridge,
        "b̂=GRU(h,n)+Wb̂  (free-run)": ho_gru,
        "b̂=GRU(h,n)  (no beh, free-run)": ho_gru_no_beh,
    }

    results = {}
    for name, preds in models.items():
        ok = np.isfinite(preds[valid, 0])
        if ok.sum() < 10:
            results[name] = np.full(K, np.nan)
            continue
        idx = valid[ok]
        r2 = np.array([r2_score(b[idx, j], preds[idx, j]) for j in range(K)])
        results[name] = r2

    # ================================================================ #
    #  Report
    # ================================================================ #
    print()
    print("=" * 100)
    print("  GRU DECODER BENCHMARK  (5-fold temporal CV, free-run R²)")
    print("=" * 100)
    col_w = 45
    header = f"  {'Model':<{col_w}s}" + "".join(f"{m:>8s}" for m in mode_names)
    print(header)
    print("  " + "-" * (col_w + 8 * K))

    for name in models:
        vals = results.get(name, np.full(K, np.nan))
        parts = [f"  {name:<{col_w}s}"]
        for v in vals[:K]:
            parts.append(f"{v:8.3f}" if np.isfinite(v) else f"{'---':>8s}")
        print("".join(parts))

    print()

    # ================================================================ #
    #  Variance ratio (amplitude calibration)
    # ================================================================ #
    print("  Variance ratio (std_pred / std_gt):")
    header2 = f"  {'Model':<{col_w}s}" + "".join(f"{m:>8s}" for m in mode_names)
    print(header2)
    for name, preds in models.items():
        ok = np.isfinite(preds[valid, 0])
        idx = valid[ok]
        if len(idx) < 10:
            continue
        parts = [f"  {name:<{col_w}s}"]
        for j in range(K):
            ratio = np.std(preds[idx, j]) / max(np.std(b[idx, j]), 1e-10)
            parts.append(f"{ratio:8.2f}")
        print("".join(parts))
    print()

    # ================================================================ #
    #  Summary figure
    # ================================================================ #
    out_dir = Path("output_plots/eigenworms")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Panel 1: R² bar chart
    ax = axes[0]
    x = np.arange(K)
    width = 0.25
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for i, (name, vals) in enumerate(results.items()):
        ax.bar(x + i * width, np.clip(vals, -0.5, 1.0), width,
               label=name, color=colors[i % len(colors)])
    ax.set_xticks(x + width)
    ax.set_xticklabels(mode_names)
    ax.set_ylabel("R² (free-run)")
    ax.set_title("GRU Decoder Benchmark — 5-fold temporal CV")
    ax.legend(fontsize=8)
    ax.axhline(0, color="k", linewidth=0.5)

    # Panel 2: time traces for a1, a2
    ax2 = axes[1]
    t_axis = np.arange(T) * dt
    seg = slice(200, min(600, T))
    ax2.plot(t_axis[seg], b[seg, 1], "k-", linewidth=1.5, label="GT a₂", alpha=0.8)
    for name, preds in models.items():
        ok = np.isfinite(preds[seg, 1])
        t_ok = t_axis[seg][ok]
        ax2.plot(t_ok, preds[seg, 1][ok], linewidth=1, alpha=0.7,
                 label=name.split("(")[0].strip())
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("a₂")
    ax2.set_title("a₂ predictions (sample segment)")
    ax2.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    fig_path = out_dir / "gru_decoder_benchmark.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Figure saved to {fig_path}")

    # Save text summary
    txt_path = out_dir / "gru_decoder_benchmark.txt"
    with open(txt_path, "w") as f:
        f.write("GRU Decoder Benchmark: 5-fold temporal CV, free-run R²\n")
        f.write(f"h5: {args.h5}\n")
        f.write(f"hidden={args.hidden}, epochs={args.epochs}, lr={args.lr}\n\n")
        for name in models:
            vals = results.get(name, np.full(K, np.nan))
            f.write(f"{name:45s}: " + " ".join(f"{v:.3f}" for v in vals) + "\n")
    print(f"  Summary saved to {txt_path}")
    print("\n  Done.")


if __name__ == "__main__":
    main()
