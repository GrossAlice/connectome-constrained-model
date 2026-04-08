#!/usr/bin/env python
"""
MLP baseline hyperparameter sweep
===================================
Sweep hidden size, depth, dropout, weight_decay on worm 2022-08-02-01.
3-fold temporal CV, K=5 context window.
Reports one-step R² and windowed LOO R² for each configuration.
"""
import sys, os, time, json, itertools
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/agross/Downloads/connectome-constrained model/src")

H5 = "data/used/behaviour+neuronal activity atanas (2023)/2/2022-08-02-01.h5"
SAVE_ROOT = "output_plots/stage2/mlp_baseline_sweep"
os.makedirs(SAVE_ROOT, exist_ok=True)

K = 5                  # context window
N_FOLDS = 3
WINDOW_SIZE = 50       # LOO window
MAX_EPOCHS = 200
PATIENCE = 20
LOO_N = 30             # number of LOO neurons
LOO_SEED = 0
LOO_MODE = "variance"

# ──────────────────────────────────────────────────────────────────────
#  Sweep grid
# ──────────────────────────────────────────────────────────────────────
HIDDEN_SIZES = [
    (64,),
    (128,),
    (256,),
    (64, 64),
    (128, 64),
]
DROPOUTS = [0.0, 0.2, 0.5]
WEIGHT_DECAYS = [1e-5, 1e-3, 1e-2]

# Full grid would be 5×3×3 = 45 configs.  That's manageable.
CONFIGS = []
for hs, dp, wd in itertools.product(HIDDEN_SIZES, DROPOUTS, WEIGHT_DECAYS):
    tag = f"h{'_'.join(map(str,hs))}_dp{dp}_wd{wd}"
    CONFIGS.append(dict(tag=tag, hidden=hs, dropout=dp, weight_decay=wd))

print(f"MLP sweep: {len(CONFIGS)} configurations, {N_FOLDS}-fold CV, K={K}")

# ──────────────────────────────────────────────────────────────────────
#  Load data
# ──────────────────────────────────────────────────────────────────────
import h5py

h5_path = os.path.join(
    "/home/agross/Downloads/connectome-constrained model/src", H5)
with h5py.File(h5_path, "r") as f:
    u = f["stage1/u_mean"][:]          # (T, N)
    labels = [s.decode() if isinstance(s, bytes) else s
              for s in f["gcamp/neuron_labels"][:]]
T, N = u.shape
print(f"Data: T={T}, N={N}")

# Choose LOO subset: top-30 highest-variance neurons (same as retrain_loo default)
var_order = np.argsort(np.nanvar(u, axis=0))[::-1]
loo_neurons = sorted(int(i) for i in var_order[:LOO_N])
print(f"LOO neurons ({len(loo_neurons)}): {loo_neurons[:10]}...")

# ──────────────────────────────────────────────────────────────────────
#  Temporal folds
# ──────────────────────────────────────────────────────────────────────
from stage2.train import _make_temporal_folds

raw_folds = _make_temporal_folds(T, N_FOLDS)
# Each fold: (train_start, train_end, test_start, test_end)
folds = []
for te_s, te_e in raw_folds:
    if te_s == raw_folds[0][0]:
        folds.append((te_e, T, te_s, te_e))
    else:
        folds.append((0, te_s, te_s, te_e))


# ──────────────────────────────────────────────────────────────────────
#  MLP model
# ──────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, d_in, d_out, hidden=(256,), dropout=0.0):
        super().__init__()
        layers = []
        d = d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / max(ss_tot, 1e-12)


# ──────────────────────────────────────────────────────────────────────
#  Train + evaluate one config
# ──────────────────────────────────────────────────────────────────────
def run_one_config(cfg, u, folds, loo_neurons, device="cuda"):
    x = u.astype(np.float32)
    T, N = x.shape
    din = K * N
    dev = torch.device(device)

    os_pred = np.full((T, N), np.nan, np.float32)
    loo_pred = np.full((T, N), np.nan, np.float32)

    for fi, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        u_tr = x[tr_s:tr_e]

        # Build sliding-window data
        indices = np.arange(K, len(u_tr))
        X_np = np.stack([u_tr[t - K:t].flatten() for t in indices])
        Y_np = u_tr[indices]

        # Train/val split (last 15%)
        nv = max(int(X_np.shape[0] * 0.15), 1)
        nf = X_np.shape[0] - nv
        Xf = torch.tensor(X_np[:nf], dtype=torch.float32, device=dev)
        Xv = torch.tensor(X_np[nf:], dtype=torch.float32, device=dev)
        Yf = torch.tensor(Y_np[:nf], dtype=torch.float32, device=dev)
        Yv = torch.tensor(Y_np[nf:], dtype=torch.float32, device=dev)

        model = MLP(din, N, hidden=cfg["hidden"],
                     dropout=cfg["dropout"]).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3,
                                weight_decay=cfg["weight_decay"])

        bvl, bst, w = float("inf"), None, 0
        for ep in range(MAX_EPOCHS):
            model.train()
            perm = torch.randperm(nf, device=dev)
            for bs in range(0, nf, 256):
                idx = perm[bs:bs + 256]
                loss = nn.functional.mse_loss(model(Xf[idx]), Yf[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vl = nn.functional.mse_loss(model(Xv), Yv).item()
            if vl < bvl - 1e-6:
                bvl = vl
                bst = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                w = 0
            else:
                w += 1
                if w >= PATIENCE:
                    break
        if bst:
            model.load_state_dict({k: v.to(dev) for k, v in bst.items()})
        model.eval()

        # ── One-step predictions on test fold ──
        first = max(K, te_s)
        n_te = te_e - first
        if n_te > 0:
            X_te = np.stack([x[t - K:t].flatten() for t in range(first, te_e)])
            with torch.no_grad():
                for bs in range(0, n_te, 512):
                    be = min(bs + 512, n_te)
                    ct = torch.tensor(X_te[bs:be], dtype=torch.float32, device=dev)
                    os_pred[first + bs:first + be] = model(ct).cpu().numpy()

        # ── Windowed LOO ──
        T_te = te_e - te_s
        for ni in loo_neurons:
            pred_arr = np.full(T_te, np.nan, np.float32)
            for s in range(0, T_te, WINDOW_SIZE):
                e = min(s + WINDOW_SIZE, T_te)
                pred_arr[s] = x[te_s + s, ni]
                for t_loc in range(s, e - 1):
                    t_abs = te_s + t_loc
                    ctx_start = t_abs - K + 1
                    if ctx_start < 0:
                        continue
                    ctx = x[ctx_start:t_abs + 1].copy()
                    for k in range(K):
                        pa_idx = t_loc - (K - 1) + k
                        if (pa_idx >= s and pa_idx <= t_loc
                                and np.isfinite(pred_arr[pa_idx])):
                            ctx[k, ni] = pred_arr[pa_idx]
                    ctx_flat = ctx.flatten()
                    with torch.no_grad():
                        ct = torch.tensor(ctx_flat, dtype=torch.float32,
                                          device=dev).unsqueeze(0)
                        pred_arr[t_loc + 1] = model(ct).cpu().numpy()[0, ni]
            loo_pred[te_s:te_e, ni] = pred_arr

        del model, opt
        torch.cuda.empty_cache()

    # ── Compute R² ──
    onestep_r2 = np.full(N, np.nan)
    for i in range(N):
        m = np.isfinite(os_pred[:, i])
        if m.sum() > 3:
            onestep_r2[i] = _r2(u[m, i], os_pred[m, i])

    loo_r2 = np.full(len(loo_neurons), np.nan)
    for ki, ni in enumerate(loo_neurons):
        m = np.isfinite(loo_pred[:, ni])
        if m.sum() > 3:
            loo_r2[ki] = _r2(u[m, ni], loo_pred[m, ni])

    # Count parameters
    tmp = MLP(din, N, hidden=cfg["hidden"], dropout=cfg["dropout"])
    n_params = sum(p.numel() for p in tmp.parameters())

    return {
        "onestep_r2_mean": float(np.nanmean(onestep_r2)),
        "onestep_r2_median": float(np.nanmedian(onestep_r2)),
        "loo_r2_mean": float(np.nanmean(loo_r2)),
        "loo_r2_median": float(np.nanmedian(loo_r2)),
        "n_params": n_params,
    }


# ──────────────────────────────────────────────────────────────────────
#  Main sweep
# ──────────────────────────────────────────────────────────────────────
results = []
for ci, cfg in enumerate(CONFIGS):
    tag = cfg["tag"]
    result_path = os.path.join(SAVE_ROOT, f"{tag}.json")

    # Skip if done
    if os.path.exists(result_path):
        print(f"[{ci+1}/{len(CONFIGS)}] SKIP {tag}")
        with open(result_path) as f:
            row = json.load(f)
        results.append(row)
        continue

    print(f"\n[{ci+1}/{len(CONFIGS)}] {tag}  "
          f"hidden={cfg['hidden']} drop={cfg['dropout']} wd={cfg['weight_decay']}",
          flush=True)

    t0 = time.time()
    metrics = run_one_config(cfg, u, folds, loo_neurons)
    elapsed = time.time() - t0

    row = {**cfg, **metrics, "elapsed_sec": round(elapsed, 1)}
    row["hidden"] = list(row["hidden"])  # JSON-serializable
    results.append(row)

    with open(result_path, "w") as f:
        json.dump(row, f, indent=2)

    print(f"  1step={metrics['onestep_r2_mean']:.4f}/{metrics['onestep_r2_median']:.4f}  "
          f"LOO={metrics['loo_r2_mean']:.4f}/{metrics['loo_r2_median']:.4f}  "
          f"params={metrics['n_params']:,}  time={elapsed:.0f}s")

# ──────────────────────────────────────────────────────────────────────
#  Summary table
# ──────────────────────────────────────────────────────────────────────
print("\n\n" + "=" * 100)
print("MLP BASELINE SWEEP RESULTS")
print("=" * 100)
header = (f"{'Config':<35s} {'Params':>8s} {'1step_mean':>10s} "
          f"{'1step_med':>10s} {'LOO_mean':>10s} {'LOO_med':>10s} {'Time':>6s}")
print(header)
print("-" * len(header))

# Sort by LOO mean R² descending
results_sorted = sorted(results, key=lambda r: r.get("loo_r2_mean", -999),
                         reverse=True)
for row in results_sorted:
    print(f"{row['tag']:<35s} {row['n_params']:>8,} "
          f"{row['onestep_r2_mean']:>10.4f} {row['onestep_r2_median']:>10.4f} "
          f"{row['loo_r2_mean']:>10.4f} {row['loo_r2_median']:>10.4f} "
          f"{row['elapsed_sec']:>6.0f}")

with open(os.path.join(SAVE_ROOT, "sweep_summary.json"), "w") as f:
    json.dump(results_sorted, f, indent=2)

# Print the winner
best = results_sorted[0]
print(f"\n🏆 BEST: {best['tag']}")
print(f"   hidden={best['hidden']}, dropout={best['dropout']}, "
      f"weight_decay={best['weight_decay']}")
print(f"   LOO R² mean={best['loo_r2_mean']:.4f}, "
      f"one-step R² mean={best['onestep_r2_mean']:.4f}, "
      f"params={best['n_params']:,}")
print(f"\nSaved to {SAVE_ROOT}/sweep_summary.json")
