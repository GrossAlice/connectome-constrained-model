"""Ridge warm-start: initialise Stage-2 lag parameters from per-neuron Ridge.

Strategy
--------
1.  For each neuron *i*, build a feature matrix identical to what the ODE
    sees through its lag buffer, and fit sklearn RidgeCV.
2.  Map the Ridge coefficients  (β, intercept) → S2 parameters
    (λ, g, α, G, I0) with λ·g = 1 so that θ ≡ β numerically.
3.  Overwrite the model parameters in-place.

The mapping (choosing λ = 0.5, g = 2.0 so λ·g = 1):

    effective coeff on u_i(t)   = (1 − λ) + λ·g·α[0]  →  α[0] = β_self − (1−λ)
    effective coeff on u_i(t−k) = λ·g·α[k]             →  α[k] = β_lag_k
    effective coeff on u_j(t−k) = λ·g·G[k,i,j]         →  G[k,i,j] = β_nbr_{j,k}
    intercept                   = λ·I0                  →  I0 = intercept / λ
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import RidgeCV


def warmstart_from_ridge(
    model,
    u: torch.Tensor,
    *,
    train_mask: torch.Tensor | None = None,
    alphas: np.ndarray | None = None,
    lambda_target: float = 0.5,
    verbose: bool = True,
) -> dict:
    """Set model lag params from per-neuron Ridge fit on *u*.

    Parameters
    ----------
    model : Stage2ModelPT
        Model whose parameters are overwritten in-place.
    u : (T, N) tensor
        Smoothed neural activity (z-scored Stage-1 posterior mean).
    train_mask : (T-1,) bool tensor, optional
        True for time-steps used in training (False = held-out).  If None,
        all frames are used.
    alphas : 1-D array, optional
        Ridge penalty grid for RidgeCV.  Default ``logspace(-3, 6, 20)``.
    lambda_target : float
        Value of λ_i to impose (default 0.5, middle of sigmoid range).
        g_i is set to 1/λ so that λ·g = 1.
    verbose : bool
        Print summary statistics.

    Returns
    -------
    info : dict   with keys 'mean_r2', 'alphas_chosen', etc.
    """
    device = next(model.parameters()).device
    u_np = u.detach().cpu().numpy().astype(np.float64)  # (T, N)
    T, N = u_np.shape
    K = model._lag_order
    if K <= 0:
        if verbose:
            print("[warmstart] lag_order=0 – nothing to do")
        return {}

    if alphas is None:
        alphas = np.logspace(-3, 6, 20)

    g_target = 1.0 / lambda_target  # so λ·g = 1

    # ── Build connectome mask (union, same as Ridge baseline) ────────
    T_e = model.T_e.detach().cpu().numpy()
    T_sv = model.T_sv.detach().cpu().numpy() if hasattr(model, "T_sv") else np.zeros((N, N))
    T_dcv = model.T_dcv.detach().cpu().numpy() if hasattr(model, "T_dcv") else np.zeros((N, N))
    # mask[i,j] = True iff j→i exists somewhere in connectome
    adj = np.abs(T_e) + np.abs(T_sv) + np.abs(T_dcv)
    mask_conn = adj.T > 0  # [post, pre]
    np.fill_diagonal(mask_conn, False)

    # ── Build lag feature buffer  (T, K, N) ──────────────────────────
    #   lag_feat[t, k, :] = u[t − 1 − k]   (zero-padded for early t)
    #   At loop step t (predicting u[t] from u[t-1]):
    #     k=0  → u[t-1]  (= Ridge's u_i(t))
    #     k=1  → u[t-2]  (= Ridge's u_i(t−1))
    #     ...
    u_pad = np.zeros((K + T, N), dtype=np.float64)
    u_pad[K:] = u_np
    lag_feat = np.stack(
        [u_pad[K - 1 - k: K - 1 - k + T] for k in range(K)], axis=1
    )  # (T, K, N)

    # Valid time range: t = K .. T-1  (need K history and 1 target ahead)
    # At step t we predict u[t] from lag_feat[t], so target = u[t].
    # But S2 loop predicts u[t] from u[t-1], and we align "features at step t"
    # with lag_feat[t].  Target = u[t].
    # Minimum valid t: when lag_feat[t, K-1] = u[t-K] has actual data → t ≥ K.
    t_start = K
    feat_all = lag_feat[t_start:]     # (T_eff, K, N)
    y_all = u_np[t_start:]           # (T_eff, N)
    T_eff = T - t_start

    # Apply train_mask (it's indexed on transitions 0..T-2,
    # our features are at steps t_start..T-1 so offset accordingly).
    if train_mask is not None:
        tm = train_mask.detach().cpu().numpy()
        # train_mask[t-1] governs the transition u[t-1]→u[t],
        # our row index r corresponds to t = t_start + r, transition idx = t-1.
        sel = tm[t_start - 1: T - 1]  # length T_eff
        feat_tr = feat_all[sel]
        y_tr = y_all[sel]
    else:
        feat_tr = feat_all
        y_tr = y_all

    # ── Per-neuron Ridge ─────────────────────────────────────────────
    alpha_init = np.zeros((K, N), dtype=np.float64)
    I0_init = np.zeros(N, dtype=np.float64)
    r2_scores = np.zeros(N)
    chosen_alphas = np.zeros(N)

    # Prepare G storage (union or per-type)
    per_type = hasattr(model, "_lag_nbr_types")
    if per_type:
        G_inits = {}
        type_masks = {}
        for prefix in model._lag_nbr_types:
            G_inits[prefix] = np.zeros((K, N, N), dtype=np.float64)
            type_masks[prefix] = getattr(model, f"_lag_nbr_mask_{prefix}").cpu().numpy()
    elif hasattr(model, "_lag_G"):
        G_init = np.zeros((K, N, N), dtype=np.float64)
    else:
        G_init = None

    for i in range(N):
        # Self features: feat_tr[:, :, i]  →  (n_train, K)
        X_self = feat_tr[:, :, i]

        # Neighbor features
        nbr_idx = np.where(mask_conn[i])[0]
        if len(nbr_idx) > 0:
            X_nbr = feat_tr[:, :, nbr_idx].reshape(feat_tr.shape[0], -1)  # (n, K*|N(i)|)
        else:
            X_nbr = np.zeros((feat_tr.shape[0], 0))

        X = np.hstack([X_self, X_nbr])
        y = y_tr[:, i]

        ridge = RidgeCV(alphas=alphas, fit_intercept=True)
        ridge.fit(X, y)

        beta = ridge.coef_
        r2_scores[i] = ridge.score(X, y)
        chosen_alphas[i] = ridge.alpha_

        # ── Map β → S2 params ───────────────────────────────────────
        # Self-lag: first K coefficients
        beta_self = beta[:K]
        # k=0 feature is u_i(t-1) = u_prev  →  effective: (1-λ) + λ·g·α[0]
        #   with λ·g = 1: α[0] = β_self[0] - (1 - λ)
        alpha_init[0, i] = beta_self[0] - (1.0 - lambda_target)
        alpha_init[1:, i] = beta_self[1:]

        # Intercept  →  λ · I0  →  I0 = intercept / λ
        I0_init[i] = ridge.intercept_ / lambda_target

        # Neighbor-lag: next K * |N(i)| coefficients
        if len(nbr_idx) > 0:
            beta_nbr = beta[K:].reshape(K, len(nbr_idx))
            if per_type:
                # Assign each edge to its first matching type
                for ki, j in enumerate(nbr_idx):
                    for prefix in model._lag_nbr_types:
                        if type_masks[prefix][i, j] > 0:
                            for kk in range(K):
                                G_inits[prefix][kk, i, j] += beta_nbr[kk, ki]
                            break  # assign to first type only
            elif G_init is not None:
                for kk in range(K):
                    G_init[kk, i, nbr_idx] = beta_nbr[kk]

    # ── Inject into model ────────────────────────────────────────────
    with torch.no_grad():
        # Lambda: set to lambda_target via inverse reparametrization
        lo, hi = model._lambda_u_lo, model._lambda_u_hi
        frac = (lambda_target - lo) / max(hi - lo, 1e-12)
        frac = np.clip(frac, 1e-6, 1 - 1e-6)
        lam_raw_val = float(np.log(frac / (1 - frac)))  # logit
        model._lambda_u_raw.fill_(lam_raw_val)

        # Input gain g
        if model._learn_input_gain:
            model._g_raw.fill_(g_target)
        else:
            # Even if not learnable, set the buffer so λ·g ≈ 1
            # (if g is a buffer it defaults to 1.0, so we need λ·g = 1
            #  → scale alpha/G/I0 by 1/(λ·1) instead)
            # Re-scale params since effective coeff = λ * 1 * θ = λ·θ
            scale = 1.0 / lambda_target  # undo the λ factor
            alpha_init *= scale
            I0_init *= scale
            if per_type:
                for prefix in G_inits:
                    G_inits[prefix] *= scale
            elif G_init is not None:
                G_init *= scale

        # I0
        model.I0.copy_(torch.tensor(I0_init, dtype=torch.float32, device=device))

        # Self-lag alpha
        model._lag_alpha.copy_(
            torch.tensor(alpha_init, dtype=torch.float32, device=device))

        # Neighbor lag G
        if per_type:
            for prefix in model._lag_nbr_types:
                param = getattr(model, f"_lag_G_{prefix}")
                mask_p = getattr(model, f"_lag_nbr_mask_{prefix}").cpu().numpy()
                G_p = G_inits[prefix] * mask_p[np.newaxis]  # re-mask
                param.copy_(torch.tensor(G_p, dtype=torch.float32, device=device))
        elif hasattr(model, "_lag_G"):
            mask_u = model._lag_nbr_mask.cpu().numpy()
            G_masked = G_init * mask_u[np.newaxis]
            model._lag_G.copy_(
                torch.tensor(G_masked, dtype=torch.float32, device=device))

    info = {
        "mean_r2": float(np.mean(r2_scores)),
        "median_r2": float(np.median(r2_scores)),
        "mean_alpha": float(np.mean(chosen_alphas)),
        "lambda_set": lambda_target,
        "g_set": g_target if model._learn_input_gain else 1.0,
    }
    if verbose:
        print(f"[warmstart] Ridge R²: mean={info['mean_r2']:.4f}  "
              f"median={info['median_r2']:.4f}  "
              f"mean α_ridge={info['mean_alpha']:.2f}")
        print(f"[warmstart] Set λ={lambda_target:.3f}, "
              f"g={info['g_set']:.3f}")
    return info
