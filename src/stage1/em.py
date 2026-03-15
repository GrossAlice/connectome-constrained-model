from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple

from .config import Stage1Config

__all__ = ["fit_stage1_all_neurons", "kalman_smoother_pairwise"]

_LOG2PI = float(np.log(2.0 * np.pi))

# ── Utilities ──

def robust_sigma_y(y_centered: np.ndarray) -> float:
    """Estimate observation noise std from differenced trace (robust to slow drift)."""
    dy = np.diff(y_centered)
    s = float(np.nanstd(dy) / np.sqrt(2.0))
    if not np.isfinite(s) or s <= 0:
        s = float(np.nanstd(y_centered))
    if not np.isfinite(s) or s <= 0:
        s = 1.0
    return s


def clip_pos(x: float, floor: float = 1e-6) -> float:
    """Clip to a positive floor, handling NaN/Inf."""
    if not np.isfinite(x):
        return floor
    return float(max(x, floor))


def build_A(rho: float, lam: float) -> np.ndarray:
    """State transition matrix for [u, c]."""
    return np.array([[rho, 0.0],
                     [lam, 1.0 - lam]], dtype=float)

# ── Kalman smoother ──

def _symmetrise(P: np.ndarray, var_floor: float = 0.0) -> np.ndarray:
    P = 0.5 * (P + P.T)
    if var_floor > 0.0:
        d = np.diag(P)
        bad = ~np.isfinite(d) | (d < var_floor)
        if np.any(bad):
            np.fill_diagonal(P, np.where(bad, var_floor, d))
    return P


def kalman_smoother_pairwise(
    y: np.ndarray,
    A: np.ndarray,
    Q: np.ndarray,
    C: np.ndarray,
    R_obs: float,
    beta: float,
    init_mean: np.ndarray,
    init_cov: np.ndarray,
    var_floor: float = 0.0,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Kalman filter + RTS smoother.

    Returns (ll, sm_m[T,n], sm_P[T,n,n], pair_P[T,n,n]).
    pair_P holds Cov[x_t, x_{t-1} | Y] for t=1..T-1.
    """
    y = np.asarray(y, dtype=np.float64)
    T = y.shape[0]
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    C = np.asarray(C, dtype=np.float64)
    if C.ndim == 1:
        C = C.reshape(1, -1)
    R_obs = float(R_obs)
    var_floor = float(var_floor)

    AT = A.T
    CT = C.T

    filt_m = np.empty((T, n), dtype=np.float64)
    filt_P = np.empty((T, n, n), dtype=np.float64)
    pred_m = np.empty((T, n), dtype=np.float64)
    pred_P = np.empty((T, n, n), dtype=np.float64)
    J = np.empty((max(T - 1, 0), n, n), dtype=np.float64)

    # Forward pass
    ll = 0.0
    m_pred = init_mean.astype(np.float64).copy()
    P_pred = init_cov.astype(np.float64).copy()

    for t in range(T):
        pred_m[t] = m_pred
        pred_P[t] = P_pred

        yt = y[t]
        if np.isfinite(yt):
            CP = C @ P_pred
            S = float(CP @ CT) + R_obs
            if S > 0.0:
                K = (P_pred @ CT) / S
                innov = yt - float(C @ m_pred) - beta
                m_f = m_pred + K[:, 0] * innov
                P_f = P_pred - K @ CP
                ll += -0.5 * (_LOG2PI + np.log(S) + innov * innov / S)
            else:
                m_f, P_f = m_pred, P_pred
        else:
            m_f, P_f = m_pred, P_pred

        filt_m[t] = m_f
        filt_P[t] = _symmetrise(P_f, var_floor)

        if t < T - 1:
            m_pred = A @ m_f
            P_pred = _symmetrise(A @ filt_P[t] @ AT + Q, var_floor)

    # Backward pass (RTS)
    sm_m = filt_m.copy()
    sm_P = filt_P.copy()

    for t in range(T - 2, -1, -1):
        Pp_next = pred_P[t + 1]
        try:
            Jt = filt_P[t] @ AT @ np.linalg.inv(Pp_next)
        except np.linalg.LinAlgError:
            Jt = filt_P[t] @ AT @ np.linalg.pinv(Pp_next)
        J[t] = Jt

        sm_m[t] = filt_m[t] + Jt @ (sm_m[t + 1] - pred_m[t + 1])
        Pt = filt_P[t] + Jt @ (sm_P[t + 1] - Pp_next) @ Jt.T
        sm_P[t] = _symmetrise(Pt, var_floor)

    # Pairwise cross-covariances
    pair_P = np.zeros((T, n, n), dtype=np.float64)
    for t in range(1, T):
        pair_P[t] = sm_P[t] @ J[t - 1].T

    return ll, sm_m, sm_P, pair_P

# ── EM state ──

@dataclass
class _EMState:
    """EM parameter state."""
    rho: np.ndarray          # (N,) drive AR coefficient
    lam_c: np.ndarray        # (N,) calcium coupling
    sigma_u2: np.ndarray     # (N,) drive process noise variance
    sigma_c2: np.ndarray     # (N,) calcium process noise variance
    sigma_y2: np.ndarray     # (N,) observation noise variance
    alpha: np.ndarray        # (N,) observation gain
    beta: np.ndarray         # (N,) observation offset
    init_mean: np.ndarray    # (N, 2) initial state mean
    init_cov: np.ndarray     # (N, 2, 2) initial state covariance
    sigma_y_floor2: np.ndarray  # (N,) per-neuron sigma_y floor (variance)


@dataclass
class _SharedAccumulators:
    """Shared M-step accumulators."""
    rho_num: float = 0.0
    rho_den: float = 0.0
    lam_zd: float = 0.0
    lam_d2: float = 0.0
    sc_z2: float = 0.0
    sc_zd: float = 0.0
    sc_d2: float = 0.0
    sc_cnt: int = 0

# ── EM initialisation ──

def _init_em_state(
    X: np.ndarray,
    cfg: Stage1Config,
    active_idx: np.ndarray,
) -> _EMState:
    """Initialise all EM parameters from data and config."""
    T, N = X.shape
    dt = 1.0 / float(cfg.sample_rate_hz)

    # Dynamics
    rho_init = float(np.exp(-dt / cfg.tau_u_init_sec)) if cfg.tau_u_init_sec > 0 else 0.99
    rho_init = float(np.clip(rho_init, cfg.rho_clip[0], cfg.rho_clip[1]))
    lam_init = 1.0 - float(np.exp(-dt / cfg.tau_c_init_sec)) if cfg.tau_c_init_sec > 0 else 0.5
    lam_init = float(np.clip(lam_init, cfg.lambda_clip[0], cfg.lambda_clip[1]))

    rho = np.full(N, rho_init, dtype=np.float64)
    lam_c = np.full(N, lam_init, dtype=np.float64)
    sigma_c2 = np.full(N, cfg.sigma_c_init ** 2, dtype=np.float64)

    # Observation
    alpha = np.full(N, float(cfg.alpha_value), dtype=np.float64)
    beta = np.zeros(N, dtype=np.float64)
    sigma_y2 = np.ones(N, dtype=np.float64)
    sigma_u2 = np.ones(N, dtype=np.float64)

    for i in active_idx:
        y = X[:, i]
        mu = float(np.nanmean(y))
        beta[i] = mu if np.isfinite(mu) else 0.0
        s_y = robust_sigma_y(y - beta[i])
        sigma_y2[i] = clip_pos(s_y ** 2, floor=cfg.eps_var)
        sigma_u2[i] = clip_pos((cfg.sigma_u_scale_init * s_y) ** 2, floor=cfg.eps_var)

    # Per-neuron adaptive sigma_y floor
    sigma_y_floor2 = np.full(N, cfg.sigma_y_floor ** 2, dtype=np.float64)
    if cfg.sigma_y_floor_frac > 0:
        for i in active_idx:
            adaptive = cfg.sigma_y_floor_frac * np.sqrt(sigma_y2[i])
            sigma_y_floor2[i] = max(cfg.sigma_y_floor ** 2, adaptive ** 2)

    # Kalman init
    init_mean = np.zeros((N, 2), dtype=np.float64)
    init_cov = np.tile(np.eye(2, dtype=np.float64), (N, 1, 1))

    return _EMState(
        rho=rho, lam_c=lam_c,
        sigma_u2=sigma_u2, sigma_c2=sigma_c2, sigma_y2=sigma_y2,
        alpha=alpha, beta=beta,
        init_mean=init_mean, init_cov=init_cov,
        sigma_y_floor2=sigma_y_floor2,
    )

# ── Per-neuron E + M step ──

def _em_step_neuron(
    i: int,
    X: np.ndarray,
    obs_mask: np.ndarray,
    s: _EMState,
    acc: _SharedAccumulators,
    cfg: Stage1Config,
) -> float:
    """E-step + M-step for neuron *i*. Updates *s* and *acc* in place; returns LL."""
    T = X.shape[0]
    y = X[:, i]
    rho_i = float(s.rho[i])
    lam_i = float(s.lam_c[i])

    # E-step
    A_i = build_A(rho_i, lam_i)
    Q_i = np.diag([clip_pos(s.sigma_u2[i], cfg.eps_var),
                   clip_pos(s.sigma_c2[i], cfg.eps_var)])
    C_i = np.array([[0.0, s.alpha[i]]])

    ll, sm_m, sm_P, pair_P = kalman_smoother_pairwise(
        y=y, A=A_i, Q=Q_i, C=C_i,
        R_obs=float(s.sigma_y2[i]),
        beta=float(s.beta[i]),
        init_mean=s.init_mean[i],
        init_cov=s.init_cov[i],
        var_floor=float(cfg.eps_var),
    )
    s.init_mean[i] = sm_m[0]
    s.init_cov[i] = sm_P[0]

    # Sufficient statistics
    Eu = sm_m[:, 0]
    Ec = sm_m[:, 1]
    Euu = Eu ** 2 + sm_P[:, 0, 0]
    Ecc = Ec ** 2 + sm_P[:, 1, 1]
    Euc = Eu * Ec + sm_P[:, 0, 1]
    Eu1u0 = pair_P[1:, 0, 0] + Eu[1:] * Eu[:-1]
    Ec1u0 = pair_P[1:, 1, 0] + sm_m[1:, 1] * Eu[:-1]
    Ec1c0 = pair_P[1:, 1, 1] + sm_m[1:, 1] * Ec[:-1]

    obs_i = obs_mask[:, i]
    nobs = int(np.sum(obs_i))

    # M-step: observation
    if not cfg.fix_alpha and nobs >= 2:
        S_s = float(np.sum(Ec[obs_i]))
        S_ss = float(np.sum(Ecc[obs_i]))
        S_y = float(np.sum(y[obs_i]))
        S_sy = float(np.sum(y[obs_i] * Ec[obs_i]))
        det = S_ss * nobs - S_s * S_s
        if abs(det) > 1e-12:
            s.alpha[i] = max((nobs * S_sy - S_s * S_y) / det, cfg.alpha_floor)
            s.beta[i] = (S_y - s.alpha[i] * S_s) / nobs
    elif cfg.fix_alpha and nobs >= 1:
        s.beta[i] = (float(np.sum(y[obs_i])) -
                     s.alpha[i] * float(np.sum(Ec[obs_i]))) / nobs

    if nobs >= 1:
        resid = y[obs_i] - s.alpha[i] * Ec[obs_i] - s.beta[i]
        var_term = s.alpha[i] ** 2 * sm_P[obs_i, 1, 1]
        s.sigma_y2[i] = clip_pos(float(np.mean(resid ** 2 + var_term)), floor=cfg.eps_var)
        s.sigma_y2[i] = max(s.sigma_y2[i], s.sigma_y_floor2[i])

    if T <= 1:
        return ll

    # M-step: drive (rho, sigma_u)
    rho_num_i = float(np.sum(Eu1u0))
    rho_den_i = float(np.sum(Euu[:-1]))

    if cfg.share_rho:
        acc.rho_num += rho_num_i
        acc.rho_den += rho_den_i
    elif rho_den_i > 1e-12:
        s.rho[i] = float(np.clip(rho_num_i / rho_den_i,
                                  cfg.rho_clip[0], cfg.rho_clip[1]))

    rho_new = float(s.rho[i])  # use updated rho
    mu_pred = rho_new * Eu[:-1]
    resid_u = Eu[1:] - mu_pred
    var_u = (sm_P[1:, 0, 0]
             + rho_new ** 2 * sm_P[:-1, 0, 0]
             - 2.0 * rho_new * pair_P[1:, 0, 0])
    s.sigma_u2[i] = clip_pos(
        float(np.mean(resid_u ** 2 + np.clip(var_u, 0, None))),
        floor=cfg.eps_var)

    # M-step: calcium (lambda_c, sigma_c)
    E_zd = float(np.sum(Ec1u0 - Ec1c0 - Euc[:-1] + Ecc[:-1]))
    E_d2 = float(np.sum(Euu[:-1] - 2.0 * Euc[:-1] + Ecc[:-1]))
    E_z2 = float(np.sum(Ecc[1:] - 2.0 * Ec1c0 + Ecc[:-1]))

    if cfg.share_lambda_c:
        acc.lam_zd += E_zd
        acc.lam_d2 += E_d2
    elif E_d2 > 1e-12:
        s.lam_c[i] = float(np.clip(E_zd / E_d2,
                                    cfg.lambda_clip[0], cfg.lambda_clip[1]))

    if cfg.share_sigma_c:
        acc.sc_z2 += E_z2
        acc.sc_zd += E_zd
        acc.sc_d2 += E_d2
        acc.sc_cnt += int(T - 1)
    else:
        li = float(s.lam_c[i])
        sc2 = (E_z2 - 2.0 * li * E_zd + li ** 2 * E_d2) / max(T - 1, 1)
        s.sigma_c2[i] = clip_pos(sc2, floor=cfg.eps_var)

    return ll

# ── Shared M-step ──

def _m_step_shared(
    s: _EMState,
    acc: _SharedAccumulators,
    cfg: Stage1Config,
    active_idx: np.ndarray,
) -> None:
    """Update shared parameters after the neuron loop."""
    if cfg.share_rho and acc.rho_den > 1e-12:
        s.rho[:] = float(np.clip(acc.rho_num / acc.rho_den,
                                  cfg.rho_clip[0], cfg.rho_clip[1]))

    if cfg.share_lambda_c and acc.lam_d2 > 1e-12:
        s.lam_c[:] = float(np.clip(acc.lam_zd / acc.lam_d2,
                                    cfg.lambda_clip[0], cfg.lambda_clip[1]))
        if cfg.share_rho:
            s.rho[:] = float(np.clip(s.rho[active_idx[0]],
                                      cfg.rho_clip[0], cfg.rho_clip[1]))
        else:
            for i in active_idx:
                s.rho[i] = float(np.clip(s.rho[i],
                                          cfg.rho_clip[0], cfg.rho_clip[1]))

    if cfg.share_sigma_c and acc.sc_cnt > 0:
        li = float(s.lam_c[active_idx[0]])
        sc2 = (acc.sc_z2 - 2.0 * li * acc.sc_zd + li ** 2 * acc.sc_d2) / acc.sc_cnt
        s.sigma_c2[:] = clip_pos(sc2, floor=cfg.eps_var)

# ── Logging ──

def _log_em_iteration(
    it: int,
    ll_total: float,
    s: _EMState,
    cfg: Stage1Config,
    active_idx: np.ndarray,
    dt: float,
) -> None:
    """One-line EM iteration summary."""

    def _tau(rate: float) -> float:
        return -dt / np.log(rate) if 0 < rate < 1 else float("inf")

    # rho / tau_u
    if cfg.share_rho:
        r = float(s.rho[active_idx[0]])
        rho_str = f"rho={r:.6f} (tau_u={_tau(r):.3f}s)"
    else:
        r = float(np.nanmedian(s.rho[active_idx]))
        rho_str = f"rho_med={r:.4f} (tau_u~{_tau(r):.2f}s)"

    # lambda_c / tau_c
    if cfg.share_lambda_c:
        l = float(s.lam_c[active_idx[0]])
        lam_str = f"lam={l:.6f} (tau_c={_tau(1 - l):.3f}s)"
    else:
        l = float(np.nanmedian(s.lam_c[active_idx]))
        lam_str = f"lam_med={l:.6f} (tau_c~{_tau(1 - l):.1f}s)"

    # sigma_c
    if cfg.share_sigma_c:
        sc_str = f"sigma_c={np.sqrt(s.sigma_c2[active_idx[0]]):.6f}"
    else:
        sc_med = float(np.nanmedian(np.sqrt(s.sigma_c2[active_idx])))
        sc_str = f"sigma_c_med={sc_med:.6f}"

    print(f"[EM] iter {it + 1:02d}/{cfg.em_max_iters}"
          f"  ll={ll_total:.1f}  {rho_str}  {lam_str}  {sc_str}")

# ── Final smooth ──

def _final_smooth(
    X: np.ndarray,
    s: _EMState,
    cfg: Stage1Config,
    active_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Re-run smoother with converged parameters."""
    T, N = X.shape
    u_mean = np.full((T, N), np.nan, dtype=np.float32)
    u_var = np.full((T, N), np.nan, dtype=np.float32)
    c_mean = np.full((T, N), np.nan, dtype=np.float32)
    c_var = np.full((T, N), np.nan, dtype=np.float32)

    for i in active_idx:
        A_i = build_A(float(s.rho[i]), float(s.lam_c[i]))
        Q_i = np.diag([clip_pos(s.sigma_u2[i], cfg.eps_var),
                       clip_pos(s.sigma_c2[i], cfg.eps_var)])
        C_i = np.array([[0.0, s.alpha[i]]])

        _, sm_m, sm_P, _ = kalman_smoother_pairwise(
            y=X[:, i], A=A_i, Q=Q_i, C=C_i,
            R_obs=float(s.sigma_y2[i]),
            beta=float(s.beta[i]),
            init_mean=s.init_mean[i],
            init_cov=s.init_cov[i],
            var_floor=float(cfg.eps_var),
        )
        u_mean[:, i] = sm_m[:, 0].astype(np.float32)
        u_var[:, i] = sm_P[:, 0, 0].astype(np.float32)
        c_mean[:, i] = sm_m[:, 1].astype(np.float32)
        c_var[:, i] = sm_P[:, 1, 1].astype(np.float32)

    return u_mean, u_var, c_mean, c_var

# ── Package outputs ──

def _package_outputs(
    s: _EMState,
    cfg: Stage1Config,
    active_idx: np.ndarray,
    trace_mean: np.ndarray,
    ll_hist: list[float],
    u_mean: np.ndarray,
    u_var: np.ndarray,
    c_mean: np.ndarray,
    c_var: np.ndarray,
) -> Dict[str, object]:
    """Collect fitted parameters and smoothed outputs."""
    beta_out = s.beta + trace_mean if cfg.center_traces else s.beta.copy()

    def _scalar_or_array(arr: np.ndarray, shared: bool) -> object:
        return float(arr[active_idx[0]]) if shared else arr.astype(np.float32)

    return {
        "u_mean": u_mean,
        "u_var": u_var,
        "c_mean": c_mean,
        "c_var": c_var,
        "alpha": s.alpha.astype(np.float32),
        "beta": beta_out.astype(np.float32),
        "sigma_y": np.sqrt(s.sigma_y2).astype(np.float32),
        "sigma_u": np.sqrt(s.sigma_u2).astype(np.float32),
        "rho": _scalar_or_array(s.rho, cfg.share_rho),
        "lambda_c": _scalar_or_array(s.lam_c, cfg.share_lambda_c),
        "sigma_c": (float(np.sqrt(s.sigma_c2[active_idx[0]])) if cfg.share_sigma_c
                    else np.sqrt(s.sigma_c2).astype(np.float32)),
        "ll_hist": np.array(ll_hist, dtype=np.float64),
        "trace_mean": trace_mean if cfg.center_traces else None,
    }

# ── Main entry point ──

def fit_stage1_all_neurons(
    X: np.ndarray,
    cfg: Stage1Config,
) -> Dict[str, object]:
    X = np.asarray(X, dtype=np.float64)
    T, N = X.shape
    dt = 1.0 / float(cfg.sample_rate_hz)

    # Trace centering
    trace_mean = np.zeros(N, dtype=np.float64)
    if cfg.center_traces:
        trace_mean = np.nanmean(X, axis=0)
        trace_mean = np.where(np.isfinite(trace_mean), trace_mean, 0.0)
        X = X - trace_mean[None, :]

    # Active neurons
    obs_mask = np.isfinite(X)
    nan_frac = 1.0 - obs_mask.mean(axis=0)
    active = nan_frac < 1.0
    active_idx = np.flatnonzero(active)

    if len(active_idx) == 0:
        print("[EM] No neurons have finite data — nothing to fit")
        empty = np.full((T, N), np.nan, dtype=np.float32)
        return {
            "u_mean": empty, "u_var": empty,
            "c_mean": empty, "c_var": empty,
            "alpha": np.ones(N, dtype=np.float32),
            "beta": np.zeros(N, dtype=np.float32),
            "sigma_y": np.ones(N, dtype=np.float32),
            "sigma_u": np.ones(N, dtype=np.float32),
            "rho": 0.99, "lambda_c": 0.5, "sigma_c": 0.01,
            "ll_hist": np.array([]), "trace_mean": None,
        }

    for i in np.flatnonzero(~active):
        print(f"[EM] Skipping neuron {i}: all-NaN")

    # Initialise
    s = _init_em_state(X, cfg, active_idx)
    ll_hist: list[float] = []

    # EM loop
    for it in range(int(cfg.em_max_iters)):
        acc = _SharedAccumulators()
        ll_total = 0.0

        for i in active_idx:
            ll_total += _em_step_neuron(i, X, obs_mask, s, acc, cfg)

        _m_step_shared(s, acc, cfg, active_idx)

        ll_hist.append(ll_total)
        _log_em_iteration(it, ll_total, s, cfg, active_idx, dt)

        # Convergence check
        if it >= 1:
            prev = ll_hist[-2]
            if np.isfinite(prev) and abs(prev) > 1e-9:
                rel = abs((ll_total - prev) / prev)
                if rel < cfg.em_tol_rel_ll:
                    print(f"[EM] Converged at iteration {it + 1} (rel={rel:.3e})")
                    break
    else:
        print(f"[EM] Did not converge after {cfg.em_max_iters} iterations")

    # Final smooth
    u_mean, u_var, c_mean, c_var = _final_smooth(X, s, cfg, active_idx)

    n_fitted = int(np.sum(active))
    print(f"[EM] Fitted {n_fitted}/{N} neurons ({N - n_fitted} skipped)")

    return _package_outputs(s, cfg, active_idx, trace_mean, ll_hist,
                            u_mean, u_var, c_mean, c_var)
