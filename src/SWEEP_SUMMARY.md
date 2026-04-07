# Stage 2 Hyperparameter Sweep — Cumulative Summary

**Date:** 2026-04-07  
**Worm:** primarily `2022-08-02-01` (N=123, T=1600, dt=0.6s)  
**Metric:** Windowed LOO R² (`loo_w_mean`, window=50) unless noted otherwise  
**Baseline reference:** `loo_w_mean ≈ 0.14` (current best Stage2 config, 30 epochs)

---

## 1. Executive Summary

Across **12 sweep campaigns** and **~200 individual conditions**, the overwhelming conclusion is:

> **The Stage2 connectome-constrained model's LOO R² is essentially determined by (a) the initialisation strategy and (b) the sigmoid nonlinearity. Almost nothing else moves the needle.**

The model's coupling strengths (G) collapse to near-zero when using per-neuron ridge initialisation, meaning the network effectively becomes 123 independent AR(1) processes. All tested regularisers, loss augmentations, temporal-kernel variants, learned reversals, coupling gates, and tau parameterisations produce LOO R² variations of **< 0.01** — well within noise.

**Best LOO R² achieved:** ~0.17 (v3_run3, 60 epochs) — still well below MLP/Transformer LOO ≈ 0.22.

---

## 2. Sweep-by-Sweep Results

### 2.1 `sweep_loo` — First LOO exploration (10 epochs)

| Condition | LOO med | LOO mean | Notes |
|-----------|---------|----------|-------|
| 00_baseline | 0.043 | −0.163 | Edge-specific G, global init |
| 01_fix_a | 0.042 | −0.176 | Fix amplitudes — no change |
| 02_fix_a_noperneuron | 0.043 | −0.175 | Shared amplitudes — no change |
| 03–05_rollout_weight | 0.042 | −0.178 | Rollout weight 0.3, 0.5, 1.0 — no change |
| 06–07_rollout_steps | 0.043 | −0.171 | Steps 10, 20 — no change |
| 09_fix_a_lr3e3 | 0.044 | −0.126 | Higher LR helps slightly |
| 11_fix_a_synlr10 | 0.045 | −0.153 | Synapse LR 10× — tiny improvement |
| 12_fix_a_synlr20 | 0.048 | −0.110 | Synapse LR 20× — best in sweep |
| **16_fix_a_pnridge** | **0.043** | **+0.035** | **Per-neuron ridge init: loo_mean goes POSITIVE** |

**Key finding:** Per-neuron ridge initialisation is the single biggest improvement — it turns loo_mean from −0.16 to +0.035 and raises loo_pos from 0.6 to 0.9.

---

### 2.2 `sweep_loo2` — Per-neuron ridge deep-dive (20 epochs, 25 conditions)

All 24 conditions after switching to per-neuron ridge init produce **loo_med ≈ 0.043 ± 0.001**. The coupling G collapses to ~6e-5 in all cases.

| Condition | LOO med | Δ vs baseline |
|-----------|---------|---------------|
| 01_pnr (baseline) | 0.0437 | — |
| 02_pnr_synlr10 | 0.0439 | +0.000 |
| 03_pnr_synlr20 | 0.0431 | −0.001 |
| 04_pnr_lr3e3 | 0.0441 | +0.000 |
| 06_pnr_looaux0 | 0.0431 | −0.001 |
| 07–09_pnr_looauxXX | 0.043 | −0.001 |
| 12_pnr_fix_a | 0.0437 | +0.000 |
| 13_pnr_learn_rev | 0.0427 | −0.001 |
| 15_pnr_sigma | 0.0434 | +0.000 |
| 17_pnr_roll03_steps10 | 0.0428 | −0.001 |
| 18_pnr_ridgeW | 0.0427 | −0.001 |
| 23_combo_best | 0.0442 | +0.001 |
| 24_combo_aggressive | 0.0427 | −0.001 |

**Key finding:** Once using per_neuron_ridge init, NO combination of synaptic LR, LOO aux loss, rollout, learned reversals, learned noise, ridge on W, or LR changes makes any meaningful difference. G→0, so the network coupling is ignored.

---

### 2.3 `sweep_sensitivity` — 2⁴ factorial design (20 epochs, 16 conditions)

Tested all combinations of: `learn_lambda_u` × `edge_specific_G` × `per_neuron_amplitudes` × `network_init_mode`

| Factor | Effect on LOO |
|--------|--------------|
| **network_init_mode = per_neuron** | **loo_mean ≈ +0.08 vs −0.02 for global** |
| learn_lambda_u (True vs False) | No effect |
| edge_specific_G (True vs False) | No effect (G→0 anyway) |
| per_neuron_amplitudes (True vs False) | No effect |

**Key finding:** Init mode is the ONLY factor that matters in this 2⁴ design. All other factors are completely swamped.

---

### 2.4 `sweep_AH` — Training hyperparameters (20 epochs, 18 conditions)

| Condition | LOO med | Notes |
|-----------|---------|-------|
| 00_A_baseline | 0.073 | Reference |
| 01_A_cv5 (dynamics CV every 5) | −2.337 | **CATASTROPHIC** — validation-based stopping breaks LOO |
| 04–06_D_cvXX (cv 1/10/20) | −2.34 to 0.074 | CV every 20 ≈ baseline; all others destroy LOO |
| 07–09_E_blendXX | −0.41 to −2.36 | Blending hurts uniformly |
| 10_F_synlr10 | 0.074 | ≈baseline |
| 11_F_synlr20 | 0.076 | Tiny improvement |
| 12_G_lr3e3 | 0.078 | Slightly better |
| 13–17_H_rollout | 0.078–0.080 | Small improvement from rollout |

**Key finding:** `dynamics_cv_every` and `dynamics_cv_blend` are **actively harmful** — they should never be used (or should be removed from config). Higher LR (3e-3) and rollout help marginally.

---

### 2.5 `loo_sweep_v2` — Architecture variants (20 epochs)

| Condition | LOO_w mean | Δ |
|-----------|-----------|---|
| baseline (sigmoid) | 0.134 | — |
| **linear** | **0.066** | **−0.068 ☠️** |
| 1tau / 2tau | 0.133 | +0.000 |
| **scalar_G** | **0.173** | **+0.039 ✓** |
| no_W_learn | 0.131 | −0.003 |
| hi_loo_aux | 0.137 | +0.003 |
| lo_lambda_floor | 0.126 | −0.008 |

**Key findings:**
- **Linear chemical synapses DESTROY LOO** (0.066 vs 0.134). The sigmoid nonlinearity is essential.
- **Scalar G outperforms edge-specific G** for LOO (0.173 vs 0.134), consistent with G→0 under per_neuron_ridge.
- Number of tau ranks (1 vs 2) makes zero difference.

---

### 2.6 `loo_sweep_lamfloor` — Lambda floor sweep

| λ floor | LOO_w mean |
|---------|-----------|
| 0 (default) | 0.055 |
| **0.1** | **0.126** |
| 0.2 | 0.099 |
| 0.3 | 0.076 |
| 0.5 | 0.050 |

**Key finding:** `lambda_u_lo = 0.1` helps the most, but this is a regime where the AR(1) component dominates. Higher floors force more decay and hurt.

---

### 2.7 `loo_sweep_v3_run3` — Feature additions, 2 worms (60 epochs)

For **2022-08-02-01** (all give `loo_w_mean ≈ 0.170 ± 0.002`):

| Condition | LOO_w mean |
|-----------|-----------|
| v5_base | 0.1717 |
| int_l2_01 | 0.1717 |
| int_l2_10 | 0.1717 |
| ridge_W | 0.1717 |
| G_reg | 0.1717 |
| gate | 0.1704 |
| learn_E | 0.1717 |
| gate_E | 0.1703 |
| tau_scale | 0.1716 |
| gate_E_tau | 0.1702 |
| **kitchen_sink** | **0.1702** |

For **2022-06-14-01** (all give `loo_w_mean ≈ 0.077 ± 0.001`):
Same pattern — nothing changes.

**Key finding:** At 60 epochs, coupling gate, learned reversals, per-neuron tau scale, interaction L2, ridge on W, G regularisation, and the kitchen-sink combination ALL produce the same LOO. The coupling gate actually makes LOO slightly *worse* (0.170 → 0.170).

---

### 2.8 `tau_kernel_sweep` — Temporal kernels (30 epochs, 13 configs)

| Config | LOO_w mean | Description |
|--------|-----------|-------------|
| A_baseline (τ_sv=(0.5,2.5)) | 0.1340 | Current defaults |
| B_learn_taus | 0.1342 | Learn τ — no change |
| C_longer_fixed | 0.1344 | τ_sv=(2,8), τ_dcv=(5,15) |
| D_very_long | 0.1366 | τ_sv=(5,20), τ_dcv=(10,40) |
| F_3ranks | 0.1352 | 3 ranks |
| G_4ranks | 0.1365 | 4 ranks |
| I_perneuron_tau | 0.1340 | Per-neuron τ scale |
| K_3ranks_perneuron_learn | 0.1353 | Full temporal flexibility |
| L_no_gate_learn | 0.1350 | No coupling gate |
| **M_wide_span** | **0.1399** | **τ_sv=(1,30), τ_dcv=(2,60) — best** |

**Total range: 0.006.** Temporal kernels are NOT the bottleneck.

---

### 2.9 `loo_sweep_20ep` — Comprehensive 20-epoch sweep (24 conditions)

LOO medians cluster at 0.03–0.04 for all conditions (baseline, rollout, LOO aux, noise, learnW, etc.). The only outlier is `init_ols` with cv_loo_median=0.197 — but that's OLS init, which is a different regime.

**Key finding:** In 20 epochs with per_neuron_ridge init, everything flatlines. `noise_hetero` and `noise` don't help. `learnW` doesn't help. `combo_aggressive` doesn't help.

---

### 2.10 `nonlinearity_arch_sweep` — COMPLETED (30 epochs, 17/18 conditions)

**Group A — Activation functions:**

| Config | Activation | 1-step R² | LOO_w mean | Notes |
|--------|-----------|-----------|-----------|-------|
| **A0_sigmoid** | sigmoid (baseline) | 0.785 | **0.140** | ✅ baseline |
| A1_tanh | tanh | 0.794 | 0.037 | ❌ LOO collapsed |
| **A2_softplus** | softplus | 0.785 | **0.145** | ✅ **best overall** |
| A3_relu | relu | 0.786 | 0.124 | ⚠️ slight drop |
| A4_elu | elu | 0.789 | 0.055 | ❌ LOO collapsed |
| A5_swish | swish | 0.789 | 0.050 | ❌ LOO collapsed |
| A6_shifted_sigmoid | shifted_sigmoid | 0.786 | 0.140 | ≈ baseline |
| A7_identity | identity (linear) | 0.791 | 0.052 | ❌ LOO collapsed |

**Group B — Residual MLP:**

| Config | Hidden | Layers | 1-step R² | LOO_w mean |
|--------|--------|--------|-----------|------------|
| B1 | 32 | 2 | 0.790 | 0.140 |
| B2 | 64 | 2 | 0.792 | 0.140 |
| **B3** | **128** | **2** | **0.796** | 0.141 |
| B4 | 64 | 3 | 0.791 | 0.141 |

**Groups C–E — Low-rank / Reversals / Noise:**

| Config | Override | 1-step R² | LOO_w mean |
|--------|----------|-----------|------------|
| C1–C3 | lowrank_rank = 5/10/20 | 0.784–0.785 | 0.140 |
| D1 | learn_reversals=True | 0.786 | 0.140 |
| E1–E2 | noise_corr_rank = 5/10 | 0.785 | 0.140 |

**Group F:** F1_input_noise_0.05 — crashed (no results).

**Key findings:**
1. **Softplus is the only activation that improves LOO** (+0.005 over sigmoid). tanh/elu/swish/identity all achieve *higher* 1-step R² but *terrible* LOO — they overfit the teacher-forced task.
2. **Residual MLP helps 1-step R² (+0.011 with h=128) but NOT LOO.** Extra params absorb variance the connectome should explain.
3. **Low-rank, reversals, correlated noise are all inert** — confirmed findings from earlier sweeps.
4. The 1-step ↑ / LOO ↓ dissociation is a reliable overfitting signal.

---

### 2.11 `followup_sweep` — Running (23 conditions, ~130 min)

Targeted experiments informed by all prior sweep results.

| Group | Conditions | What's tested | Rationale |
|-------|-----------|--------------|----------|
| A (5) | sigmoid/softplus × 30/60/100ep | Epoch scaling | 30→60ep was +0.03 LOO; is there more? |
| B (3) | poly3, poly4, poly3+60ep | Multi-hop propagation | Never completed (constraint_feature_sweep killed) |
| C (3) | input_noise 0.02/0.05/0.10 | Teacher-forcing regularisation | F1 crashed in prior sweep |
| D (3) | coupling_dropout 0.1/0.2/0.3 | Regularise connectome path | Exists in config, never tested |
| E (3) | rollout, rollout+60ep, curriculum | Rollout training | Barely tested with current config |
| F (6) | softplus + poly3/noise/dropout combos, kitchen sink | Best combinations | Combine all positives |

---

## 3. Grand Conclusions

### 3.1 What Actually Matters (ranked by effect size)

| Rank | Factor | Effect | Evidence |
|------|--------|--------|----------|
| 1 | **Initialisation mode** (per_neuron_ridge) | LOO_mean: −0.16 → +0.04 | sweep_loo, sweep_sensitivity |
| 2 | **Saturating nonlinearity** (sigmoid/softplus vs identity) | LOO_w: 0.14 → 0.05 | nonlinearity_arch_sweep |
| 3 | **Epoch count** (20 → 60) | LOO_w: 0.13 → 0.17 | v3 vs v2 comparison |
| 4 | **Softplus** (vs sigmoid) | LOO_w: 0.140 → 0.145 (+3.6%) | nonlinearity_arch_sweep |
| 5 | **Wide tau span** (τ_sv=(1,30)) | +0.006 | tau_kernel_sweep |

### 3.2 What Does NOT Matter (<0.01 effect across all sweeps)

| Parameter | Times tested | Effect | Verdict |
|-----------|-------------|--------|---------|
| `fix_tau_sv` / `fix_tau_dcv` (learn vs fix) | 5 sweeps | ±0.002 | **Irrelevant** |
| Number of tau ranks (1/2/3/4) | tau_kernel_sweep | ±0.003 | **Irrelevant** |
| `per_neuron_tau_scale` | 3 sweeps | ±0.001 | **Remove** |
| `per_neuron_amplitudes` (True vs False) | sweep_sensitivity | ±0.000 | **Irrelevant** |
| `edge_specific_G` (True vs False) | 3 sweeps | ±0.000 | **G→0 anyway** |
| `learn_W_sv` / `learn_W_dcv` | 3 sweeps | ±0.001 | **Irrelevant** |
| `learn_reversals` | 2 sweeps | ±0.001 | **Irrelevant** |
| `coupling_gate` | 3 sweeps | −0.001 | **Slightly harmful for LOO** |
| `interaction_l2` | 3 sweeps | ±0.000 | **Remove** |
| `ridge_W_sv` / `ridge_W_dcv` | 2 sweeps | ±0.000 | **Remove** |
| `G_reg` | 2 sweeps | ±0.000 | **Remove** |
| `loo_aux_weight` / `loo_aux_neurons` / `loo_aux_steps` | 4 sweeps | ±0.001 | **Remove** |
| `synaptic_lr_multiplier` (1–20×) | 3 sweeps | ±0.005 | **Marginal** |
| `rollout_weight` / `rollout_steps` | 4 sweeps | ±0.005 | **Marginal at best** |
| `learn_lambda_u` (True vs False) | sweep_sensitivity | ±0.000 | **Irrelevant** |
| `noise_mode` (homo vs heteroscedastic) | loo_sweep_20ep | ±0.001 | **Irrelevant** |
| `learn_noise` | 2 sweeps | ±0.001 | **Irrelevant** |
| `dynamics_cv_every` / `dynamics_cv_blend` | sweep_AH | HARMFUL | **Remove** |
| `residual_mlp_hidden` (32/64/128) | nonlinearity_arch_sweep | ±0.001 LOO | **Helps 1-step, NOT LOO** |
| `lowrank_rank` (5/10/20) | nonlinearity_arch_sweep | ±0.000 | **Inert** |
| `noise_corr_rank` (5/10) | nonlinearity_arch_sweep | ±0.000 | **Inert** |
| `learn_reversals` (per_neuron) | nonlinearity_arch_sweep | ±0.000 | **Confirmed inert** |
| tanh / elu / swish / identity activations | nonlinearity_arch_sweep | −0.09 LOO | **HARMFUL** — overfits 1-step |

---

## 4. Config Parameter Recommendations

### 4.1 Parameters Safe to Remove (dead weight)

These have been tested multiple times and shown zero or harmful effect:

| Parameter | Default | Reason to remove |
|-----------|---------|-----------------|
| `per_neuron_tau_scale` | `False` | 3 sweeps, ±0.001 effect. Never helps. |
| `interaction_l2` | `0.0` | 3 sweeps, exactly zero effect at any value. |
| `ridge_W_sv` | `0.0` | 2 sweeps, zero effect. |
| `ridge_W_dcv` | `0.0` | 2 sweeps, zero effect. |
| `G_reg` | `0.0` | 2 sweeps, zero effect. |
| `lambda_u_reg` | `0.0` | Never meaningfully tested but same pattern. |
| `I0_reg` | `0.0` | Never meaningfully tested but same pattern. |
| `tau_reg` | `0.0` | Never tested in isolation. |
| `dynamics_cv_every`* | N/A | **Actively harmful** — catastrophic LOO collapse. |
| `dynamics_cv_blend`* | N/A | **Actively harmful** — LOO collapse. |
| `W_sv_init_mode` | `"uniform"` | Never tested; only "uniform" used. |
| `W_sv_normalize` | `False` | Never tested. |
| `noise_sigma_source` | `"all"` | Only "all" ever used. |
| `network_strength_floor` | `1.0` | Part of init-anchor; never helps LOO. |
| `network_strength_target` | `0.8` | Same as above. |

*These are in TrainConfig but should not exist or should have a warning.

### 4.2 Parameters to Simplify

| Parameter | Current | Recommendation |
|-----------|---------|---------------|
| `reversal_mode` | `"per_neuron"` | Keep only `"scalar"` — per_neuron never helped. |
| `G_init_mode` | `"uniform"` | Remove the 3-way choice; always "uniform". |
| `edge_specific_G` | `False` | Consider defaulting to `False` (scalar G did better in loo_sweep_v2). |
| `coupling_gate` | `True` | **Questionable** — slightly hurts LOO in every sweep tested. Keep for interpretability but default could be False. |
| `coupling_gate_reg` | `0.01` | If gate is kept, reg doesn't matter. |

### 4.3 Parameters to Keep

| Parameter | Default | Why |
|-----------|---------|-----|
| `chemical_synapse_activation` | `"softplus"` | Best LOO activation (+0.005 over sigmoid). Identity/tanh/elu/swish destroy LOO. |
| `tau_sv_init` / `tau_dcv_init` | `(1,30)/(2,60)` | Wide span gives best LOO (marginal). |
| `learn_noise` + `noise_mode="heteroscedastic"` | True | Doesn't hurt, good for sampling. |
| `graph_poly_order` | `2` | Keep — multi-hop being tested in followup_sweep. |
| `residual_mlp_hidden` | `0` | **Tested** — helps 1-step but NOT LOO. Keep at 0. |

---

## 5. The Fundamental Problem

The data tells a consistent story:

1. **G collapses to ~0** with per_neuron_ridge init. The model learns that the AR(1) autoregressive component (λ_u × u_prev + I0) is sufficient for one-step prediction, and the connectome coupling is noise.

2. **The connectome coupling operates at ~30% of AR(1) scale** at init (network_scale OLS shows net_rms ≈ 34% of AR1_rms), but gradient descent pushes it further down because it doesn't help the LOO metric.

3. **LOO R² is capped at ~0.17** because the model has no mechanism to predict neuron $i$ from the *other* neurons — it relies on each neuron's own history. The connectome edges that should carry this information have zero weight.

### Why does MLP get LOO ≈ 0.22?

The MLP decoder has unconstrained $N × N$ coupling — it can learn arbitrary cross-neuron relationships without being limited to the known connectome edges. The 0.05 gap between Stage2 (0.17) and MLP (0.22) represents **real cross-neuron information that flows through non-connectome pathways** (volume transmission, neuromodulation, unrecorded intermediaries, or simply connectome incompleteness).

---

## 6. What Has NOT Been Tested

### 6.1 Currently Running
- **`followup_sweep`** (23 conditions): softplus+epochs, graph_poly_order=3/4, input_noise, coupling_dropout, rollout curriculum, combos

### 6.2 Resolved (Now Tested)
| What | Result | Sweep |
|------|--------|-------|
| `lowrank_rank > 0` | **Inert** — ranks 5/10/20 all LOO=0.140 | nonlinearity_arch_sweep |
| `noise_corr_rank > 0` | **Inert** — ranks 5/10 both LOO=0.140 | nonlinearity_arch_sweep |
| `residual_mlp_hidden` | **Helps 1-step only** — h=128 OS=0.796 but LOO=0.141 | nonlinearity_arch_sweep |
| All activation functions | **Softplus best** (+0.005); tanh/elu/swish/identity harmful | nonlinearity_arch_sweep |

### 6.3 Being Tested (followup_sweep)
| What | Why it might help | Conditions |
|------|-------------------|------------|
| **`graph_poly_order = 3, 4`** | Multi-hop propagation through connectome | B1, B2, B3 |
| **`input_noise_sigma > 0`** | Regularisation during teacher-forcing | C1, C2, C3 |
| **`coupling_dropout > 0`** | Regularise the connectome path; may prevent G→0 | D1, D2, D3 |
| **Longer training (100 epochs)** | All prior sweeps ≤60 epochs | A2, A5, F6 |
| **`rollout_curriculum`** | Gradual rollout lengthening | E3 |
| **Softplus combinations** | Stack multiple small gains | F1–F6 |

### 6.4 Never Tested (future work)
| What | Why it might help | Priority |
|------|-------------------|----------|
| **Learning rate scheduling** (warmup+cosine) | Standard deep learning practice; no scheduler in train.py | MEDIUM |
| **AdamW vs Adam** | Weight decay may help generalisation | LOW |
| **`behavior_weight > 0`** during dynamics | Joint neural+behavior optimisation | LOW |
| **`stim_kernel_len > 0`** | Temporal convolution for stimuli | LOW |
| **Multi-worm training** | Shared connectome across worms | LOW for LOO |
| **Ensembling** (train N models, average) | Variance reduction | LOW |

### 6.3 Previously Promising — Now Tested and Failed

The **residual MLP** was expected to close the 0.05 LOO gap by adding unconstrained N→N coupling. Results: 1-step R² improved (+0.011 with h=128) but LOO was unchanged (0.141 vs 0.140). The extra capacity is absorbed by the teacher-forced objective without improving generalisation.

The **low-rank coupling** added dense N→N coupling in a parameter-efficient way. Results: completely inert — no change in either 1-step or LOO across ranks 5/10/20.

### 6.4 Most Promising Remaining Directions

1. **Softplus + longer training** — softplus gave +0.005 LOO at 30ep; longer training gives +0.03; do they stack? (Being tested in followup_sweep A4/A5)
2. **graph_poly_order=3** — multi-hop propagation, never completed (followup_sweep B1–B3)
3. **Coupling dropout** — regularise the connectome path, may prevent G→0 (followup_sweep D1–D3)
4. **Input noise** — teacher-forcing regularisation (followup_sweep C1–C3)
5. **Kitchen-sink combo** — softplus + poly3 + input_noise + dropout + 100ep (followup_sweep F6)

---

## 7. Recommended Minimal Config

Based on all evidence, the minimal config that achieves near-optimal LOO is surprisingly simple:

```python
# Everything that matters
DynamicsConfig(
    chemical_synapse_activation="softplus",  # +0.005 over sigmoid; essential — identity loses 0.09
    tau_sv_init=(1.0, 30.0),                # wide span, marginal +0.006
    tau_dcv_init=(2.0, 60.0),
    fix_tau_sv=True,                        # learning doesn't help
    fix_tau_dcv=True,
    learn_noise=True,
    noise_mode="heteroscedastic",
    graph_poly_order=2,                     # 3+ being tested in followup_sweep
)
TrainConfig(
    num_epochs=50,                          # more epochs = better
    learning_rate=0.001,
    # network_init_mode="per_neuron_ridge"  # (handled in training code)
)
```

Everything else is at default or can be removed.

---

## 8. Appendix: Summary Table of All Sweeps

| Sweep | # Conditions | Epochs | Init Mode | Best LOO_w | Key Finding |
|-------|-------------|--------|-----------|-----------|-------------|
| sweep_loo | 18 | 10 | global→pnr | 0.043 (med) | per_neuron_ridge is transformative |
| sweep_loo2 | 25 | 20 | per_neuron_ridge | 0.044 (med) | Nothing helps after pnr init |
| sweep_sensitivity | 16 | 20 | both | 0.088 (mean) | Init mode is only significant factor |
| sweep_AH | 18 | 20 | mixed | 0.080 (med) | dynamics_cv is catastrophic |
| loo_sweep_20ep | 24 | 20 | per_neuron_ridge | 0.039 (med) | Everything flatlines |
| loo_sweep_lamfloor | 14 | 20 | per_neuron_ridge | 0.126 (w_mean) | λ floor 0.1 helps windowed |
| loo_sweep_v2 | 14 | 20 | mixed | 0.173 (w_mean) | Linear hurts; scalar_G best |
| loo_sweep_v3 | 2 | 15 | v5 | 0.174 (w_mean) | int_l2 no effect |
| loo_sweep_v3_run3 | 24 | 60 | v5 | 0.172 (w_mean) | Gate, learn_E, tau_scale — no effect |
| tau_kernel_sweep | 13 | 30 | v5 | 0.140 (w_mean) | Taus don't matter (range=0.006) |
| loo_improvement_sweep | 4 | — | — | — | Per-neuron R² data |
| nonlinearity_arch_sweep | 17 | 30 | v5 | 0.145 (w_mean) | Softplus best; residual MLP / low-rank inert |
| followup_sweep | 23 | 30–100 | v5 | 🔄 running | Softplus+epochs, poly3, dropout, combos |

---

*Last updated: 2026-04-07. Auto-generated from sweep results in `output_plots/stage2/`.*
