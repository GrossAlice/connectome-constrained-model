# Connectome-Constrained Neural Activity Decoder v5

## Overview

This experiment replicates v4's neural activity prediction but uses **connectome-defined neighbors** instead of all recorded neurons. This tests whether the structural connectome provides useful priors for neural dynamics prediction.

## Connectome Structure

### Gap Junctions (T_e)
- **Type**: Electrical synapses
- **Property**: Symmetric (bidirectional)
- **Mechanism**: Direct electrical coupling, fast
- **Usage in model**: **Concurrent features** u_j(t)
- **Avg neighbors**: ~7 per neuron (atlas-wide)

### Chemical Synapses (T_sv + T_dcv)
- **T_sv**: Synaptic vesicles (classical neurotransmitters)
- **T_dcv**: Dense core vesicles (neuromodulators)
- **Property**: Asymmetric (directed: pre → post)
- **Mechanism**: Chemical transmission, slower with delay
- **Usage in model**: **Causal/lagged features** u_j(t-1..t-K)
- **Avg neighbors**: ~12 (T_sv) + ~29 (T_dcv) per neuron

## Conditions (6 total)

| Condition | Gap (conc) | Syn (lag) | Self (lag) | Description |
|-----------|------------|-----------|------------|-------------|
| self | ❌ | ❌ | ✅ | Pure autoregressive |
| conc | ✅ | ❌ | ❌ | Gap junction concurrent only |
| causal | ❌ | ✅ | ❌ | Synaptic lagged only |
| conc_causal | ✅ | ✅ | ❌ | Gap + synaptic (no self) |
| conc_self | ✅ | ❌ | ✅ | Gap + self-history |
| causal_self | ❌ | ✅ | ✅ | Synaptic + self-history |

## Stage2 Model Design Recommendations

Based on v4 and v5 results, here's how to structure a stage2 model for both LOO and free-run:

### 1. Self-History (AR Component) — ESSENTIAL
```python
# Always include, dominant predictor (R² ~ 0.84 from v4)
h_self = MLP(u_i[t-K:t])  # or LSTM/Transformer on own history
```

### 2. Gap Junction Coupling (Concurrent)
```python
# Use concurrent features from gap junction partners
# Gap junctions are fast enough for concurrent effects
gap_neighbors = T_e[j, i] > 0  # or T_e[i, j] > 0 (symmetric)
h_gap = attention(u[t], mask=gap_neighbors)  # soft/gated
# Or: h_gap = sum(T_e[j,i] * u_j[t] for j in gap_neighbors)
```

### 3. Synaptic Coupling (Causal/Lagged)
```python
# Use lagged features from presynaptic partners
syn_presynaptic = T_syn[j, i] > 0  # j sends to i
h_syn = attention(u[t-K:t], mask=syn_presynaptic)  # soft/gated
# Or: h_syn = sum(T_syn[j,i] * u_j[t-k] for j in syn_pre, k=1..K)
```

### 4. Full Stage2 Architecture
```python
class ConnectomeConstrainedModel:
    def __init__(self, T_e, T_syn, N, K):
        self.T_e = T_e      # gap junction weights
        self.T_syn = T_syn  # synapse weights
        
        # Self-history encoder (per-neuron or shared)
        self.ar_encoder = TransformerEncoder(d_in=1, d_model=64, n_layers=2)
        
        # Gap junction aggregator (concurrent)
        self.gap_attn = MultiHeadAttention(d_model=64, n_heads=4)
        
        # Synapse aggregator (causal)
        self.syn_attn = MultiHeadAttention(d_model=64, n_heads=4)
        
        # Combiner
        self.combiner = MLP([64*3, 128, N])
        
    def forward(self, u, t):
        # u: (T, N) neural activity
        # t: current time step
        
        # 1. Self-history
        h_self = self.ar_encoder(u[t-K:t])  # (N, d)
        
        # 2. Gap junction (concurrent)
        gap_mask = self.T_e > 0  # (N, N)
        h_gap = self.gap_attn(u[t:t+1], mask=gap_mask)  # (N, d)
        
        # 3. Synaptic (causal)
        syn_mask = self.T_syn > 0  # (N, N)
        h_syn = self.syn_attn(u[t-K:t], mask=syn_mask)  # (N, d)
        
        # 4. Combine and predict
        h = torch.cat([h_self, h_gap, h_syn], dim=-1)  # (N, 3d)
        u_pred = self.combiner(h)  # (N,)
        
        return u_pred
```

### 5. Weight Initialization Options

```python
# Option A: Use connectome as mask (sparse)
# Zero out connections not in connectome
gap_attn.weight *= (T_e > 0)
syn_attn.weight *= (T_syn > 0)

# Option B: Use connectome as initialization
# Initialize weights proportional to connection strength
gap_attn.weight = softmax(T_e, dim=-1)
syn_attn.weight = softmax(T_syn, dim=-1)

# Option C: Learnable with connectome prior
# Add regularization to stay close to connectome
loss += λ * ||W - normalize(T)||^2
```

### 6. LOO vs Free-Run Considerations

**For LOO (one-step prediction):**
- Self-history is from ground truth (teacher forcing)
- Gap junction concurrent features are from ground truth
- Only causal synaptic features matter for generalization

**For Free-Run (multi-step simulation):**
- Self-history is from own predictions (error accumulates)
- Strong AR component helps stability
- Connectome structure may help prevent drift
- Consider noise injection during training (scheduled sampling)

## Output Files

For each worm, the script saves:
- `results.json`: Per-neuron R² and correlation for all conditions
- `T_gap_{worm}.npy`: Mapped gap junction matrix (N_worm × N_worm)
- `T_syn_{worm}.npy`: Mapped synapse matrix (N_worm × N_worm)
- `connectome_info_{worm}.json`: Neighbor lists and design recommendations
- Plots: heatmaps, connectivity stats, R² vs connectivity

## Usage

```bash
# Run on all 8 worms (same as v4)
python -m scripts.neural_activity_decoder_v5_connectome \
    --data_dir "data/used/behaviour+neuronal activity atanas (2023)/2" \
    --worm_ids 2022-06-14-01 2022-06-14-07 2022-07-15-06 2022-07-15-12 \
               2022-12-21-06 2023-01-05-01 2023-01-06-15 2023-01-09-08 \
    --neurons all \
    --device cuda
```
