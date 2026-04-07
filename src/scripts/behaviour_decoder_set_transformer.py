#!/usr/bin/env python
"""Multi-worm behaviour decoder using Set Transformer with neuron identity embeddings.

Architecture
------------
Unlike the per-worm models which have fixed input dimension (N neurons for that worm),
this model uses neuron identity embeddings to pool across all worms:

1. Each neuron's activity x_i is combined with its identity embedding e_i:
   h_i = MLP([x_i; e_i])  or  h_i = x_i * W + e_i

2. The set of neuron representations is processed by Set Transformer blocks
   (permutation invariant via self-attention over neurons).

3. A learned "behavior query" token attends to all neurons to produce behavior output.

This allows training on ALL worms simultaneously despite different neuron subsets.

Key differences from Atlas Transformer:
- Atlas TRF: Embeds into fixed 302-dim space with zero-padding for missing neurons
- Set TRF: Processes variable-size sets natively, no padding needed

Author: Copilot
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stage2.io_multi import _load_full_atlas, _read_neuron_labels, _match_labels_to_atlas

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SetTransformerConfig:
    """Configuration for Set Transformer behaviour decoder."""
    # Neuron embedding
    n_atlas: int = 302  # Total neurons in C. elegans atlas (use full vocab for indexing)
    neuron_embed_dim: int = 64  # Dimension of neuron identity embeddings
    
    # Set Transformer architecture
    d_model: int = 128  # Hidden dimension
    n_heads: int = 4
    n_encoder_layers: int = 2  # SAB layers
    n_decoder_layers: int = 1  # PMA layers (usually 1)
    d_ff: int = 256  # Feedforward dimension
    dropout: float = 0.1
    
    # Temporal context
    context_length: int = 5  # K lags of neural history
    
    # Behavior output
    n_beh: int = 6  # Eigenworm modes
    
    # Training
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 200
    patience: int = 25
    
    # Pooling strategy
    pooling: str = "pma"  # "pma" (Set Transformer) or "mean" (DeepSet)
    
    # Free-run evaluation
    reclamp_frames: int = 10


# ══════════════════════════════════════════════════════════════════════════════
# SET TRANSFORMER BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════════════════════

class MultiheadAttentionBlock(nn.Module):
    """Multihead Attention Block (MAB) from Set Transformer paper."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, 
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Q: (B, Nq, D) query
        K: (B, Nk, D) key/value
        key_padding_mask: (B, Nk) True = ignore
        """
        # Pre-norm attention
        Q_norm = self.norm1(Q)
        K_norm = self.norm1(K)
        attn_out, _ = self.attn(Q_norm, K_norm, K_norm, key_padding_mask=key_padding_mask)
        H = Q + attn_out
        
        # Pre-norm feedforward
        H = H + self.ff(self.norm2(H))
        return H


class SetAttentionBlock(nn.Module):
    """Set Attention Block (SAB) - self-attention over set elements."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.mab = MultiheadAttentionBlock(d_model, n_heads, d_ff, dropout)
    
    def forward(self, X: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """X: (B, N, D), mask: (B, N) True = ignore"""
        return self.mab(X, X, key_padding_mask=mask)


class InducedSetAttentionBlock(nn.Module):
    """Induced Set Attention Block (ISAB) - reduces O(N²) to O(NM) via inducing points."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, 
                 n_inducing: int = 16, dropout: float = 0.1):
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(1, n_inducing, d_model) * 0.02)
        self.mab1 = MultiheadAttentionBlock(d_model, n_heads, d_ff, dropout)
        self.mab2 = MultiheadAttentionBlock(d_model, n_heads, d_ff, dropout)
    
    def forward(self, X: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """X: (B, N, D)"""
        B = X.size(0)
        I = self.inducing.expand(B, -1, -1)  # (B, M, D)
        H = self.mab1(I, X, key_padding_mask=mask)  # (B, M, D)
        return self.mab2(X, H)  # (B, N, D)


class PoolingByMultiheadAttention(nn.Module):
    """Pooling by Multihead Attention (PMA) - aggregate set to fixed-size output."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 n_outputs: int = 1, dropout: float = 0.1):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, n_outputs, d_model) * 0.02)
        self.mab = MultiheadAttentionBlock(d_model, n_heads, d_ff, dropout)
    
    def forward(self, X: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """X: (B, N, D) -> (B, n_outputs, D)"""
        B = X.size(0)
        S = self.seeds.expand(B, -1, -1)
        return self.mab(S, X, key_padding_mask=mask)


# ══════════════════════════════════════════════════════════════════════════════
# SET TRANSFORMER FOR BEHAVIOUR DECODING
# ══════════════════════════════════════════════════════════════════════════════

class SetTransformerBehaviourDecoder(nn.Module):
    """Set Transformer that predicts behaviour from variable neuron sets.
    
    Input per sample:
      - neural_activity: (N_observed, K) - K timesteps of activity per neuron
      - neuron_indices: (N_observed,) - atlas index for each observed neuron
      - beh_context: (K, n_beh) - behaviour history (optional)
    
    Output:
      - behaviour: (n_beh,) - predicted next-step behaviour
    """
    
    def __init__(self, cfg: SetTransformerConfig):
        super().__init__()
        self.cfg = cfg
        
        # Neuron identity embeddings (learnable)
        self.neuron_embed = nn.Embedding(cfg.n_atlas, cfg.neuron_embed_dim)
        
        # Project neural activity + identity to d_model
        # Input: K timesteps of activity + neuron embedding
        neural_input_dim = cfg.context_length + cfg.neuron_embed_dim
        self.neural_proj = nn.Sequential(
            nn.Linear(neural_input_dim, cfg.d_model),
            nn.GELU(),
            nn.LayerNorm(cfg.d_model),
        )
        
        # Behaviour context encoding (optional)
        self.beh_proj = nn.Linear(cfg.context_length * cfg.n_beh, cfg.d_model)
        
        # Set Transformer encoder (process neurons as a set)
        self.encoder_layers = nn.ModuleList([
            SetAttentionBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_encoder_layers)
        ])
        
        # Pooling: aggregate neuron set to fixed representation
        if cfg.pooling == "mean":
            # DeepSet-style mean pooling
            self.pma = None
            self.pool_mlp = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            )
            pool_out_dim = cfg.d_model
        else:
            # PMA pooling (Set Transformer default)
            self.pma = PoolingByMultiheadAttention(
                cfg.d_model, cfg.n_heads, cfg.d_ff, n_outputs=2, dropout=cfg.dropout
            )
            self.pool_mlp = None
            pool_out_dim = cfg.d_model * 2
        
        # Decoder layers (optional refinement, PMA only)
        self.decoder_layers = nn.ModuleList([
            SetAttentionBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_decoder_layers)
        ]) if cfg.pooling != "mean" else nn.ModuleList()
        
        # Final behaviour prediction head
        self.beh_head = nn.Sequential(
            nn.Linear(pool_out_dim + cfg.d_model, cfg.d_model),  # pooled + beh context
            nn.GELU(),
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.n_beh),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
    
    def forward(
        self,
        neural_activity: torch.Tensor,  # (B, N_max, K) padded
        neuron_indices: torch.Tensor,   # (B, N_max) long, padded with -1
        beh_context: torch.Tensor,      # (B, K, n_beh)
        neuron_mask: Optional[torch.Tensor] = None,  # (B, N_max) True = padding
    ) -> torch.Tensor:
        """
        neural_activity: (B, N_max, K) - K timesteps per neuron, padded
        neuron_indices: (B, N_max) - atlas indices, -1 for padding
        beh_context: (B, K, n_beh) - behaviour history
        neuron_mask: (B, N_max) - True where neuron is padding
        
        Returns: (B, n_beh)
        """
        B, N_max, K = neural_activity.shape
        
        # Create mask from indices if not provided
        if neuron_mask is None:
            neuron_mask = (neuron_indices < 0)  # (B, N_max)
        
        # Clamp indices for embedding lookup (invalid will be masked anyway)
        safe_indices = neuron_indices.clamp(min=0)  # (B, N_max)
        
        # Get neuron identity embeddings
        neuron_emb = self.neuron_embed(safe_indices)  # (B, N_max, embed_dim)
        
        # Concatenate activity with identity
        # neural_activity: (B, N_max, K)
        # neuron_emb: (B, N_max, embed_dim)
        x = torch.cat([neural_activity, neuron_emb], dim=-1)  # (B, N_max, K + embed_dim)
        
        # Project to d_model
        x = self.neural_proj(x)  # (B, N_max, d_model)
        
        # Zero out padding positions
        x = x.masked_fill(neuron_mask.unsqueeze(-1), 0.0)
        
        # Set Transformer encoder
        for layer in self.encoder_layers:
            x = layer(x, mask=neuron_mask)
        
        # Pool to fixed size
        if self.pma is not None:
            pooled = self.pma(x, mask=neuron_mask)  # (B, 2, d_model)
            pooled = pooled.reshape(B, -1)  # (B, 2*d_model)
        else:
            # DeepSet mean pooling
            x_masked = x.masked_fill(neuron_mask.unsqueeze(-1), 0.0)
            valid_count = (~neuron_mask).float().sum(dim=-1, keepdim=True).clamp(min=1)
            pooled = x_masked.sum(dim=1) / valid_count  # (B, d_model)
            pooled = self.pool_mlp(pooled)
        
        # Encode behaviour context
        beh_flat = beh_context.reshape(B, -1)  # (B, K * n_beh)
        beh_enc = self.beh_proj(beh_flat)  # (B, d_model)
        
        # Decoder refinement (PMA only)
        for layer in self.decoder_layers:
            pooled_reshaped = pooled.reshape(B, 2, self.cfg.d_model)
            pooled_reshaped = layer(pooled_reshaped)
            pooled = pooled_reshaped.reshape(B, -1)
        
        # Combine and predict
        combined = torch.cat([pooled, beh_enc], dim=-1)  # (B, 2*d_model + d_model)
        beh_pred = self.beh_head(combined)  # (B, n_beh)
        
        return beh_pred


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

class MultiWormBehaviourDataset(Dataset):
    """Dataset that pools samples from multiple worms.
    
    Each sample is a (neural_context, beh_context, beh_target, neuron_indices) tuple.
    Different worms have different neuron subsets - handled via variable-length sets.
    """
    
    def __init__(
        self,
        worm_data: List[Dict[str, Any]],
        context_length: int = 5,
        n_beh: int = 6,
    ):
        """
        worm_data: List of dicts with keys:
            - 'u': (T, N) neural activity
            - 'b': (T, n_beh) behaviour
            - 'atlas_idx': (N,) int array of atlas indices
            - 'worm_id': str
        """
        self.K = context_length
        self.n_beh = n_beh
        
        # Build sample index: (worm_idx, time_idx)
        self.samples = []
        self.worm_data = worm_data
        
        for worm_idx, data in enumerate(worm_data):
            T = data['u'].shape[0]
            for t in range(self.K, T):
                self.samples.append((worm_idx, t))
        
        # Find max neurons across all worms for padding
        self.max_neurons = max(d['u'].shape[1] for d in worm_data)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        worm_idx, t = self.samples[idx]
        data = self.worm_data[worm_idx]
        
        # Extract context windows
        u = data['u'][t - self.K : t]  # (K, N)
        b = data['b'][t - self.K : t]  # (K, n_beh)
        b_target = data['b'][t]        # (n_beh,)
        atlas_idx = data['atlas_idx']  # (N,)
        
        # Transpose neural: (K, N) -> (N, K) for set processing
        u = u.T  # (N, K)
        N = u.shape[0]
        
        # Pad to max_neurons
        u_padded = np.zeros((self.max_neurons, self.K), dtype=np.float32)
        u_padded[:N] = u
        
        idx_padded = np.full(self.max_neurons, -1, dtype=np.int64)
        idx_padded[:N] = atlas_idx
        
        mask = np.ones(self.max_neurons, dtype=bool)
        mask[:N] = False  # False = valid neuron
        
        return {
            'neural': torch.from_numpy(u_padded),
            'neuron_idx': torch.from_numpy(idx_padded),
            'neuron_mask': torch.from_numpy(mask),
            'beh_context': torch.from_numpy(b.astype(np.float32)),
            'beh_target': torch.from_numpy(b_target.astype(np.float32)),
            'worm_idx': worm_idx,
            't': t,
        }


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_worm_from_h5(h5_path: str, full_atlas: List[str], n_beh: int = 6) -> Dict[str, Any]:
    """Load one worm's data with atlas alignment."""
    with h5py.File(h5_path, 'r') as f:
        # Neural activity
        if 'stage1/u_mean' in f:
            u = f['stage1/u_mean'][:]
        else:
            u = f['gcamp/trace_array_original'][:]
        
        # Ensure (T, N)
        if u.shape[0] < u.shape[1]:
            u = u.T
        
        # Behaviour (eigenworms)
        for ds in ['behaviour/eigenworms_stephens', 'behaviour/eigenworms_calc_6',
                   'behavior/eigenworm_projection']:
            if ds in f:
                b = f[ds][:]
                break
        else:
            raise ValueError(f"No eigenworm data in {h5_path}")
        
        if b.shape[0] < b.shape[1]:
            b = b.T
        b = b[:, :n_beh]
        
        # Neuron labels -> atlas indices
        labels = _read_neuron_labels(f)
        if labels is None:
            raise ValueError(f"No neuron labels in {h5_path}")
        
        matched, atlas_idx = _match_labels_to_atlas(labels, full_atlas)
    
    # Align lengths
    T = min(u.shape[0], b.shape[0])
    u = u[:T].astype(np.float32)
    b = b[:T].astype(np.float32)
    
    # Normalize neural activity per neuron
    u_mean = u.mean(axis=0, keepdims=True)
    u_std = u.std(axis=0, keepdims=True) + 1e-6
    u = (u - u_mean) / u_std
    
    # Normalize behaviour
    b_mean = b.mean(axis=0, keepdims=True)
    b_std = b.std(axis=0, keepdims=True) + 1e-6
    b = (b - b_mean) / b_std
    
    worm_id = Path(h5_path).stem
    
    return {
        'u': u,
        'b': b,
        'b_mean': b_mean.squeeze(),
        'b_std': b_std.squeeze(),
        'atlas_idx': atlas_idx,
        'matched_labels': matched,
        'worm_id': worm_id,
        'h5_path': h5_path,
    }


def _load_motor_neuron_names() -> set:
    """Load motor neuron names from file."""
    motor_path = Path(__file__).parent.parent / "data/used/masks+motor neurons/motor_neurons_with_control.txt"
    if motor_path.exists():
        return set(ln.strip() for ln in motor_path.read_text().splitlines() if ln.strip())
    return set()


def load_all_worms(h5_dir: str, n_beh: int = 6, motor_only: bool = False,
                   worm_ids: List[str] = None) -> Tuple[List[Dict], List[str]]:
    """Load all worms from directory."""
    full_atlas = _load_full_atlas()
    h5_files = sorted(glob.glob(os.path.join(h5_dir, "*.h5")))
    
    # Filter by worm_ids if provided
    if worm_ids:
        worm_id_set = set(worm_ids)
        h5_files = [f for f in h5_files if Path(f).stem in worm_id_set]
    
    print(f"Found {len(h5_files)} H5 files")
    
    # Load motor neuron names if needed
    motor_names = _load_motor_neuron_names() if motor_only else None
    if motor_only:
        print(f"Motor-only mode: {len(motor_names)} motor neuron names")
    
    worm_data = []
    for h5_path in h5_files:
        try:
            data = load_worm_from_h5(h5_path, full_atlas, n_beh)
            
            # Filter to motor neurons if requested
            if motor_only and motor_names:
                motor_mask = np.array([label in motor_names for label in data['matched_labels']])
                if motor_mask.sum() == 0:
                    print(f"  SKIP {data['worm_id']}: no motor neurons found")
                    continue
                    
                data['u'] = data['u'][:, motor_mask]
                data['atlas_idx'] = data['atlas_idx'][motor_mask]
                data['matched_labels'] = [l for l, m in zip(data['matched_labels'], motor_mask) if m]
            
            worm_data.append(data)
            label = "motor" if motor_only else "all"
            print(f"  Loaded {data['worm_id']}: T={data['u'].shape[0]}, N={data['u'].shape[1]} ({label})")
        except Exception as e:
            print(f"  Failed {Path(h5_path).stem}: {e}")
    
    return worm_data, full_atlas


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_set_transformer(
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: SetTransformerConfig,
    device: torch.device,
) -> SetTransformerBehaviourDecoder:
    """Train the Set Transformer model."""
    
    model = SetTransformerBehaviourDecoder(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params:,} parameters")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_epochs)
    
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    for epoch in range(cfg.max_epochs):
        # Training
        model.train()
        train_loss = 0.0
        n_train = 0
        
        for batch in train_loader:
            neural = batch['neural'].to(device)
            neuron_idx = batch['neuron_idx'].to(device)
            neuron_mask = batch['neuron_mask'].to(device)
            beh_context = batch['beh_context'].to(device)
            beh_target = batch['beh_target'].to(device)
            
            optimizer.zero_grad()
            pred = model(neural, neuron_idx, beh_context, neuron_mask)
            loss = F.mse_loss(pred, beh_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item() * neural.size(0)
            n_train += neural.size(0)
        
        train_loss /= n_train
        
        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        
        with torch.no_grad():
            for batch in val_loader:
                neural = batch['neural'].to(device)
                neuron_idx = batch['neuron_idx'].to(device)
                neuron_mask = batch['neuron_mask'].to(device)
                beh_context = batch['beh_context'].to(device)
                beh_target = batch['beh_target'].to(device)
                
                pred = model(neural, neuron_idx, beh_context, neuron_mask)
                loss = F.mse_loss(pred, beh_target)
                
                val_loss += loss.item() * neural.size(0)
                n_val += neural.size(0)
        
        val_loss /= n_val
        scheduler.step()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: train={train_loss:.4f}, val={val_loss:.4f}, best={best_val_loss:.4f}")
        
        if patience_counter >= cfg.patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    if best_state:
        model.load_state_dict(best_state)
    model.eval().cpu()
    
    return model


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_freerun(
    model: SetTransformerBehaviourDecoder,
    worm_data: Dict[str, Any],
    cfg: SetTransformerConfig,
    device: torch.device,
    reclamp_every: int = 10,
) -> np.ndarray:
    """Free-run evaluation on a single worm with periodic re-clamping."""
    model.eval().to(device)
    
    u = worm_data['u']  # (T, N)
    b = worm_data['b']  # (T, n_beh)
    atlas_idx = worm_data['atlas_idx']
    
    T, N = u.shape
    K = cfg.context_length
    
    # Prepare neuron indices (constant for this worm)
    neuron_idx = torch.from_numpy(atlas_idx).long().unsqueeze(0).to(device)  # (1, N)
    neuron_mask = torch.zeros(1, N, dtype=torch.bool, device=device)  # No padding
    
    predictions = np.full((T, cfg.n_beh), np.nan)
    
    with torch.no_grad():
        for t in range(K, T):
            # Get neural context (always from ground truth)
            u_ctx = torch.from_numpy(u[t-K:t].T).float().unsqueeze(0).to(device)  # (1, N, K)
            
            # Get behaviour context (from predictions or GT depending on reclamp)
            if t == K or (t - K) % reclamp_every == 0:
                # Re-clamp to ground truth
                b_ctx = torch.from_numpy(b[t-K:t]).float().unsqueeze(0).to(device)  # (1, K, n_beh)
            else:
                # Use previous predictions
                b_ctx_np = np.zeros((K, cfg.n_beh), dtype=np.float32)
                for i, ti in enumerate(range(t-K, t)):
                    if np.isnan(predictions[ti]).any():
                        b_ctx_np[i] = b[ti]
                    else:
                        b_ctx_np[i] = predictions[ti]
                b_ctx = torch.from_numpy(b_ctx_np).float().unsqueeze(0).to(device)
            
            # Predict
            pred = model(u_ctx, neuron_idx, b_ctx, neuron_mask)
            predictions[t] = pred.cpu().numpy().squeeze()
    
    return predictions


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Compute R² and correlation per mode."""
    valid = ~np.isnan(pred).any(axis=1)
    pred_v = pred[valid]
    gt_v = gt[valid]
    
    n_modes = pred_v.shape[1]
    r2_per = []
    corr_per = []
    
    for i in range(n_modes):
        ss_res = np.sum((gt_v[:, i] - pred_v[:, i]) ** 2)
        ss_tot = np.sum((gt_v[:, i] - gt_v[:, i].mean()) ** 2) + 1e-12
        r2_per.append(1 - ss_res / ss_tot)
        corr_per.append(np.corrcoef(gt_v[:, i], pred_v[:, i])[0, 1])
    
    return {
        'r2_mean': float(np.mean(r2_per)),
        'r2_per_mode': r2_per,
        'corr_mean': float(np.mean(corr_per)),
        'corr_per_mode': corr_per,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def leave_one_out_cv(
    worm_data: List[Dict[str, Any]],
    cfg: SetTransformerConfig,
    device: torch.device,
    out_dir: Path,
) -> Dict[str, Any]:
    """Leave-one-worm-out cross-validation.
    
    For each worm:
    1. Train on all OTHER worms
    2. Evaluate (free-run) on the held-out worm
    
    This tests true generalization to unseen worms.
    """
    n_worms = len(worm_data)
    results = []
    
    for hold_out_idx in range(n_worms):
        held_out = worm_data[hold_out_idx]
        train_worms = [w for i, w in enumerate(worm_data) if i != hold_out_idx]
        
        print(f"\n[{hold_out_idx+1}/{n_worms}] Hold out: {held_out['worm_id']} "
              f"(N={held_out['u'].shape[1]}), train on {len(train_worms)} worms")
        
        # Create datasets
        # Split training worms temporally for train/val
        train_data_split = []
        val_data_split = []
        
        for w in train_worms:
            T = w['u'].shape[0]
            t_split = int(T * 0.85)
            train_data_split.append({
                'u': w['u'][:t_split],
                'b': w['b'][:t_split],
                'atlas_idx': w['atlas_idx'],
                'worm_id': w['worm_id'],
            })
            val_data_split.append({
                'u': w['u'][t_split:],
                'b': w['b'][t_split:],
                'atlas_idx': w['atlas_idx'],
                'worm_id': w['worm_id'],
            })
        
        train_ds = MultiWormBehaviourDataset(train_data_split, cfg.context_length, cfg.n_beh)
        val_ds = MultiWormBehaviourDataset(val_data_split, cfg.context_length, cfg.n_beh)
        
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
        
        print(f"  Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
        
        # Train
        model = train_set_transformer(train_loader, val_loader, cfg, device)
        
        # Evaluate on held-out worm (free-run)
        pred = evaluate_freerun(model, held_out, cfg, device, cfg.reclamp_frames)
        metrics = compute_metrics(pred, held_out['b'])
        
        print(f"  Held-out {held_out['worm_id']}: R²={metrics['r2_mean']:.3f}, "
              f"Corr={metrics['corr_mean']:.3f}")
        
        results.append({
            'worm_id': held_out['worm_id'],
            'n_neurons': held_out['u'].shape[1],
            **metrics,
        })
        
        # Save per-worm results
        _plot_traces(pred, held_out['b'], held_out['worm_id'], out_dir, cfg)
    
    return results


def _plot_traces(pred: np.ndarray, gt: np.ndarray, worm_id: str, out_dir: Path, cfg: SetTransformerConfig):
    """Plot behaviour traces."""
    T, n_modes = gt.shape
    fig, axes = plt.subplots(n_modes, 1, figsize=(12, 2 * n_modes), sharex=True)
    
    # Plot frames 400-600 or whatever is available
    fs, fe = 400, min(600, T)
    t_ax = np.arange(fs, fe) / 4.0  # Assuming 4Hz
    
    for i, ax in enumerate(axes):
        ax.plot(t_ax, gt[fs:fe, i], 'k-', lw=1, label='GT')
        ax.plot(t_ax, pred[fs:fe, i], 'r-', lw=1, alpha=0.8, label='SetTRF')
        ax.set_ylabel(f'Mode {i+1}')
        if i == 0:
            ax.legend(loc='upper right')
    
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f'Set Transformer (K={cfg.context_length}) — {worm_id}', fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_dir / f'traces_settrf_{worm_id}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_summary(results: List[Dict], out_dir: Path, cfg: SetTransformerConfig):
    """Plot summary across all worms."""
    worm_ids = [r['worm_id'] for r in results]
    r2_means = [r['r2_mean'] for r in results]
    corr_means = [r['corr_mean'] for r in results]
    n_neurons = [r['n_neurons'] for r in results]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Bar plot of R²
    ax = axes[0]
    ax.bar(range(len(worm_ids)), r2_means, color='steelblue')
    ax.axhline(np.mean(r2_means), color='red', linestyle='--', label=f'Mean={np.mean(r2_means):.3f}')
    ax.set_xticks(range(len(worm_ids)))
    ax.set_xticklabels([w[:10] for w in worm_ids], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('R²')
    ax.set_title('Leave-one-out R² per worm')
    ax.legend()
    
    # Bar plot of Correlation
    ax = axes[1]
    ax.bar(range(len(worm_ids)), corr_means, color='coral')
    ax.axhline(np.mean(corr_means), color='red', linestyle='--', label=f'Mean={np.mean(corr_means):.3f}')
    ax.set_xticks(range(len(worm_ids)))
    ax.set_xticklabels([w[:10] for w in worm_ids], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Correlation')
    ax.set_title('Leave-one-out Correlation per worm')
    ax.legend()
    
    # Scatter: R² vs N_neurons
    ax = axes[2]
    ax.scatter(n_neurons, r2_means, c='steelblue', s=50, alpha=0.7)
    ax.set_xlabel('Number of neurons')
    ax.set_ylabel('R²')
    ax.set_title('R² vs Neuron count')
    
    # Add correlation line
    z = np.polyfit(n_neurons, r2_means, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(n_neurons), max(n_neurons), 100)
    ax.plot(x_line, p(x_line), 'r--', alpha=0.5)
    
    fig.suptitle(f'Set Transformer Leave-One-Out CV (K={cfg.context_length})', fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_dir / 'summary_settrf_loo.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nSaved summary to {out_dir / 'summary_settrf_loo.png'}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Set Transformer multi-worm behaviour decoder")
    parser.add_argument("--h5_dir", type=str, required=True, help="Directory with H5 files")
    parser.add_argument("--out", type=str, default="output_plots/behaviour_decoder/set_transformer",
                        help="Output directory")
    parser.add_argument("--context_length", type=int, default=5, help="Context length K")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of encoder layers")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--max_epochs", type=int, default=200, help="Max training epochs")
    parser.add_argument("--max_worms", type=int, default=None, help="Limit number of worms (for quick testing)")
    parser.add_argument("--motor_only", action="store_true", help="Use motor neurons only")
    parser.add_argument("--pooling", type=str, default="pma", choices=["pma", "mean"],
                        help="Pooling strategy: pma (Set Transformer) or mean (DeepSet)")
    parser.add_argument("--worm_ids", type=str, nargs="+", default=None, help="Specific worm IDs to process")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Configuration
    cfg = SetTransformerConfig(
        context_length=args.context_length,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_encoder_layers=args.n_layers,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        pooling=args.pooling,
    )
    
    # Load data
    print(f"\nLoading worms from {args.h5_dir}")
    worm_data, full_atlas = load_all_worms(
        args.h5_dir, 
        motor_only=args.motor_only,
        worm_ids=args.worm_ids
    )
    
    # Limit worms if requested
    if args.max_worms and args.max_worms < len(worm_data):
        worm_data = worm_data[:args.max_worms]
        print(f"Limited to {len(worm_data)} worms (--max_worms={args.max_worms})")
    
    print(f"Using {len(worm_data)} worms, atlas has {len(full_atlas)} neurons")
    
    # Run leave-one-out CV
    print("\n" + "="*60)
    print("LEAVE-ONE-WORM-OUT CROSS-VALIDATION")
    print("="*60)
    
    results = leave_one_out_cv(worm_data, cfg, device, out_dir)
    
    # Summary
    _plot_summary(results, out_dir, cfg)
    
    # Save results
    with open(out_dir / 'results_loo.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    r2_all = [r['r2_mean'] for r in results]
    corr_all = [r['corr_mean'] for r in results]
    print(f"Mean R²: {np.mean(r2_all):.3f} ± {np.std(r2_all):.3f}")
    print(f"Mean Corr: {np.mean(corr_all):.3f} ± {np.std(corr_all):.3f}")
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
