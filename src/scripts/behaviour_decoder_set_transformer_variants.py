#!/usr/bin/env python
"""Test different Set Transformer architecture variants for behavior decoding.

Architecture variants to test:
1. Baseline: Current PMA pooling with 2 seeds
2. DeepSet: Mean pooling instead of attention-based pooling
3. TemporalConv: 1D conv over time before set attention
4. CrossAttention: Behavior query attends to neural representations
5. Larger: Deeper model with more capacity
6. NeuronMLP: MLP per neuron before set attention

Author: Copilot
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
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
# SHARED COMPONENTS
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
        Q_norm = self.norm1(Q)
        K_norm = self.norm1(K)
        attn_out, _ = self.attn(Q_norm, K_norm, K_norm, key_padding_mask=key_padding_mask)
        x = Q + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class SetAttentionBlock(nn.Module):
    """Self-attention over set elements."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.mab = MultiheadAttentionBlock(d_model, n_heads, d_ff, dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.mab(x, x, key_padding_mask=mask)


class PoolingByMultiheadAttention(nn.Module):
    """PMA: Pool set to fixed size using learned seed vectors."""
    def __init__(self, d_model: int, n_heads: int, n_seeds: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, n_seeds, d_model) * 0.02)
        self.mab = MultiheadAttentionBlock(d_model, n_heads, d_ff, dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = x.size(0)
        seeds = self.seeds.expand(B, -1, -1)
        return self.mab(seeds, x, key_padding_mask=mask)


# ══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE VARIANTS
# ══════════════════════════════════════════════════════════════════════════════

class SetTransformerBaseline(nn.Module):
    """Variant 1: Baseline - PMA pooling (current architecture)."""
    
    def __init__(self, n_atlas: int = 302, K: int = 5, n_beh: int = 6,
                 d_model: int = 128, n_heads: int = 4, n_layers: int = 2,
                 neuron_embed_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.name = "baseline"
        self.K = K
        self.d_model = d_model
        
        # Neuron identity embeddings
        self.neuron_embed = nn.Embedding(n_atlas + 1, neuron_embed_dim, padding_idx=n_atlas)
        
        # Project [temporal_features; neuron_embed] to d_model
        self.neural_proj = nn.Linear(K + neuron_embed_dim, d_model)
        
        # Set Transformer encoder
        d_ff = d_model * 2
        self.encoder_layers = nn.ModuleList([
            SetAttentionBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        
        # Pool to fixed size
        self.pooling = PoolingByMultiheadAttention(d_model, n_heads, 2, d_ff, dropout)
        
        # Behavior context encoder
        self.beh_proj = nn.Linear(K * n_beh, d_model)
        
        # Output head
        self.beh_head = nn.Sequential(
            nn.Linear(2 * d_model + d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_beh),
        )
    
    def forward(self, neural_context, neuron_idx, beh_context, neuron_mask):
        B, N, K = neural_context.shape
        
        # Get neuron embeddings
        idx_safe = neuron_idx.clamp(min=0)
        idx_safe[neuron_idx < 0] = self.neuron_embed.padding_idx
        neuron_emb = self.neuron_embed(idx_safe)
        
        # Combine temporal features with neuron identity
        x = torch.cat([neural_context, neuron_emb], dim=-1)
        x = self.neural_proj(x)
        x = x.masked_fill(neuron_mask.unsqueeze(-1), 0.0)
        
        # Encoder
        for layer in self.encoder_layers:
            x = layer(x, mask=neuron_mask)
        
        # Pool
        pooled = self.pooling(x, mask=neuron_mask)
        pooled = pooled.reshape(B, -1)
        
        # Behavior context
        beh_enc = self.beh_proj(beh_context.reshape(B, -1))
        
        # Predict
        combined = torch.cat([pooled, beh_enc], dim=-1)
        return self.beh_head(combined)


class DeepSetModel(nn.Module):
    """Variant 2: DeepSet - Mean pooling instead of attention."""
    
    def __init__(self, n_atlas: int = 302, K: int = 5, n_beh: int = 6,
                 d_model: int = 128, n_heads: int = 4, n_layers: int = 2,
                 neuron_embed_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.name = "deepset"
        self.K = K
        self.d_model = d_model
        
        # Neuron identity embeddings
        self.neuron_embed = nn.Embedding(n_atlas + 1, neuron_embed_dim, padding_idx=n_atlas)
        
        # Per-element MLP (phi)
        self.phi = nn.Sequential(
            nn.Linear(K + neuron_embed_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Post-pooling MLP (rho)
        self.rho = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Behavior context encoder
        self.beh_proj = nn.Linear(K * n_beh, d_model)
        
        # Output head
        self.beh_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_beh),
        )
    
    def forward(self, neural_context, neuron_idx, beh_context, neuron_mask):
        B, N, K = neural_context.shape
        
        # Get neuron embeddings
        idx_safe = neuron_idx.clamp(min=0)
        idx_safe[neuron_idx < 0] = self.neuron_embed.padding_idx
        neuron_emb = self.neuron_embed(idx_safe)
        
        # Combine and transform per-element
        x = torch.cat([neural_context, neuron_emb], dim=-1)
        x = self.phi(x)  # (B, N, d_model)
        
        # Mask invalid neurons
        x = x.masked_fill(neuron_mask.unsqueeze(-1), 0.0)
        
        # Mean pooling (accounting for mask)
        valid_count = (~neuron_mask).float().sum(dim=-1, keepdim=True).clamp(min=1)  # (B, 1)
        pooled = x.sum(dim=1) / valid_count  # (B, d_model)
        
        # Post-pooling transformation
        pooled = self.rho(pooled)
        
        # Behavior context
        beh_enc = self.beh_proj(beh_context.reshape(B, -1))
        
        # Predict
        combined = torch.cat([pooled, beh_enc], dim=-1)
        return self.beh_head(combined)


class TemporalConvSetTransformer(nn.Module):
    """Variant 3: 1D conv over time before set attention."""
    
    def __init__(self, n_atlas: int = 302, K: int = 5, n_beh: int = 6,
                 d_model: int = 128, n_heads: int = 4, n_layers: int = 2,
                 neuron_embed_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.name = "temporal_conv"
        self.K = K
        self.d_model = d_model
        
        # Neuron identity embeddings
        self.neuron_embed = nn.Embedding(n_atlas + 1, neuron_embed_dim, padding_idx=n_atlas)
        
        # Temporal convolution (per neuron)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(1, d_model // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model // 2, d_model // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),  # Pool temporal to 1
        )
        
        # Project [conv_features; neuron_embed] to d_model
        self.neural_proj = nn.Linear(d_model // 2 + neuron_embed_dim, d_model)
        
        # Set Transformer encoder
        d_ff = d_model * 2
        self.encoder_layers = nn.ModuleList([
            SetAttentionBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        
        # Pool to fixed size
        self.pooling = PoolingByMultiheadAttention(d_model, n_heads, 2, d_ff, dropout)
        
        # Behavior context encoder
        self.beh_proj = nn.Linear(K * n_beh, d_model)
        
        # Output head
        self.beh_head = nn.Sequential(
            nn.Linear(2 * d_model + d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_beh),
        )
    
    def forward(self, neural_context, neuron_idx, beh_context, neuron_mask):
        B, N, K = neural_context.shape
        
        # Temporal convolution per neuron
        # Reshape: (B, N, K) -> (B*N, 1, K)
        x = neural_context.reshape(B * N, 1, K)
        x = self.temporal_conv(x)  # (B*N, d_model//2, 1)
        x = x.squeeze(-1).reshape(B, N, -1)  # (B, N, d_model//2)
        
        # Get neuron embeddings
        idx_safe = neuron_idx.clamp(min=0)
        idx_safe[neuron_idx < 0] = self.neuron_embed.padding_idx
        neuron_emb = self.neuron_embed(idx_safe)
        
        # Combine
        x = torch.cat([x, neuron_emb], dim=-1)
        x = self.neural_proj(x)
        x = x.masked_fill(neuron_mask.unsqueeze(-1), 0.0)
        
        # Encoder
        for layer in self.encoder_layers:
            x = layer(x, mask=neuron_mask)
        
        # Pool
        pooled = self.pooling(x, mask=neuron_mask)
        pooled = pooled.reshape(B, -1)
        
        # Behavior context
        beh_enc = self.beh_proj(beh_context.reshape(B, -1))
        
        # Predict
        combined = torch.cat([pooled, beh_enc], dim=-1)
        return self.beh_head(combined)


class CrossAttentionSetTransformer(nn.Module):
    """Variant 4: Behavior query tokens attend to neural representations."""
    
    def __init__(self, n_atlas: int = 302, K: int = 5, n_beh: int = 6,
                 d_model: int = 128, n_heads: int = 4, n_layers: int = 2,
                 neuron_embed_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.name = "cross_attention"
        self.K = K
        self.n_beh = n_beh
        self.d_model = d_model
        
        # Neuron identity embeddings
        self.neuron_embed = nn.Embedding(n_atlas + 1, neuron_embed_dim, padding_idx=n_atlas)
        
        # Project to d_model
        self.neural_proj = nn.Linear(K + neuron_embed_dim, d_model)
        
        # Set Transformer encoder (self-attention over neurons)
        d_ff = d_model * 2
        self.encoder_layers = nn.ModuleList([
            SetAttentionBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        
        # Behavior query tokens (one per behavior mode)
        self.beh_queries = nn.Parameter(torch.randn(1, n_beh, d_model) * 0.02)
        
        # Behavior context encoder
        self.beh_context_proj = nn.Linear(K, d_model)
        
        # Cross-attention: behavior queries attend to neurons
        self.cross_attn = MultiheadAttentionBlock(d_model, n_heads, d_ff, dropout)
        
        # Per-mode output heads
        self.beh_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
            ) for _ in range(n_beh)
        ])
    
    def forward(self, neural_context, neuron_idx, beh_context, neuron_mask):
        B, N, K = neural_context.shape
        
        # Get neuron embeddings
        idx_safe = neuron_idx.clamp(min=0)
        idx_safe[neuron_idx < 0] = self.neuron_embed.padding_idx
        neuron_emb = self.neuron_embed(idx_safe)
        
        # Combine and project
        x = torch.cat([neural_context, neuron_emb], dim=-1)
        x = self.neural_proj(x)
        x = x.masked_fill(neuron_mask.unsqueeze(-1), 0.0)
        
        # Self-attention over neurons
        for layer in self.encoder_layers:
            x = layer(x, mask=neuron_mask)
        
        # Behavior queries attend to neural representations
        queries = self.beh_queries.expand(B, -1, -1)  # (B, n_beh, d_model)
        beh_repr = self.cross_attn(queries, x, key_padding_mask=neuron_mask)  # (B, n_beh, d_model)
        
        # Add behavior context per mode
        beh_ctx = beh_context.permute(0, 2, 1)  # (B, n_beh, K)
        beh_ctx_enc = self.beh_context_proj(beh_ctx)  # (B, n_beh, d_model)
        
        # Combine and predict each mode
        combined = torch.cat([beh_repr, beh_ctx_enc], dim=-1)  # (B, n_beh, 2*d_model)
        
        outputs = []
        for i, head in enumerate(self.beh_heads):
            out_i = head(combined[:, i])  # (B, 1)
            outputs.append(out_i)
        
        return torch.cat(outputs, dim=-1)  # (B, n_beh)


class LargerSetTransformer(nn.Module):
    """Variant 5: Larger model with more capacity."""
    
    def __init__(self, n_atlas: int = 302, K: int = 5, n_beh: int = 6,
                 d_model: int = 256, n_heads: int = 8, n_layers: int = 4,
                 neuron_embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.name = "larger"
        self.K = K
        self.d_model = d_model
        
        # Neuron identity embeddings (larger)
        self.neuron_embed = nn.Embedding(n_atlas + 1, neuron_embed_dim, padding_idx=n_atlas)
        
        # Project to d_model
        self.neural_proj = nn.Linear(K + neuron_embed_dim, d_model)
        
        # Deeper encoder
        d_ff = d_model * 4
        self.encoder_layers = nn.ModuleList([
            SetAttentionBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        
        # Pool with more seeds
        self.pooling = PoolingByMultiheadAttention(d_model, n_heads, 4, d_ff, dropout)
        
        # Behavior context encoder
        self.beh_proj = nn.Sequential(
            nn.Linear(K * n_beh, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        
        # Deeper output head
        self.beh_head = nn.Sequential(
            nn.Linear(4 * d_model + d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_beh),
        )
    
    def forward(self, neural_context, neuron_idx, beh_context, neuron_mask):
        B, N, K = neural_context.shape
        
        # Get neuron embeddings
        idx_safe = neuron_idx.clamp(min=0)
        idx_safe[neuron_idx < 0] = self.neuron_embed.padding_idx
        neuron_emb = self.neuron_embed(idx_safe)
        
        # Combine and project
        x = torch.cat([neural_context, neuron_emb], dim=-1)
        x = self.neural_proj(x)
        x = x.masked_fill(neuron_mask.unsqueeze(-1), 0.0)
        
        # Encoder
        for layer in self.encoder_layers:
            x = layer(x, mask=neuron_mask)
        
        # Pool
        pooled = self.pooling(x, mask=neuron_mask)
        pooled = pooled.reshape(B, -1)
        
        # Behavior context
        beh_enc = self.beh_proj(beh_context.reshape(B, -1))
        
        # Predict
        combined = torch.cat([pooled, beh_enc], dim=-1)
        return self.beh_head(combined)


class NeuronMLPSetTransformer(nn.Module):
    """Variant 6: Per-neuron MLP before set attention."""
    
    def __init__(self, n_atlas: int = 302, K: int = 5, n_beh: int = 6,
                 d_model: int = 128, n_heads: int = 4, n_layers: int = 2,
                 neuron_embed_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.name = "neuron_mlp"
        self.K = K
        self.d_model = d_model
        
        # Neuron identity embeddings
        self.neuron_embed = nn.Embedding(n_atlas + 1, neuron_embed_dim, padding_idx=n_atlas)
        
        # Per-neuron MLP (shared weights, but uses neuron embedding)
        self.neuron_mlp = nn.Sequential(
            nn.Linear(K + neuron_embed_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        
        # Set attention (lighter)
        d_ff = d_model * 2
        self.encoder_layers = nn.ModuleList([
            SetAttentionBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        
        # Pool
        self.pooling = PoolingByMultiheadAttention(d_model, n_heads, 2, d_ff, dropout)
        
        # Behavior context encoder
        self.beh_proj = nn.Linear(K * n_beh, d_model)
        
        # Output head
        self.beh_head = nn.Sequential(
            nn.Linear(2 * d_model + d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_beh),
        )
    
    def forward(self, neural_context, neuron_idx, beh_context, neuron_mask):
        B, N, K = neural_context.shape
        
        # Get neuron embeddings
        idx_safe = neuron_idx.clamp(min=0)
        idx_safe[neuron_idx < 0] = self.neuron_embed.padding_idx
        neuron_emb = self.neuron_embed(idx_safe)
        
        # Per-neuron MLP
        x = torch.cat([neural_context, neuron_emb], dim=-1)
        x = self.neuron_mlp(x)
        x = x.masked_fill(neuron_mask.unsqueeze(-1), 0.0)
        
        # Set attention
        for layer in self.encoder_layers:
            x = layer(x, mask=neuron_mask)
        
        # Pool
        pooled = self.pooling(x, mask=neuron_mask)
        pooled = pooled.reshape(B, -1)
        
        # Behavior context
        beh_enc = self.beh_proj(beh_context.reshape(B, -1))
        
        # Predict
        combined = torch.cat([pooled, beh_enc], dim=-1)
        return self.beh_head(combined)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING (same as original)
# ══════════════════════════════════════════════════════════════════════════════

def load_worm_from_h5(h5_path: str, full_atlas: List[str], n_beh: int = 6) -> Dict[str, Any]:
    """Load a single worm's data from H5 file."""
    with h5py.File(h5_path, 'r') as f:
        # Neural activity
        if 'gcamp/trace_array' in f:
            u = f['gcamp/trace_array'][:]
        else:
            u = f['gcamp/trace_array_original'][:]
        
        if u.shape[0] < u.shape[1]:
            u = u.T
        
        # Behaviour
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
        
        labels = _read_neuron_labels(f)
        if labels is None:
            raise ValueError(f"No neuron labels in {h5_path}")
        
        matched, atlas_idx = _match_labels_to_atlas(labels, full_atlas)
    
    T = min(u.shape[0], b.shape[0])
    u = u[:T].astype(np.float32)
    b = b[:T].astype(np.float32)
    
    # Normalize
    u_mean = u.mean(axis=0, keepdims=True)
    u_std = u.std(axis=0, keepdims=True) + 1e-6
    u = (u - u_mean) / u_std
    
    b_mean = b.mean(axis=0, keepdims=True)
    b_std = b.std(axis=0, keepdims=True) + 1e-6
    b = (b - b_mean) / b_std
    
    worm_id = Path(h5_path).stem
    
    return {
        'u': u, 'b': b, 'b_mean': b_mean.squeeze(), 'b_std': b_std.squeeze(),
        'atlas_idx': atlas_idx, 'matched_labels': matched, 'worm_id': worm_id,
        'h5_path': h5_path,
    }


class MultiWormDataset(Dataset):
    """Dataset that pools samples from multiple worms."""
    
    def __init__(self, worm_data: List[Dict], K: int = 5, n_beh: int = 6):
        self.K = K
        self.n_beh = n_beh
        self.samples = []
        self.worm_data = worm_data
        
        for worm_idx, data in enumerate(worm_data):
            T = data['u'].shape[0]
            for t in range(K, T):
                self.samples.append((worm_idx, t))
        
        self.max_neurons = max(d['u'].shape[1] for d in worm_data)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        worm_idx, t = self.samples[idx]
        data = self.worm_data[worm_idx]
        
        u = data['u'][t - self.K : t].T  # (N, K)
        b = data['b'][t - self.K : t]     # (K, n_beh)
        b_target = data['b'][t]           # (n_beh,)
        atlas_idx = data['atlas_idx']     # (N,)
        N = u.shape[0]
        
        # Pad
        u_padded = np.zeros((self.max_neurons, self.K), dtype=np.float32)
        u_padded[:N] = u
        
        idx_padded = np.full(self.max_neurons, -1, dtype=np.int64)
        idx_padded[:N] = atlas_idx
        
        mask = np.ones(self.max_neurons, dtype=bool)
        mask[:N] = False
        
        return {
            'neural': torch.from_numpy(u_padded),
            'neuron_idx': torch.from_numpy(idx_padded),
            'beh_context': torch.from_numpy(b.astype(np.float32)),
            'beh_target': torch.from_numpy(b_target.astype(np.float32)),
            'mask': torch.from_numpy(mask),
        }


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def train_model(model, train_loader, val_loader, device, max_epochs=100, patience=15, lr=1e-3):
    """Train a model with early stopping."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(max_epochs):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            neural = batch['neural'].to(device)
            neuron_idx = batch['neuron_idx'].to(device)
            beh_ctx = batch['beh_context'].to(device)
            beh_target = batch['beh_target'].to(device)
            mask = batch['mask'].to(device)
            
            optimizer.zero_grad()
            pred = model(neural, neuron_idx, beh_ctx, mask)
            loss = F.mse_loss(pred, beh_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                neural = batch['neural'].to(device)
                neuron_idx = batch['neuron_idx'].to(device)
                beh_ctx = batch['beh_context'].to(device)
                beh_target = batch['beh_target'].to(device)
                mask = batch['mask'].to(device)
                
                pred = model(neural, neuron_idx, beh_ctx, mask)
                val_loss += F.mse_loss(pred, beh_target).item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    if best_state:
        model.load_state_dict(best_state)
    
    return model, best_val_loss


def evaluate_freerun(model, worm_data, K, n_beh, device, reclamp_every=10):
    """Free-run evaluation on a single worm."""
    model.eval()
    
    u = worm_data['u']
    b = worm_data['b']
    atlas_idx = worm_data['atlas_idx']
    T, N = u.shape
    
    neuron_idx = torch.from_numpy(atlas_idx).long().unsqueeze(0).to(device)
    neuron_mask = torch.zeros(1, N, dtype=torch.bool, device=device)
    
    predictions = np.full((T, n_beh), np.nan)
    
    with torch.no_grad():
        for t in range(K, T):
            u_ctx = torch.from_numpy(u[t-K:t].T).float().unsqueeze(0).to(device)
            
            if t == K or (t - K) % reclamp_every == 0:
                b_ctx = torch.from_numpy(b[t-K:t]).float().unsqueeze(0).to(device)
            else:
                b_ctx_np = np.zeros((K, n_beh), dtype=np.float32)
                for i, ti in enumerate(range(t-K, t)):
                    if np.isnan(predictions[ti]).any():
                        b_ctx_np[i] = b[ti]
                    else:
                        b_ctx_np[i] = predictions[ti]
                b_ctx = torch.from_numpy(b_ctx_np).float().unsqueeze(0).to(device)
            
            pred = model(u_ctx, neuron_idx, b_ctx, neuron_mask)
            predictions[t] = pred.cpu().numpy().squeeze()
    
    return predictions


def compute_metrics(pred, gt):
    """Compute R² and correlation."""
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
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

VARIANTS = {
    'baseline': SetTransformerBaseline,
    'deepset': DeepSetModel,
    'temporal_conv': TemporalConvSetTransformer,
    'cross_attention': CrossAttentionSetTransformer,
    'larger': LargerSetTransformer,
    'neuron_mlp': NeuronMLPSetTransformer,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_dir', required=True, help='Directory with H5 files')
    parser.add_argument('--out', required=True, help='Output directory')
    parser.add_argument('--test_worms', nargs='+', default=None, 
                        help='Specific worm IDs to test (leave-one-out). If not specified, uses worst 5.')
    parser.add_argument('--variants', nargs='+', default=list(VARIANTS.keys()),
                        help='Variants to test')
    parser.add_argument('--K', type=int, default=5, help='Context length')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load all worms
    full_atlas = _load_full_atlas()
    h5_files = sorted(glob.glob(os.path.join(args.h5_dir, "*.h5")))
    
    all_worms = []
    for h5_path in h5_files:
        try:
            data = load_worm_from_h5(h5_path, full_atlas)
            all_worms.append(data)
        except Exception as e:
            print(f"Failed to load {h5_path}: {e}")
    
    print(f"Loaded {len(all_worms)} worms")
    
    # Select test worms (default: worst 5 from previous run)
    if args.test_worms is None:
        test_worm_ids = ['2023-01-19-15', '2023-01-17-14', '2023-01-13-07', 
                         '2023-01-19-01', '2023-01-19-22']
    else:
        test_worm_ids = args.test_worms
    
    print(f"Test worms: {test_worm_ids}")
    
    # Results storage
    results = {variant: {} for variant in args.variants}
    
    # Run each variant
    for variant_name in args.variants:
        print(f"\n{'='*70}")
        print(f"VARIANT: {variant_name}")
        print(f"{'='*70}")
        
        ModelClass = VARIANTS[variant_name]
        
        for test_worm_id in test_worm_ids:
            print(f"\n  Testing on: {test_worm_id}")
            
            # Split data: train on all except test worm
            train_worms = [w for w in all_worms if w['worm_id'] != test_worm_id]
            test_worm = next((w for w in all_worms if w['worm_id'] == test_worm_id), None)
            
            if test_worm is None:
                print(f"    Worm {test_worm_id} not found, skipping")
                continue
            
            # Create datasets
            train_dataset = MultiWormDataset(train_worms, K=args.K)
            
            # Split train into train/val (90/10)
            n_train = int(0.9 * len(train_dataset))
            train_subset, val_subset = torch.utils.data.random_split(
                train_dataset, [n_train, len(train_dataset) - n_train],
                generator=torch.Generator().manual_seed(42)
            )
            
            train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=0)
            
            # Create model
            model = ModelClass(K=args.K)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"    Model params: {n_params:,}")
            
            # Train
            t0 = time.time()
            model, val_loss = train_model(model, train_loader, val_loader, device, 
                                          max_epochs=args.max_epochs, patience=15)
            train_time = time.time() - t0
            print(f"    Train time: {train_time:.0f}s, Val loss: {val_loss:.4f}")
            
            # Evaluate
            pred = evaluate_freerun(model, test_worm, args.K, 6, device)
            metrics = compute_metrics(pred, test_worm['b'])
            
            print(f"    R²={metrics['r2_mean']:.3f}, Corr={metrics['corr_mean']:.3f}")
            
            results[variant_name][test_worm_id] = {
                'r2_mean': metrics['r2_mean'],
                'corr_mean': metrics['corr_mean'],
                'r2_per_mode': metrics['r2_per_mode'],
                'corr_per_mode': metrics['corr_per_mode'],
                'n_neurons': test_worm['u'].shape[1],
                'train_time': train_time,
                'n_params': n_params,
            }
    
    # Save results
    with open(out / 'results_variants.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for variant in args.variants:
        r2s = [results[variant][w]['r2_mean'] for w in results[variant]]
        corrs = [results[variant][w]['corr_mean'] for w in results[variant]]
        if r2s:
            print(f"{variant:20s}: R²={np.mean(r2s):.3f}±{np.std(r2s):.3f}, Corr={np.mean(corrs):.3f}±{np.std(corrs):.3f}")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # R² comparison
    ax = axes[0]
    x = np.arange(len(test_worm_ids))
    width = 0.8 / len(args.variants)
    
    for i, variant in enumerate(args.variants):
        r2s = [results[variant].get(w, {}).get('r2_mean', np.nan) for w in test_worm_ids]
        ax.bar(x + i * width, r2s, width, label=variant, alpha=0.8)
    
    ax.set_xticks(x + width * (len(args.variants) - 1) / 2)
    ax.set_xticklabels([w[:10] for w in test_worm_ids], rotation=45, ha='right')
    ax.set_ylabel('R²')
    ax.set_title('R² by Variant (worst worms)')
    ax.legend(fontsize=8)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Mean comparison
    ax = axes[1]
    means = []
    stds = []
    for variant in args.variants:
        r2s = [results[variant][w]['r2_mean'] for w in results[variant]]
        means.append(np.mean(r2s) if r2s else 0)
        stds.append(np.std(r2s) if r2s else 0)
    
    ax.bar(args.variants, means, yerr=stds, capsize=5, alpha=0.8)
    ax.set_ylabel('Mean R²')
    ax.set_title('Mean R² across test worms')
    ax.tick_params(axis='x', rotation=45)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(out / 'comparison_variants.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out / 'comparison_variants.png'}")


if __name__ == '__main__':
    main()
