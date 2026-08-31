"""
attention.py -- CBAM 注意力 + Transformer Bottleneck

CBAM: Convolutional Block Attention Module
  - 通道注意力: 全局平均池化 -> FC -> GELU -> FC -> sigmoid
  - 空间注意力: avg_pool + max_pool -> Conv1d -> sigmoid

TransformerBlock: Pre-Norm Transformer encoder block
  - LayerNorm -> MultiheadAttention -> residual
  - LayerNorm -> FFN -> GELU -> residual
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ChannelAttention(nn.Module):
    """通道注意力: GAP -> FC -> GELU -> FC -> sigmoid"""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.GELU(),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        b, c, l = x.shape
        # Global Average Pooling
        gap = x.mean(dim=-1)  # [B, C]
        attn = torch.sigmoid(self.fc(gap))  # [B, C]
        return x * attn.unsqueeze(-1)  # [B, C, L]


class SpatialAttention(nn.Module):
    """空间注意力: avg_pool + max_pool -> Conv1d(2->1) -> sigmoid"""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        avg_pool = x.mean(dim=1, keepdim=True)   # [B, 1, L]
        max_pool = x.max(dim=1, keepdim=True)[0]  # [B, 1, L]
        cat_pool = torch.cat([avg_pool, max_pool], dim=1)  # [B, 2, L]
        attn = torch.sigmoid(self.conv(cat_pool))  # [B, 1, L]
        return x * attn


class CBAM(nn.Module):
    """CBAM: 通道注意力 + 空间注意力"""

    def __init__(self, channels: int, reduction: int = 8, spatial_kernel: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class TransformerBlock(nn.Module):
    """Pre-Norm Transformer encoder block."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_ff: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] (batch_first=True)
        Returns:
            [B, L, D]
        """
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out

        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerBottleneck(nn.Module):
    """
    多个 Transformer blocks 组成的 bottleneck.
    输入 [B, C, L] -> 转置为 [B, L, C] -> N 个 TransformerBlock -> 转置回 [B, C, L]
    """

    def __init__(
        self,
        d_model: int,
        n_blocks: int = 3,
        n_heads: int = 8,
        d_ff: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, L] -> [B, L, C]
        x = x.transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        # [B, L, C] -> [B, C, L]
        return x.transpose(1, 2)
