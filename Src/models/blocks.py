"""
blocks.py — 1D-UNet 基础构件

ConvBlock: Depthwise-Separable Conv + GroupNorm + GELU，带残差连接
Downsample / Upsample 操作在 UNet 主类中显式处理
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    两层 Depthwise-Separable Conv + 残差连接，不改变空间尺寸。
    in_ch → out_ch (若不同则用 1x1 投影)。
    """

    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int = 7, gn_groups: int = 8):
        super().__init__()
        g = min(gn_groups, out_ch)
        padding = kernel_size // 2

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, kernel_size, padding=padding,
                      groups=in_ch, bias=False),
            nn.Conv1d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding,
                      groups=out_ch, bias=False),
            nn.Conv1d(out_ch, out_ch, 1, bias=False),
            nn.GroupNorm(g, out_ch),
        )
        self.act = nn.GELU()

        self.need_proj = (in_ch != out_ch)
        if self.need_proj:
            self.proj = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, bias=False),
                nn.GroupNorm(g, out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.proj(x) if self.need_proj else x
        return self.act(self.conv2(self.conv1(x)) + res)
