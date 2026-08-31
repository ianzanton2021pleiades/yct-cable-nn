"""
blocks_v2.py -- CableFormer V2 基础构件

ResBlock: 标准卷积 + 空洞卷积 + GroupNorm + GELU, 带残差连接
相比 V1 的 DepthwiseSeparable ConvBlock:
  - 使用标准卷积 (非 depthwise-separable), 特征质量更高
  - 支持空洞卷积 (dilation), 不增加参数扩大感受野
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """
    两层标准卷积残差块, 支持空洞卷积.
    in_ch -> out_ch (若不同则用 1x1 投影).
    不改变空间尺寸.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        dilations: tuple[int, int] = (1, 1),
        gn_groups: int = 16,
    ):
        super().__init__()
        g1 = min(gn_groups, out_ch)
        g2 = min(gn_groups, out_ch)

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_ch, out_ch, kernel_size=3,
                padding=dilations[0], dilation=dilations[0], bias=False,
            ),
            nn.GroupNorm(g1, out_ch),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                out_ch, out_ch, kernel_size=3,
                padding=dilations[1], dilation=dilations[1], bias=False,
            ),
            nn.GroupNorm(g2, out_ch),
        )
        self.act = nn.GELU()

        self.need_proj = (in_ch != out_ch)
        if self.need_proj:
            self.proj = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, bias=False),
                nn.GroupNorm(min(gn_groups, out_ch), out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.proj(x) if self.need_proj else x
        return self.act(self.conv2(self.conv1(x)) + res)


class DSConvBlock(nn.Module):
    """
    轻量 depthwise-separable 卷积残差块.
    用于 decoder 阶段, 在 skip-connection concat 后大通道数场景下控制参数量.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 7,
        gn_groups: int = 16,
    ):
        super().__init__()
        g = min(gn_groups, out_ch)
        p = kernel_size // 2

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, kernel_size, padding=p, groups=in_ch, bias=False),
            nn.Conv1d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=p, groups=out_ch, bias=False),
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
