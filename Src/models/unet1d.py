"""
unet1d.py — 可调宽 1D-UNet 电缆缺陷检测模型

输入:  [B, 2, 2400] (impulse + step response, 归一化到 [-1,1])
输出:  logits [B, 2400], probs [B, 2400] ∈ [0,1]

架构:  4级编码器 → bottleneck → 4级解码器，skip-connection 连接对应层
       每级: ConvBlock(同分辨率处理) → Conv1d stride=2(下采样)
       解码: ConvTranspose1d(上采样) → concat skip → ConvBlock

三档宽度:
  narrow (C=16): ~100k params
  medium (C=32): ~400k params
  wide   (C=64): ~1.6M params
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ConvBlock


class UNet1D(nn.Module):

    def __init__(
        self,
        base_ch: int = 32,
        in_channels: int = 2,
        out_channels: int = 1,
        kernel_size: int = 7,
    ):
        super().__init__()
        C = base_ch

        # ── 输入投影 ──
        self.input_conv = ConvBlock(in_channels, C, kernel_size)  # [B,C,2400]

        # ── 编码器 (处理 + 下采样) ──
        self.enc1_conv = ConvBlock(C, C, kernel_size)       # [B,C,2400]
        self.enc1_down = nn.Conv1d(C, C, 3, stride=2, padding=1)  # → [B,C,1200]

        self.enc2_conv = ConvBlock(C, 2 * C, kernel_size)   # [B,2C,1200]
        self.enc2_down = nn.Conv1d(2 * C, 2 * C, 3, stride=2, padding=1)  # → [B,2C,600]

        self.enc3_conv = ConvBlock(2 * C, 4 * C, kernel_size)  # [B,4C,600]
        self.enc3_down = nn.Conv1d(4 * C, 4 * C, 3, stride=2, padding=1)  # → [B,4C,300]

        self.enc4_conv = ConvBlock(4 * C, 8 * C, kernel_size)  # [B,8C,300]
        self.enc4_down = nn.Conv1d(8 * C, 8 * C, 3, stride=2, padding=1)  # → [B,8C,150]

        # ── Bottleneck ──
        self.bottleneck = ConvBlock(8 * C, 8 * C, kernel_size)  # [B,8C,150]

        # ── 解码器 (上采样 + concat skip + 处理) ──
        # skip 来自编码器 conv 输出(下采样前)，通道数是上采样输出的 2 倍
        self.dec4_up = nn.ConvTranspose1d(8 * C, 4 * C, 4, stride=2, padding=1)  # → [B,4C,300]
        self.dec4_conv = ConvBlock(4 * C + 8 * C, 4 * C, kernel_size)  # concat skip3(8C)

        self.dec3_up = nn.ConvTranspose1d(4 * C, 2 * C, 4, stride=2, padding=1)  # → [B,2C,600]
        self.dec3_conv = ConvBlock(2 * C + 4 * C, 2 * C, kernel_size)  # concat skip2(4C)

        self.dec2_up = nn.ConvTranspose1d(2 * C, C, 4, stride=2, padding=1)  # → [B,C,1200]
        self.dec2_conv = ConvBlock(C + 2 * C, C, kernel_size)    # concat skip1(2C)

        self.dec1_up = nn.ConvTranspose1d(C, C, 4, stride=2, padding=1)  # → [B,C,2400]
        self.dec1_conv = ConvBlock(C + C, C, kernel_size)        # concat skip0(C)

        # ── 输出头 ──
        self.head = nn.Conv1d(C, out_channels, 1)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, in_channels, 2400]
        Returns:
            logits: [B, 2400]
            probs:  [B, 2400] ∈ [0,1]
        """
        # 输入
        x0 = self.input_conv(x)             # [B, C, 2400]

        # 编码
        s0 = self.enc1_conv(x0)             # skip0: [B, C, 2400]
        x1 = self.enc1_down(s0)             # [B, C, 1200]

        s1 = self.enc2_conv(x1)             # skip1: [B, 2C, 1200]
        x2 = self.enc2_down(s1)             # [B, 2C, 600]

        s2 = self.enc3_conv(x2)             # skip2: [B, 4C, 600]
        x3 = self.enc3_down(s2)             # [B, 4C, 300]

        s3 = self.enc4_conv(x3)             # skip3: [B, 8C, 300]
        x4 = self.enc4_down(s3)             # [B, 8C, 150]

        # Bottleneck
        x4 = self.bottleneck(x4)            # [B, 8C, 150]

        # 解码
        d3 = self.dec4_up(x4)              # [B, 4C, 300]
        d3 = torch.cat([d3, s3], dim=1)     # [B, 8C, 300]
        d3 = self.dec4_conv(d3)             # [B, 4C, 300]

        d2 = self.dec3_up(d3)              # [B, 2C, 600]
        d2 = torch.cat([d2, s2], dim=1)     # [B, 4C, 600]
        d2 = self.dec3_conv(d2)             # [B, 2C, 600]

        d1 = self.dec2_up(d2)              # [B, C, 1200]
        d1 = torch.cat([d1, s1], dim=1)     # [B, 2C, 1200]
        d1 = self.dec2_conv(d1)             # [B, C, 1200]

        d0 = self.dec1_up(d1)              # [B, C, 2400]
        d0 = torch.cat([d0, s0], dim=1)     # [B, 2C, 2400]
        d0 = self.dec1_conv(d0)             # [B, C, 2400]

        # 输出
        logits = self.head(d0).squeeze(1)   # [B, 2400]
        probs = torch.sigmoid(logits)
        return logits, probs


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for name, C in [("narrow", 16), ("medium", 32), ("wide", 64)]:
        m = UNet1D(base_ch=C)
        n = count_parameters(m)
        x = torch.randn(2, 2, 2400)
        logits, probs = m(x)
        print(f"{name} (C={C}): {n:,} params, "
              f"output={tuple(probs.shape)}, "
              f"range=[{probs.min():.4f}, {probs.max():.4f}]")
