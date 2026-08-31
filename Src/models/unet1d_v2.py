"""
unet1d_v2.py -- UNet1D V2: 改进版 1D-UNet 电缆缺陷检测模型

相比 V1 的核心改进:
  1. 编码器: depthwise-separable conv → 标准卷积 + 空洞卷积 (ResBlock)
  2. 注意力: 无 → CBAM (通道+空间注意力)
  3. 编码级数: 4 级 → 5 级 (bottleneck 150→75 点)
  4. 下采样: 裸 Conv1d → Conv1d + GroupNorm + GELU
  5. 解码器: 保持 ConvBlock (轻量)

输入:  [B, 2, 2400] (impulse + step response, 归一化到 [-1,1])
输出:  logits [B, 2400], probs [B, 2400] ∈ [0,1]
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ConvBlock
from .blocks_v2 import ResBlock
from .attention import CBAM


class UNet1DV2(nn.Module):

    def __init__(
        self,
        base_ch: int = 48,
        in_channels: int = 2,
        out_channels: int = 1,
        use_cbam: bool = True,
    ):
        super().__init__()
        C = base_ch

        # ── Stem ──
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, C, kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(min(16, C), C),
            nn.GELU(),
        )

        # ── Encoder (5 stages: ResBlock + CBAM + enhanced downsample) ──
        # Stage 1: C → C, dilation (1,1)
        self.enc1_block = ResBlock(C, C, dilations=(1, 1))
        self.enc1_cbam = CBAM(C) if use_cbam else nn.Identity()
        self.enc1_down = nn.Sequential(
            nn.Conv1d(C, C, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(16, C), C),
            nn.GELU(),
        )

        # Stage 2: C → 2C, dilation (1,2)
        self.enc2_block = ResBlock(C, 2 * C, dilations=(1, 2))
        self.enc2_cbam = CBAM(2 * C) if use_cbam else nn.Identity()
        self.enc2_down = nn.Sequential(
            nn.Conv1d(2 * C, 2 * C, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(16, 2 * C), 2 * C),
            nn.GELU(),
        )

        # Stage 3: 2C → 4C, dilation (1,4)
        self.enc3_block = ResBlock(2 * C, 4 * C, dilations=(1, 4))
        self.enc3_cbam = CBAM(4 * C) if use_cbam else nn.Identity()
        self.enc3_down = nn.Sequential(
            nn.Conv1d(4 * C, 4 * C, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(16, 4 * C), 4 * C),
            nn.GELU(),
        )

        # Stage 4: 4C → 8C, dilation (1,4)
        self.enc4_block = ResBlock(4 * C, 8 * C, dilations=(1, 4))
        self.enc4_cbam = CBAM(8 * C) if use_cbam else nn.Identity()
        self.enc4_down = nn.Sequential(
            nn.Conv1d(8 * C, 8 * C, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(16, 8 * C), 8 * C),
            nn.GELU(),
        )

        # Stage 5: 8C → 16C, dilation (1,2)
        self.enc5_block = ResBlock(8 * C, 16 * C, dilations=(1, 2))
        self.enc5_cbam = CBAM(16 * C) if use_cbam else nn.Identity()
        self.enc5_down = nn.Sequential(
            nn.Conv1d(16 * C, 16 * C, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(16, 16 * C), 16 * C),
            nn.GELU(),
        )

        # ── Bottleneck ──
        self.bottleneck = ResBlock(16 * C, 16 * C, dilations=(1, 2))

        # ── Decoder (5 stages: ConvTranspose1d + cat skip + ConvBlock) ──
        # Dec5: 16C → 8C, concat skip4 (16C) → 24C → 8C
        self.dec5_up = nn.ConvTranspose1d(16 * C, 8 * C, 4, stride=2, padding=1)
        self.dec5_conv = ConvBlock(8 * C + 16 * C, 8 * C)

        # Dec4: 8C → 4C, concat skip3 (8C) → 12C → 4C
        self.dec4_up = nn.ConvTranspose1d(8 * C, 4 * C, 4, stride=2, padding=1)
        self.dec4_conv = ConvBlock(4 * C + 8 * C, 4 * C)

        # Dec3: 4C → 2C, concat skip2 (4C) → 6C → 2C
        self.dec3_up = nn.ConvTranspose1d(4 * C, 2 * C, 4, stride=2, padding=1)
        self.dec3_conv = ConvBlock(2 * C + 4 * C, 2 * C)

        # Dec2: 2C → C, concat skip1 (2C) → 3C → C
        self.dec2_up = nn.ConvTranspose1d(2 * C, C, 4, stride=2, padding=1)
        self.dec2_conv = ConvBlock(C + 2 * C, C)

        # Dec1: C → C, concat skip0 (C) → 2C → C
        self.dec1_up = nn.ConvTranspose1d(C, C, 4, stride=2, padding=1)
        self.dec1_conv = ConvBlock(C + C, C)

        # ── Output Head ──
        self.head = nn.Conv1d(C, out_channels, 1)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, in_channels, 2400]
        Returns:
            logits: [B, 2400]
            probs:  [B, 2400] ∈ [0,1]
        """
        # Stem
        x0 = self.stem(x)                           # [B, C, 2400]

        # Encoder
        s0 = self.enc1_cbam(self.enc1_block(x0))    # skip0: [B, C, 2400]
        x1 = self.enc1_down(s0)                     # [B, C, 1200]

        s1 = self.enc2_cbam(self.enc2_block(x1))    # skip1: [B, 2C, 1200]
        x2 = self.enc2_down(s1)                     # [B, 2C, 600]

        s2 = self.enc3_cbam(self.enc3_block(x2))    # skip2: [B, 4C, 600]
        x3 = self.enc3_down(s2)                     # [B, 4C, 300]

        s3 = self.enc4_cbam(self.enc4_block(x3))    # skip3: [B, 8C, 300]
        x4 = self.enc4_down(s3)                     # [B, 8C, 150]

        s4 = self.enc5_cbam(self.enc5_block(x4))    # skip4: [B, 16C, 150]
        x5 = self.enc5_down(s4)                     # [B, 16C, 75]

        # Bottleneck
        x5 = self.bottleneck(x5)                    # [B, 16C, 75]

        # Decoder
        d4 = self.dec5_up(x5)                       # [B, 8C, 150]
        d4 = torch.cat([d4, s4], dim=1)             # [B, 24C, 150]
        d4 = self.dec5_conv(d4)                     # [B, 8C, 150]

        d3 = self.dec4_up(d4)                       # [B, 4C, 300]
        d3 = torch.cat([d3, s3], dim=1)             # [B, 12C, 300]
        d3 = self.dec4_conv(d3)                     # [B, 4C, 300]

        d2 = self.dec3_up(d3)                       # [B, 2C, 600]
        d2 = torch.cat([d2, s2], dim=1)             # [B, 6C, 600]
        d2 = self.dec3_conv(d2)                     # [B, 2C, 600]

        d1 = self.dec2_up(d2)                       # [B, C, 1200]
        d1 = torch.cat([d1, s1], dim=1)             # [B, 3C, 1200]
        d1 = self.dec2_conv(d1)                     # [B, C, 1200]

        d0 = self.dec1_up(d1)                       # [B, C, 2400]
        d0 = torch.cat([d0, s0], dim=1)             # [B, 2C, 2400]
        d0 = self.dec1_conv(d0)                     # [B, C, 2400]

        # Output
        logits = self.head(d0).squeeze(1)           # [B, 2400]
        probs = torch.sigmoid(logits)
        return logits, probs


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for name, C in [("small", 32), ("medium", 48), ("large", 64)]:
        m = UNet1DV2(base_ch=C)
        n = count_parameters(m)
        x = torch.randn(2, 2, 2400)
        logits, probs = m(x)
        print(f"{name} (C={C}): {n:,} params, "
              f"output={tuple(probs.shape)}, "
              f"range=[{probs.min():.4f}, {probs.max():.4f}]")
