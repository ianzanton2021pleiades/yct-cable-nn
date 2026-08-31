"""
cableformer.py -- CableFormer: CNN-Transformer 混合架构电缆缺陷检测

输入:  [B, 2, 2400] (impulse + step response, 归一化到 [-1,1])
输出:  logits [B, 2400], probs [B, 2400] in [0,1]

架构: 5 级 CNN 编码器 (标准卷积+空洞卷积+CBAM)
      -> Transformer Bottleneck (全局自注意力, 带投影降维)
      -> 5 级解码器 (DSConvBlock + skip connection)
      -> 1x1 Head + sigmoid

参数量控制策略:
  - Encoder: 标准卷积 + 空洞卷积 (特征质量优先)
  - Transformer: 投影到固定 d_model=256 (避免 16C 维度的二次增长)
  - Decoder: depthwise-separable conv (concat 后通道数大, 用轻量卷积)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .blocks_v2 import ResBlock, DSConvBlock
from .attention import CBAM, TransformerBottleneck


class CableFormer(nn.Module):

    def __init__(
        self,
        base_ch: int = 48,
        in_channels: int = 2,
        out_channels: int = 1,
        transformer_dim: int = 256,
        n_transformer_blocks: int = 3,
        n_heads: int = 8,
        use_cbam: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        C = base_ch

        # -- Stem --
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, C, kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(min(16, C), C),
            nn.GELU(),
        )

        # -- Encoder (5 stages, standard conv + dilated + CBAM) --
        # Stage 1: C -> C, dilations (1,1)
        self.enc1_block = ResBlock(C, C, dilations=(1, 1))
        self.enc1_cbam = CBAM(C) if use_cbam else nn.Identity()
        self.enc1_down = nn.Conv1d(C, C, 3, stride=2, padding=1, bias=False)

        # Stage 2: C -> 2C, dilations (1,2)
        self.enc2_block = ResBlock(C, 2 * C, dilations=(1, 2))
        self.enc2_cbam = CBAM(2 * C) if use_cbam else nn.Identity()
        self.enc2_down = nn.Conv1d(2 * C, 2 * C, 3, stride=2, padding=1, bias=False)

        # Stage 3: 2C -> 4C, dilations (1,4)
        self.enc3_block = ResBlock(2 * C, 4 * C, dilations=(1, 4))
        self.enc3_cbam = CBAM(4 * C) if use_cbam else nn.Identity()
        self.enc3_down = nn.Conv1d(4 * C, 4 * C, 3, stride=2, padding=1, bias=False)

        # Stage 4: 4C -> 8C, dilations (1,4)
        self.enc4_block = ResBlock(4 * C, 8 * C, dilations=(1, 4))
        self.enc4_cbam = CBAM(8 * C) if use_cbam else nn.Identity()
        self.enc4_down = nn.Conv1d(8 * C, 8 * C, 3, stride=2, padding=1, bias=False)

        # Stage 5: 8C -> 16C, dilations (1,2)
        self.enc5_block = ResBlock(8 * C, 16 * C, dilations=(1, 2))
        self.enc5_down = nn.Conv1d(16 * C, 16 * C, 3, stride=2, padding=1, bias=False)

        # -- Transformer Bottleneck (with projection to control params) --
        # 16C -> transformer_dim -> Transformer -> transformer_dim -> 16C
        self.tf_proj_down = nn.Sequential(
            nn.Conv1d(16 * C, transformer_dim, 1, bias=False),
            nn.GroupNorm(min(16, transformer_dim), transformer_dim),
        )
        self.transformer = TransformerBottleneck(
            d_model=transformer_dim,
            n_blocks=n_transformer_blocks,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.tf_proj_up = nn.Sequential(
            nn.Conv1d(transformer_dim, 16 * C, 1, bias=False),
            nn.GroupNorm(min(16, 16 * C), 16 * C),
            nn.GELU(),
        )

        # -- Decoder (5 stages, DSConvBlock + skip connections) --
        # Dec5: 16C -> 8C, concat skip4 (16C) -> 24C -> 8C
        self.dec5_up = nn.ConvTranspose1d(16 * C, 8 * C, 4, stride=2, padding=1, bias=False)
        self.dec5_block = DSConvBlock(8 * C + 16 * C, 8 * C)

        # Dec4: 8C -> 4C, concat skip3 (8C) -> 12C -> 4C
        self.dec4_up = nn.ConvTranspose1d(8 * C, 4 * C, 4, stride=2, padding=1, bias=False)
        self.dec4_block = DSConvBlock(4 * C + 8 * C, 4 * C)

        # Dec3: 4C -> 2C, concat skip2 (4C) -> 6C -> 2C
        self.dec3_up = nn.ConvTranspose1d(4 * C, 2 * C, 4, stride=2, padding=1, bias=False)
        self.dec3_block = DSConvBlock(2 * C + 4 * C, 2 * C)

        # Dec2: 2C -> C, concat skip1 (2C) -> 3C -> C
        self.dec2_up = nn.ConvTranspose1d(2 * C, C, 4, stride=2, padding=1, bias=False)
        self.dec2_block = DSConvBlock(C + 2 * C, C)

        # Dec1: C -> C, concat skip0 (C) -> 2C -> C
        self.dec1_up = nn.ConvTranspose1d(C, C, 4, stride=2, padding=1, bias=False)
        self.dec1_block = DSConvBlock(C + C, C)

        # -- Output Head --
        self.head = nn.Conv1d(C, out_channels, 1)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, in_channels, 2400]
        Returns:
            logits: [B, 2400]
            probs:  [B, 2400] in [0,1]
        """
        # Stem
        x0 = self.stem(x)                       # [B, C, 2400]

        # Encoder
        s0 = self.enc1_cbam(self.enc1_block(x0))  # skip0: [B, C, 2400]
        x1 = self.enc1_down(s0)                     # [B, C, 1200]

        s1 = self.enc2_cbam(self.enc2_block(x1))  # skip1: [B, 2C, 1200]
        x2 = self.enc2_down(s1)                     # [B, 2C, 600]

        s2 = self.enc3_cbam(self.enc3_block(x2))  # skip2: [B, 4C, 600]
        x3 = self.enc3_down(s2)                     # [B, 4C, 300]

        s3 = self.enc4_cbam(self.enc4_block(x3))  # skip3: [B, 8C, 300]
        x4 = self.enc4_down(s3)                     # [B, 8C, 150]

        s4 = self.enc5_block(x4)                    # skip4: [B, 16C, 150]
        x5 = self.enc5_down(s4)                     # [B, 16C, 75]

        # Transformer Bottleneck (with projection)
        x5 = self.tf_proj_down(x5)                 # [B, tf_dim, 75]
        x5 = self.transformer(x5)                   # [B, tf_dim, 75]
        x5 = self.tf_proj_up(x5)                   # [B, 16C, 75]

        # Decoder
        d4 = self.dec5_up(x5)                       # [B, 8C, 150]
        d4 = torch.cat([d4, s4], dim=1)            # [B, 24C, 150]
        d4 = self.dec5_block(d4)                    # [B, 8C, 150]

        d3 = self.dec4_up(d4)                       # [B, 4C, 300]
        d3 = torch.cat([d3, s3], dim=1)            # [B, 12C, 300]
        d3 = self.dec4_block(d3)                    # [B, 4C, 300]

        d2 = self.dec3_up(d3)                       # [B, 2C, 600]
        d2 = torch.cat([d2, s2], dim=1)            # [B, 6C, 600]
        d2 = self.dec3_block(d2)                    # [B, 2C, 600]

        d1 = self.dec2_up(d2)                       # [B, C, 1200]
        d1 = torch.cat([d1, s1], dim=1)            # [B, 3C, 1200]
        d1 = self.dec2_block(d1)                    # [B, C, 1200]

        d0 = self.dec1_up(d1)                       # [B, C, 2400]
        d0 = torch.cat([d0, s0], dim=1)            # [B, 2C, 2400]
        d0 = self.dec1_block(d0)                    # [B, C, 2400]

        # Output
        logits = self.head(d0).squeeze(1)           # [B, 2400]
        probs = torch.sigmoid(logits)
        return logits, probs


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for name, C in [("small", 32), ("medium", 48), ("large", 64)]:
        m = CableFormer(base_ch=C)
        n = count_parameters(m)
        x = torch.randn(2, 2, 2400)
        logits, probs = m(x)
        print(f"{name} (C={C}): {n:,} params, "
              f"output={tuple(probs.shape)}, "
              f"range=[{probs.min():.4f}, {probs.max():.4f}]")
