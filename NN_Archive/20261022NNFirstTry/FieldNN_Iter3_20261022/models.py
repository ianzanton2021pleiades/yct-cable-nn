"""Hierarchical 1D models for 8-channel FDR localization inputs.

Input shape is ``[batch, 8, 8192]``. One input channel carries normalized
distance coordinates. Outputs are event, defect-interval, and terminal logits
at the original 8192-point resolution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class _ResidualConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1) -> None:
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, padding=padding,
            dilation=dilation, bias=False,
        )
        self.norm1 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, padding=padding,
            dilation=dilation, bias=False,
        )
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.activation(x + residual)


class _DownsampleStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.down = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.norm = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.activation = nn.GELU()
        self.refine = _ResidualConv1d(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(self.activation(self.norm(self.down(x))))


class _Encoder(nn.Module):
    """Seven scales: 8192, 4096, 2048, 1024, 512, 256, and 128 points."""

    def __init__(self, input_channels: int, channels: list[int]) -> None:
        super().__init__()
        self.full_resolution = _ResidualConv1d(input_channels, channels[0])
        self.downsamples = nn.ModuleList(
            _DownsampleStage(channels[index], channels[index + 1])
            for index in range(len(channels) - 1)
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = [self.full_resolution(x)]
        for stage in self.downsamples:
            features.append(stage(features[-1]))
        return features


class _Decoder(nn.Module):
    def __init__(self, channels: list[int]) -> None:
        super().__init__()
        self.stages = nn.ModuleList(
            _ResidualConv1d(channels[index + 1] + channels[index], channels[index])
            for index in range(len(channels) - 2, -1, -1)
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        x = features[-1]
        for stage, skip in zip(self.stages, reversed(features[:-1])):
            x = F.interpolate(
                x, size=skip.shape[-1], mode="linear", align_corners=True
            )
            x = stage(torch.cat((x, skip), dim=1))
        return x


class _UNetBottleneck(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dilated = nn.Sequential(
            *[_ResidualConv1d(channels, channels, dilation=rate) for rate in (1, 2, 4, 8)]
        )
        self.global_projection = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dilated(x)
        global_context = self.global_projection(x.mean(dim=-1, keepdim=True))
        return x + global_context


class _TransformerBottleneck(nn.Module):
    def __init__(self, channels: int, depth: int) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=4,
            dim_feedforward=channels * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.transpose(1, 2)).transpose(1, 2)


class _LocalizationModel(nn.Module):
    def __init__(
        self,
        name: str,
        input_channels: int,
        width: int,
        depth: int,
    ) -> None:
        super().__init__()
        channel_limit = 192 if width <= 32 else 256
        channels = [min(width * (2**scale), channel_limit) for scale in range(7)]
        self.name = name
        self.input_channels = input_channels
        self.encoder = _Encoder(input_channels, channels)
        self.bottleneck = (
            _UNetBottleneck(channels[-1])
            if name == "unet"
            else _TransformerBottleneck(channels[-1], depth)
        )
        self.decoder = _Decoder(channels)
        self.event_head = nn.Conv1d(channels[0], 1, kernel_size=1)
        self.interval_head = nn.Conv1d(channels[0], 1, kernel_size=1)
        self.terminal_head = nn.Conv1d(channels[0], 1, kernel_size=1)
        nn.init.constant_(self.event_head.bias, -3.0)
        nn.init.constant_(self.interval_head.bias, -2.0)
        nn.init.constant_(self.terminal_head.bias, -3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("input must have shape [batch, channels, length]")
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"expected {self.input_channels} input channels, got {x.shape[1]}"
            )
        features = self.encoder(x)
        features[-1] = self.bottleneck(features[-1])
        decoded = self.decoder(features)
        return torch.cat(
            (
                self.event_head(decoded),
                self.interval_head(decoded),
                self.terminal_head(decoded),
            ),
            dim=1,
        )


def build_model(
    name: str,
    input_channels: int = 8,
    width: int = 32,
    depth: int = 3,
    state_size: int = 8,
) -> nn.Module:
    """Build a U-Net or Transformer with 8192-point, three-head output.

    ``depth`` controls Transformer bottleneck layers. ``state_size`` is accepted
    for compatibility with existing checkpoint loaders.
    """
    normalized_name = name.lower()
    if normalized_name not in {"unet", "transformer"}:
        raise ValueError("name must be one of: unet, transformer")
    if input_channels < 1 or width < 1 or depth < 1 or state_size < 1:
        raise ValueError("input_channels, width, depth, and state_size must be positive")
    return _LocalizationModel(normalized_name, input_channels, width, depth)
