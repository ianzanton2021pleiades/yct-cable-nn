"""Pure-PyTorch 1D localization models for distance-domain FDR inputs.

Input channels: normalized impulse, normalized step, log-RMS impulse,
log-RMS step. Output channels are event, interval, terminal logits (no sigmoid).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ChannelLayerNorm(nn.Module):
    """LayerNorm over channels for a [batch, channels, length] tensor."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class _ConvNormAct(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1, kernel_size: int = 3
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )
        self.norm = _ChannelLayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.conv(x)))


class _ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm1 = _ChannelLayerNorm(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = _ChannelLayerNorm(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.gelu(self.norm1(x)))
        h = self.conv2(F.gelu(self.norm2(h)))
        return self.skip(x) + h


class _SharedStem(nn.Module):
    """Three stride-2 stages with full-resolution and multiscale skip features."""

    def __init__(self, input_channels: int, width: int) -> None:
        super().__init__()
        self.full = _ConvNormAct(input_channels, width)
        self.half_stage = _ConvNormAct(width, width, stride=2)
        self.quarter = _ConvNormAct(width, width * 2, stride=2)
        self.eighth = _ConvNormAct(width * 2, width * 4, stride=2)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        full = self.full(x)
        half = self.half_stage(full)
        quarter = self.quarter(half)
        eighth = self.eighth(quarter)
        return full, half, quarter, eighth


class _DecoderStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = _ResidualConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        return self.block(torch.cat((x, skip), dim=1))


class _SharedDecoder(nn.Module):
    def __init__(self, width: int, output_channels: int = 3) -> None:
        super().__init__()
        self.eighth_to_quarter = _DecoderStage(width * 4, width * 2, width * 2)
        self.quarter_to_half = _DecoderStage(width * 2, width, width)
        self.half_to_full = _DecoderStage(width, width, width)
        self.output = nn.Conv1d(width, output_channels, kernel_size=1)

    def forward(
        self,
        eighth: torch.Tensor,
        quarter: torch.Tensor,
        half: torch.Tensor,
        full: torch.Tensor,
    ) -> torch.Tensor:
        x = self.eighth_to_quarter(eighth, quarter)
        x = self.quarter_to_half(x, half)
        x = self.half_to_full(x, full)
        return self.output(x)


def affine_prefix_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Inclusive parallel scan of h[t] = a[t] * h[t-1] + b[t], h[-1] = 0.

    The affine-pair composition is associative, so the logarithmic scan is
    equivalent to sequential recurrence while retaining an ordinary autograd
    graph. No tensor is modified in place.
    """

    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")
    if a.ndim < 2:
        raise ValueError("scan tensors need a batch and time dimension")

    length = a.shape[1]
    offset = 1
    while offset < length:
        a_current = a[:, offset:]
        a_previous = a[:, :-offset]
        b_current = b[:, offset:]
        b_previous = b[:, :-offset]
        next_a = torch.cat((a[:, :offset], a_current * a_previous), dim=1)
        next_b = torch.cat((b[:, :offset], b_current + a_current * b_previous), dim=1)
        a, b = next_a, next_b
        offset *= 2
    return b


def selective_scan(
    u: torch.Tensor, delta: torch.Tensor, b_input: torch.Tensor, a_log: torch.Tensor
) -> torch.Tensor:
    """Input-selective diagonal SSM scan; returns all states in FP32.

    Shapes: u/delta [B,T,C], b_input [B,T,N], a_log [C,N].
    This uses the standard Mamba selective discretization
    A_bar=exp(delta*A), B_bar*u=delta*B*u.
    """

    u32 = u.float()
    delta32 = delta.float()
    b32 = b_input.float()
    a = -torch.exp(a_log.float())
    transition = torch.exp(delta32.unsqueeze(-1) * a.unsqueeze(0).unsqueeze(0))
    input_term = (
        delta32.unsqueeze(-1) * b32.unsqueeze(2) * u32.unsqueeze(-1)
    )
    return affine_prefix_scan(transition, input_term)


class _MambaBlock(nn.Module):
    """Bidirectional Mamba-style block with shared forward/reverse parameters."""

    def __init__(self, model_dim: int, state_size: int, expand: int = 2) -> None:
        super().__init__()
        inner_dim = model_dim * expand
        dt_rank = max(1, math.ceil(model_dim / 16))
        self.inner_dim = inner_dim
        self.state_size = state_size
        self.dt_rank = dt_rank

        self.norm = nn.LayerNorm(model_dim)
        self.in_proj = nn.Linear(model_dim, inner_dim * 2, bias=False)
        self.causal_conv = nn.Conv1d(
            inner_dim,
            inner_dim,
            kernel_size=4,
            groups=inner_dim,
            bias=True,
        )
        self.x_proj = nn.Linear(inner_dim, dt_rank + state_size * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, inner_dim, bias=True)
        self.out_proj = nn.Linear(inner_dim, model_dim, bias=False)

        initial_a = torch.arange(1, state_size + 1, dtype=torch.float32).log()
        self.A_log = nn.Parameter(initial_a.unsqueeze(0).repeat(inner_dim, 1))
        self.D = nn.Parameter(torch.ones(inner_dim, dtype=torch.float32))

        with torch.no_grad():
            log_min = math.log(1e-3)
            log_max = math.log(1e-1)
            initial_delta = torch.exp(
                torch.rand(inner_dim) * (log_max - log_min) + log_min
            )
            self.dt_proj.bias.copy_(
                initial_delta + torch.log(-torch.expm1(-initial_delta))
            )

    def _causal_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = F.pad(x, (self.causal_conv.kernel_size[0] - 1, 0))
        return F.silu(self.causal_conv(x).transpose(1, 2))

    def _scan_direction(
        self, u: torch.Tensor, reverse: bool
    ) -> torch.Tensor:
        projected = self.x_proj(u)
        dt_low_rank, b_input, c_readout = torch.split(
            projected,
            (self.dt_rank, self.state_size, self.state_size),
            dim=-1,
        )
        delta = F.softplus(self.dt_proj(dt_low_rank))

        if reverse:
            u = u.flip(1)
            delta = delta.flip(1)
            b_input = b_input.flip(1)
            c_readout = c_readout.flip(1)

        states = selective_scan(u, delta, b_input, self.A_log)
        y = (states * c_readout.float().unsqueeze(2)).sum(dim=-1)
        y = y + self.D.float().view(1, 1, -1) * u.float()
        if reverse:
            y = y.flip(1)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        content, gate = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        forward_features = self._causal_features(content)
        reverse_features = self._causal_features(content.flip(1)).flip(1)

        forward_y = self._scan_direction(forward_features, reverse=False)
        reverse_y = self._scan_direction(reverse_features, reverse=True)
        y = 0.5 * (forward_y + reverse_y)
        y = y * F.silu(gate).float()
        return x + self.out_proj(y.to(dtype=x.dtype))


class _MambaBackbone(nn.Module):
    def __init__(self, model_dim: int, depth: int, state_size: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_MambaBlock(model_dim, state_size) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        return x.transpose(1, 2)


class _SinusoidalPositionEncoding(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length, channels = x.shape[1], x.shape[2]
        position = torch.arange(length, device=x.device, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(
            torch.arange(0, channels, 2, device=x.device, dtype=torch.float32)
            * (-math.log(10000.0) / channels)
        )
        encoding = torch.zeros(length, channels, device=x.device, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * scale)
        encoding[:, 1::2] = torch.cos(position * scale[: encoding[:, 1::2].shape[1]])
        return x + encoding.to(dtype=x.dtype).unsqueeze(0)


class _TransformerBackbone(nn.Module):
    def __init__(self, model_dim: int, depth: int) -> None:
        super().__init__()
        self.position = _SinusoidalPositionEncoding()
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=4,
            dim_feedforward=model_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.position(x)
        x = self.encoder(x)
        return x.transpose(1, 2)


class _UNetBackbone(nn.Module):
    def __init__(self, width: int, depth: int) -> None:
        super().__init__()
        self.half_blocks = nn.Sequential(
            *[_ResidualConvBlock(width, width) for _ in range(depth)]
        )
        self.quarter_blocks = nn.Sequential(
            *[_ResidualConvBlock(width * 2, width * 2) for _ in range(depth)]
        )
        self.eighth_blocks = nn.Sequential(
            *[_ResidualConvBlock(width * 4, width * 4) for _ in range(depth)]
        )

    def forward(
        self, half: torch.Tensor, quarter: torch.Tensor, eighth: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.half_blocks(half),
            self.quarter_blocks(quarter),
            self.eighth_blocks(eighth),
        )


class _FieldLocalizationModel(nn.Module):
    def __init__(
        self,
        name: str,
        input_channels: int,
        width: int,
        depth: int,
        state_size: int,
    ) -> None:
        super().__init__()
        self.name = name
        self.input_channels = input_channels
        model_dim = width * 4
        self.stem = _SharedStem(input_channels, width)

        if name == "mamba":
            self.backbone = _MambaBackbone(model_dim, depth, state_size)
        elif name == "transformer":
            self.backbone = _TransformerBackbone(model_dim, depth)
        else:
            self.backbone = _UNetBackbone(width, depth)

        self.decoder = _SharedDecoder(width, output_channels=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("input must have shape [batch, channels, length]")
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"expected {self.input_channels} input channels, got {x.shape[1]}"
            )

        full, half, quarter, eighth = self.stem(x)
        if self.name == "unet":
            half, quarter, eighth = self.backbone(half, quarter, eighth)
        else:
            eighth = self.backbone(eighth)
        return self.decoder(eighth, quarter, half, full)


def build_model(
    name: str,
    input_channels: int = 4,
    width: int = 24,
    depth: int = 2,
    state_size: int = 8,
) -> nn.Module:
    """Build a model returning event/interval/terminal logits at input resolution."""

    normalized_name = name.lower()
    if normalized_name not in {"mamba", "transformer", "unet"}:
        raise ValueError("name must be one of: mamba, transformer, unet")
    if input_channels < 1 or width < 1 or depth < 1 or state_size < 1:
        raise ValueError("input_channels, width, depth, and state_size must be positive")
    return _FieldLocalizationModel(
        normalized_name, input_channels, width, depth, state_size
    )
