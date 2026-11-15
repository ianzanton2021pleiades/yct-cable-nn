"""Phase-aware 1D detector with a shared single-phase encoder."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


TASK_COUNT = 4


class ChannelLayerNorm(nn.Module):
    """Layer normalization over channels independently at each distance."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1) -> None:
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, padding=padding,
            dilation=dilation, bias=False,
        )
        self.norm1 = ChannelLayerNorm(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = ChannelLayerNorm(out_channels)
        self.activation = nn.GELU()
        self.residual = (
            nn.Identity() if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        mask = valid.unsqueeze(1).to(dtype=x.dtype)
        residual = self.residual(x) * mask
        x = self.activation(self.norm1(self.conv1(x * mask))) * mask
        x = self.norm2(self.conv2(x))
        return self.activation(x + residual) * mask


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.down = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.norm = ChannelLayerNorm(out_channels)
        self.activation = nn.GELU()
        self.refine = ConvBlock(out_channels, out_channels)

    def forward(
        self, x: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        next_valid = valid[..., ::2]
        mask = next_valid.unsqueeze(1).to(dtype=x.dtype)
        x = self.activation(self.norm(self.down(x * valid.unsqueeze(1)))) * mask
        return self.refine(x, next_valid), next_valid


class SharedEncoder(nn.Module):
    def __init__(self, in_channels: int, width: int) -> None:
        super().__init__()
        self.stem = ConvBlock(in_channels, width)
        self.down1 = DownBlock(width, width * 2)
        self.down2 = DownBlock(width * 2, width * 4)
        self.down3 = DownBlock(width * 4, width * 4)
        self.bottleneck = nn.ModuleList(
            ConvBlock(width * 4, width * 4, dilation=dilation)
            for dilation in (1, 2, 4, 8)
        )

    def forward(
        self, x: torch.Tensor, valid: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        x0 = self.stem(x, valid)
        x1, valid1 = self.down1(x0, valid)
        x2, valid2 = self.down2(x1, valid1)
        x3, valid3 = self.down3(x2, valid2)
        for block in self.bottleneck:
            x3 = block(x3, valid3)
        return (x0, x1, x2, x3), (valid, valid1, valid2, valid3)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")[..., : skip.shape[-1]]
        return self.block(torch.cat((x, skip), dim=1), valid)


class PhaseDetector(nn.Module):
    """Predict joint, terminal, point-anomaly and range-anomaly logits per phase."""

    def __init__(self, in_channels: int = 5, width: int = 16) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.width = width
        self.encoder = SharedEncoder(in_channels, width)
        bottleneck_channels = width * 4
        self.phase_context_projection = nn.Conv1d(
            bottleneck_channels, bottleneck_channels, kernel_size=1, bias=False
        )
        self.up2 = UpBlock(width * 4, width * 4, width * 4)
        self.up1 = UpBlock(width * 4, width * 2, width * 2)
        self.up0 = UpBlock(width * 2, width, width)
        self.head = nn.Conv1d(width, TASK_COUNT, kernel_size=1)

    def export_encoder_state_dict(self) -> dict[str, torch.Tensor]:
        """Return only shared single-phase encoder weights for B-arm initialization."""
        return self.encoder.state_dict()

    def load_encoder_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load a shared encoder state dict and require an exact architecture match."""
        self.encoder.load_state_dict(state_dict, strict=True)

    @staticmethod
    def _mask_for(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return x * valid.unsqueeze(1).to(dtype=x.dtype)

    def forward(
        self,
        x: torch.Tensor,
        valid: torch.Tensor,
        phase_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, phases, channels, original_length = x.shape
        phase_mask = phase_mask.bool()
        valid = valid.bool() & phase_mask.unsqueeze(-1)
        x = x * valid.unsqueeze(2).to(dtype=x.dtype)

        pad_length = (-original_length) % 8
        if pad_length:
            x = F.pad(x, (0, pad_length))
            valid = F.pad(valid, (0, pad_length), value=False)

        padded_length = x.shape[-1]
        flat_x = x.reshape(batch * phases, channels, padded_length)
        flat_valid = valid.reshape(batch * phases, padded_length)
        features, masks = self.encoder(flat_x, flat_valid)
        f0, f1, f2, f3 = features
        m0, m1, m2, m3 = masks

        bottleneck_length = f3.shape[-1]
        grouped = f3.reshape(batch, phases, f3.shape[1], bottleneck_length)
        available = m3.reshape(batch, phases, bottleneck_length)
        weights = available.unsqueeze(2).to(dtype=f3.dtype)
        mean_phase = (grouped * weights).sum(dim=1, keepdim=True)
        mean_phase = mean_phase / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        context_difference = mean_phase - grouped
        flat_difference = context_difference.reshape(batch * phases, f3.shape[1], bottleneck_length)
        fused = f3 + self.phase_context_projection(flat_difference)
        fused = self._mask_for(fused, m3)

        decoded = self.up2(fused, f2, m2)
        decoded = self.up1(decoded, f1, m1)
        decoded = self.up0(decoded, f0, m0)
        logits = self.head(decoded)
        logits = logits.reshape(batch, phases, TASK_COUNT, padded_length)
        logits = logits[..., :original_length]
        return logits * valid[..., :original_length].unsqueeze(2).to(dtype=logits.dtype)
