"""1D U-Net with role/mechanism heads + terminal regression head.

Input: fixed 0.25 m distance grid, GRID_BINS bins (0-2752 m), channels:
  0 impulse (raw)  1 signed-log impulse  2 step  3 normalized distance axis
Output heads at stride-1:
  role  (4 logits), mechanism (11 logits), terminal (end position + visibility)

v2 (UNet1DV2): MultiScaleFront (parallel depthwise k=3/9/31/99 + GELU,
concatenated with the input before the stem) + optional depthwise-separable
blocks (--dsconv) so width=48 stays near v1 parameter counts.

v3: both variants add head_z0, a per-pixel 1-channel z0-deviation regression
head (output "z0", shape (B, N)); supervised on synthetic samples only.
"""
from __future__ import annotations

import torch
import torch.nn as nn

GRID_BINS = 11008
N_ROLE = 4
N_MECH = 11


def conv_block(cin, cout, k=5):
    return nn.Sequential(
        nn.Conv1d(cin, cout, k, padding=k // 2),
        nn.GroupNorm(min(8, cout), cout),
        nn.GELU(),
        nn.Conv1d(cout, cout, k, padding=k // 2),
        nn.GroupNorm(min(8, cout), cout),
        nn.GELU(),
    )


def dsconv_block(cin, cout, k=5):
    """Depthwise + pointwise replacement for conv_block (same receptive field,
    ~k-times fewer params in the spatial conv) — enables wider (48ch) nets at
    v1-level parameter counts."""
    return nn.Sequential(
        nn.Conv1d(cin, cin, k, padding=k // 2, groups=cin),
        nn.Conv1d(cin, cout, 1),
        nn.GroupNorm(min(8, cout), cout),
        nn.GELU(),
        nn.Conv1d(cout, cout, k, padding=k // 2, groups=cout),
        nn.Conv1d(cout, cout, 1),
        nn.GroupNorm(min(8, cout), cout),
        nn.GELU(),
    )


class MultiScaleFront(nn.Module):
    """Parallel depthwise convs with very different kernel sizes + GELU.
    Large kernels (31/99 bins = 7.75/24.75 m) surface broad low-amplitude
    far-end reflections that sit at ~1e-4 next to near-end features; small
    kernels keep sharp near-end edges.  Branch outputs are concatenated with
    the raw input, so the stem sees both scales side by side."""
    def __init__(self, cin: int = 4, scales: tuple = (3, 9, 31, 99)):
        super().__init__()
        self.cin = cin
        self.scales = tuple(scales)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(cin, cin, k, padding=k // 2, groups=cin),
                nn.GroupNorm(cin, cin),
                nn.GELU(),
            ) for k in self.scales])

    @property
    def out_channels(self) -> int:
        return self.cin * (1 + len(self.scales))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x] + [b(x) for b in self.branches], dim=1)


class UNet1D(nn.Module):
    def __init__(self, cin: int = 4, width: int = 32, levels: int = 4):
        super().__init__()
        chans = [width * 2 ** i for i in range(levels + 1)]  # 32..512
        self.stem = conv_block(cin, chans[0])
        self.down = nn.ModuleList()
        for i in range(levels):
            self.down.append(nn.ModuleDict(dict(
                pool=nn.Conv1d(chans[i], chans[i], 3, stride=2, padding=1),
                block=conv_block(chans[i], chans[i + 1]))))
        self.bridge = conv_block(chans[-1], chans[-1])
        self.up = nn.ModuleList()
        for i in range(levels, 0, -1):
            self.up.append(nn.ModuleDict(dict(
                expand=nn.ConvTranspose1d(chans[i], chans[i - 1], 4, stride=2, padding=1),
                block=conv_block(chans[i - 1] * 2, chans[i - 1]))))
        self.head_role = nn.Conv1d(chans[0], N_ROLE, 1)
        self.head_mech = nn.Conv1d(chans[0], N_MECH, 1)
        self.head_z0 = nn.Conv1d(chans[0], 1, 1)
        self.head_term = nn.Sequential(
            nn.AdaptiveAvgPool1d(64), nn.Flatten(),
            nn.Linear(chans[0] * 64, 128), nn.GELU(),
            nn.Linear(128, 3))  # (end_pos_frac, visibility logit, spare)

    def forward(self, x: torch.Tensor) -> dict:
        skips = []
        h = self.stem(x)
        for d in self.down:
            skips.append(h)
            h = d["block"](d["pool"](h))
        h = self.bridge(h)
        for u in self.up:  # appended 5..1; first module consumes the bridge
            h = u["block"](torch.cat([u["expand"](h), skips.pop()], dim=1))
        return dict(role=self.head_role(h), mech=self.head_mech(h),
                    z0=self.head_z0(h).squeeze(1), term=self.head_term(h))


class UNet1DV2(nn.Module):
    """v2: MultiScaleFront before the stem + optional depthwise-separable
    blocks.  Same I/O contract and head layout as UNet1D (role 4 / mech 11 /
    terminal 3 at stride 1), drop-in with --arch v2."""
    def __init__(self, cin: int = 4, width: int = 32, levels: int = 4,
                 scales: tuple = (3, 9, 31, 99), dsconv: bool = True):
        super().__init__()
        blk = dsconv_block if dsconv else conv_block
        chans = [width * 2 ** i for i in range(levels + 1)]
        self.front = MultiScaleFront(cin, scales)
        self.stem = blk(self.front.out_channels, chans[0])
        self.down = nn.ModuleList()
        for i in range(levels):
            self.down.append(nn.ModuleDict(dict(
                pool=nn.Conv1d(chans[i], chans[i], 3, stride=2, padding=1),
                block=blk(chans[i], chans[i + 1]))))
        self.bridge = blk(chans[-1], chans[-1])
        self.up = nn.ModuleList()
        for i in range(levels, 0, -1):
            self.up.append(nn.ModuleDict(dict(
                expand=nn.ConvTranspose1d(chans[i], chans[i - 1], 4, stride=2, padding=1),
                block=blk(chans[i - 1] * 2, chans[i - 1]))))
        self.head_role = nn.Conv1d(chans[0], N_ROLE, 1)
        self.head_mech = nn.Conv1d(chans[0], N_MECH, 1)
        self.head_z0 = nn.Conv1d(chans[0], 1, 1)
        self.head_term = nn.Sequential(
            nn.AdaptiveAvgPool1d(64), nn.Flatten(),
            nn.Linear(chans[0] * 64, 128), nn.GELU(),
            nn.Linear(128, 3))

    def forward(self, x: torch.Tensor) -> dict:
        skips = []
        h = self.stem(self.front(x))
        for d in self.down:
            skips.append(h)
            h = d["block"](d["pool"](h))
        h = self.bridge(h)
        for u in self.up:
            h = u["block"](torch.cat([u["expand"](h), skips.pop()], dim=1))
        return dict(role=self.head_role(h), mech=self.head_mech(h),
                    z0=self.head_z0(h).squeeze(1), term=self.head_term(h))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
