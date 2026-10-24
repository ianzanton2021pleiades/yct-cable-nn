"""Iter8 input adapter around the existing Iter3 U-Net and Transformer."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch
from torch import nn


HERE = Path(__file__).resolve().parent
ITER3_MODELS = HERE.parent / "FieldNN_Iter3_20261022" / "models.py"
spec = importlib.util.spec_from_file_location("iter8_iter3_models", ITER3_MODELS)
_base_models = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = _base_models
spec.loader.exec_module(_base_models)


class InputStemModel(nn.Module):
    def __init__(
        self,
        name: str,
        input_channels: int,
        width: int,
        depth: int,
        state_size: int,
    ) -> None:
        super().__init__()
        self.name = name.lower()
        self.input_channels = input_channels
        self.input_stem = nn.Conv1d(input_channels, 8, kernel_size=1, bias=False)
        self.backbone = _base_models.build_model(
            self.name,
            input_channels=8,
            width=width,
            depth=depth,
            state_size=state_size,
        )

        with torch.no_grad():
            self.input_stem.weight.zero_()
            self.input_stem.weight[:, :8, 0] = torch.eye(8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.input_stem(x))


def build_model(
    name: str,
    input_channels: int = 12,
    width: int = 32,
    depth: int = 3,
    state_size: int = 8,
) -> nn.Module:
    """Build Iter3's U-Net or Transformer with an identity-initialized 12→8 stem."""
    normalized_name = name.lower()
    if normalized_name not in {"unet", "transformer"}:
        raise ValueError("name must be one of: unet, transformer")
    return InputStemModel(
        normalized_name,
        input_channels=input_channels,
        width=width,
        depth=depth,
        state_size=state_size,
    )
