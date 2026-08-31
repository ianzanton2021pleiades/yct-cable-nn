"""External library wrapper for UNet1D V1 full-training baseline."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .common import PROJECT_ROOT, build_input_channels, ensure_src_on_path, make_result, run_torch_model


MODEL_INFO = {
    "key": "unet1d_baseline_full",
    "name": "UNet1D Baseline (Full Train)",
    "description": "2通道 V1 模型完整训练版本（29 epoch early-stop），Recall=0.905。",
    "checkpoint": "AgentsStorage/experiments/exp_20260617_150349_unet1d_medium_baseline/best.pt",
    "threshold": 0.2,
}


def _build_model():
    ensure_src_on_path()
    from models.unet1d import UNet1D

    return UNet1D(base_ch=32, in_channels=2, out_channels=1, kernel_size=7)


def analyze(
    s11: Optional[Dict[str, np.ndarray]],
    distance: np.ndarray,
    impulse_response: np.ndarray,
    step_response: np.ndarray,
    visible_distance_range: Optional[Tuple[float, float]] = None,
    device: Optional[str] = None,
    **_: Any,
) -> Dict[str, Any]:
    grid, channels = build_input_channels(distance, impulse_response, step_response, in_channels=2)
    checkpoint_path = PROJECT_ROOT / MODEL_INFO["checkpoint"]
    confidence = run_torch_model(_build_model(), checkpoint_path, channels, device=device)
    confidence = np.clip(confidence, 0.0, 1.0)
    return make_result(
        model_name=MODEL_INFO["name"],
        grid=grid,
        confidence=confidence,
        visible_distance_range=visible_distance_range,
        threshold=float(MODEL_INFO["threshold"]),
    )
