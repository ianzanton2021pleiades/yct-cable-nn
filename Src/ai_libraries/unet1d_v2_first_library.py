"""External library wrapper for UNet1D V2 first training run."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .common import PROJECT_ROOT, build_input_channels, ensure_src_on_path, make_result, run_torch_model


MODEL_INFO = {
    "key": "unet1d_v2_first_run",
    "name": "UNet1D V2 First Run",
    "description": "4通道 V2 模型第一次训练，End Recall=0.941。",
    "checkpoint": "AgentsStorage/experiments/exp_20260618_170217_unet1d_v2_medium/best.pt",
    "threshold": 0.2,
}


def _build_model():
    ensure_src_on_path()
    from models.unet1d_v2 import UNet1DV2

    return UNet1DV2(base_ch=48, in_channels=4, out_channels=1, use_cbam=True)


def analyze(
    s11: Optional[Dict[str, np.ndarray]],
    distance: np.ndarray,
    impulse_response: np.ndarray,
    step_response: np.ndarray,
    visible_distance_range: Optional[Tuple[float, float]] = None,
    device: Optional[str] = None,
    **_: Any,
) -> Dict[str, Any]:
    grid, channels = build_input_channels(distance, impulse_response, step_response, in_channels=4)
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
