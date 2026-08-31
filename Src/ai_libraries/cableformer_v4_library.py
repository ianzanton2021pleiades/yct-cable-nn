"""External library wrapper for CableFormer V4 (relaxed constraints)."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .common import PROJECT_ROOT, build_input_channels, ensure_src_on_path, make_result, run_torch_model


MODEL_INFO = {
    "key": "cableformer_v4_relaxed",
    "name": "CableFormer V4 Relaxed",
    "description": "2通道 CNN-Transformer 混合模型，放松约束训练版本。",
    "checkpoint": "AgentsStorage/experiments/exp_20260618_035421_cableformer_medium_v4/best.pt",
    "threshold": 0.15,
}


def _build_model():
    ensure_src_on_path()
    from models.cableformer import CableFormer

    return CableFormer(
        base_ch=48,
        in_channels=2,
        out_channels=1,
        transformer_dim=256,
        n_transformer_blocks=3,
        n_heads=8,
        use_cbam=True,
        dropout=0.1,
    )


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
