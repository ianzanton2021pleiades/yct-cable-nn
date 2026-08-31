"""External library wrapper for ShortBNCNet V2."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .common import PROJECT_ROOT, import_torch, make_result, restrict_to_visible_range


MODEL_INFO = {
    "key": "short_bncnet_v2",
    "name": "ShortBNCNet V2",
    "description": "5通道短电缆 TCN 模型 V2 版本。",
    "checkpoint": "ShortModel/experiments/short_bncnet_v2/best.pt",
    "threshold": 0.30,
    "end_threshold": 0.30,
    "prominence": 0.04,
    "min_peak_distance_m": 1.0,
    "block_until_m": 8.0,
}


def _moving_average(values: np.ndarray, width: int) -> np.ndarray:
    width = max(3, int(width) | 1)
    kernel = np.ones(width, dtype=np.float64) / width
    return np.convolve(values, kernel, mode="same")


def _robust_scale(values: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    q = np.percentile(np.abs(arr), 99.5)
    scale = max(float(q), floor)
    return np.clip(arr / scale, -3.0, 3.0) / 3.0


def _build_short_features(
    distance: np.ndarray,
    impulse_response: np.ndarray,
    step_response: np.ndarray,
    grid_max_m: float = 220.0,
    grid_step_m: float = 0.1,
    local_start_m: float = 15.0,
    energy_window_m: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray]:
    distance = np.asarray(distance, dtype=np.float64)
    impulse = np.asarray(impulse_response)
    if np.iscomplexobj(impulse):
        impulse = impulse.real
    impulse = np.asarray(impulse, dtype=np.float64)
    step = np.asarray(step_response, dtype=np.float64)
    if min(len(distance), len(impulse), len(step)) < 2:
        raise ValueError("too few points")

    n_points = int(round(grid_max_m / grid_step_m))
    grid = np.linspace(0.0, grid_max_m, n_points, endpoint=False) + grid_step_m / 2.0
    imp_grid = np.interp(grid, distance, impulse)
    step_grid = np.interp(grid, distance, step)

    local_start_idx = max(0, min(len(grid) - 1, int(round(local_start_m / grid_step_m))))
    impulse_global = _robust_scale(imp_grid)

    local_scale = max(float(np.percentile(np.abs(imp_grid[local_start_idx:]), 99.0)), 1e-10)
    impulse_local = np.clip(imp_grid / local_scale, -3.0, 3.0) / 3.0
    impulse_local[:local_start_idx] = np.clip(impulse_local[:local_start_idx], -0.5, 0.5)

    trend_width = max(11, int(round(8.0 / grid_step_m)))
    step_trend = _moving_average(step_grid, trend_width)
    step_detrended = _robust_scale(step_grid - step_trend)

    step_gradient = _robust_scale(np.gradient(step_grid, grid_step_m))

    energy_width = max(3, int(round(energy_window_m / grid_step_m)))
    local_energy = _robust_scale(np.sqrt(_moving_average(imp_grid ** 2, energy_width)))

    features = np.stack(
        [impulse_global, impulse_local, step_detrended, step_gradient, local_energy],
        axis=0,
    ).astype(np.float32)
    return grid.astype(np.float64), features


def _load_short_model_class():
    model_path = PROJECT_ROOT / "ShortModel" / "src" / "model.py"
    spec = importlib.util.spec_from_file_location("shortmodel_external_model", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load ShortModel: {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ShortBNCNet


def _run_model(features: np.ndarray, device: Optional[str] = None) -> np.ndarray:
    torch = import_torch()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = PROJECT_ROOT / MODEL_INFO["checkpoint"]
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")

    ShortBNCNet = _load_short_model_class()
    cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    model_cfg = cfg.get("model", {})
    model = ShortBNCNet(
        in_channels=int(model_cfg.get("in_channels", 5)),
        hidden_channels=int(model_cfg.get("hidden_channels", 48)),
        out_channels=int(model_cfg.get("out_channels", 3)),
        dropout=float(model_cfg.get("dropout", 0.03)),
    )

    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    x = torch.from_numpy(features).unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        _, probs_t = model(x)
    return probs_t.squeeze(0).detach().cpu().numpy().astype(np.float64)


def _extract_events(
    probs: np.ndarray,
    grid: np.ndarray,
    threshold: float,
    prominence: float,
    min_distance_m: float,
    block_until_m: float,
) -> List[Dict[str, float]]:
    try:
        from scipy.signal import find_peaks
        dd = float(grid[1] - grid[0]) if len(grid) > 1 else 0.1
        min_distance = max(1, int(round(min_distance_m / dd)))
        peaks, props = find_peaks(probs, height=threshold, prominence=prominence, distance=min_distance)
        events = []
        for j, idx in enumerate(peaks):
            pos = float(grid[idx])
            if pos < block_until_m:
                continue
            events.append({"position_m": pos, "confidence": float(probs[idx])})
        return events
    except Exception:
        events = []
        dd = float(grid[1] - grid[0]) if len(grid) > 1 else 0.1
        min_sep = max(1, int(round(min_distance_m / dd)))
        for idx in range(1, len(probs) - 1):
            if grid[idx] < block_until_m:
                continue
            if probs[idx] >= threshold and probs[idx] >= probs[idx - 1] and probs[idx] >= probs[idx + 1]:
                if not events or idx - int(round(events[-1]["position_m"] / dd)) >= min_sep:
                    events.append({"position_m": float(grid[idx]), "confidence": float(probs[idx])})
        return events


def analyze(
    s11: Optional[Dict[str, np.ndarray]],
    distance: np.ndarray,
    impulse_response: np.ndarray,
    step_response: np.ndarray,
    visible_distance_range: Optional[Tuple[float, float]] = None,
    device: Optional[str] = None,
    **_: Any,
) -> Dict[str, Any]:
    grid, features = _build_short_features(distance, impulse_response, step_response)
    probs = _run_model(features, device=device)
    if probs.shape[0] < 2:
        raise ValueError(f"ShortModel output shape error: {probs.shape}")

    bnc_confidence = np.clip(probs[0], 0.0, 1.0)
    end_confidence = np.clip(probs[1], 0.0, 1.0)
    confidence = np.maximum(bnc_confidence, end_confidence)

    plot_grid, plot_confidence = restrict_to_visible_range(grid, confidence, visible_distance_range)
    plot_bnc_grid, plot_bnc = restrict_to_visible_range(grid, bnc_confidence, visible_distance_range)
    plot_end_grid, plot_end = restrict_to_visible_range(grid, end_confidence, visible_distance_range)

    defect_events = _extract_events(
        plot_bnc, plot_bnc_grid,
        threshold=float(MODEL_INFO["threshold"]),
        prominence=float(MODEL_INFO["prominence"]),
        min_distance_m=float(MODEL_INFO["min_peak_distance_m"]),
        block_until_m=float(MODEL_INFO["block_until_m"]),
    )
    end_events = _extract_events(
        plot_end, plot_end_grid,
        threshold=float(MODEL_INFO["end_threshold"]),
        prominence=float(MODEL_INFO["prominence"]),
        min_distance_m=float(MODEL_INFO["min_peak_distance_m"]),
        block_until_m=float(MODEL_INFO["block_until_m"]),
    )
    end_event = max(end_events, key=lambda item: item["confidence"], default=None)

    result = make_result(
        model_name=MODEL_INFO["name"],
        grid=plot_grid,
        confidence=plot_confidence,
        visible_distance_range=None,
        threshold=float(MODEL_INFO["threshold"]),
    )
    result.update({
        "end_position_m": None if end_event is None else end_event["position_m"],
        "end_confidence": None if end_event is None else end_event["confidence"],
        "defect_positions_m": [item["position_m"] for item in defect_events],
        "defect_confidences": [item["confidence"] for item in defect_events],
    })
    return result
