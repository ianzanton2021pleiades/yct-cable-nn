"""Shared helpers for external TDR neural-network analysis libraries."""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "Src"
DEFAULT_MODEL_D_MAX = 1200.0
DEFAULT_MODEL_DD = 0.5


def ensure_src_on_path() -> None:
    import sys

    src = str(SRC_ROOT)
    if src not in sys.path:
        sys.path.insert(0, src)


def as_real_array(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if np.iscomplexobj(arr):
        arr = arr.real
    return arr.astype(np.float64, copy=False)


def normalize_by_abs(values: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    arr = as_real_array(values)
    scale = max(float(np.max(np.abs(arr))) if arr.size else 0.0, eps)
    return arr / scale


def fixed_grid_from_distance(
    distance: np.ndarray,
    impulse: np.ndarray,
    step: np.ndarray,
    d_max: float = DEFAULT_MODEL_D_MAX,
    dd: float = DEFAULT_MODEL_DD,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    distance = as_real_array(distance)
    impulse = as_real_array(impulse)
    step = as_real_array(step)
    if distance.ndim != 1 or impulse.ndim != 1 or step.ndim != 1:
        raise ValueError("distance, impulse_response, step_response 必须是一维数组")
    if min(len(distance), len(impulse), len(step)) < 2:
        raise ValueError("距离域响应点数过少，无法进行神经网络分析")

    n_points = int(round(d_max / dd))
    grid = np.linspace(0.0, d_max, n_points, endpoint=False) + dd / 2.0
    imp_grid = np.interp(grid, distance, impulse)
    step_grid = np.interp(grid, distance, step)
    return grid, imp_grid.astype(np.float64), step_grid.astype(np.float64)


def build_input_channels(
    distance: np.ndarray,
    impulse: np.ndarray,
    step: np.ndarray,
    in_channels: int,
    d_max: float = DEFAULT_MODEL_D_MAX,
    dd: float = DEFAULT_MODEL_DD,
    local_skip_m: float = 15.0,
) -> Tuple[np.ndarray, np.ndarray]:
    grid, imp_grid, step_grid = fixed_grid_from_distance(distance, impulse, step, d_max=d_max, dd=dd)

    if in_channels == 2:
        channels = np.stack(
            [normalize_by_abs(imp_grid), normalize_by_abs(step_grid)],
            axis=0,
        )
    elif in_channels == 4:
        skip_idx = max(0, min(len(grid) - 1, int(round(local_skip_m / dd))))
        imp_global = normalize_by_abs(imp_grid)
        imp_local = imp_grid / max(float(np.max(np.abs(imp_grid[skip_idx:]))), 1e-10)
        imp_local[:skip_idx] = np.clip(imp_local[:skip_idx], -1.0, 1.0)

        step_global = normalize_by_abs(step_grid)
        step_local = step_grid / max(float(np.max(np.abs(step_grid[skip_idx:]))), 1e-10)
        step_local[:skip_idx] = np.clip(step_local[:skip_idx], -1.0, 1.0)
        channels = np.stack([imp_global, imp_local, step_global, step_local], axis=0)
    else:
        raise ValueError(f"不支持的模型输入通道数: {in_channels}")

    return grid, channels.astype(np.float32)


def import_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local runtime
        raise RuntimeError(
            "当前 Python 环境无法导入 torch。请用项目要求的 Python 3.11(gpushare_cu124) 环境运行。"
        ) from exc
    return torch


def load_checkpoint_state(checkpoint_path: Path) -> Dict[str, Any]:
    torch = import_torch()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到模型权重文件: {checkpoint_path}")

    try:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    for key in ("model_state_dict", "state_dict", "model"):
        if isinstance(checkpoint, dict) and key in checkpoint:
            state = checkpoint[key]
            break
    else:
        state = checkpoint

    if not isinstance(state, dict):
        raise ValueError(f"无法从权重文件解析 state_dict: {checkpoint_path}")

    cleaned = {}
    for key, value in state.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value
    return cleaned


def run_torch_model(model: Any, checkpoint_path: Path, channels: np.ndarray, device: Optional[str] = None) -> np.ndarray:
    torch = import_torch()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    state = load_checkpoint_state(checkpoint_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    critical_missing = [key for key in missing if not key.startswith("head.")]
    if critical_missing or unexpected:
        raise ValueError(
            f"权重与模型结构不匹配: missing={list(missing)[:6]}, unexpected={list(unexpected)[:6]}"
        )

    model.to(device)
    model.eval()
    x = torch.from_numpy(channels).unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        output = model(x)
        probs = output[1] if isinstance(output, (tuple, list)) else output
        probs = probs.squeeze().detach().cpu().numpy()
    return np.asarray(probs, dtype=np.float64)


def select_peaks(
    grid: np.ndarray,
    confidence: np.ndarray,
    threshold: float,
    min_distance_m: float = 2.0,
    max_peaks: int = 8,
) -> List[Dict[str, float]]:
    grid = as_real_array(grid)
    confidence = as_real_array(confidence)
    if len(grid) != len(confidence):
        raise ValueError("grid 与 confidence 长度不一致")

    dd = float(np.median(np.diff(grid))) if len(grid) > 2 else DEFAULT_MODEL_DD
    min_sep = max(1, int(round(min_distance_m / max(dd, 1e-9))))
    candidates: List[int] = []
    for idx in range(1, len(confidence) - 1):
        if confidence[idx] >= threshold and confidence[idx] >= confidence[idx - 1] and confidence[idx] >= confidence[idx + 1]:
            candidates.append(idx)

    candidates.sort(key=lambda i: float(confidence[i]), reverse=True)
    chosen: List[int] = []
    for idx in candidates:
        if all(abs(idx - old) >= min_sep for old in chosen):
            chosen.append(idx)
        if len(chosen) >= max_peaks:
            break
    chosen.sort()
    return [{"position_m": float(grid[i]), "confidence": float(confidence[i])} for i in chosen]


def infer_end_and_defects(peaks: List[Dict[str, float]]) -> Tuple[Optional[Dict[str, float]], List[Dict[str, float]]]:
    if not peaks:
        return None, []
    end_event = max(peaks, key=lambda item: item["position_m"])
    defects = [item for item in peaks if item is not end_event]
    return end_event, defects


def restrict_to_visible_range(
    distance: np.ndarray,
    confidence: np.ndarray,
    visible_distance_range: Optional[Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    if visible_distance_range is None:
        return distance, confidence
    x0, x1 = visible_distance_range
    lo, hi = sorted((float(x0), float(x1)))
    mask = (distance >= lo) & (distance <= hi)
    if not np.any(mask):
        return distance, confidence
    return distance[mask], confidence[mask]


def make_result(
    model_name: str,
    grid: np.ndarray,
    confidence: np.ndarray,
    visible_distance_range: Optional[Tuple[float, float]],
    threshold: float,
) -> Dict[str, Any]:
    plot_distance, plot_confidence = restrict_to_visible_range(grid, confidence, visible_distance_range)
    peaks = select_peaks(plot_distance, plot_confidence, threshold=threshold)
    end_event, defects = infer_end_and_defects(peaks)
    return {
        "model_name": model_name,
        "distance": plot_distance,
        "confidence": plot_confidence,
        "events": peaks,
        "end_position_m": None if end_event is None else end_event["position_m"],
        "end_confidence": None if end_event is None else end_event["confidence"],
        "defect_positions_m": [item["position_m"] for item in defects],
        "defect_confidences": [item["confidence"] for item in defects],
        "threshold": threshold,
    }


def discover_libraries() -> Dict[str, Dict[str, Any]]:
    libs: Dict[str, Dict[str, Any]] = {}
    package = __package__ or "ai_libraries"
    for path in sorted(Path(__file__).resolve().parent.glob("*_library.py")):
        module_name = f"{package}.{path.stem}"
        try:
            module = importlib.import_module(module_name)
            analyze = getattr(module, "analyze", None)
            info = getattr(module, "MODEL_INFO", None)
            if callable(analyze) and isinstance(info, dict):
                key = str(info.get("key") or path.stem)
                libs[key] = {"module": module, "info": info, "analyze": analyze}
        except Exception as exc:
            key = path.stem
            libs[key] = {
                "module": None,
                "info": {"key": key, "name": key, "load_error": str(exc)},
                "analyze": None,
            }
    return libs
