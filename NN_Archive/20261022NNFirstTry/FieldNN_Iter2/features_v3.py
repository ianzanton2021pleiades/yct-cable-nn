"""8-channel distance-domain features built from raw impulse/step responses.

The key change from the first version is distance-dependent gain plus local
detrending/normalization, which keeps far-end reflections visible instead of
letting the near-end fixture dominate the global RMS.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "DifferentAgent" / "CodeX" / "Core16_Field"))
from field_responses import algorithm1


def rolling_rms(values: np.ndarray, width_points: int) -> np.ndarray:
    return np.sqrt(uniform_filter1d(values * values, size=width_points, mode="nearest") + 1e-12)


def rolling_mean(values: np.ndarray, width_points: int) -> np.ndarray:
    return uniform_filter1d(values, size=width_points, mode="nearest")


def robust_norm(values: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    scale = float(np.percentile(np.abs(values), percentile))
    return values / scale if scale > 1e-12 else values


def distance_features(imp: np.ndarray, step: np.ndarray, d: np.ndarray, gain_tau: float = 200.0,
                      local_width_m: float = 50.0) -> np.ndarray:
    spacing = float(d[1] - d[0])
    local_points = max(5, int(round(local_width_m / spacing)))
    gain = 1.0 + d / float(gain_tau)
    detrend = step - rolling_mean(step, local_points * 2)
    detrend_local = detrend / rolling_rms(step, local_points)
    detrend_local = np.clip(detrend_local, -8.0, 8.0) / 8.0
    detrend_gain = robust_norm(detrend * gain)
    step_diff = robust_norm(np.abs(np.gradient(step, d)) * gain)
    imp_gain = robust_norm(np.abs(imp) * gain)
    imp_abs = robust_norm(np.abs(imp))
    log_imp = np.clip((np.log10(np.abs(imp) * gain + 1e-6) + 6.0) / 8.0, 0.0, 1.0)
    return np.stack([
        detrend_local,
        np.clip(detrend_gain, -2.0, 2.0),
        np.clip(robust_norm(detrend), -2.0, 2.0),
        np.clip(step_diff, 0.0, 2.0),
        np.clip(imp_gain, 0.0, 2.0),
        np.clip(imp_abs, 0.0, 2.0),
        log_imp,
        d / d[-1],
    ]).astype(np.float32)


def preprocess_s11_v3(frequency_hz, s11, cfg, gain_tau: float = 200.0, local_width_m: float = 50.0) -> np.ndarray:
    f, g = np.asarray(frequency_hz, dtype=float), np.asarray(s11, dtype=complex)
    if f.ndim != 1 or g.shape != f.shape or len(f) < 4:
        raise ValueError("S11 must contain at least four frequency/complex pairs")
    if not np.isfinite(f).all() or not np.isfinite(g).all() or np.any(np.diff(f) <= 0):
        raise ValueError("S11 frequency must be finite and strictly increasing; S11 must be finite")
    band = cfg["frequency"]
    if f[0] > band["start_hz"] * (1 + 1e-9) or f[-1] < band["stop_hz"] * (1 - 1e-9):
        raise ValueError("Input does not cover the configured frequency band")
    keep = (f >= band["start_hz"]) & (f <= band["stop_hz"])
    fi = np.unique(np.r_[band["start_hz"], f[keep], band["stop_hz"]])
    gi = np.interp(fi, f, g.real) + 1j * np.interp(fi, f, g.imag)
    r = cfg["response"]
    result = algorithm1(fi, gi, velocity_m_s=r["velocity_m_s"], distance_max_m=r["distance_max_m"])
    d = np.linspace(0.0, r["distance_max_m"], r["points"])
    imp = np.interp(d, result["distance_m"], result["impulse_raw"])
    step = np.interp(d, result["distance_m"], result["step_rho"])
    return distance_features(imp, step, d, gain_tau, local_width_m)


def raw_responses_from_old_inputs(inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover raw impulse/step from the first-version 4-channel input."""
    imp = inputs[0] * (10.0 ** inputs[2])
    step = inputs[1] * (10.0 ** inputs[3])
    return imp.astype(np.float32), step.astype(np.float32)
