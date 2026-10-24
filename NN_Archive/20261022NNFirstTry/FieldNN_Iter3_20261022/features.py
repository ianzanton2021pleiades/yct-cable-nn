"""Signed multiscale TDR features; identical path for synthetic and measured S11."""
from pathlib import Path
import sys
import numpy as np
from scipy.ndimage import uniform_filter1d
CORE = Path(__file__).resolve().parents[3] / "DifferentAgent" / "CodeX" / "Core16_Field"
sys.path.append(str(CORE))
from field_responses import algorithm1


def distance_features(imp, step, d):
    imp, step = np.asarray(imp, dtype=np.float64), np.asarray(step, dtype=np.float64)
    ir = max(float(np.sqrt(np.mean(imp * imp))), 1e-12)
    sr = max(float(np.sqrt(np.mean(step * step))), 1e-12)
    width = max(3, int(round(30.0 / (d[1] - d[0]))))
    local = np.sqrt(uniform_filter1d(imp * imp, width, mode="nearest") + (0.01 * ir)**2)
    detrend = step - uniform_filter1d(step, width, mode="nearest")
    return np.stack([
        np.clip(imp / ir, -16, 16) / 4,
        np.clip(step / sr, -8, 8) / 4,
        np.full_like(imp, np.log10(ir) / 6),
        np.full_like(imp, np.log10(sr) / 6),
        np.clip(imp / local, -6, 6) / 3,
        np.clip(detrend / sr, -4, 4),
        np.clip((np.log10(local + 1e-12) + 8) / 8, -1, 1),
        d / d[-1],
    ]).astype(np.float32)


def preprocess(frequency_hz, s11, cfg):
    f, g = np.asarray(frequency_hz, float), np.asarray(s11, complex)
    band, r = cfg["frequency"], cfg["response"]
    if f[0] > band["start_hz"] * (1 + 1e-9) or f[-1] < band["stop_hz"] * (1 - 1e-9):
        raise ValueError("Measured S11 does not cover the configured band")
    keep = (f >= band["start_hz"]) & (f <= band["stop_hz"])
    fi = np.unique(np.r_[band["start_hz"], f[keep], band["stop_hz"]])
    gi = np.interp(fi, f, g.real) + 1j * np.interp(fi, f, g.imag)
    result = algorithm1(fi, gi, velocity_m_s=r["velocity_m_s"], distance_max_m=r["distance_max_m"])
    d = np.linspace(0, r["distance_max_m"], r["points"])
    return distance_features(np.interp(d, result["distance_m"], result["impulse_raw"]),
                             np.interp(d, result["distance_m"], result["step_rho"]), d)
