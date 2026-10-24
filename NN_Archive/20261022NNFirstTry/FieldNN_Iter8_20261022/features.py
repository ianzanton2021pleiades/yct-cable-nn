"""Iter8 signed TDR features: Iter3's eight channels plus broad step scales."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import uniform_filter1d


HERE = Path(__file__).resolve().parent
CORE = HERE.parents[2] / "DifferentAgent" / "CodeX" / "Core16_Field"
if str(CORE) not in sys.path:
    sys.path.append(str(CORE))

from field_responses import algorithm1


def distance_features(imp, step, d):
    imp = np.asarray(imp, dtype=np.float64)
    step = np.asarray(step, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)

    ir = max(float(np.sqrt(np.mean(imp * imp))), 1e-12)
    sr = max(float(np.sqrt(np.mean(step * step))), 1e-12)
    dx = d[1] - d[0]
    width30 = max(3, int(round(30.0 / dx)))
    local30 = np.sqrt(
        uniform_filter1d(imp * imp, width30, mode="nearest") + (0.01 * ir) ** 2
    )
    detrend30 = step - uniform_filter1d(step, width30, mode="nearest")

    # Preserve the Iter3 channels and their arithmetic unchanged.
    features = [
        np.clip(imp / ir, -16, 16) / 4,
        np.clip(step / sr, -8, 8) / 4,
        np.full_like(imp, np.log10(ir) / 6),
        np.full_like(imp, np.log10(sr) / 6),
        np.clip(imp / local30, -6, 6) / 3,
        np.clip(detrend30 / sr, -4, 4),
        np.clip((np.log10(local30 + 1e-12) + 8) / 8, -1, 1),
        d / d[-1],
    ]

    for window_m in (90.0, 300.0):
        width = max(3, int(round(window_m / dx)))
        detrend = step - uniform_filter1d(step, width, mode="nearest")
        local = np.sqrt(
            uniform_filter1d(detrend * detrend, width, mode="nearest")
            + (0.001 * sr) ** 2
        )
        features.extend((
            np.clip(detrend / sr, -4, 4),
            np.clip(detrend / local, -6, 6) / 3,
        ))

    return np.stack(features).astype(np.float32)


def preprocess(frequency_hz, s11, cfg):
    f = np.asarray(frequency_hz, dtype=float)
    g = np.asarray(s11, dtype=complex)
    band, response = cfg["frequency"], cfg["response"]
    if (
        f[0] > band["start_hz"] * (1 + 1e-9)
        or f[-1] < band["stop_hz"] * (1 - 1e-9)
    ):
        raise ValueError("Measured S11 does not cover the configured band")

    keep = (f >= band["start_hz"]) & (f <= band["stop_hz"])
    fi = np.unique(np.r_[band["start_hz"], f[keep], band["stop_hz"]])
    gi = np.interp(fi, f, g.real) + 1j * np.interp(fi, f, g.imag)
    result = algorithm1(
        fi,
        gi,
        velocity_m_s=response["velocity_m_s"],
        distance_max_m=response["distance_max_m"],
    )
    d = np.linspace(
        0.0, response["distance_max_m"], response["points"], dtype=np.float64
    )
    imp = np.interp(d, result["distance_m"], result["impulse_raw"])
    step = np.interp(d, result["distance_m"], result["step_rho"])
    return distance_features(imp, step, d)
