"""S11 -> spatial responses using the instrument's own convention.

Reverse-engineered 2026-10-29 (see DifferentAgent/Zcode/DG/02_变换口径标定.md):
the .fdr files' embedded StepResponse/ImpulseResponse tables equal, to float32
precision, the discrete sine-transform

    step(t) = (2/pi) * sum_i (CL_i + CL_{i+1})/2 * (cos(w_i t) - cos(w_{i+1} t)) / t
    CL_i = Re(S11_i) / w_i,          w = 2*pi*f
    t = linspace(1ns, 3*L/v, N_t),   dist = v/2 * t - offset
    impulse[k] = (step[k+1]-step[k]) / (dt * c * g),  g = 0.6813 (per-file const)

using ONLY Re(S11), no window, no mean removal, valid on the native log grid.
It reproduces the vendor table of the 770 m hospital cable with peak
positions/ammplitudes/floor all equal (corr 1-1e-10).  offset = v * tau_fixture
(=4.3817 m for the 25.5 ns clamp fixture), so the junction sits at 0 m.

This is the canonical transform for synthetic AND field curves.  Outputs are
resampled onto a fixed 0.25 m distance grid.
"""
from __future__ import annotations

import numpy as np

C_M_S = 3.0e8
IMPULSE_G = 0.681305944919586
DISTANCE_STEP_M = 0.25


def vendor_step(f_hz: np.ndarray, s11_re: np.ndarray, t_s: np.ndarray,
                chunk: int = 128) -> np.ndarray:
    """step(t) on an arbitrary time grid, from S11 real part (any freq grid).

    step(t) = (2/pi)/t * sum_i (CL_i+CL_{i+1})/2 * (cos(w_i t) - cos(w_{i+1} t))
    """
    w = 2.0 * np.pi * np.asarray(f_hz, np.float64)
    cl = np.asarray(s11_re, np.float64) / w
    weights = (0.5 * (cl[:-1] + cl[1:])).astype(np.float32)
    w32 = w.astype(np.float32)
    step = np.empty(t_s.size, dtype=np.float32)
    for s in range(0, t_s.size, chunk):
        tc = t_s[s:s + chunk].astype(np.float32)
        cosw = np.cos(np.outer(tc, w32))          # (n_t, n_f)
        acc = (cosw[:, :-1] - cosw[:, 1:]) @ weights
        step[s:s + chunk] = (acc / tc).astype(np.float32) * (2.0 / np.pi)
    return step


def s11_to_responses(f_log: np.ndarray, s11: np.ndarray, v_axis_m_per_s: float,
                     length_setting_m: float, offset_m: float,
                     distance_step_m: float = DISTANCE_STEP_M,
                     dist_max_m: float | None = None):
    """Return (distance_m, impulse, step, coverage) on the 0.25 m grid."""
    f = np.asarray(f_log, np.float64)
    re = np.real(np.asarray(s11, np.complex128))
    if dist_max_m is None:
        dist_max_m = 1.5 * length_setting_m
    n_out = int(np.floor(float(dist_max_m) / distance_step_m + 1e-12)) + 1
    dist_out = np.arange(n_out, dtype=np.float64) * distance_step_m
    # evaluate step directly on the requested distance grid (t = 2(dist+offset)/v)
    t_s = 2.0 * (dist_out + offset_m) / v_axis_m_per_s
    step = vendor_step(f, re, t_s)
    impulse = np.zeros_like(step)
    impulse[1:] = np.diff(step) / (distance_step_m * 2.0 / v_axis_m_per_s * C_M_S * IMPULSE_G)
    coverage = {
        "distance_step_m": distance_step_m,
        "distance_max_m": float(dist_out[-1]),
        "transform": "vendor_sine_transform_g0.6813",
        "v_axis_m_per_s": v_axis_m_per_s,
        "offset_m": float(offset_m),
        "band_hz": (float(f[0]), float(f[-1])),
    }
    return dist_out, impulse, step, coverage
