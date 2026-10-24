"""Eleven-channel TDR front end: the frozen eight Iter3 channels plus three
local-contrast views at scales the baseline does not expose.

Why extra channels
------------------
The baseline feature stack only carries one local-contrast scale (30 m, channel
4) and one detrended-step scale (30 m, channel 5).  Reflections from a joint or
a water tree are 2-10 m wide, and the fixture ringing that buries them is
20-80 m wide.  A single 30 m window cannot separate the two, so the network has
to infer the separation from the raw signed trace.  Three extra channels give it
the separation directly:

    8  impulse minus its 8 m local mean, divided by the 8 m local RMS   sharp transients
    9  impulse minus its 100 m local mean, divided by the global RMS    broad reflections
    10 step minus its 8 m local mean, divided by the global step RMS    local impedance slope

The base eight channels keep their positions so the distance axis stays at index
7 and the cached field tensors remain reusable.

The traces are clipped to the same 16-RMS bound that channel 0 already applies,
in both the raw and the reconstructed path, so the two paths produce identical
extra channels instead of differing inside the near-end fixture spike.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CORE = PROJECT_ROOT / "DifferentAgent" / "CodeX" / "Core16_Field"
ITER3 = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_Iter3_20261022"
for path in (CORE, ITER3):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from features import distance_features  # noqa: E402


BASE_CHANNELS = 8
FEATURE_CHANNELS = 11
SHARP_WINDOW_M = 8.0
BROAD_WINDOW_M = 100.0
CLIP_IN_RMS = 16.0


def _windows(distance: np.ndarray) -> tuple[int, int]:
    spacing = float(distance[1] - distance[0])
    return (
        max(3, int(round(SHARP_WINDOW_M / spacing))),
        max(3, int(round(BROAD_WINDOW_M / spacing))),
    )


def contrast_channels(impulse, step, distance, ir: float, sr: float) -> np.ndarray:
    """The three new channels given the global RMS of each trace."""
    dx_sharp, dx_broad = _windows(distance)
    impulse = np.clip(np.asarray(impulse, dtype=np.float64), -CLIP_IN_RMS * ir, CLIP_IN_RMS * ir)
    step = np.clip(np.asarray(step, dtype=np.float64), -CLIP_IN_RMS * sr, CLIP_IN_RMS * sr)

    sharp_mean = uniform_filter1d(impulse, dx_sharp, mode="nearest")
    sharp_rms = np.sqrt(uniform_filter1d(impulse * impulse, dx_sharp, mode="nearest") + (0.01 * ir) ** 2)
    channel_sharp = np.clip((impulse - sharp_mean) / sharp_rms, -6.0, 6.0) / 3.0

    broad_mean = uniform_filter1d(impulse, dx_broad, mode="nearest")
    channel_broad = np.clip((impulse - broad_mean) / ir, -4.0, 4.0) / 2.0

    step_local = uniform_filter1d(step, dx_sharp, mode="nearest")
    channel_step = np.clip((step - step_local) / sr, -4.0, 4.0) / 2.0

    return np.stack((channel_sharp, channel_broad, channel_step)).astype(np.float32)


def distance_features_v2(impulse, step, distance) -> np.ndarray:
    """Full eleven-channel stack for a raw trace pair."""
    impulse = np.asarray(impulse, dtype=np.float64)
    step = np.asarray(step, dtype=np.float64)
    ir = max(float(np.sqrt(np.mean(impulse * impulse))), 1e-12)
    sr = max(float(np.sqrt(np.mean(step * step))), 1e-12)
    base = distance_features(impulse, step, distance)
    return np.concatenate((base, contrast_channels(impulse, step, distance, ir, sr)), axis=0)


def extend_features(base: np.ndarray, distance) -> np.ndarray:
    """Extend cached eight-channel features without the raw traces.

    The base stack is invertible up to the symmetric clip at 16 RMS: channel 2
    carries ``log10(impulse RMS)/6`` and channel 0 carries ``impulse/(4*RMS)``,
    so the trace is recovered as ``channel0*4*10**(6*channel2)``.  Both paths
    then clip at 16 RMS, so the reconstruction is exact rather than approximate.
    """
    base = np.asarray(base, dtype=np.float32)
    if base.shape[0] < BASE_CHANNELS:
        raise ValueError("base features must have at least eight channels")
    impulse_rms = float(10.0 ** (6.0 * float(np.median(base[2]))))
    step_rms = float(10.0 ** (6.0 * float(np.median(base[3]))))
    impulse = base[0].astype(np.float64) * 4.0 * impulse_rms
    step = base[1].astype(np.float64) * 4.0 * step_rms
    return np.concatenate(
        (base[:BASE_CHANNELS], contrast_channels(impulse, step, distance, impulse_rms, step_rms)),
        axis=0,
    )
