"""Independent Client-compatible S11-to-response implementation for DG V3."""
from __future__ import annotations

from typing import Any

import numpy as np

from storage import DatasetProtocolError

SPEED_OF_LIGHT = 299_792_458.0


def _arrays(frequency_hz: np.ndarray, s11: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    values = np.asarray(s11, dtype=np.complex128)
    if frequency.ndim != 1 or values.ndim != 1 or len(frequency) != len(values) or len(frequency) < 3:
        raise DatasetProtocolError("frequency and s11 must be aligned one-dimensional arrays (at least 3 points)")
    if not np.all(np.isfinite(frequency)) or not np.all(np.isfinite(values.real)) or not np.all(np.isfinite(values.imag)):
        raise DatasetProtocolError("frequency and s11 must be finite")
    order = np.argsort(frequency)
    frequency, values = frequency[order], values[order]
    if np.any(frequency <= 0) or np.any(np.diff(frequency) <= 0):
        raise DatasetProtocolError("frequency must be strictly increasing and positive")
    return frequency, values


def fft_shift(arr: np.ndarray, inverse: bool = False) -> np.ndarray:
    """The odd-length half exchange used by the Client implementation."""
    values = np.asarray(arr)
    k = len(values) // 2
    if inverse:
        return np.concatenate((values[k:], values[:k]))
    return np.concatenate((values[-k:], values[:-k]))


def estimate_first_step(frequency_hz: np.ndarray, quantile: float = 5.0) -> float:
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    differences = np.diff(np.unique(np.sort(frequency)))
    differences = differences[differences > 0]
    if not len(differences):
        raise DatasetProtocolError("at least two distinct frequencies are required")
    step = float(np.percentile(differences, quantile))
    mean_step = float(np.mean(differences))
    if len(differences) > 1 and step < mean_step / 10.0:
        step = mean_step
    return step


def _window(n: int, name: str) -> np.ndarray:
    if name == "hann":
        return np.hanning(n)
    if name in {"rectangular", "none"}:
        return np.ones(n)
    raise DatasetProtocolError(f"client_hann_v1 only permits Hann; got window={name!r}")


def build_equally_spaced_spectrum(frequency_hz: np.ndarray, s11: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Construct the centered odd-length Hermitian spectrum without valid-data extrapolation."""
    frequency, values = _arrays(frequency_hz, s11)
    first_step = estimate_first_step(frequency)
    fmax = float(frequency[-1])
    steps = int(np.floor(fmax / first_step))
    if steps < 2:
        raise DatasetProtocolError("frequency span is too small for Client IFFT")
    # Positive bins are all inside the measured interval.  Only bin zero uses
    # the Client's explicit DC extrapolation below.
    positive_frequency = np.arange(1, steps + 1, dtype=np.float64) * first_step
    if positive_frequency[0] < frequency[0] - np.finfo(np.float64).eps * max(1.0, frequency[0]):
        raise DatasetProtocolError(
            "frequency grid would require extrapolation below the measured minimum; "
            "supply a sweep whose first step is within the measured band"
        )
    positive = np.interp(positive_frequency, frequency, values.real) + 1j * np.interp(positive_frequency, frequency, values.imag)
    spectrum = np.zeros(2 * steps + 1, dtype=np.complex128)
    spectrum[steps + 1 :] = positive
    spectrum[:steps] = np.conj(positive[::-1])
    # Client-compatible magnitude/phase linear DC extrapolation.
    abs_dc = 2.0 * abs(positive[0]) - abs(positive[1])
    phase_dc = 2.0 * np.angle(positive[0]) - np.angle(positive[1])
    spectrum[steps] = abs_dc * np.exp(1j * phase_dc)
    return spectrum, first_step, steps


def s11_to_responses(
    frequency_hz: np.ndarray,
    s11: np.ndarray,
    *,
    epsr: float = 2.23,
    distance_step_m: float = 0.25,
    target_distance_max_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Convert S11 to 0.25 m distance responses with variable (coverage-limited) length.

    The returned coverage dictionary is part of the protocol.  Interpolation is
    performed only on the native IFFT interval; no valid-distance extrapolation
    is performed.  DC is the sole explicit Client extrapolation.
    """
    epsr = float(epsr)
    distance_step_m = float(distance_step_m)
    if not np.isfinite(epsr) or epsr <= 0 or not np.isfinite(distance_step_m) or distance_step_m != 0.25:
        raise DatasetProtocolError("DG V3 requires positive epsr and distance_step_m=0.25")
    frequency, values = _arrays(frequency_hz, s11)
    spectrum, first_step, steps = build_equally_spaced_spectrum(frequency, values)
    spectrum *= _window(len(spectrum), "hann")
    time_domain = np.fft.ifft(fft_shift(spectrum, inverse=True))
    dt_s = 1.0 / (first_step * len(spectrum))
    native_distance = np.arange(len(spectrum), dtype=np.float64) * dt_s * SPEED_OF_LIGHT / (2.0 * np.sqrt(epsr))
    native_step = np.real(np.cumsum(time_domain) * dt_s)
    requested_max = float(target_distance_max_m)
    if not np.isfinite(requested_max) or requested_max <= 0:
        raise DatasetProtocolError("target_distance_max_m must be positive")
    valid_max = min(requested_max, float(native_distance[-1]))
    n_grid = int(np.floor(valid_max / distance_step_m + 1.0e-12)) + 1
    distance = np.arange(n_grid, dtype=np.float64) * distance_step_m
    # np.interp is used only over [native_distance[0], native_distance[-1]].
    impulse_real = np.interp(distance, native_distance, time_domain.real)
    impulse_imag = np.interp(distance, native_distance, time_domain.imag)
    step = np.interp(distance, native_distance, native_step)
    coverage = {
        "source_frequency_min_hz": float(frequency[0]),
        "source_frequency_max_hz": float(frequency[-1]),
        "first_step_hz": float(first_step),
        "native_distance_max_m": float(native_distance[-1]),
        "target_distance_max_m": requested_max,
        "distance_min_m": float(distance[0]),
        "distance_max_m": float(distance[-1]),
        "valid_distance_max_m": float(distance[-1]),
        "distance_step_m": distance_step_m,
        "point_count": int(len(distance)),
        "dc_mode": "client_extrapolation",
        "window": "hann",
        "algorithm": "client_hann_v1",
        "valid_distance_extrapolated": False,
        "truncated_by_ifft_range": bool(native_distance[-1] + distance_step_m < requested_max),
    }
    # The public protocol stores real/imag separately and has a real step curve.
    return distance, impulse_real, impulse_imag, step, coverage


def client_hann_v1(
    frequency_hz: np.ndarray,
    s11: np.ndarray,
    *,
    epsr: float = 2.23,
    distance_step_m: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Named profile entry point used by callers selecting the DG V3 profile."""
    return s11_to_responses(frequency_hz, s11, epsr=epsr, distance_step_m=distance_step_m)
