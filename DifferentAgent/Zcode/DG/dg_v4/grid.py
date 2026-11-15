"""Frequency grids matching the field wideband impedance analyzer.

Measured fact (2026-10 survey of 188 .fdr files): the instrument sweeps a
logarithmic frequency axis  f_i = f_start * r**i  with r = (fmax/fmin)**(1/(N-1)).
Bands seen in the field: 9 kHz-1 GHz @ 50000 pt (standard), 9 kHz-1 GHz @ 10000 pt
(fast), 9 kHz-200 MHz @ 50000 pt (long cables / 220 kV).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

BANDS: dict[str, dict] = {
    # name: (fmin, fmax, points, share)
    "1ghz_std":  {"fmin": 9.0e3, "fmax": 1.0e9, "points": 50000, "share": 0.80},
    "1ghz_fast": {"fmin": 9.0e3, "fmax": 1.0e9, "points": 10000, "share": 0.15},
    "200mhz":    {"fmin": 9.0e3, "fmax": 2.0e8, "points": 50000, "share": 0.05},
}


def log_frequencies(fmin_hz: float, fmax_hz: float, points: int) -> np.ndarray:
    ratio = (fmax_hz / fmin_hz) ** (1.0 / (points - 1))
    return fmin_hz * ratio ** np.arange(points, dtype=np.float64)


@dataclass(frozen=True)
class BandAxis:
    name: str
    freq_hz: np.ndarray

    @property
    def fmax(self) -> float:
        return float(self.freq_hz[-1])


def get_band_axis(name: str) -> BandAxis:
    spec = BANDS[name]
    return BandAxis(name, log_frequencies(spec["fmin"], spec["fmax"], spec["points"]))


def sample_band_variant(rng: np.random.Generator) -> str:
    names = list(BANDS)
    weights = np.asarray([BANDS[n]["share"] for n in names])
    return str(rng.choice(names, p=weights / weights.sum()))
