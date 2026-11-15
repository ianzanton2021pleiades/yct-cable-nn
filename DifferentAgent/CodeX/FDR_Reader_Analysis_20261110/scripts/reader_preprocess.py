"""Display-only MovingMean used by FDR_Reader's Analysis comparison charts.

Source: embedded Analyser.Calculation.Movmean(double[,], int, int), decompiled
to extracted/ui/Calculation.cs lines 42-74. This helper returns a new array;
it never changes the reader's saved response table.
"""
from __future__ import annotations

import numpy as np


def movmean(data: np.ndarray, k: int, column: int = 1) -> np.ndarray:
    """Match the reader's clipped, centered window over one response column."""
    source = np.asarray(data, dtype=np.float64)
    result = np.empty((source.shape[0], 2), dtype=np.float64)
    result[:, 0] = source[:, 0]
    if k > 1:
        radius = k // 2
        for i in range(source.shape[0]):
            lo = max(0, i - radius)
            hi = min(source.shape[0] - 1, i + radius)
            result[i, 1] = np.mean(source[lo:hi + 1, column])
    else:
        result[:, 1] = source[:, column]
    return result


def support_points(k: int) -> int:
    """Number of samples in an interior window; even K uses K+1 samples."""
    return 1 if k <= 1 else 2 * (k // 2) + 1


def sample_step_m(data: np.ndarray) -> float:
    distance = np.asarray(data, dtype=np.float64)[:, 0]
    return float(np.median(np.diff(distance)))


def span_m(k: int, delta_m: float) -> float:
    return (support_points(k) - 1) * delta_m
