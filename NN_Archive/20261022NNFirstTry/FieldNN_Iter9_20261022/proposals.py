"""Physical proposal generation and candidate-centered patch extraction."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks


PROPOSAL_CHANNELS = (0, 1, 4, 5, 6, 8, 9, 10, 11)
PATCH_POINTS = 256
MIN_CENTER_M = 30.0
MAX_CENTER_M = 3240.0
MIN_SEPARATION_M = 10.0
MAX_PEAKS_PER_MAP = 20


def _local_maxima(score: np.ndarray, distance: np.ndarray) -> list[tuple[float, float]]:
    spacing_m = float(np.median(np.diff(distance)))
    separation_points = MIN_SEPARATION_M / spacing_m
    peak_indices, _ = find_peaks(score, distance=separation_points)
    in_range = (distance[peak_indices] >= MIN_CENTER_M) & (distance[peak_indices] <= MAX_CENTER_M)
    peak_indices = peak_indices[in_range]
    ordered = sorted(peak_indices.tolist(), key=lambda i: (-float(score[i]), i))
    selected = ordered[:MAX_PEAKS_PER_MAP]
    return [(float(distance[i]), float(score[i])) for i in selected]


def _score_maps(features: np.ndarray, distance: np.ndarray) -> tuple[np.ndarray, ...]:
    spacing_m = float(np.median(np.diff(distance)))
    width90 = max(3, int(round(90.0 / spacing_m)))
    if width90 % 2 == 0:
        width90 += 1
    pulse = np.asarray(features[0], dtype=np.float64)
    local_rms90 = np.sqrt(uniform_filter1d(pulse * pulse, width90, mode="nearest"))
    pulse_over_rms90 = np.abs(pulse) / np.maximum(local_rms90, 1e-12)
    return (
        np.abs(features[4]),
        pulse_over_rms90,
        np.abs(features[8]),
        np.abs(features[10]),
    )


def propose(features: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Return sorted nominal-distance centers from four physical response maps."""
    features = np.asarray(features)
    distance = np.asarray(distance, dtype=np.float64)
    maps = _score_maps(features, distance)

    candidates: list[tuple[float, float, int, int]] = []
    for map_index, score_map in enumerate(maps):
        peaks = _local_maxima(np.asarray(score_map), distance)
        map_scale = max((strength for _, strength in peaks), default=0.0)
        for rank, (center, strength) in enumerate(peaks):
            relative_strength = strength / map_scale if map_scale > 0.0 else 0.0
            candidates.append((center, relative_strength, map_index, rank))

    # Map-relative peak strength avoids comparing scores with different scales.
    candidates.sort(key=lambda item: (-item[1], item[3], item[0], item[2]))
    selected: list[tuple[float, float, int, int]] = []
    for candidate in candidates:
        if all(
            abs(candidate[0] - kept[0]) >= MIN_SEPARATION_M
            for kept in selected
        ):
            selected.append(candidate)

    return np.asarray(sorted(item[0] for item in selected), dtype=np.float32)


def extract_patch(
    features: np.ndarray,
    distance: np.ndarray,
    center: float,
) -> np.ndarray:
    """Extract two normalized views as a float32 ``[20, 256]`` patch."""
    features = np.asarray(features, dtype=np.float32)
    distance = np.asarray(distance, dtype=np.float64)
    views: list[np.ndarray] = []

    for half_width_m in (50.0, 200.0):
        view_distance = np.linspace(
            float(center) - half_width_m,
            float(center) + half_width_m,
            PATCH_POINTS,
            dtype=np.float64,
        )
        view = np.stack(
            [np.interp(view_distance, distance, features[channel]) for channel in PROPOSAL_CHANNELS]
        ).astype(np.float32, copy=False)

        pulse_rms = float(np.sqrt(np.mean(view[0] * view[0])))
        view[0] /= max(pulse_rms, 1e-12)

        view[1] -= float(np.median(view[1]))
        step_rms = float(np.sqrt(np.mean(view[1] * view[1])))
        view[1] /= max(step_rms, 1e-12)

        np.clip(view, -6.0, 6.0, out=view)
        view /= 3.0
        relative_coordinate = np.linspace(-1.0, 1.0, PATCH_POINTS, dtype=np.float32)
        views.append(np.concatenate((view, relative_coordinate[None, :]), axis=0))

    return np.concatenate(views, axis=0).astype(np.float32, copy=False)
