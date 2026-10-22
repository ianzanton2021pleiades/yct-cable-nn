from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


DEFAULT_THRESHOLDS = {"event": 0.5, "interval": 0.5, "terminal": 0.5}
TERMINAL_EDGE_EXCLUSION_M = 10.0


def _response_grid(cfg: dict[str, Any], length: int | None = None) -> np.ndarray:
    response = cfg["response"]
    points = int(response["points"])
    distance_max_m = float(response["distance_max_m"])
    if length is not None and length != points:
        raise ValueError(f"probability length {length} does not match cfg.response.points={points}")
    return np.linspace(0.0, distance_max_m, points, dtype=np.float64)


def _local_peaks(values: np.ndarray, threshold: float) -> list[int]:
    """Return one center index per above-threshold local-maximum plateau."""
    peaks: list[int] = []
    index = 0
    while index < values.size:
        end = index
        while end + 1 < values.size and values[end + 1] == values[index]:
            end += 1
        value = float(values[index])
        left = float(values[index - 1]) if index > 0 else -np.inf
        right = float(values[end + 1]) if end + 1 < values.size else -np.inf
        if value >= threshold and value >= left and value >= right and (value > left or value > right):
            peaks.append((index + end) // 2)
        index = end + 1
    return peaks


def _separated_peaks(values: np.ndarray, distances: np.ndarray, threshold: float,
                     minimum_separation_m: float) -> list[int]:
    candidates = _local_peaks(values, threshold)
    candidates.sort(key=lambda index: (-float(values[index]), float(distances[index])))
    selected: list[int] = []
    for candidate in candidates:
        if all(abs(float(distances[candidate] - distances[chosen])) >= minimum_separation_m
               for chosen in selected):
            selected.append(candidate)
    return sorted(selected, key=lambda index: float(distances[index]))


def _threshold_intervals(values: np.ndarray, distances: np.ndarray, threshold: float,
                         minimum_width_m: float) -> list[dict[str, float]]:
    above = values >= threshold
    intervals: list[dict[str, float]] = []
    index = 0
    while index < above.size:
        if not above[index]:
            index += 1
            continue
        start = index
        while index + 1 < above.size and above[index + 1]:
            index += 1
        end = index
        width = float(distances[end] - distances[start])
        if width >= minimum_width_m:
            intervals.append({
                "start_m": float(distances[start]),
                "end_m": float(distances[end]),
                "confidence": float(np.mean(values[start:end + 1])),
            })
        index += 1
    return intervals


def decode_predictions(probabilities: np.ndarray, cfg: dict[str, Any],
                       thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    """Decode event, interval, and independent-terminal probability tracks.

    The input has three rows in the order event, defect interval, terminal. The
    configured point count is the model's distance grid and must match L.
    """
    tracks = np.asarray(probabilities, dtype=np.float64)
    if tracks.ndim != 2 or tracks.shape[0] != 3:
        raise ValueError("probabilities must have shape (3, L): event, interval, terminal")
    distances = _response_grid(cfg, tracks.shape[1])
    settings = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        settings.update({key: float(value) for key, value in thresholds.items()})
    evaluation = cfg["evaluation"]

    event_indices = _separated_peaks(
        tracks[0], distances, settings["event"], float(evaluation["event_min_separation_m"])
    )
    events = [
        {"distance_m": float(distances[index]), "confidence": float(tracks[0, index])}
        for index in event_indices
    ]

    intervals = _threshold_intervals(
        tracks[1], distances, settings["interval"], float(evaluation["interval_min_width_m"])
    )

    terminal_indices = _local_peaks(tracks[2], settings["terminal"])
    distance_max_m = float(cfg["response"]["distance_max_m"])
    terminal_indices = [
        index for index in terminal_indices
        if TERMINAL_EDGE_EXCLUSION_M < float(distances[index])
        < distance_max_m - TERMINAL_EDGE_EXCLUSION_M
    ]
    terminal_index = max(
        terminal_indices,
        key=lambda index: (float(tracks[2, index]), -float(distances[index])),
        default=None,
    )
    terminal = None if terminal_index is None else {
        "distance_m": float(distances[terminal_index]),
        "confidence": float(tracks[2, terminal_index]),
    }
    return {"events": events, "intervals": intervals, "terminal": terminal}


def _position(value: Any) -> float:
    if isinstance(value, dict):
        return float(value.get("distance_m", value.get("position_m")))
    return float(value)


def _interval(value: Any) -> tuple[float, float]:
    if isinstance(value, dict):
        start = float(value["start_m"])
        end = float(value["end_m"])
    else:
        start, end = float(value[0]), float(value[1])
    return (start, end) if start <= end else (end, start)


def _interval_iou(left: tuple[float, float], right: tuple[float, float]) -> float:
    intersection = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    if union == 0.0:
        return 1.0 if left == right else 0.0
    return float(intersection / union)


def _maximum_cardinality_matching(left_count: int, right_count: int,
                                  pair_cost: np.ndarray, compatible: np.ndarray,
                                  maximum_compatible_cost: float) -> list[tuple[int, int]]:
    """Maximize compatible one-to-one matches, then minimize their total cost."""
    if left_count == 0 or right_count == 0:
        return []
    unmatched_cost = (min(left_count, right_count) + 1) * (maximum_compatible_cost + 1.0)
    forbidden_cost = (left_count + right_count + 1) * unmatched_cost * 2.0
    size = left_count + right_count
    costs = np.zeros((size, size), dtype=np.float64)
    costs[:left_count, :right_count] = np.where(compatible, pair_cost, forbidden_cost)
    costs[:left_count, right_count:] = unmatched_cost
    costs[left_count:, :right_count] = unmatched_cost
    rows, columns = linear_sum_assignment(costs)
    return [
        (int(row), int(column))
        for row, column in zip(rows, columns)
        if row < left_count and column < right_count and compatible[row, column]
    ]


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _prf(matched: int, predicted: int, actual: int) -> dict[str, float | int | None]:
    precision = _safe_ratio(matched, predicted)
    recall = _safe_ratio(matched, actual)
    f1 = float(2 * matched / (predicted + actual)) if predicted + actual else 1.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _event_metrics(predictions: list[dict[str, Any]], truth: list[dict[str, Any]],
                   tolerances: Sequence[float]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for tolerance in tolerances:
        matched_count = 0
        prediction_count = 0
        truth_count = 0
        errors: list[float] = []
        for prediction, target in zip(predictions, truth):
            predicted_positions = [_position(item) for item in prediction.get("events", [])]
            truth_positions = [_position(item) for item in target.get("event_positions_m", [])]
            prediction_count += len(predicted_positions)
            truth_count += len(truth_positions)
            if not predicted_positions or not truth_positions:
                continue
            distances = np.abs(np.asarray(truth_positions)[:, None] - np.asarray(predicted_positions)[None, :])
            compatible = distances <= tolerance
            matches = _maximum_cardinality_matching(
                len(truth_positions), len(predicted_positions), distances, compatible, float(tolerance)
            )
            matched_count += len(matches)
            errors.extend(float(distances[left, right]) for left, right in matches)

        key = str(int(tolerance)) if float(tolerance).is_integer() else str(float(tolerance))
        result[key] = {
            "tolerance_m": float(tolerance),
            "curve_count": len(truth),
            "truth_event_count": truth_count,
            "prediction_event_count": prediction_count,
            "matched_event_count": matched_count,
            **_prf(matched_count, prediction_count, truth_count),
            "matched_truth_coverage": _safe_ratio(matched_count, truth_count),
            "matched_prediction_coverage": _safe_ratio(matched_count, prediction_count),
            "mean_matched_mae_m": float(np.mean(errors)) if errors else None,
        }
    return result


def _interval_metrics(predictions: list[dict[str, Any]], truth: list[dict[str, Any]],
                      cfg: dict[str, Any]) -> dict[str, Any]:
    iou_threshold = 0.5
    predicted_count = 0
    truth_count = 0
    matched_count = 0
    matched_ious: list[float] = []
    distances = _response_grid(cfg)
    grid_intersection = 0
    grid_union = 0

    for prediction, target in zip(predictions, truth):
        predicted_intervals = [_interval(item) for item in prediction.get("intervals", [])]
        truth_intervals = [_interval(item) for item in target.get("defect_intervals_m", [])]
        predicted_count += len(predicted_intervals)
        truth_count += len(truth_intervals)

        predicted_mask = np.zeros(distances.size, dtype=bool)
        truth_mask = np.zeros(distances.size, dtype=bool)
        for start, end in predicted_intervals:
            predicted_mask |= (distances >= start) & (distances <= end)
        for start, end in truth_intervals:
            truth_mask |= (distances >= start) & (distances <= end)
        grid_intersection += int(np.count_nonzero(predicted_mask & truth_mask))
        grid_union += int(np.count_nonzero(predicted_mask | truth_mask))

        if not predicted_intervals or not truth_intervals:
            continue
        ious = np.asarray([
            [_interval_iou(actual, predicted) for predicted in predicted_intervals]
            for actual in truth_intervals
        ], dtype=np.float64)
        compatible = ious >= iou_threshold
        matches = _maximum_cardinality_matching(
            len(truth_intervals), len(predicted_intervals), 1.0 - ious,
            compatible, 1.0 - iou_threshold,
        )
        matched_count += len(matches)
        matched_ious.extend(float(ious[left, right]) for left, right in matches)

    return {
        "iou_threshold": iou_threshold,
        "curve_count": len(truth),
        "truth_interval_count": truth_count,
        "prediction_interval_count": predicted_count,
        "matched_interval_count": matched_count,
        **_prf(matched_count, predicted_count, truth_count),
        "matched_truth_coverage": _safe_ratio(matched_count, truth_count),
        "matched_prediction_coverage": _safe_ratio(matched_count, predicted_count),
        "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else None,
        "grid_overall_iou": float(grid_intersection / grid_union) if grid_union else 1.0,
        "grid_intersection_points": grid_intersection,
        "grid_union_points": grid_union,
        "grid_point_count_total": len(truth) * int(cfg["response"]["points"]),
    }


def _terminal_metrics(predictions: list[dict[str, Any]], truth: list[dict[str, Any]],
                      tolerance_m: float) -> dict[str, Any]:
    truth_count = 0
    prediction_count = 0
    matched_count = 0
    all_pair_errors: list[float] = []
    matched_errors: list[float] = []
    for prediction, target in zip(predictions, truth):
        predicted = prediction.get("terminal")
        actual = target.get("terminal_position_m")
        predicted_position = None if predicted is None else _position(predicted)
        actual_position = None if actual is None else _position(actual)
        truth_count += actual_position is not None
        prediction_count += predicted_position is not None
        if predicted_position is None or actual_position is None:
            continue
        error = abs(predicted_position - actual_position)
        all_pair_errors.append(error)
        if error <= tolerance_m:
            matched_count += 1
            matched_errors.append(error)

    return {
        "tolerance_m": float(tolerance_m),
        "curve_count": len(truth),
        "truth_terminal_curve_count": int(truth_count),
        "prediction_terminal_curve_count": int(prediction_count),
        "terminal_hit_count": matched_count,
        "terminal_miss_count": int(truth_count - matched_count),
        "terminal_false_positive_count": int(prediction_count - matched_count),
        "terminal_hit_rate": _safe_ratio(matched_count, int(truth_count)),
        **_prf(matched_count, int(prediction_count), int(truth_count)),
        "mean_terminal_error_m": float(np.mean(all_pair_errors)) if all_pair_errors else None,
        "mean_matched_terminal_error_m": float(np.mean(matched_errors)) if matched_errors else None,
    }


def evaluate_predictions(predictions: list[dict[str, Any]], truth: list[dict[str, Any]],
                         cfg: dict[str, Any]) -> dict[str, Any]:
    """Evaluate predictions against complete synthetic truth only.

    Unlabeled field records must not be passed here: an absent field label is
    unknown, not a negative. Empty annotation lists are real negatives only for
    fully labeled synthetic curves.
    """
    if len(predictions) != len(truth):
        raise ValueError("predictions and truth must contain the same number of curves")
    if any(item.get("label_status") in {"unknown", "unlabeled", "unlabelled"} for item in truth):
        raise ValueError("complete truth is required; unknown field labels are not negatives")
    _response_grid(cfg)
    evaluation = cfg["evaluation"]
    tolerances = [float(value) for value in evaluation["event_tolerances_m"]]
    event_metrics = _event_metrics(predictions, truth, tolerances)
    return {
        "curve_count": len(truth),
        "event_metrics_by_tolerance_m": event_metrics,
        "interval_metrics": _interval_metrics(predictions, truth, cfg),
        "terminal_metrics": _terminal_metrics(
            predictions, truth, float(evaluation["terminal_tolerance_m"])
        ),
    }
