"""Shared positive-only field metrics and checkpoint inference helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from decode import decode_predictions
from models import build_model


WET_CASE = "2026-6某地进行的1500m不同位置电缆接头浸水试验"


def load_model(checkpoint_path: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    model = build_model(checkpoint["model_name"], **cfg["model"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model, checkpoint, cfg


@torch.inference_mode()
def collect_probabilities(
    models: list[torch.nn.Module],
    dataset,
    device: torch.device,
    batch_size: int = 64,
    amp: bool = True,
) -> tuple[list[np.ndarray], list[int]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    probabilities: list[np.ndarray] = []
    indices: list[int] = []
    for batch in loader:
        if len(batch) == 2:
            inputs, item = batch
        elif len(batch) == 3:
            inputs, _, item = batch
        else:
            inputs, _, item, _ = batch
        inputs = inputs.to(device, non_blocking=device.type == "cuda")
        total = None
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            for model in models:
                value = model(inputs).float().sigmoid()
                total = value if total is None else total + value
        total = total / max(len(models), 1)
        probabilities.extend(total.cpu().numpy())
        indices.extend(int(value) for value in item.tolist())
    return probabilities, indices


def match_events(predicted: list[dict[str, Any]], truth: list[float], tolerance: float) -> int:
    used: set[int] = set()
    hits = 0
    for position in truth:
        best: tuple[float, int] | None = None
        for index, event in enumerate(predicted):
            if index in used:
                continue
            error = abs(float(event["distance_m"]) - float(position))
            if error <= tolerance and (best is None or error < best[0]):
                best = (error, index)
        if best is not None:
            used.add(best[1])
            hits += 1
    return hits


def metrics_from_pairs(
    predictions: list[dict[str, Any]],
    truth_positions: list[list[float]],
    group_names: list[str],
) -> dict[str, Any]:
    tolerances = (10.0, 20.0, 30.0)
    rows = []
    for prediction, truth, group in zip(predictions, truth_positions, group_names):
        if not truth:
            continue
        rows.append({
            "group": group,
            "truth": len(truth),
            "candidates": len(prediction["events"]),
            **{f"hits_{int(t)}": match_events(prediction["events"], truth, t) for t in tolerances},
        })
    if not rows:
        return {"curves": 0, "truth_events": 0, "candidates": 0, "candidates_per_curve": None}
    result: dict[str, Any] = {
        "curves": len(rows),
        "truth_events": sum(row["truth"] for row in rows),
        "candidates": sum(row["candidates"] for row in rows),
        "candidates_per_curve": float(np.mean([row["candidates"] for row in rows])),
    }
    for tolerance in tolerances:
        key = f"hits_{int(tolerance)}"
        result[f"hit_rate_{int(tolerance)}"] = float(
            sum(row[key] for row in rows) / max(sum(row["truth"] for row in rows), 1)
        )
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(row)
    result["group_macro_hit_20"] = float(np.mean([
        sum(item["hits_20"] for item in values) / max(sum(item["truth"] for item in values), 1)
        for values in groups.values()
    ])) if groups else None
    result["groups"] = len(groups)
    return result


def decode_all(
    probabilities: list[np.ndarray],
    cfg: dict[str, Any],
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    return [decode_predictions(item, cfg, thresholds) for item in probabilities]


def hybrid_metrics(
    probabilities: list[np.ndarray],
    records: list[dict[str, Any]],
    cfg: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    predictions = decode_all(probabilities, cfg, thresholds)
    truth = [
        sorted({round(float(event["position_m"]), 3) for event in record["injected_events"]})
        for record in records
    ]
    groups = [str(record.get("case_group") or "") for record in records]
    result = metrics_from_pairs(predictions, truth, groups)
    strict = [index for index, record in enumerate(records) if record.get("strict")]
    if strict:
        result["strict"] = metrics_from_pairs(
            [predictions[index] for index in strict],
            [truth[index] for index in strict],
            [groups[index] for index in strict],
        )
    non_wet = [
        index for index, record in enumerate(records)
        if record.get("case_group") != WET_CASE
    ]
    if non_wet:
        result["non_wet"] = metrics_from_pairs(
            [predictions[index] for index in non_wet],
            [truth[index] for index in non_wet],
            [groups[index] for index in non_wet],
        )
    return result


def field_metrics(
    probabilities: list[np.ndarray],
    records: list[dict[str, Any]],
    cfg: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    predictions = decode_all(probabilities, cfg, thresholds)
    truth = [list(record.get("known_positions_nominal_m") or []) for record in records]
    groups = [str(record.get("case_group") or "") for record in records]
    result = metrics_from_pairs(predictions, truth, groups)
    strict = [index for index, record in enumerate(records) if record.get("strict")]
    if strict:
        result["strict"] = metrics_from_pairs(
            [predictions[index] for index in strict],
            [truth[index] for index in strict],
            [groups[index] for index in strict],
        )
    non_wet = [
        index for index, record in enumerate(records)
        if record.get("case_group") != WET_CASE
    ]
    if non_wet:
        result["non_wet"] = metrics_from_pairs(
            [predictions[index] for index in non_wet],
            [truth[index] for index in non_wet],
            [groups[index] for index in non_wet],
        )
    wet = [
        index for index, record in enumerate(records)
        if record.get("case_group") == WET_CASE
    ]
    if wet:
        result["wet_1500"] = metrics_from_pairs(
            [predictions[index] for index in wet],
            [truth[index] for index in wet],
            [groups[index] for index in wet],
        )
    return result


def calibration_metrics(
    probabilities: list[np.ndarray],
    cfg: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    false_events = 0
    false_intervals = 0
    any_false = 0
    for probability in probabilities:
        prediction = decode_predictions(probability, cfg, thresholds)
        events = [event for event in prediction["events"] if float(event["distance_m"]) >= 30.0]
        intervals = [item for item in prediction["intervals"] if float(item["end_m"]) >= 30.0]
        false_events += len(events)
        false_intervals += len(intervals)
        any_false += int(bool(events or intervals))
    count = max(len(probabilities), 1)
    return {
        "references": len(probabilities),
        "false_event_count": false_events,
        "false_events_per_reference": false_events / count,
        "false_interval_count": false_intervals,
        "fraction_with_false_event_or_interval": any_false / count,
    }
