"""Export existing FieldNN predictions, positive-only field metrics, and case plots."""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
ITER5_DIR = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_Iter5_20261022"
ITER6_DIR = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_Iter6_20261022"
ITER8_DIR = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_Iter8_20261022"
ITER8_DATA = Path(r"E:\CableNNDataSet\CodeX\FieldNN_Iter8_20261022")
ITER8_OUTPUT = ITER8_DIR / "output" / "iter8"
ITER5_UNET_PREDICTIONS = (
    ITER5_DIR / "output" / "iter5" / "unet" / "field_e06" / "predictions.jsonl"
)
ITER8_UNET_PREDICTIONS = (
    ITER8_OUTPUT / "paired" / "candidate" / "unet" / "field_fixed_e000" / "predictions.jsonl"
)
ITER8_TRANSFORMER_PREDICTIONS = (
    ITER8_OUTPUT / "paired" / "candidate" / "transformer" / "field_fixed_e010" / "predictions.jsonl"
)
ITER8_UNET_CHECKPOINT = ITER8_OUTPUT / "unet" / "best.pt"
ITER8_TRANSFORMER_CHECKPOINT = ITER8_OUTPUT / "transformer" / "best.pt"
ITER8_SUMMARY = ITER8_OUTPUT / "iteration8_summary.json"
TOLERANCES_M = (10.0, 20.0, 30.0)
MAX_CASES = 3
MIN_CASE_POSITIONS = 3
EVENT_CHANNEL = 0
STEP_CHANNEL = 1
FIELD_AXES_MAX_M = 3250.0

CSV_FIELDS = (
    "row_type",
    "model",
    "threshold",
    "split",
    "population",
    "groups",
    "bags",
    "curves",
    "known_positions",
    "hits_at_10m",
    "hits_at_20m",
    "hits_at_30m",
    "micro_hit_at_10m",
    "micro_hit_at_20m",
    "micro_hit_at_30m",
    "group_macro_hit_at_10m",
    "group_macro_hit_at_20m",
    "group_macro_hit_at_30m",
    "mean_matched_error_at_10m",
    "mean_matched_error_at_20m",
    "mean_matched_error_at_30m",
    "median_matched_error_at_20m",
    "candidate_events_total",
    "candidate_events_per_curve",
    "candidate_events_unverified_or_matched",
    "false_events",
    "calibration_references",
    "synthetic_view",
    "synthetic_event_precision_at_5m",
    "synthetic_event_recall_at_5m",
    "synthetic_event_f1_at_5m",
    "synthetic_interval_f1_iou_0_5",
    "synthetic_terminal_f1_at_10m",
    "parameters",
    "model_state_tensor_bytes",
    "inference_package_bytes",
    "checkpoint_bytes",
    "latency_median_ms",
    "latency_p95_ms",
    "latency_scope",
    "notes",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_field_evaluator():
    path = ITER6_DIR / "evaluate_field.py"
    spec = importlib.util.spec_from_file_location("iter10_field_evaluator", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load field evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prediction_index(
    records: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    threshold: float,
    evaluator,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    by_key = {
        evaluator.record_key(row.get("record") or {}): row
        for row in prediction_rows
    }
    result: dict[int, dict[str, Any]] = {}
    for record in records:
        key = evaluator.record_key(record)
        source = by_key.get(key)
        if source is None:
            continue
        row = copy.deepcopy(source)
        prediction = row.get("prediction") or {}
        prediction["events"] = [
            event
            for event in prediction.get("events", [])
            if float(event.get("confidence", 0.0)) >= threshold
        ]
        row["prediction"] = prediction
        result[int(record["index"])] = row

    scored_rows = [row for row in prediction_rows if row.get("status") == "predicted"]
    total_events = sum(
        sum(
            float(event.get("confidence", 0.0)) >= threshold
            for event in (row.get("prediction") or {}).get("events", [])
        )
        for row in scored_rows
    )
    return result, {
        "prediction_rows": len(prediction_rows),
        "predicted_rows": len(scored_rows),
        "candidate_events": int(total_events),
        "mean_candidates_per_predicted_curve": (
            total_events / len(scored_rows) if scored_rows else None
        ),
        "missing_field_record_predictions": len(records) - len(result),
    }


def matched_error_summary(bag_reports: list[dict[str, Any]], tolerance: float, evaluator):
    claims, _ = evaluator.position_claims(bag_reports, tolerance)
    errors = [
        float(value["error_m"])
        for value in claims.values()
        if value.get("hit") and value.get("error_m") is not None
    ]
    return {
        "position_count": len(claims),
        "matched_count": len(errors),
        "hit_rate": len(errors) / len(claims) if claims else None,
        "mean_error_m": sum(errors) / len(errors) if errors else None,
        "median_error_m": median(errors) if errors else None,
        "errors_m": errors,
    }


def indices_for_bags(bags: list[dict[str, Any]]) -> set[int]:
    return {
        int(index)
        for bag in bags
        for index in bag.get("indices", [])
    }


def candidate_burden(indices: set[int], predictions_by_index: dict[int, dict[str, Any]]):
    counts = [
        len((predictions_by_index[index].get("prediction") or {}).get("events", []))
        for index in sorted(indices)
        if index in predictions_by_index
        and predictions_by_index[index].get("status") == "predicted"
    ]
    return {
        "curves": len(indices),
        "predicted_curves": len(counts),
        "candidate_events": sum(counts),
        "mean_candidates_per_curve": sum(counts) / len(counts) if counts else None,
    }


def positive_population_report(
    split: str,
    bags: list[dict[str, Any]],
    records_by_index: dict[int, dict[str, Any]],
    predictions_by_index: dict[int, dict[str, Any]],
    evaluator,
) -> dict[str, Any]:
    bag_reports = evaluator.positive_summary(bags, records_by_index, predictions_by_index)
    macro = bag_reports["independent_group_macro"]
    errors = {
        str(int(tolerance)): matched_error_summary(bag_reports["bags"], tolerance, evaluator)
        for tolerance in TOLERANCES_M
    }
    curves = indices_for_bags(bags)
    burden = candidate_burden(curves, predictions_by_index)
    return {
        "split": split,
        "bags": bags,
        "summary": bag_reports,
        "groups": bag_reports["groups"],
        "macro": macro,
        "errors": errors,
        "burden": burden,
        "curves": curves,
    }


def group_row(
    group_report: dict[str, Any],
    population: dict[str, Any],
    label: str,
    threshold: float,
    evaluator,
) -> dict[str, Any]:
    group = group_report["group"]
    bag_reports = [row for row in population["summary"]["bags"] if row["group"] == group]
    positions = group_report["by_tolerance"]["20"]["unique_bag_position_claims"]
    hits = {
        str(int(tolerance)): group_report["by_tolerance"][str(int(tolerance))][
            "matched_bag_position_claims"
        ]
        for tolerance in TOLERANCES_M
    }
    errors = {
        str(int(tolerance)): matched_error_summary(bag_reports, tolerance, evaluator)
        for tolerance in TOLERANCES_M
    }
    candidate_total = group_report["by_tolerance"]["20"]["candidate_total_unverified"]
    candidate_sweeps = group_report["by_tolerance"]["20"]["selected_best_sweeps"]
    case_groups = group_report.get("case_groups", [])
    return {
        "row_type": "field_group",
        "model": label,
        "threshold": threshold,
        "split": "test",
        "population": "nonwet_test_group",
        "groups": 1,
        "bags": group_report["bag_count"],
        "curves": sum(len(row["applicable_indices"]) for row in bag_reports),
        "known_positions": positions,
        "hits_at_10m": hits["10"],
        "hits_at_20m": hits["20"],
        "hits_at_30m": hits["30"],
        "group_macro_hit_at_10m": group_report["by_tolerance"]["10"]["hit_rate"],
        "group_macro_hit_at_20m": group_report["by_tolerance"]["20"]["hit_rate"],
        "group_macro_hit_at_30m": group_report["by_tolerance"]["30"]["hit_rate"],
        "micro_hit_at_10m": group_report["by_tolerance"]["10"]["hit_rate"],
        "micro_hit_at_20m": group_report["by_tolerance"]["20"]["hit_rate"],
        "micro_hit_at_30m": group_report["by_tolerance"]["30"]["hit_rate"],
        "mean_matched_error_at_10m": errors["10"]["mean_error_m"],
        "mean_matched_error_at_20m": errors["20"]["mean_error_m"],
        "mean_matched_error_at_30m": errors["30"]["mean_error_m"],
        "median_matched_error_at_20m": errors["20"]["median_error_m"],
        "candidate_events_total": candidate_total,
        "candidate_events_per_curve": (
            candidate_total / candidate_sweeps if candidate_sweeps else None
        ),
        "candidate_events_unverified_or_matched": candidate_total,
        "notes": "; ".join(case_groups),
    }


def population_row(label: str, threshold: float, population: dict[str, Any], name: str):
    macro = population["macro"]
    errors = population["errors"]
    burden = population["burden"]
    group_reports = population["groups"]
    positions = macro["20"]["unique_positions"] if "20" in macro else 0
    return {
        "row_type": "field_population",
        "model": label,
        "threshold": threshold,
        "split": population["split"],
        "population": name,
        "groups": len(group_reports),
        "bags": len(population["bags"]),
        "curves": burden["curves"],
        "known_positions": positions,
        "hits_at_10m": macro["10"]["matched_positions"],
        "hits_at_20m": macro["20"]["matched_positions"],
        "hits_at_30m": macro["30"]["matched_positions"],
        "micro_hit_at_10m": macro["10"]["micro_hit_rate"],
        "micro_hit_at_20m": macro["20"]["micro_hit_rate"],
        "micro_hit_at_30m": macro["30"]["micro_hit_rate"],
        "group_macro_hit_at_10m": macro["10"]["macro_hit_rate"],
        "group_macro_hit_at_20m": macro["20"]["macro_hit_rate"],
        "group_macro_hit_at_30m": macro["30"]["macro_hit_rate"],
        "mean_matched_error_at_10m": errors["10"]["mean_error_m"],
        "mean_matched_error_at_20m": errors["20"]["mean_error_m"],
        "mean_matched_error_at_30m": errors["30"]["mean_error_m"],
        "median_matched_error_at_20m": errors["20"]["median_error_m"],
        "candidate_events_total": burden["candidate_events"],
        "candidate_events_per_curve": burden["mean_candidates_per_curve"],
        "candidate_events_unverified_or_matched": burden["candidate_events"],
        "notes": "正位置袋监督；未匹配候选仍未核验，不计算现场precision/recall/F1。",
    }


def checkpoint_artifact(path: Path) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model") or checkpoint.get("state_dict")
    tensor_bytes = sum(
        int(value.numel()) * int(value.element_size())
        for value in state.values()
        if hasattr(value, "numel") and hasattr(value, "element_size")
    ) if isinstance(state, dict) else None
    run_info_path = path.parent / "run_info.json"
    run_info = read_json(run_info_path) if run_info_path.is_file() else {}
    config = checkpoint.get("config") or run_info.get("config") or {}
    return {
        "state_tensor_bytes": tensor_bytes,
        "checkpoint_bytes": path.stat().st_size,
        "parameters": run_info.get("parameters"),
        "model_name": checkpoint.get("model_name") or run_info.get("model"),
        "epoch": checkpoint.get("epoch"),
        "config": config,
        "run_info": run_info,
    }


def benchmark_info(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {
            "median_ms": None,
            "p95_ms": None,
            "parameters": None,
            "weight_tensor_bytes": None,
            "inference_package_bytes": None,
            "checkpoint_bytes": None,
            "scope": "未测；没有nn_benchmark.json",
        }
    raw = read_json(path)
    payload = raw.get("benchmark", raw.get("inference", raw))
    return {
        "median_ms": payload.get("median_ms"),
        "p95_ms": payload.get("p95_ms"),
        "parameters": payload.get("parameters"),
        "weight_tensor_bytes": payload.get("weight_tensor_bytes", payload.get("weight_bytes")),
        "inference_package_bytes": payload.get("inference_package_bytes"),
        "checkpoint_bytes": payload.get("checkpoint_bytes"),
        "scope": payload.get(
            "scope",
            "NN-only、输入已预载入；不含H2D、FFT或FDR读取，按benchmark文件说明",
        ),
    }


def calibration_at_threshold(calibration: dict[str, Any] | None, threshold: float):
    if calibration is None:
        return None
    result = copy.deepcopy(calibration)
    source_threshold = float(calibration.get("event_threshold", threshold))
    if threshold < source_threshold:
        result["threshold_note"] = (
            f"原报告只保存threshold={source_threshold:g}的候选，无法下推到更低threshold={threshold:g}。"
        )
        return result
    for split, item in result["splits"].items():
        entries = item.get("entries", [])
        if not entries:
            continue
        count = 0
        for entry in entries:
            events = entry.get("false_cable_events", [])
            count += sum(
                float(event.get("confidence", 0.0)) >= threshold
                for event in events
            )
        item["false_event_count"] = int(count)
        item["false_events_per_reference"] = (
            count / item["references"] if item.get("references") else None
        )
    result["event_threshold"] = float(threshold)
    result["threshold_note"] = "根据保存的higher/equal-threshold候选置信度离线计数；未重新推理。"
    return result


def synthetic_metrics_for(checkpoint_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for view in ("mixed", "clean"):
        path = checkpoint_dir / f"evaluation_test_{view}" / "evaluation.json"
        if not path.is_file():
            continue
        report = read_json(path)
        metrics = report["metrics"]
        result[view] = {
            "curve_count": metrics["curve_count"],
            "event_at_5m": metrics["event_metrics_by_tolerance_m"]["5"],
            "interval_f1": metrics["interval_metrics"]["f1"],
            "terminal_f1": metrics["terminal_metrics"]["f1"],
            "thresholds": report.get("thresholds"),
        }
    return result


def model_file_row(
    label: str,
    threshold: float,
    checkpoint: Path,
    parameters: int | None,
    state_tensor_bytes: int | None,
    latency: dict[str, Any],
) -> dict[str, Any]:
    return {
        "row_type": "model_artifact",
        "model": label,
        "threshold": threshold,
        "parameters": parameters,
        "model_state_tensor_bytes": state_tensor_bytes,
        "checkpoint_bytes": latency.get("checkpoint_bytes")
        or (checkpoint.stat().st_size if checkpoint.is_file() else None),
        "latency_median_ms": latency["median_ms"],
        "latency_p95_ms": latency["p95_ms"],
        "inference_package_bytes": latency.get("inference_package_bytes"),
        "latency_scope": latency["scope"],
        "notes": "parameters×4是FP32参数估值；checkpoint文件另含训练状态。",
    }


def choose_cases(population: dict[str, Any], evaluator) -> list[dict[str, Any]]:
    candidates = []
    for group_report in population["groups"]:
        group = str(group_report["group"])
        if "rg58" in group.casefold() or group_report["by_tolerance"]["20"][
            "unique_bag_position_claims"
        ] < MIN_CASE_POSITIONS:
            continue
        group_bags = [
            row
            for row in population["summary"]["bags"]
            if row["group"] == group and not row["known_lab"]
        ]
        if any(
            "rg58" in str(path).casefold()
            for row in group_bags
            for path in row.get("source_relative_paths", [])
        ):
            continue
        hit_count = group_report["by_tolerance"]["20"]["matched_bag_position_claims"]
        if hit_count <= 0:
            continue
        case_bags = []
        for bag in group_bags:
            best = bag["bag_best_single_sweep"].get("20")
            if best is None or best["matched_count"] <= 0:
                continue
            mean_error = best.get("mean_error_m")
            case_bags.append((
                best["matched_count"] / max(1, bag["position_count"]),
                best["matched_count"],
                -(float(mean_error) if mean_error is not None else float("inf")),
                bag,
                best,
            ))
        if not case_bags:
            continue
        _, _, _, bag, best = max(case_bags, key=lambda item: item[:3])
        positions = list(bag["positions_source_m"])
        candidates.append(
            {
                "group": group,
                "case_group": bag["case_group"],
                "source_relative_path": bag["source_identity"],
                "bag": bag,
                "best_sweep": best,
                "positions_source_m": positions,
                "position_count": bag["position_count"],
                "matched_count": best["matched_count"],
                "hit_rate_at_20m": best["matched_count"] / max(1, best["position_count"]),
                "group_hit_rate_at_20m": group_report["by_tolerance"]["20"]["hit_rate"],
                "mean_error_at_20m": best.get("mean_error_m"),
                "candidate_count": best["candidate_count"],
                "group_hit_count": hit_count,
                "group_position_count": group_report["by_tolerance"]["20"][
                    "unique_bag_position_claims"
                ],
            }
        )
    candidates.sort(
        key=lambda item: (
            -float(item["group_hit_rate_at_20m"] or 0.0),
            -int(item["group_hit_count"]),
            float(item["mean_error_at_20m"] or math.inf),
            -int(item["group_position_count"]),
        )
    )
    return candidates[:MAX_CASES]


def short_case_name(value: str) -> str:
    if "南岭" in value or "nanling" in value.casefold():
        return "南岭一线 F55"
    if "四季线" in value or "坂田南学校" in value:
        return "四季线 F65"
    if "湖中线" in value or "龙山工业区" in value:
        return "湖中线 F23"
    return value.split(":", 1)[-1][-24:]


def plot_case(
    case: dict[str, Any],
    record: dict[str, Any],
    feature_array: np.ndarray,
    prediction_row: dict[str, Any],
    output_dir: Path,
    evaluator,
) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams["font.family"] = ["Times New Roman", "SimHei"]
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    index = int(case["best_sweep"]["index"])
    features = np.asarray(feature_array[index], dtype=np.float32)
    ratio = float(record.get("source_scale_ratio") or 1.0)
    nominal_axis = np.linspace(0.0, FIELD_AXES_MAX_M, features.shape[-1], dtype=np.float64)
    source_axis = nominal_axis * ratio
    positions = [float(value) for value in case["positions_source_m"]]
    events = [
        event
        for event in (prediction_row.get("prediction") or {}).get("events", [])
    ]
    event_distances = [float(event["distance_m"]) for event in events]
    matched = evaluator.one_to_one_match(positions, event_distances, 20.0)
    matched_events = {round(float(item["event_m"]), 5) for item in matched["matches"]}

    fig, axes = plt.subplots(3, 1, figsize=(13, 10.5), sharex=False,
                             gridspec_kw={"height_ratios": [1.2, 1.0, 1.2]})
    axes[0].plot(source_axis, features[EVENT_CHANNEL], color="#1f4e79", linewidth=0.8,
                 label="归一化脉冲特征（Iter8 ch0）")
    axes[1].plot(source_axis, features[STEP_CHANNEL], color="#9c5700", linewidth=0.8,
                 label="归一化阶跃特征（Iter8 ch1）")
    local_low = max(float(source_axis[0]), min(positions) - 50.0)
    local_high = min(float(source_axis[-1]), max(positions) + 50.0)
    local_mask = (source_axis >= local_low) & (source_axis <= local_high)
    axes[2].plot(source_axis[local_mask], features[EVENT_CHANNEL, local_mask],
                 color="#1f4e79", linewidth=1.2, label="归一化脉冲特征")
    for position in positions:
        for ax in axes:
            ax.axvspan(position - 20.0, position + 20.0, color="#e7a5c8", alpha=0.22)
            ax.axvline(position, color="#a00066", linestyle="--", linewidth=1.1)
    for event in events:
        event_m = float(event["distance_m"])
        is_matched = round(event_m, 5) in matched_events
        color = "#17824a" if is_matched else "#777777"
        axes[0].axvline(event_m, color=color, linestyle="-", linewidth=0.9,
                        alpha=0.9 if is_matched else 0.55)
        axes[1].axvline(event_m, color=color, linestyle="-", linewidth=0.8,
                        alpha=0.75 if is_matched else 0.35)
        if local_low <= event_m <= local_high:
            axes[2].axvline(event_m, color=color, linestyle="-", linewidth=1.0,
                            alpha=0.9 if is_matched else 0.55)

    axes[0].set_ylabel("特征值")
    axes[1].set_ylabel("特征值")
    axes[2].set_ylabel("局部特征值")
    axes[0].set_xlabel("FDR source/report distance (m)")
    axes[1].set_xlabel("FDR source/report distance (m)")
    axes[2].set_xlabel("FDR source/report distance (m)")
    phase = str(record.get("phase") or "unknown")
    if phase in {"A", "B", "C"}:
        phase += "相"
    axes[0].set_title(
        f"{short_case_name(case['case_group'])} | {phase} | "
        f"hit {matched['matched_count']}/{matched['position_count']} @20m, "
        f"threshold {case.get('threshold', 0.0):.2f}",
        fontsize=12,
    )
    axes[2].set_title("局部脉冲视图（已知位置附近；仅放大显示，不改候选）", fontsize=10)
    legend_handles = [
        Line2D([0], [0], color="#a00066", linestyle="--", label="已知位置"),
        Patch(facecolor="#e7a5c8", alpha=0.35, label="已知位置 ±20 m 匹配容差"),
        Line2D([0], [0], color="#17824a", linestyle="-", label="匹配到已知位置的候选"),
        Line2D([0], [0], color="#777777", linestyle="-", label="其他候选（未核验）"),
    ]
    axes[0].legend(handles=legend_handles, loc="upper right", fontsize=9, ncol=2)
    for ax in axes:
        ax.tick_params(direction="in", top=True, right=True)
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.grid(True, alpha=0.2, linewidth=0.5)
    axes[0].set_xlim(float(source_axis[0]), float(source_axis[-1]))
    axes[1].set_xlim(float(source_axis[0]), float(source_axis[-1]))
    axes[2].set_xlim(local_low, local_high)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = "".join(char if char.isalnum() else "_" for char in case["group"]).strip("_")
    path = output_dir / f"case_{slug[:70]}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def selected_group_row(case: dict[str, Any], image_path: Path, label: str, threshold: float):
    best = case["best_sweep"]
    return {
        "row_type": "selected_case",
        "model": label,
        "threshold": threshold,
        "split": "test",
        "population": "nonwet_test_group",
        "groups": 1,
        "bags": 1,
        "curves": 1,
        "known_positions": case["position_count"],
        "hits_at_20m": case["matched_count"],
        "group_macro_hit_at_20m": case["group_hit_rate_at_20m"],
        "mean_matched_error_at_20m": case["mean_error_at_20m"],
        "candidate_events_total": case["candidate_count"],
        "candidate_events_per_curve": case["candidate_count"],
        "candidate_events_unverified_or_matched": case["candidate_count"],
        "notes": f"{case['group']} | {case['source_relative_path']} | image={image_path}",
    }


def append_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in CSV_FIELDS})


def fmt(value: Any, places: int = 3) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return "—"
        return f"{float(value):.{places}f}"
    return str(value)


def report_markdown(
    rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    case_rows: list[dict[str, Any]],
    calibration: dict[str, Any],
    synthetic: dict[str, Any],
    artifact: dict[str, Any],
    benchmark: dict[str, Any],
    selected_threshold: float,
    label: str,
    data_path: Path,
    prediction_path: Path,
    checkpoint_path: Path,
    all_prediction_summary: dict[str, Any],
) -> str:
    primary = next(
        (row for row in rows if row.get("model") == label and row.get("population") == "nonwet_test"),
        {},
    )
    wet = next(
        (row for row in rows if row.get("model") == label and row.get("population") == "wet_1500m_test"),
        {},
    )
    lines = [
        f"# {label} 现场结果导出",
        "",
        f"本报告由 `{Path(__file__).name}` 从现有预测、标签袋、缓存和检查点元信息导出；未重新运行模型推理。当前工作点：**{label}，event threshold={selected_threshold:.3f}**。",
        "",
        "完整现场 Accuracy、precision、recall、F1 不可计算：实测位置标签是正例袋，不完整真值；未标注位置没有被当作负例，未匹配候选保持未核验。有效独立现场组数量在各测试人群中单独给出，不能把 sweep 数当作独立电缆数。",
        "",
        "## 阈值对照",
        "",
        "阈值由现场验证组确定，test 只在冻结阈值后统计。U-Net .94 使用对应 Iter5 U-Net .6 验证候选预算；Transformer .98 使用对应 Iter5 Transformer .6 预算；跨架构候选预算不代表严格相同。密度约束不等于 precision。",
        "",
        "## 每模型综合性能表",
        "",
        "| 模型/工作点 | 阈值 | 非湿test hit@10 | hit@20 | hit@30 | macro@20 | 候选数/曲线 | mean error@20 (m) | 参数量 | tensor权重(MiB) | checkpoint(MiB) | NN median/p95 (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison_rows:
        lines.append(
            "| {model} | {threshold} | {h10} | {h20} | {h30} | {macro} | {burden} | {error} | {params} | {weights} | {checkpoint} | {latency} |".format(
                model=row["model"],
                threshold=fmt(row.get("threshold")),
                h10=(f"{row.get('hits_at_10m')}/{row.get('known_positions')} ({fmt(row.get('micro_hit_at_10m'))})"
                     if row.get("known_positions") else "—"),
                h20=(f"{row.get('hits_at_20m')}/{row.get('known_positions')} ({fmt(row.get('micro_hit_at_20m'))})"
                     if row.get("known_positions") else "—"),
                h30=(f"{row.get('hits_at_30m')}/{row.get('known_positions')} ({fmt(row.get('micro_hit_at_30m'))})"
                     if row.get("known_positions") else "—"),
                macro=fmt(row.get("group_macro_hit_at_20m")),
                burden=f"{fmt(row.get('candidate_events_total'),0)} / {fmt(row.get('candidate_events_per_curve'))}",
                error=fmt(row.get("mean_matched_error_at_20m")),
                params=fmt(row.get("parameters"),0),
                weights=fmt((row.get("model_state_tensor_bytes") or 0)/1024/1024,2),
                checkpoint=fmt((row.get("checkpoint_bytes") or 0)/1024/1024,2),
                latency=f"{fmt(row.get('latency_median_ms'))}/{fmt(row.get('latency_p95_ms'))}",
            )
        )
    lines += [
        "",
        "hit@tolerance 以一对一匹配已知位置计算。括号内为逐位置 micro 命中率；group macro 对独立组命中率等权平均。候选数包含命中候选，其他候选未核验。",
        "",
        "## 全部现场预测候选负担",
        "",
        f"预测文件覆盖 {all_prediction_summary['predicted_rows']}/{all_prediction_summary['prediction_rows']} 行；按threshold={selected_threshold:.3f}离线筛选后为 {all_prediction_summary['candidate_events']} 个候选，平均 {fmt(all_prediction_summary['mean_candidates_per_predicted_curve'])}/已预测曲线。该统计包括未标注现场曲线，候选均未核验，不能解释为误报数。",
        "",
        "## 当前阈值的现场分组结果",
        "",
        "| 测试人群 | 组数 | 袋数 | sweep数 | 已知位置数 | hit@10 | hit@20 | hit@30 | macro@20 | 候选数 | 候选/曲线 | mean error@20 (m) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in (primary, wet):
        if not row:
            continue
        lines.append(
            f"| {row.get('population')} | {fmt(row.get('groups'),0)} | {fmt(row.get('bags'),0)} | "
            f"{fmt(row.get('curves'),0)} | {fmt(row.get('known_positions'),0)} | "
            f"{fmt(row.get('hits_at_10m'),0)}/{fmt(row.get('known_positions'),0)} | "
            f"{fmt(row.get('hits_at_20m'),0)}/{fmt(row.get('known_positions'),0)} | "
            f"{fmt(row.get('hits_at_30m'),0)}/{fmt(row.get('known_positions'),0)} | "
            f"{fmt(row.get('group_macro_hit_at_20m'))} | {fmt(row.get('candidate_events_total'),0)} | "
            f"{fmt(row.get('candidate_events_per_curve'))} | {fmt(row.get('mean_matched_error_at_20m'))} |"
        )
    lines += [
        "",
        "## 非湿 test 组明细",
        "",
        "| Group | 已知位置 | hit@10 | hit@20 | hit@30 | 候选/最佳sweep | mean error@20 (m) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    group_rows = [row for row in rows if row.get("row_type") == "field_group" and row.get("model") == label]
    for row in group_rows:
        lines.append(
            f"| {row['notes']} | {fmt(row.get('known_positions'),0)} | "
            f"{fmt(row.get('hits_at_10m'),0)}/{fmt(row.get('known_positions'),0)} | "
            f"{fmt(row.get('hits_at_20m'),0)}/{fmt(row.get('known_positions'),0)} | "
            f"{fmt(row.get('hits_at_30m'),0)}/{fmt(row.get('known_positions'),0)} | "
            f"{fmt(row.get('candidate_events_per_curve'))} | {fmt(row.get('mean_matched_error_at_20m'))} |"
        )
    lines += ["", "## 成功案例图", ""]
    for row in case_rows:
        lines += [
            f"### {row['group']} — {row['matched_count']}/{row['position_count']} known positions hit@20m",
            "",
            f"File/sweep: `{row['source_relative_path']}` / {row['sweep_rowid']} ({row['phase']}).",
            "",
            f"![{row['group']} {label} known-positive example]({row['image_path']})",
            "",
        ]
    lines += [
        "Figures show a full-range pulse/step overview with all thresholded candidates and a local pulse magnification around the known positions. They illustrate candidate-to-known-position matches only; unmatched candidates are not labeled false positives.",
        "Each selected figure is one sweep (N=1) and is an illustrative hit example; it cannot provide sample-level Accuracy/F1 or establish complete field performance.",
        "",
        "## 校准与合成测试",
        "",
        "### 短接参考伪事件",
        "",
        "| Split | 短接参考数 | event false events |",
        "|---|---:|---:|",
    ]
    if calibration is None:
        lines.append("| — | — | 当前检查点未提供校准评估 |")
    else:
        lines.append(
            f"校准数据语义：{calibration.get('semantics', '见校准报告')}；"
            f"统计阈值={calibration.get('event_threshold')}，{calibration.get('threshold_note', '')}"
        )
        for split in ("train", "val", "test"):
            item = calibration["splits"][split]
            lines.append(f"| {split} | {item['references']} | {item['false_event_count']} |")
    lines += [
        "",
        "这些参考是用户确认的夹具短接、未接被测电缆数据，不是健康电缆。",
        "",
        "| Synthetic test view | curves | event P@5m | event R@5m | event F1@5m | interval F1 IoU≥0.5 | terminal F1@10m |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for view in ("mixed", "clean"):
        item = synthetic.get(view)
        if item is None:
            lines.append(f"| {view} | — | — | — | — | — | — |")
        else:
            lines.append(
                f"| {view} | {fmt(item.get('curve_count'),0)} | "
                f"{fmt(item['event_at_5m']['precision'])} | {fmt(item['event_at_5m']['recall'])} | "
                f"{fmt(item['event_at_5m']['f1'])} | {fmt(item['interval_f1'])} | "
                f"{fmt(item['terminal_f1'])} |"
            )
    lines += [
        "",
        "Synthetic metrics use generated geometric truth and do not establish field accuracy.",
        "",
        "## 模型文件与延迟",
        "",
        f"- Parameters: {fmt(artifact.get('parameters'),0)}; FP32 parameter-size estimate: {fmt((artifact.get('parameters') or 0)*4/1024/1024,2)} MiB.",
        f"- State-dict tensor storage: {fmt((artifact.get('state_tensor_bytes') or 0)/1024/1024,2)} MiB; full checkpoint file: {fmt((artifact.get('checkpoint_bytes') or 0)/1024/1024,2)} MiB. The checkpoint contains training state as well as weights.",
        f"- Inference package size: {fmt(artifact.get('inference_package_bytes'),0)} bytes.",
        f"- Inference latency: {fmt(benchmark.get('median_ms'))} ms median / {fmt(benchmark.get('p95_ms'))} ms p95; scope: {benchmark.get('scope')}.",
        "",
        "## 输入与来源",
        "",
        f"- Cache: `{data_path}`",
        f"- Prediction source: `{prediction_path}`; event threshold applied offline in this export.",
        f"- Checkpoint metadata source: `{checkpoint_path}`. The checkpoint was loaded with `map_location=cpu` only to count saved model tensors; no model forward pass ran.",
        "- Existing source/report distance mapping uses the record's `source_scale_ratio`; waveform feature x-axes are converted from the Iter8 nominal response grid for the case plots.",
        "",
    ]
    return "\n".join(lines)


def model_results_markdown(
    label: str,
    threshold: float,
    checkpoint_path: Path,
    artifact: dict[str, Any],
    benchmark: dict[str, Any],
    nonwet: dict[str, Any],
    wet: dict[str, Any],
    cases: list[dict[str, Any]],
    calibration: dict[str, Any] | None,
    synthetic: dict[str, Any],
    all_prediction_summary: dict[str, Any],
) -> str:
    run_info_path = checkpoint_path.parent / "run_info.json"
    config_path = checkpoint_path.parent / "config.json"
    selection_path = checkpoint_path.parent / "selection.json"
    run_info = read_json(run_info_path) if run_info_path.is_file() else artifact.get("run_info", {})
    config = read_json(config_path) if config_path.is_file() else artifact.get("config", {})
    selection = read_json(selection_path) if selection_path.is_file() else None
    training = config.get("training") or run_info.get("training") or {}
    adaptation = (
        config.get("adaptation8")
        or config.get("adaptation")
        or run_info.get("adaptation")
        or {}
    )
    source_checkpoint = (
        adaptation.get("source_checkpoint")
        or run_info.get("source_checkpoint")
        or "未记录"
    )
    selection_text = (
        f"`{selection_path}`: `{json.dumps(selection, ensure_ascii=False, separators=(',', ':'))}`"
        if selection is not None
        else f"best checkpoint epoch={artifact.get('epoch')}；阈值由命令行参数给出。"
    )
    if (
        selection is None
        and checkpoint_path.resolve() == ITER8_UNET_CHECKPOINT.resolve()
        and abs(threshold - 0.94) < 1e-9
    ):
        source = read_json(ITER8_SUMMARY)["density_matched_analysis"]
        selection_text = (
            f"`{ITER8_SUMMARY}` density_matched_analysis；field validation 选定 "
            f"U-Net event threshold={source['selected_thresholds_from_validation']['unet']}."
        )
    macro20 = nonwet["macro"]["20"]
    errors20 = nonwet["errors"]["20"]
    lines = [
        f"# {label} 模型结果",
        "",
        "本页只描述传入的 checkpoint 与既有现场预测。exporter 对缓存预测按 confidence 离线筛选，没有重新做模型 forward。完整现场真值未知，未标注位置没有作为负例，未匹配候选不是已确认误报。",
        "",
        "## checkpoint 与选择",
        "",
        f"- Checkpoint: `{checkpoint_path}`。",
        f"- Model label: {label}; field event threshold: {threshold:.3f}。",
        f"- Selection record: {selection_text}",
        f"- Training source checkpoint: `{source_checkpoint}`。",
        f"- Training config: `{config_path if config_path.is_file() else run_info_path if run_info_path.is_file() else '未找到'}`。",
        f"- Training values: `{json.dumps(training, ensure_ascii=False, separators=(',', ':'))}`。",
        f"- Adaptation values: `{json.dumps(adaptation, ensure_ascii=False, separators=(',', ':'))}`。",
        "",
        "## 现场结果",
        "",
        "| 人群 | 组数 | bag数 | sweep数 | 已知位置 | hit@10 | hit@20 | hit@30 | group macro@20 | mean matched error@20 (m) | 候选数 | 候选/曲线 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, population in (("nonwet_test", nonwet), ("wet_1500m_test", wet)):
        macro = population["macro"]
        error = population["errors"]["20"]
        burden = population["burden"]
        n = macro["20"]["unique_positions"]
        lines.append(
            f"| {name} | {macro['20']['independent_groups']} | {len(population['bags'])} | "
            f"{burden['curves']} | {n} | {macro['10']['matched_positions']}/{n} | "
            f"{macro['20']['matched_positions']}/{n} | {macro['30']['matched_positions']}/{n} | "
            f"{fmt(macro['20']['macro_hit_rate'])} | {fmt(error['mean_error_m'])} | "
            f"{burden['candidate_events']} | {fmt(burden['mean_candidates_per_curve'])} |"
        )
    lines += [
        "",
        f"该预测文件共有 {all_prediction_summary['prediction_rows']} 行、"
        f"{all_prediction_summary['predicted_rows']} 条 status=predicted；按阈值后候选总数 "
        f"{all_prediction_summary['candidate_events']}，均值 "
        f"{fmt(all_prediction_summary['mean_candidates_per_predicted_curve'])}/条。未标注曲线候选保持未核验。",
        "",
        "## 案例",
        "",
    ]
    for case in cases:
        lines += [
            f"### {case['group']} — {case['matched_count']}/{case['position_count']} hit@20m",
            "",
            f"`{case['source_relative_path']}` sweep {case['sweep_rowid']} ({case['phase']}); "
            f"mean matched error={fmt(case['mean_error_at_20m'])} m; candidate count={case['candidate_count']}.",
            "",
            f"![{case['group']} known-position example]({case['image_path']})",
            "",
        ]
    lines += ["## 校准参考与合成评价", ""]
    if calibration is None:
        lines.append("当前 checkpoint 目录没有 calibration_iter8.json；校准伪事件数未测。")
    else:
        lines += [
            f"Calibration semantics: {calibration.get('semantics', '见校准报告')}。",
            "",
            "| Split | references | event false events |",
            "|---|---:|---:|",
        ]
        for split in ("train", "val", "test"):
            item = calibration["splits"][split]
            lines.append(f"| {split} | {item['references']} | {item['false_event_count']} |")
    lines.append("")
    if synthetic:
        for view, item in synthetic.items():
            lines.append(
                f"- Synthetic test `{view}`: curves={item['curve_count']}; "
                f"event F1@5m={fmt(item['event_at_5m']['f1'])}; "
                f"interval F1={fmt(item['interval_f1'])}; terminal F1@10m={fmt(item['terminal_f1'])}."
            )
    else:
        lines.append("当前 checkpoint 目录没有 mixed/clean synthetic evaluation.json；合成测试未测。")
    lines += [
        "",
        "合成几何标签不代表现场完整真值。短接参考是用户确认的无测试电缆夹具数据，不是健康电缆。",
        "",
        "## 模型资源与延迟",
        "",
        f"- Parameters: {fmt(artifact.get('parameters'),0)}; FP32 size estimate: {fmt((artifact.get('parameters') or 0)*4/1024/1024,2)} MiB.",
        f"- Model tensor bytes: {fmt((artifact.get('state_tensor_bytes') or 0)/1024/1024,2)} MiB; inference package: {fmt(artifact.get('inference_package_bytes'),0)} bytes; full checkpoint: {fmt((artifact.get('checkpoint_bytes') or 0)/1024/1024,2)} MiB.",
        f"- NN latency: median={fmt(benchmark.get('median_ms'))} ms, p95={fmt(benchmark.get('p95_ms'))} ms; {benchmark.get('scope')}",
        "",
        "Latency is a neural-network-only measurement when provided; it excludes FDR reading, FFT/Algorithm 1, and host-to-device transfer according to the benchmark scope.",
        "",
        "For the original Iter8 architecture and historical training context, see `MODEL_CARD_Iter8.md`. That card is a historical Iter8 description and does not replace this per-checkpoint result when a later model is selected.",
        "The combined comparison table and full per-group metrics are in `performance.md` and `performance.csv`.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ITER8_DATA)
    parser.add_argument("--predictions", type=Path, default=ITER8_UNET_PREDICTIONS)
    parser.add_argument("--checkpoint", type=Path, default=ITER8_UNET_CHECKPOINT)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--threshold", type=float, default=0.94)
    parser.add_argument("--output", type=Path, default=HERE / "output" / "existing_iter8")
    parser.add_argument("--label", default="Iter8-U-Net")
    parser.add_argument("--benchmark", type=Path, default=HERE / "output" / "nn_benchmark.json")
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(output / "matplotlib_cache")
    os.environ["MPLBACKEND"] = "Agg"

    evaluator = load_field_evaluator()
    records = read_jsonl(args.data / "field_records.jsonl")
    bag_data = read_json(args.data / "field_bags.json")
    bags = bag_data["bags"] if isinstance(bag_data, dict) else bag_data
    records_by_index = {int(row["index"]): row for row in records}
    prediction_rows = read_jsonl(args.predictions)
    current_by_index, current_total = prediction_index(
        records, prediction_rows, args.threshold, evaluator
    )
    test_bags = evaluator.bag_rows("test", bags, records_by_index)
    nonwet_bags = [bag for bag in test_bags if not bag.get("known_lab")]
    wet_bags = [bag for bag in test_bags if bag.get("known_lab")]
    current_nonwet = positive_population_report(
        "test", nonwet_bags, records_by_index, current_by_index, evaluator
    )
    current_wet = positive_population_report(
        "test", wet_bags, records_by_index, current_by_index, evaluator
    )

    checkpoint_info = checkpoint_artifact(args.checkpoint) if args.checkpoint.is_file() else {}
    artifact = dict(checkpoint_info)
    benchmark = benchmark_info(args.benchmark)
    if benchmark.get("parameters") is not None:
        artifact["parameters"] = int(benchmark["parameters"])
    if benchmark.get("weight_tensor_bytes") is not None:
        artifact["state_tensor_bytes"] = int(benchmark["weight_tensor_bytes"])
    for key in ("inference_package_bytes", "checkpoint_bytes"):
        if benchmark.get(key) is not None:
            artifact[key] = int(benchmark[key])
    artifact.setdefault("checkpoint_bytes", args.checkpoint.stat().st_size if args.checkpoint.is_file() else None)

    calibration_path = args.calibration or (args.checkpoint.parent / "calibration_iter8.json")
    calibration = (
        calibration_at_threshold(read_json(calibration_path), args.threshold)
        if calibration_path.is_file()
        else None
    )
    synthetic = synthetic_metrics_for(args.checkpoint.parent)

    output_rows: list[dict[str, Any]] = []
    for population, name in ((current_nonwet, "nonwet_test"), (current_wet, "wet_1500m_test")):
        output_rows.append(population_row(args.label, args.threshold, population, name))
    for group_report in current_nonwet["groups"]:
        output_rows.append(group_row(group_report, current_nonwet, args.label, args.threshold, evaluator))

    all_candidate_row = {
        "row_type": "all_field_predictions_unverified",
        "model": args.label,
        "threshold": args.threshold,
        "split": "all",
        "population": "all_prediction_rows",
        "curves": current_total["predicted_rows"],
        "candidate_events_total": current_total["candidate_events"],
        "candidate_events_per_curve": current_total["mean_candidates_per_predicted_curve"],
        "candidate_events_unverified_or_matched": current_total["candidate_events"],
        "notes": "Unmatched and unlabeled-curve candidates remain unverified.",
    }
    output_rows.append(all_candidate_row)

    comparison_specs = [
        ("Iter5-U-Net@0.6", ITER5_UNET_PREDICTIONS, 0.60, ITER5_DIR / "output" / "iter5" / "unet" / "best.pt"),
        ("Iter8-U-Net@0.94", ITER8_UNET_PREDICTIONS, 0.94, ITER8_UNET_CHECKPOINT),
        ("Iter8-Transformer@0.98", ITER8_TRANSFORMER_PREDICTIONS, 0.98, ITER8_TRANSFORMER_CHECKPOINT),
        ("Iter8-U-Net-fixed@0.6", ITER8_UNET_PREDICTIONS, 0.60, ITER8_UNET_CHECKPOINT),
    ]
    current_key = (str(args.predictions.resolve()), round(float(args.threshold), 9))
    field_predictions: list[tuple[str, Path, float, Path, bool]] = []
    current_in_comparison = False
    for label, path, threshold, checkpoint in comparison_specs:
        if not path.is_file():
            continue
        is_current = (
            str(path.resolve()), round(float(threshold), 9)
        ) == current_key
        if is_current:
            label = args.label
            checkpoint = args.checkpoint
            current_in_comparison = True
        field_predictions.append((label, path, threshold, checkpoint, is_current))
    if not current_in_comparison:
        field_predictions.append((args.label, args.predictions, args.threshold, args.checkpoint, True))

    comparison_rows: list[dict[str, Any]] = []
    artifact_cache: dict[str, dict[str, Any]] = {str(args.checkpoint.resolve()): artifact}
    for model_label, path, threshold, checkpoint, is_current in field_predictions:
        rows = prediction_rows if is_current else read_jsonl(path)
        by_index, _ = (current_by_index, current_total) if is_current else prediction_index(
            records, rows, threshold, evaluator
        )
        nonwet = positive_population_report(
            "test", nonwet_bags, records_by_index, by_index, evaluator
        )
        base = population_row(model_label, threshold, nonwet, "nonwet_test")
        checkpoint_key = str(checkpoint.resolve())
        if checkpoint_key not in artifact_cache:
            artifact_cache[checkpoint_key] = checkpoint_artifact(checkpoint) if checkpoint.is_file() else {}
        item_artifact = artifact_cache[checkpoint_key]
        item_benchmark = benchmark if is_current else benchmark_info(None)
        base.update(
            {
                "parameters": item_artifact.get("parameters"),
                "model_state_tensor_bytes": item_benchmark.get("weight_tensor_bytes")
                or item_artifact.get("state_tensor_bytes"),
                "checkpoint_bytes": item_benchmark.get("checkpoint_bytes")
                or item_artifact.get("checkpoint_bytes"),
                "inference_package_bytes": item_benchmark.get("inference_package_bytes"),
                "latency_median_ms": item_benchmark.get("median_ms"),
                "latency_p95_ms": item_benchmark.get("p95_ms"),
                "latency_scope": item_benchmark.get("scope"),
            }
        )
        comparison_rows.append(base)
        output_rows.append(base)
        wet_by_index = current_by_index if is_current else by_index
        wet_population = positive_population_report(
            "test", wet_bags, records_by_index, wet_by_index, evaluator
        )
        output_rows.append(population_row(model_label, threshold, wet_population, "wet_1500m_test"))

    current_selected_cases = choose_cases(current_nonwet, evaluator)
    feature_array = np.load(args.data / "field_features.npy", mmap_mode="r")
    case_rows: list[dict[str, Any]] = []
    plot_paths: list[Path] = []
    for case in current_selected_cases:
        case["threshold"] = args.threshold
        index = int(case["best_sweep"]["index"])
        record = records_by_index[index]
        case["source_relative_path"] = record["source_relative_path"]
        filtered_prediction = current_by_index.get(index)
        if filtered_prediction is None:
            continue
        image_path = plot_case(
            case,
            record,
            feature_array,
            filtered_prediction,
            output / "assets",
            evaluator,
        )
        plot_paths.append(image_path)
        case_rows.append(
            {
                **case,
                "sweep_rowid": record.get("sweep_rowid"),
                "phase": record.get("phase", "unknown"),
                "image_path": image_path.relative_to(output).as_posix(),
            }
        )
        output_rows.append(selected_group_row(case, image_path.relative_to(output), args.label, args.threshold))

    if calibration is not None:
        for split in ("train", "val", "test"):
            item = calibration["splits"][split]
            output_rows.append(
                {
                    "row_type": "calibration_reference",
                    "model": args.label,
                    "threshold": calibration.get("event_threshold"),
                    "split": split,
                    "population": "confirmed_short_no_cable_reference",
                    "false_events": item["false_event_count"],
                    "calibration_references": item["references"],
                    "notes": calibration.get("threshold_note") or calibration.get("semantics"),
                }
            )

    for view, item in synthetic.items():
        output_rows.append(
            {
                "row_type": "synthetic_test",
                "model": args.label,
                "split": "test",
                "population": "synthetic",
                "curves": item["curve_count"],
                "synthetic_view": view,
                "synthetic_event_precision_at_5m": item["event_at_5m"]["precision"],
                "synthetic_event_recall_at_5m": item["event_at_5m"]["recall"],
                "synthetic_event_f1_at_5m": item["event_at_5m"]["f1"],
                "synthetic_interval_f1_iou_0_5": item["interval_f1"],
                "synthetic_terminal_f1_at_10m": item["terminal_f1"],
                "notes": "合成几何真值；不代表现场准确率。",
            }
        )

    current_row = model_file_row(
        args.label,
        args.threshold,
        args.checkpoint,
        artifact.get("parameters"),
        artifact.get("state_tensor_bytes"),
        benchmark,
    )
    current_row["inference_package_bytes"] = artifact.get("inference_package_bytes")
    current_row["checkpoint_bytes"] = artifact.get("checkpoint_bytes")
    output_rows.append(current_row)
    csv_path = output / "performance.csv"
    append_csv_rows(csv_path, output_rows)

    report_text = report_markdown(
        output_rows,
        comparison_rows,
        case_rows,
        calibration,
        synthetic,
        artifact,
        benchmark,
        args.threshold,
        args.label,
        args.data,
        args.predictions,
        args.checkpoint,
        current_total,
    )
    performance_markdown_path = output / "performance.md"
    performance_markdown_path.write_text(report_text, encoding="utf-8")

    results_text = model_results_markdown(
        args.label,
        args.threshold,
        args.checkpoint,
        artifact,
        benchmark,
        current_nonwet,
        current_wet,
        case_rows,
        calibration,
        synthetic,
        current_total,
    )
    model_results_path = output / "MODEL_RESULTS.md"
    model_results_path.write_text(results_text, encoding="utf-8")

    case_index = output / "selected_cases.csv"
    case_fields = [
        "group", "case_group", "source_relative_path", "sweep_rowid", "phase",
        "position_count", "matched_count", "hit_rate_at_20m", "group_hit_rate_at_20m",
        "mean_error_at_20m",
        "candidate_count", "image_path",
    ]
    with case_index.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=case_fields)
        writer.writeheader()
        for row in case_rows:
            writer.writerow({key: row.get(key) for key in case_fields})

    print(json.dumps({
        "label": args.label,
        "threshold": args.threshold,
        "data": str(args.data),
        "predictions": str(args.predictions),
        "checkpoint": str(args.checkpoint),
        "output": str(output),
        "nonwet_test": {
            "groups": current_nonwet["macro"]["20"]["independent_groups"],
            "positions": current_nonwet["macro"]["20"]["unique_positions"],
            "hits_at_20m": current_nonwet["macro"]["20"]["matched_positions"],
            "group_macro_at_20m": current_nonwet["macro"]["20"]["macro_hit_rate"],
        },
        "selected_cases": [row["group"] for row in case_rows],
        "csv": str(csv_path),
        "markdown": str(performance_markdown_path),
        "model_results": str(model_results_path),
        "calibration": str(calibration_path) if calibration is not None else None,
        "synthetic_views": sorted(synthetic),
        "figures": [str(path) for path in plot_paths],
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
