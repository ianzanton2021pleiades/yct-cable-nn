"""Build the comprehensive field/synthetic metrics table and significance checks."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from decode import decode_predictions  # noqa: E402
from eval_utils import collect_probabilities, load_model, match_events  # noqa: E402
from hybrid_data import FieldDataset  # noqa: E402


WET_CASE = "2026-6某地进行的1500m不同位置电缆接头浸水试验"


def metric(result: dict, key: str, default=None):
    try:
        value = result
        for part in key.split("."):
            value = value[part]
        return float(value) if value is not None else default
    except (KeyError, TypeError, ValueError):
        return default


def group_field_cases(
    probabilities: list[np.ndarray],
    records: list[dict[str, Any]],
    cfg: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, dict[str, float]]:
    cases: dict[str, dict[str, float]] = defaultdict(
        lambda: {"truth": 0.0, "hits": 0.0, "candidates": 0.0, "curves": 0.0}
    )
    for probability, record in zip(probabilities, records):
        truth = [float(value) for value in (record.get("known_positions_nominal_m") or [])]
        if not truth:
            continue
        prediction = decode_predictions(probability, cfg, thresholds)
        hits = match_events(prediction["events"], truth, 20.0)
        case = str(record.get("case_group") or "")
        cases[case]["truth"] += len(truth)
        cases[case]["hits"] += hits
        cases[case]["candidates"] += len(prediction["events"])
        cases[case]["curves"] += 1
    return dict(cases)


def paired_bootstrap(
    left: list[float], right: list[float], samples: int = 10000, seed: int = 0
) -> tuple[float, float]:
    if not left or not right or len(left) != len(right):
        return 0.0, 1.0
    difference = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    observed = float(difference.mean())
    if np.allclose(difference, 0.0):
        return observed, 1.0
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, difference.size, size=(samples, difference.size))
    means = difference[indices].mean(axis=1)
    p_value = 2.0 * min(float((means <= 0.0).mean()), float((means >= 0.0).mean()))
    return observed, min(1.0, p_value)


def latency_and_size(checkpoints: list[Path], device: torch.device) -> tuple[float, float, float]:
    models = []
    parameters = 0
    checkpoint_bytes = 0
    for path in checkpoints:
        model, _, _ = load_model(path, device)
        models.append(model)
        parameters += sum(parameter.numel() for parameter in model.parameters())
        checkpoint_bytes += Path(path).stat().st_size
    inputs = torch.randn(1, 8, 8192, device=device)
    with torch.inference_mode():
        for _ in range(5):
            for model in models:
                model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(30):
            for model in models:
                model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        milliseconds = (time.perf_counter() - started) / 30.0 * 1000.0
    return (
        float(milliseconds),
        parameters * 4.0 / (1024.0**2),
        checkpoint_bytes / (1024.0**2),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=HERE / "output" / "hybrid")
    parser.add_argument("--evaluation-name", default="evaluation_e05")
    parser.add_argument("--data", type=Path, default=Path(r"E:\CableNNDataSet\DeepSeek_CMDC\FieldNN_Hybrid_20261022"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(1)
    device = torch.device(args.device)
    checkpoint_paths = {
        "unet": [args.run / "unet" / "best.pt"],
        "transformer": [args.run / "transformer" / "best.pt"],
        "ensemble": [args.run / "unet" / "best.pt", args.run / "transformer" / "best.pt"],
    }
    results: dict[str, Any] = {}
    field_cases: dict[str, dict[str, dict[str, float]]] = {}
    cfg = None
    for name, checkpoints in checkpoint_paths.items():
        evaluation_path = args.run / name / args.evaluation_name / "evaluation.json"
        if not evaluation_path.is_file():
            continue
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        results[name] = evaluation
        thresholds = {key: float(value) for key, value in evaluation["thresholds"].items()}
        models = []
        for path in checkpoints:
            model, _, model_cfg = load_model(path, device)
            models.append(model)
            cfg = model_cfg
        dataset = FieldDataset(args.data, "test")
        probabilities, indices = collect_probabilities(models, dataset, device, 64)
        records = [dataset.records[index] for index in indices]
        field_cases[name] = group_field_cases(probabilities, records, cfg, thresholds)

    if not results:
        raise SystemExit(f"No evaluation results under {args.run}/{args.evaluation_name}")

    latencies = {name: latency_and_size(paths, device) for name, paths in checkpoint_paths.items() if name in results}
    non_wet_cases = sorted({
        case for cases in field_cases.values() for case in cases if case != WET_CASE
    })
    table = []
    for name in ("unet", "transformer", "ensemble"):
        if name not in results:
            continue
        evaluation = results[name]
        event = evaluation["synthetic_test"]["clean"]["report"]["event_metrics_by_tolerance_m"]["5"]
        interval = evaluation["synthetic_test"]["clean"]["report"]["interval_metrics"]
        terminal = evaluation["synthetic_test"]["clean"]["report"]["terminal_metrics"]
        field = evaluation["field_test"]
        wet = field.get("wet_1500", {})
        short = evaluation.get("calibration_test", {})
        short_specificity = None
        if short.get("references"):
            short_specificity = 1.0 - float(short.get("fraction_with_false_event_or_interval", 0.0))
        cases = field_cases[name]
        non_wet_truth = float(field.get("non_wet", {}).get("truth_events") or 0.0)
        non_wet_candidates = float(field.get("non_wet", {}).get("candidates") or 0.0)
        non_wet_sensitivity = float(field.get("non_wet", {}).get("hit_rate_20") or 0.0)
        annotated_precision = (
            non_wet_sensitivity * non_wet_truth / non_wet_candidates
            if non_wet_candidates > 0 else None
        )
        table.append({
            "method": name,
            "synthetic_event_precision_5m": event.get("precision"),
            "synthetic_event_recall_5m": event.get("recall"),
            "synthetic_event_f1_5m": event.get("f1"),
            "synthetic_interval_f1": interval.get("f1"),
            "synthetic_terminal_f1": terminal.get("f1"),
            "synthetic_macro_f1": float(np.mean([
                event.get("f1") or 0.0,
                interval.get("f1") or 0.0,
                terminal.get("f1") or 0.0,
            ])),
            "field_nonwet_sensitivity_20": non_wet_sensitivity,
            "field_nonwet_annotated_precision_20": annotated_precision,
            "field_nonwet_candidates_per_curve": field.get("non_wet", {}).get("candidates_per_curve"),
            "field_strict_sensitivity_20": field.get("strict", {}).get("hit_rate_20"),
            "field_strict_candidates_per_curve": field.get("strict", {}).get("candidates_per_curve"),
            "wet_sensitivity_20": wet.get("hit_rate_20"),
            "wet_candidates_per_curve": wet.get("candidates_per_curve"),
            "short_specificity": short_specificity,
            "latency_ms_per_curve": latencies[name][0],
            "parameters_mb": latencies[name][1],
            "checkpoint_mb": latencies[name][2],
            "p_vs_unet": None,
        })
        if name != "unet":
            baseline_map = {
                case: field_cases["unet"][case]["hits"] / field_cases["unet"][case]["truth"]
                for case in non_wet_cases
                if case in field_cases["unet"] and field_cases["unet"][case]["truth"] > 0
            }
            right_map = {
                case: cases[case]["hits"] / cases[case]["truth"]
                for case in non_wet_cases
                if case in cases and cases[case]["truth"] > 0
            }
            keys = sorted(set(baseline_map) & set(right_map))
            if keys:
                _, p_value = paired_bootstrap(
                    [right_map[key] for key in keys],
                    [baseline_map[key] for key in keys],
                )
                table[-1]["p_vs_unet"] = p_value

    tag = "e05" if "e05" in args.evaluation_name else "e06"
    output = args.output or (args.run / f"metrics_table_{tag}.md")
    lines = [
        "| Method | Syn P@5m | Syn R@5m | Syn F1@5m | Syn interval F1 | Syn terminal F1 | Macro F1 | Field non-wet Sens@20 | P_ann | Cand/curve | Strict Sens@20 | Wet Sens@20 | Short specificity | Latency ms | Params MB | p vs U-Net |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        def fmt(value, digits=4):
            return "—" if value is None else f"{value:.{digits}f}"
        lines.append(
            "| " + " | ".join([
                row["method"],
                fmt(row["synthetic_event_precision_5m"]),
                fmt(row["synthetic_event_recall_5m"]),
                fmt(row["synthetic_event_f1_5m"]),
                fmt(row["synthetic_interval_f1"]),
                fmt(row["synthetic_terminal_f1"]),
                fmt(row["synthetic_macro_f1"]),
                fmt(row["field_nonwet_sensitivity_20"]),
                fmt(row["field_nonwet_annotated_precision_20"]),
                fmt(row["field_nonwet_candidates_per_curve"], 3),
                fmt(row["field_strict_sensitivity_20"]),
                fmt(row["wet_sensitivity_20"]),
                fmt(row["short_specificity"]),
                fmt(row["latency_ms_per_curve"], 3),
                fmt(row["parameters_mb"], 2),
                fmt(row["p_vs_unet"]),
            ]) + " |"
        )
    lines.append("")
    lines.append("P_ann = hits@20m / all predicted events on non-wet field cases; it is a lower bound because unannotated real events count against it.")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.run / f"metrics_table_{tag}.json").write_text(
        json.dumps(table, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    import csv

    with (args.run / f"metrics_table_{tag}.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)
    print("\n".join(lines))
    print(f"\ntable: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
