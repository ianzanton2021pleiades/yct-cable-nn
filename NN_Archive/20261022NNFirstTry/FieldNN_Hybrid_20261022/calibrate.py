"""Calibrate event/interval/terminal thresholds on synthetic + hybrid validation.

The event threshold is selected with a precision-aware hybrid rule: among
thresholds that keep the injected-event candidate density within budget, it
maximizes a balanced synthetic-F1 / hybrid-hit@20 score.  Interval and terminal
follow the synthetic precision-first rule.  Test data is never used.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
LEGACY = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_20261022"
if str(LEGACY) not in sys.path:
    sys.path.append(str(LEGACY))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from decode import decode_predictions  # noqa: E402
from eval_utils import collect_probabilities, field_metrics, hybrid_metrics, load_model  # noqa: E402
from hybrid_data import FieldDataset, HybridDataset, SyntheticWrapper  # noqa: E402
from metrics import evaluate_predictions  # noqa: E402


THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)
CANDIDATE_BUDGET = 3.0


def synthetic_truths(dataset, indices: list[int]) -> list[dict[str, Any]]:
    rows = []
    for index in indices:
        metadata = dataset.metadata[index]
        rows.append({
            "event_positions_m": metadata.get("event_positions_m", []),
            "defect_intervals_m": metadata.get("defect_intervals_m", []),
            "terminal_position_m": metadata.get("terminal_position_m"),
            "sample_kind": metadata.get("sample_kind"),
        })
    return rows


def precision_key(metric: dict[str, Any], target: float = 0.90) -> tuple:
    precision = float(metric.get("precision") or 0.0)
    recall = float(metric.get("recall") or 0.0)
    fhalf = 1.25 * precision * recall / (0.25 * precision + recall) if precision + recall else 0.0
    return (int(precision >= target and recall > 0), recall if precision >= target else fhalf, precision)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--synthetic", type=Path, default=Path(r"E:\CableNNDataSet\CodeX\FieldNN_Iter4_20261022"))
    parser.add_argument("--checkpoint", required=True, type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    torch.set_num_threads(1)
    device = torch.device(args.device)
    models = []
    configs = []
    for path in args.checkpoint:
        model, checkpoint, cfg = load_model(path, device)
        models.append(model)
        configs.append((path, checkpoint, cfg))
    cfg = configs[0][2]

    syn_val = SyntheticWrapper(args.synthetic, "val", train=False, view="mixed")
    hyb_val = HybridDataset(args.data, "val", augment=False)
    field_val = FieldDataset(args.data, "val")
    syn_probs, syn_indices = collect_probabilities(models, syn_val, device, args.batch_size)
    syn_truth = synthetic_truths(syn_val, syn_indices)
    hyb_probs, hyb_indices = collect_probabilities(models, hyb_val, device, args.batch_size)
    hyb_records = [hyb_val.records[index] for index in hyb_indices]
    field_probs, field_indices = collect_probabilities(models, field_val, device, args.batch_size)
    field_records = [field_val.records[index] for index in field_indices]

    curves = {"event": [], "interval": [], "terminal": []}
    for threshold in THRESHOLDS:
        decoded = [
            decode_predictions(
                probability,
                cfg,
                {"event": threshold, "interval": threshold, "terminal": threshold},
            )
            for probability in syn_probs
        ]
        report = evaluate_predictions(decoded, syn_truth, cfg)
        event = report["event_metrics_by_tolerance_m"]["5"]
        interval = report["interval_metrics"]
        terminal = report["terminal_metrics"]
        hybrid_prediction = hybrid_metrics(
            hyb_probs,
            hyb_records,
            cfg,
            {"event": threshold, "interval": 0.99, "terminal": 0.4},
        )
        field_prediction = field_metrics(
            field_probs,
            field_records,
            cfg,
            {"event": threshold, "interval": 0.99, "terminal": 0.4},
        )
        curves["event"].append({
            "threshold": threshold,
            "synthetic_precision": event.get("precision"),
            "synthetic_recall": event.get("recall"),
            "synthetic_f1": event.get("f1"),
            "hybrid_hit_rate_20": hybrid_prediction.get("hit_rate_20"),
            "hybrid_hit_rate_10": hybrid_prediction.get("hit_rate_10"),
            "hybrid_candidates_per_curve": hybrid_prediction.get("candidates_per_curve"),
            "field_hit_rate_20": field_prediction.get("hit_rate_20"),
            "field_hit_rate_10": field_prediction.get("hit_rate_10"),
            "field_candidates_per_curve": field_prediction.get("candidates_per_curve"),
        })
        curves["interval"].append({"threshold": threshold, **interval})
        curves["terminal"].append({"threshold": threshold, **terminal})

    event_rows = [
        row for row in curves["event"]
        if (row.get("hybrid_candidates_per_curve") is not None
            and float(row["hybrid_candidates_per_curve"]) <= CANDIDATE_BUDGET
            and row.get("field_candidates_per_curve") is not None
            and float(row["field_candidates_per_curve"]) <= CANDIDATE_BUDGET)
    ] or curves["event"]
    event_row = max(
        event_rows,
        key=lambda row: (
            0.4 * float(row.get("synthetic_f1") or 0.0)
            + 0.3 * float(row.get("hybrid_hit_rate_20") or 0.0)
            + 0.3 * float(row.get("field_hit_rate_20") or 0.0),
            float(row.get("field_hit_rate_20") or 0.0),
            float(row.get("synthetic_f1") or 0.0),
        ),
    )
    interval_row = max(curves["interval"], key=lambda row: (*precision_key(row), row["threshold"]))
    terminal_row = max(curves["terminal"], key=lambda row: (*precision_key(row), row["threshold"]))
    thresholds = {
        "event": float(event_row["threshold"]),
        "interval": float(interval_row["threshold"]),
        "terminal": float(terminal_row["threshold"]),
    }

    output = args.output or Path(args.checkpoint[0]).parent
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "calibration_split": "val",
        "checkpoints": [str(path.resolve()) for path in args.checkpoint],
        "thresholds": thresholds,
        "policy": (
            "event: maximize 0.4*synthetic F1@5m + 0.3*hybrid hit@20m + "
            "0.3*field hit@20m subject to hybrid and field candidates/curve "
            f"<= {CANDIDATE_BUDGET}; interval/terminal: synthetic precision-first "
            "(recall if precision>=0.90 else F0.5)"
        ),
        "candidate_budget": CANDIDATE_BUDGET,
        "curves": curves,
    }
    (output / "thresholds.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output / "thresholds.json"),
        "thresholds": thresholds,
        "selected_event": event_row,
        "selected_interval": interval_row,
        "selected_terminal": terminal_row,
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
