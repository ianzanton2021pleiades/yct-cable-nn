"""Evaluate synthetic, injected-hybrid, original-field, wet-1500, and short references."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
LEGACY = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_20261022"
if str(LEGACY) not in sys.path:
    sys.path.append(str(LEGACY))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from decode import decode_predictions  # noqa: E402
from eval_utils import (  # noqa: E402
    calibration_metrics,
    collect_probabilities,
    field_metrics,
    hybrid_metrics,
    load_model,
)
from hybrid_data import (  # noqa: E402
    CalibrationDataset,
    FieldDataset,
    HybridDataset,
    SyntheticWrapper,
)
from metrics import evaluate_predictions  # noqa: E402


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


def evaluate_synthetic(
    models, dataset, cfg, thresholds, device: torch.device, batch_size: int
) -> dict[str, Any]:
    probabilities, indices = collect_probabilities(models, dataset, device, batch_size)
    truths = synthetic_truths(dataset, indices)
    decoded = [decode_predictions(item, cfg, thresholds) for item in probabilities]
    report = evaluate_predictions(decoded, truths, cfg)
    return {"report": report, "curves": len(decoded)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--synthetic", type=Path, default=Path(r"E:\CableNNDataSet\CodeX\FieldNN_Iter4_20261022"))
    parser.add_argument("--checkpoint", required=True, type=Path, nargs="+")
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    torch.set_num_threads(1)
    device = torch.device(args.device)
    models = []
    checkpoints = []
    for path in args.checkpoint:
        model, checkpoint, cfg = load_model(path, device)
        models.append(model)
        checkpoints.append({"path": str(path.resolve()), "model_name": checkpoint["model_name"], "epoch": int(checkpoint["epoch"])})

    threshold_path = args.thresholds or (Path(args.checkpoint[0]).parent / "thresholds.json")
    calibration = json.loads(Path(threshold_path).read_text(encoding="utf-8"))
    thresholds = {key: float(value) for key, value in calibration["thresholds"].items()}

    result: dict[str, Any] = {
        "data": str(args.data.resolve()),
        "checkpoints": checkpoints,
        "thresholds": thresholds,
        "threshold_policy": calibration.get("policy"),
        "label_policy": (
            "synthetic curves have complete truth; hybrid injected events are exact; "
            "measured-field engineer positions are weak positives and every other "
            "location is unverified, never a negative."
        ),
    }

    synthetic: dict[str, Any] = {}
    for view in ("clean", "mixed"):
        dataset = SyntheticWrapper(args.synthetic, "test", train=False, view=view)
        synthetic[view] = evaluate_synthetic(models, dataset, cfg, thresholds, device, args.batch_size)
    result["synthetic_test"] = synthetic

    hybrid = HybridDataset(args.data, "test", augment=False)
    hyb_probs, hyb_indices = collect_probabilities(models, hybrid, device, args.batch_size)
    hyb_records = [hybrid.records[index] for index in hyb_indices]
    result["hybrid_test"] = hybrid_metrics(hyb_probs, hyb_records, cfg, thresholds)
    result["hybrid_test"]["curves_total"] = len(hybrid)

    field = FieldDataset(args.data, "test")
    field_probs, field_indices = collect_probabilities(models, field, device, args.batch_size)
    field_records = [field.records[index] for index in field_indices]
    result["field_test"] = field_metrics(field_probs, field_records, cfg, thresholds)
    result["field_test"]["curves_total"] = len(field)

    calibration_dataset = CalibrationDataset(args.data, "test", cfg)
    if len(calibration_dataset):
        cal_probs, _ = collect_probabilities(models, calibration_dataset, device, args.batch_size)
        result["calibration_test"] = calibration_metrics(cal_probs, cfg, thresholds)
    else:
        result["calibration_test"] = {"references": 0}

    output = args.output or Path(args.checkpoint[0]).parent / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    summary = {
        "output": str(output / "evaluation.json"),
        "thresholds": thresholds,
        "synthetic_clean_event_f1_5m": synthetic["clean"]["report"]["event_metrics_by_tolerance_m"]["5"]["f1"],
        "synthetic_mixed_event_f1_5m": synthetic["mixed"]["report"]["event_metrics_by_tolerance_m"]["5"]["f1"],
        "synthetic_clean_interval_f1": synthetic["clean"]["report"]["interval_metrics"]["f1"],
        "synthetic_clean_terminal_f1": synthetic["clean"]["report"]["terminal_metrics"]["f1"],
        "hybrid_hit20": result["hybrid_test"].get("hit_rate_20"),
        "hybrid_candidates_per_curve": result["hybrid_test"].get("candidates_per_curve"),
        "hybrid_strict": result["hybrid_test"].get("strict"),
        "field_hit20": result["field_test"].get("hit_rate_20"),
        "field_candidates_per_curve": result["field_test"].get("candidates_per_curve"),
        "field_strict": result["field_test"].get("strict"),
        "wet_1500": result["field_test"].get("wet_1500"),
        "calibration_false_events": result["calibration_test"].get("false_event_count"),
        "calibration_references": result["calibration_test"].get("references"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
