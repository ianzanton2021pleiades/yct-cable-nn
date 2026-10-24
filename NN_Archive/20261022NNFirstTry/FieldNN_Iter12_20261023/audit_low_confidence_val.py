"""One CPU-only validation pass for the Iter12 T5-initialized Transformer."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
I5 = HERE.parent / "FieldNN_Iter5_20261022"
I8 = HERE.parent / "FieldNN_Iter8_20261022"
I6 = HERE.parent / "FieldNN_Iter6_20261022"
DATA = Path(r"E:\CableNNDataSet\CodeX\FieldNN_Iter8_20261022")
CHECKPOINT = I5 / "output" / "iter5" / "transformer" / "best.pt"
OUTPUT = HERE / "output" / "low_confidence_val_audit" / "summary.json"
DECODE_MINIMUM = 0.05
THRESHOLDS = tuple(round(step / 20.0, 2) for step in range(1, 21))
U5_VAL_CURVES = 31
U5_VAL_CANDIDATES = 79
TOLERANCE_M = 20.0


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def decode_source_events(probabilities, cfg, record, decoder) -> dict[str, Any]:
    prediction = decoder.decode_predictions(
        probabilities,
        cfg,
        {"event": DECODE_MINIMUM, "interval": 0.99, "terminal": 0.4},
    )
    scale = float(record.get("source_scale_ratio") or 1.0)
    for event in prediction["events"]:
        event["distance_m"] = float(event["distance_m"]) * scale
    return prediction


def filter_events(
    predictions: dict[int, dict[str, Any]], threshold: float,
) -> dict[int, dict[str, Any]]:
    return {
        index: {
            **row,
            "prediction": {
                **row["prediction"],
                "events": [
                    event for event in row["prediction"]["events"]
                    if float(event["confidence"]) >= threshold
                ],
            },
        }
        for index, row in predictions.items()
    }


def main() -> int:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if str(I8) not in sys.path:
        sys.path.insert(0, str(I8))

    models = load_module(I8 / "models.py", "iter12_lowconf_i8_models")
    i8_data = load_module(I8 / "data.py", "iter12_lowconf_i8_data")
    evaluator = load_module(I6 / "evaluate_field.py", "iter12_lowconf_evaluator")
    decoder = load_module(I5 / "decode.py", "iter12_lowconf_decoder")

    initial = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    cfg = copy.deepcopy(initial["config"])
    cfg["model"]["input_channels"] = 12
    model = models.build_model("transformer", **cfg["model"]).cpu().eval()
    model.backbone.load_state_dict(initial["model"])

    field_features = np.load(DATA / "field_features.npy", mmap_mode="r")
    records = evaluator.read_jsonl(DATA / "field_records.jsonl")
    val_records = [record for record in records if record["split"] == "val"]
    val_records_by_index = {int(record["index"]): record for record in val_records}
    val_indices = sorted(val_records_by_index)
    val_bag_data = json.loads((DATA / "field_bags.json").read_text(encoding="utf-8"))
    all_bags = val_bag_data["bags"] if isinstance(val_bag_data, dict) else val_bag_data
    val_bags = evaluator.bag_rows("val", all_bags, val_records_by_index)

    calibration = i8_data.CalibrationDataset(DATA, "val")
    calibration_samples = [calibration[index] for index in range(len(calibration))]
    calibration_features = np.stack([sample[0] for sample in calibration_samples]).astype(
        np.float32, copy=False
    )
    calibration_indices = [int(sample[2]) for sample in calibration_samples]
    val_field_features = np.array(field_features[val_indices], dtype=np.float32, copy=True)
    all_features = np.concatenate((val_field_features, calibration_features), axis=0)

    with torch.inference_mode():
        logits = model(torch.from_numpy(all_features)).float().numpy()
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    del logits, all_features, val_field_features, calibration_features

    field_predictions: dict[int, dict[str, Any]] = {}
    for offset, index in enumerate(val_indices):
        record = val_records_by_index[index]
        field_predictions[index] = {
            "record": record,
            "status": "predicted",
            "prediction": decode_source_events(
                probabilities[offset], cfg, record, decoder
            ),
            "distance_axis": "source_report",
        }

    calibration_event_lists = []
    calibration_offset = len(val_indices)
    for offset, index in enumerate(calibration_indices):
        prediction = decoder.decode_predictions(
            probabilities[calibration_offset + offset],
            cfg,
            {"event": DECODE_MINIMUM, "interval": 0.99, "terminal": 0.4},
        )
        calibration_event_lists.append(prediction["events"])
    del probabilities

    rows = []
    for threshold in THRESHOLDS:
        filtered = filter_events(field_predictions, threshold)
        positive = evaluator.positive_summary(
            val_bags, val_records_by_index, filtered, set(val_indices)
        )
        macro = positive["independent_group_macro"][str(int(TOLERANCE_M))]
        candidates = sum(
            len(row["prediction"]["events"]) for row in filtered.values()
        )
        per_reference = [
            sum(float(event["confidence"]) >= threshold for event in events)
            for events in calibration_event_lists
        ]
        rows.append({
            "threshold": threshold,
            "group_macro_hit20": float(macro["macro_hit_rate"] or 0.0),
            "group_macro_hit20_rounded_12": round(
                float(macro["macro_hit_rate"] or 0.0), 12
            ),
            "matched_positions": int(macro["matched_positions"]),
            "unique_positions": int(macro["unique_positions"]),
            "candidate_total_unverified": candidates,
            "candidates_per_val_curve": candidates / len(val_indices),
            "budget_feasible": candidates <= U5_VAL_CANDIDATES,
            "calibration_false_event_count": sum(per_reference),
            "calibration_references_with_event": sum(count > 0 for count in per_reference),
        })

    baseline = next(row for row in rows if row["threshold"] == 0.6)
    for row in rows:
        row["higher_known_hit_than_0_6"] = (
            row["group_macro_hit20_rounded_12"]
            > baseline["group_macro_hit20_rounded_12"]
        )
        row["zero_calibration_events_and_budget_feasible"] = (
            row["calibration_false_event_count"] == 0 and row["budget_feasible"]
        )
    qualifying = [
        row for row in rows
        if row["higher_known_hit_than_0_6"]
        and row["zero_calibration_events_and_budget_feasible"]
    ]
    best_qualifying = max(
        qualifying,
        key=lambda row: (
            row["group_macro_hit20_rounded_12"],
            -row["candidate_total_unverified"],
            row["threshold"],
        ),
        default=None,
    )

    result = {
        "schema": "fieldnn-iter12-low-confidence-val-audit-1",
        "model_state": {
            "description": "step-0 I12 initialization: Iter8 12-channel Transformer identity input stem plus Iter5 Transformer backbone",
            "checkpoint": str(CHECKPOINT),
            "running_i12_checkpoint_used": False,
            "training_or_weight_update": False,
        },
        "scope": {
            "split": "field val and calibration val only",
            "field_val_curves": len(val_indices),
            "field_val_bags": len(val_bags),
            "field_val_groups": len({bag["group"] for bag in val_bags}),
            "calibration_val_references": len(calibration),
            "calibration_semantics": "short/no-test-cable fixture references; not healthy-cable false-positive evidence",
            "cpu_threads": 1,
            "forward_passes": 1,
            "event_decode_minimum": DECODE_MINIMUM,
            "thresholds": list(THRESHOLDS),
            "known_position_tolerance_m": TOLERANCE_M,
            "candidate_budget_total": U5_VAL_CANDIDATES,
            "candidate_budget_per_curve": U5_VAL_CANDIDATES / U5_VAL_CURVES,
            "test_used_for_selection": False,
        },
        "inputs": {
            "field_features": str(DATA / "field_features.npy"),
            "field_records": str(DATA / "field_records.jsonl"),
            "field_bags": str(DATA / "field_bags.json"),
            "calibration_cache": str(DATA),
        },
        "baseline_threshold_0_6": baseline,
        "qualifying_zero_calibration_event_and_budget_workpoint": best_qualifying,
        "qualifying_workpoint_exists": best_qualifying is not None,
        "threshold_results": rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "baseline_threshold_0_6": baseline,
        "qualifying_workpoint_exists": best_qualifying is not None,
        "best_qualifying_workpoint": best_qualifying,
        "threshold_results": rows,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
