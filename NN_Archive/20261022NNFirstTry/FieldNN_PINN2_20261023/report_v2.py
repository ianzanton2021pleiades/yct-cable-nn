"""Collect the numbers quoted in the report from the logged artefacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PINN1 = HERE.parents[0] / "FieldNN_PINN_20261023" / "output"


def history(path: Path, key: str) -> list[tuple[int, float]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    values = []
    for row in rows:
        raw = row.get(key, "")
        if raw in ("", "None"):
            continue
        values.append((int(float(row["epoch"])), float(raw)))
    return values


def summary(name: str, path: Path, keys=("train_supervision", "val_supervision")) -> dict:
    output = {"name": name}
    for key in keys:
        values = history(path, key)
        if not values:
            continue
        best = min(values, key=lambda item: item[1])
        output[key] = {
            "final": values[-1][1],
            "minimum": best[1],
            "minimum_epoch": best[0],
            "at_100": next((value for epoch, value in values if epoch == 100), None),
            "at_50": next((value for epoch, value in values if epoch == 50), None),
        }
    return output


def field_block(payload: dict) -> dict:
    field = payload["field_test"]
    annotation = field.get("annotation_summary", {})
    return {
        "epoch": payload.get("epoch"),
        "non_wet": {key: field["non_wet"].get(key) for key in
                    ("hit_rate_20", "candidates_per_curve", "group_macro_hit_20")},
        "annotation": {key: annotation.get(key) for key in
                       ("P_ann", "curves_all_positions_hit", "curves_any_position_hit",
                        "annotated_curves", "mae_matched_m", "terminal_error_median_m")},
        "wet": {key: field.get("wet_1500", {}).get(key) for key in ("hit_rate_20",)},
        "calibration": payload.get("calibration_test", {}).get("false_events_per_reference"),
    }


def synthetic_block(payload: dict) -> dict:
    output = {}
    for view in ("clean", "mixed"):
        block = payload["synthetic_test"][view]
        metrics = block["event_metrics_by_tolerance_m"]["5"]
        output[view] = {
            "f1": metrics["f1"], "precision": metrics["precision"], "recall": metrics["recall"],
            "f1_2m": block["event_metrics_by_tolerance_m"]["2"]["f1"],
            "interval_f1": block["interval_metrics"]["f1"],
            "terminal_f1": block["terminal_metrics"]["f1"],
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-history", type=Path, default=HERE / "output" / "v2_100_cont" / "unet" / "history_stitched.csv")
    parser.add_argument("--v1-history", type=Path, default=PINN1 / "hist_150" / "unet" / "history.csv")
    parser.add_argument("--v2-pretrain", type=Path, default=HERE / "output" / "v2_100_cont" / "unet" / "evaluation.json")
    parser.add_argument("--v2-field", type=Path, default=HERE / "output" / "field_ft" / "best.pt")
    args = parser.parse_args()

    result = {"loss": [], "metrics": {}}
    if args.v2_history.exists():
        result["loss"].append(summary("v2", args.v2_history))
    if args.v1_history.exists():
        result["loss"].append(summary("v1", args.v1_history))
    if args.v2_pretrain.exists():
        payload = json.loads(args.v2_pretrain.read_text(encoding="utf-8"))
        result["metrics"]["v2_pretrain"] = {
            "synthetic": synthetic_block(payload), "field": field_block(payload)
        }
    field_dir = args.v2_field.parent
    field_eval = field_dir / "evaluation.json"
    if field_eval.exists():
        payload = json.loads(field_eval.read_text(encoding="utf-8"))
        result["metrics"]["v2_field_finetuned"] = {
            "synthetic": synthetic_block(payload), "field": field_block(payload)
        }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
