"""Compare the Iter8 validation threshold grid with exact confidence cutoffs."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ITER8 = HERE.parent / "FieldNN_Iter8_20261022"
ITER6 = HERE.parent / "FieldNN_Iter6_20261022"
DENSITY_REPORT = ITER8 / "output" / "iter8" / "density_analysis.json"
OUTPUT = HERE / "output" / "threshold_resolution_val.json"
GRID = tuple(index / 100.0 for index in range(60, 101))
TOLERANCE_M = 20.0
U5_VAL_CANDIDATES = 79


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate_threshold(
    threshold: float,
    predictions: dict[int, dict[str, Any]],
    val_bags: list[dict[str, Any]],
    records_by_index: dict[int, dict[str, Any]],
    val_indices: set[int],
    evaluator,
    density_tools,
) -> dict[str, Any]:
    filtered = density_tools.filter_by_confidence(predictions, threshold)
    density = density_tools.density_summary(val_indices, filtered, evaluator)
    positive = evaluator.positive_summary(
        val_bags, records_by_index, filtered, val_indices
    )
    macro = positive["independent_group_macro"][str(int(TOLERANCE_M))]
    candidates = int(density["event_candidates_unverified"])
    return {
        "threshold": float(threshold),
        "group_macro_hit20": float(macro["macro_hit_rate"] or 0.0),
        "group_macro_hit20_rounded_12": round(float(macro["macro_hit_rate"] or 0.0), 12),
        "micro_hit20": float(macro["micro_hit_rate"] or 0.0),
        "matched_positions": int(macro["matched_positions"]),
        "unique_positions": int(macro["unique_positions"]),
        "independent_groups": int(macro["independent_groups"]),
        "candidate_total_unverified": candidates,
        "candidates_per_val_curve": candidates / len(val_indices),
        "density_feasible": candidates <= U5_VAL_CANDIDATES,
    }


def select_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    feasible = [row for row in rows if row["density_feasible"]]
    return max(
        feasible,
        key=lambda row: (
            row["group_macro_hit20_rounded_12"],
            -row["candidate_total_unverified"],
            row["threshold"],
        ),
    )


def main() -> int:
    density_report = json.loads(DENSITY_REPORT.read_text(encoding="utf-8"))
    data_root = Path(density_report["data"])
    evaluator = load_module(ITER6 / "evaluate_field.py", "threshold_resolution_evaluator")
    density_tools = load_module(
        ITER6 / "calibrate_field_density.py", "threshold_resolution_density"
    )

    records = evaluator.read_jsonl(data_root / "field_records.jsonl")
    val_records = [row for row in records if row["split"] == "val"]
    records_by_index = {int(row["index"]): row for row in val_records}
    val_indices = set(records_by_index)
    bag_data = json.loads((data_root / "field_bags.json").read_text(encoding="utf-8"))
    all_bags = bag_data["bags"] if isinstance(bag_data, dict) else bag_data
    val_bags = evaluator.bag_rows("val", all_bags, records_by_index)

    prediction_paths = {
        "unet": Path(density_report["prediction_files"]["unet_candidate"]),
        "transformer": Path(density_report["prediction_files"]["transformer_candidate"]),
    }
    reports: dict[str, Any] = {}
    for model, prediction_path in prediction_paths.items():
        keyed = evaluator.load_prediction_files(
            [f"{model}={prediction_path}"]
        )[model]
        predictions = evaluator.prediction_by_index(val_records, keyed)
        if set(predictions) != val_indices or any(
            row.get("status") != "predicted" for row in predictions.values()
        ):
            raise ValueError(f"{model} does not contain predictions for all val curves")

        confidence_values = sorted({
            float(event["confidence"])
            for row in predictions.values()
            for event in (row.get("prediction") or {}).get("events", [])
        })
        # Keep both ends of the threshold domain so the exact-confidence sweep
        # can represent the same boundary choices as the original 0.01 grid.
        exact_thresholds = sorted(set(confidence_values) | {GRID[0], GRID[-1]})
        grid_rows = [
            evaluate_threshold(t, predictions, val_bags, records_by_index,
                               val_indices, evaluator, density_tools)
            for t in GRID
        ]
        exact_rows = [
            evaluate_threshold(t, predictions, val_bags, records_by_index,
                               val_indices, evaluator, density_tools)
            for t in exact_thresholds
        ]
        grid_best = select_best(grid_rows)
        exact_best = select_best(exact_rows)
        macro_delta = (
            exact_best["group_macro_hit20_rounded_12"]
            - grid_best["group_macro_hit20_rounded_12"]
        )
        candidate_delta = (
            exact_best["candidate_total_unverified"]
            - grid_best["candidate_total_unverified"]
        )
        reports[model] = {
            "prediction_file": str(prediction_path),
            "unique_val_confidence_count": len(confidence_values),
            "grid": {
                "threshold_count": len(grid_rows),
                "feasible_threshold_count": sum(row["density_feasible"] for row in grid_rows),
                "selected": grid_best,
            },
            "exact_confidence": {
                "threshold_count_including_endpoints": len(exact_rows),
                "feasible_threshold_count": sum(row["density_feasible"] for row in exact_rows),
                "selected": exact_best,
            },
            "exact_minus_grid": {
                "group_macro_hit20_rounded_12": macro_delta,
                "candidate_total_unverified": candidate_delta,
                "same_macro_fewer_candidates": macro_delta == 0 and candidate_delta < 0,
            },
        }

    result = {
        "schema": "fieldnn-iter10-threshold-resolution-val-1",
        "scope": {
            "split": "val only",
            "val_curves": len(val_indices),
            "val_bags": len(val_bags),
            "val_groups": len({bag["group"] for bag in val_bags}),
            "tolerance_m": TOLERANCE_M,
            "budget_source": "Iter5 U-Net locked validation density",
            "candidate_budget_total": U5_VAL_CANDIDATES,
            "candidate_budget_per_curve": U5_VAL_CANDIDATES / len(val_indices),
            "selection_tie_break": "maximize group macro rounded to 12 decimals, then fewer candidates, then higher threshold",
            "test_used_for_selection": False,
            "candidate_limit": "offline confidence filtering of Iter8 predictions decoded at 0.6; omitted peaks cannot be recovered",
            "unmatched_candidates": "unverified, not treated as false positives",
        },
        "inputs": {
            "density_analysis": str(DENSITY_REPORT),
            "field_data_root": str(data_root),
            "field_records": str(data_root / "field_records.jsonl"),
            "field_bags": str(data_root / "field_bags.json"),
        },
        "models": reports,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["models"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
