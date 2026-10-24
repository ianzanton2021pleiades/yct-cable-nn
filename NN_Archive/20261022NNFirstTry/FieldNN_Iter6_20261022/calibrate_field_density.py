"""Select a field-event confidence threshold under a validation density cap.

This is candidate-subset analysis: it filters saved decoded peaks by their
stored confidence and does not rerun decoding or inference.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_DATA = Path(r"E:\CableNNDataSet\CodeX\FieldNN_Iter6_20261022")
DEFAULT_OUTPUT = HERE / "output" / "density_analysis" / "field_density_analysis.json"
ITER6_PAIRED = HERE / "output" / "iter6" / "paired"
ITER5_OUTPUT = Path(r"D:\GitRepository\Cable-NN\NN_Archive\20261022NNFirstTry\FieldNN_Iter5_20261022\output\iter5")

MODEL_FILES = {
    "unet": {
        "candidate": ITER6_PAIRED / "candidate" / "unet" / "field_fixed_e000" / "predictions.jsonl",
        "baseline": ITER5_OUTPUT / "unet" / "field_e06" / "predictions.jsonl",
    },
    "transformer": {
        "candidate": ITER6_PAIRED / "candidate" / "transformer" / "field_fixed_e001" / "predictions.jsonl",
        "baseline": ITER5_OUTPUT / "transformer" / "field_e007" / "predictions.jsonl",
    },
}

GRID = tuple(index / 100.0 for index in range(60, 101))
TOLERANCE_M = 20.0


def load_evaluator():
    path = HERE / "evaluate_field.py"
    spec = importlib.util.spec_from_file_location("iter6_density_evaluator", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load evaluator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def filter_by_confidence(
    predictions_by_index: dict[int, dict[str, Any]], threshold: float | None,
) -> dict[int, dict[str, Any]]:
    filtered: dict[int, dict[str, Any]] = {}
    for index, source_row in predictions_by_index.items():
        row = copy.deepcopy(source_row)
        if row.get("status") == "predicted":
            events = (row.get("prediction") or {}).get("events") or []
            if threshold is None:
                events = []
            else:
                events = [event for event in events if float(event["confidence"]) >= threshold]
            row["prediction"]["events"] = events
        filtered[int(index)] = row
    return filtered


def density_summary(
    indices: set[int], predictions_by_index: dict[int, dict[str, Any]], evaluator,
) -> dict[str, Any]:
    status_counts = Counter()
    event_total = 0
    for index in indices:
        row = predictions_by_index.get(index)
        status = "missing_prediction" if row is None else str(row.get("status", "missing_status"))
        status_counts[status] += 1
        if status == "predicted":
            event_total += len(evaluator.predicted_events(row))
    predicted = status_counts["predicted"]
    return {
        "applicable_curves": len(indices),
        "predicted_curves": predicted,
        "missing_or_nonpredicted_curves": len(indices) - predicted,
        "event_candidates_unverified": event_total,
        "mean_candidates_per_predicted_curve": event_total / predicted if predicted else None,
        "status_counts": dict(status_counts),
    }


def macro_hit_summary(
    split_bags: list[dict[str, Any]], records_by_index: dict[int, dict[str, Any]],
    predictions_by_index: dict[int, dict[str, Any]], allowed_indices: set[int], evaluator,
) -> dict[str, Any]:
    report = evaluator.positive_summary(
        split_bags, records_by_index, predictions_by_index, allowed_indices
    )
    macro = report["independent_group_macro"][str(int(TOLERANCE_M))]
    return {
        "tolerance_m": TOLERANCE_M,
        "groups": macro["independent_groups"],
        "matched_positions": macro["matched_positions"],
        "unique_positions": macro["unique_positions"],
        "group_macro_hit_rate": macro["macro_hit_rate"],
        "micro_hit_rate": macro["micro_hit_rate"],
        "mean_candidates_per_group_best_sweep": macro[
            "mean_unverified_candidates_per_group_best_sweep"
        ],
    }


def population_metrics(
    label: str, split_bags: list[dict[str, Any]], records_by_index: dict[int, dict[str, Any]],
    predictions_by_index: dict[int, dict[str, Any]],
    allowed_indices: set[int], evaluator,
) -> dict[str, Any]:
    positive = evaluator.positive_summary(
        split_bags, records_by_index, predictions_by_index, allowed_indices
    )
    macro = positive["independent_group_macro"][str(int(TOLERANCE_M))]
    known_lab_bags = [row for row in positive["bags"] if row.get("known_lab")]
    return {
        "population": label,
        "bag_count": len(split_bags),
        "group_count": len({bag["group"] for bag in split_bags}),
        "density": density_summary(allowed_indices, predictions_by_index, evaluator),
        "positive_labels": {
            "tolerance_m": TOLERANCE_M,
            "groups": macro["independent_groups"],
            "matched_positions": macro["matched_positions"],
            "unique_positions": macro["unique_positions"],
            "group_macro_hit_rate": macro["macro_hit_rate"],
            "micro_hit_rate": macro["micro_hit_rate"],
            "mean_candidates_per_group_best_sweep": macro[
                "mean_unverified_candidates_per_group_best_sweep"
            ],
        },
        "known_lab_positions": evaluator.known_lab_summary(known_lab_bags),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    for model, files in MODEL_FILES.items():
        parser.add_argument(f"--candidate-{model}", type=Path, default=files["candidate"])
        parser.add_argument(f"--baseline-{model}", type=Path, default=files["baseline"])
    args = parser.parse_args()

    evaluator = load_evaluator()
    data_root = args.data.expanduser().resolve()
    records = evaluator.read_jsonl(data_root / "field_records.jsonl")
    bag_data = json.loads((data_root / "field_bags.json").read_text(encoding="utf-8"))
    bags = bag_data["bags"] if isinstance(bag_data, dict) else bag_data
    records_by_index = {int(record["index"]): record for record in records}
    all_predictions: dict[str, dict[int, dict[str, Any]]] = {}
    prediction_paths: dict[str, str] = {}

    for model in MODEL_FILES:
        candidate_path = getattr(args, f"candidate_{model}").expanduser().resolve()
        baseline_path = getattr(args, f"baseline_{model}").expanduser().resolve()
        candidate_keyed = evaluator.load_prediction_files([f"candidate={candidate_path}"])["candidate"]
        baseline_keyed = evaluator.load_prediction_files([f"baseline={baseline_path}"])["baseline"]
        all_predictions[f"{model}_candidate"] = evaluator.prediction_by_index(records, candidate_keyed)
        all_predictions[f"{model}_baseline"] = evaluator.prediction_by_index(records, baseline_keyed)
        prediction_paths[f"{model}_candidate"] = str(candidate_path)
        prediction_paths[f"{model}_baseline"] = str(baseline_path)

    val_indices_by_model: dict[str, set[int]] = {}
    threshold_reports: dict[str, Any] = {}
    selected_thresholds: dict[str, float] = {}

    # Thresholds are selected only from the common, applicable validation rows.
    for model in MODEL_FILES:
        baseline = all_predictions[f"{model}_baseline"]
        candidate = all_predictions[f"{model}_candidate"]
        val_records = [record for record in records if record["split"] == "val"]
        val_indices = {int(record["index"]) for record in val_records}
        common_val = {
            index for index in val_indices
            if index in baseline and index in candidate
            and baseline[index].get("status") == "predicted"
            and candidate[index].get("status") == "predicted"
        }
        val_indices_by_model[model] = common_val
        val_bags = evaluator.bag_rows("val", bags, records_by_index)
        baseline_density = density_summary(common_val, baseline, evaluator)
        baseline_macro = macro_hit_summary(
            val_bags, records_by_index, baseline, common_val, evaluator
        )

        grid_reports = []
        feasible = []
        for threshold in GRID:
            filtered = filter_by_confidence(candidate, threshold)
            density = density_summary(common_val, filtered, evaluator)
            macro = macro_hit_summary(
                val_bags, records_by_index, filtered, common_val, evaluator
            )
            row = {
                "threshold": threshold,
                "density": density,
                "positive_labels": macro,
                "density_feasible": (
                    density["mean_candidates_per_predicted_curve"] is not None
                    and density["mean_candidates_per_predicted_curve"]
                    <= baseline_density["mean_candidates_per_predicted_curve"] + 1e-12
                ),
            }
            grid_reports.append(row)
            if row["density_feasible"]:
                score = macro["group_macro_hit_rate"]
                feasible.append((
                    -1.0 if score is None else float(score),
                    -float(density["event_candidates_unverified"]),
                    threshold,
                    filtered,
                    row,
                ))

        if feasible:
            # Maximize group-macro hit; ties use fewer candidates, then the higher threshold.
            chosen = max(feasible, key=lambda item: (item[0], item[1], item[2]))
            threshold = chosen[2]
            selected = chosen[3]
            selection = {
                "rule": "max validation group-macro best-single-sweep hit@20m under baseline validation candidate-density cap",
                "tie_break": "fewer candidates, then higher threshold",
                "feasible_threshold_count": len(feasible),
                "fallback_no_feasible_threshold": False,
                "selected_grid_row": chosen[4],
            }
        else:
            threshold = 1.0
            selected = filter_by_confidence(candidate, None)
            selection = {
                "rule": "no feasible threshold; fallback to 1.0 with all candidate events suppressed",
                "tie_break": None,
                "feasible_threshold_count": 0,
                "fallback_no_feasible_threshold": True,
                "selected_grid_row": None,
            }
        selected_thresholds[model] = threshold
        threshold_reports[model] = {
            "baseline_val_density_at_locked_threshold": baseline_density,
            "baseline_val_positive_labels": baseline_macro,
            "common_applicable_val_curves": len(common_val),
            "selected_threshold": threshold,
            "selection": selection,
            "grid": grid_reports,
        }

    # Test is evaluated only after each model's validation-only threshold is fixed.
    test_report: dict[str, Any] = {}
    test_records = [record for record in records if record["split"] == "test"]
    wet_indices = {int(record["index"]) for record in test_records if record.get("known_lab")}
    other_indices = {int(record["index"]) for record in test_records if not record.get("known_lab")}
    test_bags = evaluator.bag_rows("test", bags, records_by_index)
    wet_bags = [bag for bag in test_bags if bag["known_lab"]]
    other_bags = [bag for bag in test_bags if not bag["known_lab"]]

    for model in MODEL_FILES:
        baseline = all_predictions[f"{model}_baseline"]
        candidate_fixed = all_predictions[f"{model}_candidate"]
        candidate_selected = filter_by_confidence(candidate_fixed, selected_thresholds[model])
        if threshold_reports[model]["selection"]["fallback_no_feasible_threshold"]:
            candidate_selected = filter_by_confidence(candidate_fixed, None)
        populations = {}
        for label, population_bags, population_indices in (
            ("wet_1500m_known_positive", wet_bags, wet_indices),
            ("other_test_groups", other_bags, other_indices),
        ):
            populations[label] = {
                "baseline_at_locked_threshold": population_metrics(
                    label, population_bags, records_by_index, baseline, population_indices, evaluator
                ),
                "candidate_fixed_0.6": population_metrics(
                    label, population_bags, records_by_index, candidate_fixed, population_indices, evaluator
                ),
                "candidate_validation_selected": population_metrics(
                    label, population_bags, records_by_index, candidate_selected, population_indices, evaluator
                ),
                "applicable_records": len(population_indices),
            }
        test_report[model] = {
            "selected_threshold_from_val": selected_thresholds[model],
            "populations": populations,
        }

    result = {
        "schema": "fieldnn-iter6-density-calibration-1",
        "data": str(data_root),
        "prediction_files": prediction_paths,
        "baseline_threshold": 0.6,
        "threshold_grid": [round(value, 2) for value in GRID],
        "selection_population": "field_records in val split, using only matched predicted rows",
        "density_constraint": "candidate event count per applicable val curve must not exceed the matched Iter5 baseline at its locked 0.6 threshold",
        "objective": "maximize Iter6 evaluator independent-group macro of best single-sweep one-to-one hit@20m",
        "confidence_filter_assumption": (
            "The saved candidate predictions were decoded at event threshold 0.6. Raising the threshold filters only "
            "stored decoded events by confidence; it cannot recover peaks already omitted at 0.6 and does not rerun inference."
        ),
        "interpretation_limit": "The density cap matches candidate volume, not precision; unmatched candidates are unverified and labels are positive-only.",
        "validation_selection": threshold_reports,
        "test_after_selection": test_report,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
