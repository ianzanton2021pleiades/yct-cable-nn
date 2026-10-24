"""Evaluate Iter6 field detections against case-grouped, partial field labels.

Positions and predictions are compared on the source/report distance axis. The
known annotations are positive-only evidence; unmatched predictions remain
unverified candidates and are never counted as false positives.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


TOLERANCES_M = (10.0, 20.0, 30.0)
SPLITS = ("train", "val", "test")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def record_key(record: dict[str, Any]) -> tuple[str, str]:
    source = str(record.get("source_relative_path") or record.get("input_path") or "")
    source = source.replace("\\", "/").casefold()
    return source, str(record.get("sweep_rowid"))


def one_to_one_match(
    positions_m: list[float], events_m: list[float], tolerance_m: float
) -> dict[str, Any]:
    """Maximize hit count first, then minimize total distance error."""
    truth = np.asarray(positions_m, dtype=np.float64)
    events = np.asarray(events_m, dtype=np.float64)
    n_truth, n_events = len(truth), len(events)
    if n_truth == 0:
        return {
            "position_count": 0,
            "candidate_count": n_events,
            "matched_count": 0,
            "hit_rate": None,
            "mean_error_m": None,
            "matches": [],
        }

    unmatched_cost = (n_truth + 1) * (tolerance_m + 1.0)
    costs = np.full((n_truth, n_events + n_truth), unmatched_cost * 2.0)
    if n_events:
        errors = np.abs(truth[:, None] - events[None, :])
        costs[:, :n_events] = np.where(errors <= tolerance_m, errors, unmatched_cost * 2.0)
    costs[:, n_events:] = unmatched_cost

    rows, columns = linear_sum_assignment(costs)
    matches = []
    for row, column in zip(rows.tolist(), columns.tolist()):
        if column >= n_events:
            continue
        error = float(abs(truth[row] - events[column]))
        if error <= tolerance_m:
            matches.append({
                "position_m": float(truth[row]),
                "event_m": float(events[column]),
                "error_m": error,
            })
    matches.sort(key=lambda item: item["position_m"])
    errors = [item["error_m"] for item in matches]
    return {
        "position_count": n_truth,
        "candidate_count": n_events,
        "matched_count": len(matches),
        "hit_rate": len(matches) / n_truth,
        "mean_error_m": sum(errors) / len(errors) if errors else None,
        "matches": matches,
    }


def load_prediction_files(specs: list[str]) -> dict[str, dict[tuple[str, str], dict[str, Any]]]:
    predictions: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for spec in specs:
        label, filename = spec.split("=", 1)
        rows = read_jsonl(Path(filename))
        by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            by_key[record_key(row.get("record") or {})] = row
        predictions[label] = by_key
    return predictions


def prediction_by_index(
    records: list[dict[str, Any]],
    predictions: dict[tuple[str, str], dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    return {
        int(record["index"]): predictions[record_key(record)]
        for record in records
        if record_key(record) in predictions
    }


def predicted_events(row: dict[str, Any]) -> list[float]:
    prediction = row.get("prediction") or {}
    return [float(event["distance_m"]) for event in prediction.get("events", [])]


def bag_rows(
    split: str,
    bags: list[dict[str, Any]],
    records_by_index: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for bag in bags:
        if bag["split"] != split:
            continue
        indices = sorted({int(index) for index in bag["indices"]})
        indices = [
            index for index in indices
            if index in records_by_index and records_by_index[index]["split"] == split
        ]
        positions = sorted({round(float(value), 6) for value in bag["positions_source_m"]})
        source_paths = sorted({
            str(records_by_index[index].get("source_relative_path") or "")
            .replace("\\", "/").casefold()
            for index in indices
        })
        source_identity = str(bag.get("source_relative_path") or "")
        if not source_identity:
            source_identity = "|".join(source_paths)
        source_identity = source_identity.replace("\\", "/").casefold()
        label_text = " ".join(str(bag.get(key) or "") for key in ("group", "case_group", "case_name"))
        known_lab = bool(bag.get("known_lab")) or (
            "1500" in label_text.lower()
            and ("wet" in label_text.lower() or "浸水" in label_text)
        )
        rows.append({
            **bag,
            "split": split,
            "group": str(bag["group"]),
            "case_group": str(bag["case_group"]),
            "indices": indices,
            "positions_source_m": positions,
            "source_relative_paths": source_paths,
            "source_identity": source_identity,
            "known_lab": known_lab,
        })
    return rows


def sweep_evaluation(
    index: int,
    bag: dict[str, Any],
    record: dict[str, Any],
    prediction: dict[str, Any] | None,
) -> dict[str, Any]:
    status = "missing_prediction" if prediction is None else str(prediction.get("status", "missing_status"))
    row = {
        "index": index,
        "group": bag["group"],
        "case_group": bag["case_group"],
        "source_relative_path": record.get("source_relative_path"),
        "sweep_rowid": record.get("sweep_rowid"),
        "status": status,
    }
    if status != "predicted":
        return row

    events = predicted_events(prediction)
    row["candidate_count"] = len(events)
    row["by_tolerance"] = {
        str(int(tolerance)): one_to_one_match(
            bag["positions_source_m"], events, tolerance
        )
        for tolerance in TOLERANCES_M
    }
    return row


def best_single_sweep(sweeps: list[dict[str, Any]], tolerance: float) -> dict[str, Any] | None:
    available = [row for row in sweeps if row["status"] == "predicted"]
    if not available:
        return None

    def rank(row: dict[str, Any]) -> tuple[float, float]:
        metric = row["by_tolerance"][str(int(tolerance))]
        error_sum = sum(item["error_m"] for item in metric["matches"])
        return metric["matched_count"], -error_sum

    selected = max(available, key=rank)
    metric = selected["by_tolerance"][str(int(tolerance))]
    return {
        "index": selected["index"],
        "source_relative_path": selected["source_relative_path"],
        "sweep_rowid": selected["sweep_rowid"],
        **metric,
    }


def bag_evaluation(
    bag: dict[str, Any],
    records_by_index: dict[int, dict[str, Any]],
    predictions_by_index: dict[int, dict[str, Any]],
    allowed_indices: set[int] | None = None,
) -> dict[str, Any]:
    indices = [index for index in bag["indices"] if allowed_indices is None or index in allowed_indices]
    sweeps = [
        sweep_evaluation(
            index,
            bag,
            records_by_index[index],
            predictions_by_index.get(index),
        )
        for index in indices
    ]
    statuses = Counter(row["status"] for row in sweeps)
    by_tolerance = {}
    for tolerance in TOLERANCES_M:
        by_tolerance[str(int(tolerance))] = best_single_sweep(sweeps, tolerance)
    return {
        "split": bag["split"],
        "group": bag["group"],
        "case_group": bag["case_group"],
        "known_lab": bag["known_lab"],
        "source_relative_paths": bag["source_relative_paths"],
        "source_identity": bag["source_identity"],
        "applicable_indices": indices,
        "position_count": len(bag["positions_source_m"]),
        "positions_source_m": bag["positions_source_m"],
        "sweep_coverage": {
            "expected": len(indices),
            "predicted": statuses["predicted"],
            "missing": statuses["missing_prediction"],
            "other_status": {key: value for key, value in statuses.items()
                              if key not in {"predicted", "missing_prediction"}},
        },
        "bag_best_single_sweep": by_tolerance,
        "per_sweep": sweeps,
    }


def position_claims(
    bag_reports: list[dict[str, Any]],
    tolerance: float,
    per_sweep: bool = False,
) -> tuple[dict[tuple[Any, ...], dict[str, Any]], dict[int, int]]:
    """Deduplicate the same source, applicable sweeps, and position claim."""
    claims: dict[tuple[Any, ...], dict[str, Any]] = {}
    candidates_by_sweep: dict[int, int] = {}
    key = str(int(tolerance))

    for bag in bag_reports:
        context = (
            bag["source_identity"],
            tuple(bag["applicable_indices"]),
        )
        if per_sweep:
            scored_sweeps = [
                row for row in bag["per_sweep"] if row["status"] == "predicted"
            ]
        else:
            best = bag["bag_best_single_sweep"][key]
            scored_sweeps = [] if best is None else [best]

        for sweep in scored_sweeps:
            index = int(sweep["index"])
            candidate_count = int(sweep["candidate_count"])
            candidates_by_sweep[index] = candidate_count
            matches = {
                round(float(item["position_m"]), 6): item
                for item in sweep["by_tolerance"][key]["matches"]
            } if per_sweep else {
                round(float(item["position_m"]), 6): item
                for item in sweep["matches"]
            }
            for position in bag["positions_source_m"]:
                position_key = round(float(position), 6)
                claim_key = (*context, index, position_key) if per_sweep else (*context, position_key)
                match = matches.get(position_key)
                value = {
                    "hit": match is not None,
                    "error_m": None if match is None else float(match["error_m"]),
                }
                previous = claims.get(claim_key)
                if previous is None or (
                    value["hit"] and not previous["hit"]
                ) or (
                    value["hit"] == previous["hit"]
                    and value["error_m"] is not None
                    and (previous["error_m"] is None or value["error_m"] < previous["error_m"])
                ):
                    claims[claim_key] = value

        if not per_sweep:
            # A bag with no usable prediction still contributes its labeled
            # positions as misses; its missing coverage is reported separately.
            best = bag["bag_best_single_sweep"][key]
            if best is None:
                for position in bag["positions_source_m"]:
                    claim_key = (*context, round(float(position), 6))
                    claims.setdefault(claim_key, {"hit": False, "error_m": None})

    return claims, candidates_by_sweep


def group_evaluation(
    group: str,
    bag_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    by_tolerance = {}
    for tolerance in TOLERANCES_M:
        claims, candidates = position_claims(bag_reports, tolerance)
        hit_count = sum(value["hit"] for value in claims.values())
        position_count = len(claims)
        candidate_total = sum(candidates.values())
        by_tolerance[str(int(tolerance))] = {
            "matched_bag_position_claims": hit_count,
            "unique_bag_position_claims": position_count,
            # Retained as an alias for reports that used the previous name.
            "unique_positions": position_count,
            "unique_positions_scope": (
                "claims deduplicated by source, applicable sweep indices, and position; "
                "different origin/condition claims remain separate"
            ),
            "hit_rate": hit_count / position_count if position_count else None,
            "selected_best_sweeps": len(candidates),
            "candidate_total_unverified": candidate_total,
            "mean_unverified_candidates_per_best_sweep": (
                candidate_total / len(candidates) if candidates else None
            ),
        }
    return {
        "group": group,
        "case_groups": sorted({row["case_group"] for row in bag_reports}),
        "bag_count": len(bag_reports),
        "by_tolerance": by_tolerance,
    }


def positive_summary(
    bags: list[dict[str, Any]],
    records_by_index: dict[int, dict[str, Any]],
    predictions_by_index: dict[int, dict[str, Any]],
    allowed_indices: set[int] | None = None,
) -> dict[str, Any]:
    bag_reports = [
        bag_evaluation(bag, records_by_index, predictions_by_index, allowed_indices)
        for bag in bags
    ]
    grouped_bags: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in bag_reports:
        grouped_bags[row["group"]].append(row)
    group_reports = [
        group_evaluation(group, rows)
        for group, rows in sorted(grouped_bags.items())
    ]

    per_sweep = {}
    group_macro = {}
    for tolerance in TOLERANCES_M:
        key = str(int(tolerance))
        sweep_claims, sweep_candidates_by_index = position_claims(
            bag_reports, tolerance, per_sweep=True
        )
        sweep_matched = sum(value["hit"] for value in sweep_claims.values())
        sweep_positions = len(sweep_claims)
        sweep_candidates = sum(sweep_candidates_by_index.values())
        per_sweep[key] = {
            "sweeps": len(sweep_candidates_by_index),
            "bag_position_observations": sweep_positions,
            "matched_bag_position_observations": sweep_matched,
            "position_observations": sweep_positions,
            "matched_observations": sweep_matched,
            "hit_rate": sweep_matched / sweep_positions if sweep_positions else None,
            "candidate_total_unverified": sweep_candidates,
            "mean_candidates_per_sweep": (
                sweep_candidates / len(sweep_candidates_by_index) if sweep_candidates_by_index else None
            ),
        }

        scored_groups = [
            row["by_tolerance"][key] for row in group_reports
        ]
        scored_groups = [row for row in scored_groups if row["unique_bag_position_claims"]]
        rates = [row["hit_rate"] for row in scored_groups]
        candidate_means = [
            row["mean_unverified_candidates_per_best_sweep"] for row in scored_groups
            if row["mean_unverified_candidates_per_best_sweep"] is not None
        ]
        total_positions = sum(row["unique_bag_position_claims"] for row in scored_groups)
        total_hits = sum(row["matched_bag_position_claims"] for row in scored_groups)
        group_macro[key] = {
            "independent_groups": len(scored_groups),
            "macro_hit_rate": sum(rates) / len(rates) if rates else None,
            "micro_hit_rate": total_hits / total_positions if total_positions else None,
            "matched_positions": total_hits,
            "matched_bag_position_claims": total_hits,
            "unique_bag_position_claims": total_positions,
            "unique_positions": total_positions,
            "unique_positions_scope": (
                "claims deduplicated by source, applicable sweep indices, and position; "
                "different origin/condition claims remain separate"
            ),
            "mean_unverified_candidates_per_group_best_sweep": (
                sum(candidate_means) / len(candidate_means) if candidate_means else None
            ),
            "candidate_total_unverified": sum(
                row["candidate_total_unverified"] for row in scored_groups
            ),
            "groups_with_candidate_coverage": len(candidate_means),
        }

    return {
        "per_sweep_micro": per_sweep,
        "bags": bag_reports,
        "independent_group_macro": group_macro,
        "groups": group_reports,
    }


def known_lab_summary(bag_reports: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not bag_reports:
        return None
    by_tolerance = {}
    for tolerance in TOLERANCES_M:
        claims, candidates = position_claims(bag_reports, tolerance)
        hits = sum(value["hit"] for value in claims.values())
        positions = len(claims)
        by_tolerance[str(int(tolerance))] = {
            "tolerance_m": tolerance,
            "known_event_hits": hits,
            "known_positions": positions,
            "known_event_hit_rate": hits / positions if positions else None,
            "candidate_total_unverified_or_matched": sum(candidates.values()),
        }
    return {
        "scope": "known positive positions in the held-out wet-1500 m test group; not full truth",
        "bags": len(bag_reports),
        "by_tolerance": by_tolerance,
    }


def unknown_curve_summary(
    split: str,
    records: list[dict[str, Any]],
    bags: list[dict[str, Any]],
    predictions_by_index: dict[int, dict[str, Any]],
    allowed_indices: set[int] | None = None,
) -> dict[str, Any]:
    labeled_indices = {index for bag in bags for index in bag["indices"]}
    unknown = [
        row for row in records
        if row["split"] == split
        and int(row["index"]) not in labeled_indices
        and (allowed_indices is None or int(row["index"]) in allowed_indices)
    ]
    statuses = Counter()
    event_counts = []
    group_counts: dict[str, list[int]] = defaultdict(list)
    for record in unknown:
        index = int(record["index"])
        prediction = predictions_by_index.get(index)
        status = "missing_prediction" if prediction is None else str(prediction.get("status", "missing_status"))
        statuses[status] += 1
        if status == "predicted":
            count = len(predicted_events(prediction))
            event_counts.append(count)
            group_counts[str(record.get("group") or record.get("case_group") or "unknown")].append(count)
    return {
        "scope": "unlabeled curves; candidates are unverified and not false positives",
        "curves": len(unknown),
        "predicted_curves": len(event_counts),
        "status_counts": dict(statuses),
        "events_total_unverified": sum(event_counts),
        "mean_candidates_per_curve": sum(event_counts) / len(event_counts) if event_counts else None,
        "median_candidates_per_curve": (
            median(event_counts) if event_counts else None
        ),
        "group_candidate_means": {
            group: sum(values) / len(values) for group, values in sorted(group_counts.items())
        },
    }


def split_report(
    split: str,
    records: list[dict[str, Any]],
    bags: list[dict[str, Any]],
    records_by_index: dict[int, dict[str, Any]],
    predictions_by_index: dict[int, dict[str, Any]],
    extra_prediction_keys: int,
    allowed_indices: set[int] | None = None,
) -> dict[str, Any]:
    split_records = [row for row in records if row["split"] == split]
    considered = [
        row for row in split_records
        if allowed_indices is None or int(row["index"]) in allowed_indices
    ]
    statuses = Counter()
    for record in considered:
        prediction = predictions_by_index.get(int(record["index"]))
        statuses["missing_prediction" if prediction is None else str(prediction.get("status", "missing_status"))] += 1
    split_bags = bag_rows(split, bags, records_by_index)
    if allowed_indices is not None:
        split_bags = [
            {**bag, "indices": [index for index in bag["indices"] if index in allowed_indices]}
            for bag in split_bags
        ]
    positive = positive_summary(split_bags, records_by_index, predictions_by_index, allowed_indices)
    wet1500 = [
        row for row in positive["bags"]
        if split == "test" and row["known_lab"]
    ]
    return {
        "coverage": {
            "records": len(considered),
            "prediction_rows_matched_by_sweep_key": len(considered) - statuses["missing_prediction"],
            "predicted": statuses["predicted"],
            "missing_prediction": statuses["missing_prediction"],
            "other_status": {key: value for key, value in statuses.items()
                             if key not in {"predicted", "missing_prediction"}},
            "unmatched_prediction_keys_in_file": extra_prediction_keys,
        },
        "positive_labels": positive,
        "unknown_curves": unknown_curve_summary(
            split, records, split_bags, predictions_by_index, allowed_indices
        ),
        "known_lab_1500m_test": known_lab_summary(wet1500),
    }


def comparison_report(
    baseline_label: str,
    candidate_label: str,
    split: str,
    records: list[dict[str, Any]],
    bags: list[dict[str, Any]],
    records_by_index: dict[int, dict[str, Any]],
    baseline_by_index: dict[int, dict[str, Any]],
    candidate_by_index: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    common = {
        int(row["index"])
        for row in records
        if row["split"] == split
        and int(row["index"]) in baseline_by_index
        and int(row["index"]) in candidate_by_index
        and baseline_by_index[int(row["index"])].get("status") == "predicted"
        and candidate_by_index[int(row["index"])].get("status") == "predicted"
    }
    base = split_report(
        split, records, bags, records_by_index, baseline_by_index, 0, common
    )
    cand = split_report(
        split, records, bags, records_by_index, candidate_by_index, 0, common
    )

    def values(report: dict[str, Any]) -> dict[str, Any]:
        return {
            "group_macro_hit_rate@20m": report["positive_labels"]["independent_group_macro"]["20"]["macro_hit_rate"],
            "mean_unverified_candidates_per_group_best_sweep@20m": report["positive_labels"]["independent_group_macro"]["20"]["mean_unverified_candidates_per_group_best_sweep"],
            "unknown_curve_mean_candidates": report["unknown_curves"]["mean_candidates_per_curve"],
        }

    base_values, candidate_values = values(base), values(cand)
    deltas = {
        key: (
            candidate_values[key] - base_values[key]
            if candidate_values[key] is not None and base_values[key] is not None
            else None
        )
        for key in base_values
    }
    return {
        "baseline": baseline_label,
        "candidate": candidate_label,
        "common_predicted_sweeps": len(common),
        "scope": "same split and same sweeps predicted by both; candidate counts remain unverified",
        "baseline_metrics": base_values,
        "candidate_metrics": candidate_values,
        "candidate_minus_baseline": deltas,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = read_jsonl(args.data / "field_records.jsonl")
    bag_data = json.loads((args.data / "field_bags.json").read_text(encoding="utf-8"))
    bags = bag_data["bags"] if isinstance(bag_data, dict) else bag_data
    records_by_index = {int(record["index"]): record for record in records}
    prediction_files = load_prediction_files(args.predictions)
    labels = list(prediction_files)
    reports: dict[str, Any] = {}
    by_index: dict[str, dict[int, dict[str, Any]]] = {}

    for label in labels:
        source_predictions = prediction_files[label]
        by_index[label] = prediction_by_index(records, source_predictions)
        expected_keys = {record_key(record) for record in records}
        extra_keys = len(set(source_predictions) - expected_keys)
        reports[label] = {
            "prediction_file": next(spec.split("=", 1)[1] for spec in args.predictions
                                     if spec.split("=", 1)[0] == label),
            "splits": {
                split: split_report(
                    split, records, bags, records_by_index, by_index[label], extra_keys
                )
                for split in SPLITS
            },
        }

    comparisons = {}
    if len(labels) >= 2:
        baseline = labels[0]
        for candidate in labels[1:]:
            comparisons[f"{baseline}_vs_{candidate}"] = {
                split: comparison_report(
                    baseline, candidate, split, records, bags, records_by_index,
                    by_index[baseline], by_index[candidate],
                )
                for split in SPLITS
            }

    result = {
        "schema": "fieldnn-iter6-field-evaluation-1",
        "data": str(args.data.resolve()),
        "matching": {
            "tolerances_m": list(TOLERANCES_M),
            "assignment": "maximum one-to-one match count, then minimum total absolute distance error",
            "bag_best": "best single sweep per bag at each tolerance; do not OR hits across sweeps",
            "group_macro": "deduplicate positions within each independent group before macro averaging",
            "field_precision_recall": None,
            "candidate_semantics": "unmatched candidates are unverified; unknown curves are not negatives",
        },
        "models": reports,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
