#!/usr/bin/env python3
"""Attach measured curves to the nearest TXT length evidence and summarize scope."""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath


BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "output"
MEASUREMENTS_PATH = OUTPUT / "measurements.csv"
EVIDENCE_PATH = Path(__file__).resolve().parent / "scope_evidence.json"
SCOPED_PATH = OUTPUT / "scope_measurements.csv"
SUMMARY_PATH = OUTPUT / "short_scope_summary.json"

BUCKETS = ("up_to_2500m", "over_2500m", "unknown")
MANUAL_DOC_DIRECTORY = "2023-12-26-无锡蓝安+藕光"
MANUAL_DOC_NOTE = (
    "主代理已从该目录DOCX核对蓝安#1约1070m、蓝安#2约560m、藕光约1020m；"
    "只提供目录级短线范围旁证，不将藕光_new、蓝安431-1100米等文件名精确对齐到线路。"
)


def normalized_path(value: str) -> str:
    return str(value or "").replace("\\", "/").strip("/")


def nearest_assessment(source_relative_path: str, assessments: dict[str, dict]):
    parts = PurePosixPath(normalized_path(source_relative_path)).parts[:-1]
    for depth in range(len(parts), 0, -1):
        candidate = "/".join(parts[:depth])
        if candidate in assessments:
            return assessments[candidate], candidate, depth
    return None, "", 0


def as_bool(value) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes"}


def as_float(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def numeric_summary(values):
    clean = sorted(value for value in values if value is not None and math.isfinite(value))
    if not clean:
        return {"n": 0, "min": None, "q05": None, "q50": None, "q95": None, "max": None}
    return {
        "n": len(clean),
        "min": clean[0],
        "q05": quantile(clean, 0.05),
        "q50": quantile(clean, 0.50),
        "q95": quantile(clean, 0.95),
        "max": clean[-1],
    }


def value_distribution(values):
    counts = Counter(
        format(value, ".12g")
        for value in values
        if value is not None and math.isfinite(value)
    )
    return dict(sorted(counts.items(), key=lambda item: float(item[0])))


def quantile(sorted_values, probability):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def valid_rows(rows):
    return [row for row in rows if as_bool(row.get("valid_curve"))]


def scope_counts(rows):
    result = {}
    for bucket in BUCKETS:
        bucket_rows = [row for row in rows if row["scope_bucket"] == bucket]
        curves = valid_rows(bucket_rows)
        result[bucket] = {
            "measurement_row_count": len(bucket_rows),
            "valid_curve_count": len(curves),
            "source_file_count": len({row["source_relative_path"] for row in curves}),
            "source_directory_count": len(
                {normalized_path(PurePosixPath(normalized_path(row["source_relative_path"])).parent.as_posix()) for row in curves}
            ),
            "matched_evidence_directory_count": len(
                {row["matched_evidence_directory"] for row in curves if row["matched_evidence_directory"]}
            ),
            "top_level_case_count": len({row["top_level_case"] for row in curves if row["top_level_case"]}),
            "valid_curve_provenance_counts": dict(Counter(row["scope_basis"] for row in curves)),
        }
    return result


def short_scope_statistics(rows, bands):
    short_curves = [
        row for row in rows
        if row["scope_bucket"] == "up_to_2500m" and as_bool(row.get("valid_curve"))
    ]

    band_summaries = []
    for lower_hz, upper_hz, median_column in bands:
        by_case = defaultdict(list)
        for row in short_curves:
            value = as_float(row.get(median_column))
            if value is not None:
                by_case[row["top_level_case"]].append(value)
        case_medians = [statistics.median(values) for values in by_case.values()]
        band_summaries.append({
            "frequency_band_hz": [lower_hz, upper_hz],
            "curve_count_with_band_value": sum(len(values) for values in by_case.values()),
            "top_level_case_count": len(case_medians),
            "case_median_abs_s11_quantiles": numeric_summary(case_medians),
        })

    velocity_pairs = set()
    for row in short_curves:
        value = as_float(row.get("wave_velocity_raw"))
        if value is not None:
            velocity_pairs.add((row["source_relative_path"], value))
    velocity_values = [value for _, value in velocity_pairs]
    velocity_files_by_value = Counter(format(value, ".12g") for _, value in velocity_pairs)

    start_actual = [as_float(row.get("frequency_first_hz")) for row in short_curves]
    stop_actual = [as_float(row.get("frequency_last_hz")) for row in short_curves]
    start_setting = [as_float(row.get("frequency_start_setting_hz")) for row in short_curves]
    stop_setting = [as_float(row.get("frequency_stop_setting_hz")) for row in short_curves]
    point_counts = [as_float(row.get("finite_point_count")) for row in short_curves]
    row_counts = [as_float(row.get("row_count")) for row in short_curves]
    median_steps = [as_float(row.get("frequency_median_diff_hz")) for row in short_curves]

    grid_counts = Counter()
    monotonic_counts = Counter()
    s11_gt_curve_count = 0
    s11_gt_source_files = set()
    for row in short_curves:
        linear = as_bool(row.get("frequency_linear_grid"))
        logarithmic = as_bool(row.get("frequency_log_grid"))
        if linear and logarithmic:
            grid_counts["both_linear_and_log_flags"] += 1
        elif linear:
            grid_counts["linear"] += 1
        elif logarithmic:
            grid_counts["logarithmic"] += 1
        else:
            grid_counts["other_or_uneven"] += 1
        monotonic_counts["strictly_increasing" if as_bool(row.get("frequency_strictly_increasing")) else "not_strictly_increasing"] += 1
        ratio = as_float(row.get("s11_abs_gt_1_01_ratio"))
        if ratio is not None and ratio > 0:
            s11_gt_curve_count += 1
            s11_gt_source_files.add(row["source_relative_path"])

    return {
        "description": (
            "描述性参考，不是唯一经验参数估计。只用up_to_2500m有效曲线；"
            "先按顶层case对各频带abs median取中位数，再对case中位数等权计算q05/q50/q95。"
        ),
        "valid_curve_count": len(short_curves),
        "top_level_case_count": len({row["top_level_case"] for row in short_curves}),
        "provenance_counts": dict(Counter(row["scope_basis"] for row in short_curves)),
        "frequency_band_abs_median_by_case_quantiles": band_summaries,
        "wave_velocity_setting": {
            "unit_as_stored": "measurements.csv wave_velocity_raw",
            "unique_source_file_value_pair_count": len(velocity_pairs),
            "unique_source_file_count": len({source for source, _ in velocity_pairs}),
            "distribution_by_value_counting_each_source_file_once": dict(sorted(velocity_files_by_value.items())),
            "quantiles": numeric_summary(velocity_values),
        },
        "frequency_and_grid_distributions": {
            "actual_start_hz": numeric_summary(start_actual),
            "actual_stop_hz": numeric_summary(stop_actual),
            "setting_start_hz": numeric_summary(start_setting),
            "setting_stop_hz": numeric_summary(stop_setting),
            "actual_start_hz_curve_counts": value_distribution(start_actual),
            "actual_stop_hz_curve_counts": value_distribution(stop_actual),
            "setting_start_hz_curve_counts": value_distribution(start_setting),
            "setting_stop_hz_curve_counts": value_distribution(stop_setting),
            "finite_point_count": numeric_summary(point_counts),
            "row_count": numeric_summary(row_counts),
            "finite_point_count_curve_counts": value_distribution(point_counts),
            "row_count_curve_counts": value_distribution(row_counts),
            "median_frequency_step_hz": numeric_summary(median_steps),
            "grid_type_curve_counts": dict(grid_counts),
            "strictly_increasing_curve_counts": dict(monotonic_counts),
        },
        "s11_abs_greater_than_1_01": {
            "curve_count_with_any_point_above_threshold": s11_gt_curve_count,
            "source_file_count": len(s11_gt_source_files),
            "criterion": "measurements.csv s11_abs_gt_1_01_ratio > 0",
        },
    }


def main():
    with MEASUREMENTS_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        input_rows = list(csv.DictReader(handle))
    with MEASUREMENTS_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        fieldnames = next(csv.reader(handle))

    with EVIDENCE_PATH.open("r", encoding="utf-8") as handle:
        evidence = json.load(handle)
    assessment_rows = evidence.get("directory_assessments", [])
    assessments = {
        normalized_path(item["relative_directory"]): item
        for item in assessment_rows
        if item.get("relative_directory")
    }

    scoped_rows = []
    manual_doc_source_count = 0
    for original in input_rows:
        row = dict(original)
        source_path = row.get("source_relative_path", "")
        source_normalized = normalized_path(source_path)
        top_level_case = PurePosixPath(source_normalized).parts[0] if source_normalized else ""
        assessment, matched_directory, matched_depth = nearest_assessment(source_path, assessments)
        raw_bucket = assessment.get("length_bucket") if assessment else None
        basis = "no_matching_evidence_directory"
        bucket = "unknown"
        manual_note = ""

        if top_level_case == MANUAL_DOC_DIRECTORY:
            bucket = "up_to_2500m"
            basis = "manual_doc_source_directory_level_short_range_only"
            manual_note = MANUAL_DOC_NOTE
            manual_doc_source_count += 1
        elif raw_bucket == "up_to_2500m":
            bucket = "up_to_2500m"
            basis = "txt_directory_assessment"
        elif raw_bucket == "over_2500m":
            bucket = "over_2500m"
            basis = "txt_directory_assessment"
        elif raw_bucket == "unknown_or_conflicting":
            bucket = "unknown"
            basis = "txt_unknown_or_conflicting_length_evidence"

        row.update({
            "scope_bucket": bucket,
            "scope_basis": basis,
            "top_level_case": top_level_case,
            "matched_evidence_directory": matched_directory,
            "matched_evidence_depth": matched_depth,
            "evidence_assessment_bucket": raw_bucket or "no_matching_evidence_directory",
            "evidence_bucket_basis": assessment.get("bucket_basis", "") if assessment else "",
            "manual_doc_source": manual_note,
        })
        scoped_rows.append(row)

    output_fields = fieldnames + [
        "scope_bucket",
        "scope_basis",
        "top_level_case",
        "matched_evidence_directory",
        "matched_evidence_depth",
        "evidence_assessment_bucket",
        "evidence_bucket_basis",
        "manual_doc_source",
    ]
    with SCOPED_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(scoped_rows)

    band_columns = []
    for name in fieldnames:
        match = re.fullmatch(r"band_(\d+)_(\d+)Hz_s11_abs_median", name)
        if match:
            band_columns.append((int(match.group(1)), int(match.group(2)), name))
    band_columns.sort()

    summary = {
        "inputs": {
            "measurements_csv": str(MEASUREMENTS_PATH),
            "evidence_records_json": str(EVIDENCE_PATH),
            "measurement_row_count": len(scoped_rows),
            "evidence_directory_assessment_count": len(assessment_rows),
        },
        "scope_policy": (
            "按source_relative_path父目录向上匹配最近的directory_assessments；子目录证据优先于祖先。"
            "TXT长度冲突归unknown。2023无锡只采用主代理DOCX目录级短线旁证，不精确映射到单个线路文件。"
            "整线超过2500m不截取前段作为短线。"
        ),
        "scope_bucket_counts": scope_counts(scoped_rows),
        "manual_doc_source_measurement_rows": manual_doc_source_count,
        "up_to_2500m_descriptive_statistics": short_scope_statistics(scoped_rows, band_columns),
    }
    with SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps({
        "measurement_rows": len(scoped_rows),
        "scope_bucket_counts": {
            key: {
                "valid_curve_count": value["valid_curve_count"],
                "source_file_count": value["source_file_count"],
                "source_directory_count": value["source_directory_count"],
            }
            for key, value in summary["scope_bucket_counts"].items()
        },
        "manual_doc_source_measurement_rows": manual_doc_source_count,
        "outputs": [str(SCOPED_PATH), str(SUMMARY_PATH)],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
