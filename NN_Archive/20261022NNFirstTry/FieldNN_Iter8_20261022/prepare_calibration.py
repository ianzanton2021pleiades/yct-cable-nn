"""Cache raw Algorithm 1 responses from source FDR CalibrationData tables."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from tqdm import tqdm


sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
ITER3_DIR = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_Iter3_20261022"
ITER4_DIR = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_Iter4_20261022"
ITER6_DIR = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_Iter6_20261022"
CORE16_DIR = PROJECT_ROOT / "DifferentAgent" / "CodeX" / "Core16_Field"
MANIFEST_PATH = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_20261022" / "output" / "real_manifest.jsonl"
CALIBRATION_INDEX_PATH = CORE16_DIR / "output" / "calibrations.csv"
CALIBRATION_SOURCE_LOG = Path(r"E:\CableNNDataSet\CodeX\FieldNN_Iter4_20261022\calibration_sources_v2.jsonl")
SOURCE_ROOT = Path(r"E:\20261010截至的全部fdr文件")
RULES_PATH = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_20261022" / "real_group_rules.json"
ITER4_CONFIG_PATH = ITER4_DIR / "config.json"
ITER4_PREPARE_PATH = ITER4_DIR / "prepare.py"
SPLITS = ("train", "val", "test")

sys.path.insert(0, str(CORE16_DIR))
from analyze_corpus import load_reader_module  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def norm_path(value: str) -> str:
    return str(value).replace("\\", "/").strip("/").casefold()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_calibration_index(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = norm_path(str(row.get("source_relative_path") or ""))
            if not key:
                continue
            current = result.get(key)
            point_count = int(float(row.get("row_count") or 0))
            if current is None or point_count > int(current.get("row_count") or 0):
                result[key] = {**row, "row_count": point_count}
    return result


def build_source_rows(
    manifest: list[dict[str, Any]],
    calibration_index: dict[str, dict[str, Any]],
    field_prepare,
    iter4_source_log: list[dict[str, Any]],
    frequency_band: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    case_to_group, group_rows = field_prepare.build_groups(manifest, RULES_PATH)
    split_for_group, forced_reason = field_prepare.assign_splits(
        manifest, case_to_group, group_rows, iter4_source_log
    )

    manifest_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest:
        manifest_by_source[norm_path(str(row["source_relative_path"]))].append(row)

    source_rows = []
    excluded_sources = []
    nonempty_sources = 0
    for source_key in sorted(set(manifest_by_source) & set(calibration_index)):
        index_record = calibration_index[source_key]
        if int(index_record["row_count"]) <= 0:
            excluded_sources.append({"source_relative_path": index_record["source_relative_path"],
                                     "reason": "empty_calibration_table"})
            continue
        nonempty_sources += 1
        unique_count = (int(float(index_record["finite_point_count"]))
                        - int(float(index_record["duplicate_frequency_count"])))
        if unique_count < 3:
            excluded_sources.append({"source_relative_path": index_record["source_relative_path"],
                                     "reason": "insufficient_distinct_frequencies",
                                     "distinct_frequency_count": unique_count,
                                     "table_row_count": int(index_record["row_count"]),
                                     "frequency_min_hz": float(index_record["frequency_min_hz"]),
                                     "frequency_max_hz": float(index_record["frequency_max_hz"])})
            continue
        low, high = float(index_record["frequency_min_hz"]), float(index_record["frequency_max_hz"])
        if low > float(frequency_band["start_hz"]) or high < float(frequency_band["stop_hz"]):
            excluded_sources.append({"source_relative_path": index_record["source_relative_path"],
                                     "reason": "calibration_band_not_covered",
                                     "frequency_min_hz": low, "frequency_max_hz": high})
            continue
        manifest_rows = manifest_by_source[source_key]
        cases = sorted({
            str(row.get("source_case_group") or row.get("case_group") or "")
            for row in manifest_rows
        })
        groups = sorted({case_to_group[case] for case in cases if case in case_to_group})
        if len(groups) != 1:
            raise ValueError(f"Source {source_key} maps to {len(groups)} connected groups")
        group = groups[0]
        split = split_for_group[group]
        paths = {str(row["source_relative_path"]) for row in manifest_rows}
        if len(paths) != 1:
            raise ValueError(f"Normalized source key has multiple manifest spellings: {source_key}")
        source_relative_path = next(iter(paths))
        cable_groups = sorted({
            str(row.get("cable_group") or "") for row in manifest_rows if row.get("cable_group")
        })
        source_statuses = sorted({
            str(row.get("development_status") or "") for row in manifest_rows
        })
        source_rows.append({
            "source_relative_path": source_relative_path,
            "source_key": source_key,
            "source_case_group": cases[0] if cases else "",
            "case_groups": cases,
            "cable_group": cable_groups[0] if len(cable_groups) == 1 else cable_groups,
            "group": group,
            "split": split,
            "development_status": source_statuses,
            "calibration_index_row_count": int(index_record["row_count"]),
            "split_forced_reason": forced_reason.get(group),
        })

    split_group_counts = {
        split: len({group for group, value in split_for_group.items() if value == split})
        for split in SPLITS
    }
    with CALIBRATION_INDEX_PATH.open("r", encoding="utf-8-sig", newline="") as stream:
        calibration_index_rows = sum(1 for _ in csv.DictReader(stream))
    audit = {
        "manifest_rows": len(manifest),
        "manifest_unique_sources": len(manifest_by_source),
        "calibration_index_rows": calibration_index_rows,
        "calibration_index_unique_sources": len(calibration_index),
        "manifest_sources_with_calibration_rows": nonempty_sources,
        "usable_calibration_sources": len(source_rows),
        "excluded_sources": excluded_sources,
        "connected_groups_by_split": split_group_counts,
        "split_assignment_reuse": "Iter6 prepare_field.build_groups and assign_splits over the full manifest, using the Iter4 calibration_sources_v2 log",
    }
    return source_rows, audit


def decode_calibration(
    record: dict[str, Any], cfg: dict[str, Any], distance: np.ndarray, fdr_reader,
    raw_root: Path, raw_function,
) -> tuple[int, np.ndarray, np.ndarray, dict[str, Any]]:
    source_path = raw_root.joinpath(*PurePosixPath(record["source_relative_path"].replace("\\", "/")).parts)
    plaintext, _ = fdr_reader.FdrCrypto().decrypt(source_path.read_bytes())
    with sqlite3.connect(":memory:") as connection:
        connection.deserialize(plaintext)
        tables = {
            str(row[0])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "CalibrationData" not in tables:
            raise ValueError(f"CalibrationData table is missing: {record['source_relative_path']}")
        rows = connection.execute(
            "SELECT Freq, Real, Image FROM CalibrationData ORDER BY Freq"
        ).fetchall()

    raw = np.asarray(rows, dtype=np.float64)
    raw = raw[np.isfinite(raw).all(axis=1)]
    order = np.argsort(raw[:, 0], kind="stable")
    raw = raw[order]
    _, first = np.unique(raw[:, 0], return_index=True)
    raw = raw[np.sort(first)]
    frequency = raw[:, 0]
    s11 = raw[:, 1] + 1j * raw[:, 2]
    impulse, step, native_dx = raw_function(frequency, s11, cfg, distance)
    metadata = {
        "calibration_data_table_rows": int(len(rows)),
        "finite_rows": int(np.isfinite(np.asarray(rows, dtype=np.float64)).all(axis=1).sum()),
        "unique_frequency_rows": int(raw.shape[0]),
        "frequency_min_hz": float(frequency[0]),
        "frequency_max_hz": float(frequency[-1]),
        "algorithm1_native_distance_step_m": float(native_dx),
    }
    return int(record["index"]), impulse, step, metadata


def make_record(source: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "index": index,
        "source_relative_path": source["source_relative_path"],
        "source_case_group": source["source_case_group"],
        "case_groups": source["case_groups"],
        "cable_group": source["cable_group"],
        "group": source["group"],
        "split": source["split"],
        "development_status": source["development_status"],
        "calibration_index_row_count": source["calibration_index_row_count"],
        "split_forced_reason": source["split_forced_reason"],
        "sample_kind": "calibration_short_no_cable_fixture",
        "known_semantics": "CalibrationData is a fixture short-circuit measurement with no tested cable connected; user-confirmed.",
        "label_status": "fixture_nuisance_only",
        "event_positions_m": [],
        "defect_intervals_m": [],
        "event_negative_from_m": 30.0,
        "interval_negative_from_m": 30.0,
        "terminal_position_m": -1.0,
        "terminal_label": -1,
        "terminal_label_policy": "ignore the terminal head everywhere; no remote cable end is present",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="Exact output directory; existing unrelated files are preserved")
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--smoke", action="store_true",
                        help="After full group split, keep at most two source files per split")
    args = parser.parse_args()

    field_prepare = load_module(ITER6_DIR / "prepare_field.py", "iter6_prepare_field_for_calibration")
    iter4_prepare = load_module(ITER4_PREPARE_PATH, "iter4_prepare_for_calibration")
    iter4_config = json.loads(ITER4_CONFIG_PATH.read_text(encoding="utf-8"))
    manifest = read_jsonl(MANIFEST_PATH)
    calibration_sources = read_jsonl(CALIBRATION_SOURCE_LOG)
    calibration_index = load_calibration_index(CALIBRATION_INDEX_PATH)
    source_rows, audit = build_source_rows(
        manifest, calibration_index, field_prepare, calibration_sources, iter4_config["frequency"]
    )

    full_split_counts = Counter(row["split"] for row in source_rows)
    if args.smoke:
        selected = []
        for split in SPLITS:
            selected.extend([row for row in source_rows if row["split"] == split][:2])
        source_rows = selected

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = [make_record(row, index) for index, row in enumerate(source_rows)]
    points = int(iter4_config["response"]["points"])
    distance_max_m = float(iter4_config["response"]["distance_max_m"])
    distance = np.linspace(0.0, distance_max_m, points, dtype=np.float64)
    impulses = np.empty((len(records), points), dtype=np.float32)
    steps = np.empty_like(impulses)

    fdr_reader = load_reader_module()
    futures = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for record, source in zip(records, source_rows):
            future = pool.submit(
                decode_calibration, record, iter4_config, distance, fdr_reader,
                SOURCE_ROOT, iter4_prepare._raw_responses,
            )
            futures[future] = record
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="CalibrationData -> full-axis Algorithm 1", unit="FDR"):
            index, impulse, step, extraction = future.result()
            impulses[index] = impulse
            steps[index] = step
            records[index].update(extraction)

    np.save(output / "calibration_raw_imp.npy", impulses)
    np.save(output / "calibration_raw_step.npy", steps)
    with (output / "calibration_records.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")

    split_counts = Counter(record["split"] for record in records)
    group_counts = {
        split: len({record["group"] for record in records if record["split"] == split})
        for split in SPLITS
    }
    source_counts_by_split = {
        split: sum(1 for record in records if record["split"] == split)
        for split in SPLITS
    }
    calibration_config = {
        "schema": "fieldnn-iter8-calibration-raw-config-1",
        "dataset_kind": "fixture short-circuit, no tested cable connected",
        "semantics_source": "user-confirmed CalibrationData meaning",
        "source_root_read_only": str(SOURCE_ROOT),
        "manifest": str(MANIFEST_PATH),
        "calibration_index": str(CALIBRATION_INDEX_PATH),
        "split_source_log": str(CALIBRATION_SOURCE_LOG),
        "frequency": {
            "start_hz": float(iter4_config["frequency"]["start_hz"]),
            "stop_hz": float(iter4_config["frequency"]["stop_hz"]),
            "input_grid": "original CalibrationData frequencies in-band; Iter4._raw_responses inserts band endpoints by interpolation",
        },
        "response": iter4_config["response"],
        "transform": "Iter4 prepare._raw_responses / Algorithm 1, unmodified full response axis; no plateau subtraction, near-field taper, or truncation",
        "outputs": {
            "impulse": "calibration_raw_imp.npy [N, points], impulse_raw",
            "step": "calibration_raw_step.npy [N, points], step_rho",
            "records": "calibration_records.jsonl",
        },
        "label_policy": {
            "event_negative_from_m": 30.0,
            "interval_negative_from_m": 30.0,
            "terminal_label": -1,
            "terminal_policy": "ignore the terminal head across the full grid",
            "scope": "fixture nuisance response only; not a healthy-cable sample and not terminal truth",
        },
    }
    write_json(output / "calibration_config.json", calibration_config)
    summary = {
        "schema": "fieldnn-iter8-calibration-raw-summary-1",
        "status": "complete",
        "mode": "smoke" if args.smoke else "full",
        "output": str(output),
        "workers": args.workers,
        "split_audit": audit,
        "full_manifest_intersection_sources_by_split": dict(full_split_counts),
        "output_sources_by_split": dict(split_counts),
        "connected_groups_in_output_by_split": group_counts,
        "output_source_count": len(records),
        "raw_array_shape": list(impulses.shape),
        "raw_dtype": str(impulses.dtype),
        "label_policy": calibration_config["label_policy"],
        "source_records": [
            {
                "index": row["index"],
                "source_relative_path": row["source_relative_path"],
                "group": row["group"],
                "split": row["split"],
            }
            for row in records
        ],
    }
    write_json(output / "calibration_summary.json", summary)
    print(json.dumps({
        "output": str(output),
        "sources": len(records),
        "excluded_sources": audit["excluded_sources"],
        "sources_by_split": dict(split_counts),
        "groups_by_split": group_counts,
        "shape": list(impulses.shape),
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
