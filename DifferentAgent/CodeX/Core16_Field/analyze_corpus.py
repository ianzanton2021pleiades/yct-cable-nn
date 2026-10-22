from __future__ import annotations

import argparse
import csv
import importlib.machinery
import importlib.util
import json
import math
import re
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_ROOT = Path(r"E:\20261010截至的全部fdr文件")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output"
CONVERTER_PATH = PROJECT_ROOT / "REF" / "FDR数据文件快速转CSV-V2极速版.pyw"
BANDS_HZ = (9_000.0, 100_000.0, 1_000_000.0, 5_000_000.0, 10_000_000.0,
            50_000_000.0, 100_000_000.0, 200_000_000.0, 1_000_000_000.0)
BAND_LABELS = tuple(f"{int(a)}_{int(b)}Hz" for a, b in zip(BANDS_HZ[:-1], BANDS_HZ[1:]))
UNCURATED_CASE_GROUPS = {"未筛选", "其他没有标注参考的数据"}
TARGET_SIGNAL_MARKERS = (
    ("1500m", "浸水试验"),
    ("2026-7-20", "两段电缆"),
    ("2026-9-14", "从接头开始测"),
)


def load_reader_module():
    loader = importlib.machinery.SourceFileLoader("fdr_reader_for_corpus", str(CONVERTER_PATH))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


def json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    return str(value).strip().casefold() in {"true", "1", "yes", "y"}


def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def table_columns(con: sqlite3.Connection, table: str) -> list[str]:
    return [d[0] for d in con.execute(f"SELECT * FROM {qident(table)} LIMIT 0").description]


def table_records(con: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    cur = con.execute(f"SELECT * FROM {qident(table)}")
    names = [d[0] for d in cur.description]
    return [{name: json_value(value) for name, value in zip(names, row)} for row in cur.fetchall()]


def case_classification(case_group: str) -> tuple[str, str]:
    if case_group in UNCURATED_CASE_GROUPS:
        return "uncurated", "top_level_group_is_unfiltered_or_unlabeled_reference"
    return "curated", "top_level_archived_case_directory; diagnosis_not_inferred"


def is_target_signal_group(case_group: str) -> bool:
    text = case_group.casefold()
    return any(all(marker.casefold() in text for marker in markers) for markers in TARGET_SIGNAL_MARKERS)


def quantile_record(values: list[float] | np.ndarray) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "q05": None, "q50": None, "q95": None, "max": None}
    return {
        "count": int(arr.size),
        "q05": float(np.quantile(arr, 0.05)),
        "q50": float(np.quantile(arr, 0.50)),
        "q95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }


def curve_statistics(rows: list[tuple[Any, Any, Any]]) -> dict[str, Any]:
    data = np.asarray(rows, dtype=np.float64)
    if data.size == 0:
        return {
            "row_count": 0, "finite_point_count": 0, "nonfinite_count": 0,
            "frequency_first_hz": None, "frequency_last_hz": None,
            "frequency_min_hz": None, "frequency_max_hz": None,
            "frequency_median_diff_hz": None, "frequency_median_ratio": None,
            "frequency_linear_grid": None, "frequency_log_grid": None,
            "frequency_strictly_increasing": None, "duplicate_frequency_count": 0,
            "s11_abs_q05": None, "s11_abs_q50": None, "s11_abs_q95": None,
            "s11_abs_max": None, "s11_abs_gt_1_01_ratio": None,
            "bands": {label: {"point_count": 0, "s11_abs_median": None} for label in BAND_LABELS},
        }

    freq, real, imag = data[:, 0], data[:, 1], data[:, 2]
    finite_rows = np.isfinite(freq) & np.isfinite(real) & np.isfinite(imag)
    finite_frequency = np.isfinite(freq)
    magnitude = np.abs(real[finite_rows] + 1j * imag[finite_rows])
    sorted_freq = np.sort(freq[finite_frequency])
    unique_freq = np.unique(sorted_freq)
    diffs = np.diff(unique_freq)
    positive_diffs = diffs[diffs > 0]
    ratios = unique_freq[1:] / unique_freq[:-1] if unique_freq.size > 1 and np.all(unique_freq[:-1] > 0) else np.array([])
    med_diff = float(np.median(positive_diffs)) if positive_diffs.size else None
    med_ratio = float(np.median(ratios)) if ratios.size else None
    linear_grid = bool(np.allclose(diffs, med_diff, rtol=1e-4, atol=max(1e-9, abs(med_diff or 0.0) * 1e-8))) if diffs.size else None
    log_grid = bool(np.allclose(ratios, med_ratio, rtol=1e-4, atol=1e-12)) if ratios.size else None
    original_sorted = freq[finite_frequency]
    strictly_increasing = bool(original_sorted.size < 2 or np.all(np.diff(original_sorted) > 0))
    stats: dict[str, Any] = {
        "row_count": int(data.shape[0]),
        "finite_point_count": int(np.count_nonzero(finite_rows)),
        "nonfinite_count": int(data.shape[0] - np.count_nonzero(finite_rows)),
        "frequency_first_hz": float(freq[0]) if np.isfinite(freq[0]) else None,
        "frequency_last_hz": float(freq[-1]) if np.isfinite(freq[-1]) else None,
        "frequency_min_hz": float(np.min(freq[finite_frequency])) if np.any(finite_frequency) else None,
        "frequency_max_hz": float(np.max(freq[finite_frequency])) if np.any(finite_frequency) else None,
        "frequency_median_diff_hz": med_diff,
        "frequency_median_ratio": med_ratio,
        "frequency_linear_grid": linear_grid,
        "frequency_log_grid": log_grid,
        "frequency_strictly_increasing": strictly_increasing,
        "duplicate_frequency_count": int(finite_frequency.sum() - unique_freq.size),
        "s11_abs_q05": float(np.quantile(magnitude, 0.05)) if magnitude.size else None,
        "s11_abs_q50": float(np.quantile(magnitude, 0.50)) if magnitude.size else None,
        "s11_abs_q95": float(np.quantile(magnitude, 0.95)) if magnitude.size else None,
        "s11_abs_max": float(np.max(magnitude)) if magnitude.size else None,
        "s11_abs_gt_1_01_ratio": float(np.count_nonzero(magnitude > 1.01) / magnitude.size) if magnitude.size else None,
        "bands": {},
    }
    valid_f = freq[finite_rows]
    for index, label in enumerate(BAND_LABELS):
        low, high = BANDS_HZ[index], BANDS_HZ[index + 1]
        mask = (valid_f >= low) & ((valid_f < high) if index < len(BAND_LABELS) - 1 else (valid_f <= high))
        band_mag = magnitude[mask]
        stats["bands"][label] = {
            "point_count": int(band_mag.size),
            "s11_abs_median": float(np.median(band_mag)) if band_mag.size else None,
        }
    return stats


def signal_filename(index: int, source_path: Path, sweep_id: int) -> str:
    stem = re.sub(r"[^\w.-]+", "_", source_path.stem, flags=re.UNICODE).strip("._") or "fdr"
    return f"{index:05d}_{stem}_s{sweep_id}.npz"


def write_signal(path: Path, freq: np.ndarray, real: np.ndarray, imag: np.ndarray,
                 response_tables: dict[str, tuple[np.ndarray, np.ndarray]], metadata: dict[str, str]) -> None:
    payload: dict[str, Any] = {
        "frequency_hz": freq.astype(np.float64, copy=False),
        "s11_real": real.astype(np.float64, copy=False),
        "s11_imag": imag.astype(np.float64, copy=False),
        "s11_complex": real.astype(np.float64, copy=False) + 1j * imag.astype(np.float64, copy=False),
    }
    for name, (distance, rho) in response_tables.items():
        payload[f"vendor_{name}_distance"] = distance
        payload[f"vendor_{name}_rho"] = rho
    for name, value in metadata.items():
        payload[name] = np.asarray(value, dtype=np.str_)
    np.savez_compressed(path, **payload)


def process_file(index: int, total: int, source_path: Path, source_root: Path, output_dir: Path,
                 preserve_all: bool, preserve_first: bool, reader_module: Any) -> dict[str, Any]:
    started = time.perf_counter()
    relative = source_path.relative_to(source_root).as_posix()
    case_group = source_path.relative_to(source_root).parts[0]
    classification, classification_reason = case_classification(case_group)
    raw = source_path.read_bytes()
    plaintext, page_size = reader_module.FdrCrypto().decrypt(raw)
    con = sqlite3.connect(":memory:")
    con.deserialize(plaintext)
    try:
        tables = {str(row[0]) for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        setting_rows = table_records(con, "MeasureSetting") if "MeasureSetting" in tables else []
        nameplate_columns = table_columns(con, "Nameplate") if "Nameplate" in tables else []
        nameplate_rows = table_records(con, "Nameplate") if "Nameplate" in tables else []
        inventory: dict[str, Any] = {
            "source_relative_path": relative,
            "case_group": case_group,
            "classification": classification,
            "classification_reason": classification_reason,
            "file_size_bytes": len(raw),
            "sqlite_page_size": page_size,
            "tables": sorted(tables),
            "measure_setting": setting_rows[0] if setting_rows else None,
            "nameplate_fields": nameplate_columns,
            "nameplate_rows": nameplate_rows,
            "sweeps": [],
        }
        measurements: list[dict[str, Any]] = []
        if "CalibrationData" in tables:
            calibration_rows = con.execute("SELECT Freq, Real, Image FROM CalibrationData").fetchall()
            calibration_stats = curve_statistics(calibration_rows)
            calibration_entry: dict[str, Any] = {
                "source_relative_path": relative,
                "case_group": case_group,
                "row_count": calibration_stats["row_count"],
                **{key: value for key, value in calibration_stats.items() if key != "bands"},
            }
            for label, band in calibration_stats["bands"].items():
                calibration_entry[f"band_{label}_point_count"] = band["point_count"]
                calibration_entry[f"band_{label}_s11_abs_median"] = band["s11_abs_median"]
        else:
            calibration_entry = {
                "source_relative_path": relative, "case_group": case_group,
                "row_count": 0, "finite_point_count": 0,
            }

        if "Sweeps" in tables:
            sweep_cursor = con.execute("SELECT rowid, * FROM Sweeps ORDER BY rowid")
            sweep_columns = [d[0] for d in sweep_cursor.description]
            for sweep_row in sweep_cursor.fetchall():
                sweep_id = int(sweep_row[0])
                sweep = dict(zip(sweep_columns[1:], sweep_row[1:]))
                table_name = f"MeasureData_{sweep_id}"
                table_exists = table_name in tables
                rows = con.execute(f"SELECT Freq, Real, Image FROM {qident(table_name)}").fetchall() if table_exists else []
                stats = curve_statistics(rows)
                measured = as_bool(sweep.get("IsMeasured"))
                valid_curve = bool(stats["finite_point_count"])
                response_tables: dict[str, tuple[np.ndarray, np.ndarray]] = {}
                freq_arr = real_arr = imag_arr = np.array([], dtype=np.float64)
                if rows:
                    arr = np.asarray(rows, dtype=np.float64)
                    freq_arr, real_arr, imag_arr = arr[:, 0], arr[:, 1], arr[:, 2]
                should_save = preserve_all or (preserve_first and valid_curve and not any(x.get("_preserved_first") for x in inventory["sweeps"]))
                signal_relpath = None
                if should_save and rows:
                    for kind, prefix in (("impulse", "ImpulseResponse_"), ("step", "StepResponse_")):
                        response_table = f"{prefix}{sweep_id}"
                        if response_table in tables:
                            response = con.execute(f"SELECT Distance, Rho FROM {qident(response_table)}").fetchall()
                            response_arr = np.asarray(response, dtype=np.float64) if response else np.empty((0, 2), dtype=np.float64)
                            response_tables[kind] = (response_arr[:, 0], response_arr[:, 1])
                    signal_name = signal_filename(index, source_path, sweep_id)
                    signal_path = output_dir / "signals" / signal_name
                    write_signal(signal_path, freq_arr, real_arr, imag_arr, response_tables, {
                        "source_relative_path": relative,
                        "case_group": case_group,
                        "sweep_name": str(sweep.get("SweepName") or ""),
                        "measure_table": table_name,
                        "classification": classification,
                    })
                    signal_relpath = signal_path.relative_to(output_dir).as_posix()

                row: dict[str, Any] = {
                    "source_relative_path": relative,
                    "case_group": case_group,
                    "classification": classification,
                    "classification_reason": classification_reason,
                    "sweep_rowid": sweep_id,
                    "sweep_name": json_value(sweep.get("SweepName")),
                    "comments": json_value(sweep.get("Comments")),
                    "remark": json_value(sweep.get("Remark")),
                    "is_measured": measured,
                    "measure_table": table_name,
                    "measure_table_exists": table_exists,
                    "points_setting": json_value(sweep.get("Points")),
                    "frequency_start_setting_hz": json_value(sweep.get("Frequency_Start")),
                    "frequency_stop_setting_hz": json_value(sweep.get("Frequency_Stop")),
                    "wave_velocity_raw": json_value(sweep.get("Wave_Velocity")),
                    "x_limit_min_raw": json_value(sweep.get("X_Limit_Min")),
                    "x_limit_max_raw": json_value(sweep.get("X_Limit_Max")),
                    "x_axis_offset_raw": json_value(sweep.get("X_Axis_Offset")),
                    "table_status": "missing" if not table_exists else ("empty" if not rows else "present"),
                    "valid_curve": valid_curve,
                    "signal_npz": signal_relpath,
                    **{key: value for key, value in stats.items() if key != "bands"},
                }
                for label, band in stats["bands"].items():
                    row[f"band_{label}_point_count"] = band["point_count"]
                    row[f"band_{label}_s11_abs_median"] = band["s11_abs_median"]
                measurements.append(row)
                inv_sweep = {key: json_value(value) for key, value in sweep.items()}
                inv_sweep.update({"sweep_rowid": sweep_id, "measure_table": table_name,
                                  "measure_table_exists": table_exists, "table_status": row["table_status"],
                                  "is_measured_bool": measured, "stats": stats, "signal_npz": signal_relpath})
                if preserve_first and signal_relpath:
                    inv_sweep["_preserved_first"] = True
                inventory["sweeps"].append(inv_sweep)

        inventory["sweep_count"] = len(measurements)
        inventory["measured_sweep_count"] = sum(row["is_measured"] is True for row in measurements)
        inventory["unmeasured_sweep_count"] = sum(row["is_measured"] is False for row in measurements)
        inventory["missing_measure_table_count"] = sum(row["table_status"] == "missing" for row in measurements)
        inventory["empty_measure_table_count"] = sum(row["table_status"] == "empty" for row in measurements)
        inventory["valid_curve_count"] = sum(row["valid_curve"] for row in measurements)
        for sweep in inventory["sweeps"]:
            sweep.pop("_preserved_first", None)
        elapsed = time.perf_counter() - started
        print(f"[{index}/{total}] OK {relative} sweeps={len(measurements)} elapsed={elapsed:.2f}s", flush=True)
        return {"inventory": inventory, "measurements": measurements, "calibration": calibration_entry,
                "elapsed_seconds": elapsed, "error": None}
    finally:
        con.close()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def add_band_summary(destination: dict[str, Any], selected_rows: list[dict[str, Any]]) -> None:
    result: dict[str, Any] = {}
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in selected_rows:
        by_case.setdefault(str(row["case_group"]), []).append(row)
    for label in BAND_LABELS:
        curve_medians = [float(row[f"band_{label}_s11_abs_median"]) for row in selected_rows
                         if row.get(f"band_{label}_s11_abs_median") is not None]
        case_medians = []
        for case_rows in by_case.values():
            values = [float(row[f"band_{label}_s11_abs_median"]) for row in case_rows
                      if row.get(f"band_{label}_s11_abs_median") is not None]
            if values:
                case_medians.append(float(np.median(values)))
        point_count = sum(int(row.get(f"band_{label}_point_count") or 0) for row in selected_rows)
        result[label] = {
            "point_count": point_count,
            "curve_equal_weighted_curve_median": quantile_record(curve_medians),
            "case_equal_weighted_case_median": quantile_record(case_medians),
        }
    destination["frequency_bands_hz"] = result


def category_summary(category: str, measurements: list[dict[str, Any]], inventories: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in measurements if row["classification"] == category and row["valid_curve"]]
    selected_inv = [item for item in inventories if item["classification"] == category]
    medians = [float(row["s11_abs_q50"]) for row in selected if row.get("s11_abs_q50") is not None]
    by_case: dict[str, list[float]] = {}
    for row in selected:
        by_case.setdefault(str(row["case_group"]), []).append(float(row["s11_abs_q50"]))
    case_medians = [float(np.median(values)) for values in by_case.values() if values]
    result: dict[str, Any] = {
        "classification": category,
        "file_count": len(selected_inv),
        "top_level_case_group_count": len({item["case_group"] for item in selected_inv}),
        "sweep_count": sum(item["sweep_count"] for item in selected_inv),
        "measured_sweep_count": sum(item["measured_sweep_count"] for item in selected_inv),
        "unmeasured_sweep_count": sum(item["unmeasured_sweep_count"] for item in selected_inv),
        "missing_measure_table_count": sum(item["missing_measure_table_count"] for item in selected_inv),
        "empty_measure_table_count": sum(item["empty_measure_table_count"] for item in selected_inv),
        "valid_curve_count": len(selected),
        "physical_independent_cable_count": None,
        "physical_independent_cable_count_note": "not inferred from file or case counts",
        "curve_equal_weighted_curve_median": quantile_record(medians),
        "case_equal_weighted_case_median": quantile_record(case_medians),
    }
    add_band_summary(result, selected)
    return result


def case_metrics_rows(measurements: list[dict[str, Any]], inventories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category in ("curated", "uncurated"):
        selected_inventory = [i for i in inventories if i["classification"] == category]
        selected_measurements = [m for m in measurements if m["classification"] == category and m["valid_curve"]]
        grouped: dict[str, list[dict[str, Any]]] = {}
        for m in selected_measurements:
            grouped.setdefault(str(m["case_group"]), []).append(m)
        for case_group, case_rows in sorted(grouped.items()):
            values = [float(m["s11_abs_q50"]) for m in case_rows if m.get("s11_abs_q50") is not None]
            record = {
                "scope": "top_level_case",
                "classification": category,
                "case_group": case_group,
                "file_count": sum(i["case_group"] == case_group for i in selected_inventory),
                "sweep_count": sum(i["sweep_count"] for i in selected_inventory if i["case_group"] == case_group),
                "valid_curve_count": len(case_rows),
                "curve_median_q05": quantile_record(values)["q05"],
                "curve_median_q50": quantile_record(values)["q50"],
                "curve_median_q95": quantile_record(values)["q95"],
                "curve_median_max": quantile_record(values)["max"],
                "case_median": float(np.median(values)) if values else None,
            }
            for label in BAND_LABELS:
                band_values = [float(m[f"band_{label}_s11_abs_median"]) for m in case_rows
                               if m.get(f"band_{label}_s11_abs_median") is not None]
                record[f"band_{label}_curve_median"] = float(np.median(band_values)) if band_values else None
            rows.append(record)
        class_summary = category_summary(category, measurements, inventories)
        curve_distribution = class_summary["curve_equal_weighted_curve_median"]
        case_distribution = class_summary["case_equal_weighted_case_median"]
        summary_row = {
            "scope": "classification",
            "classification": category,
            "case_group": "",
            "file_count": class_summary["file_count"],
            "sweep_count": class_summary["sweep_count"],
            "valid_curve_count": class_summary["valid_curve_count"],
            "curve_median_q05": curve_distribution["q05"],
            "curve_median_q50": curve_distribution["q50"],
            "curve_median_q95": curve_distribution["q95"],
            "curve_median_max": curve_distribution["max"],
            "case_median": case_distribution["q50"],
        }
        for label, band in class_summary["frequency_bands_hz"].items():
            summary_row[f"band_{label}_curve_median"] = band["curve_equal_weighted_curve_median"]["q50"]
        rows.append(summary_row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="只读统计 FDR 数据库中的测量 S11 与独立校正表")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if not hasattr(sqlite3.Connection, "deserialize"):
        raise RuntimeError("当前 Python sqlite3 不支持内存 deserialize；已停止，不会使用文件回退。")
    args.output.mkdir(parents=True, exist_ok=True)
    signal_dir = args.output / "signals"

    files = sorted((p for p in args.source_root.rglob("*") if p.is_file() and p.suffix.casefold() in {".fdr", ".db8"}
                    and not any("rg58" in part.casefold() or "核级" in part for part in p.relative_to(args.source_root).parts)),
                   key=lambda p: p.relative_to(args.source_root).as_posix().casefold())
    if args.limit is not None:
        files = files[:max(0, args.limit)]

    case_groups = [p.relative_to(args.source_root).parts[0] for p in files]
    curated_files = [p for p in files if case_classification(p.relative_to(args.source_root).parts[0])[0] == "curated"]
    first_five_curated = set(curated_files[:5])
    targeted_files = {p for p in files if is_target_signal_group(p.relative_to(args.source_root).parts[0])}
    preserve_files = first_five_curated | targeted_files
    if preserve_files:
        signal_dir.mkdir(parents=True, exist_ok=True)
        for stale_signal in signal_dir.glob("*.npz"):
            stale_signal.unlink()

    print(f"files={len(files)} workers={args.workers} curated_reference_files={min(5, len(first_five_curated))} "
          f"target_case_files={len(targeted_files)}", flush=True)
    reader_module = load_reader_module()
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(process_file, i, len(files), path, args.source_root, args.output,
                            path in targeted_files, path in first_five_curated, reader_module): path
            for i, path in enumerate(files, 1)
        }
        completed = 0
        for future in as_completed(future_map):
            path = future_map[future]
            completed += 1
            try:
                results.append(future.result())
            except Exception as exc:
                relative = path.relative_to(args.source_root).as_posix()
                case_group = path.relative_to(args.source_root).parts[0]
                errors.append({"source_relative_path": relative, "case_group": case_group,
                               "error_type": type(exc).__name__, "error": str(exc)})
                print(f"[{completed}/{len(files)}] ERROR {relative}: {type(exc).__name__}: {exc}", flush=True)

    results.sort(key=lambda item: item["inventory"]["source_relative_path"].casefold())
    inventories = [item["inventory"] for item in results]
    measurements = [row for item in results for row in item["measurements"]]
    calibrations = [item["calibration"] for item in results]
    inventories.extend({
        "source_relative_path": err["source_relative_path"],
        "case_group": err["case_group"],
        "classification": case_classification(err["case_group"])[0],
        "classification_reason": case_classification(err["case_group"])[1],
        "status": "error",
        "error_type": err["error_type"],
        "error": err["error"],
    } for err in errors)
    inventories.sort(key=lambda item: item["source_relative_path"].casefold())

    measurement_fields = [
        "source_relative_path", "case_group", "classification", "classification_reason",
        "sweep_rowid", "sweep_name", "comments", "remark", "is_measured", "measure_table",
        "measure_table_exists", "table_status", "valid_curve", "signal_npz", "points_setting",
        "frequency_start_setting_hz", "frequency_stop_setting_hz", "wave_velocity_raw",
        "x_limit_min_raw", "x_limit_max_raw", "x_axis_offset_raw", "row_count", "finite_point_count",
        "nonfinite_count", "frequency_first_hz", "frequency_last_hz", "frequency_min_hz", "frequency_max_hz",
        "frequency_median_diff_hz", "frequency_median_ratio", "frequency_linear_grid", "frequency_log_grid",
        "frequency_strictly_increasing", "duplicate_frequency_count", "s11_abs_q05", "s11_abs_q50",
        "s11_abs_q95", "s11_abs_max", "s11_abs_gt_1_01_ratio",
    ]
    for label in BAND_LABELS:
        measurement_fields.extend((f"band_{label}_point_count", f"band_{label}_s11_abs_median"))
    calibration_fields = [key for key in calibrations[0].keys()] if calibrations else ["source_relative_path", "case_group", "row_count"]
    write_csv(args.output / "measurements.csv", measurements, measurement_fields)
    write_csv(args.output / "calibrations.csv", calibrations, calibration_fields)
    write_csv(args.output / "case_metrics.csv", case_metrics_rows(measurements, [i for i in inventories if i.get("status") != "error"]),
              ["scope", "classification", "case_group", "file_count", "sweep_count", "valid_curve_count",
               "curve_median_q05", "curve_median_q50", "curve_median_q95", "curve_median_max", "case_median"] +
              [f"band_{label}_curve_median" for label in BAND_LABELS])

    with (args.output / "corpus_inventory.json").open("w", encoding="utf-8") as stream:
        json.dump(inventories, stream, ensure_ascii=False, indent=2, allow_nan=False)
    with (args.output / "read_errors.json").open("w", encoding="utf-8") as stream:
        json.dump(errors, stream, ensure_ascii=False, indent=2)

    valid_inventories = [i for i in inventories if i.get("status") != "error"]
    summary = {
        "source_root": str(args.source_root),
        "source_file_count": len(files),
        "read_success_count": len(results),
        "read_error_count": len(errors),
        "measurement_sweep_count": len(measurements),
        "calibration_table_count": sum(c.get("row_count", 0) > 0 for c in calibrations),
        "calibration_point_count": sum(int(c.get("row_count") or 0) for c in calibrations),
        "sweep_counts": {
            "measured": sum(i["measured_sweep_count"] for i in valid_inventories),
            "unmeasured": sum(i["unmeasured_sweep_count"] for i in valid_inventories),
            "missing_measure_table": sum(i["missing_measure_table_count"] for i in valid_inventories),
            "empty_measure_table": sum(i["empty_measure_table_count"] for i in valid_inventories),
            "valid_curve": sum(i["valid_curve_count"] for i in valid_inventories),
        },
        "classification_policy": {
            "curated": "all eligible top-level archived case directories except the two explicitly uncurated groups",
            "uncurated_group_names": sorted(UNCURATED_CASE_GROUPS),
            "rule_note": "curated means archived case directory only; no diagnosis or complete label is inferred",
            "physical_independent_cable_count": "not inferred",
        },
        "classification_summaries": {
            category: category_summary(category, measurements, valid_inventories)
            for category in ("curated", "uncurated")
        },
        "signal_exports": {
            "target_case_file_count": len(targeted_files),
            "first_five_curated_reference_file_count": len(first_five_curated),
            "npz_count": len(list(signal_dir.glob("*.npz"))) if signal_dir.exists() else 0,
            "vendor_response_label": "vendor_* arrays are stored instrument results; no algorithm identity is assigned",
        },
        "definitions": {
            "frequency_unit": "Hz",
            "reflection_coefficient": "Gamma = S11_Real + j*S11_Imag",
            "magnitude": "abs(Gamma), linear scale",
            "magnitude_gt_1_01_ratio": "count(abs(Gamma)>1.01) / finite valid point count",
            "nonfinite_count": "rows with any non-finite frequency, real, or imaginary value",
            "duplicate_frequency_count": "finite frequency row count minus unique finite frequency count",
            "frequency_grid_tolerance": "median spacing/ratio allclose rtol=1e-4; linear atol=max(1e-9, median_spacing*1e-8)",
            "frequency_bands": "adjacent edges in frequency_bands_hz; all bins half-open except the final bin includes 1 GHz",
            "curve_equal_weighted": "quantiles across valid curve-level |S11| medians, one value per measurement curve",
            "case_equal_weighted": "median curve median within each top-level case group, then quantiles with one value per case group",
            "coordinate_limit_note": "X_Limit_Min/Max and X_Axis_Offset are stored as source values and are not interpreted as cable length",
            "calibration_note": "CalibrationData is summarized separately and excluded from measurement and case summaries",
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    with (args.output / "corpus_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"DONE files={len(files)} read_ok={len(results)} errors={len(errors)} curves={len(measurements)} "
          f"elapsed={summary['elapsed_seconds']:.1f}s output={args.output}", flush=True)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
