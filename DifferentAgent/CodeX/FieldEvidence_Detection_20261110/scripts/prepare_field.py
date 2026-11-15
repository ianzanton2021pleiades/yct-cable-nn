"""Prepare accepted field FDR groups for the first-round detector."""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parents[2]
DG_IO_DIR = REPO_ROOT / "DifferentAgent" / "Zcode" / "DG" / "dg_v4"
READER_PREPROCESS_DIR = (
    REPO_ROOT / "DifferentAgent" / "CodeX" / "FDR_Reader_Analysis_20261110" / "scripts"
)
sys.path.insert(0, str(DG_IO_DIR))
sys.path.insert(0, str(READER_PREPROCESS_DIR))

from fdr_io import decrypt_fdr  # noqa: E402
from reader_preprocess import movmean  # noqa: E402


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _json_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _read_table(connection: sqlite3.Connection, name: str) -> dict | None:
    cursor = connection.cursor()
    cursor.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    )
    if cursor.fetchone() is None:
        return None
    cursor.execute(f'SELECT * FROM "{name}"')
    columns = [item[0] for item in cursor.description]
    rows = cursor.fetchall()
    return {"columns": columns, "rows": rows}


def _read_fdr(path: Path) -> dict:
    connection = sqlite3.connect(":memory:")
    try:
        connection.deserialize(decrypt_fdr(str(path)))
        cursor = connection.cursor()
        cursor.execute("SELECT rowid, * FROM Sweeps ORDER BY rowid")
        sweep_columns = [item[0] for item in cursor.description][1:]
        sweeps = {}
        for row in cursor.fetchall():
            rowid = int(row[0])
            tables = {
                key: _read_table(connection, f"{table_name}_{rowid}")
                for table_name, key in (
                    ("MeasureData", "s11"),
                    ("ImpulseResponse", "impulse"),
                    ("StepResponse", "step"),
                )
            }
            sweeps[rowid] = {
                "rowid": rowid,
                "columns": sweep_columns,
                "values": row[1:],
                "tables": tables,
            }
        globals_ = {
            key: _read_table(connection, table_name)
            for table_name, key in (
                ("MeasureSetting", "measure_setting"),
                ("Nameplate", "nameplate"),
                ("CalibrationData", "calibration"),
            )
        }
        return {
            "path": str(path),
            "sweeps": sweeps,
            "globals": globals_,
        }
    finally:
        connection.close()


def _table_matrix(table: dict | None) -> np.ndarray:
    if table is None:
        return np.empty((0, 0), dtype=np.float64)
    if not table["rows"]:
        return np.empty((0, len(table["columns"])), dtype=np.float64)
    return np.asarray(table["rows"], dtype=np.float64)


def _column_index(table: dict, *names: str) -> int:
    columns = {name.lower(): index for index, name in enumerate(table["columns"])}
    for name in names:
        if name.lower() in columns:
            return columns[name.lower()]
    raise KeyError(f"表 {table['columns']} 缺少列 {names}")


def _response_arrays(table: dict) -> tuple[np.ndarray, np.ndarray]:
    matrix = _table_matrix(table)
    distance = matrix[:, _column_index(table, "distance")]
    response = matrix[:, _column_index(table, "rho")]
    order = np.argsort(distance)
    return distance[order], response[order]


def _row_meta(sweep: dict | None) -> dict | None:
    if sweep is None:
        return None
    return {
        "rowid": sweep["rowid"],
        "columns": sweep["columns"],
        "values": [_json_value(value) for value in sweep["values"]],
        "tables": {
            key: None if table is None else {
                "columns": table["columns"],
                "rows": len(table["rows"]),
            }
            for key, table in sweep["tables"].items()
        },
    }


def _global_meta(table: dict | None) -> dict | None:
    if table is None:
        return None
    return {
        "columns": table["columns"],
        "row_count": len(table["rows"]),
    }


def _is_complete_response(sweep: dict | None) -> tuple[bool, list[str]]:
    if sweep is None:
        return False, ["Sweeps.rowid"]
    missing = []
    for key, table_name in (("impulse", "ImpulseResponse"), ("step", "StepResponse")):
        table = sweep["tables"][key]
        if table is None or not table["rows"]:
            missing.append(f"{table_name}_{sweep['rowid']}")
    return not missing, missing


def _odd_window(physical_span_m: float, source_step_m: float) -> int:
    sample_count = max(1, int(np.floor(physical_span_m / source_step_m + 0.5)) + 1)
    if sample_count % 2 == 0:
        candidates = (max(1, sample_count - 1), sample_count + 1)
        sample_count = min(
            candidates,
            key=lambda count: (abs((count - 1) * source_step_m - physical_span_m), count),
        )
    return sample_count


def _prepare_phase(
    impulse_table: dict,
    step_table: dict,
    distance_grid: np.ndarray,
    smoothing_span_m: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    impulse_distance, impulse = _response_arrays(impulse_table)
    step_distance, step = _response_arrays(step_table)
    impulse_step_m = float(np.median(np.diff(impulse_distance)))
    step_step_m = float(np.median(np.diff(step_distance)))
    impulse_k = _odd_window(smoothing_span_m, impulse_step_m)
    step_k = _odd_window(smoothing_span_m, step_step_m)
    impulse_smooth = movmean(np.column_stack((impulse_distance, impulse)), impulse_k)[:, 1]
    step_smooth = movmean(np.column_stack((step_distance, step)), step_k)[:, 1]
    step_derivative = np.gradient(step_smooth, step_distance)

    common_start = max(float(impulse_distance[0]), float(step_distance[0]))
    common_end = min(float(impulse_distance[-1]), float(step_distance[-1]))
    valid = (distance_grid >= common_start) & (distance_grid <= common_end)
    x = np.zeros((5, distance_grid.size), dtype=np.float32)
    x[0] = np.interp(
        distance_grid,
        impulse_distance,
        np.sign(impulse_smooth) * np.log1p(np.abs(impulse_smooth)),
    )
    x[1] = np.interp(distance_grid, step_distance, step)
    x[2] = np.interp(distance_grid, step_distance, step_smooth)
    x[3] = np.interp(distance_grid, step_distance, step_derivative)
    x[4] = distance_grid / 3000.0
    x[:, ~valid] = 0.0
    return x, valid, {
        "impulse_source_step_m": impulse_step_m,
        "step_source_step_m": step_step_m,
        "impulse_window_k": impulse_k,
        "step_window_k": step_k,
        "impulse_applied_span_m": (impulse_k - 1) * impulse_step_m,
        "step_applied_span_m": (step_k - 1) * step_step_m,
        "common_response_start_m": common_start,
        "common_response_end_m": common_end,
    }


def _measurement_id(value: str) -> tuple[Path, int]:
    source_path, rowid_text = value.rsplit("#Sweeps.rowid=", 1)
    return Path(source_path), int(rowid_text)


def _safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.")


def _slot_layout(members: dict) -> tuple[list[str], list[str | None]]:
    if len(members) == 1:
        slot = next(iter(members))
        return [slot], [members[slot]]
    if not set(members).issubset({"A", "B", "C"}):
        raise ValueError("多相比较组的members必须显式使用A/B/C键")
    slots = ["A", "B", "C"]
    return slots, [members.get(slot) for slot in slots]


def _label_is_supervised(label: dict | None) -> bool:
    if label is None:
        return False
    if label.get("events"):
        return True
    return any(
        bool(ranges)
        for ranges in (label.get("negative_ranges") or {}).values()
    )


def _cache_document(
    document: dict,
    token: str,
    rowid: int,
    arrays: dict[str, np.ndarray],
    columns: dict,
) -> None:
    sweep = document["sweeps"].get(rowid)
    if sweep is None:
        return
    prefix = f"{token}_rowid_{rowid}"
    for key in ("s11", "impulse", "step"):
        table = sweep["tables"][key]
        arrays[f"{prefix}_{key}"] = _table_matrix(table)
        columns[f"{prefix}_{key}"] = None if table is None else table["columns"]


def _run(args: argparse.Namespace) -> None:
    config = _load_json(EXPERIMENT_DIR / "config" / "first_round.json")
    groups_document = _load_json(args.groups)
    labels_document = _load_json(args.labels)
    labels_by_id = {
        item["measurement_id"]: item
        for item in labels_document.get("measurements", [])
    }
    preprocessing = config["preprocessing"]
    start_m = float(preprocessing["distance_start_m"])
    end_m = float(preprocessing["distance_end_m"])
    distance_step_m = float(preprocessing["distance_step_m"])
    smoothing_span_m = float(preprocessing["smoothing_physical_span_m"])
    length = int(round((end_m - start_m) / distance_step_m)) + 1
    distance_grid = start_m + np.arange(length, dtype=np.float64) * distance_step_m

    output_dir = args.output_dir
    groups_dir = output_dir / "groups"
    source_cache_dir = output_dir / "source_cache"
    review_dir = output_dir / "review"
    groups_dir.mkdir(parents=True, exist_ok=True)
    source_cache_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    documents: dict[str, dict] = {}
    document_tokens: dict[str, str] = {}
    manifest_groups = []
    summary_groups = []

    def get_document(path: Path) -> tuple[str, dict]:
        cache_key = str(path.resolve())
        if cache_key not in documents:
            token = f"source_{len(documents):02d}"
            documents[cache_key] = _read_fdr(path)
            document_tokens[cache_key] = token
        return document_tokens[cache_key], documents[cache_key]

    groups = groups_document.get("groups", [])
    for group_index, group in enumerate(groups, start=1):
        comparison_group_id = str(group["comparison_group_id"])
        print(f"[{group_index}/{len(groups)}] {comparison_group_id}", flush=True)
        case_id = str(group.get("case_id", group["case_group_id"]))
        case_group_id = str(group.get("case_group_id", case_id))
        members = group["members"]
        slots, measurement_ids = _slot_layout(members)
        phase_count = len(slots)
        x = np.zeros((phase_count, 5, length), dtype=np.float32)
        valid = np.zeros((phase_count, length), dtype=bool)
        phase_present = np.asarray([value is not None for value in measurement_ids], dtype=bool)
        phase_mask = np.zeros(phase_count, dtype=bool)
        source_relations = []
        source_documents_meta = {}
        source_cache_arrays: dict[str, np.ndarray] = {}
        source_cache_columns = {}
        smoothing_meta = []

        derived_copy_value = group.get("derived_copy")
        derived_copy_path = (
            (EXPERIMENT_DIR / derived_copy_value).resolve()
            if derived_copy_value
            else None
        )

        for phase_index, (slot, measurement_id) in enumerate(zip(slots, measurement_ids)):
            if measurement_id is None:
                source_relations.append({
                    "slot": slot,
                    "phase_name": group.get("phase_names", {}).get(slot),
                    "measurement_id": None,
                    "phase_present": False,
                    "response_status": "phase_missing",
                })
                smoothing_meta.append(None)
                continue

            source_path, rowid = _measurement_id(str(measurement_id))
            original_token, original_document = get_document(source_path)
            original_sweep = original_document["sweeps"].get(rowid)
            original_s11 = None if original_sweep is None else original_sweep["tables"]["s11"]
            _cache_document(
                original_document, original_token, rowid, source_cache_arrays, source_cache_columns
            )
            original_source_meta = source_documents_meta.setdefault(original_token, {
                "path": str(source_path),
                "role": "original_measurement_file",
                "measure_setting": _global_meta(original_document["globals"]["measure_setting"]),
                "nameplate": _global_meta(original_document["globals"]["nameplate"]),
                "calibration": _global_meta(original_document["globals"]["calibration"]),
                "sweeps": {},
            })
            original_source_meta["sweeps"][str(rowid)] = _row_meta(original_sweep)
            calibration = original_document["globals"]["calibration"]
            source_cache_arrays[f"{original_token}_calibration"] = _table_matrix(calibration)
            source_cache_columns[f"{original_token}_calibration"] = (
                None if calibration is None else calibration["columns"]
            )

            response_document = original_document
            response_sweep = original_sweep
            response_token = original_token
            response_source_path = source_path
            response_source_role = "original_measurement_file"
            derived_copy_status = None
            use_derived_copy = derived_copy_path is not None and (
                slot == "C" or len(members) == 1
            )
            if use_derived_copy:
                if derived_copy_path.exists():
                    response_token, response_document = get_document(derived_copy_path)
                    response_sweep = response_document["sweeps"].get(rowid)
                    response_source_path = derived_copy_path
                    response_source_role = "derived_copy"
                    response_source_meta = source_documents_meta.setdefault(response_token, {
                        "path": str(derived_copy_path),
                        "role": "derived_copy",
                        "measure_setting": _global_meta(response_document["globals"]["measure_setting"]),
                        "nameplate": _global_meta(response_document["globals"]["nameplate"]),
                        "calibration": _global_meta(response_document["globals"]["calibration"]),
                        "sweeps": {},
                    })
                    response_source_meta["sweeps"][str(rowid)] = _row_meta(response_sweep)
                    derived_copy_status = "present"
                    derived_calibration = response_document["globals"]["calibration"]
                    source_cache_arrays[f"{response_token}_calibration"] = _table_matrix(
                        derived_calibration
                    )
                    source_cache_columns[f"{response_token}_calibration"] = (
                        None if derived_calibration is None else derived_calibration["columns"]
                    )
                    _cache_document(
                        response_document,
                        response_token,
                        rowid,
                        source_cache_arrays,
                        source_cache_columns,
                    )
                else:
                    response_sweep = None
                    response_source_path = derived_copy_path
                    response_source_role = "derived_copy_missing"
                    derived_copy_status = "missing"

            response_complete, missing_tables = _is_complete_response(response_sweep)
            phase_name = group.get("phase_names", {}).get(slot)
            if phase_name is None and original_sweep is not None:
                sweep_columns = {
                    name.lower(): index for index, name in enumerate(original_sweep["columns"])
                }
                if "sweepname" in sweep_columns:
                    phase_name = _json_value(
                        original_sweep["values"][sweep_columns["sweepname"]]
                    )

            relation = {
                "slot": slot,
                "phase_name": phase_name,
                "measurement_id": str(measurement_id),
                "phase_present": True,
                "rowid": rowid,
                "original_file": str(source_path),
                "s11_source": str(source_path),
                "s11_present": original_s11 is not None and bool(original_s11["rows"]),
                "response_source": str(response_source_path),
                "response_source_role": response_source_role,
                "configured_derived_copy": (
                    str(derived_copy_path)
                    if use_derived_copy
                    else None
                ),
                "derived_copy_status": derived_copy_status,
                "response_status": "available" if response_complete else "response_missing",
                "missing_response_tables": missing_tables,
                "original_sweep": _row_meta(original_sweep),
                "response_sweep": _row_meta(response_sweep),
            }
            source_relations.append(relation)

            if response_complete:
                x[phase_index], valid[phase_index], phase_smoothing = _prepare_phase(
                    response_sweep["tables"]["impulse"],
                    response_sweep["tables"]["step"],
                    distance_grid,
                    smoothing_span_m,
                )
                phase_mask[phase_index] = bool(valid[phase_index].any())
                if not phase_mask[phase_index]:
                    relation["response_status"] = "no_overlap_with_public_grid"
                smoothing_meta.append(phase_smoothing)
            else:
                smoothing_meta.append(None)

        label_events = []
        label_negative_ranges = {}
        label_source_refs = {}
        for measurement_id in measurement_ids:
            if measurement_id is None:
                continue
            label = labels_by_id.get(str(measurement_id))
            if label is None:
                continue
            for event in label.get("events") or []:
                label_events.append({**event, "measurement_id": str(measurement_id)})
            label_negative_ranges[str(measurement_id)] = label.get("negative_ranges") or {}
            if label.get("source_refs"):
                label_source_refs[str(measurement_id)] = label["source_refs"]

        safe_stem = _safe_stem(comparison_group_id)
        group_file = f"groups/{safe_stem}.npz"
        source_cache_file = f"source_cache/{safe_stem}.npz"
        source_cache_arrays["metadata"] = np.asarray(
            json.dumps(
                {
                    "sources": source_documents_meta,
                    "table_columns": source_cache_columns,
                },
                ensure_ascii=False,
            )
        )
        np.savez_compressed(
            output_dir / source_cache_file,
            **source_cache_arrays,
        )

        meta = {
            "comparison_group_id": comparison_group_id,
            "case_id": case_id,
            "case_group_id": case_group_id,
            "measurement_ids": measurement_ids,
            "slots": slots,
            "phase_names": [relation["phase_name"] for relation in source_relations],
            "phase_present": phase_present.tolist(),
            "phase_mask": phase_mask.tolist(),
            "source_files": list(dict.fromkeys(
                relation["original_file"]
                for relation in source_relations
                if relation.get("original_file")
            )),
            "source_relations": source_relations,
            "source_documents": source_documents_meta,
            "source_cache_file": source_cache_file,
            "coordinate_alignment_basis": group.get("coordinate_alignment_basis"),
            "member_roles": group.get("member_roles", {}),
            "events": label_events,
            "negative_ranges": label_negative_ranges,
            "label_source_refs": label_source_refs,
            "preprocessing": {
                "distance_start_m": start_m,
                "distance_end_m": end_m,
                "distance_step_m": distance_step_m,
                "smoothing_physical_span_m": smoothing_span_m,
                "reader_display_smoothing_k": group.get("reader_display_smoothing_k"),
                "phase_windows": smoothing_meta,
            },
        }
        np.savez_compressed(
            output_dir / group_file,
            x=x,
            valid=valid,
            phase_mask=phase_mask,
            distance_m=distance_grid.astype(np.float32),
            meta=np.asarray(json.dumps(meta, ensure_ascii=False)),
        )

        supervised_measurements = [
            str(measurement_id)
            for measurement_id in measurement_ids
            if measurement_id is not None
            and _label_is_supervised(labels_by_id.get(str(measurement_id)))
        ]
        phase_mask_list = phase_mask.tolist()
        manifest_group = {
            "comparison_group_id": comparison_group_id,
            "case_id": case_id,
            "case_group_id": case_group_id,
            "file": group_file,
            "source_cache_file": source_cache_file,
            "supervised": bool(supervised_measurements),
            "supervised_measurement_ids": supervised_measurements,
            "measurement_ids": measurement_ids,
            "slots": slots,
            "phase_present": phase_present.tolist(),
            "phase_mask": phase_mask_list,
            "member_roles": group.get("member_roles", {}),
            "response_missing_measurement_ids": [
                relation["measurement_id"]
                for relation in source_relations
                if relation.get("measurement_id")
                and relation["response_status"] == "response_missing"
            ],
        }
        manifest_groups.append(manifest_group)
        summary_groups.append({
            "comparison_group_id": comparison_group_id,
            "case_group_id": case_group_id,
            "measurement_ids": measurement_ids,
            "phase_present": phase_present.tolist(),
            "phase_mask": phase_mask_list,
            "supervised_measurement_ids": supervised_measurements,
            "source_relations": source_relations,
        })

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump({"groups": manifest_groups}, stream, ensure_ascii=False, indent=2)
    with (review_dir / "accepted_labels.json").open("w", encoding="utf-8") as stream:
        json.dump(labels_document, stream, ensure_ascii=False, indent=2)
    with (output_dir / "prepare_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "config_file": str(EXPERIMENT_DIR / "config" / "first_round.json"),
                "groups_file": str(args.groups),
                "labels_file": str(args.labels),
                "group_count": len(manifest_groups),
                "source_file_count": len(documents),
                "groups": summary_groups,
            },
            stream,
            ensure_ascii=False,
            indent=2,
        )
    print(f"已写入 {len(manifest_groups)} 组：{output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="读取现场FDR并准备分组训练缓存")
    parser.add_argument(
        "--groups",
        type=Path,
        default=EXPERIMENT_DIR / "review" / "comparison_groups.json",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=EXPERIMENT_DIR / "review" / "accepted_labels.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPERIMENT_DIR / "output" / "field_cache",
    )
    _run(parser.parse_args())


if __name__ == "__main__":
    main()
