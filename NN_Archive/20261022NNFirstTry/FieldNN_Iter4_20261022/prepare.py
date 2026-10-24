"""Cache Iter3 S11 as raw Algorithm1 responses and build a calibration nuisance bank."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
CORE16 = PROJECT / "DifferentAgent" / "CodeX" / "Core16_Field"
DEFAULT_SOURCE = Path(r"E:\CableNNDataSet\CodeX\FieldNN_Iter3_20261022")
DEFAULT_OUTPUT = Path(r"E:\CableNNDataSet\CodeX\FieldNN_Iter4_20261022")
DEFAULT_CONFIG = HERE / "config.json"
SPLITS = ("train", "val", "test")
sys.path.insert(0, str(CORE16))
from field_responses import algorithm1  # noqa: E402

_WORKER_FREQUENCY: np.ndarray | None = None
_WORKER_DISTANCE: np.ndarray | None = None
_WORKER_CFG: dict[str, Any] | None = None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _worker_init(source_text: str, cfg: dict[str, Any]) -> None:
    global _WORKER_FREQUENCY, _WORKER_DISTANCE, _WORKER_CFG
    source = Path(source_text)
    _WORKER_FREQUENCY = np.load(source / "frequency.npy", mmap_mode="r")
    _WORKER_DISTANCE = np.linspace(
        0.0,
        float(cfg["response"]["distance_max_m"]),
        int(cfg["response"]["points"]),
        dtype=np.float64,
    )
    _WORKER_CFG = cfg


def _raw_responses(
    frequency_hz: np.ndarray,
    s11: np.ndarray,
    cfg: dict[str, Any],
    distance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    band = cfg["frequency"]
    frequency_hz = np.asarray(frequency_hz, dtype=np.float64)
    s11 = np.asarray(s11, dtype=np.complex128)
    keep = (frequency_hz >= float(band["start_hz"])) & (
        frequency_hz <= float(band["stop_hz"])
    )
    frequency = np.unique(
        np.r_[float(band["start_hz"]), frequency_hz[keep], float(band["stop_hz"])]
    )
    uniform_s11 = np.interp(frequency, frequency_hz, s11.real) + 1j * np.interp(
        frequency, frequency_hz, s11.imag
    )
    response_cfg = cfg["response"]
    transformed = algorithm1(
        frequency,
        uniform_s11,
        velocity_m_s=float(response_cfg["velocity_m_s"]),
        distance_max_m=float(response_cfg["distance_max_m"]),
    )
    impulse = np.interp(
        distance, transformed["distance_m"], transformed["impulse_raw"]
    )
    step = np.interp(distance, transformed["distance_m"], transformed["step_rho"])
    native_dx = float(transformed["distance_m"][1] - transformed["distance_m"][0])
    return impulse.astype(np.float32), step.astype(np.float32), native_dx


def _process_chunk(job: dict[str, Any]) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    source = Path(job["source"])
    folder = source / job["shard_path"]
    start = int(job["source_start"])
    end = start + int(job["count"])
    s11 = np.load(folder / "s11.npy", mmap_mode="r")[start:end]
    targets = np.load(folder / "targets.npy", mmap_mode="r")[start:end]
    count = end - start
    points = int(_WORKER_CFG["response"]["points"])
    impulse = np.empty((count, points), dtype=np.float32)
    step = np.empty_like(impulse)
    for index in range(count):
        impulse[index], step[index], _ = _raw_responses(
            _WORKER_FREQUENCY, s11[index], _WORKER_CFG, _WORKER_DISTANCE
        )
    return int(job["destination_start"]), impulse, step, np.asarray(targets, dtype=np.float32)


def _split_jobs(
    source: Path,
    shards: list[dict[str, Any]],
    split: str,
    count_limit: int,
    chunk_size: int,
) -> list[dict[str, Any]]:
    jobs = []
    destination = 0
    remaining = count_limit
    for shard in shards:
        if shard["split"] != split or remaining <= 0:
            continue
        take = min(int(shard["count"]), remaining)
        for source_start in range(0, take, chunk_size):
            count = min(chunk_size, take - source_start)
            jobs.append({
                "source": str(source),
                "shard_path": shard["path"],
                "source_start": source_start,
                "destination_start": destination,
                "count": count,
            })
            destination += count
        remaining -= take
    return jobs


def _write_split_metadata(
    source: Path,
    shards: list[dict[str, Any]],
    split: str,
    count_limit: int,
    output_path: Path,
) -> None:
    remaining = count_limit
    with output_path.open("w", encoding="utf-8", newline="\n") as dst:
        for shard in shards:
            if shard["split"] != split or remaining <= 0:
                continue
            path = source / shard["path"] / "metadata.jsonl"
            with path.open("r", encoding="utf-8") as src:
                for line in src:
                    if remaining <= 0:
                        break
                    dst.write(line)
                    remaining -= 1


def _split_complete(output: Path, split: str, count: int, points: int) -> bool:
    marker = output / "prepare_manifest.json"
    required = (
        output / "arrays" / f"{split}_raw_imp.npy",
        output / "arrays" / f"{split}_raw_step.npy",
        output / "arrays" / f"{split}_targets.npy",
        output / f"metadata_{split}.jsonl",
    )
    if not marker.is_file() or not all(path.is_file() for path in required):
        return False
    manifest = _read_json(marker)
    row = manifest.get("splits", {}).get(split, {})
    if not row.get("complete") or int(row.get("count", -1)) != count:
        return False
    expected_shapes = {
        f"{split}_raw_imp.npy": (count, points),
        f"{split}_raw_step.npy": (count, points),
        f"{split}_targets.npy": (count, 3, points),
    }
    for name, shape in expected_shapes.items():
        values = np.load(output / "arrays" / name, mmap_mode="r")
        valid = values.shape == shape
        mmap = getattr(values, "_mmap", None)
        if mmap is not None:
            mmap.close()
        if not valid:
            return False
    with (output / f"metadata_{split}.jsonl").open("r", encoding="utf-8") as stream:
        return sum(1 for _ in stream) == count


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT / path


def _calibration_sources(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    bank_cfg = cfg["calibration_bank"]
    manifest_path = _resolve_repo_path(bank_cfg["real_manifest"])
    index_path = _resolve_repo_path(bank_cfg["calibration_index"])
    calibration_rows = {}
    with index_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            relative = str(row["source_relative_path"]).replace("\\", "/")
            calibration_rows[relative] = int(float(row.get("row_count") or 0))

    selected = []
    seen = set()
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("development_status") != bank_cfg["development_status_required"]:
            continue
        relative = str(row.get("source_relative_path") or "").replace("\\", "/")
        if not relative or relative in seen:
            continue
        seen.add(relative)
        count = calibration_rows.get(relative, 0)
        if count:
            selected.append({
                "source_relative_path": relative,
                "source_case_group": str(row.get("source_case_group") or ""),
                "development_status": str(row["development_status"]),
                "calibration_point_count_from_index": count,
            })
        if len(selected) >= int(bank_cfg["maximum_distinct_fdr_sources"]):
            break
    if len(selected) < int(bank_cfg["minimum_distinct_fdr_sources"]):
        raise ValueError(
            f"Only {len(selected)} eligible development CalibrationData sources; "
            f"need {bank_cfg['minimum_distinct_fdr_sources']}"
        )
    return selected


def _make_calibration_bank(
    output: Path,
    cfg: dict[str, Any],
    source_root: Path,
    frequency: np.ndarray,
    distance: np.ndarray,
    resume: bool,
) -> dict[str, Any]:
    bank_cfg = cfg["calibration_bank"]
    bank_path = output / bank_cfg["array_file"]
    log_path = output / bank_cfg["source_log"]
    marker_path = output / "prepare_manifest.json"
    if resume and bank_path.is_file() and log_path.is_file() and marker_path.is_file():
        marker = _read_json(marker_path)
        bank_info = marker.get("calibration_bank", {})
        if bank_info.get("complete"):
            return bank_info

    bank_cfg = cfg["calibration_bank"]
    sources = _calibration_sources(cfg)
    reader = _resolve_repo_path("DifferentAgent/CodeX/Core16_Field/analyze_corpus.py")
    sys.path.insert(0, str(reader.parent))
    from analyze_corpus import load_reader_module

    fdr_reader = load_reader_module()
    full_until = float(bank_cfg["full_scale_until_m"])
    taper_end = float(bank_cfg["taper_to_zero_at_m"])
    active = distance <= float(bank_cfg["active_rms_region_m"])
    weights = np.zeros(distance.size, dtype=np.float64)
    weights[distance <= full_until] = 1.0
    taper = (distance > full_until) & (distance < taper_end)
    weights[taper] = 0.5 * (
        1.0 + np.cos(np.pi * (distance[taper] - full_until) / (taper_end - full_until))
    )

    bank_impulse = []
    bank_step = []
    logs = []
    for bank_index, record in enumerate(tqdm(sources, desc="CalibrationData bank", unit="FDR")):
        source_path = source_root.joinpath(*PurePosixPath(record["source_relative_path"]).parts)
        plaintext, _ = fdr_reader.FdrCrypto().decrypt(source_path.read_bytes())
        with sqlite3.connect(":memory:") as connection:
            connection.deserialize(plaintext)
            rows = connection.execute(
                "SELECT Freq, Real, Image FROM CalibrationData ORDER BY Freq"
            ).fetchall()
        raw = np.asarray(rows, dtype=np.float64)
        raw = raw[np.isfinite(raw).all(axis=1)]
        order = np.argsort(raw[:, 0], kind="stable")
        raw = raw[order]
        _, first = np.unique(raw[:, 0], return_index=True)
        raw = raw[np.sort(first)]
        s11 = raw[:, 1] + 1j * raw[:, 2]
        impulse, step, native_dx = _raw_responses(raw[:, 0], s11, cfg, distance)
        step = step.astype(np.float64)
        plateau_mask = (distance >= 200.0) & (distance <= 250.0)
        plateau = float(np.median(step[plateau_mask]))
        centered_step = step - plateau
        tapered_step = centered_step * weights
        fixed_dx = float(distance[1] - distance[0])
        derivative_scale = native_dx / fixed_dx
        tapered_impulse = np.diff(tapered_step, prepend=0.0) * derivative_scale
        scale = float(np.sqrt(np.mean(tapered_impulse[active] ** 2)))
        bank_impulse.append((tapered_impulse / scale).astype(np.float32))
        bank_step.append((tapered_step / scale).astype(np.float32))
        logs.append({
            "bank_index": bank_index,
            **record,
            "calibration_data_source": source_path.as_posix(),
            "calibration_data_point_count": int(raw.shape[0]),
            "bank_kind": "known CalibrationData short/no-cable reference nuisance",
            "cable_health_label": None,
            "terminal_label": None,
            "event_label": None,
            "active_impulse_rms_before_shared_normalization": scale,
            "active_step_rms_before_shared_normalization": float(
                np.sqrt(np.mean(tapered_step[active] ** 2))
            ),
            "removed_step_plateau_200_250m": plateau,
            "shared_amplitude_scale": scale,
            "algorithm1_native_distance_step_m": native_dx,
            "cache_fixed_distance_step_m": fixed_dx,
            "impulse_difference_scale_native_over_fixed_dx": derivative_scale,
            "full_scale_until_m": full_until,
            "taper_to_zero_at_m": taper_end,
            "taper_policy": "subtract step plateau, cosine-taper the step, then derive paired impulse from its first difference including the zero-origin start",
        })

    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        bank_path,
        impulse=np.stack(bank_impulse, axis=0),
        step=np.stack(bank_step, axis=0),
    )
    with log_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in logs:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {
        "complete": True,
        "count": len(logs),
        "array_path": bank_path.name,
        "source_log": log_path.name,
        "shape": [len(logs), int(distance.size)],
        "semantics": bank_cfg["semantics"],
    }


def _effective_config(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = _read_json(Path(args.config) if args.config else DEFAULT_CONFIG)
    source = Path(args.source).expanduser()
    output = Path(args.output).expanduser()
    source_cfg = _read_json(source / "config.json")
    source_manifest = _read_json(source / "manifest.json")
    cfg["frequency"] = source_cfg["frequency"]
    cfg["response"] = source_cfg["response"]
    cfg["evaluation"] = source_cfg["evaluation"]
    cfg["seed"] = 2026102204
    dataset = cfg["dataset"]
    dataset["source_path"] = str(source)
    dataset["output_path"] = str(output)
    dataset["source_counts"] = source_manifest["counts"]
    dataset["sample_limit_per_split"] = int(args.samples) if args.samples is not None else None
    dataset["workers"] = int(args.workers)
    if args.samples is not None and int(args.samples) < 1:
        raise ValueError("--samples must be positive")
    if int(args.workers) < 1:
        raise ValueError("--workers must be positive")
    dataset["response_source"] = "frequency.npy/s11.npy from the frozen Iter3 source"
    return cfg, source_manifest


def _update_prepare_manifest(path: Path, manifest: dict[str, Any]) -> None:
    _write_json(path, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--samples", type=int, default=None, help="maximum rows per split; intended for smoke preparation")
    parser.add_argument("--resume", action="store_true", help="skip completed splits and an existing complete bank")
    parser.add_argument("--bank-only", action="store_true", help="refresh only calibration_bank_v2.npz and its source log")
    args = parser.parse_args()

    cfg, source_manifest = _effective_config(args)
    source = Path(cfg["dataset"]["source_path"])
    output = Path(cfg["dataset"]["output_path"])
    output.mkdir(parents=True, exist_ok=True)
    if args.bank_only:
        existing_config = _read_json(output / "config.json") if (output / "config.json").is_file() else cfg
        existing_config["calibration_bank"] = cfg["calibration_bank"]
        existing_config["evaluation"] = cfg["evaluation"]
        existing_config["seed"] = cfg["seed"]
        _write_json(output / "config.json", existing_config)
        frequency = np.load(source / "frequency.npy", mmap_mode="r")
        distance = np.load(source / "distance.npy", mmap_mode="r")
        marker_path = output / "prepare_manifest.json"
        prepared = _read_json(marker_path) if marker_path.is_file() else {
            "schema": "fieldnn-iter4-prepare-manifest-1",
            "source_path": str(source),
            "split_counts": {},
            "splits": {},
        }
        prepared["calibration_bank"] = _make_calibration_bank(
            output, cfg, Path(cfg["calibration_bank"]["source_root"]),
            frequency, distance, resume=False,
        )
        _update_prepare_manifest(marker_path, prepared)
        output_manifest_path = output / "manifest.json"
        output_manifest = _read_json(output_manifest_path) if output_manifest_path.is_file() else {}
        output_manifest["calibration_bank"] = prepared["calibration_bank"]
        _write_json(output_manifest_path, output_manifest)
        print(
            f"Refreshed calibration nuisance bank: {prepared['calibration_bank']['count']} curves; "
            f"output={output / cfg['calibration_bank']['array_file']}",
            flush=True,
        )
        return

    config_path = output / "config.json"
    if config_path.exists():
        current = _read_json(config_path)
        if _canonical(current) != _canonical(cfg):
            raise ValueError(f"output config differs from request: {config_path}")
    else:
        _write_json(config_path, cfg)

    frequency = np.load(source / "frequency.npy", mmap_mode="r")
    distance = np.load(source / "distance.npy", mmap_mode="r")
    shutil_source = (
        ("frequency.npy", source / "frequency.npy"),
        ("distance.npy", source / "distance.npy"),
    )
    for name, input_path in shutil_source:
        target_path = output / name
        if not target_path.exists():
            target_path.write_bytes(input_path.read_bytes())

    marker_path = output / "prepare_manifest.json"
    if marker_path.is_file():
        prepared = _read_json(marker_path)
    else:
        prepared = {
            "schema": "fieldnn-iter4-prepare-manifest-1",
            "source_path": str(source),
            "split_counts": {},
            "splits": {},
            "calibration_bank": {"complete": False},
        }
    bank_cfg = cfg["calibration_bank"]
    source_root = Path(bank_cfg["source_root"])
    if not (args.resume and prepared.get("calibration_bank", {}).get("complete")):
        prepared["calibration_bank"] = _make_calibration_bank(
            output, cfg, source_root, frequency, distance, args.resume
        )
        _update_prepare_manifest(marker_path, prepared)

    counts = {}
    for split in SPLITS:
        total = int(source_manifest["counts"][split])
        count = min(total, int(args.samples)) if args.samples is not None else total
        counts[split] = count
        if args.resume and _split_complete(output, split, count, int(cfg["response"]["points"])):
            prepared["splits"][split] = {
                **prepared["splits"].get(split, {}),
                "complete": True,
                "count": count,
                "skipped_on_resume": True,
            }
            continue

        array_dir = output / "arrays"
        array_dir.mkdir(parents=True, exist_ok=True)
        points = int(cfg["response"]["points"])
        imp_path = array_dir / f"{split}_raw_imp.npy"
        step_path = array_dir / f"{split}_raw_step.npy"
        target_path = array_dir / f"{split}_targets.npy"
        raw_imp = np.lib.format.open_memmap(
            imp_path, mode="w+", dtype=np.float32, shape=(count, points)
        )
        raw_step = np.lib.format.open_memmap(
            step_path, mode="w+", dtype=np.float32, shape=(count, points)
        )
        targets = np.lib.format.open_memmap(
            target_path, mode="w+", dtype=np.float32, shape=(count, 3, points)
        )
        _write_split_metadata(
            source,
            source_manifest["shards"],
            split,
            count,
            output / f"metadata_{split}.jsonl",
        )
        jobs = _split_jobs(
            source,
            source_manifest["shards"],
            split,
            count,
            int(cfg["dataset"]["chunk_size"]),
        )
        started = time.perf_counter()
        with ProcessPoolExecutor(
            max_workers=int(args.workers),
            initializer=_worker_init,
            initargs=(str(source), cfg),
        ) as executor:
            futures = {executor.submit(_process_chunk, job): job for job in jobs}
            with tqdm(total=count, desc=f"Cache {split} raw responses", unit="sample") as progress:
                for future in as_completed(futures):
                    destination_start, imp_chunk, step_chunk, target_chunk = future.result()
                    end = destination_start + imp_chunk.shape[0]
                    raw_imp[destination_start:end] = imp_chunk
                    raw_step[destination_start:end] = step_chunk
                    targets[destination_start:end] = target_chunk
                    progress.update(imp_chunk.shape[0])
        raw_imp.flush()
        raw_step.flush()
        targets.flush()
        del raw_imp, raw_step, targets
        prepared["splits"][split] = {
            "complete": True,
            "count": count,
            "raw_impulse_path": f"arrays/{split}_raw_imp.npy",
            "raw_step_path": f"arrays/{split}_raw_step.npy",
            "targets_path": f"arrays/{split}_targets.npy",
            "metadata_path": f"metadata_{split}.jsonl",
            "seconds": time.perf_counter() - started,
            "source_shard_count": sum(item["split"] == split for item in source_manifest["shards"]),
        }
        prepared["split_counts"] = counts
        _update_prepare_manifest(marker_path, prepared)

    prepared["split_counts"] = counts
    prepared["calibration_bank"] = {
        **prepared["calibration_bank"],
        "complete": True,
    }
    _update_prepare_manifest(marker_path, prepared)
    _write_json(output / "manifest.json", {
        "schema": "fieldnn-iter4-raw-cache-1",
        "source_dataset": str(source),
        "counts": counts,
        "arrays": {
            split: {
                "raw_impulse": f"arrays/{split}_raw_imp.npy",
                "raw_step": f"arrays/{split}_raw_step.npy",
                "targets": f"arrays/{split}_targets.npy",
                "metadata": f"metadata_{split}.jsonl",
            }
            for split in SPLITS
        },
        "calibration_bank": prepared["calibration_bank"],
    })
    print(
        f"Prepared raw response cache: train={counts['train']}, val={counts['val']}, "
        f"test={counts['test']}; calibration bank={prepared['calibration_bank']['count']} curves; "
        f"output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
