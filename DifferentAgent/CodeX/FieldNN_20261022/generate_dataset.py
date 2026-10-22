"""Generate resumable, sharded FieldNN training data on CPU."""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from common import distance_grid, frequency_grid, load_config, write_json
from generator import generate_sample, load_case_templates, load_physics_config, sample_plan

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = Path(r"E:\CableNNDataSet\CodeX\FieldNN_20261022")
SPLITS = ("train", "val", "test")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _canonical_config(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _make_shard_jobs(plan: list[dict[str, Any]], shard_size: int):
    jobs = []
    for split in SPLITS:
        rows = [item for item in plan if item["split"] == split]
        for shard_index, start in enumerate(range(0, len(rows), shard_size)):
            jobs.append({
                "split": split,
                "shard_index": shard_index,
                "plan": rows[start:start + shard_size],
            })
    return jobs


def _shard_path(root: Path, job: dict) -> Path:
    return root / "shards" / job["split"] / f"{int(job['shard_index']):05d}"


def _shard_complete(root: Path, job: dict, cfg: dict) -> bool:
    folder = _shard_path(root, job)
    summary_path = folder / "summary.json"
    required = (
        folder / "s11.npy",
        folder / "inputs.npy",
        folder / "targets.npy",
        folder / "metadata.jsonl",
    )
    if not summary_path.is_file() or not all(path.is_file() for path in required):
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_count = len(job["plan"])
    if (
        not summary.get("complete")
        or summary.get("split") != job["split"]
        or int(summary.get("count", -1)) != expected_count
    ):
        return False
    expected_shapes = {
        "s11.npy": (expected_count, int(cfg["frequency"]["points"])),
        "inputs.npy": (expected_count, 4, int(cfg["response"]["points"])),
        "targets.npy": (expected_count, 3, int(cfg["response"]["points"])),
    }
    for name, shape in expected_shapes.items():
        values = np.load(folder / name, mmap_mode="r")
        valid = values.shape == shape
        mmap = getattr(values, "_mmap", None)
        if mmap is not None:
            mmap.close()
        if not valid:
            return False
    with (folder / "metadata.jsonl").open("r", encoding="utf-8") as handle:
        if sum(1 for _ in handle) != expected_count:
            return False
    return True


def _percentiles(values: list[float]) -> dict[str, float]:
    quantiles = np.percentile(np.asarray(values, dtype=float), [0, 50, 95, 100])
    return {
        "min": float(quantiles[0]),
        "median": float(quantiles[1]),
        "q95": float(quantiles[2]),
        "max": float(quantiles[3]),
    }


def _write_shard(root_string: str, cfg: dict, job: dict) -> dict:
    root = Path(root_string)
    folder = _shard_path(root, job)
    folder.mkdir(parents=True, exist_ok=True)
    rows = job["plan"]
    sample_count = len(rows)
    frequency_count = int(cfg["frequency"]["points"])
    distance_count = int(cfg["response"]["points"])
    s11_values = np.empty((sample_count, frequency_count), dtype=np.complex64)
    inputs = np.empty((sample_count, 4, distance_count), dtype=np.float32)
    targets = np.empty((sample_count, 3, distance_count), dtype=np.float32)
    physical_max, measured_max, physical_rms, measured_rms = [], [], [], []
    physical_passive = []
    templates = load_case_templates()

    metadata_path = folder / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8", newline="\n") as metadata_file:
        for row_index, item in enumerate(rows):
            s11, sample_inputs, sample_targets, metadata = generate_sample(item, cfg, templates)
            if sample_inputs.shape != (4, distance_count):
                raise ValueError(
                    f"input shape for sample {item['sample_id']} is {sample_inputs.shape}"
                )
            if sample_targets.shape != (3, distance_count):
                raise ValueError(
                    f"target shape for sample {item['sample_id']} is {sample_targets.shape}"
                )
            s11_values[row_index] = s11
            inputs[row_index] = sample_inputs
            targets[row_index] = sample_targets
            metadata_file.write(
                json.dumps(metadata, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                + "\n"
            )
            diagnostic = metadata["s11_diagnostics"]
            physical_max.append(diagnostic["physical_max_abs"])
            measured_max.append(diagnostic["measured_max_abs"])
            physical_rms.append(diagnostic["physical_rms_abs"])
            measured_rms.append(diagnostic["measured_rms_abs"])
            physical_passive.append(diagnostic["physical_passive_with_1e-6_tolerance"])

    np.save(folder / "s11.npy", s11_values)
    np.save(folder / "inputs.npy", inputs)
    np.save(folder / "targets.npy", targets)
    summary = {
        "schema": "fieldnn-shard-summary-1",
        "complete": True,
        "split": job["split"],
        "shard_index": int(job["shard_index"]),
        "count": sample_count,
        "sample_id_min": min(int(item["sample_id"]) for item in rows),
        "sample_id_max": max(int(item["sample_id"]) for item in rows),
        "physical_s11_max_abs": _percentiles(physical_max),
        "measured_s11_max_abs": _percentiles(measured_max),
        "physical_s11_rms_abs": _percentiles(physical_rms),
        "measured_s11_rms_abs": _percentiles(measured_rms),
        "physical_nonpassive_count": int(sum(not value for value in physical_passive)),
    }
    write_json(folder / "summary.json", summary)
    return {
        "path": str(folder),
        "split": job["split"],
        "count": sample_count,
        "shard_index": int(job["shard_index"]),
        "summary": summary,
    }


def _effective_config(args) -> dict:
    cfg = load_config(args.config)
    cfg["physical_model"] = load_physics_config()
    dataset = cfg["dataset"]
    if args.samples is not None:
        dataset["samples"] = int(args.samples)
    if args.workers is not None:
        dataset["workers"] = int(args.workers)
    if args.shard_size is not None:
        dataset["shard_size"] = int(args.shard_size)
    if int(dataset["samples"]) < 1:
        raise ValueError("samples must be positive")
    if int(dataset["workers"]) < 1:
        raise ValueError("workers must be positive")
    if int(dataset["shard_size"]) < 1:
        raise ValueError("shard-size must be positive")
    return cfg


def main():
    args = parse_args()
    cfg = _effective_config(args)
    root = Path(args.output).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    effective_config_path = root / "config.json"
    if effective_config_path.exists():
        current = json.loads(effective_config_path.read_text(encoding="utf-8"))
        if _canonical_config(current) != _canonical_config(cfg):
            raise ValueError(
                f"output config differs from requested config: {effective_config_path}"
            )
    else:
        write_json(effective_config_path, cfg)

    frequency = frequency_grid(cfg)
    distance = distance_grid(cfg)
    np.save(root / "frequency.npy", frequency)
    np.save(root / "distance.npy", distance)

    plan = sample_plan(int(cfg["dataset"]["samples"]), cfg)
    jobs = _make_shard_jobs(plan, int(cfg["dataset"]["shard_size"]))
    results = []
    todo = []
    for job in jobs:
        if args.resume and _shard_complete(root, job, cfg):
            folder = _shard_path(root, job)
            summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            results.append({
                "path": str(folder),
                "split": job["split"],
                "count": len(job["plan"]),
                "shard_index": int(job["shard_index"]),
                "summary": summary,
            })
        else:
            todo.append(job)

    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=int(cfg["dataset"]["workers"])) as executor:
        futures = {
            executor.submit(_write_shard, str(root), cfg, job): job
            for job in todo
        }
        with tqdm(total=len(futures), desc="生成分片", unit="shard") as progress:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                progress.update(1)
                elapsed = time.perf_counter() - started
                progress.set_postfix(
                    split=result["split"],
                    shard=f"{result['shard_index']:05d}",
                    seconds=f"{elapsed:.1f}",
                )

    results.sort(key=lambda item: (SPLITS.index(item["split"]), item["shard_index"]))
    counts = {split: 0 for split in SPLITS}
    manifest_shards = []
    for result in results:
        counts[result["split"]] += int(result["count"])
        relative = Path(result["path"]).relative_to(root)
        manifest_shards.append({
            "path": relative.as_posix(),
            "split": result["split"],
            "count": int(result["count"]),
        })
    manifest = {
        "schema": "fieldnn-dataset-1",
        "shards": manifest_shards,
        "counts": counts,
    }
    write_json(root / "manifest.json", manifest)
    elapsed = time.perf_counter() - started
    print(
        f"完成：{sum(counts.values())} samples "
        f"(train={counts['train']}, val={counts['val']}, test={counts['test']}) "
        f"in {elapsed:.1f}s; root={root}"
    )


if __name__ == "__main__":
    main()
