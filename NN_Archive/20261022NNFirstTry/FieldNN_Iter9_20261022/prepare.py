"""Prepare proposal-centered synthetic, field, and calibration patch caches."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from proposals import extract_patch, propose


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
ITER3_DIR = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_Iter3_20261022"
ITER4_DIR = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_Iter4_20261022"
ITER6_DIR = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_Iter6_20261022"
ITER8_DIR = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_Iter8_20261022"
MANIFEST_PATH = (
    PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_20261022"
    / "output" / "real_manifest.jsonl"
)
RULES_PATH = (
    PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_20261022"
    / "real_group_rules.json"
)
SYNTHETIC_ROOT = Path(r"E:\CableNNDataSet\CodeX\FieldNN_Iter4_20261022")
FIELD_CACHE_ROOT = Path(r"E:\CableNNDataSet\CodeX\FieldNN_Iter8_20261022")
DEFAULT_OUTPUT = Path(r"E:\CableNNDataSet\CodeX\FieldNN_Iter9_20261022")
SPLITS = ("train", "val", "test")
FEATURE_CHANNELS = (0, 1, 4, 5, 6, 8, 9, 10, 11)
PATCH_SHAPE = (20, 256)
SOURCE_RADIUS_M = 20.0
SYNTHETIC_POSITIVE_LIMIT = 4
SYNTHETIC_NEGATIVE_LIMIT = 4


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(
                json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                + "\n"
            )


def norm_path(value: str) -> str:
    return str(value).replace("\\", "/").strip("/").casefold()


def curve_key(record: dict[str, Any]) -> tuple[str, str]:
    return norm_path(str(record["source_relative_path"])), str(record["sweep_rowid"])


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class PatchWriter:
    """Stream float16 patches to disk, then wrap them in an exact-size NPY file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.raw_path = path.with_name(f".{path.name}.raw")
        self.stream = self.raw_path.open("wb")
        self.count = 0

    def append(self, patch: np.ndarray) -> None:
        np.asarray(patch, dtype=np.float16).tofile(self.stream)
        self.count += 1

    def finish(self) -> int:
        self.stream.close()
        if not self.count:
            np.save(self.path, np.empty((0, *PATCH_SHAPE), dtype=np.float16))
            self.raw_path.unlink(missing_ok=True)
            return 0

        source = np.memmap(
            self.raw_path,
            dtype=np.float16,
            mode="r",
            shape=(self.count, *PATCH_SHAPE),
        )
        destination = np.lib.format.open_memmap(
            self.path,
            mode="w+",
            dtype=np.float16,
            shape=(self.count, *PATCH_SHAPE),
        )
        for start in range(0, self.count, 1024):
            destination[start : start + 1024] = source[start : start + 1024]
        destination.flush()
        destination._mmap.close()
        source._mmap.close()
        del destination, source
        self.raw_path.unlink()
        return self.count


def load_iter8_modules():
    if str(ITER8_DIR) not in sys.path:
        sys.path.insert(0, str(ITER8_DIR))
    if str(ITER3_DIR) not in sys.path:
        sys.path.append(str(ITER3_DIR))
    features = load_module(ITER8_DIR / "features.py", "features")
    data = load_module(ITER8_DIR / "data.py", "iter9_iter8_data")
    field_prepare = load_module(
        ITER8_DIR / "prepare_field.py", "iter9_iter8_prepare_field"
    )
    group_prepare = load_module(
        ITER6_DIR / "prepare_field.py", "iter9_iter6_prepare_field"
    )
    return features, data, field_prepare, group_prepare


def _center_index(distance: np.ndarray, center: float) -> int:
    spacing = float(distance[1] - distance[0])
    index = int(round((float(center) - float(distance[0])) / spacing))
    return min(max(index, 0), distance.size - 1)


def synthetic_labels(
    centers: np.ndarray,
    distance: np.ndarray,
    targets: np.ndarray,
    event_positions: list[float],
    terminal_position: float | None,
) -> list[tuple[float, int]]:
    events = np.asarray(event_positions, dtype=np.float64)
    positive = [
        (float(center), float(np.min(np.abs(events - center))))
        for center in centers
        if events.size and float(np.min(np.abs(events - center))) <= SOURCE_RADIUS_M
    ]
    positive.sort(key=lambda item: (item[1], item[0]))
    kept_positive = {center for center, _ in positive[:SYNTHETIC_POSITIVE_LIMIT]}

    terminal_negative: float | None = None
    if terminal_position is not None:
        terminal_candidates = [
            float(center)
            for center in centers
            if abs(float(center) - terminal_position) <= SOURCE_RADIUS_M
            and (not events.size or float(np.min(np.abs(events - center))) > SOURCE_RADIUS_M)
            and targets[0, _center_index(distance, float(center))] >= 0.0
            and targets[2, _center_index(distance, float(center))] >= 0.0
        ]
        if terminal_candidates:
            terminal_negative = min(
                terminal_candidates,
                key=lambda center: (abs(center - terminal_position), center),
            )

    negative_candidates: list[tuple[float, float]] = []
    for center_value in centers:
        center = float(center_value)
        if center in kept_positive:
            continue
        distances = [abs(center - float(event)) for event in events]
        if terminal_position is not None:
            distances.append(abs(center - terminal_position))
        if distances and not all(value > 40.0 for value in distances):
            continue
        if targets[0, _center_index(distance, center)] < 0.0:
            continue
        nearest_known = min(distances) if distances else float("inf")
        negative_candidates.append((center, nearest_known))

    negative_candidates.sort(key=lambda item: (item[1], item[0]))
    kept_negative: list[float] = []
    if terminal_negative is not None:
        kept_negative.append(terminal_negative)
    for center, _ in negative_candidates:
        if center not in kept_negative:
            kept_negative.append(center)
        if len(kept_negative) == SYNTHETIC_NEGATIVE_LIMIT:
            break

    labeled = [(center, 1) for center in kept_positive]
    labeled.extend((center, 0) for center in kept_negative)
    return sorted(labeled, key=lambda item: item[0])


def prepare_synthetic(
    output: Path,
    data_module,
    workers: int,
    smoke: bool,
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        dataset = data_module.ShardDataset(SYNTHETIC_ROOT, split, train=False)
        dataset.view = "mixed"
        dataset._open_arrays()
        dataset._open_calibration_bank()
        metadata = dataset.metadata
        sample_count = min(len(dataset), 8) if smoke else len(dataset)
        patch_writer = PatchWriter(output / f"synthetic_{split}_patches.npy")
        labels: list[int] = []

        def process_curve(index: int) -> tuple[int, list[tuple[np.ndarray, int]]]:
            features, targets, curve_index = dataset[index]
            centers = propose(features, dataset.distance)
            row = metadata[int(curve_index)]
            terminal = row.get("terminal_position_m")
            if terminal is not None and float(terminal) <= 0.0:
                terminal = None
            elif terminal is not None:
                terminal = float(terminal)
            selected = synthetic_labels(
                centers,
                dataset.distance,
                targets,
                [float(value) for value in row.get("event_positions_m", [])],
                terminal,
            )
            patches = [
                (extract_patch(features, dataset.distance, center), label)
                for center, label in selected
            ]
            return int(curve_index), patches

        chunk_size = workers * 4
        with ThreadPoolExecutor(max_workers=workers) as pool:
            progress = tqdm(
                total=sample_count,
                desc=f"Synthetic {split} proposals",
                unit="curve",
            )
            for start in range(0, sample_count, chunk_size):
                indices = range(start, min(start + chunk_size, sample_count))
                for _, selected in pool.map(process_curve, indices):
                    for patch, label in selected:
                        patch_writer.append(patch)
                        labels.append(label)
                    progress.update(1)
            progress.close()

        patch_count = patch_writer.finish()
        np.save(output / f"synthetic_{split}_labels.npy", np.asarray(labels, dtype=np.int8))
        counts[split] = {
            "curves": sample_count,
            "patches": patch_count,
            "positive": int(sum(labels)),
            "negative": int(len(labels) - sum(labels)),
        }
    return counts


def _field_record(
    row: dict[str, Any],
    index: int,
    group: str,
    split: str,
    cached: dict[str, Any] | None,
    source_velocity: float | None,
    nominal_velocity_m_per_us: float,
) -> dict[str, Any]:
    record = dict(cached or {})
    if source_velocity is not None:
        source_velocity = float(source_velocity)
    if source_velocity is None and record.get("source_wave_velocity_m_per_us") is not None:
        source_velocity = float(record["source_wave_velocity_m_per_us"])
    ratio = record.get("source_scale_ratio")
    if ratio is None:
        ratio = (
            float(source_velocity) / nominal_velocity_m_per_us
            if source_velocity is not None
            else 1.0
        )
    record.update(
        {
            "index": index,
            "source_relative_path": str(row["source_relative_path"]),
            "sweep_rowid": int(row["sweep_rowid"]),
            "sweep_name": str(row.get("sweep_name") or ""),
            "phase": str(row.get("phase") or "unknown"),
            "cable_group": str(row.get("cable_group") or group),
            "case_group": str(row.get("source_case_group") or row.get("case_group") or ""),
            "group": group,
            "split": split,
            "source_wave_velocity_m_per_us": source_velocity,
            "source_scale_ratio": float(ratio),
        }
    )
    record.setdefault("source_axis_status", "source_report" if source_velocity is not None else "nominal_fallback")
    record.setdefault("positions_source_m", [])
    record.setdefault("positions_nominal_m", [])
    record.setdefault("position_windows_source_m", [])
    record.setdefault("position_windows_nominal_m", [])
    record.setdefault("label_claims", [])
    record.setdefault("known_lab", False)
    record.setdefault("unknown_length", True)
    record.setdefault("known_length_m", None)
    record.setdefault(
        "label_scope",
        "positive-only bag supervision; unannotated locations are not negatives",
    )
    return record


def prepare_field(
    output: Path,
    workers: int,
    smoke: bool,
    field_prepare,
    group_prepare,
) -> tuple[dict[str, int], dict[str, Any]]:
    manifest = read_jsonl(MANIFEST_PATH)
    iter4_calibration_sources = read_jsonl(SYNTHETIC_ROOT / "calibration_sources_v2.jsonl")
    case_to_group, group_rows = group_prepare.build_groups(manifest, RULES_PATH)
    split_for_group, forced_reason = group_prepare.assign_splits(
        manifest, case_to_group, group_rows, iter4_calibration_sources
    )
    row_groups: list[str] = []
    row_splits: list[str] = []
    for row in manifest:
        case = str(row.get("source_case_group") or row.get("case_group") or "")
        group = case_to_group[case]
        row_groups.append(group)
        row_splits.append(split_for_group[group])

    selected_groups: set[str] | None = None
    if smoke:
        selected_groups = set()
        for split in SPLITS:
            groups = sorted(group for group, value in split_for_group.items() if value == split)
            selected_groups.update(groups[:2])
    selected_manifest_indices = [
        index
        for index, group in enumerate(row_groups)
        if selected_groups is None or group in selected_groups
    ]

    cache_records = read_jsonl(FIELD_CACHE_ROOT / "field_records.jsonl")
    cache_by_key = {curve_key(record): record for record in cache_records}
    cache_features = np.load(FIELD_CACHE_ROOT / "field_features.npy", mmap_mode="r")
    feature_cfg = json.loads((field_prepare.CONFIG_PATH).read_text(encoding="utf-8"))
    nominal_velocity = float(feature_cfg["response"]["velocity_m_s"]) / 1e6

    feature_by_manifest_index: dict[int, tuple[np.ndarray, float | None]] = {}
    missing_indices: list[int] = []
    for manifest_index in selected_manifest_indices:
        row = manifest[manifest_index]
        cached = cache_by_key.get(curve_key(row))
        if cached is None:
            missing_indices.append(manifest_index)
        else:
            feature_by_manifest_index[manifest_index] = (
                np.asarray(cache_features[int(cached["index"])], dtype=np.float32),
                cached.get("source_wave_velocity_m_per_us"),
            )

    if missing_indices:
        def read_one(index: int):
            return field_prepare.read_feature(index, manifest[index], feature_cfg)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, feature, source_velocity in tqdm(
                pool.map(read_one, missing_indices),
                total=len(missing_indices),
                desc="FDR -> missing Iter8 field features",
                unit="curve",
            ):
                feature_by_manifest_index[int(index)] = (feature, source_velocity)

    manifest_to_curve: dict[int, int] = {
        manifest_index: curve_index
        for curve_index, manifest_index in enumerate(selected_manifest_indices)
    }
    field_records: list[dict[str, Any]] = []
    feature_rows: list[np.ndarray] = []
    for curve_index, manifest_index in enumerate(selected_manifest_indices):
        row = manifest[manifest_index]
        cached = cache_by_key.get(curve_key(row))
        feature, source_velocity = feature_by_manifest_index[manifest_index]
        record = _field_record(
            row,
            curve_index,
            row_groups[manifest_index],
            row_splits[manifest_index],
            cached,
            source_velocity,
            nominal_velocity,
        )
        field_records.append(record)
        feature_rows.append(feature)

    cache_index_to_manifest: dict[int, int] = {}
    manifest_index_by_key = {curve_key(row): index for index, row in enumerate(manifest)}
    for record in cache_records:
        cache_index_to_manifest[int(record["index"])] = manifest_index_by_key[curve_key(record)]

    cache_bags = json.loads((FIELD_CACHE_ROOT / "field_bags.json").read_text(encoding="utf-8"))
    field_bags: list[dict[str, Any]] = []
    for source_bag in cache_bags["bags"]:
        group = str(source_bag["group"])
        if selected_groups is not None and group not in selected_groups:
            continue
        bag = dict(source_bag)
        bag["split"] = split_for_group[group]
        remapped = []
        for old_index in source_bag["indices"]:
            manifest_index = cache_index_to_manifest[int(old_index)]
            remapped.append(manifest_to_curve[manifest_index])
        bag["indices"] = remapped
        field_bags.append(bag)

    field_candidates: list[dict[str, Any]] = []
    field_positive_bags: list[dict[str, Any]] = []
    candidate_indices_by_curve: dict[int, list[tuple[int, float]]] = defaultdict(list)
    patch_writer = PatchWriter(output / "field_patches.npy")
    distance = np.linspace(
        0.0,
        float(feature_cfg["response"]["distance_max_m"]),
        int(feature_cfg["response"]["points"]),
        dtype=np.float64,
    )
    for curve_index, (record, feature) in enumerate(
        tqdm(zip(field_records, feature_rows), total=len(field_records), desc="Field candidates", unit="curve")
    ):
        centers = propose(feature, distance)
        ratio = float(record["source_scale_ratio"])
        for center in centers:
            candidate_index = len(field_candidates)
            source_distance = float(center) * ratio
            field_candidates.append(
                {
                    "index": candidate_index,
                    "curve_index": curve_index,
                    "distance_m": source_distance,
                    "nominal_distance_m": float(center),
                }
            )
            candidate_indices_by_curve[curve_index].append((candidate_index, source_distance))
            patch_writer.append(extract_patch(feature, distance, float(center)))

    field_patch_count = patch_writer.finish()
    for bag in field_bags:
        for position in bag.get("positions_source_m", []):
            matched = sorted(
                candidate_index
                for curve_index in bag["indices"]
                for candidate_index, candidate_distance in candidate_indices_by_curve[curve_index]
                if abs(candidate_distance - float(position)) <= SOURCE_RADIUS_M
            )
            field_positive_bags.append(
                {
                    "indices": matched,
                    "split": bag["split"],
                    "group": bag["group"],
                    "position_source_m": float(position),
                    "proposal_missing": not bool(matched),
                }
            )

    write_jsonl(output / "field_candidates.jsonl", field_candidates)
    write_jsonl(output / "field_records.jsonl", field_records)
    write_json(output / "field_bags.json", {"bags": field_bags})
    write_json(output / "positive_bags.json", field_positive_bags)

    return (
        {
            "source_curves_full": len(manifest),
            "source_cache_records_available": len(cache_records),
            "source_curves_cached": len(selected_manifest_indices) - len(missing_indices),
            "source_curves_recomputed": len(missing_indices),
            "source_curves_output": len(field_records),
            "field_candidates": len(field_candidates),
            "field_patches": field_patch_count,
            "field_bags": len(field_bags),
            "positive_bag_positions": len(field_positive_bags),
            "positive_positions_missing_proposals": sum(
                bool(item["proposal_missing"]) for item in field_positive_bags
            ),
        },
        {"group_split": split_for_group, "forced_group_reasons": forced_reason},
    )


def prepare_calibration(
    output: Path,
    iter8_features,
    smoke: bool,
) -> dict[str, int]:
    source_records = read_jsonl(FIELD_CACHE_ROOT / "calibration_records.jsonl")
    selected_indices = list(range(len(source_records)))
    if smoke:
        selected_indices = []
        for split in SPLITS:
            selected_indices.extend(
                [
                    index
                    for index, record in enumerate(source_records)
                    if record["split"] == split
                ][:2]
            )

    raw_imp = np.load(FIELD_CACHE_ROOT / "calibration_raw_imp.npy", mmap_mode="r")
    raw_step = np.load(FIELD_CACHE_ROOT / "calibration_raw_step.npy", mmap_mode="r")
    cfg = json.loads((ITER4_DIR / "config.json").read_text(encoding="utf-8"))
    response = cfg["response"]
    distance = np.linspace(
        0.0,
        float(response["distance_max_m"]),
        int(response["points"]),
        dtype=np.float64,
    )
    records: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    patch_writer = PatchWriter(output / "calibration_patches.npy")
    for new_index, old_index in enumerate(
        tqdm(selected_indices, desc="Calibration candidates", unit="curve")
    ):
        source_record = source_records[old_index]
        record = dict(source_record)
        record["index"] = new_index
        records.append(record)
        feature = iter8_features.distance_features(
            np.asarray(raw_imp[old_index], dtype=np.float32),
            np.asarray(raw_step[old_index], dtype=np.float32),
            distance,
        )
        for center in propose(feature, distance):
            candidates.append(
                {
                    "index": len(candidates),
                    "curve_index": new_index,
                    "distance_m": float(center),
                    "nominal_distance_m": float(center),
                }
            )
            patch_writer.append(extract_patch(feature, distance, float(center)))

    patch_count = patch_writer.finish()
    write_jsonl(output / "calibration_candidates.jsonl", candidates)
    write_jsonl(output / "calibration_records.jsonl", records)
    return {
        "calibration_sources": len(records),
        "calibration_candidates": len(candidates),
        "calibration_patches": patch_count,
        "sources_without_proposals": sum(
            not any(row["curve_index"] == index for row in candidates)
            for index in range(len(records))
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    iter8_features, data_module, field_prepare, group_prepare = load_iter8_modules()
    write_json(
        output / "config.json",
        {
            "schema": "fieldnn-iter9-patch-cache-1",
            "mode": "smoke" if args.smoke else "full",
            "synthetic_source": str(SYNTHETIC_ROOT),
            "field_source": str(MANIFEST_PATH),
            "field_cache_read_only": str(FIELD_CACHE_ROOT),
            "model_axis_m": [0.0, 3250.0],
            "model_axis_points": 8192,
            "physical_scope_max_m": 2500.0,
            "field_distance_axis": "source/report = nominal_distance_m * source_scale_ratio",
            "feature_channels": list(FEATURE_CHANNELS),
            "patch_shape": list(PATCH_SHAPE),
            "view_half_widths_m": [50.0, 200.0],
            "view_points": PATCH_SHAPE[1],
            "proposal_maps": [
                "abs feature4 pulse/local RMS30",
                "abs feature0 pulse/local RMS90",
                "abs feature8 step detrend90",
                "abs feature10 step detrend300",
            ],
            "proposal_rules": {
                "center_range_nominal_m": [30.0, 3240.0],
                "minimum_separation_m": 10.0,
                "top_peaks_per_map": 20,
                "label_independent_on_field_validation_and_test": True,
            },
            "synthetic_label_rules": {
                "positive_within_m": SOURCE_RADIUS_M,
                "negative_beyond_events_and_terminal_m": 40.0,
                "event_target_at_negative_center_must_be_nonnegative": True,
                "max_positive_per_curve": SYNTHETIC_POSITIVE_LIMIT,
                "max_negative_per_curve": SYNTHETIC_NEGATIVE_LIMIT,
                "terminal_near_negative_limit": 1,
                "view": "deterministic Iter5 mixed view",
            },
            "field_label_rules": {
                "positive_radius_source_m": SOURCE_RADIUS_M,
                "unannotated_locations_are_negative": False,
                "split": "Iter6 connected case/cable/source groups",
            },
            "workers": args.workers,
            "smoke": {
                "enabled": bool(args.smoke),
                "synthetic_curves_per_split": 8 if args.smoke else None,
                "field_groups_per_split": 2 if args.smoke else None,
                "calibration_sources_per_split": 2 if args.smoke else None,
            },
        },
    )

    synthetic_summary = prepare_synthetic(output, data_module, args.workers, args.smoke)
    field_summary, split_audit = prepare_field(
        output,
        args.workers,
        args.smoke,
        field_prepare,
        group_prepare,
    )
    calibration_summary = prepare_calibration(output, iter8_features, args.smoke)

    summary = {
        "schema": "fieldnn-iter9-patch-cache-summary-1",
        "status": "complete",
        "mode": "smoke" if args.smoke else "full",
        "output": str(output),
        "workers": args.workers,
        "synthetic": synthetic_summary,
        "field": field_summary,
        "calibration": calibration_summary,
        "field_group_split": split_audit["group_split"],
        "forced_group_reasons": split_audit["forced_group_reasons"],
        "field_label_policy": "Known annotation positions are positive-only evidence; missing proposals are retained in positive_bags.json; unknown candidates are not negative labels.",
        "calibration_label_policy": "Every cached CalibrationData curve is a confirmed short/no-cable fixture reference; candidate patches are calibration negatives only.",
    }
    write_json(output / "prepare_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
