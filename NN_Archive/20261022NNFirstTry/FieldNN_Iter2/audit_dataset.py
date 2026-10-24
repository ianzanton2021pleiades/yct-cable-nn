"""Audit an existing FieldNN manifest and its generated NumPy shards."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_DATA = Path(r"E:\CableNNDataSet\CodeX\FieldNN_20261022")
DEFAULT_OUTPUT = HERE / "output" / "audit"
SPLITS = ("train", "val", "test")
MEASURED_SUMMARY = HERE.parents[2] / "DifferentAgent" / "CodeX" / "Core16_Field" / "output" / "short_scope_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.max_samples < 1:
        parser.error("--max-samples must be at least 1")
    return args


def _quantiles(values: list[float] | np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"n": 0, "min": None, "q05": None, "q50": None, "q95": None, "max": None}
    q05, q50, q95 = np.quantile(array, [0.05, 0.50, 0.95])
    return {
        "n": int(array.size),
        "min": float(np.min(array)),
        "q05": float(q05),
        "q50": float(q50),
        "q95": float(q95),
        "max": float(np.max(array)),
    }


def _shard_folder(root: Path, relative: str) -> Path:
    return root.joinpath(*str(relative).replace("\\", "/").split("/"))


def _read_dataset(root: Path) -> tuple[dict, dict, np.ndarray, np.ndarray, list[dict], list[str]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
    frequency = np.load(root / "frequency.npy", allow_pickle=False)
    distance = np.load(root / "distance.npy", allow_pickle=False)
    entries: list[dict] = []
    issues: list[str] = []

    for shard in manifest["shards"]:
        folder = _shard_folder(root, shard["path"])
        metadata_path = folder / "metadata.jsonl"
        if not metadata_path.is_file():
            issues.append(f"missing metadata file: {metadata_path}")
            continue
        metadata_rows = [
            json.loads(line)
            for line in metadata_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        expected_count = int(shard["count"])
        if len(metadata_rows) != expected_count:
            issues.append(
                f"{shard['path']}: manifest count={expected_count}, "
                f"metadata rows={len(metadata_rows)}"
            )
        summary_path = folder / "summary.json"
        shard_summary = (
            json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_path.is_file()
            else None
        )
        for row_index, metadata in enumerate(metadata_rows):
            entries.append({
                "split": str(shard["split"]),
                "shard_path": str(shard["path"]),
                "folder": folder,
                "row_index": row_index,
                "metadata": metadata,
                "shard_summary": shard_summary,
            })

    actual_counts = Counter(entry["split"] for entry in entries)
    for split in SPLITS:
        manifest_count = int(manifest.get("counts", {}).get(split, 0))
        if actual_counts[split] != manifest_count:
            issues.append(
                f"{split}: manifest count={manifest_count}, metadata count={actual_counts[split]}"
            )
    unexpected_splits = sorted(set(actual_counts) - set(SPLITS))
    if unexpected_splits:
        issues.append(f"unexpected split names: {unexpected_splits}")
    if frequency.ndim != 1 or distance.ndim != 1:
        issues.append("frequency.npy and distance.npy must be one-dimensional")
    return manifest, cfg, frequency, distance, entries, issues


def _balanced_quotas(entries_by_split: dict[str, list[dict]], limit: int) -> dict[str, int]:
    quotas = {split: 0 for split in SPLITS}
    remaining = min(limit, sum(len(entries_by_split[split]) for split in SPLITS))
    while remaining:
        progressed = False
        for split in SPLITS:
            if quotas[split] < len(entries_by_split[split]):
                quotas[split] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break
    return quotas


def _uniform_indices(size: int, count: int) -> list[int]:
    if count == 0:
        return []
    if count == 1:
        return [(size - 1) // 2]
    return np.rint(np.linspace(0, size - 1, count)).astype(np.int64).tolist()


def _select_entries(entries: list[dict], limit: int) -> tuple[list[dict], dict[str, int]]:
    by_split = {split: [] for split in SPLITS}
    for entry in entries:
        if entry["split"] in by_split:
            by_split[entry["split"]].append(entry)
    quotas = _balanced_quotas(by_split, limit)
    selected = []
    for split in SPLITS:
        selected.extend(by_split[split][index] for index in _uniform_indices(
            len(by_split[split]), quotas[split]
        ))
    return selected, quotas


def _plot_entries(selected: list[dict]) -> list[dict]:
    ordered = sorted(
        selected,
        key=lambda entry: (
            float(entry["metadata"].get("length_m", math.inf)),
            entry["split"],
            int(entry["metadata"].get("sample_id", -1)),
        ),
    )
    if len(ordered) <= 4:
        return ordered
    midpoint = len(ordered) // 2
    short_group = ordered[:midpoint]
    long_group = ordered[midpoint:]
    short_indices = _uniform_indices(len(short_group), min(2, len(short_group)))
    long_indices = _uniform_indices(len(long_group), min(2, len(long_group)))
    return (
        [short_group[index] for index in short_indices]
        + [long_group[index] for index in long_indices]
    )


def _metadata_coverage(entries: list[dict], cfg: dict, manifest: dict) -> dict[str, Any]:
    by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    missing_topology_by_split: Counter = Counter()
    lengths: list[float] = []
    joint_counts: Counter = Counter()
    event_counts: list[int] = []
    interval_counts: Counter = Counter()
    case_ids: Counter = Counter()
    case_titles: Counter = Counter()
    categories: Counter = Counter()
    interval_mechanisms: Counter = Counter()
    semantics: Counter = Counter()

    for entry in entries:
        meta = entry["metadata"]
        topology = meta.get("topology_id")
        if topology is None or not str(topology):
            missing_topology_by_split[entry["split"]] += 1
        else:
            by_split[entry["split"]].add(str(topology))
        if "length_m" in meta:
            lengths.append(float(meta["length_m"]))
        joint_counts[str(len(meta.get("joint_positions_m", [])))] += 1
        event_positions = meta.get("event_positions_m", [])
        event_counts.append(len(event_positions))
        intervals = meta.get("defect_intervals_m", [])
        interval_counts[str(len(intervals))] += 1
        for key, counter in (
            ("case_id", case_ids),
            ("case_title", case_titles),
            ("category", categories),
            ("defect_label_semantics", semantics),
        ):
            if meta.get(key) is not None:
                counter[str(meta[key])] += 1
        for detail in meta.get("defect_interval_details", []):
            mechanism = detail.get("case_category")
            if mechanism is not None:
                interval_mechanisms[str(mechanism)] += 1

    pairwise = {}
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1:]:
            overlap = sorted(by_split[left] & by_split[right])
            pairwise[f"{left}__{right}"] = {
                "intersection_count": len(overlap),
                "examples": overlap[:10],
            }

    length_bins = []
    configured_bins = cfg.get("randomization", {}).get("length_bins_m", [])
    for index, bounds in enumerate(configured_bins):
        lower, upper = (float(value) for value in bounds)
        is_last = index == len(configured_bins) - 1
        count = sum(
            lower <= length <= upper if is_last else lower <= length < upper
            for length in lengths
        )
        length_bins.append({
            "lower_m": lower,
            "upper_m": upper,
            "upper_inclusive": is_last,
            "sample_count": int(count),
        })

    manifest_counts = manifest.get("counts", {})
    actual_counts = Counter(entry["split"] for entry in entries)
    return {
        "sample_count_by_split": {
            split: int(actual_counts[split]) for split in SPLITS
        },
        "manifest_count_by_split": {
            split: int(manifest_counts.get(split, 0)) for split in SPLITS
        },
        "topology_unique_count_by_split": {
            split: len(by_split[split]) for split in SPLITS
        },
        "missing_topology_id_count_by_split": {
            split: int(missing_topology_by_split[split]) for split in SPLITS
        },
        "topology_pairwise_split_intersections": pairwise,
        "topology_id_semantics_note": (
            "Current sample_plan assigns a unique topology_id per sample_id; "
            "zero intersection verifies ID separation, not physical topology or "
            "mechanism equivalence."
        ),
        "length_m_distribution": _quantiles(lengths),
        "configured_length_bin_counts": length_bins,
        "joint_count_per_sample": dict(sorted(
            joint_counts.items(), key=lambda item: int(item[0])
        )),
        "event_count_per_sample": _quantiles(event_counts),
        "defect_interval_count_per_sample": dict(sorted(
            interval_counts.items(), key=lambda item: int(item[0])
        )),
        "case_id_sample_counts": dict(sorted(case_ids.items())),
        "case_title_sample_counts": dict(sorted(case_titles.items())),
        "category_sample_counts": dict(sorted(categories.items())),
        "defect_mechanism_interval_counts": dict(sorted(interval_mechanisms.items())),
        "defect_label_semantics_counts": dict(sorted(semantics.items())),
    }


def _validate_coordinates(
    metadata: dict,
    targets: np.ndarray,
    distance: np.ndarray,
    event_sigma_m: float,
) -> list[str]:
    issues: list[str] = []
    length_m = float(metadata["length_m"])
    terminal = float(metadata["terminal_position_m"])
    events = [float(value) for value in metadata.get("event_positions_m", [])]
    intervals = [
        (float(value[0]), float(value[1]))
        for value in metadata.get("defect_intervals_m", [])
    ]

    if not (distance[0] <= length_m <= distance[-1]):
        issues.append("cable length is outside the configured response grid")
    if not math.isclose(terminal, length_m, rel_tol=0.0, abs_tol=1e-5):
        issues.append("terminal_position_m differs from physical length_m")
    if any(not (0.0 <= position <= length_m) for position in events):
        issues.append("an event coordinate is outside the physical cable")
    if any(not (0.0 <= start <= end <= length_m) for start, end in intervals):
        issues.append("a defect interval is invalid or outside the physical cable")
    if any(math.isclose(position, terminal, rel_tol=0.0, abs_tol=1e-5) for position in events):
        issues.append("terminal position is duplicated in event_positions_m")

    source_events = [
        float(position)
        for position in metadata.get("joint_positions_m", [])
    ]
    for start, end in intervals:
        source_events.extend((start, end))
    expected_events = sorted({
        round(position, 6)
        for position in source_events
        if abs(position - terminal) > 1.0e-5
    })
    actual_events = sorted({round(position, 6) for position in events})
    if expected_events != actual_events:
        issues.append("event_positions_m disagrees with joint and interval boundary coordinates")

    expected = np.zeros_like(targets, dtype=np.float64)
    for position in events:
        expected[0] = np.maximum(
            expected[0],
            np.exp(-0.5 * ((distance - position) / event_sigma_m) ** 2),
        )
    for start, end in intervals:
        expected[1, (distance >= start) & (distance <= end)] = 1.0
    expected[2] = np.exp(-0.5 * ((distance - terminal) / event_sigma_m) ** 2)
    if not np.allclose(targets, expected, rtol=1e-5, atol=2e-6):
        issues.append("target tracks do not match event/interval/terminal metadata")
    return issues


def _selective_data_checks(
    selected: list[dict],
    frequency: np.ndarray,
    distance: np.ndarray,
    cfg: dict,
    plot_entries: list[dict],
) -> tuple[dict[str, Any], list[str], dict[tuple[str, int], dict]]:
    array_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    plot_keys = {
        (entry["split"], int(entry["metadata"]["sample_id"]))
        for entry in plot_entries
    }
    plot_data: dict[tuple[str, int], dict] = {}
    issues: list[str] = []
    raw_magnitude_arrays = []
    per_curve_median_magnitudes = []
    any_above_1_count = 0
    any_above_1_01_count = 0
    measured_band_values: dict[tuple[float, float], list[float]] = defaultdict(list)
    real_bands = _load_real_band_entries()
    band_pairs = [
        (float(item["frequency_band_hz"][0]), float(item["frequency_band_hz"][1]))
        for item in real_bands
    ]
    endpoint_rows = []
    selected_records = []
    sigma = float(cfg["response"]["event_sigma_m"])
    expected_shapes = (
        (frequency.size,),
        (4, distance.size),
        (3, distance.size),
    )

    for entry in selected:
        folder = entry["folder"]
        folder_key = str(folder)
        if folder_key not in array_cache:
            array_cache[folder_key] = (
                np.load(folder / "s11.npy", mmap_mode="r", allow_pickle=False),
                np.load(folder / "inputs.npy", mmap_mode="r", allow_pickle=False),
                np.load(folder / "targets.npy", mmap_mode="r", allow_pickle=False),
            )
        s11_array, input_array, target_array = array_cache[folder_key]
        index = int(entry["row_index"])
        s11 = np.asarray(s11_array[index])
        inputs = np.asarray(input_array[index])
        targets = np.asarray(target_array[index])
        meta = entry["metadata"]
        sample_id = int(meta.get("sample_id", -1))
        name = f"{entry['split']} sample_id={sample_id}"

        shapes = (s11.shape, inputs.shape, targets.shape)
        shape_ok = shapes == expected_shapes
        if not shape_ok:
            issues.append(f"{name}: shapes are {shapes}, expected {expected_shapes}")
        finite_s11 = bool(np.isfinite(s11).all())
        finite_inputs = bool(np.isfinite(inputs).all())
        finite_targets = bool(np.isfinite(targets).all())
        if not finite_s11:
            issues.append(f"{name}: S11 contains non-finite values")
        if not finite_inputs:
            issues.append(f"{name}: model input contains non-finite values")
        if not finite_targets:
            issues.append(f"{name}: target contains non-finite values")

        if shape_ok and finite_s11:
            magnitude = np.abs(s11)
            raw_magnitude_arrays.append(magnitude.reshape(-1))
            per_curve_median_magnitudes.append(float(np.median(magnitude)))
            any_above_1_count += bool(np.any(magnitude > 1.0))
            any_above_1_01_count += bool(np.any(magnitude > 1.01))
            for lower, upper in band_pairs:
                if frequency[0] > lower or frequency[-1] < upper:
                    continue
                mask = (frequency >= lower) & (frequency < upper)
                if np.any(mask):
                    measured_band_values[(lower, upper)].append(
                        float(np.median(magnitude[mask]))
                    )

        coordinate_issues = []
        if shape_ok and finite_targets and np.isfinite(distance).all():
            coordinate_issues = _validate_coordinates(meta, targets, distance, sigma)
            issues.extend(f"{name}: {issue}" for issue in coordinate_issues)

        cable_length = float(meta["length_m"])
        terminal = float(meta["terminal_position_m"])
        half_window = max(20.0, 0.1 * cable_length)
        lower_bound = max(float(distance[0]), terminal - half_window)
        upper_bound = min(float(distance[-1]), terminal + half_window)
        window = (distance >= lower_bound) & (distance <= upper_bound)
        if shape_ok and finite_inputs and np.any(window):
            local_indices = np.flatnonzero(window)
            peak_local_index = int(np.argmax(np.abs(inputs[0, window])))
            peak_index = int(local_indices[peak_local_index])
            peak_distance = float(distance[peak_index])
            offset = peak_distance - cable_length
            endpoint_rows.append({
                "split": entry["split"],
                "sample_id": sample_id,
                "physical_length_m": cable_length,
                "search_half_width_m": float(half_window),
                "search_window_m": [float(lower_bound), float(upper_bound)],
                "observed_impulse_abs_peak_distance_m": peak_distance,
                "signed_peak_offset_from_physical_endpoint_m": float(offset),
                "absolute_peak_offset_m": float(abs(offset)),
            })

        selected_records.append({
            "split": entry["split"],
            "sample_id": sample_id,
            "topology_id": meta.get("topology_id"),
            "length_m": cable_length,
            "joint_count": len(meta.get("joint_positions_m", [])),
            "event_count": len(meta.get("event_positions_m", [])),
            "defect_interval_count": len(meta.get("defect_intervals_m", [])),
            "finite": {
                "s11": finite_s11,
                "input": finite_inputs,
                "target": finite_targets,
            },
            "coordinate_validation_issues": coordinate_issues,
        })

        key = (entry["split"], sample_id)
        if key in plot_keys and shape_ok:
            plot_data[key] = {
                "metadata": meta,
                "frequency_hz": frequency,
                "distance_m": distance,
                "s11": s11.copy(),
                "inputs": inputs.copy(),
                "targets": targets.copy(),
            }

    if raw_magnitude_arrays:
        raw_magnitudes = np.concatenate(raw_magnitude_arrays)
    else:
        raw_magnitudes = np.empty(0, dtype=np.float32)

    band_results = []
    for real_entry in real_bands:
        lower, upper = (
            float(real_entry["frequency_band_hz"][0]),
            float(real_entry["frequency_band_hz"][1]),
        )
        synth_values = measured_band_values.get((lower, upper), [])
        covered = bool(frequency[0] <= lower and frequency[-1] >= upper)
        band_results.append({
            "frequency_band_hz": [lower, upper],
            "synthetic_selected_curve_median_abs_s11_quantiles": _quantiles(synth_values),
            "synthetic_band_fully_covered": covered,
            "synthetic_selected_curve_count": len(synth_values),
            "measured_top_level_case_median_abs_s11_quantiles": real_entry.get(
                "case_median_abs_s11_quantiles"
            ),
            "measured_top_level_case_count": real_entry.get("top_level_case_count"),
            "measured_curve_count_with_band_value": real_entry.get("curve_count_with_band_value"),
        })

    abs_peak_offsets = [row["absolute_peak_offset_m"] for row in endpoint_rows]
    return {
        "selected_sample_count": len(selected),
        "selected_sample_counts_by_split": dict(Counter(
            entry["split"] for entry in selected
        )),
        "sample_checks": selected_records,
        "measured_s11_abs_point_distribution": _quantiles(raw_magnitudes),
        "per_curve_median_abs_s11_distribution": _quantiles(per_curve_median_magnitudes),
        "selected_curves_with_any_abs_s11_gt_1_count": int(any_above_1_count),
        "selected_curves_with_any_abs_s11_gt_1_01_count": int(any_above_1_01_count),
        "s11_magnitude_note": (
            "Measured-field |S11| is descriptive only; values above 1 are counted, "
            "not treated as a hard audit failure. Physical passivity is reported "
            "separately only when the generator recorded it."
        ),
        "frequency_band_abs_s11_comparison": band_results,
        "physical_endpoint_to_nearby_impulse_peak": {
            "search_rule": "window = endpoint +/- max(20 m, 0.1 * physical cable length)",
            "diagnostic_only_warning": (
                "The strongest impulse-response peak in this window is not a true "
                "terminal identification; offsets are reported without moving labels."
            ),
            "sample_count": len(endpoint_rows),
            "absolute_offset_m_distribution": _quantiles(abs_peak_offsets),
            "signed_offset_m_distribution": _quantiles([
                row["signed_peak_offset_from_physical_endpoint_m"]
                for row in endpoint_rows
            ]),
            "per_sample": endpoint_rows,
        },
        "plot_data": plot_data,
    }, issues, plot_data


def _load_real_band_entries() -> list[dict]:
    if not MEASURED_SUMMARY.is_file():
        return []
    summary = json.loads(MEASURED_SUMMARY.read_text(encoding="utf-8"))
    return summary.get(
        "up_to_2500m_descriptive_statistics", {}
    ).get("frequency_band_abs_median_by_case_quantiles", [])


def _physical_s11_summary(entries: list[dict]) -> dict[str, Any]:
    passive_values = []
    physical_max_abs_values = []
    physical_rms_values = []
    for entry in entries:
        diagnostic = entry["metadata"].get("s11_diagnostics", {})
        passive = diagnostic.get("physical_passive_with_1e-6_tolerance")
        if passive is not None:
            passive_values.append(bool(passive))
        value = diagnostic.get("physical_max_abs")
        if value is not None and math.isfinite(float(value)):
            physical_max_abs_values.append(float(value))
        value = diagnostic.get("physical_rms_abs")
        if value is not None and math.isfinite(float(value)):
            physical_rms_values.append(float(value))
    return {
        "recorded_sample_count": len(passive_values),
        "physical_passive_count": int(sum(passive_values)),
        "physical_nonpassive_count": int(len(passive_values) - sum(passive_values)),
        "physical_max_abs_distribution": _quantiles(physical_max_abs_values),
        "physical_rms_abs_distribution": _quantiles(physical_rms_values),
        "interpretation": (
            "Summarized from generator-recorded physical_s11 diagnostics before "
            "measurement error/noise; no passivity conclusion is inferred from "
            "the measured S11 field."
        ),
    }


def _plot_examples(plot_data: dict[tuple[str, int], dict], output_path: Path) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": ["Times New Roman", "SimHei"],
        "font.sans-serif": ["SimHei"],
        "axes.unicode_minus": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.spines.top": True,
        "axes.spines.right": True,
        "figure.dpi": 200,
        "savefig.dpi": 200,
    })
    rows = list(plot_data.values())
    rows.sort(key=lambda item: float(item["metadata"]["length_m"]))
    if not rows:
        return []

    figure, axes = plt.subplots(
        len(rows), 3, figsize=(18, max(4.2, 4.2 * len(rows))), squeeze=False
    )
    example_manifest = []
    for row_index, data in enumerate(rows):
        meta = data["metadata"]
        sample_id = int(meta["sample_id"])
        length_m = float(meta["length_m"])
        split = str(meta["split"])
        label = "short" if row_index < math.ceil(len(rows) / 2) else "long"
        example_manifest.append({
            "split": split,
            "sample_id": sample_id,
            "length_m": length_m,
            "length_stratum": label,
        })

        frequency = data["frequency_hz"]
        distance = data["distance_m"]
        s11 = data["s11"]
        inputs = data["inputs"]
        targets = data["targets"]
        events = [float(value) for value in meta.get("event_positions_m", [])]
        intervals = [
            (float(value[0]), float(value[1]))
            for value in meta.get("defect_intervals_m", [])
        ]
        terminal = float(meta["terminal_position_m"])

        mag_axis, response_axis, target_axis = axes[row_index]
        phase_axis = mag_axis.twinx()
        mag_axis.semilogx(frequency, np.abs(s11), color="tab:blue", linewidth=0.8)
        phase_axis.semilogx(
            frequency,
            np.rad2deg(np.unwrap(np.angle(s11))),
            color="tab:orange",
            linewidth=0.65,
            alpha=0.8,
        )
        mag_axis.set_ylabel("|S11|")
        phase_axis.set_ylabel("Phase (deg)")
        mag_axis.set_xlabel("Frequency (Hz)")
        mag_axis.set_title(f"{split} sample {sample_id}; length={length_m:.1f} m")
        mag_axis.grid(True, which="both", alpha=0.18)

        response_axis.plot(
            distance, inputs[0], color="tab:blue", linewidth=0.7, label="Impulse (norm)"
        )
        response_axis.plot(
            distance, inputs[1], color="tab:orange", linewidth=0.7, label="Step (norm)"
        )
        response_axis.set_yscale("symlog", linthresh=0.01)
        for start, end in intervals:
            response_axis.axvspan(start, end, color="tab:green", alpha=0.12)
        for position in events:
            response_axis.axvline(position, color="tab:red", alpha=0.20, linewidth=0.6)
        response_axis.axvline(
            terminal, color="black", alpha=0.7, linestyle="--", linewidth=0.8,
            label="Physical terminal",
        )
        response_axis.set_ylabel("Normalized response (symlog display only)")
        response_axis.set_xlabel("Distance (m)")
        response_axis.set_title("Impulse/step; display scale only, model input unchanged")
        response_axis.grid(True, alpha=0.18)
        response_axis.legend(loc="upper right", fontsize=7)

        target_axis.plot(distance, targets[0], color="tab:red", linewidth=0.75, label="Event")
        target_axis.plot(
            distance, targets[1], color="tab:green", linewidth=0.75, label="Interval"
        )
        target_axis.plot(
            distance, targets[2], color="black", linewidth=0.75, label="Terminal"
        )
        if events:
            target_axis.scatter(
                events, [1.04] * len(events), marker="|", s=30, color="tab:red",
                label="Event truth",
            )
        target_axis.axvline(terminal, color="black", alpha=0.35, linestyle="--")
        target_axis.set_ylim(-0.05, 1.1)
        target_axis.set_ylabel("Target")
        target_axis.set_xlabel("Distance (m)")
        target_axis.grid(True, alpha=0.18)
        target_axis.legend(loc="upper right", fontsize=7)

        for axis in (mag_axis, phase_axis, response_axis, target_axis):
            axis.tick_params(direction="in", top=True, right=True)
            for spine in axis.spines.values():
                spine.set_visible(True)

    figure.suptitle("FieldNN dataset audit examples (short and long strata)")
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return example_manifest


def main() -> None:
    args = parse_args()
    data_root = args.data.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest, cfg, frequency, distance, entries, issues = _read_dataset(data_root)
    selected, quotas = _select_entries(entries, args.max_samples)
    plot_entries = _plot_entries(selected)
    sampled_audit, sample_issues, plot_data = _selective_data_checks(
        selected, frequency, distance, cfg, plot_entries
    )
    issues.extend(sample_issues)

    coverage = _metadata_coverage(entries, cfg, manifest)
    topology_intersections = coverage["topology_pairwise_split_intersections"]
    for pair, result in topology_intersections.items():
        if result["intersection_count"] != 0:
            issues.append(
                f"topology_id crosses splits for {pair}: {result['intersection_count']}"
            )
    if any(coverage["missing_topology_id_count_by_split"].values()):
        issues.append("one or more samples have no topology_id")

    plot_path = output_root / "audit_examples.png"
    examples = _plot_examples(plot_data, plot_path)
    sampled_audit.pop("plot_data")
    sampled_audit["selection_quota_by_split"] = quotas
    sampled_audit["plotted_examples"] = examples

    real_summary_status = (
        str(MEASURED_SUMMARY)
        if MEASURED_SUMMARY.is_file()
        else f"not found: {MEASURED_SUMMARY}"
    )
    report = {
        "schema": "fieldnn-dataset-audit-1",
        "integrity_status": "pass" if not issues else "fail",
        "field_realism_status": "not_established",
        "data_root": str(data_root),
        "manifest_schema": manifest.get("schema"),
        "sample_limit": int(args.max_samples),
        "dataset_sample_count": len(entries),
        "dataset_sample_count_by_split": coverage["sample_count_by_split"],
        "manifest_sample_count_by_split": coverage["manifest_count_by_split"],
        "shard_count": len(manifest.get("shards", [])),
        "frequency_grid": {
            "point_count": int(frequency.size),
            "start_hz": float(frequency[0]),
            "stop_hz": float(frequency[-1]),
        },
        "distance_grid": {
            "point_count": int(distance.size),
            "start_m": float(distance[0]),
            "stop_m": float(distance[-1]),
        },
        "sampled_audit": sampled_audit,
        "dataset_coverage": coverage,
        "generator_physical_s11_audit": _physical_s11_summary(entries),
        "measured_field_comparison_source": real_summary_status,
        "audit_issues": issues,
        "interpretation_notes": [
            "This audit checks data integrity and label geometry; it does not establish field detectability or model accuracy.",
            "Measured |S11| above 1 is counted descriptively and is not treated as an error.",
            "Synthetic band statistics summarize selected per-sample medians; the field summary reports quantiles of top-level-case medians, so their sampling grains differ.",
            "Bands not fully covered by the synthetic frequency grid are listed with no synthetic value.",
            "Endpoint-to-impulse-peak offsets diagnose label/response alignment only; they do not move truth labels or identify a real terminal.",
            "The current topology_id is unique per sample, so an empty split intersection is not proof of independent physical topologies.",
            "The symlog impulse/step axes in the PNG are display-only; model input arrays are unchanged.",
        ],
    }
    report_path = output_root / "dataset_audit.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(
        f"integrity_status={report['integrity_status']} "
        f"field_realism_status={report['field_realism_status']} "
        f"selected={len(selected)} issues={len(issues)}"
    )
    print(f"json={report_path}")
    print(f"png={plot_path}")
    for issue in issues[:20]:
        print(f"ISSUE: {issue}")


if __name__ == "__main__":
    main()
