#!/usr/bin/env python3
"""Generate the exploratory field-bound Core-16 cases through all three routes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from field_cases import build_cases
from field_physics import ROUTES, simulate
from field_responses import algorithm1, algorithm2, moving_mean


BASE = Path(__file__).resolve().parent
LAYERS = ("clean", "field")
METRIC_FIELDS = [
    "length_m", "case_id", "title", "category", "layer", "route",
    "finite_s11", "finite_responses", "max_abs_s11", "passive_clean",
    "defect_windows_m", "terminal_window_m",
    "a1_defect_impulse_diff_peak", "a1_defect_step_diff_peak",
    "a1_terminal_impulse_peak", "a1_terminal_step_peak",
    "a1_near_end_impulse_energy_ratio",
    "a2_defect_impulse_diff_peak", "a2_defect_step_diff_peak",
    "a2_terminal_impulse_peak", "a2_terminal_step_peak",
    "a2_near_end_impulse_energy_ratio",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="field_config.json")
    parser.add_argument("--lengths", nargs="+", type=float, default=None)
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--output", default="output/core16")
    parser.add_argument("--assets", default="assets/core16")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def resolve_from_base(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else BASE / path


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def length_label(length_m: float) -> str:
    return f"{length_m:g}m"


def case_windows(case: dict, length_m: float, margin_m: float) -> list[list[float]]:
    if case["id"] == "C00":
        return []
    category = case.get("category")
    if category == "local":
        center = 0.375 * length_m
        intervals = [[center - margin_m, center + margin_m]]
    elif category == "long_section":
        intervals = [[0.25 * length_m - margin_m, 0.50 * length_m + margin_m]]
    elif category == "replacement":
        intervals = [[0.50 * length_m - margin_m, 0.75 * length_m + margin_m]]
    elif category == "segmented_loss":
        intervals = [
            [0.25 * length_m - margin_m, 0.50 * length_m + margin_m],
            [0.75 * length_m - margin_m, length_m],
        ]
    else:
        intervals = []
    return [[max(0.0, start), min(length_m, end)] for start, end in intervals]


def response_bundle(
    frequency_hz: np.ndarray,
    s11: np.ndarray,
    config: dict,
    length_m: float,
    velocity_m_s: float,
    distance_max_m: float,
    response_step_m: float,
):
    response_config = config["response"]
    actual_df = float(frequency_hz[1] - frequency_hz[0])
    a1 = algorithm1(
        frequency_hz,
        s11,
        velocity_m_s=velocity_m_s,
        distance_max_m=distance_max_m,
        fft_step_hz=actual_df,
    )
    a2 = algorithm2(
        frequency_hz,
        s11,
        velocity_m_s=velocity_m_s,
        distance_max_m=distance_max_m,
        distance_step_m=response_step_m,
        step_smoothing_m=float(response_config["step_smoothing_m"]),
        impulse_smoothing_m=float(response_config["impulse_smoothing_m"]),
    )

    a1_distance = np.asarray(a1["distance_m"])
    a1_dx = float(a1_distance[1] - a1_distance[0])
    a1_step_points = max(1, round(float(response_config["step_smoothing_m"]) / a1_dx))
    a1_impulse_points = max(1, round(float(response_config["impulse_smoothing_m"]) / a1_dx))
    a1_impulse = np.asarray(a1["impulse_raw"])
    a1_step_rho = np.asarray(a1["step_rho"])
    a2_distance = np.asarray(a2["distance_m"])
    a2_impulse = np.asarray(a2["impulse_raw"])
    a2_step = np.asarray(a2["step_raw"])

    return {
        "algorithm1": {
            "distance_m": a1_distance,
            "impulse_raw": a1_impulse,
            "impulse_smoothed": moving_mean(a1_impulse, a1_impulse_points),
            "step_raw": np.asarray(a1["step_raw"]),
            "step_rho": a1_step_rho,
            "step_smoothed": moving_mean(a1_step_rho, a1_step_points),
            "fft_step_hz": float(a1["fft_step_hz"]),
            "period_distance_m": float(a1["period_distance_m"]),
            "native_max_gap_hz": float(a1["native_max_gap_hz"]),
            "dt_s": float(a1["dt_s"]),
            "distance_step_m": a1_dx,
            "step_smoothing_points": a1_step_points,
            "impulse_smoothing_points": a1_impulse_points,
        },
        "algorithm2": {
            "distance_m": a2_distance,
            "distance_impulse_m": np.asarray(a2["distance_impulse_m"]),
            "step_raw": a2_step,
            "impulse_raw": a2_impulse,
            "step_smoothed": np.asarray(a2["step_smoothed"]),
            "impulse_smoothed": np.asarray(a2["impulse_smoothed"]),
            "distance_step_m": float(a2["distance_step_m"]),
            "step_smoothing_points": int(a2["step_smoothing_points"]),
            "impulse_smoothing_points": int(a2["impulse_smoothing_points"]),
        },
    }


def save_responses(path: Path, bundle: dict, metadata: dict):
    arrays = {}
    for algorithm_name, values in bundle.items():
        for name, value in values.items():
            if isinstance(value, np.ndarray):
                arrays[f"{algorithm_name}_{name}"] = value
            else:
                arrays[f"{algorithm_name}_{name}"] = np.asarray(value)
    for key, value in metadata.items():
        arrays[key] = np.asarray(value)
    np.savez_compressed(path, **arrays)


def write_s11(path: Path, frequency_hz: np.ndarray, s11: np.ndarray):
    data = np.column_stack((frequency_hz, s11.real, s11.imag))
    np.savetxt(
        path,
        data,
        delimiter=",",
        header="Frequency,S11_Real,S11_Imaginary",
        comments="",
        fmt="%.10e",
    )


def window_peak(distance_m, values, windows):
    if not windows:
        return None
    selected = np.zeros(distance_m.size, dtype=bool)
    for start, end in windows:
        selected |= (distance_m >= start) & (distance_m <= end)
    if not np.any(selected):
        return None
    return float(np.max(np.abs(values[selected])))


def difference_peak(distance_m, values, reference, windows):
    if not windows:
        return None
    reference_distance, reference_values = reference
    if reference_values.size != values.size or not np.array_equal(reference_distance, distance_m):
        reference_values = np.interp(distance_m, reference_distance, reference_values)
    return window_peak(distance_m, np.asarray(values) - reference_values, windows)


def energy_ratio(distance_m, values, near_end_m, cable_length_m):
    near = np.asarray(values)[distance_m <= near_end_m]
    whole = np.asarray(values)[distance_m <= cable_length_m]
    denominator = float(np.sum(np.square(whole)))
    if near.size == 0 or denominator == 0.0:
        return None
    return float(np.sum(np.square(near)) / denominator)


def calculate_metrics(
    case: dict,
    length_m: float,
    layer: str,
    route: str,
    s11: np.ndarray,
    bundle: dict,
    c00_bundle: dict | None,
    near_end_m: float,
    passivity_tolerance: float,
):
    finite = bool(np.all(np.isfinite(s11.real)) and np.all(np.isfinite(s11.imag)))
    max_abs = float(np.max(np.abs(s11))) if finite else None
    passive_clean = ""
    if layer == "clean":
        passive_clean = bool(finite and max_abs <= 1.0 + passivity_tolerance)

    windows = case_windows(case, length_m, near_end_m)
    terminal_window = [max(0.0, length_m - near_end_m), length_m + near_end_m]
    result = {
        "length_m": length_m,
        "case_id": case["id"],
        "title": case["title"],
        "category": case.get("category", ""),
        "layer": layer,
        "route": route,
        "finite_s11": finite,
        "finite_responses": all(np.all(np.isfinite(value))
                                for algorithm in bundle.values()
                                for value in algorithm.values()),
        "max_abs_s11": max_abs if max_abs is not None else "",
        "passive_clean": passive_clean,
        "defect_windows_m": json.dumps(windows, ensure_ascii=False),
        "terminal_window_m": json.dumps(terminal_window),
    }
    for prefix, algorithm_name, impulse_key, step_key in (
        ("a1", "algorithm1", "impulse_raw", "step_smoothed"),
        ("a2", "algorithm2", "impulse_smoothed", "step_smoothed"),
    ):
        current = bundle[algorithm_name]
        impulse_distance = (
            current["distance_impulse_m"]
            if algorithm_name == "algorithm2"
            else current["distance_m"]
        )
        step_distance = current["distance_m"]
        result[f"{prefix}_defect_impulse_diff_peak"] = (
            difference_peak(
                impulse_distance,
                current[impulse_key],
                (
                    c00_bundle[algorithm_name]["distance_impulse_m"]
                    if algorithm_name == "algorithm2"
                    else c00_bundle[algorithm_name]["distance_m"],
                    c00_bundle[algorithm_name][impulse_key],
                ),
                windows,
            )
            if c00_bundle is not None and case["id"] != "C00"
            else ""
        )
        result[f"{prefix}_defect_step_diff_peak"] = (
            difference_peak(
                step_distance,
                current[step_key],
                (c00_bundle[algorithm_name]["distance_m"], c00_bundle[algorithm_name][step_key]),
                windows,
            )
            if c00_bundle is not None and case["id"] != "C00"
            else ""
        )
        result[f"{prefix}_terminal_impulse_peak"] = window_peak(
            impulse_distance, current[impulse_key], [terminal_window]
        )
        result[f"{prefix}_terminal_step_peak"] = window_peak(
            step_distance, current[step_key], [terminal_window]
        )
        result[f"{prefix}_near_end_impulse_energy_ratio"] = energy_ratio(
            impulse_distance, current[impulse_key], near_end_m, length_m
        )
    return result


def save_plot_for_case(
    case: dict,
    length_m: float,
    distance_max_m: float,
    route_names: tuple[str, ...],
    output_root: Path,
    asset_root: Path,
):
    os.environ["MPLCONFIGDIR"] = str((BASE / "cache" / "matplotlib").resolve())
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.rcParams["font.family"] = ["Times New Roman", "SimHei"]
    matplotlib.rcParams["font.serif"] = ["Times New Roman", "SimHei"]
    matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["figure.dpi"] = 200
    matplotlib.rcParams["savefig.dpi"] = 200
    matplotlib.rcParams["xtick.direction"] = "in"
    matplotlib.rcParams["ytick.direction"] = "in"
    import matplotlib.pyplot as plt

    colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(3, 4, figsize=(19, 11), sharex=True)
    columns = [
        ("algorithm1", "impulse_raw", "distance_m", "A1 脉冲"),
        ("algorithm1", "step_smoothed", "distance_m", "A1 阶跃"),
        ("algorithm2", "impulse_smoothed", "distance_impulse_m", "A2 脉冲"),
        ("algorithm2", "step_smoothed", "distance_m", "A2 阶跃"),
    ]
    windows = case_windows(case, length_m, max(20.0, 0.02 * length_m))
    joint_positions = [float(item["position_m"]) for item in case["joints"]]
    legend_handles = {}
    length_key = length_label(length_m)

    for row_index, route in enumerate(route_names):
        for column_index, (algorithm_name, value_key, distance_key, title) in enumerate(columns):
            ax = axes[row_index, column_index]
            for joint_position in joint_positions:
                ax.axvline(joint_position, color="0.45", linewidth=0.55, linestyle=":")
            for start, end in windows:
                ax.axvspan(start, end, color="#E45756", alpha=0.10, linewidth=0)

            if case["id"] != "C00":
                reference_path = output_root / "responses" / length_key / "field" / route / "C00.npz"
                with np.load(reference_path) as reference:
                    ref_x = reference[f"{algorithm_name}_{distance_key}"]
                    ref_y = reference[f"{algorithm_name}_{value_key}"]
                ref_line, = ax.plot(ref_x, ref_y, color="0.78", linewidth=0.7, linestyle="-.", label="C00 field ref")
                legend_handles.setdefault("C00 field ref", ref_line)

            for layer in LAYERS:
                response_path = output_root / "responses" / length_key / layer / route / f"{case['id']}.npz"
                with np.load(response_path) as response:
                    x = response[f"{algorithm_name}_{distance_key}"]
                    y = response[f"{algorithm_name}_{value_key}"]
                color = "0.35" if case["id"] == "C00" else colors(int(case["id"][1:]) % 10)
                linestyle = "--" if layer == "clean" else "-"
                label = f"{layer} {case['id']}"
                line, = ax.plot(x, y, color=color, linestyle=linestyle, linewidth=0.85, label=label)
                legend_handles.setdefault(layer, line)

            ax.set_xlim(0.0, distance_max_m)
            ax.grid(True, alpha=0.18)
            if row_index == 0:
                ax.set_title(title)
            if column_index == 0:
                ax.set_ylabel(f"{route}\n响应幅值")
            if row_index == 2:
                ax.set_xlabel("距离 (m)")

    fig.suptitle(
        f"{length_m:g} m | {case['id']} {case['title']} | 实线 field，虚线 clean；"
        "竖线为接头，阴影为异常分析窗",
        fontsize=12,
    )
    fig.legend(
        legend_handles.values(),
        legend_handles.keys(),
        loc="lower center",
        ncol=max(2, len(legend_handles)),
        frameon=True,
    )
    fig.tight_layout(rect=(0.02, 0.06, 0.99, 0.95))
    target = asset_root / length_key / f"{case['id']}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    config_path = resolve_from_base(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    output_root = resolve_from_base(args.output)
    asset_root = resolve_from_base(args.assets)
    output_root.mkdir(parents=True, exist_ok=True)
    if not args.no_plots:
        asset_root.mkdir(parents=True, exist_ok=True)

    lengths = args.lengths if args.lengths is not None else config.get("lengths_m", [600.0, 1500.0, 2400.0])
    all_cases_by_length = {float(length): build_cases(config, float(length)) for length in lengths}
    if args.cases is None or not args.cases or [item.lower() for item in args.cases] == ["all"]:
        selected_ids = None
    else:
        selected_ids = {item.upper() for item in args.cases}
    cases_by_length = {
        length: [
            case for case in cases
            if selected_ids is None or case["id"].upper() in selected_ids
        ]
        for length, cases in all_cases_by_length.items()
    }
    selected_cases = cases_by_length[next(iter(cases_by_length))]
    if not selected_cases:
        raise ValueError("No Core-16 cases selected.")
    baseline_added_for_reference = False
    if selected_ids is not None and all(case["id"] != "C00" for case in selected_cases):
        all_cases = all_cases_by_length[next(iter(all_cases_by_length))]
        selected_cases = [next(case for case in all_cases if case["id"] == "C00")] + selected_cases
        baseline_added_for_reference = True
        for length_m, cases in cases_by_length.items():
            baseline = next(case for case in all_cases_by_length[length_m] if case["id"] == "C00")
            cases_by_length[length_m] = [baseline] + cases

    start_hz = float(config["frequency"]["start_hz"])
    configured_stop_hz = float(config["frequency"]["stop_hz"])
    stop_hz = min(configured_stop_hz, 20_000_000.0) if args.quick else configured_stop_hz
    base_points = 4096 if args.quick else int(config["frequency"]["points"])
    response_config = config["response"]
    velocity_m_s = float(response_config["velocity_m_s"])
    response_step_m = 1.0 if args.quick else float(response_config["distance_step_m"])
    reference_hz = float(config["reference_hz"])
    zref = float(config.get("zref_ohm", 50.0))
    distance_multiplier = float(response_config["distance_multiplier"])
    passivity_tolerance = float(config.get("passivity_tolerance", 1.0e-9))

    truth = {
        "schema": "core16-field-truth-1",
        "parameter_status": config.get("parameter_status", "unspecified"),
        "config": config,
        "seed": config["seed"],
        "measurement_model": {
            "configuration": config.get("measurement", {}),
            "measurement_errors_applied": False,
            "random_noise_added": False,
            "note": "未识别measurement误差保持零增量；clean不加夹具，field仅加配置夹具。",
        },
        "layers": {
            "clean": {"fixture": None},
            "field": {"fixture": config.get("fixture")},
            "physical_cases_shared_between_layers": True,
        },
        "lengths": {},
    }
    metrics_rows = []
    profile_lengths = {}
    total_tasks = sum(len(cases) for cases in cases_by_length.values()) * len(ROUTES) * len(LAYERS)
    completed = 0
    overall_start = time.perf_counter()

    for length_m, cases in cases_by_length.items():
        length_key = length_label(length_m)
        distance_max_m = length_m * distance_multiplier
        minimum_points = math.ceil(4.0 * distance_max_m * stop_hz / velocity_m_s) + 1
        point_count = max(base_points, minimum_points)
        frequency_hz = np.linspace(start_hz, stop_hz, point_count, dtype=np.float64)
        frequency_step_hz = float(frequency_hz[1] - frequency_hz[0])
        near_end_m = max(20.0, 0.02 * length_m)
        full_truth_cases = all_cases_by_length[length_m]
        truth["lengths"][length_key] = {
            "length_m": length_m,
            "analysis_distance_max_m": distance_max_m,
            "sampling": {
                "frequency_start_hz": float(frequency_hz[0]),
                "frequency_stop_hz": float(frequency_hz[-1]),
                "frequency_points": int(point_count),
                "frequency_step_hz": frequency_step_hz,
                "minimum_points_for_distance_alias_bound": int(minimum_points),
            },
            "cases": [
                {
                    "id": case["id"],
                    "title": case["title"],
                    "category": case.get("category"),
                    "purpose": case.get("purpose"),
                    "segments": case["segments"],
                    "joints": case["joints"],
                    "joint_positions_m": [joint["position_m"] for joint in case["joints"]],
                    "termination": case["termination"],
                    "defect_analysis_windows_m": case_windows(
                        case, length_m, max(20.0, 0.02 * length_m)
                    ),
                }
                for case in full_truth_cases
            ],
        }

        length_start = time.perf_counter()
        c00_bundles = {}
        for case in cases:
            for route in ROUTES:
                for layer in LAYERS:
                    fixture = config.get("fixture") if layer == "field" else None
                    task_start = time.perf_counter()
                    s11 = simulate(
                        frequency_hz,
                        case["segments"],
                        case["joints"],
                        fixture=fixture,
                        route=route,
                        reference_hz=reference_hz,
                        load=case["termination"],
                        zref=zref,
                    )
                    bundle = response_bundle(
                        frequency_hz,
                        s11,
                        config,
                        length_m,
                        velocity_m_s,
                        distance_max_m,
                        response_step_m,
                    )

                    s11_path = output_root / "s11" / length_key / layer / route / f"{case['id']}.csv"
                    response_path = output_root / "responses" / length_key / layer / route / f"{case['id']}.npz"
                    s11_path.parent.mkdir(parents=True, exist_ok=True)
                    response_path.parent.mkdir(parents=True, exist_ok=True)
                    write_s11(s11_path, frequency_hz, s11)
                    save_responses(
                        response_path,
                        bundle,
                        {
                            "frequency_hz": frequency_hz,
                            "case_id": case["id"],
                            "route": route,
                            "layer": layer,
                            "length_m": length_m,
                            "distance_max_m": distance_max_m,
                        },
                    )

                    if case["id"] == "C00":
                        c00_bundles[(layer, route)] = bundle
                    metrics_rows.append(calculate_metrics(
                        case=case,
                        length_m=length_m,
                        layer=layer,
                        route=route,
                        s11=s11,
                        bundle=bundle,
                        c00_bundle=c00_bundles.get((layer, route)),
                        near_end_m=near_end_m,
                        passivity_tolerance=passivity_tolerance,
                    ))

                    completed += 1
                    elapsed = time.perf_counter() - overall_start
                    task_elapsed = time.perf_counter() - task_start
                    percent = 100.0 * completed / total_tasks
                    print(
                        f"[{completed:03d}/{total_tasks:03d} {percent:5.1f}%] "
                        f"{length_key} {case['id']} {route} {layer} "
                        f"task={task_elapsed:.1f}s elapsed={elapsed:.1f}s",
                        flush=True,
                    )

        profile_lengths[length_key] = {
            "length_m": length_m,
            "analysis_distance_max_m": distance_max_m,
            "frequency_start_hz": float(frequency_hz[0]),
            "frequency_stop_hz": float(frequency_hz[-1]),
            "frequency_points": int(point_count),
            "frequency_step_hz": frequency_step_hz,
            "distance_alias_minimum_points": int(minimum_points),
            "algorithm1": {
                "fft_step_hz": frequency_step_hz,
                "distance_sample_step_m": float(
                    c00_bundles[("clean", ROUTES[0])]["algorithm1"]["distance_step_m"]
                ),
                "nominal_distance_sample_step_m": float(velocity_m_s / (4.0 * frequency_hz[-1])),
                "bandlimited_resolution_m_approx": float(
                    velocity_m_s / (2.0 * (frequency_hz[-1] - frequency_hz[0]))
                ),
                "bandlimited_resolution_note": (
                    "近似v/(2*带宽)；Hann窗主瓣对应的实际分辨宽度更大。"
                ),
                "period_distance_m": float(
                    c00_bundles[("clean", ROUTES[0])]["algorithm1"]["period_distance_m"]
                ),
                "native_max_gap_hz": float(
                    c00_bundles[("clean", ROUTES[0])]["algorithm1"]["native_max_gap_hz"]
                ),
            },
            "algorithm2": {
                "requested_distance_step_m": response_step_m,
                "actual_distance_step_m": (
                    float(c00_bundles[("clean", ROUTES[0])]["algorithm2"]["distance_step_m"])
                ),
            },
            "smoothing": {
                "configured_step_smoothing_m": float(response_config["step_smoothing_m"]),
                "configured_impulse_smoothing_m": float(response_config["impulse_smoothing_m"]),
                "algorithm1_actual_step_smoothing_m": float(
                    c00_bundles[("clean", ROUTES[0])]["algorithm1"]["step_smoothing_points"]
                    * c00_bundles[("clean", ROUTES[0])]["algorithm1"]["distance_step_m"]
                ),
                "algorithm1_actual_impulse_smoothing_m": float(
                    c00_bundles[("clean", ROUTES[0])]["algorithm1"]["impulse_smoothing_points"]
                    * c00_bundles[("clean", ROUTES[0])]["algorithm1"]["distance_step_m"]
                ),
                "algorithm2_actual_step_smoothing_m": float(
                    c00_bundles[("clean", ROUTES[0])]["algorithm2"]["step_smoothing_points"]
                    * c00_bundles[("clean", ROUTES[0])]["algorithm2"]["distance_step_m"]
                ),
                "algorithm2_actual_impulse_smoothing_m": float(
                    c00_bundles[("clean", ROUTES[0])]["algorithm2"]["impulse_smoothing_points"]
                    * c00_bundles[("clean", ROUTES[0])]["algorithm2"]["distance_step_m"]
                ),
                "quick_requested_distance_step_m": response_step_m if args.quick else None,
            },
            "elapsed_seconds": time.perf_counter() - length_start,
            "selected_case_count": len(cases),
            "routes": list(ROUTES),
            "layers": list(LAYERS),
        }

        if not args.no_plots:
            for case in cases:
                save_plot_for_case(
                    case,
                    length_m,
                    distance_max_m,
                    ROUTES,
                    output_root,
                    asset_root,
                )

    metrics_path = output_root / "metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(metrics_rows)

    truth_path = output_root / "truth.json"
    with truth_path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(truth), handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    total_elapsed = time.perf_counter() - overall_start
    profile = {
        "parameter_status": config.get("parameter_status", "unspecified"),
        "quick_mode": args.quick,
        "quick_mode_note": (
            "quick将频率上限限制为20 MHz、基础点数4096、Algorithm 2距离步进1 m；"
            "这不是正式高频结果，点数仍受距离无模糊下限约束。"
            if args.quick else ""
        ),
        "measurement_errors_applied": False,
        "random_noise_added": False,
        "measurement_config": config.get("measurement", {}),
        "layers": {
            "clean": {"fixture": None},
            "field": {"fixture": config.get("fixture")},
            "physical_cases_shared_between_layers": True,
        },
        "selected_case_ids": [case["id"] for case in selected_cases],
        "baseline_case_added_for_comparison": baseline_added_for_reference,
        "lengths": profile_lengths,
        "total_elapsed_seconds": total_elapsed,
        "metric_reference": "同长度、同层、同路线下相对C00；C00自身差分指标留空。",
        "passivity_scope": "仅clean物理层计算passive_clean；field层不设被动性硬门槛。",
        "outputs": {
            "s11": str(output_root / "s11"),
            "responses": str(output_root / "responses"),
            "metrics_csv": str(metrics_path),
            "truth_json": str(truth_path),
            "assets": "" if args.no_plots else str(asset_root),
        },
    }
    profile_path = output_root / "profile_summary.json"
    with profile_path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(profile), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"完成：{completed}/{total_tasks} 个case-route-layer，耗时 {total_elapsed:.1f}s")
    print(f"metrics: {metrics_path}")
    print(f"truth: {truth_path}")
    print(f"profile: {profile_path}")


if __name__ == "__main__":
    main()
