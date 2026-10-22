"""Core-16 route comparison built beside, not inside, DG_Route_Evaluation.

The original CST V1 and DG V3 sources are imported read-only.  This runner
keeps the three route implementations, the external attachments, the common
measurement chain, and both FDR algorithms on one protocol while taking all
Core-16 body values from ``core16_cases.yaml``.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V1_PATH = ROOT / "DifferentAgent/CodeX/CST_Reproduction/V1/cst_fdr_reproduction.py"
DG_V3_PATH = ROOT / "DG_Update/DG_V3"
C0 = 299_792_458.0
REFERENCE_HZ = 100.0e6
ROUTES = (
    "dg_v3_rlgc",
    "cst_v1_ladder_0p1m",
    "cst_v1_continuous",
)
ROUTE_LABELS = {
    "dg_v3_rlgc": "DG V3 RLGC（100 MHz锚定）",
    "cst_v1_ladder_0p1m": "CST V1 0.1 m梯形",
    "cst_v1_continuous": "CST V1 连续固定RLGC",
}
ROUTE_COLORS = {
    "dg_v3_rlgc": "#1f77b4",
    "cst_v1_ladder_0p1m": "#7c3aed",
    "cst_v1_continuous": "#334155",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入 {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V1 = load_module("core16_cst_v1_source", V1_PATH)
sys.path.insert(0, str(DG_V3_PATH))
from dg_v3.physics import segment_abcd  # noqa: E402
from dg_v3.topology import CableSegment, RLGCMaterial  # noqa: E402


@dataclass(frozen=True)
class RLGC:
    """RLGC of one 0.4 m equivalent cell, stored in SI units."""

    r_ohm: float
    l_h: float
    g_s: float
    c_f: float

    def scaled(self, factor: float) -> "RLGC":
        return RLGC(
            self.r_ohm * factor,
            self.l_h * factor,
            self.g_s * factor,
            self.c_f * factor,
        )

    def per_m(self, base_cell_length_m: float) -> "RLGC":
        return self.scaled(1.0 / float(base_cell_length_m))


@dataclass(frozen=True)
class BodySegment:
    start_m: float
    end_m: float
    rlgc: RLGC

    @property
    def length_m(self) -> float:
        return self.end_m - self.start_m


@dataclass(frozen=True)
class Terminal:
    conductance_s: float
    capacitance_f: float


@dataclass(frozen=True)
class CoreCase:
    case_id: str
    title: str
    category: str
    purpose: str
    segments: tuple[BodySegment, ...]
    terminal: Terminal


def rlgc_from_yaml(raw: dict) -> RLGC:
    return RLGC(
        float(raw["r_ohm"]),
        float(raw["l_uh"]) * 1.0e-6,
        float(raw["g_us"]) * 1.0e-6,
        float(raw["c_pf"]) * 1.0e-12,
    )


def load_cases(cfg: dict) -> tuple[CoreCase, ...]:
    base_cell = float(cfg["cable"]["base_cell_length_m"])
    body_length = float(cfg["cable"]["body_length_m"])
    expected_ids = tuple(f"C{index:02d}" for index in range(16))
    raw_cases = cfg["cases"]
    if tuple(item["id"] for item in raw_cases) != expected_ids:
        raise ValueError("Core-16 配置必须按 C00-C15 顺序完整给出")

    cases: list[CoreCase] = []
    for raw_case in raw_cases:
        segments = tuple(
            BodySegment(
                float(item["start_m"]),
                float(item["end_m"]),
                rlgc_from_yaml(item),
            )
            for item in raw_case["segments"]
        )
        if not segments:
            raise ValueError(f"{raw_case['id']} 没有主体区段")
        previous = 0.0
        for segment in segments:
            if segment.end_m <= segment.start_m:
                raise ValueError(f"{raw_case['id']} 存在非正长度区段")
            if not np.isclose(segment.start_m, previous, rtol=0.0, atol=1.0e-9):
                raise ValueError(f"{raw_case['id']} 主体区段不连续")
            if not np.isclose(
                segment.start_m / 0.1,
                round(segment.start_m / 0.1),
                rtol=0.0,
                atol=1.0e-8,
            ) or not np.isclose(
                segment.end_m / 0.1,
                round(segment.end_m / 0.1),
                rtol=0.0,
                atol=1.0e-8,
            ):
                raise ValueError(f"{raw_case['id']} 区段边界不能落在0.1 m梯形单元内部")
            values = segment.rlgc
            if min(values.r_ohm, values.l_h, values.c_f) <= 0.0 or values.g_s < 0.0:
                raise ValueError(f"{raw_case['id']} 存在非法 RLGC 值")
            previous = segment.end_m
        if not np.isclose(previous, body_length, rtol=0.0, atol=1.0e-9):
            raise ValueError(f"{raw_case['id']} 主体长度不是 {body_length} m")
        raw_terminal = raw_case["terminal"]
        terminal = Terminal(
            float(raw_terminal["extra_shunt_g_us"]) * 1.0e-6,
            float(raw_terminal["extra_shunt_c_pf"]) * 1.0e-12,
        )
        cases.append(
            CoreCase(
                str(raw_case["id"]),
                str(raw_case["title"]),
                str(raw_case["category"]),
                str(raw_case["purpose"]),
                segments,
                terminal,
            )
        )
    if len(cases) != 16:
        raise ValueError("Core-16 配置必须有16个工况")
    return tuple(cases)


def identity(size: int):
    one = np.ones(size, dtype=np.complex128)
    zero = np.zeros(size, dtype=np.complex128)
    return one, zero, zero, one


def attachment_after(
    network,
    frequency_hz: np.ndarray,
    end_m: float,
    cable_cfg: dict,
):
    """Append the unchanged 10 m joint or 20/30 m 300 ohm TL."""

    joint_cfg = cable_cfg["attachments"]["special_joint1"]
    if np.isclose(end_m, float(joint_cfg["position_m"])):
        series = V1.series_element(
            frequency_hz,
            float(joint_cfg["series_r_ohm"]),
            float(joint_cfg["series_l_uh"]) * 1.0e-6,
        )
        shunt = V1.shunt_element(
            frequency_hz,
            1.0 / float(joint_cfg["shunt_r_ohm"]),
            float(joint_cfg["shunt_c_pf"]) * 1.0e-12,
        )
        return V1.cascade(network, V1.cascade(series, shunt))

    tl_cfg = cable_cfg["attachments"]["tl300"]
    if any(np.isclose(end_m, float(position)) for position in tl_cfg["positions_m"]):
        return V1.cascade(
            network,
            V1.lossless_transmission_line(
                frequency_hz,
                float(tl_cfg["length_m"]),
                float(tl_cfg["impedance_ohm"]),
            ),
        )
    return network


def append_front_tl(network, frequency_hz: np.ndarray, cable_cfg: dict):
    tl_cfg = cable_cfg["attachments"]["front_tl"]
    return V1.cascade(
        network,
        V1.lossless_transmission_line(
            frequency_hz,
            float(tl_cfg["length_m"]),
            float(tl_cfg["impedance_ohm"]),
        ),
    )


def append_terminal(network, frequency_hz: np.ndarray, terminal: Terminal):
    return V1.cascade(
        network,
        V1.shunt_element(
            frequency_hz,
            terminal.conductance_s,
            terminal.capacitance_f,
        ),
    )


def find_segment(case: CoreCase, start_m: float, end_m: float) -> BodySegment:
    for segment in case.segments:
        if start_m >= segment.start_m - 1.0e-9 and end_m <= segment.end_m + 1.0e-9:
            return segment
    raise ValueError(f"{case.case_id}: 找不到 {start_m}-{end_m} m 对应主体区段")


def simulate_cst_ladder(
    frequency_hz: np.ndarray,
    case: CoreCase,
    cable_cfg: dict,
) -> np.ndarray:
    """CST V1 fixed ladder, with the 0.1/0.4 cell scaling applied."""

    body_length = float(cable_cfg["body_length_m"])
    base_cell = float(cable_cfg["base_cell_length_m"])
    cell_length = float(cable_cfg["cst_ladder_cell_length_m"])
    total_cells = int(round(body_length / cell_length))
    scale = cell_length / base_cell
    network = identity(frequency_hz.size)
    network = append_front_tl(network, frequency_hz, cable_cfg)
    for index in range(total_cells):
        start_m = index * cell_length
        end_m = (index + 1) * cell_length
        source = find_segment(case, start_m, end_m)
        cell = source.rlgc.scaled(scale)
        network = V1.cascade(
            network,
            V1.standard_cell(
                frequency_hz,
                cell.r_ohm,
                cell.l_h,
                cell.g_s,
                cell.c_f,
            ),
        )
        network = attachment_after(network, frequency_hz, end_m, cable_cfg)
    network = append_terminal(network, frequency_hz, case.terminal)
    return V1.network_to_s11(network)


def simulate_cst_continuous(
    frequency_hz: np.ndarray,
    case: CoreCase,
    cable_cfg: dict,
) -> np.ndarray:
    """CST V1 continuous fixed-RLGC reference."""

    base_cell = float(cable_cfg["base_cell_length_m"])
    network = identity(frequency_hz.size)
    network = append_front_tl(network, frequency_hz, cable_cfg)
    for segment in case.segments:
        per_m = segment.rlgc.per_m(base_cell)
        network = V1.cascade(
            network,
            V1.continuous_fixed_line(
                frequency_hz,
                segment.length_m,
                per_m.r_ohm,
                per_m.l_h,
                per_m.g_s,
                per_m.c_f,
            ),
        )
        network = attachment_after(network, frequency_hz, segment.end_m, cable_cfg)
    network = append_terminal(network, frequency_hz, case.terminal)
    return V1.network_to_s11(network)


def anchored_segment(
    frequency_hz: np.ndarray,
    segment: BodySegment,
    base_cell_length_m: float,
):
    """DG V3 effective RLGC segment matched to the primary RLGC at 100 MHz."""

    primary = segment.rlgc.per_m(base_cell_length_m)
    omega_ref = 2.0 * math.pi * REFERENCE_HZ
    external_l = primary.l_h - primary.r_ohm / omega_ref
    if external_l <= 0.0:
        raise ValueError(f"{segment.start_m}-{segment.end_m} m 锚定后外部电感非正")
    z0 = math.sqrt(external_l / primary.c_f)
    velocity = 1.0 / math.sqrt(external_l * primary.c_f)
    epsr = (C0 / velocity) ** 2
    material = RLGCMaterial(
        model="effective_rlgc",
        conductor_conductivity_s_per_m=0.0,
        shunt_conductance_s_per_m=primary.g_s,
        series_resistance_100mhz_ohm_per_m=primary.r_ohm,
    )
    dg_segment = CableSegment(
        start_m=segment.start_m,
        end_m=segment.end_m,
        z0_ohm=z0,
        epsr=epsr,
        alpha_db_per_m_at_100mhz=0.0,
        tan_delta_at_100mhz=0.0,
        material=material,
    )
    return segment_abcd(frequency_hz, dg_segment)


def simulate_dg_v3(
    frequency_hz: np.ndarray,
    case: CoreCase,
    cable_cfg: dict,
) -> np.ndarray:
    """DG V3 continuous frequency-dependent RLGC route."""

    base_cell = float(cable_cfg["base_cell_length_m"])
    network = identity(frequency_hz.size)
    network = append_front_tl(network, frequency_hz, cable_cfg)
    for segment in case.segments:
        network = V1.cascade(
            network,
            anchored_segment(frequency_hz, segment, base_cell),
        )
        network = attachment_after(network, frequency_hz, segment.end_m, cable_cfg)
    network = append_terminal(network, frequency_hz, case.terminal)
    return V1.network_to_s11(network)


def simulate_clean(
    route: str,
    frequency_hz: np.ndarray,
    case: CoreCase,
    cable_cfg: dict,
) -> np.ndarray:
    if route == "dg_v3_rlgc":
        return simulate_dg_v3(frequency_hz, case, cable_cfg)
    if route == "cst_v1_ladder_0p1m":
        return simulate_cst_ladder(frequency_hz, case, cable_cfg)
    if route == "cst_v1_continuous":
        return simulate_cst_continuous(frequency_hz, case, cable_cfg)
    raise ValueError(f"未知路线: {route}")


def correlated_noise(size: int, correlation: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    white = (rng.standard_normal(size) + 1j * rng.standard_normal(size)) / math.sqrt(2.0)
    out = np.empty(size, dtype=np.complex128)
    out[0] = white[0]
    innovation = math.sqrt(1.0 - correlation**2)
    for index in range(1, size):
        out[index] = correlation * out[index - 1] + innovation * white[index]
    return out


def apply_common_measurement(
    s11: np.ndarray,
    frequency_hz: np.ndarray,
    measurement_cfg: dict,
    seed: int,
) -> np.ndarray:
    """The same measurement chain as DG_Route_Evaluation."""

    f_norm = frequency_hz / REFERENCE_HZ
    two_way_loss = np.power(
        10.0,
        -(2.0 * float(measurement_cfg["loss_db_at_100mhz"]) * f_norm) / 20.0,
    )
    delay = float(measurement_cfg["delay_ns"]) * 1.0e-9
    gamma = s11 * two_way_loss * np.exp(-1j * 4.0 * np.pi * frequency_hz * delay)
    e00 = 0.5 * float(measurement_cfg["directivity"]) * (1.0 + 1j)
    e11 = 0.5 * float(measurement_cfg["source_match"]) * (1.0 - 1j)
    tracking = float(measurement_cfg["tracking"]) * np.exp(
        1j * float(measurement_cfg["tracking_phase_rad"])
    )
    measured = e00 + tracking * gamma / (1.0 - e11 * gamma)
    scale = float(measurement_cfg["noise_sigma"]) * (
        1.0 + 0.35 / (1.0 + np.sqrt(frequency_hz / REFERENCE_HZ))
    )
    return measured + scale * correlated_noise(
        len(s11), float(measurement_cfg["noise_correlation"]), seed
    )


def save_s11(path: Path, frequency_hz: np.ndarray, s11: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("Frequency_Hz", "S11_Real", "S11_Imag"))
        writer.writerows(zip(frequency_hz, s11.real, s11.imag))


def save_response(path: Path, a1: dict, a2: dict, distance_max_m: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        algorithm1_distance_m=np.asarray(a1["distance_m"]),
        algorithm1_impulse=np.asarray(a1["impulse_real_raw"]),
        algorithm1_step=np.asarray(a1["step_raw"]),
        algorithm2_distance_m=np.asarray(a2["distance_m"]),
        algorithm2_step=np.asarray(a2["step_smoothed"]),
        algorithm2_impulse_distance_m=np.asarray(a2["distance_impulse_m"]),
        algorithm2_impulse=np.asarray(a2["impulse_smoothed"]),
        distance_max_m=np.asarray(float(distance_max_m)),
    )


def crop_mask(distance: np.ndarray, distance_max_m: float) -> np.ndarray:
    return np.asarray(distance) <= float(distance_max_m)


def peak_in_window(
    distance: np.ndarray,
    values: np.ndarray,
    low_m: float,
    high_m: float,
    absolute: bool,
) -> tuple[float, float]:
    distance = np.asarray(distance)
    values = np.asarray(values)
    mask = (
        (distance >= low_m)
        & (distance <= high_m)
        & np.isfinite(distance)
        & np.isfinite(values)
    )
    indices = np.flatnonzero(mask)
    if not len(indices):
        return math.nan, math.nan
    selected = np.abs(values[indices]) if absolute else values[indices]
    index = int(indices[np.argmax(selected)])
    return float(distance[index]), float(values[index])


def median_window(distance: np.ndarray, values: np.ndarray, low_m: float, high_m: float) -> float:
    distance = np.asarray(distance)
    values = np.asarray(values)
    mask = (
        (distance >= low_m)
        & (distance <= high_m)
        & np.isfinite(distance)
        & np.isfinite(values)
    )
    return float(np.median(values[mask])) if np.any(mask) else math.nan


def slope_window(distance: np.ndarray, values: np.ndarray, low_m: float, high_m: float) -> float:
    distance = np.asarray(distance)
    values = np.asarray(values)
    mask = (
        (distance >= low_m)
        & (distance <= high_m)
        & np.isfinite(distance)
        & np.isfinite(values)
    )
    if np.count_nonzero(mask) < 2:
        return math.nan
    return float(np.polyfit(distance[mask], values[mask], 1)[0])


def target_window(category: str) -> tuple[float, float] | None:
    """Fixed analysis windows; these do not introduce new physical locations."""

    return {
        "local": (12.0, 18.0),
        "long_section": (11.0, 19.0),
        "segmented_loss": (30.0, 36.0),
        "replacement": (21.0, 29.0),
    }.get(category)


def finite_response(run: dict) -> bool:
    arrays = (
        run["s11"],
        run["a1"]["distance_m"],
        run["a1"]["impulse_real_raw"],
        run["a1"]["step_raw"],
        run["a2"]["distance_m"],
        run["a2"]["step_smoothed"],
        run["a2"]["distance_impulse_m"],
        run["a2"]["impulse_smoothed"],
    )
    return all(bool(np.all(np.isfinite(np.asarray(array)))) for array in arrays)


def metrics_for(
    route: str,
    layer: str,
    case: CoreCase,
    run: dict,
    baseline: dict,
    distance_max_m: float,
    passivity_tolerance: float,
) -> dict:
    a1 = run["a1"]
    base_a1 = baseline["a1"]
    distance = np.asarray(a1["distance_m"])
    visible = crop_mask(distance, distance_max_m)
    base_scale = max(float(base_a1["step_scale"]), 1.0e-30)
    step = np.asarray(a1["step_raw"]) / base_scale
    base_step = np.asarray(base_a1["step_raw"]) / base_scale
    impulse = np.asarray(a1["impulse_real_raw"]) / base_scale
    base_impulse = np.asarray(base_a1["impulse_real_raw"]) / base_scale
    delta_step = step - base_step
    delta_impulse = impulse - base_impulse

    terminal_pos, terminal_amp = peak_in_window(
        distance, impulse, 37.0, distance_max_m, absolute=False
    )
    base_terminal_pos, base_terminal_amp = peak_in_window(
        distance, base_impulse, 37.0, distance_max_m, absolute=False
    )
    terminal_ratio = (
        abs(terminal_amp) / max(abs(base_terminal_amp), 1.0e-30)
        if np.isfinite(terminal_amp) and np.isfinite(base_terminal_amp)
        else math.nan
    )
    local_pos, local_amp = peak_in_window(
        distance, delta_impulse, 12.0, 18.0, absolute=True
    )
    target = target_window(case.category)
    if target is None:
        target_pos, target_amp, target_step = math.nan, math.nan, math.nan
    else:
        target_pos, target_amp = peak_in_window(
            distance, delta_impulse, target[0], target[1], absolute=True
        )
        target_step = median_window(distance, delta_step, target[0], target[1])

    checks = {
        "finite_response": finite_response(run),
    }
    if layer == "clean":
        checks["passive_clean"] = bool(
            np.max(np.abs(np.asarray(run["s11"]))) <= 1.0 + passivity_tolerance
        )
    return {
        "route": route,
        "route_label": ROUTE_LABELS[route],
        "layer": layer,
        "case": case.case_id,
        "title": case.title,
        "category": case.category,
        "max_abs_s11": float(np.max(np.abs(run["s11"]))),
        "s11_rms_delta_to_C00": float(
            np.sqrt(np.mean(np.abs(run["s11"] - baseline["s11"]) ** 2))
        ),
        "terminal_position_m": terminal_pos,
        "terminal_shift_m": terminal_pos - base_terminal_pos,
        "terminal_amplitude_ratio": terminal_ratio,
        "local_delta_peak_position_m": local_pos,
        "local_delta_peak_amplitude": local_amp,
        "target_delta_peak_position_m": target_pos,
        "target_delta_peak_amplitude": target_amp,
        "target_delta_step_median": target_step,
        "step_delta_median_16_35m": median_window(
            distance, delta_step, 16.0, 35.0
        ),
        "step_slope_16_35m": slope_window(distance, step, 16.0, 35.0),
        "baseline_step_slope_16_35m": slope_window(
            distance, base_step, 16.0, 35.0
        ),
        "algorithm2_terminal_position_m": peak_in_window(
            np.asarray(run["a2"]["distance_impulse_m"]),
            np.asarray(run["a2"]["impulse_smoothed"]),
            37.0,
            distance_max_m,
            absolute=False,
        )[0],
        "checks": checks,
        "numeric_gate_pass": bool(all(checks.values())),
    }


def configure_plot() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["SimHei", "Microsoft YaHei", "Arial Unicode MS"],
            "axes.unicode_minus": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )


def normalized_pair(
    run: dict,
    baseline: dict,
    algorithm: int,
    kind: str,
    distance_max_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if algorithm == 1:
        distance = np.asarray(run["a1"]["distance_m"])
        key = "step_raw" if kind == "step" else "impulse_real_raw"
        values = np.asarray(run["a1"][key])
        base_values = np.asarray(baseline["a1"][key])
    else:
        distance = np.asarray(
            run["a2"]["distance_m"] if kind == "step" else run["a2"]["distance_impulse_m"]
        )
        key = "step_smoothed" if kind == "step" else "impulse_smoothed"
        values = np.asarray(run["a2"][key])
        base_values = np.asarray(baseline["a2"][key])
    visible = crop_mask(distance, distance_max_m)
    center = float(np.median(base_values[distance <= 5.0])) if kind == "step" else 0.0
    scale = max(
        float(
            np.max(
                np.abs(
                    np.concatenate(
                        (values[visible] - center, base_values[visible] - center)
                    )
                )
            )
        ),
        1.0e-30,
    )
    return distance[visible], (values[visible] - center) / scale, (base_values[visible] - center) / scale


def plot_board(
    path: Path,
    case: CoreCase,
    layer: str,
    frequency_hz: np.ndarray,
    runs: dict[str, dict[str, dict]],
    distance_max_m: float,
) -> None:
    configure_plot()
    fig = plt.figure(figsize=(15.9, 17.2), dpi=200)
    grid = fig.add_gridspec(
        4, 6, height_ratios=(1.0, 1.05, 1.05, 1.25), hspace=0.34, wspace=0.34
    )
    axes = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[0, 4:6]),
        fig.add_subplot(grid[1, 0:3]),
        fig.add_subplot(grid[1, 3:6]),
        fig.add_subplot(grid[2, 0:3]),
        fig.add_subplot(grid[2, 3:6]),
    ]
    for route in ROUTES:
        color = ROUTE_COLORS[route]
        run = runs[route][case.case_id][layer]
        baseline = runs[route]["C00"][layer]
        s11 = run["s11"]
        axes[0].plot(
            frequency_hz / 1.0e6,
            s11.real,
            color=color,
            lw=0.85,
            label=ROUTE_LABELS[route],
        )
        axes[1].plot(frequency_hz / 1.0e6, np.abs(s11), color=color, lw=0.85)
        axes[2].plot(
            frequency_hz / 1.0e6,
            np.angle(s11, deg=True),
            color=color,
            lw=0.75,
        )
        for axis, algorithm, kind in (
            (axes[3], 1, "step"),
            (axes[4], 1, "impulse"),
            (axes[5], 2, "step"),
            (axes[6], 2, "impulse"),
        ):
            distance, values, baseline_values = normalized_pair(
                run, baseline, algorithm, kind, distance_max_m
            )
            if case.case_id != "C00":
                axis.plot(
                    distance,
                    baseline_values,
                    color=color,
                    lw=0.65,
                    alpha=0.35,
                    ls="--",
                )
            axis.plot(distance, values, color=color, lw=0.95)
    titles = (
        "S11 实部",
        "S11 幅值",
        "S11 相位（包裹）",
        "算法1 阶跃 / 归一化",
        "算法1 脉冲 / 归一化",
        "算法2 阶跃 / 归一化",
        "算法2 脉冲 / 归一化",
    )
    for index, (axis, title) in enumerate(zip(axes, titles)):
        axis.set_title(title)
        axis.grid(True, color="#d1d5db", lw=0.45, alpha=0.65)
        axis.tick_params(direction="in", top=True, right=True)
        axis.set_xlabel("Frequency (MHz)" if index < 3 else "Distance (m)")
    axes[0].set_ylabel("S11 Real")
    axes[1].set_ylabel("|S11|")
    axes[2].set_ylabel("Phase (deg)")
    axes[2].set_ylim(-180, 180)
    axes[0].legend(loc="best", fontsize=8)
    table_axis = fig.add_subplot(grid[3, 1:5])
    table_axis.axis("off")
    rows = []
    for route in ROUTES:
        metric = runs[route][case.case_id][layer]["metrics"]
        rows.append(
            [
                ROUTE_LABELS[route],
                f"{metric['s11_rms_delta_to_C00']:.6g}",
                "通过" if metric["numeric_gate_pass"] else "失败",
            ]
        )
    table = table_axis.table(
        cellText=rows,
        colLabels=("路线", "S11 RMS 相对C00", "数值检查"),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    fig.suptitle(
        f"Core-16 路线对比：{case.case_id} {case.title}（{layer}，距离轴0–{distance_max_m:g} m）",
        fontsize=15,
        y=0.994,
    )
    fig.text(
        0.5,
        0.006,
        "实线为当前工况；同色虚线为该路线C00基线。数值检查不等同于物理方向门槛。",
        ha="center",
        fontsize=9,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_report(
    path: Path,
    cfg: dict,
    cases: tuple[CoreCase, ...],
    metrics: list[dict],
    runtime_s: float,
) -> None:
    frequency = cfg["frequency_grid"]
    fdr = cfg["fdr"]
    lines = [
        "# Core-16 DG路线对比评估报告",
        "",
        f"- 运行时间：{runtime_s:.2f} s",
        f"- 频率网格：{frequency['start_hz']/1.0e3:g} kHz–{frequency['stop_hz']/1.0e6:g} MHz，{frequency['points']}点",
        f"- 算法2：time_points={fdr['algorithm2_time_points']}，chunk_size={fdr['algorithm2_chunk_size']}，距离分析上限={fdr['distance_max_m']} m",
        "- 输出同时保留 clean kernel 和 measured 两层；measured 由三条路线共用同一测量链和按工况固定的随机种子产生。",
        "- 本目录是独立扩展实验；原 `DG_Route_Evaluation` 目录不被脚本写入。",
        "",
        "## 1. 路线与公平性",
        "",
        "三条路线固定为：DG V3 频变连续 RLGC（100 MHz锚定）、CST V1 0.1 m固定RLGC梯形、CST V1连续固定RLGC参考。三条路线共同使用40 m主体、1 m/135 Ω前置无损线、10 m特殊接头、20/30 m的0.5 m/300 Ω无损线、末端开路及相同的FDR和测量链。",
        "",
        "CST梯形每个0.1 m单元按 `scale=0.1/0.4=0.25` 缩放每0.4 m等效RLGC；没有把0.4 m数值直接复制到0.1 m单元。DG V3按现有 effective_rlgc 方法把每米RLGC在100 MHz锚定，并保留其频变R/固定G行为。",
        "",
        "## 2. Core-16 工况",
        "",
        "| ID | 工况 | 空间类别 | 作用 | 末端附加G (μS) | 末端附加C (pF) |",
        "|---|---|---|---|---:|---:|",
    ]
    for case in cases:
        lines.append(
            f"| {case.case_id} | {case.title} | {case.category} | {case.purpose} | {case.terminal.conductance_s*1.0e6:g} | {case.terminal.capacitance_f*1.0e12:g} |"
        )
    lines += [
        "",
        "C05 的末端属于30–40 m第四段，因此末端附加支路固定为G=500 μS、C=16 pF；其他工况末端附加支路保持G=5 μS、C=16 pF。",
        "",
        "## 3. 数值检查汇总",
        "",
        "这里的 `numeric_gate_pass` 只检查响应有限性以及 clean 层 `|S11|≤1+1e-9`，不把未经本轮确认的物理方向写成硬门槛。方向、幅度和三路线差异使用 `metrics.csv` 与各工况对照图复核。",
        "",
        "| 路线 | 层 | 数值通过数 | 总数 |",
        "|---|---|---:|---:|",
    ]
    for route in ROUTES:
        for layer in ("clean", "measured"):
            rows = [
                item
                for item in metrics
                if item["route"] == route and item["layer"] == layer
            ]
            lines.append(
                f"| {ROUTE_LABELS[route]} | {layer} | {sum(item['numeric_gate_pass'] for item in rows)} | {len(rows)} |"
            )
    lines += [
        "",
        "逐条原始指标见 `output/metrics.csv` 和 `output/metrics.json`。脚本没有用任何实测曲线或距离域模板替换S11。",
        "",
        "## 4. 对照图",
        "",
    ]
    for layer in ("clean", "measured"):
        lines.append(f"### {layer}")
        lines.append("")
        for case in cases:
            lines.append(
                f"![{layer}-{case.case_id}](assets/comparison_board_{layer}_{case.case_id}.png)"
            )
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Core-16 DG/CST三路线对比评估")
    parser.add_argument("--config", type=Path, default=HERE / "core16_cases.yaml")
    parser.add_argument("--output", type=Path, default=HERE / "output")
    parser.add_argument("--assets", type=Path, default=HERE / "assets")
    parser.add_argument(
        "--report",
        type=Path,
        default=HERE / "Core-16路线对比评估报告.md",
    )
    parser.add_argument("--smoke", action="store_true", help="600频点/600时间点短时流程检查")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    configured_routes = tuple(cfg["routes"])
    if configured_routes != ROUTES:
        raise ValueError("配置中的路线必须保持三条既定路线及其顺序")
    cases = load_cases(cfg)
    point_count = 600 if args.smoke else int(cfg["frequency_grid"]["points"])
    time_points = 600 if args.smoke else int(cfg["fdr"]["algorithm2_time_points"])
    frequency_hz = np.linspace(
        float(cfg["frequency_grid"]["start_hz"]),
        float(cfg["frequency_grid"]["stop_hz"]),
        point_count,
        dtype=np.float64,
    )
    output = args.output.resolve()
    assets = args.assets.resolve()
    distance_max_m = float(cfg["fdr"]["distance_max_m"])
    started = time.perf_counter()
    all_runs: dict[str, dict[str, dict[str, dict]]] = {route: {} for route in ROUTES}
    print(f"频率网格：{frequency_hz[0]:g} Hz–{frequency_hz[-1]:g} Hz，{point_count}点")
    print(f"算法2：{time_points}时间点，chunk_size={cfg['fdr']['algorithm2_chunk_size']}，距离上限={distance_max_m:g} m")
    print("路线：" + ", ".join(ROUTE_LABELS[route] for route in ROUTES))

    for route_index, route in enumerate(ROUTES, start=1):
        print(f"\n[{route_index}/{len(ROUTES)}] {ROUTE_LABELS[route]}")
        for case_index, case in enumerate(cases, start=1):
            print(f"  [{case_index}/{len(cases)}] {case.case_id} {case.title}")
            clean = simulate_clean(route, frequency_hz, case, cfg["cable"])
            measured = apply_common_measurement(
                clean,
                frequency_hz,
                cfg["measurement"],
                int(cfg["measurement"]["seed"]) + case_index,
            )
            all_runs[route][case.case_id] = {}
            for layer, s11 in (("clean", clean), ("measured", measured)):
                a1 = V1.algorithm1_ifft(frequency_hz, s11)
                a2 = V1.algorithm2_fdr(
                    frequency_hz,
                    s11,
                    time_points=time_points,
                    chunk_size=int(cfg["fdr"]["algorithm2_chunk_size"]),
                    progress_label=f"{route}:{case.case_id}:{layer}:A2",
                )
                run = {"s11": s11, "a1": a1, "a2": a2}
                all_runs[route][case.case_id][layer] = run
                save_s11(
                    output / "s11" / layer / route / f"s11_{case.case_id}.csv",
                    frequency_hz,
                    s11,
                )
                save_response(
                    output / "responses" / layer / route / f"response_{case.case_id}.npz",
                    a1,
                    a2,
                    distance_max_m,
                )

    metrics: list[dict] = []
    passivity_tolerance = float(cfg["gates"]["passivity_tolerance"])
    for route in ROUTES:
        for layer in ("clean", "measured"):
            baseline = all_runs[route]["C00"][layer]
            for case in cases:
                run = all_runs[route][case.case_id][layer]
                item = metrics_for(
                    route,
                    layer,
                    case,
                    run,
                    baseline,
                    distance_max_m,
                    passivity_tolerance,
                )
                run["metrics"] = item
                metrics.append(item)

    for layer in ("clean", "measured"):
        for case in cases:
            plot_board(
                assets / f"comparison_board_{layer}_{case.case_id}.png",
                case,
                layer,
                frequency_hz,
                all_runs,
                distance_max_m,
            )

    output.mkdir(parents=True, exist_ok=True)
    flat_rows = []
    for item in metrics:
        row = {key: value for key, value in item.items() if key != "checks"}
        row["checks"] = json.dumps(item["checks"], ensure_ascii=False, sort_keys=True)
        flat_rows.append(row)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    runtime_s = time.perf_counter() - started
    summary = {
        "protocol": cfg,
        "runtime_s": runtime_s,
        "case_count": len(cases),
        "route_count": len(ROUTES),
        "layers": ["clean", "measured"],
        "metrics_count": len(metrics),
        "limitations": [
            "CST V1连续和梯形路线使用相同的固定主RLGC，DG V3只在100 MHz锚定后保留频变R/固定G差异。",
            "numeric_gate_pass只代表有限性和clean被动性，不代表Core-16物理方向已经通过。",
            "算法2的distance_max_m=60 m用于当前V1显示、指标和对照图裁剪；算法2核心计算仍直接调用原V1实现。",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(args.report.resolve(), cfg, cases, metrics, runtime_s)
    print(f"\n完成：{len(metrics)}条指标，{2 * len(cases)}张工况图，耗时{runtime_s:.2f}s")


if __name__ == "__main__":
    main()
