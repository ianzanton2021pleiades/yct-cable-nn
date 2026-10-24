"""Seeded single-sample synthesis for FieldNN from the Core16 RLGC mechanisms."""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
CORE16 = HERE.parents[2] / "DifferentAgent" / "CodeX" / "Core16_Field"
sys.path.insert(0, str(CORE16))

from common import make_targets, preprocess_s11, frequency_grid
from field_physics import simulate

CASE_FILE = CORE16.parent / "DG_Core16_Route_Evaluation" / "core16_cases.yaml"
ROUTE = "dg_v3_rlgc"
RLGC_KEYS = ("r_ohm", "l_uh", "g_us", "c_pf")


def load_case_templates() -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    raw = yaml.safe_load(CASE_FILE.read_text(encoding="utf-8"))
    base = raw["cable"]["base_rlgc_per_0p4m"]
    cases = {item["id"]: item for item in raw["cases"]}
    return cases, base


@lru_cache(maxsize=1)
def load_physics_config() -> dict[str, Any]:
    """Read, but never modify, the established Core16 material and fixture anchors."""
    return json.loads((CORE16 / "field_config.json").read_text(encoding="utf-8"))


def sample_plan(count: int, cfg: dict) -> list[dict[str, Any]]:
    """Create balanced case and length-bin assignments before split processing."""
    randomization = cfg["randomization"]
    cases, _ = load_case_templates()
    case_ids = sorted(cases)
    rng = np.random.default_rng(int(cfg["seed"]))

    baseline_count = int(round(count * float(randomization["case_balance"]["C00_fraction"])))
    other_count = count - baseline_count
    defect_ids = [case_id for case_id in case_ids if case_id != "C00"]
    quotient, remainder = divmod(other_count, len(defect_ids))
    assigned_cases = ["C00"] * baseline_count
    for index, case_id in enumerate(defect_ids):
        assigned_cases.extend([case_id] * (quotient + (index < remainder)))
    rng.shuffle(assigned_cases)

    bins = randomization["length_bins_m"]
    bin_ids = np.arange(count, dtype=np.int64) % len(bins)
    rng.shuffle(bin_ids)
    length_rng = np.random.default_rng(int(cfg["seed"]) + 7_919)
    plan = []
    for sample_id, (case_id, bin_id) in enumerate(zip(assigned_cases, bin_ids)):
        lower, upper = (float(value) for value in bins[int(bin_id)])
        length_m = float(length_rng.uniform(lower, upper))
        split_slot = sample_id % 10
        split = "train" if split_slot < 8 else ("val" if split_slot == 8 else "test")
        plan.append({
            "sample_id": int(sample_id),
            "topology_id": f"fieldnn_topology_{sample_id:08d}",
            "seed": int(cfg["seed"]) + int(sample_id),
            "split": split,
            "case_id": case_id,
            "length_m": length_m,
        })
    return plan


def _uniform(rng: np.random.Generator, bounds: list[float]) -> float:
    return float(rng.uniform(float(bounds[0]), float(bounds[1])))


def _factor(rng: np.random.Generator, bounds: list[float]) -> float:
    return _uniform(rng, bounds)


def _source_profiles(case: dict, base: dict[str, float]) -> list[dict[str, Any]]:
    profiles = []
    for segment in case["segments"]:
        ratios = {
            key: float(segment[key]) / float(base[key])
            for key in RLGC_KEYS
        }
        if any(abs(value - 1.0) > 1.0e-9 for value in ratios.values()):
            profiles.append({
                "ratios": ratios,
                "local_resistance_increment_ohm": float(segment["r_ohm"] - base["r_ohm"]),
                "source_start_m": float(segment["start_m"]),
                "source_end_m": float(segment["end_m"]),
            })
    return profiles


def _sample_intervals(
    case: dict,
    source_base: dict[str, float],
    length_m: float,
    rng: np.random.Generator,
    randomization: dict,
) -> list[dict[str, Any]]:
    profiles = _source_profiles(case, source_base)
    if not profiles:
        return []

    category = str(case["category"])
    margin = max(2.0, float(randomization["defect_position_margin_fraction"]) * length_m)
    intervals: list[dict[str, Any]] = []

    def make_interval(start: float, end: float, profile: dict[str, Any]) -> dict[str, Any]:
        return {
            "start_m": float(start),
            "end_m": float(end),
            "severity_factor": _uniform(rng, randomization["severity_factor"]),
            "ratios": dict(profile["ratios"]),
            "local_resistance_increment_ohm": profile["local_resistance_increment_ohm"],
            "category": category,
        }

    if category == "local":
        width = min(_uniform(rng, randomization["local_width_m"]), length_m - 2.0 * margin)
        start = float(rng.uniform(margin, length_m - margin - width))
        intervals.append(make_interval(start, start + width, profiles[0]))
        return intervals

    if category in {"long_section", "replacement"}:
        bounds = (
            randomization["long_width_fraction"]
            if category == "long_section"
            else randomization["replacement_width_fraction"]
        )
        width = _uniform(rng, bounds) * length_m
        start = float(rng.uniform(margin, length_m - margin - width))
        intervals.append(make_interval(start, start + width, profiles[0]))
        return intervals

    if category == "segmented_loss":
        if len(profiles) == 1:
            width = _uniform(rng, randomization["segmented_width_fraction"]) * length_m
            start = float(rng.uniform(margin, length_m - margin - width))
            intervals.append(make_interval(start, start + width, profiles[0]))
            return intervals

        first_width = _uniform(rng, randomization["segmented_width_fraction"]) * length_m
        first_start = float(rng.uniform(margin, max(margin, 0.42 * length_m)))
        first_end = min(first_start + first_width, 0.64 * length_m)
        intervals.append(make_interval(first_start, first_end, profiles[0]))

        second_width = _uniform(rng, randomization["segmented_width_fraction"]) * length_m
        if case["id"] == "C05":
            second_end = length_m
            second_start = max(first_end + margin * 0.25, length_m - second_width)
        else:
            second_start_min = max(first_end + margin, 0.58 * length_m)
            second_start_max = max(second_start_min, length_m - margin - second_width)
            second_start = float(rng.uniform(second_start_min, second_start_max))
            second_end = second_start + second_width
        intervals.append(make_interval(second_start, second_end, profiles[1]))
        return intervals

    raise ValueError(f"unsupported Core16 category: {category}")


def _sample_joint_positions(
    length_m: float, rng: np.random.Generator, randomization: dict
) -> list[float]:
    spec = randomization["joint_count"]
    count = int(rng.integers(int(spec["min"]), int(spec["max"]) + 1))
    margin = min(float(randomization["joint_edge_margin_m"]), length_m * 0.1)
    spacing = float(randomization["joint_min_spacing_m"])
    positions: list[float] = []
    while len(positions) < count:
        candidate = float(rng.uniform(margin, length_m - margin))
        if all(abs(candidate - existing) >= spacing for existing in positions):
            positions.append(candidate)
    return sorted(positions)


def _jittered_body(physics_config: dict, cfg: dict, rng: np.random.Generator) -> dict[str, float]:
    anchor = physics_config["body"]
    factors = cfg["randomization"]["body_factor"]
    return {
        key: float(anchor[key]) * _factor(rng, factors[key])
        for key in (
            "z0_ohm",
            "velocity_m_s",
            "r_dc_ohm_per_m",
            "r_skin_ref_ohm_per_m",
            "g_dc_s_per_m",
            "tan_delta",
        )
    } | {
        "debye_corner_hz": float(anchor["debye_corner_hz"])
    }


def _sample_fixture(physics_config: dict, cfg: dict, rng: np.random.Generator) -> dict[str, float]:
    anchor = physics_config["fixture"]
    bounds = cfg["randomization"]["fixture_factor"]
    fixture = {}
    for key in ("z0_ohm", "delay_s", "loss_db_ref", "series_r_ohm", "series_l_h"):
        fixture[key] = float(anchor[key]) * _factor(rng, bounds)
    fixture["shunt_g_s"] = float(anchor["shunt_g_s"])
    fixture["shunt_c_f"] = float(anchor["shunt_c_f"])
    return fixture


def _sample_measurement(
    rng: np.random.Generator, cfg: dict
) -> tuple[dict, complex, complex, complex, float, float]:
    prior = cfg["randomization"]["measurement_prior"]
    directivity_magnitude = _uniform(rng, prior["directivity_magnitude"])
    directivity_phase = float(rng.uniform(-np.pi, np.pi))
    source_match_magnitude = _uniform(rng, prior["source_match_magnitude"])
    source_match_phase = float(rng.uniform(-np.pi, np.pi))
    tracking_amplitude = _uniform(rng, prior["tracking_amplitude"])
    tracking_phase = _uniform(rng, prior["tracking_phase_rad"])
    directivity = directivity_magnitude * np.exp(1j * directivity_phase)
    source_match = source_match_magnitude * np.exp(1j * source_match_phase)
    tracking = tracking_amplitude * np.exp(1j * tracking_phase)
    sigma = _uniform(rng, prior["complex_noise_sigma"])
    correlation = _uniform(rng, prior["noise_correlation"])
    measurement = {
        "prior_status": cfg["randomization"]["parameter_status"],
        "model": "D + T*Gamma/(1-S*Gamma) + correlated_complex_noise",
        "directivity_real": float(directivity.real),
        "directivity_imag": float(directivity.imag),
        "source_match_real": float(source_match.real),
        "source_match_imag": float(source_match.imag),
        "tracking_amplitude": tracking_amplitude,
        "tracking_phase_rad": tracking_phase,
        "complex_noise_sigma_rms": sigma,
        "noise_correlation": correlation,
    }
    return measurement, directivity, source_match, tracking, sigma, correlation


def _correlated_noise(
    size: int, sigma: float, correlation: float, rng: np.random.Generator
) -> np.ndarray:
    white = (rng.standard_normal(size) + 1j * rng.standard_normal(size)) / np.sqrt(2.0)
    noise = np.empty(size, dtype=np.complex128)
    noise[0] = white[0]
    innovation_scale = np.sqrt(max(0.0, 1.0 - correlation * correlation))
    for index in range(1, size):
        noise[index] = correlation * noise[index - 1] + innovation_scale * white[index]
    return sigma * noise


def _build_segments(
    length_m: float,
    body: dict[str, float],
    intervals: list[dict[str, Any]],
    rng: np.random.Generator,
    randomization: dict,
) -> list[dict[str, float]]:
    background = randomization["background"]
    blocks = int(background["blocks"])
    edges = np.linspace(0.0, length_m, blocks + 1)
    z_noise = rng.normal(size=blocks)
    v_noise = rng.normal(size=blocks)
    corr = float(background["correlation"])
    for index in range(1, blocks):
        z_noise[index] = corr * z_noise[index - 1] + np.sqrt(1.0 - corr**2) * z_noise[index]
        v_noise[index] = corr * v_noise[index - 1] + np.sqrt(1.0 - corr**2) * v_noise[index]

    boundaries = sorted(set(
        [0.0, float(length_m)]
        + [float(item) for item in edges]
        + [value for interval in intervals for value in (interval["start_m"], interval["end_m"])]
    ))
    segments = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        midpoint = (start + end) / 2.0
        block = min(blocks - 1, int(midpoint / length_m * blocks))
        z0 = body["z0_ohm"] * np.exp(float(background["z_log_sigma"]) * z_noise[block])
        velocity = body["velocity_m_s"] * np.exp(
            float(background["velocity_log_sigma"]) * v_noise[block]
        )
        interval = next(
            (item for item in intervals if item["start_m"] <= midpoint < item["end_m"]),
            None,
        )
        ratios = interval["ratios"] if interval is not None else {key: 1.0 for key in RLGC_KEYS}
        severity = float(interval["severity_factor"]) if interval is not None else 1.0
        r_ratio = 1.0 + (float(ratios["r_ohm"]) - 1.0) * severity
        l_ratio = 1.0 + (float(ratios["l_uh"]) - 1.0) * severity
        g_ratio = 1.0 + (float(ratios["g_us"]) - 1.0) * severity
        c_ratio = 1.0 + (float(ratios["c_pf"]) - 1.0) * severity

        r_dc = body["r_dc_ohm_per_m"]
        r_skin = body["r_skin_ref_ohm_per_m"]
        if interval is not None and interval["category"] == "local":
            r_dc += (
                float(interval["local_resistance_increment_ohm"])
                * severity
                / max(float(interval["end_m"] - interval["start_m"]), 1.0e-6)
            )
        else:
            r_dc *= r_ratio
            r_skin *= r_ratio

        segments.append({
            "start_m": float(start),
            "end_m": float(end),
            "r_dc_ohm_per_m": float(r_dc),
            "r_skin_ref_ohm_per_m": float(r_skin),
            "l_external_h_per_m": float((z0 / velocity) * l_ratio),
            "c_inf_f_per_m": float((1.0 / (z0 * velocity)) * c_ratio),
            "g_dc_s_per_m": float(body["g_dc_s_per_m"] * g_ratio),
            "tan_delta": float(body["tan_delta"] * g_ratio / c_ratio),
            "debye_strength": 0.0,
            "debye_corner_hz": float(body["debye_corner_hz"]),
        })
    return segments


def generate_sample(plan: dict, cfg: dict, templates=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Generate one sample and return complex S11, model input, targets, metadata."""
    if templates is None:
        templates = load_case_templates()
    cases, source_base = templates
    physics_config = cfg["physical_model"]
    rng = np.random.default_rng(int(plan["seed"]))
    case = cases[plan["case_id"]]
    length_m = float(plan["length_m"])
    randomization = cfg["randomization"]
    body = _jittered_body(physics_config, cfg, rng)
    intervals = _sample_intervals(case, source_base, length_m, rng, randomization)
    segments = _build_segments(length_m, body, intervals, rng, randomization)
    joint_positions = _sample_joint_positions(length_m, rng, randomization)
    joint_cfg = physics_config["joint"]
    joint_factor = randomization["joint_strength_factor"]
    joints = []
    for position in joint_positions:
        strength = _uniform(rng, joint_factor)
        joints.append({
            "position_m": float(position),
            "series_r_ohm": float(joint_cfg["series_r_ohm"] * strength),
            "series_l_h": float(joint_cfg["series_l_h"] * strength),
            "shunt_g_s": float(joint_cfg["shunt_g_s"]),
            "shunt_c_f": float(joint_cfg["shunt_c_f"] * strength),
        })

    fixture = _sample_fixture(physics_config, cfg, rng)
    frequency = frequency_grid(cfg)
    physical_s11 = simulate(
        frequency,
        segments,
        joints,
        fixture=fixture,
        route=ROUTE,
        reference_hz=float(physics_config["reference_hz"]),
        load="open",
        zref=float(physics_config.get("zref_ohm", 50.0)),
    )
    measurement, directivity, source_match, tracking, sigma, correlation = _sample_measurement(rng, cfg)
    measured_s11 = directivity + tracking * physical_s11 / (1.0 - source_match * physical_s11)
    measured_s11 += _correlated_noise(frequency.size, sigma, correlation, rng)
    if not (np.isfinite(physical_s11).all() and np.isfinite(measured_s11).all()):
        raise ValueError(f"non-finite S11 in sample {plan['sample_id']}")

    defect_intervals = [
        [float(interval["start_m"]), float(interval["end_m"])]
        for interval in intervals
    ]
    event_positions = [float(position) for position in joint_positions]
    for start, end in defect_intervals:
        event_positions.extend([start, end])
    terminal_position = length_m
    event_positions = sorted({
        round(position, 6)
        for position in event_positions
        if abs(position - terminal_position) > 1.0e-5
    })
    metadata = {
        **plan,
        "case_title": str(case["title"]),
        "category": str(case["category"]),
        "route": ROUTE,
        "termination": "open",
        "terminal_position_m": terminal_position,
        "joint_positions_m": [float(value) for value in joint_positions],
        "joints": joints,
        "defect_intervals_m": defect_intervals,
        "event_positions_m": event_positions,
        "defect_interval_details": [
            {
                "start_m": float(item["start_m"]),
                "end_m": float(item["end_m"]),
                "case_category": item["category"],
                "severity_factor": float(item["severity_factor"]),
                "rlgc_ratio_before_severity": item["ratios"],
                "local_resistance_increment_ohm": float(
                    item["local_resistance_increment_ohm"]
                ),
            }
            for item in intervals
        ],
        "defect_label_semantics": (
            "synthetic RLGC parameter-change interval; field detectability and "
            "impedance-discontinuity status are not asserted"
        ),
        "segments": segments,
        "fixture": fixture,
        "measurement": measurement,
        "body_sample": body,
        "parameter_status": randomization["parameter_status"],
        "s11_diagnostics": {
            "physical_max_abs": float(np.max(np.abs(physical_s11))),
            "measured_max_abs": float(np.max(np.abs(measured_s11))),
            "physical_rms_abs": float(np.sqrt(np.mean(np.abs(physical_s11) ** 2))),
            "measured_rms_abs": float(np.sqrt(np.mean(np.abs(measured_s11) ** 2))),
            "physical_passive_with_1e-6_tolerance": bool(
                np.max(np.abs(physical_s11)) <= 1.0 + 1.0e-6
            ),
        },
    }
    inputs = preprocess_s11(frequency, measured_s11, cfg)
    targets = make_targets(metadata, cfg)
    return measured_s11.astype(np.complex64), inputs, targets, metadata


__all__ = [
    "generate_sample",
    "load_case_templates",
    "load_physics_config",
    "sample_plan",
]
