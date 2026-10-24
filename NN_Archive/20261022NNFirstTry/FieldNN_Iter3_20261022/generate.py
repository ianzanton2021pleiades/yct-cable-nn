"""Generate FieldNN Iter3 S11, response features, and observable labels on CPU."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
LEGACY = PROJECT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_20261022"
CORE16 = PROJECT / "DifferentAgent" / "CodeX" / "Core16_Field"
DEFAULT_OUTPUT = Path(r"E:\CableNNDataSet\CodeX\FieldNN_Iter3_20261022")
SPLITS = ("train", "val", "test")
_LEGACY_REFERENCE_HZ = 1.0e6
_DB_PER_NP = 8.685889638

sys.path.insert(0, str(LEGACY))
from generator import (  # noqa: E402
    _build_segments,
    _correlated_noise,
    _sample_intervals,
    _sample_joint_positions,
    load_case_templates,
)
from field_physics import _segment_zy, simulate  # noqa: E402
from features import distance_features  # noqa: E402
from field_responses import algorithm1  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _uniform(rng: np.random.Generator, bounds: list[float]) -> float:
    return float(rng.uniform(float(bounds[0]), float(bounds[1])))


def _frequency_grid(cfg: dict[str, Any]) -> np.ndarray:
    spec = cfg["frequency"]
    grid = np.geomspace if spec.get("spacing", "linear") == "log" else np.linspace
    return grid(
        float(spec["start_hz"]),
        float(spec["stop_hz"]),
        int(spec["points"]),
        dtype=np.float64,
    )


def _distance_grid(cfg: dict[str, Any]) -> np.ndarray:
    response = cfg["response"]
    return np.linspace(
        0.0,
        float(response["distance_max_m"]),
        int(response["points"]),
        dtype=np.float64,
    )


def _attenuation_np_per_m(
    z0_ohm: float,
    velocity_m_s: float,
    tan_delta: float,
    r_skin_100mhz_ohm_per_m: float,
    g_dc_s_per_m: float = 1e-10,
) -> float:
    """Return Re(gamma) at 100 MHz for an input Rskin specified at 100 MHz."""
    segment = {
        "r_dc_ohm_per_m": 0.0,
        "r_skin_ref_ohm_per_m": float(r_skin_100mhz_ohm_per_m)
        * math.sqrt(_LEGACY_REFERENCE_HZ / 100.0e6),
        "l_external_h_per_m": float(z0_ohm) / float(velocity_m_s),
        "c_inf_f_per_m": 1.0 / (float(z0_ohm) * float(velocity_m_s)),
        "g_dc_s_per_m": float(g_dc_s_per_m),
        "tan_delta": float(tan_delta),
        "debye_strength": 0.0,
        "debye_corner_hz": _LEGACY_REFERENCE_HZ,
    }
    f = np.asarray([100.0e6], dtype=np.float64)
    z, y = _segment_zy(f, segment, _LEGACY_REFERENCE_HZ, "dg_v3_rlgc")
    return float(np.sqrt(z * y).real[0])


def _solve_tan_delta(
    z0_ohm: float,
    velocity_m_s: float,
    dielectric_target_np_per_m: float,
    tan_delta_max: float,
    g_dc_s_per_m: float,
) -> float:
    lower, upper = 0.0, float(tan_delta_max)
    for _ in range(64):
        middle = 0.5 * (lower + upper)
        loss = _attenuation_np_per_m(
            z0_ohm, velocity_m_s, middle, 0.0, g_dc_s_per_m
        )
        if loss < dielectric_target_np_per_m:
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)


def _solve_skin_resistance(
    z0_ohm: float,
    velocity_m_s: float,
    tan_delta: float,
    target_alpha_db_per_m: float,
    g_dc_s_per_m: float,
) -> float:
    target_np = float(target_alpha_db_per_m) / _DB_PER_NP
    lower, upper = 0.0, 1.0
    while _attenuation_np_per_m(
        z0_ohm, velocity_m_s, tan_delta, upper, g_dc_s_per_m
    ) < target_np:
        upper *= 2.0
    for _ in range(64):
        middle = 0.5 * (lower + upper)
        loss = _attenuation_np_per_m(
            z0_ohm, velocity_m_s, tan_delta, middle, g_dc_s_per_m
        )
        if loss < target_np:
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)


def _sample_body(
    rng: np.random.Generator, cfg: dict[str, Any]
) -> tuple[dict[str, float], dict[str, Any]]:
    physical = cfg["physical_model"]
    prior = physical["main_body_prior"]
    if rng.random() < float(physical["legacy_high_loss_anchor_fraction"]):
        body = dict(physical["legacy_high_loss_body"])
        alpha = _attenuation_np_per_m(
            body["z0_ohm"],
            body["velocity_m_s"],
            body["tan_delta"],
            body["r_skin_ref_ohm_per_m"]
            * math.sqrt(100.0e6 / _LEGACY_REFERENCE_HZ),
            body["g_dc_s_per_m"],
        ) * _DB_PER_NP
        return body, {
            "profile": "legacy_core16_high_loss_anchor",
            "epsr_equivalent": (299_792_458.0 / body["velocity_m_s"]) ** 2,
            "alpha_db_per_m_at_100mhz": float(alpha),
            "tan_delta_at_100mhz": float(body["tan_delta"]),
            "dielectric_fraction_of_total": None,
        }

    z0 = _uniform(rng, prior["z0_ohm"])
    velocity = _uniform(rng, prior["velocity_m_s"])
    alpha_db = _uniform(rng, prior["alpha_db_per_m_at_100mhz"])
    dielectric_fraction = _uniform(
        rng, prior["dielectric_loss_fraction_of_total"]
    )
    g_dc = float(prior["g_dc_s_per_m"])
    dielectric_target_np = (
        alpha_db * dielectric_fraction / _DB_PER_NP
    )
    tan_delta = _solve_tan_delta(
        z0,
        velocity,
        dielectric_target_np,
        float(prior["tan_delta_max_at_100mhz"]),
        g_dc,
    )
    r_skin_100mhz = _solve_skin_resistance(
        z0, velocity, tan_delta, alpha_db, g_dc
    )
    body = {
        "z0_ohm": z0,
        "velocity_m_s": velocity,
        "r_dc_ohm_per_m": float(prior["r_dc_ohm_per_m"]),
        "r_skin_ref_ohm_per_m": r_skin_100mhz
        * math.sqrt(_LEGACY_REFERENCE_HZ / 100.0e6),
        "g_dc_s_per_m": g_dc,
        "tan_delta": tan_delta,
        "debye_corner_hz": float(prior["debye_corner_hz"]),
    }
    return body, {
        "profile": "joint_effective_rlgc_prior",
        "epsr_equivalent": (299_792_458.0 / velocity) ** 2,
        "alpha_db_per_m_at_100mhz": float(alpha_db),
        "tan_delta_at_100mhz": float(tan_delta),
        "dielectric_fraction_of_total": float(dielectric_fraction),
        "r_skin_100mhz_ohm_per_m": float(r_skin_100mhz),
    }


def _sample_fixture(
    rng: np.random.Generator, cfg: dict[str, Any], legacy_fixture: dict[str, Any]
) -> dict[str, float]:
    prior = cfg["physical_model"]["fixture_prior"]
    if rng.random() < float(prior["legacy_anchor_fraction"]):
        return {key: float(value) for key, value in legacy_fixture.items()}
    if rng.random() < float(prior["short_delay_fraction"]):
        delay = _uniform(rng, prior["short_delay_s"])
        delay_family = "short"
    else:
        delay = _uniform(rng, prior["long_delay_s"])
        delay_family = "long"
    loss_at_100mhz = _uniform(rng, prior["loss_db_at_100mhz"])
    return {
        "z0_ohm": _uniform(rng, prior["z0_ohm"]),
        "delay_s": delay,
        "loss_db_ref": loss_at_100mhz
        * math.sqrt(_LEGACY_REFERENCE_HZ / 100.0e6),
        "loss_db_at_100mhz_sampled": loss_at_100mhz,
        "series_r_ohm": _uniform(rng, prior["series_r_ohm"]),
        "series_l_h": _uniform(rng, prior["series_l_h"]),
        "shunt_g_s": 0.0,
        "shunt_c_f": _uniform(rng, prior["shunt_c_f"]),
        "delay_family": delay_family,
    }


def _sample_joints(
    positions: list[float], rng: np.random.Generator, cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    variants = cfg["physical_model"]["joint_variants"]
    names = list(variants)
    weights = np.asarray([variants[name]["weight"] for name in names], dtype=float)
    weights /= np.sum(weights)
    joints = []
    for position in positions:
        name = str(rng.choice(names, p=weights))
        spec = variants[name]
        joints.append({
            "position_m": float(position),
            "variant": name,
            "series_r_ohm": _uniform(rng, spec["series_r_ohm"]),
            "series_l_h": _uniform(rng, spec["series_l_h"]),
            "shunt_g_s": 0.0,
            "shunt_c_f": _uniform(rng, spec["shunt_c_f"]),
        })
    return joints


def _sample_measurement(
    rng: np.random.Generator,
    cfg: dict[str, Any],
    frequency_hz: np.ndarray,
) -> tuple[dict[str, Any], complex, complex, complex, float, float, float]:
    prior = cfg["physical_model"]["measurement_prior"]
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
    low_frequency_factor = _uniform(rng, prior["low_frequency_noise_factor"])
    envelope = 1.0 + (low_frequency_factor - 1.0) * np.exp(
        -frequency_hz / 5.0e6
    )
    metadata = {
        "model": "D + T*Gamma/(1-S*Gamma) + correlated_complex_noise",
        "prior_status": cfg["physical_model"]["parameter_status"],
        "directivity_real": float(directivity.real),
        "directivity_imag": float(directivity.imag),
        "source_match_real": float(source_match.real),
        "source_match_imag": float(source_match.imag),
        "tracking_amplitude": tracking_amplitude,
        "tracking_phase_rad": tracking_phase,
        "complex_noise_sigma_rms": sigma,
        "noise_correlation": correlation,
        "low_frequency_noise_factor": low_frequency_factor,
    }
    return metadata, directivity, source_match, tracking, sigma, correlation, low_frequency_factor


def _algorithm1_spectrum(
    frequency_hz: np.ndarray, signal: np.ndarray, cfg: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    band = cfg["frequency"]
    keep = (frequency_hz >= float(band["start_hz"])) & (
        frequency_hz <= float(band["stop_hz"])
    )
    frequency = np.unique(
        np.r_[float(band["start_hz"]), frequency_hz[keep], float(band["stop_hz"])]
    )
    uniform = np.interp(frequency, frequency_hz, signal.real) + 1j * np.interp(
        frequency, frequency_hz, signal.imag
    )
    return frequency, uniform


def _algorithm1_traces(
    frequency_hz: np.ndarray, signal: np.ndarray, cfg: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequency, uniform = _algorithm1_spectrum(frequency_hz, signal, cfg)
    response = cfg["response"]
    transformed = algorithm1(
        frequency,
        uniform,
        velocity_m_s=float(response["velocity_m_s"]),
        distance_max_m=float(response["distance_max_m"]),
    )
    distance = _distance_grid(cfg)
    impulse = np.interp(distance, transformed["distance_m"], transformed["impulse_raw"])
    step = np.interp(distance, transformed["distance_m"], transformed["step_rho"])
    return impulse, step, distance


def _segment_spectra(
    frequency_hz: np.ndarray, segments: list[dict[str, float]]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    gammas, characteristics = [], []
    for segment in segments:
        impedance, admittance = _segment_zy(
            frequency_hz, segment, _LEGACY_REFERENCE_HZ, "dg_v3_rlgc"
        )
        gammas.append(np.sqrt(impedance * admittance))
        characteristics.append(np.sqrt(impedance / admittance))
    return gammas, characteristics


def _event_reflection(
    event: dict[str, Any],
    frequency_hz: np.ndarray,
    segments: list[dict[str, float]],
    characteristics: list[np.ndarray],
    joints: list[dict[str, Any]],
    load_ohm: float,
) -> np.ndarray:
    role = str(event["role"])
    if role == "joint":
        joint = joints[int(event["joint_index"])]
        omega = 2.0 * np.pi * frequency_hz
        series = float(joint["series_r_ohm"]) + 1j * omega * float(joint["series_l_h"])
        shunt = float(joint["shunt_g_s"]) + 1j * omega * float(joint["shunt_c_f"])
        a, b, c, d = 1.0 + series * shunt, series, shunt, 1.0
        numerator = a + b / 50.0 - 50.0 * c - d
        denominator = a + b / 50.0 + 50.0 * c + d
        return numerator / denominator
    if role == "terminal":
        zc = characteristics[-1]
        return (float(load_ohm) - zc) / (float(load_ohm) + zc)

    position = float(event["physical_position_m"])
    tolerance = 1.0e-7
    left_index = next(
        index for index in range(len(segments) - 1, -1, -1)
        if abs(float(segments[index]["end_m"]) - position) <= tolerance
    )
    right_index = next(
        index for index, segment in enumerate(segments)
        if abs(float(segment["start_m"]) - position) <= tolerance
    )
    z_left, z_right = characteristics[left_index], characteristics[right_index]
    return (z_right - z_left) / (z_right + z_left)


def _main_lobe_width(distance: np.ndarray, impulse: np.ndarray) -> tuple[int, float, float]:
    magnitude = np.abs(impulse)
    peak_index = int(np.argmax(magnitude))
    peak = float(magnitude[peak_index])
    half = 0.5 * peak
    left = peak_index
    right = peak_index
    while left > 0 and magnitude[left - 1] >= half:
        left -= 1
    while right + 1 < magnitude.size and magnitude[right + 1] >= half:
        right += 1
    return peak_index, peak, float(distance[right] - distance[left])


def _propagation_kernel(
    event: dict[str, Any],
    frequency_hz: np.ndarray,
    segments: list[dict[str, float]],
    gammas: list[np.ndarray],
    reflection: np.ndarray,
    fixture: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[float, float, float]:
    position = float(event["physical_position_m"])
    path = np.zeros(frequency_hz.size, dtype=np.complex128)
    for segment, gamma in zip(segments, gammas):
        overlap = min(position, float(segment["end_m"])) - float(segment["start_m"])
        if overlap > 0.0:
            path += gamma * min(overlap, float(segment["end_m"]) - float(segment["start_m"]))
        if position <= float(segment["end_m"]):
            break
    fixture_alpha = (
        math.log(10.0) / 20.0
        * float(fixture["loss_db_ref"])
        * np.sqrt(frequency_hz / _LEGACY_REFERENCE_HZ)
    )
    fixture_gamma = fixture_alpha + 1j * 2.0 * np.pi * frequency_hz * float(fixture["delay_s"])
    kernel = reflection * np.exp(-2.0 * (path + fixture_gamma))
    distance = _distance_grid(cfg)
    kernel_frequency, uniform_kernel = _algorithm1_spectrum(frequency_hz, kernel, cfg)
    response = cfg["response"]
    transformed = algorithm1(
        kernel_frequency,
        uniform_kernel,
        velocity_m_s=float(response["velocity_m_s"]),
        distance_max_m=float(response["distance_max_m"]),
    )
    impulse = np.interp(distance, transformed["distance_m"], transformed["impulse_raw"])
    index, peak, width = _main_lobe_width(distance, impulse)
    return float(distance[index]), peak, width


def _kernel_observability(
    position_m: float,
    peak_amplitude: float,
    lobe_width_m: float,
    noise_impulse: np.ndarray,
    distance: np.ndarray,
    cfg: dict[str, Any],
    terminal: bool = False,
) -> dict[str, Any]:
    radius = max(25.0, 2.0 * float(lobe_width_m))
    local_noise = np.abs(noise_impulse[np.abs(distance - position_m) <= radius])
    median = float(np.median(local_noise)) if local_noise.size else 0.0
    mad = float(np.median(np.abs(local_noise - median))) if local_noise.size else 0.0
    noise_floor = median + 3.0 * 1.4826 * mad
    snr = float(peak_amplitude / max(noise_floor, 1.0e-12))
    if terminal:
        max_width = float(cfg["response"]["max_terminal_lobe_width_m"])
        min_snr = float(cfg["response"]["minimum_terminal_kernel_snr"])
    else:
        max_width = float(cfg["response"]["max_event_lobe_width_m"])
        min_snr = float(cfg["response"]["minimum_event_kernel_snr"])
    return {
        "observable": bool(snr >= min_snr and lobe_width_m <= max_width),
        "kernel_peak_abs_impulse": float(peak_amplitude),
        "noise_floor_abs_impulse": float(noise_floor),
        "kernel_snr": snr,
        "main_lobe_fwhm_m": float(lobe_width_m),
        "minimum_snr": min_snr,
        "maximum_lobe_width_m": max_width,
        "method": "isolated_propagation_kernel_vs_transformed_noise_weak_supervision_v1",
    }


def _make_targets(
    event_records: list[dict[str, Any]],
    interval_records: list[dict[str, Any]],
    terminal_record: dict[str, Any],
    distance: np.ndarray,
    cfg: dict[str, Any],
) -> np.ndarray:
    target = np.zeros((3, distance.size), dtype=np.float32)
    event_sigma = float(cfg["response"]["event_sigma_m"])
    ignore_window = float(cfg["response"]["ignore_window_m"])
    ignored_events = np.zeros(distance.size, dtype=bool)
    protected_events = np.zeros(distance.size, dtype=bool)
    for event in event_records:
        position = float(event["response_axis_position_m"])
        window = np.abs(distance - position) <= ignore_window
        if event["observable"]:
            target[0] = np.maximum(
                target[0],
                np.exp(-0.5 * ((distance - position) / event_sigma) ** 2),
            )
            protected_events |= np.abs(distance - position) <= 2.0 * event_sigma
        else:
            ignored_events |= window
    target[0][ignored_events & ~protected_events] = -1.0
    for interval in interval_records:
        start = float(interval["response_axis_start_m"])
        end = float(interval["response_axis_end_m"])
        if interval["observable"]:
            target[1, (distance >= start) & (distance <= end)] = 1.0
        else:
            target[1, (distance >= start - ignore_window) & (distance <= end + ignore_window)] = -1.0
    if terminal_record["observable"]:
        terminal_sigma = float(cfg["response"]["terminal_sigma_m"])
        terminal_position_m = float(terminal_record["response_axis_position_m"])
        target[2] = np.exp(
            -0.5 * ((distance - terminal_position_m) / terminal_sigma) ** 2
        )
    else:
        position = float(terminal_record["response_axis_position_m"])
        target[2, np.abs(distance - position) <= ignore_window] = -1.0
    return target


def _negative_windows(targets: np.ndarray, distance: np.ndarray) -> dict[str, Any]:
    names = ("event", "defect_interval", "terminal")
    windows = {}
    for channel, name in enumerate(names):
        ignored = np.flatnonzero(targets[channel] < 0.0)
        if ignored.size == 0:
            windows[name] = []
            continue
        breaks = np.flatnonzero(np.diff(ignored) > 1)
        starts = np.r_[ignored[0], ignored[breaks + 1]]
        ends = np.r_[ignored[breaks], ignored[-1]]
        windows[name] = [
            [float(distance[start]), float(distance[end])]
            for start, end in zip(starts, ends)
        ]
    return windows


def generate_sample(
    plan: dict[str, Any],
    cfg: dict[str, Any],
    templates: tuple[dict[str, dict[str, Any]], dict[str, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    cases, source_base = templates
    rng = np.random.default_rng(int(plan["seed"]))
    case = cases[str(plan["case_id"])]
    length_m = float(plan["length_m"])
    randomization = cfg["randomization"]
    sample_randomization = dict(randomization)
    if plan["sample_kind"] == "healthy_no_event":
        sample_randomization["joint_count"] = {"min": 0, "max": 0}
    elif plan["sample_kind"] == "joint_only":
        sample_randomization["joint_count"] = dict(randomization["joint_only_count"])

    body, body_material = _sample_body(rng, cfg)
    intervals = _sample_intervals(
        case, source_base, length_m, rng, sample_randomization
    )
    segments = _build_segments(
        length_m, body, intervals, rng, sample_randomization
    )
    joint_positions = _sample_joint_positions(
        length_m, rng, sample_randomization
    )
    joints = _sample_joints(joint_positions, rng, cfg)
    legacy_fixture = cfg["physical_model"]["legacy_fixture_anchor"]
    fixture = _sample_fixture(rng, cfg, legacy_fixture)
    frequency = _frequency_grid(cfg)
    term_prior = cfg["physical_model"]["termination_prior"]
    termination = "open" if rng.random() < float(term_prior["open_probability"]) else "short"
    load_bounds = term_prior["open_load_ohm"] if termination == "open" else term_prior["short_load_ohm"]
    load_ohm = _uniform(rng, load_bounds)
    physical_s11 = simulate(
        frequency,
        segments,
        joints,
        fixture=fixture,
        route="dg_v3_rlgc",
        reference_hz=_LEGACY_REFERENCE_HZ,
        load=load_ohm,
        zref=50.0,
    )

    measurement, directivity, source_match, tracking, sigma, correlation, low_frequency_factor = (
        _sample_measurement(rng, cfg, frequency)
    )
    measured_s11 = directivity + tracking * physical_s11 / (
        1.0 - source_match * physical_s11
    )
    noise = _correlated_noise(frequency.size, sigma, correlation, rng)
    noise *= 1.0 + (low_frequency_factor - 1.0) * np.exp(-frequency / 5.0e6)
    measured_s11 += noise

    physical_defects = [
        [float(item["start_m"]), float(item["end_m"])] for item in intervals
    ]
    physical_event_positions = sorted(
        {
            round(float(position), 6)
            for position in joint_positions
            + [edge for interval in physical_defects for edge in interval]
            if abs(float(position) - length_m) > 1.0e-5
        }
    )
    gammas, characteristics = _segment_spectra(frequency, segments)
    impulse, step, distance = _algorithm1_traces(frequency, measured_s11, cfg)
    features = distance_features(impulse, step, distance)
    noise_impulse, _, _ = _algorithm1_traces(frequency, noise, cfg)

    event_specs = []
    for joint_index, position in enumerate(joint_positions):
        event_specs.append({
            "event_id": f"joint_{joint_index}",
            "role": "joint",
            "joint_index": int(joint_index),
            "physical_position_m": float(position),
        })
    for defect_index, (start, end) in enumerate(physical_defects):
        event_specs.append({
            "event_id": f"defect_{defect_index}_start",
            "role": "defect_boundary",
            "defect_index": int(defect_index),
            "boundary": "start",
            "physical_position_m": float(start),
        })
        if abs(float(end) - length_m) > 1.0e-5:
            event_specs.append({
                "event_id": f"defect_{defect_index}_end",
                "role": "defect_boundary",
                "defect_index": int(defect_index),
                "boundary": "end",
                "physical_position_m": float(end),
            })
    terminal_spec = {
        "event_id": "terminal",
        "role": "terminal",
        "physical_position_m": float(length_m),
    }

    def locate_event(spec: dict[str, Any], terminal: bool = False) -> dict[str, Any]:
        reflection = _event_reflection(
            spec, frequency, segments, characteristics, joints, load_ohm
        )
        response_position, peak, width = _propagation_kernel(
            spec, frequency, segments, gammas, reflection, fixture, cfg
        )
        visibility = _kernel_observability(
            response_position, peak, width, noise_impulse, distance, cfg, terminal
        )
        return {
            **spec,
            "response_axis_position_m": float(response_position),
            **visibility,
        }

    event_records = [locate_event(spec) for spec in event_specs]
    terminal_record = locate_event(terminal_spec, terminal=True)
    response_intervals = []
    interval_records = []
    for defect_index, physical_interval in enumerate(physical_defects):
        start_record = next(
            item for item in event_records
            if item["event_id"] == f"defect_{defect_index}_start"
        )
        end_at_terminal = abs(float(physical_interval[1]) - length_m) <= 1.0e-5
        end_record = terminal_record if end_at_terminal else next(
            item for item in event_records
            if item["event_id"] == f"defect_{defect_index}_end"
        )
        response_start, response_end = sorted((
            float(start_record["response_axis_position_m"]),
            float(end_record["response_axis_position_m"]),
        ))
        interval_visible = bool(start_record["observable"] or end_record["observable"])
        response_intervals.append([response_start, response_end])
        interval_records.append({
            "physical_start_m": float(physical_interval[0]),
            "physical_end_m": float(physical_interval[1]),
            "response_axis_start_m": response_start,
            "response_axis_end_m": response_end,
            "observable": interval_visible,
            "visibility_basis": "at_least_one_boundary_propagation_kernel_visible",
            "end_uses_terminal_kernel": end_at_terminal,
        })
    targets = _make_targets(event_records, interval_records, terminal_record, distance, cfg)
    ignored_windows = _negative_windows(targets, distance)

    response_event_positions = sorted({
        round(float(event["response_axis_position_m"]), 6) for event in event_records
    })
    training_event_positions = sorted({
        round(float(event["response_axis_position_m"]), 6)
        for event in event_records if event["observable"]
    })
    training_intervals = [
        [record["response_axis_start_m"], record["response_axis_end_m"]]
        for record in interval_records if record["observable"]
    ]
    training_terminal_position = (
        float(terminal_record["response_axis_position_m"])
        if terminal_record["observable"] else None
    )

    actual_alpha_db = _attenuation_np_per_m(
        body["z0_ohm"], body["velocity_m_s"], body["tan_delta"],
        body["r_skin_ref_ohm_per_m"]
        * math.sqrt(100.0e6 / _LEGACY_REFERENCE_HZ),
        body["g_dc_s_per_m"],
    ) * _DB_PER_NP
    metadata = {
        **plan,
        "case_title": str(case["title"]),
        "category": str(case["category"]),
        "route": "dg_v3_rlgc",
        "termination": termination,
        "load_ohm": float(load_ohm),
        "physical_length_m": float(length_m),
        "terminal_position_m": training_terminal_position,
        "joint_positions_m": [float(value) for value in joint_positions],
        "defect_intervals_m": training_intervals,
        "event_positions_m": training_event_positions,
        "physical_terminal_position_m": float(length_m),
        "physical_joint_positions_m": [float(value) for value in joint_positions],
        "physical_defect_intervals_m": physical_defects,
        "physical_event_positions_m": physical_event_positions,
        "response_axis": {
            "method": "event reflection times exp(-2*cumulative_gamma*length) and fixture propagation, transformed by Algorithm1",
            "velocity_m_s": float(cfg["response"]["velocity_m_s"]),
            "is_exact_full_network_peak": False,
            "note": "isolated first-reflection propagation-kernel main lobe; excludes multiple-reflection interference and is not a field gold label",
        },
        "response_terminal_candidate_position_m": float(terminal_record["response_axis_position_m"]),
        "response_event_positions_m": response_event_positions,
        "response_defect_intervals_m": response_intervals,
        "training_event_positions_m": training_event_positions,
        "training_defect_intervals_m": training_intervals,
        "terminal_observable": bool(terminal_record["observable"]),
        "terminal_observability": terminal_record,
        "event_observability": event_records,
        "defect_interval_observability": interval_records,
        "weak_supervision_note": "Unobservable physical events are not positives: target=-1 masks their local windows. Visibility is a propagation-kernel/noise heuristic, not measured ground truth.",
        "target_ignore_windows_m": ignored_windows,
        "target_ignore_coordinate": "Algorithm1 response axis in meters; exact contiguous spans where targets.npy is -1",
        "defect_label_semantics": "observable synthetic RLGC parameter-change intervals; physical geometry is retained separately",
        "segments": segments,
        "joints": joints,
        "fixture": fixture,
        "measurement": measurement,
        "body_sample": {
            **body,
            "body_profile": body_material["profile"],
            "epsr_equivalent": body_material["epsr_equivalent"],
            "alpha_db_per_m_at_100mhz_sampled": body_material["alpha_db_per_m_at_100mhz"],
            "alpha_db_per_m_at_100mhz_from_legacy_rlgc": float(actual_alpha_db),
            "r_skin_100mhz_ohm_per_m_sampled": body_material.get("r_skin_100mhz_ohm_per_m"),
            "r_skin_100mhz_ohm_per_m_from_legacy_rlgc": float(
                body["r_skin_ref_ohm_per_m"]
                * math.sqrt(100.0e6 / _LEGACY_REFERENCE_HZ)
            ),
            "tan_delta_at_100mhz": body_material["tan_delta_at_100mhz"],
            "dielectric_fraction_of_total": body_material["dielectric_fraction_of_total"],
        },
        "parameter_status": cfg["physical_model"]["parameter_status"],
        "s11_diagnostics": {
            "physical_max_abs": float(np.max(np.abs(physical_s11))),
            "measured_max_abs": float(np.max(np.abs(measured_s11))),
            "physical_rms_abs": float(np.sqrt(np.mean(np.abs(physical_s11) ** 2))),
            "measured_rms_abs": float(np.sqrt(np.mean(np.abs(measured_s11) ** 2))),
        },
    }
    return measured_s11.astype(np.complex64), features, targets, metadata


def _sample_plan(count: int, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    randomization = cfg["randomization"]
    dataset = cfg["dataset"]
    rng = np.random.default_rng(int(cfg["seed"]))
    case_templates, _ = load_case_templates()
    defect_cases = sorted(case_id for case_id in case_templates if case_id != "C00")
    negative_count = int(round(count * float(dataset["healthy_zero_event_fraction"])))
    joint_only_count = int(round(count * float(dataset["joint_only_fraction"])))
    assignments = [
        ("C00", "healthy_no_event") for _ in range(negative_count)
    ] + [("C00", "joint_only") for _ in range(joint_only_count)]
    remaining = count - len(assignments)
    assignments.extend(
        (defect_cases[index % len(defect_cases)], "defect_case")
        for index in range(remaining)
    )
    rng.shuffle(assignments)

    bins = randomization["length_bins_m"]
    bin_ids = np.arange(count, dtype=np.int64) % len(bins)
    rng.shuffle(bin_ids)
    length_rng = np.random.default_rng(int(cfg["seed"]) + 7_919)
    plan = []
    for sample_id, ((case_id, sample_kind), bin_id) in enumerate(
        zip(assignments, bin_ids)
    ):
        lower, upper = (float(value) for value in bins[int(bin_id)])
        plan.append({
            "sample_id": int(sample_id),
            "topology_id": f"fieldnn_iter3_{sample_id:08d}",
            "seed": int(cfg["seed"]) + int(sample_id),
            "split": "train" if sample_id % 10 < 8 else (
                "val" if sample_id % 10 == 8 else "test"
            ),
            "case_id": case_id,
            "sample_kind": sample_kind,
            "length_m": float(length_rng.uniform(lower, upper)),
        })
    return plan


def _make_shard_jobs(plan: list[dict[str, Any]], shard_size: int) -> list[dict[str, Any]]:
    jobs = []
    for split in SPLITS:
        rows = [item for item in plan if item["split"] == split]
        for shard_index, start in enumerate(range(0, len(rows), shard_size)):
            jobs.append({
                "split": split,
                "shard_index": shard_index,
                "plan": rows[start : start + shard_size],
            })
    return jobs


def _shard_folder(root: Path, job: dict[str, Any]) -> Path:
    return root / "shards" / job["split"] / f"{int(job['shard_index']):05d}"


def _canonical_config(cfg: dict[str, Any]) -> str:
    return json.dumps(cfg, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _shard_complete(root: Path, job: dict[str, Any], cfg: dict[str, Any]) -> bool:
    folder = _shard_folder(root, job)
    required = [
        folder / "s11.npy",
        folder / "inputs.npy",
        folder / "targets.npy",
        folder / "metadata.jsonl",
        folder / "summary.json",
    ]
    if not all(path.is_file() for path in required):
        return False
    summary = _read_json(folder / "summary.json")
    count = len(job["plan"])
    if not summary.get("complete") or int(summary.get("count", -1)) != count:
        return False
    expected = {
        "s11.npy": (count, int(cfg["frequency"]["points"])),
        "inputs.npy": (count, int(cfg["model"]["input_channels"]), int(cfg["response"]["points"])),
        "targets.npy": (count, 3, int(cfg["response"]["points"])),
    }
    for name, shape in expected.items():
        values = np.load(folder / name, mmap_mode="r")
        valid = values.shape == shape
        mmap = getattr(values, "_mmap", None)
        if mmap is not None:
            mmap.close()
        if not valid:
            return False
    with (folder / "metadata.jsonl").open("r", encoding="utf-8") as stream:
        return sum(1 for _ in stream) == count


def _write_shard(root_text: str, cfg: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    root = Path(root_text)
    folder = _shard_folder(root, job)
    folder.mkdir(parents=True, exist_ok=True)
    rows = job["plan"]
    count = len(rows)
    frequency_count = int(cfg["frequency"]["points"])
    distance_count = int(cfg["response"]["points"])
    inputs = np.empty(
        (count, int(cfg["model"]["input_channels"]), distance_count), dtype=np.float32
    )
    targets = np.empty((count, 3, distance_count), dtype=np.float32)
    s11_values = np.empty((count, frequency_count), dtype=np.complex64)
    visible_events = 0
    invisible_events = 0
    visible_terminals = 0
    physical_max_abs = []
    alpha_target_residuals = []
    templates = load_case_templates()
    metadata_path = folder / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row_index, row in enumerate(rows):
            s11, sample_inputs, sample_targets, metadata = generate_sample(
                row, cfg, templates
            )
            if sample_inputs.shape != (int(cfg["model"]["input_channels"]), distance_count):
                raise ValueError(f"input shape is {sample_inputs.shape} for sample {row['sample_id']}")
            s11_values[row_index] = s11
            inputs[row_index] = sample_inputs
            targets[row_index] = sample_targets
            visible_events += len(metadata["training_event_positions_m"])
            invisible_events += sum(
                not event["observable"] for event in metadata["event_observability"]
            )
            visible_terminals += int(metadata["terminal_observable"])
            physical_max_abs.append(metadata["s11_diagnostics"]["physical_max_abs"])
            body = metadata["body_sample"]
            alpha_target_residuals.append(abs(
                float(body["alpha_db_per_m_at_100mhz_sampled"])
                - float(body["alpha_db_per_m_at_100mhz_from_legacy_rlgc"])
            ))
            stream.write(
                json.dumps(metadata, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                + "\n"
            )
    np.save(folder / "s11.npy", s11_values)
    np.save(folder / "inputs.npy", inputs)
    np.save(folder / "targets.npy", targets)
    summary = {
        "schema": "fieldnn-iter3-shard-1",
        "complete": True,
        "split": job["split"],
        "shard_index": int(job["shard_index"]),
        "count": count,
        "sample_id_min": min(int(row["sample_id"]) for row in rows),
        "sample_id_max": max(int(row["sample_id"]) for row in rows),
        "planned_healthy_zero_event_count": sum(row["sample_kind"] == "healthy_no_event" for row in rows),
        "visible_event_count": visible_events,
        "unobservable_event_count": invisible_events,
        "observable_terminal_count": visible_terminals,
        "physical_s11_max_abs": {
            "min": float(np.min(physical_max_abs)),
            "median": float(np.median(physical_max_abs)),
            "q95": float(np.percentile(physical_max_abs, 95)),
            "max": float(np.max(physical_max_abs)),
        },
        "material_alpha_target_residual_db_per_m_max": float(np.max(alpha_target_residuals)),
        "ignored_point_fraction_by_channel": {
            name: float(np.count_nonzero(targets[:, channel] < 0.0) / (count * distance_count))
            for channel, name in enumerate(("event", "defect_interval", "terminal"))
        },
    }
    _write_json(folder / "summary.json", summary)
    return {**summary, "path": folder.as_posix()}


def _effective_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = _read_json(Path(args.config) if args.config else HERE / "config.json")
    if args.samples is not None:
        cfg["dataset"]["samples"] = int(args.samples)
    if args.workers is not None:
        cfg["dataset"]["workers"] = int(args.workers)
    if args.shard_size is not None:
        cfg["dataset"]["shard_size"] = int(args.shard_size)
    if int(cfg["dataset"]["samples"]) < 1:
        raise ValueError("samples must be positive")
    if int(cfg["dataset"]["workers"]) < 1:
        raise ValueError("workers must be positive")
    if int(cfg["dataset"]["shard_size"]) < 1:
        raise ValueError("shard_size must be positive")
    if int(cfg["model"]["input_channels"]) != 8:
        raise ValueError("Iter3 features require input_channels=8")
    cfg["dataset"]["schema"] = "fieldnn-shards-v1"
    cfg["dataset"]["label_coordinate"] = "observable Algorithm1 response-axis distance"
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = _effective_config(args)
    root = Path(args.output).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "config.json"
    if config_path.exists():
        current = _read_json(config_path)
        if _canonical_config(current) != _canonical_config(cfg):
            raise ValueError(f"output config differs from request: {config_path}")
    else:
        _write_json(config_path, cfg)

    frequency = _frequency_grid(cfg)
    distance = _distance_grid(cfg)
    np.save(root / "frequency.npy", frequency)
    np.save(root / "distance.npy", distance)
    plan = _sample_plan(int(cfg["dataset"]["samples"]), cfg)
    jobs = _make_shard_jobs(plan, int(cfg["dataset"]["shard_size"]))
    results = []
    pending = []
    for job in jobs:
        if args.resume and _shard_complete(root, job, cfg):
            results.append(_read_json(_shard_folder(root, job) / "summary.json"))
        else:
            pending.append(job)

    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=int(cfg["dataset"]["workers"])) as executor:
        futures = {
            executor.submit(_write_shard, str(root), cfg, job): job
            for job in pending
        }
        with tqdm(total=len(futures), desc="Generate Iter3 shards", unit="shard") as bar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                bar.update(1)
                bar.set_postfix(split=result["split"], shard=f"{result['shard_index']:05d}")

    results.sort(key=lambda item: (SPLITS.index(item["split"]), int(item["shard_index"])))
    split_counts = {split: 0 for split in SPLITS}
    manifest_shards = []
    for result in results:
        split = str(result["split"])
        split_counts[split] += int(result["count"])
        folder = _shard_folder(root, {
            "split": split,
            "shard_index": int(result["shard_index"]),
        })
        manifest_shards.append({
            "path": folder.relative_to(root).as_posix(),
            "split": split,
            "count": int(result["count"]),
        })
    _write_json(root / "manifest.json", {
        "schema": "fieldnn-dataset-1",
        "input_channels": 8,
        "target_channels": ["event", "defect_interval", "terminal"],
        "shards": manifest_shards,
        "counts": split_counts,
    })
    elapsed = time.perf_counter() - started
    print(
        f"Complete: {sum(split_counts.values())} samples "
        f"(train={split_counts['train']}, val={split_counts['val']}, "
        f"test={split_counts['test']}) in {elapsed:.1f}s; root={root}"
    )


if __name__ == "__main__":
    main()
