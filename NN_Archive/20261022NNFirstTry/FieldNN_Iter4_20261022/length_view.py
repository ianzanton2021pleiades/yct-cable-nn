"""Independent endpoint estimate and Algorithm 1 length-view response.

The endpoint detector is AST-extracted from the REF GUI file so Tkinter and the
GUI module are never imported. The NN input and NN probability grid are untouched.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
CORE16_FIELD = HERE.parents[2] / "DifferentAgent" / "CodeX" / "Core16_Field"
REF_GUI = PROJECT_ROOT / "REF" / "算法1：IFFT" / "yct-TDR-2-GUI-multisamples-EndSearchWithoutNN.pyw"

import sys

if str(CORE16_FIELD) not in sys.path:
    sys.path.insert(0, str(CORE16_FIELD))
from field_responses import algorithm1


def _load_ref_endpoint_functions() -> tuple[Any, Any, Any]:
    tree = ast.parse(REF_GUI.read_text(encoding="utf-8"), filename=str(REF_GUI))
    required = {"_local_maxima_indices", "_median_step_jump", "detect_endpoint_from_ifft"}
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in required]
    found = {node.name for node in selected}
    if found != required:
        raise ValueError(f"REF endpoint functions are incomplete: missing {sorted(required - found)}")
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    namespace: dict[str, Any] = {"np": np}
    exec(compile(module, str(REF_GUI), "exec"), namespace)
    return tuple(namespace[name] for name in (
        "_local_maxima_indices", "_median_step_jump", "detect_endpoint_from_ifft"
    ))


_local_maxima_indices, _median_step_jump, detect_endpoint_from_ifft = _load_ref_endpoint_functions()


def _step_evidence(distance: np.ndarray, step: np.ndarray,
                   candidate_m: float, dmax_m: float) -> dict[str, Any]:
    window_m = max(10.0, min(80.0, 0.018 * float(dmax_m)))
    pre = ((distance >= max(0.0, candidate_m - 2.0 * window_m))
           & (distance <= max(0.0, candidate_m - 0.75 * window_m)))
    post = ((distance >= candidate_m + 0.75 * window_m)
            & (distance <= min(float(dmax_m), candidate_m + 2.0 * window_m)))
    if int(pre.sum()) < 3 or int(post.sum()) < 3:
        signed_jump = None
    else:
        signed_jump = float(np.median(step[post]) - np.median(step[pre]))
    absolute_jump = float(_median_step_jump(distance, step, candidate_m, dmax_m))
    return {
        "window_m": float(window_m),
        "signed_jump": signed_jump,
        "absolute_jump": absolute_jump,
        "positive_step_jump": bool(signed_jump is not None and signed_jump > 0.0 and absolute_jump > 0.0),
    }


def _pulse_evidence(distance: np.ndarray, impulse: np.ndarray,
                    candidate_m: float) -> dict[str, Any]:
    local = np.abs(distance - candidate_m) <= 5.0
    background = (np.abs(distance - candidate_m) >= 15.0) & (np.abs(distance - candidate_m) <= 40.0)
    peak = float(np.max(np.abs(impulse[local]))) if local.any() else 0.0
    background_rms = float(np.sqrt(np.mean(impulse[background] ** 2))) if background.any() else 0.0
    ratio = peak / max(background_rms, 1e-12)
    return {
        "local_peak": peak,
        "background_rms": background_rms,
        "peak_to_background_rms": float(ratio),
        "local_pulse": bool(ratio >= 2.0),
    }


def _run_algorithm1(frequency_hz: np.ndarray, s11: np.ndarray,
                    velocity_m_s: float, distance_max_m: float,
                    fft_step_hz: float | None = None) -> dict[str, Any]:
    return algorithm1(
        frequency_hz,
        s11,
        velocity_m_s=float(velocity_m_s),
        distance_max_m=float(distance_max_m),
        fft_step_hz=fft_step_hz,
    )


def analyze_length_view(
    frequency_hz: np.ndarray,
    s11: np.ndarray,
    cfg: dict[str, Any],
    source_velocity_m_s: float,
    display_source_grid_max_m: float,
) -> dict[str, Any]:
    """Estimate an endpoint without labels, then recompute Algorithm 1 to 1.2L.

    A trusted result is only a stable heuristic estimate. It is not a measured
    cable length. The returned `_plot_responses` arrays are for the caller's
    figure only and must not be used as NN inputs or serialized into JSONL.
    """
    velocity = float(source_velocity_m_s)
    full_max = float(display_source_grid_max_m)
    if velocity <= 0 or full_max <= 0:
        raise ValueError("source velocity and source display range must be positive")

    # Match the checkpoint's measured band without changing its NN input path.
    frequency = np.asarray(frequency_hz, dtype=float)
    reflection = np.asarray(s11, dtype=complex)
    band = cfg["frequency"]
    start_hz, stop_hz = float(band["start_hz"]), float(band["stop_hz"])
    if frequency[0] > start_hz or frequency[-1] < stop_hz:
        raise ValueError("Measured S11 does not cover the configured frequency band")
    keep = (frequency >= start_hz) & (frequency <= stop_hz)
    frequency = np.unique(np.r_[start_hz, frequency[keep], stop_hz])
    reflection = np.interp(frequency, frequency_hz, np.asarray(s11).real) + 1j * np.interp(
        frequency, frequency_hz, np.asarray(s11).imag
    )

    full = _run_algorithm1(frequency, reflection, velocity, full_max)
    d_full = np.asarray(full["distance_m"], dtype=float)
    impulse_full = np.asarray(full["impulse_raw"], dtype=float)
    step_full = np.asarray(full["step_rho"], dtype=float)
    full_fft_step_hz = float(full["fft_step_hz"])
    candidate_full = detect_endpoint_from_ifft(d_full, impulse_full, step_full)

    response_recomputed: dict[str, Any] = {
        "performed": False,
        "algorithm": "Core16_Field.field_responses.algorithm1",
        "velocity_used_m_per_us": velocity / 1e6,
        "frequency_point_count": int(len(frequency)),
        "frequency_band_hz": [float(frequency[0]), float(frequency[-1])],
        "full_grid_fft_step_hz": full_fft_step_hz,
        "recomputed_fft_step_hz": None,
        "fft_step_reused_from_full_grid": False,
        "distance_max_m": None,
        "point_count": 0,
        "endpoint_candidate_m": None,
    }
    recomputed = None
    candidate_recomputed = None
    pulse_full = step_full_evidence = None
    pulse_recomputed = step_recomputed_evidence = None

    if candidate_full is not None:
        candidate_full = float(candidate_full)
        recalc_max = 1.2 * candidate_full
        response_recomputed["performed"] = True
        response_recomputed["distance_max_m"] = float(recalc_max)
        # Hold frequency resampling and DC extrapolation fixed across ranges;
        # only Algorithm 1's displayed distance extent changes.
        recomputed = _run_algorithm1(frequency, reflection, velocity, recalc_max, full_fft_step_hz)
        response_recomputed["recomputed_fft_step_hz"] = float(recomputed["fft_step_hz"])
        response_recomputed["fft_step_reused_from_full_grid"] = (
            float(recomputed["fft_step_hz"]) == full_fft_step_hz
        )
        response_recomputed["point_count"] = int(len(recomputed["distance_m"]))
        d_recomputed = np.asarray(recomputed["distance_m"], dtype=float)
        impulse_recomputed = np.asarray(recomputed["impulse_raw"], dtype=float)
        step_recomputed = np.asarray(recomputed["step_rho"], dtype=float)
        candidate_recomputed = detect_endpoint_from_ifft(
            d_recomputed, impulse_recomputed, step_recomputed
        )
        if candidate_recomputed is not None:
            candidate_recomputed = float(candidate_recomputed)
            response_recomputed["endpoint_candidate_m"] = candidate_recomputed
            pulse_recomputed = _pulse_evidence(d_recomputed, impulse_recomputed, candidate_recomputed)
            step_recomputed_evidence = _step_evidence(
                d_recomputed, step_recomputed, candidate_recomputed, recalc_max
            )
        pulse_full = _pulse_evidence(d_full, impulse_full, candidate_full)
        step_full_evidence = _step_evidence(d_full, step_full, candidate_full, full_max)
    else:
        response_recomputed["reason"] = "no_endpoint_candidate_on_full_source_grid"

    tolerance = max(10.0, 0.05 * candidate_full) if candidate_full is not None else None
    near_limit = max(10.0, 0.02 * full_max)
    full_right_margin = max(10.0, 0.05 * full_max)
    if candidate_recomputed is not None and candidate_full is not None and recomputed is not None:
        recalc_max = float(response_recomputed["distance_max_m"])
        recalc_right_margin = max(10.0, 0.05 * recalc_max)
        stable = abs(candidate_recomputed - candidate_full) <= float(tolerance)
        far_from_near = candidate_full > near_limit and candidate_recomputed > max(10.0, 0.02 * recalc_max)
        far_from_right = (candidate_full < full_max - full_right_margin
                          and candidate_recomputed < recalc_max - recalc_right_margin)
        local_pulse = bool(pulse_full["local_pulse"] and pulse_recomputed["local_pulse"])
        positive_step = bool(
            step_full_evidence["positive_step_jump"]
            and step_recomputed_evidence["positive_step_jump"]
        )
    else:
        stable = far_from_near = far_from_right = local_pulse = positive_step = False

    checks = {
        "full_grid_candidate_found": candidate_full is not None,
        "recomputed_candidate_found": candidate_recomputed is not None,
        "stable_within_tolerance": bool(stable),
        "far_from_near_end": bool(far_from_near),
        "far_from_right_boundary": bool(far_from_right),
        "local_pulse_in_both_runs": bool(local_pulse),
        "positive_step_jump_in_both_runs": bool(positive_step),
    }
    trusted = all(checks.values())
    estimate_m = ((candidate_full + candidate_recomputed) / 2.0
                  if trusted and candidate_full is not None and candidate_recomputed is not None else None)
    plot_window_m = float(1.2 * estimate_m) if trusted and estimate_m is not None else full_max

    if recomputed is not None:
        plot_distance = np.asarray(recomputed["distance_m"], dtype=float)
        plot_impulse = np.asarray(recomputed["impulse_raw"], dtype=float)
        plot_step = np.asarray(recomputed["step_rho"], dtype=float)
    else:
        plot_distance, plot_impulse, plot_step = d_full, impulse_full, step_full

    return {
        "length_estimate": {
            "estimated_length_m": estimate_m,
            "trusted_heuristic": bool(trusted),
            "status": "trusted_heuristic" if trusted else ("candidate_uncertain" if candidate_full is not None else "no_candidate"),
            "full_grid_candidate_m": candidate_full,
            "recomputed_candidate_m": candidate_recomputed,
            "stability_tolerance_m": tolerance,
            "note": "Heuristic estimate only; not an independently known or measured cable length.",
        },
        "response_recomputed": response_recomputed,
        "plot_window_m": plot_window_m,
        "heuristic_evidence": {
            "checks": checks,
            "near_limit_m": near_limit,
            "full_grid_right_margin_m": full_right_margin,
            "full_grid": {"pulse": pulse_full, "step": step_full_evidence},
            "recomputed": {"pulse": pulse_recomputed, "step": step_recomputed_evidence},
        },
        "_plot_responses": {
            "full_grid": {"distance_m": d_full, "impulse_real": impulse_full, "step_response": step_full},
            "recomputed": {"distance_m": plot_distance, "impulse_real": plot_impulse, "step_response": plot_step},
        },
    }
