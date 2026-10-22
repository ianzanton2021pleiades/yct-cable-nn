from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares


HERE = Path(__file__).resolve().parent
SIGNAL_DIR = HERE / "output" / "signals"
OUTPUT_DIR = HERE / "output"
ASSET_DIR = HERE / "assets"
MPLCONFIGDIR = HERE / "cache" / "MPLCONFIGDIR"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPLCONFIGDIR)
sys.path.insert(0, str(HERE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from field_cases import build_cases
from field_physics import simulate
from field_responses import algorithm1, algorithm2


LENGTH_M = 2430.0
DISTANCE_MAX_M = 3159.0
VELOCITY_M_S = 172.0e6
FREQUENCY_START_HZ = 9.0e3
FREQUENCY_STOP_HZ = 20.0e6
FREQUENCY_POINTS = 4096
REFERENCE_HZ = 1.0e6
END_WINDOW_M = (0.9 * LENGTH_M, 1.1 * LENGTH_M)
R_SKIN_BOUNDS = (0.003, 0.2)
TAN_DELTA_BOUNDS = (1.0e-5, 0.1)
BODY_VELOCITY_BOUNDS = (1.4e8, 2.2e8)
FREQUENCY_RESIDUAL_WEIGHT = 0.05
MAX_NFEV = 40
STARTS = (
    {
        "name": "current_prior",
        "r_skin_ref_ohm_per_m": 0.03,
        "tan_delta": 0.001,
        "body_velocity_m_s": 172.0e6,
    },
    {
        "name": "loss_and_velocity_alternative",
        "r_skin_ref_ohm_per_m": 0.05,
        "tan_delta": 0.03,
        "body_velocity_m_s": 190.0e6,
    },
)


def interpolate_complex(source_f: np.ndarray, source_s11: np.ndarray,
                        target_f: np.ndarray) -> np.ndarray:
    return (
        np.interp(target_f, source_f, source_s11.real)
        + 1.0j * np.interp(target_f, source_f, source_s11.imag)
    )


def load_measured_case() -> dict[str, Any]:
    matches = []
    for path in sorted(SIGNAL_DIR.glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            relative = str(archive["source_relative_path"].item())
            sweep_name = str(archive["sweep_name"].item())
            if "2026-9-14" in relative and sweep_name.startswith("A"):
                matches.append((path, relative, sweep_name))
    if len(matches) != 1:
        raise ValueError(f"期望唯一的2026-9-14 A相NPZ，实际匹配{len(matches)}条")

    path, relative, sweep_name = matches[0]
    with np.load(path, allow_pickle=False) as archive:
        return {
            "npz_path": path,
            "source_relative_path": relative,
            "sweep_name": sweep_name,
            "frequency_hz": archive["frequency_hz"].copy(),
            "s11_complex": archive["s11_complex"].copy(),
        }


def forward(config: dict, frequency_hz: np.ndarray,
            r_skin_ref_ohm_per_m: float,
            tan_delta: float,
            body_velocity_m_s: float) -> tuple[np.ndarray, dict[str, Any]]:
    candidate = copy.deepcopy(config)
    candidate["body"]["r_skin_ref_ohm_per_m"] = float(r_skin_ref_ohm_per_m)
    candidate["body"]["tan_delta"] = float(tan_delta)
    candidate["body"]["velocity_m_s"] = float(body_velocity_m_s)
    case = next(
        item for item in build_cases(candidate, LENGTH_M) if item["id"] == "C00"
    )
    s11 = simulate(
        frequency_hz,
        case["segments"],
        case["joints"],
        fixture=candidate["fixture"],
        route="dg_v3_rlgc",
        reference_hz=REFERENCE_HZ,
        load=case["termination"],
        zref=50.0,
    )
    response = algorithm1(
        frequency_hz,
        s11,
        velocity_m_s=VELOCITY_M_S,
        distance_max_m=DISTANCE_MAX_M,
    )
    return s11, response


def end_window_mask(distance_m: np.ndarray) -> np.ndarray:
    return (distance_m >= END_WINDOW_M[0]) & (distance_m <= END_WINDOW_M[1])


def residual_function(config: dict, frequency_hz: np.ndarray,
                      measured_s11: np.ndarray, measured_a1: dict,
                      window_mask: np.ndarray, measured_window_norm: float,
                      measured_s11_norm: float):
    measured_impulse_window = measured_a1["impulse_raw"][window_mask]

    def residual(log_parameters: np.ndarray) -> np.ndarray:
        r_skin = float(10.0 ** log_parameters[0])
        tan_delta = float(10.0 ** log_parameters[1])
        body_velocity = float(10.0 ** log_parameters[2])
        model_s11, model_a1 = forward(
            config, frequency_hz, r_skin, tan_delta, body_velocity
        )
        window_difference = (
            model_a1["impulse_raw"][window_mask] - measured_impulse_window
        ) / measured_window_norm
        frequency_difference = model_s11 - measured_s11
        frequency_residual = np.sqrt(FREQUENCY_RESIDUAL_WEIGHT) * np.concatenate(
            (frequency_difference.real, frequency_difference.imag)
        ) / measured_s11_norm
        return np.concatenate((window_difference, frequency_residual))

    return residual


def peak_in_window(distance_m: np.ndarray, impulse: np.ndarray) -> dict[str, float]:
    mask = end_window_mask(distance_m) & np.isfinite(impulse)
    indices = np.flatnonzero(mask)
    index = int(indices[np.argmax(np.abs(impulse[indices]))])
    amplitude = float(impulse[index])
    return {
        "peak_position_m": float(distance_m[index]),
        "peak_amplitude_signed": amplitude,
        "peak_amplitude_abs": abs(amplitude),
    }


def window_metrics(measured_distance: np.ndarray, measured_impulse: np.ndarray,
                   model_distance: np.ndarray, model_impulse: np.ndarray) -> dict[str, Any]:
    measured_mask = end_window_mask(measured_distance)
    model_mask = end_window_mask(model_distance)
    measured_window = measured_impulse[measured_mask]
    model_window = model_impulse[model_mask]
    difference = model_window - measured_window
    measured_norm = float(np.linalg.norm(measured_window))
    measured_peak = peak_in_window(measured_distance, measured_impulse)
    model_peak = peak_in_window(model_distance, model_impulse)
    return {
        "window_m": [float(END_WINDOW_M[0]), float(END_WINDOW_M[1])],
        "sample_count": int(measured_window.size),
        "field_window_impulse_l2_norm": measured_norm,
        "window_rmse": float(np.sqrt(np.mean(difference**2))),
        "window_relative_l2_error": float(np.linalg.norm(difference) / measured_norm),
        "measured_peak": measured_peak,
        "model_peak": model_peak,
        "peak_position_difference_m": (
            model_peak["peak_position_m"] - measured_peak["peak_position_m"]
        ),
        "peak_amplitude_ratio_model_to_measured": (
            model_peak["peak_amplitude_abs"] / measured_peak["peak_amplitude_abs"]
        ),
    }


def full_metrics(frequency_hz: np.ndarray, measured_s11: np.ndarray,
                 model_s11: np.ndarray, measured_a1: dict,
                 model_a1: dict, measured_a2: dict | None,
                 model_a2: dict | None) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "algorithm1_raw_impulse_end_window": window_metrics(
            measured_a1["distance_m"], measured_a1["impulse_raw"],
            model_a1["distance_m"], model_a1["impulse_raw"],
        ),
        "frequency_complex_rmse": float(
            np.sqrt(np.mean(np.abs(model_s11 - measured_s11) ** 2))
        ),
        "frequency_complex_relative_l2_error": float(
            np.linalg.norm(model_s11 - measured_s11) / np.linalg.norm(measured_s11)
        ),
        "frequency_median_abs_error": float(np.median(np.abs(model_s11 - measured_s11))),
    }
    if measured_a2 is not None and model_a2 is not None:
        metrics["algorithm2_smoothed_impulse_end_window"] = window_metrics(
            measured_a2["distance_impulse_m"], measured_a2["impulse_smoothed"],
            model_a2["distance_impulse_m"], model_a2["impulse_smoothed"],
        )
    return metrics


def bound_status(value: float, lower: float, upper: float) -> dict[str, bool | float]:
    lower_fraction = (value - lower) / (upper - lower)
    upper_fraction = (upper - value) / (upper - lower)
    return {
        "at_lower_bound_within_1pct_range": bool(lower_fraction <= 0.01),
        "at_upper_bound_within_1pct_range": bool(upper_fraction <= 0.01),
        "fraction_from_lower": float(lower_fraction),
        "fraction_from_upper": float(upper_fraction),
    }


def assess_two_start_uniqueness(fits: list[dict[str, Any]]) -> dict[str, Any]:
    first, second = fits
    log_separation = max(
        abs(np.log10(first["r_skin_ref_ohm_per_m"] / second["r_skin_ref_ohm_per_m"])),
        abs(np.log10(first["tan_delta"] / second["tan_delta"])),
        abs(np.log10(first["body_velocity_m_s"] / second["body_velocity_m_s"])),
    )
    min_cost = min(first["cost"], second["cost"])
    cost_gap = abs(first["cost"] - second["cost"]) / max(min_cost, 1.0e-15)
    same_cost = cost_gap <= 0.01
    materially_different = log_separation >= 0.1
    if same_cost and materially_different:
        assessment = "two_starts_have_similar_cost_but_different_parameters; potential_non_uniqueness"
    elif same_cost and not materially_different:
        assessment = "two_starts_converged_to_similar_parameters; local_uniqueness_not_proven"
    else:
        assessment = "start_costs_differ; two_starts_do_not_establish_uniqueness"
    return {
        "assessment": assessment,
        "max_abs_log10_parameter_separation": float(log_separation),
        "relative_cost_gap": float(cost_gap),
        "is_full_identifiability_analysis": False,
    }


def configure_plot() -> None:
    plt.rcParams.update({
        "font.family": ["Times New Roman", "SimHei"],
        "font.serif": ["Times New Roman", "SimHei"],
        "font.sans-serif": ["SimHei", "Microsoft YaHei"],
        "axes.unicode_minus": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.spines.top": True,
        "axes.spines.right": True,
        "figure.dpi": 200,
        "savefig.dpi": 200,
    })


def make_plot(frequency_hz: np.ndarray, measured_s11: np.ndarray,
              baseline_s11: np.ndarray, best_s11: np.ndarray,
              measured_a1: dict, baseline_a1: dict, best_a1: dict,
              measured_a2: dict, baseline_a2: dict, best_a2: dict) -> None:
    configure_plot()
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    low, high = END_WINDOW_M
    colors = {"field": "#111827", "prior": "#d95f02", "fit": "#1f77b4"}

    for label, response, color in (
        ("实测 A1", measured_a1, colors["field"]),
        ("先验模型", baseline_a1, colors["prior"]),
        ("拟合模型", best_a1, colors["fit"]),
    ):
        distance = response["distance_m"]
        mask = (distance >= low) & (distance <= high)
        axes[0, 0].plot(distance[mask], response["impulse_raw"][mask],
                        color=color, linewidth=1.0, label=label)
    axes[0, 0].set_title("Algorithm 1 原始脉冲：固定末端窗")
    axes[0, 0].set_xlabel("距离 (m)")
    axes[0, 0].set_ylabel("impulse_raw")
    axes[0, 0].legend(fontsize=8)

    for label, response, color in (
        ("实测 A2", measured_a2, colors["field"]),
        ("先验模型", baseline_a2, colors["prior"]),
        ("拟合模型", best_a2, colors["fit"]),
    ):
        distance = response["distance_impulse_m"]
        mask = (distance >= low) & (distance <= high)
        axes[0, 1].plot(distance[mask], response["impulse_smoothed"][mask],
                        color=color, linewidth=1.0, label=label)
    axes[0, 1].set_title("Algorithm 2 脉冲：3 m平滑")
    axes[0, 1].set_xlabel("距离 (m)")
    axes[0, 1].set_ylabel("impulse_smoothed")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].semilogx(frequency_hz, np.abs(measured_s11), color=colors["field"],
                        linewidth=1.0, label="实测 A相")
    axes[1, 0].semilogx(frequency_hz, np.abs(baseline_s11), color=colors["prior"],
                        linewidth=1.0, label="先验模型")
    axes[1, 0].semilogx(frequency_hz, np.abs(best_s11), color=colors["fit"],
                        linewidth=1.0, label="拟合模型")
    axes[1, 0].set_title("|S11|：拟合前后与实测")
    axes[1, 0].set_xlabel("频率 (Hz)")
    axes[1, 0].set_ylabel("|Γ|")
    axes[1, 0].legend(fontsize=8)

    distance = measured_a1["distance_m"]
    mask = (distance >= low) & (distance <= high)
    baseline_error = baseline_a1["impulse_raw"][mask] - measured_a1["impulse_raw"][mask]
    best_error = best_a1["impulse_raw"][mask] - measured_a1["impulse_raw"][mask]
    axes[1, 1].plot(distance[mask], baseline_error, color=colors["prior"],
                    linewidth=1.0, label="先验−实测")
    axes[1, 1].plot(distance[mask], best_error, color=colors["fit"],
                    linewidth=1.0, label="拟合−实测")
    axes[1, 1].axhline(0.0, color="black", linewidth=0.7)
    axes[1, 1].set_title("主目标：末端窗完整脉冲向量差")
    axes[1, 1].set_xlabel("距离 (m)")
    axes[1, 1].set_ylabel("A1 impulse residual")
    axes[1, 1].legend(fontsize=8)

    for axis in axes.flat:
        axis.grid(True, which="both", alpha=0.22)
    figure.suptitle(
        "2430 m A相同源anchor的端到端损耗与有效材料速度拟合\n"
        "材料速度只影响RLGC构造；距离轴固定172 Mm/s；不可解释为导体损耗或独立验证",
        fontsize=12,
    )
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(ASSET_DIR / "body_loss_fit.png", dpi=200)
    plt.close(figure)


def previous_two_parameter_reference(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    previous = json.loads(path.read_text(encoding="utf-8"))
    if "two_parameter_reference" in previous:
        return previous["two_parameter_reference"]
    return {
        key: previous[key]
        for key in (
            "comparison",
            "objective",
            "model_snapshot",
            "fits",
            "best_fit_index",
            "two_start_parameter_non_uniqueness",
            "before_after_metrics",
        )
    }


def main() -> None:
    config = json.loads((HERE / "field_config.json").read_text(encoding="utf-8"))
    output_json = OUTPUT_DIR / "body_loss_fit.json"
    two_parameter_reference = previous_two_parameter_reference(output_json)
    measured_source = load_measured_case()
    frequency_hz = np.linspace(
        FREQUENCY_START_HZ, FREQUENCY_STOP_HZ, FREQUENCY_POINTS, dtype=np.float64
    )
    measured_s11 = interpolate_complex(
        measured_source["frequency_hz"], measured_source["s11_complex"], frequency_hz
    )
    measured_a1 = algorithm1(
        frequency_hz,
        measured_s11,
        velocity_m_s=VELOCITY_M_S,
        distance_max_m=DISTANCE_MAX_M,
    )
    window_mask = end_window_mask(measured_a1["distance_m"])
    measured_window_norm = float(np.linalg.norm(measured_a1["impulse_raw"][window_mask]))
    measured_s11_norm = float(np.linalg.norm(measured_s11))

    prior_r_skin = float(config["body"]["r_skin_ref_ohm_per_m"])
    prior_tan_delta = float(config["body"]["tan_delta"])
    prior_body_velocity = float(config["body"]["velocity_m_s"])
    baseline_s11, baseline_a1 = forward(
        config, frequency_hz, prior_r_skin, prior_tan_delta, prior_body_velocity
    )
    measured_a2 = algorithm2(
        frequency_hz,
        measured_s11,
        velocity_m_s=VELOCITY_M_S,
        distance_max_m=DISTANCE_MAX_M,
        distance_step_m=1.0,
        step_smoothing_m=1.0,
        impulse_smoothing_m=3.0,
    )
    baseline_a2 = algorithm2(
        frequency_hz,
        baseline_s11,
        velocity_m_s=VELOCITY_M_S,
        distance_max_m=DISTANCE_MAX_M,
        distance_step_m=1.0,
        step_smoothing_m=1.0,
        impulse_smoothing_m=3.0,
    )

    residual = residual_function(
        config, frequency_hz, measured_s11, measured_a1, window_mask,
        measured_window_norm, measured_s11_norm,
    )
    lower = np.log10([
        R_SKIN_BOUNDS[0], TAN_DELTA_BOUNDS[0], BODY_VELOCITY_BOUNDS[0]
    ])
    upper = np.log10([
        R_SKIN_BOUNDS[1], TAN_DELTA_BOUNDS[1], BODY_VELOCITY_BOUNDS[1]
    ])
    fits = []
    for start in STARTS:
        initial = np.log10([
            start["r_skin_ref_ohm_per_m"], start["tan_delta"],
            start["body_velocity_m_s"],
        ])
        result = least_squares(
            residual,
            initial,
            bounds=(lower, upper),
            max_nfev=MAX_NFEV,
            x_scale="jac",
            method="trf",
        )
        r_skin = float(10.0 ** result.x[0])
        tan_delta = float(10.0 ** result.x[1])
        body_velocity = float(10.0 ** result.x[2])
        fits.append({
            "start_name": start["name"],
            "start_parameters": {
                "r_skin_ref_ohm_per_m": start["r_skin_ref_ohm_per_m"],
                "tan_delta": start["tan_delta"],
                "body_velocity_m_s": start["body_velocity_m_s"],
            },
            "r_skin_ref_ohm_per_m": r_skin,
            "tan_delta": tan_delta,
            "body_velocity_m_s": body_velocity,
            "cost": float(result.cost),
            "nfev": int(result.nfev),
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "optimality": float(result.optimality),
            "bound_status": {
                "r_skin_ref_ohm_per_m": bound_status(r_skin, *R_SKIN_BOUNDS),
                "tan_delta": bound_status(tan_delta, *TAN_DELTA_BOUNDS),
                "body_velocity_m_s": bound_status(body_velocity, *BODY_VELOCITY_BOUNDS),
            },
        })
        print(f"{start['name']}: cost={result.cost:.8g}, nfev={result.nfev}, "
              f"Rskin={r_skin:.8g}, tan_delta={tan_delta:.8g}, "
              f"body_velocity={body_velocity:.8g}")

    best_index = int(np.argmin([fit["cost"] for fit in fits]))
    best_fit = fits[best_index]
    best_s11, best_a1 = forward(
        config,
        frequency_hz,
        best_fit["r_skin_ref_ohm_per_m"],
        best_fit["tan_delta"],
        best_fit["body_velocity_m_s"],
    )
    best_a2 = algorithm2(
        frequency_hz,
        best_s11,
        velocity_m_s=VELOCITY_M_S,
        distance_max_m=DISTANCE_MAX_M,
        distance_step_m=1.0,
        step_smoothing_m=1.0,
        impulse_smoothing_m=3.0,
    )

    baseline_metrics = full_metrics(
        frequency_hz, measured_s11, baseline_s11,
        measured_a1, baseline_a1, measured_a2, baseline_a2,
    )
    best_metrics = full_metrics(
        frequency_hz, measured_s11, best_s11,
        measured_a1, best_a1, measured_a2, best_a2,
    )
    params_close = assess_two_start_uniqueness(fits)
    model_snapshot = {
        "route": "dg_v3_rlgc",
        "reference_hz": REFERENCE_HZ,
        "length_m": LENGTH_M,
        "body_material_velocity_m_s_prior": prior_body_velocity,
        "body_material_velocity_m_s_fitted": best_fit["body_velocity_m_s"],
        "response_distance_velocity_m_s_fixed": VELOCITY_M_S,
        "fixed_body_fields": {
            key: value for key, value in config["body"].items()
            if key not in {"r_skin_ref_ohm_per_m", "tan_delta", "velocity_m_s"}
        },
        "fixed_joint": config["joint"],
        "fixed_joint_positions_fraction": config["joint_positions_fraction"],
        "fixed_fixture": config["fixture"],
        "termination": "open",
    }
    payload = {
        "comparison": {
            "measured_npz": str(measured_source["npz_path"].relative_to(HERE)),
            "measured_source_relative_path": measured_source["source_relative_path"],
            "sweep_name": measured_source["sweep_name"],
            "frequency_grid_hz": {
                "start": FREQUENCY_START_HZ,
                "stop": FREQUENCY_STOP_HZ,
                "points": FREQUENCY_POINTS,
                "interpolation": "linear interpolation of raw complex S11 real and imaginary parts",
            },
            "distance_max_m": DISTANCE_MAX_M,
            "end_window_m": [float(END_WINDOW_M[0]), float(END_WINDOW_M[1])],
            "response_distance_velocity_m_s_fixed": VELOCITY_M_S,
            "body_material_velocity_m_s_prior": prior_body_velocity,
            "length_source": "2430 m is the FDR annotation supplied for this anchor, not an independent geometric measurement.",
        },
        "objective": {
            "main": "full A1 raw impulse vector residual in the fixed end window",
            "window_residual": "(model - measured) / L2_norm(measured impulse in window)",
            "weak_frequency_term": "sqrt(0.05) * stacked real/imag complex S11 difference / L2_norm(measured S11)",
            "frequency_residual_weight": FREQUENCY_RESIDUAL_WEIGHT,
            "peak_position_used_as_objective": False,
            "optimizer": "scipy.optimize.least_squares, bounded trust-region reflective, log10 parameters",
            "starts": list(STARTS),
            "max_nfev_per_start": MAX_NFEV,
            "parameter_bounds": {
                "r_skin_ref_ohm_per_m_at_1mhz": list(R_SKIN_BOUNDS),
                "tan_delta": list(TAN_DELTA_BOUNDS),
                "body_material_velocity_m_s": list(BODY_VELOCITY_BOUNDS),
            },
            "body_material_velocity_changes_only_rlgc_construction": True,
            "response_distance_velocity_remains_fixed_m_s": VELOCITY_M_S,
        },
        "model_snapshot": model_snapshot,
        "fit_parameter_count": 3,
        "best_fit_parameters": {
            "r_skin_ref_ohm_per_m_at_1mhz": best_fit["r_skin_ref_ohm_per_m"],
            "tan_delta": best_fit["tan_delta"],
            "body_material_velocity_m_s": best_fit["body_velocity_m_s"],
            "response_distance_velocity_m_s": VELOCITY_M_S,
        },
        "fits": fits,
        "best_fit_index": best_index,
        "two_start_parameter_non_uniqueness": params_close,
        "before_after_metrics": {
            "current_prior": baseline_metrics,
            "three_parameter_best_fit": best_metrics,
        },
        "two_parameter_reference": two_parameter_reference,
        "interpretation_limits": [
            "This is an effective end-to-end fit for a known open termination; the objective includes unmodeled connector and material differences.",
            "The same 2430m A-phase anchor supplied both the fixture fit and this loss fit; this is not independent validation.",
            "Fitted r_skin and tan_delta are effective model parameters, not a conductor-loss measurement.",
            "The fitted body velocity changes only the RLGC construction; the response distance conversion remains fixed at 172e6 m/s.",
            "2430m is an FDR annotation, so effective body velocity is not an independent geometric or true propagation-velocity inversion.",
            "A-phase is not known healthy. C00 is only the open-termination comparison topology.",
            "End-window peak is the absolute maximum inside a fixed window, not automatic terminal detection.",
            "A two-start comparison is a limited non-uniqueness indicator, not full parameter identifiability.",
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    make_plot(
        frequency_hz, measured_s11, baseline_s11, best_s11,
        measured_a1, baseline_a1, best_a1,
        measured_a2, baseline_a2, best_a2,
    )
    print(f"A1 end-window RMSE: {baseline_metrics['algorithm1_raw_impulse_end_window']['window_rmse']:.8g} -> {best_metrics['algorithm1_raw_impulse_end_window']['window_rmse']:.8g}")
    print(f"Complex S11 RMSE: {baseline_metrics['frequency_complex_rmse']:.8g} -> {best_metrics['frequency_complex_rmse']:.8g}")
    print(f"JSON: {OUTPUT_DIR / 'body_loss_fit.json'}")
    print(f"PNG: {ASSET_DIR / 'body_loss_fit.png'}")


if __name__ == "__main__":
    main()
