from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


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
FREQUENCY_START_HZ = 9.0e3
FREQUENCY_STOP_HZ = 20.0e6
FREQUENCY_POINTS = 4096
VELOCITY_M_S = 172.0e6
END_WINDOW_FRACTION = (0.9, 1.1)
NEAR_WINDOW_FRACTION = (0.0, 0.05)
BANDS_HZ = (
    (9.0e3, 100.0e3, "9k_100k"),
    (100.0e3, 1.0e6, "100k_1M"),
    (1.0e6, 10.0e6, "1M_10M"),
    (10.0e6, 20.0e6, "10M_20M"),
)


def interp_complex(source_f: np.ndarray, source_s11: np.ndarray,
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
            if "2026-9-14" not in relative or not sweep_name.startswith("A"):
                continue
            matches.append({
                "npz_path": path,
                "source_relative_path": relative,
                "sweep_name": sweep_name,
            })
    if len(matches) != 1:
        raise ValueError(f"期望唯一的2026-9-14 A相NPZ，实际匹配{len(matches)}条")

    match = matches[0]
    with np.load(match["npz_path"], allow_pickle=False) as archive:
        match["frequency_hz"] = archive["frequency_hz"].copy()
        match["s11_complex"] = archive["s11_complex"].copy()
    return match


def measure_end_peak(distance_m: np.ndarray, values: np.ndarray) -> dict[str, float | None]:
    low_m, high_m = (END_WINDOW_FRACTION[0] * LENGTH_M,
                     END_WINDOW_FRACTION[1] * LENGTH_M)
    mask = (distance_m >= low_m) & (distance_m <= high_m) & np.isfinite(values)
    if not np.any(mask):
        return {"window_low_m": low_m, "window_high_m": high_m,
                "peak_position_m": None, "peak_amplitude_signed": None,
                "peak_amplitude_abs": None}
    indices = np.flatnonzero(mask)
    index = int(indices[np.argmax(np.abs(values[indices]))])
    amplitude = float(values[index])
    return {
        "window_low_m": low_m,
        "window_high_m": high_m,
        "peak_position_m": float(distance_m[index]),
        "peak_amplitude_signed": amplitude,
        "peak_amplitude_abs": abs(amplitude),
    }


def energy_ratio(distance_m: np.ndarray, impulse: np.ndarray) -> float:
    total_mask = (distance_m >= 0.0) & (distance_m <= DISTANCE_MAX_M)
    near_high_m = NEAR_WINDOW_FRACTION[1] * LENGTH_M
    near_mask = (distance_m >= 0.0) & (distance_m <= near_high_m)
    x_total, y_total = distance_m[total_mask], impulse[total_mask]
    x_near, y_near = distance_m[near_mask], impulse[near_mask]

    def integral(x: np.ndarray, y: np.ndarray) -> float:
        return float(np.sum(0.5 * (y[1:] ** 2 + y[:-1] ** 2) * np.diff(x)))

    return integral(x_near, y_near) / integral(x_total, y_total)


def band_medians(frequency_hz: np.ndarray,
                 s11_complex: np.ndarray) -> list[dict[str, float | int | str]]:
    records = []
    for lower_hz, upper_hz, label in BANDS_HZ:
        mask = (frequency_hz >= lower_hz) & (frequency_hz < upper_hz)
        if upper_hz == FREQUENCY_STOP_HZ:
            mask |= frequency_hz == upper_hz
        records.append({
            "band": label,
            "lower_hz": lower_hz,
            "upper_hz": upper_hz,
            "point_count": int(np.count_nonzero(mask)),
            "median_abs_gamma": float(np.median(np.abs(s11_complex[mask]))),
        })
    return records


def make_responses(frequency_hz: np.ndarray, s11_complex: np.ndarray) -> dict[str, Any]:
    response_a1 = algorithm1(
        frequency_hz,
        s11_complex,
        velocity_m_s=VELOCITY_M_S,
        distance_max_m=DISTANCE_MAX_M,
    )
    response_a2 = algorithm2(
        frequency_hz,
        s11_complex,
        velocity_m_s=VELOCITY_M_S,
        distance_max_m=DISTANCE_MAX_M,
        distance_step_m=1.0,
        step_smoothing_m=1.0,
        impulse_smoothing_m=3.0,
    )
    return {"a1": response_a1, "a2": response_a2}


def response_metrics(responses: dict[str, Any]) -> dict[str, Any]:
    a1, a2 = responses["a1"], responses["a2"]
    return {
        "algorithm1": {
            "impulse_source": "impulse_raw (native IFFT output)",
            "step_source": "step_rho (native cumulative response)",
            "end_window_peak": measure_end_peak(a1["distance_m"], a1["impulse_raw"]),
            "near_0_5pct_length_impulse_energy_ratio": energy_ratio(
                a1["distance_m"], a1["impulse_raw"]
            ),
            "distance_sample_step_m": float(np.median(np.diff(a1["distance_m"]))),
            "fft_step_hz": float(a1["fft_step_hz"]),
            "period_distance_m": float(a1["period_distance_m"]),
        },
        "algorithm2": {
            "impulse_source": "impulse_smoothed (3 m physical window)",
            "step_source": "step_smoothed (1 m physical window)",
            "end_window_peak": measure_end_peak(
                a2["distance_impulse_m"], a2["impulse_smoothed"]
            ),
            "near_0_5pct_length_impulse_energy_ratio": energy_ratio(
                a2["distance_impulse_m"], a2["impulse_smoothed"]
            ),
            "distance_sample_step_m": float(a2["distance_step_m"]),
            "step_smoothing_points": int(a2["step_smoothing_points"]),
            "impulse_smoothing_points": int(a2["impulse_smoothing_points"]),
        },
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
              model_s11: np.ndarray, measured: dict[str, Any],
              model: dict[str, Any]) -> None:
    configure_plot()
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    colors = {"measured": "#1f77b4", "model": "#d95f02"}

    axes[0, 0].semilogx(frequency_hz, measured_s11.real, color=colors["measured"],
                        label="实测 Re")
    axes[0, 0].semilogx(frequency_hz, measured_s11.imag, color=colors["measured"],
                        linestyle="--", label="实测 Im")
    axes[0, 0].semilogx(frequency_hz, model_s11.real, color=colors["model"],
                        label="模型 Re")
    axes[0, 0].semilogx(frequency_hz, model_s11.imag, color=colors["model"],
                        linestyle="--", label="模型 Im")
    axes[0, 0].set_title("S11 实部与虚部")
    axes[0, 0].set_xlabel("频率 (Hz)")
    axes[0, 0].set_ylabel("S11")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].semilogx(frequency_hz, np.abs(measured_s11), color=colors["measured"],
                        label="实测 A相")
    axes[0, 1].semilogx(frequency_hz, np.abs(model_s11), color=colors["model"],
                        label="C00模型")
    axes[0, 1].set_title("|Γ| 频带形态")
    axes[0, 1].set_xlabel("频率 (Hz)")
    axes[0, 1].set_ylabel("|S11|")
    axes[0, 1].legend(fontsize=8)

    for axis, algorithm, key, title, ylabel in (
        (axes[0, 2], "a1", "impulse_raw", "Algorithm 1 脉冲", "IFFT impulse_raw"),
        (axes[1, 0], "a1", "step_rho", "Algorithm 1 阶跃", "step_rho"),
        (axes[1, 1], "a2", "impulse_smoothed", "Algorithm 2 脉冲（3 m平滑）", "impulse_smoothed"),
        (axes[1, 2], "a2", "step_smoothed", "Algorithm 2 阶跃（1 m平滑）", "step_smoothed"),
    ):
        for label, curves in (("实测 A相", measured), ("C00模型", model)):
            response = curves[algorithm]
            distance = (response["distance_m"] if algorithm == "a1" or key == "step_smoothed"
                        else response["distance_impulse_m"])
            axis.plot(distance, response[key], color=colors[
                "measured" if label == "实测 A相" else "model"],
                linewidth=0.8, label=label)
        axis.set_xlim(0.0, DISTANCE_MAX_M)
        axis.set_title(title)
        axis.set_xlabel("单程距离 (m；按172 Mm/s换算)")
        axis.set_ylabel(ylabel)
        axis.legend(fontsize=8)

    for axis in axes.flat:
        axis.grid(True, which="both", alpha=0.22)

    figure.suptitle(
        "2430 m现场A相与Core-16 C00：仅比较形态尺度与频带\n"
        "A相不是已知健康样本，C00不要求逐点匹配其接头/异常；fixture与本A相联合拟合同源，20 MHz以上未比较",
        fontsize=12,
    )
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(ASSET_DIR / "anchor_comparison.png", dpi=200)
    plt.close(figure)


def main() -> None:
    config = json.loads((HERE / "field_config.json").read_text(encoding="utf-8"))
    measured_source = load_measured_case()
    frequency_hz = np.linspace(
        FREQUENCY_START_HZ, FREQUENCY_STOP_HZ, FREQUENCY_POINTS, dtype=np.float64
    )
    measured_s11 = interp_complex(
        measured_source["frequency_hz"], measured_source["s11_complex"], frequency_hz
    )

    cases = build_cases(config, LENGTH_M)
    case = next(item for item in cases if item["id"] == "C00")
    model_s11 = simulate(
        frequency_hz,
        case["segments"],
        case["joints"],
        fixture=config["fixture"],
        route="dg_v3_rlgc",
        reference_hz=1.0e6,
        load=case["termination"],
        zref=50.0,
    )

    measured_responses = make_responses(frequency_hz, measured_s11)
    model_responses = make_responses(frequency_hz, model_s11)
    measured_bands = band_medians(frequency_hz, measured_s11)
    model_bands = band_medians(frequency_hz, model_s11)
    band_comparison = []
    for measured_band, model_band in zip(measured_bands, model_bands):
        band_comparison.append({
            "band": measured_band["band"],
            "lower_hz": measured_band["lower_hz"],
            "upper_hz": measured_band["upper_hz"],
            "point_count": measured_band["point_count"],
            "measured_median_abs_gamma": measured_band["median_abs_gamma"],
            "model_median_abs_gamma": model_band["median_abs_gamma"],
            "model_minus_measured": (
                float(model_band["median_abs_gamma"])
                - float(measured_band["median_abs_gamma"])
            ),
        })

    measured_response_metrics = response_metrics(measured_responses)
    model_response_metrics = response_metrics(model_responses)
    end_window = [END_WINDOW_FRACTION[0] * LENGTH_M,
                  END_WINDOW_FRACTION[1] * LENGTH_M]
    near_window = [NEAR_WINDOW_FRACTION[0] * LENGTH_M,
                   NEAR_WINDOW_FRACTION[1] * LENGTH_M]
    model_prior = {
        "body": config["body"],
        "joint": config["joint"],
        "joint_positions_fraction": config["joint_positions_fraction"],
        "joint_positions_m": [round(float(item["position_m"]), 3) for item in case["joints"]],
        "fixture": config["fixture"],
    }
    payload = {
        "comparison": {
            "nominal_length_m": LENGTH_M,
            "distance_max_m": DISTANCE_MAX_M,
            "velocity_m_s": VELOCITY_M_S,
            "frequency_hz": {
                "start": FREQUENCY_START_HZ,
                "stop": FREQUENCY_STOP_HZ,
                "points": FREQUENCY_POINTS,
                "grid": "common linear; measured complex S11 interpolated real/imag separately",
                "limitation": "Comparison stops at 20 MHz; the original measured sweep extends to 1 GHz.",
            },
            "route": "dg_v3_rlgc",
            "reference_hz": 1.0e6,
            "termination": case["termination"],
            "case_id": case["id"],
            "configuration_snapshot_version": (
                "Runtime field_config.json snapshot; five nonuniform fractions derived "
                "from 637 m reference points and scaled to the 2430 m model."
            ),
        },
        "sources": {
            "measured": {
                "npz": str(measured_source["npz_path"].relative_to(HERE)),
                "source_relative_path": measured_source["source_relative_path"],
                "sweep_name": measured_source["sweep_name"],
                "arrays": ["frequency_hz", "s11_complex"],
            },
            "model": {
                "generator": "field_cases.build_cases(field_config.json, 2430.0), case C00",
                "simulator": "field_physics.simulate",
                "route": "dg_v3_rlgc",
                "arrays": ["frequency_hz", "s11_complex"],
                "configuration": model_prior,
                "configuration_snapshot_version": (
                    "Runtime field_config.json snapshot; see exact fractions and scaled "
                    "absolute joint positions in configuration."
                ),
            },
        },
        "metrics": {
            "s11_band_abs_medians": band_comparison,
            "response_window_definitions": {
                "end_window_m": end_window,
                "near_end_window_m": near_window,
                "end_window_peak_is_not_automatic_terminal_detection": True,
                "near_end_energy": "integral(impulse^2) in 0–5%L divided by integral in 0–3159m",
            },
            "measured": measured_response_metrics,
            "model": model_response_metrics,
            "s11_complex_rmse": float(np.sqrt(np.mean(np.abs(measured_s11 - model_s11) ** 2))),
            "median_abs_s11_difference": float(np.median(np.abs(measured_s11 - model_s11))),
        },
        "interpretation_limits": [
            "现场A相不是已知健康样本；C00只是开路主对照，不能逐点匹配A相的所有接头/异常。",
            "本比较只看形态尺度和9 kHz–20 MHz频带，不据此宣称逐点物理验证。",
            "fixture来自同一2430m案例A相5–200 MHz短接+测量联合拟合；本比较复用了拟合anchor，不是独立泛化验证。",
            "body的Rskin/tan_delta和joint仍是先验探索参数；config中的fixture也仅是近端等效候选。",
            "Algorithm 1使用复数S11原生IFFT输出；Algorithm 2只使用实部并报告1 m/3 m平滑输出，两算法绝对幅值口径不同。",
            "线长2430m是本次指定的案例长度；末端窗峰只是固定窗口内的响应峰，不表示自动末端识别成功。",
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "anchor_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    np.savez_compressed(
        OUTPUT_DIR / "anchor_comparison.npz",
        frequency_hz=frequency_hz,
        measured_s11_complex=measured_s11,
        model_s11_complex=model_s11,
        algorithm1_distance_m=measured_responses["a1"]["distance_m"],
        algorithm1_measured_impulse=measured_responses["a1"]["impulse_raw"],
        algorithm1_model_impulse=model_responses["a1"]["impulse_raw"],
        algorithm1_measured_step_rho=measured_responses["a1"]["step_rho"],
        algorithm1_model_step_rho=model_responses["a1"]["step_rho"],
        algorithm2_distance_m=measured_responses["a2"]["distance_m"],
        algorithm2_impulse_distance_m=measured_responses["a2"]["distance_impulse_m"],
        algorithm2_measured_impulse=measured_responses["a2"]["impulse_smoothed"],
        algorithm2_model_impulse=model_responses["a2"]["impulse_smoothed"],
        algorithm2_measured_step=measured_responses["a2"]["step_smoothed"],
        algorithm2_model_step=model_responses["a2"]["step_smoothed"],
    )
    make_plot(frequency_hz, measured_s11, model_s11,
              measured_responses, model_responses)

    print(f"NPZ: {OUTPUT_DIR / 'anchor_comparison.npz'}")
    print(f"JSON: {OUTPUT_DIR / 'anchor_comparison.json'}")
    print(f"PNG: {ASSET_DIR / 'anchor_comparison.png'}")
    print(f"S11 complex RMSE (9 kHz–20 MHz): {payload['metrics']['s11_complex_rmse']:.8g}")


if __name__ == "__main__":
    main()
