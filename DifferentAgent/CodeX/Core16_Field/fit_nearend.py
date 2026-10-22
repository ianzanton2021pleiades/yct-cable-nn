from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from field_physics import _fixture, _to_s11


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
SIGNAL_DIR = HERE / "output" / "signals"
SOURCE_ROOT = Path(r"E:\20261010截至的全部fdr文件")
CONVERTER_PATH = PROJECT_ROOT / "REF" / "FDR数据文件快速转CSV-V2极速版.pyw"
OUTPUT_JSON = HERE / "output" / "nearend_fit.json"
OUTPUT_PNG = HERE / "assets" / "nearend_fit.png"
REFERENCE_HZ = 1.0e6
ZREF_OHM = 50.0
Z0_LINE_OHM = 50.0
FREQUENCY_MIN_HZ = 5.0e6
FREQUENCY_MAX_HZ = 200.0e6
FIT_POINT_COUNT = 512
BANDS = (
    (5.0e6, 20.0e6, "5-20MHz"),
    (20.0e6, 50.0e6, "20-50MHz"),
    (50.0e6, 100.0e6, "50-100MHz"),
    (100.0e6, 200.0e6, "100-200MHz"),
)
PARAMETER_NAMES = ("Zc_ohm", "series_R_ohm", "series_L_H", "shunt_C_F", "delay_s", "loss_db_ref")
LOWER = np.asarray([5.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
UPPER = np.asarray([150.0, 20.0, 5.0e-6, 10.0e-9, 100.0e-9, 10.0], dtype=np.float64)
STARTS = (
    np.asarray([50.0, 2.0, 0.5e-6, 0.5e-9, 20.0e-9, 2.0]),
    np.asarray([85.0, 5.0, 2.0e-6, 4.0e-9, 50.0e-9, 5.0]),
    np.asarray([20.0, 0.3, 0.05e-6, 0.1e-9, 5.0e-9, 0.2]),
)
ANCHORS = (
    {
        "anchor_id": "wuxi_baian_1100m",
        "title": "Wuxi Baian 1100 m",
        "expected_line_length_m": 1100.0,
        "length_evidence_source_path": "2023-12-26-无锡蓝安+藕光/蓝安431-1100米.db8.fdr",
        "length_evidence_statement": "The source FDR filename contains '1100米'; the case TXT has no separate explicit length statement.",
        "length_evidence_boundary": "filename evidence only; not independently verified from the FDR database",
        "npz_name": "00001_蓝安431-1100米.db8_s1.npz",
    },
    {
        "anchor_id": "2026-07-20_first_segment_637m_A",
        "title": "2026-07-20 first segment 637 m, A phase",
        "expected_line_length_m": 637.0,
        "length_evidence_source_path": "2026-7-20 10.46.01深圳某地两段电缆/第一段（从柜子测）/其他信息.txt",
        "length_evidence_statement": "FDR全长标注：637m",
        "length_evidence_boundary": "case TXT statement; not independently verified from the FDR database",
        "npz_name": "00067_2026-7-20_10.46.01_s1.npz",
    },
    {
        "anchor_id": "2026-09-14_2430m_first_phase_A",
        "title": "2026-09-14 2430 m, first phase A",
        "expected_line_length_m": 2430.0,
        "length_evidence_source_path": "2026-9-14 11.14.34深圳220KV清水河站10kV布吉环保二线F12北控环保污水厂#1智能柜(从接头开始测)/其他信息.txt",
        "length_evidence_statement": "FDR全长2430m",
        "length_evidence_boundary": "case TXT statement; not independently verified from the FDR database",
        "npz_name": "00093_2026-9-14_11.14.34_s1.npz",
    },
)


def json_float(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def display_parameters(parameters: np.ndarray) -> dict[str, float]:
    return {
        "Zc_ohm": float(parameters[0]),
        "series_R_ohm": float(parameters[1]),
        "series_L_uH": float(parameters[2] * 1.0e6),
        "shunt_C_nF": float(parameters[3] * 1.0e9),
        "delay_ns": float(parameters[4] * 1.0e9),
        "loss_db_at_1MHz": float(parameters[5]),
    }


def normalized(parameters: np.ndarray) -> np.ndarray:
    return (parameters - LOWER) / (UPPER - LOWER)


def physical(normalized_parameters: np.ndarray) -> np.ndarray:
    return LOWER + normalized_parameters * (UPPER - LOWER)


def bound_report(parameters: np.ndarray) -> dict[str, Any]:
    unit_position = normalized(parameters)
    tolerance = 1.0e-3
    near_lower = [name for name, value in zip(PARAMETER_NAMES, unit_position) if value <= tolerance]
    near_upper = [name for name, value in zip(PARAMETER_NAMES, unit_position) if value >= 1.0 - tolerance]
    return {
        "near_lower_bound": near_lower,
        "near_upper_bound": near_upper,
        "at_or_near_any_bound": bool(near_lower or near_upper),
        "tolerance_fraction_of_bound_span": tolerance,
    }


def nearend_s11(frequency_hz: np.ndarray, parameters: np.ndarray, load: str | float) -> np.ndarray:
    zc, series_r, series_l, shunt_c, delay_s, loss_db = parameters
    fixture = {
        "loss_db_ref": float(loss_db),
        "delay_s": float(delay_s),
        "z0_ohm": Z0_LINE_OHM,
        "series_r_ohm": float(series_r),
        "series_l_h": float(series_l),
        "shunt_g_s": 0.0,
        "shunt_c_f": float(shunt_c),
    }
    matrix = _fixture(frequency_hz, fixture, REFERENCE_HZ, ZREF_OHM)
    return _to_s11(matrix, ZREF_OHM, "short" if load == "short" else float(zc))


def interpolate_complex_log(source_frequency: np.ndarray, source_gamma: np.ndarray,
                            target_frequency: np.ndarray) -> np.ndarray:
    mask = (np.isfinite(source_frequency) & np.isfinite(source_gamma.real)
            & np.isfinite(source_gamma.imag) & (source_frequency > 0.0))
    source_frequency = source_frequency[mask]
    source_gamma = source_gamma[mask]
    order = np.argsort(source_frequency)
    source_frequency = source_frequency[order]
    source_gamma = source_gamma[order]
    unique_frequency, first = np.unique(source_frequency, return_index=True)
    source_gamma = source_gamma[first]
    source_x = np.log(source_frequency if unique_frequency.size == source_frequency.size else unique_frequency)
    if unique_frequency.size != source_frequency.size:
        source_frequency = unique_frequency
        source_x = np.log(source_frequency)
    target_x = np.log(target_frequency)
    real = np.interp(target_x, source_x, source_gamma.real)
    imag = np.interp(target_x, source_x, source_gamma.imag)
    return real + 1j * imag


def load_reader_module():
    loader = importlib.machinery.SourceFileLoader("fdr_reader_nearend", str(CONVERTER_PATH))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


def load_calibration(reader_module: Any, source_relative_path: str,
                     target_frequency: np.ndarray) -> tuple[np.ndarray, int]:
    if not hasattr(sqlite3.Connection, "deserialize"):
        raise RuntimeError("sqlite3.Connection.deserialize unavailable; no disk fallback is allowed")
    source_path = SOURCE_ROOT / Path(source_relative_path)
    encrypted = source_path.read_bytes()
    plaintext, _page_size = reader_module.FdrCrypto().decrypt(encrypted)
    connection = sqlite3.connect(":memory:")
    connection.deserialize(plaintext)
    try:
        rows = connection.execute("SELECT Freq, Real, Image FROM CalibrationData").fetchall()
        array = np.asarray(rows, dtype=np.float64)
        gamma = array[:, 1] + 1j * array[:, 2]
        return interpolate_complex_log(array[:, 0], gamma, target_frequency), int(array.shape[0])
    finally:
        connection.close()


def complex_rmse(actual: np.ndarray, fitted: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.abs(fitted - actual) ** 2)))


def band_rmse(actual: np.ndarray, fitted: np.ndarray, frequency_hz: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    for low, high, label in BANDS:
        mask = (frequency_hz >= low) & ((frequency_hz < high) if high < FREQUENCY_MAX_HZ else (frequency_hz <= high))
        result[label] = complex_rmse(actual[mask], fitted[mask])
    return result


def residual_vector(normalized_parameters: np.ndarray, frequency_hz: np.ndarray,
                    measurement_gamma: np.ndarray, calibration_gamma: np.ndarray | None) -> np.ndarray:
    parameters = physical(normalized_parameters)
    measurement_fit = nearend_s11(frequency_hz, parameters, parameters[0])
    measurement_error = measurement_fit - measurement_gamma
    n = float(np.sqrt(frequency_hz.size))
    components = [measurement_error.real / n, measurement_error.imag / n]
    if calibration_gamma is not None:
        calibration_fit = nearend_s11(frequency_hz, parameters, "short")
        calibration_error = calibration_fit - calibration_gamma
        components.extend((calibration_error.real / n, calibration_error.imag / n))
    return np.concatenate(components)


def run_measurement_multistart(frequency_hz: np.ndarray, measurement_gamma: np.ndarray) -> tuple[Any, list[dict[str, Any]]]:
    results = []
    for index, start in enumerate(STARTS, 1):
        fit = least_squares(
            residual_vector,
            normalized(start),
            args=(frequency_hz, measurement_gamma, None),
            bounds=(np.zeros_like(LOWER), np.ones_like(UPPER)),
            x_scale="jac",
            max_nfev=450,
            ftol=1.0e-9,
            xtol=1.0e-9,
            gtol=1.0e-9,
        )
        fitted = physical(fit.x)
        entry = {
            "start_index": index,
            "initial_values": display_parameters(start),
            "fitted_values": display_parameters(fitted),
            "measurement_rmse": complex_rmse(measurement_gamma, nearend_s11(frequency_hz, fitted, fitted[0])),
            "optimizer_success": bool(fit.success),
            "optimizer_message": str(fit.message),
            "nfev": int(fit.nfev),
            "cost": float(fit.cost),
        }
        results.append((fit, entry))
    best_fit, _ = min(results, key=lambda item: item[0].cost)
    return best_fit, [entry for _, entry in results]


def fit_anchor(anchor: dict[str, Any], reader_module: Any, frequency_grid: np.ndarray) -> dict[str, Any]:
    npz_path = SIGNAL_DIR / anchor["npz_name"]
    with np.load(npz_path, allow_pickle=False) as data:
        raw_frequency = np.asarray(data["frequency_hz"], dtype=np.float64)
        raw_gamma = np.asarray(data["s11_complex"], dtype=np.complex128)
        relative_path = str(data["source_relative_path"].item())
        sweep_name = str(data["sweep_name"].item())
        measurement_gamma = interpolate_complex_log(raw_frequency, raw_gamma, frequency_grid)
    calibration_gamma, calibration_point_count = load_calibration(reader_module, relative_path, frequency_grid)

    best_measurement_fit, multistart_results = run_measurement_multistart(frequency_grid, measurement_gamma)
    measurement_only_parameters = physical(best_measurement_fit.x)
    measurement_only_prediction = nearend_s11(frequency_grid, measurement_only_parameters,
                                               measurement_only_parameters[0])
    measurement_only_short_prediction = nearend_s11(frequency_grid, measurement_only_parameters, "short")

    joint_fit = least_squares(
        residual_vector,
        best_measurement_fit.x,
        args=(frequency_grid, measurement_gamma, calibration_gamma),
        bounds=(np.zeros_like(LOWER), np.ones_like(UPPER)),
        x_scale="jac",
        max_nfev=500,
        ftol=1.0e-9,
        xtol=1.0e-9,
        gtol=1.0e-9,
    )
    joint_parameters = physical(joint_fit.x)
    joint_measurement_prediction = nearend_s11(frequency_grid, joint_parameters, joint_parameters[0])
    joint_calibration_prediction = nearend_s11(frequency_grid, joint_parameters, "short")

    measurement_only_rmse = complex_rmse(measurement_gamma, measurement_only_prediction)
    joint_measurement_rmse = complex_rmse(measurement_gamma, joint_measurement_prediction)
    calibration_only_rmse = complex_rmse(calibration_gamma, measurement_only_short_prediction)
    joint_calibration_rmse = complex_rmse(calibration_gamma, joint_calibration_prediction)
    return {
        **anchor,
        "source_relative_path": relative_path,
        "signal_npz": npz_path.name,
        "curve_name": sweep_name,
        "frequency_band_hz": [FREQUENCY_MIN_HZ, FREQUENCY_MAX_HZ],
        "fit_frequency_sample_count": int(frequency_grid.size),
        "measurement_source_point_count": int(raw_frequency.size),
        "calibration_source_point_count": calibration_point_count,
        "measurement_fit": {
            "initial_values_by_start": [display_parameters(start) for start in STARTS],
            "multistart_results": multistart_results,
            "best_start_index": int(min(multistart_results, key=lambda row: row["cost"])["start_index"]),
            "fitted_values": display_parameters(measurement_only_parameters),
            "complex_rmse": measurement_only_rmse,
            "band_complex_rmse": band_rmse(measurement_gamma, measurement_only_prediction, frequency_grid),
            "calibration_short_complex_rmse_without_short_constraint": calibration_only_rmse,
            "calibration_short_band_complex_rmse_without_short_constraint": band_rmse(
                calibration_gamma, measurement_only_short_prediction, frequency_grid),
            "at_bound": bound_report(measurement_only_parameters),
        },
        "joint_short_measurement_fit": {
            "initial_values": display_parameters(measurement_only_parameters),
            "fitted_values": display_parameters(joint_parameters),
            "measurement_complex_rmse": joint_measurement_rmse,
            "measurement_band_complex_rmse": band_rmse(measurement_gamma, joint_measurement_prediction, frequency_grid),
            "calibration_short_complex_rmse": joint_calibration_rmse,
            "calibration_short_band_complex_rmse": band_rmse(calibration_gamma, joint_calibration_prediction, frequency_grid),
            "measurement_rmse_change_vs_measurement_only": joint_measurement_rmse - measurement_only_rmse,
            "measurement_rmse_worsened_vs_measurement_only": bool(joint_measurement_rmse > measurement_only_rmse),
            "at_bound": bound_report(joint_parameters),
            "optimizer_success": bool(joint_fit.success),
            "optimizer_message": str(joint_fit.message),
            "nfev": int(joint_fit.nfev),
        },
        "_plot_data": {
            "raw_frequency_hz": raw_frequency,
            "raw_measurement_gamma": raw_gamma,
            "measurement_gamma": measurement_gamma,
            "calibration_gamma": calibration_gamma,
            "measurement_only_prediction": measurement_only_prediction,
            "measurement_only_short_prediction": measurement_only_short_prediction,
            "joint_measurement_prediction": joint_measurement_prediction,
            "joint_calibration_prediction": joint_calibration_prediction,
        },
    }


def create_plot(results: list[dict[str, Any]]) -> None:
    cache_dir = HERE / "cache" / "matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)
    os.environ["XDG_CACHE_HOME"] = str(HERE / "cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.sans-serif": ["SimHei"],
        "axes.unicode_minus": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.linewidth": 0.8,
        "font.size": 8,
        "savefig.dpi": 200,
    })
    fig, axes = plt.subplots(5, 3, figsize=(15.0, 11.8), sharex="col", constrained_layout=True)
    row_labels = ("Measurement |Γ|", "Measurement Re(Γ)", "Measurement Im(Γ)",
                  "Short Re(Γ)", "Short Im(Γ)")
    for column, result in enumerate(results):
        axes[0, column].set_title(
            f"{result['title']}\nLength evidence: {result['expected_line_length_m']:.0f} m"
        )
        data = result["_plot_data"]
        raw_f = data["raw_frequency_hz"]
        raw_g = data["raw_measurement_gamma"]
        raw_mask = (raw_f >= FREQUENCY_MIN_HZ) & (raw_f <= FREQUENCY_MAX_HZ)
        raw_f_mhz = raw_f[raw_mask] / 1.0e6
        raw_g = raw_g[raw_mask]
        stride = max(1, raw_f_mhz.size // 4000)
        raw_f_mhz = raw_f_mhz[::stride]
        raw_g = raw_g[::stride]
        grid_mhz = np.geomspace(FREQUENCY_MIN_HZ, FREQUENCY_MAX_HZ, FIT_POINT_COUNT) / 1.0e6

        axes[0, column].semilogx(raw_f_mhz, np.abs(raw_g), color="0.55", linewidth=0.6, label="Measured")
        axes[0, column].semilogx(grid_mhz, np.abs(data["measurement_only_prediction"]),
                                 color="#276FBF", linestyle="--", linewidth=1.0, label="Measurement-only")
        axes[0, column].semilogx(grid_mhz, np.abs(data["joint_measurement_prediction"]),
                                 color="#C43C39", linewidth=1.0, label="Joint with short")
        axes[1, column].semilogx(raw_f_mhz, raw_g.real, color="0.55", linewidth=0.6)
        axes[1, column].semilogx(grid_mhz, data["measurement_only_prediction"].real,
                                 color="#276FBF", linestyle="--", linewidth=1.0)
        axes[1, column].semilogx(grid_mhz, data["joint_measurement_prediction"].real,
                                 color="#C43C39", linewidth=1.0)
        axes[2, column].semilogx(raw_f_mhz, raw_g.imag, color="0.55", linewidth=0.6)
        axes[2, column].semilogx(grid_mhz, data["measurement_only_prediction"].imag,
                                 color="#276FBF", linestyle="--", linewidth=1.0)
        axes[2, column].semilogx(grid_mhz, data["joint_measurement_prediction"].imag,
                                 color="#C43C39", linewidth=1.0)

        cal = data["calibration_gamma"]
        axes[3, column].semilogx(grid_mhz, cal.real, color="0.55", linewidth=0.7, label="Measured short")
        axes[3, column].semilogx(grid_mhz, data["measurement_only_short_prediction"].real,
                                 color="#276FBF", linestyle="--", linewidth=1.0, label="Measurement-only params")
        axes[3, column].semilogx(grid_mhz, data["joint_calibration_prediction"].real,
                                 color="#C43C39", linewidth=1.0, label="Joint fit")
        axes[4, column].semilogx(grid_mhz, cal.imag, color="0.55", linewidth=0.7)
        axes[4, column].semilogx(grid_mhz, data["measurement_only_short_prediction"].imag,
                                 color="#276FBF", linestyle="--", linewidth=1.0)
        axes[4, column].semilogx(grid_mhz, data["joint_calibration_prediction"].imag,
                                 color="#C43C39", linewidth=1.0)
        for row in range(5):
            axes[row, column].set_xlim(FREQUENCY_MIN_HZ / 1.0e6, FREQUENCY_MAX_HZ / 1.0e6)
            axes[row, column].grid(True, which="both", alpha=0.22, linewidth=0.5)
            axes[row, column].tick_params(which="both", direction="in")
        axes[4, column].set_xlabel("Frequency (MHz)")
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label)
    axes[0, 0].legend(loc="best", fontsize=7)
    axes[3, 0].legend(loc="best", fontsize=7)
    fig.suptitle("Near-end equivalent fits, 5–200 MHz — non-unique, not physical component estimates", fontsize=12)
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=200)
    plt.close(fig)


def main() -> int:
    started = time.perf_counter()
    (HERE / "output").mkdir(parents=True, exist_ok=True)
    (HERE / "assets").mkdir(parents=True, exist_ok=True)
    frequency_grid = np.geomspace(FREQUENCY_MIN_HZ, FREQUENCY_MAX_HZ, FIT_POINT_COUNT)
    reader_module = load_reader_module()
    results = [fit_anchor(anchor, reader_module, frequency_grid) for anchor in ANCHORS]
    plot_results = results
    report_results = []
    for result in results:
        report_results.append({key: value for key, value in result.items() if key != "_plot_data"})
    report = {
        "method": "field_physics._fixture followed by _to_s11; a real semi-infinite Zc load represents effective cable input",
        "frequency_band_hz": [FREQUENCY_MIN_HZ, FREQUENCY_MAX_HZ],
        "fit_frequency_sampling": {
            "point_count": FIT_POINT_COUNT,
            "spacing": "uniform in log frequency",
            "source_resampling": "linear interpolation of complex real/imaginary parts versus log frequency",
            "weighting": "each log-spaced frequency has equal weight",
        },
        "band_residuals_hz": [{"band": label, "low_hz": low, "high_hz": high,
                                "high_inclusive": high == FREQUENCY_MAX_HZ}
                               for low, high, label in BANDS],
        "fixture_model": {
            "reference_impedance_ohm": ZREF_OHM,
            "fixed_fixture_line_z0_ohm": Z0_LINE_OHM,
            "fixed_shunt_conductance_s": 0.0,
            "loss_reference_frequency_hz": REFERENCE_HZ,
            "parameter_order": list(PARAMETER_NAMES),
            "bounds": {
                "Zc_ohm": [5.0, 150.0],
                "series_R_ohm": [0.0, 20.0],
                "series_L_uH": [0.0, 5.0],
                "shunt_C_nF": [0.0, 10.0],
                "delay_ns": [0.0, 100.0],
                "loss_db_at_1MHz": [0.0, 10.0],
            },
            "loss_model": "field_physics._fixture one-way loss_db_ref*sqrt(f/1MHz)",
            "short_load_model": "field_physics._to_s11(..., load='short')",
        },
        "assumptions_and_limits": [
            "The 5-200 MHz band is used because reflections from the far end of these long cables are expected to be substantially attenuated there; this is a modeling assumption, not independently proven by this fit.",
            "The fitted Zc is an effective semi-infinite cable input, not a body RLGC inversion.",
            "The fixture coefficients are near-end equivalent parameters and are not asserted to be the actual physical R, L, C, delay, or loss of the fixture.",
            "The short measurement constrains the same fixture before cable connection; with an ideal short at the fixture output, shunt C is not independently identifiable from the short response.",
            "The joint fit remains non-unique and does not invert three calibration error terms or automatically correct the original S11.",
            "No vendor response waveform was used as a template or injected into a generator.",
        ],
        "anchors": report_results,
    }
    create_plot(plot_results)
    report["elapsed_seconds"] = time.perf_counter() - started
    with OUTPUT_JSON.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({
        "output_json": str(OUTPUT_JSON),
        "output_plot": str(OUTPUT_PNG),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "anchors": [
            {
                "anchor_id": item["anchor_id"],
                "measurement_rmse_only": item["measurement_fit"]["complex_rmse"],
                "measurement_rmse_joint": item["joint_short_measurement_fit"]["measurement_complex_rmse"],
                "short_rmse_joint": item["joint_short_measurement_fit"]["calibration_short_complex_rmse"],
                "measurement_worsened": item["joint_short_measurement_fit"]["measurement_rmse_worsened_vs_measurement_only"],
            }
            for item in report_results
        ],
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
