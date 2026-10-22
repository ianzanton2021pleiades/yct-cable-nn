from __future__ import annotations

import csv
import json
import os
import re
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


EXPECTED_WET_DISTANCES_M = (100, 200, 300, 400, 500, 600, 700, 900, 1100, 1300, 1400)
NOMINAL_END_MARKER_M = 1500.0
TERMINAL_GUARD_M = 40.0
IMPULSE_SMOOTHING_M = 3.0
S11_BANDS = (
    (9.0e3, 100.0e3, "9k_100k"),
    (100.0e3, 1.0e6, "100k_1M"),
    (1.0e6, 10.0e6, "1M_10M"),
    (10.0e6, 100.0e6, "10M_100M"),
    (100.0e6, 1.0e9, "100M_1G"),
)
SOURCE_FIELDS = (
    "vendor_impulse_distance",
    "vendor_impulse_rho",
    "vendor_step_distance",
    "vendor_step_rho",
    "frequency_hz",
    "s11_complex",
    "sweep_name",
)


def _read_scalar(archive: Any, name: str) -> str:
    return str(archive[name].item())


def load_sweeps() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    baseline = None
    wet = []
    for path in sorted(SIGNAL_DIR.glob("*1500m*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            sweep_name = _read_scalar(archive, "sweep_name")
            item = {
                "source_npz": path.name,
                "sweep_name": sweep_name,
                "vendor_impulse_distance": archive["vendor_impulse_distance"].copy(),
                "vendor_impulse_rho": archive["vendor_impulse_rho"].copy(),
                "vendor_step_distance": archive["vendor_step_distance"].copy(),
                "vendor_step_rho": archive["vendor_step_rho"].copy(),
                "frequency_hz": archive["frequency_hz"].copy(),
                "s11_complex": archive["s11_complex"].copy(),
            }
        if sweep_name == "正常":
            baseline = item
        else:
            match = re.match(r"^(\d+)\s*m", sweep_name)
            if match is not None:
                item["wet_distance_m"] = int(match.group(1))
                wet.append(item)

    if baseline is None:
        raise FileNotFoundError("1500 m浸水NPZ中没有 sweep_name='正常' 的基准记录")
    wet.sort(key=lambda item: item["wet_distance_m"])
    found = tuple(item["wet_distance_m"] for item in wet)
    if found != EXPECTED_WET_DISTANCES_M:
        raise ValueError(f"浸水位置应为100–1400 m共11条，实际为{found}")
    return baseline, wet


def physical_moving_mean(distance_m: np.ndarray, values: np.ndarray,
                         width_m: float) -> tuple[np.ndarray, int, float]:
    spacing_m = float(np.median(np.diff(distance_m)))
    intervals = max(1, int(round(width_m / spacing_m)))
    points = intervals + 1
    if points % 2 == 0:
        points += 1
    kernel = np.full(points, 1.0 / points, dtype=np.float64)
    smoothed = np.convolve(values, kernel, mode="same")
    actual_width_m = float((points - 1) * spacing_m)
    return smoothed, points, actual_width_m


def half_height_width(distance_m: np.ndarray, values: np.ndarray,
                      peak_index: int, low_index: int,
                      high_index: int) -> tuple[float | None, bool, str]:
    peak = float(values[peak_index])
    if peak >= 0.0:
        return None, False, "no_negative_peak"
    level = peak / 2.0

    left = peak_index
    while left > low_index and values[left - 1] <= level:
        left -= 1
    if left == low_index and values[left] <= level:
        left_crossing = None
    else:
        x0, y0 = distance_m[left - 1], values[left - 1]
        x1, y1 = distance_m[left], values[left]
        left_crossing = float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))

    right = peak_index
    while right < high_index and values[right + 1] <= level:
        right += 1
    if right == high_index and values[right] <= level:
        right_crossing = None
    else:
        x0, y0 = distance_m[right], values[right]
        x1, y1 = distance_m[right + 1], values[right + 1]
        right_crossing = float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))

    if left_crossing is None or right_crossing is None:
        side = "both" if left_crossing is None and right_crossing is None else (
            "left" if left_crossing is None else "right"
        )
        return None, False, f"half_height_open_{side}"
    return right_crossing - left_crossing, True, "closed"


def fit_line(x: list[float], y: list[float]) -> dict[str, float | int | None]:
    if len(x) < 2:
        return {"count": len(x), "slope_per_m": None, "intercept": None,
                "r_squared": None}
    slope, intercept = np.polyfit(np.asarray(x), np.asarray(y), 1)
    predicted = slope * np.asarray(x) + intercept
    residual = np.asarray(y) - predicted
    total = np.sum((np.asarray(y) - np.mean(y)) ** 2)
    r_squared = 1.0 - float(np.sum(residual**2) / total) if total > 0.0 else 1.0
    return {"count": len(x), "slope_per_m": float(slope),
            "intercept": float(intercept), "r_squared": r_squared}


def analyze_one(baseline: dict[str, Any], wet: dict[str, Any]) -> dict[str, Any]:
    x = wet["vendor_impulse_distance"]
    base_rho = np.interp(x, baseline["vendor_impulse_distance"],
                        baseline["vendor_impulse_rho"])
    delta = wet["vendor_impulse_rho"] - base_rho
    smoothed, points, actual_width_m = physical_moving_mean(
        x, delta, IMPULSE_SMOOTHING_M
    )
    p = int(wet["wet_distance_m"])
    low_m = p - 0.1 * p - 10.0
    high_m = p + 0.15 * p + 30.0
    terminal_cap_m = NOMINAL_END_MARKER_M - TERMINAL_GUARD_M
    high_m = min(high_m, terminal_cap_m)
    low_index = int(np.searchsorted(x, low_m, side="left"))
    high_index = int(np.searchsorted(x, high_m, side="right")) - 1
    peak_index = low_index + int(np.argmin(smoothed[low_index:high_index + 1]))
    peak_position_m = float(x[peak_index])
    peak_rho = float(smoothed[peak_index])
    width_m, width_complete, width_status = half_height_width(
        x, smoothed, peak_index, low_index, high_index
    )

    residual_exclusion_m = max(15.0, width_m / 2.0 + 10.0) if width_m else 15.0
    residual_mask = (
        (x >= low_m) & (x <= high_m)
        & (np.abs(x - peak_position_m) >= residual_exclusion_m)
    )
    residual_floor = (
        float(np.sqrt(np.mean(smoothed[residual_mask] ** 2)))
        if np.any(residual_mask) else None
    )
    peak_to_floor = (
        float(abs(peak_rho) / residual_floor)
        if residual_floor is not None and residual_floor > 0.0 else None
    )
    sufficiently_distinguishable = bool(
        peak_rho < 0.0
        and residual_floor is not None
        and (peak_to_floor is None or peak_to_floor >= 3.0)
    )

    frequency = wet["frequency_hz"]
    baseline_frequency = baseline["frequency_hz"]
    baseline_s11 = baseline["s11_complex"]
    baseline_s11_on_grid = (
        np.interp(frequency, baseline_frequency, baseline_s11.real)
        + 1.0j * np.interp(frequency, baseline_frequency, baseline_s11.imag)
    )
    delta_s11 = wet["s11_complex"] - baseline_s11_on_grid
    band_metrics: dict[str, Any] = {}
    for lower_hz, upper_hz, band_name in S11_BANDS:
        mask = (frequency >= lower_hz) & (frequency < upper_hz)
        if upper_hz == S11_BANDS[-1][1]:
            mask |= frequency == upper_hz
        band_metrics[band_name] = {
            "rms": float(np.sqrt(np.mean(np.abs(delta_s11[mask]) ** 2))),
            "point_count": int(np.count_nonzero(mask)),
        }

    fit_included = bool(p < 1400 and sufficiently_distinguishable)
    return {
        "source_npz": wet["source_npz"],
        "sweep_name": wet["sweep_name"],
        "wet_distance_m": p,
        "vendor_impulse_distance_point_count": int(x.size),
        "vendor_impulse_distance_start_m": float(x[0]),
        "vendor_impulse_distance_end_m": float(x[-1]),
        "vendor_step_distance_point_count": int(wet["vendor_step_distance"].size),
        "vendor_step_distance_start_m": float(wet["vendor_step_distance"][0]),
        "vendor_step_distance_end_m": float(wet["vendor_step_distance"][-1]),
        "frequency_hz_point_count": int(frequency.size),
        "frequency_hz_start": float(frequency[0]),
        "frequency_hz_end": float(frequency[-1]),
        "source_fields": ";".join(SOURCE_FIELDS),
        "smoothing_width_requested_m": IMPULSE_SMOOTHING_M,
        "smoothing_points": points,
        "smoothing_width_actual_m": actual_width_m,
        "peak_window_low_m": float(low_m),
        "peak_window_high_m": float(high_m),
        "peak_position_m": peak_position_m,
        "peak_offset_from_label_m": peak_position_m - p,
        "negative_peak_rho": peak_rho,
        "peak_abs_rho": abs(peak_rho),
        "fwhm_m": width_m,
        "fwhm_complete": width_complete,
        "fwhm_status": width_status,
        "neighbor_residual_floor_rms": residual_floor,
        "peak_to_residual_floor": peak_to_floor,
        "sufficiently_distinguishable_3x_floor": sufficiently_distinguishable,
        "attenuation_fit_included": fit_included,
        "attenuation_fit_exclusion": (
            "near_nominal_1500m_terminal" if p == 1400 else
            (None if fit_included else "peak_not_3x_local_residual_floor")
        ),
        **{
            f"delta_s11_rms_{name}": metric["rms"]
            for name, metric in band_metrics.items()
        },
        **{
            f"delta_s11_point_count_{name}": metric["point_count"]
            for name, metric in band_metrics.items()
        },
        "_smoothed_distance": x,
        "_smoothed_delta": smoothed,
        "_band_metrics": band_metrics,
    }


def baseline_record(baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_npz": baseline["source_npz"],
        "sweep_name": baseline["sweep_name"],
        "wet_distance_m": None,
        "vendor_impulse_distance_point_count": int(baseline["vendor_impulse_distance"].size),
        "vendor_impulse_distance_start_m": float(baseline["vendor_impulse_distance"][0]),
        "vendor_impulse_distance_end_m": float(baseline["vendor_impulse_distance"][-1]),
        "vendor_step_distance_point_count": int(baseline["vendor_step_distance"].size),
        "vendor_step_distance_start_m": float(baseline["vendor_step_distance"][0]),
        "vendor_step_distance_end_m": float(baseline["vendor_step_distance"][-1]),
        "frequency_hz_point_count": int(baseline["frequency_hz"].size),
        "frequency_hz_start": float(baseline["frequency_hz"][0]),
        "frequency_hz_end": float(baseline["frequency_hz"][-1]),
        "source_fields": ";".join(SOURCE_FIELDS),
        "smoothing_width_requested_m": IMPULSE_SMOOTHING_M,
        "smoothing_points": None,
        "smoothing_width_actual_m": None,
        "peak_window_low_m": None,
        "peak_window_high_m": None,
        "peak_position_m": None,
        "peak_offset_from_label_m": None,
        "negative_peak_rho": 0.0,
        "peak_abs_rho": 0.0,
        "fwhm_m": None,
        "fwhm_complete": False,
        "fwhm_status": "baseline_is_reference",
        "neighbor_residual_floor_rms": None,
        "peak_to_residual_floor": None,
        "sufficiently_distinguishable_3x_floor": False,
        "attenuation_fit_included": False,
        "attenuation_fit_exclusion": "normal_reference_not_a_no_joint_baseline",
        **{f"delta_s11_rms_{name}": 0.0 for _, _, name in S11_BANDS},
        **{f"delta_s11_point_count_{name}": 0 for _, _, name in S11_BANDS},
    }


def make_plot(baseline: dict[str, Any], records: list[dict[str, Any]],
              fit: dict[str, Any], width_trend: dict[str, Any]) -> None:
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
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    ax_impulse, ax_amplitude, ax_width, ax_bands = axes.flat

    base_x = baseline["vendor_impulse_distance"]
    ax_impulse.plot(base_x, np.zeros_like(base_x), color="black", linestyle="--",
                    linewidth=1.0, label="正常−正常（0）")
    colors = plt.get_cmap("viridis")
    selected = {100, 400, 700, 900, 1100, 1300, 1400}
    for record in records:
        if record["wet_distance_m"] not in selected:
            continue
        color = colors((record["wet_distance_m"] - 100.0) / 1300.0)
        ax_impulse.plot(record["_smoothed_distance"], record["_smoothed_delta"],
                        color=color, linewidth=1.0,
                        label=f"{record['wet_distance_m']} m")
    ax_impulse.set_xlim(20.0, 1650.0)
    ax_impulse.set_xlabel("距离 (m)")
    ax_impulse.set_ylabel("vendor 脉冲 Δρ（浸水−正常）")
    ax_impulse.set_title("3 m物理滑动均值后的差分脉冲")
    ax_impulse.legend(ncol=2, fontsize=8)

    detectable = [r for r in records if r["sufficiently_distinguishable_3x_floor"]]
    x_amp = [r["wet_distance_m"] for r in records]
    y_amp = [r["peak_abs_rho"] for r in records]
    ax_amplitude.semilogy(x_amp, y_amp, "o", color="#1f77b4", label="位置窗负峰幅值")
    terminal = next(r for r in records if r["wet_distance_m"] == 1400)
    ax_amplitude.semilogy([1400], [terminal["peak_abs_rho"]], marker="o",
                          markerfacecolor="none", markeredgecolor="#d62728",
                          linestyle="none", markersize=8, label="1400 m（近名义末端，拟合排除）")
    if fit["count"] >= 2:
        fit_x = np.linspace(min(r["wet_distance_m"] for r in detectable if r["wet_distance_m"] < 1400),
                            max(r["wet_distance_m"] for r in detectable if r["wet_distance_m"] < 1400), 100)
        fit_y = 10.0 ** (fit["intercept"] + fit["slope_per_m"] * fit_x)
        ax_amplitude.semilogy(fit_x, fit_y, "--", color="#d62728",
                              label=f"可辨样本拟合 {fit['attenuation_db_per_km']:.2f} dB/km, n={fit['count']}")
    ax_amplitude.set_xlabel("浸水接头标注距离 (m)")
    ax_amplitude.set_ylabel("|差分负峰 Δρ|")
    ax_amplitude.set_title("差分峰随距离衰减")
    ax_amplitude.grid(True, which="both", alpha=0.25)
    ax_amplitude.legend(fontsize=8)

    full_widths = [r for r in records if r["fwhm_complete"] and r["fwhm_m"] is not None]
    width_fit_points = [
        r for r in full_widths
        if r["wet_distance_m"] < 1400
        and r["sufficiently_distinguishable_3x_floor"]
    ]
    ax_width.plot([r["wet_distance_m"] for r in width_fit_points],
                  [r["fwhm_m"] for r in width_fit_points], "o-",
                  color="#2ca02c", label="完整且可辨半高宽")
    width_fit_positions = {r["wet_distance_m"] for r in width_fit_points}
    width_unfit = [r for r in full_widths if r["wet_distance_m"] not in width_fit_positions]
    if width_unfit:
        ax_width.plot([r["wet_distance_m"] for r in width_unfit],
                      [r["fwhm_m"] for r in width_unfit], "o",
                      markerfacecolor="none", markeredgecolor="#d62728",
                      linestyle="none", label="完整但未入拟合")
    if width_trend["count"] >= 2:
        fit_x = np.linspace(min(r["wet_distance_m"] for r in width_fit_points),
                            max(r["wet_distance_m"] for r in width_fit_points), 100)
        fit_y = width_trend["intercept"] + width_trend["slope_per_m"] * fit_x
        ax_width.plot(fit_x, fit_y, "--", color="#d62728",
                      label=f"{min(r['wet_distance_m'] for r in width_fit_points)}–{max(r['wet_distance_m'] for r in width_fit_points)} m趋势 {width_trend['slope_m_per_km']:.2f} m/km")
    ax_width.set_xlabel("浸水接头标注距离 (m)")
    ax_width.set_ylabel("差分负峰半高宽 (m)")
    ax_width.set_title("峰展宽趋势（边界不闭合者留空）")
    ax_width.grid(True, alpha=0.25)
    ax_width.legend(fontsize=8)

    band_names = [name for _, _, name in S11_BANDS]
    band_labels = ["9 k–100 k", "100 k–1 M", "1–10 M", "10–100 M", "100 M–1 G"]
    norm = Normalize(vmin=100.0, vmax=1400.0)
    for record in records:
        metrics = record["_band_metrics"]
        values = [metrics[name]["rms"] for name in band_names]
        ax_bands.plot(np.arange(len(values)), values, marker="o", linewidth=1.0,
                      color=colors(norm(record["wet_distance_m"])))
    ax_bands.set_yscale("log")
    ax_bands.set_xticks(np.arange(len(band_labels)), band_labels, rotation=20, ha="right")
    ax_bands.set_xlabel("频带 (Hz)")
    ax_bands.set_ylabel("复数差分 S11 RMS（反射有效频带指标）")
    ax_bands.set_title("浸水−正常的分频带反射差分")
    ax_bands.grid(True, which="both", alpha=0.25)
    colorbar = figure.colorbar(ScalarMappable(norm=norm, cmap=colors), ax=ax_bands)
    colorbar.set_label("接头标注距离 (m)")

    figure.suptitle(
        "1500 m单接头浸水受控试验：距离衰减与展宽\n"
        "源TXT注明该长电缆介损远高于XLPE；本结果不能外推为通用电缆损耗",
        fontsize=13,
    )
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(ASSET_DIR / "wet_distance_characteristics.png", dpi=200)
    plt.close(figure)


def main() -> None:
    baseline, wet = load_sweeps()
    baseline_record_data = baseline_record(baseline)
    records = [analyze_one(baseline, item) for item in wet]

    fit_records = [r for r in records if r["attenuation_fit_included"]]
    amplitude_fit = fit_line(
        [float(r["wet_distance_m"]) for r in fit_records],
        [float(np.log10(r["peak_abs_rho"])) for r in fit_records],
    )
    attenuation = (
        -20.0 * float(amplitude_fit["slope_per_m"]) * 1000.0
        if amplitude_fit["slope_per_m"] is not None else None
    )
    amplitude_fit["attenuation_db_per_km"] = attenuation

    width_records = [
        r for r in records
        if r["fwhm_complete"] and r["fwhm_m"] is not None
        and r["wet_distance_m"] < 1400
        and r["sufficiently_distinguishable_3x_floor"]
    ]
    width_trend = fit_line(
        [float(r["wet_distance_m"]) for r in width_records],
        [float(r["fwhm_m"]) for r in width_records],
    )
    width_trend["slope_m_per_km"] = (
        1000.0 * float(width_trend["slope_per_m"])
        if width_trend["slope_per_m"] is not None else None
    )

    for record in records:
        record.pop("_smoothed_distance")
        record.pop("_smoothed_delta")
        record.pop("_band_metrics")
    output_records = [baseline_record_data, *records]
    methods = {
        "baseline": "sweep_name=正常；它是同一受控试验参照，不代表无接头电缆。",
        "impulse_difference": "浸水vendor_impulse_rho减正常曲线；统一3 m物理滑动均值。",
        "peak_window": "[p-0.1p-10, p+0.15p+30] m；为避开名义1500 m终端邻域，上界最多1460 m。",
        "peak": "限定窗内3 m平滑差分的负向最小值；半高宽为Δρ=peak/2两侧闭合交点距离，任一边界未闭合则宽度为空。",
        "residual_floor": "限定位置窗内、排除峰中心至少15 m（若宽峰则按半宽+10 m）的平滑差分RMS。",
        "distinguishable": "操作性判据：负峰绝对值至少为邻域残差RMS的3倍；不是统计显著性检验。",
        "attenuation_fit": "对可辨样本且p<1400 m拟合log10(|Δρ峰|)=a+b·p；衰减=-20b×1000 dB/km。1400 m单列，不入拟合。",
        "width_trend": "完整且可辨、p<1400 m样本的半高宽对距离线性趋势。",
        "s11_bands": "复数差分S11 wet-normal的RMS=sqrt(mean(|ΔS11|^2))；分段9k–100k、100k–1M、1–10M、10–100M、100M–1G Hz；这是反射有效频带指标，不称噪声。",
        "raw_arrays": "每条记录通过source_npz引用原始NPZ；其vendor_impulse_distance/rho、vendor_step_distance/rho、frequency_hz、s11_complex和sweep_name字段保持原样，不重复膨胀写入JSON/CSV。",
        "material_scope": "E:\\20261010截至的全部fdr文件\\2026-6某地进行的1500m不同位置电缆接头浸水试验\\其他信息.txt明确说明此长电缆正常介损比XLPE大很多；拟合只表示该试验的差分峰衰减，不能视为导体α或通用/XLPE损耗。",
    }
    payload = {
        "source_directory": str(SIGNAL_DIR.relative_to(HERE)),
        "baseline_sweep_name": baseline["sweep_name"],
        "wet_sweep_count": len(records),
        "methods": methods,
        "effective_differential_peak_attenuation": amplitude_fit,
        "differential_peak_width_trend": width_trend,
        "records": output_records,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "wet_distance_metrics.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
                         encoding="utf-8")
    csv_path = OUTPUT_DIR / "wet_distance_metrics.csv"
    fieldnames = list(output_records[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_records)

    # Recompute the small plotting payload after removing private arrays from exports.
    plot_records = [analyze_one(baseline, item) for item in wet]
    make_plot(baseline, plot_records, amplitude_fit, width_trend)
    print(f"记录：正常1条 + 浸水{len(records)}条")
    print(f"可辨样本：{amplitude_fit['count']}；差分峰有效衰减：{attenuation}")
    print(f"宽度趋势样本：{width_trend['count']}；斜率：{width_trend['slope_m_per_km']} m/km")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"PNG: {ASSET_DIR / 'wet_distance_characteristics.png'}")


if __name__ == "__main__":
    main()
