"""第三章 CST 等效电路的 Python 复现与 FDR 双算法对照。

本程序只重建 HumanDoc/Coaxial cable with loss.cst 中的集总电路，
不修改 CST 工程和 REF 算法文件。频域模型用 ABCD 级联计算一端口
开路反射系数；距离域分别复现 REF/算法1 和 REF/算法2 的实际计算流程。

正式运行：
    python cst_fdr_reproduction.py

短时冒烟验证：
    python cst_fdr_reproduction.py --smoke --output smoke_output
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass


ROOT = Path(__file__).resolve().parents[3]
HUMAN_DOC = ROOT / "HumanDoc"
REPORT_DOCX = HUMAN_DOC / (
    "20251022-宽频阻抗谱检测技术在输配电电缆状态评估与缺陷定位中的应用研究报告.docx"
)
SCHEMATIC_XML = HUMAN_DOC / "Coaxial cable with loss" / "Model" / "DS" / "schematic.xml"

C0 = 299_792_458.0
ALGORITHM2_LIGHT_SPEED_M_S = 3.0e8
CABLE_BODY_LENGTH_M = 40.0
PLOT_MAX_DISTANCE_M = 1.5 * CABLE_BODY_LENGTH_M
Z_REFERENCE_OHM = 50.0
EPSR_ALGORITHM1 = 2.23
DEFAULT_ALGORITHM2 = {
    "cable_length_m": 95.0,
    "velocity_factor": 0.6715,
    "step_smoothing_points": 5,
    "impulse_smoothing_points": 50,
    "time_points": 10_000,
    "line_offset_m": 0.0,
    "step_offset": 0.0,
    "impulse_normalization_factor": 6.5,
    "test_voltage_v": 10.0,
    "reference_impedance_ohm": 50.0,
    "skip_first_data_point": True,
}


@dataclass(frozen=True)
class Case:
    name: str
    title: str
    reference_image: str | None
    expectation: str
    kind: str = "baseline"
    value: float | None = None


CASES = (
    Case(
        "baseline",
        "基准模型",
        None,
        "约10、20、30 m存在接头响应，约40-42 m出现开路终端响应。",
    ),
    Case(
        "overall_C20pF",
        "整体电容增加：16 pF → 20 pF",
        "image3.png",
        "阶跃/阻抗整体下移，终端反射幅值接近不变，测量全长略增加。",
        "overall_c",
        20e-12,
    ),
    Case(
        "overall_G20k",
        "整体损耗增加：200 kΩ → 20 kΩ",
        "image4.png",
        "阶跃曲线整体呈下降斜率，终端反射减弱，测量全长基本不变。",
        "overall_g",
        1.0 / 20_000.0,
    ),
    Case(
        "segmented_loss",
        "分段损耗：200 kΩ / 20 kΩ / 200 kΩ / 2 kΩ",
        "image5.png",
        "损耗越大的区段下降越陡，终端反射减弱；脉冲中损耗特征不明显。",
        "segmented_g",
    ),
    Case(
        "local_C32pF_15m",
        "局部电容增加：15 m 单元 16 pF → 32 pF",
        "image6.png",
        "15 m 附近出现负极性局部脉冲，终端幅值接近不变，全长略增加。",
        "local_c",
        32e-12,
    ),
    Case(
        "local_G2k_15m",
        "局部损耗增加：15 m 单元 200 kΩ → 2 kΩ",
        "image7.png",
        "15 m 至末端阶跃整体下移，终端反射略降，测量全长基本不变。",
        "local_g",
        1.0 / 2_000.0,
    ),
    Case(
        "local_C4pF_15m",
        "局部电容减小：15 m 单元 16 pF → 4 pF",
        "image8.png",
        "15 m 附近出现正极性局部脉冲，终端幅值接近不变，全长略缩短。",
        "local_c",
        4e-12,
    ),
    Case(
        "local_R10ohm_15m",
        "局部串联电阻增加：15 m 单元 0.01 Ω → 10 Ω",
        "image9.png",
        "15 m 至末端阶跃整体上移，后方接头和终端反射减弱，全长基本不变。",
        "local_r",
        10.0,
    ),
    Case(
        "local_R50ohm_15m",
        "局部串联电阻增加：15 m 单元 0.01 Ω → 50 Ω",
        "image9.png",
        "15 m 至末端阶跃整体上移，后方接头和终端反射减弱，全长基本不变。",
        "local_r",
        50.0,
    ),
)


def progress(label: str, current: int, total: int, *, done: bool = False) -> None:
    """输出不依赖第三方库的终端进度条。"""

    width = 28
    ratio = 1.0 if total <= 0 else min(max(current / total, 0.0), 1.0)
    filled = int(round(width * ratio))
    text = f"\r{label} [{'#' * filled}{'-' * (width - filled)}] {current}/{total}"
    print(text, end="\n" if done else "", flush=True)


def cascade(
    left: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    right: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """级联两个 ABCD 网络。"""

    a, b, c, d = left
    e, f, g, h = right
    return a * e + b * g, a * f + b * h, c * e + d * g, c * f + d * h


def series_element(
    frequency_hz: np.ndarray, resistance_ohm: float, inductance_h: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z = resistance_ohm + 1j * 2.0 * np.pi * frequency_hz * inductance_h
    one = np.ones_like(z, dtype=np.complex128)
    zero = np.zeros_like(z, dtype=np.complex128)
    return one, z, zero, one


def shunt_element(
    frequency_hz: np.ndarray, conductance_s: float, capacitance_f: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = conductance_s + 1j * 2.0 * np.pi * frequency_hz * capacitance_f
    one = np.ones_like(y, dtype=np.complex128)
    zero = np.zeros_like(y, dtype=np.complex128)
    return one, zero, y, one


def standard_cell(
    frequency_hz: np.ndarray,
    resistance_ohm: float,
    inductance_h: float,
    conductance_s: float,
    capacitance_f: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """CST标准单元：输入并联C-G，随后串联R-L。"""

    return cascade(
        shunt_element(frequency_hz, conductance_s, capacitance_f),
        series_element(frequency_hz, resistance_ohm, inductance_h),
    )


def special_joint1(
    frequency_hz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """CST网表中的特殊接头：串联0.1Ω/1μH，末端并联8pF/200kΩ。"""

    return cascade(
        series_element(frequency_hz, 0.1, 1.0e-6),
        shunt_element(frequency_hz, 1.0 / 200_000.0, 8.0e-12),
    )


def lossless_transmission_line(
    frequency_hz: np.ndarray, length_m: float, impedance_ohm: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """CST TL1/TL3/TL4：εr=μr=1、衰减=0。"""

    theta = 2.0 * np.pi * frequency_hz * length_m / C0
    cos_theta = np.cos(theta).astype(np.complex128)
    sin_theta = np.sin(theta)
    return (
        cos_theta,
        1j * impedance_ohm * sin_theta,
        1j * sin_theta / impedance_ohm,
        cos_theta,
    )


def build_frequency_grid(point_count: int) -> np.ndarray:
    if point_count < 10:
        raise ValueError("频率点数必须不小于10")
    return np.linspace(9.0e3, 200.0e6, point_count, dtype=np.float64)


def case_cell_parameters(case: Case, index: int) -> tuple[float, float, float, float]:
    resistance = 0.01
    inductance = 0.3e-6
    capacitance = 16.0e-12
    conductance = 1.0 / 200_000.0

    if case.kind == "overall_c":
        capacitance = float(case.value)
    elif case.kind == "overall_g":
        conductance = float(case.value)
    elif case.kind == "segmented_g":
        if index < 25:
            conductance = 1.0 / 200_000.0
        elif index < 50:
            conductance = 1.0 / 20_000.0
        elif index < 75:
            conductance = 1.0 / 200_000.0
        else:
            conductance = 1.0 / 2_000.0
    elif index == 37 and case.kind == "local_c":
        capacitance = float(case.value)
    elif index == 37 and case.kind == "local_g":
        conductance = float(case.value)
    elif index == 37 and case.kind == "local_r":
        resistance = float(case.value)

    return resistance, inductance, conductance, capacitance


def terminal_shunt_parameters(case: Case) -> tuple[float, float]:
    """末端额外并联支路；对应网表中的第101个16 pF节点。"""

    capacitance = 16.0e-12
    conductance = 1.0 / 200_000.0
    if case.kind == "overall_c":
        capacitance = float(case.value)
    elif case.kind == "overall_g":
        conductance = float(case.value)
    elif case.kind == "segmented_g":
        # 末端属于30-40 m的第四段。
        conductance = 1.0 / 2_000.0
    return conductance, capacitance


def simulate_s11(frequency_hz: np.ndarray, case: Case) -> np.ndarray:
    """按 schematic.xml 的顺序级联并计算一端口开路S11。"""

    identity = np.ones_like(frequency_hz, dtype=np.complex128)
    zero = np.zeros_like(frequency_hz, dtype=np.complex128)
    network = (identity, zero, zero, identity)

    # 外部端口 -> TL1 -> 第一组标准单元。
    network = cascade(network, lossless_transmission_line(frequency_hz, 1.0, 135.0))
    for index in range(100):
        r, l, g, cap = case_cell_parameters(case, index)
        network = cascade(network, standard_cell(frequency_hz, r, l, g, cap))
        if index == 24:
            network = cascade(network, special_joint1(frequency_hz))
        elif index == 49:
            network = cascade(network, lossless_transmission_line(frequency_hz, 0.5, 300.0))
        elif index == 74:
            network = cascade(network, lossless_transmission_line(frequency_hz, 0.5, 300.0))

    terminal_conductance, terminal_capacitance = terminal_shunt_parameters(case)
    network = cascade(
        network,
        shunt_element(frequency_hz, terminal_conductance, terminal_capacitance),
    )

    a, _, c, _ = network
    # ZL -> ∞ 时，Zin=A/C，S11=(A-Zref*C)/(A+Zref*C)。
    with np.errstate(divide="ignore", invalid="ignore"):
        s11 = (a - Z_REFERENCE_OHM * c) / (a + Z_REFERENCE_OHM * c)
    return np.asarray(s11, dtype=np.complex128)


def validate_schematic() -> dict[str, int]:
    """读取CST结构化网表，确认关键块数量与本复现拓扑一致。"""

    if not SCHEMATIC_XML.is_file():
        raise FileNotFoundError(f"找不到CST结构化网表: {SCHEMATIC_XML}")
    root = ET.parse(SCHEMATIC_XML).getroot()
    blocks = list(root.findall("Block"))
    class_counts: dict[str, int] = {}
    for block in blocks:
        block_class = block.attrib.get("Class", "")
        class_counts[block_class] = class_counts.get(block_class, 0) + 1
    series_resistors = sum(
        block.attrib.get("Class") == "RES" and block.findtext("Label", "") == "0.01"
        for block in blocks
    )
    inductors = sum(
        block.attrib.get("Class") == "IND" and block.findtext("Label", "") == "0.3 μ"
        for block in blocks
    )
    terminal_capacitors = sum(
        block.attrib.get("Class") == "CAP" and block.findtext("Label", "") == "16 p"
        for block in blocks
    )
    parallel_resistors = sum(
        block.attrib.get("Class") == "RES" and block.findtext("Label", "") == "200000"
        for block in blocks
    )
    if (
        series_resistors != 100
        or inductors != 100
        or terminal_capacitors != 101
        or parallel_resistors != 102
        or class_counts.get("TL") != 3
        or class_counts.get("TLOPEN") != 1
        or class_counts.get("EXTPORT") != 1
    ):
        raise ValueError(
            "CST网表关键拓扑与当前复现假设不一致: "
            f"0.01Ω电阻={series_resistors}, 0.3μH电感={inductors}, "
            f"16pF电容={terminal_capacitors}, 200kΩ电阻={parallel_resistors}, "
            f"classes={class_counts}"
        )
    return {
        "block_count": len(blocks),
        "standard_series_resistors": series_resistors,
        "standard_inductors": inductors,
        "sixteen_pf_nodes": terminal_capacitors,
        "two_hundred_kohm_shunts": parallel_resistors,
        "tl_count": class_counts.get("TL", 0),
        "tl_open_count": class_counts.get("TLOPEN", 0),
        "external_port_count": class_counts.get("EXTPORT", 0),
    }


def estimate_first_step(frequencies_hz: np.ndarray, quantile: float = 5.0) -> float:
    values = np.sort(np.unique(np.asarray(frequencies_hz, dtype=np.float64)))
    if values.size < 2:
        raise ValueError("频率点过少，无法估计频率步进")
    differences = np.diff(values)
    differences = differences[differences > 0.0]
    if differences.size < 2:
        return float((values[-1] - values[0]) / max(values.size - 1, 1))
    first_step = float(np.percentile(differences, quantile))
    mean_step = float(np.mean(differences))
    if first_step < mean_step / 10.0:
        first_step = mean_step
    return first_step


def algorithm1_ifft(
    frequency_hz: np.ndarray, s11: np.ndarray
) -> dict[str, np.ndarray | float | int]:
    """复现 REF/算法1：复数S11、DC外推、Hann窗、IFFT。"""

    frequency_hz = np.asarray(frequency_hz, dtype=np.float64)
    s11 = np.asarray(s11, dtype=np.complex128)
    if frequency_hz.size > 1:
        frequency_hz = frequency_hz[1:]
        s11 = s11[1:]

    order = np.argsort(frequency_hz)
    frequency_hz = frequency_hz[order]
    s11 = s11[order]
    first_step = estimate_first_step(frequency_hz)
    steps = int(np.floor(frequency_hz[-1] / first_step))
    n = 2 * steps + 1
    f_linear = np.arange(steps + 1, dtype=np.float64) * first_step
    s_linear = np.interp(f_linear, frequency_hz, s11.real) + 1j * np.interp(
        f_linear, frequency_hz, s11.imag
    )

    spectrum = np.zeros(n, dtype=np.complex128)
    for index in range(1, steps + 1):
        spectrum[steps + index] = s_linear[index]
        spectrum[steps - index] = np.conj(s_linear[index])
    if steps > 2:
        dc_abs = 2.0 * abs(spectrum[steps + 1]) - abs(spectrum[steps + 2])
        dc_phase = 2.0 * np.angle(spectrum[steps + 1]) - np.angle(spectrum[steps + 2])
        spectrum[steps] = dc_abs * np.exp(1j * dc_phase)
    else:
        spectrum[steps] = s_linear[0]
    spectrum *= np.hanning(n)

    k = n // 2
    fft_order_spectrum = np.concatenate((spectrum[k:], spectrum[:k]))
    time_domain = np.fft.ifft(fft_order_spectrum)
    dt = 1.0 / (first_step * n)
    time_s = np.arange(n, dtype=np.float64) * dt
    impulse_real = np.real(time_domain)
    impulse_magnitude = np.abs(time_domain)
    step_real = np.real(np.cumsum(time_domain) * dt)
    max_step = float(np.max(np.abs(step_real)))
    if max_step > 0.0:
        impulse_real_ref = impulse_real / max_step
        impulse_magnitude_ref = impulse_magnitude / max_step
        step_ref = step_real / max_step
    else:
        impulse_real_ref = impulse_real
        impulse_magnitude_ref = impulse_magnitude
        step_ref = step_real

    distance_m = time_s * C0 / (2.0 * np.sqrt(EPSR_ALGORITHM1))
    baseline_raw = float(np.mean(step_real[: max(1, int(0.05 * step_real.size))]))
    gamma = np.clip(step_real - baseline_raw, -0.9999, 0.9999)
    with np.errstate(divide="ignore", invalid="ignore"):
        impedance_ohm = Z_REFERENCE_OHM * (1.0 + gamma) / (1.0 - gamma)
    baseline = float(np.median(step_ref[distance_m <= 5.0]))
    display_mask = distance_m <= PLOT_MAX_DISTANCE_M
    display_scale = float(np.max(np.abs(step_ref[display_mask] - baseline)))
    if display_scale <= 0.0:
        display_scale = 1.0
    impulse_display_scale = float(np.max(np.abs(impulse_real_ref[display_mask])))
    if impulse_display_scale <= 0.0:
        impulse_display_scale = 1.0
    return {
        "frequency_hz": frequency_hz,
        "s11": s11,
        "time_s": time_s,
        "distance_m": distance_m,
        "impulse_real_raw": impulse_real,
        "step_raw": step_real,
        "step_scale": max_step,
        "impulse_real_ref": impulse_real_ref,
        "impulse_magnitude_ref": impulse_magnitude_ref,
        "step_ref": step_ref,
        "gamma": gamma,
        "impedance_ohm": impedance_ohm,
        "display_impulse": impulse_real_ref / impulse_display_scale,
        "display_step": (step_ref - baseline) / display_scale,
        "first_step_hz": first_step,
        "spectrum_point_count": n,
    }


def matlab_movmean(values: np.ndarray, window_points: int) -> np.ndarray:
    """复现 REF/算法2 的 MATLAB movmean shrink endpoint 口径。"""

    array = np.asarray(values, dtype=np.float64)
    if window_points <= 1 or array.size == 0:
        return array.copy()
    if window_points % 2:
        left_count = right_count = window_points // 2
    else:
        left_count = window_points // 2
        right_count = window_points // 2 - 1
    positions = np.arange(array.size)
    lower = np.maximum(positions - left_count, 0)
    upper = np.minimum(positions + right_count + 1, array.size)
    if not np.all(np.isfinite(array)):
        result = np.empty(array.size, dtype=np.float64)
        for index in range(array.size):
            result[index] = np.mean(array[lower[index] : upper[index]])
        return result
    cumulative = np.concatenate(([0.0], np.cumsum(array, dtype=np.float64)))
    return (cumulative[upper] - cumulative[lower]) / (upper - lower)


def algorithm2_step_integral(
    frequency_hz: np.ndarray,
    s11_real: np.ndarray,
    time_s: np.ndarray,
    test_voltage_v: float,
    chunk_size: int,
    progress_label: str,
) -> np.ndarray:
    """按 REF/算法2 公式计算余弦积分；分块仅改变内存占用。"""

    frequency_hz = np.asarray(frequency_hz, dtype=np.float64)
    s11_real = np.asarray(s11_real, dtype=np.float64)
    time_s = np.asarray(time_s, dtype=np.float64)
    omega = 2.0 * np.pi * frequency_hz
    cable_loss = (s11_real - float(np.mean(s11_real))) / omega
    omega_left = omega[:-1]
    omega_right = omega[1:]
    trapezoid_loss = (cable_loss[:-1] + cable_loss[1:]) / 2.0
    scale = test_voltage_v * 2.0 / np.pi
    response = np.empty(time_s.size, dtype=np.float64)
    total = math.ceil(time_s.size / chunk_size)
    for chunk_index, start in enumerate(range(0, time_s.size, chunk_size), start=1):
        stop = min(start + chunk_size, time_s.size)
        current_time = time_s[start:stop]
        with np.errstate(divide="ignore", invalid="ignore"):
            cosine_difference = np.cos(omega_left[:, None] * current_time[None, :]) - np.cos(
                omega_right[:, None] * current_time[None, :]
            )
            response[start:stop] = np.sum(
                scale * trapezoid_loss[:, None] * cosine_difference / current_time[None, :],
                axis=0,
            )
        progress(progress_label, chunk_index, total, done=chunk_index == total)
    return response


def algorithm2_fdr(
    frequency_hz: np.ndarray,
    s11: np.ndarray,
    *,
    time_points: int,
    chunk_size: int,
    progress_label: str,
) -> dict[str, np.ndarray | float | int]:
    """复现 REF/算法2 默认配置；算法本体严格只读取S11实部。"""

    parameters = dict(DEFAULT_ALGORITHM2)
    frequency_hz = np.asarray(frequency_hz, dtype=np.float64)
    s11_real = np.asarray(s11, dtype=np.complex128).real.astype(np.float64)
    if parameters["skip_first_data_point"] and frequency_hz.size > 1:
        frequency_hz = frequency_hz[1:]
        s11_real = s11_real[1:]
    parameters["time_points"] = int(time_points)
    end_time_s = min(
        3.0
        * parameters["cable_length_m"]
        / (ALGORITHM2_LIGHT_SPEED_M_S * parameters["velocity_factor"]),
        1.0e-4,
    )
    time_s = np.linspace(1.0e-9, end_time_s, time_points, dtype=np.float64)
    distance_uncorrected_m = (
        time_s * ALGORITHM2_LIGHT_SPEED_M_S * parameters["velocity_factor"] / 2.0
    )
    distance_m = distance_uncorrected_m - parameters["line_offset_m"]
    step_raw = algorithm2_step_integral(
        frequency_hz,
        s11_real,
        time_s,
        parameters["test_voltage_v"],
        chunk_size,
        progress_label,
    )
    step_raw = step_raw + parameters["step_offset"]
    impulse_raw = np.diff(step_raw) / (
        np.diff(time_s)
        * ALGORITHM2_LIGHT_SPEED_M_S
        * parameters["impulse_normalization_factor"]
    )
    step_smoothed = matlab_movmean(step_raw, parameters["step_smoothing_points"])
    impulse_smoothed = matlab_movmean(impulse_raw, parameters["impulse_smoothing_points"])

    distance_impulse_uncorrected = distance_uncorrected_m[1:]
    after_half = np.flatnonzero(
        distance_impulse_uncorrected > parameters["cable_length_m"] / 2.0
    )
    if after_half.size == 0:
        raise ValueError("算法2时间窗口未覆盖半电缆长度")
    end_index = int(after_half[0] + np.argmax(impulse_raw[after_half]))
    detected_end_distance = float(distance_impulse_uncorrected[end_index])
    compensation_start = int(
        np.flatnonzero(distance_impulse_uncorrected > detected_end_distance / 5.0)[0]
    )
    compensation_end = int(
        np.flatnonzero(distance_impulse_uncorrected > 4.0 * detected_end_distance / 5.0)[0]
    )
    impulse_compensation = float(np.mean(impulse_raw[compensation_start : compensation_end + 1]))
    compensated_impulse = impulse_raw - impulse_compensation
    with np.errstate(divide="ignore", invalid="ignore"):
        magnitude_final_raw_db = 20.0 * np.log10(np.abs(compensated_impulse))
        inner_smoothed = matlab_movmean(compensated_impulse, parameters["step_smoothing_points"])
        magnitude_double_raw_db = 20.0 * np.log10(np.abs(inner_smoothed))
        impedance_raw = np.abs(
            (1.0 + step_raw) / (1.0 - step_raw) * parameters["reference_impedance_ohm"]
        )
    magnitude_final_db = matlab_movmean(
        magnitude_final_raw_db, parameters["impulse_smoothing_points"]
    )
    magnitude_double_db = matlab_movmean(
        magnitude_double_raw_db, parameters["impulse_smoothing_points"]
    )
    impedance_smoothed = matlab_movmean(
        impedance_raw, parameters["step_smoothing_points"]
    )

    baseline = float(np.median(step_smoothed[distance_m <= 5.0]))
    display_mask = distance_m <= PLOT_MAX_DISTANCE_M
    display_scale = float(np.max(np.abs(step_smoothed[display_mask] - baseline)))
    if display_scale <= 0.0:
        display_scale = 1.0
    impulse_display_scale = float(
        np.max(np.abs(impulse_smoothed[distance_impulse_uncorrected <= PLOT_MAX_DISTANCE_M]))
    )
    if impulse_display_scale <= 0.0:
        impulse_display_scale = 1.0
    return {
        "frequency_hz": frequency_hz,
        "s11_real": s11_real,
        "time_s": time_s,
        "distance_m": distance_m,
        "distance_impulse_m": distance_impulse_uncorrected,
        "step_raw": step_raw,
        "step_smoothed": step_smoothed,
        "impulse_raw": impulse_raw,
        "impulse_smoothed": impulse_smoothed,
        "impulse_compensation": impulse_compensation,
        "magnitude_final_db": magnitude_final_db,
        "magnitude_double_db": magnitude_double_db,
        "impedance_smoothed_ohm": impedance_smoothed,
        "display_step": (step_smoothed - baseline) / display_scale,
        "display_impulse": impulse_smoothed / impulse_display_scale,
        "detected_end_distance_m": detected_end_distance,
        "time_points": time_points,
    }


def extract_report_images(output_dir: Path) -> dict[str, Path]:
    references: dict[str, Path] = {}
    reference_dir = output_dir / "report_reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    if not REPORT_DOCX.is_file():
        return references
    with zipfile.ZipFile(REPORT_DOCX) as archive:
        for case in CASES:
            if case.reference_image is None:
                continue
            archive_name = f"word/media/{case.reference_image}"
            if archive_name not in archive.namelist():
                continue
            target = reference_dir / case.reference_image
            target.write_bytes(archive.read(archive_name))
            references[case.reference_image] = target
    return references


def save_csv_outputs(
    output_dir: Path,
    s11_output_dir: Path,
    case: Case,
    frequency_hz: np.ndarray,
    s11: np.ndarray,
    a1: dict[str, np.ndarray | float | int],
    a2: dict[str, np.ndarray | float | int],
) -> None:
    pd.DataFrame(
        {
            "Frequency_Hz": frequency_hz,
            "S11_Real": s11.real,
            "S11_Imag": s11.imag,
        }
    ).to_csv(s11_output_dir / f"s11_{case.name}.csv", index=False, float_format="%.15g")
    pd.DataFrame(
        {
            "Time_s": a1["time_s"],
            "Distance_m": a1["distance_m"],
            "Impulse_Real_Raw": a1["impulse_real_raw"],
            "Step_Raw": a1["step_raw"],
            "Impulse_Real_REF": a1["impulse_real_ref"],
            "Impulse_Magnitude_REF": a1["impulse_magnitude_ref"],
            "Step_REF": a1["step_ref"],
            "Display_Impulse": a1["display_impulse"],
            "Display_Step": a1["display_step"],
            "Gamma": a1["gamma"],
            "Impedance_Ohm": a1["impedance_ohm"],
        }
    ).to_csv(output_dir / f"algorithm1_{case.name}.csv", index=False, float_format="%.15g")
    pd.DataFrame(
        {
            "Time_s": a2["time_s"],
            "Distance_m": a2["distance_m"],
            "Step_Raw": a2["step_raw"],
            "Step_Smoothed": a2["step_smoothed"],
            "Display_Step": a2["display_step"],
            "Distance_Impulse_m": np.r_[a2["distance_impulse_m"], np.nan],
            "Impulse_Raw": np.r_[a2["impulse_raw"], np.nan],
            "Impulse_Smoothed": np.r_[a2["impulse_smoothed"], np.nan],
            "Display_Impulse": np.r_[a2["display_impulse"], np.nan],
            "Magnitude_Final_dB": np.r_[a2["magnitude_final_db"], np.nan],
            "Magnitude_Double_dB": np.r_[a2["magnitude_double_db"], np.nan],
            "Impedance_Ohm": a2["impedance_smoothed_ohm"],
        }
    ).to_csv(output_dir / f"algorithm2_{case.name}.csv", index=False, float_format="%.15g")


def finite_array(value: object) -> bool:
    return bool(np.all(np.isfinite(np.asarray(value))))


def peak_in_window(
    distance_m: np.ndarray, values: np.ndarray, low_m: float, high_m: float
) -> tuple[float, float]:
    mask = (distance_m >= low_m) & (distance_m <= high_m) & np.isfinite(values)
    if not np.any(mask):
        return float("nan"), float("nan")
    indices = np.flatnonzero(mask)
    index = int(indices[np.argmax(np.abs(values[indices]))])
    return float(distance_m[index]), float(values[index])


def positive_peak_in_window(
    distance_m: np.ndarray, values: np.ndarray, low_m: float, high_m: float
) -> tuple[float, float]:
    mask = (distance_m >= low_m) & (distance_m <= high_m) & np.isfinite(values)
    if not np.any(mask):
        return float("nan"), float("nan")
    indices = np.flatnonzero(mask)
    index = int(indices[np.argmax(values[indices])])
    return float(distance_m[index]), float(values[index])


def median_in_window(
    distance_m: np.ndarray, values: np.ndarray, low_m: float, high_m: float
) -> float:
    mask = (distance_m >= low_m) & (distance_m <= high_m) & np.isfinite(values)
    return float(np.median(values[mask])) if np.any(mask) else float("nan")


def linear_slope(
    distance_m: np.ndarray, values: np.ndarray, low_m: float, high_m: float
) -> float:
    mask = (distance_m >= low_m) & (distance_m <= high_m) & np.isfinite(values)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    return float(np.polyfit(distance_m[mask], values[mask], 1)[0])


def difference_region_metrics(
    distance_m: np.ndarray,
    delta: np.ndarray,
    regions: tuple[tuple[str, float, float], ...],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name, low_m, high_m in regions:
        mask = (
            (distance_m >= low_m)
            & (distance_m <= high_m)
            & np.isfinite(delta)
        )
        values = delta[mask]
        if values.size == 0:
            result[name] = {
                "rms": float("nan"),
                "peak_abs": float("nan"),
                "mean": float("nan"),
            }
        else:
            result[name] = {
                "rms": float(np.sqrt(np.mean(values**2))),
                "peak_abs": float(np.max(np.abs(values))),
                "mean": float(np.mean(values)),
            }
    return result


def compute_case_metrics(
    case: Case,
    frequency_hz: np.ndarray,
    s11: np.ndarray,
    baseline_s11: np.ndarray,
    a1: dict[str, np.ndarray | float | int],
    baseline_a1: dict[str, np.ndarray | float | int],
    a2: dict[str, np.ndarray | float | int],
    baseline_a2: dict[str, np.ndarray | float | int],
) -> dict[str, object]:
    distance = np.asarray(a1["distance_m"])
    impulse = np.asarray(a1["impulse_real_ref"])
    step = np.asarray(a1["step_ref"])
    baseline_step = np.asarray(baseline_a1["step_ref"])
    baseline_impulse = np.asarray(baseline_a1["impulse_real_ref"])
    terminal_pos, terminal_amp = peak_in_window(
        distance, impulse, 37.0, PLOT_MAX_DISTANCE_M
    )
    joint_positions = []
    for low, high in ((8.0, 14.0), (18.0, 24.0), (27.0, 34.0)):
        pos, amp = peak_in_window(distance, impulse, low, high)
        joint_positions.append({"position_m": pos, "amplitude": amp})

    delta_step = step - baseline_step
    delta_impulse = impulse - baseline_impulse
    local_delta_pos, local_delta_amp = peak_in_window(distance, delta_impulse, 12.0, 18.0)
    baseline_distance = np.asarray(baseline_a1["distance_m"])
    baseline_impulse_values = np.asarray(baseline_a1["impulse_real_ref"])
    baseline_step_values = np.asarray(baseline_a1["step_ref"])
    baseline_terminal_pos, baseline_terminal_amp = positive_peak_in_window(
        baseline_distance, baseline_impulse_values, 37.0, PLOT_MAX_DISTANCE_M
    )
    terminal_pos, terminal_amp = positive_peak_in_window(
        distance, impulse, 37.0, PLOT_MAX_DISTANCE_M
    )
    terminal_amp_ratio = (
        abs(terminal_amp) / abs(baseline_terminal_amp)
        if np.isfinite(terminal_amp) and np.isfinite(baseline_terminal_amp) and baseline_terminal_amp != 0.0
        else float("nan")
    )
    terminal_shift = terminal_pos - baseline_terminal_pos
    baseline_slope = linear_slope(baseline_distance, baseline_step_values, 16.0, 35.0)
    step_delta = median_in_window(distance, delta_step, 16.0, 35.0)
    slope = linear_slope(distance, step, 16.0, 35.0)
    regions = (
        ("pre_0_10m", 0.0, 10.0),
        ("local_12_18m", 12.0, 18.0),
        ("post_18_60m", 18.0, PLOT_MAX_DISTANCE_M),
    )
    locality = {
        "algorithm1_step": {},
        "algorithm1_impulse": {},
    }
    common_scale = float(baseline_a1["step_scale"])
    if common_scale <= 0.0:
        common_scale = 1.0
    common_step_delta = (
        np.asarray(a1["step_raw"]) - np.asarray(baseline_a1["step_raw"])
    ) / common_scale
    common_impulse_delta = (
        np.asarray(a1["impulse_real_raw"])
        - np.asarray(baseline_a1["impulse_real_raw"])
    ) / common_scale
    locality["algorithm1_step"] = difference_region_metrics(
        distance, common_step_delta, regions
    )
    locality["algorithm1_impulse"] = difference_region_metrics(
        distance, common_impulse_delta, regions
    )
    a2_step_distance = np.asarray(a2["distance_m"])
    a2_step_delta = np.asarray(a2["step_smoothed"]) - np.asarray(
        baseline_a2["step_smoothed"]
    )
    a2_impulse_distance = np.asarray(a2["distance_impulse_m"])
    a2_impulse_delta = np.asarray(a2["impulse_smoothed"]) - np.asarray(
        baseline_a2["impulse_smoothed"]
    )
    locality["algorithm2_step"] = difference_region_metrics(
        a2_step_distance, a2_step_delta, regions
    )
    locality["algorithm2_impulse"] = difference_region_metrics(
        a2_impulse_distance, a2_impulse_delta, regions
    )
    checks: dict[str, bool] = {}
    if case.kind == "baseline":
        checks["joint_and_terminal_windows_have_events"] = all(
            np.isfinite(item["position_m"]) for item in joint_positions
        ) and np.isfinite(terminal_pos)
    elif case.kind == "overall_c":
        checks["post_region_step_lower"] = bool(step_delta < 0.0)
        checks["terminal_amplitude_near_unchanged"] = bool(0.75 <= terminal_amp_ratio <= 1.25)
        checks["terminal_later"] = bool(terminal_shift > 0.0)
    elif case.kind == "overall_g":
        checks["post_region_step_lower"] = bool(step_delta < 0.0)
        checks["terminal_amplitude_reduced"] = bool(terminal_amp_ratio < 1.0)
        checks["terminal_position_preserved"] = bool(abs(terminal_shift) <= 1.0)
    elif case.kind == "segmented_g":
        checks["post_region_step_lower"] = bool(step_delta < 0.0)
        checks["terminal_amplitude_reduced"] = bool(terminal_amp_ratio < 1.0)
        checks["slope_is_not_increased"] = bool(slope <= baseline_slope)
    elif case.kind == "local_c":
        expected_negative = float(case.value) > 16.0e-12
        checks["local_pulse_polarity"] = bool(
            local_delta_amp < 0.0 if expected_negative else local_delta_amp > 0.0
        )
        checks["terminal_amplitude_near_unchanged"] = bool(0.75 <= terminal_amp_ratio <= 1.25)
        checks["terminal_position_direction"] = bool(
            terminal_shift > 0.0 if expected_negative else terminal_shift < 0.0
        )
    elif case.kind == "local_g":
        checks["post_region_step_lower"] = bool(step_delta < 0.0)
        checks["terminal_amplitude_reduced"] = bool(terminal_amp_ratio < 1.0)
        checks["terminal_position_preserved"] = bool(abs(terminal_shift) <= 1.0)
    elif case.kind == "local_r":
        checks["post_region_step_higher"] = bool(step_delta > 0.0)
        checks["terminal_amplitude_reduced"] = bool(terminal_amp_ratio < 1.0)
        checks["terminal_position_preserved"] = bool(abs(terminal_shift) <= 1.0)
    observed = (
        f"终端正峰={terminal_pos:.4g} m/{terminal_amp:.4g}，"
        f"相对基准位置变化={terminal_shift:.4g} m、幅值比={terminal_amp_ratio:.4g}；"
        f"16–35 m阶跃中位差={step_delta:.4g}、斜率={slope:.4g}；"
        f"15 m差分脉冲峰={local_delta_pos:.4g} m/{local_delta_amp:.4g}。"
    )
    return {
        "case": case.name,
        "title": case.title,
        "s11_abs_min": float(np.min(np.abs(s11))),
        "s11_abs_max": float(np.max(np.abs(s11))),
        "s11_complex_rms_vs_baseline": float(
            np.sqrt(np.mean(np.abs(s11 - baseline_s11) ** 2))
        ),
        "terminal_peak_position_m": terminal_pos,
        "terminal_peak_amplitude_ref": terminal_amp,
        "terminal_position_shift_m": terminal_shift,
        "terminal_amplitude_ratio_vs_baseline": terminal_amp_ratio,
        "joint_peaks": joint_positions,
        "step_median_0_5m": median_in_window(distance, step, 0.0, 5.0),
        "step_median_16_35m": median_in_window(distance, step, 16.0, 35.0),
        "step_slope_16_35m": slope,
        "delta_step_median_16_35m": step_delta,
        "local_delta_peak_position_m": local_delta_pos,
        "local_delta_peak_amplitude": local_delta_amp,
        "all_finite": all(
            finite_array(value)
            for value in (frequency_hz, s11.real, s11.imag, distance, impulse, step)
        ),
        "report_expectation": case.expectation,
        "observed": observed,
        "qualitative_checks": checks,
        "qualitative_match": bool(all(checks.values())),
        "locality_metrics": locality,
        "comparison_mode": "报告内嵌图定性对照，无CST逐点数值基准",
    }


def plot_board(
    output_dir: Path,
    case: Case,
    frequency_hz: np.ndarray,
    s11: np.ndarray,
    baseline_s11: np.ndarray,
    a1: dict[str, np.ndarray | float | int],
    baseline_a1: dict[str, np.ndarray | float | int],
    a2: dict[str, np.ndarray | float | int],
    baseline_a2: dict[str, np.ndarray | float | int],
    references: dict[str, Path],
) -> None:
    reference = references.get(case.reference_image or "")
    has_reference = reference is not None
    row_count = 4 if has_reference else 3
    # 给报告参考图更多垂直空间，再由 set_box_aspect 保持其原始比例。
    height_ratios = [1.0, 1.0, 1.0, 1.35] if has_reference else [1.0, 1.0, 1.0]
    figure_height = 17.5 if has_reference else 12.0
    fig = plt.figure(figsize=(16.0, figure_height), dpi=200)
    grid = fig.add_gridspec(row_count, 6, height_ratios=height_ratios)
    axes: list[plt.Axes] = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[0, 4:6]),
        fig.add_subplot(grid[1, 0:3]),
        fig.add_subplot(grid[1, 3:6]),
        fig.add_subplot(grid[2, 0:3]),
        fig.add_subplot(grid[2, 3:6]),
    ]

    frequency_mhz = frequency_hz / 1.0e6
    base_s11_real = np.asarray(baseline_s11).real
    case_s11_real = np.asarray(s11).real
    axes[0].plot(frequency_mhz, base_s11_real, label="基准", lw=0.8)
    axes[0].plot(frequency_mhz, case_s11_real, label=case.title, lw=0.8)
    axes[0].set_title("S11 实部")
    axes[0].set_ylabel("S11 Real", fontname="Times New Roman")

    axes[1].plot(frequency_mhz, np.abs(baseline_s11), label="基准", lw=0.8)
    axes[1].plot(frequency_mhz, np.abs(s11), label=case.title, lw=0.8)
    axes[1].set_title("S11 幅值 / 幅频特性")
    axes[1].set_ylabel("|S11|", fontname="Times New Roman")

    axes[2].plot(
        frequency_mhz,
        np.angle(baseline_s11, deg=True),
        label="基准",
        lw=0.8,
    )
    axes[2].plot(
        frequency_mhz,
        np.angle(s11, deg=True),
        label=case.title,
        lw=0.8,
    )
    axes[2].set_title("S11 相位 / 相频特性（包裹相位）")
    axes[2].set_ylabel("Phase (deg)", fontname="Times New Roman")
    axes[2].set_ylim(-180.0, 180.0)

    a1_panels = (
        (axes[3], "display_step", "算法1 阶跃 / 归一化对照"),
        (axes[4], "display_impulse", "算法1 脉冲 / 归一化对照"),
    )
    for axis, key, title in a1_panels:
        distance = np.asarray(a1["distance_m"])
        base_distance = np.asarray(baseline_a1["distance_m"])
        mask = distance <= PLOT_MAX_DISTANCE_M
        base_mask = base_distance <= PLOT_MAX_DISTANCE_M
        axis.plot(
            base_distance[base_mask],
            np.asarray(baseline_a1[key])[base_mask],
            label="基准",
            lw=0.8,
        )
        axis.plot(
            distance[mask], np.asarray(a1[key])[mask], label=case.title, lw=0.8
        )
        axis.set_title(title)

    a2_panels = (
        (axes[5], "display_step", "算法2 阶跃 / 归一化对照", "distance_m"),
        (axes[6], "display_impulse", "算法2 脉冲 / 归一化对照", "distance_impulse_m"),
    )
    for axis, key, title, xkey in a2_panels:
        distance = np.asarray(a2[xkey])
        base_distance = np.asarray(baseline_a2[xkey])
        mask = distance <= PLOT_MAX_DISTANCE_M
        base_mask = base_distance <= PLOT_MAX_DISTANCE_M
        axis.plot(
            base_distance[base_mask],
            np.asarray(baseline_a2[key])[base_mask],
            label="基准",
            lw=0.8,
        )
        axis.plot(
            distance[mask], np.asarray(a2[key])[mask], label=case.title, lw=0.8
        )
        axis.set_title(title)

    if has_reference:
        reference_axis = fig.add_subplot(grid[3, 0:6])
        reference_image_data = mpimg.imread(reference)
        reference_axis.imshow(reference_image_data, aspect="equal")
        reference_axis.set_box_aspect(
            reference_image_data.shape[0] / reference_image_data.shape[1]
        )
        reference_axis.set_title("报告对应 CST 图")
        reference_axis.set_anchor("C")
        reference_axis.axis("off")
        axes.append(reference_axis)

    for axis in axes[:7]:
        axis.set_xlabel(
            "Frequency (MHz)" if axis in axes[:3] else "Distance (m)",
            fontname="Times New Roman",
        )
        if axis not in axes[:3]:
            axis.set_ylabel("Normalized amplitude", fontname="Times New Roman")
            axis.set_xlim(0.0, PLOT_MAX_DISTANCE_M)
        axis.legend(fontsize=7)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.7)
        axis.tick_params(direction="in", top=True, right=True, width=0.7)
        axis.grid(True, which="major", linewidth=0.3, alpha=0.25)
    fig.suptitle(
        f"第三章 CST复现：{case.title}（距离轴0–{PLOT_MAX_DISTANCE_M:g} m）",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    fig.savefig(
        output_dir / f"comparison_board_{case.name}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_locality_board(
    output_dir: Path,
    case: Case,
    a1: dict[str, np.ndarray | float | int],
    baseline_a1: dict[str, np.ndarray | float | int],
    a2: dict[str, np.ndarray | float | int],
    baseline_a2: dict[str, np.ndarray | float | int],
) -> None:
    """绘制基准-工况差分，区分缺陷邻域和其余距离区间。"""

    common_scale = float(baseline_a1["step_scale"])
    if common_scale <= 0.0:
        common_scale = 1.0
    series = (
        (
            np.asarray(a1["distance_m"]),
            (
                np.asarray(a1["step_raw"])
                - np.asarray(baseline_a1["step_raw"])
            )
            / common_scale,
            "算法1 阶跃差分（共用基准尺度）",
        ),
        (
            np.asarray(a1["distance_m"]),
            (
                np.asarray(a1["impulse_real_raw"])
                - np.asarray(baseline_a1["impulse_real_raw"])
            )
            / common_scale,
            "算法1 脉冲差分（共用基准尺度）",
        ),
        (
            np.asarray(a2["distance_m"]),
            np.asarray(a2["step_smoothed"])
            - np.asarray(baseline_a2["step_smoothed"]),
            "算法2 阶跃差分（默认配置）",
        ),
        (
            np.asarray(a2["distance_impulse_m"]),
            np.asarray(a2["impulse_smoothed"])
            - np.asarray(baseline_a2["impulse_smoothed"]),
            "算法2 脉冲差分（默认配置）",
        ),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=200)
    for ax, (distance, delta, title) in zip(axes.flat, series):
        mask = (
            (distance >= 0.0)
            & (distance <= PLOT_MAX_DISTANCE_M)
            & np.isfinite(delta)
        )
        ax.plot(distance[mask], delta[mask], color="tab:red", lw=0.8)
        ax.axhline(0.0, color="black", lw=0.6)
        ax.axvline(15.0, color="tab:blue", linestyle="--", lw=0.8, label="15 m")
        ax.axvspan(12.0, 18.0, color="tab:orange", alpha=0.12, label="局部观察区")
        ax.set_title(title)
        ax.set_xlabel("Distance (m)", fontname="Times New Roman")
        ax.set_ylabel("Variant - baseline", fontname="Times New Roman")
        ax.set_xlim(0.0, PLOT_MAX_DISTANCE_M)
        ax.legend(fontsize=7)
        ax.grid(True, which="major", linewidth=0.3, alpha=0.25)
        ax.tick_params(direction="in", top=True, right=True, width=0.7)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.7)
    fig.suptitle(
        f"局部性诊断：{case.title}\n橙色区域为12–18 m，差分越集中越接近局部响应",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(
        output_dir / f"locality_board_{case.name}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def write_report(
    report_path: Path,
    metrics: list[dict[str, object]],
    runtime_s: float,
    point_count: int,
    time_points: int,
) -> None:
    lines = [
        "# 第三章 CST 等效电路 Python 复现报告",
        "",
        f"- 运行时间：{runtime_s:.3f} s",
        f"- 频率网格：9 kHz–200 MHz，{point_count}点",
        f"- 算法2时间点：{time_points}",
        "- 电路模型：4组×25个标准RLCG单元，标准单元距离映射为0.4 m",
        "- 对比口径：报告内嵌图定性对照；当前没有CST ASCII S11，因此不报告逐点误差或RMSE",
        f"- 图像横轴：0–{PLOT_MAX_DISTANCE_M:g} m（{CABLE_BODY_LENGTH_M:g} m电缆本体的1.5倍）",
        f"- 显示归一化：阶跃按0–5 m基线和0–{PLOT_MAX_DISTANCE_M:g} m尺度；脉冲按自身0–{PLOT_MAX_DISTANCE_M:g} m最大绝对值；定量检查使用REF原始输出",
        "",
        "## 模型和算法边界",
        "",
        "标准单元按 schematic.xml 的电气连接顺序实现为输入并联 C-G、随后串联 R-L。保留 TL1（1 m、135 Ω）、TL3/TL4（0.5 m、300 Ω）、特殊接头和末端开路。",
        "",
        "算法1保留复数 S11，使用自动 DC 外推、共轭双边谱、Hann窗和 IFFT。算法2严格只使用 S11 实部，使用 REF 默认参数；其 NumPy 分块只用于内存控制，公式未改变。",
        "",
        "## 工况对照",
        "",
        "| 工况 | 终端峰位置(m) | 终端峰幅值 | 15m局部差分峰位置(m) | 15m局部差分峰幅值 | 阶跃16–35m斜率 | 检查 | 是否吻合 |",
        "|---|---:|---:|---:|---:|---:|---|:---:|",
    ]
    for item in metrics:
        checks = "；".join(
            f"{name}={'是' if value else '否'}"
            for name, value in item["qualitative_checks"].items()
        )
        lines.append(
            "| {case} | {terminal_peak_position_m:.4g} | {terminal_peak_amplitude_ref:.4g} | "
            "{local_delta_peak_position_m:.4g} | {local_delta_peak_amplitude:.4g} | "
                "{step_slope_16_35m:.4g} | {checks} | {match} |".format(
                checks=checks,
                match="是" if item["qualitative_match"] else "否",
                **item,
            )
        )
    lines.extend(
        [
            "",
            "## 逐项观察",
            "",
        ]
    )
    for item in metrics:
        lines.extend(
            [
                f"### {item['title']}",
                "",
                f"- 观察：{item['observed']}",
                f"- 报告预期：{item['report_expectation']}",
                f"- 局部性RMS（前0–10 m / 局部12–18 m / 后18–{PLOT_MAX_DISTANCE_M:g} m；算法1共用基准尺度、算法2使用默认原始尺度）："
                f"算法1阶跃={item['locality_metrics']['algorithm1_step']['pre_0_10m']['rms']:.4g}/"
                f"{item['locality_metrics']['algorithm1_step']['local_12_18m']['rms']:.4g}/"
                f"{item['locality_metrics']['algorithm1_step']['post_18_60m']['rms']:.4g}；"
                f"算法1脉冲={item['locality_metrics']['algorithm1_impulse']['pre_0_10m']['rms']:.4g}/"
                f"{item['locality_metrics']['algorithm1_impulse']['local_12_18m']['rms']:.4g}/"
                f"{item['locality_metrics']['algorithm1_impulse']['post_18_60m']['rms']:.4g}；"
                f"算法2阶跃={item['locality_metrics']['algorithm2_step']['pre_0_10m']['rms']:.4g}/"
                f"{item['locality_metrics']['algorithm2_step']['local_12_18m']['rms']:.4g}/"
                f"{item['locality_metrics']['algorithm2_step']['post_18_60m']['rms']:.4g}；"
                f"算法2脉冲={item['locality_metrics']['algorithm2_impulse']['pre_0_10m']['rms']:.4g}/"
                f"{item['locality_metrics']['algorithm2_impulse']['local_12_18m']['rms']:.4g}/"
                f"{item['locality_metrics']['algorithm2_impulse']['post_18_60m']['rms']:.4g}",
                f"- 方向性检查：{'通过' if item['qualitative_match'] else '未全部通过'}（仅表示与报告文字/图像的定性一致性）",
                "",
            ]
        )
    lines.extend(["## 图像引用", ""])
    for item in metrics:
        case_name = item["case"]
        lines.extend(
            [
                f"### {item['title']}",
                "",
                f"![完整复现对照板](assets/comparison_board_{case_name}.png)",
                f"![局部性差分诊断](assets/locality_board_{case_name}.png)",
                "",
            ]
        )
    lines.extend(
        [
            "## 结果文件",
            "",
            "S11 文件位于 `s11_output/s11_*.csv`；算法结果位于 `output/algorithm1_*.csv` 和 `output/algorithm2_*.csv`。",
            "",
            "本报告中的“吻合”只表示响应方向、局部极性、衰减区域和终端行为与报告文字/图形相符；报告图片没有原始数值，不能据此计算 CST 逐点误差。",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def json_default(value: object) -> object:
    """将NumPy标量转换为标准JSON标量。"""

    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"无法写入JSON的类型: {type(value).__name__}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="第三章 CST 等效电路 Python 复现")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="算法结果和summary输出目录，默认是本试验目录下的output",
    )
    parser.add_argument(
        "--assets",
        type=Path,
        default=Path(__file__).resolve().parent / "assets",
        help="对照图和报告参考图目录，默认是本试验目录下的assets",
    )
    parser.add_argument(
        "--s11-output",
        type=Path,
        default=Path(__file__).resolve().parent / "s11_output",
        help="S11 CSV目录，默认是本试验目录下的s11_output",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(__file__).resolve().parent / "comparison_report.md",
        help="Markdown报告路径，默认是本试验目录下的comparison_report.md",
    )
    parser.add_argument("--points", type=int, default=10_000, help="频率点数，正式默认10000")
    parser.add_argument(
        "--algorithm2-time-points",
        type=int,
        default=10_000,
        help="算法2时间点，正式默认10000",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=128,
        help="算法2分块时间点数，仅影响内存和进度，不改变公式",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="短时冒烟模式：默认频率点和算法2时间点均改为300",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_size < 1:
        raise ValueError("--chunk-size必须大于0")
    point_count = 300 if args.smoke else args.points
    time_points = 300 if args.smoke else args.algorithm2_time_points
    output_dir = args.output.resolve()
    assets_dir = args.assets.resolve()
    s11_output_dir = args.s11_output.resolve()
    report_path = args.report.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    s11_output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "SimHei",
            "font.sans-serif": ["SimHei", "Microsoft YaHei", "Microsoft YaHei UI"],
            "axes.unicode_minus": False,
            "figure.dpi": 200,
        }
    )

    started = time.perf_counter()
    frequency_hz = build_frequency_grid(point_count)
    references = extract_report_images(assets_dir)
    print(f"算法结果目录: {output_dir}")
    print(f"对照图目录: {assets_dir}")
    print(f"S11 CSV目录: {s11_output_dir}")
    print(f"频率网格: {frequency_hz[0]:.6g} Hz - {frequency_hz[-1]:.6g} Hz, {point_count}点")
    print(f"算法2时间点: {time_points}，分块大小: {args.chunk_size}")
    schematic_summary = validate_schematic()
    print(f"CST网表: {SCHEMATIC_XML}")
    print(
        "电路自检通过: "
        f"{schematic_summary['standard_series_resistors']}个0.01Ω电阻、"
        f"{schematic_summary['standard_inductors']}个0.3μH电感、"
        f"{schematic_summary['tl_count']}个TL、1个开路块；"
        "15 m局部缺陷=第2组第13单元（全局索引37）"
    )

    results: dict[str, dict[str, object]] = {}
    baseline_s11: np.ndarray | None = None
    baseline_a1: dict[str, object] | None = None
    baseline_a2: dict[str, object] | None = None
    metrics: list[dict[str, object]] = []
    for case_index, case in enumerate(CASES, start=1):
        print(f"\n[{case_index}/{len(CASES)}] {case.title}")
        s11 = simulate_s11(frequency_hz, case)
        a1 = algorithm1_ifft(frequency_hz, s11)
        a2 = algorithm2_fdr(
            frequency_hz,
            s11,
            time_points=time_points,
            chunk_size=args.chunk_size,
            progress_label=f"算法2 {case.name}",
        )
        if baseline_s11 is None:
            baseline_s11 = s11
            baseline_a1 = a1
            baseline_a2 = a2
        assert baseline_a1 is not None and baseline_a2 is not None and baseline_s11 is not None
        save_csv_outputs(
            output_dir,
            s11_output_dir,
            case,
            frequency_hz,
            s11,
            a1,
            a2,
        )
        plot_board(
            assets_dir,
            case,
            frequency_hz,
            s11,
            baseline_s11,
            a1,
            baseline_a1,
            a2,
            baseline_a2,
            references,
        )
        plot_locality_board(assets_dir, case, a1, baseline_a1, a2, baseline_a2)
        case_metrics = compute_case_metrics(
            case,
            frequency_hz,
            s11,
            baseline_s11,
            a1,
            baseline_a1,
            a2,
            baseline_a2,
        )
        metrics.append(case_metrics)
        results[case.name] = {"case": case_metrics}
        print(
            f"  S11|范围=[{case_metrics['s11_abs_min']:.4g}, {case_metrics['s11_abs_max']:.4g}], "
            f"终端峰={case_metrics['terminal_peak_position_m']:.4g} m"
        )

    runtime_s = time.perf_counter() - started
    summary = {
        "runtime_s": runtime_s,
        "frequency_start_hz": float(frequency_hz[0]),
        "frequency_stop_hz": float(frequency_hz[-1]),
        "frequency_points": point_count,
        "algorithm1_epsr": EPSR_ALGORITHM1,
        "algorithm2": {**DEFAULT_ALGORITHM2, "time_points": time_points},
        "algorithm2_light_speed_m_s": ALGORITHM2_LIGHT_SPEED_M_S,
        "plot_max_distance_m": PLOT_MAX_DISTANCE_M,
        "output_paths": {
            "algorithm_results": str(output_dir),
            "assets": str(assets_dir),
            "s11_output": str(s11_output_dir),
            "report": str(report_path),
        },
        "schematic_summary": schematic_summary,
        "model_distance_mapping": {
            "standard_cell_length_m": 0.4,
            "standard_cell_count": 100,
            "local_defect_cell_index_zero_based": 37,
            "tl1_length_m": 1.0,
            "tl3_length_m": 0.5,
            "tl4_length_m": 0.5,
        },
        "cases": results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=json_default,
        ),
        encoding="utf-8",
    )
    write_report(report_path, metrics, runtime_s, point_count, time_points)
    print(f"\n完成：{len(CASES)}个工况，耗时 {runtime_s:.3f} s")
    print(f"Markdown报告: {report_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断。")
        sys.exit(130)
