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
import contextlib
import importlib.util
import io
import json
import math
import sys
import time
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

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


ROOT = Path(__file__).resolve().parents[4]
HUMAN_DOC = ROOT / "HumanDoc"
REPORT_DOCX = HUMAN_DOC / (
    "20251022-宽频阻抗谱检测技术在输配电电缆状态评估与缺陷定位中的应用研究报告.docx"
)
SCHEMATIC_XML = HUMAN_DOC / "Coaxial cable with loss" / "Model" / "DS" / "schematic.xml"

C0 = 299_792_458.0
ALGORITHM2_LIGHT_SPEED_M_S = 3.0e8
CABLE_BODY_LENGTH_M = 40.0
CABLE_SEGMENT_LENGTH_M = 10.0
BASE_CELL_LENGTH_M = 0.4
DEFAULT_CELL_LENGTH_M = 0.1
FORMAL_CELL_LENGTHS = (0.4, 0.2, 0.1, 0.05)
LOCAL_DEFECT_START_M = 14.8
LOCAL_DEFECT_END_M = 15.2
RLGC_REFERENCE_HZ = 100.0e6
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


@dataclass(frozen=True)
class CircuitVariant:
    """一个可复现实验模型；V0参数只在此处按模型类型组织。"""

    name: str
    title: str
    kind: str
    cell_length_m: float | None
    include_special_joint: bool = True
    include_tl300: bool = True
    include_tl1: bool = True


MODEL_VARIANTS = {
    "fixed_ladder_0p4m": CircuitVariant(
        "fixed_ladder_0p4m", "固定RLGC梯形：0.4 m单元", "fixed_ladder", 0.4
    ),
    "fixed_ladder_0p2m": CircuitVariant(
        "fixed_ladder_0p2m", "固定RLGC梯形：0.2 m单元", "fixed_ladder", 0.2
    ),
    "fixed_ladder_0p1m": CircuitVariant(
        "fixed_ladder_0p1m", "固定RLGC梯形：0.1 m单元", "fixed_ladder", 0.1
    ),
    "fixed_ladder_0p05m": CircuitVariant(
        "fixed_ladder_0p05m", "固定RLGC梯形：0.05 m单元", "fixed_ladder", 0.05
    ),
    "fixed_continuous": CircuitVariant(
        "fixed_continuous", "固定RLGC连续线参考", "fixed_continuous", None
    ),
    "dg_loss_0p1m": CircuitVariant(
        "dg_loss_0p1m", "频变损耗对照：0.1 m单元", "dg_loss", 0.1
    ),
    "ablation_no_special": CircuitVariant(
        "ablation_no_special", "消融：去除特殊RLCG接头", "fixed_ladder", 0.1,
        include_special_joint=False,
    ),
    "ablation_no_tl300": CircuitVariant(
        "ablation_no_tl300", "消融：去除300 Ω TL", "fixed_ladder", 0.1,
        include_tl300=False,
    ),
    "ablation_no_internal": CircuitVariant(
        "ablation_no_internal", "消融：去除全部内部接头", "fixed_ladder", 0.1,
        include_special_joint=False, include_tl300=False,
    ),
    "ablation_body_only": CircuitVariant(
        "ablation_body_only", "消融：仅保留电缆主体", "fixed_ladder", 0.1,
        include_special_joint=False, include_tl300=False, include_tl1=False,
    ),
}


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


def validate_cell_length(cell_length_m: float) -> tuple[int, int]:
    """Return total cells and cells per 10 m segment for a formal grid."""

    cell_length_m = float(cell_length_m)
    if cell_length_m <= 0.0 or not np.isfinite(cell_length_m):
        raise ValueError("单元长度必须是正的有限数值")
    cells_per_segment = int(round(CABLE_SEGMENT_LENGTH_M / cell_length_m))
    total_cells = int(round(CABLE_BODY_LENGTH_M / cell_length_m))
    if not np.isclose(cells_per_segment * cell_length_m, CABLE_SEGMENT_LENGTH_M):
        raise ValueError("单元长度必须能够精确分割10 m段长")
    if not np.isclose(total_cells * cell_length_m, CABLE_BODY_LENGTH_M):
        raise ValueError("单元长度必须能够精确分割40 m主体")
    return total_cells, cells_per_segment


def case_cell_parameters(
    case: Case,
    start_m: float,
    end_m: float,
    cell_length_m: float,
) -> tuple[float, float, float, float]:
    """Return one fixed-RLGC cell's values while preserving V0 semantics."""

    scale = float(cell_length_m) / BASE_CELL_LENGTH_M
    resistance = 0.01 * scale
    inductance = 0.3e-6 * scale
    capacitance = 16.0e-12 * scale
    conductance = (1.0 / 200_000.0) * scale
    center_m = 0.5 * (float(start_m) + float(end_m))

    if case.kind == "overall_c":
        capacitance = float(case.value) * scale
    elif case.kind == "overall_g":
        conductance = float(case.value) * scale
    elif case.kind == "segmented_g":
        if center_m < 10.0:
            conductance = (1.0 / 200_000.0) * scale
        elif center_m < 20.0:
            conductance = (1.0 / 20_000.0) * scale
        elif center_m < 30.0:
            conductance = (1.0 / 200_000.0) * scale
        else:
            conductance = (1.0 / 2_000.0) * scale
    elif (
        float(start_m) < LOCAL_DEFECT_END_M
        and float(end_m) > LOCAL_DEFECT_START_M
    ):
        if case.kind == "local_c":
            capacitance = float(case.value) * scale
        elif case.kind == "local_g":
            conductance = float(case.value) * scale
        elif case.kind == "local_r":
            resistance = float(case.value) * scale

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


def network_to_s11(
    network: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
) -> np.ndarray:
    """Convert an ABCD network ending in an open load to 50 Ω S11."""

    a, _, c, _ = network
    with np.errstate(divide="ignore", invalid="ignore"):
        s11 = (a - Z_REFERENCE_OHM * c) / (a + Z_REFERENCE_OHM * c)
    return np.asarray(s11, dtype=np.complex128)


def continuous_fixed_line(
    frequency_hz: np.ndarray,
    length_m: float,
    resistance_per_m: float,
    inductance_per_m: float,
    conductance_per_m: float,
    capacitance_per_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ABCD of a continuous line using fixed V0 primary parameters."""

    omega = 2.0 * np.pi * frequency_hz
    series = resistance_per_m + 1j * omega * inductance_per_m
    shunt = conductance_per_m + 1j * omega * capacitance_per_m
    gamma = np.sqrt(series * shunt)
    characteristic = np.sqrt(series / shunt)
    gl = gamma * float(length_m)
    sinh_gl = np.sinh(gl)
    return (
        np.cosh(gl),
        characteristic * sinh_gl,
        sinh_gl / characteristic,
        np.cosh(gl),
    )


def dg_loss_cell(
    frequency_hz: np.ndarray,
    resistance_ohm: float,
    inductance_h: float,
    conductance_s: float,
    capacitance_f: float,
    cell_length_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A frequency-dependent loss comparison anchored to V0 at 100 MHz."""

    scale = float(cell_length_m) / BASE_CELL_LENGTH_M
    resistance_per_m_ref = resistance_ohm / float(cell_length_m)
    inductance_per_m_ref = inductance_h / float(cell_length_m)
    conductance_per_m_ref = conductance_s / float(cell_length_m)
    capacitance_per_m_ref = capacitance_f / float(cell_length_m)
    omega = 2.0 * np.pi * frequency_hz
    omega_ref = 2.0 * np.pi * RLGC_REFERENCE_HZ
    ratio = np.sqrt(np.maximum(frequency_hz, 1.0) / RLGC_REFERENCE_HZ)
    resistance_per_m = resistance_per_m_ref * ratio
    inductance_external = inductance_per_m_ref - resistance_per_m_ref / omega_ref
    inductance_per_m = inductance_external + resistance_per_m / omega
    conductance_per_m = conductance_per_m_ref * (
        np.maximum(frequency_hz, 1.0) / RLGC_REFERENCE_HZ
    )
    return standard_cell(
        frequency_hz,
        resistance_per_m * float(cell_length_m),
        inductance_per_m * float(cell_length_m),
        conductance_per_m * float(cell_length_m),
        np.full_like(frequency_hz, capacitance_per_m_ref * float(cell_length_m)),
    )


def terminal_shunt_element(
    frequency_hz: np.ndarray,
    case: Case,
    frequency_dependent_loss: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    conductance, capacitance = terminal_shunt_parameters(case)
    if frequency_dependent_loss:
        conductance = conductance * (
            np.maximum(frequency_hz, 1.0) / RLGC_REFERENCE_HZ
        )
    return shunt_element(frequency_hz, conductance, capacitance)


def simulate_fixed_ladder(
    frequency_hz: np.ndarray,
    case: Case,
    cell_length_m: float,
    *,
    frequency_dependent_loss: bool = False,
    include_special_joint: bool = True,
    include_tl300: bool = True,
    include_tl1: bool = True,
) -> np.ndarray:
    """Cascade a scaled fixed-RLGC ladder with the selected CST structures."""

    total_cells, cells_per_segment = validate_cell_length(cell_length_m)
    identity = np.ones_like(frequency_hz, dtype=np.complex128)
    zero = np.zeros_like(frequency_hz, dtype=np.complex128)
    network = (identity, zero, zero, identity)
    if include_tl1:
        network = cascade(
            network, lossless_transmission_line(frequency_hz, 1.0, 135.0)
        )

    for index in range(total_cells):
        start_m = index * float(cell_length_m)
        end_m = (index + 1) * float(cell_length_m)
        r, l, g, cap = case_cell_parameters(
            case, start_m, end_m, float(cell_length_m)
        )
        if frequency_dependent_loss:
            element = dg_loss_cell(
                frequency_hz, r, l, g, cap, float(cell_length_m)
            )
        else:
            element = standard_cell(frequency_hz, r, l, g, cap)
        network = cascade(network, element)
        if index + 1 == cells_per_segment and include_special_joint:
            network = cascade(network, special_joint1(frequency_hz))
        elif index + 1 in (2 * cells_per_segment, 3 * cells_per_segment):
            if include_tl300:
                network = cascade(
                    network, lossless_transmission_line(frequency_hz, 0.5, 300.0)
                )

    network = cascade(
        network,
        terminal_shunt_element(
            frequency_hz, case, frequency_dependent_loss=frequency_dependent_loss
        ),
    )
    return network_to_s11(network)


def simulate_continuous_fixed(
    frequency_hz: np.ndarray,
    case: Case,
    *,
    include_special_joint: bool = True,
    include_tl300: bool = True,
    include_tl1: bool = True,
) -> np.ndarray:
    """Continuous fixed-RLGC reference with the same CST boundaries."""

    identity = np.ones_like(frequency_hz, dtype=np.complex128)
    zero = np.zeros_like(frequency_hz, dtype=np.complex128)
    network = (identity, zero, zero, identity)
    if include_tl1:
        network = cascade(
            network, lossless_transmission_line(frequency_hz, 1.0, 135.0)
        )

    boundaries = [0.0, 10.0, 20.0, 30.0, 40.0]
    if case.kind in {"local_c", "local_g", "local_r"}:
        boundaries.extend([LOCAL_DEFECT_START_M, LOCAL_DEFECT_END_M])
    boundaries = sorted(set(boundaries))
    for start_m, end_m in zip(boundaries[:-1], boundaries[1:]):
        r, l, g, cap = case_cell_parameters(
            case, start_m, end_m, BASE_CELL_LENGTH_M
        )
        network = cascade(
            network,
            continuous_fixed_line(
                frequency_hz,
                end_m - start_m,
                r / BASE_CELL_LENGTH_M,
                l / BASE_CELL_LENGTH_M,
                g / BASE_CELL_LENGTH_M,
                cap / BASE_CELL_LENGTH_M,
            ),
        )
        if np.isclose(end_m, 10.0) and include_special_joint:
            network = cascade(network, special_joint1(frequency_hz))
        elif np.isclose(end_m, 20.0) or np.isclose(end_m, 30.0):
            if include_tl300:
                network = cascade(
                    network, lossless_transmission_line(frequency_hz, 0.5, 300.0)
                )

    network = cascade(network, terminal_shunt_element(frequency_hz, case))
    return network_to_s11(network)


def simulate_s11(
    frequency_hz: np.ndarray,
    case: Case,
    model_variant: str = "fixed_ladder_0p1m",
) -> np.ndarray:
    """Simulate one case under a named V1 model variant."""

    if model_variant not in MODEL_VARIANTS:
        raise ValueError(f"未知模型变体: {model_variant}")
    variant = MODEL_VARIANTS[model_variant]
    if variant.kind == "fixed_continuous":
        return simulate_continuous_fixed(frequency_hz, case)
    if variant.kind == "dg_loss":
        return simulate_fixed_ladder(
            frequency_hz,
            case,
            float(variant.cell_length_m),
            frequency_dependent_loss=True,
        )
    return simulate_fixed_ladder(
        frequency_hz,
        case,
        float(variant.cell_length_m),
        include_special_joint=variant.include_special_joint,
        include_tl300=variant.include_tl300,
        include_tl1=variant.include_tl1,
    )


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


def algorithm2_ref_parity_check(
    frequency_hz: np.ndarray,
    s11: np.ndarray,
    *,
    time_points: int,
) -> dict[str, object]:
    """在小型输入上将本地实现与 REF/算法2 核心逐数组比较。"""

    ref_path = ROOT / "REF" / "算法2" / "fdr_response_core.py"
    if not ref_path.is_file():
        return {
            "available": False,
            "reason": f"找不到 REF 核心文件: {ref_path}",
        }
    check_frequency_count = min(int(np.asarray(frequency_hz).size), 257)
    check_time_points = min(int(time_points), 256)
    if check_frequency_count < 3 or check_time_points < 2:
        return {
            "available": False,
            "reason": "用于对照的频率点或时间点不足",
        }
    spec = importlib.util.spec_from_file_location("ref_algorithm2_core", ref_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 REF 算法2核心: {ref_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    frequency = np.asarray(frequency_hz, dtype=np.float64)[:check_frequency_count]
    s11_array = np.asarray(s11, dtype=np.complex128)[:check_frequency_count]
    ours_stdout = io.StringIO()
    with contextlib.redirect_stdout(ours_stdout):
        ours = algorithm2_fdr(
            frequency,
            s11_array,
            time_points=check_time_points,
            chunk_size=64,
            progress_label="",
        )
    ref_data = module.FrequencyData(
        Path("V1 internal parity"),
        np.ascontiguousarray(frequency[1:], dtype=np.float64),
        np.ascontiguousarray(s11_array.real[1:], dtype=np.float64),
    )
    ref_parameters = module.AnalysisParameters(time_points=check_time_points)
    reference = module.compute_response(ref_data, ref_parameters)
    comparisons = {
        "step_raw_max_abs": float(np.max(np.abs(ours["step_raw"] - reference.step_raw))),
        "step_smoothed_max_abs": float(
            np.max(np.abs(ours["step_smoothed"] - reference.step_smoothed))
        ),
        "impulse_raw_max_abs": float(
            np.max(np.abs(ours["impulse_raw"] - reference.impulse_raw))
        ),
        "impulse_smoothed_max_abs": float(
            np.max(np.abs(ours["impulse_smoothed"] - reference.impulse_smoothed))
        ),
    }
    return {
        "available": True,
        "frequency_points_after_skip": int(frequency[1:].size),
        "time_points": check_time_points,
        "max_abs_error": max(comparisons.values()),
        **comparisons,
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


def save_variant_outputs(
    output_dir: Path,
    s11_output_dir: Path,
    variant_name: str,
    case: Case,
    frequency_hz: np.ndarray,
    s11: np.ndarray,
    a1: dict[str, np.ndarray | float | int],
    a2: dict[str, np.ndarray | float | int],
) -> None:
    """Save one model/case without mixing S11 sources or padding arrays."""

    algorithm_dir = output_dir / variant_name
    s11_dir = s11_output_dir / variant_name
    algorithm_dir.mkdir(parents=True, exist_ok=True)
    s11_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "Frequency_Hz": frequency_hz,
            "S11_Real": s11.real,
            "S11_Imag": s11.imag,
        }
    ).to_csv(
        s11_dir / f"s11_{case.name}.csv",
        index=False,
        float_format="%.15g",
    )
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
    ).to_csv(
        algorithm_dir / f"algorithm1_{case.name}.csv",
        index=False,
        float_format="%.15g",
    )
    pd.DataFrame(
        {
            "Time_s": a2["time_s"],
            "Distance_m": a2["distance_m"],
            "Step_Raw": a2["step_raw"],
            "Step_Smoothed": a2["step_smoothed"],
            "Display_Step": a2["display_step"],
            "Impedance_Ohm": a2["impedance_smoothed_ohm"],
        }
    ).to_csv(
        algorithm_dir / f"algorithm2_step_{case.name}.csv",
        index=False,
        float_format="%.15g",
    )
    pd.DataFrame(
        {
            "Time_s": np.asarray(a2["time_s"])[1:],
            "Distance_m": a2["distance_impulse_m"],
            "Impulse_Raw": a2["impulse_raw"],
            "Impulse_Smoothed": a2["impulse_smoothed"],
            "Display_Impulse": a2["display_impulse"],
            "Magnitude_Final_dB": a2["magnitude_final_db"],
            "Magnitude_Double_dB": a2["magnitude_double_db"],
        }
    ).to_csv(
        algorithm_dir / f"algorithm2_impulse_{case.name}.csv",
        index=False,
        float_format="%.15g",
    )


def model_physical_metadata(variant_name: str) -> dict[str, float | int | str | None]:
    """Return physical bookkeeping used in the diagnostic report."""

    variant = MODEL_VARIANTS[variant_name]
    if variant.cell_length_m is None:
        return {
            "model_variant": variant_name,
            "model_title": variant.title,
            "cell_length_m": None,
            "cell_count": None,
            "cells_per_10m": None,
            "cutoff_frequency_hz": None,
            "electrical_length_at_200mhz_rad": None,
            "z0_high_frequency_ohm": math.sqrt(0.3e-6 / 16.0e-12),
            "model_native_velocity_factor": 1.0 / math.sqrt(
                (0.3e-6 / BASE_CELL_LENGTH_M)
                * (16.0e-12 / BASE_CELL_LENGTH_M)
            )
            / C0,
        }
    total_cells, cells_per_segment = validate_cell_length(float(variant.cell_length_m))
    scale = float(variant.cell_length_m) / BASE_CELL_LENGTH_M
    cell_l = 0.3e-6 * scale
    cell_c = 16.0e-12 * scale
    return {
        "model_variant": variant_name,
        "model_title": variant.title,
        "cell_length_m": float(variant.cell_length_m),
        "cell_count": total_cells,
        "cells_per_10m": cells_per_segment,
        "cutoff_frequency_hz": 1.0 / (np.pi * math.sqrt(cell_l * cell_c)),
        "electrical_length_at_200mhz_rad": 2.0
        * np.pi
        * 200.0e6
        * math.sqrt(cell_l * cell_c),
        "z0_high_frequency_ohm": math.sqrt(cell_l / cell_c),
        "model_native_velocity_factor": float(variant.cell_length_m)
        / math.sqrt(cell_l * cell_c)
        / C0,
    }


def joint_tail_metric(
    distance_m: np.ndarray,
    values: np.ndarray,
    *,
    joint_low_m: float = 8.0,
    joint_high_m: float = 18.0,
    tail_start_m: float = 1.5,
    tail_end_m: float = 6.0,
) -> dict[str, float | None]:
    """Measure the post-joint tail relative to the largest joint response."""

    distance_m = np.asarray(distance_m, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    joint_mask = (
        (distance_m >= joint_low_m)
        & (distance_m < joint_high_m)
        & np.isfinite(values)
    )
    if not np.any(joint_mask):
        return {
            "joint_peak_position_m": None,
            "joint_peak_amplitude": None,
            "tail_rms": None,
            "tail_peak": None,
            "tail_to_joint_rms": None,
            "tail_to_joint_peak": None,
        }
    joint_indices = np.flatnonzero(joint_mask)
    joint_index = int(joint_indices[np.argmax(np.abs(values[joint_indices]))])
    joint_position = float(distance_m[joint_index])
    joint_amplitude = float(abs(values[joint_index]))
    tail_mask = (
        (distance_m >= joint_position + tail_start_m)
        & (distance_m < joint_position + tail_end_m)
        & np.isfinite(values)
    )
    if not np.any(tail_mask) or joint_amplitude <= 0.0:
        return {
            "joint_peak_position_m": joint_position,
            "joint_peak_amplitude": joint_amplitude,
            "tail_rms": None,
            "tail_peak": None,
            "tail_to_joint_rms": None,
            "tail_to_joint_peak": None,
        }
    tail_values = values[tail_mask]
    tail_rms = float(np.sqrt(np.mean(tail_values**2)))
    tail_peak = float(np.max(np.abs(tail_values)))
    return {
        "joint_peak_position_m": joint_position,
        "joint_peak_amplitude": joint_amplitude,
        "tail_rms": tail_rms,
        "tail_peak": tail_peak,
        "tail_to_joint_rms": tail_rms / joint_amplitude,
        "tail_to_joint_peak": tail_peak / joint_amplitude,
    }


def _band_median_db(
    frequency_hz: np.ndarray,
    s11: np.ndarray,
    low_mhz: float,
    high_mhz: float,
) -> float | None:
    mask = (frequency_hz >= low_mhz * 1.0e6) & (frequency_hz < high_mhz * 1.0e6)
    if not np.any(mask):
        return None
    magnitude_db = 20.0 * np.log10(np.maximum(np.abs(s11[mask]), 1.0e-15))
    return float(np.median(magnitude_db))


def _phase_slope_deg_per_mhz(
    frequency_hz: np.ndarray,
    s11: np.ndarray,
    low_mhz: float,
    high_mhz: float,
) -> float | None:
    mask = (frequency_hz >= low_mhz * 1.0e6) & (frequency_hz < high_mhz * 1.0e6)
    if np.count_nonzero(mask) < 3:
        return None
    x = frequency_hz[mask] / 1.0e6
    y = np.unwrap(np.angle(s11[mask])) * 180.0 / np.pi
    return float(np.polyfit(x, y, 1)[0])


def frequency_edge_metrics(
    frequency_hz: np.ndarray, s11: np.ndarray
) -> dict[str, float | None]:
    """Summarize the 145 MHz amplitude and phase edge used in the report."""

    before = _band_median_db(frequency_hz, s11, 135.0, 145.0)
    after = _band_median_db(frequency_hz, s11, 145.0, 155.0)
    before_phase = _phase_slope_deg_per_mhz(frequency_hz, s11, 135.0, 145.0)
    after_phase = _phase_slope_deg_per_mhz(frequency_hz, s11, 145.0, 155.0)
    return {
        "magnitude_median_db_135_145": before,
        "magnitude_median_db_145_155": after,
        "magnitude_change_db": (
            None if before is None or after is None else after - before
        ),
        "phase_slope_deg_per_mhz_135_145": before_phase,
        "phase_slope_deg_per_mhz_145_155": after_phase,
        "phase_slope_change_deg_per_mhz": (
            None
            if before_phase is None or after_phase is None
            else after_phase - before_phase
        ),
    }


def model_baseline_diagnostics(
    variant_name: str,
    frequency_hz: np.ndarray,
    s11: np.ndarray,
    a1: dict[str, np.ndarray | float | int],
    a2: dict[str, np.ndarray | float | int],
) -> dict[str, object]:
    diagnostics = model_physical_metadata(variant_name)
    diagnostics["frequency_edge"] = frequency_edge_metrics(frequency_hz, s11)
    variant = MODEL_VARIANTS[variant_name]
    if variant.include_special_joint:
        diagnostics["algorithm1_joint_tail"] = joint_tail_metric(
            np.asarray(a1["distance_m"]), np.asarray(a1["impulse_real_ref"])
        )
        diagnostics["algorithm2_joint_tail"] = joint_tail_metric(
            np.asarray(a2["distance_impulse_m"]), np.asarray(a2["impulse_smoothed"])
        )
    else:
        diagnostics["algorithm1_joint_tail"] = {
            "joint_peak_position_m": None,
            "joint_peak_amplitude": None,
            "tail_rms": None,
            "tail_peak": None,
            "tail_to_joint_rms": None,
            "tail_to_joint_peak": None,
        }
        diagnostics["algorithm2_joint_tail"] = dict(
            diagnostics["algorithm1_joint_tail"]
        )
    diagnostics["s11_abs_mean"] = float(np.mean(np.abs(s11)))
    diagnostics["s11_abs_std"] = float(np.std(np.abs(s11)))
    diagnostics["all_finite"] = all(
        finite_array(value)
        for value in (
            frequency_hz,
            s11.real,
            s11.imag,
            a1["distance_m"],
            a1["impulse_real_ref"],
            a1["step_ref"],
            a2["distance_m"],
            a2["step_smoothed"],
            a2["distance_impulse_m"],
            a2["impulse_smoothed"],
        )
    )
    diagnostics["event_positions_m"] = baseline_event_positions(a1, a2)
    return diagnostics


def relative_rms(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    scale = float(np.sqrt(np.mean(reference**2)))
    if scale <= 0.0:
        scale = 1.0
    return float(np.sqrt(np.mean((candidate - reference) ** 2)) / scale)


def complex_relative_rms(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate = np.asarray(candidate, dtype=np.complex128)
    reference = np.asarray(reference, dtype=np.complex128)
    scale = float(np.sqrt(np.mean(np.abs(reference) ** 2)))
    if scale <= 0.0:
        scale = 1.0
    return float(np.sqrt(np.mean(np.abs(candidate - reference) ** 2)) / scale)


def compare_model_to_reference(
    candidate: dict[str, object],
    reference: dict[str, object],
) -> dict[str, float]:
    """Compare equal-grid frequency and distance arrays against a reference."""

    candidate_a1 = candidate["a1"]
    reference_a1 = reference["a1"]
    candidate_a2 = candidate["a2"]
    reference_a2 = reference["a2"]
    return {
        "s11_complex_relative_rms": complex_relative_rms(
            np.asarray(candidate["s11"]), np.asarray(reference["s11"])
        ),
        "algorithm1_step_relative_rms": relative_rms(
            np.asarray(candidate_a1["step_raw"]),
            np.asarray(reference_a1["step_raw"]),
        ),
        "algorithm1_impulse_relative_rms": relative_rms(
            np.asarray(candidate_a1["impulse_real_raw"]),
            np.asarray(reference_a1["impulse_real_raw"]),
        ),
        "algorithm2_step_relative_rms": relative_rms(
            np.asarray(candidate_a2["step_smoothed"]),
            np.asarray(reference_a2["step_smoothed"]),
        ),
        "algorithm2_impulse_relative_rms": relative_rms(
            np.asarray(candidate_a2["impulse_smoothed"]),
            np.asarray(reference_a2["impulse_smoothed"]),
        ),
    }


def load_v0_case_reference(
    case: Case,
    frequency_hz: np.ndarray,
    *,
    time_points: int,
    chunk_size: int,
) -> dict[str, object]:
    """Load frozen V0 outputs, falling back to read-only recomputation."""

    v0_dir = Path(__file__).resolve().parent.parent / "V0"
    s11_path = v0_dir / "s11_output" / f"s11_{case.name}.csv"
    a1_path = v0_dir / "output" / f"algorithm1_{case.name}.csv"
    a2_path = v0_dir / "output" / f"algorithm2_{case.name}.csv"
    saved_matches_grid = False
    if s11_path.is_file() and a1_path.is_file() and a2_path.is_file():
        saved_frequency = pd.read_csv(s11_path, usecols=["Frequency_Hz"])[
            "Frequency_Hz"
        ].to_numpy(np.float64)
        saved_matches_grid = (
            saved_frequency.size == frequency_hz.size
            and np.allclose(saved_frequency, frequency_hz, rtol=0.0, atol=1.0e-6)
        )
    if saved_matches_grid:
        s11_frame = pd.read_csv(s11_path)
        a1_frame = pd.read_csv(a1_path)
        a2_frame = pd.read_csv(a2_path)
        a1 = {
            "time_s": a1_frame["Time_s"].to_numpy(np.float64),
            "distance_m": a1_frame["Distance_m"].to_numpy(np.float64),
            "impulse_real_raw": a1_frame["Impulse_Real_Raw"].to_numpy(np.float64),
            "step_raw": a1_frame["Step_Raw"].to_numpy(np.float64),
            "impulse_real_ref": a1_frame["Impulse_Real_REF"].to_numpy(np.float64),
            "impulse_magnitude_ref": a1_frame["Impulse_Magnitude_REF"].to_numpy(
                np.float64
            ),
            "step_ref": a1_frame["Step_REF"].to_numpy(np.float64),
            "display_impulse": a1_frame["Display_Impulse"].to_numpy(np.float64),
            "display_step": a1_frame["Display_Step"].to_numpy(np.float64),
            "gamma": a1_frame["Gamma"].to_numpy(np.float64),
            "impedance_ohm": a1_frame["Impedance_Ohm"].to_numpy(np.float64),
            "step_scale": float(np.max(np.abs(a1_frame["Step_Raw"]))),
        }
        step_distance = a2_frame["Distance_m"].to_numpy(np.float64)
        impulse_mask = np.isfinite(a2_frame["Distance_Impulse_m"])
        a2 = {
            "time_s": a2_frame["Time_s"].to_numpy(np.float64),
            "distance_m": step_distance,
            "step_raw": a2_frame["Step_Raw"].to_numpy(np.float64),
            "step_smoothed": a2_frame["Step_Smoothed"].to_numpy(np.float64),
            "display_step": a2_frame["Display_Step"].to_numpy(np.float64),
            "distance_impulse_m": a2_frame.loc[
                impulse_mask, "Distance_Impulse_m"
            ].to_numpy(np.float64),
            "impulse_raw": a2_frame.loc[impulse_mask, "Impulse_Raw"].to_numpy(
                np.float64
            ),
            "impulse_smoothed": a2_frame.loc[
                impulse_mask, "Impulse_Smoothed"
            ].to_numpy(np.float64),
            "display_impulse": a2_frame.loc[
                impulse_mask, "Display_Impulse"
            ].to_numpy(np.float64),
            "magnitude_final_db": a2_frame.loc[
                impulse_mask, "Magnitude_Final_dB"
            ].to_numpy(np.float64),
            "magnitude_double_db": a2_frame.loc[
                impulse_mask, "Magnitude_Double_dB"
            ].to_numpy(np.float64),
            "impedance_smoothed_ohm": a2_frame["Impedance_Ohm"].to_numpy(
                np.float64
            ),
            "detected_end_distance_m": float(
                np.nanmax(a2_frame["Distance_Impulse_m"])
            ),
        }
        s11 = s11_frame["S11_Real"].to_numpy(np.float64) + 1j * s11_frame[
            "S11_Imag"
        ].to_numpy(np.float64)
        return {"s11": s11, "a1": a1, "a2": a2, "source": "V0 saved outputs"}

    v0_path = v0_dir / "cst_fdr_reproduction.py"
    spec = importlib.util.spec_from_file_location("cst_v0_reference", v0_path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"找不到V0程序或无法加载: {v0_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    s11 = module.simulate_s11(frequency_hz, case)
    a1 = module.algorithm1_ifft(frequency_hz, s11)
    with contextlib.redirect_stdout(io.StringIO()):
        a2 = module.algorithm2_fdr(
            frequency_hz,
            s11,
            time_points=time_points,
            chunk_size=chunk_size,
            progress_label="",
        )
    return {"s11": s11, "a1": a1, "a2": a2, "source": "V0 read-only recompute"}


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "SimHei",
            "font.sans-serif": ["SimHei", "Microsoft YaHei", "Microsoft YaHei UI"],
            "axes.unicode_minus": False,
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )


def style_axis(axis: plt.Axes, *, frequency: bool = False) -> None:
    axis.tick_params(direction="in", top=True, right=True, width=0.7)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
    axis.grid(True, which="major", linewidth=0.3, alpha=0.25)
    axis.set_xlabel(
        "Frequency (MHz)" if frequency else "Distance (m)",
        fontname="Times New Roman",
    )


def plot_resolution_comparison(
    assets_dir: Path,
    frequency_hz: np.ndarray,
    runs: dict[str, dict[str, object]],
) -> None:
    configure_plot_style()
    fig, axes = plt.subplots(4, 2, figsize=(15.5, 15.0), dpi=200)
    colors = {"fixed_ladder_0p4m": "#1f77b4", "fixed_ladder_0p2m": "#ff7f0e",
              "fixed_ladder_0p1m": "#2ca02c", "fixed_ladder_0p05m": "#d62728"}
    labels = {name: MODEL_VARIANTS[name].title for name in runs}
    for name, run in runs.items():
        color = colors.get(name, None)
        s11 = np.asarray(run["s11"])
        a1 = run["a1"]
        a2 = run["a2"]
        axes[0, 0].plot(frequency_hz / 1e6, s11.real, lw=0.75, color=color, label=labels[name])
        axes[0, 1].plot(frequency_hz / 1e6, 20*np.log10(np.maximum(np.abs(s11), 1e-15)), lw=0.75, color=color, label=labels[name])
        axes[1, 0].plot(frequency_hz / 1e6, np.angle(s11, deg=True), lw=0.75, color=color, label=labels[name])
        axes[1, 1].plot(np.asarray(a1["distance_m"]), np.asarray(a1["display_step"]), lw=0.75, color=color, label=labels[name])
        axes[2, 0].plot(np.asarray(a1["distance_m"]), np.asarray(a1["display_impulse"]), lw=0.75, color=color, label=labels[name])
        axes[2, 1].plot(np.asarray(a2["distance_m"]), np.asarray(a2["display_step"]), lw=0.75, color=color, label=labels[name])
        axes[3, 0].plot(np.asarray(a2["distance_impulse_m"]), np.asarray(a2["display_impulse"]), lw=0.75, color=color, label=labels[name])
    titles = [
        "S11 实部", "S11 幅值 (dB)", "S11 相位 (包裹)",
        "算法1 阶跃 / 归一化", "算法1 脉冲 / 归一化",
        "算法2 阶跃 / 归一化", "算法2 脉冲 / 归一化",
    ]
    for index, axis in enumerate(axes.flat[:7]):
        axis.set_title(titles[index])
        style_axis(axis, frequency=index < 3)
        axis.legend(fontsize=7)
        if index < 3:
            axis.axvline(145.0, color="black", linestyle="--", lw=0.6)
        else:
            axis.set_xlim(0.0, PLOT_MAX_DISTANCE_M)
    axes[1, 0].set_ylim(-180.0, 180.0)
    axes[3, 1].axis("off")
    axes[3, 1].text(
        0.02, 0.95,
        "固定参数、相同总长、相同接头\n仅改变梯形单元尺寸\n虚线：145 MHz",
        va="top", fontsize=10,
    )
    fig.suptitle("CST_Reproduction V1：梯形单元尺寸收敛", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(assets_dir / "resolution_convergence.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_baseline_variant_comparison(
    assets_dir: Path,
    runs: dict[str, dict[str, object]],
    filename: str,
    title: str,
) -> None:
    configure_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.5), dpi=200)
    for name, run in runs.items():
        a1 = run["a1"]
        a2 = run["a2"]
        label = MODEL_VARIANTS[name].title
        axes[0, 0].plot(a1["distance_m"], a1["display_step"], lw=0.8, label=label)
        axes[0, 1].plot(a1["distance_m"], a1["display_impulse"], lw=0.8, label=label)
        axes[1, 0].plot(a2["distance_m"], a2["display_step"], lw=0.8, label=label)
        axes[1, 1].plot(a2["distance_impulse_m"], a2["display_impulse"], lw=0.8, label=label)
    for axis, title_text in zip(
        axes.flat,
        ("算法1 阶跃", "算法1 脉冲", "算法2 阶跃", "算法2 脉冲"),
    ):
        axis.set_title(title_text)
        axis.set_xlim(0.0, PLOT_MAX_DISTANCE_M)
        axis.legend(fontsize=7)
        style_axis(axis)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(assets_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_v0_v1_case_board(
    output_dir: Path,
    case: Case,
    frequency_hz: np.ndarray,
    v0_run: dict[str, object],
    v1_run: dict[str, object],
    references: dict[str, Path],
) -> None:
    configure_plot_style()
    reference = references.get(case.reference_image or "")
    has_reference = reference is not None
    height_ratios = [1.0, 1.0, 1.0, 1.35] if has_reference else [1.0, 1.0, 1.0]
    fig = plt.figure(figsize=(16.0, 17.5 if has_reference else 12.0), dpi=200)
    grid = fig.add_gridspec(4 if has_reference else 3, 6, height_ratios=height_ratios)
    axes = [
        fig.add_subplot(grid[0, 0:2]), fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[0, 4:6]), fig.add_subplot(grid[1, 0:3]),
        fig.add_subplot(grid[1, 3:6]), fig.add_subplot(grid[2, 0:3]),
        fig.add_subplot(grid[2, 3:6]),
    ]
    v0_s11, v1_s11 = np.asarray(v0_run["s11"]), np.asarray(v1_run["s11"])
    axes[0].plot(frequency_hz/1e6, v0_s11.real, lw=0.8, label="V0")
    axes[0].plot(frequency_hz/1e6, v1_s11.real, lw=0.8, label="V1 0.1 m")
    axes[1].plot(frequency_hz/1e6, 20*np.log10(np.maximum(np.abs(v0_s11),1e-15)), lw=0.8, label="V0")
    axes[1].plot(frequency_hz/1e6, 20*np.log10(np.maximum(np.abs(v1_s11),1e-15)), lw=0.8, label="V1 0.1 m")
    axes[2].plot(frequency_hz/1e6, np.angle(v0_s11, deg=True), lw=0.8, label="V0")
    axes[2].plot(frequency_hz/1e6, np.angle(v1_s11, deg=True), lw=0.8, label="V1 0.1 m")
    v0_a1, v1_a1 = v0_run["a1"], v1_run["a1"]
    v0_a2, v1_a2 = v0_run["a2"], v1_run["a2"]
    for axis, key in ((axes[3], "display_step"), (axes[4], "display_impulse")):
        axis.plot(v0_a1["distance_m"], v0_a1[key], lw=0.8, label="V0")
        axis.plot(v1_a1["distance_m"], v1_a1[key], lw=0.8, label="V1 0.1 m")
    axes[5].plot(v0_a2["distance_m"], v0_a2["display_step"], lw=0.8, label="V0")
    axes[5].plot(v1_a2["distance_m"], v1_a2["display_step"], lw=0.8, label="V1 0.1 m")
    axes[6].plot(v0_a2["distance_impulse_m"], v0_a2["display_impulse"], lw=0.8, label="V0")
    axes[6].plot(v1_a2["distance_impulse_m"], v1_a2["display_impulse"], lw=0.8, label="V1 0.1 m")
    for index, axis in enumerate(axes):
        axis.set_title(("S11 实部", "S11 幅值 (dB)", "S11 相位 (包裹)",
                        "算法1 阶跃", "算法1 脉冲", "算法2 阶跃", "算法2 脉冲")[index])
        style_axis(axis, frequency=index < 3)
        axis.legend(fontsize=7)
        if index < 3:
            axis.axvline(145.0, color="black", linestyle="--", lw=0.6)
        else:
            axis.set_xlim(0.0, PLOT_MAX_DISTANCE_M)
    axes[2].set_ylim(-180.0, 180.0)
    if has_reference:
        ref_axis = fig.add_subplot(grid[3, 0:6])
        image = mpimg.imread(reference)
        ref_axis.imshow(image, aspect="equal")
        ref_axis.set_box_aspect(image.shape[0] / image.shape[1])
        ref_axis.set_title("报告对应 CST 图")
        ref_axis.set_anchor("C")
        ref_axis.axis("off")
    fig.suptitle(f"V0/V1 对照：{case.title}", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    fig.savefig(output_dir / f"v0_vs_v1_{case.name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def read_external_s11(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    columns = {str(column).lower().replace("_", ""): column for column in frame.columns}
    frequency_column = next(
        column for name, column in columns.items() if "freq" in name
    )
    real_column = next(
        column for name, column in columns.items() if "real" in name
    )
    imaginary_column = next(
        column for name, column in columns.items() if "imag" in name
    )
    frequency = pd.to_numeric(frame[frequency_column], errors="coerce").to_numpy(
        np.float64
    )
    s11 = pd.to_numeric(frame[real_column], errors="coerce").to_numpy(
        np.float64
    ) + 1j * pd.to_numeric(frame[imaginary_column], errors="coerce").to_numpy(
        np.float64
    )
    mask = np.isfinite(frequency) & np.isfinite(s11.real) & np.isfinite(s11.imag)
    order = np.argsort(frequency[mask])
    return frequency[mask][order], s11[mask][order]


def plot_field_reference(assets_dir: Path, data_root: Path) -> list[dict[str, object]]:
    """Plot read-only field S11 references and return their metadata."""

    candidates = (
        ("常州220kV", data_root / "无校准S11" / "常州-5100m" / "常州220kV输电电缆96线-0517-A相.csv"),
        ("现场250m", data_root / "无校准S11" / "某地-250m" / "2025-10-29 15.52.10-A相[末端开路].csv"),
        ("现场300m", data_root / "无校准S11" / "某地-300m" / "2025-9-16 12.02.15-A相[末端开路].csv"),
        ("现场780m", data_root / "无校准S11" / "某地-780m" / "2025-6-5 12.48.52-A相[末端开路].csv"),
    )
    traces: list[tuple[str, np.ndarray, np.ndarray]] = []
    metadata: list[dict[str, object]] = []
    for label, path in candidates:
        if not path.is_file():
            continue
        frequency, s11 = read_external_s11(path)
        traces.append((label, frequency, s11))
        metadata.append(
            {
                "label": label,
                "path": str(path),
                "point_count": int(frequency.size),
                "frequency_start_hz": float(frequency[0]),
                "frequency_stop_hz": float(frequency[-1]),
                "edge": frequency_edge_metrics(frequency, s11),
            }
        )
    if not traces:
        return metadata
    configure_plot_style()
    fig, axes = plt.subplots(2, 1, figsize=(14.0, 8.0), dpi=200, sharex=False)
    for label, frequency, s11 in traces:
        axes[0].plot(
            frequency / 1e6,
            20.0 * np.log10(np.maximum(np.abs(s11), 1e-15)),
            lw=0.75,
            label=label,
        )
        axes[1].plot(
            frequency / 1e6,
            np.angle(s11, deg=True),
            lw=0.75,
            label=label,
        )
    axes[0].set_title("现场 S11 幅频参考（只读数据）")
    axes[1].set_title("现场 S11 相频参考（只读数据）")
    for axis in axes:
        style_axis(axis, frequency=True)
        axis.axvline(145.0, color="black", linestyle="--", lw=0.7, label="145 MHz")
        axis.set_xlim(0.0, 200.0)
        axis.legend(fontsize=8)
    axes[1].set_ylim(-180.0, 180.0)
    fig.tight_layout()
    fig.savefig(assets_dir / "field_amplitude_phase_reference.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return metadata


def run_rg58_reference_validation(
    assets_dir: Path,
    output_dir: Path,
    data_root: Path,
    *,
    time_points: int,
    chunk_size: int,
) -> list[dict[str, object]]:
    """Use the existing two FDR algorithms as an external locality check."""

    root = data_root / "RG58缺陷制造实验"
    patterns = {
        "Health-1": "*Health-1.csv",
        "Health-2": "*Health-2.csv",
        "CutPVC": "*CutPVC-1.csv",
        "guhua": "*guhua.csv",
    }
    traces: dict[str, tuple[np.ndarray, np.ndarray, dict[str, object], dict[str, object]]] = {}
    for label, pattern in patterns.items():
        matches = list(root.glob(pattern))
        if not matches:
            return []
        frequency, s11 = read_external_s11(matches[0])
        a1 = algorithm1_ifft(frequency, s11)
        with contextlib.redirect_stdout(io.StringIO()):
            a2 = algorithm2_fdr(
                frequency,
                s11,
                time_points=time_points,
                chunk_size=chunk_size,
                progress_label="",
            )
        traces[label] = (frequency, s11, a1, a2)
    h1, h2 = traces["Health-1"], traces["Health-2"]
    h1_step_scale = np.sqrt(np.mean(((h1[2]["step_raw"] + h2[2]["step_raw"]) / 2.0) ** 2))
    h1_impulse_scale = np.max(np.abs((h1[2]["impulse_real_raw"] + h2[2]["impulse_real_raw"]) / 2.0))
    h2_step_scale = np.sqrt(np.mean(((h1[3]["step_smoothed"] + h2[3]["step_smoothed"]) / 2.0) ** 2))
    h2_impulse_scale = np.max(np.abs((h1[3]["impulse_smoothed"] + h2[3]["impulse_smoothed"]) / 2.0))
    health_a1_step = (h1[2]["step_raw"] + h2[2]["step_raw"]) / 2.0
    health_a1_impulse = (h1[2]["impulse_real_raw"] + h2[2]["impulse_real_raw"]) / 2.0
    health_a2_step = (h1[3]["step_smoothed"] + h2[3]["step_smoothed"]) / 2.0
    health_a2_impulse = (h1[3]["impulse_smoothed"] + h2[3]["impulse_smoothed"]) / 2.0
    records: list[dict[str, object]] = []
    configure_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.0), dpi=200)
    def peak_position(x: np.ndarray, y: np.ndarray, low: float, high: float) -> float:
        mask = (x >= low) & (x < high)
        return float(x[mask][np.argmax(np.abs(y[mask]))])

    for label in ("CutPVC", "guhua"):
        frequency, _, a1, a2 = traces[label]
        delta_a1_step = (a1["step_raw"] - health_a1_step) / max(h1_step_scale, 1e-30)
        delta_a1_impulse = (a1["impulse_real_raw"] - health_a1_impulse) / max(h1_impulse_scale, 1e-30)
        delta_a2_step = (a2["step_smoothed"] - health_a2_step) / max(h2_step_scale, 1e-30)
        delta_a2_impulse = (a2["impulse_smoothed"] - health_a2_impulse) / max(h2_impulse_scale, 1e-30)
        axes[0, 0].plot(a1["distance_m"], delta_a1_step, lw=0.8, label=label)
        axes[0, 1].plot(a1["distance_m"], delta_a1_impulse, lw=0.8, label=label)
        axes[1, 0].plot(a2["distance_m"], delta_a2_step, lw=0.8, label=label)
        axes[1, 1].plot(a2["distance_impulse_m"], delta_a2_impulse, lw=0.8, label=label)
        def rms_window(x: np.ndarray, y: np.ndarray, low: float, high: float) -> float:
            mask = (x >= low) & (x < high)
            return float(np.sqrt(np.mean(y[mask] ** 2)))
        if label == "CutPVC":
            local_a1 = rms_window(a1["distance_m"], delta_a1_impulse, 45.0, 60.0)
            local_a2 = rms_window(a2["distance_impulse_m"], delta_a2_impulse, 45.0, 60.0)
        else:
            local_a1 = rms_window(a1["distance_m"], delta_a1_impulse, 45.0, 60.0)
            local_a2 = rms_window(a2["distance_impulse_m"], delta_a2_impulse, 45.0, 60.0)
        post_a1 = rms_window(a1["distance_m"], delta_a1_impulse, 60.0, 80.0)
        post_a2 = rms_window(a2["distance_impulse_m"], delta_a2_impulse, 60.0, 80.0)
        records.append(
            {
                "state": label,
                "algorithm1_local_peak_position_m": peak_position(
                    a1["distance_m"], delta_a1_impulse, 45.0, 60.0
                ),
                "algorithm1_local_rms_45_60": local_a1,
                "algorithm1_post_rms_60_80": post_a1,
                "algorithm1_post_to_local_rms": post_a1 / local_a1,
                "algorithm2_local_peak_position_m": peak_position(
                    a2["distance_impulse_m"], delta_a2_impulse, 45.0, 60.0
                ),
                "algorithm2_local_rms_45_60": local_a2,
                "algorithm2_post_rms_60_80": post_a2,
                "algorithm2_post_to_local_rms": post_a2 / local_a2,
            }
        )
    for axis, title in zip(axes.flat, ("算法1 阶跃差分", "算法1 脉冲差分", "算法2 阶跃差分", "算法2 脉冲差分")):
        axis.set_title(title)
        axis.set_xlim(0.0, PLOT_MAX_DISTANCE_M)
        axis.axvspan(45.0, 60.0, color="tab:orange", alpha=0.12)
        axis.axvspan(60.0, 80.0, color="tab:green", alpha=0.08)
        axis.legend(fontsize=8)
        style_axis(axis)
    fig.suptitle("RG58 缺陷制造实验：两套 FDR 局部性参考", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(assets_dir / "rg58_defect_locality_reference.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(records).to_csv(output_dir / "rg58_reference_metrics.csv", index=False, encoding="utf-8-sig")
    return records


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


def baseline_event_positions(
    a1: dict[str, np.ndarray | float | int],
    a2: dict[str, np.ndarray | float | int],
) -> dict[str, dict[str, float]]:
    """Record the FDR-marked positions of the three joints and terminal peak."""

    a1_distance = np.asarray(a1["distance_m"])
    a1_impulse = np.asarray(a1["impulse_real_ref"])
    a2_distance = np.asarray(a2["distance_impulse_m"])
    a2_impulse = np.asarray(a2["impulse_smoothed"])
    windows = (
        ("joint_10m_m", 8.0, 14.0),
        ("joint_20m_m", 18.0, 26.0),
        ("joint_30m_m", 27.0, 38.0),
    )
    result: dict[str, dict[str, float]] = {"algorithm1": {}, "algorithm2": {}}
    for name, low_m, high_m in windows:
        result["algorithm1"][name] = peak_in_window(
            a1_distance, a1_impulse, low_m, high_m
        )[0]
        result["algorithm2"][name] = peak_in_window(
            a2_distance, a2_impulse, low_m, high_m
        )[0]
    result["algorithm1"]["terminal_m"] = positive_peak_in_window(
        a1_distance, a1_impulse, 37.0, PLOT_MAX_DISTANCE_M
    )[0]
    result["algorithm2"]["terminal_m"] = positive_peak_in_window(
        a2_distance, a2_impulse, 37.0, PLOT_MAX_DISTANCE_M
    )[0]
    return result


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


def register_custom_variant(cell_length_m: float) -> str:
    cell_length_m = float(cell_length_m)
    if cell_length_m <= 0.0 or not np.isfinite(cell_length_m):
        raise ValueError("--cell-length必须是正的有限数值")
    total_cells, cells_per_segment = validate_cell_length(cell_length_m)
    del total_cells, cells_per_segment
    name = f"custom_ladder_{cell_length_m:g}".replace(".", "p")
    MODEL_VARIANTS[name] = CircuitVariant(
        name, f"自定义固定RLGC梯形：{cell_length_m:g} m单元", "fixed_ladder", cell_length_m
    )
    return name


def selected_variant_names(args: argparse.Namespace) -> list[str]:
    resolution = [
        "fixed_ladder_0p4m",
        "fixed_ladder_0p2m",
        "fixed_ladder_0p1m",
        "fixed_ladder_0p05m",
    ]
    main_models = ["fixed_ladder_0p1m", "fixed_continuous", "dg_loss_0p1m"]
    ablations = [
        "ablation_no_special",
        "ablation_no_tl300",
        "ablation_no_internal",
        "ablation_body_only",
    ]
    if args.model == "custom":
        return [register_custom_variant(args.cell_length)]
    if args.model != "all":
        selected = [args.model]
        if args.resolution_sweep:
            selected.extend(name for name in resolution if name not in selected)
        if args.include_continuous and "fixed_continuous" not in selected:
            selected.append("fixed_continuous")
        if args.ablation:
            selected.extend(name for name in ablations if name not in selected)
        return selected
    selected = []
    for name in resolution + main_models + ablations:
        if name not in selected:
            selected.append(name)
    return selected


def cases_for_variant(variant_name: str, requested_model: str) -> tuple[Case, ...]:
    if requested_model == "all":
        variant = MODEL_VARIANTS[variant_name]
        if variant_name.startswith("fixed_ladder_0p") and variant_name != "fixed_ladder_0p1m":
            return (CASES[0],)
        if variant_name.startswith("ablation_"):
            return (CASES[0],)
    return CASES


def run_variant(
    variant_name: str,
    cases: tuple[Case, ...],
    frequency_hz: np.ndarray,
    output_dir: Path,
    assets_dir: Path,
    s11_output_dir: Path,
    references: dict[str, Path],
    *,
    time_points: int,
    chunk_size: int,
) -> dict[str, object]:
    variant_asset_dir = assets_dir / variant_name
    variant_asset_dir.mkdir(parents=True, exist_ok=True)
    runs: dict[str, dict[str, object]] = {}
    case_metrics: list[dict[str, object]] = []
    baseline_run: dict[str, object] | None = None
    print(f"\n模型：{MODEL_VARIANTS[variant_name].title}")
    for index, case in enumerate(cases, start=1):
        print(f"  [{index}/{len(cases)}] {case.title}")
        s11 = simulate_s11(frequency_hz, case, variant_name)
        a1 = algorithm1_ifft(frequency_hz, s11)
        a2 = algorithm2_fdr(
            frequency_hz,
            s11,
            time_points=time_points,
            chunk_size=chunk_size,
            progress_label=f"算法2 {variant_name} {case.name}",
        )
        run = {"s11": s11, "a1": a1, "a2": a2}
        runs[case.name] = run
        save_variant_outputs(
            output_dir,
            s11_output_dir,
            variant_name,
            case,
            frequency_hz,
            s11,
            a1,
            a2,
        )
        if case.kind == "baseline":
            baseline_run = run
        if baseline_run is not None and case.kind != "baseline":
            case_metrics.append(
                compute_case_metrics(
                    case,
                    frequency_hz,
                    s11,
                    np.asarray(baseline_run["s11"]),
                    a1,
                    baseline_run["a1"],
                    a2,
                    baseline_run["a2"],
                )
            )
    if baseline_run is None:
        raise RuntimeError(f"模型没有基准工况: {variant_name}")
    diagnostics = model_baseline_diagnostics(
        variant_name,
        frequency_hz,
        np.asarray(baseline_run["s11"]),
        baseline_run["a1"],
        baseline_run["a2"],
    )
    if variant_name == "fixed_ladder_0p1m":
        for case in cases:
            if case.kind != "baseline":
                plot_locality_board(
                    variant_asset_dir,
                    case,
                    runs[case.name]["a1"],
                    baseline_run["a1"],
                    runs[case.name]["a2"],
                    baseline_run["a2"],
                )
    print(
        f"  基准S11|均值={diagnostics['s11_abs_mean']:.6g}，"
        f"算法1尾波比={diagnostics['algorithm1_joint_tail']['tail_to_joint_rms']}, "
        f"算法2尾波比={diagnostics['algorithm2_joint_tail']['tail_to_joint_rms']}"
    )
    return {
        "variant": variant_name,
        "runs": runs,
        "baseline": baseline_run,
        "diagnostics": diagnostics,
        "case_metrics": case_metrics,
    }


def write_v1_report(
    report_path: Path,
    runtime_s: float,
    point_count: int,
    time_points: int,
    model_runs: dict[str, dict[str, object]],
    resolution_convergence: dict[str, object],
    field_metadata: list[dict[str, object]],
    rg58_metrics: list[dict[str, object]],
    algorithm2_ref_parity: dict[str, object],
) -> None:
    lines = [
        "# CST_Reproduction V1：细分梯形单元与双算法对照报告",
        "",
        f"- 运行时间：{runtime_s:.3f} s",
        f"- 频率网格：9 kHz–200 MHz，{point_count}点",
        f"- 算法2时间点：{time_points}",
        "- V0保持冻结；本报告只记录V1程序生成的模型对照，不修改CST工程、REF程序或现场原始数据。",
        "- 50 Ω端口、135/300 Ω TL、特殊接头和开路边界均按V0保留。",
        "",
        "## 1. 既定模型与实验假设",
        "",
        "固定梯形细分按单位长度等比例缩放 R/L/G/C；0.1 m模型为V1默认模型，连续固定RLGC仅作收敛参考。15 m缺陷保持14.8–15.2 m物理范围。",
        "",
        "算法1和算法2均使用当前REF参数；S11是唯一频域源，不按算法拆分。",
        "",
        "## 2. 基准模型量化结果",
        "",
        "| 模型 | 单元(m) | 单元数 | 截止频率(MHz) | 200MHz单元电长度(rad) | Z0(Ω) | VF(模型) | 算法1尾波/主峰 | 算法2尾波/主峰 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, payload in model_runs.items():
        d = payload["diagnostics"]
        fmt = lambda value: "—" if value is None else f"{float(value):.6g}"
        lines.append(
            f"| {name} | {fmt(d['cell_length_m'])} | {d['cell_count'] or '—'} | "
            f"{fmt(None if d['cutoff_frequency_hz'] is None else d['cutoff_frequency_hz']/1e6)} | "
            f"{fmt(d['electrical_length_at_200mhz_rad'])} | {fmt(d['z0_high_frequency_ohm'])} | "
            f"{fmt(d['model_native_velocity_factor'])} | "
            f"{fmt(d['algorithm1_joint_tail']['tail_to_joint_rms'])} | "
            f"{fmt(d['algorithm2_joint_tail']['tail_to_joint_rms'])} |"
        )
    lines.extend([
        "",
        "145 MHz边缘指标同时保存于 `output/model_summary.csv` 和 `output/summary.json`；这些指标用于区分离散网络的截止边缘，不把现场文件直接当作CST真值。",
        "",
        "## 3. 分辨率收敛",
        "",
        "| 模型 | S11相对RMS | 算法1阶跃相对RMS | 算法1脉冲相对RMS | 算法2阶跃相对RMS | 算法2脉冲相对RMS |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name, values in resolution_convergence.items():
        lines.append(
            f"| {name} | {values['s11_complex_relative_rms']:.6g} | "
            f"{values['algorithm1_step_relative_rms']:.6g} | "
            f"{values['algorithm1_impulse_relative_rms']:.6g} | "
            f"{values['algorithm2_step_relative_rms']:.6g} | "
            f"{values['algorithm2_impulse_relative_rms']:.6g} |"
        )
    lines.extend(["", "## 4. 验收判定", ""])
    if "fixed_ladder_0p1m" in model_runs:
        default_diagnostic = model_runs["fixed_ladder_0p1m"]["diagnostics"]
        a1_tail = default_diagnostic["algorithm1_joint_tail"]["tail_to_joint_rms"]
        a2_tail = default_diagnostic["algorithm2_joint_tail"]["tail_to_joint_rms"]
        tail_pass = (
            a1_tail is not None and a2_tail is not None
            and float(a1_tail) <= 0.05 and float(a2_tail) <= 0.05
        )
        lines.append(
            f"- 0.1 m接头后尾波/主峰：算法1={a1_tail:.6g}、算法2={a2_tail:.6g}，"
            f"阈值0.05：{'通过' if tail_pass else '未通过'}。"
        )
        lines.append(
            f"- 0.1 m模型数组有限性：{'通过' if default_diagnostic['all_finite'] else '未通过'}。"
        )
    if "fixed_ladder_0p1m" in resolution_convergence:
        values = resolution_convergence["fixed_ladder_0p1m"]
        convergence_pass = all(
            float(values[key]) <= 0.05
            for key in (
                "s11_complex_relative_rms",
                "algorithm1_step_relative_rms",
                "algorithm1_impulse_relative_rms",
                "algorithm2_step_relative_rms",
                "algorithm2_impulse_relative_rms",
            )
        )
        lines.append(
            "- 0.1 m相对连续固定RLGC参考的五项相对RMS阈值0.05："
            f"{'通过' if convergence_pass else '未通过'}；当前结果说明0.1 m在尾波上稳定，"
            "但尚未达到全频域/全距离域数值收敛。"
        )
    default_run = model_runs.get("fixed_ladder_0p1m")
    default_diagnostic = None if default_run is None else default_run["diagnostics"]
    cutoff_parts = []
    for name, label in (
        ("fixed_ladder_0p4m", "0.4 m"),
        ("fixed_ladder_0p1m", "0.1 m"),
        ("fixed_ladder_0p05m", "0.05 m"),
    ):
        if name in model_runs:
            cutoff_parts.append(
                f"{label}={float(model_runs[name]['diagnostics']['cutoff_frequency_hz']) / 1.0e6:.3f} MHz"
            )
    cutoff_text = "、".join(cutoff_parts) if cutoff_parts else "当前选择的模型未提供梯形截止估计"
    algorithm1_tail_text = (
        f"0.1 m基准的接头后尾波/主峰指标为 {float(default_diagnostic['algorithm1_joint_tail']['tail_to_joint_rms']):.6g}。"
        if default_diagnostic is not None
        else "当前选择的模型未包含0.1 m默认模型，未计算该指标。"
    )
    position_text = (
        "按当前两套FDR速度标尺，0.1 m基准的事件位置为："
        f"算法1接头={default_diagnostic['event_positions_m']['algorithm1']['joint_10m_m']:.4g}/"
        f"{default_diagnostic['event_positions_m']['algorithm1']['joint_20m_m']:.4g}/"
        f"{default_diagnostic['event_positions_m']['algorithm1']['joint_30m_m']:.4g} m、"
        f"终端={default_diagnostic['event_positions_m']['algorithm1']['terminal_m']:.4g} m；"
        f"算法2接头={default_diagnostic['event_positions_m']['algorithm2']['joint_10m_m']:.4g}/"
        f"{default_diagnostic['event_positions_m']['algorithm2']['joint_20m_m']:.4g}/"
        f"{default_diagnostic['event_positions_m']['algorithm2']['joint_30m_m']:.4g} m、"
        f"终端={default_diagnostic['event_positions_m']['algorithm2']['terminal_m']:.4g} m。"
        "它们相对10/20/30/40 m的偏移是速度标尺差异，不在V1中通过修改电路长度强行校正。"
        if default_diagnostic is not None
        else "当前选择的模型未包含0.1 m默认模型，未写入默认事件位置。"
    )
    parity_text = (
        f"最大绝对误差={float(algorithm2_ref_parity['max_abs_error']):.6g}，"
        f"频率点（跳过首点）={algorithm2_ref_parity['frequency_points_after_skip']}，"
        f"时间点={algorithm2_ref_parity['time_points']}"
        if algorithm2_ref_parity.get("available")
        else f"未完成（{algorithm2_ref_parity.get('reason', '未知原因')}）"
    )
    lines.extend([
        "",
        "## 5. 频域、FDR 与伪影判别",
        "",
        "- S11频域依据：本次运行的梯形截止估计为 "
        + cutoff_text
        + "；0.4 m接近145 MHz，而细分到0.1/0.05 m后截止移到200 MHz以上，"
        "因此145 MHz附近的幅相断崖属于粗梯形网络的离散色散证据，不能当作现场电缆的统一截止。",
        "- 算法1依据：复数S11保留相位，经过自动DC外推、共轭双边谱、Hann窗和IFFT；"
        + algorithm1_tail_text,
        "- 空间位置依据：" + position_text,
        "- 算法2依据：严格只取S11实部，按REF的余弦积分、5点阶跃平滑、差分、50点脉冲平滑和补偿流程；"
        "本地实现与REF核心的独立小规模逐数组对照为 " + parity_text + "，误差应按浮点误差理解。",
        "- 算法伪影判别：有限9 kHz–200 MHz带宽、算法1的Hann旁瓣以及算法2的实部积分/平滑会改变峰宽、尾波和显示幅值；"
        "这些是变换显示效应，不能单独解释为新增的物理缺陷。",
        "- 模型离散伪影判别：0.4 m模型尾波明显更强，细分后显著下降；0.05 m相对连续模型的全频S11误差虽已降低，"
        "但距离域脉冲仍未达到预设0.05阈值，因此V1不能声称已对连续线完全收敛。",
        "- 现场限制：现场参考为只读、未校准或未统一标注的数据；它可以检验145 MHz是否具有共性、以及缺陷后是否出现同量级新峰，"
        "不能反推出统一的YJV RLGC参数，也不能替代CST ASCII S11逐点真值。",
        "",
        "## 6. 9个报告工况",
        "",
    ])
    for name in ("fixed_ladder_0p1m", "fixed_continuous", "dg_loss_0p1m"):
        if name not in model_runs:
            continue
        lines.extend([f"### {MODEL_VARIANTS[name].title}", "", "| 工况 | 终端峰(m) | 局部差分峰(m) | 局部差分幅值 | 是否吻合 |", "|---|---:|---:|---:|:---:|"])
        for item in model_runs[name]["case_metrics"]:
            lines.append(
                f"| {item['case']} | {item['terminal_peak_position_m']:.6g} | "
                f"{item['local_delta_peak_position_m']:.6g} | "
                f"{item['local_delta_peak_amplitude']:.6g} | "
                f"{'是' if item['qualitative_match'] else '否'} |"
            )
        lines.append("")
    lines.extend([
        "## 7. 现场外部参照",
        "",
        "现场参考只读使用，未参与V1参数建模。无校准S11中的未标注电缆不被强行标记为YJV。RG58缺陷制造实验用于检验两套算法对局部差分的稳定性。",
        "",
        f"- 现场幅相参考样本数：{len(field_metadata)}",
        f"- RG58缺陷参考状态数：{len(rg58_metrics)}",
        "",
        "| RG58状态 | 算法1局部峰(m) | 算法2局部峰(m) | 算法1后区/局部RMS | 算法2后区/局部RMS |",
        "|---|---:|---:|---:|---:|",
    ])
    for item in rg58_metrics:
        lines.append(
            f"| {item['state']} | {item['algorithm1_local_peak_position_m']:.6g} | "
            f"{item['algorithm2_local_peak_position_m']:.6g} | "
            f"{item['algorithm1_post_to_local_rms']:.6g} | "
            f"{item['algorithm2_post_to_local_rms']:.6g} |"
        )
    lines.extend([
        "",
        "现场幅频/相频图中的145 MHz线仅用于检查V0离散截止是否具有现场共性，不用于反推现场电缆参数。",
        "",
        "## 8. 图像和结果文件",
        "",
        "- `assets/resolution_convergence.png`：单元尺寸收敛。",
        "- `assets/component_ablation.png`：接头组件消融。",
        "- `assets/loss_model_comparison.png`：固定与频变损耗对照。",
        "- `assets/field_amplitude_phase_reference.png`：现场幅频/相频参考。",
        "- `assets/rg58_defect_locality_reference.png`：RG58双算法局部性参考。",
        "- `assets/v0_vs_v1_*.png`：9个工况的V0/V1对照。",
        "- `output/field_reference_metrics.csv`：现场145 MHz幅相参考指标。",
        "- `output/rg58_reference_metrics.csv`：RG58缺陷局部性指标。",
        "- `s11_output/<model>/s11_<case>.csv`：三列S11源文件。",
        "- `output/<model>/algorithm*.csv`：两套算法派生结果。",
        "",
        "当前没有CST ASCII S11导出，因此不报告CST逐点RMSE或完全一致结论。",
    ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sanitize_json(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    if isinstance(value, np.generic):
        return sanitize_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CST_Reproduction V1：细分梯形单元与双算法对照")
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
    model_choices = ["all", "custom", *MODEL_VARIANTS.keys()]
    parser.add_argument(
        "--model",
        choices=model_choices,
        default="all",
        help="运行指定模型；默认all会执行分辨率、连续、频变损耗和组件消融",
    )
    parser.add_argument(
        "--cell-length",
        type=float,
        default=DEFAULT_CELL_LENGTH_M,
        help="--model custom时的单元长度，必须能够整除10 m",
    )
    parser.add_argument(
        "--resolution-sweep",
        action="store_true",
        help="指定单个模型时额外加入0.4/0.2/0.1/0.05 m分辨率对照",
    )
    parser.add_argument(
        "--include-continuous",
        action="store_true",
        help="指定单个模型时额外加入连续固定RLGC参考",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="指定单个模型时额外加入接头组件消融",
    )
    parser.add_argument(
        "--field-reference-root",
        type=Path,
        default=Path(r"E:\FDR案例-csv"),
        help="只读现场参考数据根目录，不向该目录写入",
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
    configure_plot_style()

    started = time.perf_counter()
    frequency_hz = build_frequency_grid(point_count)
    references = extract_report_images(assets_dir)
    variant_names = selected_variant_names(args)
    print(f"算法结果目录: {output_dir}")
    print(f"对照图目录: {assets_dir}")
    print(f"S11 CSV目录: {s11_output_dir}")
    print(f"频率网格: {frequency_hz[0]:.6g} Hz - {frequency_hz[-1]:.6g} Hz, {point_count}点")
    print(f"算法2时间点: {time_points}，分块大小: {args.chunk_size}")
    print(f"模型变体: {', '.join(variant_names)}")
    schematic_summary = validate_schematic()
    print(f"CST网表: {SCHEMATIC_XML}")
    print(
        "电路自检通过: "
        f"{schematic_summary['standard_series_resistors']}个0.01Ω电阻、"
        f"{schematic_summary['standard_inductors']}个0.3μH电感、"
        f"{schematic_summary['tl_count']}个TL、1个开路块；"
        "15 m局部缺陷物理范围=14.8–15.2 m"
    )

    model_runs: dict[str, dict[str, object]] = {}
    for variant_name in variant_names:
        cases = cases_for_variant(variant_name, args.model)
        model_runs[variant_name] = run_variant(
            variant_name,
            cases,
            frequency_hz,
            output_dir,
            assets_dir,
            s11_output_dir,
            references,
            time_points=time_points,
            chunk_size=args.chunk_size,
        )

    resolution_names = [
        name for name in (
            "fixed_ladder_0p4m",
            "fixed_ladder_0p2m",
            "fixed_ladder_0p1m",
            "fixed_ladder_0p05m",
        ) if name in model_runs
    ]
    resolution_runs = {
        name: model_runs[name]["baseline"] for name in resolution_names
    }
    resolution_convergence: dict[str, dict[str, float]] = {}
    if "fixed_continuous" in model_runs:
        continuous_baseline = model_runs["fixed_continuous"]["baseline"]
        for name, run in resolution_runs.items():
            resolution_convergence[name] = compare_model_to_reference(
                run,
                continuous_baseline,
            )
        if resolution_runs:
            plot_resolution_comparison(assets_dir, frequency_hz, resolution_runs)

    ablation_names = [
        name for name in (
            "fixed_ladder_0p1m",
            "ablation_no_special",
            "ablation_no_tl300",
            "ablation_no_internal",
        ) if name in model_runs
    ]
    if len(ablation_names) >= 2:
        plot_baseline_variant_comparison(
            assets_dir,
            {name: model_runs[name]["baseline"] for name in ablation_names},
            "component_ablation.png",
            "CST_Reproduction V1：接头组件消融（含完整模型）",
        )

    main_names = [
        name for name in ("fixed_ladder_0p1m", "fixed_continuous", "dg_loss_0p1m")
        if name in model_runs
    ]
    if main_names:
        plot_baseline_variant_comparison(
            assets_dir,
            {name: model_runs[name]["baseline"] for name in main_names},
            "loss_model_comparison.png",
            "CST_Reproduction V1：固定/连续与 DG V3 风格频变损耗对照",
        )

    if "fixed_ladder_0p1m" in model_runs:
        v1_default = model_runs["fixed_ladder_0p1m"]
        for case in CASES:
            v0_run = load_v0_case_reference(
                case,
                frequency_hz,
                time_points=time_points,
                chunk_size=args.chunk_size,
            )
            plot_v0_v1_case_board(
                assets_dir,
                case,
                frequency_hz,
                v0_run,
                v1_default["runs"][case.name],
                references,
            )

    field_metadata = plot_field_reference(assets_dir, args.field_reference_root)
    pd.DataFrame(
        [
            {
                "label": item["label"],
                "path": item["path"],
                "point_count": item["point_count"],
                "frequency_start_hz": item["frequency_start_hz"],
                "frequency_stop_hz": item["frequency_stop_hz"],
                **item["edge"],
            }
            for item in field_metadata
        ]
    ).to_csv(output_dir / "field_reference_metrics.csv", index=False, encoding="utf-8-sig")
    rg58_metrics = run_rg58_reference_validation(
        assets_dir,
        output_dir,
        args.field_reference_root,
        time_points=time_points,
        chunk_size=args.chunk_size,
    )

    parity_source = next(iter(model_runs.values()))["baseline"]
    algorithm2_ref_parity = algorithm2_ref_parity_check(
        frequency_hz,
        np.asarray(parity_source["s11"]),
        time_points=time_points,
    )
    print(
        "Algorithm 2 REF核心对照: "
        + (
            f"最大绝对误差={algorithm2_ref_parity['max_abs_error']:.6g}"
            if algorithm2_ref_parity.get("available")
            else str(algorithm2_ref_parity.get("reason"))
        )
    )

    runtime_s = time.perf_counter() - started
    model_summary = {
        name: payload["diagnostics"] for name, payload in model_runs.items()
    }
    pd.DataFrame(
        [
            {
                "model_variant": name,
                "cell_length_m": diagnostic["cell_length_m"],
                "cell_count": diagnostic["cell_count"],
                "cutoff_frequency_hz": diagnostic["cutoff_frequency_hz"],
                "electrical_length_at_200mhz_rad": diagnostic["electrical_length_at_200mhz_rad"],
                "z0_high_frequency_ohm": diagnostic["z0_high_frequency_ohm"],
                "model_native_velocity_factor": diagnostic["model_native_velocity_factor"],
                "s11_abs_mean": diagnostic["s11_abs_mean"],
                "s11_abs_std": diagnostic["s11_abs_std"],
                "algorithm1_tail_to_joint_rms": diagnostic["algorithm1_joint_tail"]["tail_to_joint_rms"],
                "algorithm2_tail_to_joint_rms": diagnostic["algorithm2_joint_tail"]["tail_to_joint_rms"],
                "algorithm1_joint_10m_position_m": diagnostic["event_positions_m"]["algorithm1"]["joint_10m_m"],
                "algorithm1_joint_20m_position_m": diagnostic["event_positions_m"]["algorithm1"]["joint_20m_m"],
                "algorithm1_joint_30m_position_m": diagnostic["event_positions_m"]["algorithm1"]["joint_30m_m"],
                "algorithm1_terminal_position_m": diagnostic["event_positions_m"]["algorithm1"]["terminal_m"],
                "algorithm2_joint_10m_position_m": diagnostic["event_positions_m"]["algorithm2"]["joint_10m_m"],
                "algorithm2_joint_20m_position_m": diagnostic["event_positions_m"]["algorithm2"]["joint_20m_m"],
                "algorithm2_joint_30m_position_m": diagnostic["event_positions_m"]["algorithm2"]["joint_30m_m"],
                "algorithm2_terminal_position_m": diagnostic["event_positions_m"]["algorithm2"]["terminal_m"],
                "edge_magnitude_change_db": diagnostic["frequency_edge"]["magnitude_change_db"],
                "edge_phase_slope_change_deg_per_mhz": diagnostic["frequency_edge"]["phase_slope_change_deg_per_mhz"],
                "all_finite": diagnostic["all_finite"],
            }
            for name, diagnostic in model_summary.items()
        ]
    ).to_csv(output_dir / "model_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [{"model_variant": name, **values} for name, values in resolution_convergence.items()]
    ).to_csv(output_dir / "resolution_convergence.csv", index=False, encoding="utf-8-sig")

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
        "model_variants": model_summary,
        "resolution_convergence": resolution_convergence,
        "case_metrics": {
            name: payload["case_metrics"] for name, payload in model_runs.items()
        },
        "field_reference": field_metadata,
        "rg58_reference": rg58_metrics,
        "algorithm2_ref_parity": algorithm2_ref_parity,
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
    write_v1_report(
        report_path,
        runtime_s,
        point_count,
        time_points,
        model_runs,
        resolution_convergence,
        field_metadata,
        rg58_metrics,
        algorithm2_ref_parity,
    )
    print(f"\n完成：{len(model_runs)}个模型，耗时 {runtime_s:.3f} s")
    print(f"Markdown报告: {report_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断。")
        sys.exit(130)
