"""V2.7 RG58/Field 路线对报告仿真章节 9 工况的还原对比。

参考路径全部只读：
- DG_V2.7/core/s11_generator.py      —— V2.7 频变 RLGC 传播核（几何反解 + 趋肤效应 + tanδ 介损）
- DifferentAgent/CodeX/CST_Reproduction/V1/cst_fdr_reproduction.py
  —— 报告 CST 等效电路的结构元件（TL1 / special_joint1 / TL300 / 末端并联支路）、
     算法1 / 算法2、工况定义与参考图。

路线定义（与 CodeX/DG_Route_Evaluation 的做法同构）：
- 每条路线 = V2.7 RLGC 传播核 + 报告场景结构。电缆主体按报告拓扑分段
  （0-10 / 10-14.8 / 14.8-15.2 / 15.2-20 / 20-30 / 30-40 m），段参数取该路线的
  基准介质（z0、epsr、alpha@100MHz、tanδ），并按报告 9 工况做相对修改：
    * C 变化 → epsr 等比缩放（固定几何下 C ∝ epsr，Z0、相速随之与 sqrt(L/C) 一致）
    * G 变化 → tanδ 等比缩放（V2.7 用介损 tanδ 表示并联损耗，G ∝ f·C·tanδ）
    * R 变化 → 在 15 m 单元（0.4 m）段内注入常数串联电阻 R'（Ω/m）。
      V2.7 参数空间没有常数串联 R 旋钮（其导体损耗是几何反解的趋肤 R ∝ √f），
      因此把报告要求的额外常数 R' 直接加进该段单位长度串联阻抗 Z，
      这是唯一必要的最小扩展，报告中已记录。
- 与 CodeX 相同：9 kHz–200 MHz / 10000 点统一频率网格，clean 与
  common_measurement 两层，算法1/算法2 均由同一 S11 重新计算。

正式运行：
    python compare_v27_routes.py
短时冒烟：
    python compare_v27_routes.py --smoke
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import io
import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # D:\GitRepository\Cable-NN
V1_PATH = ROOT / "DifferentAgent/CodeX/CST_Reproduction/V1/cst_fdr_reproduction.py"
V27_CORE_PATH = ROOT / "DG_V2.7/core/s11_generator.py"
REPORT_REFERENCE_DIR = ROOT / "DifferentAgent/CodeX/CST_Reproduction/V1/assets/report_reference"

C0 = 299_792_458.0
REFERENCE_HZ = 100.0e6
BASE_CELL_LENGTH_M = 0.4
BASE_CELL_C = 16.0e-12          # 报告基准单元电容（每 0.4 m）
BASE_CELL_G = 1.0 / 200_000.0   # 报告基准单元并联电导（每 0.4 m）
LOCAL_START_M = 14.8
LOCAL_END_M = 15.2
SEGMENT_BOUNDARIES = (0.0, 10.0, 14.8, 15.2, 20.0, 30.0, 40.0)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入 {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V1 = load_module("v27_route_cst_v1", V1_PATH)
GEN = load_module("v27_route_s11_generator", V27_CORE_PATH)
SegmentParams = GEN.SegmentParams


def case_by_name(name: str):
    return next(item for item in V1.CASES if item.name == name)


def identity(size: int):
    one = np.ones(size, dtype=np.complex128)
    zero = np.zeros(size, dtype=np.complex128)
    return one, zero, zero, one


# ═══════════════════════════════════════════════
# 工况 → 路线段参数映射
# ═══════════════════════════════════════════════

def route_segment(route: dict, case, start_m: float, end_m: float):
    """把报告工况映射为一条 V2.7 路线段。返回 (SegmentParams, r_extra_per_m)。"""
    epsr = float(route["epsr"])
    tan_delta = float(route["tan_delta_100mhz"])
    r_extra_per_m = 0.0
    kind = case.kind
    center_m = 0.5 * (start_m + end_m)
    is_local = start_m < LOCAL_END_M and end_m > LOCAL_START_M

    if kind == "overall_c":
        epsr *= float(case.value) / BASE_CELL_C
    elif kind == "overall_g":
        tan_delta *= float(case.value) / BASE_CELL_G
    elif kind == "segmented_g":
        g_mult = 1.0 if center_m < 10.0 else 10.0 if center_m < 20.0 else 1.0 if center_m < 30.0 else 100.0
        tan_delta *= g_mult
    elif is_local:
        if kind == "local_c":
            epsr *= float(case.value) / BASE_CELL_C
        elif kind == "local_g":
            tan_delta *= float(case.value) / BASE_CELL_G
        elif kind == "local_r":
            # 报告：15 m 单元（0.4 m）串联电阻 0.01 Ω → case.value Ω。
            # 基线 0.01 Ω 已隐含在路线基准中，额外注入 (case.value − 0.01)/0.4 Ω/m。
            r_extra_per_m = max(float(case.value) - 0.01, 0.0) / BASE_CELL_LENGTH_M

    seg = SegmentParams(
        length_m=end_m - start_m,
        z0_ohm=float(route["z0_ohm"]),
        epsr=epsr,
        alpha_db_per_m_100mhz=float(route["alpha_db_per_m_100mhz"]),
        sigma_dielectric=float(route["sigma_dielectric"]),
        tan_delta_100mhz=tan_delta,
    )
    return seg, r_extra_per_m


# ═══════════════════════════════════════════════
# V2.7 RLGC 传播核 → 单段 ABCD
# ═══════════════════════════════════════════════

def v27_segment_abcd(frequency_hz: np.ndarray, seg, r_extra_per_m: float = 0.0):
    """逐行复刻 DG_V2.7 core/s11_generator.py::_compute_s11_for_cable 的单段
    频变 RLGC 计算（几何反解、趋肤效应 R、tanδ 介损、Debye 项），把递归输入
    阻抗替换为 ABCD 参数，以便与 CST 场景结构级联。r_extra_per_m 是本实验
    注入的额外常数串联电阻（仅 local_R 工况使用）。
    """
    freq = np.asarray(frequency_hz, dtype=np.float64)
    omega = 2.0 * GEN.PI * freq

    epsr_ref = max(float(seg.epsr), 1.05)
    log_term = max(float(seg.z0_ohm) * math.sqrt(epsr_ref) / 60.0, 1e-6)

    tan_delta = max(float(getattr(seg, "tan_delta_100mhz", 0.0)), 0.0)
    debye_delta = max(float(getattr(seg, "debye_delta_epsr", 0.0)), 0.0)
    debye_corner = max(float(getattr(seg, "debye_corner_hz", 80e6)), 1.0)
    debye_exp = float(np.clip(getattr(seg, "debye_exponent", 1.0), 0.55, 1.35))

    x_ref = (REFERENCE_HZ / debye_corner) ** debye_exp
    eps_debye_ref = debye_delta / complex(1.0, x_ref)
    effective_tan_ref = tan_delta
    if eps_debye_ref.real + epsr_ref > 0:
        effective_tan_ref += max(-eps_debye_ref.imag / (epsr_ref + eps_debye_ref.real), 0.0)
    beta_ref = 2.0 * GEN.PI * REFERENCE_HZ * math.sqrt(epsr_ref + max(eps_debye_ref.real, 0.0)) / C0
    alpha_dielectric_db = 8.686 * 0.5 * beta_ref * effective_tan_ref
    conductor_alpha_target = max(float(seg.alpha_db_per_m_100mhz) - alpha_dielectric_db, 2.0e-4)

    rc_m, rs_m = GEN._target_to_geometry(seg.z0_ohm, epsr_ref, conductor_alpha_target)
    skin_depth = np.sqrt(2.0 / np.maximum(omega * GEN.MU0 * seg.sigma_cu, 1e-30))

    debye_den = 1.0 + 1j * np.power(np.maximum(freq, 0.0) / debye_corner, debye_exp)
    eps_complex = epsr_ref + debye_delta / debye_den
    c_complex = 2.0 * GEN.PI * GEN.EPS0 * eps_complex / log_term
    c_real = 2.0 * GEN.PI * GEN.EPS0 * np.maximum(eps_complex.real, 1.01) / log_term
    g_conductive = 2.0 * GEN.PI * float(seg.sigma_dielectric) / log_term
    g_loss = omega * c_real * tan_delta

    r_skin = (1.0 / (2.0 * GEN.PI * rc_m * skin_depth * seg.sigma_cu)
              + 1.0 / (2.0 * GEN.PI * rs_m * skin_depth * seg.sigma_cu))
    R = r_skin + r_extra_per_m
    L = (GEN.MU0 / (2.0 * GEN.PI) * log_term
         + GEN.MU0 / (4.0 * GEN.PI) * (1.0 / rc_m + 1.0 / rs_m) * skin_depth)

    Z = R + 1j * omega * L
    Y = g_conductive + g_loss + 1j * omega * c_complex
    z0 = np.sqrt(Z / Y)
    gamma = np.sqrt(Z * Y)

    gl = gamma * float(seg.length_m)
    cosh_gl = np.cosh(gl)
    sinh_gl = np.sinh(gl)
    return cosh_gl, z0 * sinh_gl, sinh_gl / z0, cosh_gl


def simulate_route(route: dict, frequency_hz: np.ndarray, case_name: str) -> np.ndarray:
    """V2.7 路线传播核 + 与 CodeX 相同的 CST 场景结构（TL1 / 接头 / TL300 / 末端）。"""
    case = case_by_name(case_name)
    network = identity(frequency_hz.size)
    network = V1.cascade(network, V1.lossless_transmission_line(frequency_hz, 1.0, 135.0))
    for start_m, end_m in zip(SEGMENT_BOUNDARIES[:-1], SEGMENT_BOUNDARIES[1:]):
        if end_m <= start_m:
            continue
        seg, r_extra = route_segment(route, case, start_m, end_m)
        network = V1.cascade(network, v27_segment_abcd(frequency_hz, seg, r_extra))
        if np.isclose(end_m, 10.0):
            network = V1.cascade(network, V1.special_joint1(frequency_hz))
        elif np.isclose(end_m, 20.0) or np.isclose(end_m, 30.0):
            network = V1.cascade(network, V1.lossless_transmission_line(frequency_hz, 0.5, 300.0))
    network = V1.cascade(network, V1.terminal_shunt_element(frequency_hz, case))
    return V1.network_to_s11(network)


# ═══════════════════════════════════════════════
# 统一测量层（与 CodeX 完全一致的误差模型）
# ═══════════════════════════════════════════════

def correlated_noise(size: int, correlation: float, seed: int):
    rng = np.random.default_rng(seed)
    white = (rng.standard_normal(size) + 1j * rng.standard_normal(size)) / math.sqrt(2.0)
    out = np.empty(size, dtype=np.complex128)
    out[0] = white[0]
    innovation = math.sqrt(1.0 - correlation**2)
    for index in range(1, size):
        out[index] = correlation * out[index - 1] + innovation * white[index]
    return out


def apply_common_measurement(s11: np.ndarray, frequency_hz: np.ndarray, cfg: dict, seed: int):
    p = cfg["measurement"]
    f_norm = frequency_hz / REFERENCE_HZ
    two_way_loss = np.power(10.0, -(2.0 * float(p["loss_db_at_100mhz"]) * f_norm) / 20.0)
    delay = float(p["delay_ns"]) * 1.0e-9
    gamma = s11 * two_way_loss * np.exp(-1j * 4.0 * np.pi * frequency_hz * delay)
    e00 = 0.5 * float(p["directivity"]) * (1.0 + 1j)
    e11 = 0.5 * float(p["source_match"]) * (1.0 - 1j)
    tracking = float(p["tracking"]) * np.exp(1j * float(p["tracking_phase_rad"]))
    measured = e00 + tracking * gamma / (1.0 - e11 * gamma)
    scale = float(p["noise_sigma"]) * (1.0 + 0.35 / (1.0 + np.sqrt(frequency_hz / REFERENCE_HZ)))
    return measured + scale * correlated_noise(len(s11), float(p["noise_correlation"]), seed)


# ═══════════════════════════════════════════════
# 指标（沿用 CodeX 的方向性检查）
# ═══════════════════════════════════════════════

def window_values(distance, values, low, high):
    distance = np.asarray(distance)
    values = np.asarray(values)
    mask = (distance >= low) & (distance <= high) & np.isfinite(values)
    return values[mask]


def positive_peak(distance, values, low=37.0, high=60.0):
    distance = np.asarray(distance)
    values = np.asarray(values)
    mask = (distance >= low) & (distance <= high) & np.isfinite(values)
    idx = np.flatnonzero(mask)
    if not len(idx):
        return math.nan, math.nan
    pick = idx[int(np.argmax(values[idx]))]
    return float(distance[pick]), float(values[pick])


def signed_peak(distance, values, low=12.0, high=18.0):
    distance = np.asarray(distance)
    values = np.asarray(values)
    mask = (distance >= low) & (distance <= high) & np.isfinite(values)
    idx = np.flatnonzero(mask)
    if not len(idx):
        return math.nan, math.nan
    pick = idx[int(np.argmax(np.abs(values[idx])))]
    return float(distance[pick]), float(values[pick])


def metrics_for(route_name: str, layer: str, case_name: str, run: dict, baseline: dict):
    a1, base = run["a1"], baseline["a1"]
    d = np.asarray(a1["distance_m"])
    scale = max(float(base["step_scale"]), 1e-30)
    step = np.asarray(a1["step_raw"]) / scale
    base_step = np.asarray(base["step_raw"]) / scale
    impulse = np.asarray(a1["impulse_real_raw"]) / scale
    base_impulse = np.asarray(base["impulse_real_raw"]) / scale
    delta_step = step - base_step
    delta_impulse = impulse - base_impulse
    step_window = window_values(d, delta_step, 16.0, 35.0)
    slope_mask = (d >= 16.0) & (d <= 35.0)
    slope = float(np.polyfit(d[slope_mask], step[slope_mask], 1)[0])
    base_slope = float(np.polyfit(d[slope_mask], base_step[slope_mask], 1)[0])
    terminal_pos, terminal_amp = positive_peak(d, impulse)
    base_terminal_pos, base_terminal_amp = positive_peak(d, base_impulse)
    local_pos, local_amp = signed_peak(d, delta_impulse)
    terminal_ratio = abs(terminal_amp) / max(abs(base_terminal_amp), 1e-30)
    terminal_shift = terminal_pos - base_terminal_pos
    step_delta = float(np.median(step_window)) if len(step_window) else math.nan
    checks = {}
    if case_name == "baseline":
        checks["passive_clean"] = bool(layer != "clean" or np.max(np.abs(run["s11"])) <= 1.0 + 1e-9)
    elif case_name == "overall_C20pF":
        checks = {"step_lower": step_delta < 0, "terminal_later": terminal_shift > 0,
                  "terminal_amp_near": 0.75 <= terminal_ratio <= 1.25}
    elif case_name == "overall_G20k":
        checks = {"step_lower": step_delta < 0, "terminal_reduced": terminal_ratio < 1,
                  "terminal_position": abs(terminal_shift) <= 1.0}
    elif case_name == "segmented_loss":
        checks = {"terminal_reduced": terminal_ratio < 1, "slope_not_increased": slope <= base_slope}
    elif case_name == "local_C32pF_15m":
        checks = {"local_negative": local_amp < 0, "terminal_amp_near": 0.75 <= terminal_ratio <= 1.25,
                  "terminal_not_earlier": terminal_shift >= -0.26}
    elif case_name == "local_G2k_15m":
        checks = {"step_lower": step_delta < 0, "terminal_reduced": terminal_ratio < 1,
                  "terminal_position": abs(terminal_shift) <= 1.0}
    elif case_name == "local_C4pF_15m":
        checks = {"local_positive": local_amp > 0, "terminal_amp_near": 0.75 <= terminal_ratio <= 1.25,
                  "terminal_not_later": terminal_shift <= 0.26}
    elif case_name.startswith("local_R"):
        checks = {"step_higher": step_delta > 0, "terminal_reduced": terminal_ratio < 1,
                  "terminal_position": abs(terminal_shift) <= 1.0}
    return {
        "route": route_name,
        "layer": layer,
        "case": case_name,
        "max_abs_s11": float(np.max(np.abs(run["s11"]))),
        "s11_rms_delta": float(np.sqrt(np.mean(np.abs(run["s11"] - baseline["s11"]) ** 2))),
        "terminal_position_m": terminal_pos,
        "terminal_shift_m": terminal_shift,
        "terminal_amplitude_ratio": terminal_ratio,
        "local_delta_peak_position_m": local_pos,
        "local_delta_peak_amplitude": local_amp,
        "step_delta_median_16_35m": step_delta,
        "step_slope_16_35m": slope,
        "checks": checks,
        "case_pass": bool(all(checks.values())),
    }


# ═══════════════════════════════════════════════
# 绘图（与 CodeX comparison_board 同版式）
# ═══════════════════════════════════════════════

def configure_plot():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["SimHei", "Microsoft YaHei", "Arial Unicode MS"],
        "axes.unicode_minus": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.spines.top": True,
        "axes.spines.right": True,
    })


def normalized_pair(run: dict, baseline: dict, algorithm: int, kind: str):
    if algorithm == 1:
        d = np.asarray(run["a1"]["distance_m"])
        key = "step_raw" if kind == "step" else "impulse_real_raw"
        y = np.asarray(run["a1"][key])
        y0 = np.asarray(baseline["a1"][key])
    else:
        dkey = "distance_m" if kind == "step" else "distance_impulse_m"
        key = "step_smoothed" if kind == "step" else "impulse_smoothed"
        d = np.asarray(run["a2"][dkey])
        y = np.asarray(run["a2"][key])
        y0 = np.asarray(baseline["a2"][key])
    mask = d <= 60.0
    center = float(np.median(y0[d <= 5.0])) if kind == "step" else 0.0
    scale = max(float(np.max(np.abs(np.concatenate((y[mask] - center, y0[mask] - center))))), 1e-30)
    return d, (y - center) / scale, (y0 - center) / scale


def plot_board(path: Path, case_name: str, layer: str, frequency_hz: np.ndarray,
               runs: dict, route_names: tuple, labels: dict, colors: dict, reference_path: Path | None):
    configure_plot()
    case = case_by_name(case_name)
    fig = plt.figure(figsize=(15.9, 17.2), dpi=200)
    grid = fig.add_gridspec(4, 6, height_ratios=(1.0, 1.05, 1.05, 1.25), hspace=0.34, wspace=0.34)
    axes = [fig.add_subplot(grid[0, 0:2]), fig.add_subplot(grid[0, 2:4]), fig.add_subplot(grid[0, 4:6]),
            fig.add_subplot(grid[1, 0:3]), fig.add_subplot(grid[1, 3:6]),
            fig.add_subplot(grid[2, 0:3]), fig.add_subplot(grid[2, 3:6])]
    base_runs = {name: runs[name]["baseline"] for name in route_names}
    case_runs = {name: runs[name][case_name] for name in route_names}
    for name in route_names:
        color = colors[name]
        s11 = case_runs[name]["s11"]
        axes[0].plot(frequency_hz / 1e6, s11.real, color=color, lw=0.85, label=labels[name])
        axes[1].plot(frequency_hz / 1e6, np.abs(s11), color=color, lw=0.85)
        axes[2].plot(frequency_hz / 1e6, np.angle(s11, deg=True), color=color, lw=0.75)
        for axis, algorithm, kind in ((axes[3], 1, "step"), (axes[4], 1, "impulse"),
                                      (axes[5], 2, "step"), (axes[6], 2, "impulse")):
            d, y, y0 = normalized_pair(case_runs[name], base_runs[name], algorithm, kind)
            mask = d <= 60.0
            if case_name != "baseline":
                axis.plot(d[mask], y0[mask], color=color, lw=0.65, alpha=0.35, ls="--")
            axis.plot(d[mask], y[mask], color=color, lw=0.95)
    titles = ("S11 实部", "S11 幅值", "S11 相位（包裹）",
              "算法1 阶跃 / 归一化", "算法1 脉冲 / 归一化",
              "算法2 阶跃 / 归一化", "算法2 脉冲 / 归一化")
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
    ref_axis = fig.add_subplot(grid[3, 1:5])
    ref_axis.axis("off")
    if reference_path is not None and reference_path.exists():
        ref_axis.imshow(plt.imread(reference_path), aspect="equal")
        ref_axis.set_title("报告对应 CST 图")
    else:
        ref_axis.text(0.5, 0.5, "基准工况（报告无对应 CST 图）",
                      ha="center", va="center", fontsize=12, transform=ref_axis.transAxes)
    fig.suptitle(f"V2.7路线还原：{case.title}（{layer}，距离轴0–60 m）", fontsize=15, y=0.994)
    fig.text(0.5, 0.006, "实线为当前工况；同色虚线为该路线基线。定量判定使用未归一化数组。",
             ha="center", fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════
# 输出与报告
# ═══════════════════════════════════════════════

def save_s11(path: Path, frequency_hz: np.ndarray, s11: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("Frequency_Hz", "S11_Real", "S11_Imag"))
        writer.writerows(zip(frequency_hz, s11.real, s11.imag))


def save_response(path: Path, a1: dict, a2: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "algorithm1_distance_m": np.asarray(a1["distance_m"]),
        "algorithm1_impulse": np.asarray(a1["impulse_real_raw"]),
        "algorithm1_step": np.asarray(a1["step_raw"]),
        "algorithm2_distance_m": np.asarray(a2["distance_m"]),
        "algorithm2_step": np.asarray(a2["step_smoothed"]),
        "algorithm2_impulse_distance_m": np.asarray(a2["distance_impulse_m"]),
        "algorithm2_impulse": np.asarray(a2["impulse_smoothed"]),
    }
    np.savez(path, **payload)


def write_report(path: Path, cfg: dict, metrics: list[dict], runtime_s: float, route_names: tuple, labels: dict):
    cases = cfg["cases"]
    lines = [
        "# V2.7 路线还原报告仿真章节 9 工况",
        "",
        f"- 运行时间：{runtime_s:.2f} s",
        f"- 统一频率网格：9 kHz–200 MHz，{cfg['frequency_grid']['points']}点",
        "- S11 为唯一权威信号；算法1/算法2 结果均由同一 S11 重新计算。",
        "- 每条路线 = DG V2.7 core RLGC 传播核（几何反解 + 趋肤效应 R + tanδ 介损）+ 报告 CST 场景结构"
        "（TL1 135Ω×1 m、10 m 特殊接头、20/30 m 后 300Ω×0.5 m 线、末端 16 pF‖200 kΩ 支路），"
        "与 CodeX/DG_Route_Evaluation 中 dg_v3 候选的组装方式同构。",
        "",
        "## 1. DG V2.7 是否基于 RLGC 生成：是",
        "",
        "`DG_V2.7/core/s11_generator.py::_compute_s11_for_cable` 对每个 `SegmentParams`：",
        "1. 由 (Z0, epsr, alpha@100MHz) 二分反解同轴几何 (rc, rs)（`_target_to_geometry`）；",
        "2. 按频率计算趋肤深度下的 R(∝√f)、L（含内电感）、C（复数，含 Debye 弛豫）、"
        "G（σ介 + ωC·tanδ）；",
        "3. 由 Z=sqrt(Z/Y)、γ=sqrt(ZY) 逐段从末端向首端递推输入阻抗，最后对 50 Ω 参考取 S11。",
        "即 S11 完全由频变 RLGC 传输线模型（电报方程）级联生成，不存在距离域回写。",
        "",
        "## 2. 路线定义",
        "",
        "| 路线 | Z0 (Ω) | epsr | alpha @100MHz (dB/m) | tanδ @100MHz |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in route_names:
        r = cfg["routes"][name]
        lines.append(f"| {labels[name]} | {r['z0_ohm']} | {r['epsr']} | {r['alpha_db_per_m_100mhz']} | {r['tan_delta_100mhz']} |")
    lines += [
        "",
        "## 3. 工况映射（报告 9 工况 → V2.7 参数空间）",
        "",
        "| 工况 | 报告参数变化 | V2.7 映射 |",
        "|---|---|---|",
        "| 基准 | 全部标准参数 | 路线基准段参数 |",
        "| 整体电容增加 | C:16→20 pF | epsr ×1.25（全段） |",
        "| 整体损耗增加 | G:1/200k→1/20k | tanδ ×10（全段） |",
        "| 分段损耗 | 200k/20k/200k/2k | tanδ ×1/×10/×1/×100（按 0-10/10-20/20-30/30-40 m 分段） |",
        "| 局部电容增加 | 15 m 单元 C:16→32 pF | 14.8–15.2 m 段 epsr ×2 |",
        "| 局部损耗增加 | 15 m 单元 G:1/200k→1/2k | 14.8–15.2 m 段 tanδ ×100 |",
        "| 局部电容减小 | 15 m 单元 C:16→4 pF | 14.8–15.2 m 段 epsr ×0.25（V2.7 内部下限 1.05，实际约 ×0.47） |",
        "| 局部串联电阻增加 | 15 m 单元 R:0.01→10 Ω | 14.8–15.2 m 段注入常数串联电阻 25 Ω/m |",
        "| 局部串联电阻增加 | 15 m 单元 R:0.01→50 Ω | 14.8–15.2 m 段注入常数串联电阻 125 Ω/m |",
        "",
        "## 4. 已知模型差异（路线本身的表达能力边界）",
        "",
        "1. V2.7 的并联损耗以 tanδ 表示（G ∝ f），报告 CST 网表为常数 G；两者在 100 MHz 锚定后低频差异明显。",
        "2. V2.7 的导体损耗是几何反解的趋肤 R（∝ √f）；local_R 工况所需的常数串联 R 无法用该参数空间表示，"
        "已作为最小扩展直接加进该段单位长度串联阻抗。",
        "3. local_C4pF 工况要求 epsr 0.56（<1），物理同轴不可实现，V2.7 内部下限 1.05 使该工况欠强。",
        "4. RG58 路线在 V2.7 中还有 BNC 接头随机反射注入与噪声层，属于随机真实感层，"
        "为保证 9 工况确定性对比，本实验未启用（与 CodeX 的 clean 层做法一致）。",
        "",
        "## 5. 工况方向检查",
        "",
        "| 路线 | clean 通过数/9 | common_measurement 通过数/9 | clean 最大|S11| |",
        "|---|---:|---:|---:|",
    ]
    for name in route_names:
        clean = [m for m in metrics if m["route"] == name and m["layer"] == "clean"]
        common = [m for m in metrics if m["route"] == name and m["layer"] == "common_measurement"]
        lines.append(f"| {labels[name]} | {sum(m['case_pass'] for m in clean)}/9 | "
                     f"{sum(m['case_pass'] for m in common)}/9 | "
                     f"{max(m['max_abs_s11'] for m in clean):.6g} |")
    lines += [
        "",
        "逐工况原始指标和布尔检查见 `output/metrics.csv` 与 `output/metrics.json`。",
        "",
        "## 6. 输出图",
        "",
    ]
    for layer in ("clean", "common_measurement"):
        lines.append(f"### {layer}")
        lines.append("")
        for case_name in cases:
            lines.append(f"![{layer}-{case_name}](assets/comparison_board_{layer}_{case_name}.png)")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ═══════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="V2.7 RG58/Field 路线还原报告9工况")
    parser.add_argument("--config", type=Path, default=HERE / "evaluation_cases.yaml")
    parser.add_argument("--output", type=Path, default=HERE / "output")
    parser.add_argument("--assets", type=Path, default=HERE / "assets")
    parser.add_argument("--report", type=Path, default=HERE / "V27路线还原对比报告.md")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    point_count = 600 if args.smoke else int(cfg["frequency_grid"]["points"])
    time_points = 600 if args.smoke else int(cfg["fdr"]["algorithm2_time_points"])
    frequency_hz = np.linspace(float(cfg["frequency_grid"]["start_hz"]),
                               float(cfg["frequency_grid"]["stop_hz"]), point_count)
    route_names = tuple(cfg["routes"].keys())
    labels = {name: cfg["routes"][name]["label"] for name in route_names}
    colors = {name: cfg["routes"][name]["color"] for name in route_names}

    output = args.output.resolve()
    assets = args.assets.resolve()
    start = time.perf_counter()
    all_runs = {name: {} for name in route_names}
    metrics = []
    total_runs = len(route_names) * len(cfg["cases"])
    done = 0
    for route_index, route_name in enumerate(route_names, start=1):
        route = cfg["routes"][route_name]
        print(f"\n[{route_index}/{len(route_names)}] {labels[route_name]}")
        for case_index, case_name in enumerate(cfg["cases"], start=1):
            clean = simulate_route(route, frequency_hz, case_name)
            common = apply_common_measurement(clean, frequency_hz, cfg,
                                              int(cfg["measurement"]["seed"]) + case_index)
            for layer, s11 in (("clean", clean), ("common_measurement", common)):
                a1 = V1.algorithm1_ifft(frequency_hz, s11)
                with contextlib.redirect_stdout(io.StringIO()):
                    a2 = V1.algorithm2_fdr(
                        frequency_hz, s11, time_points=time_points,
                        chunk_size=int(cfg["fdr"]["algorithm2_chunk_size"]),
                        progress_label="",
                    )
                run = {"s11": s11, "a1": a1, "a2": a2}
                all_runs[route_name].setdefault(case_name, {})[layer] = run
                tag = "smoke_" if args.smoke else ""
                save_s11(output / "s11" / layer / route_name / f"{tag}s11_{case_name}.csv",
                         frequency_hz, s11)
                if not args.smoke:
                    save_response(output / "responses" / layer / route_name / f"response_{case_name}.npz",
                                  a1, a2)
            done += 1
            print(f"  [{done}/{total_runs}] {case_name} 完成 "
                  f"(耗时 {time.perf_counter() - start:.1f} s)")

    for route_name in route_names:
        for layer in ("clean", "common_measurement"):
            baseline = all_runs[route_name]["baseline"][layer]
            for case_name in cfg["cases"]:
                run = all_runs[route_name][case_name][layer]
                item = metrics_for(route_name, layer, case_name, run, baseline)
                run["metrics"] = item
                metrics.append(item)

    if not args.smoke:
        for layer in ("clean", "common_measurement"):
            layer_runs = {name: {case: all_runs[name][case][layer] for case in cfg["cases"]}
                          for name in route_names}
            for case_name in cfg["cases"]:
                reference = case_by_name(case_name).reference_image
                reference_path = (REPORT_REFERENCE_DIR / reference) if reference else None
                plot_board(assets / f"comparison_board_{layer}_{case_name}.png", case_name, layer,
                           frequency_hz, layer_runs, route_names, labels, colors, reference_path)

    output.mkdir(parents=True, exist_ok=True)
    flat_rows = []
    for item in metrics:
        row = {key: value for key, value in item.items() if key != "checks"}
        row["checks"] = json.dumps(item["checks"], ensure_ascii=False, sort_keys=True)
        flat_rows.append(row)
    with (output / ("metrics_smoke.csv" if args.smoke else "metrics.csv")).open(
            "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    runtime_s = time.perf_counter() - start
    summary = {
        "protocol_note": "每条路线 = DG V2.7 core 频变RLGC传播核 + 报告CST场景结构；参考路径只读",
        "runtime_s": runtime_s,
        "smoke": bool(args.smoke),
        "metrics": metrics,
    }
    (output / ("summary_smoke.json" if args.smoke else "summary.json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.smoke:
        write_report(args.report.resolve(), cfg, metrics, runtime_s, route_names, labels)
    print(f"\n完成：{len(metrics)}条指标，{'冒烟' if args.smoke else str(2 * len(cfg['cases'])) + '张工况图'}，"
          f"耗时 {runtime_s:.2f} s")


if __name__ == "__main__":
    main()
