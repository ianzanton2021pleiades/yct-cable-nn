"""Independent DG-route comparison; existing DG/CST sources stay read-only."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.dont_write_bytecode = True

import matplotlib.pyplot as plt
import numpy as np
import yaml


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V0_PATH = ROOT / "DifferentAgent/CodeX/CST_Reproduction/V0/cst_fdr_reproduction.py"
V1_PATH = ROOT / "DifferentAgent/CodeX/CST_Reproduction/V1/cst_fdr_reproduction.py"
DG_V3_PATH = ROOT / "DG_Update/DG_V3"
REAL_ROOT = Path(r"E:\FDR案例-csv")
C0 = 299_792_458.0
REFERENCE_HZ = 100.0e6

CANDIDATES = (
    "dg_v3_rlgc",
    "cst_v0_ladder_0p4m",
    "cst_v1_ladder_0p1m",
    "cst_v1_continuous",
)
LABELS = {
    "dg_v3_rlgc": "DG V3 RLGC (100 MHz锚定)",
    "cst_v0_ladder_0p4m": "CST V0 0.4 m梯形",
    "cst_v1_ladder_0p1m": "CST V1 0.1 m梯形",
    "cst_v1_continuous": "CST V1 连续固定RLGC",
}
COLORS = {
    "dg_v3_rlgc": "#1f77b4",
    "cst_v0_ladder_0p4m": "#d97706",
    "cst_v1_ladder_0p1m": "#7c3aed",
    "cst_v1_continuous": "#334155",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入 {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V0 = load_module("dg_route_cst_v0", V0_PATH)
V1 = load_module("dg_route_cst_v1", V1_PATH)
sys.path.insert(0, str(DG_V3_PATH))
from dg_v3.physics import segment_abcd  # noqa: E402
from dg_v3.topology import CableSegment, RLGCMaterial  # noqa: E402


def case_by_name(module, name: str):
    return next(item for item in module.CASES if item.name == name)


def identity(size: int):
    one = np.ones(size, dtype=np.complex128)
    zero = np.zeros(size, dtype=np.complex128)
    return one, zero, zero, one


def anchored_segment(
    frequency_hz: np.ndarray,
    start_m: float,
    end_m: float,
    resistance_per_m: float,
    inductance_per_m: float,
    conductance_per_m: float,
    capacitance_per_m: float,
):
    """Build a DG V3 effective-RLGC segment matching primary RLGC at 100 MHz."""
    omega_ref = 2.0 * math.pi * REFERENCE_HZ
    external_l = inductance_per_m - resistance_per_m / omega_ref
    if external_l <= 0.0:
        raise ValueError("100 MHz锚定后外部电感非正")
    z0 = math.sqrt(external_l / capacitance_per_m)
    velocity = 1.0 / math.sqrt(external_l * capacitance_per_m)
    epsr = (C0 / velocity) ** 2
    material = RLGCMaterial(
        model="effective_rlgc",
        conductor_conductivity_s_per_m=0.0,
        shunt_conductance_s_per_m=conductance_per_m,
        series_resistance_100mhz_ohm_per_m=resistance_per_m,
    )
    segment = CableSegment(
        start_m=start_m,
        end_m=end_m,
        z0_ohm=z0,
        epsr=epsr,
        alpha_db_per_m_at_100mhz=0.0,
        tan_delta_at_100mhz=0.0,
        material=material,
    )
    return segment_abcd(frequency_hz, segment)


def dg_case_primary(case_name: str, start_m: float, end_m: float):
    r, l, g, c = 0.025, 0.75e-6, 12.5e-6, 40.0e-12
    center = 0.5 * (start_m + end_m)
    local = start_m < 15.2 and end_m > 14.8
    if case_name == "overall_C20pF":
        c *= 1.25
    elif case_name == "overall_G20k":
        g *= 10.0
    elif case_name == "segmented_loss":
        g *= 1.0 if center < 10.0 else 10.0 if center < 20.0 else 1.0 if center < 30.0 else 100.0
    elif local and case_name == "local_C32pF_15m":
        c *= 2.0
    elif local and case_name == "local_G2k_15m":
        g *= 100.0
    elif local and case_name == "local_C4pF_15m":
        c *= 0.25
    elif local and case_name == "local_R10ohm_15m":
        r = 25.0
    elif local and case_name == "local_R50ohm_15m":
        r = 125.0
    return r, l, g, c


def simulate_dg_v3_matched(frequency_hz: np.ndarray, case_name: str) -> np.ndarray:
    """DG V3 propagation kernel with the same CST external joints and terminal."""
    case = case_by_name(V1, case_name)
    network = identity(frequency_hz.size)
    network = V1.cascade(network, V1.lossless_transmission_line(frequency_hz, 1.0, 135.0))
    boundaries = (0.0, 10.0, 14.8, 15.2, 20.0, 30.0, 40.0)
    for start_m, end_m in zip(boundaries[:-1], boundaries[1:]):
        if end_m <= start_m:
            continue
        r, l, g, c = dg_case_primary(case_name, start_m, end_m)
        network = V1.cascade(
            network,
            anchored_segment(frequency_hz, start_m, end_m, r, l, g, c),
        )
        if np.isclose(end_m, 10.0):
            network = V1.cascade(network, V1.special_joint1(frequency_hz))
        elif np.isclose(end_m, 20.0) or np.isclose(end_m, 30.0):
            network = V1.cascade(
                network, V1.lossless_transmission_line(frequency_hz, 0.5, 300.0)
            )
    network = V1.cascade(network, V1.terminal_shunt_element(frequency_hz, case))
    return V1.network_to_s11(network)


def simulate_clean(candidate: str, frequency_hz: np.ndarray, case_name: str) -> np.ndarray:
    if candidate == "dg_v3_rlgc":
        return simulate_dg_v3_matched(frequency_hz, case_name)
    if candidate == "cst_v0_ladder_0p4m":
        return V0.simulate_s11(frequency_hz, case_by_name(V0, case_name))
    if candidate == "cst_v1_ladder_0p1m":
        return V1.simulate_s11(frequency_hz, case_by_name(V1, case_name), "fixed_ladder_0p1m")
    if candidate == "cst_v1_continuous":
        return V1.simulate_s11(frequency_hz, case_by_name(V1, case_name), "fixed_continuous")
    raise ValueError(candidate)


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


def metrics_for(candidate: str, layer: str, case_name: str, run: dict, baseline: dict):
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
        checks = {"step_lower": step_delta < 0, "terminal_later": terminal_shift > 0, "terminal_amp_near": 0.75 <= terminal_ratio <= 1.25}
    elif case_name == "overall_G20k":
        checks = {"step_lower": step_delta < 0, "terminal_reduced": terminal_ratio < 1, "terminal_position": abs(terminal_shift) <= 1.0}
    elif case_name == "segmented_loss":
        checks = {"terminal_reduced": terminal_ratio < 1, "slope_not_increased": slope <= base_slope}
    elif case_name == "local_C32pF_15m":
        checks = {"local_negative": local_amp < 0, "terminal_amp_near": 0.75 <= terminal_ratio <= 1.25, "terminal_not_earlier": terminal_shift >= -0.26}
    elif case_name == "local_G2k_15m":
        checks = {"step_lower": step_delta < 0, "terminal_reduced": terminal_ratio < 1, "terminal_position": abs(terminal_shift) <= 1.0}
    elif case_name == "local_C4pF_15m":
        checks = {"local_positive": local_amp > 0, "terminal_amp_near": 0.75 <= terminal_ratio <= 1.25, "terminal_not_later": terminal_shift <= 0.26}
    elif case_name.startswith("local_R"):
        checks = {"step_higher": step_delta > 0, "terminal_reduced": terminal_ratio < 1, "terminal_position": abs(terminal_shift) <= 1.0}
    return {
        "candidate": candidate,
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


def plot_board(path: Path, case_name: str, layer: str, frequency_hz: np.ndarray, runs: dict, reference_path: Path | None):
    configure_plot()
    fig = plt.figure(figsize=(15.9, 17.2), dpi=200)
    grid = fig.add_gridspec(4, 6, height_ratios=(1.0, 1.05, 1.05, 1.25), hspace=0.34, wspace=0.34)
    axes = [fig.add_subplot(grid[0, 0:2]), fig.add_subplot(grid[0, 2:4]), fig.add_subplot(grid[0, 4:6]),
            fig.add_subplot(grid[1, 0:3]), fig.add_subplot(grid[1, 3:6]),
            fig.add_subplot(grid[2, 0:3]), fig.add_subplot(grid[2, 3:6])]
    base_runs = {name: runs[name]["baseline"] for name in CANDIDATES}
    case_runs = {name: runs[name][case_name] for name in CANDIDATES}
    for name in CANDIDATES:
        color = COLORS[name]
        s11 = case_runs[name]["s11"]
        axes[0].plot(frequency_hz / 1e6, s11.real, color=color, lw=0.85, label=LABELS[name])
        axes[1].plot(frequency_hz / 1e6, np.abs(s11), color=color, lw=0.85)
        axes[2].plot(frequency_hz / 1e6, np.angle(s11, deg=True), color=color, lw=0.75)
        for axis, algorithm, kind in ((axes[3], 1, "step"), (axes[4], 1, "impulse"), (axes[5], 2, "step"), (axes[6], 2, "impulse")):
            d, y, y0 = normalized_pair(case_runs[name], base_runs[name], algorithm, kind)
            mask = d <= 60.0
            if case_name != "baseline":
                axis.plot(d[mask], y0[mask], color=color, lw=0.65, alpha=0.35, ls="--")
            axis.plot(d[mask], y[mask], color=color, lw=0.95)
    titles = ("S11 实部", "S11 幅值", "S11 相位（包裹）", "算法1 阶跃 / 归一化", "算法1 脉冲 / 归一化", "算法2 阶跃 / 归一化", "算法2 脉冲 / 归一化")
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
        rows = []
        for name in CANDIDATES:
            rows.append([LABELS[name], "通过" if runs[name][case_name]["metrics"]["case_pass"] else "未通过"])
        table = ref_axis.table(cellText=rows, colLabels=("候选", "工况方向检查"), loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.5)
    title = case_by_name(V1, case_name).title
    fig.suptitle(f"DG路线对比：{title}（{layer}，距离轴0–60 m）", fontsize=15, y=0.994)
    fig.text(0.5, 0.006, "实线为当前工况；同色虚线为该候选基线。定量判定使用未归一化数组。", ha="center", fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def inventory_real_data():
    files = list(REAL_ROOT.rglob("*.csv")) if REAL_ROOT.exists() else []
    def count(token):
        token = token.lower()
        return sum(token in str(path).lower() for path in files)
    return {
        "as_of": time.strftime("%Y-%m-%d %H:%M:%S"),
        "root": str(REAL_ROOT),
        "total_csv": len(files),
        "ifft_csv": count("ifft现场数据汇总"),
        "unscaled_s11_csv": count("无校准s11"),
        "rg58_74m_csv": count("rg58-74m(40+4+30)"),
        "rg58_3lines_csv": count("rg58-3lines"),
        "rg58_defect_csv": count("rg58缺陷制造实验"),
        "wet_1500m_csv": count("1500m电缆单个接头浸水试验"),
    }


def write_report(path: Path, cfg: dict, metrics: list[dict], runtime_s: float, inventory: dict):
    lines = [
        "# DG 路线对比评估报告",
        "",
        f"- 运行时间：{runtime_s:.2f} s",
        f"- 统一频率网格：9 kHz–200 MHz，{cfg['bands']['points']}点" if "bands" in cfg else f"- 统一频率网格：9 kHz–200 MHz，{cfg['frequency_grid']['points']}点",
        "- S11为唯一权威信号；算法1/2结果均由同一S11重新计算。",
        "- V2.7只作历史真实感参照，不参与路线获胜判定。",
        "",
        "## 1. 候选与公平性",
        "",
        "DG V3在九工况中使用其`effective_rlgc`传播核，并在100 MHz锚定CST每米R/L/G/C；四路共用CST接头、端点和频率网格。DG V3的R随平方根频率变化，而CST的R固定，因此差异属于模型路线本身。统一测量层给四路施加完全相同的延迟、损耗、VNA误差和相关噪声。",
        "",
        "## 2. 工况硬门槛",
        "",
        "| 候选 | clean通过数/9 | common-measurement通过数/9 | clean最大|S11| |",
        "|---|---:|---:|---:|",
    ]
    for candidate in CANDIDATES:
        clean = [m for m in metrics if m["candidate"] == candidate and m["layer"] == "clean"]
        common = [m for m in metrics if m["candidate"] == candidate and m["layer"] == "common_measurement"]
        lines.append(f"| {LABELS[candidate]} | {sum(m['case_pass'] for m in clean)}/9 | {sum(m['case_pass'] for m in common)}/9 | {max(m['max_abs_s11'] for m in clean):.6g} |")
    lines += ["", "逐工况原始指标和布尔检查见`metrics.csv`与`metrics.json`。", "", "## 3. 已确认的路线证据", "",
              "1. CST V0的0.4 m梯形在约145 MHz出现离散截止，算法1接头后尾波/主峰约0.159；该纹理是离散伪影，不是真实制造不均匀。",
              "2. CST V1 0.1 m梯形把截止推到约581 MHz、尾波降至约0.00362，但相对连续模型的全频S11和脉冲误差仍未达到既有0.05阈值，不能宣称已收敛。",
              "3. CST固定并联G与DG介损模型频率依赖不同。报告中16 pF/200 kΩ在100 kHz实际对应tanδ约49.7%，不是0.5%；因此报告曲线只能作为工况趋势，不能作为介损数值真值。",
              "4. 现有DG V3 RLGC实测验证在30/30条Core留出曲线上优于旧V3，但该验证没有测量链，不能单独证明完整实测域真实性。",
              "5. V2.7的本体纹理和受潮形态提供了有用目标，但其距离域核反算、末端重塑和局部修形违反当前S11权威边界。",
              "", "## 4. 实测证据状态", "",
              f"当前只读目录共发现{inventory['total_csv']}个CSV，其中IFFT目录{inventory['ifft_csv']}个、无校准S11 {inventory['unscaled_s11_csv']}个、RG58缺陷制造{inventory['rg58_defect_csv']}个、1500 m浸水{inventory['wet_1500m_csv']}个。",
              "", "现有`AgentsStorage/DG_V3_calibration`统计早于当前文件树，不能作为本轮最终分布门槛。应在本评估目录重新生成聚合统计；未标注现场文件只承担分布真实性，不承担缺陷类型和位置真值。",
              "", "## 5. 路线结论", "",
              "当前证据不支持用CST梯形网络整体替换DG V3。V0存在明确带内离散伪影；V1 0.1 m虽然改善，但对2.5 km电缆意味着约25000个单元，计算成本高且仍未证明数值收敛；连续固定RLGC又缺少真实导体和介质的频变损耗。",
              "", "建议路线是：保留DG V3连续频变RLGC和S11权威协议，把CST九工况固化为物理回归集；下一版在健康段增加小幅、空间相关的Z0/epsr/导体损耗/tanδ随机场，使完好电缆自然产生可解释的本体纹理；受潮继续通过平滑耦合的介电常数与介损变化生成，不移植V2.7距离域修形。",
              "", "## 6. 输出图", ""]
    for layer in ("clean", "common_measurement"):
        lines.append(f"### {layer}")
        lines.append("")
        for case_name in cfg["cases"]:
            lines.append(f"![{layer}-{case_name}](assets/comparison_board_{layer}_{case_name}.png)")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="DG V3/CST V0/CST V1路线评估")
    parser.add_argument("--config", type=Path, default=HERE / "evaluation_cases.yaml")
    parser.add_argument("--output", type=Path, default=HERE / "output")
    parser.add_argument("--assets", type=Path, default=HERE / "assets")
    parser.add_argument("--report", type=Path, default=HERE / "DG路线对比评估报告.md")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    point_count = 600 if args.smoke else int(cfg["frequency_grid"]["points"])
    time_points = 600 if args.smoke else int(cfg["fdr"]["algorithm2_time_points"])
    frequency_hz = np.linspace(float(cfg["frequency_grid"]["start_hz"]), float(cfg["frequency_grid"]["stop_hz"]), point_count)
    output = args.output.resolve()
    assets = args.assets.resolve()
    start = time.perf_counter()
    all_runs = {candidate: {} for candidate in CANDIDATES}
    metrics = []
    for candidate_index, candidate in enumerate(CANDIDATES, start=1):
        print(f"\n[{candidate_index}/{len(CANDIDATES)}] {LABELS[candidate]}")
        for case_index, case_name in enumerate(cfg["cases"], start=1):
            print(f"  [{case_index}/{len(cfg['cases'])}] {case_name}")
            clean = simulate_clean(candidate, frequency_hz, case_name)
            common = apply_common_measurement(clean, frequency_hz, cfg, int(cfg["measurement"]["seed"]) + case_index)
            for layer, s11 in (("clean", clean), ("common_measurement", common)):
                a1 = V1.algorithm1_ifft(frequency_hz, s11)
                a2 = V1.algorithm2_fdr(
                    frequency_hz, s11, time_points=time_points,
                    chunk_size=int(cfg["fdr"]["algorithm2_chunk_size"]),
                    progress_label=f"{candidate}:{case_name}:{layer}:A2",
                )
                run = {"s11": s11, "a1": a1, "a2": a2}
                all_runs[candidate].setdefault(case_name, {})[layer] = run
                save_s11(output / "s11" / layer / candidate / f"s11_{case_name}.csv", frequency_hz, s11)
                save_response(output / "responses" / layer / candidate / f"response_{case_name}.npz", a1, a2)
        for layer in ("clean", "common_measurement"):
            baseline = all_runs[candidate]["baseline"][layer]
            for case_name in cfg["cases"]:
                run = all_runs[candidate][case_name][layer]
                item = metrics_for(candidate, layer, case_name, run, baseline)
                run["metrics"] = item
                metrics.append(item)

    board_runs = {
        candidate: {
            case: {layer: all_runs[candidate][case][layer] for layer in ("clean", "common_measurement")}
            for case in cfg["cases"]
        }
        for candidate in CANDIDATES
    }
    for layer in ("clean", "common_measurement"):
        layer_runs = {candidate: {case: board_runs[candidate][case][layer] for case in cfg["cases"]} for candidate in CANDIDATES}
        for case_name in cfg["cases"]:
            reference = case_by_name(V1, case_name).reference_image
            reference_path = None if reference is None else ROOT / "DifferentAgent/CodeX/CST_Reproduction/V1/assets/report_reference" / reference
            plot_board(assets / f"comparison_board_{layer}_{case_name}.png", case_name, layer, frequency_hz, layer_runs, reference_path)

    output.mkdir(parents=True, exist_ok=True)
    flat_rows = []
    for item in metrics:
        row = {key: value for key, value in item.items() if key != "checks"}
        row["checks"] = json.dumps(item["checks"], ensure_ascii=False, sort_keys=True)
        flat_rows.append(row)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    inventory = inventory_real_data()
    runtime_s = time.perf_counter() - start
    summary = {
        "protocol": cfg,
        "runtime_s": runtime_s,
        "real_data_inventory": inventory,
        "metrics": metrics,
        "limitations": [
            "CST没有ASCII S11真值；报告图只作定性趋势参照",
            "DG V3工况适配在100 MHz锚定CST primary RLGC，不是全频逐点等效",
            "本脚本未使用V2.7距离域回写结果参与路线胜负",
        ],
    }
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(args.report.resolve(), cfg, metrics, runtime_s, inventory)
    print(f"\n完成：{len(metrics)}条指标，{2 * len(cfg['cases'])}张工况图，耗时{runtime_s:.2f}s")


if __name__ == "__main__":
    main()
