"""跨路线对比分析：V2.7 RG58/Field 路线 vs CodeX 的 DG V3 / CST V1 候选。

数据来源（全部只读）：
- 本目录 output/s11、output/responses        —— V2.7 RG58 / Field 两条路线
- DifferentAgent/CodeX/DG_Route_Evaluation/output/s11、responses
  —— dg_v3_rlgc、cst_v1_ladder_0p1m、cst_v1_continuous（CodeX 正式运行存档）

两侧使用同一频率网格（9 kHz–200 MHz，10000 点）与同一测量层噪声实现
（seed = 20260902 + 工况序号），距离域响应均由各自 S11 计算，可直接对比。

输出：
- cross_comparison/output/metrics_cross.csv  逐工况跨路线指标
- cross_comparison/assets/cross_board_*.png  5 候选同板对比图
- cross_comparison/跨路线对比分析报告.md

运行：
    python cross_compare_with_codex.py
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
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

HERE = Path(__file__).resolve().parent            # .../cross_comparison
EXPERIMENT_DIR = HERE.parent                      # .../V27_Route_Report_Cases
sys.path.insert(0, str(EXPERIMENT_DIR))
import compare_v27_routes as base                 # noqa: E402  （只读导入，含 V1/GEN）

CODEX_OUTPUT = base.ROOT / "DifferentAgent/CodeX/DG_Route_Evaluation/output"
CASES = (
    "baseline", "overall_C20pF", "overall_G20k", "segmented_loss",
    "local_C32pF_15m", "local_G2k_15m", "local_C4pF_15m",
    "local_R10ohm_15m", "local_R50ohm_15m",
)
LAYERS = ("clean", "common_measurement")
CANDIDATES = (
    "v27_rg58", "v27_field", "dg_v3_rlgc", "cst_v1_ladder_0p1m", "cst_v1_continuous",
)
LABELS = {
    "v27_rg58": "V2.7 RG58 路线",
    "v27_field": "V2.7 Field 路线",
    "dg_v3_rlgc": "DG V3 RLGC (100 MHz锚定)",
    "cst_v1_ladder_0p1m": "CST V1 0.1 m梯形",
    "cst_v1_continuous": "CST V1 连续固定RLGC",
}
COLORS = {
    "v27_rg58": "#1f77b4",
    "v27_field": "#d62728",
    "dg_v3_rlgc": "#2ca02c",
    "cst_v1_ladder_0p1m": "#7c3aed",
    "cst_v1_continuous": "#334155",
}
REFERENCE_CANDIDATES = ("dg_v3_rlgc", "cst_v1_ladder_0p1m")


def s11_path(root: Path, layer: str, candidate: str, case: str) -> Path:
    return root / "s11" / layer / candidate / f"s11_{case}.csv"


def response_path(root: Path, layer: str, candidate: str, case: str) -> Path:
    return root / "responses" / layer / candidate / f"response_{case}.npz"


def load_s11(path: Path):
    freq = []
    real = []
    imag = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        next(reader)
        for row in reader:
            freq.append(float(row[0]))
            real.append(float(row[1]))
            imag.append(float(row[2]))
    return np.asarray(freq), np.asarray(real) + 1j * np.asarray(imag)


def load_run(root: Path, layer: str, candidate: str, case: str, frequency_hz: np.ndarray):
    """从存档 S11 + npz 重建 run 结构（响应均由同一 S11 计算得到）。"""
    freq, s11 = load_s11(s11_path(root, layer, candidate, case))
    if freq.size != frequency_hz.size or not np.allclose(freq, frequency_hz, rtol=0.0, atol=1.0e-6):
        raise ValueError(f"频率网格不一致: {s11_path(root, layer, candidate, case)}")
    npz = np.load(response_path(root, layer, candidate, case))
    step = np.asarray(npz["algorithm1_step"])
    a1 = {
        "distance_m": np.asarray(npz["algorithm1_distance_m"]),
        "impulse_real_raw": np.asarray(npz["algorithm1_impulse"]),
        "step_raw": step,
        "step_scale": float(np.max(np.abs(step))),
    }
    a2 = {
        "distance_m": np.asarray(npz["algorithm2_distance_m"]),
        "step_smoothed": np.asarray(npz["algorithm2_step"]),
        "distance_impulse_m": np.asarray(npz["algorithm2_impulse_distance_m"]),
        "impulse_smoothed": np.asarray(npz["algorithm2_impulse"]),
    }
    return {"s11": s11, "a1": a1, "a2": a2}


def complex_relative_rms(candidate: np.ndarray, reference: np.ndarray) -> float:
    scale = float(np.sqrt(np.mean(np.abs(reference) ** 2)))
    if scale <= 0.0:
        scale = 1.0
    return float(np.sqrt(np.mean(np.abs(candidate - reference) ** 2)) / scale)


def plot_cross_board(path: Path, case_name: str, layer: str, frequency_hz: np.ndarray,
                     runs: dict, reference_path: Path | None):
    base.configure_plot()
    case = base.case_by_name(case_name)
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
        axes[0].plot(frequency_hz / 1e6, s11.real, color=color, lw=0.7, label=LABELS[name])
        axes[1].plot(frequency_hz / 1e6, np.abs(s11), color=color, lw=0.7)
        axes[2].plot(frequency_hz / 1e6, np.angle(s11, deg=True), color=color, lw=0.6)
        for axis, algorithm, kind in ((axes[3], 1, "step"), (axes[4], 1, "impulse"),
                                      (axes[5], 2, "step"), (axes[6], 2, "impulse")):
            d, y, y0 = base.normalized_pair(case_runs[name], base_runs[name], algorithm, kind)
            mask = d <= 60.0
            if case_name != "baseline":
                axis.plot(d[mask], y0[mask], color=color, lw=0.55, alpha=0.3, ls="--")
            axis.plot(d[mask], y[mask], color=color, lw=0.9)
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
    axes[0].legend(loc="best", fontsize=7)
    ref_axis = fig.add_subplot(grid[3, 1:5])
    ref_axis.axis("off")
    if reference_path is not None and reference_path.exists():
        ref_axis.imshow(plt.imread(reference_path), aspect="equal")
        ref_axis.set_title("报告对应 CST 图")
    else:
        ref_axis.text(0.5, 0.5, "基准工况（报告无对应 CST 图）",
                      ha="center", va="center", fontsize=12, transform=ref_axis.transAxes)
    fig.suptitle(f"跨路线对比：{case.title}（{layer}，距离轴0–60 m）", fontsize=15, y=0.994)
    fig.text(0.5, 0.006, "实线为当前工况；同色虚线为该候选基线。定量判定使用未归一化数组。",
             ha="center", fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_report(path: Path, metrics: list[dict], rel_rms: list[dict], runtime_s: float):
    lines = [
        "# 跨路线对比分析：V2.7 路线 vs DG V3 / CST V1",
        "",
        f"- 运行时间：{runtime_s:.2f} s",
        "- 数据来源：本实验 output（V2.7 RG58/Field 路线）与 CodeX/DG_Route_Evaluation output"
        "（dg_v3_rlgc、cst_v1_ladder_0p1m、cst_v1_continuous），全部只读。",
        "- 两侧共用 9 kHz–200 MHz / 10000 点频率网格；common_measurement 层使用同一噪声实现"
        "（seed = 20260902 + 工况序号），距离域响应均由各自 S11 计算。",
        "- 注意：V2.7 两条路线使用各自的介质参数（50 Ω、epsr 2.23/2.3），"
        "CodeX 候选在 100 MHz 锚定报告一次 RLGC（Z0≈137 Ω、40 pF/m），"
        "因此 S11 绝对形态不同属预期，对比重点是工况方向性与事件定位的一致性。",
        "",
        "## 1. 工况方向检查通过数（/9）",
        "",
        "| 候选 | clean | common_measurement |",
        "|---|---:|---:|",
    ]
    for name in CANDIDATES:
        clean = [m for m in metrics if m["route"] == name and m["layer"] == "clean"]
        common = [m for m in metrics if m["route"] == name and m["layer"] == "common_measurement"]
        lines.append(f"| {LABELS[name]} | {sum(m['case_pass'] for m in clean)}/9 | "
                     f"{sum(m['case_pass'] for m in common)}/9 |")
    lines += [
        "",
        "## 2. S11 复数相对 RMS（跨候选，clean 层）",
        "",
        "| 工况 | " + " | ".join(f"vs {LABELS[r]}" for r in REFERENCE_CANDIDATES) + "（列为 V2.7 两条路线对该参考的偏差） |",
        f"|---|---|---|",
    ]
    by_case = {}
    for item in rel_rms:
        if item["layer"] != "clean":
            continue
        by_case.setdefault(item["case"], {})[item["candidate"]] = item
    for case_name in CASES:
        row = [case_name]
        for ref in REFERENCE_CANDIDATES:
            parts = []
            for name in ("v27_rg58", "v27_field"):
                item = by_case[case_name].get(f"{name}__vs__{ref}")
                parts.append("-" if item is None else f"{item['relative_rms']:.4f}")
            row.append(" / ".join(parts))
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "注：单元格为 RG58路线 / Field路线 对该参考候选的复数相对 RMS。",
        "",
        "## 3. 逐工况指标",
        "",
        "完整逐工况指标（终端位置/幅值比、局部差分峰、阶跃差分中值与方向性布尔检查）"
        "见 `output/metrics_cross.csv`。",
        "",
        "## 4. 对比板",
        "",
    ]
    for layer in LAYERS:
        lines.append(f"### {layer}")
        lines.append("")
        for case_name in CASES:
            lines.append(f"![{layer}-{case_name}](assets/cross_board_{layer}_{case_name}.png)")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="V2.7路线 vs CodeX候选 跨路线对比")
    parser.add_argument("--output", type=Path, default=HERE / "output")
    parser.add_argument("--assets", type=Path, default=HERE / "assets")
    parser.add_argument("--report", type=Path, default=HERE / "跨路线对比分析报告.md")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load((EXPERIMENT_DIR / "evaluation_cases.yaml").read_text(encoding="utf-8"))
    frequency_hz = np.linspace(float(cfg["frequency_grid"]["start_hz"]),
                               float(cfg["frequency_grid"]["stop_hz"]),
                               int(cfg["frequency_grid"]["points"]))
    roots = {
        "v27_rg58": EXPERIMENT_DIR / "output",
        "v27_field": EXPERIMENT_DIR / "output",
        "dg_v3_rlgc": CODEX_OUTPUT,
        "cst_v1_ladder_0p1m": CODEX_OUTPUT,
        "cst_v1_continuous": CODEX_OUTPUT,
    }
    start = time.perf_counter()
    all_runs = {name: {} for name in CANDIDATES}
    total = len(CANDIDATES) * len(CASES) * len(LAYERS)
    done = 0
    for name in CANDIDATES:
        print(f"[加载] {LABELS[name]}")
        for case_name in CASES:
            for layer in LAYERS:
                all_runs[name].setdefault(case_name, {})[layer] = load_run(
                    roots[name], layer, name, case_name, frequency_hz)
                done += 1
                print(f"  [{done}/{total}] {case_name} / {layer}")

    metrics = []
    for name in CANDIDATES:
        for layer in LAYERS:
            baseline = all_runs[name]["baseline"][layer]
            for case_name in CASES:
                run = all_runs[name][case_name][layer]
                metrics.append(base.metrics_for(name, layer, case_name, run, baseline))

    rel_rms = []
    for layer in LAYERS:
        for case_name in CASES:
            for name in ("v27_rg58", "v27_field"):
                for ref in REFERENCE_CANDIDATES:
                    rel_rms.append({
                        "layer": layer,
                        "case": case_name,
                        "candidate": f"{name}__vs__{ref}",
                        "relative_rms": complex_relative_rms(
                            all_runs[name][case_name][layer]["s11"],
                            all_runs[ref][case_name][layer]["s11"]),
                    })

    for layer in LAYERS:
        layer_runs = {name: {case: all_runs[name][case][layer] for case in CASES}
                      for name in CANDIDATES}
        for case_name in CASES:
            reference = base.case_by_name(case_name).reference_image
            reference_path = (base.REPORT_REFERENCE_DIR / reference) if reference else None
            plot_cross_board(args.assets / f"cross_board_{layer}_{case_name}.png",
                             case_name, layer, frequency_hz, layer_runs, reference_path)
            print(f"[绘图] {layer} / {case_name}")

    args.output.mkdir(parents=True, exist_ok=True)
    flat_rows = []
    for item in metrics:
        row = {key: value for key, value in item.items() if key != "checks"}
        row["checks"] = json.dumps(item["checks"], ensure_ascii=False, sort_keys=True)
        flat_rows.append(row)
    with (args.output / "metrics_cross.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    with (args.output / "relative_rms.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["layer", "case", "candidate", "relative_rms"])
        writer.writeheader()
        writer.writerows(rel_rms)
    runtime_s = time.perf_counter() - start
    write_report(args.report.resolve(), metrics, rel_rms, runtime_s)
    print(f"\n完成：{len(metrics)}条指标、{len(rel_rms)}条相对RMS、{2 * len(CASES)}张跨路线对比板，"
          f"耗时 {runtime_s:.2f} s")


if __name__ == "__main__":
    main()
