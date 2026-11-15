"""Render three source-backed candidate reviews from read-only FDR files."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[2]
sys.path.insert(0, str(REPO_ROOT / "DifferentAgent" / "Zcode" / "DG"))
sys.path.insert(
    0,
    str(REPO_ROOT / "DifferentAgent" / "CodeX" / "FDR_Reader_Analysis_20261110" / "scripts"),
)

from dg_v4.fdr_io import decrypt_fdr  # noqa: E402
from reader_preprocess import movmean, sample_step_m, span_m  # noqa: E402


OUTPUT_DIR = EXPERIMENT_ROOT / "output" / "candidate_review_20261111"
TARGET_SPAN_M = 0.5
PHASE_COLORS = {
    "A": "#1769aa",
    "B": "#c46b00",
    "C": "#218739",
}
FONT_EN = FontProperties(family="Times New Roman")
FONT_ZH = FontProperties(family="SimHei")

CASES = {
    "A01": {
        "name": "北京城区白塔寺耐压前",
        "source_path": r"E:\20261010截至的全部fdr文件\2025-2-17 11.25.49北京城区白塔寺\2025-2-17 11.25.49城区白塔寺.fdr",
        "file_reference": "2025-2-17 11.25.49城区白塔寺.fdr（耐压前）",
        "evidence_file_reference": "2025-2-17 11.25.49城区白塔寺.fdr（耐压前）；2025-2-17 12.30.59白塔寺复测.fdr（耐压后）",
        "evidence_ids": ["A01-E01"],
        "rowids": [1, 2, 3],
        "x_max_m": 3000.0,
        "global_window": (150.0, 330.0),
        "local_windows": [(150.0, 330.0)],
        "candidate": {
            "position_m": 240.0,
            "phase_explicit": False,
            "note": "A01-E01同时提到耐压前和耐压后文件；本图只读取耐压前文件。原文没有相位引用。",
        },
    },
    "A08": {
        "name": "三医院宿舍公用柜",
        "source_path": r"E:\20261010截至的全部fdr文件\2026-5-21 11.54.20深圳市第三人民医院宿舍公用智能柜\2026-5-21 11.54.20.fdr",
        "file_reference": "2026-5-21 11.54.20.fdr",
        "evidence_ids": ["A08-E01", "A08-E02", "A08-E03"],
        "rowids": [1, 2, 3],
        "x_max_m": 780.0,
        "fdr_length_estimate_m": 770.0,
        "global_window": (0.0, 850.0),
        "local_windows": [(40.0, 200.0), (600.0, 810.0)],
        "candidate": {
            "engineer_joint_m": [55.0, 91.0, 175.0, 638.0, 682.0],
            "low_voltage_pulse_m": [55.0, 93.0, 174.0, 633.0, 684.0, 765.0],
            "low_voltage_terminal_m": 765.0,
            "note": "E02工程师FDR接头标记；E01为现场低压脉冲位置，来源分开标绘。",
        },
    },
    "B07": {
        "name": "两段电缆第一段（从柜子测）",
        "source_path": r"E:\20261010截至的全部fdr文件\2026-7-20 10.46.01深圳某地两段电缆\第一段（从柜子测）\2026-7-20 10.46.01.fdr",
        "file_reference": "2026-7-20 10.46.01.fdr",
        "evidence_ids": ["B07-E05"],
        "rowids": [1, 2, 3],
        "x_max_m": 700.0,
        "fdr_length_estimate_m": 637.0,
        "global_window": (0.0, 650.0),
        "local_windows": [(0.0, 500.0)],
        "candidate": {
            "ranges_m": [[58.0, 201.0], [337.0, 459.0]],
            "note": "B07-E05的file_reference=2026-7-20 10.46.01.fdr；measurement_index将其唯一对应到第一段（从柜子测）。",
        },
    },
    "B09": {
        "name": "金坑线F18南岭黄金坑#4",
        "source_path": r"E:\20261010截至的全部fdr文件\2026-8-28 15.26.01深圳110kV樟树布变电站10kV金坑线F18南岭黄金坑#4智能柜\2026-8-28 15.26.01.fdr",
        "file_reference": "2026-8-28 15.26.01.fdr",
        "evidence_ids": ["B09-E01", "B09-E02"],
        "rowids": [1, 2, 3],
        "x_max_m": 2800.0,
        "fdr_length_estimate_m": 2488.0,
        "cable_label_length_m": 2482.0,
        "global_window": (0.0, 2700.0),
        "local_windows": [(80.0, 230.0), (2350.0, 2650.0)],
        "candidate": {
            "engineer_joint_m": 159.0,
            "note": "E01工程师称159m接头问题最大；E02记录电缆标签2482m及FDR末端估计2488m。",
        },
    },
}


def set_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman"],
        "font.sans-serif": ["SimHei"],
        "axes.unicode_minus": False,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def field(row: dict, name: str):
    key = next((key for key in row if str(key).lower() == name.lower()), None)
    return row.get(key) if key is not None else None


def is_measured(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def read_sweeps_in_memory(source_path: Path, rowids: list[int]) -> list[dict]:
    plain = decrypt_fdr(str(source_path))
    connection = sqlite3.connect(":memory:")
    try:
        connection.deserialize(plain)
        columns = [row[1] for row in connection.execute('PRAGMA table_info("Sweeps")')]
        raw_rows = connection.execute('SELECT rowid,* FROM "Sweeps" ORDER BY rowid').fetchall()
        sweeps = [dict(zip(["__rowid__"] + columns, row)) for row in raw_rows]
        by_rowid = {int(sweep["__rowid__"]): sweep for sweep in sweeps}
        result = []
        for rowid in rowids:
            row = by_rowid[rowid]
            if not is_measured(field(row, "IsMeasured")):
                raise ValueError(f"{source_path}: Sweeps.rowid={rowid}未标记实测")
            response_data = {}
            for response, table_prefix in (("impulse", "ImpulseResponse"), ("step", "StepResponse")):
                table = f"{table_prefix}_{rowid}"
                rows = connection.execute(
                    f'SELECT "Distance","Rho" FROM "{table}" ORDER BY rowid'
                ).fetchall()
                data = np.asarray(rows, dtype=np.float64)
                response_data[response] = data
            result.append({
                "rowid": rowid,
                "phase_name": field(row, "SweepName"),
                "impulse": response_data["impulse"],
                "step": response_data["step"],
            })
        return result
    finally:
        connection.close()


def odd_window_for_span(data: np.ndarray, span_target_m: float) -> tuple[int, float, float]:
    delta_m = abs(sample_step_m(data))
    target_points = span_target_m / delta_m + 1.0
    center = int(round(target_points))
    candidates = [k for k in range(max(1, center - 3), center + 4) if k % 2 == 1]
    k = min(candidates, key=lambda value: abs((value - 1) * delta_m - span_target_m))
    return k, delta_m, span_m(k, delta_m)


def add_smoothing(sweeps: list[dict]) -> list[dict]:
    for sweep in sweeps:
        sweep["smoothing"] = {}
        for response in ("impulse", "step"):
            raw = sweep[response]
            k, delta_m, actual_span_m = odd_window_for_span(raw, TARGET_SPAN_M)
            sweep["smoothing"][response] = {
                "data": movmean(raw, k),
                "k": k,
                "delta_m": delta_m,
                "span_m": actual_span_m,
            }
    return sweeps


def phase_key(name: str) -> str:
    for phase in ("A", "B", "C"):
        if phase in name:
            return phase
    return "A"


def line_annotations(ax, case_id: str, window: tuple[float, float]) -> None:
    lo, hi = window
    case = CASES[case_id]
    candidate = case["candidate"]
    if case_id == "A01":
        position = candidate["position_m"]
        if lo <= position <= hi:
            ax.axvline(
                position, color="#b42318", linestyle="--", linewidth=1.0, alpha=0.86,
                label="原文候选 240m（相位未指明）",
            )
    elif case_id == "A08":
        engineer = candidate["engineer_joint_m"]
        pulse = candidate["low_voltage_pulse_m"]
        first_engineer = True
        first_pulse = True
        for position in engineer:
            if lo <= position <= hi:
                ax.axvline(
                    position, color="#b42318", linestyle="--", linewidth=0.9, alpha=0.78,
                    label="原文候选（工程师FDR接头）" if first_engineer else "_nolegend_",
                )
                first_engineer = False
        for position in pulse:
            if lo <= position <= hi:
                ax.axvline(
                    position, color="#6842a6", linestyle=":", linewidth=0.9, alpha=0.72,
                    label="原文位置（现场低压脉冲）" if first_pulse else "_nolegend_",
                )
                first_pulse = False
        estimate = case["fdr_length_estimate_m"]
        if lo <= estimate <= hi:
            ax.axvline(
                estimate, color="#8b6b00", linestyle="-.", linewidth=1.0, alpha=0.9,
                label="FDR全长估计 770m",
            )
    elif case_id == "B07":
        for index, (start, end) in enumerate(candidate["ranges_m"]):
            start = max(start, lo)
            end = min(end, hi)
            if end > start:
                ax.axvspan(
                    start, end, color="#dc9b28", alpha=0.16,
                    label="原文候选区间" if index == 0 else "_nolegend_",
                )
                ax.text(
                    max(start + 2.0, lo + 2.0), 0.96 - index * 0.08,
                    f"原文候选 {int(candidate['ranges_m'][index][0])}–{int(candidate['ranges_m'][index][1])}m",
                    transform=ax.get_xaxis_transform(), fontproperties=FONT_ZH,
                    fontsize=8, color="#8a5a00", va="top",
                )
    elif case_id == "B09":
        point = candidate["engineer_joint_m"]
        if lo <= point <= hi:
            ax.axvline(
                point, color="#b42318", linestyle="--", linewidth=1.0, alpha=0.85,
                label="原文候选 159m",
            )
        estimate = case["fdr_length_estimate_m"]
        if lo <= estimate <= hi:
            ax.axvline(
                estimate, color="#8b6b00", linestyle="-.", linewidth=1.0, alpha=0.9,
                label="FDR末端估计 2488m",
            )
        cable_label = case["cable_label_length_m"]
        if lo <= cable_label <= hi:
            ax.axvline(
                cable_label, color="#555555", linestyle=":", linewidth=1.0, alpha=0.85,
                label="标签长度 2482m",
            )


def draw_panel(ax, case_id: str, sweeps: list[dict], response: str,
               window: tuple[float, float], show_legend: bool = True) -> None:
    lo, hi = window
    for sweep in sweeps:
        phase = phase_key(sweep["phase_name"])
        color = PHASE_COLORS[phase]
        raw = sweep[response]
        smooth = sweep["smoothing"][response]["data"]
        mask_raw = (raw[:, 0] >= lo) & (raw[:, 0] <= hi)
        mask_smooth = (smooth[:, 0] >= lo) & (smooth[:, 0] <= hi)
        ax.plot(
            raw[mask_raw, 0], raw[mask_raw, 1], color=color, alpha=0.34,
            linewidth=0.62, label=f"{sweep['phase_name']} 原始",
        )
        ax.plot(
            smooth[mask_smooth, 0], smooth[mask_smooth, 1], color=color,
            linewidth=1.15, label=f"{sweep['phase_name']} 平滑",
        )
    line_annotations(ax, case_id, window)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("Distance (m)", fontproperties=FONT_EN)
    ax.set_ylabel("脉冲响应 ρ" if response == "impulse" else "阶跃响应 ρ", fontproperties=FONT_ZH)
    ax.tick_params(direction="in", top=True, right=True, labelsize=9)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.28)
    for spine in ax.spines.values():
        spine.set_visible(True)
    if show_legend:
        ax.legend(
            loc="best", prop=FontProperties(family="SimHei", size=7.3),
            ncol=2, frameon=True, framealpha=0.9,
        )


def file_note(case_id: str, sweeps: list[dict]) -> str:
    case = CASES[case_id]
    source = Path(case["source_path"]).name
    rowids = "、".join(
        f"rowid={sweep['rowid']} {sweep['phase_name']}"
        for sweep in sweeps
    )
    smooth = "；".join(
        f"{sweep['phase_name']} K={sweep['smoothing']['step']['k']}"
        f"（Δx={sweep['smoothing']['step']['delta_m']:.4g}m，"
        f"跨度≈{sweep['smoothing']['step']['span_m']:.4g}m）"
        for sweep in sweeps
    )
    return f"FDR：{source}；{rowids}。浅色=原始，深色=整表先平滑后裁窗；{smooth}。"


def save_figure(case_id: str, sweeps: list[dict], windows: list[tuple[float, float]],
                kind: str, output_path: Path) -> None:
    rows, cols = 2, len(windows)
    figsize = (8.0 * cols, 6.5) if cols > 1 else (9.4, 7.4)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False, sharex="col")
    for col, window in enumerate(windows):
        draw_panel(axes[0, col], case_id, sweeps, "impulse", window)
        draw_panel(axes[1, col], case_id, sweeps, "step", window)
        axes[0, col].set_title(
            f"脉冲响应 · {window[0]:g}–{window[1]:g}m", fontproperties=FONT_ZH
        )
        axes[1, col].set_title(
            f"阶跃响应 · {window[0]:g}–{window[1]:g}m", fontproperties=FONT_ZH
        )
    case = CASES[case_id]
    full_range = f"X轴显示上限={case['x_max_m']:g}m；FDR全长估计={case['fdr_length_estimate_m']:g}m"
    fig.suptitle(f"{case_id} {case['name']} · {kind} · {full_range}", fontproperties=FONT_ZH, fontsize=13)
    fig.text(
        0.01, 0.005, file_note(case_id, sweeps),
        fontproperties=FONT_ZH, fontsize=7.5, ha="left", va="bottom",
    )
    fig.tight_layout(rect=(0.0, 0.045, 1.0, 0.95))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_a01_figure(sweeps: list[dict], output_path: Path) -> dict[str, float]:
    window = (150.0, 330.0)
    baseline = (150.0, 200.0)
    by_phase = {phase_key(sweep["phase_name"]): sweep for sweep in sweeps}
    reference = by_phase["A"]["step"]
    reference_mask = (reference[:, 0] >= baseline[0]) & (reference[:, 0] <= baseline[1])
    reference_mean = float(np.mean(reference[reference_mask, 1]))
    offsets = {}
    aligned = {}
    for sweep in sweeps:
        phase = phase_key(sweep["phase_name"])
        step = sweep["step"]
        mask = (step[:, 0] >= baseline[0]) & (step[:, 0] <= baseline[1])
        offset = reference_mean - float(np.mean(step[mask, 1]))
        shifted = step.copy()
        shifted[:, 1] += offset
        offsets[phase] = offset
        aligned[phase] = shifted

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.5), sharex=True)
    draw_panel(axes[0, 0], "A01", sweeps, "impulse", window)
    axes[0, 0].set_title("脉冲响应：原始浅色 + 0.5m平滑深色", fontproperties=FONT_ZH)
    draw_panel(axes[0, 1], "A01", sweeps, "step", window)
    axes[0, 1].set_title("阶跃响应：原始浅色 + 0.5m平滑深色", fontproperties=FONT_ZH)

    for sweep in sweeps:
        phase = phase_key(sweep["phase_name"])
        color = PHASE_COLORS[phase]
        raw = sweep["step"]
        mask = (raw[:, 0] >= window[0]) & (raw[:, 0] <= window[1])
        axes[1, 0].plot(
            raw[mask, 0], raw[mask, 1], color=color, linewidth=1.0,
            label=f"{sweep['phase_name']} 原始绝对响应",
        )
        shifted = aligned[phase]
        shifted_mask = (shifted[:, 0] >= window[0]) & (shifted[:, 0] <= window[1])
        axes[1, 1].plot(
            shifted[shifted_mask, 0], shifted[shifted_mask, 1], color=color,
            linewidth=1.0, label=f"{sweep['phase_name']} +{offsets[phase]:.4g}",
        )
    for ax in axes[1]:
        line_annotations(ax, "A01", window)
        ax.set_xlim(*window)
        ax.set_xlabel("Distance (m)", fontproperties=FONT_EN)
        ax.set_ylabel("阶跃响应 ρ", fontproperties=FONT_ZH)
        ax.tick_params(direction="in", top=True, right=True, labelsize=9)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.28)
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.legend(loc="best", prop=FontProperties(family="SimHei", size=7.5),
                  ncol=2, frameon=True, framealpha=0.9)

    for ax in axes.flat:
        ax.set_xlim(*window)
    phase_names = "；".join(
        f"{phase}偏移={offsets[phase]:+.5g}"
        for phase in ("A", "B", "C")
    )
    fig.suptitle(
        "A01 白塔寺耐压前 · rowid 1/2/3 · 原文候选240m（相位未指明）",
        fontproperties=FONT_ZH, fontsize=13,
    )
    fig.text(
        0.01, 0.005,
        f"仅使用耐压前FDR；Y对齐仅用于显示，参考区间150–200m对齐A相，只加常数，不缩放/去趋势。{phase_names}",
        fontproperties=FONT_ZH, fontsize=8, ha="left", va="bottom",
    )
    fig.tight_layout(rect=(0.0, 0.045, 1.0, 0.95))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return offsets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(CASES), help="仅生成指定案例")
    args = parser.parse_args()
    set_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index_path = OUTPUT_DIR / "index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        existing_cases = {case["case_id"]: case for case in index.get("cases", [])}
        existing_order = [case["case_id"] for case in index.get("cases", [])]
    else:
        index = {}
        existing_cases = {}
        existing_order = []
    index["source_policy"] = "只读FDR；RC4解密后用sqlite3 in-memory deserialize读取Sweeps和响应表。"
    index["smoothing"] = {
        "method": "reader_preprocess.movmean",
        "target_physical_span_m": TARGET_SPAN_M,
        "applied_before_window_crop": True,
        "phase_axis_shift": 0.0,
    }
    selected_case_ids = [args.case] if args.case else list(CASES)

    for case_id in selected_case_ids:
        case = CASES[case_id]
        source_path = Path(case["source_path"])
        sweeps = add_smoothing(read_sweeps_in_memory(source_path, case["rowids"]))
        for sweep in sweeps:
            step = sweep["step"]
            delta = abs(sample_step_m(step))
            sweep["sample_step_m"] = delta
        if case_id == "A01":
            figure_path = OUTPUT_DIR / "A01_pre_240m_local.png"
            offsets = save_a01_figure(sweeps, figure_path)
            outputs = [figure_path.name]
        else:
            global_path = OUTPUT_DIR / f"{case_id}_global.png"
            local_path = OUTPUT_DIR / f"{case_id}_local.png"
            save_figure(case_id, sweeps, [case["global_window"]], "整体", global_path)
            save_figure(case_id, sweeps, case["local_windows"], "局部", local_path)
            offsets = {}
            outputs = [global_path.name, local_path.name]

        case_record = {
            "case_id": case_id,
            "case_name": case["name"],
            "evidence_ids": case["evidence_ids"],
            "file_reference": case["file_reference"],
            "evidence_file_reference": case.get("evidence_file_reference"),
            "source_path": str(source_path),
            "source_mapping_note": case["candidate"]["note"],
            "fdr_length_estimate_m": case.get("fdr_length_estimate_m"),
            "source_axis_max_m": case["x_max_m"],
            "candidate_marks": case["candidate"],
            "display_alignment_offsets_y": offsets,
            "sweeps": [
                {
                    "rowid": sweep["rowid"],
                    "phase_name": sweep["phase_name"],
                    "response_points": {
                        "impulse": int(sweep["impulse"].shape[0]),
                        "step": int(sweep["step"].shape[0]),
                    },
                    "distance_step_m": sweep["sample_step_m"],
                    "smoothing": {
                        response: {
                            "k": sweep["smoothing"][response]["k"],
                            "span_m": sweep["smoothing"][response]["span_m"],
                        }
                        for response in ("impulse", "step")
                    },
                }
                for sweep in sweeps
            ],
            "outputs": outputs,
        }
        existing_cases[case_id] = case_record
        if case_id not in existing_order:
            existing_order.append(case_id)

    index["cases"] = [existing_cases[case_id] for case_id in existing_order]
    with index_path.open("w", encoding="utf-8") as stream:
        json.dump(index, stream, ensure_ascii=False, indent=2)
    print(f"review output: {OUTPUT_DIR}")
    for case_id in selected_case_ids:
        case = existing_cases[case_id]
        print(f"{case['case_id']}: {', '.join(case['outputs'])}; "
              f"source={case['source_path']}; "
              f"rowids={','.join(str(sweep['rowid']) for sweep in case['sweeps'])}")


if __name__ == "__main__":
    main()
