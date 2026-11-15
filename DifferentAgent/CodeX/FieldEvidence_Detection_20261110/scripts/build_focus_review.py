"""Build the four-case focus page with FDR Reader display smoothing."""
from __future__ import annotations

import html
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[2]
sys.path.insert(0, str(REPO_ROOT / "DifferentAgent" / "Zcode" / "DG"))
sys.path.insert(0, str(EXPERIMENT_ROOT / "scripts"))
sys.path.insert(
    0,
    str(REPO_ROOT / "DifferentAgent" / "CodeX" / "FDR_Reader_Analysis_20261110" / "scripts"),
)

from build_review_packet import (  # noqa: E402
    field,
    is_measured,
    load_batch,
    render_evidence,
    response_arrays,
    sweep_catalog,
)
from dg_v4.fdr_io import read_fdr  # noqa: E402
from reader_preprocess import movmean, sample_step_m, span_m, support_points  # noqa: E402


OUTPUT_PATH = EXPERIMENT_ROOT / "output" / "review_packet" / "focus_review.html"
GROUPS_PATH = EXPERIMENT_ROOT / "review" / "comparison_groups.json"
INPUTS = (
    EXPERIMENT_ROOT / "evidence" / "batch_a.json",
    EXPERIMENT_ROOT / "evidence" / "batch_b.json",
)

CASE_WINDOWS = {
    "A02": [(100.0, 350.0)],
    "A03": [(250.0, 600.0)],
    "A04": [(200.0, 550.0)],
    "B06": [(0.0, 220.0), (500.0, 720.0)],
}
CASE_NOTES = {
    "A02": "200m起始边界只对应C相复测rowid=4；A相、B相仅作同次上下文。C相初测rowid=3独立保留。",
    "A03": "364–476m为用户确认的异常区间，阶跃为主、脉冲为辅；来源文字未指明适用相位。",
    "A04": "B相约270m及三相约420m为用户确认的局部异常位置；C相rowid=3缺厂商脉冲/阶跃响应表。相位未知的rowid=4不并入三相比较。",
    "B06": "原文保留64–130m与580–630m两个候选区间；用户当前无法从图中辨认，保持unknown，不作正负标签。",
}
COLORS = ("#1769aa", "#c43c39", "#218739", "#7c4d9e")


def esc(value: object) -> str:
    return "" if value is None else html.escape(str(value), quote=True)


def fmt(value: float) -> str:
    return f"{value:.5g}"


def build_case_lookup() -> dict[str, dict]:
    cases: dict[str, dict] = {}
    for path in INPUTS:
        for case in load_batch(path).get("cases") or []:
            cases[case["case_id"]] = case
    return cases


def parse_rowid(measurement_id: str) -> int:
    match = re.search(r"#Sweeps\.rowid=(\d+)$", measurement_id)
    if not match:
        raise ValueError(f"measurement_id缺少Sweeps.rowid：{measurement_id}")
    return int(match.group(1))


def load_sweeps(source_path: Path) -> dict[int, dict]:
    """Pair reader phases to real Sweeps.rowid before selecting any measurement."""
    data = read_fdr(str(source_path))
    catalog = sweep_catalog(source_path)
    measured_rows = [row for row in catalog if is_measured(field(row, "IsMeasured"))]
    phases = list(data.get("phases") or [])
    if len(measured_rows) != len(phases):
        raise ValueError(
            f"{source_path}: reader phase数{len(phases)}与实测Sweeps数"
            f"{len(measured_rows)}不一致，不能按rowid配对。"
        )
    result = {}
    for phase_index, (row, phase) in enumerate(zip(measured_rows, phases), start=1):
        rowid = row["__rowid__"]
        row_name = field(row, "SweepName")
        if str(row_name) != str(phase.get("name")):
            raise ValueError(
                f"{source_path}: Sweeps.rowid={rowid}名称与reader phase不一致。"
            )
        result[int(rowid)] = {
            "row": row,
            "phase": phase,
            "phase_index": phase_index,
        }
    return result


def response_record(sweep: dict, key: str) -> dict | None:
    values = response_arrays(sweep["phase"], {}, key)
    if values is None:
        return None
    distance, rho = values
    data = np.column_stack((distance, rho)).astype(np.float64, copy=False)
    return {"data": data, "points": int(data.shape[0])}


def sample_info(record: dict | None, k_values: tuple[int, ...] = ()) -> str:
    if record is None:
        return "缺表"
    data = record["data"]
    delta = abs(sample_step_m(data)) if len(data) > 1 else float("nan")
    bits = [f"{record['points']}点", f"Δx≈{fmt(delta)}m"]
    for k in k_values:
        bits.append(f"K={k}跨度≈{fmt(span_m(k, delta))}m/{support_points(k)}点")
    return "；".join(bits)


def crop_series(data: np.ndarray, xmin: float, xmax: float) -> tuple[np.ndarray, np.ndarray]:
    # Callers smooth the complete source table first; cropping is display-only.
    mask = np.isfinite(data[:, 0]) & np.isfinite(data[:, 1])
    mask &= (data[:, 0] >= xmin) & (data[:, 0] <= xmax)
    selected = data[mask]
    if selected.size == 0:
        return np.empty(0), np.empty(0)
    selected = selected[np.argsort(selected[:, 0], kind="stable")]
    return selected[:, 0], selected[:, 1]


def decimate(x: np.ndarray, y: np.ndarray, limit: int = 1600) -> tuple[np.ndarray, np.ndarray]:
    if x.size <= limit:
        return x, y
    bucket_count = max(1, limit // 2)
    selected = []
    for bucket in range(bucket_count):
        lo = bucket * x.size // bucket_count
        hi = (bucket + 1) * x.size // bucket_count
        if hi > lo:
            block = y[lo:hi]
            selected.extend((lo + int(np.argmin(block)), lo + int(np.argmax(block))))
    indexes = np.asarray(sorted(set(selected)), dtype=np.int64)
    return x[indexes], y[indexes]


def curve_svg(
    series: list[dict],
    title: str,
    xmin: float,
    xmax: float,
    annotations: list[dict] | None = None,
) -> str:
    prepared = []
    all_y = []
    for item in series:
        data = item["data"]
        x, y = crop_series(data, xmin, xmax)
        if x.size == 0:
            continue
        x, y = decimate(x, y)
        prepared.append((item, x, y))
        all_y.append(y)
    if not prepared:
        return '<p class="missing">该距离窗口没有有效曲线数据。</p>'

    y_values = np.concatenate(all_y)
    ymin, ymax = float(np.min(y_values)), float(np.max(y_values))
    if ymax == ymin:
        pad = max(abs(ymin) * 0.05, 1.0)
    else:
        pad = (ymax - ymin) * 0.06
    ymin, ymax = ymin - pad, ymax + pad

    width, height = 820, 390
    left, right, top, bottom = 86.0, 18.0, 62.0, 50.0
    plot_width, plot_height = width - left - right, height - top - bottom

    def sx(value: float) -> float:
        return left + (value - xmin) / (xmax - xmin) * plot_width

    def sy(value: float) -> float:
        return top + (ymax - value) / (ymax - ymin) * plot_height

    pieces = [
        f'<svg class="curve" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">',
        '<style>text{font-family:"Times New Roman",SimHei,sans-serif;font-size:12px;fill:#222}'
        '.frame{fill:#fff;stroke:#333;stroke-width:1.2}.tick{stroke:#333;stroke-width:1}'
        '.grid{stroke:#ddd;stroke-width:.7}.curve-line{fill:none;stroke-width:1.35}'
        '.annotation{fill:#b42318;font-size:11px}.band{fill:#f3a847;fill-opacity:.14}</style>',
        f'<text x="{width / 2:.1f}" y="20" text-anchor="middle">{esc(title)}</text>',
        f'<rect class="frame" x="{left}" y="{top}" width="{plot_width}" height="{plot_height}"/>',
    ]
    for annotation in annotations or []:
        if annotation["type"] == "range":
            start = max(xmin, float(annotation["start"]))
            end = min(xmax, float(annotation["end"]))
            if end > start:
                pieces.append(
                    f'<rect class="band" x="{sx(start):.2f}" y="{top}" '
                    f'width="{sx(end) - sx(start):.2f}" height="{plot_height}"/>'
                )
                pieces.append(
                    f'<text class="annotation" x="{sx(start) + 3:.2f}" y="{top + 14:.2f}">'
                    f'{esc(annotation["label"])}</text>'
                )
        elif annotation["type"] == "line":
            x = float(annotation["x"])
            if xmin <= x <= xmax:
                pieces.append(
                    f'<path d="M {sx(x):.2f} {top} V {top + plot_height}" '
                    'stroke="#b42318" stroke-width="1.6" stroke-dasharray="6 4"/>'
                )
                pieces.append(
                    f'<text class="annotation" x="{sx(x) + 3:.2f}" y="{top + 14:.2f}">'
                    f'{esc(annotation["label"])}</text>'
                )

    for index in range(1, 6):
        y = top + plot_height * index / 5
        pieces.append(f'<path class="grid" d="M {left} {y:.2f} h {plot_width}"/>')
    for index in range(6):
        xv = xmin + (xmax - xmin) * index / 5
        xp = sx(xv)
        yv = ymin + (ymax - ymin) * index / 5
        yp = sy(yv)
        pieces.extend([
            f'<path class="tick" d="M {xp:.2f} {top + plot_height} v -7"/>',
            f'<text x="{xp:.2f}" y="{height - 22}" text-anchor="middle">{fmt(xv)}</text>',
            f'<path class="tick" d="M {left} {yp:.2f} h 7"/>',
            f'<text x="{left - 10}" y="{yp + 4:.2f}" text-anchor="end">{fmt(yv)}</text>',
        ])
    for index, (item, x, y) in enumerate(prepared):
        points = " ".join(f"{sx(float(a)):.2f},{sy(float(b)):.2f}" for a, b in zip(x, y))
        dash = ' stroke-dasharray="7 4"' if item.get("dash") else ""
        pieces.append(
            f'<polyline class="curve-line" stroke="{item["color"]}"{dash} points="{points}"/>'
        )
    for index, (item, _, _) in enumerate(prepared):
        x = left + index * (plot_width / max(1, len(prepared)))
        dash = ' stroke-dasharray="7 4"' if item.get("dash") else ""
        pieces.extend([
            f'<path d="M {x:.2f} 39 h 24" stroke="{item["color"]}" stroke-width="2"'
            f'{dash}/>',
            f'<text x="{x + 29:.2f}" y="43">{esc(item["label"])}</text>',
        ])
    pieces.extend([
        f'<text x="{left + plot_width / 2:.2f}" y="{height - 4}" text-anchor="middle">Distance (m)</text>',
        f'<text transform="translate(18 {top + plot_height / 2:.2f}) rotate(-90)" text-anchor="middle">rho</text>',
        "</svg>",
    ])
    return "".join(pieces)


def annotations_for(case_id: str, sweep: dict) -> list[dict]:
    rowid = int(sweep["row"]["__rowid__"])
    if case_id == "A02" and rowid == 4:
        return [{"type": "line", "x": 200.0, "label": "C相复测起始边界 200m"}]
    if case_id == "A03":
        return [{"type": "range", "start": 364.0, "end": 476.0, "label": "确认区间 364–476m；相位未指定"}]
    if case_id == "A04":
        out = [{"type": "line", "x": 420.0, "label": "三相约420m"}]
        if rowid == 2:
            out.append({"type": "line", "x": 270.0, "label": "B相约270m"})
        return out
    if case_id == "B06":
        return [
            {"type": "range", "start": 64.0, "end": 130.0, "label": "64–130m 原文候选区间"},
            {"type": "range", "start": 580.0, "end": 630.0, "label": "580–630m 原文候选区间"},
        ]
    return []


def sweep_label(sweep: dict, phase_label: str, role: str) -> str:
    row = sweep["row"]
    return (
        f"{phase_label} · rowid={row['__rowid__']} · "
        f"{role} · {field(row, 'Fstart')}–{field(row, 'Fstop')} Hz · {field(row, 'Points')}点"
    )


def render_measurement(
    case_id: str,
    sweep: dict,
    phase_label: str,
    role: str,
    windows: list[tuple[float, float]],
    smooth_k: int | None,
    chart_counter: list[int],
    missing_counter: list[int],
) -> str:
    row = sweep["row"]
    label = sweep_label(sweep, phase_label, role)
    responses = {key: response_record(sweep, key) for key in ("impulse", "step")}
    for key, record in responses.items():
        if record is None:
            missing_counter[0] += 1
    parts = [
        f'<details class="measurement"><summary>{esc(label)}</summary>',
        '<table class="meta"><tbody>',
        f'<tr><th>measurement_id</th><td>{esc(sweep["source_path"])}#Sweeps.rowid={row["__rowid__"]}</td></tr>',
        f'<tr><th>phase_index</th><td>{sweep["phase_index"]}</td></tr>',
        f'<tr><th>波速</th><td>{esc(field(row, "Wave_Velocity"))}</td></tr>',
        f'<tr><th>X_Max</th><td>{esc(field(row, "X_Max"))}m</td></tr>',
    ]
    for key, title in (("impulse", "厂商脉冲"), ("step", "厂商阶跃")):
        extra = (smooth_k,) if smooth_k and responses[key] is not None else ()
        parts.append(f'<tr><th>{title}距离采样</th><td>{esc(sample_info(responses[key], extra))}</td></tr>')
    parts.extend(["</tbody></table>"])
    if responses["impulse"] is None and responses["step"] is None:
        parts.append('<p class="missing">厂商脉冲响应表与阶跃响应表均缺失；没有从S11补算曲线。</p>')
    for xmin, xmax in windows:
        parts.append(f'<div class="window"><h4>{fmt(xmin)}–{fmt(xmax)}m</h4><div class="plot-row">')
        for key, title in (("impulse", "厂商脉冲响应"), ("step", "厂商阶跃响应")):
            record = responses[key]
            if record is None:
                parts.append(f'<p class="missing">{esc(title)}表缺失。</p>')
                continue
            raw = record["data"]
            series = [{"label": f"{phase_label} 原始", "data": raw, "color": COLORS[0]}]
            if smooth_k:
                smoothed = movmean(raw, smooth_k)
                delta = abs(sample_step_m(raw))
                series.append({
                    "label": f"K={smooth_k}平滑 · {support_points(smooth_k)}点 · 跨度≈{fmt(span_m(smooth_k, delta))}m",
                    "data": smoothed,
                    "color": COLORS[1],
                    "dash": True,
                })
            chart_counter[0] += 1
            parts.append(
                f'<figure><figcaption>{esc(title)}'
                f'{" · 原始与阅读版平滑对照" if smooth_k else " · 原始绝对响应"}</figcaption>'
                f'{curve_svg(series, f"{phase_label} {title}", xmin, xmax, annotations_for(case_id, sweep))}'
                f'</figure>'
            )
        parts.append("</div></div>")
    parts.append("</details>")
    return "".join(parts)


def align_step_series(sweeps: list[dict], k: int) -> tuple[list[dict], list[dict]]:
    by_phase = {sweep["phase_label"]: sweep for sweep in sweeps}
    reference = by_phase["A相"]
    reference_record = response_record(reference, "step")
    if reference_record is None:
        raise ValueError("石杨A相阶跃表缺失，不能按A相均值对齐。")
    reference_smoothed = movmean(reference_record["data"], k)
    ref_mask = (reference_smoothed[:, 0] >= 100.0) & (reference_smoothed[:, 0] <= 180.0)
    reference_mean = float(np.mean(reference_smoothed[ref_mask, 1]))
    series, offsets = [], []
    for index, sweep in enumerate(sweeps):
        record = response_record(sweep, "step")
        if record is None:
            continue
        smoothed = movmean(record["data"], k)
        base_mask = (smoothed[:, 0] >= 100.0) & (smoothed[:, 0] <= 180.0)
        offset = reference_mean - float(np.mean(smoothed[base_mask, 1]))
        label = sweep["phase_label"]
        delta = abs(sample_step_m(record["data"]))
        display = smoothed.copy()
        display[:, 1] += offset
        series.append({
            "label": f"{label} rowid={sweep['row']['__rowid__']} · +{fmt(offset)}",
            "data": display,
            "color": COLORS[index % len(COLORS)],
        })
        offsets.append({
            "K": k,
            "phase": label,
            "rowid": sweep["row"]["__rowid__"],
            "offset": offset,
            "delta_m": delta,
            "span_m": span_m(k, delta),
            "support_points": support_points(k),
        })
    return series, offsets


def render_shiyang_alignment(sweeps: list[dict], chart_counter: list[int]) -> str:
    main = [sweep for sweep in sweeps if sweep["phase_label"] in {"A相", "B相", "C相复测"}]
    phase_order = {"A相": 0, "B相": 1, "C相复测": 2}
    main.sort(key=lambda sweep: phase_order[sweep["phase_label"]])
    parts = [
        '<section class="alignment"><h4>石杨三相阶跃：原始绝对响应与显示对齐</h4>',
        '<p>显示窗口100–350m。K=2、K=75均按原表完整阶跃序列先平滑，再裁窗口、降采样；'
        '对齐区间为100–180m。每条曲线只加常数Y偏移，未缩放、未去趋势。</p>',
    ]
    raw_series = []
    for index, sweep in enumerate(main):
        record = response_record(sweep, "step")
        if record is not None:
            raw_series.append({
                "label": f"{sweep['phase_label']} rowid={sweep['row']['__rowid__']}",
                "data": record["data"],
                "color": COLORS[index],
            })
    chart_counter[0] += 1
    parts.append(
        '<figure><figcaption>原始绝对阶跃响应，不作相间对齐</figcaption>'
        + curve_svg(raw_series, "石杨三相原始阶跃响应", 100.0, 350.0)
        + "</figure>"
    )
    all_offsets = []
    for k in (2, 75):
        series, offsets = align_step_series(main, k)
        all_offsets.extend(offsets)
        chart_counter[0] += 1
        parts.append(
            f'<figure><figcaption>K={k}阅读版平滑后，按100–180m均值对齐A相</figcaption>'
            + curve_svg(series, f"石杨三相阶跃 · K={k} · 对齐A相", 100.0, 350.0)
            + "</figure>"
        )
    parts.append('<table class="offsets"><thead><tr><th>K</th><th>相位/rowid</th><th>加入Y的偏移量</th><th>步长</th><th>窗口点数</th><th>近似跨度</th></tr></thead><tbody>')
    for row in all_offsets:
        parts.append(
            f'<tr><td>{row["K"]}</td><td>{esc(row["phase"])} / {row["rowid"]}</td>'
            f'<td>{fmt(row["offset"])}</td><td>{fmt(row["delta_m"])}m</td>'
            f'<td>{row["support_points"]}</td><td>{fmt(row["span_m"])}m</td></tr>'
        )
    parts.append("</tbody></table></section>")
    return "".join(parts)


def main() -> None:
    groups_payload = json.loads(GROUPS_PATH.read_text(encoding="utf-8"))
    groups = {group["case_id"]: group for group in groups_payload["groups"]}
    evidence_cases = build_case_lookup()
    chart_counter, missing_counter = [0], [0]
    file_cache: dict[Path, dict[int, dict]] = {}
    file_count = 0
    selected_sweeps = 0
    case_sections = []

    for case_id in ("A02", "A03", "A04", "B06"):
        group = groups[case_id]
        source_path = Path(group["source_path"])
        if source_path not in file_cache:
            file_cache[source_path] = load_sweeps(source_path)
            file_count += 1
        source_sweeps = file_cache[source_path]
        measurements = []
        for phase_label, measurement_id in group["members"].items():
            rowid = parse_rowid(measurement_id)
            sweep = source_sweeps[rowid]
            sweep.update({
                "source_path": str(source_path),
                "phase_label": group.get("phase_names", {}).get(phase_label, phase_label),
                "role": group.get("member_roles", {}).get(phase_label, "context"),
            })
            measurements.append(sweep)
        extra_measurements = []
        separately_retained = group.get("excluded_measurements") or [] if case_id == "A02" else []
        for excluded in separately_retained:
            rowid = parse_rowid(excluded["measurement_id"])
            sweep = source_sweeps[rowid]
            sweep.update({
                "source_path": str(source_path),
                "phase_label": field(sweep["row"], "SweepName"),
                "role": "独立保留，不并入主比较组",
                "exclude_reason": excluded["reason"],
            })
            extra_measurements.append(sweep)

        case = evidence_cases[case_id]
        section = [
            f'<section class="case" id="case-{case_id}">',
            f'<h2>{esc(case_id)}</h2>',
            f'<p class="summary">{esc(case.get("summary"))}</p>',
            f'<p class="case-note">{esc(CASE_NOTES[case_id])}</p>',
            f'<p><b>对比组：</b>{esc(group["comparison_group_id"])} · '
            f'<b>源FDR：</b>{esc(source_path)} · <b>窗口：</b>'
            f'{esc("；".join(f"{fmt(a)}–{fmt(b)}m" for a,b in CASE_WINDOWS[case_id]))}</p>',
            render_evidence(case),
            '<h3>相位响应</h3>',
        ]
        for sweep in measurements:
            selected_sweeps += 1
            smooth_k = 15 if case_id == "A03" else None
            section.append(render_measurement(
                case_id,
                sweep,
                sweep["phase_label"],
                sweep["role"],
                CASE_WINDOWS[case_id],
                smooth_k,
                chart_counter,
                missing_counter,
            ))
        if case_id == "A02":
            section.append(render_shiyang_alignment(measurements, chart_counter))
        if extra_measurements:
            section.append('<h3>独立保留的测次</h3>')
            for sweep in extra_measurements:
                selected_sweeps += 1
                section.append(
                    f'<p class="note">{esc(sweep.get("exclude_reason"))}</p>'
                )
                section.append(render_measurement(
                    case_id,
                    sweep,
                    sweep["phase_label"],
                    sweep["role"],
                    CASE_WINDOWS[case_id],
                    None,
                    chart_counter,
                    missing_counter,
                ))
        section.append("</section>")
        case_sections.append("".join(section))

    page = (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>四个代表案例局部审阅</title><style>'
        'body{font-family:"Times New Roman",SimHei,sans-serif;line-height:1.5;color:#20242a;'
        'max-width:1250px;margin:20px auto;padding:0 18px;background:#f4f6f8}'
        'nav{position:sticky;top:0;background:#fff;padding:12px;border:1px solid #c8ccd2;z-index:2}'
        'nav a{margin-right:18px}.case{background:#fff;border:1px solid #c8ccd2;padding:18px;margin:22px 0}'
        'details{border:1px solid #d3d7dc;margin:10px 0;padding:10px;background:#fff}'
        'summary{cursor:pointer;font-weight:600}.window{border-top:1px solid #ddd;margin-top:12px;padding-top:4px}'
        '.plot-row{display:grid;grid-template-columns:1fr 1fr;gap:12px}figure{margin:6px 0;min-width:0}'
        'figcaption{font-weight:600}.curve{width:100%;height:auto}.meta,.offsets{border-collapse:collapse;margin:10px 0;max-width:100%}'
        'th,td{border:1px solid #d6d9de;padding:4px 8px;text-align:left}.case-note{background:#f7f9fb;padding:9px 12px}'
        '.evidence{background:#f7f9fb;border-left:3px solid #73849a;padding:8px 12px;margin:12px 0}'
        'blockquote{white-space:pre-wrap;margin:6px 0}.note{color:#555}.missing{color:#9b2525;background:#fff0ef;padding:8px}'
        '.alignment{margin-top:20px;border-top:2px solid #bbb}.alignment figure{max-width:900px}'
        '@media(max-width:800px){.plot-row{grid-template-columns:1fr}nav{position:static}}'
        '</style></head><body><h1>四个代表案例局部审阅</h1>'
        '<p>FDR Reader平滑按完整厂商响应表执行，之后才裁剪距离窗口并为图形降采样。'
        'K=2对应3点居中均值，K=15对应15点，K=75对应75点；边缘窗口按可用点数收缩。'
        '表格报告各相脉冲/阶跃的采样步长和所示平滑支持跨度。石杨相间对齐只用于显示比较。</p>'
        '<nav><b>案例：</b><a href="#case-A02">A02</a><a href="#case-A03">A03</a>'
        '<a href="#case-A04">A04</a><a href="#case-B06">B06</a></nav>'
        + "".join(case_sections)
        + "</body></html>"
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(page, encoding="utf-8")
    print(f"FDR files read: {file_count}")
    print(f"selected measurements: {selected_sweeps}")
    print(f"charts: {chart_counter[0]}")
    print(f"missing response tables: {missing_counter[0]}")
    print(f"focus review: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
