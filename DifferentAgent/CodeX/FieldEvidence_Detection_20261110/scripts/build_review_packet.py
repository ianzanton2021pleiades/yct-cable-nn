"""Build an offline case review packet from the curated A/B evidence batches."""
from __future__ import annotations

import html
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[2]
sys.path.insert(0, str(REPO_ROOT / "DifferentAgent" / "Zcode" / "DG"))
from dg_v4.fdr_io import decrypt_fdr, phase_arrays, read_fdr


INPUTS = (
    EXPERIMENT_ROOT / "evidence" / "batch_a.json",
    EXPERIMENT_ROOT / "evidence" / "batch_b.json",
)
OUTPUT_DIR = EXPERIMENT_ROOT / "output" / "review_packet"


def esc(value: object) -> str:
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def field(row: dict, name: str):
    key = next((k for k in row if str(k).lower() == name.lower()), None)
    return row.get(key) if key is not None else None


def is_measured(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def sweep_catalog(path: Path) -> list[dict]:
    """Read real Sweeps rowids in memory; the source FDR stays read-only."""
    plain = decrypt_fdr(str(path))
    con = sqlite3.connect(":memory:")
    try:
        deserialize = getattr(con, "deserialize", None)
        if deserialize is None:
            raise RuntimeError("SQLite deserialize() is unavailable in this Python build")
        deserialize(plain)
        columns = [r[1] for r in con.execute('PRAGMA table_info("Sweeps")')]
        rows = con.execute('SELECT rowid, * FROM "Sweeps" ORDER BY rowid').fetchall()
        return [dict(zip(["__rowid__"] + columns, row)) for row in rows]
    finally:
        con.close()


def response_arrays(phase: dict, arrays: dict, key: str):
    distance_key, rho_key = key + "_dist", key + "_rho"
    if distance_key in arrays and rho_key in arrays:
        return np.asarray(arrays[distance_key]).ravel(), np.asarray(arrays[rho_key]).ravel()
    raw = phase.get(key)
    if raw is None:
        return None
    columns, rows = raw
    indexes = {str(name).lower(): i for i, name in enumerate(columns)}
    if "distance" not in indexes or "rho" not in indexes:
        raise ValueError(f"{key} table lacks Distance/Rho columns")
    values = np.asarray(rows, dtype=np.float64)
    return values[:, indexes["distance"]], values[:, indexes["rho"]]


def display_samples(x: np.ndarray, y: np.ndarray, limit: int = 1600):
    """Min/max decimation for display only; the prepared training data is untouched."""
    if x.size <= limit:
        return x, y
    buckets = max(1, limit // 2)
    selected = []
    for b in range(buckets):
        lo = b * x.size // buckets
        hi = (b + 1) * x.size // buckets
        if hi <= lo:
            continue
        block = y[lo:hi]
        selected.extend((lo + int(np.argmin(block)), lo + int(np.argmax(block))))
    indexes = np.asarray(sorted(set(selected)), dtype=np.int64)
    return x[indexes], y[indexes]


def fmt(value: float) -> str:
    return f"{value:.4g}"


def curve_svg(x_raw, y_raw, title: str) -> str:
    x = np.asarray(x_raw, dtype=np.float64).ravel()
    y = np.asarray(y_raw, dtype=np.float64).ravel()
    n = min(x.size, y.size)
    valid = np.isfinite(x[:n]) & np.isfinite(y[:n])
    x, y = x[:n][valid], y[:n][valid]
    if x.size == 0:
        return '<p class="missing">曲线表没有有效数值。</p>'
    order = np.argsort(x, kind="stable")
    x, y = display_samples(x[order], y[order])
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    if xmax == xmin:
        xmax = xmin + 1.0
    if ymax == ymin:
        pad = max(abs(ymin) * 0.05, 1.0)
        ymin, ymax = ymin - pad, ymax + pad
    else:
        pad = (ymax - ymin) * 0.05
        ymin, ymax = ymin - pad, ymax + pad

    width, height = 760, 330
    left, right, top, bottom = 76.0, 20.0, 34.0, 48.0
    plot_w, plot_h = width - left - right, height - top - bottom

    def sx(v):
        return left + (v - xmin) / (xmax - xmin) * plot_w

    def sy(v):
        return top + (ymax - v) / (ymax - ymin) * plot_h

    points = " ".join(f"{sx(float(a)):.2f},{sy(float(b)):.2f}" for a, b in zip(x, y))
    ticks = []
    for i in range(6):
        xv = xmin + (xmax - xmin) * i / 5
        xp = sx(xv)
        ticks.append(
            f'<path d="M {xp:.2f} {height - bottom:.2f} v -7" class="tick"/>'
            f'<text x="{xp:.2f}" y="{height - 25}" text-anchor="middle">{esc(fmt(xv))}</text>'
        )
        yv = ymin + (ymax - ymin) * i / 5
        yp = sy(yv)
        ticks.append(
            f'<path d="M {left:.2f} {yp:.2f} h 7" class="tick"/>'
            f'<text x="{left - 10:.2f}" y="{yp + 4:.2f}" text-anchor="end">{esc(fmt(yv))}</text>'
        )
    return (
        f'<svg class="curve" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{esc(title)}">'
        '<style>text{font-family:"Times New Roman",SimHei,sans-serif;font-size:12px;fill:#222}'
        '.frame{fill:#fff;stroke:#333;stroke-width:1.2}.tick{stroke:#333;stroke-width:1}'
        '.grid{stroke:#ddd;stroke-width:.7}.curve-line{fill:none;stroke:#1769aa;stroke-width:1.35}</style>'
        f'<text x="{width / 2}" y="20" text-anchor="middle">{esc(title)}</text>'
        f'<rect class="frame" x="{left}" y="{top}" width="{plot_w}" height="{plot_h}"/>'
        + "".join(
            f'<path class="grid" d="M {left:.2f} {top + plot_h * i / 5:.2f} h {plot_w:.2f}"/>'
            for i in range(1, 5)
        )
        + "".join(ticks)
        + f'<polyline class="curve-line" points="{points}"/>'
        + f'<text x="{left + plot_w / 2:.2f}" y="{height - 5}" text-anchor="middle">Distance (m)</text>'
        + f'<text transform="translate(17 {top + plot_h / 2:.2f}) rotate(-90)" text-anchor="middle">rho</text>'
        + "</svg>"
    )


def resolve_source(raw_path: str, case: dict) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    base = Path(str(case.get("source_folder") or ""))
    return base / path if str(base) else EXPERIMENT_ROOT / path


def phase_metadata(phase: dict, row: dict | None) -> dict:
    return {
        "phase_name": phase.get("name") if phase else field(row or {}, "SweepName"),
        "frequency_start_hz": phase.get("fstart") if phase else field(row or {}, "Fstart"),
        "frequency_stop_hz": phase.get("fstop") if phase else field(row or {}, "Fstop"),
        "points": phase.get("points") if phase else field(row or {}, "Points"),
        "wave_velocity": phase.get("wave_velocity") if phase else field(row or {}, "Wave_Velocity"),
        "source_axis_max_m": phase.get("x_max_m") if phase else field(row or {}, "X_Max"),
        "x_offset": field(row or {}, "X_Offset"),
    }


def render_file(case: dict, raw_path: str, file_index: int) -> tuple[dict, str]:
    source_path = resolve_source(raw_path, case)
    file_record = {
        "source_path": str(source_path),
        "source_path_in_batch": raw_path,
        "status": "ok",
        "calibration_table_present": None,
        "calibration_rows": None,
        "sweeps": [],
        "errors": [],
        "alignment_warnings": [],
    }
    html_parts = [f'<details class="source-file"><summary>{esc(source_path)}</summary>']
    if not source_path.is_file():
        file_record["status"] = "source_missing"
        file_record["errors"].append("源FDR文件不存在或不可读。")
        html_parts.append('<p class="missing">源FDR文件不存在或不可读。</p></details>')
        return file_record, "".join(html_parts)

    catalog = []
    try:
        catalog = sweep_catalog(source_path)
    except Exception as exc:
        file_record["errors"].append(f"Sweeps目录读取失败：{type(exc).__name__}: {exc}")

    data = None
    try:
        data = read_fdr(str(source_path))
        calibration = data.get("calibration")
        file_record["calibration_table_present"] = calibration is not None
        file_record["calibration_rows"] = len(calibration) if calibration is not None else 0
    except Exception as exc:
        file_record["status"] = "read_failed"
        file_record["errors"].append(f"FDR读取失败：{type(exc).__name__}: {exc}")

    phases = list((data or {}).get("phases") or [])
    measured_rows = [r for r in catalog if is_measured(field(r, "IsMeasured"))]
    aligned = len(measured_rows) == len(phases)
    if not aligned:
        warning = (
            f"reader返回{len(phases)}个phase，Sweeps中标记实测{len(measured_rows)}行；"
            "无法可靠对应rowid，未构造measurement_id。"
        )
        file_record["alignment_warnings"].append(warning)
    phase_by_rowid = {}
    if aligned:
        for phase_index, (row, phase) in enumerate(zip(measured_rows, phases), start=1):
            rowid = row.get("__rowid__")
            row_name = field(row, "SweepName")
            if str(row_name) != str(phase.get("name")):
                file_record["alignment_warnings"].append(
                    f"Sweeps.rowid={rowid}名称与reader phase不一致，未配对响应表。"
                )
                continue
            phase_by_rowid[rowid] = (phase_index, phase)

    outer_rows = catalog or []
    if not outer_rows and phases:
        file_record["alignment_warnings"].append("存在phase但无法读取Sweeps rowid。")
        for phase_index, phase in enumerate(phases, start=1):
            record = {
                "measurement_id": None,
                "sweep_rowid": None,
                "sweep_record_position": None,
                "phase_index": phase_index,
                "phase_name": phase.get("name"),
                "is_measured": True,
                **phase_metadata(phase, None),
                "tables_present": {},
                "curve_points": {},
                "errors": [],
            }
            axis_max = record["source_axis_max_m"]
            record["source_axis_over_2500m"] = (
                float(axis_max) > 2500.0 if axis_max not in (None, "") else None
            )
            html_parts.append(
                f'<details class="phase"><summary>rowid未知 · phase={esc(record["phase_name"])}</summary>'
            )
            html_parts.append(
                '<p class="warning">Sweeps.rowid不可读；保留reader phase信息，不构造measurement_id。</p>'
            )
            arrays = {}
            if "s11" in phase:
                try:
                    arrays = phase_arrays(phase)
                except Exception as exc:
                    record["errors"].append(
                        f"phase_arrays读取失败：{type(exc).__name__}: {exc}"
                    )
            for key, label in (("impulse", "厂商脉冲响应"), ("step", "厂商阶跃响应")):
                record["tables_present"][key] = key in phase
                try:
                    values = response_arrays(phase, arrays, key)
                    if values is None:
                        record["curve_points"][key] = 0
                        html_parts.append(f'<p class="missing">{esc(label)}表缺失。</p>')
                    else:
                        x, y = values
                        record["curve_points"][key] = int(min(x.size, y.size))
                        html_parts.append(curve_svg(x, y, label))
                except Exception as exc:
                    record["curve_points"][key] = 0
                    record["errors"].append(
                        f"{label}表读取失败：{type(exc).__name__}: {exc}"
                    )
                    html_parts.append(
                        f'<p class="missing">{esc(label)}表读取失败：'
                        f'{esc(type(exc).__name__)}: {esc(exc)}</p>'
                    )
            file_record["sweeps"].append(record)
            html_parts.append("</details>")

    for sweep_position, row in enumerate(outer_rows, start=1):
        rowid = row.get("__rowid__")
        measured = is_measured(field(row, "IsMeasured"))
        phase_entry = phase_by_rowid.get(rowid) if measured else None
        phase_index, phase = phase_entry if phase_entry else (None, None)
        raw_name = field(row, "SweepName")
        measurement_id = (
            f"{source_path}#Sweeps.rowid={rowid}" if rowid is not None else None
        )
        record = {
            "measurement_id": measurement_id,
            "sweep_rowid": rowid,
            "sweep_record_position": sweep_position,
            "phase_index": phase_index,
            "phase_name": phase.get("name") if phase else raw_name,
            "is_measured": measured,
            **phase_metadata(phase, row),
            "tables_present": {},
            "curve_points": {},
            "errors": [],
        }
        axis_max = record["source_axis_max_m"]
        record["source_axis_over_2500m"] = (
            float(axis_max) > 2500.0 if axis_max not in (None, "") else None
        )
        anchor = f"m-{file_index}-{sweep_position}"
        html_parts.append(f'<details class="phase" id="{anchor}">')
        html_parts.append(
            f'<summary>rowid={esc(rowid)} · phase={esc(record["phase_name"])} · '
            f'测量={ "是" if measured else "否" }</summary>'
        )
        html_parts.append('<div class="meta"><table>')
        for label, key in (
            ("measurement_id", "measurement_id"),
            ("phase_index", "phase_index"),
            ("频率起点 Hz", "frequency_start_hz"),
            ("频率终点 Hz", "frequency_stop_hz"),
            ("频点数", "points"),
            ("波速", "wave_velocity"),
            ("来源响应轴上限 m", "source_axis_max_m"),
            ("X_Offset", "x_offset"),
        ):
            html_parts.append(f"<tr><th>{esc(label)}</th><td>{esc(record.get(key))}</td></tr>")
        html_parts.append("</table></div>")

        if not measured:
            html_parts.append('<p class="note">该Sweeps行未标记为实测，没有对应响应表。</p>')
        elif phase is None:
            html_parts.append('<p class="missing">没有与真实rowid可靠对齐的phase；响应曲线未配对。</p>')
            record["errors"].append("phase与Sweeps.rowid未可靠对齐。")
        else:
            arrays = {}
            if "s11" in phase:
                try:
                    arrays = phase_arrays(phase)
                except Exception as exc:
                    record["errors"].append(
                        f"phase_arrays读取失败：{type(exc).__name__}: {exc}"
                    )
            for key, label in (("impulse", "厂商脉冲响应"), ("step", "厂商阶跃响应")):
                record["tables_present"][key] = key in phase
                try:
                    values = response_arrays(phase, arrays, key)
                    if values is None:
                        html_parts.append(
                            f'<p class="missing">{esc(label)}表缺失。</p>'
                        )
                        record["curve_points"][key] = 0
                    else:
                        x, y = values
                        record["curve_points"][key] = int(min(x.size, y.size))
                        html_parts.append(curve_svg(x, y, label))
                except Exception as exc:
                    record["curve_points"][key] = 0
                    record["errors"].append(
                        f"{label}表读取失败：{type(exc).__name__}: {exc}"
                    )
                    html_parts.append(
                        f'<p class="missing">{esc(label)}表读取失败：'
                        f'{esc(type(exc).__name__)}: {esc(exc)}</p>'
                    )
        file_record["sweeps"].append(record)
        html_parts.append("</details>")

    if not catalog and not phases:
        html_parts.append('<p class="missing">没有读取到Sweeps行或实测phase。</p>')
    if file_record["calibration_table_present"] is not None:
        present = "有" if file_record["calibration_table_present"] else "无"
        html_parts.append(
            f'<p class="note">CalibrationData表：{present}；'
            f'记录数：{file_record["calibration_rows"]}。</p>'
        )
    for error in file_record["errors"]:
        html_parts.append(f'<p class="missing">{esc(error)}</p>')
    for warning in file_record["alignment_warnings"]:
        html_parts.append(f'<p class="warning">{esc(warning)}</p>')
    if file_record["status"] == "ok" and (file_record["errors"] or file_record["alignment_warnings"]):
        file_record["status"] = "partial"
    html_parts.append("</details>")
    return file_record, "".join(html_parts)


def render_evidence(case: dict) -> str:
    parts = ['<section><h3>来源证据</h3>']
    for item in case.get("evidence") or []:
        parts.append('<article class="evidence">')
        for label, key in (
            ("证据ID", "id"),
            ("来源", "source_path"),
            ("来源类型", "source_kind"),
            ("文件引用", "file_reference"),
            ("相位引用", "phase_reference"),
            ("状态引用", "state_reference"),
            ("位置原文", "position_text"),
            ("支持内容", "supports"),
            ("不确定性", "uncertainty"),
        ):
            value = item.get(key)
            if value not in (None, ""):
                parts.append(f'<p><b>{esc(label)}：</b>{esc(value)}</p>')
        quote = item.get("quote")
        if quote:
            parts.append(f'<blockquote>{esc(quote)}</blockquote>')
        parts.append("</article>")
    if not case.get("evidence"):
        parts.append('<p class="note">该案例没有来源摘录。</p>')
    parts.append("</section>")
    return "".join(parts)


def render_case(case: dict, case_index: int) -> tuple[dict, str]:
    raw_files = case.get("measurement_files") or []
    if isinstance(raw_files, str):
        raw_files = [raw_files]
    raw_files = list(dict.fromkeys(str(p) for p in raw_files))
    file_records, file_html = [], []
    for file_index, raw_path in enumerate(raw_files, start=1):
        record, markup = render_file(case, raw_path, file_index)
        file_records.append(record)
        file_html.append(markup)

    axis_values = [
        float(s["source_axis_max_m"])
        for f in file_records
        for s in f["sweeps"]
        if s.get("is_measured") and s.get("source_axis_max_m") not in (None, "")
    ]
    length_claims = [
        {
            "evidence_id": item.get("id"),
            "source_path": item.get("source_path"),
            "position_text": item.get("position_text"),
            "quote": item.get("quote"),
        }
        for item in case.get("evidence") or []
        if "长度" in str(item.get("source_kind") or "")
    ]
    structured_physical_m = case.get("physical_length_m")
    physical_over = (
        float(structured_physical_m) > 2500.0
        if isinstance(structured_physical_m, (int, float))
        else None
    )
    if str(case.get("case_id")) == "A10":
        screening_status = "reference_only"
        screening_note = "本轮指定为长缆关系参考案例；仍保留读取，用于核对同缆关系。"
    else:
        screening_status = "not_assessed"
        screening_note = "仅列来源轴上限与原文长度记录；本包不据此判定训练资格。"
    length_screen = {
        "source_axis_max_m": max(axis_values) if axis_values else None,
        "source_axis_over_2500m": max(axis_values) > 2500.0 if axis_values else None,
        "physical_length_m": structured_physical_m,
        "physical_length_over_2500m": physical_over,
        "physical_length_claims": length_claims,
        "training_screening": screening_status,
        "screening_note": screening_note,
    }
    parts = [
        f'<section class="case" id="case-{case_index}">',
        f'<h2>{esc(case.get("case_id") or ("case-" + str(case_index)))}</h2>',
        f'<p class="summary">{esc(case.get("summary"))}</p>',
        f'<p class="note"><b>长度核对：</b>来源响应轴最大值 '
        f'{esc(length_screen["source_axis_max_m"])} m；'
        f'来源轴超过2500m：{esc(length_screen["source_axis_over_2500m"])}；'
        f'结构化物理长度：{esc(length_screen["physical_length_m"])} m；'
        f'物理长度超过2500m：{esc(length_screen["physical_length_over_2500m"])}。'
        f'筛选状态：{esc(screening_status)}。{esc(screening_note)}</p>',
        '<h3>物理长度来源记录</h3><ul>',
    ]
    for claim in length_claims:
        parts.append(
            f'<li>{esc(claim.get("evidence_id"))} · {esc(claim.get("source_path"))} '
            f'· {esc(claim.get("quote"))}</li>'
        )
    if not length_claims:
        parts.append("<li>没有独立的长度记录。</li>")
    parts.extend([
        "</ul>",
        '<h3>来源文字</h3><ul>',
    ])
    for source in case.get("source_texts") or []:
        parts.append(f"<li>{esc(source)}</li>")
    parts.append("</ul>")
    parts.append(render_evidence(case))
    parts.append("<h3>测量文件与内部sweep/phase</h3>")
    parts.extend(file_html)
    parts.append("<h3>未决项</h3><ul>")
    for item in case.get("unresolved") or []:
        parts.append(f"<li>{esc(item)}</li>")
    if not case.get("unresolved"):
        parts.append("<li>无。</li>")
    parts.append("</ul>")
    related = case.get("related_cases") or []
    if related:
        parts.append("<h3>相关案例</h3><ul>")
        for item in related:
            parts.append(f"<li>{esc(item)}</li>")
        parts.append("</ul>")
    parts.append("</section>")
    return {
        "case_id": case.get("case_id"),
        "source_folder": case.get("source_folder"),
        "source_texts": case.get("source_texts") or [],
        "summary": case.get("summary"),
        "length_screen": length_screen,
        "evidence_count": len(case.get("evidence") or []),
        "unresolved": case.get("unresolved") or [],
        "files": file_records,
    }, "".join(parts)


def load_batch(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def main() -> None:
    missing = [str(path) for path in INPUTS if not path.is_file()]
    if missing:
        raise SystemExit("输入批次尚未齐全：\n" + "\n".join(missing))

    index = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "inputs": [str(path) for path in INPUTS],
        "cases": [],
    }
    sections = []
    for batch_path in INPUTS:
        batch = load_batch(batch_path)
        for case in batch.get("cases") or []:
            record, markup = render_case(case, len(index["cases"]) + 1)
            record["batch"] = batch.get("batch", batch_path.stem)
            index["cases"].append(record)
            sections.append(markup)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index_path = OUTPUT_DIR / "measurement_index.json"
    html_path = OUTPUT_DIR / "index.html"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    page = (
        "<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>现场案例静态审阅包</title><style>"
        "body{font-family:'Times New Roman',SimHei,sans-serif;line-height:1.55;color:#20242a;"
        "max-width:1180px;margin:24px auto;padding:0 20px;background:#f5f6f8}"
        "h1,h2,h3{font-family:'Times New Roman',SimHei,sans-serif}"
        ".case{background:#fff;border:1px solid #c8ccd2;border-radius:6px;padding:20px;margin:22px 0}"
        ".summary{font-size:1.05em}.evidence{border-left:3px solid #718096;padding:8px 14px;margin:14px 0;background:#f8fafc}"
        "blockquote{white-space:pre-wrap;margin:8px 0;color:#353b45}"
        "details{border:1px solid #cbd0d6;border-radius:4px;background:#fff;margin:10px 0;padding:10px}"
        "summary{cursor:pointer;font-weight:600}.meta table{border-collapse:collapse;margin:10px 0}"
        "th,td{border:1px solid #d6d9de;padding:4px 8px;text-align:left}"
        ".curve{display:block;width:min(100%,760px);height:auto;margin:12px 0}"
        ".missing{color:#9b2525;background:#fff2f0;padding:8px}.warning{color:#805600;background:#fff8df;padding:8px}"
        ".note{color:#555}.source-file{margin-top:12px}"
        "</style></head><body><h1>现场案例静态审阅包</h1>"
        "<p>曲线来自FDR厂商脉冲/阶跃表；显示降采样只用于图形。页面不自动挑峰、不把物理位置映射为响应峰。"
        "测量身份使用源FDR路径与Sweeps真实rowid，phase名称用于显示。</p>"
        + "".join(sections)
        + "</body></html>"
    )
    html_path.write_text(page, encoding="utf-8")

    files = [f for case in index["cases"] for f in case["files"]]
    sweeps = [s for file in files for s in file["sweeps"]]
    print(f"cases: {len(index['cases'])}")
    print(f"FDR files: {len(files)}; sweep rows: {len(sweeps)}")
    print(f"read failures: {sum(f['status'] != 'ok' for f in files)}")
    print(f"alignment warnings: {sum(len(f['alignment_warnings']) for f in files)}")
    print(f"index: {html_path}")
    print(f"measurement index: {index_path}")


if __name__ == "__main__":
    main()
