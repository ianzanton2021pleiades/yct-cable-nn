"""Render one concise model-comparison table from an exporter performance.csv."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "output" / "existing_iter8" / "performance.csv"


def number(row: dict[str, str], key: str) -> float | None:
    value = (row.get(key) or "").strip()
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def first_nonempty(rows: list[dict[str, str]], key: str) -> str:
    return next(((row.get(key) or "").strip() for row in rows if (row.get(key) or "").strip()), "")


def fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def count_rate(row: dict[str, str], hit_key: str) -> str:
    hits = number(row, hit_key)
    positions = number(row, "known_positions")
    if hits is None or positions is None:
        return "—"
    rate = number(row, hit_key.replace("hits_at_", "micro_hit_at_"))
    suffix = f" ({rate:.0%})" if rate is not None else ""
    return f"{hits:.0f}/{positions:.0f}{suffix}"


def benchmark_matches(csv_label: str, benchmark_label: str) -> bool:
    csv_key = "".join(char for char in csv_label.casefold() if char.isalnum())
    bench_key = "".join(char for char in benchmark_label.casefold() if char.isalnum())
    aliases = {
        "u5": "iter5unet",
        "u8": "iter8unet",
        "t8": "iter8transformer",
        "i10": "iter10",
        "i11": "iter11",
    }
    prefix = aliases.get(bench_key)
    if prefix is None and bench_key.startswith("i10"):
        prefix = "iter10"
    if prefix is None and bench_key.startswith("i11"):
        prefix = "iter11"
    prefix = prefix or bench_key
    return csv_key == bench_key or (prefix and csv_key.startswith(prefix))


def model_rows(
    source_rows: list[dict[str, str]],
    benchmark_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    populations: dict[tuple[str, str], list[dict[str, str]]] = {}
    artifacts: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in source_rows:
        model = (row.get("model") or "").strip()
        threshold = (row.get("threshold") or "").strip()
        key = (model, threshold)
        if row.get("row_type") == "field_population" and row.get("population") in {
            "nonwet_test",
            "wet_1500m_test",
        }:
            populations.setdefault(key, []).append(row)
        elif row.get("row_type") == "model_artifact":
            artifacts.setdefault(key, []).append(row)

    output = []
    for (model, threshold), rows in populations.items():
        nonwet = next((row for row in rows if row.get("population") == "nonwet_test"), {})
        wet = next((row for row in rows if row.get("population") == "wet_1500m_test"), {})
        artifact_rows = artifacts.get((model, threshold), [])
        artifact = artifact_rows[-1] if artifact_rows else {}
        benchmark = next(
            (entry for entry in benchmark_rows if benchmark_matches(model, str(entry.get("label", "")))),
            {},
        )
        params = number(nonwet, "parameters") or number(artifact, "parameters")
        weight_bytes = number(nonwet, "model_state_tensor_bytes") or number(
            artifact, "model_state_tensor_bytes"
        )
        checkpoint_bytes = number(nonwet, "checkpoint_bytes") or number(artifact, "checkpoint_bytes")
        inference_package_bytes = number(nonwet, "inference_package_bytes") or number(
            artifact, "inference_package_bytes"
        )
        median_ms = number(nonwet, "latency_median_ms") or number(artifact, "latency_median_ms")
        p95_ms = number(nonwet, "latency_p95_ms") or number(artifact, "latency_p95_ms")
        params = params or benchmark.get("parameters")
        weight_bytes = weight_bytes or benchmark.get("weight_tensor_bytes")
        checkpoint_bytes = checkpoint_bytes or benchmark.get("checkpoint_bytes")
        median_ms = median_ms or benchmark.get("median_ms")
        p95_ms = p95_ms or benchmark.get("p95_ms")
        inference_package_bytes = inference_package_bytes or benchmark.get("inference_package_bytes")

        missing = []
        if params is None:
            missing.append("参数量未记录")
        if weight_bytes is None:
            missing.append("模型tensor字节未记录")
        if checkpoint_bytes is None:
            missing.append("checkpoint大小未记录")
        if inference_package_bytes is None:
            missing.append("推理包大小未测")
        if median_ms is None or p95_ms is None:
            missing.append("NN latency median/p95未测")
        if not nonwet or number(nonwet, "known_positions") is None:
            missing.append("非湿test没有可用正位置指标")

        nonwet_curves = number(nonwet, "curves")
        wet_curves = number(wet, "curves")
        nonwet_candidates = number(nonwet, "candidate_events_total")
        wet_candidates = number(wet, "candidate_events_total")
        output.append(
            {
                "model": model,
                "threshold": float(threshold) if threshold else None,
                "nonwet_hits": "\n".join(
                    count_rate(nonwet, key).split(" ", 1)[0]
                    for key in ("hits_at_10m", "hits_at_20m", "hits_at_30m")
                ),
                "wet_hits": "\n".join(
                    count_rate(wet, key).split(" ", 1)[0]
                    for key in ("hits_at_10m", "hits_at_20m", "hits_at_30m")
                ),
                "groups_positions": (
                    f"{number(nonwet, 'groups') or 0:.0f}组/{number(nonwet, 'known_positions') or 0:.0f}位"
                    if nonwet
                    else "—"
                ),
                "macro20": number(nonwet, "group_macro_hit_at_20m"),
                "candidates": (
                    f"{nonwet_candidates:.0f}/{nonwet_curves:.0f}="
                    f"{fmt(nonwet_candidates / nonwet_curves if nonwet_curves else None)}"
                    if nonwet_candidates is not None and nonwet_curves is not None
                    else "—"
                ),
                "matched_error": number(nonwet, "mean_matched_error_at_20m"),
                "parameters": f"{params:,.0f}" if params is not None else "—",
                "sizes": (
                    f"{fmt(weight_bytes / 1024**2)} / {fmt(checkpoint_bytes / 1024**2)} MiB"
                    if weight_bytes is not None and checkpoint_bytes is not None
                    else "—"
                ),
                "latency": (
                    f"{fmt(median_ms)} / {fmt(p95_ms)} ms"
                    if median_ms is not None and p95_ms is not None
                    else "—"
                ),
                "missing": "；".join(missing) if missing else "—",
                "wet_candidates": wet_candidates,
                "inference_package_bytes": inference_package_bytes,
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--title", default="FieldNN 现场结果综合表")
    args = parser.parse_args()

    output = args.output or args.input.with_name("performance_table.png")
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(output.parent / "matplotlib_cache")
    os.environ["MPLBACKEND"] = "Agg"

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "SimHei"],
            "font.sans-serif": ["SimHei"],
            "axes.unicode_minus": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "savefig.dpi": 200,
            "figure.dpi": 200,
            "font.size": 9,
        }
    )

    with args.input.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    benchmark_payload = (
        json.loads(args.benchmark.read_text(encoding="utf-8"))
        if args.benchmark is not None and args.benchmark.is_file()
        else {}
    )
    benchmark_rows = benchmark_payload.get("models", [])
    summary = model_rows(rows, benchmark_rows)
    if not summary:
        raise ValueError(f"No field_population nonwet_test rows found in {args.input}")

    columns = [
        "模型 / event阈值",
        "非湿 test hit@10/20/30",
        "湿接头 test hit@10/20/30",
        "组 / 正位置",
        "group macro@20",
        "候选数 / 曲线",
        "matched error@20 (m)",
        "参数量",
        "tensor权重 / checkpoint",
        "NN median / p95",
        "缺项说明",
    ]
    cell_rows = []
    for row in summary:
        display_model = row['model'].split(' (', 1)[0].replace('-Transformer', '-\nTransformer')
        if 'same checkpoint' in row['model']:
            display_model += '\n覆盖点（CPU缓存）'
        elif 'historical fallback' in row['model']:
            display_model += '\n历史参考'
        elif '(bestU)' in row['model']:
            display_model += '\n主工作点'
        cell_rows.append(
            [
                f"{display_model}\n{fmt(row['threshold'], 3)}",
                row["nonwet_hits"],
                row["wet_hits"],
                row["groups_positions"],
                fmt(row["macro20"], 3),
                row["candidates"],
                fmt(row["matched_error"], 2),
                row["parameters"],
                row["sizes"],
                row["latency"],
                row["missing"],
            ]
        )

    fig_height = max(7.0, 3.0 + 1.15 * len(cell_rows))
    fig, ax = plt.subplots(figsize=(20, fig_height))
    ax.axis("off")
    ax.set_title(args.title, fontsize=15, pad=18)
    widths = [0.10, 0.115, 0.105, 0.075, 0.075, 0.09, 0.085, 0.07, 0.10, 0.09, 0.095]
    table = ax.table(
        cellText=cell_rows,
        colLabels=columns,
        colWidths=widths,
        cellLoc="center",
        colLoc="center",
        bbox=[0.0, 0.16, 1.0, 0.78],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.8)
    for (row_idx, _), cell in table.get_celld().items():
        cell.set_edgecolor("#606060")
        cell.set_linewidth(0.55)
        cell.get_text().set_wrap(True)
        if row_idx == 0:
            cell.set_facecolor("#2e5578")
            cell.get_text().set_color("white")
        else:
            cell.set_facecolor("#f1f4f7" if row_idx % 2 == 0 else "#ffffff")
            if _ == 10 and summary[row_idx - 1]["missing"] != "—":
                cell.set_facecolor("#fff2d6")

    fig.text(
        0.01,
        0.07,
        "Field指标是正位置袋 hit，不代表完整现场precision/recall/F1；未标注区域不是负例，未匹配候选仍未核验。"
        "“—”表示来源文件未提供或该项未测；推理时延只在NN benchmark存在时填写。",
        fontsize=9,
        ha="left",
        va="center",
    )
    benchmark_scope = benchmark_payload.get("scope") or next(
        (row.get("scope") for row in benchmark_rows if row.get("scope")),
        "NN-only forward；输入预载并驻留device，不含H2D/FFT/FDR/后处理。",
    )
    fig.text(
        0.01,
        0.035,
        f"主工作点由验证集确定；固定0.60为预设覆盖补充，阈值不同不等于相同候选预算。每配置单次训练N=1；std/p未测，完整现场Accuracy不可计算。权重tensor与checkpoint整文件大小分列。时延范围：{benchmark_scope}",
        fontsize=8,
        ha="left",
        va="center",
    )
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output} ({len(cell_rows)} model/workpoint rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
