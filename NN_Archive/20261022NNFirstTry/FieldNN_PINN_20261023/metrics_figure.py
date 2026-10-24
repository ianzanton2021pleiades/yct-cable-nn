"""Grouped bar chart of quality metrics and operating cost across checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.sans-serif"] = ["Times New Roman", "SimHei"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True


def unwrap(node):
    if isinstance(node, dict) and "report" in node:
        return node["report"]
    return node


def pick(node, *keys, default=None):
    current = node
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def extract(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    clean = unwrap(pick(data, "synthetic_test", "clean", default={}))
    field = pick(data, "field_test", default={}) or {}
    hybrid = pick(data, "hybrid_test", default={}) or {}
    annotation = pick(field, "annotation_summary", default={}) or {}
    return {
        "syn_f1_5": pick(clean, "event_metrics_by_tolerance_m", "5", "f1"),
        "syn_f1_2": pick(clean, "event_metrics_by_tolerance_m", "2", "f1"),
        "syn_interval": pick(clean, "interval_metrics", "f1"),
        "syn_terminal": pick(clean, "terminal_metrics", "f1"),
        "hybrid_hit20": pick(hybrid, "non_wet", "hit_rate_20"),
        "field_hit20": pick(field, "non_wet", "hit_rate_20"),
        "field_hit10": pick(field, "non_wet", "hit_rate_10"),
        "strict_hit20": pick(field, "strict", "hit_rate_20"),
        "wet_hit20": pick(field, "wet_1500", "hit_rate_20"),
        "p_ann": annotation.get("P_ann"),
        "field_cand": pick(field, "non_wet", "candidates_per_curve"),
        "hybrid_cand": pick(hybrid, "non_wet", "candidates_per_curve"),
        "all_hit": annotation.get("curves_all_positions_hit"),
        "any_hit": annotation.get("curves_any_position_hit"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", nargs="+", required=True, help="label=path")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    labels, rows = [], []
    for entry in args.entry:
        label, _, path = entry.partition("=")
        labels.append(label)
        rows.append(extract(Path(path)))

    quality = [
        ("Syn F1@5m", "syn_f1_5"), ("Syn F1@2m", "syn_f1_2"), ("Syn interval", "syn_interval"),
        ("Syn terminal", "syn_terminal"), ("Hybrid hit@20", "hybrid_hit20"),
        ("Field hit@20", "field_hit20"), ("Field hit@10", "field_hit10"),
        ("Strict hit@20", "strict_hit20"), ("Wet1500 hit@20", "wet_hit20"), ("P_ann", "p_ann"),
    ]
    colors = ["#9e9e9e", "#78909c", "#1f4e9c", "#2e7d32", "#ef6c00"]

    figure, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), gridspec_kw={"width_ratios": [2.6, 0.9, 0.9]})
    axis = axes[0]
    width = 0.8 / len(rows)
    index = np.arange(len(quality))
    for number, (label, row) in enumerate(zip(labels, rows)):
        values = [row[key] if row[key] is not None else np.nan for _, key in quality]
        axis.bar(index + number * width - 0.4 + width / 2, values, width * 0.92,
                 label=label, color=colors[number % len(colors)], edgecolor="white", linewidth=0.4)
    axis.set_xticks(index)
    axis.set_xticklabels([name for name, _ in quality], rotation=38, ha="right", fontsize=8)
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("score")
    axis.set_title("Quality metrics (higher is better)", fontsize=10)
    axis.legend(frameon=False, fontsize=8)
    axis.grid(axis="y", alpha=0.25)

    axis = axes[1]
    cost = [("Field", "field_cand"), ("Hybrid", "hybrid_cand")]
    width = 0.8 / len(rows)
    index = np.arange(len(cost))
    for number, (label, row) in enumerate(zip(labels, rows)):
        values = [row[key] if row[key] is not None else np.nan for _, key in cost]
        axis.bar(index + number * width - 0.4 + width / 2, values, width * 0.92,
                 label=label, color=colors[number % len(colors)], edgecolor="white", linewidth=0.4)
    axis.set_xticks(index)
    axis.set_xticklabels([name for name, _ in cost], fontsize=9)
    axis.set_ylabel("candidates / curve")
    axis.set_title("Candidate density", fontsize=10)
    axis.grid(axis="y", alpha=0.25)

    axis = axes[2]
    coverage = [("all hit", "all_hit"), ("any hit", "any_hit")]
    width = 0.8 / len(rows)
    index = np.arange(len(coverage))
    for number, (label, row) in enumerate(zip(labels, rows)):
        values = [row[key] if row[key] is not None else np.nan for _, key in coverage]
        axis.bar(index + number * width - 0.4 + width / 2, values, width * 0.92,
                 label=label, color=colors[number % len(colors)], edgecolor="white", linewidth=0.4)
        for position, value in enumerate(values):
            if value == value:
                axis.text(position + number * width - 0.4 + width / 2, value + 0.4, f"{int(value)}",
                          ha="center", fontsize=7)
    axis.set_xticks(index)
    axis.set_xticklabels([name for name, _ in coverage], fontsize=9)
    axis.set_ylabel("annotated curves")
    axis.set_title("Annotated coverage (43 curves)", fontsize=10)
    axis.grid(axis="y", alpha=0.25)

    figure.suptitle("PINN U-Net vs CMDC baseline", fontsize=12)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)
    plt.close(figure)
    print(json.dumps({label: row for label, row in zip(labels, rows)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
