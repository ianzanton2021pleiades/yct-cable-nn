"""Build a markdown comparison table from several evaluation.json files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def unwrap(node: Any) -> Any:
    if isinstance(node, dict) and "report" in node:
        return node["report"]
    return node


def pick(node: Any, *keys, default=None):
    current = node
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def row_from(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    clean = unwrap(pick(data, "synthetic_test", "clean", default={}))
    mixed = unwrap(pick(data, "synthetic_test", "mixed", default={}))
    field = pick(data, "field_test", default={}) or {}
    hybrid = pick(data, "hybrid_test", default={}) or {}
    calibration = pick(data, "calibration_test", default={}) or {}
    annotation = pick(field, "annotation_summary", default={}) or {}
    return {
        "path": str(path),
        "epoch": data.get("epoch"),
        "syn_f1_5": pick(clean, "event_metrics_by_tolerance_m", "5", "f1", default=float("nan")),
        "syn_p_5": pick(clean, "event_metrics_by_tolerance_m", "5", "precision", default=float("nan")),
        "syn_r_5": pick(clean, "event_metrics_by_tolerance_m", "5", "recall", default=float("nan")),
        "syn_f1_mixed": pick(mixed, "event_metrics_by_tolerance_m", "5", "f1", default=float("nan")),
        "interval_f1": pick(clean, "interval_metrics", "f1", default=float("nan")),
        "terminal_f1": pick(clean, "terminal_metrics", "f1", default=float("nan")),
        "hybrid_hit20": pick(hybrid, "non_wet", "hit_rate_20", default=float("nan")),
        "hybrid_cand": pick(hybrid, "non_wet", "candidates_per_curve", default=float("nan")),
        "field_hit20": pick(field, "non_wet", "hit_rate_20", default=float("nan")),
        "field_cand": pick(field, "non_wet", "candidates_per_curve", default=float("nan")),
        "field_strict": pick(field, "strict", "hit_rate_20", default=float("nan")),
        "field_wet": pick(field, "wet_1500", "hit_rate_20", default=float("nan")),
        "p_ann": annotation.get("P_ann", float("nan")),
        "curves_all_hit": annotation.get("curves_all_positions_hit"),
        "curves_any_hit": annotation.get("curves_any_position_hit"),
        "annotated": annotation.get("annotated_curves"),
        "short_false": calibration.get("false_events_per_reference", float("nan")),
    }


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and value != value:
        return "-"
    return f"{float(value):.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", nargs="+", required=True, help="label=path pairs")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for entry in args.entry:
        label, _, path = entry.partition("=")
        row = row_from(Path(path))
        row["label"] = label
        rows.append(row)

    header = (
        "| 方案 | Syn F1@5m | Syn P@5m | Syn interval F1 | Syn terminal F1 | "
        "Hybrid hit@20 | 候选/条 | 现场 hit@20 | 现场候选/条 | strict | wet1500 | P_ann | 全命中曲线 |"
    )
    separator = "|" + "---|" * 13
    lines = [header, separator]
    for row in rows:
        lines.append(
            "| {label} | {a} | {b} | {c} | {d} | {e} | {f} | {g} | {h} | {i} | {j} | {k} | {l} |".format(
                label=row["label"],
                a=fmt(row["syn_f1_5"]), b=fmt(row["syn_p_5"]),
                c=fmt(row["interval_f1"]), d=fmt(row["terminal_f1"]),
                e=fmt(row["hybrid_hit20"]), f=fmt(row["hybrid_cand"], 2),
                g=fmt(row["field_hit20"]), h=fmt(row["field_cand"], 2),
                i=fmt(row["field_strict"]), j=fmt(row["field_wet"]),
                k=fmt(row["p_ann"]), l=row["curves_all_hit"],
            )
        )
    text = "\n".join(lines)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
