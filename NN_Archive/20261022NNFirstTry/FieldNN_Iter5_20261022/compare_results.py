"""Compare Iter4 and Iter5 synthetic/field results into one markdown table."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ITER4 = ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_Iter4_20261022"
ITER5 = HERE


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def head(report: dict) -> dict:
    m = report["metrics"]
    return {
        "event": m["event_metrics_by_tolerance_m"]["5"],
        "interval": m["interval_metrics"],
        "terminal": m["terminal_metrics"],
        "fp": report.get("false_positive_summary", {}),
        "epoch": report.get("checkpoint_epoch"),
        "thresholds": report.get("thresholds"),
    }


def field_row(summary_path: Path, predictions_path: Path) -> dict | None:
    summary = load(summary_path)
    if summary is None or not predictions_path.is_file():
        return None
    rows = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    predicted = [row for row in rows if row.get("status") == "predicted"]
    counts = sorted(len(row["prediction"]["events"]) for row in predicted)
    known = (summary.get("known_lab_positive_event_summary") or {}).get("by_tolerance", {})

    def hits(tolerance: str):
        return known.get(tolerance, {}).get("known_event_hits")

    sweeps = (summary.get("known_lab_positive_event_summary") or {}).get("sweeps_evaluated")
    return {
        "known": [hits("10"), hits("20"), hits("30")],
        "sweeps": sweeps,
        "events_total": sum(counts),
        "events_mean": sum(counts) / len(counts) if counts else None,
        "events_median": counts[len(counts) // 2] if counts else None,
        "near_end_suppression_m": summary.get("near_end_suppression_m", 0.0),
    }


def fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def known_text(field: dict | None) -> str:
    if not field or field.get("known") is None:
        return "—"
    a, b, c = field["known"]
    return f"{a}/{b}/{c} of {field['sweeps']}"


def main() -> None:
    summary5 = load(ITER5 / "output" / "iter5" / "iteration5_summary.json")
    print("| 模型 | 版本 | event P/R/F1@5m | interval F1 | terminal F1 | 现场已知@10/20/30m | 候选/条 | 近端抑制 |")
    print("|---|---|---|---|---|---|---|---|")
    for model in ("unet", "transformer"):
        sub = "field_e022" if model == "unet" else "field_e016"
        report4 = load(
            ITER4 / "output" / "adaptation4" / model / "evaluation_test_mixed" / "evaluation.json"
        )
        if report4:
            h = head(report4)
            f4 = field_row(
                ITER4 / "output" / "adaptation4" / model / sub / "summary.json",
                ITER4 / "output" / "adaptation4" / model / sub / "predictions.jsonl",
            )
            print(
                f"| {model} | Iter4 | {fmt(h['event']['precision'])}/{fmt(h['event']['recall'])}/"
                f"{fmt(h['event']['f1'])} | {fmt(h['interval']['f1'])} | {fmt(h['terminal']['f1'])} | "
                f"{known_text(f4)} | {fmt(f4['events_mean'] if f4 else None, 2)} | 0 m |"
            )
        entry = summary5["models"].get(model) if summary5 else None
        if entry:
            for view in ("mixed", "clean"):
                syn = entry["synthetic"].get(view)
                if not syn:
                    continue
                f = entry.get("field") if view == "mixed" else None
                counts = f["counts"] if f else None
                known = f["known_lab"]["by_tolerance"] if f else None
                if known:
                    known_value = (
                        f"{known['10']['known_event_hits']}/{known['20']['known_event_hits']}/"
                        f"{known['30']['known_event_hits']} of "
                        f"{known['10']['known_positive_sweeps_evaluated']}"
                    )
                else:
                    known_value = "—"
                print(
                    f"| {model} | Iter5 {view} | {fmt(syn['event']['precision'])}/"
                    f"{fmt(syn['event']['recall'])}/{fmt(syn['event']['f1'])} | "
                    f"{fmt(syn['interval']['f1'])} | {fmt(syn['terminal']['f1'])} | "
                    f"{known_value} | {fmt(counts['events_per_curve_mean'] if counts else None, 2)} | 30 m |"
                )
            print(f"  thresholds={entry.get('thresholds')}")
    print()
    if summary5:
        for model, entry in summary5["models"].items():
            counts = (entry.get("field") or {}).get("counts")
            if counts:
                print(f"\n{model} field group event medians:")
                for group, value in counts["group_event_medians"].items():
                    print(f"  {value:4.1f}  {group}")


if __name__ == "__main__":
    main()
