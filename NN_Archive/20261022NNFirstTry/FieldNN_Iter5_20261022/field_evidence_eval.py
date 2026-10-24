"""Score field predictions against engineer-written positions in field_evidence.json.

The annotations are human FDR readings, not independent ground truth. A hit means
a predicted event lies within the tolerance of an annotated position. Extra
candidates are reported as unverified, never as false positives.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "field_evidence.json"
TOLERANCES = (10.0, 20.0, 30.0)


def key(source: str, rowid) -> tuple[str, str]:
    return (str(source).replace("\\", "/").casefold(), str(rowid))


def load_evidence(path: Path) -> dict[tuple[str, str], list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    index: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in data["records"]:
        positions = [float(value) for value in (row.get("positions_m") or [])]
        rowids = row.get("sweep_rowids") or []
        if not positions or not rowids:
            continue
        for rowid in rowids:
            index[key(row["source_relative_path"], rowid)].append({
                "positions": positions,
                "claim": row.get("claim"),
                "confidence": row.get("confidence"),
                "evidence_type": row.get("evidence_type"),
                "case_group": row.get("case_group"),
                "quote": row.get("quote"),
            })
    return index


def evaluate(predictions_path: Path, index: dict, exclude_case: str | None) -> dict:
    rows = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    predicted = [row for row in rows if row.get("status") == "predicted"]
    checks = []
    case_checks: dict[tuple[str, str, float], list[float]] = defaultdict(list)
    candidates_total = 0
    curves = 0
    for row in predicted:
        record = row.get("record", {})
        k = key(record.get("source_relative_path") or record.get("input_path") or "", record.get("sweep_rowid"))
        annotations = index.get(k, [])
        if exclude_case and any(item["case_group"] == exclude_case for item in annotations):
            continue
        events = [float(event["distance_m"]) for event in row["prediction"]["events"]]
        candidates_total += len(events)
        curve_has_annotation = False
        for item in annotations:
            if item["claim"] not in {"defect_position", "joint_position"}:
                continue
            curve_has_annotation = True
            for position in item["positions"]:
                error = min((abs(event - position) for event in events), default=None)
                nearest = None if error is None else min(events, key=lambda e: abs(e - position))
                checks.append({
                    "case_group": item["case_group"],
                    "claim": item["claim"],
                    "confidence": item["confidence"],
                    "position_m": position,
                    "nearest_m": nearest,
                    "error_m": error,
                    "hit10": bool(error is not None and error <= 10.0),
                    "hit20": bool(error is not None and error <= 20.0),
                    "hit30": bool(error is not None and error <= 30.0),
                })
                if error is not None:
                    case_checks[(item["case_group"], item["claim"], position)].append(error)
                else:
                    case_checks.setdefault((item["case_group"], item["claim"], position), [])
        if curve_has_annotation:
            curves += 1

    def summary(items):
        count = len(items)
        return {
            "positions": count,
            "hit10": sum(item["hit10"] for item in items),
            "hit20": sum(item["hit20"] for item in items),
            "hit30": sum(item["hit30"] for item in items),
            "hit10_rate": sum(item["hit10"] for item in items) / count if count else None,
            "hit20_rate": sum(item["hit20"] for item in items) / count if count else None,
            "hit30_rate": sum(item["hit30"] for item in items) / count if count else None,
            "median_error_m": sorted(item["error_m"] for item in items if item["error_m"] is not None)[count // 2] if count else None,
        }

    case_items = []
    for (case, claim, position), errors in case_checks.items():
        best = min(errors) if errors else None
        case_items.append({
            "case_group": case,
            "claim": claim,
            "position_m": position,
            "error_m": best,
            "hit10": bool(best is not None and best <= 10.0),
            "hit20": bool(best is not None and best <= 20.0),
            "hit30": bool(best is not None and best <= 30.0),
        })
    per_case: dict[str, list[bool]] = defaultdict(list)
    for item in case_items:
        per_case[item["case_group"]].append(item["hit20"])

    return {
        "curves_with_annotations": curves,
        "candidates_on_annotated_curves": candidates_total,
        "candidates_per_annotated_curve": candidates_total / curves if curves else None,
        "per_sweep_position": summary(checks),
        "case_position": summary(case_items),
        "case_position_by_claim": {
            claim: summary([item for item in case_items if item["claim"] == claim])
            for claim in ("defect_position", "joint_position")
        },
        "by_claim": {
            claim: summary([item for item in checks if item["claim"] == claim])
            for claim in ("defect_position", "joint_position")
        },
        "by_confidence": {
            confidence: summary([item for item in checks if item["confidence"] == confidence])
            for confidence in ("high", "medium", "low")
        },
        "per_case_hit20": {
            case: {"positions": len(values), "hits20": sum(values)}
            for case, values in sorted(per_case.items())
        },
        "checks": checks,
        "case_checks": case_items,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", action="append", required=True,
                    help="repeat for each model/version; label=path")
    ap.add_argument("--exclude-case", default="2026-6某地进行的1500m不同位置电缆接头浸水试验")
    ap.add_argument("--output", type=Path, default=HERE / "output" / "field_evidence_eval.json")
    args = ap.parse_args()
    index = load_evidence(EVIDENCE)
    result = {}
    for item in args.predictions:
        label, text = item.split("=", 1)
        result[label] = evaluate(Path(text), index, args.exclude_case)
        print(label, json.dumps({
            "curves_with_annotations": result[label]["curves_with_annotations"],
            "candidates_per_annotated_curve": result[label]["candidates_per_annotated_curve"],
            "case_position": result[label]["case_position"],
            "case_position_by_claim": result[label]["case_position_by_claim"],
        }, ensure_ascii=False))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("written", args.output)


if __name__ == "__main__":
    main()
