"""Build the final Iter5 summary from the locked checkpoints and evaluations."""

from __future__ import annotations

import json
from pathlib import Path

import field_evidence_eval as evidence
from run_iteration5 import field_counts

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ITER4 = ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_Iter4_20261022"
OUT = HERE / "output" / "iter5"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def synthetic(path: Path) -> dict | None:
    report = load(path)
    if report is None:
        return None
    return {
        "checkpoint_epoch": report["checkpoint_epoch"],
        "thresholds": report["thresholds"],
        "event": report["metrics"]["event_metrics_by_tolerance_m"]["5"],
        "interval": report["metrics"]["interval_metrics"],
        "terminal": report["metrics"]["terminal_metrics"],
        "false_positive_summary": report["false_positive_summary"],
    }


def main() -> None:
    index = evidence.load_evidence(evidence.EVIDENCE)
    summary = {
        "schema": "fieldnn-iter5-final-1",
        "data": r"E:\CableNNDataSet\CodeX\FieldNN_Iter4_20261022",
        "training": "fine-tuned from Iter4 best with width-aware targets and 1-6x fixture bank",
        "near_end_suppression_m": 30.0,
        "operating_point": "event threshold selected by val precision target 0.90; U-Net locked at 0.6",
        "models": {},
    }
    for model in ("unet", "transformer"):
        run = OUT / model
        entry = {
            "thresholds": load(run / "thresholds.json")["thresholds"],
            "synthetic": {
                view: synthetic(run / f"evaluation_test_{view}" / "evaluation.json")
                for view in ("mixed", "clean")
            },
        }
        latest = load(run / "latest_field.json")
        if latest:
            field_dir = Path(latest["output"])
            field = load(field_dir / "summary.json")
            entry["field"] = {
                "output": str(field_dir),
                "known_lab": field.get("known_lab_positive_event_summary"),
                "counts": field_counts(field_dir / "predictions.jsonl"),
                "evidence": {
                    key: value
                    for key, value in evidence.evaluate(
                        field_dir / "predictions.jsonl", index,
                        "2026-6某地进行的1500m不同位置电缆接头浸水试验",
                    ).items()
                    if key not in {"checks", "case_checks", "per_case_hit20"}
                },
            }
        for label, path in (
            ("iter4_baseline_mixed", ITER4 / "output" / "adaptation4" / "baseline" / model / "evaluation_test" / "evaluation.json"),
            ("iter4_candidate_mixed", ITER4 / "output" / "adaptation4" / model / "evaluation_test_mixed" / "evaluation.json"),
            ("iter4_candidate_clean", ITER4 / "output" / "adaptation4" / model / "evaluation_test_clean" / "evaluation.json"),
        ):
            report = synthetic(path)
            if report:
                entry.setdefault("reference", {})[label] = report
        summary["models"][model] = entry
    (OUT / "iteration5_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("written", OUT / "iteration5_summary.json")


if __name__ == "__main__":
    main()
