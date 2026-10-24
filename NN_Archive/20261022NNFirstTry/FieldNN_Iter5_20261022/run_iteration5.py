"""Iter5: width-aware retraining, precision-first calibration, field evaluation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ITER3 = HERE.parent / "FieldNN_Iter3_20261022"
ITER4 = HERE.parent / "FieldNN_Iter4_20261022"
for path in (ITER3, HERE):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from run_iteration import reserve_one_core


def run_step(name: str, command: list[str], output: Path) -> dict:
    log = output / "logs" / (name + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print("\n=== " + name + " ===\n" + subprocess.list2cmdline(command), flush=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write("\n" + time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
        stream.write(subprocess.list2cmdline(command) + "\n")
        with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            cwd=str(HERE),
        ) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)
    return {"stage": name, "seconds": time.perf_counter() - started, "log": str(log)}


def field_counts(predictions_path: Path) -> dict:
    rows = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    predicted = [row for row in rows if row.get("status") == "predicted"]
    events = [len(row["prediction"]["events"]) for row in predicted]
    intervals = [len(row["prediction"]["intervals"]) for row in predicted]
    by_group: dict[str, list[int]] = {}
    for row in predicted:
        if row.get("record", {}).get("label_status") != "unknown":
            continue
        group = row["record"].get("cable_group", "unknown")
        by_group.setdefault(group, []).append(len(row["prediction"]["events"]))
    return {
        "curves": len(predicted),
        "events_total": int(sum(events)),
        "events_per_curve_mean": float(sum(events) / len(events)) if events else None,
        "events_per_curve_median": float(sorted(events)[len(events) // 2]) if events else None,
        "intervals_total": int(sum(intervals)),
        "intervals_per_curve_mean": float(sum(intervals) / len(intervals)) if intervals else None,
        "near_end_policy": "first 30 m suppressed at decode",
        "group_event_medians": {
            group: float(sorted(values)[len(values) // 2])
            for group, values in sorted(by_group.items())
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path(r"E:\CableNNDataSet\CodeX\FieldNN_Iter4_20261022"))
    ap.add_argument("--output", type=Path, default=HERE / "output" / "iter5")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip-field", action="store_true")
    ap.add_argument("--models", default="unet,transformer")
    args = ap.parse_args()
    os.environ.update({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONUTF8": "1",
        "PYTHONUNBUFFERED": "1",
    })
    cpu = reserve_one_core()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(args.output / "matplotlib_cache")
    prefix = [sys.executable, "-X", "utf8", "-B"]
    completed = []

    def stage(name, script, *params, view="mixed"):
        os.environ["FIELDNN_EVAL_VIEW"] = view
        completed.append(run_step(name, prefix + [str(script)] + [str(p) for p in params], args.output))
        (args.output / "pipeline_status.json").write_text(
            json.dumps(
                {"cpu": cpu, "data": str(args.data), "stages": completed},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    print(json.dumps({"cpu": cpu, "data": str(args.data), "output": str(args.output)}), flush=True)
    models = [name.strip() for name in args.models.split(",") if name.strip()]
    for model in models:
        run = args.output / model
        source_checkpoint = ITER4 / "output" / "adaptation4" / model / "best.pt"
        if args.resume and (run / "last.pt").exists():
            restoration = ["--resume"]
        else:
            restoration = ["--init-checkpoint", str(source_checkpoint)]
        stage(
            f"{model}_train",
            HERE / "train.py",
            "--data", args.data, "--output", args.output, "--model", model,
            "--epochs", args.epochs, "--batch-size", args.batch_size,
            "--workers", args.workers, "--device", args.device, *restoration,
        )
        stage(
            f"{model}_val",
            HERE / "evaluate.py",
            "--data", args.data, "--checkpoint", run / "best.pt",
            "--split", "val", "--calibrate", "--precision-target", "0.90",
            "--batch-size", args.batch_size, "--device", args.device,
        )
        stage(
            f"{model}_test_mixed",
            HERE / "evaluate.py",
            "--data", args.data, "--checkpoint", run / "best.pt",
            "--split", "test", "--batch-size", args.batch_size, "--device", args.device,
            "--output", run / "evaluation_test_mixed",
        )
        stage(
            f"{model}_test_clean",
            HERE / "evaluate.py",
            "--data", args.data, "--checkpoint", run / "best.pt",
            "--split", "test", "--batch-size", args.batch_size, "--device", args.device,
            "--output", run / "evaluation_test_clean", view="clean",
        )
        if not args.skip_field:
            calibration = json.loads((run / "thresholds.json").read_text(encoding="utf-8"))
            field_out = run / f"field_e{int(calibration['checkpoint_epoch']) + 1:03d}"
            if not (field_out / "summary.json").exists():
                stage(
                    f"{model}_field",
                    HERE / "field_eval5.py",
                    "--checkpoint", run / "best.pt", "--output", field_out,
                    "--subset", "all", "--distance-axis", "source",
                    "--device", args.device, "--known-lab",
                )
            (run / "latest_field.json").write_text(
                json.dumps({"output": str(field_out)}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    summary = {"cpu": cpu, "data": str(args.data), "models": {}}
    for model in ("unet", "transformer"):
        run = args.output / model
        if not (run / "thresholds.json").is_file():
            continue
        entry = {"thresholds": None, "synthetic": {}, "field": None}
        threshold_path = run / "thresholds.json"
        if threshold_path.is_file():
            entry["thresholds"] = json.loads(threshold_path.read_text(encoding="utf-8"))["thresholds"]
        for view in ("mixed", "clean"):
            path = run / f"evaluation_test_{view}" / "evaluation.json"
            if path.is_file():
                report = json.loads(path.read_text(encoding="utf-8"))
                entry["synthetic"][view] = {
                    "checkpoint_epoch": report["checkpoint_epoch"],
                    "event": report["metrics"]["event_metrics_by_tolerance_m"]["5"],
                    "interval": report["metrics"]["interval_metrics"],
                    "terminal": report["metrics"]["terminal_metrics"],
                    "false_positive_summary": report["false_positive_summary"],
                }
        latest = run / "latest_field.json"
        if latest.is_file():
            field_out = Path(json.loads(latest.read_text(encoding="utf-8"))["output"])
            summary_path = field_out / "summary.json"
            if summary_path.is_file():
                field = json.loads(summary_path.read_text(encoding="utf-8"))
                entry["field"] = {
                    "known_lab": field.get("known_lab_positive_event_summary"),
                    "counts": field_counts(field_out / "predictions.jsonl"),
                    "output": str(field_out),
                }
        baseline = ITER4 / "output" / "adaptation4" / "baseline" / model / "evaluation_test" / "evaluation.json"
        previous = ITER4 / "output" / "adaptation4" / model / "evaluation_test_mixed" / "evaluation.json"
        for label, path in (("iter4_baseline_mixed", baseline), ("iter4_candidate_mixed", previous)):
            if path.is_file():
                report = json.loads(path.read_text(encoding="utf-8"))
                entry.setdefault("reference", {})[label] = {
                    "event": report["metrics"]["event_metrics_by_tolerance_m"]["5"],
                    "interval": report["metrics"]["interval_metrics"],
                    "terminal": report["metrics"]["terminal_metrics"],
                    "false_positive_summary": report["false_positive_summary"],
                }
        summary["models"][model] = entry
    (args.output / "iteration5_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Complete: {args.output / 'iteration5_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
