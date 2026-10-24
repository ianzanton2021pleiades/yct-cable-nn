"""Run Iter6 field adaptation, synthetic evaluation, and paired field review."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
ITER3 = HERE.parent / "FieldNN_Iter3_20261022"
ITER5 = HERE.parent / "FieldNN_Iter5_20261022"
if str(ITER3) not in sys.path:
    sys.path.insert(0, str(ITER3))

from run_iteration import reserve_one_core


DEFAULT_FIELD_DATA = Path(r"E:\CableNNDataSet\CodeX\FieldNN_Iter6_20261022")
DEFAULT_SYNTHETIC = Path(r"E:\CableNNDataSet\CodeX\FieldNN_Iter4_20261022")
FIXED_THRESHOLDS = {"event": 0.6, "interval": 0.99, "terminal": 0.4}
BASELINE_FIELD_OUTPUTS = {
    "unet": ITER5 / "output" / "iter5" / "unet" / "field_e06",
    "transformer": ITER5 / "output" / "iter5" / "transformer" / "field_e007",
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def run_step(name: str, command: list[str], output: Path, env: dict[str, str]) -> dict:
    log = output / "logs" / f"{name}.log"
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
            env=env,
        ) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)
    return {"stage": name, "seconds": time.perf_counter() - started, "log": str(log)}


def field_output_path(model_run: Path, kind: str, model_name: str, epoch: int) -> Path:
    base = model_run / f"field_{kind}_e{epoch:03d}"
    if base.exists() and any(base.iterdir()):
        suffix = time.strftime("%Y%m%d_%H%M%S")
        candidate = model_run / f"field_{kind}_e{epoch:03d}_{suffix}"
        serial = 2
        while candidate.exists() and any(candidate.iterdir()):
            candidate = model_run / f"field_{kind}_e{epoch:03d}_{suffix}_{serial}"
            serial += 1
        return candidate
    return base


def evaluation_summary(path: Path) -> dict | None:
    report = load_json(path)
    if report is None:
        return None
    return {
        "split": report.get("split"),
        "view": report.get("evaluation_view"),
        "checkpoint_epoch": report.get("checkpoint_epoch"),
        "thresholds": report.get("thresholds"),
        "metrics": report.get("metrics"),
        "false_positive_summary": report.get("false_positive_summary"),
    }


def field_summary(path: Path) -> dict | None:
    report = load_json(path / "summary.json")
    if report is None:
        return None
    return {
        "output": str(path),
        "model_name": report.get("model_name"),
        "checkpoint_epoch": report.get("checkpoint_epoch"),
        "thresholds": report.get("thresholds"),
        "processed": report.get("processed"),
        "not_processed": report.get("not_processed"),
        "known_lab_positive_event_summary": report.get("known_lab_positive_event_summary"),
    }


def training_summary(model_run: Path) -> dict:
    rows = []
    history = model_run / "history.csv"
    if history.is_file():
        with history.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    initial = load_json(model_run / "initial_validation.json")
    best_row = min(rows, key=lambda row: float(row["selection_score"])) if rows else None
    initial_score = float(initial["selection_score"]) if initial else None
    selected_epoch = -1
    if best_row is not None and initial_score is not None:
        if float(best_row["selection_score"]) < initial_score:
            selected_epoch = int(best_row["epoch"]) - 1
    return {
        "run_info": load_json(model_run / "run_info.json"),
        "initial_validation": initial,
        "history_rows": len(rows),
        "last_epoch": rows[-1] if rows else None,
        "lowest_history_row": best_row,
        "selected_checkpoint_epoch_from_training_rule": selected_epoch,
        "history_row_is_selected_checkpoint": selected_epoch >= 0,
    }


def baseline_synthetic_summary(model: str) -> dict:
    baseline_run = ITER5 / "output" / "iter5" / model
    return {
        view: evaluation_summary(
            baseline_run / f"evaluation_test_{view}" / "evaluation.json"
        )
        for view in ("mixed", "clean")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_FIELD_DATA, help="Prepared Iter6 field cache")
    parser.add_argument("--synthetic", type=Path, default=DEFAULT_SYNTHETIC)
    parser.add_argument("--output", type=Path, default=HERE / "output" / "iter6")
    parser.add_argument("--models", default="unet,transformer")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run one train and one validation batch per model")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--precision-target", type=float, default=0.90)
    parser.add_argument("--field-plots", type=int, default=8)
    args = parser.parse_args()

    models = [name.strip().lower() for name in args.models.split(",") if name.strip()]
    if not models or any(name not in {"unet", "transformer"} for name in models):
        parser.error("--models must be a comma-separated subset of unet,transformer")

    if args.smoke:
        args.data = HERE / "smoke_field_cache"
        args.output = HERE / "smoke_output"
    args.data = args.data.expanduser().resolve()
    args.synthetic = args.synthetic.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    mpl_dir = args.output / "matplotlib_cache"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONUTF8": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "MPLCONFIGDIR": str(mpl_dir),
    })
    cpu = reserve_one_core()
    prefix = [sys.executable, "-X", "utf8", "-B"]
    train_script = HERE / "train.py"
    prepare_script = HERE / "prepare_field.py"
    prepare_workers = 2 if args.smoke else 7
    stages: list[dict] = []
    results: dict[str, dict] = {}
    epochs = 1 if args.smoke else args.epochs
    max_train_batches = 1 if args.smoke else args.max_train_batches
    max_val_batches = 1 if args.smoke else args.max_val_batches

    def write_status(status: str) -> None:
        write_json(args.output / "pipeline_status.json", {
            "status": status,
            "cpu": cpu,
            "data": str(args.data),
            "synthetic": str(args.synthetic),
            "stages": stages,
        })

    def stage(name: str, command: list[str], view: str | None = None) -> None:
        step_env = env.copy()
        if view is not None:
            step_env["FIELDNN_EVAL_VIEW"] = view
        stages.append(run_step(name, command, args.output, step_env))
        write_status("running")

    print(json.dumps({
        "data": str(args.data), "synthetic": str(args.synthetic),
        "output": str(args.output), "models": models,
        "device": args.device, "smoke": args.smoke,
    }, ensure_ascii=False), flush=True)

    summary = {
        "schema": "fieldnn-iter6-run-summary-1",
        "status": "running",
        "mode": "smoke" if args.smoke else "full",
        "data": str(args.data),
        "synthetic": str(args.synthetic),
        "output": str(args.output),
        "device": args.device,
        "settings": {
            "epochs": epochs,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "max_train_batches": max_train_batches,
            "max_val_batches": max_val_batches,
            "precision_target": args.precision_target,
            "fixed_pair_thresholds": FIXED_THRESHOLDS,
        },
        "threshold_policy": "calibrate on synthetic validation only at the requested precision target",
        "field_metric_policy": "positive-only, case-grouped; unmatched candidates remain unverified",
        "models": results,
    }

    write_status("running")
    write_json(args.output / "iteration6_summary.json", summary)
    cache_ready = all(
        (args.data / name).is_file()
        for name in ("field_features.npy", "field_records.jsonl", "field_bags.json", "summary.json")
    )
    cache_summary = load_json(args.data / "summary.json") if cache_ready else None
    if args.resume and cache_ready and bool(cache_summary.get("smoke")) == bool(args.smoke):
        print(f"Reuse prepared field cache: {args.data}", flush=True)
        stages.append({"stage": "prepare_reused", "seconds": 0.0, "cache": str(args.data)})
        write_status("running")
    else:
        prepare_command = prefix + [
            str(prepare_script), "--output", str(args.data),
            "--workers", str(prepare_workers),
        ]
        if args.smoke:
            prepare_command.append("--smoke")
        stage("prepare", prepare_command)

    for model in models:
        run = args.output / model
        train_command = prefix + [
            str(train_script),
            "--data", str(args.data),
            "--synthetic", str(args.synthetic),
            "--model", model,
            "--output", str(args.output),
            "--epochs", str(epochs),
            "--batch-size", str(args.batch_size),
            "--workers", str(args.workers),
            "--device", args.device,
        ]
        if args.resume:
            if (run / "last.pt").is_file():
                train_command.append("--resume")
        if max_train_batches:
            train_command += ["--max-train-batches", str(max_train_batches)]
        if max_val_batches:
            train_command += ["--max-val-batches", str(max_val_batches)]
        stage(f"{model}_train", train_command)

        model_result: dict[str, object] = {"training": training_summary(run)}
        results[model] = model_result
        checkpoint = run / "best.pt"
        evaluate_script = HERE / "evaluate.py"
        stage(f"{model}_val_calibrate", prefix + [
            str(evaluate_script), "--data", str(args.synthetic),
            "--checkpoint", str(checkpoint), "--split", "val", "--calibrate",
            "--precision-target", str(args.precision_target),
            "--batch-size", str(args.batch_size), "--device", args.device,
            *( ["--limit", "2"] if args.smoke else [] ),
        ])
        for view in ("mixed", "clean"):
            stage(f"{model}_test_{view}", prefix + [
                str(evaluate_script), "--data", str(args.synthetic),
                "--checkpoint", str(checkpoint), "--split", "test",
                "--batch-size", str(args.batch_size), "--device", args.device,
                "--output", str(run / f"evaluation_test_{view}"),
                *( ["--limit", "2"] if args.smoke else [] ),
            ], view=view)

        thresholds_path = run / "thresholds.json"
        calibration = load_json(thresholds_path)
        if calibration is None:
            raise FileNotFoundError(f"Calibration file was not written: {thresholds_path}")
        epoch = int(calibration["checkpoint_epoch"])

        if args.smoke:
            smoke_field = field_output_path(run, "smoke", model, epoch)
            stage(f"{model}_field_calibrated", prefix + [
                str(HERE / "field_infer.py"), "--checkpoint", str(checkpoint),
                "--manifest", str(args.data / "field_records.jsonl"),
                "--output", str(smoke_field), "--subset", "all",
                "--distance-axis", "source", "--device", args.device,
                "--limit", "1", "--plots", "0",
            ])
            smoke_match = args.output / model / "smoke_field_match.json"
            stage(f"{model}_smoke_field_match", prefix + [
                str(HERE / "evaluate_field.py"), "--data", str(args.data),
                "--predictions", f"iter6_smoke={smoke_field / 'predictions.jsonl'}",
                "--output", str(smoke_match),
            ])
            model_result.update({
                "baseline_iter5_test": baseline_synthetic_summary(model),
                "calibrated_thresholds": calibration,
                "test_mixed": evaluation_summary(run / "evaluation_test_mixed" / "evaluation.json"),
                "test_clean": evaluation_summary(run / "evaluation_test_clean" / "evaluation.json"),
                "smoke_field_inference": field_summary(smoke_field),
                "smoke_field_match_only": load_json(smoke_match),
            })
            summary["models"] = results
            continue

        calibrated_field = field_output_path(run, "calibrated", model, epoch)
        stage(f"{model}_field_calibrated", prefix + [
            str(HERE / "field_infer.py"), "--checkpoint", str(checkpoint),
            "--output", str(calibrated_field), "--subset", "all",
            "--distance-axis", "source", "--device", args.device,
            "--known-lab", "--length-view", "--plots", str(args.field_plots),
        ])

        paired_model = args.output / "paired" / "candidate" / model
        paired_model.mkdir(parents=True, exist_ok=True)
        paired_checkpoint = paired_model / "best.pt"
        shutil.copy2(checkpoint, paired_checkpoint)
        write_json(paired_model / "thresholds.json", {
            "calibration_split": "val",
            "model_name": model,
            "checkpoint_epoch": epoch,
            "thresholds": FIXED_THRESHOLDS,
            "policy": "fixed Iter5 comparison point; event=0.6, interval=0.99, terminal=0.4",
        })
        fixed_field = field_output_path(paired_model, "fixed", model, epoch)
        stage(f"{model}_field_fixed_pair", prefix + [
            str(HERE / "field_infer.py"), "--checkpoint", str(paired_checkpoint),
            "--output", str(fixed_field), "--subset", "all",
            "--distance-axis", "source", "--device", args.device,
            "--known-lab", "--length-view", "--plots", str(args.field_plots),
        ])

        baseline_field = BASELINE_FIELD_OUTPUTS[model] / "predictions.jsonl"
        if not baseline_field.is_file():
            raise FileNotFoundError(f"Iter5 paired baseline predictions are missing: {baseline_field}")
        paired_report = args.output / "paired" / f"field_comparison_{model}.json"
        stage(f"{model}_field_paired_compare", prefix + [
            str(HERE / "evaluate_field.py"), "--data", str(args.data),
            "--predictions", f"baseline={baseline_field}",
            "--predictions", f"iter6_fixed={fixed_field / 'predictions.jsonl'}",
            "--output", str(paired_report),
        ])

        model_result.update({
            "baseline_iter5_test": baseline_synthetic_summary(model),
            "calibrated_thresholds": calibration,
            "test_mixed": evaluation_summary(run / "evaluation_test_mixed" / "evaluation.json"),
            "test_clean": evaluation_summary(run / "evaluation_test_clean" / "evaluation.json"),
            "field_calibrated": field_summary(calibrated_field),
            "field_fixed_threshold_pair": field_summary(fixed_field),
            "fixed_threshold_field_comparison": load_json(paired_report),
            "paired_baseline_predictions": str(baseline_field),
        })
        summary["models"] = results
        summary["stages"] = stages

    summary["models"] = results
    summary["stages"] = stages
    summary["status"] = "complete"
    write_json(args.output / "iteration6_summary.json", summary)
    write_status("complete")
    print(f"Complete: {args.output / 'iteration6_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
