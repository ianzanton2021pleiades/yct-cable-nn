"""Iter8: broader step inputs and confirmed short-reference supervision."""

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
from typing import Any


HERE = Path(__file__).resolve().parent
ITER3 = HERE.parent / "FieldNN_Iter3_20261022"
ITER5 = HERE.parent / "FieldNN_Iter5_20261022"
ITER6 = HERE.parent / "FieldNN_Iter6_20261022"
ITER6_OUTPUT = ITER6 / "output" / "iter6"
ITER7_OUTPUT = HERE.parent / 'FieldNN_Iter7_20261022' / 'output' / 'iter7'
DEFAULT_DATA = Path(r"E:\CableNNDataSet\CodeX\FieldNN_Iter8_20261022")
DEFAULT_SYNTHETIC = Path(r"E:\CableNNDataSet\CodeX\FieldNN_Iter4_20261022")
FIXED_THRESHOLDS = {"event": 0.6, "interval": 0.99, "terminal": 0.4}
BASELINE_FIELD_OUTPUTS = {
    "unet": ITER5 / "output" / "iter5" / "unet" / "field_e06",
    "transformer": ITER5 / "output" / "iter5" / "transformer" / "field_e007",
}

sys.path.insert(0, str(ITER6))
from run_iteration6 import reserve_one_core  # noqa: E402


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def run_step(name: str, command: list[str], output: Path, env: dict[str, str]) -> dict[str, Any]:
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


def field_output_path(model_run: Path, epoch: int) -> Path:
    base = model_run / f"field_fixed_e{epoch:03d}"
    if base.exists() and any(base.iterdir()):
        suffix = time.strftime("%Y%m%d_%H%M%S")
        candidate = model_run / f"field_fixed_e{epoch:03d}_{suffix}"
        serial = 2
        while candidate.exists() and any(candidate.iterdir()):
            candidate = model_run / f"field_fixed_e{epoch:03d}_{suffix}_{serial}"
            serial += 1
        return candidate
    return base


def compact_evaluation(path: Path) -> dict[str, Any] | None:
    report = load_json(path)
    if report is None:
        return None
    metrics = report["metrics"]
    event = metrics["event_metrics_by_tolerance_m"]["5"]
    return {
        "split": report.get("split"),
        "view": report.get("evaluation_view"),
        "checkpoint_epoch": report.get("checkpoint_epoch"),
        "thresholds": report.get("thresholds"),
        "event_at_5m": {
            "precision": event.get("precision"),
            "recall": event.get("recall"),
            "f1": event.get("f1"),
        },
        "interval_f1": metrics.get("interval_metrics", {}).get("f1"),
        "terminal_f1": metrics.get("terminal_metrics", {}).get("f1"),
    }


def training_summary(model_run: Path) -> dict[str, Any]:
    history_path = model_run / "history.csv"
    rows: list[dict[str, str]] = []
    if history_path.is_file():
        with history_path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    initial = load_json(model_run / "initial_validation.json")
    best_row = min(rows, key=lambda row: float(row["selection_score"])) if rows else None
    selected_epoch = -1
    if best_row is not None and initial is not None:
        if float(best_row["selection_score"]) < float(initial["selection_score"]):
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


def field_comparison_summary(path: Path) -> dict[str, Any] | None:
    report = load_json(path)
    if report is None:
        return None
    compact: dict[str, Any] = {"path": str(path), "models": {}}
    for label in ("baseline5", "iter7", "iter8_fixed"):
        source = report["models"].get(label)
        if source is None:
            continue
        compact["models"][label] = {}
        for split in ("train", "val", "test"):
            result = source["splits"][split]
            hit20 = result["positive_labels"]["independent_group_macro"]["20"]
            per_sweep20 = result["positive_labels"]["per_sweep_micro"]["20"]
            split_summary = {
                "coverage": result["coverage"],
                "group_macro_hit_rate_20m": hit20.get("macro_hit_rate"),
                "group_mean_candidates_per_best_sweep": hit20.get(
                    "mean_unverified_candidates_per_group_best_sweep"
                ),
                "per_sweep_hit_rate_20m": per_sweep20.get("hit_rate"),
                "per_sweep_mean_candidates": per_sweep20.get("mean_candidates_per_sweep"),
                "known_lab_1500m_test": result.get("known_lab_1500m_test"),
            }
            if split == "test":
                other_groups = [
                    group for group in result["positive_labels"]["groups"]
                    if group["group"] != "development_experiment:1500m_wet_joint"
                ]
                other20 = [group["by_tolerance"]["20"] for group in other_groups]
                group_candidate_means = [
                    metric["mean_unverified_candidates_per_best_sweep"]
                    for metric in other20
                    if metric["mean_unverified_candidates_per_best_sweep"] is not None
                ]
                position_count = sum(metric["unique_bag_position_claims"] for metric in other20)
                matched_count = sum(metric["matched_bag_position_claims"] for metric in other20)
                split_summary["other_test_groups_20m"] = {
                    "groups": len(other_groups),
                    "unique_bag_position_claims": position_count,
                    "matched_bag_position_claims": matched_count,
                    "micro_hit_rate": matched_count / position_count if position_count else None,
                    "macro_hit_rate": (
                        sum(metric["hit_rate"] for metric in other20) / len(other20)
                        if other20 else None
                    ),
                    "mean_candidates_per_group_best_sweep": (
                        sum(group_candidate_means) / len(group_candidate_means)
                        if group_candidate_means else None
                    ),
                }
            compact["models"][label][split] = split_summary
    return compact


def density_analysis_summary(path: Path) -> dict[str, Any] | None:
    report = load_json(path)
    if report is None:
        return None
    selected = report.get("validation_selection", {})
    return {
        "path": str(path),
        "selected_thresholds_from_validation": {
            model: value.get("selected_threshold") for model, value in selected.items()
        },
        "test_after_selection": report.get("test_after_selection"),
        "density_constraint": report.get("density_constraint"),
        "interpretation_limit": report.get("interpretation_limit"),
    }


def previous_fixed_prediction(model: str) -> Path:
    summary_path = ITER7_OUTPUT / "iteration7_summary.json"
    summary = load_json(summary_path)
    if summary is None:
        raise FileNotFoundError(f"Iter7 summary was not found: {summary_path}")
    return Path(summary['models'][model]['fixed_field_predictions'])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="unet,transformer")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    models = [name.strip().lower() for name in args.models.split(",") if name.strip()]
    if not models or any(name not in {"unet", "transformer"} for name in models):
        parser.error("--models must be a comma-separated subset of unet,transformer")

    smoke = bool(args.smoke)
    data = HERE / "smoke_data" if smoke else DEFAULT_DATA
    synthetic = DEFAULT_SYNTHETIC
    output = HERE / "smoke_output" if smoke else HERE / "output" / "iter8"
    epochs = 1 if smoke else args.epochs
    max_train_batches = 3 if smoke else 0
    max_val_batches = 1 if smoke else 0
    output.mkdir(parents=True, exist_ok=True)
    mpl_dir = output / "matplotlib_cache"
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
    results: dict[str, Any] = {}
    stages: list[dict[str, Any]] = []
    summary_path = output / "iteration8_summary.json"
    summary: dict[str, Any] = {
        "schema": "fieldnn-iter8-run-summary-1",
        "status": "running",
        "mode": "smoke" if smoke else "full",
        "data": str(data),
        "synthetic": str(synthetic),
        "output": str(output),
        "device": args.device,
        "cpu": cpu,
        "settings": {
            "epochs": epochs,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "max_train_batches": max_train_batches,
            "max_val_batches": max_val_batches,
            "resume": args.resume,
            "fixed_field_thresholds": FIXED_THRESHOLDS,
        },
        "iteration8_change": (
            "Add 90/300 m step features through an identity-initialized input stem. "
            "Replace frozen-teacher preservation with supervised short/no-cable references."
        ),
        "held_constant": (
            "U-Net/Transformer backbones, Iter5 checkpoint initialization, Iter6 field label/group split, "
            "Iter4 synthetic replay, positive MIL and training cadence."
        ),
        "field_metric_policy": "positive-only, group-held-out; unmatched candidates are unverified, not false positives",
        "models": results,
        "stages": stages,
    }

    def save_summary(status: str) -> None:
        summary["status"] = status
        summary["models"] = results
        summary["stages"] = stages
        write_json(summary_path, summary)
        write_json(output / "pipeline_status.json", {
            "status": status,
            "cpu": cpu,
            "data": str(data),
            "synthetic": str(synthetic),
            "stages": stages,
        })

    def stage(name: str, command: list[str], view: str | None = None) -> None:
        step_env = env.copy()
        if view is not None:
            step_env["FIELDNN_EVAL_VIEW"] = view
        stages.append(run_step(name, command, output, step_env))
        save_summary("running")

    print(json.dumps({
        "data": str(data), "synthetic": str(synthetic), "output": str(output),
        "models": models, "device": args.device, "smoke": smoke,
    }, ensure_ascii=False), flush=True)
    save_summary("running")

    train_script = HERE / "train.py"
    prepare_files = {
        'field': ('field_features.npy', 'field_bags.json', 'field_records.jsonl', 'summary.json'),
        'calibration': ('calibration_raw_imp.npy', 'calibration_raw_step.npy',
                        'calibration_records.jsonl', 'calibration_summary.json'),
    }
    for kind, files in prepare_files.items():
        if args.resume and all((data / name).is_file() for name in files):
            stages.append({'stage': f'prepare_{kind}_reused', 'seconds': 0., 'data': str(data)})
            print(f'Reuse completed {kind} preparation: {data}', flush=True)
            save_summary('running')
        else:
            stage(f'prepare_{kind}', prefix + [str(HERE / f'prepare_{kind}.py'),
                  '--output', str(data), '--workers', '2' if smoke else '7',
                  *(['--smoke'] if smoke else [])])
    for model in models:
        model_run = output / model
        train_command = prefix + [
            str(train_script),
            "--data", str(data),
            "--synthetic", str(synthetic),
            "--model", model,
            "--output", str(output),
            "--epochs", str(epochs),
            "--batch-size", str(args.batch_size),
            "--workers", str(args.workers),
            "--device", args.device,
        ]
        if args.resume and (model_run / "last.pt").is_file():
            train_command.append("--resume")
        if max_train_batches:
            train_command += ["--max-train-batches", str(max_train_batches)]
        if max_val_batches:
            train_command += ["--max-val-batches", str(max_val_batches)]
        stage(f"{model}_train", train_command)

        model_result: dict[str, Any] = {
            "training": training_summary(model_run),
            "source_checkpoint": str(ITER5 / "output" / "iter5" / model / "best.pt"),
        }
        results[model] = model_result
        checkpoint = model_run / "best.pt"

        evaluate_script = HERE / "evaluate.py"
        limit = ['--limit', '2'] if smoke else []
        stage(f"{model}_val_calibrate", prefix + [
            str(evaluate_script),
            "--data", str(synthetic),
            "--checkpoint", str(checkpoint),
            "--split", "val",
            "--calibrate",
            "--precision-target", "0.90",
            "--batch-size", str(args.batch_size),
            "--device", args.device,
            *limit,
        ])
        for view in ("mixed", "clean"):
            stage(f"{model}_test_{view}", prefix + [
                str(evaluate_script),
                "--data", str(synthetic),
                "--checkpoint", str(checkpoint),
                "--split", "test",
                "--batch-size", str(args.batch_size),
                "--device", args.device,
                "--output", str(model_run / f"evaluation_test_{view}"),
                *limit,
            ], view=view)

        calibration = load_json(model_run / "thresholds.json")
        if calibration is None:
            raise FileNotFoundError(f"Calibration file was not written: {model_run / 'thresholds.json'}")
        epoch = int(calibration["checkpoint_epoch"])
        for version, calibration_checkpoint in (
            ('baseline5', ITER5 / 'output' / 'iter5' / model / 'best.pt'),
            ('iter8', checkpoint),
        ):
            stage(f'{model}_short_{version}', prefix + [str(HERE / 'evaluate_calibration.py'),
                  '--data', str(data), '--checkpoint', str(calibration_checkpoint),
                  '--output', str(model_run / f'calibration_{version}.json'), '--device', args.device,
                  *(['--limit', '2'] if smoke else [])])
        if smoke:
            smoke_field = model_run / 'field_smoke'
            stage(f'{model}_field_smoke', prefix + [str(HERE / 'field_infer.py'),
                  '--checkpoint', str(checkpoint), '--manifest', str(data / 'field_records.jsonl'),
                  '--output', str(smoke_field), '--subset', 'all', '--limit', '1',
                  '--plots', '0', '--device', args.device])
            continue

        pair_model = output / "paired" / "candidate" / model
        pair_model.mkdir(parents=True, exist_ok=True)
        pair_checkpoint = pair_model / "best.pt"
        shutil.copy2(checkpoint, pair_checkpoint)
        write_json(pair_model / "thresholds.json", {
            "calibration_split": "val",
            "model_name": model,
            "checkpoint_epoch": epoch,
            "thresholds": FIXED_THRESHOLDS,
            "policy": "fixed Iter5/Iter7 comparison thresholds: event=0.6, interval=0.99, terminal=0.4",
        })
        fixed_field = field_output_path(pair_model, epoch)
        stage(f"{model}_field_fixed", prefix + [
            str(HERE / "field_infer.py"),
            "--checkpoint", str(pair_checkpoint),
            "--output", str(fixed_field),
            "--subset", "all",
            "--distance-axis", "source",
            "--device", args.device,
            "--known-lab",
            "--plots", "0",
        ])

        baseline_field = BASELINE_FIELD_OUTPUTS[model] / "predictions.jsonl"
        previous_field = previous_fixed_prediction(model)
        paired_report = output / "paired" / f"field_comparison_{model}.json"
        stage(f"{model}_field_paired_compare", prefix + [
            str(ITER6 / "evaluate_field.py"),
            "--data", str(data),
            "--predictions", f"baseline5={baseline_field}",
            "--predictions", f"iter7={previous_field}",
            "--predictions", f"iter8_fixed={fixed_field / 'predictions.jsonl'}",
            "--output", str(paired_report),
        ])

        model_result.update({
            "calibrated_thresholds": calibration,
            "synthetic_test_three_way": {
                "baseline5": {
                    view: compact_evaluation(
                        ITER5 / "output" / "iter5" / model / f"evaluation_test_{view}" / "evaluation.json"
                    ) for view in ("mixed", "clean")
                },
                "iter7": {
                    view: compact_evaluation(
                        ITER7_OUTPUT / model / f"evaluation_test_{view}" / "evaluation.json"
                    ) for view in ("mixed", "clean")
                },
                "iter8": {
                    view: compact_evaluation(
                        model_run / f"evaluation_test_{view}" / "evaluation.json"
                    ) for view in ("mixed", "clean")
                },
            },
            "fixed_field_output": str(fixed_field),
            "fixed_field_predictions": str(fixed_field / "predictions.jsonl"),
            "paired_comparison_path": str(paired_report),
            "paired_comparison": field_comparison_summary(paired_report),
            "iter7_fixed_predictions": str(previous_field),
            "calibration_reference_tests": {version: load_json(model_run / f'calibration_{version}.json')
                                            for version in ('baseline5', 'iter8')},
            "baseline5_predictions": str(baseline_field),
        })
        save_summary("running")

    if not smoke and set(models) == {"unet", "transformer"}:
        density_path = output / "density_analysis.json"
        density_script = ITER6 / "calibrate_field_density.py"
        stage("density_matched", prefix + [
            str(density_script),
            "--data", str(data),
            "--output", str(density_path),
            "--candidate-unet", results["unet"]["fixed_field_predictions"],
            "--candidate-transformer", results["transformer"]["fixed_field_predictions"],
        ])
        summary["density_matched_analysis"] = density_analysis_summary(density_path)
    elif smoke:
        summary["density_matched_analysis"] = None
        summary["smoke_scope"] = (
            "Three training batches, limited synthetic calibration/test, short-reference evaluation, "
            "one FDR inference per model; no full field or density-matched performance evaluation."
        )
    else:
        summary["density_matched_analysis"] = None
        summary["density_matched_analysis_note"] = (
            "Run the paired validation-density analysis when both models are included."
        )

    save_summary("complete")
    print(f"Complete: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
