"""Iter4: reuse S11, augment waveforms and calibration nuisances, then evaluate."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
HERE = Path(__file__).resolve().parent
ITER3 = HERE.parent / "FieldNN_Iter3_20261022"
sys.path.append(str(ITER3))
from run_iteration import reserve_one_core, run_step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path(r"E:\CableNNDataSet\CodeX\FieldNN_Iter4_20261022"))
    ap.add_argument("--source", type=Path, default=Path(r"E:\CableNNDataSet\CodeX\FieldNN_Iter3_20261022"))
    ap.add_argument("--output", type=Path, default=HERE / "output" / "adaptation4")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip-prepare", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.environ.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                       "NUMEXPR_NUM_THREADS": "1", "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"})
    cpu = reserve_one_core()
    if args.smoke:
        args.data, args.output = HERE / "smoke_cache", HERE / "smoke_output"
        args.epochs, args.workers = 1, 2
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(args.output / "matplotlib_cache")
    prefix = [sys.executable, "-X", "utf8", "-B"]
    completed = []
    def stage(name, script, *params, view="mixed"):
        os.environ["FIELDNN_EVAL_VIEW"] = view
        result = run_step(name, prefix + [str(script)] + list(map(str, params)), args.output)
        completed.append(result)
        (args.output / "pipeline_status.json").write_text(json.dumps({"cpu": cpu, "smoke_only": args.smoke,
            "completed_stages": completed}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"cpu": cpu, "data": str(args.data), "output": str(args.output), "smoke_only": args.smoke}, ensure_ascii=False), flush=True)
    if not args.skip_prepare:
        stage("01_prepare", HERE / "prepare.py", "--source", args.source, "--output", args.data,
              "--workers", 2 if args.smoke else 7, *(["--samples", "20"] if args.smoke else []),
              *(["--resume"] if args.resume else []))
    for model in ("unet", "transformer"):
        source_checkpoint = ITER3 / "output" / "iteration3" / model / "best.pt"
        baseline = args.output / "baseline" / model
        baseline.mkdir(parents=True, exist_ok=True)
        if not (baseline / "best.pt").exists():
            shutil.copyfile(source_checkpoint, baseline / "best.pt")
        stage(model + "_baseline_val", HERE / "evaluate.py", "--data", args.data, "--checkpoint", baseline / "best.pt",
              "--split", "val", "--calibrate", "--batch-size", args.batch_size, "--device", args.device)
        stage(model + "_baseline_mixed", HERE / "evaluate.py", "--data", args.data, "--checkpoint", baseline / "best.pt",
              "--split", "test", "--batch-size", args.batch_size, "--device", args.device)
        run = args.output / model
        restoration = ["--resume"] if args.resume and (run / "last.pt").exists() else ["--init-checkpoint", source_checkpoint]
        stage(model + "_train", HERE / "train.py", "--data", args.data, "--output", args.output,
              "--model", model, "--epochs", args.epochs, "--batch-size", args.batch_size,
              "--workers", args.workers, "--device", args.device, *restoration,
              *(["--max-train-batches", "1", "--max-val-batches", "1"] if args.smoke else []))
        stage(model + "_val", HERE / "evaluate.py", "--data", args.data, "--checkpoint", run / "best.pt",
              "--split", "val", "--calibrate", "--batch-size", args.batch_size, "--device", args.device)
        stage(model + "_test_mixed", HERE / "evaluate.py", "--data", args.data, "--checkpoint", run / "best.pt",
              "--split", "test", "--batch-size", args.batch_size, "--device", args.device,
              "--output", run / "evaluation_test_mixed")
        stage(model + "_test_clean", HERE / "evaluate.py", "--data", args.data, "--checkpoint", run / "best.pt",
              "--split", "test", "--batch-size", args.batch_size, "--device", args.device,
              "--output", run / "evaluation_test_clean", view="clean")
        calibration = json.loads((run / "thresholds.json").read_text(encoding="utf-8"))
        field_out = run / f"field_e{int(calibration['checkpoint_epoch']) + 1:03d}"
        if not (field_out / "summary.json").exists():
            stage(model + "_field", ITER3 / "field_eval.py", "--checkpoint", run / "best.pt",
                  "--output", field_out, "--subset", "development" if args.smoke else "all",
                  "--distance-axis", "source", "--device", args.device,
                  *(["--limit", "1"] if args.smoke else ["--known-lab"]))
        (run / "latest_field.json").write_text(json.dumps({"output": str(field_out)}, ensure_ascii=False, indent=2), encoding="utf-8")
    result = {"smoke_only": args.smoke, "cpu": cpu, "data": str(args.data), "models": {}}
    for model in ("unet", "transformer"):
        run = args.output / model
        field_out = Path(json.loads((run / "latest_field.json").read_text(encoding="utf-8"))["output"])
        field = json.loads((field_out / "summary.json").read_text(encoding="utf-8"))
        result["models"][model] = {
            "baseline_mixed": json.loads((args.output / "baseline" / model / "evaluation_test" / "evaluation.json").read_text(encoding="utf-8")),
            "candidate_mixed": json.loads((run / "evaluation_test_mixed" / "evaluation.json").read_text(encoding="utf-8")),
            "candidate_clean": json.loads((run / "evaluation_test_clean" / "evaluation.json").read_text(encoding="utf-8")),
            "known_lab": field.get("known_lab_positive_event_summary"), "field_output": str(field_out),
            "field_processed": field["processed"], "field_not_processed": field["not_processed"],
        }
    (args.output / "adaptation_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Complete: {args.output / 'adaptation_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
