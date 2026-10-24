"""One-command runner: prepare -> train -> calibrate -> evaluate -> ensemble."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_PYTHON = Path(r"E:\Anaconda\envs\gpushare_cu124\python.exe")
DEFAULT_DATA = Path(r"E:\CableNNDataSet\DeepSeek_CMDC\FieldNN_Hybrid_20261022")
ITER5 = HERE.parent / "FieldNN_Iter5_20261022"


def run(python: Path, args: list[str], log_path: Path | None = None) -> None:
    command = [str(python), "-X", "utf8", "-B", *args]
    print("+", " ".join(command), flush=True)
    if log_path is None:
        subprocess.run(command, cwd=HERE, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as stream:
        process = subprocess.Popen(
            command, cwd=HERE, stdout=stream, stderr=subprocess.STDOUT, text=True
        )
        code = process.wait()
    if code != 0:
        raise SystemExit(f"command failed ({code}); see {log_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--init-root", type=Path)
    parser.add_argument("--output", type=Path, default=HERE / "output" / "hybrid")
    parser.add_argument("--variants-train", type=int)
    parser.add_argument("--variants-val", type=int)
    parser.add_argument("--variants-test", type=int)
    parser.add_argument("--models", nargs="+", default=["unet", "transformer"], choices=["unet", "transformer"])
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    data = args.data if not args.smoke else args.data.with_name(args.data.name + "_smoke")
    output = args.output if not args.smoke else HERE / "output" / "smoke_hybrid"
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    if not args.skip_prepare:
        prepare_args = [
            str(HERE / "prepare_hybrid.py"),
            "--output", str(data),
        ]
        if args.smoke:
            prepare_args += ["--smoke", "--smoke-cases", "2", "--smoke-variants", "2"]
        else:
            prepare_args += ["--workers", "7"]
            for flag, value in (
                ("--variants-train", args.variants_train),
                ("--variants-val", args.variants_val),
                ("--variants-test", args.variants_test),
            ):
                if value is not None:
                    prepare_args += [flag, str(value)]
        run(args.python, prepare_args, logs / "prepare.log")

    if not args.skip_train:
        init_root = args.init_root or (ITER5 / "output" / "iter5")
        for model in args.models:
            init = init_root / model / "best.pt"
            command = [
                str(HERE / "train.py"),
                "--data", str(data),
                "--model", model,
                "--output", str(output),
                "--epochs", str(1 if args.smoke else args.epochs),
                "--batch-size", str(args.batch_size),
                "--workers", str(args.workers),
            ]
            if args.config:
                command += ["--config", str(args.config)]
            if args.freeze_encoder:
                command += ["--freeze-encoder"]
            if init.is_file():
                command += ["--init-checkpoint", str(init)]
            if args.smoke:
                command += ["--max-train-batches", "3", "--max-val-batches", "1"]
            run(args.python, command, logs / f"{model}_train.log")

    for model in args.models:
        checkpoint = output / model / "best.pt"
        if not checkpoint.is_file():
            continue
        run(
            args.python,
            [
                str(HERE / "calibrate.py"),
                "--data", str(data),
                "--checkpoint", str(checkpoint),
                "--output", str(checkpoint.parent),
            ],
            logs / f"{model}_calibrate.log",
        )
        run(
            args.python,
            [
                str(HERE / "evaluate.py"),
                "--data", str(data),
                "--checkpoint", str(checkpoint),
                "--thresholds", str(checkpoint.parent / "thresholds.json"),
                "--output", str(checkpoint.parent / "evaluation"),
            ],
            logs / f"{model}_evaluate.log",
        )

    if len(args.models) >= 2:
        checkpoints = [str(output / model / "best.pt") for model in args.models if (output / model / "best.pt").is_file()]
        if len(checkpoints) >= 2:
            run(
                args.python,
                [
                    str(HERE / "calibrate.py"),
                    "--data", str(data),
                    "--checkpoint", *checkpoints,
                    "--output", str(output / "ensemble"),
                ],
                logs / "ensemble_calibrate.log",
            )
            run(
                args.python,
                [
                    str(HERE / "evaluate.py"),
                    "--data", str(data),
                    "--checkpoint", *checkpoints,
                    "--thresholds", str(output / "ensemble" / "thresholds.json"),
                    "--output", str(output / "ensemble" / "evaluation"),
                ],
                logs / "ensemble_evaluate.log",
            )

    summary = {"data": str(data), "output": str(output), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    for name in [*args.models, "ensemble"]:
        path = output / name / "evaluation" / "evaluation.json"
        if path.is_file():
            summary[name] = json.loads(path.read_text(encoding="utf-8"))
    (output / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
