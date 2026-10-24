"""One command: generate -> train two models -> calibrate -> test -> field inference.

Long stages are intentionally started by the user. --smoke runs only 20 synthetic
samples and one training batch per model; its scores are not performance evidence.
"""
import argparse
import ctypes
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def reserve_one_core():
    if os.name != "nt":
        return {"cpu_budget": "six workers + one main thread"}
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    size = ctypes.c_ulong(0)
    kernel.GetLogicalProcessorInformationEx(0, None, ctypes.byref(size))
    buffer = ctypes.create_string_buffer(size.value)
    if not kernel.GetLogicalProcessorInformationEx(0, buffer, ctypes.byref(size)):
        raise ctypes.WinError(ctypes.get_last_error())
    masks, offset = [], 0
    while offset < size.value:
        record_size = int.from_bytes(buffer.raw[offset + 4:offset + 8], "little")
        group_count = int.from_bytes(buffer.raw[offset + 30:offset + 32], "little")
        if group_count != 1:
            raise RuntimeError("This launcher expects a single Windows processor group")
        masks.append(int.from_bytes(buffer.raw[offset + 32:offset + 40], "little"))
        offset += record_size
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    allowed = 0
    for mask in masks[:-1]:
        allowed |= mask
    if not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), allowed):
        raise ctypes.WinError(ctypes.get_last_error())
    return {"physical_cores_used": len(masks) - 1, "physical_cores_reserved": 1,
            "affinity_mask": hex(allowed)}


def run_step(name, command, output):
    log = output / "logs" / (name + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print(f"\n=== {name} ===\n{subprocess.list2cmdline(command)}", flush=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write("\n" + datetime.datetime.now().isoformat() + "\n" + subprocess.list2cmdline(command) + "\n")
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace", bufsize=1, cwd=HERE) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)
    return {"stage": name, "seconds": time.perf_counter() - started, "log": str(log)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path(r"E:\CableNNDataSet\CodeX\FieldNN_Iter3_20261022"))
    ap.add_argument("--output", type=Path, default=HERE / "output" / "iteration3")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--field-subset", choices=["development", "all"], default="all")
    args = ap.parse_args()
    os.environ.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                       "NUMEXPR_NUM_THREADS": "1", "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"})
    cpu = reserve_one_core()
    if args.smoke:
        args.data = HERE / "smoke_data_v2"
        args.output = HERE / "smoke_output"
        args.epochs, args.workers = 1, 2
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(args.output / "matplotlib_cache")
    print(json.dumps({"cpu": cpu, "data": str(args.data), "output": str(args.output),
                      "smoke_only": args.smoke}, ensure_ascii=False), flush=True)
    prefix = [sys.executable, "-X", "utf8", "-B"]
    stages = []
    def stage(name, file, *params):
        stages.append(run_step(name, prefix + [str(HERE / file)] + list(map(str, params)), args.output))
        (args.output / "pipeline_status.json").write_text(json.dumps({"smoke_only": args.smoke,
            "cpu": cpu, "completed_stages": stages}, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.skip_generate:
        extra = (["--samples", "20", "--workers", "2"] if args.smoke else ["--workers", "7"])
        stage("01_generate", "generate.py", "--output", args.data, *extra,
              *(["--resume"] if args.resume else []))
    for number, model in enumerate(["unet", "transformer"], 2):
        run = args.output / model
        stage(f"{number}_{model}_train", "train.py", "--data", args.data, "--output", args.output,
              "--model", model, "--epochs", args.epochs, "--batch-size", args.batch_size,
              "--workers", args.workers, "--device", args.device,
              *(["--max-train-batches", "1", "--max-val-batches", "1"] if args.smoke else []),
              *(["--resume"] if args.resume and (run / "last.pt").exists() else []))
        stage(f"{number}_{model}_val", "evaluate.py", "--data", args.data, "--checkpoint", run / "best.pt",
              "--split", "val", "--calibrate", "--batch-size", args.batch_size, "--device", args.device)
        stage(f"{number}_{model}_test", "evaluate.py", "--data", args.data, "--checkpoint", run / "best.pt",
              "--split", "test", "--batch-size", args.batch_size, "--device", args.device)
        stage(f"{number}_{model}_field", "field_eval.py", "--checkpoint", run / "best.pt",
              "--output", run / "field", "--subset", "development" if args.smoke else args.field_subset,
              "--device", args.device, *(["--limit", "1"] if args.smoke else ["--known-lab"]))
    summary = {"smoke_only": args.smoke, "data": str(args.data), "cpu": cpu, "models": {}}
    for model in ("unet", "transformer"):
        run = args.output / model
        summary["models"][model] = {"test": json.loads((run / "evaluation_test" / "evaluation.json").read_text(encoding="utf-8")),
                                     "run": str(run), "field_output": str(run / "field")}
    (args.output / "iteration_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nComplete. Results: {args.output / 'iteration_summary.json'}", flush=True)


if __name__ == "__main__":
    main()




