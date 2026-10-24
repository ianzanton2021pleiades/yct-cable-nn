"""Export the locked 0.60 I11 field workpoint and fixed-threshold curves."""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
I10 = HERE.parent / "FieldNN_Iter10_20261023"
I8 = HERE.parent / "FieldNN_Iter8_20261022"
RUN = HERE / "output" / "iter11_50ep" / "unet"
DATA = Path(r"E:\CableNNDataSet\CodeX\FieldNN_Iter8_20261022")
PREDICTIONS = HERE / "output" / "coverage_workpoint" / "CPU_cache_predictions.jsonl"
HISTORY = RUN / "history_steps.csv"
CHECKPOINT = RUN / "best.pt"
BENCHMARK = RUN / "nn_benchmark.json"
THRESHOLD = 0.60
OUT = HERE / "output" / "coverage_workpoint"
EXPORT = OUT / "cpu_cache_export_060"


def load_module(path: Path, name: str):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def generate_cache_predictions() -> tuple[list[dict], dict[str, int]]:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    i8_models = load_module(I8 / "models.py", "iter11_coverage_i8_models")
    training = load_module(I10 / "train_recorded.py", "iter11_coverage_training_helpers")
    field = training.FieldCache.from_root(DATA)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    model = i8_models.build_model(checkpoint["model_name"], **cfg["model"]).cpu().eval()
    model.load_state_dict(checkpoint["model"])

    PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    split_counts = {"train": 0, "val": 0, "test": 0}
    with PREDICTIONS.open("w", encoding="utf-8", newline="\n") as stream:
        with torch.inference_mode():
            for record in tqdm(sorted(field.records, key=lambda row: int(row["index"])),
                               desc="CPU batch-1 annotated field cache", unit="curve"):
                index = int(record["index"])
                features = np.array(field.features[index:index + 1], dtype=np.float32, copy=True)
                logits = model(torch.from_numpy(features)).float()
                probabilities = torch.sigmoid(logits[0]).numpy()
                prediction = training._source_axis_prediction(
                    probabilities, record, cfg, THRESHOLD
                )
                row = {
                    "record": record,
                    "status": "predicted",
                    "prediction": prediction,
                    "distance_axis": "source_report",
                }
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                rows.append(row)
                split_counts[str(record["split"])] += 1
    return rows, split_counts


def to_int(value: str | None) -> int | None:
    return None if value in (None, "") else int(float(value))


def to_float(value: str | None) -> float | None:
    return None if value in (None, "") else float(value)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cpu_predictions, split_counts = generate_cache_predictions()
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(OUT / "matplotlib_cache")
    env["MPLBACKEND"] = "Agg"
    exporter = I10 / "export_existing.py"
    subprocess.run(
        [
            sys.executable, "-X", "utf8", str(exporter),
            "--data", str(DATA),
            "--predictions", str(PREDICTIONS),
            "--checkpoint", str(CHECKPOINT),
            "--benchmark", str(BENCHMARK),
            "--threshold", str(THRESHOLD),
            "--label", "Iter11-U-Net-fixed0.60",
            "--output", str(EXPORT),
        ],
        check=True,
        env=env,
    )

    performance_path = EXPORT / "performance.md"
    performance_text = performance_path.read_text(encoding="utf-8")
    performance_text = performance_text.replace(
        "阈值由现场验证组确定，test 只在冻结阈值后统计。U-Net .94 使用对应 Iter5 U-Net .6 验证候选预算；Transformer .98 使用对应 Iter5 Transformer .6 预算；跨架构候选预算不代表严格相同。密度约束不等于 precision。",
        "本补充工作点使用 I11 checkpoint 对 138 条已标注缓存曲线做的 CPU batch-1 前向，event threshold 固定沿用既有 Iter5 U-Net 0.60 预设；未按 test 调阈值。该缓存不是完整 213 条现场曲线。I11 主工作点仍为 selection.json 中的 0.94。密度约束不等于 precision。",
    )
    performance_text = re.sub(
        r"(预测文件覆盖 \d+/\d+ 行；.*?不能解释为误报数。)",
        r"\1 本文件仅覆盖 138 条已标注缓存曲线，不代表完整 213 条现场曲线。",
        performance_text,
        count=1,
    )
    performance_path.write_text(performance_text, encoding="utf-8")
    model_results_path = EXPORT / "MODEL_RESULTS.md"
    model_results_text = model_results_path.read_text(encoding="utf-8")
    model_results_text = re.sub(
        r"(该预测文件共有 \d+ 行、\d+ 条 status=predicted；.*?候选。)",
        r"\1 这些是已标注缓存曲线，不代表完整 213 条现场曲线。",
        model_results_text,
        count=1,
    )
    model_results_lines = model_results_text.splitlines()
    for line_index, line in enumerate(model_results_lines):
        if line.startswith("- Model label:"):
            model_results_lines.insert(
                line_index + 1,
                "- Threshold origin: supplementary 0.60 is the pre-existing Iter5 U-Net preset; its predictions come from a CPU batch-1 forward over the 138 annotated cache curves. No test-based tuning; the main selected workpoint remains 0.94.",
            )
            model_results_lines.insert(
                line_index + 2,
                "- Prediction coverage: 138 annotated cache curves only; this is not a full 213-sweep field evaluation.",
            )
            break
    model_results_path.write_text("\n".join(model_results_lines) + "\n", encoding="utf-8")

    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": ["Times New Roman", "SimHei"],
        "font.serif": ["Times New Roman"],
        "font.sans-serif": ["SimHei"],
        "axes.unicode_minus": False,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.dpi": 200,
        "savefig.dpi": 200,
    })

    with HISTORY.open("r", encoding="utf-8-sig", newline="") as stream:
        history = list(csv.DictReader(stream))
    steps = [int(row["optimizer_step"]) for row in history]
    train_hit = [float(row["train_hit20_fixed_060"]) for row in history]
    val_hit = [float(row["val_hit20_fixed_060"]) for row in history]
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(steps, train_hit, color="#2878b5", lw=1.8, label="Train group macro")
    ax.plot(steps, val_hit, color="#d9534f", lw=1.8, label="Validation group macro")
    ax.set_xlabel("Optimizer updates")
    ax.set_ylabel("Known-position group macro hit@20 m")
    ax.set_title("I11 U-Net field localization at fixed event threshold 0.60")
    ax.set_xlim(min(steps), max(steps))
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(direction="in", top=True, right=True)
    ax.grid(axis="y", color="#b8bec5", lw=0.6, alpha=0.55)
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.legend(frameon=True)
    fig.tight_layout()
    curve_path = OUT / "fixed_060_hit_curve.png"
    fig.savefig(curve_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    with (EXPORT / "performance.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        performance_rows = list(csv.DictReader(stream))
    with (EXPORT / "selected_cases.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        case_rows = list(csv.DictReader(stream))

    selected_populations = {}
    for population in ("nonwet_test", "wet_1500m_test", "all_prediction_rows"):
        row = next(
            item for item in performance_rows
            if item.get("model") == "Iter11-U-Net-fixed0.60"
            and item.get("population") == population
        )
        selected_populations[population] = {
            "split": row.get("split"),
            "groups": to_int(row.get("groups")),
            "bags": to_int(row.get("bags")),
            "curves": to_int(row.get("curves")),
            "known_positions": to_int(row.get("known_positions")),
            "hits_at_20m": to_int(row.get("hits_at_20m")),
            "group_macro_hit_at_20m": to_float(row.get("group_macro_hit_at_20m")),
            "candidate_events_total": to_int(row.get("candidate_events_total")),
            "candidate_events_per_curve": to_float(row.get("candidate_events_per_curve")),
        }

    result = {
        "schema": "fieldnn-iter11-fixed060-coverage-workpoint-1",
        "workpoint": {
            "model": "I11 U-Net",
            "event_threshold": THRESHOLD,
            "threshold_source": "pre-existing Iter5 U-Net locked preset; not selected or adjusted from test results",
            "status": "supplementary coverage-priority report; main 0.94 workpoint is unchanged",
            "candidate_semantics": "candidate events include matched events; unmatched and unlabeled candidates are unverified, not false positives",
        },
        "inputs": {
            "predictions": str(PREDICTIONS),
            "history_steps": str(HISTORY),
            "data_cache": str(DATA),
            "checkpoint": str(CHECKPOINT),
            "cpu_prediction_scope": "138 annotated cache curves only; not the full 213-sweep corpus",
        },
        "cpu_inference": {
            "curve_count": len(cpu_predictions),
            "split_counts": split_counts,
            "device": "cpu",
            "threads": 1,
            "batch_size": 1,
            "event_decode_threshold": THRESHOLD,
            "prediction_file": str(PREDICTIONS),
        },
        "fixed_threshold_populations": selected_populations,
        "selected_cases": [
            {
                "group": row.get("group"),
                "case_group": row.get("case_group"),
                "source_relative_path": row.get("source_relative_path"),
                "sweep_rowid": row.get("sweep_rowid"),
                "phase": row.get("phase"),
                "known_positions": to_int(row.get("position_count")),
                "hits_at_20m": to_int(row.get("matched_count")),
                "candidate_events": to_int(row.get("candidate_count")),
                "image": row.get("image_path"),
            }
            for row in case_rows
        ],
        "fixed_060_curve": {
            "metric": "train_hit20_fixed_060 and val_hit20_fixed_060 from history_steps.csv",
            "threshold": THRESHOLD,
            "record_points": len(history),
            "first_optimizer_update": steps[0],
            "last_optimizer_update": steps[-1],
            "train_final_group_macro_hit20": train_hit[-1],
            "val_final_group_macro_hit20": val_hit[-1],
            "image": str(curve_path),
        },
        "tables_and_case_report": {
            "performance_csv": str(EXPORT / "performance.csv"),
            "performance_markdown": str(EXPORT / "performance.md"),
            "model_results_markdown": str(EXPORT / "MODEL_RESULTS.md"),
            "selected_cases_csv": str(EXPORT / "selected_cases.csv"),
        },
    }
    summary_path = OUT / "coverage_workpoint_summary.json"
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
