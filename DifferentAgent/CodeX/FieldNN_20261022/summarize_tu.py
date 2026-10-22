"""Summarize Transformer/U-Net pilot artifacts and plot their learning curves."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_RUN = Path("output") / "tu_baseline_20261022"
MODELS = ("transformer", "unet")
TOLERANCES_M = (2, 5, 10)


def _read_json(path: Path, pending: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        pending.append(str(path))
        return None
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _read_history(path: Path, pending: list[str]) -> list[dict[str, float]] | None:
    if not path.is_file():
        pending.append(str(path))
        return None
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for raw in csv.DictReader(stream):
            row = {}
            for key in ("epoch", "train_loss", "val_loss", "seconds", "peak_gpu_gib"):
                value = raw.get(key)
                try:
                    parsed = float(value) if value not in (None, "") else None
                except ValueError:
                    parsed = None
                if parsed is not None and math.isfinite(parsed):
                    row[key] = parsed
            if row:
                rows.append(row)
    if not rows:
        pending.append(f"{path} (no complete epoch rows)")
    return rows


def _event_metrics(report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {}
    result = report.get("metrics", {}).get("event_metrics_by_tolerance_m", {})
    selected = {}
    for tolerance in TOLERANCES_M:
        record = result.get(str(tolerance), result.get(f"{tolerance:.1f}", {}))
        selected[str(tolerance)] = {
            "f1": record.get("f1"),
            "mean_matched_mae_m": record.get("mean_matched_mae_m"),
            "matched_event_count": record.get("matched_event_count"),
            "truth_event_count": record.get("truth_event_count"),
            "prediction_event_count": record.get("prediction_event_count"),
        }
    return selected


def _count_distribution(values: list[int]) -> dict[str, Any] | None:
    if not values:
        return None
    return {
        "n": len(values),
        "median": float(median(values)),
        "min": min(values),
        "max": max(values),
    }


def _read_field_predictions(path: Path, pending: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        pending.append(str(path))
        return None

    status_counts: Counter[str] = Counter()
    consistency_counts: Counter[str] = Counter()
    event_counts: list[int] = []
    interval_counts: list[int] = []
    terminal_candidates = 0
    missing_consistency = 0
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            status = str(row.get("status", "missing_status"))
            status_counts[status] += 1
            if status != "predicted":
                continue

            prediction = row.get("prediction") or {}
            events = prediction.get("events")
            intervals = prediction.get("intervals")
            if isinstance(events, list):
                event_counts.append(len(events))
            if isinstance(intervals, list):
                interval_counts.append(len(intervals))
            if prediction.get("terminal") is not None:
                terminal_candidates += 1

            consistency = prediction.get("terminal_consistency")
            consistency_status = consistency.get("status") if isinstance(consistency, dict) else None
            if consistency_status is None:
                missing_consistency += 1
            else:
                consistency_counts[str(consistency_status)] += 1

    return {
        "predictions_jsonl_record_count": int(sum(status_counts.values())),
        "prediction_status_counts": dict(sorted(status_counts.items())),
        "predicted_record_count": int(status_counts.get("predicted", 0)),
        "not_processed_record_count": int(status_counts.get("not_processed", 0)),
        "event_count_per_predicted_curve": _count_distribution(event_counts),
        "interval_count_per_predicted_curve": _count_distribution(interval_counts),
        "terminal_candidate_count": terminal_candidates,
        "terminal_consistency_status_counts": {
            key: int(consistency_counts.get(key, 0))
            for key in ("consistent_candidate", "uncertain_candidate", "no_candidate")
        },
        "other_terminal_consistency_status_counts": {
            key: int(value)
            for key, value in sorted(consistency_counts.items())
            if key not in {"consistent_candidate", "uncertain_candidate", "no_candidate"}
        },
        "terminal_consistency_missing_count": missing_consistency,
        "interpretation": (
            "Prediction coverage and heuristic terminal consistency only; "
            "not field accuracy or ground-truth validation."
        ),
    }


def _evaluation_summary(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if report is None:
        return None
    metrics = report.get("metrics", {})
    interval = metrics.get("interval_metrics", {})
    terminal = metrics.get("terminal_metrics", {})
    counts = report.get("counts", {})
    timing = report.get("timing", {})
    return {
        "split": report.get("split"),
        "checkpoint_epoch": report.get("checkpoint_epoch"),
        "evaluation_scope": counts.get("evaluation_scope"),
        "evaluated_sample_count": counts.get("evaluated_sample_count"),
        "split_sample_count": counts.get("split_sample_count"),
        "event_metrics_by_tolerance_m": _event_metrics(report),
        "interval_metrics": {
            "iou_threshold": interval.get("iou_threshold"),
            "grid_overall_iou": interval.get("grid_overall_iou"),
            "mean_matched_iou": interval.get("mean_matched_iou"),
            "f1": interval.get("f1"),
            "matched_interval_count": interval.get("matched_interval_count"),
            "truth_interval_count": interval.get("truth_interval_count"),
        },
        "terminal_metrics": {
            "tolerance_m": terminal.get("tolerance_m"),
            "terminal_hit_count": terminal.get("terminal_hit_count"),
            "truth_terminal_curve_count": terminal.get("truth_terminal_curve_count"),
            "terminal_hit_rate": terminal.get("terminal_hit_rate"),
            "mean_terminal_error_m": terminal.get("mean_terminal_error_m"),
            "mean_matched_terminal_error_m": terminal.get("mean_matched_terminal_error_m"),
        },
        "inference_seconds": timing.get("inference_seconds"),
        "total_seconds": timing.get("total_seconds_before_report_write"),
    }


def _model_summary(run: Path, name: str) -> dict[str, Any]:
    model_dir = run / name
    pending: list[str] = []
    history = _read_history(model_dir / "history.csv", pending)
    run_info = _read_json(model_dir / "run_info.json", pending)
    val_report = _read_json(model_dir / "evaluation_val" / "evaluation.json", pending)
    test_report = _read_json(model_dir / "evaluation_test" / "evaluation.json", pending)
    field_report = _read_json(model_dir / "field_development" / "summary.json", pending)
    field_predictions = _read_field_predictions(
        model_dir / "field_development" / "predictions.jsonl", pending
    )

    complete_history = history or []
    candidates = [
        row for row in complete_history
        if row.get("val_loss") is not None and row.get("epoch") is not None
    ]
    best = min(candidates, key=lambda row: row["val_loss"]) if candidates else None

    return {
        "model": name,
        "artifact_status": "ready" if not pending else "pending",
        "pending_artifacts": pending,
        "best_epoch": None if best is None else int(best["epoch"]),
        "best_val_loss": None if best is None else best["val_loss"],
        "epochs_recorded": len(complete_history),
        "training_seconds": (
            sum(row["seconds"] for row in complete_history if "seconds" in row)
            if complete_history and all("seconds" in row for row in complete_history)
            else None
        ),
        "peak_gpu_gib": (
            max(row["peak_gpu_gib"] for row in complete_history if "peak_gpu_gib" in row)
            if complete_history and any("peak_gpu_gib" in row for row in complete_history)
            else None
        ),
        "parameters": None if run_info is None else run_info.get("parameters"),
        "gpu": None if run_info is None else run_info.get("gpu"),
        "validation": _evaluation_summary(val_report),
        "test": _evaluation_summary(test_report),
        "field_development": (
            None if field_report is None and field_predictions is None else {
                "processed": None if field_report is None else field_report.get("processed"),
                "not_processed": None if field_report is None else field_report.get("not_processed"),
                "seconds": None if field_report is None else field_report.get("seconds"),
                "field_metrics": None if field_report is None else field_report.get("field_metrics"),
                "evidence_boundary": None if field_report is None else field_report.get("evidence_boundary"),
                "scope": None if field_report is None else field_report.get("scope"),
                "predictions": field_predictions,
            }
        ),
        "_history": complete_history,
    }


def _fmt(value: Any, digits: int = 3, suffix: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value}{suffix}"
    if isinstance(value, float):
        return f"{value:.{digits}f}{suffix}"
    return str(value)


def _metric_row(model: dict[str, Any]) -> str:
    test = model["test"] or {}
    events = test.get("event_metrics_by_tolerance_m", {})
    event_f1 = "/".join(_fmt(events.get(str(t), {}).get("f1")) for t in TOLERANCES_M)
    mae_5 = _fmt(events.get("5", {}).get("mean_matched_mae_m"), 2, " m")
    interval = test.get("interval_metrics", {})
    interval_text = (
        f"IoU {_fmt(interval.get('grid_overall_iou'))}/"
        f"{_fmt(interval.get('mean_matched_iou'))}; F1 "
        f"{_fmt(interval.get('f1'))}"
    )
    terminal = test.get("terminal_metrics", {})
    terminal_text = (
        f"{_fmt(terminal.get('terminal_hit_count'))}/"
        f"{_fmt(terminal.get('truth_terminal_curve_count'))}; "
        f"{_fmt(terminal.get('mean_matched_terminal_error_m'), 2, ' m')}"
    )
    field = model["field_development"] or {}
    field_count = (
        f"{_fmt(field.get('processed'))}/"
        f"{_fmt(field.get('processed') + field.get('not_processed'))}"
        if isinstance(field.get("processed"), int)
        and isinstance(field.get("not_processed"), int)
        else "pending"
    )
    run_text = (
        f"{_fmt(model['training_seconds'], 1, ' s')}; "
        f"{_fmt(model['peak_gpu_gib'], 2, ' GiB')}; "
        f"{_fmt(model['parameters'], 0)}"
    )
    best = f"{_fmt(model['best_epoch'], 0)} / {_fmt(model['best_val_loss'])}"
    scope = test.get("evaluation_scope") or "pending"
    return (
        f"| {model['model']} | {best} | {event_f1} | {mae_5} | "
        f"{interval_text} | {terminal_text} | {run_text} | {field_count} | {scope} |"
    )


def _distribution_text(value: dict[str, Any] | None) -> str:
    if value is None:
        return "pending/无成功预测"
    return f"{_fmt(value.get('median'), 1)} [{_fmt(value.get('min'), 0)}, {_fmt(value.get('max'), 0)}]"


def _status_counts_text(value: dict[str, Any] | None) -> str:
    if value is None:
        return "pending"
    counts = value.get("prediction_status_counts", {})
    return ", ".join(f"{key}={count}" for key, count in counts.items()) or "无记录"


def _consistency_text(value: dict[str, Any] | None) -> str:
    if value is None:
        return "pending"
    counts = value.get("terminal_consistency_status_counts", {})
    return ", ".join(
        f"{key}={counts.get(key, 0)}"
        for key in ("consistent_candidate", "uncertain_candidate", "no_candidate")
    )


def _plot_curves(models: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": ["Times New Roman", "SimHei"],
        "font.sans-serif": ["SimHei"],
        "axes.unicode_minus": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.dpi": 200,
        "savefig.dpi": 200,
    })
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, model in zip(axes, models):
        history = model["_history"]
        train = [(row["epoch"], row["train_loss"]) for row in history if "train_loss" in row]
        validation = [(row["epoch"], row["val_loss"]) for row in history if "val_loss" in row]
        if train:
            axis.plot(*zip(*train), marker="o", markersize=3, label="Train loss")
        if validation:
            axis.plot(*zip(*validation), marker="o", markersize=3, label="Validation loss")
        if not train and not validation:
            axis.text(0.5, 0.5, "Pending", ha="center", va="center", transform=axis.transAxes)
        axis.set_title(model["model"])
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.grid(True, alpha=0.25)
        if train or validation:
            axis.legend(loc="best")
        for spine in axis.spines.values():
            spine.set_visible(True)
        axis.tick_params(direction="in", top=True, right=True)
    figure.suptitle("Transformer/U-Net learning curves")
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _markdown(run: Path, models: list[dict[str, Any]]) -> str:
    lines = [
        "# Transformer / U-Net 首轮基线",
        "",
        f"运行目录：{run}",
        "",
        "本次对比基于冻结生成器与现有 10k 合成样本，训练各 5 epoch。指标评估的是合成几何真值，不证明现场可检测性或现场准确率。",
        "",
        "| 模型 | 最佳 epoch / val loss | Test event F1 @2/5/10 m | MAE @5 m | Interval grid/matched IoU; F1 | Terminal 命中/总数; 匹配误差 | 训练秒; 峰值显存; 参数量 | Field 成功/总数 | Test 范围 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    lines.extend(_metric_row(model) for model in models)
    lines.extend([
        "",
        "## Field development 预测覆盖描述",
        "",
        "| 模型 | JSONL status | 每条成功曲线事件数中位数[范围] | 区间数中位数[范围] | terminal 候选数 | terminal 一致性状态 |",
        "|---|---|---:|---:|---:|---|",
    ])
    for model in models:
        field = model["field_development"] or {}
        predictions = field.get("predictions") or {}
        lines.append(
            f"| {model['model']} | {_status_counts_text(predictions)} | "
            f"{_distribution_text(predictions.get('event_count_per_predicted_curve'))} | "
            f"{_distribution_text(predictions.get('interval_count_per_predicted_curve'))} | "
            f"{_fmt(predictions.get('terminal_candidate_count'), 0)} | "
            f"{_consistency_text(predictions)} |"
        )
    lines.extend([
        "",
        "以上只表示处理覆盖和启发式末端候选一致性，绝非现场准确率；现场标签未知，field_metrics 为空。",
        "",
        "Field 列只报告 development 子集的成功处理数；现场标签未知，field_metrics 应为空，不计算或声称现场准确率。若某个 artifact 尚未生成，相关值显示为 pending/—，不会用 0 代替。",
        "",
        "5 epoch 只用于观察损失趋势和首轮基线。任何 test 指标仍受该合成分布与标签定义限制；不能据此把生成器称为已通过现场域验证。",
        "",
        "## 待产出文件",
        "",
    ])
    pending = [
        f"- {model['model']}: {path}"
        for model in models
        for path in model["pending_artifacts"]
    ]
    lines.extend(pending or ["- 无"])
    lines.extend([
        "",
        "## 学习曲线",
        "",
        "见 learning_curves.png。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    run = args.run.expanduser()
    if not run.is_absolute():
        run = HERE / run
    run = run.resolve()
    run.mkdir(parents=True, exist_ok=True)

    models = [_model_summary(run, name) for name in MODELS]
    status = "ready" if all(model["artifact_status"] == "ready" for model in models) else "pending"
    learning_plot = run / "learning_curves.png"
    _plot_curves(models, learning_plot)
    for model in models:
        model.pop("_history")

    report = {
        "schema": "fieldnn-tu-baseline-summary-1",
        "status": status,
        "run_directory": str(run),
        "training_scope": {
            "models": list(MODELS),
            "epochs": 5,
            "synthetic_sample_count": 10000,
            "field_realism_status": "not_established",
            "field_metrics": None,
        },
        "models": models,
        "learning_curves_png": str(learning_plot),
    }
    (run / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (run / "comparison.md").write_text(_markdown(run, models), encoding="utf-8")
    print(f"status={status}")
    print(f"comparison={run / 'comparison.md'}")
    print(f"learning_curves={learning_plot}")
    print(f"json={run / 'comparison.json'}")


if __name__ == "__main__":
    main()
