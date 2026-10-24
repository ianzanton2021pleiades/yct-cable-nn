"""Finalize the Iter12 comparison after Iter10/11/12 runs are complete.

The default path is CPU-only and reads saved predictions. It does not run
model inference. Pass --run-benchmark only when the six-model CUDA timing run
is explicitly intended.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
ITER5 = PROJECT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_Iter5_20261022"
ITER6 = PROJECT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_Iter6_20261022"
ITER8 = PROJECT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_Iter8_20261022"
ITER10 = PROJECT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_Iter10_20261023"
ITER11 = PROJECT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_Iter11_20261023"
ITER12 = HERE
EXPORTER = ITER10 / "export_existing.py"
PLOT_TABLE = ITER10 / "plot_table.py"
BENCHMARK = ITER10 / "benchmark_models.py"
TEMPLATE = HERE / "MODEL_CARD_TEMPLATE.md"
OUTPUT = HERE / "output" / "final_delivery"
DEFAULT_DATA = Path(r"E:\CableNNDataSet\CodeX\FieldNN_Iter8_20261022")
COVERAGE_DIR = ITER11 / "output" / "coverage_workpoint"
COVERAGE_SUMMARY = COVERAGE_DIR / "coverage_workpoint_summary.json"
COVERAGE_EXPORT_DIR = COVERAGE_DIR / "cpu_cache_export_060"
COVERAGE_CSV = COVERAGE_EXPORT_DIR / "performance.csv"
COVERAGE_MD = COVERAGE_EXPORT_DIR / "performance.md"
COVERAGE_CASES = COVERAGE_EXPORT_DIR / "selected_cases.csv"
COVERAGE_ASSETS = COVERAGE_EXPORT_DIR / "assets"
COVERAGE_CURVE = COVERAGE_DIR / "fixed_060_hit_curve.png"
U8_FIXED060_CSV = ITER10 / "output" / "existing_iter8" / "performance.csv"
MIN_COMPLETE_EPOCHS = 50
SIGNIFICANT_HIT_DROP = 4
MAX_CANDIDATE_MULTIPLE_OF_U5 = 1.25


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载现有模块：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def require_files(paths: list[Path], context: str) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{context} 尚未完整，缺少：\n" + "\n".join(missing))


def run_candidate(label: str, architecture: str, run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir.parent / "iteration10_summary.json"
    required = [
        summary_path,
        run_dir / "best.pt",
        run_dir / "selection.json",
        run_dir / "run_info.json",
        run_dir / "config.json",
        run_dir / "initial_validation.json",
        run_dir / "evaluation" / "field_report.json",
        run_dir / "evaluation" / "predictions.jsonl",
        run_dir / "history_steps.csv",
        run_dir / "optimizer_steps.csv",
        run_dir / "figures" / "field_loss.png",
        run_dir / "figures" / "field_hit20.png",
        run_dir / "figures" / "validation_operating_point.png",
    ]
    require_files(required, f"{label} 结果")
    summary = read_json(summary_path)
    if summary.get("status") != "complete":
        raise RuntimeError(f"{label} 尚未完成：{summary_path} status={summary.get('status')!r}")

    selection = read_json(run_dir / "selection.json")
    run_info = read_json(run_dir / "run_info.json")
    epochs_equivalent = float(run_info.get("epochs_equivalent", 0.0))
    if epochs_equivalent < MIN_COMPLETE_EPOCHS:
        raise RuntimeError(
            f"{label} 只有 {epochs_equivalent:g} 个等效 epoch；"
            f"bestU/bestT 只比较至少 {MIN_COMPLETE_EPOCHS} epoch 的完整轨迹。"
        )
    selected_source = selection.get("selected_source") or run_info.get("selected_source", "")
    selected_reference = "reference" in str(selected_source).casefold()
    selected_step = run_info.get("selected_step")
    if selected_step is None:
        selected_step = selection.get("step", selection.get("optimizer_step"))
    selected_initial = (
        not selected_reference
        and (
            (selected_step is not None and int(selected_step) <= 0)
            or "initial" in str(selected_source).casefold()
        )
    )
    field_thresholds_path = run_dir / "field_thresholds.json"
    if field_thresholds_path.is_file():
        field_thresholds = read_json(field_thresholds_path)
        threshold = float(field_thresholds["thresholds"]["event"])
        threshold_source = str(field_thresholds_path)
    else:
        validation = selection["validation"]
        threshold = float(selection.get("threshold", validation["val_threshold"]))
        if validation.get("val_density_fallback"):
            threshold = 1.000001
        step = selection.get("step", selection.get("optimizer_step"))
        steps_per_epoch = run_info.get("steps_per_epoch")
        epoch = (
            max(0, (int(step) - 1) // int(steps_per_epoch))
            if step is not None and steps_per_epoch
            else None
        )
        field_thresholds = {
            "model_name": architecture,
            "checkpoint_epoch": epoch,
            "calibration_split": "val",
            "thresholds": {"event": threshold},
            "policy": "现场event阈值由 selection.json 的验证工作点重建；未读取合成 thresholds.json。",
        }
        threshold_source = str(run_dir / "selection.json")

    return {
        "label": label,
        "architecture": architecture,
        "run_dir": run_dir,
        "checkpoint_path": run_dir / "best.pt",
        "package_path": run_dir / "inference.pt",
        "prediction_path": run_dir / "evaluation" / "predictions.jsonl",
        "threshold": threshold,
        "threshold_source": threshold_source,
        "field_thresholds": field_thresholds,
        "config_path": run_dir / "config.json",
        "run_info": run_info,
        "selection": selection,
        "selected_source": selected_source,
        "best_checkpoint_is_reference": selected_reference,
        "best_checkpoint_is_initial": selected_initial,
        "validation_tie": {
            "candidate_events": selection.get("validation", {}).get("val_candidates_total_selected"),
            "field_loss": selection.get("validation", {}).get("val_field_loss"),
            "source": str(run_dir / "selection.json"),
        },
        "priority": {"Iter10-U-Net": 3, "Iter11-U-Net": 4, "Iter12-Transformer": 5}[label],
        "candidate_class": (
            "complete_50_epoch_reference_fallback" if selected_reference else "complete_50_epoch_run"
        ),
    }


def candidates_after_runs() -> list[dict[str, Any]]:
    iter5 = ITER5 / "output" / "iter5" / "unet"
    iter8 = ITER8 / "output" / "iter8"
    old = [
        {
            "label": "Iter5-U-Net",
            "architecture": "unet",
            "run_dir": iter5,
            "checkpoint_path": iter5 / "best.pt",
            "prediction_path": iter5 / "field_e06" / "predictions.jsonl",
            "threshold": 0.60,
            "threshold_source": "Iter10 comparison baseline: field event threshold 0.60",
            "candidate_class": "baseline",
            "priority": 0,
            "validation_tie": {"candidate_events": None, "field_loss": None, "source": "not recorded"},
        },
        {
            "label": "Iter8-U-Net",
            "architecture": "unet",
            "run_dir": iter8 / "unet",
            "checkpoint_path": iter8 / "unet" / "best.pt",
            "prediction_path": iter8 / "paired" / "candidate" / "unet" / "field_fixed_e000" / "predictions.jsonl",
            "threshold": 0.94,
            "threshold_source": "Iter10 comparison baseline: field validation event threshold 0.94",
            "candidate_class": "historical_reference",
            "priority": 1,
        },
        {
            "label": "Iter8-Transformer",
            "architecture": "transformer",
            "run_dir": iter8 / "transformer",
            "checkpoint_path": iter8 / "transformer" / "best.pt",
            "prediction_path": iter8 / "paired" / "candidate" / "transformer" / "field_fixed_e010" / "predictions.jsonl",
            "threshold": 0.98,
            "threshold_source": "Iter10 comparison baseline: field validation event threshold 0.98",
            "candidate_class": "historical_reference",
            "priority": 2,
        },
    ]
    require_files(
        [path for item in old for path in (item["checkpoint_path"], item["prediction_path"])],
        "U5/U8/T8 历史比较结果",
    )
    for item in old:
        item["package_path"] = item["run_dir"] / "inference.pt"
        item["field_thresholds"] = {
            "model_name": item["architecture"],
            "calibration_split": "val",
            "thresholds": {"event": item["threshold"]},
            "policy": item["threshold_source"],
        }
        item["config_path"] = item["run_dir"] / "config.json"
        run_info_path = item["run_dir"] / "run_info.json"
        item["run_info"] = read_json(run_info_path) if run_info_path.is_file() else {}
        selection_path = item["run_dir"] / "selection.json"
        item["selection"] = read_json(selection_path) if selection_path.is_file() else {}

    recent = [
        run_candidate("Iter10-U-Net", "unet", ITER10 / "output" / "iter10_50ep" / "unet"),
        run_candidate("Iter11-U-Net", "unet", ITER11 / "output" / "iter11_50ep" / "unet"),
        run_candidate("Iter12-Transformer", "transformer", ITER12 / "output" / "iter12_50ep" / "transformer"),
    ]
    for item, reference_run in (
        (old[1], recent[1]),
        (old[2], recent[2]),
    ):
        reference_path = reference_run["run_dir"] / "reference_validation.json"
        require_files([reference_path], f"{item['label']} 验证参照")
        item["reference_validation_path"] = reference_path
        item["validation_tie"] = {
            "candidate_events": None,
            "field_loss": None,
            "source": str(reference_path),
        }
        if reference_path.is_file():
            reference = read_json(reference_path)
            item["validation_tie"].update(
                candidate_events=reference.get("val_candidates_total_selected"),
                field_loss=reference.get("val_field_loss"),
            )
    return [old[0], old[1], old[2], *recent]


def evaluate_candidate(
    candidate: dict[str, Any],
    records: list[dict[str, Any]],
    bags: list[dict[str, Any]],
    records_by_index: dict[int, dict[str, Any]],
    evaluator,
    exporter,
) -> None:
    prediction_rows = exporter.read_jsonl(candidate["prediction_path"])
    predictions_by_index, coverage = exporter.prediction_index(
        records, prediction_rows, float(candidate["threshold"]), evaluator
    )
    test_bags = evaluator.bag_rows("test", bags, records_by_index)
    val_bags = evaluator.bag_rows("val", bags, records_by_index)
    populations = {
        "nonwet_test": [bag for bag in test_bags if not bag.get("known_lab")],
        "wet_1500m_test": [bag for bag in test_bags if bag.get("known_lab")],
    }
    reports = {
        name: exporter.positive_population_report(
            "test", population_bags, records_by_index, predictions_by_index, evaluator
        )
        for name, population_bags in populations.items()
    }
    candidate["prediction_coverage"] = {
        "prediction_rows": len(prediction_rows),
        **coverage,
    }
    candidate["metrics"] = {
        name: exporter.population_row(candidate["label"], candidate["threshold"], report, name)
        for name, report in reports.items()
    }
    val_nonwet = [bag for bag in val_bags if not bag.get("known_lab")]
    val_report = exporter.positive_population_report(
        "val", val_nonwet, records_by_index, predictions_by_index, evaluator
    )
    candidate["validation_row"] = exporter.population_row(
        candidate["label"], candidate["threshold"], val_report, "nonwet_val"
    )
    validation_tie = candidate.setdefault("validation_tie", {})
    if validation_tie.get("candidate_events") is None:
        validation_tie["candidate_events"] = candidate["validation_row"].get("candidate_events_total")
        validation_tie["source"] = str(candidate["prediction_path"]) + " (Iter6 val replay)"


def csv_write(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def selection_key(candidate: dict[str, Any]) -> tuple[float, int, float, float, float, int]:
    row = candidate["metrics"]["nonwet_test"]
    macro = row.get("group_macro_hit_at_20m")
    hits = row.get("hits_at_20m")
    candidates = row.get("candidate_events_total")
    val_tie = candidate.get("validation_tie") or {}
    val_candidates = val_tie.get("candidate_events")
    val_loss = val_tie.get("field_loss")
    return (
        round(float(macro), 12) if macro is not None else -1.0,
        int(hits) if hits is not None else -1,
        -float(candidates) if candidates is not None else -math.inf,
        -float(val_candidates) if val_candidates is not None else -math.inf,
        -float(val_loss) if val_loss is not None else -math.inf,
        int(candidate.get("priority", 0)),
    )


def choose_architecture_candidate(
    candidates: list[dict[str, Any]], architecture: str, u5: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    eligible = [
        item for item in candidates
        if item["architecture"] == architecture
        and item.get("candidate_class") == "complete_50_epoch_run"
    ]
    if not eligible:
        reference_label = "Iter8-U-Net" if architecture == "unet" else "Iter8-Transformer"
        reference = next(item for item in candidates if item["label"] == reference_label)
        return reference, {
            "fallback": True,
            "reason": "本轮更新权重未超过验证参考，run 内 best.pt 仍选择历史 reference 或初始权重。",
            "selected_50_epoch_candidate": None,
            "reference": reference_label,
        }
    selected = max(eligible, key=selection_key)
    baseline = u5["metrics"]["nonwet_test"]
    selected_row = selected["metrics"]["nonwet_test"]
    hit_drop = int(baseline.get("hits_at_20m") or 0) - int(selected_row.get("hits_at_20m") or 0)
    baseline_candidates = float(baseline.get("candidate_events_total") or 0)
    selected_candidates = float(selected_row.get("candidate_events_total") or 0)
    candidate_limit = baseline_candidates * MAX_CANDIDATE_MULTIPLE_OF_U5
    reasons = []
    if hit_drop >= SIGNIFICANT_HIT_DROP:
        reasons.append(
            f"known hit@20 少于 U5 基线 {hit_drop} 个（回退阈值 {SIGNIFICANT_HIT_DROP}）"
        )
    if selected_candidates > candidate_limit:
        reasons.append(
            f"候选数 {selected_candidates:g} 超过 U5 的 1.25 倍预算 {candidate_limit:g}"
        )
    if reasons:
        reference_label = "Iter8-U-Net" if architecture == "unet" else "Iter8-Transformer"
        reference = next(item for item in candidates if item["label"] == reference_label)
        return reference, {
            "fallback": True,
            "reason": "；".join(reasons),
            "selected_50_epoch_candidate": selected["label"],
            "reference": reference_label,
        }
    return selected, {
        "fallback": False,
        "reason": "从至少 50 epoch 的完整新轨迹中按现场指标选择；历史权重仅作参照。",
        "selected_50_epoch_candidate": selected["label"],
        "reference": None,
    }


def format_hits(row: dict[str, Any], tolerance: int) -> str:
    hits = row.get(f"hits_at_{tolerance}m")
    positions = row.get("known_positions")
    return f"{int(hits)}/{int(positions)}" if hits is not None and positions is not None else "—"


def format_number(value: Any, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}" if value is not None else "未记录"


def format_summary(row: dict[str, Any]) -> str:
    prefix = (
        f"{row.get('groups', 0)} 组、{row.get('known_positions', 0)} 个已知位置；"
        f"hit@10/20/30 m = {format_hits(row, 10)}/{format_hits(row, 20)}/{format_hits(row, 30)}；"
        f"group macro@20 m = {float(row['group_macro_hit_at_20m']):.4f}；"
        f"候选 {int(row.get('candidate_events_total') or 0)} 个 / "
        f"{int(row.get('curves') or 0)} 条 sweep"
    )
    error = row.get("mean_matched_error_at_20m")
    return prefix + (f"；matched mean error@20 m = {float(error):.2f} m" if error is not None else "；暂无可计算的匹配误差")


def interpretation_notes(candidates: list[dict[str, Any]]) -> str:
    by_label = {candidate["label"]: candidate for candidate in candidates}
    u5 = by_label["Iter5-U-Net"]["metrics"]["nonwet_test"]
    u8 = by_label["Iter8-U-Net"]["metrics"]["nonwet_test"]
    i11 = by_label["Iter11-U-Net"]["metrics"]["nonwet_test"]

    def values(row: dict[str, Any]) -> tuple[int, int, float, int]:
        return (
            int(row.get("hits_at_20m") or 0),
            int(row.get("known_positions") or 0),
            float(row.get("group_macro_hit_at_20m") or 0.0),
            int(row.get("candidate_events_total") or 0),
        )

    u5_hits, u5_positions, u5_macro, u5_candidates = values(u5)
    u8_hits, u8_positions, u8_macro, u8_candidates = values(u8)
    i11_hits, i11_positions, i11_macro, i11_candidates = values(i11)
    lines = [
        f"Iter11 相对 U5：hit@20 {u5_hits}/{u5_positions}→{i11_hits}/{i11_positions}，"
        f"group macro {u5_macro:.4f}→{i11_macro:.4f}，候选 {u5_candidates}→{i11_candidates}。"
        "这是重复使用保留组上的已知位置覆盖比较，不是完整准确率。"
    ]
    if (i11_hits, i11_positions, round(i11_macro, 12)) == (
        u8_hits,
        u8_positions,
        round(u8_macro, 12),
    ):
        delta = i11_candidates - u8_candidates
        lines.append(
            f"Iter11 相对 U8 保持相同 hit@20 与 group macro，候选数变化 {delta:+d}；"
            "候选仍未核验，因此把它解释为长轨迹下保持覆盖与稳定适应，不宣称准确率提升。"
        )
    else:
        lines.append(
            f"Iter11 相对 U8：hit@20 {u8_hits}/{u8_positions}→{i11_hits}/{i11_positions}，"
            f"group macro {u8_macro:.4f}→{i11_macro:.4f}，候选 {u8_candidates}→{i11_candidates}；"
            "候选差异不等于误报差异。"
        )
    return "\n\n".join(lines)


def comparison_markdown(
    rows: list[dict[str, Any]], benchmark_path: Path | None, notes: str
) -> str:
    lines = [
        "# FieldNN 现场结果比较",
        "",
        "主表只列 Iter5-U-Net 基线、至少完成 50 epoch 的 U-Net 优胜者和 Transformer 优胜者；历史 U8/T8 只在回退条件触发时作为安全参照。",
        "所有现场指标复用 Iter6 `evaluate_field.py` 的正位置匹配 API；候选数是待核验负担，不是误报数。",
        "",
        "| 模型 | event 阈值 | 非湿 hit@10/20/30 | group macro@20 | 候选 / sweep | matched mean error@20 (m) | 湿接头 hit@20 | 湿组候选 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        wet = row["wet"]
        nonwet = row["nonwet"]
        wet_hits = format_hits(wet, 20)
        lines.append(
            f"| {markdown_link(row['display'], row['candidate']['checkpoint_path'])} | {row['threshold']:.6g} | "
            f"{format_hits(nonwet, 10)}/{format_hits(nonwet, 20)}/{format_hits(nonwet, 30)} | "
            f"{float(nonwet['group_macro_hit_at_20m']):.4f} | "
            f"{int(nonwet.get('candidate_events_total') or 0)} / {int(nonwet.get('curves') or 0)} | "
            f"{float(nonwet['mean_matched_error_at_20m']):.2f} | {wet_hits} | "
            f"{int(wet.get('candidate_events_total') or 0)} |"
        )
    lines += [
        "",
        "Iter11/Iter12 与历史候选的比较来自反复用于开发的保留组；本表不构成新盲测。",
        "现场未标注位置仍为 unknown，未匹配候选仍未核验；因此不计算完整现场 Accuracy、precision、recall 或 F1。",
        "1500 m 湿接头组单列，未并入非湿组排名。测试组上的优胜者名称仅用于事后汇总，不用于重选权重或阈值。",
        notes,
    ]
    if benchmark_path is None:
        lines.append("NN-only latency 尚未汇总；如需 GPU 计时，显式运行 `finalize_delivery.py --run-benchmark`。")
    else:
        lines.append(f"NN-only benchmark：`{benchmark_path}`；时延仅覆盖模型 forward，不含 FDR、FFT、特征处理或后处理。")
    return "\n".join(lines) + "\n"


def training_description(candidate: dict[str, Any]) -> str:
    config_path = candidate.get("config_path")
    if not config_path or not config_path.is_file():
        run_info_path = Path(candidate["run_dir"]) / "run_info.json"
        if run_info_path.is_file():
            return f"历史 checkpoint；训练信息见 {markdown_link('run_info.json', run_info_path)}。"
        return "历史 checkpoint；旧轮次未保存可复用的训练配置说明。"
    config = read_json(config_path)
    model_config = config.get("model_config") or config.get("model") or {}
    parts = [f"实验：`{candidate['label']}`"]
    if model_config:
        parts.append("模型配置：" + ", ".join(f"{key}={value}" for key, value in model_config.items()))
    for key, label in (
        ("epochs_equivalent", "等效 epoch"),
        ("steps", "更新步数"),
        ("batch_size", "batch size"),
        ("backbone_learning_rate", "主干学习率"),
        ("stem_learning_rate", "stem 学习率"),
        ("stem_lr", "stem 学习率"),
        ("weak_weight", "现场弱监督权重"),
        ("calibration_weight", "校准参考权重"),
        ("init_iter8", "Iter8 初始化"),
        ("stem_only", "仅训练输入 stem"),
    ):
        if key in config:
            parts.append(f"{label}={config[key]}")
    selection = candidate.get("selection") or {}
    if selection:
        parts.append(
            f"验证选择 checkpoint step={selection.get('step', '未记录')}、"
            f"field event threshold={candidate['threshold']:.6g}"
        )
    return "；".join(parts) + "。"


def absolute_target(path: Path) -> str:
    return "/" + path.expanduser().resolve().as_posix()


def markdown_link(label: str, path: Path) -> str:
    return f"[{label}](<{absolute_target(path)}>)"


def markdown_image(label: str, path: Path) -> str:
    return f"![{label}](<{absolute_target(path)}>)"


def summarize_data(
    records: list[dict[str, Any]],
    bags: list[dict[str, Any]],
    records_by_index: dict[int, dict[str, Any]],
    evaluator,
    prediction_count: int,
) -> dict[str, Any]:
    records_by_split = {
        split: sum(row.get("split") == split for row in records)
        for split in ("train", "val", "test")
    }
    train_bags = evaluator.bag_rows("train", bags, records_by_index)
    val_bags = evaluator.bag_rows("val", bags, records_by_index)
    test_bags = evaluator.bag_rows("test", bags, records_by_index)
    nonwet_test_groups = {row["group"] for row in test_bags if not row.get("known_lab")}
    wet_test_groups = {row["group"] for row in test_bags if row.get("known_lab")}
    labeled_indices = {int(index) for bag in bags for index in bag.get("indices", [])}
    total_curves = max(len(records), prediction_count)
    return {
        "total_curves": total_curves,
        "labeled_curves": len(labeled_indices),
        "curves_by_split": records_by_split,
        "train_groups": len({row["group"] for row in train_bags}),
        "validation_groups": len({row["group"] for row in val_bags}),
        "nonwet_test_groups": len(nonwet_test_groups),
        "wet_test_groups": len(wet_test_groups),
        "unknown_curves": total_curves - len(labeled_indices),
    }


def format_data_summary(summary: dict[str, Any], winner: dict[str, Any]) -> str:
    splits = summary["curves_by_split"]
    coverage = winner["prediction_coverage"]
    return (
        f"全场推理共有 {summary['total_curves']} 条记录，其中带标签的 field split 有 "
        f"{splits['train']}/{splits['val']}/{splits['test']} 条（train/val/test）；"
        f"val 有 {summary['validation_groups']} 组，非湿 test 有 "
        f"{summary['nonwet_test_groups']} 组，1500 m 湿接头 test 有 "
        f"{summary['wet_test_groups']} 组。全场有 {summary['unknown_curves']} 条曲线没有已知正位置，"
        f"仍保留在全场推理；当前赢家保存了 {coverage.get('predicted_rows', 0)}/"
        f"{coverage.get('prediction_rows', 0)} 条预测记录。"
    )


def architecture_description(architecture: str) -> str:
    shared = (
        "输入为 12 通道、8192 点，12→8 输入 stem 接现有 Iter3 主干；三个输出头分别预测 event、interval、terminal。"
        "工作网格为 0–3250 m、9 kHz–200 MHz；项目目标为不超过 2.5 km 的短电缆。"
        "现场解码在 nominal 距离轴固定抑制首端 30 m 的 event/interval 候选。"
    )
    if architecture == "unet":
        detail = (
            "U-Net 采用七尺度 1D 编码/解码，特征宽度为 32/64/128/192/192/192/192；"
            "bottleneck 串联 dilation 1/2/4/8 残差块，并通过全局均值投影回注上下文；"
            "解码使用线性上采样和跳连。"
        )
    else:
        detail = (
            "Transformer 使用相同七尺度编码/解码与 32/64/128/192/192/192/192 宽度；"
            "bottleneck 为 4-head、d_model=192、feed-forward=768、depth=3 的 Transformer encoder。"
        )
    return shared + detail


def training_objective(candidates: list[dict[str, Any]]) -> str:
    settings = []
    for candidate in candidates:
        if not str(candidate.get("candidate_class", "")).startswith("complete_50_epoch"):
            continue
        config = read_json(candidate["config_path"])
        adaptation = config.get("adaptation10", {})
        weak = config.get("weak_weight", adaptation.get("weak_weight"))
        calibration = config.get("calibration_weight", adaptation.get("calibration_weight", 0.5))
        frequency = config.get("field_every_steps", adaptation.get("field_every_steps", 2))
        settings.append(
            f"{candidate['label']}: λMIL={float(weak):g}, λcal={float(calibration):g}, field/cal 每 {int(frequency)} 个训练步"
        )
    return (
        "合成批次使用 Iter8 三头目标：event/terminal Gaussian focal，interval 为全网格加权 BCE 与 Dice。"
        "每隔配置指定的训练步加入同定义的正位置跨 sweep bag MIL 与短接/无电缆校准损失；"
        "MIL 每个已知位置通过 bag 内 max pooling 产生一个正例，unknown 不作为负例。"
        "校准参考只对 30 m 以外的 event/interval mask 施加负约束，不监督 terminal，也不是健康电缆。"
        "train/val 使用相同正例 MIL loss；现场曲线另记录固定 0.94 event 阈值下按组平均的 hit@20 m。\n\n"
        "三轮实际设置：" + "；".join(settings) + "。"
    )


def candidate_run_markdown(candidates: list[dict[str, Any]], winner: dict[str, Any]) -> str:
    recent = [
        item for item in candidates
        if str(item.get("candidate_class", "")).startswith("complete_50_epoch")
    ]
    winner_is_old = winner.get("candidate_class") != "complete_50_epoch_run"
    lines = []
    if winner_is_old:
        lines.append(
            f"当前回退赢家为历史模型 {winner['label']}；以下 Iter10/11/12 曲线是各自独立的 50 epoch 探索轨迹，"
            "不是赢家的训练历史，不拼接。"
        )
    else:
        lines.append("每轮训练曲线单独展示；不同实验轨迹不拼接。赢家曲线只对应赢家自身训练过程。")
    for candidate in recent:
        info = candidate["run_info"]
        config = read_json(candidate["config_path"])
        selection = candidate.get("selection") or {}
        initial_validation = read_json(candidate["run_dir"] / "initial_validation.json")
        history_path = candidate["run_dir"] / "history_steps.csv"
        with history_path.open("r", encoding="utf-8-sig", newline="") as stream:
            recorded_points = sum(1 for _ in csv.DictReader(stream))
        optimizer_path = candidate["run_dir"] / "optimizer_steps.csv"
        with optimizer_path.open("r", encoding="utf-8-sig", newline="") as stream:
            optimizer_rows = list(csv.DictReader(stream))
        amp_skips = sum(int(float(row.get("amp_skipped") or 0)) for row in optimizer_rows)
        total_parameters = info.get("model_parameters", info.get("parameters", "未记录"))
        trainable = info.get("trainable_parameters", "未记录")
        source = candidate.get("selected_source") or info.get("selected_source", selection.get("selected_source", "未记录"))
        step = info.get("selected_step")
        if step is None:
            step = selection.get("step", selection.get("optimizer_step", "未记录"))
        if step is None:
            step = "no update step (validation reference)"
        init = info.get("initialization_checkpoint", config.get("initialization_checkpoint", "未记录"))
        train_settings = (
            f"batch={config.get('batch_size', '未记录')}，"
            f"λMIL={config.get('weak_weight', '未记录')}，"
            f"stem LR={config.get('stem_learning_rate', '未记录')}，"
            f"init_iter8={config.get('init_iter8', '未记录')}，"
            f"stem_only={config.get('stem_only', '未记录')}"
        )
        init_path = Path(init) if init and init != "未记录" else None
        init_display = (
            markdown_link("初始化 checkpoint", init_path)
            if init_path is not None and init_path.is_file()
            else f"`{init}`"
        )
        if candidate.get("best_checkpoint_is_reference"):
            role = "独立训练轨迹；best.pt 为历史 reference"
            curve_label = f"{candidate['label']} exploration curve (best.pt is reference)"
            checkpoint_label = "best.pt (historical reference)"
        elif candidate.get("best_checkpoint_is_initial"):
            role = (
                "本轮选择；best.pt 仍为初始化 checkpoint"
                if candidate["label"] == winner["label"]
                else "独立训练轨迹；best.pt 仍为初始化 checkpoint"
            )
            curve_label = f"{candidate['label']} exploration curve (best.pt is initialization)"
            checkpoint_label = "best.pt (initialization checkpoint)"
        elif candidate["label"] == winner["label"]:
            role = "本轮赢家训练轨迹"
            curve_label = f"{candidate['label']} training trajectory"
            checkpoint_label = "best.pt"
        else:
            role = "独立训练轨迹"
            curve_label = f"{candidate['label']} independent training trajectory"
            checkpoint_label = "best.pt"
        lines += [
            f"### {candidate['label']}（{role}）",
            "",
            f"等效 epoch={info.get('epochs_equivalent', '未记录')}；成功 optimizer updates="
            f"{info.get('optimizer_steps', '未记录')}；训练 attempt={len(optimizer_rows)}，"
            f"AMP skip={amp_skips}；真实 history 点={recorded_points}；"
            f"最佳 source=`{source}`，step={step}；总参数={total_parameters}，可训练参数={trainable}；"
            f"{train_settings}。",
            f"训练前验证基线：train/val MIL loss="
            f"{format_number(initial_validation.get('train_field_loss'))}/"
            f"{format_number(initial_validation.get('val_field_loss'))}，"
            f"val group macro@20={format_number(initial_validation.get('val_macro_hit20_selected'))}，"
            f"阈值={format_number(initial_validation.get('val_threshold'))}，"
            f"候选={initial_validation.get('val_candidates_total_selected', '未记录')}。",
            f"初始化 checkpoint：{init_display}。配置与产物："
            f"{markdown_link('config.json', candidate['config_path'])}、"
            f"{markdown_link('run_info.json', candidate['run_dir'] / 'run_info.json')}、"
            f"{markdown_link('initial_validation.json', candidate['run_dir'] / 'initial_validation.json')}、"
            f"{markdown_link(checkpoint_label, candidate['checkpoint_path'])}、"
            f"{markdown_link('history_steps.csv', history_path)}、"
            f"{markdown_link('optimizer_steps.csv', candidate['run_dir'] / 'optimizer_steps.csv')}。",
            "",
            markdown_image(f"{curve_label}: field loss", candidate['run_dir'] / "figures" / "field_loss.png"),
            "",
            markdown_image(f"{curve_label}: fixed-.94 group hit@20", candidate['run_dir'] / "figures" / "field_hit20.png"),
            "",
            markdown_image(f"{curve_label}: validation operating point", candidate['run_dir'] / "figures" / "validation_operating_point.png"),
            "",
        ]
    pilot_dir = ITER12 / "output" / "pilot_from_t8_stem" / "transformer"
    pilot_info_path = pilot_dir / "run_info.json"
    pilot_selection_path = pilot_dir / "selection.json"
    pilot_history_path = pilot_dir / "history_steps.csv"
    if all(path.is_file() for path in (pilot_info_path, pilot_selection_path, pilot_history_path)):
        pilot_info = read_json(pilot_info_path)
        pilot_selection = read_json(pilot_selection_path)
        with pilot_history_path.open("r", encoding="utf-8-sig", newline="") as stream:
            pilot_points = sum(1 for _ in csv.DictReader(stream))
        pilot_validation = pilot_selection.get("validation", {})
        pilot_init = pilot_info.get("initialization_checkpoint", "未记录")
        pilot_init_path = Path(pilot_init) if pilot_init and pilot_init != "未记录" else None
        pilot_init_link = (
            markdown_link("Iter8 Transformer checkpoint", pilot_init_path)
            if pilot_init_path is not None and pilot_init_path.is_file()
            else f"`{pilot_init}`"
        )
        lines += [
            "### Iter12 T8-stem pilot（独立试跑；不纳入正式 50 epoch 结果与候选排名）",
            "",
            f"该 pilot 冻结 Iter8 Transformer 主干，仅训练输入 stem；记录 {pilot_info.get('epochs_equivalent', '未记录')} 个等效 epoch、"
            f"{pilot_info.get('optimizer_step', '未记录')} updates、{pilot_points} 个真实 history 点。"
            f"初始来源：{pilot_init_link}。当时 train/val field MIL loss="
            f"{float(pilot_validation['train_field_loss']):.6f}/{float(pilot_validation['val_field_loss']):.6f}；"
            "正例梯度趋于饱和，validation 未改善。该 pilot 未评分 test。",
            "正式 I12 是从 Iter5 Transformer 主干重新启动的独立完整 50 epoch；config 中 `init_iter8=false`，"
            "T8 仅保留为 validation reference。pilot 曲线不与正式轨迹拼接。",
            "保留文件："
            f"{markdown_link('config.json', pilot_dir / 'config.json')}、"
            f"{markdown_link('run_info.json', pilot_info_path)}、"
            f"{markdown_link('selection.json', pilot_selection_path)}、"
            f"{markdown_link('history_steps.csv', pilot_history_path)}、"
            f"{markdown_link('optimizer_steps.csv', pilot_dir / 'optimizer_steps.csv')}。",
            "",
        ]
    return "\n".join(lines)


def benchmark_entry(candidate: dict[str, Any], benchmark_path: Path | None) -> dict[str, Any]:
    if benchmark_path is not None:
        payload = read_json(benchmark_path)
        row = next(
            (item for item in payload.get("models", []) if item.get("label") == candidate["label"]),
            None,
        )
        if row is not None:
            return row
    local = Path(candidate["run_dir"]) / "nn_benchmark.json"
    if local.is_file():
        return read_json(local)
    return {}


def enrich_metric_row(
    row: dict[str, Any], candidate: dict[str, Any], benchmark_path: Path | None
) -> dict[str, Any]:
    benchmark = benchmark_entry(candidate, benchmark_path)
    info = candidate.get("run_info") or {}
    row["parameters"] = benchmark.get("parameters") or info.get("model_parameters") or info.get("parameters")
    row["model_state_tensor_bytes"] = benchmark.get("weight_tensor_bytes") or info.get("model_weight_tensor_bytes")
    row["checkpoint_bytes"] = benchmark.get("checkpoint_bytes") or candidate["checkpoint_path"].stat().st_size
    package_path = Path(candidate.get("final_package_path") or candidate["package_path"])
    row["inference_package_bytes"] = benchmark.get("inference_package_bytes") or (
        package_path.stat().st_size if package_path.is_file() else None
    )
    row["latency_median_ms"] = benchmark.get("median_ms")
    row["latency_p95_ms"] = benchmark.get("p95_ms")
    row["latency_scope"] = benchmark.get("scope")
    return row


def runtime_summary(
    candidate: dict[str, Any], benchmark_path: Path | None, package_parameters: int
) -> str:
    benchmark = benchmark_entry(candidate, benchmark_path)
    parameter_count = benchmark.get("parameters") or package_parameters
    if benchmark.get("median_ms") is None or benchmark.get("p95_ms") is None:
        return (
            f"参数量 {parameter_count:,}；NN-only latency/warmup/repetition 尚未测量，"
            "本卡不填推理速度。"
        )
    warmup = benchmark.get("warmup_iterations", benchmark.get("warmups", "未记录"))
    measured = benchmark.get("measured_iterations", benchmark.get("repetitions", "未记录"))
    amp = benchmark.get("amp", benchmark.get("autocast", "未记录"))
    shape = benchmark.get("input_shape", "未记录")
    return (
        f"参数量 {parameter_count:,}；batch={benchmark.get('batch_size', 1)}，input shape={shape}，"
        f"autocast={amp}；预热 {warmup} 次、同步测量 {measured} 次；"
        f"NN forward median/p95={float(benchmark['median_ms']):.3f}/{float(benchmark['p95_ms']):.3f} ms。"
        f"范围：{benchmark.get('scope', 'NN-only；输入预载入，不含FDR/FFT/特征和后处理。')}"
    )


def coverage_workpoint_rows(
    i11: dict[str, Any], benchmark_path: Path | None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = read_json(COVERAGE_SUMMARY)
    with COVERAGE_CSV.open("r", encoding="utf-8-sig", newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    rows = [
        dict(row)
        for row in source_rows
        if row.get("row_type") == "field_population"
        and row.get("model") == "Iter11-U-Net-fixed0.60"
        and row.get("population") in {"nonwet_test", "wet_1500m_test"}
    ]
    by_population = {row["population"]: row for row in rows}
    rows = [by_population["nonwet_test"], by_population["wet_1500m_test"]]
    for row in rows:
        row["model"] = "Iter11-U-Net (same checkpoint; CPU cache 138)"
        row["notes"] = (
            "Pre-existing Iter5 U-Net fixed event threshold 0.60; CPU FP32 batch-1 prediction over 138 annotated cache curves only. "
            "Not full 213-sweep inference and not test-tuned. Candidate events remain unverified; parameters and NN-only timing, if shown, are inherited from the same I11 checkpoint benchmark."
        )
        enrich_metric_row(row, i11, benchmark_path)
    return summary, rows


def coverage_workpoint_markdown(
    summary: dict[str, Any],
    i11: dict[str, Any],
    inline: bool,
    best_unet_label: str,
    coverage_csv_output: Path | None,
    coverage_table_output: Path | None,
    package_thresholds: Path | None,
) -> str:
    fixed = summary["fixed_threshold_populations"]
    fixed_nonwet = fixed["nonwet_test"]
    fixed_wet = fixed["wet_1500m_test"]
    main_nonwet = i11["metrics"]["nonwet_test"]
    main_wet = i11["metrics"]["wet_1500m_test"]
    curve = summary["fixed_060_curve"]
    cpu = summary["cpu_inference"]
    source = summary["workpoint"]["threshold_source"]
    with U8_FIXED060_CSV.open("r", encoding="utf-8-sig", newline="") as stream:
        u8_rows = list(csv.DictReader(stream))
    u8_row = next(
        row for row in u8_rows
        if row.get("row_type") == "field_population"
        and row.get("model") == "Iter8-U-Net-fixed@0.6"
        and row.get("population") == "nonwet_test"
    )
    u8_candidates = int(float(u8_row["candidate_events_total"]))

    if inline:
        table_link = "主 `performance.csv`/`performance_table.png` 中已添加同一 I11 checkpoint 的 .60 次工作点行。"
    else:
        table_link = (
            f"bestU 是 {best_unet_label}；"
            f".60 点独立放在 {markdown_link('coverage_point.csv', coverage_csv_output)} 与 "
            f"{markdown_link('coverage_point_table.png', coverage_table_output)}，不参加 bestU/bestT 排名。"
        )
    lines = [
        "### I11 固定 0.60 覆盖工作点（同一模型的补充 operating point）",
        "",
        f"阈值 `{float(summary['workpoint']['event_threshold']):.2f}` 沿用 {source}，没有依据 test 调整；"
        "它是同一 I11 checkpoint 的第二工作点，不是第 4 个模型；不替换主 .94 阈值，也不改变 bestU/bestT 权重选择。",
        "",
        "| I11 工作点 | 评估范围 | 非湿 hit@20 m | group macro@20 | 候选 / sweep | 湿组 hit@20 m | 湿组候选 / sweep |",
        "|---|---|---:|---:|---:|---:|---:|",
        f"| .94 主点 | 全场 213 sweep CUDA FP16 推理；已标注 test 组评分 | "
        f"{format_hits(main_nonwet, 20)} | {float(main_nonwet['group_macro_hit_at_20m']):.4f} | "
        f"{int(main_nonwet['candidate_events_total'])}/{int(main_nonwet['curves'])}="
        f"{float(main_nonwet['candidate_events_per_curve']):.3f} | {format_hits(main_wet, 20)} | "
        f"{int(main_wet['candidate_events_total'])}/{int(main_wet['curves'])} |",
        f"| .60 固定覆盖点 | CPU FP32 batch 1；{cpu['curve_count']} 条已标注 cache 曲线 | "
        f"{int(fixed_nonwet['hits_at_20m'])}/{int(fixed_nonwet['known_positions'])} | "
        f"{float(fixed_nonwet['group_macro_hit_at_20m']):.4f} | "
        f"{int(fixed_nonwet['candidate_events_total'])}/{int(fixed_nonwet['curves'])}="
        f"{float(fixed_nonwet['candidate_events_per_curve']):.3f} | "
        f"{int(fixed_wet['hits_at_20m'])}/{int(fixed_wet['known_positions'])} | "
        f"{int(fixed_wet['candidate_events_total'])}/{int(fixed_wet['curves'])} |",
        "",
        f"固定 .60 曲线包含 {curve['record_points']} 个真实点（update {curve['first_optimizer_update']}–"
        f"{curve['last_optimizer_update']}）；固定 .60 group hit@20 的 train/val 末点为 "
        f"{float(curve['train_final_group_macro_hit20']):.5f}/{float(curve['val_final_group_macro_hit20']):.5f}。"
        f"I11 的验证选中 step={i11['run_info'].get('selected_step', i11['selection'].get('step'))}；"
        "末点只是训练轨迹记录，不是 checkpoint 选择点。这是已知位置覆盖指标，不是完整 Accuracy。",
        f"I11 .60 CPU cache 有 {int(fixed_nonwet['candidate_events_total'])} 个非湿候选；"
        f"U8 既有 .60 结果为 {u8_candidates} 个候选，但 U8 数来自全 213 sweep 推理，I11 .60 只覆盖 138 条标注缓存曲线。"
        "两者范围不同，不能把表面上的 2 个差值称作改进；所有未匹配候选仍未核验。",
        "表格中 .60 行的参数量与 NN-only latency（若已测）沿用同一个 I11 checkpoint benchmark；CPU FP32 batch-1 是现场覆盖预测的计算路径，不是 GPU latency 的计算路径。",
        table_link,
        "",
        "审计资料："
        f"{markdown_link('coverage_workpoint_summary.json', COVERAGE_SUMMARY)}、"
        f"{markdown_link('coverage workpoint performance.csv', COVERAGE_CSV)}、"
        f"{markdown_link('coverage workpoint performance.md', COVERAGE_MD)}、"
        f"{markdown_link('fixed_060_hit_curve.png', COVERAGE_CURVE)}、"
        f"{markdown_link('selected_cases.csv', COVERAGE_CASES)}。",
    ]
    if package_thresholds is not None:
        lines += [
            "",
            f"同一 I11 推理包的可选 .60 阈值配置：{markdown_link('thresholds_coverage_060.json', package_thresholds)}。"
            "默认 `thresholds.json`/`field_thresholds.json` 保持 .94，不复制第二份权重。",
        ]
    case_images = []
    with COVERAGE_CASES.open("r", encoding="utf-8-sig", newline="") as stream:
        cases = list(csv.DictReader(stream))
    for index, case in enumerate(cases, start=1):
        image_path = (COVERAGE_ASSETS / Path(case["image_path"]).name).resolve()
        if image_path.is_file():
            case_images.append(markdown_image(f"I11 fixed .60 case {index}", image_path))
    if case_images:
        lines.extend(["", *case_images])
    return "\n".join(lines)


def case_images(case_csv: Path, export_dir: Path) -> str:
    if not case_csv.is_file():
        return "案例图未生成。"
    with case_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    images = []
    for index, row in enumerate(rows[:4], start=1):
        image_path = (export_dir / (row.get("image_path") or "").replace("\\", "/")).resolve()
        if image_path.is_file():
            images.append(markdown_image(f"现场案例 {index}", image_path))
    return "\n\n".join(images) if images else "当前 exporter 未选出可展示案例。"


def package_winner(candidate: dict[str, Any], output: Path) -> tuple[Path, int, int]:
    import torch

    source = Path(candidate["package_path"])
    if not source.is_file():
        source = Path(candidate["checkpoint_path"])
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    model_state = checkpoint.get("model") or checkpoint.get("state_dict")
    epoch = checkpoint.get("epoch")
    if epoch is None:
        raise ValueError(f"选中 checkpoint 未记录 epoch，拒绝生成不完整推理包：{source}")
    selection = checkpoint.get("selection") or candidate.get("selection") or {}
    model_name = checkpoint.get("model_name") or candidate["architecture"]
    config = checkpoint.get("config")
    if config is None:
        raise ValueError(f"选中 checkpoint 未包含 config：{source}")
    parameter_count = sum(int(tensor.numel()) for tensor in model_state.values())
    package_dir = output / "best_model"
    package_dir.mkdir(parents=True, exist_ok=True)
    package_path = package_dir / "inference.pt"
    torch.save(
        {
            "model_name": model_name,
            "config": config,
            "model": model_state,
            "epoch": int(epoch),
            "selection": selection,
        },
        package_path,
    )
    thresholds = dict(candidate.get("field_thresholds") or {})
    thresholds.update(
        model_name=model_name,
        checkpoint_epoch=int(epoch),
        calibration_split="val",
        thresholds={"event": float(candidate["threshold"]), "interval": 0.99, "terminal": 0.4},
        policy=(
            "现场event阈值来自验证工作点；interval=0.99、terminal=0.4 为现场推理默认解码阈值，"
            "未使用合成校准 thresholds 覆盖。"
        ),
    )
    write_json(package_dir / "field_thresholds.json", thresholds)
    write_json(package_dir / "thresholds.json", thresholds)
    candidate["final_package_path"] = package_path
    candidate["packed_epoch"] = int(epoch)
    candidate["parameter_count"] = parameter_count
    candidate["packed_source"] = source
    return package_path, int(epoch), parameter_count


def run_benchmark(candidates: list[dict[str, Any]], data: Path, output: Path, device: str) -> Path:
    benchmark_output = output / "benchmark.json"
    command = [
        sys.executable,
        str(BENCHMARK),
        "--data",
        str(data),
        "--device",
        device,
        "--output",
        str(benchmark_output),
    ]
    for candidate in candidates:
        command.extend(
            ["--checkpoint", f"{candidate['label']}={candidate['checkpoint_path']}"]
        )
    subprocess.run(command, cwd=BENCHMARK.parent, check=True)
    return benchmark_output


def run_best_exporter(
    winner: dict[str, Any],
    data: Path,
    export_dir: Path,
    benchmark_path: Path | None,
    output: Path,
) -> None:
    command = [
        sys.executable,
        str(EXPORTER),
        "--data",
        str(data),
        "--predictions",
        str(winner["prediction_path"]),
        "--checkpoint",
        str(winner["checkpoint_path"]),
        "--threshold",
        str(winner["threshold"]),
        "--output",
        str(export_dir),
        "--label",
        winner["label"],
    ]
    calibration = Path(winner["run_dir"]) / "calibration_iter8.json"
    if calibration.is_file():
        command.extend(["--calibration", str(calibration)])
    local_benchmark = Path(winner["run_dir"]) / "nn_benchmark.json"
    export_benchmark = local_benchmark if local_benchmark.is_file() else None
    if benchmark_path is not None:
        payload = read_json(benchmark_path)
        entry = next(
            (row for row in payload.get("models", []) if row.get("label") == winner["label"]),
            None,
        )
        if entry is not None:
            export_benchmark = output / "best_model_benchmark.json"
            write_json(
                export_benchmark,
                {
                    "median_ms": entry.get("median_ms"),
                    "p95_ms": entry.get("p95_ms"),
                    "parameters": entry.get("parameters"),
                    "weight_tensor_bytes": entry.get("weight_tensor_bytes"),
                    "inference_package_bytes": entry.get("inference_package_bytes"),
                    "checkpoint_bytes": entry.get("checkpoint_bytes"),
                    "scope": entry.get("scope"),
                },
            )
    if export_benchmark is not None:
        command.extend(["--benchmark", str(export_benchmark)])
    subprocess.run(command, cwd=EXPORTER.parent, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--benchmark-json", type=Path)
    parser.add_argument("--run-benchmark", action="store_true")
    parser.add_argument("--benchmark-device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.run_benchmark and args.benchmark_json is not None:
        parser.error("--run-benchmark 与 --benchmark-json 不能同时使用")
    args.data = args.data.expanduser().resolve()

    require_files(
        [
            args.data / "field_records.jsonl",
            args.data / "field_bags.json",
            args.data / "field_features.npy",
            EXPORTER,
            PLOT_TABLE,
            BENCHMARK,
            TEMPLATE,
            COVERAGE_SUMMARY,
            COVERAGE_CSV,
            COVERAGE_MD,
            COVERAGE_CASES,
            COVERAGE_CURVE,
            U8_FIXED060_CSV,
        ],
        "FieldNN finalizer 输入",
    )
    candidates = candidates_after_runs()
    evaluator = load_module(ITER6 / "evaluate_field.py", "iter12_delivery_field_evaluator")
    exporter = load_module(EXPORTER, "iter12_delivery_exporter")
    records = exporter.read_jsonl(args.data / "field_records.jsonl")
    bag_data = read_json(args.data / "field_bags.json")
    bags = bag_data["bags"] if isinstance(bag_data, dict) else bag_data
    records_by_index = {int(row["index"]): row for row in records}

    for candidate in candidates:
        evaluate_candidate(candidate, records, bags, records_by_index, evaluator, exporter)
    full_prediction_count = max(
        item["prediction_coverage"]["prediction_rows"] for item in candidates
    )
    data_summary = summarize_data(
        records, bags, records_by_index, evaluator, full_prediction_count
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    benchmark_path = args.benchmark_json.expanduser().resolve() if args.benchmark_json else None
    if benchmark_path is not None and not benchmark_path.is_file():
        raise FileNotFoundError(f"benchmark JSON 不存在：{benchmark_path}")
    if args.run_benchmark:
        benchmark_path = run_benchmark(candidates, args.data, OUTPUT, args.benchmark_device)

    u5 = next(item for item in candidates if item["label"] == "Iter5-U-Net")
    best_unet, best_unet_decision = choose_architecture_candidate(candidates, "unet", u5)
    best_transformer, best_transformer_decision = choose_architecture_candidate(candidates, "transformer", u5)
    best_unet["decision"] = best_unet_decision
    best_transformer["decision"] = best_transformer_decision
    overall_best = max([best_unet, best_transformer], key=selection_key)

    package_target, packed_epoch, parameter_count = package_winner(overall_best, OUTPUT)
    i11 = next(item for item in candidates if item["label"] == "Iter11-U-Net")
    coverage_summary, coverage_rows = coverage_workpoint_rows(i11, benchmark_path)
    coverage_inline = best_unet["label"] == "Iter11-U-Net"
    coverage_csv_output = None if coverage_inline else OUTPUT / "coverage_point.csv"
    coverage_table_output = None if coverage_inline else OUTPUT / "coverage_point_table.png"
    coverage_threshold_path = None
    if coverage_inline and overall_best["label"] == "Iter11-U-Net":
        coverage_threshold_path = OUTPUT / "best_model" / "thresholds_coverage_060.json"
        coverage_thresholds = read_json(OUTPUT / "best_model" / "thresholds.json")
        coverage_thresholds["calibration_split"] = "pre-existing Iter5 U-Net locked preset; not selected on I11 test"
        coverage_thresholds["thresholds"] = {"event": 0.6, "interval": 0.99, "terminal": 0.4}
        coverage_thresholds["policy"] = (
            "Supplementary I11 coverage point: pre-existing Iter5 U-Net event=0.60 preset; "
            "not selected or adjusted from test results. CPU FP32 batch-1 evaluation covers 138 labeled cache curves only."
        )
        coverage_thresholds["evaluation_scope"] = "138 annotated cache curves; not full 213-sweep inference"
        write_json(coverage_threshold_path, coverage_thresholds)
    for candidate in candidates:
        for population in ("nonwet_test", "wet_1500m_test"):
            candidate["metrics"][population] = enrich_metric_row(
                candidate["metrics"][population], candidate, benchmark_path
            )
        candidate["validation_row"] = enrich_metric_row(
            candidate["validation_row"], candidate, benchmark_path
        )

    main_specs = [
        (candidates[0], candidates[0]["label"]),
        (
            best_unet,
            best_unet["label"] + (" (bestU, historical fallback)" if best_unet_decision["fallback"] else " (bestU)"),
        ),
        (
            best_transformer,
            best_transformer["label"] + (" (bestT, historical fallback)" if best_transformer_decision["fallback"] else " (bestT)"),
        ),
    ]
    main_rows: list[dict[str, Any]] = []
    comparison_rows = []
    for candidate, display in main_specs:
        nonwet = dict(candidate["metrics"]["nonwet_test"])
        wet = dict(candidate["metrics"]["wet_1500m_test"])
        nonwet["model"] = display
        wet["model"] = display
        main_rows.extend([nonwet, wet])
        if coverage_inline and candidate["label"] == best_unet["label"]:
            main_rows.extend(coverage_rows)
        comparison_rows.append(
            {
                "display": display,
                "threshold": candidate["threshold"],
                "nonwet": nonwet,
                "wet": wet,
                "candidate": candidate,
            }
        )

    csv_fields = list(exporter.CSV_FIELDS)
    csv_write(OUTPUT / "performance.csv", main_rows, csv_fields)
    if not coverage_inline:
        csv_write(coverage_csv_output, coverage_rows, csv_fields)
    historical_rows = []
    for candidate in candidates:
        for population in ("nonwet_test", "wet_1500m_test"):
            row = dict(candidate["metrics"][population])
            checkpoint_note = (
                f"best.pt selected source={candidate.get('selected_source')}; this 50-epoch run kept a historical reference checkpoint. "
                if candidate.get("best_checkpoint_is_reference")
                else f"best.pt selected source={candidate.get('selected_source')}; this 50-epoch run kept the initial checkpoint. "
                if candidate.get("best_checkpoint_is_initial")
                else f"best.pt selected source={candidate.get('selected_source', 'historical model')}. "
            )
            row["notes"] = (
                f"{candidate['threshold_source']}; {checkpoint_note}prediction coverage: "
                f"{candidate['prediction_coverage']['predicted_rows']}/"
                f"{candidate['prediction_coverage']['prediction_rows']} predicted rows."
            )
            historical_rows.append(row)
    csv_write(OUTPUT / "historical_results.csv", historical_rows, csv_fields)
    comparison_notes = interpretation_notes(candidates)
    coverage_section = coverage_workpoint_markdown(
        coverage_summary,
        i11,
        coverage_inline,
        best_unet["label"],
        coverage_csv_output,
        coverage_table_output,
        coverage_threshold_path,
    )
    (OUTPUT / "comparison.md").write_text(
        comparison_markdown(comparison_rows, benchmark_path, comparison_notes)
        + "\n\n"
        + coverage_section
        + "\n",
        encoding="utf-8",
    )

    export_dir = OUTPUT / "best_exporter"
    run_best_exporter(overall_best, args.data, export_dir, benchmark_path, OUTPUT)

    comparison_table = "\n".join(
        "| {model} | {threshold:.6g} | {nonwet} | {macro:.4f} | {candidates}/{curves} | {error} | {wet} | {wet_candidates} |".format(
            model=markdown_link(row["display"], row["candidate"]["checkpoint_path"]),
            threshold=row["threshold"],
            nonwet=(
                f"{format_hits(row['nonwet'], 10)}/{format_hits(row['nonwet'], 20)}/"
                f"{format_hits(row['nonwet'], 30)}"
            ),
            macro=float(row["nonwet"]["group_macro_hit_at_20m"]),
            candidates=int(row["nonwet"].get("candidate_events_total") or 0),
            curves=int(row["nonwet"].get("curves") or 0),
            error=(
                f"{float(row['nonwet']['mean_matched_error_at_20m']):.2f}"
                if row["nonwet"].get("mean_matched_error_at_20m") is not None
                else "—"
            ),
            wet=format_hits(row["wet"], 20),
            wet_candidates=int(row["wet"].get("candidate_events_total") or 0),
        )
        for row in comparison_rows
    )
    template = TEMPLATE.read_text(encoding="utf-8")
    threshold_source = overall_best["threshold_source"]
    threshold_source_path = Path(threshold_source)
    threshold_source_display = (
        markdown_link(threshold_source_path.name, threshold_source_path)
        if threshold_source_path.is_file()
        else threshold_source
    )
    model_reference_lines = []
    for candidate in candidates:
        source_path = Path(candidate["threshold_source"])
        source_display = (
            markdown_link(source_path.name, source_path)
            if source_path.is_file()
            else candidate["threshold_source"]
        )
        tie_source = Path((candidate.get("validation_tie") or {}).get("source", ""))
        tie_display = (
            markdown_link(tie_source.name, tie_source)
            if tie_source.is_file()
            else markdown_link("saved predictions", candidate["prediction_path"])
        )
        reference_label = candidate["label"]
        if candidate.get("best_checkpoint_is_reference"):
            reference_label += " (best.pt is Iter8 reference)"
        elif candidate.get("best_checkpoint_is_initial"):
            reference_label += " (best.pt is initialization)"
        model_reference_lines.append(
            f"| {reference_label} | {markdown_link('checkpoint', candidate['checkpoint_path'])} | "
            f"{candidate['threshold']:.6g} | {source_display} | {tie_display} |"
        )
    decision_text = (
        f"bestU：{best_unet_decision['reason']} bestT：{best_transformer_decision['reason']} "
        f"当前最终模型为 {overall_best['label']}。比较优胜者按非湿保留组 group macro@20 m "
        f"（保留 12 位小数）、命中数、候选数、validation 候选数和 validation MIL loss 排序。"
    )
    card_values = {
        "{{MODEL_NAME}}": overall_best["label"],
        "{{ARCHITECTURE}}": "U-Net" if overall_best["architecture"] == "unet" else "Transformer",
        "{{EVENT_THRESHOLD}}": f"{overall_best['threshold']:.6g}",
        "{{THRESHOLD_SOURCE}}": threshold_source_display,
        "{{TRAINING_DESCRIPTION}}": training_description(overall_best),
        "{{ARCHITECTURE_DETAIL}}": architecture_description(overall_best["architecture"]),
        "{{DATA_SUMMARY}}": format_data_summary(data_summary, overall_best),
        "{{TRAINING_OBJECTIVE}}": training_objective(candidates),
        "{{EXPERIMENTS}}": candidate_run_markdown(candidates, overall_best),
        "{{RUNTIME_SUMMARY}}": runtime_summary(overall_best, benchmark_path, parameter_count),
        "{{MODEL_REFERENCES}}": "\n".join(model_reference_lines),
        "{{SELECTION_DECISION}}": decision_text,
        "{{INTERPRETATION_NOTES}}": comparison_notes,
        "{{COVERAGE_POINT}}": coverage_section,
        "{{NONWET_SUMMARY}}": format_summary(overall_best["metrics"]["nonwet_test"]),
        "{{WET_SUMMARY}}": format_summary(overall_best["metrics"]["wet_1500m_test"]),
        "{{COMPARISON_TABLE}}": comparison_table,
        "{{PACKAGE_PATH}}": markdown_link("inference.pt", package_target),
        "{{PACKAGE_BYTES}}": str(package_target.stat().st_size),
        "{{PACKED_EPOCH}}": str(packed_epoch),
        "{{PACKED_SOURCE}}": markdown_link(str(overall_best["packed_source"]), Path(overall_best["packed_source"])),
        "{{THRESHOLDS_PATH}}": (
            markdown_link("field_thresholds.json", OUTPUT / "best_model" / "field_thresholds.json")
            + "、"
            + markdown_link("thresholds.json", OUTPUT / "best_model" / "thresholds.json")
        ),
        "{{CASE_IMAGES}}": case_images(export_dir / "selected_cases.csv", export_dir),
        "{{DELIVERY_INDEX}}": markdown_link("model_index.json", OUTPUT / "model_index.json"),
        "{{COMPARISON_ARTIFACTS}}": "、".join(
            markdown_link(name, OUTPUT / filename)
            for name, filename in (
                ("comparison.md", "comparison.md"),
                ("performance.csv", "performance.csv"),
                ("performance_table.png", "performance_table.png"),
                ("historical_results.csv", "historical_results.csv"),
            )
        ) + (
            "、" + markdown_link("coverage_point.csv", coverage_csv_output)
            if coverage_csv_output is not None
            else ""
        ),
    }
    card = template
    for token, value in card_values.items():
        card = card.replace(token, value)
    (HERE / "MODEL_CARD_BEST.md").write_text(card, encoding="utf-8")
    (OUTPUT / "MODEL_CARD_BEST.md").write_text(card, encoding="utf-8")

    table_command = [
        sys.executable,
        str(PLOT_TABLE),
        "--input",
        str(OUTPUT / "performance.csv"),
        "--output",
        str(OUTPUT / "performance_table.png"),
    ]
    if benchmark_path is not None:
        table_command.extend(["--benchmark", str(benchmark_path)])
    subprocess.run(table_command, cwd=PLOT_TABLE.parent, check=True)
    if not coverage_inline:
        coverage_table_command = [
            sys.executable,
            str(PLOT_TABLE),
            "--input",
            str(coverage_csv_output),
            "--output",
            str(coverage_table_output),
            "--title",
            "I11 固定0.60 CPU缓存覆盖工作点（138条标注曲线）",
        ]
        if benchmark_path is not None:
            coverage_table_command.extend(["--benchmark", str(benchmark_path)])
        subprocess.run(coverage_table_command, cwd=PLOT_TABLE.parent, check=True)

    history_index = []
    for candidate in candidates:
        history_index.append(
            {
                "label": candidate["label"],
                "architecture": candidate["architecture"],
                "threshold": candidate["threshold"],
                "threshold_source": candidate["threshold_source"],
                "candidate_class": candidate.get("candidate_class"),
                "selected_source": candidate.get("selected_source"),
                "best_checkpoint_is_reference": candidate.get("best_checkpoint_is_reference", False),
                "best_checkpoint_is_initial": candidate.get("best_checkpoint_is_initial", False),
                "validation_tie": candidate.get("validation_tie"),
                "decision": candidate.get("decision"),
                "checkpoint": str(candidate["checkpoint_path"]),
                "predictions": str(candidate["prediction_path"]),
                "run_info": candidate.get("run_info"),
                "prediction_coverage": candidate["prediction_coverage"],
                "nonwet_test": candidate["metrics"]["nonwet_test"],
                "wet_1500m_test": candidate["metrics"]["wet_1500m_test"],
            }
        )
    write_json(
        OUTPUT / "model_index.json",
        {
            "schema": "fieldnn-final-delivery-1",
            "field_cache": str(args.data.resolve()),
            "data_summary": data_summary,
            "evaluator": str(ITER6 / "evaluate_field.py"),
            "field_metric_scope": "known positive positions, Iter6 source/report axis matching, single best sweep per bag",
            "selection_rule": {
                "eligibility": f"bestU/bestT must have a complete >= {MIN_COMPLETE_EPOCHS} epoch trajectory and must not select iter8_reference as best.pt. A selected step-0 initialization is explicitly identified as such. Historical U8/T8 are reference-only unless fallback triggers",
                "ranking": "group macro hit@20m rounded to 12 decimals, matched hits, fewer test candidates, fewer validation candidates, lower validation MIL loss, then newer experiment",
                "fallback": f"if the new winner loses at least {SIGNIFICANT_HIT_DROP} known hit@20m positions against U5 or has more than {MAX_CANDIDATE_MULTIPLE_OF_U5:.2f}x U5 test candidates, use the corresponding Iter8 reference and report why",
                "interpretation": "post-hoc comparison only; weights and thresholds remain validation-selected",
            },
            "best_unet": best_unet["label"],
            "best_transformer": best_transformer["label"],
            "best_unet_decision": best_unet_decision,
            "best_transformer_decision": best_transformer_decision,
            "overall_best": {
                "label": overall_best["label"],
                "architecture": overall_best["architecture"],
                "threshold": overall_best["threshold"],
                "threshold_source": overall_best["threshold_source"],
                "source_checkpoint": str(overall_best["checkpoint_path"]),
                "package": str(package_target),
                "packed_source": str(overall_best["packed_source"]),
                "packed_epoch": packed_epoch,
                "parameter_count": parameter_count,
                "field_threshold_config": str(OUTPUT / "best_model" / "field_thresholds.json"),
                "threshold_config": str(OUTPUT / "best_model" / "thresholds.json"),
                "nonwet_test": overall_best["metrics"]["nonwet_test"],
                "wet_1500m_test": overall_best["metrics"]["wet_1500m_test"],
            },
            "coverage_point_060": {
                "same_model": "Iter11-U-Net",
                "integrated_in_main_table": coverage_inline,
                "event_threshold": coverage_summary["workpoint"]["event_threshold"],
                "event_threshold_source": coverage_summary["workpoint"]["threshold_source"],
                "scope": coverage_summary["cpu_inference"],
                "fixed_threshold_populations": coverage_summary["fixed_threshold_populations"],
                "curve": str(COVERAGE_CURVE),
                "summary": str(COVERAGE_SUMMARY),
                "performance_csv": str(COVERAGE_CSV),
                "performance_markdown": str(COVERAGE_MD),
                "selected_cases": str(COVERAGE_CASES),
                "assets": str(COVERAGE_ASSETS),
                "supplementary_csv": str(coverage_csv_output) if coverage_csv_output is not None else None,
                "supplementary_table": str(coverage_table_output) if coverage_table_output is not None else None,
                "optional_package_thresholds": str(coverage_threshold_path) if coverage_threshold_path is not None else None,
                "candidate_semantics": "unmatched/unlabeled candidates are unverified; .60 threshold was not tuned on test",
            },
            "benchmark": str(benchmark_path) if benchmark_path is not None else None,
            "history": history_index,
            "limits": [
                "现场 unknown 位置未作为负例；未匹配候选保持未核验。",
                "完整现场 Accuracy、precision、recall、F1 不可计算。",
                "1500 m 湿接头 test 组单列，不并入非湿 test 排名。",
                "保留组在多轮开发中被重复使用；当前比较不是新盲测。",
                "Iter11 与 Iter12 的架构、学习率和监督权重设置不同，不能单独归因架构差异。",
            ],
        },
    )
    print(f"最终交付已生成：{OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
