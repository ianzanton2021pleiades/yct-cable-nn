"""CPU-only, validation-selected fusion of saved Iter8 field candidates.

This operates on the decoded candidate subsets already saved at confidence >= 0.6.
It does not rerun models or recover peaks omitted by the saved decoder.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ITER6_EVALUATOR = HERE.parent / "FieldNN_Iter6_20261022" / "evaluate_field.py"
VAL_THRESHOLD_GRID = tuple(index / 100.0 for index in range(60, 100))
MATCH_RADII_M = (10.0, 20.0)
UNMATCHED_FACTORS = (0.5, 0.75, 1.0)
U8_SOLO_THRESHOLD = 0.94


def load_evaluator():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("iter6_field_fusion_evaluator", ITER6_EVALUATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def filter_events(row: dict[str, Any], threshold: float) -> dict[str, Any]:
    result = copy.deepcopy(row)
    if result.get("status") == "predicted":
        result["prediction"]["events"] = [
            event for event in result["prediction"].get("events", [])
            if float(event["confidence"]) >= threshold
        ]
    return result


def fuse_events(
    primary_row: dict[str, Any], secondary_row: dict[str, Any],
    match_radius_m: float, unmatched_factor: float,
) -> dict[str, Any]:
    """Greedily pair nearest U/T events one-to-one; retain primary position for pairs."""
    result = copy.deepcopy(primary_row)
    if result.get("status") != "predicted":
        return result

    primary = (primary_row.get("prediction") or {}).get("events", [])
    secondary = (secondary_row.get("prediction") or {}).get("events", [])
    edges = sorted(
        (abs(float(u["distance_m"]) - float(t["distance_m"])), ui, ti)
        for ui, u in enumerate(primary)
        for ti, t in enumerate(secondary)
        if abs(float(u["distance_m"]) - float(t["distance_m"])) <= match_radius_m
    )
    used_u: set[int] = set()
    used_t: set[int] = set()
    fused = []
    for _, ui, ti in edges:
        if ui in used_u or ti in used_t:
            continue
        used_u.add(ui)
        used_t.add(ti)
        p_u = float(primary[ui]["confidence"])
        p_t = float(secondary[ti]["confidence"])
        fused.append({
            "distance_m": float(primary[ui]["distance_m"]),
            "confidence": max(p_u * unmatched_factor, math.sqrt(p_u * p_t)),
            "fusion": "matched_geometric_mean_with_primary_floor",
        })

    for ui, event in enumerate(primary):
        if ui not in used_u:
            fused.append({
                "distance_m": float(event["distance_m"]),
                "confidence": float(event["confidence"]) * unmatched_factor,
                "fusion": "unmatched_primary_penalty",
            })
    for ti, event in enumerate(secondary):
        if ti not in used_t:
            fused.append({
                "distance_m": float(event["distance_m"]),
                "confidence": float(event["confidence"]) * unmatched_factor,
                "fusion": "unmatched_secondary_penalty",
            })
    fused.sort(key=lambda event: (event["distance_m"], -event["confidence"]))
    result["prediction"]["events"] = fused
    return result


def prediction_density(indices: set[int], predictions: dict[int, dict[str, Any]], evaluator) -> dict[str, Any]:
    predicted = [predictions[index] for index in indices if index in predictions
                 and predictions[index].get("status") == "predicted"]
    candidate_count = sum(len(evaluator.predicted_events(row)) for row in predicted)
    return {
        "applicable_sweeps": len(indices),
        "predicted_sweeps": len(predicted),
        "event_candidates_unverified": candidate_count,
        "candidates_per_predicted_sweep": candidate_count / len(predicted) if predicted else None,
    }


def metric_summary(
    bags: list[dict[str, Any]], records_by_index: dict[int, dict[str, Any]],
    predictions: dict[int, dict[str, Any]], indices: set[int], evaluator,
) -> dict[str, Any]:
    report = evaluator.positive_summary(bags, records_by_index, predictions, indices)
    result = {}
    for tolerance in (10, 20):
        metric = report["independent_group_macro"][str(tolerance)]
        result[str(tolerance)] = {
            "independent_groups": metric["independent_groups"],
            "matched_positions": metric["matched_positions"],
            "known_positive_positions": metric["unique_positions"],
            "group_macro_hit_rate": metric["macro_hit_rate"],
            "micro_hit_rate": metric["micro_hit_rate"],
            "mean_candidates_per_group_best_sweep": metric[
                "mean_unverified_candidates_per_group_best_sweep"
            ],
        }
    return {"hit_metrics": result, "density": prediction_density(indices, predictions, evaluator)}


def with_threshold(
    base: dict[int, dict[str, Any]], indices: set[int], threshold: float,
) -> dict[int, dict[str, Any]]:
    return {index: filter_events(base[index], threshold) for index in indices if index in base}


def main() -> None:
    evaluator = load_evaluator()
    out_dir = HERE / "output" / "fusion_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = load_json(HERE / "output" / "iter8" / "iteration8_summary.json")
    density_report = load_json(HERE / "output" / "iter8" / "density_analysis.json")
    data_root = Path(summary["data"])
    records = evaluator.read_jsonl(data_root / "field_records.jsonl")
    bags_obj = load_json(data_root / "field_bags.json")
    bags = bags_obj["bags"] if isinstance(bags_obj, dict) else bags_obj
    records_by_index = {int(record["index"]): record for record in records}

    prediction_paths = {
        "U8": Path(summary["models"]["unet"]["fixed_field_predictions"]),
        "T8": Path(summary["models"]["transformer"]["fixed_field_predictions"]),
        "U5": Path(summary["models"]["unet"]["baseline5_predictions"]),
        "T5": Path(summary["models"]["transformer"]["baseline5_predictions"]),
    }
    predictions = {}
    for label, path in prediction_paths.items():
        keyed = evaluator.load_prediction_files([f"{label}={path}"])[label]
        predictions[label] = evaluator.prediction_by_index(records, keyed)

    split_indices = {
        split: {int(record["index"]) for record in records if record["split"] == split}
        for split in ("val", "test")
    }
    common = {
        split: {
            index for index in split_indices[split]
            if all(index in predictions[label] and predictions[label][index].get("status") == "predicted"
                   for label in ("U8", "T8", "U5", "T5"))
        }
        for split in ("val", "test")
    }
    bags_by_split = {
        split: evaluator.bag_rows(split, bags, records_by_index)
        for split in ("val", "test")
    }

    baseline_density = prediction_density(common["val"], predictions["U5"], evaluator)
    density_cap = baseline_density["candidates_per_predicted_sweep"]
    val_u8 = with_threshold(predictions["U8"], common["val"], U8_SOLO_THRESHOLD)
    val_u8_metrics = metric_summary(
        bags_by_split["val"], records_by_index, val_u8, common["val"], evaluator
    )

    grid = []
    for secondary in ("T8", "T5"):
        for match_radius in MATCH_RADII_M:
            for unmatched_factor in UNMATCHED_FACTORS:
                merged = {
                    index: fuse_events(
                        predictions["U8"][index], predictions[secondary][index],
                        match_radius, unmatched_factor,
                    )
                    for index in common["val"]
                }
                for threshold in VAL_THRESHOLD_GRID:
                    filtered = with_threshold(merged, common["val"], threshold)
                    metric = metric_summary(
                        bags_by_split["val"], records_by_index, filtered, common["val"], evaluator
                    )
                    density = metric["density"]
                    density_value = density["candidates_per_predicted_sweep"]
                    grid.append({
                        "secondary_model": secondary,
                        "match_radius_m": match_radius,
                        "unmatched_factor": unmatched_factor,
                        "threshold": round(threshold, 2),
                        "val_group_macro_hit20": metric["hit_metrics"]["20"]["group_macro_hit_rate"],
                        "val_micro_hit20": metric["hit_metrics"]["20"]["micro_hit_rate"],
                        "val_candidate_density": density_value,
                        "val_candidate_total": density["event_candidates_unverified"],
                        "density_feasible": density_value <= density_cap + 1e-12,
                    })

    feasible = [row for row in grid if row["density_feasible"]]
    chosen = max(feasible, key=lambda row: (
        -1.0 if row["val_group_macro_hit20"] is None else row["val_group_macro_hit20"],
        -row["val_candidate_total"],
        row["threshold"],
        -row["match_radius_m"],
        -row["unmatched_factor"],
        row["secondary_model"] == "T8",
    ))

    selected_secondary = chosen["secondary_model"]
    selected_match_radius = chosen["match_radius_m"]
    selected_unmatched_factor = chosen["unmatched_factor"]
    selected_threshold = chosen["threshold"]
    u8_test = with_threshold(predictions["U8"], common["test"], U8_SOLO_THRESHOLD)
    merged_test = {
        index: fuse_events(
            predictions["U8"][index], predictions[selected_secondary][index],
            selected_match_radius, selected_unmatched_factor,
        )
        for index in common["test"]
    }
    fused_test = with_threshold(merged_test, common["test"], selected_threshold)

    test_records = [record for record in records if record["split"] == "test"]
    wet_indices = {int(record["index"]) for record in test_records if record.get("known_lab")}
    other_indices = {int(record["index"]) for record in test_records if not record.get("known_lab")}
    test_bags = bags_by_split["test"]
    wet_bags = [bag for bag in test_bags if bag["known_lab"]]
    other_bags = [bag for bag in test_bags if not bag["known_lab"]]
    populations = {}
    for label, allowed, relevant_bags in (
        ("all_test", common["test"], test_bags),
        ("six_nonwet_groups", common["test"] & other_indices, other_bags),
        ("wet_1500m_group", common["test"] & wet_indices, wet_bags),
    ):
        populations[label] = {
            "u8_alone_threshold_0_94": metric_summary(
                relevant_bags, records_by_index, u8_test, allowed, evaluator
            ),
            "validation_selected_fusion": metric_summary(
                relevant_bags, records_by_index, fused_test, allowed, evaluator
            ),
        }

    existing_u8_threshold = density_report["validation_selection"]["unet"]["selected_threshold"]
    result = {
        "schema": "fieldnn-iter8-candidate-fusion-1",
        "scope": "CPU analysis of existing decoded field candidates; no inference or model changes",
        "candidate_subset_limit": (
            "Inputs contain only events saved at confidence >=0.6. This is candidate-level fusion, "
            "not a full probability average or a complete NN ensemble; omitted peaks cannot be recovered."
        ),
        "prediction_files": {label: str(path) for label, path in prediction_paths.items()},
        "field_data": str(data_root),
        "fixed_group_split": {
            "val_records": len(split_indices["val"]),
            "common_predicted_val_sweeps": len(common["val"]),
            "test_records": len(split_indices["test"]),
            "common_predicted_test_sweeps": len(common["test"]),
            "common_predicted_nonwet_test_sweeps": len(common["test"] & other_indices),
            "common_predicted_wet_test_sweeps": len(common["test"] & wet_indices),
            "nonwet_independent_groups": len({bag["group"] for bag in other_bags}),
            "wet_independent_groups": len({bag["group"] for bag in wet_bags}),
        },
        "selection": {
            "selection_split": "val only",
            "objective": "maximum evaluator independent-group macro hit@20m",
            "density_cap": {
                "source": "Iter5 U-Net field_e06 at locked confidence 0.6",
                **baseline_density,
            },
            "candidate_grid": {
                "primary": "Iter8 U-Net fixed candidate output (U8)",
                "secondary_options": ["Iter8 Transformer fixed candidate output (T8)", "Iter5 Transformer field_e007 (T5)"],
                "match_radii_m": list(MATCH_RADII_M),
                "unmatched_factors": list(UNMATCHED_FACTORS),
                "thresholds": [round(value, 2) for value in VAL_THRESHOLD_GRID],
                "pairing": "greedy nearest one-to-one event pairs within the selected radius",
                "matched_event": "keep U8 position; score=max(pU8*factor, sqrt(pU8*pT))",
                "unmatched_event": "keep event position; score=its source confidence*factor",
                "tie_break": "fewer validation candidates, higher threshold, smaller merge radius, lower unmatched factor, then T8",
            },
            "iter8_existing_u8_threshold": existing_u8_threshold,
            "selected": chosen,
            "feasible_configurations": len(feasible),
            "u8_alone_threshold_0_94_val": val_u8_metrics,
            "selected_fusion_val": metric_summary(
                bags_by_split["val"], records_by_index,
                with_threshold(
                    {
                        index: fuse_events(
                            predictions["U8"][index], predictions[selected_secondary][index],
                            selected_match_radius, selected_unmatched_factor,
                        )
                        for index in common["val"]
                    }, common["val"], selected_threshold,
                ), common["val"], evaluator,
            ),
        },
        "test_after_selection": populations,
        "interpretation": (
            "Unmatched candidates are unverified under positive-only annotations. A higher hit rate is not precision, "
            "and the existing field curves have already participated in development."
        ),
        "validation_grid": grid,
    }

    (out_dir / "fusion_analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_markdown(result, out_dir / "fusion_analysis.md")
    print(json.dumps({
        "selected": chosen,
        "density_cap": density_cap,
        "u8_val_hit20": val_u8_metrics["hit_metrics"]["20"]["group_macro_hit_rate"],
        "fusion_test": populations,
        "outputs": [str(out_dir / "fusion_analysis.json"), str(out_dir / "fusion_analysis.md")],
    }, ensure_ascii=False, indent=2))


def write_markdown(result: dict[str, Any], path: Path) -> None:
    selection = result["selection"]
    chosen = selection["selected"]
    lines = [
        "# Iter8 现场候选融合快速分析",
        "",
        "仅使用已保存的现场候选进行 CPU 后处理。融合不是完整概率集成，不能恢复解码时未保存的低置信峰；未命中标签的候选仍属未核实。",
        "",
        f"验证集只选参数：U8 + {chosen['secondary_model']}，合并半径 {chosen['match_radius_m']} m，未匹配折扣 {chosen['unmatched_factor']}，融合阈值 {chosen['threshold']:.2f}。",
        f"Iter5 U-Net 验证候选密度上限为 {selection['density_cap']['candidates_per_predicted_sweep']:.3f}/条；融合验证密度 {chosen['val_candidate_density']:.3f}/条。",
        f"U8 单独阈值0.94的验证组平均 hit@20m 为 {selection['u8_alone_threshold_0_94_val']['hit_metrics']['20']['group_macro_hit_rate']:.3f}；融合验证为 {chosen['val_group_macro_hit20']:.3f}。",
        "",
        "| 测试范围 | 扫频数 | 独立组数 | U8@0.94 hit@10 | 融合 hit@10 | U8@0.94 hit@20 | 融合 hit@20 | U8候选/条 | 融合候选/条 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    split_sweeps = result["fixed_group_split"]
    for label, display in (
        ("six_nonwet_groups", "6个非浸水测试组"),
        ("wet_1500m_group", "1500m浸水组"),
        ("all_test", "全部测试组"),
    ):
        row = result["test_after_selection"][label]
        u8 = row["u8_alone_threshold_0_94"]
        fusion = row["validation_selected_fusion"]
        sweeps = (split_sweeps["common_predicted_nonwet_test_sweeps"] if label == "six_nonwet_groups"
                  else split_sweeps["common_predicted_wet_test_sweeps"] if label == "wet_1500m_group"
                  else split_sweeps["common_predicted_test_sweeps"])
        groups = u8["hit_metrics"]["20"]["independent_groups"]
        lines.append(
            f"| {display} | {sweeps} | {groups} | "
            f"{u8['hit_metrics']['10']['group_macro_hit_rate']:.3f} | {fusion['hit_metrics']['10']['group_macro_hit_rate']:.3f} | "
            f"{u8['hit_metrics']['20']['group_macro_hit_rate']:.3f} | {fusion['hit_metrics']['20']['group_macro_hit_rate']:.3f} | "
            f"{u8['density']['candidates_per_predicted_sweep']:.3f} | {fusion['density']['candidates_per_predicted_sweep']:.3f} |"
        )
    test_all = result["test_after_selection"]["all_test"]
    u8_hit = test_all["u8_alone_threshold_0_94"]["hit_metrics"]["20"]["group_macro_hit_rate"]
    fusion_hit = test_all["validation_selected_fusion"]["hit_metrics"]["20"]["group_macro_hit_rate"]
    verdict = "高于" if fusion_hit > u8_hit else "未高于" if fusion_hit == u8_hit else "低于"
    lines += [
        "",
        f"按全测试独立组 macro hit@20m，验证集选出的融合结果{verdict} U8 单独0.94工作点（{fusion_hit:.3f} 对 {u8_hit:.3f}）。这只是已知正位置覆盖指标，不能据此称为precision或真实现场性能提升。",
        "",
        "## 复现",
        "",
        "```powershell",
        "conda run --no-capture-output -n gpushare_cu124 python `",
        f'  "{(HERE / "analyze_candidate_fusion.py").as_posix()}"',
        "```",
        "",
        "完整候选网格及各阈值的验证选择指标见同目录 `fusion_analysis.json`。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
