"""Evaluate one FieldNN checkpoint on a synthetic validation or test split."""
from __future__ import annotations

import argparse
import itertools
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import DEFAULT_DATA, HERE, ShardDataset, load_config, write_json, frequency_grid
from metrics import decode_predictions, evaluate_predictions
from train import model_from_checkpoint


THRESHOLD_CANDIDATES = (0.2, 0.35, 0.5, 0.65, 0.8)
DEFAULT_THRESHOLDS = {"event": 0.5, "interval": 0.5, "terminal": 0.5}
TRUTH_SEMANTICS = "synthetic_geometric"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_data_and_checkpoint(root: Path, split: str, dataset: ShardDataset,
                                  checkpoint: dict[str, Any]) -> dict[str, Any]:
    data_cfg = load_config(root / "config.json")
    checkpoint_cfg = checkpoint["config"]
    for section in ("frequency", "response", "model"):
        if checkpoint_cfg.get(section) != data_cfg.get(section):
            raise ValueError(f"Checkpoint {section} config differs from dataset config")

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    expected_count = int(manifest["counts"][split])
    if len(dataset) != expected_count:
        raise ValueError(f"{split} metadata count differs from manifest: {len(dataset)} != {expected_count}")

    expected_frequency = frequency_grid(data_cfg)
    stored_frequency = np.load(root / "frequency.npy", mmap_mode="r")
    if stored_frequency.shape != expected_frequency.shape or not np.allclose(
        stored_frequency, expected_frequency, rtol=1e-12, atol=1e-9
    ):
        raise ValueError("Dataset frequency.npy differs from its configured preprocessing grid")

    response_cfg = data_cfg["response"]
    expected_distance = np.linspace(
        0.0, float(response_cfg["distance_max_m"]), int(response_cfg["points"]), dtype=np.float64
    )
    stored_distance = np.load(root / "distance.npy", mmap_mode="r")
    if stored_distance.shape != expected_distance.shape or not np.allclose(
        stored_distance, expected_distance, rtol=1e-12, atol=1e-9
    ):
        raise ValueError("Dataset distance.npy differs from its configured response grid")

    if not len(dataset):
        raise ValueError(f"Dataset split is empty: {split}")
    sample_x, sample_y, _ = dataset[0]
    expected_x_shape = (int(data_cfg["model"]["input_channels"]), int(response_cfg["points"]))
    expected_y_shape = (3, int(response_cfg["points"]))
    if sample_x.shape != expected_x_shape or sample_y.shape != expected_y_shape:
        raise ValueError(
            f"Dataset sample shapes differ from config: x={sample_x.shape}, y={sample_y.shape}"
        )
    return data_cfg


def _threshold_path(checkpoint_path: Path, split: str, requested: Path | None) -> tuple[Path | None, str]:
    if requested is not None:
        return requested, "explicit_threshold_file"
    default_path = checkpoint_path.parent / "thresholds.json"
    if default_path.exists():
        return default_path, "checkpoint_threshold_file"
    if split == "test":
        raise FileNotFoundError(
            f"Validation thresholds are required for test evaluation: {default_path}; "
            "run evaluate.py --split val --calibrate first"
        )
    return None, "default_thresholds"


def _load_threshold_file(path: Path, checkpoint: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("calibration_split") != "val":
        raise ValueError("Threshold file must record calibration_split='val'")
    if value.get("model_name") != checkpoint["model_name"]:
        raise ValueError("Threshold file model_name does not match this checkpoint")
    checkpoint_epoch = int(checkpoint["epoch"])
    if int(value.get("checkpoint_epoch", -1)) != checkpoint_epoch:
        raise ValueError("Threshold file checkpoint_epoch does not match this checkpoint")
    thresholds = {**DEFAULT_THRESHOLDS, **{key: float(v) for key, v in value["thresholds"].items()}}
    source = {
        "kind": "explicit_threshold_file" if path else "checkpoint_threshold_file",
        "path": str(path.resolve()),
        "calibration_split": "val",
        "calibration_sample_count": int(value["sample_count"]),
        "calibration_split_sample_count": int(value.get("split_sample_count", value["sample_count"])),
        "complete_split_calibration": bool(value.get("complete_split_calibration", True)),
        "checkpoint_epoch": checkpoint_epoch,
        "model_name": checkpoint["model_name"],
    }
    return thresholds, source


def _truth_for_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_positions_m": metadata["event_positions_m"],
        "defect_intervals_m": metadata["defect_intervals_m"],
        "terminal_position_m": metadata["terminal_position_m"],
    }


def _collect_predictions(model: torch.nn.Module, dataset: ShardDataset, cfg: dict[str, Any],
                         device: torch.device, batch_size: int, max_batches: int) -> tuple[list[np.ndarray], list[dict[str, Any]], list[dict[str, Any]], list[int], int, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    batches = loader if max_batches == 0 else itertools.islice(loader, max_batches)
    probabilities_by_sample: list[np.ndarray] = []
    truth: list[dict[str, Any]] = []
    metadata_by_sample: list[dict[str, Any]] = []
    indices_by_sample: list[int] = []
    batch_count = 0
    started = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for x, _, indices in batches:
            logits = model(x.to(device))
            expected_shape = (len(indices), 3, int(cfg["response"]["points"]))
            if tuple(logits.shape) != expected_shape:
                raise ValueError(f"Model output shape differs from response grid: {tuple(logits.shape)}")
            probabilities = torch.sigmoid(logits).float().cpu().numpy()
            for offset, index in enumerate(indices.tolist()):
                metadata = dataset.metadata[int(index)]
                if metadata.get("split") != dataset.shards[0]["split"]:
                    raise ValueError("Shard metadata split does not match the requested dataset split")
                probabilities_by_sample.append(probabilities[offset])
                truth.append(_truth_for_metadata(metadata))
                metadata_by_sample.append(metadata)
                indices_by_sample.append(int(index))
            batch_count += 1
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return probabilities_by_sample, truth, metadata_by_sample, indices_by_sample, batch_count, time.perf_counter() - started


def _decode_all(probabilities: list[np.ndarray], cfg: dict[str, Any],
                thresholds: dict[str, float]) -> list[dict[str, Any]]:
    return [decode_predictions(sample, cfg, thresholds) for sample in probabilities]


def _calibrate_thresholds(probabilities: list[np.ndarray], truth: list[dict[str, Any]],
                          cfg: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    objectives = {
        "event": ("event_f1_at_5m", "5", "event_metrics_by_tolerance_m"),
        "interval": ("interval_iou_0p5_f1", "interval_metrics", None),
        "terminal": ("terminal_f1_at_configured_tolerance", "terminal_metrics", None),
    }
    scores_by_head = {head: [] for head in objectives}
    for threshold in THRESHOLD_CANDIDATES:
        # Each decoder and objective reads only its own output track, so one
        # full-split decode/evaluation supplies all three independent scores.
        candidate_thresholds = {head: threshold for head in objectives}
        decoded = _decode_all(probabilities, cfg, candidate_thresholds)
        report = evaluate_predictions(decoded, truth, cfg)
        for head, (_, key, subkey) in objectives.items():
            metric = report[subkey][key] if subkey else report[key]
            scores_by_head[head].append({"threshold": threshold, "f1": float(metric["f1"])})

    selected = dict(DEFAULT_THRESHOLDS)
    details: dict[str, Any] = {}
    for head, (objective_name, _, _) in objectives.items():
        candidate_scores = scores_by_head[head]
        best = max(candidate_scores, key=lambda item: (item["f1"], item["threshold"]))
        selected[head] = float(best["threshold"])
        details[head] = {
            "objective": objective_name,
            "candidate_scores": candidate_scores,
            "selected_threshold": selected[head],
        }
    return selected, details


def _prediction_row(metadata: dict[str, Any], index: int, split: str,
                    model_name: str, checkpoint_epoch: int,
                    thresholds: dict[str, float], threshold_source: dict[str, Any],
                    prediction: dict[str, Any], truth: dict[str, Any]) -> dict[str, Any]:
    return {
        "split": split,
        "sample_index": int(index),
        "sample_id": int(metadata["sample_id"]),
        "topology_id": metadata.get("topology_id"),
        "case_id": metadata.get("case_id"),
        "case_title": metadata.get("case_title"),
        "length_m": float(metadata["length_m"]),
        "model_name": model_name,
        "checkpoint_epoch": checkpoint_epoch,
        "epoch_number": checkpoint_epoch + 1,
        "truth_semantics": TRUTH_SEMANTICS,
        "thresholds": thresholds,
        "threshold_source": threshold_source,
        "truth": truth,
        "prediction": prediction,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.data).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select --device cpu explicitly for a CPU check")

    model, checkpoint = model_from_checkpoint(checkpoint_path, device)
    checkpoint_epoch = int(checkpoint["epoch"])
    model_name = str(checkpoint["model_name"])
    dataset = ShardDataset(root, args.split)
    cfg = _validate_data_and_checkpoint(root, args.split, dataset, checkpoint)

    if args.calibrate:
        if args.split != "val":
            raise ValueError("--calibrate is allowed only with --split val")
        if args.thresholds is not None:
            raise ValueError("Use either --calibrate or --thresholds, not both")
        threshold_file = None
        threshold_source_kind = "calibrated_on_current_validation_split"
    else:
        threshold_file, threshold_source_kind = _threshold_path(checkpoint_path, args.split, args.thresholds)

    started = time.perf_counter()
    probabilities, truth, metadata, sample_indices, batch_count, inference_seconds = _collect_predictions(
        model, dataset, cfg, device, args.batch_size, args.max_batches
    )
    if not probabilities:
        raise ValueError(f"No samples were evaluated from split {args.split}")

    calibration_details = None
    if args.calibrate:
        thresholds, calibration_details = _calibrate_thresholds(probabilities, truth, cfg)
        threshold_source = {
            "kind": threshold_source_kind,
            "calibration_split": "val",
            "calibration_sample_count": len(truth),
            "calibration_split_sample_count": len(dataset),
            "complete_split_calibration": len(truth) == len(dataset),
            "checkpoint_epoch": checkpoint_epoch,
            "model_name": model_name,
        }
        threshold_record = {
            "calibration_split": "val",
            "data_root": str(root),
            "thresholds": thresholds,
            "sample_count": len(truth),
            "split_sample_count": len(dataset),
            "complete_split_calibration": len(truth) == len(dataset),
            "checkpoint_epoch": checkpoint_epoch,
            "epoch_number": checkpoint_epoch + 1,
            "model_name": model_name,
            "objectives": calibration_details,
            "calibrated_at_utc": _utc_now(),
        }
        threshold_file = checkpoint_path.parent / "thresholds.json"
        write_json(threshold_file, threshold_record)
        threshold_source["path"] = str(threshold_file.resolve())
    elif threshold_file is None:
        thresholds = dict(DEFAULT_THRESHOLDS)
        threshold_source = {"kind": "default_thresholds", "thresholds": dict(thresholds)}
    else:
        thresholds, threshold_source = _load_threshold_file(threshold_file, checkpoint)
        threshold_source["kind"] = threshold_source_kind

    predictions = _decode_all(probabilities, cfg, thresholds)
    metrics = evaluate_predictions(predictions, truth, cfg)
    full_split = len(truth) == len(dataset)
    output_dir = Path(args.output).expanduser() if args.output else checkpoint_path.parent / f"evaluation_{args.split}"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8", newline="\n") as stream:
        for index, meta, prediction, target in zip(sample_indices, metadata, predictions, truth):
            row = _prediction_row(
                meta, index, args.split, model_name, checkpoint_epoch,
                thresholds, threshold_source, prediction, target,
            )
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")

    report = {
        "schema": "fieldnn-evaluation-1",
        "checkpoint": str(checkpoint_path),
        "model_name": model_name,
        "checkpoint_epoch": checkpoint_epoch,
        "epoch_number": checkpoint_epoch + 1,
        "split": args.split,
        "data_root": str(root),
        "truth_semantics": TRUTH_SEMANTICS,
        "truth_note": "Synthetic geometric truth: generated event positions, changed-RLGC intervals, and configured open-end position; not measured-field ground truth.",
        "preprocessing_config": {"frequency": cfg["frequency"], "response": cfg["response"], "model": cfg["model"]},
        "thresholds": thresholds,
        "threshold_source": threshold_source,
        "threshold_calibration": calibration_details,
        "counts": {
            "split_sample_count": len(dataset),
            "evaluated_sample_count": len(truth),
            "batch_count": batch_count,
            "max_batches": args.max_batches,
            "full_split_evaluation": full_split,
            "evaluation_scope": "complete_split" if full_split else "partial_batch_check",
        },
        "timing": {
            "inference_seconds": inference_seconds,
            "total_seconds_before_report_write": time.perf_counter() - started,
            "generated_at_utc": _utc_now(),
        },
        "metrics": metrics,
        "predictions_jsonl": str(predictions_path),
    }
    report_path = output_dir / "evaluation.json"
    report["evaluation_report"] = str(report_path)
    write_json(report_path, report)
    print(json.dumps({"model": model_name, "split": args.split, "samples": len(truth),
                      "event_f1_at_5m": metrics["event_metrics_by_tolerance_m"]["5"]["f1"],
                      "interval_f1_iou_0p5": metrics["interval_metrics"]["f1"],
                      "terminal_f1": metrics["terminal_metrics"]["f1"],
                      "report": str(report_path)}, ensure_ascii=False, allow_nan=False))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--calibrate", action="store_true", help="Select thresholds on the validation split")
    parser.add_argument("--thresholds", type=Path, help="Validation-calibrated thresholds JSON")
    parser.add_argument("--output", type=Path, help="Output directory for evaluation.json and predictions.jsonl")
    parser.add_argument("--max-batches", type=int, default=0, help="Small pipeline check only; 0 evaluates the full split")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.max_batches < 0:
        parser.error("--max-batches must be zero or positive")
    if args.calibrate and args.split != "val":
        parser.error("--calibrate is only allowed with --split val")
    if args.calibrate and args.thresholds is not None:
        parser.error("Use either --calibrate or --thresholds, not both")
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
