"""Evaluation of a PINN checkpoint on the synthetic test set, injected hybrid test set and raw field test set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
CMDC = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_Hybrid_20261022"
CODEX = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_20261022"
for path in (HERE, CMDC, CODEX):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data_pinn import PinnDataset  # noqa: E402
from decode import decode_predictions  # noqa: E402
from eval_utils import calibration_metrics, field_metrics, hybrid_metrics  # noqa: E402
from hybrid_data import CalibrationDataset, FieldDataset, HybridDataset  # noqa: E402
from metrics import evaluate_predictions  # noqa: E402
from models_pinn import build_model  # noqa: E402


FIELD_ROOT = Path(r"E:\CableNNDataSet\Deepseek_CMDC\FieldNN_Hybrid_Long_20261022")
SYNTHETIC = Path(r"E:\CableNNDataSet\CodeX\FieldNN_Iter4_20261022")
THRESHOLDS = {"event": 0.5, "interval": 0.95, "terminal": 0.6}


def collect(model, dataset, device, batch_size=64):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    probabilities, indices = [], []
    with torch.inference_mode():
        for batch in loader:
            inputs, item = (batch[0], batch[1]) if len(batch) == 2 else (batch[0], batch[2])
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                output = model(inputs.to(device, non_blocking=device.type == "cuda"))
            probabilities.extend(output[:, :3].float().sigmoid().cpu().numpy())
            indices.extend(int(value) for value in item.tolist())
    return probabilities, indices


def synthetic_block(model, device, view: str, config) -> dict[str, Any]:
    dataset = PinnDataset(SYNTHETIC, "test", train=False, view=view)
    probabilities, indices = collect(model, dataset, device)
    truth = []
    for index in indices:
        metadata = dataset.metadata[index]
        truth.append({
            "event_positions_m": metadata.get("event_positions_m", []),
            "defect_intervals_m": metadata.get("defect_intervals_m", []),
            "terminal_position_m": metadata.get("terminal_position_m"),
            "sample_kind": metadata.get("sample_kind"),
        })
    report = evaluate_predictions(
        [decode_predictions(item, config, THRESHOLDS) for item in probabilities], truth, config
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--synthetic-only", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    model = build_model(
        state["model_name"], input_channels=8, width=32, depth=3,
        dropout=config["model"].get("dropout", 0.15),
    ).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    result: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "epoch": int(state.get("epoch", -1)),
        "thresholds": THRESHOLDS,
        "synthetic_test": {},
    }
    for view in ("clean", "mixed"):
        result["synthetic_test"][view] = synthetic_block(model, device, view, config)
        f1 = result["synthetic_test"][view]["event_metrics_by_tolerance_m"]["5"]["f1"]
        print(f"synthetic {view}: F1@5m {f1:.4f}", flush=True)

    from losses_pinn import compute_losses, total_loss
    from torch.utils.data import Subset
    probe_root = PinnDataset(SYNTHETIC, "train", train=False, view="clean")
    probe = Subset(probe_root, list(range(0, len(probe_root), max(1, len(probe_root) // 300)))[:300])
    loader = DataLoader(probe, batch_size=64, shuffle=False, num_workers=0)
    probe_total = 0.0
    probe_seen = 0
    with torch.inference_mode():
        for batch in loader:
            inputs, targets = batch[0].to(device), batch[1].to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                output = model(inputs)
            value = total_loss(compute_losses(output[:, :3], targets), 0.0)
            probe_total += float(value) * inputs.shape[0]
            probe_seen += int(inputs.shape[0])
    result["train_clean_probe"] = {
        "curves": probe_seen,
        "supervision_loss": probe_total / max(probe_seen, 1),
    }
    print("train clean probe:", json.dumps(result["train_clean_probe"], ensure_ascii=False), flush=True)

    if args.synthetic_only:
        output = args.output or args.checkpoint.parent / "evaluation_synthetic.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0

    hybrid_test = HybridDataset(FIELD_ROOT, "test", augment=False)
    probabilities, indices = collect(model, hybrid_test, device)
    result["hybrid_test"] = hybrid_metrics(
        probabilities, [hybrid_test.records[i] for i in indices], config, THRESHOLDS
    )
    print("hybrid test:", json.dumps(result["hybrid_test"]["non_wet"], ensure_ascii=False), flush=True)

    field_test = FieldDataset(FIELD_ROOT, "test")
    probabilities, indices = collect(model, field_test, device)
    records = [field_test.records[i] for i in indices]
    result["field_test"] = field_metrics(probabilities, records, config, THRESHOLDS)
    print("field test:", json.dumps(result["field_test"]["non_wet"], ensure_ascii=False), flush=True)

    from eval_utils import WET_CASE, match_events
    predictions = [decode_predictions(item, config, THRESHOLDS) for item in probabilities]
    total_hits = 0
    total_predicted = 0
    all_hit_curves = 0
    any_hit_curves = 0
    annotated = 0
    per_curve = []
    matched_errors = []
    for prediction, record in zip(predictions, records):
        truth = [float(value) for value in (record.get("known_positions_nominal_m") or [])]
        if not truth:
            continue
        hits = match_events(prediction["events"], truth, 20.0)
        wet = str(record.get("case_group") or "") == WET_CASE
        per_curve.append({
            "case_group": record.get("case_group"),
            "sweep_name": record.get("sweep_name"),
            "wet": wet,
            "truth": truth,
            "hits": hits,
            "candidates": len(prediction["events"]),
            "terminal_m": (prediction.get("terminal") or {}).get("distance_m"),
            "known_length_m": record.get("known_length_m"),
        })
        if wet:
            continue
        annotated += 1
        total_hits += hits
        total_predicted += len(prediction["events"])
        all_hit_curves += int(hits == len(truth))
        any_hit_curves += int(hits > 0)
        for position in truth:
            errors = [abs(float(event["distance_m"]) - position) for event in prediction["events"]]
            nearest = min(errors) if errors else None
            if nearest is not None and nearest <= 20.0:
                matched_errors.append(nearest)
    per_curve.sort(key=lambda row: (-row["hits"] / max(len(row["truth"]), 1), row["candidates"]))
    terminal_errors = []
    for row in per_curve:
        if row["terminal_m"] is not None and row["known_length_m"]:
            terminal_errors.append(abs(float(row["terminal_m"]) - float(row["known_length_m"])))
    result["field_test"]["annotation_summary"] = {
        "annotated_curves": annotated,
        "curves_all_positions_hit": all_hit_curves,
        "curves_any_position_hit": any_hit_curves,
        "P_ann": total_hits / max(total_predicted, 1),
        "hits_total": total_hits,
        "predicted_total": total_predicted,
        "mae_matched_m": float(np.mean(matched_errors)) if matched_errors else None,
        "terminal_error_median_m": float(np.median(terminal_errors)) if terminal_errors else None,
        "terminal_within_50m": float(np.mean([value <= 50.0 for value in terminal_errors])) if terminal_errors else None,
        "terminal_within_100m": float(np.mean([value <= 100.0 for value in terminal_errors])) if terminal_errors else None,
        "terminal_evaluated": len(terminal_errors),
    }
    result["field_test"]["per_curve"] = per_curve
    print("annotation summary:", json.dumps(result["field_test"]["annotation_summary"], ensure_ascii=False),
          flush=True)

    calibration = CalibrationDataset(FIELD_ROOT, "test", config)
    probabilities, _ = collect(model, calibration, device)
    result["calibration_test"] = calibration_metrics(probabilities, config, THRESHOLDS)

    output = args.output or args.checkpoint.parent / "evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("written", output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
