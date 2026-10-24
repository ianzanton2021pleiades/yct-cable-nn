"""Evaluate v3 checkpoints on synthetic val/test with validation-calibrated thresholds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_v3 import ROOT, V3Dataset, worker_init
from metrics import decode_predictions, evaluate_predictions
from train_v3 import model_from_checkpoint

CANDIDATES = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)
DEFAULTS = {"event": 0.5, "interval": 0.5, "terminal": 0.5}


def truth(meta):
    return {"event_positions_m": meta["event_positions_m"],
            "defect_intervals_m": meta["defect_intervals_m"],
            "terminal_position_m": meta["terminal_position_m"]}


def collect(model, dataset, cfg, device, batch_size, workers):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                        worker_init_fn=worker_init, pin_memory=device.type == "cuda")
    probs, truths = [], []
    model.eval()
    with torch.inference_mode():
        for x, _, indices in loader:
            p = model(x.to(device)).sigmoid().float().cpu().numpy()
            for offset, index in enumerate(indices.tolist()):
                probs.append(p[offset])
                truths.append(truth(dataset.metadata[index]))
    return probs, truths


def calibrate(probs, truths, cfg):
    scores = {k: [] for k in DEFAULTS}
    for threshold in CANDIDATES:
        decoded = [decode_predictions(p, cfg, {k: threshold for k in DEFAULTS}) for p in probs]
        report = evaluate_predictions(decoded, truths, cfg)
        scores["event"].append((threshold, report["event_metrics_by_tolerance_m"]["5"]["f1"]))
        scores["interval"].append((threshold, report["interval_metrics"]["f1"]))
        scores["terminal"].append((threshold, report["terminal_metrics"]["f1"]))
    selected = {k: max(v, key=lambda t: (t[1], t[0]))[0] for k, v in scores.items()}
    return selected, {k: [{"threshold": a, "f1": b} for a, b in v] for k, v in scores.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    model, state = model_from_checkpoint(args.checkpoint, device)
    cfg = state["config"]
    dataset = V3Dataset(args.data, args.split, train=False)
    probs, truths = collect(model, dataset, cfg, device, args.batch_size, args.workers)
    if args.calibrate:
        thresholds, details = calibrate(probs, truths, cfg)
        threshold_path = Path(args.checkpoint).parent / "thresholds.json"
        threshold_path.write_text(json.dumps({
            "calibration_split": "val", "thresholds": thresholds,
            "sample_count": len(truths), "model_name": state["model_name"],
            "checkpoint_epoch": int(state["epoch"]), "objectives": details,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print("thresholds", thresholds, flush=True)
    else:
        threshold_path = Path(args.checkpoint).parent / "thresholds.json"
        thresholds = json.loads(threshold_path.read_text(encoding="utf-8"))["thresholds"]
    decoded = [decode_predictions(p, cfg, thresholds) for p in probs]
    report = evaluate_predictions(decoded, truths, cfg)
    out = Path(args.output) if args.output else Path(args.checkpoint).parent / f"evaluation_{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    result = {"model_name": state["model_name"], "checkpoint_epoch": int(state["epoch"]),
              "split": args.split, "thresholds": thresholds, "metrics": report,
              "evaluated": len(truths)}
    (out / "evaluation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"model": state["model_name"], "split": args.split,
                      "event_f1_5m": report["event_metrics_by_tolerance_m"]["5"]["f1"],
                      "interval_f1": report["interval_metrics"]["f1"],
                      "terminal_f1": report["terminal_metrics"]["f1"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
