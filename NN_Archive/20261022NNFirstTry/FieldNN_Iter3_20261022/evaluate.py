"""Validation-only precision-first calibration and frozen held-out evaluation."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import build_model
from decode import decode_predictions
HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent / "FieldNN_20261022"))
from metrics import evaluate_predictions

CANDIDATES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99)


def collect(model, dataset, device, batch_size):
    probabilities, truths = [], []
    ignored, total = np.zeros(3, dtype=np.int64), 0
    with torch.inference_mode():
        for x, targets, indices in tqdm(DataLoader(dataset, batch_size=batch_size, num_workers=0), desc="predict"):
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                out = model(x.to(device)).float().sigmoid().cpu().numpy()
            mask = targets.numpy() < 0
            ignored += mask.sum(axis=(0, 2))
            total += mask.shape[0] * mask.shape[2]
            out[mask] = 0.0
            probabilities.extend(out)
            for i in indices.tolist():
                m = dataset.metadata[i]
                truths.append({**{k: m[k] for k in ("event_positions_m", "defect_intervals_m", "terminal_position_m")}, "sample_kind": m.get("sample_kind")})
    return probabilities, truths, {"ignored_fraction_by_head": dict(zip(("event", "interval", "terminal"), (ignored / max(1, total)).tolist())), "policy": "Predictions in undefined synthetic target windows are excluded; measured-field inference remains full-grid."}


def precision_key(metric, target):
    p, r = metric.get("precision") or 0.0, metric.get("recall") or 0.0
    fhalf = 1.25 * p * r / (0.25 * p + r) if p + r else 0.0
    return (int(p >= target and r > 0), r if p >= target else fhalf, p)


def get_heads(report):
    return {"event": report["event_metrics_by_tolerance_m"]["5"],
            "interval": report["interval_metrics"], "terminal": report["terminal_metrics"]}


def extra_stats(decoded, truths):
    healthy = [i for i, t in enumerate(truths) if t.get("sample_kind") == "healthy_no_event"]
    return {"zero_event_curve_count": len(healthy),
            "false_events_per_zero_event_curve": sum(len(decoded[i]["events"]) for i in healthy) / len(healthy) if healthy else None,
            "events_per_curve_median": float(np.median([len(v["events"]) for v in decoded])),
            "post_terminal_event_count": sum(sum(e["distance_m"] > t["terminal_position_m"] + 10 for e in v["events"])
                                             for v, t in zip(decoded, truths) if t["terminal_position_m"] is not None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--precision-target", type=float, default=0.85)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.calibrate and args.split != "val":
        ap.error("Threshold calibration requires --split val")
    torch.set_num_threads(1)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    from data import ShardDataset
    dataset = ShardDataset(args.data, args.split)
    model = build_model(checkpoint["model_name"], **cfg["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    probs, truths, coverage = collect(model, dataset, device, args.batch_size)
    threshold_path = args.checkpoint.parent / "thresholds.json"
    if args.calibrate:
        tables = {k: [] for k in ("event", "interval", "terminal")}
        for t in tqdm(CANDIDATES, desc="precision calibration"):
            decoded = [decode_predictions(p, cfg, {k: t for k in tables}) for p in probs]
            heads = get_heads(evaluate_predictions(decoded, truths, cfg))
            for k in tables:
                tables[k].append({"threshold": t, **heads[k]})
        selected = {k: max(rows, key=lambda r: (*precision_key(r, args.precision_target), r["threshold"]))["threshold"]
                    for k, rows in tables.items()}
        result = {"calibration_split": "val", "model_name": checkpoint["model_name"],
                  "checkpoint_epoch": checkpoint["epoch"], "thresholds": selected,
                  "precision_target": args.precision_target, "supervision_coverage": coverage,
                  "policy": "maximize recall if precision reaches target; otherwise maximize F0.5",
                  "curves": tables}
        threshold_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        calibration = json.loads(threshold_path.read_text(encoding="utf-8"))
        if calibration["checkpoint_epoch"] != checkpoint["epoch"] or calibration["model_name"] != checkpoint["model_name"]:
            raise ValueError("Thresholds must be calibrated on this checkpoint")
        selected = calibration["thresholds"]
    decoded = [decode_predictions(p, cfg, selected) for p in probs]
    report = evaluate_predictions(decoded, truths, cfg)
    out = args.output or args.checkpoint.parent / f"evaluation_{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    result = {"model_name": checkpoint["model_name"], "checkpoint_epoch": checkpoint["epoch"],
              "split": args.split, "thresholds": selected, "metrics": report,
              "false_positive_summary": extra_stats(decoded, truths), "evaluated": len(truths), "supervision_coverage": coverage,
              "label_axis": "nominal-velocity response distance; physical geometry retained separately"}
    (out / "evaluation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"split": args.split, "heads": get_heads(report), "false_positive_summary": result["false_positive_summary"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()



