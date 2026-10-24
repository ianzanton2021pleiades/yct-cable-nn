"""Evaluate a synthetic-pretrained checkpoint on the complete synthetic test set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
LEGACY = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_20261022"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(LEGACY) not in sys.path:
    sys.path.append(str(LEGACY))

from decode import decode_predictions  # noqa: E402
from hybrid_data import SyntheticWrapper  # noqa: E402
from metrics import evaluate_predictions  # noqa: E402
from models import build_model  # noqa: E402


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", type=Path, default=Path(r"E:\CableNNDataSet\CodeX\FieldNN_Iter4_20261022"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    torch.set_num_threads(1)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    model = build_model(checkpoint["model_name"], **cfg["model"]).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    dataset = SyntheticWrapper(args.synthetic, args.split, train=False, view="clean")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    probabilities, indices = [], []
    for batch in loader:
        inputs, item = batch[0], batch[2]
        logits = model(inputs.to(device, non_blocking=device.type == "cuda"))
        probabilities.extend(logits.float().sigmoid().cpu().numpy())
        indices.extend(int(value) for value in item.tolist())
    truths = []
    for index in indices:
        metadata = dataset.metadata[index]
        truths.append({
            "event_positions_m": metadata.get("event_positions_m", []),
            "defect_intervals_m": metadata.get("defect_intervals_m", []),
            "terminal_position_m": metadata.get("terminal_position_m"),
            "sample_kind": metadata.get("sample_kind"),
        })
    decoded = [
        decode_predictions(probability, cfg, {"event": 0.5, "interval": 0.95, "terminal": 0.5})
        for probability in probabilities
    ]
    report = evaluate_predictions(decoded, truths, cfg)
    event = report["event_metrics_by_tolerance_m"]["5"]
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "model_name": checkpoint["model_name"],
        "split": args.split,
        "thresholds": {"event": 0.5, "interval": 0.95, "terminal": 0.5},
        "event_metrics_by_tolerance_m": report["event_metrics_by_tolerance_m"],
        "interval_metrics": report["interval_metrics"],
        "terminal_metrics": report["terminal_metrics"],
        "event_f1_5m": event.get("f1"),
        "event_precision_5m": event.get("precision"),
        "event_recall_5m": event.get("recall"),
    }
    output = args.output or (args.checkpoint.parent / f"evaluation_{args.split}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "checkpoint": result["checkpoint"],
        "epoch": result["checkpoint_epoch"],
        "event_f1_5m": result["event_f1_5m"],
        "event_precision_5m": result["event_precision_5m"],
        "event_recall_5m": result["event_recall_5m"],
        "interval_f1": result["interval_metrics"].get("f1"),
        "terminal_f1": result["terminal_metrics"].get("f1"),
        "output": str(output),
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
