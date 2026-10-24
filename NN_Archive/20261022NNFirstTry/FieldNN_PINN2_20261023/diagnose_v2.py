"""Where the event loss actually goes, and where recall is lost by distance.

Two questions the training log cannot answer:

1. The focal term is divided by the number of positive centres, so its value is
   dominated by the background probability level rather than by localisation.
   This script splits ``event`` into its positive and negative parts and reports
   the background probability quantiles.
2. Field failures cluster beyond 1500 m.  This script buckets event recall by
   true distance on the synthetic validation view, where the truth is exact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
for path in (
    HERE,
    PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_Hybrid_20261022",
    PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_Iter5_20261022",
    PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_20261022",
    PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_PINN_20261023",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data_v2 import PinnV2Dataset  # noqa: E402
from decode import decode_predictions  # noqa: E402
from eval_utils import match_events  # noqa: E402
from models_pinn import build_model  # noqa: E402

SYNTHETIC = Path(r"E:\CableNNDataSet\CodeX\FieldNN_Iter4_20261022")
THRESHOLDS = {"event": 0.5, "interval": 0.95, "terminal": 0.6}
BUCKETS = ((0.0, 500.0), (500.0, 1000.0), (1000.0, 1500.0), (1500.0, 2000.0), (2000.0, 2700.0))


def _legacy_dataset(view: str):
    """Eight-channel view, for measuring the PINN-v1 checkpoints on the same terms."""
    from data_pinn import PinnDataset

    return PinnDataset(SYNTHETIC, "val", train=False, view=view)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--view", default="mixed")
    parser.add_argument("--channels", type=int, default=11,
                        help="11 for PINN-v2 checkpoints, 8 for PINN-v1 checkpoints")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    model = build_model(
        state["model_name"],
        input_channels=int(config["model"]["input_channels"]),
        width=int(config["model"].get("width", 32)),
        depth=int(config["model"].get("depth", 3)),
        dropout=float(config["model"].get("dropout", 0.15)),
    ).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    dataset = (
        PinnV2Dataset(SYNTHETIC, "val", train=False, view=args.view)
        if args.channels == 11
        else _legacy_dataset(args.view)
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    distance = dataset.base.distance

    background_probabilities = []
    positive_loss = 0.0
    negative_loss = 0.0
    positive_count = 0.0
    predictions = []
    truths = []
    with torch.inference_mode():
        for batch in loader:
            inputs, targets, items = batch[0].to(device), batch[1].to(device), batch[2]
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                output = model(inputs)
            logits = output[:, 0].float()
            label = targets[:, 0].float()
            probability = logits.sigmoid()
            valid = (label >= 0.0) & (targets[:, 3].float() > 0.0)
            positive = (label >= 0.99) & valid
            positive_loss += float((-torch.nn.functional.logsigmoid(logits) * (1 - probability) ** 2
                                    * positive.float()).sum())
            positive_count += float(positive.sum())
            negative_loss += float((-torch.nn.functional.logsigmoid(-logits) * probability ** 2
                                    * (1 - label).pow(4) * valid.float()).sum())
            background_probabilities.append((probability * (label < 0.05) * valid).cpu().numpy())
            probabilities = output[:, :3].float().sigmoid().cpu().numpy()
            for index, item in enumerate(items.tolist()):
                predictions.append(decode_predictions(probabilities[index], config, THRESHOLDS))
                metadata = dataset.metadata[int(item)]
                truths.append([float(value) for value in metadata.get("event_positions_m", [])])

    background = np.concatenate([item.ravel() for item in background_probabilities])
    positive_part = positive_loss / max(positive_count, 1)
    negative_part = negative_loss / max(positive_count, 1)

    buckets = []
    for low, high in BUCKETS:
        hits = total = 0
        for prediction, truth in zip(predictions, truths):
            for position in truth:
                if low <= position < high:
                    total += 1
                    hits += int(bool(match_events(prediction["events"], [position], 5.0)))
        buckets.append({
            "range_m": [low, high], "events": total,
            "recall_at_5m": hits / max(total, 1),
        })

    result = {
        "checkpoint": str(args.checkpoint),
        "view": args.view,
        "event_loss": positive_part + negative_part,
        "positive_part": positive_part,
        "negative_part": negative_part,
        "background_probability": {
            "mean": float(background.mean()),
            "p90": float(np.quantile(background, 0.90)),
            "p99": float(np.quantile(background, 0.99)),
            "p999": float(np.quantile(background, 0.999)),
            "max": float(background.max()),
            "share_above_0_02": float((background > 0.02).mean()),
        },
        "recall_by_distance": buckets,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    output = args.output or args.checkpoint.parent / f"diagnose_{args.view}.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
