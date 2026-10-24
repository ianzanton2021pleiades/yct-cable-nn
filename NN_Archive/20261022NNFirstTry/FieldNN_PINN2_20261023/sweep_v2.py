"""Event-threshold sensitivity on the measured-field test split.

Reported as a sensitivity curve only.  The headline operating point stays at the
threshold the CMDC baseline used so no test-set tuning enters the comparison.
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

from data_v2 import FieldV2  # noqa: E402
from decode import decode_predictions  # noqa: E402
from eval_utils import WET_CASE, match_events  # noqa: E402
from models_pinn import build_model  # noqa: E402

FIELD_ROOT = Path(r"E:\CableNNDataSet\Deepseek_CMDC\FieldNN_Hybrid_Long_20261022")
THRESHOLD_GRID = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)


def collect(model, dataset, device):
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    probabilities, indices = [], []
    with torch.inference_mode():
        for batch in loader:
            inputs, item = (batch[0], batch[1]) if len(batch) == 2 else (batch[0], batch[2])
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                output = model(inputs.to(device, non_blocking=device.type == "cuda"))
            probabilities.extend(output[:, :3].float().sigmoid().cpu().numpy())
            indices.extend(int(value) for value in item.tolist())
    return probabilities, indices


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
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

    dataset = FieldV2(FIELD_ROOT, "test")
    probabilities, indices = collect(model, dataset, device)
    records = [dataset.records[index] for index in indices]

    curve = []
    for threshold in THRESHOLD_GRID:
        thresholds = {"event": threshold, "interval": 0.95, "terminal": 0.60}
        hits = truth_total = candidates = annotated = 0
        curves_all = curves_any = 0
        for probability, record in zip(probabilities, records):
            truth = [float(value) for value in (record.get("known_positions_nominal_m") or [])]
            if not truth or str(record.get("case_group") or "") == WET_CASE:
                continue
            prediction = decode_predictions(probability, config, thresholds)
            matched = match_events(prediction["events"], truth, 20.0)
            annotated += 1
            hits += matched
            truth_total += len(truth)
            candidates += len(prediction["events"])
            curves_all += int(matched == len(truth))
            curves_any += int(matched > 0)
        curve.append({
            "event_threshold": threshold,
            "hit_rate_20": hits / max(truth_total, 1),
            "candidates_per_curve": candidates / max(annotated, 1),
            "P_ann": hits / max(candidates, 1),
            "curves_all_hit": curves_all,
            "curves_any_hit": curves_any,
            "annotated": annotated,
        })
        print(json.dumps(curve[-1]), flush=True)

    output = args.output or args.checkpoint.parent / "threshold_sweep.json"
    output.write_text(json.dumps({"checkpoint": str(args.checkpoint), "curve": curve},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print("written", output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
