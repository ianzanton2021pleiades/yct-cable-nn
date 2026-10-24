"""Event-threshold sensitivity on the measured-field test set.

The annotated validation split has nine curves, which is too few to select a
threshold.  This script therefore reports a sensitivity curve on the test split
and labels it as such; the operating point quoted in the report is the one the
CMDC baseline used, so no test-set tuning enters the headline numbers.
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
CMDC = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_Hybrid_20261022"
CODEX = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_20261022"
for path in (HERE, CMDC, CODEX):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from decode import decode_predictions  # noqa: E402
from eval_utils import WET_CASE, match_events  # noqa: E402
from hybrid_data import FieldDataset  # noqa: E402
from models_pinn import build_model  # noqa: E402


FIELD_ROOT = Path(r"E:\CableNNDataSet\Deepseek_CMDC\FieldNN_Hybrid_Long_20261022")


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
    parser.add_argument("--output", type=Path, default=HERE / "output" / "threshold_sweep.json")
    parser.add_argument("--device", default="cuda")
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

    dataset = FieldDataset(FIELD_ROOT, "test")
    probabilities, indices = collect(model, dataset, device)
    records = [dataset.records[index] for index in indices]

    curve = []
    for threshold in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        thresholds = {"event": threshold, "interval": 0.95, "terminal": 0.60}
        hits = total_truth = candidates = 0
        curves_all = curves_any = annotated = 0
        for probability, record in zip(probabilities, records):
            truth = [float(value) for value in (record.get("known_positions_nominal_m") or [])]
            if not truth or str(record.get("case_group") or "") == WET_CASE:
                continue
            prediction = decode_predictions(probability, config, thresholds)
            hit = match_events(prediction["events"], truth, 20.0)
            hits += hit
            total_truth += len(truth)
            candidates += len(prediction["events"])
            annotated += 1
            curves_all += int(hit == len(truth))
            curves_any += int(hit > 0)
        curve.append({
            "event_threshold": threshold,
            "hit_rate_20": hits / max(total_truth, 1),
            "candidates_per_curve": candidates / max(annotated, 1),
            "curves_all_hit": curves_all,
            "curves_any_hit": curves_any,
            "annotated": annotated,
            "P_ann": hits / max(candidates, 1),
        })
        print(json.dumps(curve[-1], ensure_ascii=False), flush=True)

    # 事件阈值对末端无影响，这里再单独统计末端的长度误差
    terminal_rows = []
    for probability, record in zip(probabilities, records):
        length = record.get("known_length_m")
        prediction = decode_predictions(probability, config, {"event": 0.5, "interval": 0.95, "terminal": 0.6})
        terminal = prediction.get("terminal")
        if not length or terminal is None:
            continue
        terminal_rows.append(abs(float(terminal["distance_m"]) - float(length)))
    summary = {
        "checkpoint": str(args.checkpoint),
        "curve": curve,
        "terminal_abs_error_m": {
            "n": len(terminal_rows),
            "median": float(np.median(terminal_rows)) if terminal_rows else None,
            "mean": float(np.mean(terminal_rows)) if terminal_rows else None,
            "within_50m": float(np.mean([value <= 50.0 for value in terminal_rows])) if terminal_rows else None,
            "within_100m": float(np.mean([value <= 100.0 for value in terminal_rows])) if terminal_rows else None,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["terminal_abs_error_m"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
