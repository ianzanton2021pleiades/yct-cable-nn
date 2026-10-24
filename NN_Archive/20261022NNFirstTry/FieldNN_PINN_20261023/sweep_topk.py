"""Top-k event decoding: fix the candidate budget, then measure hit rate.

A global probability threshold produces a variable number of candidates per
curve, so hit rates are not comparable between models.  This script keeps the
k strongest peaks only, which turns the comparison into "same budget, whose
peaks are right" and matches how the candidates are counted in the field
reports.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import find_peaks
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
CMDC = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_Hybrid_20261022"
CODEX = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_20261022"
for path in (HERE, CMDC, CODEX):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval_utils import WET_CASE, match_events  # noqa: E402
from hybrid_data import FieldDataset  # noqa: E402
from models_pinn import build_model  # noqa: E402


FIELD_ROOT = Path(r"E:\CableNNDataSet\Deepseek_CMDC\FieldNN_Hybrid_Long_20261022")
NEAR_M = 30.0
EDGE_M = 10.0


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


def topk_events(probability: np.ndarray, config: dict, k: int) -> list[float]:
    distance = np.linspace(0.0, float(config["response"]["distance_max_m"]), probability.shape[-1])
    spacing = distance[1] - distance[0]
    options = config.get("evaluation", {})
    peaks, _ = find_peaks(
        probability[0],
        height=0.05,
        distance=max(1, int(round(options.get("event_min_separation_m", 3.0) / spacing))),
        prominence=float(options.get("event_prominence", 0.10)),
    )
    valid = [index for index in peaks
             if distance[index] >= NEAR_M and distance[index] <= distance[-1] - EDGE_M]
    ranked = sorted(valid, key=lambda index: -float(probability[0, index]))[:k]
    return sorted(float(distance[index]) for index in ranked)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    model = build_model(state["model_name"], input_channels=8, width=32, depth=3,
                        dropout=config["model"].get("dropout", 0.15)).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    dataset = FieldDataset(FIELD_ROOT, "test")
    probabilities, indices = collect(model, dataset, device)
    records = [dataset.records[index] for index in indices]

    curve = []
    for k in (1, 2, 3, 4, 5, 6, 8):
        hits = truth_total = predicted = annotated = all_hit = any_hit = 0
        errors = []
        for probability, record in zip(probabilities, records):
            truth = [float(value) for value in (record.get("known_positions_nominal_m") or [])]
            if not truth or str(record.get("case_group") or "") == WET_CASE:
                continue
            events = topk_events(probability, config, k)
            hit = match_events([{"distance_m": value} for value in events], truth, 20.0)
            hits += hit
            truth_total += len(truth)
            predicted += len(events)
            annotated += 1
            all_hit += int(hit == len(truth))
            any_hit += int(hit > 0)
            for position in truth:
                if events:
                    nearest = min(abs(value - position) for value in events)
                    if nearest <= 20.0:
                        errors.append(nearest)
        curve.append({
            "k": k,
            "hit_rate_20": hits / max(truth_total, 1),
            "candidates_per_curve": predicted / max(annotated, 1),
            "P_ann": hits / max(predicted, 1),
            "curves_all_hit": all_hit,
            "curves_any_hit": any_hit,
            "annotated": annotated,
            "mae_matched_m": float(np.mean(errors)) if errors else None,
        })
        print(json.dumps(curve[-1], ensure_ascii=False), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"checkpoint": str(args.checkpoint), "curve": curve},
                                      ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
