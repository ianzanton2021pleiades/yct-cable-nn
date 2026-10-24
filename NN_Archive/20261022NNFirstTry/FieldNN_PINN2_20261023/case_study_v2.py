"""Measured-field case figures with the terminal-after-last-event physics rule."""

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
THRESHOLDS = {"event": 0.5, "interval": 0.95, "terminal": 0.6}
TERMINAL_MARGIN_M = 20.0


def style() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["font.family"] = ["Times New Roman", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.top"] = True
    plt.rcParams["ytick.right"] = True
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["figure.dpi"] = 200


def physics_terminal(probability: np.ndarray, prediction: dict, config: dict) -> dict:
    events = prediction["events"]
    if not events:
        return prediction
    distance = np.linspace(0.0, float(config["response"]["distance_max_m"]), probability.shape[-1])
    allowed = (distance >= max(float(event["distance_m"]) for event in events) + TERMINAL_MARGIN_M)
    allowed &= distance <= distance[-1] - 10.0
    if not allowed.any():
        return prediction
    curve = np.where(allowed, probability[2], -1.0)
    best = int(np.argmax(curve))
    if curve[best] < float(THRESHOLDS["terminal"]):
        return {**prediction, "terminal": None}
    return {**prediction, "terminal": {"distance_m": float(distance[best]),
                                       "confidence": float(curve[best])}}


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
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "Image")
    parser.add_argument("--prefix", type=str, default="20261023_PINN2")
    parser.add_argument("--count", type=int, default=4)
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
    predictions = [physics_terminal(prob, decode_predictions(prob, config, THRESHOLDS), config)
                   for prob in probabilities]

    rows = []
    for item, record, prediction, probability in zip(indices, records, predictions, probabilities):
        truth = [float(value) for value in (record.get("known_positions_nominal_m") or [])]
        if not truth:
            continue
        hits = match_events(prediction["events"], truth, 20.0)
        rows.append({
            "index": int(item), "case_group": str(record.get("case_group") or ""),
            "sweep_name": str(record.get("sweep_name") or ""), "truth": truth,
            "hits20": hits, "hit_rate20": hits / max(len(truth), 1),
            "candidates": len(prediction["events"]), "prediction": prediction,
            "probability": probability, "record": record,
        })

    def score(row):
        return row["hit_rate20"] + 0.08 * min(len(row["truth"]), 5) - 0.02 * row["candidates"]

    non_wet = sorted([row for row in rows if row["case_group"] != WET_CASE], key=score, reverse=True)
    wet = sorted([row for row in rows if row["case_group"] == WET_CASE and row["hit_rate20"] > 0],
                 key=score, reverse=True)
    selected = (wet[:1] + non_wet[: max(args.count - 1, 0)])[: args.count]

    summary = {"cases": []}
    for number, row in enumerate(selected, 1):
        feature = np.asarray(dataset.features[row["index"]], dtype=np.float32)
        distance = np.linspace(0.0, float(config["response"]["distance_max_m"]), feature.shape[-1])
        truth, prediction = row["truth"], row["prediction"]
        xmax = 1700.0 if row["case_group"] == WET_CASE else min(2500.0, max(max(truth) + 200.0, 400.0))

        style()
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(3, 1, figsize=(11, 8.2), sharex=True)
        axes[0].plot(distance, feature[0], color="tab:blue", linewidth=0.55)
        axes[0].set_ylabel("Impulse feature")
        axes[1].plot(distance, feature[1], color="tab:orange", linewidth=0.55)
        axes[1].set_ylabel("Step feature")
        axes[2].plot(distance, row["probability"][0], color="tab:blue", linewidth=0.75,
                     label="Event probability")
        axes[2].axhline(float(THRESHOLDS["event"]), color="0.45", linestyle="--", linewidth=0.7,
                        label="Event threshold")
        for position in truth:
            hit = any(abs(float(event["distance_m"]) - position) <= 20.0
                      for event in prediction["events"])
            axes[2].axvline(position, color="green" if hit else "red", linestyle=":", linewidth=1.0)
            axes[2].text(position, 1.015, f"{position:.0f}", rotation=90, fontsize=7,
                         color="green" if hit else "red", va="bottom", ha="center")
        for event in prediction["events"]:
            axes[2].plot(event["distance_m"], event["confidence"], "v", color="tab:red", markersize=4)
        for interval in prediction["intervals"]:
            axes[2].axvspan(interval["start_m"], interval["end_m"], color="tab:red", alpha=0.10)
        if prediction.get("terminal") is not None:
            axes[2].axvline(prediction["terminal"]["distance_m"], color="tab:green",
                            linestyle="--", linewidth=0.9, label="Terminal candidate")
        known_length = row["record"].get("known_length_m")
        if known_length:
            axes[0].axvline(float(known_length), color="black", linestyle="-.", linewidth=0.8,
                            label=f"Known length {float(known_length):.0f} m")
            axes[0].legend(fontsize=7, loc="upper right", frameon=True)
        for axis in axes:
            axis.set_xlim(0.0, xmax)
            axis.grid(alpha=0.18)
            axis.tick_params(top=True, right=True)
        axes[2].set_xlabel("Response distance (m), nominal 172 m/us")
        axes[2].set_ylabel("Event probability")
        axes[2].set_ylim(0.0, 1.05)
        axes[2].legend(fontsize=7, loc="upper right", frameon=True)
        figure.suptitle(
            f"{row['case_group']} | {row['sweep_name']} | "
            f"hit@20m {row['hits20']}/{len(truth)}, candidates {row['candidates']}"
        )
        figure.tight_layout()
        path = args.output_dir / f"{args.prefix}_case{number}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=200)
        plt.close(figure)
        summary["cases"].append({
            "case": number, "figure": str(path), "case_group": row["case_group"],
            "sweep_name": row["sweep_name"], "truth_m": truth, "hits20": row["hits20"],
            "candidates": row["candidates"],
            "terminal_m": (prediction.get("terminal") or {}).get("distance_m"),
            "known_length_m": known_length,
        })
        print(json.dumps(summary["cases"][-1], ensure_ascii=False), flush=True)

    output = args.output_dir / f"{args.prefix}_cases.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"cases": len(summary["cases"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
