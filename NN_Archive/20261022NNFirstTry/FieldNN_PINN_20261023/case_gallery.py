"""Six-panel measured-field case gallery for slide use."""

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
THRESHOLDS = {"event": 0.5, "interval": 0.95, "terminal": 0.6}


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


def physics_terminal(probability: np.ndarray, prediction: dict, config: dict) -> dict:
    events = prediction["events"]
    if not events:
        return prediction
    distance = np.linspace(0.0, float(config["response"]["distance_max_m"]), probability.shape[-1])
    last_event = max(float(event["distance_m"]) for event in events)
    allowed = (distance >= last_event + 20.0) & (distance <= distance[-1] - 10.0)
    if not allowed.any():
        return prediction
    curve = np.where(allowed, probability[2], -1.0)
    best = int(np.argmax(curve))
    if curve[best] < float(THRESHOLDS["terminal"]):
        return {**prediction, "terminal": None}
    return {**prediction,
            "terminal": {"distance_m": float(distance[best]), "confidence": float(curve[best])}}


def plot_single_case(row: dict, feature: np.ndarray, config: dict, path: Path, plt) -> None:
    """Three stacked panels: impulse feature, step feature, event probability."""
    distance = np.linspace(0.0, float(config["response"]["distance_max_m"]), feature.shape[-1])
    truth = row["truth"]
    prediction = row["prediction"]
    terminal = prediction.get("terminal")

    figure, axes = plt.subplots(3, 1, figsize=(11.0, 8.2), sharex=True)
    axes[0].plot(distance, feature[0], color="tab:blue", linewidth=0.55)
    axes[0].set_ylabel("Impulse feature")
    axes[1].plot(distance, feature[1], color="tab:orange", linewidth=0.55)
    axes[1].set_ylabel("Step feature")

    axes[2].plot(distance, row["probability"][0], color="tab:blue", linewidth=0.75,
                 label="Event probability")
    axes[2].axhline(float(THRESHOLDS["event"]), color="0.45", linestyle="--", linewidth=0.7,
                    label="Event threshold")
    for position in truth:
        hit = any(abs(float(event["distance_m"]) - position) <= 20.0 for event in prediction["events"])
        axes[2].axvline(position, color="#1b7f3b" if hit else "#c62828", linestyle=":", linewidth=1.0)
        axes[2].text(position, 1.015, f"{position:.0f}", rotation=90, fontsize=7,
                     color="#1b7f3b" if hit else "#c62828", va="bottom", ha="center")
    for event in prediction["events"]:
        axes[2].plot(event["distance_m"], event["confidence"], "v", color="#c62828", markersize=4)
    for interval in prediction["intervals"]:
        axes[2].axvspan(interval["start_m"], interval["end_m"], color="#c62828", alpha=0.10)
    if terminal is not None:
        axes[2].axvline(float(terminal["distance_m"]), color="#1b7f3b", linestyle="--", linewidth=0.9,
                        label="Terminal candidate")
    if row["length"]:
        axes[0].axvline(float(row["length"]), color="black", linestyle="-.", linewidth=0.8,
                        label=f"Known length {float(row['length']):.0f} m")
        axes[0].legend(fontsize=7, loc="upper right", frameon=True)

    if row["length"]:
        xmax = min(2600.0, float(row["length"]) * 1.15)
    else:
        xmax = min(2600.0, max(truth) * 1.35 + 120.0)
    xmax = max(xmax, max(truth) + 60.0)
    for axis in axes:
        axis.set_xlim(0.0, xmax)
        axis.grid(alpha=0.18)
        axis.tick_params(top=True, right=True, direction="in")
    axes[2].set_xlabel("Response distance (m), nominal 172 m/μs")
    axes[2].set_ylabel("Event probability")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].legend(fontsize=7, loc="upper right", frameon=True)

    title = row["record"].get("case_group", "")
    figure.suptitle(
        f"{title} | {row['record'].get('sweep_name','')} | "
        f"hit@20m {row['hits']}/{len(truth)}, candidates {row['candidates']}",
        fontsize=10,
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--per-case-dir", type=Path, default=None)
    parser.add_argument("--per-case-prefix", type=str, default="20261023_PINN150_")
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

    rows = []
    for probability, record in zip(probabilities, records):
        truth = [float(value) for value in (record.get("known_positions_nominal_m") or [])]
        if not truth:
            continue
        prediction = physics_terminal(probability, decode_predictions(probability, config, THRESHOLDS), config)
        hits = match_events(prediction["events"], truth, 20.0)
        terminal = prediction.get("terminal")
        length = record.get("known_length_m")
        terminal_error = None
        if terminal and length:
            terminal_error = abs(float(terminal["distance_m"]) - float(length))
        rows.append({
            "record": record, "truth": truth, "prediction": prediction, "probability": probability,
            "hits": hits, "all_hit": hits == len(truth), "candidates": len(prediction["events"]),
            "terminal_error": terminal_error, "length": length,
            "wet": str(record.get("case_group") or "") == WET_CASE,
        })

    def rank(row):
        terminal_ok = row["terminal_error"] is not None and row["terminal_error"] <= 60.0
        tier = 0 if (row["all_hit"] and terminal_ok) else (1 if row["all_hit"] else 2)
        return (tier, -len(row["truth"]), 0.02 * row["candidates"])

    ordered = sorted(rows, key=rank)
    selected: list[dict] = []
    used_groups: set[str] = set()
    for tier_limit in (0, 1, 2):
        for row in ordered:
            if len(selected) >= args.count:
                break
            if rank(row)[0] > tier_limit:
                continue
            if row["hits"] / max(len(row["truth"]), 1) < 0.66:
                continue
            group = str(row["record"].get("case_group") or "")
            if group in used_groups:
                continue
            selected.append(row)
            used_groups.add(group)
        if len(selected) >= args.count:
            break
    if len(selected) < args.count:
        chosen = {id(row) for row in selected}
        selected += [row for row in ordered if id(row) not in chosen][: args.count - len(selected)]

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

    columns = 2
    counts = int(np.ceil(len(selected) / columns))
    figure, axes = plt.subplots(counts, columns, figsize=(15.0, 3.0 * counts), squeeze=False)
    summary = []
    for position, row in enumerate(selected):
        axis = axes[position // columns][position % columns]
        feature = np.asarray(dataset.features[row["record"]["index"]], dtype=np.float32)
        distance = np.linspace(0.0, float(config["response"]["distance_max_m"]), feature.shape[-1])
        truth = row["truth"]
        prediction = row["prediction"]
        xmax = min(2500.0, max(max(truth) + 200.0, 400.0))
        if row["wet"]:
            xmax = 1700.0

        envelope = np.abs(feature[0])
        envelope = envelope / max(float(envelope.max()), 1e-6)
        axis.fill_between(distance, 0.0, 0.85 * envelope, color="0.75", alpha=0.45, linewidth=0)
        axis.plot(distance, row["probability"][0], color="#1f4e9c", linewidth=1.0)
        axis.axhline(float(THRESHOLDS["event"]), color="0.5", linestyle="--", linewidth=0.7)
        for value in truth:
            hit = any(abs(float(event["distance_m"]) - value) <= 20.0 for event in prediction["events"])
            axis.axvline(value, color="#1b7f3b" if hit else "#c62828", linestyle=":", linewidth=1.1)
        for event in prediction["events"]:
            axis.plot(event["distance_m"], event["confidence"], "v", color="#c62828", markersize=5)
        terminal = prediction.get("terminal")
        if terminal:
            axis.axvline(float(terminal["distance_m"]), color="#1b7f3b", linestyle="--", linewidth=1.0)
        if row["length"]:
            axis.axvline(float(row["length"]), color="black", linestyle="-.", linewidth=0.9)
        axis.set_xlim(0.0, xmax)
        axis.set_ylim(0.0, 1.05)
        axis.grid(alpha=0.2)
        title = row["record"].get("case_group", "")
        marker = title.find("深圳")
        if marker > 0:
            title = title[marker:]
        title = title[:24] + "..." if len(title) > 24 else title
        terminal_text = "n/a" if row["terminal_error"] is None else f"{row['terminal_error']:.0f} m"
        axis.set_title(
            f"{title} | {row['record'].get('sweep_name','')}\n"
            f"hit@20m {row['hits']}/{len(truth)} | 候选 {row['candidates']} | 末端误差 {terminal_text}"
            + (" | 全中" if row["all_hit"] else ""),
            fontsize=8.5,
        )
        if position // columns == counts - 1:
            axis.set_xlabel("distance (m)")
        if position % columns == 0:
            axis.set_ylabel("event prob.")
        summary.append({
            "case_group": row["record"].get("case_group"),
            "sweep": row["record"].get("sweep_name"),
            "truth_m": truth,
            "hits20": row["hits"],
            "candidates": row["candidates"],
            "terminal_m": (terminal or {}).get("distance_m"),
            "known_length_m": row["length"],
        })

    figure.suptitle("PINN U-Net measured-field cases (red triangle: prediction, dotted: annotated position, "
                    "dash-dot: known length)", fontsize=10)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)
    plt.close(figure)

    if args.per_case_dir is not None:
        args.per_case_dir.mkdir(parents=True, exist_ok=True)
        for number, row in enumerate(selected, 1):
            path = args.per_case_dir / f"{args.per_case_prefix}case{number}.png"
            plot_single_case(row, np.asarray(dataset.features[row["record"]["index"]], dtype=np.float32),
                             config, path, plt)
            summary[number - 1]["figure"] = str(path)

    print(json.dumps({"cases": summary}, ensure_ascii=False, indent=2))
    args.output.with_suffix(".json").write_text(
        json.dumps({"cases": summary}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
