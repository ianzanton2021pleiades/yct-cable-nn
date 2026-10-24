"""Select and plot two well-recognized measured-field FDR cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from eval_utils import collect_probabilities, load_model, match_events  # noqa: E402
from hybrid_data import FieldDataset  # noqa: E402


WET_CASE = "2026-6某地进行的1500m不同位置电缆接头浸水试验"


def style() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["font.family"] = ["Times New Roman", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["figure.dpi"] = 200


def case_rows(
    probabilities: list[np.ndarray],
    records: list[dict[str, Any]],
    cfg: dict[str, Any],
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    from decode import decode_predictions

    rows = []
    for probability, record in zip(probabilities, records):
        truth = [float(value) for value in (record.get("known_positions_nominal_m") or [])]
        if not truth:
            continue
        prediction = decode_predictions(probability, cfg, thresholds)
        hits20 = match_events(prediction["events"], truth, 20.0)
        rows.append({
            "index": int(record["index"]),
            "case_group": str(record.get("case_group") or ""),
            "source_relative_path": str(record.get("source_relative_path") or ""),
            "sweep_name": str(record.get("sweep_name") or ""),
            "truth": truth,
            "hits20": hits20,
            "hit_rate20": hits20 / max(len(truth), 1),
            "candidates": len(prediction["events"]),
            "candidates_per_curve": len(prediction["events"]),
            "prediction": prediction,
            "probability": probability,
            "record": record,
        })
    return rows


def select_cases(rows: list[dict[str, Any]], count: int = 4) -> list[dict[str, Any]]:
    def score(row):
        return (
            row["hit_rate20"]
            + 0.10 * min(len(row["truth"]), 4)
            - 0.03 * row["candidates"]
        )

    wet = sorted(
        [row for row in rows if row["case_group"] == WET_CASE and row["hit_rate20"] > 0],
        key=lambda row: (max(row["truth"]), -row["candidates"]),
        reverse=True,
    )
    non_wet = sorted(
        [row for row in rows if row["case_group"] != WET_CASE],
        key=score,
        reverse=True,
    )
    if count <= 2:
        wanted_wet, wanted_non_wet = 1, 1
    elif count == 3:
        wanted_wet, wanted_non_wet = 1, 2
    else:
        wanted_wet, wanted_non_wet = 2, 2
    selected = wet[:wanted_wet] + non_wet[:wanted_non_wet]
    if len(selected) < count:
        selected_ids = {id(row) for row in selected}
        remaining = [row for row in rows if id(row) not in selected_ids]
        selected.extend(sorted(remaining, key=score, reverse=True)[: count - len(selected)])
    return selected[:count]


def plot_case(
    row: dict[str, Any],
    features: np.ndarray,
    cfg: dict[str, Any],
    thresholds: dict[str, float],
    path: Path,
) -> None:
    style()
    import matplotlib.pyplot as plt

    distance = np.linspace(
        0.0,
        float(cfg["response"]["distance_max_m"]),
        int(cfg["response"]["points"]),
    )
    prediction = row["prediction"]
    truth = row["truth"]
    xmax = min(2500.0, max(max(truth) + 150.0, 1100.0))
    if row["case_group"] == WET_CASE:
        xmax = 1700.0

    fig, axes = plt.subplots(3, 1, figsize=(11, 8.2), sharex=True)
    axes[0].plot(distance, features[0], color="tab:blue", linewidth=0.55)
    axes[0].set_ylabel("Impulse feature")
    axes[1].plot(distance, features[1], color="tab:orange", linewidth=0.55)
    axes[1].set_ylabel("Step feature")
    axes[2].plot(distance, row["probability"][0], color="tab:blue", linewidth=0.75, label="Event probability")
    axes[2].axhline(float(thresholds["event"]), color="0.45", linestyle="--", linewidth=0.7, label="Event threshold")

    for position in truth:
        hit = any(abs(float(event["distance_m"]) - position) <= 20.0 for event in prediction["events"])
        axes[2].axvline(position, color="green" if hit else "red", linestyle=":", linewidth=1.0)
        axes[2].text(position, 1.015, f"{position:.0f}", rotation=90, fontsize=7,
                     color="green" if hit else "red", va="bottom", ha="center")
    for event in prediction["events"]:
        axes[2].plot(event["distance_m"], event["confidence"], "v", color="tab:red", markersize=4)
    for interval in prediction["intervals"]:
        axes[2].axvspan(interval["start_m"], interval["end_m"], color="tab:red", alpha=0.10)
    if prediction.get("terminal") is not None:
        axes[2].axvline(prediction["terminal"]["distance_m"], color="tab:green",
                        linestyle="--", linewidth=0.8, label="Terminal candidate")

    for axis in axes:
        axis.set_xlim(0.0, xmax)
        axis.grid(alpha=0.18)
        axis.tick_params(top=True, right=True)
    axes[2].set_xlabel("Response distance (m), nominal 172 m/μs")
    axes[2].set_ylabel("Event probability")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].legend(fontsize=7, loc="upper right", frameon=True)
    fig.suptitle(
        f"{row['case_group']} | {row['sweep_name']} | "
        f"hit@20m {row['hits20']}/{len(truth)}, candidates {row['candidates']}"
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path, nargs="+")
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "Image")
    parser.add_argument("--candidates-json", type=Path, default=HERE / "output" / "hybrid_long" / "case_candidates.json")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    torch.set_num_threads(1)
    device = torch.device(args.device)
    models = []
    cfg = None
    for path in args.checkpoint:
        model, checkpoint, config = load_model(path, device)
        models.append(model)
        cfg = config
    threshold_path = args.thresholds or (Path(args.checkpoint[0]).parent / "thresholds.json")
    calibration = json.loads(Path(threshold_path).read_text(encoding="utf-8"))
    thresholds = {key: float(value) for key, value in calibration["thresholds"].items()}

    dataset = FieldDataset(args.data, "test")
    probabilities, indices = collect_probabilities(models, dataset, device, args.batch_size)
    records = [dataset.records[index] for index in indices]
    rows = case_rows(probabilities, records, cfg, thresholds)
    rows.sort(key=lambda row: row["hit_rate20"] - 0.04 * row["candidates"], reverse=True)
    args.candidates_json.parent.mkdir(parents=True, exist_ok=True)
    args.candidates_json.write_text(
        json.dumps(
            [
                {key: value for key, value in row.items()
                 if key not in {"prediction", "probability", "record"}}
                for row in rows
            ],
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    selected = select_cases(rows, args.count)
    for number, row in enumerate(selected, 1):
        index = int(row["index"])
        feature = np.asarray(dataset.features[index], dtype=np.float32)
        path = args.output_dir / f"20261022_FieldNN_Hybrid_case{number}.png"
        plot_case(row, feature, cfg, thresholds, path)
        print(
            json.dumps({
                "case": number,
                "figure": str(path),
                "case_group": row["case_group"],
                "sweep_name": row["sweep_name"],
                "truth_m": row["truth"],
                "hits20": row["hits20"],
                "candidates": row["candidates"],
            }, ensure_ascii=False)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
