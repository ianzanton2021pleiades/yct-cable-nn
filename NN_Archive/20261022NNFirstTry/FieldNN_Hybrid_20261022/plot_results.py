"""Summarize and plot U-Net / Transformer / ensemble evaluation results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
IMAGE_DIR = PROJECT_ROOT / "Image"


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


def load_results(run: Path, evaluation_name: str) -> dict:
    result = {}
    for name in ("unet", "transformer", "ensemble"):
        path = run / name / evaluation_name / "evaluation.json"
        if path.is_file():
            result[name] = json.loads(path.read_text(encoding="utf-8"))
    return result


def metric(result: dict, key: str, default=None):
    try:
        value = result
        for part in key.split("."):
            value = value[part]
        return float(value) if value is not None else default
    except (KeyError, TypeError, ValueError):
        return default


def rows(results: dict) -> list[dict]:
    output = []
    for name in ("unet", "transformer", "ensemble"):
        result = results.get(name)
        if not isinstance(result, dict) or "synthetic_test" not in result:
            continue
        output.append({
            "model": name,
            "syn_clean_f1_5m": metric(result, "synthetic_test.clean.report.event_metrics_by_tolerance_m.5.f1"),
            "syn_mixed_f1_5m": metric(result, "synthetic_test.mixed.report.event_metrics_by_tolerance_m.5.f1"),
            "syn_interval_f1": metric(result, "synthetic_test.clean.report.interval_metrics.f1"),
            "syn_terminal_f1": metric(result, "synthetic_test.clean.report.terminal_metrics.f1"),
            "field_nonwet_hit20": metric(result, "field_test.non_wet.hit_rate_20"),
            "field_nonwet_cand": metric(result, "field_test.non_wet.candidates_per_curve"),
            "field_strict_hit20": metric(result, "field_test.strict.hit_rate_20"),
            "field_strict_cand": metric(result, "field_test.strict.candidates_per_curve"),
            "wet_hit20": metric(result, "field_test.wet_1500.hit_rate_20"),
            "wet_cand": metric(result, "field_test.wet_1500.candidates_per_curve"),
            "hybrid_hit20": metric(result, "hybrid_test.hit_rate_20"),
            "hybrid_cand": metric(result, "hybrid_test.candidates_per_curve"),
            "cal_false_events": metric(result, "calibration_test.false_event_count"),
        })
    return output


def plot_metrics(table: list[dict], output_path: Path) -> None:
    style()
    import matplotlib.pyplot as plt

    labels = [row["model"] for row in table]
    x = np.arange(len(labels))
    width = 0.22

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4))

    axis = axes[0, 0]
    for offset, key, label in (
        (-width, "syn_clean_f1_5m", "synthetic clean F1@5m"),
        (0.0, "syn_mixed_f1_5m", "synthetic mixed F1@5m"),
        (width, "syn_interval_f1", "synthetic interval F1"),
    ):
        axis.bar(x + offset, [row.get(key) or 0.0 for row in table], width, label=label)
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_ylim(0, 1.02)
    axis.set_title("Synthetic test", fontsize=10)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=6.5, frameon=True)

    axis = axes[0, 1]
    for offset, key, label in (
        (-width / 2, "field_nonwet_hit20", "non-wet hit@20m"),
        (width / 2, "field_strict_hit20", "strict 5-cable hit@20m"),
    ):
        axis.bar(x + offset, [row.get(key) or 0.0 for row in table], width, label=label)
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_ylim(0, 1.02)
    axis.set_title("Measured field: known-position hit", fontsize=10)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=7, frameon=True)

    axis = axes[1, 0]
    for offset, key, label in (
        (-width, "field_nonwet_cand", "non-wet"),
        (0.0, "field_strict_cand", "strict 5-cable"),
        (width, "wet_cand", "wet-1500"),
    ):
        axis.bar(x + offset, [row.get(key) or 0.0 for row in table], width, label=label)
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_title("Candidates per curve (lower is better)", fontsize=10)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=7, frameon=True)

    axis = axes[1, 1]
    for offset, key, label in (
        (-width, "wet_hit20", "wet-1500 hit@20m"),
        (0.0, "hybrid_hit20", "injected hybrid hit@20m"),
        (width, "cal_false_events", "short false events"),
    ):
        axis.bar(x + offset, [row.get(key) or 0.0 for row in table], width, label=label)
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_title("Wet joint / injected / short reference", fontsize=10)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=6.5, frameon=True)

    for axis in axes.flat:
        axis.tick_params(top=True, right=True)
    fig.suptitle("FieldNN Hybrid: measured-S11 anchored evaluation", fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_training(run: Path, output_path: Path) -> None:
    style()
    import matplotlib.pyplot as plt

    histories = {}
    for name in ("unet", "transformer"):
        path = run / name / "history.csv"
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as stream:
            histories[name] = list(csv.DictReader(stream))
    if not histories:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for name, rows_ in histories.items():
        epochs = [int(row["epoch"]) for row in rows_]
        axes[0].plot(epochs, [float(row["val_syn"]) for row in rows_], label=f"{name} synthetic val")
        axes[0].plot(epochs, [float(row["val_hyb"]) for row in rows_], linestyle="--", label=f"{name} hybrid val")
        axes[1].plot(epochs, [float(row["field_hit20"]) for row in rows_], marker="o", label=f"{name} field hit@20m")
        axes[1].plot(epochs, [float(row["hyb_hit20"]) for row in rows_], marker="s", linestyle="--", label=f"{name} hybrid hit@20m")
        axes[1].plot(epochs, [float(row["field_cand"]) for row in rows_], marker="^", linestyle=":", label=f"{name} field candidates")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("validation loss")
    axes[0].set_title("Validation losses", fontsize=10)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("hit rate / candidates per curve")
    axes[1].set_title("Field and injected validation", fontsize=10)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.tick_params(top=True, right=True)
        axis.legend(fontsize=6.5, frameon=True)
    fig.suptitle("FieldNN Hybrid training history", fontsize=11)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=HERE / "output" / "hybrid")
    parser.add_argument("--evaluation-name", default="evaluation_e05")
    parser.add_argument("--image", type=Path, default=IMAGE_DIR / "20261022_FieldNN_Hybrid_metrics_e05.png")
    parser.add_argument("--training-image", type=Path, default=IMAGE_DIR / "20261022_FieldNN_Hybrid_training.png")
    args = parser.parse_args()

    results = load_results(args.run, args.evaluation_name)
    table = rows(results)
    if not table:
        raise SystemExit(f"No evaluation results under {args.run}")
    (args.run / "comparison.json").write_text(
        json.dumps(table, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    columns = list(table[0])
    print("| " + " | ".join(columns) + " |")
    print("|" + "|".join(["---"] * len(columns)) + "|")
    for row in table:
        print("| " + " | ".join(
            f"{row[key]:.4f}" if isinstance(row[key], float) else str(row[key])
            for key in columns
        ) + " |")
    plot_metrics(table, args.image)
    plot_training(args.run, args.training_image)
    print(f"\nfigure: {args.image}")
    print(f"training figure: {args.training_image}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
