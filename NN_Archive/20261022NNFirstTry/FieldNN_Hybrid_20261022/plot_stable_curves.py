"""Plot stable long training: loss and accuracy against training steps."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def column(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if values.size < window:
        return values
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="same")


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=HERE / "output" / "hybrid_stable" / "unet")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "Image" / "20261022_FieldNN_Hybrid_stable_curves.png")
    args = parser.parse_args()

    history_path = args.run / "history.csv"
    steps_path = args.run / "steps.csv"
    if not history_path.is_file() or not steps_path.is_file():
        raise SystemExit(f"missing history/steps under {args.run}")
    history = read_csv(history_path)
    steps = read_csv(steps_path)
    if not history or not steps:
        raise SystemExit("empty history/steps")

    epochs = column(history, "epoch")
    steps_per_epoch = max(int(float(row["step"])) for row in steps if int(float(row["epoch"])) == 1)
    val_x = epochs * steps_per_epoch
    scores = np.minimum(column(history, "score_raw"), column(history, "score_ema"))
    best_index = int(np.argmin(scores))
    best_epoch = float(epochs[best_index])
    best_step = best_epoch * steps_per_epoch

    synthetic_steps = [row for row in steps if int(float(row["domain"])) == 0]
    hybrid_steps = [row for row in steps if int(float(row["domain"])) == 1]
    synthetic = {
        "step": np.asarray([float(row["step"]) for row in synthetic_steps]),
        "loss": np.asarray([float(row["loss"]) for row in synthetic_steps]),
    }
    hybrid = {
        "step": np.asarray([float(row["step"]) for row in hybrid_steps]),
        "loss": np.asarray([float(row["loss"]) for row in hybrid_steps]),
    }

    style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.4), sharex=True)
    axis = axes[0]
    axis.plot(synthetic["step"], smooth(synthetic["loss"], 80), color="tab:blue",
              linewidth=1.0, label="train synthetic (smoothed)")
    axis.plot(val_x, column(history, "val_syn"), color="tab:blue", linestyle="--",
              marker="o", markersize=2.5, linewidth=1.0, label="validation synthetic")
    axis.plot(hybrid["step"], smooth(hybrid["loss"], 80), color="tab:orange",
              linewidth=1.0, label="train injected (smoothed)")
    axis.plot(val_x, column(history, "val_hyb"), color="tab:orange", linestyle="--",
              marker="s", markersize=2.5, linewidth=1.0, label="validation injected")
    axis.axvline(best_step, color="tab:green", linestyle=":", linewidth=1.0,
                 label=f"selected step {best_step:.0f}")
    axis.set_ylabel("loss")
    axis.set_title("Train / validation loss vs training step", fontsize=10)
    axis.grid(alpha=0.2)
    axis.legend(fontsize=7, frameon=True, ncol=2)

    axis = axes[1]
    axis.plot(val_x, column(history, "train_syn_f1"), color="tab:blue", marker="o",
              markersize=2.5, linewidth=1.1, label="train synthetic F1@5m (clean probe)")
    axis.plot(val_x, column(history, "syn_f1"), color="tab:orange", marker="s",
              markersize=2.5, linewidth=1.1, label="validation synthetic F1@5m")
    axis.plot(val_x, column(history, "field_hit20"), color="tab:green", marker="^",
              markersize=2.5, linewidth=1.1, label="field known-positive hit@20m")
    axis.plot(val_x, column(history, "hyb_hit20"), color="tab:purple", marker="D",
              markersize=2.0, linestyle="--", linewidth=0.9, label="injected hit@20m")
    axis.axvline(best_step, color="tab:green", linestyle=":", linewidth=1.0)
    axis.set_xlabel("training step")
    axis.set_ylabel("F1 / hit rate")
    axis.set_ylim(0.0, 1.02)
    axis.set_title("Accuracy (F1 / hit rate) vs training step", fontsize=10)
    axis.grid(alpha=0.2)
    axis.legend(fontsize=7, frameon=True, ncol=2)

    for axis in axes:
        axis.tick_params(top=True, right=True)
    fig.suptitle("FieldNN Hybrid stable long training (frozen encoder, 50 epochs)", fontsize=12)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(f"figure: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
