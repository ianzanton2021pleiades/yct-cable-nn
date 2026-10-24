"""Plot synthetic pretraining loss and F1 curves against training steps."""

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
    parser.add_argument("--run", type=Path, default=HERE / "output" / "pretrain50" / "unet")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "Image" / "20261022_FieldNN_Hybrid_pretrain50_curves.png")
    args = parser.parse_args()

    history = read_csv(args.run / "history.csv")
    steps = read_csv(args.run / "steps.csv")
    if not history or not steps:
        raise SystemExit("empty history/steps")
    epochs = column(history, "epoch")
    steps_per_epoch = max(int(float(row["step"])) for row in steps if int(float(row["epoch"])) == 1)
    val_x = epochs * steps_per_epoch
    scores = column(history, "best_score")
    best_index = int(np.argmin(scores))
    best_epoch = float(epochs[best_index])
    best_step = best_epoch * steps_per_epoch

    step_x = np.asarray([float(row["step"]) for row in steps])
    train_loss = np.asarray([float(row["loss"]) for row in steps])

    style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True)
    axis = axes[0]
    axis.plot(step_x, smooth(train_loss, 80), color="tab:blue", linewidth=1.0, label="training loss (smoothed)")
    axis.plot(val_x, column(history, "val_loss"), color="tab:orange", marker="o",
              markersize=3, linewidth=1.1, label="validation loss")
    axis.axvline(best_step, color="tab:green", linestyle=":", linewidth=1.0,
                 label=f"selected step {best_step:.0f}")
    axis.set_ylabel("loss")
    axis.set_title("Synthetic pretraining: train / validation loss", fontsize=10)
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8, frameon=True)

    axis = axes[1]
    axis.plot(val_x, column(history, "train_f1"), color="tab:blue", marker="o",
              markersize=3, linewidth=1.1, label="training event F1@5m (clean probe)")
    axis.plot(val_x, column(history, "val_f1"), color="tab:orange", marker="s",
              markersize=3, linewidth=1.1, label="validation event F1@5m")
    axis.plot(val_x, column(history, "val_terminal_f1"), color="tab:green", marker="^",
              markersize=2.5, linestyle="--", linewidth=0.9, label="validation terminal F1")
    axis.axvline(best_step, color="tab:green", linestyle=":", linewidth=1.0)
    axis.set_xlabel("training step")
    axis.set_ylabel("F1")
    axis.set_ylim(0.0, 1.02)
    axis.set_title("Synthetic pretraining: accuracy (F1) curves", fontsize=10)
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8, frameon=True)

    for axis in axes:
        axis.tick_params(top=True, right=True)
    fig.suptitle("FieldNN Hybrid: 50-epoch synthetic pretraining", fontsize=12)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(f"figure: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
