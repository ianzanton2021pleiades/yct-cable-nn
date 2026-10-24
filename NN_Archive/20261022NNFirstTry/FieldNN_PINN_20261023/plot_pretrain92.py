"""92-epoch pretraining curves in the same layout as the CMDC pretrain100 figure."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.sans-serif"] = ["Times New Roman", "SimHei"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.linewidth"] = 0.9
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True

STEPS_PER_EPOCH = 150


def read(path: Path, limit: int = 0) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return rows[:limit] if limit else rows


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if values.size <= window:
        return values
    kernel = np.ones(window) / window
    padded = np.concatenate((np.full(window // 2, values[0]), values, np.full(window // 2, values[-1])))
    return np.convolve(padded, kernel, mode="valid")[: values.size]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1", type=Path, required=True)
    parser.add_argument("--phase2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase1-epochs", type=int, default=50)
    parser.add_argument("--phase2-epochs", type=int, default=42)
    args = parser.parse_args()

    hist1 = read(args.phase1 / "history50.csv", args.phase1_epochs)
    hist2 = read(args.phase2 / "history.csv", args.phase2_epochs)
    for row in hist2:
        row["epoch"] = str(int(row["epoch"]) + len(hist1))
    history = hist1 + hist2

    steps1 = read(args.phase1 / "steps.csv", args.phase1_epochs * STEPS_PER_EPOCH)
    steps2 = read(args.phase2 / "steps.csv", args.phase2_epochs * STEPS_PER_EPOCH)
    for row in steps2:
        row["step"] = str(int(row["step"]) + len(steps1))
    steps = steps1 + steps2

    train_step = np.array([float(row["step"]) for row in steps])
    train_loss = np.array([float(row["loss"]) for row in steps])
    epoch = np.array([int(row["epoch"]) for row in history])
    val_loss = np.array([float(row["val_supervision"]) for row in history])
    val_f1 = np.array([float(row["val_f1"]) for row in history])
    val_f1_2 = np.array([float(row["val_f1_2m"]) for row in history])
    val_term = np.array([float(row["val_terminal_f1"]) for row in history])
    physics = np.array([float(row["val_physics"]) for row in history])
    val_step = epoch * STEPS_PER_EPOCH

    scores = (1.0 - val_f1) + 0.1 * val_loss
    best_index = int(np.argmin(scores))

    figure, axes = plt.subplots(2, 1, figsize=(11.0, 7.2), sharex=True)
    axis = axes[0]
    axis.plot(train_step, smooth(train_loss, 60), color="#1f77b4", linewidth=1.1,
              label="training loss (smoothed)")
    axis.plot(val_step, val_loss, color="#ff7f0e", linewidth=1.4, marker="o", markersize=3.2,
              label="validation loss")
    axis.axvline(val_step[best_index], color="#2ca02c", linestyle=":", linewidth=1.0,
                 label=f"selected step {int(val_step[best_index])}")
    axis.axvline(len(hist1) * STEPS_PER_EPOCH, color="0.45",
                 linestyle="--", linewidth=0.9, label="phase-2 restart")
    axis.set_ylabel("loss")
    axis.set_title("Synthetic pretraining: train / validation loss", fontsize=10)
    axis.legend(frameon=True, fontsize=8, loc="upper right")
    axis.grid(alpha=0.25)
    axis.set_ylim(0.0, float(max(smooth(train_loss, 60).max(), val_loss.max())) * 1.06)

    axis = axes[1]
    axis.plot(val_step, val_f1, color="#1f77b4", linewidth=1.4, marker="o", markersize=3.2,
              label="validation event F1@5m")
    axis.plot(val_step, val_f1_2, color="#17becf", linewidth=1.2, marker="^", markersize=2.8,
              label="validation event F1@2m")
    axis.plot(val_step, val_term, color="#2ca02c", linewidth=1.4, marker="*", markersize=4.0,
              label="validation terminal F1")
    axis.axvline(val_step[best_index], color="#2ca02c", linestyle=":", linewidth=1.0)
    axis.axvline(len(hist1) * STEPS_PER_EPOCH, color="0.45", linestyle="--", linewidth=0.9)
    axis.set_xlabel("training step")
    axis.set_ylabel("F1")
    axis.set_title("Synthetic pretraining: accuracy (F1) curves", fontsize=10)
    axis.legend(frameon=True, fontsize=8, loc="lower right")
    axis.grid(alpha=0.25)
    axis.set_ylim(0.0, 1.02)

    figure.suptitle(
        f"PINN U-Net: {len(history)}-epoch synthetic pretraining "
        f"({len(hist1)} + {len(hist2)} epochs)", fontsize=12
    )
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)

    second = plt.figure(figsize=(6.0, 3.2))
    ax = second.add_subplot(111)
    ax.plot(epoch, physics, color="#6a1b9a", linewidth=1.5, marker="o", markersize=3.0)
    ax.set_xlabel("epoch")
    ax.set_ylabel("physics residual")
    ax.set_title("Forward-model residual (validation)", fontsize=10)
    ax.grid(alpha=0.25)
    second.tight_layout()
    second.savefig(args.output.with_name(args.output.stem + "_physics.png"), dpi=200)
    plt.close("all")

    summary = {
        "epochs": len(history),
        "best_epoch": int(epoch[best_index]),
        "best_step": int(val_step[best_index]),
        "best_val_f1": float(val_f1[best_index]),
        "best_val_loss": float(val_loss[best_index]),
        "at_50": {"val_loss": float(val_loss[49]), "val_f1": float(val_f1[49]),
                  "val_f1_2m": float(val_f1_2[49]), "terminal": float(val_term[49])},
        "at_92": {"val_loss": float(val_loss[-1]), "val_f1": float(val_f1[-1]),
                  "val_f1_2m": float(val_f1_2[-1]), "terminal": float(val_term[-1])},
        "min_val_loss": float(val_loss.min()), "min_val_loss_epoch": int(epoch[int(np.argmin(val_loss))]),
        "max_val_f1": float(val_f1.max()), "max_val_f1_epoch": int(epoch[int(np.argmax(val_f1))]),
        "physics_first": float(physics[0]), "physics_last": float(physics[-1]),
        "physics_min": float(physics.min()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    marks = [1, 10, 20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 90, 92]
    print("\n epoch  val_loss  val_F1@5m  val_F1@2m  terminal  physics")
    for value in marks:
        if value <= len(history):
            index = value - 1
            print(f"{value:6d} {val_loss[index]:9.4f} {val_f1[index]:10.4f} "
                  f"{val_f1_2[index]:10.4f} {val_term[index]:9.4f} {physics[index]:8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
