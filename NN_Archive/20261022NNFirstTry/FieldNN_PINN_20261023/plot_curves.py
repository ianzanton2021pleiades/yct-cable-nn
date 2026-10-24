"""Figures: 50/100-epoch comparison against the CMDC baseline plus physics residuals."""

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

BASELINE_50 = Path(r"D:\GitRepository\Cable-NN\NN_Archive\20261022NNFirstTry\FieldNN_Hybrid_20261022\output\pretrain50\unet\history.csv")
BASELINE_100 = Path(r"D:\GitRepository\Cable-NN\NN_Archive\20261022NNFirstTry\FieldNN_Hybrid_20261022\output\pretrain100\unet\history.csv")
BASELINE_STEPS_50 = Path(r"D:\GitRepository\Cable-NN\NN_Archive\20261022NNFirstTry\FieldNN_Hybrid_20261022\output\pretrain50\unet\steps.csv")
BASELINE_STEPS_100 = Path(r"D:\GitRepository\Cable-NN\NN_Archive\20261022NNFirstTry\FieldNN_Hybrid_20261022\output\pretrain100\unet\steps.csv")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    return np.array([float(row[name]) for row in rows], dtype=np.float64)


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size <= window:
        return values
    kernel = np.ones(window) / window
    padded = np.concatenate((np.full(window // 2, values[0]), values, np.full(window // 2, values[-1])))
    return np.convolve(padded, kernel, mode="valid")[: values.size]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--steps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", type=str, default="PINN")
    args = parser.parse_args()

    mine = read_csv(args.history)
    mine_steps = read_csv(args.steps)
    base50 = read_csv(BASELINE_50)
    base100 = read_csv(BASELINE_100)

    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

    axis = axes[0]
    if base50:
        axis.plot(column(base50, "epoch"), smooth(column(base50, "val_loss"), 3),
                  color="#9e9e9e", linewidth=1.4, linestyle="--", label="baseline val (50 ep)")
    if base100:
        axis.plot(column(base100, "epoch"), smooth(column(base100, "val_loss"), 3),
                  color="#616161", linewidth=1.4, linestyle=":", label="baseline val (100 ep)")
    axis.plot(column(mine, "epoch"), smooth(column(mine, "val_supervision"), 3),
              color="#1f4e9c", linewidth=1.8, label="PINN val")
    axis.plot(column(mine, "epoch"), smooth(column(mine, "train_supervision"), 3),
              color="#c62828", linewidth=1.2, alpha=0.85, label="PINN train")
    axis.set_xlabel("epoch")
    axis.set_ylabel("supervision loss")
    axis.set_title("Supervision loss (identical definition)")
    axis.legend(frameon=False, fontsize=8)
    axis.grid(alpha=0.25)

    axis = axes[1]
    if base50:
        axis.plot(column(base50, "epoch"), column(base50, "val_f1"),
                  color="#9e9e9e", linewidth=1.4, linestyle="--", label="baseline val (50 ep)")
    if base100:
        axis.plot(column(base100, "epoch"), column(base100, "val_f1"),
                  color="#616161", linewidth=1.4, linestyle=":", label="baseline val (100 ep)")
    axis.plot(column(mine, "epoch"), column(mine, "val_f1"),
              color="#1f4e9c", linewidth=1.8, label="PINN val")
    axis.plot(column(mine, "epoch"), column(mine, "val_f1_2m"),
              color="#2e7d32", linewidth=1.2, label="PINN val @2 m")
    axis.set_xlabel("epoch")
    axis.set_ylabel("event F1")
    axis.set_title("Validation event F1")
    axis.set_ylim(0.0, 1.0)
    axis.legend(frameon=False, fontsize=8)
    axis.grid(alpha=0.25)

    axis = axes[2]
    axis.plot(column(mine, "epoch"), column(mine, "val_physics"),
              color="#6a1b9a", linewidth=1.6, label="val physics residual")
    axis.plot(column(mine, "epoch"), column(mine, "train_physics"),
              color="#ab47bc", linewidth=1.1, alpha=0.85, label="train physics residual")
    axis.set_xlabel("epoch")
    axis.set_ylabel("physics loss")
    axis.set_title("Forward-model residual")
    axis.legend(frameon=False, fontsize=8)
    axis.grid(alpha=0.25)

    for axis in axes:
        for spine in axis.spines.values():
            spine.set_visible(True)
    figure.suptitle(f"{args.label}: 100-epoch synthetic pretraining", fontsize=12)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)
    plt.close(figure)

    summary = {
        "epochs": int(column(mine, "epoch")[-1]),
        "train_supervision_last": float(column(mine, "train_supervision")[-1]),
        "val_supervision_last": float(column(mine, "val_supervision")[-1]),
        "gap_last": float(column(mine, "val_supervision")[-1] - column(mine, "train_supervision")[-1]),
        "val_f1_last": float(column(mine, "val_f1")[-1]),
        "best_val_f1": float(column(mine, "best_f1")[-1]),
        "physics_last": float(column(mine, "val_physics")[-1]),
    }
    if len(mine) >= 50:
        summary.update({
            "train_supervision_50": float(column(mine, "train_supervision")[49]),
            "val_supervision_50": float(column(mine, "val_supervision")[49]),
            "val_f1_50": float(column(mine, "val_f1")[49]),
            "val_supervision_min": float(column(mine, "val_supervision").min()),
        })
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
