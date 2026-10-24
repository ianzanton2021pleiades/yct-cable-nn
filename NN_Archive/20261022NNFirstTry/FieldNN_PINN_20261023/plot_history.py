"""Epoch-based training curves with baseline overlays, plus a physics panel."""

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

BASE50 = Path(r"D:\GitRepository\Cable-NN\NN_Archive\20261022NNFirstTry\FieldNN_Hybrid_20261022\output\pretrain50\unet\history.csv")
BASE100 = Path(r"D:\GitRepository\Cable-NN\NN_Archive\20261022NNFirstTry\FieldNN_Hybrid_20261022\output\pretrain100\unet\history.csv")


def read(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def col(rows: list[dict[str, str]], name: str) -> np.ndarray:
    values = []
    for row in rows:
        raw = row.get(name, "")
        values.append(float(raw) if raw not in ("", None) else np.nan)
    return np.array(values, dtype=np.float64)


def solid(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    index = np.arange(values.size)
    keep = np.isfinite(values)
    return index[keep], values[keep]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", type=str, default="PINN U-Net")
    parser.add_argument("--boundaries", type=str, default="",
                        help="comma separated epoch numbers where a new cosine phase started")
    args = parser.parse_args()

    mine = read(args.history)
    base50 = read(BASE50)
    base100 = read(BASE100)
    epoch = col(mine, "epoch")

    figure, axes = plt.subplots(1, 3, figsize=(16.0, 4.6))

    axis = axes[0]
    if base50:
        x, y = solid(col(base50, "val_loss"))
        axis.plot(x, y, color="#9e9e9e", linewidth=1.3, linestyle="--", label="baseline 50 ep")
    if base100:
        x, y = solid(col(base100, "val_loss"))
        axis.plot(x, y, color="#616161", linewidth=1.3, linestyle=":", label="baseline 100 ep")
    axis.plot(epoch, col(mine, "val_supervision"), color="#1f4e9c", linewidth=1.6, label=f"{args.label} val")
    axis.plot(epoch, col(mine, "train_supervision"), color="#c62828", linewidth=1.0, alpha=0.8,
              label=f"{args.label} train (augmented)")
    for value in [item for item in args.boundaries.split(",") if item.strip()]:
        axis.axvline(float(value), color="0.5", linestyle="--", linewidth=0.8)
    axis.set_xlabel("epoch")
    axis.set_ylabel("supervision loss")
    axis.set_title("Supervision loss", fontsize=10)
    axis.legend(frameon=False, fontsize=8)
    axis.grid(alpha=0.25)

    axis = axes[1]
    if base50:
        x, y = solid(col(base50, "val_f1"))
        axis.plot(x, y, color="#9e9e9e", linewidth=1.3, linestyle="--", label="baseline 50 ep")
    if base100:
        x, y = solid(col(base100, "val_f1"))
        axis.plot(x, y, color="#616161", linewidth=1.3, linestyle=":", label="baseline 100 ep")
    axis.plot(epoch, col(mine, "val_f1"), color="#1f4e9c", linewidth=1.6, label=f"{args.label} F1@5m")
    axis.plot(epoch, col(mine, "val_f1_2m"), color="#2e7d32", linewidth=1.2, label=f"{args.label} F1@2m")
    axis.plot(epoch, col(mine, "val_terminal_f1"), color="#ef6c00", linewidth=1.2, label="terminal F1")
    for value in [item for item in args.boundaries.split(",") if item.strip()]:
        axis.axvline(float(value), color="0.5", linestyle="--", linewidth=0.8)
    axis.set_xlabel("epoch")
    axis.set_ylabel("F1")
    axis.set_title("Validation F1", fontsize=10)
    axis.set_ylim(0.0, 1.02)
    axis.legend(frameon=False, fontsize=8)
    axis.grid(alpha=0.25)

    axis = axes[2]
    axis.plot(epoch, col(mine, "val_physics"), color="#6a1b9a", linewidth=1.6, label="validation")
    axis.plot(epoch, col(mine, "train_physics"), color="#ab47bc", linewidth=1.0, alpha=0.8, label="training")
    for value in [item for item in args.boundaries.split(",") if item.strip()]:
        axis.axvline(float(value), color="0.5", linestyle="--", linewidth=0.8)
    axis.set_xlabel("epoch")
    axis.set_ylabel("physics residual")
    axis.set_title("Forward-model residual", fontsize=10)
    axis.legend(frameon=False, fontsize=8)
    axis.grid(alpha=0.25)

    figure.suptitle(f"{args.label}: {int(epoch[-1])} epochs", fontsize=12)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)
    plt.close(figure)

    val = col(mine, "val_supervision")
    val_f1 = col(mine, "val_f1")
    physics = col(mine, "val_physics")
    summary = {
        "epochs": int(epoch[-1]),
        "val_loss": float(val[-1]),
        "val_f1": float(val_f1[-1]),
        "val_f1_2m": float(col(mine, "val_f1_2m")[-1]),
        "terminal_f1": float(col(mine, "val_terminal_f1")[-1]),
        "min_val_loss": float(np.nanmin(val)),
        "min_val_loss_epoch": int(epoch[int(np.nanargmin(val))]),
        "max_val_f1": float(np.nanmax(val_f1)),
        "max_val_f1_epoch": int(epoch[int(np.nanargmax(val_f1))]),
        "mean_val_last10": float(np.nanmean(val[-10:])),
        "physics_first": float(physics[0]),
        "physics_last": float(physics[-1]),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    args.output.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
