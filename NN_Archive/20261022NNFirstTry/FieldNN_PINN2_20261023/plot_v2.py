"""Loss, physics-residual and F1 curves for PINN-v2 against the PINN-v1 baseline."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True

HERE = Path(__file__).resolve().parent
PINN1 = HERE.parents[0] / "FieldNN_PINN_20261023" / "output" / "hist_150" / "unet" / "history.csv"


def read_history(path: Path) -> dict[str, list[float]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    columns: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if value in ("", "None", None):
                columns.setdefault(key, []).append(float("nan"))
            else:
                columns.setdefault(key, []).append(float(value))
    return columns


def family(axis, title: str, ylabel: str, path: Path) -> None:
    axis.set_title(title, fontsize=10)
    axis.set_xlabel("Epoch", fontsize=9)
    axis.set_ylabel(ylabel, fontsize=9)
    axis.tick_params(labelsize=8)
    axis.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    axis.legend(fontsize=8, frameon=False)
    axis.figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(axis.figure)
    print("written", path, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2", type=Path, default=HERE / "output" / "v2_100" / "unet" / "history.csv")
    parser.add_argument("--v1", type=Path, default=PINN1)
    parser.add_argument("--output", type=Path, default=HERE / "assets")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    v2 = read_history(args.v2)
    epochs_v2 = v2["epoch"][: args.limit]
    v1 = read_history(args.v1) if args.v1.exists() else None
    epochs_v1 = v1["epoch"][: args.limit] if v1 else []

    figure, axis = plt.subplots(figsize=(4.6, 3.2))
    axis.plot(epochs_v1, v1["train_supervision"][: args.limit], color="#9a9a9a", linewidth=1.1,
              linestyle="--", label="PINN-v1 train")
    axis.plot(epochs_v1, v1["val_supervision"][: args.limit], color="#c0504d", linewidth=1.1,
              linestyle="--", label="PINN-v1 val")
    axis.plot(epochs_v2, v2["train_supervision"], color="#4f81bd", linewidth=1.3, label="PINN-v2 train")
    axis.plot(epochs_v2, v2["val_supervision"], color="#1f3864", linewidth=1.3, label="PINN-v2 val")
    family(axis, "Supervision loss (event + interval + terminal)", "Loss",
           args.output / "loss_curves.png")

    figure, axis = plt.subplots(figsize=(4.6, 3.2))
    axis.plot(epochs_v1, v1["train_loss"][: args.limit], color="#9a9a9a", linewidth=1.1,
              linestyle="--", label="PINN-v1 train")
    axis.plot(epochs_v1, v1["val_loss"][: args.limit], color="#c0504d", linewidth=1.1,
              linestyle="--", label="PINN-v1 val")
    axis.plot(epochs_v2, v2["train_total"], color="#4f81bd", linewidth=1.3, label="PINN-v2 train")
    axis.plot(epochs_v2, v2["val_total"], color="#1f3864", linewidth=1.3, label="PINN-v2 val")
    family(axis, "Total loss (supervision + physics)", "Loss", args.output / "total_curves.png")

    figure, axis = plt.subplots(figsize=(4.6, 3.2))
    axis.plot(epochs_v1, v1["train_physics"][: args.limit], color="#9a9a9a", linewidth=1.1,
              linestyle="--", label="PINN-v1 train")
    axis.plot(epochs_v1, v1["val_physics"][: args.limit], color="#c0504d", linewidth=1.1,
              linestyle="--", label="PINN-v1 val")
    axis.plot(epochs_v2, v2["train_physics"], color="#4f81bd", linewidth=1.3, label="PINN-v2 train")
    axis.plot(epochs_v2, v2["val_physics"], color="#1f3864", linewidth=1.3, label="PINN-v2 val")
    family(axis, "Physics residual", "Weighted residual", args.output / "physics_curves.png")

    figure, axis = plt.subplots(figsize=(4.6, 3.2))
    finite_v1 = [(epoch, value) for epoch, value in zip(epochs_v1, v1["val_f1"][: args.limit])
                 if value == value]
    finite_v2 = [(epoch, value) for epoch, value in zip(epochs_v2, v2["val_f1"])
                 if value == value]
    axis.plot([item[0] for item in finite_v1], [item[1] for item in finite_v1],
              color="#c0504d", linewidth=1.1, linestyle="--", label="PINN-v1 val")
    axis.plot([item[0] for item in finite_v2], [item[1] for item in finite_v2],
              color="#1f3864", linewidth=1.3, marker="o", markersize=2.5, label="PINN-v2 val")
    family(axis, "Event F1 at 5 m on the mixed validation view", "F1",
           args.output / "f1_curves.png")

    figure, axis = plt.subplots(figsize=(4.6, 3.2))
    axis.plot(epochs_v2, v2["val_event"], color="#1f3864", linewidth=1.3, label="event")
    axis.plot(epochs_v2, v2["val_background"], color="#c0504d", linewidth=1.1, label="background top-128")
    axis.plot(epochs_v2, v2["val_physics"], color="#9bbb59", linewidth=1.1, label="physics")
    family(axis, "PINN-v2 validation components", "Loss", args.output / "components_curves.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
