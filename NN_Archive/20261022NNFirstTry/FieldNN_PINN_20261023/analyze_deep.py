"""Diagnostics for the finished PINN run: overfitting onset, component losses, field error structure."""

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
BUCKETS = [(500.0, 1000.0), (1000.0, 1500.0), (1500.0, 2000.0), (2000.0, 2600.0), (2600.0, 4000.0)]


def read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def col(rows: list[dict[str, str]], name: str) -> np.ndarray:
    values = []
    for row in rows:
        raw = row.get(name, "")
        values.append(float(raw) if raw not in ("", None) else np.nan)
    return np.array(values, dtype=np.float64)


def bucket_curve(entry: dict) -> list[dict]:
    per_curve = entry.get("field_test", {}).get("per_curve", [])
    output = []
    for low, high in BUCKETS:
        hits = truth = curves = 0
        for row in per_curve:
            length = row.get("known_length_m") or row.get("end_estimate_m")
            if not length or row.get("wet"):
                continue
            if not (low <= float(length) < high):
                continue
            hits += int(row["hits"])
            truth += len(row["truth"])
            curves += 1
        output.append({
            "range": f"{int(low)}-{int(high)}",
            "hits": hits,
            "truth": truth,
            "curves": curves,
            "hit_rate": hits / truth if truth else None,
        })
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--pretrain", type=Path, required=True)
    parser.add_argument("--field", type=Path, required=True)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    history = read(args.history)
    pretrain = json.loads(args.pretrain.read_text(encoding="utf-8"))
    field = json.loads(args.field.read_text(encoding="utf-8"))
    sweep = json.loads(args.sweep.read_text(encoding="utf-8"))
    base50 = read(BASE50)

    epoch = col(history, "epoch")
    val = col(history, "val_supervision")
    val_f1 = col(history, "val_f1")
    train = col(history, "train_supervision")
    best_index = int(np.nanargmin(val))
    onset_index = None
    for index in range(best_index + 1, len(val)):
        if np.isfinite(val[index]) and val[index] > val[best_index] + 0.05:
            onset_index = index
            break

    figure, axes = plt.subplots(2, 2, figsize=(14.5, 8.6))

    axis = axes[0][0]
    axis.plot(epoch, val, color="#1f4e9c", linewidth=1.6, label="validation loss")
    axis.plot(epoch, train, color="#c62828", linewidth=1.0, alpha=0.8, label="training loss (augmented)")
    if base50:
        axis.plot(col(base50, "epoch"), col(base50, "val_loss"), color="#9e9e9e",
                  linewidth=1.2, linestyle="--", label="baseline validation")
    axis.axvline(epoch[best_index], color="#2e7d32", linestyle=":", linewidth=1.1)
    for boundary in (50, 92):
        axis.axvline(boundary, color="0.55", linestyle="--", linewidth=0.8)
    axis.annotate(f"min {val[best_index]:.3f} @ {int(epoch[best_index])}",
                  xy=(epoch[best_index], val[best_index]),
                  xytext=(epoch[best_index] - 34, val[best_index] + 0.30),
                  fontsize=8, color="#2e7d32",
                  arrowprops=dict(arrowstyle="->", color="#2e7d32", linewidth=0.7))
    if onset_index is not None:
        axis.axvspan(epoch[onset_index], epoch[-1], color="#c62828", alpha=0.06)
        axis.annotate("loss rises after the restart",
                      xy=(epoch[onset_index] + 6, val[onset_index] + 0.20),
                      fontsize=8, color="#c62828")
    axis.set_xlabel("epoch")
    axis.set_ylabel("loss")
    axis.set_title("(a) supervision loss and overfitting onset", fontsize=10)
    axis.legend(frameon=False, fontsize=8)
    axis.grid(alpha=0.25)

    axis = axes[0][1]
    axis.plot(epoch, col(history, "val_event"), color="#1f4e9c", linewidth=1.4, label="event")
    axis.plot(epoch, col(history, "val_interval") * 10.0, color="#ef6c00", linewidth=1.4,
              label="interval x10")
    axis.plot(epoch, col(history, "val_terminal") * 10.0, color="#2e7d32", linewidth=1.4,
              label="terminal x10")
    axis.set_xlabel("epoch")
    axis.set_ylabel("validation loss component")
    axis.set_title("(b) which term drives the rise", fontsize=10)
    axis.legend(frameon=False, fontsize=8)
    axis.grid(alpha=0.25)

    axis = axes[1][0]
    curve = sweep["curve"]
    thresholds = [row["event_threshold"] for row in curve]
    axis.plot([row["candidates_per_curve"] for row in curve], [row["hit_rate_20"] for row in curve],
              color="#1f4e9c", marker="o", linewidth=1.5, label="PINN 150 field-tuned")
    for row in curve:
        axis.annotate(f"{row['event_threshold']:.2f}",
                      xy=(row["candidates_per_curve"], row["hit_rate_20"]),
                      xytext=(3, -9), textcoords="offset points", fontsize=7, color="#1f4e9c")
    axis.scatter([3.60], [0.426], marker="*", s=140, color="#616161", zorder=5,
                 label="baseline 0.5 point")
    axis.set_xlabel("candidates per curve")
    axis.set_ylabel("event hit@20m")
    axis.set_title("(c) operating curve on annotated field curves", fontsize=10)
    axis.legend(frameon=False, fontsize=8)
    axis.grid(alpha=0.25)

    axis = axes[1][1]
    pre_buckets = bucket_curve(pretrain)
    field_buckets = bucket_curve(field)
    index = np.arange(len(BUCKETS))
    width = 0.36
    axis.bar(index - width / 2, [row["hit_rate"] or 0.0 for row in pre_buckets], width,
             color="#78909c", label="pretrain 150")
    axis.bar(index + width / 2, [row["hit_rate"] or 0.0 for row in field_buckets], width,
             color="#1f4e9c", label="field-tuned")
    for position, row in enumerate(field_buckets):
        axis.text(position + width / 2, (row["hit_rate"] or 0.0) + 0.015, str(row["curves"]),
                  ha="center", fontsize=7, color="#333333")
    axis.set_xticks(index)
    axis.set_xticklabels([row["range"] for row in pre_buckets], fontsize=8)
    axis.set_xlabel("cable length (m), number above bar = curves")
    axis.set_ylabel("event hit@20m")
    axis.set_ylim(0.0, 0.65)
    axis.set_title("(d) field hit rate by cable length", fontsize=10)
    axis.legend(frameon=False, fontsize=8)
    axis.grid(axis="y", alpha=0.25)

    figure.suptitle("PINN U-Net 150-epoch diagnostics", fontsize=12)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)
    plt.close(figure)

    summary = {
        "epochs": int(epoch[-1]),
        "min_val_loss": float(val[best_index]),
        "min_val_loss_epoch": int(epoch[best_index]),
        "val_loss_final": float(val[-1]),
        "val_f1_at_min_loss": float(val_f1[best_index]),
        "val_f1_max": float(np.nanmax(val_f1)),
        "val_f1_max_epoch": int(epoch[int(np.nanargmax(val_f1))]),
        "overfit_onset_epoch": int(epoch[onset_index]) if onset_index is not None else None,
        "val_event_final": float(col(history, "val_event")[-1]),
        "val_interval_final": float(col(history, "val_interval")[-1]),
        "val_terminal_final": float(col(history, "val_terminal")[-1]),
        "field_buckets_pretrain": pre_buckets,
        "field_buckets_tuned": field_buckets,
        "sweep": curve,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    args.output.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
