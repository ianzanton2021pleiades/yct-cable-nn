"""Plot pretraining and measured-field adaptation loss curves."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]


def read_history(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def values(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def smooth(values_: np.ndarray, window: int = 3) -> np.ndarray:
    if values_.size < window:
        return values_
    kernel = np.ones(window, dtype=float) / window
    padded = np.r_[values_[0], values_, values_[-1]]
    return np.convolve(padded, kernel, mode="valid")


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
    parser.add_argument("--pretrain", type=Path, default=PROJECT_ROOT / "NN_Archive/20261022NNFirstTry/FieldNN_Iter5_20261022/output/iter5/unet/history.csv")
    parser.add_argument("--finetune", type=Path, default=HERE / "output/hybrid_long/unet/history.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "Image/20261022_FieldNN_Hybrid_loss_curves.png")
    args = parser.parse_args()

    style()
    import matplotlib.pyplot as plt

    pretrain = read_history(args.pretrain)
    finetune = read_history(args.finetune)
    epochs_pre = values(pretrain, "epoch")
    epochs_fine = values(finetune, "epoch")
    scores = np.minimum(values(finetune, "score_raw"), values(finetune, "score_ema"))
    best_index = int(np.argmin(scores))
    best_epoch = float(epochs_fine[best_index])

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    axis = axes[0, 0]
    axis.plot(epochs_pre, values(pretrain, "train_loss"), color="tab:blue", linewidth=1.2, label="train")
    axis.plot(epochs_pre, values(pretrain, "val_loss"), color="tab:orange", linewidth=1.2, label="validation")
    axis.plot(epochs_pre, smooth(values(pretrain, "train_loss")), color="tab:blue", linewidth=0.7, linestyle="--", alpha=0.7)
    axis.plot(epochs_pre, smooth(values(pretrain, "val_loss")), color="tab:orange", linewidth=0.7, linestyle="--", alpha=0.7)
    axis.set_title("Synthetic pretraining (Iter5 U-Net)", fontsize=10)
    axis.set_xlabel("epoch")
    axis.set_ylabel("loss")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=7, frameon=True)

    axis = axes[0, 1]
    axis.plot(epochs_fine, values(finetune, "train_syn_loss"), color="tab:blue", linewidth=1.2, label="train synthetic")
    axis.plot(epochs_fine, values(finetune, "val_syn"), color="tab:orange", linewidth=1.2, label="validation synthetic")
    axis.axvline(best_epoch, color="tab:green", linestyle="--", linewidth=0.9, label=f"selected epoch {best_epoch:.0f}")
    axis.axvspan(best_epoch, epochs_fine[-1] + 0.5, color="tab:red", alpha=0.06)
    axis.set_title("Measured-field fine-tuning: synthetic stream", fontsize=10)
    axis.set_xlabel("epoch")
    axis.set_ylabel("loss")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=7, frameon=True)

    axis = axes[1, 0]
    axis.plot(epochs_fine, values(finetune, "train_hyb_loss"), color="tab:blue", linewidth=1.2, label="train injected")
    axis.plot(epochs_fine, values(finetune, "val_hyb"), color="tab:orange", linewidth=1.2, label="validation injected")
    axis.axvline(best_epoch, color="tab:green", linestyle="--", linewidth=0.9, label=f"selected epoch {best_epoch:.0f}")
    axis.axvspan(best_epoch, epochs_fine[-1] + 0.5, color="tab:red", alpha=0.06)
    axis.set_title("Measured-field fine-tuning: injected stream", fontsize=10)
    axis.set_xlabel("epoch")
    axis.set_ylabel("loss")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=7, frameon=True)

    axis = axes[1, 1]
    axis.plot(epochs_fine, values(finetune, "field_hit20"), color="tab:green", marker="o", markersize=3, label="field hit@20m")
    axis.plot(epochs_fine, values(finetune, "hyb_hit20"), color="tab:purple", marker="s", markersize=3, linestyle="--", label="injected hit@20m")
    axis.axvline(best_epoch, color="tab:green", linestyle="--", linewidth=0.9)
    axis.axvspan(best_epoch, epochs_fine[-1] + 0.5, color="tab:red", alpha=0.06)
    axis.set_xlabel("epoch")
    axis.set_ylabel("hit rate")
    axis.set_ylim(0.0, 0.8)
    axis.grid(alpha=0.2)
    axis2 = axis.twinx()
    axis2.plot(epochs_fine, values(finetune, "field_cand"), color="tab:red", marker="^", markersize=3, linestyle=":", label="field candidates/curve")
    axis2.set_ylabel("candidates / curve")
    lines1, labels1 = axis.get_legend_handles_labels()
    lines2, labels2 = axis2.get_legend_handles_labels()
    axis.legend(lines1 + lines2, labels1 + labels2, fontsize=6.5, frameon=True, loc="upper left")
    axis.set_title("Field validation: hit and candidate density", fontsize=10)

    for axis in axes.flat:
        axis.tick_params(top=True, right=True)
    fig.suptitle("FieldNN Hybrid: pretraining and measured-field adaptation", fontsize=12)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(f"figure: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
