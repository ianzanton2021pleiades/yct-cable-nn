"""Assemble the numbers quoted in the technical note from the real history files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

BASE50 = Path(r"D:\GitRepository\Cable-NN\NN_Archive\20261022NNFirstTry\FieldNN_Hybrid_20261022\output\pretrain50\unet\history.csv")
BASE100 = Path(r"D:\GitRepository\Cable-NN\NN_Archive\20261022NNFirstTry\FieldNN_Hybrid_20261022\output\pretrain100\unet\history.csv")


def read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def at(rows: list[dict[str, str]], epoch: int, key: str) -> float:
    return float(rows[epoch - 1][key])


def summarize(rows: list[dict[str, str]], train_key: str, val_key: str) -> dict:
    epochs = [int(row["epoch"]) for row in rows]

    def block(epoch: int) -> dict:
        return {
            "train": at(rows, epoch, train_key),
            "val": at(rows, epoch, val_key),
            "gap": at(rows, epoch, val_key) - at(rows, epoch, train_key),
            "f1": at(rows, epoch, "val_f1"),
        }

    first = epochs[0]
    last = epochs[-1]
    return {
        "epochs": len(rows),
        "first": block(first),
        "at50": block(50) if 50 in epochs else None,
        "at100": block(100) if 100 in epochs else None,
        "last": block(last),
        "val_min": min(float(row[val_key]) for row in rows),
        "val_min_epoch": min(rows, key=lambda row: float(row[val_key]))["epoch"],
        "train_min": min(float(row[train_key]) for row in rows),
        "f1_max": max(float(row["val_f1"]) for row in rows),
        "f1_max_epoch": max(rows, key=lambda row: float(row["val_f1"]))["epoch"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mine", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    mine = read(args.mine)
    result = {
        "baseline50": summarize(read(BASE50), "train_loss", "val_loss"),
        "baseline100": summarize(read(BASE100), "train_loss", "val_loss"),
        "pinn": summarize(mine, "train_supervision", "val_supervision"),
    }
    physics = [float(row["val_physics"]) for row in mine if row.get("val_physics")]
    if physics:
        result["physics_val"] = {"first": physics[0], "at50": physics[49] if len(physics) >= 50 else None,
                                 "last": physics[-1]}
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
