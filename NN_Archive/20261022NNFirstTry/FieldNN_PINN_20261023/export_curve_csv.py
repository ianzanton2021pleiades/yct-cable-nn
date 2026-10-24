"""Export every series drawn in the 150-epoch curve figure to one CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


BASELINE_50 = Path(r"D:\GitRepository\Cable-NN\NN_Archive\20261022NNFirstTry\FieldNN_Hybrid_20261022\output\pretrain50\unet\history.csv")
BASELINE_100 = Path(r"D:\GitRepository\Cable-NN\NN_Archive\20261022NNFirstTry\FieldNN_Hybrid_20261022\output\pretrain100\unet\history.csv")

COLUMNS = [
    "epoch",
    "pinn_train_loss_augmented",
    "pinn_val_loss",
    "pinn_val_f1_5m",
    "pinn_val_f1_2m",
    "pinn_val_terminal_f1",
    "pinn_train_physics",
    "pinn_val_physics",
    "baseline50_val_loss",
    "baseline50_val_f1_5m",
    "baseline100_val_loss",
    "baseline100_val_f1_5m",
]


def read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def value(rows: list[dict[str, str]], index: int, key: str) -> float | None:
    if index >= len(rows):
        return None
    raw = rows[index].get(key, "")
    if raw in ("", None):
        return None
    return float(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pinn = read(args.history)
    base50 = read(BASELINE_50)
    base100 = read(BASELINE_100)

    rows = []
    for index in range(max(len(pinn), len(base50), len(base100))):
        record = {
            "epoch": index + 1,
            "pinn_train_loss_augmented": value(pinn, index, "train_supervision"),
            "pinn_val_loss": value(pinn, index, "val_supervision"),
            "pinn_val_f1_5m": value(pinn, index, "val_f1"),
            "pinn_val_f1_2m": value(pinn, index, "val_f1_2m"),
            "pinn_val_terminal_f1": value(pinn, index, "val_terminal_f1"),
            "pinn_train_physics": value(pinn, index, "train_physics"),
            "pinn_val_physics": value(pinn, index, "val_physics"),
            "baseline50_val_loss": value(base50, index, "val_loss"),
            "baseline50_val_f1_5m": value(base50, index, "val_f1"),
            "baseline100_val_loss": value(base100, index, "val_loss"),
            "baseline100_val_f1_5m": value(base100, index, "val_f1"),
        }
        rows.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        for record in rows:
            writer.writerow({
                key: ("" if record[key] is None else (f"{record[key]:.6f}" if isinstance(record[key], float) else record[key]))
                for key in COLUMNS
            })
    print(f"{len(rows)} rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
