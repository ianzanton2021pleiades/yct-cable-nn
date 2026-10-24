"""Stitch consecutive training segments into one continuous history."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment", action="append", required=True,
                        help="dir=count; count 0 keeps every row, otherwise keep the first count rows")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    fields: list[str] | None = None
    history: list[dict[str, str]] = []
    steps: list[dict[str, str]] = []
    step_fields: list[str] | None = None
    step_offset = 0
    epoch_offset = 0

    for spec in args.segment:
        directory, _, count = spec.partition("=")
        folder = Path(directory)
        limit = int(count or 0)
        history_file = folder / "history50.csv"
        if not history_file.is_file():
            history_file = folder / "history.csv"
        fields_local, rows = read(history_file)
        if limit:
            rows = rows[:limit]
        fields = fields or fields_local
        for row in rows:
            item = {key: row.get(key, "") for key in fields}
            item["epoch"] = str(int(row["epoch"]) + epoch_offset)
            history.append(item)

        step_file = folder / "steps.csv"
        if step_file.is_file():
            step_fields_local, step_rows = read(step_file)
            step_fields = step_fields or step_fields_local
            for row in step_rows:
                row = dict(row)
                row["step"] = str(int(row["step"]) + step_offset)
                row["epoch"] = str(int(row["epoch"]) + epoch_offset)
                steps.append(row)
            step_offset += len(step_rows)
        epoch_offset += len(rows)

    output = args.output_dir / "unet"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    if step_fields and steps:
        trimmed = [{key: row.get(key, "") for key in step_fields} for row in steps]
        with (output / "steps.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=step_fields)
            writer.writeheader()
            writer.writerows(trimmed)

    print(f"combined epochs {len(history)} steps {len(steps)} -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
