"""Thin Iter6 entry point for the established Iter5 synthetic evaluator."""

from __future__ import annotations

import runpy
import sys
import argparse
import importlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
ITER3 = HERE.parent / "FieldNN_Iter3_20261022"
ITER5 = HERE.parent / "FieldNN_Iter5_20261022"
CODEX = HERE.parent / "FieldNN_20261022"

sys.path[:0] = [str(ITER5), str(ITER3), str(CODEX)]


def main() -> None:
    limit_parser = argparse.ArgumentParser(add_help=False)
    limit_parser.add_argument("--limit", type=int)
    options, remaining = limit_parser.parse_known_args(sys.argv[1:])
    if options.limit is not None and options.limit < 1:
        limit_parser.error("--limit must be positive")

    if options.limit is not None:
        data_module = importlib.import_module("data")
        parent_dataset = data_module.ShardDataset
        limit = options.limit

        class LimitedShardDataset(parent_dataset):
            def __len__(self) -> int:
                return min(super().__len__(), limit)

        data_module.ShardDataset = LimitedShardDataset

    target = ITER5 / "evaluate.py"
    sys.argv = [str(target), *remaining]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
