"""Thin Iter6 entry point for the existing full-grid Iter5 field inference."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ITER5 = HERE.parent / "FieldNN_Iter5_20261022"
ITER3 = HERE.parent / "FieldNN_Iter3_20261022"

sys.path[:0] = [str(ITER5), str(ITER3)]


def main() -> None:
    target = ITER5 / "field_eval5.py"
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
