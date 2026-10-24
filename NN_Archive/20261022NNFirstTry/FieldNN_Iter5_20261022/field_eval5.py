"""Run Iter3 field inference with the Iter5 near-end-suppressing decoder."""

from __future__ import annotations

import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ITER3 = HERE.parent / "FieldNN_Iter3_20261022"
sys.path.insert(0, str(ITER3))
sys.path.insert(0, str(HERE))

from decode import NEAR_END_SUPPRESSION_M, decode_predictions

import field_eval as fe

fe.decode_predictions = decode_predictions


def main() -> int:
    args = sys.argv[1:]
    sys.argv = [str(ITER3 / "field_eval.py"), *args]
    code = fe.main()
    output = None
    if "--output" in args:
        output = Path(args[args.index("--output") + 1])
    else:
        checkpoint = Path(args[args.index("--checkpoint") + 1])
        axis = args[args.index("--distance-axis") + 1] if "--distance-axis" in args else "source"
        output = ITER3 / "output" / f"field_eval_{checkpoint.stem}_{axis}"
    summary_path = output / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["near_end_suppression_m"] = NEAR_END_SUPPRESSION_M
        summary["decode_policy"] = (
            "Event and interval candidates within the first "
            f"{NEAR_END_SUPPRESSION_M:.0f} m of the response axis are suppressed as "
            "fixture-dominated; terminal detection and all other candidates are unchanged. "
            "The model input is never cropped."
        )
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
