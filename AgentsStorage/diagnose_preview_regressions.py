from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "Src" / "DG_dataset_max2.5km.py"


def load_dg_module():
    spec = importlib.util.spec_from_file_location("dg_dataset_max2p5km", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def terminal_metrics(row: dict) -> dict:
    distance = row["d1"]
    impulse = np.abs(row["imp1"].real)
    length = float(row["cable"].total_length)
    end_mask = (distance > length - 80.0) & (distance < length + 120.0)
    pre_end_mask = (distance > length - 180.0) & (distance < length - 40.0)
    if not end_mask.any():
        return {}
    local_idx = int(np.nanargmax(impulse[end_mask]))
    end_distance = distance[end_mask]
    end_impulse = impulse[end_mask]
    return {
        "terminal_peak_distance": float(end_distance[local_idx]),
        "terminal_peak": float(end_impulse[local_idx]),
        "pre_end_p95": float(np.nanpercentile(impulse[pre_end_mask], 95)) if pre_end_mask.any() else float("nan"),
        "expected_length": length,
    }


def moisture_step_metrics(row: dict) -> dict:
    defect = [d for d in row["cable"].defect_info if d["type"] == "moisture_distributed"][0]
    distance = row["d2"]
    step = row["step2"].real
    masks = {
        "pre": (distance > defect["start"] - 180.0) & (distance < defect["start"] - 40.0),
        "early": (distance > defect["start"] + 80.0) & (distance < defect["start"] + 260.0),
        "mid": (distance > defect["start"] + 450.0) & (distance < defect["end"] - 450.0),
        "late": (distance > defect["end"] - 280.0) & (distance < defect["end"] - 80.0),
        "terminal": (distance > row["cable"].total_length - 80.0) & (distance < row["cable"].total_length + 120.0),
    }
    return {
        name: float(np.nanmedian(step[mask])) if mask.any() else float("nan")
        for name, mask in masks.items()
    }


def row_for_case(dg, case_name: str, use_dirty: bool = True) -> dict:
    return next(row for row in dg.generate_defect_case_rows(seed=20260623, use_dirty=use_dirty) if row["case_name"] == case_name)


if __name__ == "__main__":
    dg = load_dg_module()
    for dirty in [False, True]:
        short_row = row_for_case(dg, "Field short", use_dirty=dirty)
        print("Field short", "dirty" if dirty else "clean", terminal_metrics(short_row))

    moisture_row = row_for_case(dg, "Field moisture_distributed", use_dirty=True)
    print("Moisture dirty", moisture_step_metrics(moisture_row))
