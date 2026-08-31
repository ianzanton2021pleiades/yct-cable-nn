from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "Src" / "DG_dataset_max2.5km.py"
spec = importlib.util.spec_from_file_location("dg_dataset_max2p5km", MODULE_PATH)
dg = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = dg
spec.loader.exec_module(dg)

print("case,end_peak,mid_p95,near_p95,step_pre,step_inside,step_after")
for row in dg.generate_defect_case_rows(seed=20260623):
    if row["profile"] != "field":
        continue
    d = row["d1"]
    impulse = np.real(row["imp1"])
    step = np.real(row["step1"])
    length = row["cable"].total_length
    end_mask = (d > length - 35.0) & (d < length + 35.0)
    mid_mask = (d > max(80.0, length * 0.2)) & (d < length - 80.0)
    near_mask = d < 80.0
    end_peak = float(np.nanmax(np.abs(impulse[end_mask]))) if end_mask.any() else 0.0
    mid_p95 = float(np.nanpercentile(np.abs(impulse[mid_mask]), 95)) if mid_mask.any() else 0.0
    near_p95 = float(np.nanpercentile(np.abs(impulse[near_mask]), 95)) if near_mask.any() else 0.0
    step_pre = float("nan")
    step_inside = float("nan")
    step_after = float("nan")
    defects = row["cable"].defect_info
    if defects:
        defect = defects[-1]
        pre_mask = (d > max(0.0, defect["start"] - 160.0)) & (d < defect["start"] - 30.0)
        in_mask = (d > defect["start"] + 30.0) & (d < defect["end"] - 30.0)
        after_mask = (d > defect["end"] + 30.0) & (d < min(length, defect["end"] + 220.0))
        if pre_mask.any():
            step_pre = float(np.nanmedian(step[pre_mask]))
        if in_mask.any():
            step_inside = float(np.nanmedian(step[in_mask]))
        if after_mask.any():
            step_after = float(np.nanmedian(step[after_mask]))
    print(
        f"{row['case_name']},{end_peak:.6g},{mid_p95:.6g},{near_p95:.6g},"
        f"{step_pre:.6g},{step_inside:.6g},{step_after:.6g}"
    )
