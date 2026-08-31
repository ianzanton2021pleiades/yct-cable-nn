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


def summarize(row: dict) -> dict:
    defect = [d for d in row["cable"].defect_info if d["type"] == "moisture_distributed"][0]
    distance = row["d2"]
    step = row["step2"].real
    impulse_abs = np.abs(row["imp2"].real)
    masks = {
        "pre": (distance > defect["start"] - 180.0) & (distance < defect["start"] - 40.0),
        "early": (distance > defect["start"] + 40.0) & (distance < defect["start"] + 220.0),
        "inside": (distance > defect["start"] + 320.0) & (distance < defect["end"] - 180.0),
        "late": (distance > defect["end"] - 220.0) & (distance < defect["end"] - 40.0),
        "terminal": (distance > row["cable"].total_length - 60.0) & (distance < row["cable"].total_length + 80.0),
    }
    out = {}
    for name, mask in masks.items():
        out[f"step_{name}"] = float(np.nanmedian(step[mask])) if mask.any() else float("nan")
        out[f"imp_{name}_max"] = float(np.nanmax(impulse_abs[mask])) if mask.any() else float("nan")
    out["defect_z0"] = float(defect["z0"])
    out["defect_epsr"] = float(defect["epsr"])
    out["defect_alpha"] = float(defect["alpha"])
    return out


def make_case(z0_mult: float, epsr_delta: float, alpha_mult: float) -> dict:
    return {
        "name": "diagnose moisture",
        "profile": "field",
        "length_m": 1800.0,
        "defects": [{
            "type": "moisture_distributed",
            "start_m": 360.0,
            "length_m": 1370.0,
            "z0_mult": z0_mult,
            "epsr_delta": epsr_delta,
            "alpha_mult": alpha_mult,
            "label_amplitude": 0.68,
        }],
    }


if __name__ == "__main__":
    dg = load_dg_module()
    original_coherent = dg.apply_field_lowfreq_coherent_phase
    case = make_case(0.80, 0.55, 1.65)
    cable = dg.cable_from_defect_case(case)
    for mode in ["orig", "off"]:
        if mode == "off":
            dg.apply_field_lowfreq_coherent_phase = lambda out, freq_hz, cable, rng=None: out
        else:
            dg.apply_field_lowfreq_coherent_phase = original_coherent
        rng = np.random.RandomState(20260631)
        band_1ghz, band_200mhz = dg.generate_defect_case_bands(cable, "field", rng, True, None, None)
        row = {
            "case_name": f"coherent={mode}",
            "profile": "field",
            "cable": cable,
            "f1": band_1ghz[0],
            "s1": band_1ghz[1],
            "d1": band_1ghz[2],
            "imp1": band_1ghz[3],
            "step1": band_1ghz[4],
            "f2": band_200mhz[0],
            "s2": band_200mhz[1],
            "d2": band_200mhz[2],
            "imp2": band_200mhz[3],
            "step2": band_200mhz[4],
        }
        print(row["case_name"], summarize(row))
    dg.apply_field_lowfreq_coherent_phase = original_coherent

    variants = [
        (0.80, 0.55, 1.65),
        (0.88, 0.55, 1.85),
        (0.94, 0.55, 2.20),
        (0.98, 0.55, 2.60),
        (1.00, 0.55, 2.80),
        (1.04, 0.55, 2.40),
        (1.10, 0.55, 2.40),
        (1.18, 0.55, 2.40),
        (0.35, 0.55, 2.40),
        (0.50, 0.55, 2.40),
        (0.65, 0.55, 2.40),
    ]
    for z0_mult, epsr_delta, alpha_mult in variants:
        cable = dg.cable_from_defect_case(make_case(z0_mult, epsr_delta, alpha_mult))
        rng = np.random.RandomState(20260623 + 8)
        band_1ghz, band_200mhz = dg.generate_defect_case_bands(
            cable,
            "field",
            rng,
            True,
            None,
            None,
        )
        row = {
            "case_name": f"z0={z0_mult} alpha={alpha_mult}",
            "profile": "field",
            "cable": cable,
            "f1": band_1ghz[0],
            "s1": band_1ghz[1],
            "d1": band_1ghz[2],
            "imp1": band_1ghz[3],
            "step1": band_1ghz[4],
            "f2": band_200mhz[0],
            "s2": band_200mhz[1],
            "d2": band_200mhz[2],
            "imp2": band_200mhz[3],
            "step2": band_200mhz[4],
        }
        print(row["case_name"], summarize(row))
