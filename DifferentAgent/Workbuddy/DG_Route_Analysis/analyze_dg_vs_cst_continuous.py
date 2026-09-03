"""Quantify how close DG V3 RLGC and CST V1 fixed-continuous really are.

Compares the S11 and FDR outputs of compare_dg_routes.py between
- dg_v3_rlgc (frequency-dependent RLGC anchored at 100 MHz)
- cst_v1_continuous (fixed continuous RLGC)
and uses cst_v1_ladder_0p1m as a scaling reference.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np

HERE = Path(__file__).resolve().parent
EVAL = Path(r"D:/GitRepository/Cable-NN/DifferentAgent/CodeX/DG_Route_Evaluation")
OUT = EVAL / "output"

A = "dg_v3_rlgc"
B = "cst_v1_continuous"
REF = "cst_v1_ladder_0p1m"

CASES = [
    "baseline", "overall_C20pF", "overall_G20k", "segmented_loss",
    "local_C32pF_15m", "local_G2k_15m", "local_C4pF_15m",
    "local_R10ohm_15m", "local_R50ohm_15m",
]
LAYERS = ("clean", "common_measurement")


def load_s11(layer: str, cand: str, case: str) -> tuple[np.ndarray, np.ndarray]:
    path = OUT / "s11" / layer / cand / f"s11_{case}.csv"
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    f = data["Frequency_Hz"]
    s = data["S11_Real"] + 1j * data["S11_Imag"]
    return f, s


def load_response(layer: str, cand: str, case: str) -> dict:
    path = OUT / "responses" / layer / cand / f"response_{case}.npz"
    return dict(np.load(path))


def curve_diff_metrics(d, y1, y2, low=None, high=None):
    """Return RMS/max of (y1-y2) over a distance window, normalized by peak abs."""
    d = np.asarray(d, dtype=float)
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)
    if low is not None:
        keep = (d >= low) & (d <= high)
    else:
        keep = np.ones_like(d, dtype=bool)
    keep &= np.isfinite(y1) & np.isfinite(y2)
    if not np.any(keep):
        return float("nan"), float("nan")
    a1, a2 = y1[keep], y2[keep]
    scale = max(float(np.max(np.abs(np.concatenate((a1, a2))))), 1e-30)
    diff = a1 - a2
    return float(np.sqrt(np.mean(diff**2)) / scale), float(np.max(np.abs(diff)) / scale)


def band_rms(f, diff):
    bands = [(0.009, 1.0), (1.0, 30.0), (30.0, 100.0), (100.0, 200.0)]
    out = []
    for lo, hi in bands:
        mask = (f >= lo * 1e6) & (f <= hi * 1e6)
        out.append(float(np.sqrt(np.mean(np.abs(diff[mask]) ** 2))) if np.any(mask) else float("nan"))
    return out


def main():
    results = {"layers": {}}
    print("=" * 100)
    print("对比 A=dg_v3_rlgc 与 B=cst_v1_continuous；REF=cst_v1_ladder_0p1m 作为量级参照")
    print("=" * 100)
    for layer in LAYERS:
        layer_rows = []
        print(f"\n##### 层: {layer} #####")
        for case in CASES:
            f, sA = load_s11(layer, A, case)
            _, sB = load_s11(layer, B, case)
            _, sR = load_s11(layer, REF, case)
            diff_AB = sA - sB
            rms_ab = float(np.sqrt(np.mean(np.abs(diff_AB) ** 2)))
            peak_ab = float(np.max(np.abs(diff_AB)))
            rms_br = float(np.sqrt(np.mean(np.abs(sB - sR) ** 2)))
            scale = float(np.sqrt(np.mean(np.abs(sA) ** 2)))  # signal RMS
            band = band_rms(f, diff_AB)

            rA = load_response(layer, A, case)
            rB = load_response(layer, B, case)
            rR = load_response(layer, REF, case)

            # algorithm1 step & impulse, algorithm2 step & impulse
            d1 = rA["algorithm1_distance_m"]
            a1_step = curve_diff_metrics(d1, rA["algorithm1_step"], rB["algorithm1_step"])
            a1_step_ref = curve_diff_metrics(d1, rA["algorithm1_step"], rR["algorithm1_step"])
            a1_imp = curve_diff_metrics(d1, rA["algorithm1_impulse"], rB["algorithm1_impulse"])
            a1_imp_ref = curve_diff_metrics(d1, rA["algorithm1_impulse"], rR["algorithm1_impulse"])
            d2 = rA["algorithm2_distance_m"]
            a2_step = curve_diff_metrics(d2, rA["algorithm2_step"], rB["algorithm2_step"])
            a2_step_ref = curve_diff_metrics(d2, rA["algorithm2_step"], rR["algorithm2_step"])
            d2i = rA["algorithm2_impulse_distance_m"]
            a2_imp = curve_diff_metrics(d2i, rA["algorithm2_impulse"], rB["algorithm2_impulse"])
            a2_imp_ref = curve_diff_metrics(d2i, rA["algorithm2_impulse"], rR["algorithm2_impulse"])

            row = {
                "case": case,
                "s11_rms_diff_AB": rms_ab,
                "s11_peak_diff_AB": peak_ab,
                "s11_rms_diff_B_vs_REF": rms_br,
                "s11_signal_rms": scale,
                "s11_rms_ratio_AB_pct": 100.0 * rms_ab / scale,
                "s11_band_rms_0_1MHz": band[0],
                "s11_band_rms_1_30MHz": band[1],
                "s11_band_rms_30_100MHz": band[2],
                "s11_band_rms_100_200MHz": band[3],
                "a1_step_rms_AB": a1_step[0],
                "a1_step_rms_BvsREF": a1_step_ref[0],
                "a1_imp_rms_AB": a1_imp[0],
                "a1_imp_rms_BvsREF": a1_imp_ref[0],
                "a2_step_rms_AB": a2_step[0],
                "a2_step_rms_BvsREF": a2_step_ref[0],
                "a2_imp_rms_AB": a2_imp[0],
                "a2_imp_rms_BvsREF": a2_imp_ref[0],
            }
            layer_rows.append(row)
            print(f"\n[{case}]")
            print(f"  S11: RMS|A-B|={rms_ab:.3e}  peak|A-B|={peak_ab:.3e}  信号RMS={scale:.4f}  "
                  f"比值={100*rms_ab/scale:.4f}%   (B vs REF 0.1m梯形 RMS={rms_br:.3e})")
            print(f"   频段RMS(A-B): <1MHz={band[0]:.2e}  1-30MHz={band[1]:.2e}  "
                  f"30-100MHz={band[2]:.2e}  100-200MHz={band[3]:.2e}")
            print(f"   FDR归一化RMS (A vs B | A vs 0.1mREF):")
            print(f"     A1阶跃: {a1_step[0]*100:.4f}% | {a1_step_ref[0]*100:.4f}%"
                  f"    A1脉冲: {a1_imp[0]*100:.4f}% | {a1_imp_ref[0]*100:.4f}%")
            print(f"     A2阶跃: {a2_step[0]*100:.4f}% | {a2_step_ref[0]*100:.4f}%"
                  f"    A2脉冲: {a2_imp[0]*100:.4f}% | {a2_imp_ref[0]*100:.4f}%")
        results["layers"][layer] = layer_rows

    (HERE / "dg_vs_cst_continuous_metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n-> {HERE / 'dg_vs_cst_continuous_metrics.json'}")


if __name__ == "__main__":
    main()
