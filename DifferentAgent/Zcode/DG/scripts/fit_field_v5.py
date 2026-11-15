"""Fit the DG_V5 forward model to annotated field files (v5, relaxed bounds).

Differences to the first-round fit_field.py (whose output field_fit.json was
never consumed):
- bounds relaxed where round 1 pinned at the ceiling: alpha100 (0.005,0.15),
  fixture sqrt-loss (0.5,8.0) dB, cable z0 (8,40) ohm
- 12 phase files across 8 cables (301-4714 m, engineers' joint annotations,
  non-blind folders) instead of 5
- reads the pre-extracted phases npz (field_data/phases) instead of .fdr

Writes output/field_fit_v5.json.  The aggregated distribution feeds
topology.P["field"] (z0, alpha) and measurement._RANGES["field"] (line_z /
delay / loss) - see calibration_notes_v5.md.

Usage:  python fit_field_v5.py [--maxiter 30]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dg_v4.generate import cable_abcd, template_seed
from dg_v4.measurement import ChainParams, evaluate as chain_evaluate, roughness_gamma
from dg_v4.topology import build_topology

PHASES = Path(r"E:\CableNNDataSet\ZCode\field_data\phases")

CASES = [
    dict(name="金嘉一线301_A", stem="2026-8-22 12.01.01__A相[末端开路]", length=301.0, joints=[]),
    dict(name="金嘉一线301_B", stem="2026-8-22 12.01.01__B相[末端开路]", length=301.0, joints=[]),
    dict(name="平湖辅城坳598_A", stem="2026-5-25 10.48.04__A相", length=598.0,
         joints=[222, 240, 255, 285, 355, 421]),
    dict(name="平湖辅城坳598_B", stem="2026-5-25 10.48.04__B相", length=598.0,
         joints=[222, 240, 255, 285, 355, 421]),
    dict(name="鼎盛鼎康945_A", stem="2026-9-16 10.45.47__A相[末端开路]", length=945.0,
         joints=[213, 338, 644, 945]),
    dict(name="鼎盛鼎康945_B", stem="2026-9-16 10.45.47__B相[末端开路]", length=945.0,
         joints=[213, 338, 644, 945]),
    dict(name="白塔寺1100_耐压前A", stem="2025-2-17 11.25.49城区白塔寺__A相", length=1100.0, joints=[240]),
    dict(name="白塔寺1100_复测A", stem="2025-2-17 12.30.59白塔寺复测__A相", length=1100.0, joints=[240]),
    dict(name="金富线1724_A", stem="2026-8-22 12.15.49__A相[末端开路]", length=1724.0, joints=[154]),
    dict(name="金富线1724_B", stem="2026-8-22 12.15.49__B相[末端开路]", length=1724.0, joints=[154]),
    dict(name="逸翠园3870_A", stem="2026-7-19 9.55.15__A相", length=3870.0,
         joints=[100, 355, 446, 826, 985, 1200, 1482, 1626, 2062, 2350, 2449, 2862, 3055, 3370, 3592]),
    dict(name="清水河4714_A", stem="2026-9-14 11.14.34__A相", length=4714.0,
         joints=[396, 667, 1004, 1367, 1383]),
]

N_FIT = 2600
EPSR_FIXED = 3.037


def load_case(case: dict):
    with np.load(PHASES / f"{case['stem']}.npz", allow_pickle=False) as z:
        f, s11 = z["freq"], z["s11"].astype(np.complex128)
    sel = np.unique(np.linspace(0, len(f) - 1, N_FIT).astype(int))
    return f[sel], s11[sel]


def model_s11(f_sub, params, case, normalize_short=True):
    z0, alpha, lzx, ldly, lloss, tipc, dscale, jscale, ph0 = params
    overrides = dict(length_m=case["length"], termination="open",
                     joint_positions_m=case["joints"], defects=[],
                     base=dict(z0=z0, epsr=EPSR_FIXED, alpha=alpha, tand=6e-4),
                     joint_scale=jscale)
    model = build_topology("field", template_seed(3), overrides=overrides)
    if model is None:
        return None
    chain = ChainParams("D", lzx, ldly, 0.6, lloss, 0.9, tipc, 150.0, 80.0, dscale, 0.03, 0.02, 0.004, 0.95)
    net = cable_abcd(f_sub, model)
    g_rough = roughness_gamma(model, f_sub, np.random.default_rng(11))
    return chain_evaluate(chain, "field", f_sub, net, complex(model.z_load_ohm), 50.0, 7,
                          gamma_rough=g_rough, normalize_short=normalize_short,
                          phase_ref_rad=ph0)


def _de_residual(x, f_sub, s11, case, ns):
    g = model_s11(f_sub, x, case, ns)
    if g is None:
        return 1e3
    return float(np.sqrt(np.mean(np.abs(g - s11) ** 2)))


def fit_case(case: dict, maxiter: int) -> dict:
    f_sub, s11 = load_case(case)
    bounds = [(8.0, 40.0), (0.005, 0.15), (50.0, 90.0), (12.0, 35.0), (0.5, 8.0),
              (20.0, 200.0), (0.005, 0.09), (0.2, 5.0), (-np.pi, np.pi)]
    best = None
    for ns in (False, True):
        de = differential_evolution(_de_residual, bounds,
                                    args=(f_sub, s11, case, ns), maxiter=maxiter,
                                    popsize=12, tol=1e-4, seed=5, polish=True,
                                    workers=-1)
        if best is None or de.fun < best[0]:
            best = (de.fun, de.x, ns)
    rms = best[0]
    names = ["z0", "alpha100", "line_z", "line_delay_ns", "line_loss", "tip_shunt_pf",
             "D_scale", "joint_scale", "phase_ref"]
    return dict(name=case["name"], stem=case["stem"], length_m=case["length"],
                rms_complex=float(rms), normalize_short=bool(best[2]),
                epsr_fixed=EPSR_FIXED, **{n: float(v) for n, v in zip(names, best[1])})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maxiter", type=int, default=30)
    args = ap.parse_args()
    results = []
    for case in CASES:
        try:
            r = fit_case(case, args.maxiter)
            print(f"{r['name']}: rms={r['rms_complex']:.4f} ns={r['normalize_short']} "
                  f"z0={r['z0']:.1f} alpha={r['alpha100']:.4f} "
                  f"line=({r['line_z']:.0f}ohm,{r['line_delay_ns']:.0f}ns,{r['line_loss']:.2f}dB) "
                  f"tipC={r['tip_shunt_pf']:.0f}pF D={r['D_scale']:.3f} jscale={r['joint_scale']:.2f}",
                  flush=True)
            results.append(r)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"{case['name']}: FAILED {exc}", flush=True)
    out = Path(__file__).resolve().parents[1] / "output" / "field_fit_v5.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved ->", out)


if __name__ == "__main__":
    main()
