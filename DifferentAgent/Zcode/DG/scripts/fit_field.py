"""Fit the DG_V4 forward model to annotated field files.

For each labelled cable (length + joint positions from the engineer txt
annotations, open end) we fit
  [z0, epsr, alpha100, tand, fix_series, fix_shunt, fix_delay_ns, fix_loss,
   D_scale, S_scale, rough_sigma, joint_scale]
by least squares on the complex S11 (log-subsampled, 9 kHz-250 MHz), then
aggregate the fitted values into frozen generator ranges.

Usage:  python fit_field.py            (writes output/field_fit.json)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dg_v4.fdr_io import read_fdr, phase_arrays
from dg_v4.generate import cable_abcd, template_seed
from dg_v4.measurement import ChainParams, evaluate as chain_evaluate, roughness_gamma
from dg_v4.topology import build_topology

BASE = r"E:\20261010截至的全部fdr文件"

CASES = [
    dict(name="三院宿舍770m", path=BASE + r"\2026-5-21 11.54.20深圳市第三人民医院宿舍公用智能柜\2026-5-21 11.54.20.fdr",
         length=770.0, joints=[55, 91, 175, 638, 682], phase=0),
    dict(name="平湖辅城坳线598m", path=BASE + r"\2026-5-25 10.48.04深圳110kV平湖站10kv辅城坳线F17平湖山厦外国语学校智能柜\2026-5-25 10.48.04.fdr",
         length=598.0, joints=[222, 240, 255, 285, 355, 421], phase=0),
    dict(name="白塔寺1100m耐压前", path=BASE + r"\2025-2-17 11.25.49北京城区白塔寺", length=1100.0,
         joints=[], phase=0),
    dict(name="逸翠园3870m", path=BASE + r"\2026-7-19 9.55.15深圳110kV布吉变电站10KV禾坑线F01信义逸翠园五栋#2智能柜",
         length=3870.0, joints=[100, 355, 446, 826, 985, 1200, 1482, 1626, 2062, 2449, 3055, 3370], phase=0),
    dict(name="金嘉一线301m", path=BASE + r"\2026-8-22 12.01.01深圳110kV金融站10kV金嘉一线F07白龙头#1智能柜",
         length=301.0, joints=[], phase=0),
]

N_FIT = 2600


def load_case(case: dict):
    p = Path(case["path"])
    if p.is_dir():
        fdrs = sorted(p.glob("*.fdr"))
        p = fdrs[0]
    data = read_fdr(str(p))
    ph = data["phases"][case["phase"]]
    a = phase_arrays(ph)
    f, s11 = a["freq"], a["s11"]
    sel = np.unique(np.linspace(0, len(f) - 1, N_FIT).astype(int))
    return f[sel], s11[sel]


EPSR_FIXED = 3.037

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
    g = model_s11(f_sub, x, case, normalize_short=ns)
    if g is None:
        return 1e3
    return float(np.sqrt(np.mean(np.abs(g - s11) ** 2)))


def fit_case(case: dict) -> dict:
    f_sub, s11 = load_case(case)
    bounds = [(9.0, 36.0), (0.005, 0.05), (50.0, 90.0), (12.0, 35.0), (0.6, 3.5),
              (20.0, 200.0), (0.005, 0.09), (0.2, 5.0), (-np.pi, np.pi)]

    best = None
    for ns in (True, False):
        de = differential_evolution(_de_residual, bounds,
                                    args=(f_sub, s11, case, ns), maxiter=40,
                                    popsize=14, tol=1e-4, seed=5, polish=True,
                                    workers=-1)
        if best is None or de.fun < best[0]:
            best = (de.fun, de.x, ns)
    rms = best[0]
    names = ["z0", "alpha100", "line_z", "line_delay_ns", "line_loss", "tip_shunt_pf",
             "D_scale", "joint_scale", "phase_ref"]
    out = dict(name=case["name"], rms_complex=float(rms), normalize_short=bool(best[2]),
               epsr_fixed=EPSR_FIXED, **{n: float(v) for n, v in zip(names, best[1])})
    return out


def main() -> None:
    results = []
    for case in CASES:
        try:
            r = fit_case(case)
            print(f"{r['name']}: rms={r['rms_complex']:.4f} ns={r['normalize_short']} "
                  f"z0={r['z0']:.1f} alpha={r['alpha100']:.4f} "
                  f"line=({r['line_z']:.0f}ohm,{r['line_delay_ns']:.0f}ns,{r['line_loss']:.2f}dB) "
                  f"tipC={r['tip_shunt_pf']:.0f}pF D={r['D_scale']:.3f} jscale={r['joint_scale']:.2f}")
            results.append(r)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"{case['name']}: FAILED {exc}")
            import traceback; traceback.print_exc()
    out = Path(__file__).resolve().parents[1] / "output" / "field_fit.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved ->", out)


if __name__ == "__main__":
    main()
