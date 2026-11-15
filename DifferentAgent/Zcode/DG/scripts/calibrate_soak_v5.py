"""Calibrate DEFECT_SPEC["moisture_local"] against the 1500 m soak trial.

Controlled experiment (2026-6): one 1500 m cable, one joint soaked at positions
100..1400 m -> 11 sweeps + reference in one .fdr (12 measured sweeps, 10 kHz-1 GHz
@ 10000 pt, vendor axis v=200 m/us).  The vendor tables for all 12 sweeps are in
field_data/phases; the differential (soak - normal) isolates the defect response
(fixture, cable clutter and joints cancel).

Method:
  1. field dips A_f(P) = min(diff impulse) in [0.8P, 1.3P]; positions inside the
     clamp ring-down (100 m) or end-transition windows (1300/1400 m) are flagged
  2. synthetic twin: 1500 m field cable, epsr from the measured end position
     (v_true = 200*1500/end_axis -> epsr), one moisture_local at P, differential
     of the same pipeline with/without the defect under identical chain/noise
  3. alpha fit: base alpha scanned so the *decay* of log(dip) vs P matches
     (round-trip attenuation is the only P-dependent term)
  4. severity per position: dip ~ Gamma(epsr_delta) with the mixture relation
     z0_factor = sqrt(epsr/(epsr+depsr))  ->  depsr*(P) = depsr_ref * A_f/A_s
  5. DEFECT_SPEC ranges set to cover [min, max] of depsr*(P) (+padding);
     verification: each position re-simulated at its fitted severity

Writes output/defect_calibration_v5.json; final spec printed for write-back.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dg_stats_v5 import grid_distance, resample_table
from dg_v4.grid import get_band_axis
from dg_v4.generate import Z_REF, cable_abcd
from dg_v4.measurement import evaluate as chain_evaluate, roughness_gamma, sample_chain
from dg_v4.topology import DEFECT_SPEC, _alpha_factor, build_topology
from dg_v4.transform import s11_to_responses

PHASES = Path(r"E:\CableNNDataSet\ZCode\field_data\phases")
PREFIX = "1500m电缆单个接头浸水试验__"
OUT = Path(__file__).resolve().parents[1] / "output"

LENGTH = 1500.0
V_FILE = 200.0            # instrument axis setting of the soak .fdr
EPSR_SIM = 2.538          # from end anchor 1593.25 m (v_true = 188.3 m/us)
Z0_SIM = 19.0
TAND_SIM = 0.001          # must satisfy the base dielectric budget (alpha ~0.02)
ALPHA_GRID = (0.012, 0.016, 0.020, 0.025, 0.030, 0.040)
DEPSR_REF = 0.30
DEFECT_LEN = 8.0          # compact soaked joint; spec len_m lower bound -> 5 m
POSITIONS = (200, 300, 400, 500, 600, 700, 900, 1100)   # clean window
POS_ALL = (100, 200, 300, 400, 500, 600, 700, 900, 1100, 1300, 1400)
N_CHAIN = 8               # chain draws per configuration (paired medians)
AXIS_SCALE = 172.0 / 188.3   # generator v_axis (field) / true cable velocity
F = get_band_axis("1ghz_fast").freq_hz


def gamma_of_depsr(depsr: float, epsr: float) -> float:
    """Edge reflection of a moisture section with z0f = sqrt(epsr/(epsr+d))."""
    return -0.5 * depsr / (epsr + depsr)


def depsr_of_gamma(gamma: float, epsr: float) -> float:
    g = min(abs(gamma), 0.45)
    return 2.0 * g * epsr / (1.0 - 2.0 * g)


def field_dips() -> dict:
    dg = grid_distance()
    def load(name):
        with np.load(PHASES / f"{PREFIX}{name}.npz", allow_pickle=False) as z:
            return (resample_table(z["impulse_dist"], z["impulse_rho"], dg),
                    resample_table(z["step_dist"], z["step_rho"], dg))
    imp0, _ = load("正常")
    out = {}
    for P in POS_ALL:
        imp, _ = load(f"{P}m接头浸水")
        d = imp - imp0
        lo, hi = int(0.8 * P / 0.25), int(1.3 * P / 0.25)
        out[P] = float(np.min(d[lo:hi]))
    return out


def synth_config(args_tuple):
    """Per-chain differential impulse dips for one (position, alpha, depsr)."""
    P, alpha, depsr, seed0 = args_tuple
    z0f = float(np.sqrt(EPSR_SIM / (EPSR_SIM + depsr)))
    defect = dict(mechanism="moisture_local", length_m=DEFECT_LEN,
                  start_m=P - DEFECT_LEN / 2, end_m=P + DEFECT_LEN / 2,
                  z0_factor=z0f, epsr_delta=depsr, alpha_factor=6.0,
                  tan_val=0.006, debye=DEFECT_SPEC["moisture_local"]["debye"],
                  debye_corner=12.0e6, grad=DEFECT_SPEC["moisture_local"]["grad"],
                  profile_shape="smooth", profile_width_m=0.0)
    rng = np.random.default_rng(seed0)
    defect["alpha_factor"] = _alpha_factor(rng, defect, alpha, TAND_SIM, EPSR_SIM)
    dips = []
    for c in range(N_CHAIN):
        seed = seed0 * 97 + c
        chain = sample_chain("field", seed * 131 + 5)
        models = []
        for defs in ([], [defect]):
            m = build_topology("field", seed, overrides=dict(
                length_m=LENGTH, termination="open", joint_positions_m=[], defects=defs,
                base=dict(z0=Z0_SIM, epsr=EPSR_SIM, alpha=alpha,
                          tand=min(TAND_SIM, 0.35 * alpha / 15.9))))
            if m is None:
                return None
            models.append(m)
        dips_c = []
        for m in models:
            net = cable_abcd(F, m)
            g_rough = roughness_gamma(m, F, np.random.default_rng(seed * 977 + 3))
            s11 = chain_evaluate(chain, "field", F, net, complex(m.z_load_ohm), Z_REF, seed,
                                 gamma_rough=g_rough, normalize_short=False)
            v_axis, tau = m.v_axis_m_per_s, chain.line_delay_ns * 1e-9
            offset = v_axis * tau
            _, imp, _, _ = s11_to_responses(F, s11, v_axis, 1560.0, offset,
                                            dist_max_m=1750.0)
            dips_c.append(imp)
        d = dips_c[1] - dips_c[0]
        p_ax = P * AXIS_SCALE                    # defect position on the synth axis
        lo, hi = int(0.8 * p_ax / 0.25), int(1.3 * p_ax / 0.25)
        dips.append(float(np.min(d[lo:hi])))
    return dips


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed0", type=int, default=3000)
    args = ap.parse_args()

    fd = field_dips()
    print("field dips:", {k: round(v, 6) for k, v in fd.items()})

    # --- alpha scan: match the decay of log(|dip|) with position -----------
    scan = []
    with mp.Pool(args.workers) as pool:
        for alpha in ALPHA_GRID:
            jobs = [(P, alpha, DEPSR_REF, args.seed0) for P in POSITIONS]
            res = pool.map(synth_config, jobs, chunksize=1)
            scan.append((alpha, dict(zip(POSITIONS, res))))
            print(f"alpha={alpha:.3f} dips:",
                  {P: (None if r is None else round(float(np.median(np.abs(r))), 6))
                   for P, r in zip(POSITIONS, res)},
                  flush=True)

    lf = np.array([np.log(abs(fd[P])) for P in POSITIONS])
    best = None
    for alpha, dips in scan:
        if any(v is None for v in dips.values()):
            continue
        ls = np.array([np.log(np.median(np.abs(dips[P]))) for P in POSITIONS])
        d = lf - ls                       # log(field/synth) per position
        resid = float(np.std(d - np.mean(d)))
        if best is None or resid < best[0]:
            best = (resid, alpha, dict(zip(POSITIONS, d.tolist())))
    resid, alpha, resid_by_pos = best
    print(f"\nalpha*={alpha:.3f} (decay residual {resid:.3f} in log10), "
          f"log offsets {[(P, round(v, 2)) for P, v in resid_by_pos.items()]}")

    # per-position severity: direct search on a depsr log grid (the dip is
    # monotonic but sublinear in depsr - linear Gamma inversion overshoots),
    # paired chains, closest median ratio to 1 in log space
    DEPSR_GRID = (0.02, 0.035, 0.05, 0.08, 0.12, 0.18, 0.28, 0.45, 0.7, 1.1)
    with mp.Pool(args.workers) as pool:
        jobs = [(P, alpha, dg_, args.seed0) for P in POSITIONS for dg_ in DEPSR_GRID]
        res = pool.map(synth_config, jobs, chunksize=1)
    table = {}
    n_ok = 0
    depsr_fit = {}
    for i, P in enumerate(POSITIONS):
        best_d = None
        for j, dg_ in enumerate(DEPSR_GRID):
            chains = res[i * len(DEPSR_GRID) + j]
            if chains is None:
                continue
            ratio = float(np.median([abs(c) / abs(fd[P]) for c in chains]))
            if best_d is None or abs(np.log(ratio)) < abs(np.log(best_d[1])):
                best_d = (dg_, ratio)
        if best_d is None:
            continue
        depsr_fit[P] = best_d[0]
        ok = 0.5 <= best_d[1] <= 2.0
        n_ok += ok
        table[str(P)] = dict(field_dip=fd[P],
                             synth_dip=best_d[1] * abs(fd[P]),
                             ratio=best_d[1], depsr=best_d[0], within_2x=bool(ok))
        print(f"P={P:>5}: field {fd[P]:.3e} synth {best_d[1] * abs(fd[P]):.3e} "
              f"ratio {best_d[1]:5.2f} depsr* {best_d[0]:.3f} {'OK' if ok else 'OUT'}")
    print(f"within +-2x: {n_ok}/{len(POSITIONS)}")

    depsr_vals = np.array(list(depsr_fit.values()))
    depsr_lo = float(max(0.05, 0.8 * depsr_vals.min()))
    depsr_hi = float(min(1.5, 1.25 * depsr_vals.max()))
    z0_hi = float(np.sqrt(3.04 / (3.04 + depsr_lo)))
    z0_lo = float(np.sqrt(3.04 / (3.04 + depsr_hi)))
    spec = dict(alpha_fit=alpha, decay_residual_log=resid,
                depsr_range=[depsr_lo, depsr_hi],
                z0_range=[z0_lo, z0_hi],
                len_m_lower=5.0, len_note="soaked joint intrinsic width 1-3 m "
                "(dip width 0.8-1.8 m at 100-200 m is resolution-limited)",
                alpha_factor_range=[3.0, 12.0],
                verification=table, n_within_2x=n_ok)
    OUT.mkdir(exist_ok=True)
    (OUT / "defect_calibration_v5.json").write_text(
        json.dumps(dict(field_dips={str(k): v for k, v in fd.items()},
                        spec=spec), ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nrecommended DEFECT_SPEC['moisture_local'] updates:")
    print("  depsr:", (depsr_lo, depsr_hi), " z0:", (round(z0_lo, 3), round(z0_hi, 3)),
          " len_m lower: 5.0")
    print("saved ->", OUT / "defect_calibration_v5.json")


if __name__ == "__main__":
    main()
