"""Amplitude calibration of the DG_V5 field profile against the field corpus.

Iterative loop (<= 4 rounds):
  1. probe: 200 field-profile samples (multiprocessing; knobs passed to the
     workers through the pool initializer because Windows spawn re-imports)
  2. scale knobs by field/synth median ratios (small-reflection regime: peak
     amplitudes scale linearly with the parameters)
       joint lump params  P["field"]["joint_series_ohm"] / ["joint_shunt_pf"]
       roughness sigma    measurement.FIELD_ROUGH_SIGMA (clutter RMS)
       terminal overflow  fixture sqrt-loss range, only if synth/field > 1.4
  3. every round's params + stats -> output/amplitude_calibration_v5.json

Convergence gate: every joint/terminal x depth-bucket amplitude median ratio
and both clutter ratios inside [0.7, 1.4].  The terminal ratios per bucket are
additionally regressed against depth (log-ratio slope) to expose how much of
the mismatch is attenuation (alpha) rather than fixture loss; the slope is
reported but alpha itself comes from fit_field_v5.

Usage:  python calibrate_amplitudes_v5.py [--probe 200] [--rounds 4] [--workers 14]
Final parameters are printed for write-back into topology.py / measurement.py.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dg_v4.measurement as M
import dg_v4.topology as T
from dg_stats_v5 import (BUCKETS, _bucket_index, empty_agg, field_reference,
                         fmt_ratio_table, median_table, ratio_table,
                         response_stats)
from dg_v4.generate import generate_sample

FIELD_PHASES = Path(r"E:\CableNNDataSet\ZCode\field_data\phases")
FIELD_MANIFEST = Path(r"E:\CableNNDataSet\ZCode\field_data\manifest.csv")
OUT = Path(__file__).resolve().parents[1] / "output"
GATE = 0.70, 1.40

# current knob values (absolute), overlaid onto freshly-spawned workers
KNOBS = {
    "joint_series_ohm": None,   # default: source-file values
    "joint_shunt_pf": None,
    "rough_sigma": None,
    "loss_sqrt": None,
}


def _init_worker(knobs: dict) -> None:
    if knobs.get("joint_series_ohm"):
        T.P["field"]["joint_series_ohm"] = tuple(knobs["joint_series_ohm"])
        T.P["field"]["joint_shunt_pf"] = tuple(knobs["joint_shunt_pf"])
    if knobs.get("rough_sigma"):
        M.FIELD_ROUGH_SIGMA = knobs["rough_sigma"]
    if knobs.get("loss_sqrt"):
        M._RANGES["field"]["D"]["loss_sqrt"] = tuple(knobs["loss_sqrt"])


def probe_sample(seed: int):
    return generate_sample(seed, profile="field")


def probe_stats(samples: list) -> dict:
    agg = empty_agg()
    for s in samples:
        if s is None or s["profile"] != "field":
            continue
        st = response_stats(s["impulse"], float(s["length_m"]), s["events"])
        agg["n_files"] += 1
        agg["lengths"].append(float(s["length_m"]))
        agg["near_0_30"].append(st["near_0_30"])
        for k, v in st["clutter"].items():
            if np.isfinite(v):
                agg["clutter"][k].append(v)
        for ev in st["events"]:
            b = _bucket_index(ev["anchor_m"])
            if b is not None and ev["observable"]:
                agg["event_amps"][ev["role"]][b].append(ev["amp"])
                agg["event_snrs"][ev["role"]].append(ev["snr"])
    return agg


def current_params() -> dict:
    p = T.P["field"]
    return dict(joint_series_ohm=[round(v, 4) for v in p["joint_series_ohm"]],
                joint_shunt_pf=[round(v, 4) for v in p["joint_shunt_pf"]],
                rough_sigma_field=round(M.FIELD_ROUGH_SIGMA, 6),
                alpha_db_per_m_100mhz=list(p["alpha_db_per_m_100mhz"]),
                loss_sqrt=[round(v, 4) for v in M._RANGES["field"]["D"]["loss_sqrt"]])


def knobs_now() -> dict:
    return dict(joint_series_ohm=list(T.P["field"]["joint_series_ohm"]),
                joint_shunt_pf=list(T.P["field"]["joint_shunt_pf"]),
                rough_sigma=M.FIELD_ROUGH_SIGMA,
                loss_sqrt=list(M._RANGES["field"]["D"]["loss_sqrt"]))


def gate_check(ratios: dict) -> tuple[bool, list[str]]:
    bad = []
    for role in ("joint", "terminal"):
        for i, r in enumerate(ratios["amps"][role]):
            if r == r and not (GATE[0] <= r <= GATE[1]):
                bad.append(f"amp {role} bucket {BUCKETS[i]} ratio {r:.2f}")
    for k, r in ratios["clutter"].items():
        if r == r and not (GATE[0] <= r <= GATE[1]):
            bad.append(f"clutter {k} ratio {r:.2f}")
    return (not bad), bad


def terminal_depth_slope(ratios: dict) -> float | None:
    """log(field/synth) vs bucket-center slope over terminal buckets with data."""
    pts = []
    for i, (lo, hi) in enumerate(BUCKETS):
        r = ratios["amps"]["terminal"][i]
        if r == r and r > 0:
            pts.append((0.5 * (lo + hi), np.log(r)))
    if len(pts) < 2:
        return None
    d = np.asarray(pts)
    return float(np.polyfit(d[:, 0], d[:, 1], 1)[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed0", type=int, default=500000)
    args = ap.parse_args()

    print("field reference ...", flush=True)
    field = field_reference(FIELD_PHASES, FIELD_MANIFEST)
    fmed = median_table(field)

    log = {"rounds": [], "final_params": None, "converged": False}
    for rnd in range(args.rounds):
        t0 = time.time()
        seeds = list(range(args.seed0 + rnd * 10000, args.seed0 + rnd * 10000 + args.probe))
        with mp.Pool(args.workers, initializer=_init_worker, initargs=(knobs_now(),)) as pool:
            samples = pool.map(probe_sample, seeds, chunksize=2)
        agg = probe_stats(samples)
        smed = median_table(agg)
        ratios = ratio_table(field, agg)
        ok, bad = gate_check(ratios)
        slope = terminal_depth_slope(ratios)
        log["rounds"].append(dict(round=rnd, params=current_params(),
                                  n_synth=agg["n_files"], field=jclean(fmed),
                                  synth=jclean(smed), ratios=jclean(ratios),
                                  gated_ok=ok, gated_bad=bad,
                                  terminal_logratio_depth_slope=slope))
        print(f"\n=== round {rnd} ({time.time()-t0:.0f}s, {agg['n_files']} synth) "
              f"gate {'PASS' if ok else 'FAIL'} ===", flush=True)
        print(fmt_ratio_table(ratios, fmed, smed), flush=True)
        if slope is not None:
            alpha_implied = -slope / 0.2303   # synth alpha short by this many dB/m
            print(f"terminal log-ratio vs depth slope {slope:.3e} /m "
                  f"-> synth alpha low by ~{alpha_implied:.4f} dB/m", flush=True)
        if ok:
            log["converged"] = True
            break
        acts = []
        # depth law: terminal log-ratio slope vs depth -> cable alpha shift
        # (round-trip amplitude ~ 10^(-alpha*d/10): slope = -0.2303 * d_alpha).
        # Only fires while terminal buckets sit outside the gate; alpha was
        # proven insensitive once, which motivated the base-dispersion model.
        tb_bad = [r for r in ratios["amps"]["terminal"] if r == r and not (GATE[0] <= r <= GATE[1])]
        slope = log["rounds"][-1]["terminal_logratio_depth_slope"]
        if tb_bad and slope is not None and slope < -1e-5:
            delta = float(np.clip(-slope / 0.2303, 0.0, 0.02))
            lo, hi = T.P["field"]["alpha_db_per_m_100mhz"]
            T.P["field"]["alpha_db_per_m_100mhz"] = (lo + delta, hi + delta)
            acts.append(f"alpha += {delta:.4f} -> ({lo + delta:.4f}, {hi + delta:.4f})")
        # joints: peak ~ lump params; use buckets <= 1500 m (deep windows can
        # sit at clutter level and bias the median); act only when out of gate
        jb_all = [(r, n, bhi) for r, n, (blo, bhi) in
                  zip(ratios["amps"]["joint"], fmed["amp_n"]["joint"], BUCKETS)
                  if r == r and r > 0 and n >= 3 and bhi <= 1500]
        jb_bad = [r for r, n, bhi in jb_all if not (GATE[0] <= r <= GATE[1])]
        if jb_bad:
            jb = [r for r, n, bhi in jb_all if bhi <= 500] or [r for r, n, bhi in jb_all]
            k = float(np.exp(np.mean(np.log(jb))))
            for key in ("joint_series_ohm", "joint_shunt_pf"):
                lo, hi = T.P["field"][key]
                T.P["field"][key] = (lo * k, hi * k)
            acts.append(f"joint params *= {k:.3f}")
        # clutter: RMS ~ sigma; act only when out of gate
        cr_all = {k: v for k, v in ratios["clutter"].items() if v == v and v > 0}
        cr_bad = {k: v for k, v in cr_all.items() if not (GATE[0] <= v <= GATE[1])}
        if cr_bad:
            kc = float(np.exp(np.mean(np.log(list(cr_bad.values())))))
            M.FIELD_ROUGH_SIGMA *= kc
            acts.append(f"rough_sigma *= {kc:.3f} -> {M.FIELD_ROUGH_SIGMA:.5f}")
        print("actions:", acts, flush=True)

    log["final_params"] = current_params()
    OUT.mkdir(exist_ok=True)
    (OUT / "amplitude_calibration_v5.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nfinal params:", json.dumps(log["final_params"], indent=1))
    print("converged:", log["converged"])
    print("saved ->", OUT / "amplitude_calibration_v5.json")


def jclean(obj):
    if isinstance(obj, dict):
        return {k: jclean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jclean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


if __name__ == "__main__":
    main()
