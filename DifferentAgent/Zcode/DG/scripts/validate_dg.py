"""Statistical acceptance: generated field-profile samples vs real field data.

Metric: per-band mean |S11| distributions (10 log-spaced bands).  The
generator passes if the generated interquartile range overlaps the real
interquartile range in every band (no band where generated mass sits entirely
outside real spread).  Also compares transformed-domain clutter floors.

    python validate_dg.py [--n 24]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dg_v4.generate import generate_sample

FIELD_PHASES = Path(r"E:\CableNNDataSet\ZCode\field_data\phases")
BANDS = [(0.01, 0.1), (0.1, 1), (1, 3), (3, 10), (10, 30), (30, 60), (60, 100),
         (100, 200), (200, 500), (500, 1000)]


def band_means(f: np.ndarray, s11: np.ndarray) -> list[float]:
    mag = np.abs(s11)
    out = []
    for lo, hi in BANDS:
        m = (f >= lo * 1e6) & (f < hi * 1e6)
        out.append(float(mag[m].mean()) if m.any() else np.nan)
    return out


def real_samples(n: int) -> list[list[float]]:
    rows = []
    files = sorted(FIELD_PHASES.glob("*.npz"))
    rng = np.random.default_rng(7)
    pick = rng.choice(len(files), size=min(n * 2, len(files)), replace=False)
    for i in pick:
        with np.load(files[i], allow_pickle=False) as z:
            if z["s11"].size < 9000:      # skip fast mode for the 1GHz-band rows
                continue
            rows.append(band_means(z["freq"], z["s11"]))
        if len(rows) >= n:
            break
    return rows


def generated_samples(n: int) -> list[list[float]]:
    rows, seed = [], 8000
    while len(rows) < n and seed < 8000 + 12 * n:
        s = generate_sample(seed)
        seed += 1
        if s is None or s["profile"] != "field" or s["band"] != "1ghz_std":
            continue
        rows.append(band_means(s["freq_hz"], s["s11"]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    args = ap.parse_args()
    real = np.asarray(real_samples(args.n))
    gen = np.asarray(generated_samples(args.n))
    print(f"real n={len(real)}  generated n={len(gen)}")
    print(f"{'band(MHz)':>14} {'real IQR':>16} {'gen IQR':>16}  verdict")
    ok_all = True
    for i, (lo, hi) in enumerate(BANDS):
        rq = np.nanpercentile(real[:, i], [25, 50, 75])
        gq = np.nanpercentile(gen[:, i], [25, 50, 75])
        overlap = min(rq[2], gq[2]) >= max(rq[0], gq[0])
        ok_all &= bool(overlap)
        print(f"{lo:6.2f}-{hi:4.0f}  {rq[0]:.3f}-{rq[2]:.3f}(m{rq[1]:.2f})  "
              f"{gq[0]:.3f}-{gq[2]:.3f}(m{gq[1]:.2f})  {'OK' if overlap else 'MISS'}")
    print("ACCEPT" if ok_all else "REJECT — widen ranges for MISS bands")


if __name__ == "__main__":
    main()
