"""Generate the DG_V5 synthetic dataset -> E:\\CableNNDataSet\\ZCode\\DG_V5_S1.

30000 samples, REALIZATIONS=50 (600 templates x 50 realizations):
  template t, realization r = seed layout (t = seed // 50, r = seed % 50)
  r < 4  -> same cable topology re-tested (independent chain/noise/band)
  r >= 4 -> independent topology per realization
Splits by template id: t % 20 == 10 -> synth_test (checked FIRST, fixes the v4
inclusion bug where t%20==10 also matched t%10==0 and landed in synth_val),
elif t % 10 == 0 -> synth_val, else train.

npz contract: s11 complex64, impulse/step float32 (valid_len,), role/mechanism
int8 (valid_len,), z0_dev float32 (valid_len,), meta json (valid_len, profile,
band, template_id, realization, length_m, length_setting_m, offset_m,
v_axis_m_per_s, events[+weight], chain, seed, split).  The frequency axis is
NOT stored per sample; band names the global freq_<band>.npy grid.

Resumable: existing sample files are skipped, manifest rebuilt incrementally.
    python generate_dataset.py [--n 30000] [--workers 14] [--limit 200]
                                [--out E:\\CableNNDataSet\\ZCode\\DG_V5_S1]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dg_v4.generate import REALIZATIONS, generate_sample
from dg_v4.grid import get_band_axis

DEFAULT_OUT = Path(r"E:\CableNNDataSet\ZCode\DG_V5_S1")
MANIFEST_COLS = ["file", "split", "profile", "length_m", "n_events",
                 "template_id", "realization", "band"]


def split_of(template_id: int) -> str:
    if template_id % 20 == 10:
        return "synth_test"
    if template_id % 10 == 0:
        return "synth_val"
    return "train"


def process_one(job):
    seed, out_dir = job
    out_path = out_dir / f"sample_{seed:07d}.npz"
    if out_path.exists():
        return ("skip", 0.0)
    t0 = time.time()
    sample = generate_sample(seed)
    if sample is None:
        return ("infeasible", 0.0)
    events = sample.pop("events")
    chain = sample.pop("chain")
    meta = dict(seed=sample["seed"], split=split_of(sample["template_id"]),
                valid_len=sample["valid_len"], profile=sample["profile"],
                band=sample["band"], template_id=sample["template_id"],
                realization=sample["realization"], length_m=sample["length_m"],
                length_setting_m=sample["length_setting_m"],
                offset_m=sample["offset_m"],
                v_axis_m_per_s=sample["v_axis_m_per_s"],
                distance_max_m=sample["distance_max_m"],
                events=events, chain=chain)
    tmp = out_path.with_suffix(".tmp.npz")
    with open(tmp, "wb") as fh:
        np.savez(fh,
                 s11=sample["s11"], impulse=sample["impulse"], step=sample["step"],
                 role=sample["role"], mechanism=sample["mechanism"],
                 z0_dev=sample["z0_dev"],
                 meta=np.array(json.dumps(meta, ensure_ascii=False)))
    tmp.rename(out_path)
    return ("ok", time.time() - t0)


def rebuild_manifest(out_dir: Path) -> int:
    rows = []
    for p in sorted((out_dir / "phases").glob("sample_*.npz")):
        try:
            with np.load(p, allow_pickle=False) as z:
                meta = json.loads(str(z["meta"]))
        except Exception:
            continue
        rows.append(dict(file=p.name, split=meta["split"], profile=meta["profile"],
                         length_m=round(float(meta["length_m"]), 2),
                         n_events=len(meta["events"]),
                         template_id=meta["template_id"],
                         realization=meta["realization"], band=meta["band"]))
    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke mode: only ~limit seeds, spread over the full range")
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = ap.parse_args()
    out_dir = Path(args.out)
    (out_dir / "phases").mkdir(parents=True, exist_ok=True)

    for band in ("1ghz_std", "1ghz_fast", "200mhz"):
        np.save(out_dir / f"freq_{band}.npy", get_band_axis(band).freq_hz)

    if args.limit > 0:
        idx = np.unique(np.linspace(0, args.n - 1, args.limit).astype(int))
        jobs = [(int(i), out_dir / "phases") for i in idx]
    else:
        jobs = [(seed, out_dir / "phases") for seed in range(args.n)]
    print(f"{len(jobs)} samples queued, {args.workers} workers -> {out_dir}")

    import multiprocessing as mp
    counts = {"ok": 0, "skip": 0, "infeasible": 0}
    t_start = time.time()
    with mp.Pool(args.workers) as pool:
        for i, (status, dt) in enumerate(pool.imap_unordered(process_one, jobs, chunksize=4)):
            counts[status] += 1
            if (i + 1) % 50 == 0 or i + 1 == len(jobs):
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                eta = (len(jobs) - i - 1) / max(rate, 1e-9) / 60
                bar = int(30 * (i + 1) / len(jobs))
                print(f"[{'#' * bar}{'.' * (30 - bar)}] {i + 1}/{len(jobs)} "
                      f"ok={counts['ok']} skip={counts['skip']} "
                      f"infeasible={counts['infeasible']} {rate:.2f}/s eta {eta:.0f}min",
                      flush=True)
    n = rebuild_manifest(out_dir)
    print(f"done: {counts}, manifest {n} rows -> {out_dir}")


if __name__ == "__main__":
    main()
