"""Build field fine-tune/validation/blind samples in the synthetic npz format.

Uses the .fdr files' EMBEDDED vendor impulse/step tables (identical to the
canonical transform, verified to float32) resampled onto the fixed 0.25 m
grid, with labels anchored from the curated engineer annotations (response
peak located near each annotated position, observability from the response).

    python prep_field_train.py
  -> E:\\CableNNDataSet\\ZCode\\field_train\\phases\\*.npz + manifest.csv
     splits: field_train (fine-tune), field_val, blind (sealed)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dg_v4.labeling import sigma_m, SNR_OBSERVABLE

FIELD_PHASES = Path(r"E:\CableNNDataSet\ZCode\field_data")
OUT = Path(r"E:\CableNNDataSet\ZCode\field_train")
GRID_BINS = 11008
DIST_STEP = 0.25
GRID_M = GRID_BINS * DIST_STEP
N_VALID_MIN = 400


def grid_distance() -> np.ndarray:
    return np.arange(GRID_BINS, dtype=np.float64) * DIST_STEP


def resample_table(dist: np.ndarray, rho: np.ndarray, dgrid: np.ndarray) -> np.ndarray:
    if dist.size < 2:
        return np.zeros_like(dgrid)
    order = np.argsort(dist)
    return np.interp(dgrid, dist[order], rho[order]).astype(np.float32)


def anchor_near(impulse: np.ndarray, dgrid: np.ndarray, pos_m: float) -> tuple[float, float]:
    """Peak anchor and SNR near an annotated position."""
    w = max(8.0, 0.04 * pos_m)
    lo = int(max(0, (pos_m - w) / DIST_STEP))
    hi = int(min(GRID_BINS - 1, (pos_m + w) / DIST_STEP))
    if hi <= lo:
        return pos_m, 0.0
    k = lo + int(np.argmax(np.abs(impulse[lo:hi + 1])))
    f_lo = min(GRID_BINS - 1, int((pos_m + 2 * w) / DIST_STEP))
    f_hi = min(GRID_BINS - 1, int((pos_m + 6 * w) / DIST_STEP))
    if f_hi > f_lo:
        floor = float(np.sqrt(np.mean(impulse[f_lo:f_hi + 1] ** 2))) + 1e-9
    else:
        floor = float(np.sqrt(np.mean(impulse ** 2))) + 1e-9
    return float(dgrid[k]), float(abs(impulse[k]) / floor)


def process_phase(row: dict, src: Path, dgrid: np.ndarray) -> dict | None:
    with np.load(src, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        if not meta.get("valid"):
            return None
        imp = resample_table(z["impulse_dist"], z["impulse_rho"], dgrid)
        stp = resample_table(z["step_dist"], z["step_rho"], dgrid)
    curated = meta.get("curated") or {}
    length_m = (curated.get("length_m") or meta.get("x_max_m") or 0) or 0
    events = []
    for i, jp in enumerate(curated.get("joints_m") or []):
        if 0 < jp < GRID_M:
            anchor, snr = anchor_near(imp, dgrid, float(jp))
            events.append(dict(event_id=f"joint_{i:02d}", role="joint", mechanism="joint",
                               physical_center_m=float(jp), axis_pos_m=float(jp),
                               anchor_m=anchor, snr=snr, observable=True,  # engineer-annotated: visible by definition
                               severity=0.5))
    for i, (s0, s1) in enumerate(curated.get("defect_intervals_m") or []):
        c = 0.5 * (s0 + s1)
        if 0 < c < GRID_M:
            anchor, snr = anchor_near(imp, dgrid, float(c))
            events.append(dict(event_id=f"defect_{i:02d}", role="defect",
                               mechanism="moisture_local", physical_center_m=float(c),
                               axis_pos_m=float(c), anchor_m=anchor, snr=snr,
                               observable=True, severity=0.7))
    if length_m > 0 and length_m < GRID_M:
        # Response-anchored end: plate lengths differ from electrical length by
        # 10-100+ m in this corpus; the strongest far peak is the physical end.
        lo = int(0.6 * length_m / DIST_STEP)
        hi = int(min(GRID_M, 1.35 * length_m) / DIST_STEP)
        k = lo + int(np.argmax(np.abs(imp[lo:hi])))
        end_anchor = float(dgrid[k])
        floor_zone = imp[int(0.7 * length_m / DIST_STEP):int(0.95 * length_m / DIST_STEP)]
        snr = float(abs(imp[k]) / (np.sqrt(np.mean(floor_zone ** 2)) + 1e-9)) if floor_zone.size else 0.0
        events.append(dict(event_id="terminal", role="terminal", mechanism="open",
                           physical_center_m=float(length_m), axis_pos_m=float(length_m),
                           anchor_m=end_anchor, snr=snr, observable=True,
                           severity=1.0, plate_m=float(length_m)))
        length_m = end_anchor  # valid-length window follows the electrical end
    valid_len = int(min(GRID_BINS, (length_m * 1.15 + 10) / DIST_STEP if length_m else GRID_BINS))
    valid_len = max(valid_len, N_VALID_MIN)
    imp[valid_len:] = 0.0
    stp[valid_len:] = 0.0
    logc = np.sign(imp) * np.log1p(np.abs(imp) * 200.0)
    dist_ch = (np.arange(GRID_BINS, dtype=np.float32) * DIST_STEP) / GRID_M
    x = np.stack([imp, logc, stp, dist_ch])
    x[:, valid_len:] = 0.0
    stem = row["stem"]
    np.savez(OUT / "phases" / f"{stem}.npz",
             impulse=imp.astype(np.float32), step=stp.astype(np.float32),
             role=np.zeros(0, dtype=np.int8), mechanism=np.zeros(0, dtype=np.int8),
             meta=np.array(json.dumps(dict(events=events, valid_len=valid_len,
                                           split=row["split"], profile="field",
                                           length_m=length_m,
                                           source=row["file"], folder=row["folder"],
                                           phase=row["phase"]), ensure_ascii=False)))
    return dict(file=f"{stem}.npz", split=row["split"], profile="field",
                length_m=length_m, n_events=len(events))


def main() -> None:
    (OUT / "phases").mkdir(parents=True, exist_ok=True)
    dgrid = grid_distance()
    rows = []
    with (FIELD_PHASES / "manifest.csv").open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    # split: blind stays blind; other field phases hash-split 85/15 train/val
    out_rows = []
    for row in rows:
        if row.get("error") or not row.get("stem"):
            continue
        split = row["split"]
        if split != "blind":
            h = hash(row["stem"]) % 100
            split = "field_val" if h < 15 else "field_train"
        if not row.get("valid") or row["valid"] in ("False", "false", "0"):
            continue
        row = dict(row, split=split)
        rec = process_phase(row, FIELD_PHASES / "phases" / f"{row['stem']}.npz", dgrid)
        if rec:
            out_rows.append(rec)
    with (OUT / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    counts = {}
    for r in out_rows:
        counts[r["split"]] = counts.get(r["split"], 0) + 1
    print(f"field samples: {counts} -> {OUT}")


if __name__ == "__main__":
    main()
