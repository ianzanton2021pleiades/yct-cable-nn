"""Shared field-vs-synthetic response statistics (DG_V5 calibration/acceptance).

Same metric conventions on both domains (0.25 m grid, vendor-transform impulse):
- event peak: max |impulse| in +-max(8 m, 4 % of axis position) around the anchor
- event SNR: peak / RMS floor in 2w..6w ahead (terminal) / behind (other events)
- clutter RMS: 20-60 %L and 60-90 %L spans, event windows excluded
- near-end RMS: 0-30 m
- depth buckets: 0-500/500-1000/1000-1500/1500-2000/2000-2500 m
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DIST_STEP = 0.25
GRID_BINS = 11008
GRID_M = GRID_BINS * DIST_STEP
BUCKETS = [(0.0, 500.0), (500.0, 1000.0), (1000.0, 1500.0), (1500.0, 2000.0), (2000.0, 2500.0)]
SNR_OBSERVABLE = 4.0
CLUTTER_SPANS = (("20_60", 0.20, 0.60), ("60_90", 0.60, 0.90))


def grid_distance() -> np.ndarray:
    return np.arange(GRID_BINS, dtype=np.float64) * DIST_STEP


def resample_table(dist: np.ndarray, rho: np.ndarray, dgrid: np.ndarray) -> np.ndarray:
    if dist.size < 2:
        return np.zeros_like(dgrid)
    order = np.argsort(dist)
    return np.interp(dgrid, dist[order], rho[order]).astype(np.float32)


def event_window(pos_m: float) -> float:
    return max(8.0, 0.04 * pos_m)


def _bucket_index(pos_m: float) -> int | None:
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= pos_m < hi:
            return i
    return None


def response_stats(imp: np.ndarray, length_m: float, events: list[dict]) -> dict:
    """Stats for one response curve.  events: dicts with role, mechanism,
    anchor_m, axis_pos_m, snr, observable (as produced by prep/label_events).
    Events whose window overlaps the terminal window are dropped from the
    role/bucket statistics (their window max is contaminated by the end rise —
    real in both domains, but far stronger for sharp synthetic ends)."""
    n = imp.size
    out_events = []
    excluded = np.zeros(n, dtype=bool)
    windows = []
    for ev in events:
        anchor = float(ev["anchor_m"])
        if not (0.0 <= anchor < GRID_M):
            continue
        w = event_window(float(ev.get("axis_pos_m", anchor)) or anchor)
        windows.append((ev["role"], anchor, w))
    term = next(((a, w) for r, a, w in windows if r == "terminal"), None)
    for ev, (role, anchor, w) in zip([e for e in events
                                      if 0.0 <= float(e["anchor_m"]) < GRID_M], windows):
        lo = max(0, int((anchor - w) / DIST_STEP))
        hi = min(n - 1, int((anchor + w) / DIST_STEP))
        if hi > lo:
            amp = float(np.max(np.abs(imp[lo:hi + 1])))
            excluded[lo:hi + 1] = True
        else:
            amp = 0.0
        contaminated = (term is not None and role != "terminal"
                        and not (anchor + w < term[0] - term[1] or anchor - w > term[0] + term[1]))
        out_events.append(dict(role=role, mechanism=ev.get("mechanism", ""),
                               anchor_m=anchor, amp=amp,
                               snr=float(ev.get("snr", 0.0)),
                               observable=bool(ev.get("observable", False))
                               and not contaminated))
    clutter = {}
    for name, lo_f, hi_f in CLUTTER_SPANS:
        lo, hi = int(lo_f * length_m / DIST_STEP), int(hi_f * length_m / DIST_STEP)
        hi = min(hi, n)
        lo = min(lo, n)
        seg = imp[lo:hi][~excluded[lo:hi]]
        clutter[name] = float(np.sqrt(np.mean(seg ** 2))) if seg.size else np.nan
    near = imp[:min(n, int(30.0 / DIST_STEP))]
    return dict(events=out_events,
                clutter=clutter,
                near_0_30=float(np.sqrt(np.mean(near ** 2))) if near.size else np.nan,
                length_m=float(length_m))


def end_anchor(imp: np.ndarray, length_m: float) -> tuple[float, float]:
    """Response-anchored end + SNR (floor zone 0.7-0.95 L), prep_field_train convention."""
    n = imp.size
    lo = int(0.6 * length_m / DIST_STEP)
    hi = int(min(GRID_M, 1.35 * length_m) / DIST_STEP)
    hi = min(hi, n)
    if hi <= lo:
        return float(length_m), 0.0
    k = lo + int(np.argmax(np.abs(imp[lo:hi])))
    floor_zone = imp[int(0.7 * length_m / DIST_STEP):int(0.95 * length_m / DIST_STEP)]
    snr = float(abs(imp[k]) / (np.sqrt(np.mean(floor_zone ** 2)) + 1e-9)) if floor_zone.size else 0.0
    return float(k * DIST_STEP), snr


def snr_at(imp: np.ndarray, pos_m: float) -> tuple[float, float]:
    """(peak amp, SNR) at pos_m with the labeling.floor convention (2w..6w behind),
    same population logic as the synthetic label_events observable gate."""
    n = imp.size
    w = event_window(pos_m)
    lo = max(0, int((pos_m - w) / DIST_STEP))
    hi = min(n - 1, int((pos_m + w) / DIST_STEP))
    if hi <= lo:
        return 0.0, 0.0
    k = lo + int(np.argmax(np.abs(imp[lo:hi + 1])))
    amp = float(np.abs(imp[k]))
    f_lo = min(n - 1, int((pos_m + 2 * w) / DIST_STEP))
    f_hi = min(n - 1, int((pos_m + 6 * w) / DIST_STEP))
    floor = float(np.sqrt(np.mean(imp[f_lo:f_hi + 1] ** 2))) + 1e-9 if f_hi > f_lo \
        else float(np.sqrt(np.mean(imp ** 2))) + 1e-9
    return amp, amp / floor


def field_reference(phases_dir: Path, manifest_csv: Path) -> dict:
    """Aggregate response statistics over the field corpus (vendor tables)."""
    import csv
    rows = []
    with open(manifest_csv, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    dgrid = grid_distance()
    agg = {"event_amps": {r: {i: [] for i in range(len(BUCKETS))} for r in ("joint", "terminal", "defect")},
           "event_snrs": {r: [] for r in ("joint", "terminal", "defect")},
           "clutter": {n: [] for n, _, _ in CLUTTER_SPANS},
           "near_0_30": [], "lengths": [], "n_files": 0, "n_skipped": 0}
    for row in rows:
        if row.get("error") or not row.get("stem") or not row.get("valid"):
            continue
        if row["valid"] not in ("True", "true", "1"):
            continue
        src = phases_dir / f"{row['stem']}.npz"
        if not src.exists():
            continue
        with np.load(src, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            imp = resample_table(z["impulse_dist"], z["impulse_rho"], dgrid)
        curated = meta.get("curated") or {}
        length_m = curated.get("length_m") or 0.0
        events = []
        if length_m <= 0:
            # no annotation: anchor the end on the strongest far peak
            xmax = float(meta.get("x_max_m") or 0)
            if xmax <= 100:
                agg["n_skipped"] += 1
                continue
            length_m = min(xmax * 0.95, GRID_M)
        for jp in curated.get("joints_m") or []:
            if 0 < float(jp) < GRID_M:
                amp, snr = snr_at(imp, float(jp))
                events.append(dict(role="joint", mechanism="joint", anchor_m=float(jp),
                                   axis_pos_m=float(jp), snr=snr,
                                   observable=snr >= SNR_OBSERVABLE, amp=amp))
        for s0, s1 in curated.get("defect_intervals_m") or []:
            c = 0.5 * (float(s0) + float(s1))
            if 0 < c < GRID_M:
                amp, snr = snr_at(imp, c)
                events.append(dict(role="defect", mechanism="moisture_local", anchor_m=c,
                                   axis_pos_m=c, snr=snr,
                                   observable=snr >= SNR_OBSERVABLE, amp=amp))
        anchor, snr = end_anchor(imp, length_m)
        events.append(dict(role="terminal", mechanism="open", anchor_m=anchor,
                           axis_pos_m=anchor, snr=snr, observable=snr >= SNR_OBSERVABLE))
        st = response_stats(imp, length_m, events)
        agg["n_files"] += 1
        agg["lengths"].append(length_m)
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


def synth_reference_from_dir(data_dir: Path, max_files: int | None = None) -> dict:
    """Aggregate response statistics over a generated npz dataset directory."""
    files = sorted((data_dir / "phases").glob("sample_*.npz"))
    if max_files:
        files = files[:max_files]
    agg = empty_agg()
    for p in files:
        with np.load(p, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            imp = z["impulse"]
        if meta.get("profile") != "field":
            continue
        st = response_stats(imp, float(meta["length_m"]), meta["events"])
        agg["n_files"] += 1
        agg["lengths"].append(float(meta["length_m"]))
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


def empty_agg() -> dict:
    return {"event_amps": {r: {i: [] for i in range(len(BUCKETS))} for r in ("joint", "terminal", "defect")},
            "event_snrs": {r: [] for r in ("joint", "terminal", "defect")},
            "clutter": {n: [] for n, _, _ in CLUTTER_SPANS},
            "near_0_30": [], "lengths": [], "n_files": 0, "n_skipped": 0}


def median_table(agg: dict) -> dict:
    """role -> bucket -> median amplitude (nan when empty), plus snr/clutter."""
    out = {"amps": {}, "snr": {}, "clutter": {}, "near_0_30": float(np.nanmedian(agg["near_0_30"]))
           if agg["near_0_30"] else float("nan"),
           "clutter_n": {}, "amp_n": {}}
    for role, buckets in agg["event_amps"].items():
        out["amps"][role] = [float(np.median(v)) if v else float("nan") for k, v in sorted(buckets.items())]
        out["amp_n"][role] = [len(v) for _, v in sorted(buckets.items())]
        sn = agg["event_snrs"][role]
        out["snr"][role] = float(np.median(sn)) if sn else float("nan")
    for k, v in agg["clutter"].items():
        out["clutter"][k] = float(np.median(v)) if v else float("nan")
        out["clutter_n"][k] = len(v)
    return out


def ratio_table(field: dict, synth: dict) -> dict:
    """field/synth ratios of the median tables (the calibration target)."""
    fm, sm = median_table(field), median_table(synth)
    out = {"amps": {}, "snr": {}, "clutter": {}, "near_0_30": fm["near_0_30"] / sm["near_0_30"]
           if sm["near_0_30"] == sm["near_0_30"] and sm["near_0_30"] > 0 else float("nan")}
    for role in fm["amps"]:
        out["amps"][role] = [f / s if (f == f and s == s and s > 0) else float("nan")
                             for f, s in zip(fm["amps"][role], sm["amps"][role])]
        out["snr"][role] = fm["snr"][role] / sm["snr"][role] if (sm["snr"][role] == sm["snr"][role]
                                                                 and sm["snr"][role] > 0) else float("nan")
    for k in fm["clutter"]:
        out["clutter"][k] = fm["clutter"][k] / sm["clutter"][k] if (sm["clutter"][k] == sm["clutter"][k]
                                                                   and sm["clutter"][k] > 0) else float("nan")
    return out


def fmt_ratio_table(ratios: dict, field_med: dict, synth_med: dict) -> str:
    lines = [f"{'role/bucket':>16} {'field':>10} {'synth':>10} {'f/s':>6}"]
    for role in ("joint", "terminal", "defect"):
        for i, (lo, hi) in enumerate(BUCKETS):
            label = f"{role}_{lo:.0f}-{hi:.0f}"
            lines.append(f"{label:>16} {field_med['amps'][role][i]:10.3e} "
                         f"{synth_med['amps'][role][i]:10.3e} {ratios['amps'][role][i]:6.2f}")
    for k in field_med["clutter"]:
        lines.append(f"{'clutter_' + k:>16} {field_med['clutter'][k]:10.3e} "
                     f"{synth_med['clutter'][k]:10.3e} {ratios['clutter'][k]:6.2f}")
    lines.append(f"{'near_0_30':>16} {field_med['near_0_30']:10.3e} "
                 f"{synth_med['near_0_30']:10.3e} {ratios['near_0_30']:6.2f}")
    for role in ("joint", "terminal", "defect"):
        lines.append(f"{'snr_' + role:>16} {field_med['snr'][role]:10.2f} "
                     f"{synth_med['snr'][role]:10.2f} {ratios['snr'][role]:6.2f}")
    return "\n".join(lines)
