"""Decode predictions with an explicit near-end fixture suppression zone."""
import numpy as np
from scipy.ndimage import binary_closing, label
from scipy.signal import find_peaks


NEAR_END_SUPPRESSION_M = 30.0
EVENT_PROMINENCE = 0.10


def decode_predictions(probabilities, cfg, thresholds=None):
    p = np.asarray(probabilities, dtype=float)
    t = {"event": 0.5, "interval": 0.5, "terminal": 0.5, **(thresholds or {})}
    d = np.linspace(0, cfg["response"]["distance_max_m"], p.shape[-1])
    dx = d[1] - d[0]
    opts = cfg.get("evaluation", {})
    near = float(opts.get("near_exclusion_m", NEAR_END_SUPPRESSION_M))
    valid = (d >= near) & (d <= d[-1] - 10.0)
    peaks, _ = find_peaks(
        p[0],
        height=t["event"],
        distance=max(1, int(round(opts.get("event_min_separation_m", 3.0) / dx))),
        prominence=float(opts.get("event_prominence", EVENT_PROMINENCE)),
    )
    events = [
        {"distance_m": float(d[i]), "confidence": float(p[0, i])}
        for i in peaks
        if valid[i]
    ]
    low = min(t["interval"], 0.7 * t["interval"])
    mask = (p[1] >= low) & valid
    gap = max(1, int(round(opts.get("interval_merge_gap_m", 2.0) / dx)))
    mask = binary_closing(mask, structure=np.ones(gap + 1)) & valid
    groups, count = label(mask)
    intervals = []
    for k in range(1, count + 1):
        ix = np.flatnonzero(groups == k)
        if np.max(p[1, ix]) < t["interval"]:
            continue
        if d[ix[-1]] - d[ix[0]] < opts.get("interval_min_width_m", 3.0):
            continue
        intervals.append({
            "start_m": float(d[ix[0]]),
            "end_m": float(d[ix[-1]]),
            "confidence": float(np.mean(p[1, ix])),
        })
    ix, _ = find_peaks(p[2], height=t["terminal"])
    ix = [i for i in ix if valid[i]]
    best = max(ix, key=lambda i: p[2, i], default=None)
    terminal = (
        None
        if best is None
        else {"distance_m": float(d[best]), "confidence": float(p[2, best])}
    )
    return {"events": events, "intervals": intervals, "terminal": terminal}
