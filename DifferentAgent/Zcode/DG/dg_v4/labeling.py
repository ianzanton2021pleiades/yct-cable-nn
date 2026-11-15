"""Response-derived labels: each event's label anchor is the actual peak found
in the transformed response near its theoretical position (response overrides
geometry on conflict), with a per-event observability score.
"""
from __future__ import annotations

import numpy as np

ROLE_BACKGROUND, ROLE_JOINT, ROLE_TERMINAL, ROLE_DEFECT = 0, 1, 2, 3
MECH = {"background": 0, "joint": 1, "moisture_local": 2, "moisture_distributed": 3,
        "aging": 4, "low_resistance_short": 5, "series_resistance": 6, "void": 7,
        "terminal_open": 8, "terminal_short": 9, "terminal_load": 10}
SNR_OBSERVABLE = 4.0


def sigma_m(axis_pos_m: float) -> float:
    return max(3.0, 0.02 * axis_pos_m)


def label_events(model, distance: np.ndarray, impulse: np.ndarray,
                 offset_m: float = 0.0) -> list[dict]:
    """Locate every event on the response axis and score its observability."""
    step = float(distance[1] - distance[0])
    out = []
    n = distance.size
    for ev in model.events:
        pos = model.travel_delay_rt(ev.center_m) * 0.5 * model.v_axis_m_per_s - offset_m
        w = max(8.0, 0.04 * pos)
        lo = max(0, int((pos - w) / step))
        hi = min(n - 1, int((pos + w) / step))
        if hi <= lo:
            anchor, snr = pos, 0.0
        else:
            k = lo + int(np.argmax(np.abs(impulse[lo:hi + 1])))
            anchor = float(distance[k])
            if ev.role == "terminal":
                f_lo, f_hi = max(0, int((pos - 6 * w) / step)), max(0, int((pos - 2 * w) / step))
            else:
                f_lo, f_hi = min(n - 1, int((pos + 2 * w) / step)), min(n - 1, int((pos + 6 * w) / step))
            if f_hi > f_lo:
                floor = float(np.sqrt(np.mean(np.abs(impulse[f_lo:f_hi + 1]) ** 2))) + 1e-9
            else:
                floor = float(np.sqrt(np.mean(np.abs(impulse) ** 2))) + 1e-9
            snr = float(np.abs(impulse[k]) / floor)
        start_pos = model.travel_delay_rt(ev.start_m) * 0.5 * model.v_axis_m_per_s - offset_m
        end_pos = model.travel_delay_rt(ev.end_m) * 0.5 * model.v_axis_m_per_s - offset_m
        out.append(dict(event_id=ev.event_id, role=ev.role, mechanism=ev.mechanism,
                        physical_center_m=ev.center_m, physical_start_m=ev.start_m,
                        physical_end_m=ev.end_m,
                        axis_pos_m=float(pos), anchor_m=anchor,
                        axis_start_m=float(start_pos), axis_end_m=float(end_pos),
                        snr=snr, observable=bool(snr >= SNR_OBSERVABLE),
                        severity=float(ev.severity)))
    return out


def build_label_grids(events: list[dict], distance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    step = float(distance[1] - distance[0])
    n = distance.size
    role = np.zeros(n, dtype=np.int8)
    mech = np.zeros(n, dtype=np.int8)
    for ev in events:
        if ev["role"] == "joint":
            role_id, mech_id, s, e = ROLE_JOINT, MECH["joint"], ev["anchor_m"], ev["anchor_m"]
        elif ev["role"] == "terminal":
            role_id, mech_id = ROLE_TERMINAL, MECH[f"terminal_{ev['mechanism']}"]
            s, e = ev["anchor_m"], ev["anchor_m"]
        else:
            role_id, mech_id = ROLE_DEFECT, MECH[ev["mechanism"]]
            s, e = ev["axis_start_m"], ev["axis_end_m"]
        half = sigma_m(max(abs(s), abs(e)))
        lo = max(0, int((s - half) / step))
        hi = min(n - 1, int((e + half) / step))
        if hi >= lo:
            role[lo:hi + 1] = role_id
            mech[lo:hi + 1] = mech_id
    return role, mech
