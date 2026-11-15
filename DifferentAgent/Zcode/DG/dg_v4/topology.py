"""Random cable topologies with mechanism-labelled defects.

Sampling follows the 2026-10 field/RG58 surveys and the HumanDoc Chinese
technical reports (defect type -> step/impulse response signatures):
moisture   = local C and G increase with gradated axial boundaries,
aging      = milder C/G increase over a longer interval,
short      = hard Z0 collapse over a short interval,
series     = shield corrosion / bad crimp modelled as a series resistance lump
             (signature: downstream reflections weakened),
void       = local capacitance decrease (positive pulse).

Every draw is rejection-sampled: if any segment's loss split is infeasible the
whole draw is resampled, so generation never hard-fails.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .materials import (SegmentSpec, derive_material, dielectric_alpha_db_per_m,
                        group_delay_s_per_m)

F_DELAY_REF = 100.0e6

# mechanism quota (first-round failure: moisture+aging designed >=40 %, realized
# 3.9 % because infeasible defect params forced a whole-cable reseed)
MOISTURE_MECHS = ("moisture_local", "moisture_distributed", "aging")
MA_QUOTA = 0.40
DEFECT_PARAM_REDRAWS = 40     # per-cable budget for (z0, epsr, alpha) redraws
DEFECT_DROPS_AFTER = 12       # redraws before a defect is dropped

# ---------------------------------------------------------------------------
# parameter ranges (initial field-calibration estimates; frozen after fitting)
# ---------------------------------------------------------------------------
P = {
    "field": {
        "length_m": (80.0, 2600.0),
        "z0_ohm": (14.0, 30.0),        # field_fit_v5: 10-90 pct 15.8-23.7 + padding
        "epsr": (2.90, 3.25),
        "alpha_db_per_m_100mhz": (0.026, 0.059),  # soak-trial decay anchor + T3
        # terminal-depth regression, frozen at run5 (calibration_notes_v5.md §3/§4)
        "tan_delta_100mhz": (0.0001, 0.0012),
        "joint_spacing_m": (80.0, 260.0),
        "joint_series_ohm": (0.085, 2.56),  # T3: field/synth joint-peak medians
        "joint_shunt_pf": (0.43, 8.5),      # (run5, matched-SNR populations)
        "open_load_ohm": (800.0, 20000.0),
        "short_load_ohm": (0.05, 2.0),
        "termination_weights": {"open": 0.60, "load": 0.30, "short": 0.10},
        "defect_mechanism_weights": {
            "moisture_local": 0.24, "moisture_distributed": 0.13,
            "aging": 0.17, "low_resistance_short": 0.20,
            "series_resistance": 0.14, "void": 0.06,
        },
        "defect_count_probs": {  # 0..5 defects by length bucket
            "short":  [0.35, 0.35, 0.20, 0.08, 0.02, 0.0],
            "mid":    [0.25, 0.32, 0.25, 0.12, 0.05, 0.01],
            "long":   [0.20, 0.28, 0.26, 0.14, 0.08, 0.04],
        },
    },
    "rg58": {
        "length_m": (8.0, 100.0),
        "z0_ohm": (47.0, 48.5),        # COMSOL high-frequency Z0 ~ 47.6
        "epsr": (2.20, 2.34),          # vp 199.3 Mm/s -> 2.26
        "alpha_db_per_m_100mhz": (0.09, 0.13),  # COMSOL 0.109
        "tan_delta_100mhz": (0.00005, 0.0006),
        "joint_spacing_m": (25.0, 60.0),
        "joint_series_ohm": (0.005, 0.10),   # BNC nearly transparent
        "joint_shunt_pf": (0.05, 1.5),
        "open_load_ohm": (2000.0, 50000.0),
        "short_load_ohm": (0.02, 0.5),
        "termination_weights": {"open": 0.75, "load": 0.15, "short": 0.10},
        "defect_mechanism_weights": {
            "moisture_local": 0.15, "aging": 0.10,
            "low_resistance_short": 0.25, "series_resistance": 0.35, "void": 0.15,
        },
        "defect_count_probs": {
            "short": [0.55, 0.30, 0.12, 0.03, 0.0, 0.0],
        },
    },
}

DEFECT_SPEC = {
    # interval fraction of length, abs length bounds, z0 factor, epsr delta,
    # alpha factor, debye (delta_epsr scale of epsr_delta, corner, exponent), tan-delta range
    "moisture_local": dict(frac=(0.04, 0.12), len_m=(5.0, 220.0), z0=(0.881, 0.977),
                           depsr=(0.14, 0.88), alpha=(3.0, 16.0), tan=(0.002, 0.015),
                           debye=(0.45, 8.0e6, 20.0e6), grad=(0.12, 0.60, 0.25)),
    "moisture_distributed": dict(frac=(0.18, 0.50), len_m=(90.0, 900.0), z0=(0.92, 0.99),
                                 depsr=(0.15, 0.55), alpha=(3.0, 10.0), tan=(0.002, 0.012),
                                 debye=(0.40, 6.0e6, 14.0e6), grad=(0.10, 0.52, 0.30)),
    "aging": dict(frac=(0.08, 0.20), len_m=(25.0, 380.0), z0=(0.94, 1.06),
                  depsr=(0.03, 0.18), alpha=(1.5, 5.0), tan=(0.0008, 0.005),
                  debye=(0.30, 25.0e6, 45.0e6), grad=(0.18, 0.58, 0.10)),
    "low_resistance_short": dict(frac=(0.0, 0.0), len_m=(0.3, 3.0), z0=(0.15, 0.60),
                                 depsr=(0.0, 0.10), alpha=(5.0, 20.0), tan=(0.001, 0.004),
                                 debye=None, grad=None),
    "void": dict(frac=(0.0, 0.0), len_m=(0.5, 5.0), z0=(1.15, 1.50),
                 depsr=(-0.40, -0.10), alpha=(1.0, 1.8), tan=(0.0002, 0.0012),
                 debye=None, grad=None),
}


@dataclass
class Lump:
    position_m: float
    role: str            # "joint" | "defect"
    mechanism: str       # "joint" | "series_resistance"
    series_ohm: float
    shunt_pf: float
    severity: float


@dataclass
class Event:
    event_id: str
    role: str            # "terminal" | "joint" | "defect"
    mechanism: str
    start_m: float
    center_m: float
    end_m: float
    severity: float
    params: dict = field(default_factory=dict)


@dataclass
class CableModel:
    profile: str
    length_m: float
    base_z0_ohm: float
    base_epsr: float
    base_alpha_db_per_m_100mhz: float
    base_tan_delta_100mhz: float
    temp_c: float
    termination: str
    z_load_ohm: float
    seg_bounds: np.ndarray          # (n+1,)
    seg_specs: list                 # SegmentSpec per segment
    materials: list
    lumps: list                     # Lump sorted by position
    events: list                    # Event sorted by center
    v_axis_m_per_s: float           # distance-axis velocity used by the instrument GUI

    def travel_delay_rt(self, position_m: float) -> float:
        """Round-trip group delay to a physical position at F_DELAY_REF."""
        total = 0.0
        remaining = float(position_m)
        for spec, material in zip(self.seg_specs, self.materials):
            if remaining <= 0:
                break
            take = min(remaining, spec.length_m)
            total += take * group_delay_s_per_m(spec, material)
            remaining -= take
        return 2.0 * total

    def as_dict(self) -> dict:
        return {
            "profile": self.profile, "length_m": self.length_m,
            "base_z0_ohm": self.base_z0_ohm, "base_epsr": self.base_epsr,
            "base_alpha_db_per_m_100mhz": self.base_alpha_db_per_m_100mhz,
            "base_tan_delta_100mhz": self.base_tan_delta_100mhz,
            "temp_c": self.temp_c, "termination": self.termination,
            "z_load_ohm": self.z_load_ohm, "v_axis_m_per_s": self.v_axis_m_per_s,
            "n_segments": len(self.seg_specs),
            "lumps": [l.__dict__ for l in self.lumps],
            "events": [e.__dict__ for e in self.events],
        }


def _u(rng, lo, hi):
    return float(rng.uniform(lo, hi))


def _ulog(rng, lo, hi):
    return float(np.exp(rng.uniform(math.log(lo), math.log(hi))))


def _long_defect_weights(kind: str, norm_pos: float, grad: tuple) -> tuple[float, float]:
    """(electrical_weight, loss_weight) along a long defect, 0..1."""
    if kind == "aging":
        w = math.sin(math.pi * norm_pos) ** 2
        return w, w
    attack, release_start, residual = grad
    smooth = lambda v: float(np.clip(v, 0.0, 1.0)) ** 2 * (3.0 - 2.0 * float(np.clip(v, 0.0, 1.0)))
    electrical = smooth(norm_pos / attack) * (1.0 - smooth((norm_pos - release_start) / (1.0 - release_start)))
    loss = 1.0 - (1.0 - residual) * smooth((norm_pos - min(0.78, release_start + 0.15)) / (1.0 - min(0.78, release_start + 0.15)))
    return electrical, smooth(norm_pos / attack) * loss


def _alpha_factor(rng, d: dict, base_alpha: float, base_tand: float, epsr: float) -> float:
    """Draw alpha_factor >= the segment's own dielectric-loss floor.

    A wet/aged segment's total attenuation can never sit below the loss of its
    own dielectric (tan-delta + Debye); drawing below that floor is what made
    moisture/aging draws infeasible and collapsed the first-round quota.
    """
    spec = DEFECT_SPEC.get(d["mechanism"])
    if spec is None:
        return 1.0
    floor = 0.0
    if d.get("debye") is not None and d["mechanism"] in MOISTURE_MECHS:
        delta = float(np.clip(abs(d["epsr_delta"]) * d["debye"][0], 0.0, 0.5))
        seg = SegmentSpec("field", 1.0, 20.0, epsr + d["epsr_delta"], 1.0,
                          float(d.get("tan_val", spec["tan"][0])),
                          debye_delta_epsr=delta,
                          debye_corner_hz=float(d.get("debye_corner") or d["debye"][1]))
        base_seg = SegmentSpec("field", 1.0, 20.0, epsr, 1.0, base_tand)
        base_cond = max(base_alpha - dielectric_alpha_db_per_m(base_seg), 0.0)
        floor = max((dielectric_alpha_db_per_m(seg) + base_cond) / base_alpha, 0.0)
    lo = max(spec["alpha"][0], floor)
    hi = max(spec["alpha"][1], lo * 1.25)
    return _u(rng, lo, hi)


def _sample_defect(rng, profile: str, length_m: float, mechanism: str) -> dict:
    if mechanism == "series_resistance":   # point lump, interval unused
        return dict(mechanism=mechanism, length_m=0.0, z0_factor=1.0,
                    epsr_delta=0.0, alpha_factor=1.0, tan_lo=0.0, tan_hi=0.0,
                    debye=None, grad=None, profile_shape="smooth", profile_width_m=0.0)
    spec = DEFECT_SPEC[mechanism]
    if spec["frac"] == (0.0, 0.0):
        dlen = _u(rng, *spec["len_m"])
    else:
        dlen = float(np.clip(length_m * _u(rng, *spec["frac"]), *spec["len_m"]))
    dlen = min(dlen, 0.6 * length_m)
    shape, width = "smooth", 0.0
    if spec["grad"] is not None and rng.random() < 0.40:   # gauss gradient family
        shape = "gauss"
        width = _u(rng, 0.15, 0.35) * dlen
    tan_val = _u(rng, *spec["tan"])
    corner = _u(rng, spec["debye"][1], spec["debye"][2]) if spec["debye"] else 80.0e6
    return dict(mechanism=mechanism, length_m=dlen,
                z0_factor=_u(rng, *spec["z0"]),
                epsr_delta=_u(rng, *spec["depsr"]),
                alpha_factor=_u(rng, *spec["alpha"]),
                tan_val=tan_val, debye_corner=corner,
                debye=spec["debye"], grad=spec["grad"],
                profile_shape=shape, profile_width_m=width)


def _resample_defect_params(rng, d: dict) -> None:
    """Redraw only the severity combo of one defect (feasibility repair)."""
    if d["mechanism"] == "series_resistance":
        return
    spec = DEFECT_SPEC[d["mechanism"]]
    d["z0_factor"] = _u(rng, *spec["z0"])
    d["epsr_delta"] = _u(rng, *spec["depsr"])
    d["alpha_factor"] = _u(rng, *spec["alpha"])
    d["tan_val"] = _u(rng, *spec["tan"])
    if spec["debye"]:
        d["debye_corner"] = _u(rng, spec["debye"][1], spec["debye"][2])


def _draw_mechanisms(rng, p: dict, count: int) -> list[str]:
    """Draw defect mechanisms with the moisture+aging >= 40 % quota enforced by
    redrawing the mechanism draw only (parameters/positions stay independent)."""
    if count <= 0:
        return []
    names = list(p["defect_mechanism_weights"])
    w = np.asarray([p["defect_mechanism_weights"][m] for m in names])
    w = w / w.sum()
    need = math.ceil(MA_QUOTA * count)
    for _ in range(20):
        mechs = [str(rng.choice(names, p=w)) for _ in range(count)]
        if sum(m in MOISTURE_MECHS for m in mechs) >= need:
            return mechs
    ma_names = [m for m in names if m in MOISTURE_MECHS]
    rest = [m for m in names if m not in MOISTURE_MECHS]
    fill = [str(rng.choice(ma_names,
                           p=np.asarray([p["defect_mechanism_weights"][m] for m in ma_names])
                           / sum(p["defect_mechanism_weights"][m] for m in ma_names)))
            for _ in range(need)]
    if rest:
        w_rest = np.asarray([p["defect_mechanism_weights"][m] for m in rest])
        w_rest = w_rest / w_rest.sum()
        fill += [str(rng.choice(rest, p=w_rest)) for _ in range(count - need)]
    rng.shuffle(fill)
    return fill


def _defect_count(rng, profile: str, length_m: float) -> int:
    p = P[profile]
    if profile == "field":
        bucket = "short" if length_m < 300 else ("mid" if length_m < 900 else "long")
        probs = p["defect_count_probs"][bucket]
    else:
        probs = p["defect_count_probs"]["short"]
    return int(rng.choice(len(probs), p=np.asarray(probs) / np.sum(probs)))


def build_topology(profile: str, seed: int, overrides: dict | None = None) -> CableModel | None:
    """One labelled cable topology per seed.  Returns None on infeasible draw."""
    rng = np.random.default_rng(seed)
    req = dict(overrides or {})
    p = P[profile]
    length_m = float(req.get("length_m", _u(rng, *p["length_m"])))
    if req.get("length_m") is not None:
        length_m = float(req["length_m"])
        if not (p["length_m"][0] <= length_m <= p["length_m"][1]):
            return None
    z0 = _u(rng, *p["z0_ohm"])
    epsr = _u(rng, *p["epsr"])
    alpha = _u(rng, *p["alpha_db_per_m_100mhz"])
    tand = _u(rng, *p["tan_delta_100mhz"])
    temp_c = float(req.get("temp_c", _u(rng, 15.0, 45.0)))
    base = req.get("base") or {}
    z0 = float(base.get("z0", z0))
    epsr = float(base.get("epsr", epsr))
    alpha = float(base.get("alpha", alpha))
    tand = float(base.get("tand", tand))
    # per-cable base dispersion (frequency-dependent velocity): a weak Debye
    # relaxation smears deep reflections the way real cables do (soak trial:
    # dip width 0.8 m @100 m -> 73 m @1100 m; pure attenuation cannot produce
    # this, and terminal amplitudes at depth come out 3-10x too strong without
    # it).  HF (100 MHz) properties are untouched.  The relaxation strength is
    # capped so its 100 MHz dielectric loss never eats the attenuation budget.
    coef = dielectric_alpha_db_per_m(SegmentSpec(profile, 1.0, z0, epsr, 1.0, 1.0))
    if profile == "field":
        debye_base = (_u(rng, 0.10, 0.28), _u(rng, 0.15e6, 0.9e6), 0.9)
        debye_only = dielectric_alpha_db_per_m(
            SegmentSpec(profile, 1.0, z0, epsr, 1.0, 0.0,
                        debye_delta_epsr=debye_base[0], debye_corner_hz=debye_base[1]))
        share = max(debye_only - coef, 0.0)
        if share > 0.4 * alpha:
            debye_base = (debye_base[0] * (0.4 * alpha / share), debye_base[1], 0.9)
    else:
        debye_base = (0.0, 80.0e6, 0.9)
    # a healthy cable's tan-delta can never eat the whole attenuation budget:
    # redraw tand until the dielectric share (tan-delta + base Debye) stays
    # below 85 % of alpha
    debye_share = 0.0
    if debye_base[0] > 0:
        debye_share = max(dielectric_alpha_db_per_m(
            SegmentSpec(profile, 1.0, z0, epsr, 1.0, 0.0,
                        debye_delta_epsr=debye_base[0],
                        debye_corner_hz=debye_base[1])) - coef, 0.0)
    budget = max(0.85 * alpha - debye_share, 0.05 * alpha)
    for _ in range(20):
        if coef * tand < budget:
            break
        tand = _u(rng, *p["tan_delta_100mhz"])

    # --- joints -----------------------------------------------------------
    joint_scale = float(req.get("joint_scale", 1.0))
    joints: list[tuple[float, float, float]] = []  # pos, series, shunt
    spacing_lo, spacing_hi = p["joint_spacing_m"]
    pos = _u(rng, spacing_lo * 0.5, spacing_hi)
    while pos < length_m - max(5.0, length_m * 0.02):
        joints.append((pos, _ulog(rng, *p["joint_series_ohm"]) * joint_scale,
                       _ulog(rng, *p["joint_shunt_pf"]) * joint_scale))
        pos += _u(rng, spacing_lo, spacing_hi)
    if req.get("joint_positions_m") is not None:
        joints = [(float(x), _ulog(rng, *p["joint_series_ohm"]) * joint_scale,
                   _ulog(rng, *p["joint_shunt_pf"]) * joint_scale)
                  for x in req["joint_positions_m"] if 0 < float(x) < length_m]

    # --- defects ----------------------------------------------------------
    defects: list[dict] = []
    if req.get("defects") is not None:
        defects = [dict(d) for d in req["defects"]]
    else:
        count = _defect_count(rng, profile, length_m)
        margin = max(2.0, length_m * 0.02)
        for mech in _draw_mechanisms(rng, p, count):
            d = _sample_defect(rng, profile, length_m, mech)
            d["alpha_factor"] = _alpha_factor(rng, d, alpha, tand, epsr)
            for _ in range(40):  # place without overlap
                start = _u(rng, margin, max(margin + 0.1, length_m - margin - d["length_m"]))
                end = start + d["length_m"]
                if any(not (end < s["start_m"] - 3.0 or start > s["end_m"] + 3.0) for s in defects):
                    continue
                if any(abs((start + end) / 2 - jp) < 3.0 for jp, _, _ in joints):
                    continue
                d.update(start_m=start, end_m=end)
                defects.append(d)
                break
            else:
                continue
    if not defects and req.get("force_defect"):
        return None

    # --- termination ------------------------------------------------------
    term = req.get("termination")
    if term is None:
        names = list(p["termination_weights"])
        w = np.asarray([p["termination_weights"][n] for n in names])
        term = str(rng.choice(names, p=w / w.sum()))
    if term == "open":
        z_load = _u(rng, *p["open_load_ohm"])
    elif term == "short":
        z_load = _u(rng, *p["short_load_ohm"])
    else:
        z_load = float(z0 * _u(rng, 0.7, 1.6))

    # --- segment lattice ---------------------------------------------------
    active = list(defects)

    def defect_weight(d: dict, mid: float):
        """(w_electrical, w_loss) of one defect at position mid, else None."""
        if d["start_m"] - 1e-9 <= mid <= d["end_m"] + 1e-9:
            if d["mechanism"] in ("low_resistance_short", "void"):
                return 1.0, 1.0
            if d.get("profile_shape") == "gauss":
                w = math.exp(-0.5 * ((mid - 0.5 * (d["start_m"] + d["end_m"]))
                                     / max(d["profile_width_m"], 1e-9)) ** 2)
                return w, w
            norm = (mid - d["start_m"]) / max(d["end_m"] - d["start_m"], 1e-9)
            return _long_defect_weights(d["mechanism"], norm, d["grad"])
        if d.get("profile_shape") == "gauss":   # gaussian tails beyond the interval
            w = math.exp(-0.5 * ((mid - 0.5 * (d["start_m"] + d["end_m"]))
                                 / max(d["profile_width_m"], 1e-9)) ** 2)
            if w > 1e-3:
                return w, w
        return None

    def defect_at(mid: float):
        for d in active:
            if d["start_m"] - 1e-9 <= mid <= d["end_m"] + 1e-9:
                w = defect_weight(d, mid)
                return d, w[0], w[1]
        best = None
        for d in active:
            if d.get("profile_shape") != "gauss":
                continue
            w = defect_weight(d, mid)
            if w is not None and (best is None or w[0] > best[1]):
                best = (d, w[0], w[1])
        return best if best is not None else (None, 0.0, 0.0)

    # per-cable base dispersion ... (drawn above, before the tand budget check)

    def build_specs(active_defs: list[dict]):
        bounds = {0.0, length_m}
        for d in active_defs:
            if d["mechanism"] in ("low_resistance_short", "void"):
                bounds.update((d["start_m"], d["end_m"]))
                continue
            lo_b, hi_b = d["start_m"], d["end_m"]
            if d.get("profile_shape") == "gauss":
                lo_b = max(0.0, lo_b - d["profile_width_m"])
                hi_b = min(length_m, hi_b + d["profile_width_m"])
            n_sub = int(np.clip(round((hi_b - lo_b) / (0.02 * length_m if length_m > 100 else 2.0)), 9, 61))
            bounds.update(np.linspace(lo_b, hi_b, n_sub + 1).tolist())
        for jp, _, _ in joints:
            bounds.add(jp)
        ordered = np.asarray(sorted(bounds))
        specs: list[SegmentSpec] = []
        for start, end in zip(ordered[:-1], ordered[1:]):
            mid = 0.5 * (start + end)
            d, w_e, w_l = defect_at(mid)
            seg_z0, seg_eps, seg_alpha, seg_tand = z0, epsr, alpha, tand
            debye = debye_base
            if d is not None:
                seg_z0 = z0 * (1.0 + (d["z0_factor"] - 1.0) * w_e)
                seg_eps = epsr + d["epsr_delta"] * w_e
                seg_alpha = alpha * (1.0 + (d["alpha_factor"] - 1.0) * w_l)
                tan_val = d.get("tan_val")
                if tan_val is None:
                    tan_val = _u(rng, d["tan_lo"], d["tan_hi"])
                seg_tand = tand + (tan_val - tand) * w_l
                if d["debye"] is not None:
                    scale, corner_lo, corner_hi = d["debye"]
                    corner = d.get("debye_corner") or _u(rng, corner_lo, corner_hi)
                    delta = float(np.clip(abs(d["epsr_delta"]) * scale * w_e, 0.0, 0.5))
                    debye = (delta if d["epsr_delta"] > 0 else 0.0, corner, 0.9)
            specs.append(SegmentSpec(profile, float(end - start), float(seg_z0), float(seg_eps),
                                     float(seg_alpha), float(seg_tand),
                                     debye_delta_epsr=debye[0], debye_corner_hz=debye[1],
                                     temp_c=temp_c))
        return ordered, specs

    # infeasible loss split -> redraw the culprit defect's severity combo only
    # (never reseed the whole cable); drop a defect after repeated failures
    ordered = specs = None
    materials = None
    for _ in range(DEFECT_PARAM_REDRAWS):
        ordered, specs = build_specs(active)
        materials = [derive_material(s) for s in specs]
        bad = [i for i, m in enumerate(materials) if m is None]
        if not bad:
            break
        mid = 0.5 * (ordered[bad[0]] + ordered[bad[0] + 1])
        culprit, w_best = None, -1.0
        for d in active:
            w = defect_weight(d, mid)
            if w is not None and w[1] > w_best:
                culprit, w_best = d, w[1]
        if culprit is None:
            return None
        culprit["n_redraw"] = culprit.get("n_redraw", 0) + 1
        if culprit["n_redraw"] > DEFECT_DROPS_AFTER:
            active.remove(culprit)
        else:
            _resample_defect_params(rng, culprit)
            culprit["alpha_factor"] = _alpha_factor(rng, culprit, alpha, tand, epsr)
    if materials is None or any(m is None for m in materials):
        return None
    defects = active

    lumps: list[Lump] = []
    events: list[Event] = []
    for i, (jp, so, sp) in enumerate(joints):
        sev = min(1.0, so / 6.0 + sp / 20.0)
        lumps.append(Lump(jp, "joint", "joint", so, sp, sev))
        events.append(Event(f"joint_{i:02d}", "joint", "joint", jp, jp, jp, sev,
                            {"series_ohm": so, "shunt_pf": sp}))
    di = 0
    for d in defects:
        if d["mechanism"] == "series_resistance":
            so = _ulog(rng, 2.0, 15.0)
            pos = 0.5 * (d["start_m"] + d["end_m"])
            lumps.append(Lump(pos, "defect", "series_resistance", so, 0.0, min(1.0, so / 15.0)))
            events.append(Event(f"defect_{di:02d}", "defect", "series_resistance",
                                pos, pos, pos, min(1.0, so / 15.0), {"series_ohm": so}))
        else:
            params = {k: float(v) for k, v in d.items()
                      if k in ("z0_factor", "epsr_delta", "alpha_factor", "length_m")}
            if d.get("profile_shape") == "gauss":
                params["profile_shape"] = "gauss"
                params["profile_width_m"] = float(d["profile_width_m"])
            events.append(Event(f"defect_{di:02d}", "defect", d["mechanism"],
                                float(d["start_m"]), 0.5 * (d["start_m"] + d["end_m"]),
                                float(d["end_m"]),
                                min(1.0, abs(d["z0_factor"] - 1.0) * 2.0
                                    + abs(d["epsr_delta"]) * 0.8 + (d["alpha_factor"] - 1.0) * 0.05),
                                params))
        di += 1
    events.append(Event("terminal", "terminal", term, length_m, length_m, length_m, 1.0,
                        {"load_ohm": z_load}))
    events.sort(key=lambda e: e.center_m)

    v_axis = 172.0e6 if profile == "field" else 199.3e6
    return CableModel(profile, length_m, z0, epsr, alpha, tand, temp_c, term, z_load,
                      ordered, specs, materials, sorted(lumps, key=lambda l: l.position_m),
                      events, v_axis)
