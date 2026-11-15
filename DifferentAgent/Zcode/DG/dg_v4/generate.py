"""Public generation API: one deterministic sample per seed.

seed layout: template_id = seed // REALIZATIONS, realization = seed % REALIZATIONS.
Topology seed depends on BOTH (first-round failure: topology depended only on
template_id, so 64 realizations shared one cable and the model memorized it):
  realization < 4  -> shared template topology (same cable re-tested, independent
                      measurement chain / noise / band)
  realization >= 4 -> independent topology per realization
so one template group = 1 cable measured 4 times + 46 independent cables.
"""
from __future__ import annotations

import numpy as np

from .grid import get_band_axis, sample_band_variant
from .labeling import build_label_grids, label_events
from .materials import SegmentSpec, derive_material, segment_propagation
from .measurement import evaluate as chain_evaluate, roughness_gamma, sample_chain
from .physics import cascade, lump_abcd, segment_abcd
from .topology import build_topology
from .transform import s11_to_responses

REALIZATIONS = 50
GRID_BINS = 11008            # 0.25 m x 11008 = 2752 m panel grid
DIST_STEP = 0.25
Z_REF = 50.0
F_Z0_REF = 100.0e6           # reference frequency for the z0_dev profile


def template_seed(template_id: int) -> int:
    return 7919 * int(template_id) + 13


def topology_seed(template_id: int, realization: int) -> int:
    t, r = int(template_id), int(realization)
    if r < 4:
        return template_seed(t)
    return template_seed(t) + 104729 * (r - 3)


def template_profile(template_id: int, ratio_rg58: float = 0.25) -> str:
    return "rg58" if (np.random.default_rng(template_seed(template_id) + 991).random() < ratio_rg58) else "field"


def cable_abcd(f: np.ndarray, model) -> tuple:
    net = None
    lump_at = {round(l.position_m, 6): l for l in model.lumps}
    for i, (spec, material) in enumerate(zip(model.seg_specs, model.materials)):
        seg = segment_abcd(f, spec, material)
        net = seg if net is None else cascade(net, seg)
        end_m = float(model.seg_bounds[i + 1])
        if end_m < model.length_m - 1e-9:
            lump = lump_at.get(round(end_m, 6))
            if lump is not None:
                net = cascade(net, lump_abcd(f, lump.series_ohm, lump.shunt_pf))
    return net


def z0_dev_profile(model, n_points: int) -> np.ndarray:
    """Per-grid-point normalized impedance deviation |Z0_seg|-|Z0_base| / |Z0_base|.

    Zc of every lattice segment (and of a defect-free base segment with the same
    base parameters) evaluated at 100 MHz from the complex propagation constant;
    expanded onto the 0.25 m grid via seg_bounds.  Point lumps (joints,
    series_resistance) are not distributed Z0 changes and stay at 0.
    """
    f_ref = np.asarray([F_Z0_REF])
    base_spec = SegmentSpec(model.profile, 1.0, model.base_z0_ohm, model.base_epsr,
                            model.base_alpha_db_per_m_100mhz, model.base_tan_delta_100mhz,
                            temp_c=model.temp_c)
    zc_base = abs(segment_propagation(f_ref, base_spec, derive_material(base_spec))[1][0])
    zc = np.asarray([abs(segment_propagation(f_ref, s, m)[1][0])
                     for s, m in zip(model.seg_specs, model.materials)])
    dev = (zc - zc_base) / zc_base
    idx = np.clip(np.searchsorted(model.seg_bounds, np.arange(n_points) * DIST_STEP,
                                  side="right") - 1, 0, dev.size - 1)
    return dev[idx].astype(np.float32)


def generate_sample(seed: int, profile: str = "mixed", band: str | None = None,
                    overrides: dict | None = None) -> dict | None:
    """Generate one sample; returns None if the template draw is infeasible."""
    seed = int(seed)
    template_id, realization = divmod(seed, REALIZATIONS)
    prof = template_profile(template_id) if profile == "mixed" else profile
    if overrides and overrides.get("profile"):
        prof = overrides["profile"]

    model = None
    for attempt in range(12):
        model = build_topology(prof, topology_seed(template_id, realization) + attempt,
                               overrides=overrides)
        if model is not None:
            break
    if model is None:
        return None

    rng = np.random.default_rng(seed * 31 + 7)
    band = band or sample_band_variant(rng)
    axis = get_band_axis(band)
    chain = sample_chain(prof, seed * 131 + 5)

    f = axis.freq_hz
    net = cable_abcd(f, model)
    g_rough = roughness_gamma(model, f, np.random.default_rng(seed * 977 + 3))
    s11 = chain_evaluate(chain, prof, f, net, complex(model.z_load_ohm), Z_REF, seed,
                         gamma_rough=g_rough, normalize_short=False)

    # instrument-convention response: offset centers the clamp junction at 0 m
    v_axis = model.v_axis_m_per_s
    tau_fix = chain.line_delay_ns * 1.0e-9
    offset_m = v_axis * tau_fix
    length_setting = max(1.03 * model.length_m, model.length_m + 5.0)
    valid_len = int(min(GRID_BINS, (1.3 * model.length_m + 10.0) / DIST_STEP + 1.0))
    distance, impulse, step, _ = s11_to_responses(
        f, s11, v_axis, length_setting, offset_m, dist_max_m=(valid_len - 1) * DIST_STEP)
    events = label_events(model, distance, impulse)
    for ev in events:
        ev["weight"] = 1.0
    role, mech = build_label_grids(events, distance)

    return {
        "seed": seed, "template_id": template_id, "realization": realization,
        "profile": prof, "band": band,
        "valid_len": int(valid_len),
        "s11": s11.astype(np.complex64),
        "distance_max_m": float(distance[-1]),
        "impulse": impulse.astype(np.float32),
        "step": step.astype(np.float32),
        "role": role, "mechanism": mech,
        "z0_dev": z0_dev_profile(model, valid_len),
        "events": events,
        "chain": chain.as_dict(),
        "length_m": float(model.length_m),
        "length_setting_m": float(length_setting),
        "offset_m": float(offset_m),
        "v_axis_m_per_s": float(v_axis),
    }


def iter_samples(count: int, start_seed: int = 0, profile: str = "mixed",
                 workers: int = 1):
    from concurrent.futures import ThreadPoolExecutor
    seeds = list(range(start_seed, start_seed + count))
    if workers <= 1:
        for s in seeds:
            yield generate_sample(s, profile)
        return
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for out in ex.map(lambda s: generate_sample(s, profile), seeds):
            yield out
