"""Measurement chain: clamp fixture + wideband analyzer front end.

Field reality being matched (2026-10 survey):
- per-test embedded short calibration -> residual directivity/tracking remain
- near-end pulse peaks can exceed 1 (fixture ringing survives de-embedding)
- complex residual rms vs fitted near-end model ~0.089 on real files
- repeatability drift is smooth and systematic, not white

Model: gamma = D(f) + T(f) * gamma_lin / (1 - S(f) * gamma_lin)
with D/S smooth random complex functions on the log-frequency axis (random-walk
knots interpolated on log(f), with enhanced low-frequency drift), plus
correlated additive noise.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

# distributed-reflection (clutter) sigma for the field profile, normalized to a
# per-cell std of FIELD_ROUGH_SIGMA * Z0; T3-calibrated against the field corpus
# clutter RMS (matched-SNR populations, run4)
FIELD_ROUGH_SIGMA = 0.00223


def _smooth_complex(f: np.ndarray, rng: np.random.Generator, scale: float,
                    low_boost: float, n_knots: int = 24) -> np.ndarray:
    knots = np.linspace(0.0, 1.0, n_knots)
    steps = (rng.standard_normal(n_knots) + 1j * rng.standard_normal(n_knots)) * scale / math.sqrt(n_knots)
    series = np.cumsum(steps)
    series -= series.mean()
    peak = float(np.max(np.abs(series))) + 1e-12
    series = series * (scale / peak)
    u = np.log(f / f[0]) / max(math.log(f[-1] / f[0]), 1e-12)
    base = np.interp(u, knots, series.real) + 1j * np.interp(u, knots, series.imag)
    return base * (1.0 + low_boost / (1.0 + f / 5.0e6))


def _ar1_noise(size: int, sigma_f: np.ndarray, rho: float,
               rng: np.random.Generator) -> np.ndarray:
    white = (rng.standard_normal(size) + 1j * rng.standard_normal(size)) / math.sqrt(2.0)
    innovation = math.sqrt(max(1.0 - rho * rho, 1e-6))
    out = np.empty(size, dtype=np.complex128)
    out[0] = white[0]
    for i in range(1, size):
        out[i] = rho * out[i - 1] + innovation * white[i]
    return out * sigma_f


@dataclass(frozen=True)
class ChainParams:
    kind: str                       # "D" clamp+leads | "E" clamp direct
    line_z_ohm: float
    line_delay_ns: float
    loss_flat_db: float             # frequency-flat one-way loss (contact/transition)
    loss_sqrt_db_100mhz: float      # sqrt(f) one-way loss at 100 MHz
    tip_series_ohm: float
    tip_shunt_pf: float             # extra capacitance across the tips
    tip_ground_ohm: float           # clamp-point RF bond to earth (armour/cabinet strap)
    tip_ground_nf: float            # bond series capacitance (RF-couples the shunt)
    directivity: float
    source_match: float
    tracking_ripple: float
    noise_sigma: float
    noise_correlation: float

    def as_dict(self) -> dict:
        return dict(self.__dict__)


# Measured fixture S-parameters (AgentsStorage/ZCode_fixture_sparam, 2026-10-29)
# anchor the D type (Z0~66 ohm, tau 23.2 ns, sqrt(f) loss 1.2-3.3 dB @100 MHz).
# The observed per-test spread of the mid-band plateau (0.16 威海 .. 0.82 三院)
# is clip/lead state, so D-type ranges are widened to span junction |Gamma|
# ~ 0.2-0.85 across tests.
_RANGES = {
    "field": dict(kind_w={"D": 0.8, "E": 0.2},
                  D=dict(line_z=(48.0, 80.0), delay=(25.0, 45.0),
                         loss_flat=(0.25, 0.7), loss_sqrt=(1.0, 4.0),
                         series=(0.3, 2.0), shunt=(10.0, 420.0)),
                  E=dict(line_z=(180.0, 225.0), delay=(3.2, 4.5),
                         loss_flat=(0.05, 0.3), loss_sqrt=(0.02, 0.15),
                         series=(0.1, 0.3), shunt=(0.0, 10.0)),
                  direct=(0.01, 0.06), smatch=(0.01, 0.06), ripple=(0.005, 0.03),
                  sigma=(0.0015, 0.008), corr=(0.95, 0.998)),
    "rg58": dict(kind_w={"D": 1.0},
                 D=dict(line_z=(46.0, 56.0), delay=(1.0, 8.0),
                        loss_flat=(0.005, 0.05), loss_sqrt=(0.02, 0.2),
                        series=(0.05, 0.8), shunt=(0.1, 2.0)),
                 E=dict(line_z=(46.0, 56.0), delay=(1.0, 8.0),
                        loss_flat=(0.005, 0.05), loss_sqrt=(0.02, 0.2),
                        series=(0.05, 0.8), shunt=(0.1, 2.0)),
                 direct=(0.004, 0.02), smatch=(0.008, 0.04), ripple=(0.003, 0.015),
                 sigma=(0.0004, 0.0025), corr=(0.85, 0.99)),
}


def sample_chain(profile: str, seed: int) -> ChainParams:
    rng = np.random.default_rng(seed)
    r = _RANGES[profile]
    kinds = list(r["kind_w"])
    kind = str(rng.choice(kinds, p=[r["kind_w"][k] for k in kinds])) if len(kinds) > 1 else kinds[0]
    spec = r[kind]
    if profile == "field":
        # 60 % of field clamps sit near an earthed strap/armour -> low-ohmic
        # shunt that creates the smooth 0.7-0.85 mid-band plateau; the rest
        # are isolated (weak plateau, like 威海 hi/lo=0.16).
        grounded = rng.random() < 0.92
        rg = float(rng.uniform(2.0, 6.0)) if grounded else float(rng.uniform(25.0, 400.0))
        cg = float(rng.uniform(0.5, 2.0))
    else:
        rg = float(rng.uniform(200.0, 2000.0))
        cg = float(rng.uniform(5.0, 40.0))
    return ChainParams(
        kind, float(rng.uniform(*spec["line_z"])), float(rng.uniform(*spec["delay"])),
        float(rng.uniform(*spec["loss_flat"])), float(rng.uniform(*spec["loss_sqrt"])),
        float(rng.uniform(*spec["series"])),
        float(rng.uniform(*spec["shunt"])), rg, cg, float(rng.uniform(*r["direct"])),
        float(rng.uniform(*r["smatch"])), float(rng.uniform(*r["ripple"])),
        float(rng.uniform(*r["sigma"])), float(rng.uniform(*r["corr"])),
    )


def fixture_abcd(f: np.ndarray, chain: ChainParams) -> tuple:
    """Clamp/lead structure: lossy line of impedance Z_fix (sqrt-f loss)
    followed by tip series/shunt parasitics and the clamp-point shunt to
    earth (R_g in parallel with the tip capacitance)."""
    theta = 2.0 * np.pi * f * chain.line_delay_ns * 1.0e-9
    # measured D-type through loss fits a flat + linear-in-f law
    # (one-way ~1.0 dB at 30-100 MHz, ~5.4 dB at 1 GHz)
    attenuation = (np.log(10.0) / 20.0) * (
        chain.loss_flat_db + chain.loss_sqrt_db_100mhz * f / 1.0e8)
    gamma_l = attenuation + 1j * theta
    ch = np.cosh(gamma_l)
    sh = np.sinh(gamma_l)
    zf = chain.line_z_ohm
    line = (ch, zf * sh, sh / zf, ch)
    series = chain.tip_series_ohm * np.ones_like(f, dtype=np.complex128)
    omega = 2.0 * np.pi * f
    z_bond = chain.tip_ground_ohm + 1.0 / (1j * omega * chain.tip_ground_nf * 1.0e-9)
    y_shunt = 1.0 / z_bond + 1j * omega * chain.tip_shunt_pf * 1.0e-12
    # tip = series R then shunt (R_g || C)
    tip = (1.0 + series * y_shunt, series, y_shunt, np.ones_like(series))
    a, b, c, d = line
    e, g, h, i2 = tip
    return a * e + b * h, a * g + b * i2, c * e + d * h, c * g + d * i2


def evaluate(chain: ChainParams, profile: str, f: np.ndarray,
             cable_abcd, z_load: complex, z_ref: float, noise_seed: int,
             gamma_rough: np.ndarray | None = None,
             normalize_short: bool = True, phase_ref_rad: float = 0.0) -> np.ndarray:
    """cable ABCD (segments + lumps, terminated by z_load) -> measured S11.

    The field instrument performs a response (short) normalization at the
    clamp tip: Gamma_meas = Gamma_raw / Gamma_short, which de-embeds the
    common fixture response but leaves residual error terms; D/S model those.
    """
    rng = np.random.default_rng(noise_seed)
    fa, fb, fc, fd = fixture_abcd(f, chain)
    a = fa * cable_abcd[0] + fb * cable_abcd[2]
    b = fa * cable_abcd[1] + fb * cable_abcd[3]
    c2 = fc * cable_abcd[0] + fd * cable_abcd[2]
    d2 = fc * cable_abcd[1] + fd * cable_abcd[3]
    zin = (a * z_load + b) / (c2 * z_load + d2)
    gamma_lin = (zin - z_ref) / (zin + z_ref)
    if normalize_short:
        zin_short = fb / fd                   # short at the clamp tip (fixture only)
        gamma_short = (zin_short - z_ref) / (zin_short + z_ref)
        gamma_lin = gamma_lin / gamma_short
    if gamma_rough is not None:
        gamma_lin = gamma_lin + gamma_rough
    gamma_lin = gamma_lin * np.exp(1j * phase_ref_rad)
    D = _smooth_complex(f, rng, chain.directivity * 0.7, 0.6)
    S = _smooth_complex(f, rng, chain.source_match * 0.7, 0.3)
    T = 1.0 + _smooth_complex(f, rng, chain.tracking_ripple * 0.5, 0.0)
    gamma = D + T * gamma_lin / (1.0 - S * gamma_lin)
    gamma = np.clip(gamma, -1.05 - 1.05j, 1.05 + 1.05j)  # keep near-passive
    low_boost = 1.5 if profile == "field" else 0.6
    sigma_f = chain.noise_sigma * (1.0 + low_boost / (1.0 + f / 1.0e6))
    return gamma + _ar1_noise(f.size, sigma_f, chain.noise_correlation, rng)


def roughness_gamma(model, f: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Distributed weak reflections from cable micro-structure (background
    clutter).  Real cables show a clutter continuum in the impulse response;
    without it synthetic curves are unnaturally clean and detectors trained on
    them false-alarm on field clutter (the documented first-round failure)."""
    length = model.length_m
    n_cells = max(2, int(np.ceil(length / 10.0)))
    centers = (np.arange(n_cells) + 0.5) * (length / n_cells)
    sigma = 0.001 if model.profile == "rg58" else FIELD_ROUGH_SIGMA
    white = rng.standard_normal(n_cells)
    profile_z = np.empty(n_cells)
    acc = 0.0
    for i in range(n_cells):          # AR(1) spatial correlation ~0.5
        acc = 0.5 * acc + np.sqrt(1.0 - 0.25) * white[i]
        profile_z[i] = acc
    profile_z *= sigma * model.base_z0_ohm / max(float(np.std(profile_z)), 1e-9)
    gamma_i = (profile_z / (2.0 * model.base_z0_ohm)).astype(np.float64)
    alpha_np100 = model.base_alpha_db_per_m_100mhz / 8.686
    att = np.exp(-2.0 * alpha_np100 * np.sqrt(f / 1.0e8)[None, :] * centers[:, None])
    phase = np.exp((-2j * np.pi) * f[None, :] * (2.0 * centers[:, None] / model.v_axis_m_per_s))
    return (gamma_i[:, None] * att * phase).sum(axis=0)
