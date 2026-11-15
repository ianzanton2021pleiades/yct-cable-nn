"""Per-segment distributed RLGC models.

Two families:
- rg58: COMSOL tabulated R'(f)/L'(f) with constant C'/G' (lab-validated golden
  reference, E:\\RG58-FDR\\COMSOL RG58 RLGC仿真结果).  Temperature shifts epsr
  by -0.054 %/degC (measured vp +1.08 % over 40 degC) and scales R slightly.
- field: effective RLGC for MV XLPE cables (phase-to-ground mode).  R(f) =
  R100*sqrt(f/100MHz) with the associated (1+j) skin internal inductance;
  complex permittivity via a Debye relaxation for moisture/aging regions.

derive_material returns None when the requested loss split is infeasible
(dielectric loss alone exceeds the alpha target); callers rejection-sample.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

C0 = 299_792_458.0
EPS0 = 8.854_187_817e-12
MU0 = 4.0e-7 * math.pi
F_REF = 100.0e6

# COMSOL RG58-RLGC.mph table (2 kHz .. 200 MHz), C'=105 pF/m, G'=5.29e-14 S/m.
_RG58_F = np.asarray([2e3, 1e4, 5e4, 1e5, 5e5, 1e6, 2e6, 5e6, 1e7, 2e7,
                      5e7, 1e8, 2e8], dtype=np.float64)
_RG58_R = np.asarray([0.02631, 0.02677, 0.03425, 0.04432, 0.09028, 0.12493,
                      0.17398, 0.27063, 0.38124, 0.53707, 0.84460, 1.19249,
                      1.68683], dtype=np.float64)
_RG58_L = np.asarray([326.0e-9, 326.0e-9, 310.0e-9, 294.0e-9, 264.0e-9, 256.0e-9,
                      251.0e-9, 246.0e-9, 243.0e-9, 242.0e-9, 240.0e-9, 239.0e-9,
                      239.0e-9], dtype=np.float64)
RG58_C_PER_M = 105.0e-12
RG58_G_PER_M = 5.29e-14


def rg58_r_of_f(f: np.ndarray, temp_c: float = 20.0) -> np.ndarray:
    r = np.exp(np.interp(np.log(np.maximum(f, 2.0e3)), np.log(_RG58_F), np.log(_RG58_R)))
    return r * (1.0 + 0.003 * (temp_c - 20.0))


def rg58_l_of_f(f: np.ndarray) -> np.ndarray:
    return np.exp(np.interp(np.log(np.maximum(f, 2.0e3)), np.log(_RG58_F), np.log(_RG58_L)))


@dataclass(frozen=True)
class SegmentSpec:
    """Physical description of one homogeneous segment."""
    profile: str
    length_m: float
    z0_ohm: float
    epsr: float
    alpha_db_per_m_at_100mhz: float
    tan_delta_at_100mhz: float
    debye_delta_epsr: float = 0.0
    debye_corner_hz: float = 80.0e6
    debye_exponent: float = 0.9
    temp_c: float = 20.0


@dataclass(frozen=True)
class SegmentMaterial:
    model: str            # "rg58_comsol" | "field_effective"
    r100_ohm_per_m: float  # field_effective only

    def as_dict(self) -> dict:
        return {"model": self.model, "r100_ohm_per_m": self.r100_ohm_per_m}


def _eps_complex(f: np.ndarray, spec: SegmentSpec) -> np.ndarray:
    if spec.debye_delta_epsr == 0.0:
        return np.full_like(f, spec.epsr)
    ratio = np.power(np.maximum(f, 1.0) / spec.debye_corner_hz, spec.debye_exponent)
    return spec.epsr + spec.debye_delta_epsr / (1.0 + 1j * ratio)


def segment_rlgc(f: np.ndarray, spec: SegmentSpec, material: SegmentMaterial):
    """Return R, L, G, C arrays (SI units per metre) for one segment."""
    omega = 2.0 * math.pi * f
    eps_c = _eps_complex(f, spec)
    eps_re = np.maximum(eps_c.real, 1.01)
    if material.model == "rg58_comsol":
        temp_scale = 1.0 - 0.00054 * (spec.temp_c - 20.0)  # epsr shift with temperature
        c = RG58_C_PER_M * temp_scale
        resistance = rg58_r_of_f(f, spec.temp_c)
        inductance = rg58_l_of_f(f)
        conductance = np.full_like(f, RG58_G_PER_M) + omega * c * spec.tan_delta_at_100mhz
        capacitance = np.full_like(f, c)
        return resistance, inductance, conductance, capacitance
    if material.model != "field_effective":
        raise ValueError(f"unknown material model {material.model!r}")
    v = C0 / math.sqrt(spec.epsr)
    l_ext = spec.z0_ohm / v
    c_inf = 1.0 / (spec.z0_ohm * v)
    r = material.r100_ohm_per_m * np.sqrt(np.maximum(f, 1.0) / F_REF)
    inductance = l_ext + r / np.maximum(omega, 1e-30)
    capacitance = c_inf * eps_c / spec.epsr
    conductance = omega * c_inf * eps_re / spec.epsr * spec.tan_delta_at_100mhz
    return r, inductance, conductance, capacitance


def segment_propagation(f: np.ndarray, spec: SegmentSpec, material: SegmentMaterial):
    r, l, g, c = segment_rlgc(f, spec, material)
    series = r + 1j * 2.0 * math.pi * f * l
    shunt = g + 1j * 2.0 * math.pi * f * c
    return np.sqrt(series * shunt), np.sqrt(series / shunt)


def group_delay_s_per_m(spec: SegmentSpec, material: SegmentMaterial,
                        reference_hz: float = F_REF) -> float:
    h = reference_hz * 1.0e-3
    gamma = segment_propagation(np.asarray([reference_hz - h, reference_hz + h]),
                                spec, material)[0]
    beta = np.unwrap(gamma.imag)
    return float((beta[1] - beta[0]) / (2.0 * math.pi * 2.0 * h))


def dielectric_alpha_db_per_m(spec: SegmentSpec) -> float:
    """alpha contribution of tan-delta (+ Debye loss) at 100 MHz, analytic."""
    eps_c = _eps_complex(np.asarray([F_REF]), spec)[0]
    eps_re = max(eps_c.real, 1.01)
    # Debye imaginary part adds an effective tan-delta term.
    tan_eff = spec.tan_delta_at_100mhz + max(-eps_c.imag, 0.0) / eps_re
    beta = 2.0 * math.pi * F_REF * math.sqrt(eps_re) / C0
    return 8.686 * 0.5 * beta * tan_eff


def derive_field_material(spec: SegmentSpec) -> SegmentMaterial | None:
    """Invert alpha = alpha_dielectric + alpha_conductor(R100) for R100.

    Returns None when dielectric loss alone already exceeds the target.
    """
    alpha_diel = dielectric_alpha_db_per_m(spec)
    alpha_target = spec.alpha_db_per_m_at_100mhz
    if alpha_diel >= alpha_target:
        return None
    alpha_cond_np = (alpha_target - alpha_diel) / 8.686
    omega = 2.0 * math.pi * F_REF
    v = C0 / math.sqrt(spec.epsr)
    l_ext = spec.z0_ohm / v
    c_inf = 1.0 / (spec.z0_ohm * v)

    def attenuation(r100: float) -> float:
        # skin impedance R*(1+j) includes internal inductance r/(omega)
        series = r100 + 1j * (omega * l_ext + r100)
        shunt = 1j * omega * c_inf
        return float(np.sqrt(series * shunt).real)

    lo, hi = 0.0, 1.0
    while attenuation(hi) < alpha_cond_np:
        hi *= 2.0
        if hi > 1e4:
            return None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if attenuation(mid) < alpha_cond_np:
            lo = mid
        else:
            hi = mid
    return SegmentMaterial("field_effective", r100_ohm_per_m=0.5 * (lo + hi))


def derive_rg58_material(spec: SegmentSpec) -> SegmentMaterial:
    return SegmentMaterial("rg58_comsol", r100_ohm_per_m=float(rg58_r_of_f(np.asarray([F_REF]))[0]))


def derive_material(spec: SegmentSpec) -> SegmentMaterial | None:
    return derive_rg58_material(spec) if spec.profile == "rg58" else derive_field_material(spec)
