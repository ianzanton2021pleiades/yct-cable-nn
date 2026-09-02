"""Random RG58/Field physical topologies for DG V3."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any

import numpy as np

from .config import GeneratorConfig

C0 = 299_792_458.0
EPS0 = 8.854_187_817e-12
MU0 = 4.0e-7 * math.pi
FREQUENCY_REFERENCE_HZ = 100.0e6


@dataclass(frozen=True)
class RLGCMaterial:
    """Derived immutable material parameters for one RLGC segment."""

    model: str
    conductor_conductivity_s_per_m: float
    dielectric_conductivity_s_per_m: float = 0.0
    shunt_conductance_s_per_m: float = 0.0
    effective_inner_radius_m: float | None = None
    effective_shield_radius_m: float | None = None
    series_resistance_100mhz_ohm_per_m: float | None = None

    def as_dict(self) -> dict[str, float | str | None]:
        return {
            "model": self.model,
            "conductor_conductivity_s_per_m": self.conductor_conductivity_s_per_m,
            "dielectric_conductivity_s_per_m": self.dielectric_conductivity_s_per_m,
            "shunt_conductance_s_per_m": self.shunt_conductance_s_per_m,
            "effective_inner_radius_m": self.effective_inner_radius_m,
            "effective_shield_radius_m": self.effective_shield_radius_m,
            "series_resistance_100mhz_ohm_per_m": self.series_resistance_100mhz_ohm_per_m,
        }


@dataclass(frozen=True)
class CableSegment:
    start_m: float
    end_m: float
    z0_ohm: float
    epsr: float
    alpha_db_per_m_at_100mhz: float
    tan_delta_at_100mhz: float
    region: str = "healthy"
    defect_id: int | None = None
    debye_delta_epsr: float = 0.0
    debye_corner_hz: float = 80.0e6
    debye_exponent: float = 1.0
    material: RLGCMaterial | None = None

    @property
    def length_m(self) -> float:
        return self.end_m - self.start_m


def _complex_permittivity(frequencies_hz: np.ndarray, segment: CableSegment) -> np.ndarray:
    f = np.asarray(frequencies_hz, dtype=np.float64)
    ratio = np.power(np.maximum(f, 0.0) / segment.debye_corner_hz,
                     segment.debye_exponent)
    return segment.epsr + segment.debye_delta_epsr / (1.0 + 1j * ratio)


def segment_rlgc(frequencies_hz: np.ndarray, segment: CableSegment) -> tuple[np.ndarray, ...]:
    """Return the segment's distributed R, L, G and complex C arrays."""
    if segment.material is None:
        raise ValueError("CableSegment is missing its derived RLGC material")
    material = segment.material
    f = np.asarray(frequencies_hz, dtype=np.float64)
    omega = 2.0 * math.pi * f
    eps_complex = _complex_permittivity(f, segment)

    if material.model == "coax_rlgc":
        rc = float(material.effective_inner_radius_m)
        rs = float(material.effective_shield_radius_m)
        sigma = material.conductor_conductivity_s_per_m
        log_term = math.log(rs / rc)
        skin_depth = np.sqrt(2.0 / np.maximum(omega * MU0 * sigma, 1e-30))
        resistance = (1.0 / (2.0 * math.pi * rc * skin_depth * sigma)
                      + 1.0 / (2.0 * math.pi * rs * skin_depth * sigma))
        inductance = (MU0 * log_term / (2.0 * math.pi)
                      + MU0 * (1.0 / rc + 1.0 / rs) * skin_depth / (4.0 * math.pi))
        capacitance = 2.0 * math.pi * EPS0 * eps_complex / log_term
        capacitance_real = 2.0 * math.pi * EPS0 * np.maximum(eps_complex.real, 1.01) / log_term
        conductance = (2.0 * math.pi * material.dielectric_conductivity_s_per_m / log_term
                       + omega * capacitance_real * segment.tan_delta_at_100mhz)
        return resistance, inductance, conductance, capacitance

    if material.model != "effective_rlgc":
        raise ValueError(f"unsupported RLGC material model: {material.model!r}")
    velocity = C0 / math.sqrt(segment.epsr)
    inductance_external = segment.z0_ohm / velocity
    capacitance_infinite = 1.0 / (segment.z0_ohm * velocity)
    resistance_ref = float(material.series_resistance_100mhz_ohm_per_m)
    resistance = resistance_ref * np.sqrt(np.maximum(f, 1.0) / FREQUENCY_REFERENCE_HZ)
    # The equal real/imaginary skin impedance includes the associated internal
    # inductance without inventing an effective coaxial radius for Field cable.
    inductance = inductance_external + resistance / np.maximum(omega, 1e-30)
    capacitance = capacitance_infinite * eps_complex / segment.epsr
    capacitance_real = capacitance_infinite * np.maximum(eps_complex.real, 1.01) / segment.epsr
    conductance = (material.shunt_conductance_s_per_m
                   + omega * capacitance_real * segment.tan_delta_at_100mhz)
    return resistance, inductance, conductance, capacitance


def segment_propagation(frequencies_hz: np.ndarray,
                        segment: CableSegment) -> tuple[np.ndarray, np.ndarray]:
    """Return propagation constant and characteristic impedance from RLGC."""
    f = np.asarray(frequencies_hz, dtype=np.float64)
    omega = 2.0 * math.pi * f
    resistance, inductance, conductance, capacitance = segment_rlgc(f, segment)
    series = resistance + 1j * omega * inductance
    shunt = conductance + 1j * omega * capacitance
    return np.sqrt(series * shunt), np.sqrt(series / shunt)


def segment_group_delay_s_per_m(segment: CableSegment,
                                reference_hz: float = FREQUENCY_REFERENCE_HZ) -> float:
    """Numerical derivative d(beta)/d(omega) at the truth reference frequency."""
    half_span_hz = reference_hz * 1.0e-3
    frequencies = np.asarray([reference_hz - half_span_hz,
                              reference_hz + half_span_hz], dtype=np.float64)
    gamma, _ = segment_propagation(frequencies, segment)
    beta = np.unwrap(gamma.imag)
    return float((beta[1] - beta[0]) / (2.0 * math.pi * (2.0 * half_span_hz)))


@dataclass(frozen=True)
class Joint:
    position_m: float
    variant: str
    series_ohm: float
    shunt_pf: float


@dataclass(frozen=True)
class TruthRecord:
    event_id: str
    role: str
    geometry: str
    mechanism: str
    physical_start_m: float
    physical_center_m: float
    physical_end_m: float
    delay_start_s: float
    delay_center_s: float
    delay_end_s: float
    severity: float
    electrical_change: dict[str, float | str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "role": self.role,
            "geometry": self.geometry,
            "mechanism": self.mechanism,
            "physical_start_m": self.physical_start_m,
            "physical_center_m": self.physical_center_m,
            "physical_end_m": self.physical_end_m,
            "delay_start_s": self.delay_start_s,
            "delay_center_s": self.delay_center_s,
            "delay_end_s": self.delay_end_s,
            "severity": self.severity,
            "electrical_change": self.electrical_change,
        }


@dataclass(frozen=True)
class DefectRegion:
    index: int
    type: str
    start_m: float
    end_m: float
    z0_factor: float
    epsr_delta: float
    alpha_factor: float
    severity: float


@dataclass
class CableTopology:
    profile: str
    length_m: float
    z_ref_ohm: float
    base_z0_ohm: float
    base_epsr: float
    base_alpha_db_per_m_at_100mhz: float
    base_tan_delta_at_100mhz: float
    segments: list[CableSegment]
    joints: list[Joint]
    termination: str
    z_load_ohm: float
    defect_regions: list[DefectRegion] = field(default_factory=list)
    delay_reference_hz: float = FREQUENCY_REFERENCE_HZ

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "physical_length_m": self.length_m,
            "reference_impedance_ohm": self.z_ref_ohm,
            "base_z0_ohm": self.base_z0_ohm,
            "base_epsr": self.base_epsr,
            "base_alpha_db_per_m_at_100mhz": self.base_alpha_db_per_m_at_100mhz,
            "base_tan_delta_at_100mhz": self.base_tan_delta_at_100mhz,
            "truth_delay_reference_hz": self.delay_reference_hz,
            "termination": self.termination,
            "load_ohm": self.z_load_ohm,
            "segments": [
                {
                    "physical_start_m": segment.start_m,
                    "physical_end_m": segment.end_m,
                    "z0_ohm": segment.z0_ohm,
                    "epsr": segment.epsr,
                    "alpha_db_per_m_at_100mhz": segment.alpha_db_per_m_at_100mhz,
                    "tan_delta_at_100mhz": segment.tan_delta_at_100mhz,
                    "debye_delta_epsr": segment.debye_delta_epsr,
                    "debye_corner_hz": segment.debye_corner_hz,
                    "debye_exponent": segment.debye_exponent,
                    "material": None if segment.material is None else segment.material.as_dict(),
                    "region": segment.region,
                }
                for segment in self.segments
            ],
        }

    def travel_time_s_at(self, position_m: float) -> float:
        """One-way group delay to a position at ``delay_reference_hz``."""
        remaining = float(position_m)
        time_s = 0.0
        for segment in self.segments:
            if remaining <= 0:
                break
            length = min(remaining, segment.length_m)
            time_s += length * segment_group_delay_s_per_m(segment, self.delay_reference_hz)
            remaining -= length
        return time_s

    def truth_records(self) -> list[TruthRecord]:
        records: list[TruthRecord] = []
        for defect in self.defect_regions:
            center = (defect.start_m + defect.end_m) / 2.0
            geometry = "point" if defect.type == "short" else "interval"
            records.append(TruthRecord(
                f"defect_{defect.index}", "defect", geometry, defect.type,
                defect.start_m, center, defect.end_m,
                2.0 * self.travel_time_s_at(defect.start_m),
                2.0 * self.travel_time_s_at(center),
                2.0 * self.travel_time_s_at(defect.end_m),
                defect.severity,
                {
                    "z0_before_ohm": self.base_z0_ohm,
                    "z0_inside_ohm": self.base_z0_ohm * defect.z0_factor,
                    "z0_factor": defect.z0_factor,
                    "epsr_before": self.base_epsr,
                    "epsr_inside": self.base_epsr + defect.epsr_delta,
                    "epsr_delta": defect.epsr_delta,
                    "loss_factor": defect.alpha_factor,
                    "impedance_direction": "up" if defect.z0_factor > 1.0 else "down",
                },
            ))
        for index, joint in enumerate(self.joints):
            delay = 2.0 * self.travel_time_s_at(joint.position_m)
            severity = min(1.0, 0.7 * joint.series_ohm / 3.5 + 0.3 * joint.shunt_pf / 2.5)
            records.append(TruthRecord(
                f"joint_{index}", "joint", "point", "joint",
                joint.position_m, joint.position_m, joint.position_m,
                delay, delay, delay, severity,
                {"series_ohm": joint.series_ohm, "shunt_pf": joint.shunt_pf},
            ))
        terminal_delay = 2.0 * self.travel_time_s_at(self.length_m)
        records.append(TruthRecord(
            "terminal_0", "terminal", "point", self.termination,
            self.length_m, self.length_m, self.length_m,
            terminal_delay, terminal_delay, terminal_delay, 1.0,
            {"load_ohm": self.z_load_ohm},
        ))
        return sorted(records, key=lambda item: item.physical_center_m)


def _uniform(rng: np.random.Generator, bounds: dict[str, float]) -> float:
    return float(rng.uniform(float(bounds["min"]), float(bounds["max"])))


def _sample_variant(rng: np.random.Generator, variants: dict[str, dict[str, Any]]) -> str:
    names = list(variants)
    probabilities = np.asarray([float(variants[name]["weight"]) for name in names], dtype=float)
    probabilities /= probabilities.sum()
    return str(rng.choice(names, p=probabilities))


def _defect_count(profile: str, length_m: float, params: dict[str, Any], rng: np.random.Generator) -> int:
    if profile == "rg58":
        probabilities = [0.60, 0.32, 0.08]
    elif length_m < 250.0:
        probabilities = [0.45, 0.35, 0.17, 0.03, 0.0, 0.0]
    elif length_m < 500.0:
        probabilities = [0.28, 0.34, 0.24, 0.10, 0.03, 0.01]
    elif length_m < 800.0:
        probabilities = [0.22, 0.31, 0.27, 0.13, 0.05, 0.02]
    elif length_m < 1500.0:
        probabilities = [0.16, 0.28, 0.29, 0.17, 0.07, 0.03]
    else:
        probabilities = [0.12, 0.24, 0.29, 0.21, 0.10, 0.04]
    values = np.arange(len(probabilities))
    return int(rng.choice(values, p=np.asarray(probabilities, dtype=float)))


def _sample_defect(profile: str, length_m: float, rng: np.random.Generator,
                   params: dict[str, Any], forced_type: str | None = None) -> tuple[str, float, float, float, float, float]:
    if forced_type is not None:
        if forced_type not in params["defects"]:
            raise ValueError(f"unsupported defect type: {forced_type}")
        if profile == "rg58" and forced_type != "short":
            raise ValueError("RG58 only supports short defects")
        defect_type = forced_type
    elif profile == "rg58":
        defect_type = "short"
    else:
        if length_m < 250.0:
            long_probability = 0.0
        elif length_m < 500.0:
            long_probability = 0.03
        elif length_m < 800.0:
            long_probability = 0.06
        elif length_m < 1500.0:
            long_probability = 0.10
        else:
            long_probability = 0.16
        defect_type = (
            str(rng.choice(["aging", "moisture_local", "moisture_distributed"], p=[0.50, 0.25, 0.25]))
            if rng.random() < long_probability
            else "short"
        )
    spec = params["defects"][defect_type]
    if "length_fraction" in spec:
        defect_length = float(np.clip(length_m * _uniform(rng, spec["length_fraction"]),
                                      float(spec["length_m"]["min"]), float(spec["length_m"]["max"])))
    else:
        defect_length = min(_uniform(rng, spec["length_m"]), length_m * 0.35)
    z0_factor = _uniform(rng, spec["z0_factor"])
    epsr_delta = _uniform(rng, spec["epsr_delta"])
    alpha_factor = _uniform(rng, spec["alpha_factor"])
    defect_length = min(defect_length, 0.65 * length_m)
    severity = min(1.0, abs(z0_factor - 1.0) * 4.0 + abs(epsr_delta) * 0.5 + (alpha_factor - 1.0) * 0.04)
    return defect_type, defect_length, z0_factor, epsr_delta, alpha_factor, severity


def _place_defects(length_m: float, count: int, profile: str, rng: np.random.Generator,
                   params: dict[str, Any], forced_type: str | None = None) -> list[DefectRegion]:
    if count == 0:
        return []
    margin = max(1.0, length_m * 0.025)
    specifications = [list(_sample_defect(profile, length_m, rng, params, forced_type)) for _ in range(count)]
    maximum_defect_length = length_m - 2.0 * margin - 2.0 * max(count - 1, 0)
    if maximum_defect_length <= 0:
        raise ValueError("cable is too short for the requested defect count")

    # PLAN-DG explicitly requires long defects to downgrade to short when the
    # sampled cable has insufficient healthy spacing.
    while sum(float(item[1]) for item in specifications) > maximum_defect_length:
        long_indices = [index for index, item in enumerate(specifications) if item[0] != "short"]
        if not long_indices:
            scale = maximum_defect_length / sum(float(item[1]) for item in specifications)
            for item in specifications:
                item[1] = float(item[1]) * scale
            break
        target = max(long_indices, key=lambda index: float(specifications[index][1]))
        specifications[target] = list(_sample_defect(profile, length_m, rng, params, "short"))

    fixed_gaps = np.asarray([margin] + [2.0] * max(count - 1, 0) + [margin], dtype=float)
    extra_healthy = length_m - sum(float(item[1]) for item in specifications) - float(fixed_gaps.sum())
    gaps = fixed_gaps + rng.dirichlet(np.ones(count + 1)) * max(extra_healthy, 0.0)
    regions: list[DefectRegion] = []
    cursor = float(gaps[0])
    for index, item in enumerate(specifications):
        defect_type, defect_length, z0_factor, epsr_delta, alpha_factor, severity = item
        start = cursor
        end = start + float(defect_length)
        regions.append(DefectRegion(index, str(defect_type), start, end,
                                    float(z0_factor), float(epsr_delta), float(alpha_factor), float(severity)))
        cursor = end + float(gaps[index + 1])
    return regions


def _place_joints(length_m: float, count: int, defects: list[DefectRegion],
                  rng: np.random.Generator, params: dict[str, Any]) -> list[Joint]:
    if count == 0:
        return []
    variants = params["joints"]["variants"]
    result: list[Joint] = []
    margin = max(2.0, length_m * 0.01)
    candidates = np.linspace(margin, length_m - margin, max(32, int(length_m) + 1))
    for value in candidates[rng.permutation(len(candidates))]:
        position = float(value)
        if any(abs(position - item.position_m) < 5.0 for item in result):
            continue
        if any(item.start_m - 2.0 <= position <= item.end_m + 2.0 for item in defects):
            continue
        variant = _sample_variant(rng, variants)
        variant_spec = variants[variant]
        result.append(Joint(position, variant, _uniform(rng, variant_spec["series_ohm"]),
                            _uniform(rng, variant_spec["shunt_pf"])))
        if len(result) == count:
            break
    return sorted(result, key=lambda item: item.position_m)


def _debye_parameters(defect: DefectRegion | None,
                      defect_params: dict[str, Any]) -> tuple[float, float, float]:
    if defect is None or defect.type == "short":
        return 0.0, 80.0e6, 1.0
    defaults = {
        "aging": (0.22, 0.015, 0.12, 35.0e6),
        "moisture_local": (0.48, 0.07, 0.38, 13.0e6),
        "moisture_distributed": (0.42, 0.06, 0.34, 9.0e6),
    }
    scale_default, minimum_default, maximum_default, corner_default = defaults[defect.type]
    spec = defect_params[defect.type].get("debye", {})
    scale = float(spec.get("delta_epsr_scale", scale_default))
    minimum = float(spec.get("delta_epsr_min", minimum_default))
    maximum = float(spec.get("delta_epsr_max", maximum_default))
    delta = float(np.clip(defect.epsr_delta * scale, minimum, maximum))
    return delta, float(spec.get("corner_hz", corner_default)), float(spec.get("exponent", 0.90))


def _dielectric_budget_db_per_m(segment: CableSegment) -> float:
    ratio = (FREQUENCY_REFERENCE_HZ / segment.debye_corner_hz) ** segment.debye_exponent
    debye = segment.debye_delta_epsr / complex(1.0, ratio)
    effective_epsr = segment.epsr + max(debye.real, 0.0)
    effective_tan = segment.tan_delta_at_100mhz
    if segment.epsr + debye.real > 0.0:
        effective_tan += max(-debye.imag / (segment.epsr + debye.real), 0.0)
    beta = 2.0 * math.pi * FREQUENCY_REFERENCE_HZ * math.sqrt(effective_epsr) / C0
    return 8.686 * 0.5 * beta * effective_tan


def _coax_geometry(z0_ohm: float, epsr: float, conductor_alpha_db_per_m: float,
                   conductivity: float,
                   dielectric_conductivity: float) -> tuple[float, float]:
    log_term = z0_ohm * math.sqrt(epsr) / 60.0
    radius_ratio = math.exp(log_term)
    omega = 2.0 * math.pi * FREQUENCY_REFERENCE_HZ

    def alpha_for_radius(inner_radius: float) -> float:
        shield_radius = inner_radius * radius_ratio
        capacitance = 2.0 * math.pi * epsr * EPS0 / log_term
        conductance = 2.0 * math.pi * dielectric_conductivity / log_term
        skin_depth = math.sqrt(2.0 / (omega * MU0 * conductivity))
        resistance = (1.0 / (2.0 * math.pi * inner_radius * skin_depth * conductivity)
                      + 1.0 / (2.0 * math.pi * shield_radius * skin_depth * conductivity))
        inductance = (MU0 * log_term / (2.0 * math.pi)
                      + MU0 * (1.0 / inner_radius + 1.0 / shield_radius)
                      * skin_depth / (4.0 * math.pi))
        gamma = np.sqrt(complex(resistance, omega * inductance)
                        * complex(conductance, omega * capacitance))
        return float(gamma.real * 8.686)

    lower, upper = 1.0e-5, 5.0e-3
    if not alpha_for_radius(lower) >= conductor_alpha_db_per_m >= alpha_for_radius(upper):
        raise ValueError("RG58 conductor attenuation is outside the effective coax geometry range")
    for _ in range(80):
        middle = 0.5 * (lower + upper)
        if alpha_for_radius(middle) > conductor_alpha_db_per_m:
            lower = middle
        else:
            upper = middle
    inner = 0.5 * (lower + upper)
    return inner, inner * radius_ratio


def _effective_series_resistance(segment: CableSegment,
                                 shunt_conductance_s_per_m: float) -> float:
    target_np = segment.alpha_db_per_m_at_100mhz / 8.685889638
    omega = 2.0 * math.pi * FREQUENCY_REFERENCE_HZ
    velocity = C0 / math.sqrt(segment.epsr)
    inductance = segment.z0_ohm / velocity
    capacitance_inf = 1.0 / (segment.z0_ohm * velocity)
    eps_complex = _complex_permittivity(np.asarray([FREQUENCY_REFERENCE_HZ]), segment)[0]
    capacitance = capacitance_inf * eps_complex / segment.epsr
    capacitance_real = capacitance_inf * max(eps_complex.real, 1.01) / segment.epsr
    shunt = (shunt_conductance_s_per_m
             + omega * capacitance_real * segment.tan_delta_at_100mhz
             + 1j * omega * capacitance)

    def attenuation(resistance: float) -> float:
        series = resistance + 1j * (omega * inductance + resistance)
        return float(np.sqrt(series * shunt).real)

    if attenuation(0.0) >= target_np:
        raise ValueError("dielectric loss exceeds the requested 100 MHz attenuation")
    upper = 1.0
    while attenuation(upper) < target_np:
        upper *= 2.0
    lower = 0.0
    for _ in range(80):
        middle = 0.5 * (lower + upper)
        if attenuation(middle) < target_np:
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)


def derive_material(segment: CableSegment, material_spec: dict[str, Any]) -> RLGCMaterial:
    model = str(material_spec["model"])
    if model == "coax_rlgc":
        conductivity = float(material_spec["conductor_conductivity_s_per_m"])
        dielectric_conductivity = float(material_spec["dielectric_conductivity_s_per_m"])
        conductor_target = segment.alpha_db_per_m_at_100mhz - _dielectric_budget_db_per_m(segment)
        if conductor_target <= 0.0:
            raise ValueError("dielectric loss exceeds the requested 100 MHz attenuation")
        inner, shield = _coax_geometry(segment.z0_ohm, segment.epsr,
                                       conductor_target, conductivity,
                                       dielectric_conductivity)
        return RLGCMaterial(model, conductivity, dielectric_conductivity,
                            effective_inner_radius_m=inner,
                            effective_shield_radius_m=shield)
    if model == "effective_rlgc":
        shunt_conductance = float(material_spec["shunt_conductance_s_per_m"])
        resistance = _effective_series_resistance(segment, shunt_conductance)
        return RLGCMaterial(model, 0.0,
                            shunt_conductance_s_per_m=shunt_conductance,
                            series_resistance_100mhz_ohm_per_m=resistance)
    raise ValueError(f"unsupported material model: {model!r}")


def _smoothstep(value: float) -> float:
    clipped = float(np.clip(value, 0.0, 1.0))
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _long_defect_weights(defect: DefectRegion, normalized_position: float,
                         defect_params: dict[str, Any]) -> tuple[float, float]:
    if defect.type == "aging":
        weight = math.sin(math.pi * normalized_position) ** 2
        return weight, weight
    defaults = {
        "moisture_local": (0.12, 0.62, 0.30),
        "moisture_distributed": (0.10, 0.52, 0.38),
    }
    attack_default, release_default, residual_default = defaults[defect.type]
    spec = defect_params[defect.type].get("debye", {})
    attack_fraction = float(spec.get("attack_fraction", attack_default))
    release_start = float(spec.get("release_start", release_default))
    residual_loss = float(spec.get("residual_loss", residual_default))
    attack = _smoothstep(normalized_position / attack_fraction)
    release = 1.0 - _smoothstep((normalized_position - release_start) / (1.0 - release_start))
    electrical = attack * release
    loss_release_start = min(0.78, release_start + 0.15)
    loss_release = 1.0 - (1.0 - residual_loss) * _smoothstep(
        (normalized_position - loss_release_start) / (1.0 - loss_release_start)
    )
    return electrical, attack * loss_release


def build_topology(
    profile: str,
    seed: int,
    config: GeneratorConfig,
    overrides: dict[str, Any] | None = None,
) -> CableTopology:
    """Create one labelled physical cable topology from one seed."""
    if profile not in {"rg58", "field"}:
        raise ValueError("profile must be 'rg58' or 'field'")
    rng = np.random.default_rng(seed)
    params = dict(config.parameters)
    profile_params = params["profiles"][profile]
    requested = dict(overrides or {})
    length_m = float(requested.get("length_m", _uniform(rng, profile_params["length_m"])))
    if not profile_params["length_m"]["min"] <= length_m <= profile_params["length_m"]["max"]:
        raise ValueError(f"{profile} length outside configured range")
    z_ref = 50.0
    base_z0 = _uniform(rng, profile_params["base_z0_ohm"])
    base_epsr = float(requested.get("epsr", _uniform(rng, profile_params["epsr"])))
    if base_epsr <= 1.0:
        raise ValueError("epsr must be greater than 1")
    base_alpha = _uniform(rng, profile_params["alpha_db_per_m_at_100mhz"])
    base_tan = _uniform(rng, profile_params["tan_delta_at_100mhz"])
    material_spec = profile_params.get("material")
    if material_spec is None:
        raise ValueError(f"{profile} profile is missing its RLGC material configuration")
    defect_count = int(requested.get("defect_count", _defect_count(profile, length_m, params, rng)))
    if defect_count < 0 or defect_count > int(profile_params["defect_count"]["max"]):
        raise ValueError("defect_count outside configured range")
    defects = _place_defects(
        length_m,
        defect_count,
        profile,
        rng,
        params,
        requested.get("defect_type"),
    )
    requested_joint_positions = requested.get("joint_positions_m")
    if requested_joint_positions is not None:
        variants = params["joints"]["variants"]
        joints = []
        for value in requested_joint_positions:
            position = float(value)
            if not 0.0 < position < length_m:
                raise ValueError("joint position outside cable")
            variant = _sample_variant(rng, variants)
            spec = variants[variant]
            joints.append(Joint(position, variant, _uniform(rng, spec["series_ohm"]), _uniform(rng, spec["shunt_pf"])))
        joints.sort(key=lambda item: item.position_m)
    else:
        joint_limit = profile_params["joint_count"]
        joint_margin = max(2.0, length_m * 0.01)
        feasible_joint_count = max(0, int((length_m - 2.0 * joint_margin) // 5.0) + 1)
        joint_maximum = min(int(joint_limit["max"]), feasible_joint_count)
        joint_minimum = min(int(joint_limit["min"]), joint_maximum)
        joint_count = int(rng.integers(joint_minimum, joint_maximum + 1))
        joints = _place_joints(length_m, joint_count, defects, rng, params)
    termination = requested.get("termination")
    if termination is None:
        termination = "open"
        if profile == "field" and rng.random() >= float(params["profiles"]["field_open_probability"]):
            termination = "short"
    if termination not in {"open", "short"} or (profile == "rg58" and termination != "open"):
        raise ValueError("unsupported termination for profile")
    load_key = "open_load_ohm" if termination == "open" else "short_load_ohm"
    z_load = _uniform(rng, profile_params[load_key])

    boundaries = {0.0, length_m}
    for region in defects:
        boundaries.update((region.start_m, region.end_m))
        if region.type != "short":
            debye_spec = params["defects"][region.type].get("debye", {})
            spacing_defaults = {"aging": 18.0, "moisture_local": 8.0,
                                "moisture_distributed": 14.0}
            spacing = float(debye_spec.get("spacing_m", spacing_defaults[region.type]))
            part_count = int(np.clip(round((region.end_m - region.start_m) / spacing), 13, 81))
            boundaries.update(np.linspace(region.start_m, region.end_m, part_count + 1).tolist())
    boundaries.update(item.position_m for item in joints)
    ordered = sorted(boundaries)
    segments: list[CableSegment] = []
    for start, end in zip(ordered[:-1], ordered[1:]):
        midpoint = (start + end) / 2.0
        region_name = "healthy"
        defect_id = None
        z_factor, eps_delta, alpha_factor = 1.0, 0.0, 1.0
        tan_delta = base_tan
        debye_delta, debye_corner, debye_exponent = 0.0, 80.0e6, 1.0
        for defect in defects:
            if defect.start_m <= midpoint <= defect.end_m:
                region_name, defect_id = defect.type, defect.index
                if defect.type == "short":
                    z_factor, eps_delta, alpha_factor = defect.z0_factor, defect.epsr_delta, defect.alpha_factor
                    tan_delta = base_tan * (1.0 + 0.45 * (alpha_factor - 1.0))
                else:
                    normalized = (midpoint - defect.start_m) / (defect.end_m - defect.start_m)
                    electrical_weight, loss_weight = _long_defect_weights(defect, normalized, params["defects"])
                    debye_peak, debye_corner, debye_exponent = _debye_parameters(defect, params["defects"])
                    z0_weights = {"aging": 0.22, "moisture_local": 0.34,
                                  "moisture_distributed": 0.0}
                    z0_weight = float(params["defects"][defect.type].get(
                        "debye", {}).get("z0_weight", z0_weights[defect.type]))
                    z_factor = 1.0 + (defect.z0_factor - 1.0) * z0_weight * electrical_weight
                    eps_delta = defect.epsr_delta * electrical_weight
                    alpha_factor = 1.0 + (defect.alpha_factor - 1.0) * loss_weight
                    debye_delta = debye_peak * electrical_weight
                    debye_spec = params["defects"][defect.type].get("debye", {})
                    tan_defaults = {
                        "aging": (1.5, 0.55, 6.0e-4, 5.0e-3),
                        "moisture_local": (2.8, 1.05, 3.0e-3, 2.4e-2),
                        "moisture_distributed": (2.7, 0.95, 2.2e-3, 2.0e-2),
                    }
                    multiplier_default, ratio_scale_default, tan_min_default, tan_max_default = \
                        tan_defaults[defect.type]
                    tan_target = base_tan * (
                        float(debye_spec.get("tan_multiplier_base", multiplier_default))
                        + float(debye_spec.get("tan_alpha_ratio_scale", ratio_scale_default))
                        * defect.alpha_factor
                    )
                    tan_target = float(np.clip(
                        tan_target,
                        float(debye_spec.get("tan_delta_min", tan_min_default)),
                        float(debye_spec.get("tan_delta_max", tan_max_default)),
                    ))
                    tan_delta = base_tan + (tan_target - base_tan) * loss_weight
                break
        segment = CableSegment(
            float(start), float(end), float(base_z0 * z_factor),
            float(base_epsr + eps_delta), float(base_alpha * alpha_factor),
            float(tan_delta), region_name, defect_id,
            float(debye_delta), float(debye_corner), float(debye_exponent),
        )
        segments.append(replace(segment, material=derive_material(segment, material_spec)))
    return CableTopology(profile, length_m, z_ref, base_z0, base_epsr, base_alpha,
                         base_tan, segments, joints, termination, z_load, defects)
