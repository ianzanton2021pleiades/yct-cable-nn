"""
[V2]DG_dataset_max2.5km.py - DirtyGenerator for <=2.5 km cable S11 datasets.

Generates two Client-style CSV files per sample:
  * 9 kHz-1 GHz, 50000 points
  * 9 kHz-200 MHz, 5000 points

The label grid is independent from the CSV distance axis:
  D_MAX=2500 m, DD=0.25 m, N_GRID=10000.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

import matplotlib
if "tk" not in matplotlib.get_backend().lower():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from core.label import build_label_vector
from core.s11_generator import CableSample, SegmentParams, SweepConfig, generate_s11
from core.tdr_signal import s11_to_responses, estimate_first_step, apply_window, fft_shift


D_MAX = 2500.0
DD = 0.25
N_GRID = int(round(D_MAX / DD))
LABEL_GRID = np.linspace(0, D_MAX, N_GRID, endpoint=False) + DD / 2.0

SWEEP_1GHZ = SweepConfig(start_hz=9e3, stop_hz=1e9, n_points=50000)
SWEEP_200MHZ = SweepConfig(start_hz=9e3, stop_hz=200e6, n_points=5000)

CSV_HEADER = [
    "Frequency",
    "S11_Real",
    "S11_Imaginary",
    "Distance",
    "ImpulseResponse",
    "StepResponse",
]

SUPPORTED_DEFECT_TYPES = ["short", "aging", "moisture_local", "moisture_distributed"]


@dataclass
class DirtyParams:
    profile: str
    additive_scale: float
    multiplicative_scale: float
    ripple_scale: float
    phase_scale_rad: float
    fixture_scale: float
    template_slow_scale: float
    template_mix_scale: float
    highfreq_decay_strength: float
    event_hf_damping: float


def _safe_float(value: str) -> float:
    text = str(value).strip().replace("\ufeff", "")
    if not text:
        return float("nan")
    return float(text)


def read_s11_csv_compatible(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read RG58/field CSV files with fuzzy Frequency/Real/Imag column matching."""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        fieldnames = [name for name in reader.fieldnames if name is not None]
        lower = {name: name.lower() for name in fieldnames}

        freq_col = next((name for name, low in lower.items() if "freq" in low), None)
        real_col = next((name for name, low in lower.items() if "real" in low), None)
        imag_col = next((name for name, low in lower.items() if "imag" in low), None)
        if not freq_col or not real_col or not imag_col:
            raise ValueError(f"Cannot find freq/real/imag columns in {path}")

        freqs: list[float] = []
        vals: list[complex] = []
        for row in reader:
            try:
                f_hz = _safe_float(row.get(freq_col, ""))
                re_v = _safe_float(row.get(real_col, ""))
                im_v = _safe_float(row.get(imag_col, ""))
            except ValueError:
                continue
            if np.isfinite(f_hz) and f_hz > 0 and np.isfinite(re_v) and np.isfinite(im_v):
                freqs.append(f_hz)
                vals.append(complex(re_v, im_v))

    if len(freqs) < 16:
        raise ValueError(f"Too few valid S11 rows in {path}")
    freqs_arr = np.asarray(freqs, dtype=np.float64)
    s11_arr = np.asarray(vals, dtype=np.complex128)
    order = np.argsort(freqs_arr)
    return freqs_arr[order], s11_arr[order]


def infer_length_from_path(path: Path, default_m: float = 300.0, clip: bool = True) -> float:
    """Infer approximate cable length from known RG58 names or folder text."""
    text = str(path)
    if "RG58-74M" in text:
        return 74.0
    if "RG58-3Lines" in text:
        name = path.name
        total = 0.0
        if "LineA" in name:
            total += 40.0
        if "LineB" in name:
            total += 30.0
        if "LineC" in name:
            total += 25.0
        if "CUT1" in name:
            total += 1.0
        if total > 0:
            return total
        return 25.0

    for part in [path.parent.name, path.name]:
        matches = [float(m) for m in re.findall(r"(\d+(?:\.\d+)?)\s*m", part, flags=re.I)]
        if matches:
            value = matches[-1]
            return float(np.clip(value, 25.0, 2500.0) if clip else value)
        cn_matches = [float(m) for m in re.findall(r"(\d+(?:\.\d+)?)\s*米", part)]
        if cn_matches:
            value = cn_matches[-1]
            return float(np.clip(value, 25.0, 2500.0) if clip else value)
    return float(np.clip(default_m, 25.0, 2500.0) if clip else default_m)


_TEMPLATE_QUALITY_CACHE: dict[tuple[str, str], float] = {}


def _looks_like_calibration_file(path: Path) -> bool:
    name = path.name.lower()
    if "校正数据" in path.name or "calibration" in name or "cal" in name:
        return True
    # Field calibration captures in the supplied corpus are roughly one fifth
    # the size/point count of normal 50k-point measurements.  This fallback
    # also survives incorrectly decoded Chinese ZIP filenames.
    try:
        return path.stat().st_size < 900_000 and "rg58" not in str(path).lower()
    except OSError:
        return False


def assess_template_quality(path: Path, profile: str) -> float:
    """Return a conservative 0..1 template score.

    V2 never transfers the template's coherent phase, so this score only needs
    to reject corrupt, clipped, or spectrally pathological files.
    """
    key = (str(path), str(profile))
    cached = _TEMPLATE_QUALITY_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        f, s11 = read_s11_csv_compatible(path)
        mag = np.abs(s11)
        finite = np.isfinite(f) & np.isfinite(s11.real) & np.isfinite(s11.imag)
        if int(finite.sum()) < 4000:
            score = 0.0
        else:
            f = f[finite]
            s11 = s11[finite]
            mag = mag[finite]
            span = float(f[-1] - f[0])
            p50, p95, p995 = np.nanpercentile(mag, [50, 95, 99.5])
            diff_scale = float(np.nanpercentile(np.abs(np.diff(s11)), 95)) / max(float(p95), 1e-6)
            clip_fraction = float(np.mean(mag >= 1.245))
            flat_fraction = float(np.mean(np.abs(np.diff(s11)) < 1e-11))
            score = 1.0
            if span < 180e6:
                score -= 0.55
            elif span < 800e6:
                score -= 0.15
            if p995 > 1.35:
                score -= min(0.45, (p995 - 1.35) * 0.9)
            if p50 < 1e-4 or p95 < 5e-4:
                score -= 0.35
            if diff_scale > 0.55:
                score -= min(0.40, (diff_scale - 0.55) * 0.55)
            score -= min(0.45, clip_fraction * 8.0)
            score -= min(0.35, flat_fraction * 5.0)
            score = float(np.clip(score, 0.0, 1.0))
    except Exception:
        score = 0.0
    _TEMPLATE_QUALITY_CACHE[key] = score
    return score


def discover_real_files(real_data_root: Path) -> tuple[list[Path], list[Path], list[Path]]:
    """Discover reference files without depending on one directory layout.

    RG58 is identified by path text.  Non-RG58 small captures or explicitly
    named files are treated as fixture calibration; all other CSVs are field
    measurements.
    """
    rg58_files: list[Path] = []
    field_files: list[Path] = []
    calibration_files: list[Path] = []
    if not real_data_root.exists():
        return rg58_files, field_files, calibration_files
    for path in sorted(real_data_root.rglob("*.csv")):
        path_text = str(path).lower()
        if "rg58" in path_text:
            rg58_files.append(path)
        elif _looks_like_calibration_file(path):
            calibration_files.append(path)
        else:
            field_files.append(path)
    return rg58_files, field_files, calibration_files


def _segment(
    length_m: float,
    z0: float,
    epsr: float,
    alpha: float,
    defect: bool = False,
    defect_type: str = "short",
    label_amplitude: float | None = None,
    defect_group: str | None = None,
    tan_delta: float = 2.5e-4,
    debye_delta_epsr: float = 0.0,
    debye_corner_hz: float = 80e6,
    debye_exponent: float = 1.0,
) -> SegmentParams:
    segment = SegmentParams(
        length_m=max(float(length_m), 0.05),
        z0_ohm=float(z0),
        epsr=float(epsr),
        alpha_db_per_m_100mhz=float(alpha),
        is_defect=defect,
        tan_delta_100mhz=float(max(tan_delta, 0.0)),
        debye_delta_epsr=float(max(debye_delta_epsr, 0.0)),
        debye_corner_hz=float(max(debye_corner_hz, 1.0)),
        debye_exponent=float(np.clip(debye_exponent, 0.55, 1.35)),
    )
    if defect:
        segment.defect_type = str(defect_type)
        if defect_group is not None:
            segment.defect_group = str(defect_group)
        if label_amplitude is not None:
            segment.label_amplitude = float(label_amplitude)
    return segment


def defect_count_policy(profile: str, total_length: float, override: int | None = None) -> dict:
    length_m = float(total_length)
    if override is not None:
        count = int(override)
        max_count = 5 if profile == "field" else 2
        probabilities = [0.0] * (max_count + 1)
        probabilities[max(0, min(count, max_count))] = 1.0
        return {
            "profile": profile,
            "length_m": length_m,
            "source": "override",
            "max_defects": max_count,
            "probabilities": probabilities,
        }

    if profile == "field":
        if length_m < 80.0:
            band = "<80m"
            probabilities = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        elif length_m < 250.0:
            band = "80-250m"
            probabilities = [0.45, 0.35, 0.17, 0.03, 0.0, 0.0]
        elif length_m < 500.0:
            band = "250-500m"
            probabilities = [0.28, 0.34, 0.24, 0.10, 0.03, 0.01]
        elif length_m < 800.0:
            band = "500-800m"
            probabilities = [0.22, 0.31, 0.27, 0.13, 0.05, 0.02]
        elif length_m < 1500.0:
            band = "800-1500m"
            probabilities = [0.16, 0.28, 0.29, 0.17, 0.07, 0.03]
        else:
            band = "1500-2500m"
            probabilities = [0.12, 0.24, 0.29, 0.21, 0.10, 0.04]
        return {
            "profile": profile,
            "length_m": length_m,
            "source": "length_stratified",
            "length_band": band,
            "max_defects": 5,
            "probabilities": probabilities,
        }

    if profile == "rg58_random":
        if length_m < 50.0:
            band = "10-50m"
            probabilities = [0.70, 0.25, 0.05]
        elif length_m < 120.0:
            band = "50-120m"
            probabilities = [0.58, 0.32, 0.10]
        else:
            band = "120-200m"
            probabilities = [0.50, 0.35, 0.15]
        return {
            "profile": profile,
            "length_m": length_m,
            "source": "rg58_random_length_stratified",
            "length_band": band,
            "max_defects": 2,
            "probabilities": probabilities,
        }

    return {
        "profile": profile,
        "length_m": length_m,
        "source": "known_template",
        "max_defects": None,
        "probabilities": None,
    }


def sample_defect_count(profile: str, total_length: float, rng: np.random.RandomState, override: int | None = None) -> int:
    policy = defect_count_policy(profile, total_length, override=override)
    probabilities = policy.get("probabilities")
    if probabilities is None:
        return 0
    values = np.arange(len(probabilities), dtype=int)
    return int(rng.choice(values, p=np.asarray(probabilities, dtype=np.float64)))


def defect_type_policy(profile: str, total_length: float) -> dict:
    length_m = float(total_length)
    if profile != "field":
        return {
            "profile": profile,
            "length_m": length_m,
            "allowed_types": ["short"],
            "long_probability_per_defect": 0.0,
            "max_long_defects": 0,
            "long_type_probabilities": {"aging": 0.0, "moisture_local": 0.0, "moisture_distributed": 0.0},
        }

    if length_m < 250.0:
        probability = 0.0
    elif length_m < 500.0:
        probability = 0.03
    elif length_m < 800.0:
        probability = 0.06
    elif length_m < 1500.0:
        probability = 0.10
    else:
        probability = 0.16
    return {
        "profile": profile,
        "length_m": length_m,
        "allowed_types": ["short", "aging", "moisture_local", "moisture_distributed"],
        "long_probability_per_defect": probability,
        "max_long_defects": 1 if length_m < 800.0 else 2,
        "long_type_probabilities": {"aging": 0.60, "moisture_local": 0.25, "moisture_distributed": 0.15},
    }


def normalize_allowed_defect_types(profile: str, allowed_types: Iterable[str] | str | None = None) -> list[str]:
    if allowed_types is None:
        return list(defect_type_policy(profile, 1000.0)["allowed_types"] if profile == "field" else ["short"])
    if isinstance(allowed_types, str):
        raw_items = re.split(r"[,+/;|\s]+", allowed_types.strip())
    else:
        raw_items = [str(item) for item in allowed_types]
    normalized: list[str] = []
    for item in raw_items:
        text = item.strip()
        if not text or text.lower() == "all":
            continue
        if text == "moisture":
            text = "moisture_local"
        if text in SUPPORTED_DEFECT_TYPES and text not in normalized:
            normalized.append(text)
    if not normalized:
        normalized = list(SUPPORTED_DEFECT_TYPES if profile == "field" else ["short"])
    if profile != "field":
        return ["short"]
    return [item for item in normalized if item in SUPPORTED_DEFECT_TYPES] or ["short"]


def sample_defect_types(
    profile: str,
    total_length: float,
    n_defects: int,
    rng: np.random.RandomState,
    allowed_types: Iterable[str] | str | None = None,
) -> list[str]:
    count = int(max(n_defects, 0))
    policy = defect_type_policy(profile, total_length)
    if allowed_types is not None:
        allowed = normalize_allowed_defect_types(profile, allowed_types)
        return [str(rng.choice(allowed)) for _ in range(count)]
    if count == 0 or policy["long_probability_per_defect"] <= 0:
        return ["short"] * count

    types: list[str] = []
    long_count = 0
    max_long = int(policy["max_long_defects"])
    for _ in range(count):
        if long_count < max_long and rng.random() < float(policy["long_probability_per_defect"]):
            draw = rng.random()
            if draw < 0.60:
                defect_type = "aging"
            elif draw < 0.85:
                defect_type = "moisture_local"
            else:
                defect_type = "moisture_distributed"
            long_count += 1
        else:
            defect_type = "short"
        types.append(defect_type)
    return types


def defect_length_for_type(defect_type: str, total_length: float, rng: np.random.RandomState) -> float:
    defect_type = "moisture_local" if defect_type == "moisture" else defect_type
    if defect_type == "aging":
        return float(np.clip(total_length * rng.uniform(0.05, 0.16), 20.0, 320.0))
    if defect_type == "moisture_local":
        return float(np.clip(total_length * rng.uniform(0.03, 0.10), 15.0, 220.0))
    if defect_type == "moisture_distributed":
        return float(np.clip(total_length * rng.uniform(0.18, 0.55), 120.0, max(130.0, total_length * 0.72)))
    return float(rng.uniform(0.8, min(8.0, total_length * 0.03)))


def downgrade_long_defects_if_needed(
    defect_types: list[str],
    defect_lengths: list[float],
    total_length: float,
    rng: np.random.RandomState,
    min_healthy_m: float = 5.0,
) -> tuple[list[str], list[float]]:
    types = list(defect_types)
    lengths = list(defect_lengths)
    max_defect_total = max(float(total_length) - min_healthy_m * (len(types) + 1), 0.0)
    while sum(lengths) > max_defect_total:
        long_indices = [i for i, t in enumerate(types) if t in {"aging", "moisture", "moisture_local", "moisture_distributed"}]
        if not long_indices:
            break
        idx = max(long_indices, key=lambda i: lengths[i])
        types[idx] = "short"
        lengths[idx] = defect_length_for_type("short", total_length, rng)
    return types, lengths


def append_gradual_defect_segments(
    segments: list[SegmentParams],
    length_m: float,
    base_z0: float,
    base_epsr: float,
    base_alpha: float,
    target_z0: float,
    target_epsr: float,
    target_alpha: float,
    defect_type: str,
    label_amplitude: float | None,
    group_id: str,
) -> dict | None:
    """Append one defect using only frequency-domain physical parameters.

    Long aging/moisture regions are represented by a raised-cosine taper of
    Z0, permittivity, dielectric loss, and Debye relaxation.  The endpoints
    return exactly to the healthy cable, eliminating the hard artificial
    boundaries that produced ringing in V1.
    """
    length_m = float(length_m)
    if defect_type not in {"aging", "moisture_local", "moisture_distributed"}:
        segments.append(_segment(
            length_m, target_z0, target_epsr, target_alpha, True, defect_type,
            label_amplitude, group_id, tan_delta=3.0e-4,
        ))
        return None

    start_m = float(sum(seg.length_m for seg in segments))
    alpha_ratio = float(np.clip(target_alpha / max(base_alpha, 1e-6), 1.0, 10.0))
    base_tan = float(np.clip(1.8e-4 + 0.020 * base_alpha, 1.8e-4, 8.0e-4))
    if defect_type == "aging":
        z0_weight = 0.24
        target_tan = float(np.clip(base_tan * (1.5 + 0.55 * alpha_ratio), 6e-4, 5e-3))
        debye_peak = float(np.clip((target_epsr - base_epsr) * 0.22, 0.015, 0.12))
        debye_corner = 35e6
        default_severity = 0.56
        spacing = 18.0
    elif defect_type == "moisture_local":
        z0_weight = 0.52
        target_tan = float(np.clip(base_tan * (2.2 + 0.85 * alpha_ratio), 0.002, 0.018))
        debye_peak = float(np.clip((target_epsr - base_epsr) * 0.42, 0.06, 0.34))
        debye_corner = 16e6
        default_severity = 0.62
        spacing = 10.0
    else:
        z0_weight = 0.34
        target_tan = float(np.clip(base_tan * (2.0 + 0.72 * alpha_ratio), 0.0015, 0.014))
        debye_peak = float(np.clip((target_epsr - base_epsr) * 0.36, 0.05, 0.30))
        debye_corner = 11e6
        default_severity = 0.62
        spacing = 16.0

    n_parts = int(np.clip(round(length_m / spacing), 11, 61))
    x = np.linspace(0.0, 1.0, n_parts)
    # Zero-valued endpoints and a broad plateau: smooth physical boundaries.
    edge = np.sin(math.pi * x) ** 2
    if defect_type == "moisture_distributed":
        edge = np.power(edge, 0.58)
    elif defect_type == "moisture_local":
        edge = np.power(edge, 0.78)
    else:
        edge = np.power(edge, 0.95)

    for weight in edge:
        w = float(weight)
        z0 = base_z0 + (target_z0 - base_z0) * z0_weight * w
        epsr = base_epsr + (target_epsr - base_epsr) * w
        alpha = base_alpha + (target_alpha - base_alpha) * w
        tan_delta = base_tan + (target_tan - base_tan) * w
        segments.append(_segment(
            length_m / n_parts, z0, epsr, alpha, False,
            tan_delta=tan_delta,
            debye_delta_epsr=debye_peak * w,
            debye_corner_hz=debye_corner,
            debye_exponent=0.92,
        ))

    severity = float(label_amplitude if label_amplitude is not None else default_severity)
    return {
        "type": defect_type,
        "start": start_m,
        "end": start_m + length_m,
        "position": start_m + length_m / 2.0,
        "length": length_m,
        "z0": float(target_z0),
        "epsr": float(target_epsr),
        "alpha": float(target_alpha),
        "severity": severity,
        "group": str(group_id),
    }


def make_known_rg58_cable(kind: str, rng: np.random.RandomState, epsr: float | None = None) -> CableSample:
    eps = float(epsr if epsr is not None else rng.uniform(2.18, 2.32))
    alpha = float(rng.uniform(0.045, 0.135))
    if kind == "rg58_74m":
        segments = [
            _segment(40.0, 50.0, eps, alpha),
            _segment(4.0, rng.uniform(49.2, 50.9), eps + rng.normal(0, 0.03), alpha, True),
            _segment(30.0, 50.0, eps, alpha),
        ]
    elif kind == "rg58_3lines_long":
        segments = [
            _segment(40.0, 50.0, eps, alpha),
            _segment(1.0, rng.uniform(49.0, 51.1), eps + rng.normal(0, 0.04), alpha, True),
            _segment(25.0, 50.0, eps, alpha),
            _segment(30.0, 50.0, eps, alpha),
        ]
    elif kind == "rg58_3lines_ab":
        segments = [
            _segment(40.0, 50.0, eps, alpha),
            _segment(1.0, rng.uniform(49.0, 51.1), eps + rng.normal(0, 0.04), alpha, True),
            _segment(30.0, 50.0, eps, alpha),
        ]
    else:
        segments = [_segment(rng.uniform(25.0, 120.0), 50.0, eps, alpha)]
    cable = CableSample(segments=segments, epsr=eps, seed=int(rng.randint(0, 2**31)))
    cable.has_joint_reflections = False
    cable.joint_positions = cumulative_internal_positions(cable)
    cable.z_load_open = float(rng.uniform(650.0, 6500.0))
    cable.termination = "finite_open"
    cable.defect_count_policy = defect_count_policy("rg58", cable.total_length)
    cable.defect_type_policy = defect_type_policy("rg58", cable.total_length)
    return cable


def make_random_rg58_cable(
    rng: np.random.RandomState,
    total_length: float | None = None,
    epsr: float | None = None,
    n_defects_override: int | None = None,
) -> CableSample:
    """Generate a variable-segment RG58 topology while keeping RG58-like physics."""
    eps = float(epsr if epsr is not None else rng.uniform(2.18, 2.32))
    alpha = float(rng.uniform(0.045, 0.135))
    if total_length is None:
        total_length = float(rng.uniform(10.0, 200.0))
    total_length = float(np.clip(total_length, 10.0, 200.0))
    policy = defect_count_policy("rg58_random", total_length, override=n_defects_override)
    n_defects = sample_defect_count("rg58_random", total_length, rng, override=n_defects_override)

    # Use Dirichlet partitioning like the field generator, but keep enough
    # minimum length to avoid many sub-meter numerical artifacts.
    n_segments = max(int(rng.randint(1, 8)), n_defects * 2 + 1)
    if n_segments == 1:
        lengths = np.asarray([total_length], dtype=np.float64)
    else:
        min_len = min(2.0, total_length / (2.0 * n_segments))
        free_length = max(total_length - min_len * n_segments, 0.0)
        weights = rng.dirichlet(np.ones(n_segments))
        lengths = min_len + weights * free_length
        lengths *= total_length / lengths.sum()

    base_z0 = float(rng.uniform(49.2, 50.8))
    segments: list[SegmentParams] = []
    candidate_indices = list(range(1, max(n_segments - 1, 1)))
    if n_defects > 0 and candidate_indices:
        defect_indices = set(rng.choice(candidate_indices, size=min(n_defects, len(candidate_indices)), replace=False).tolist())
    else:
        defect_indices = set()
    for idx, length_m in enumerate(lengths):
        is_internal_adapter = idx in defect_indices
        z_jitter = rng.normal(0.0, 0.45)
        eps_jitter = rng.normal(0.0, 0.025)
        if is_internal_adapter:
            z_jitter += rng.choice([-1.0, 1.0]) * rng.uniform(0.8, 2.2)
            eps_jitter += rng.normal(0.0, 0.035)
        segments.append(_segment(
            float(length_m),
            float(np.clip(base_z0 + z_jitter, 46.0, 54.0)),
            float(max(1.8, eps + eps_jitter)),
            float(alpha * rng.uniform(0.85, 1.25)),
            is_internal_adapter,
        ))

    cable = CableSample(segments=segments, epsr=eps, seed=int(rng.randint(0, 2**31)))
    cable.has_joint_reflections = False
    cable.joint_positions = cumulative_internal_positions(cable)
    cable.z_load_open = float(rng.uniform(650.0, 6500.0))
    cable.termination = "finite_open"
    cable.defect_count_policy = {**policy, "sampled_count": int(len(cable.defect_info))}
    cable.defect_type_policy = {**defect_type_policy("rg58_random", cable.total_length), "sampled_types": [d["type"] for d in cable.defect_info]}
    return cable


def make_field_cable(
    rng: np.random.RandomState,
    total_length: float | None = None,
    epsr: float | None = None,
    termination: str | None = None,
    n_defects_override: int | None = None,
    allowed_defect_types: Iterable[str] | str | None = None,
) -> CableSample:
    eps = float(epsr if epsr is not None else rng.uniform(2.1, 3.2))
    if total_length is None:
        total_length = float(np.clip(rng.lognormal(np.log(700.0), 0.75), 100.0, 2500.0))
    total_length = float(np.clip(total_length, 30.0, 2500.0))

    policy = defect_count_policy("field", total_length, override=n_defects_override)
    type_policy = defect_type_policy("field", total_length)
    n_defects = sample_defect_count("field", total_length, rng, override=n_defects_override)
    # Field power cables retain visible long-range coherent reflections in the
    # measured S11. The previous 0.05-0.16 dB/m @100MHz range attenuated
    # 1-2 km terminal events by tens of dB and forced later non-physical boosts.
    healthy_alpha = float(rng.uniform(0.003, 0.026))
    segments: list[SegmentParams] = []
    distributed_moisture_regions: list[dict] = []
    distributed_long_regions: list[dict] = []
    if n_defects == 0 or total_length < 80.0:
        segments.append(_segment(total_length, rng.uniform(42.0, 75.0), eps, healthy_alpha))
    else:
        defect_types = sample_defect_types("field", total_length, n_defects, rng, allowed_types=allowed_defect_types)
        defect_lengths = [defect_length_for_type(t, total_length, rng) for t in defect_types]
        defect_types, defect_lengths = downgrade_long_defects_if_needed(defect_types, defect_lengths, total_length, rng)
        healthy_total = total_length - sum(defect_lengths)
        weights = rng.dirichlet(np.ones(n_defects + 1))
        healthy_lengths = np.maximum(weights * healthy_total, 5.0)
        healthy_lengths *= healthy_total / healthy_lengths.sum()
        base_z0 = float(rng.uniform(42.0, 75.0))
        for i in range(n_defects):
            segments.append(_segment(healthy_lengths[i], base_z0, eps + rng.normal(0, 0.04), healthy_alpha))
            defect_type = defect_types[i]
            if defect_type == "aging":
                mismatch = rng.choice([-1.0, 1.0]) * rng.uniform(0.02, 0.08)
                defect_epsr = eps + rng.uniform(0.05, 0.25)
                defect_alpha = healthy_alpha * rng.uniform(1.4, 3.0)
                label_amplitude = rng.uniform(0.45, 0.65)
            elif defect_type == "moisture":
                defect_type = "moisture_local"
                mismatch = -rng.uniform(0.03, 0.12)
                defect_epsr = eps + rng.uniform(0.20, 0.80)
                defect_alpha = healthy_alpha * rng.uniform(2.0, 6.0)
                label_amplitude = rng.uniform(0.55, 0.75)
            elif defect_type == "moisture_local":
                mismatch = -rng.uniform(0.03, 0.12)
                defect_epsr = eps + rng.uniform(0.20, 0.80)
                defect_alpha = healthy_alpha * rng.uniform(2.0, 6.0)
                label_amplitude = rng.uniform(0.55, 0.75)
            elif defect_type == "moisture_distributed":
                mismatch = -rng.uniform(0.20, 0.38)
                defect_epsr = eps + rng.uniform(0.18, 0.70)
                defect_alpha = healthy_alpha * rng.uniform(2.8, 6.0)
                label_amplitude = rng.uniform(0.52, 0.72)
            else:
                mismatch = rng.choice([-1.0, 1.0]) * rng.uniform(0.03, 0.18)
                defect_epsr = max(1.5, eps + rng.normal(0, 0.18))
                defect_alpha = healthy_alpha * rng.uniform(0.8, 2.5)
                label_amplitude = None
            region = append_gradual_defect_segments(
                segments,
                defect_lengths[i],
                base_z0,
                eps,
                healthy_alpha,
                max(15.0, base_z0 * (1.0 + mismatch)),
                max(1.5, defect_epsr),
                defect_alpha,
                defect_type,
                label_amplitude,
                f"field-{i}",
            )
            if region is not None:
                if region.get("type") == "moisture_distributed":
                    distributed_moisture_regions.append(region)
                else:
                    distributed_long_regions.append(region)
        segments.append(_segment(healthy_lengths[-1], base_z0, eps + rng.normal(0, 0.04), healthy_alpha))

    cable = CableSample(segments=segments, epsr=eps, seed=int(rng.randint(0, 2**31)))
    cable.distributed_moisture_regions = distributed_moisture_regions
    cable.distributed_long_regions = distributed_long_regions
    cable.defect_count_policy = {**policy, "sampled_count": int(len(cable.defect_info))}
    cable.defect_type_policy = {**type_policy, "sampled_types": [d["type"] for d in cable.defect_info]}
    cable.has_joint_reflections = False
    # Model construction boundaries are not physical joints.  V1 sampled from
    # every internal segment boundary, so the many taper segments inside a wet
    # region could become unlabeled reflection peaks.  Draw actual joints as a
    # separate sparse topology and keep them away from defect boundaries.
    expected_joints = max(total_length / 650.0, 0.0)
    n_joints = int(np.clip(rng.poisson(expected_joints), 0, 6))
    defect_ranges = [
        (float(d.get("start", d["position"] - d["length"] / 2.0)),
         float(d.get("end", d["position"] + d["length"] / 2.0)))
        for d in cable.defect_info
    ]
    joints: list[float] = []
    for _ in range(160):
        if len(joints) >= n_joints or total_length <= 120.0:
            break
        candidate = float(rng.uniform(45.0, total_length - 45.0))
        if any(start - 35.0 <= candidate <= end + 35.0 for start, end in defect_ranges):
            continue
        if any(abs(candidate - existing) < 120.0 for existing in joints):
            continue
        joints.append(candidate)
    cable.joint_positions = sorted(joints)
    if termination is None:
        termination = rng.choice(["open", "weak_open", "short"], p=[0.58, 0.22, 0.20])
    termination = str(termination)
    if termination == "short":
        # A field short must remain a true negative-reflection termination.
        cable.z_load_open = float(rng.uniform(0.4, 12.0))
    elif termination == "weak_open":
        cable.z_load_open = float(rng.uniform(110.0, 420.0))
    else:
        # V1 used 55-120 ohm for an "open", which is almost matched and
        # inevitably erased the terminal step.  Use a finite but genuinely
        # open termination; line loss and the causal terminal kernel below
        # determine its observed amplitude.
        if total_length < 500.0:
            cable.z_load_open = float(rng.uniform(900.0, 9000.0))
        else:
            cable.z_load_open = float(rng.uniform(450.0, 4500.0))
        termination = "open"
    cable.termination = termination
    return cable


def cumulative_internal_positions(cable: CableSample) -> list[float]:
    positions: list[float] = []
    pos = 0.0
    for seg in cable.segments[:-1]:
        pos += seg.length_m
        positions.append(float(pos))
    return positions


def is_rg58_profile(profile: str) -> bool:
    return profile in {"rg58", "rg58_random"}


def choose_profile(profile: str, rng: np.random.RandomState) -> str:
    if profile == "mixed":
        return "rg58" if rng.random() < 0.35 else "field"
    return profile


def make_cable_for_profile(profile: str, rng: np.random.RandomState, total_length: float | None = None) -> CableSample:
    if profile == "rg58_random":
        return make_random_rg58_cable(rng, total_length=total_length)
    if profile == "rg58":
        if total_length is not None:
            eps = float(rng.uniform(2.18, 2.32))
            cable = CableSample(
                segments=[_segment(total_length, 50.0, eps, rng.uniform(0.045, 0.135))],
                epsr=eps,
                has_joint_reflections=False,
                seed=int(rng.randint(0, 2**31)),
            )
            cable.z_load_open = float(rng.uniform(650.0, 6500.0))
            cable.termination = "finite_open"
            cable.defect_count_policy = defect_count_policy("rg58", cable.total_length)
            cable.defect_type_policy = defect_type_policy("rg58", cable.total_length)
            return cable
        return make_known_rg58_cable(rng.choice(["rg58_74m", "rg58_3lines_ab", "rg58_3lines_long"]), rng)
    return make_field_cable(rng, total_length=total_length)


def interpolate_s11(freq_src: np.ndarray, s_src: np.ndarray, freq_dst: np.ndarray) -> np.ndarray:
    re_part = np.interp(freq_dst, freq_src, s_src.real, left=s_src.real[0], right=s_src.real[-1])
    im_part = np.interp(freq_dst, freq_src, s_src.imag, left=s_src.imag[0], right=s_src.imag[-1])
    return re_part + 1j * im_part


def smooth_random_curve(rng: np.random.RandomState, n: int, scale: float, knots: int = 24) -> np.ndarray:
    x = np.linspace(0.0, 1.0, n)
    xp = np.linspace(0.0, 1.0, knots)
    yp = rng.normal(0.0, scale, size=knots)
    return np.interp(x, xp, yp)


def load_calibration_template(path: Path | None, freq_hz: np.ndarray) -> np.ndarray | None:
    if path is None:
        return None
    try:
        f_cal, s_cal = read_s11_csv_compatible(path)
    except Exception:
        return None
    return interpolate_s11(f_cal, s_cal, freq_hz)


def load_measured_template(path: Path | None, freq_hz: np.ndarray) -> np.ndarray | None:
    if path is None:
        return None
    try:
        f_real, s_real = read_s11_csv_compatible(path)
    except Exception:
        return None
    return interpolate_s11(f_real, s_real, freq_hz)


def smooth_array(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values)
    if window <= 3 or len(arr) < window:
        return arr.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones(window, dtype=np.float64) / float(window)
    if np.iscomplexobj(arr):
        re = np.convolve(np.pad(arr.real, pad, mode="edge"), kernel, mode="valid")
        im = np.convolve(np.pad(arr.imag, pad, mode="edge"), kernel, mode="valid")
        return re + 1j * im
    return np.convolve(np.pad(arr, pad, mode="edge"), kernel, mode="valid")


def bandwise_real_limits(template_s11: np.ndarray | None, profile: str) -> list[tuple[float, float, float]]:
    if template_s11 is None:
        if profile == "field":
            return [(0.0, 200e6, 0.85), (200e6, 400e6, 0.70), (400e6, 1e9, 0.55)]
        return [(0.0, 200e6, 0.42), (200e6, 400e6, 0.11), (400e6, 1e9, 0.12)]
    return []


def constrain_fast_real_energy(
    out: np.ndarray,
    freq_hz: np.ndarray,
    profile: str,
    rng: np.random.RandomState,
    params: DirtyParams,
    template_s11: np.ndarray | None,
) -> np.ndarray:
    """
    Limit fast oscillatory energy by measured band percentiles.

    This targets the failure mode where synthetic high-frequency ripples become
    stronger than the real data after 200-400 MHz.
    """
    n = len(out)
    slow_window = max(51, int(n / (120 if profile == "field" else 90)))
    slow = smooth_array(out, slow_window)
    fast = out - slow

    if template_s11 is not None:
        bands = [(0.0, 200e6), (200e6, 400e6), (400e6, 1e9)]
        limits = []
        for lo, hi in bands:
            mask = (freq_hz >= lo) & (freq_hz <= min(hi, freq_hz[-1]))
            if int(mask.sum()) < 20:
                continue
            tpl_p95 = float(np.nanpercentile(np.abs(template_s11.real[mask]), 95))
            floor = 0.22 if profile == "field" else 0.060
            ceiling = 0.95 if profile == "field" else 0.78
            hi_factor = rng.uniform(0.85, 1.12) if profile == "field" else rng.uniform(1.00, 1.28)
            limits.append((lo, hi, float(np.clip(tpl_p95 * hi_factor, floor, ceiling))))
    else:
        limits = bandwise_real_limits(None, profile)

    for lo, hi, limit in limits:
        mask = (freq_hz >= lo) & (freq_hz <= min(hi, freq_hz[-1]))
        if int(mask.sum()) < 20:
            continue
        current = float(np.nanpercentile(np.abs((slow + fast).real[mask]), 95))
        if current <= limit or current <= 1e-12:
            continue
        scale = max(0.12, min(1.0, (limit / current) ** params.event_hf_damping))
        fast[mask] *= scale
    return slow + fast


def apply_measured_template_shape(
    out: np.ndarray,
    freq_hz: np.ndarray,
    rng: np.random.RandomState,
    params: DirtyParams,
    template_s11: np.ndarray | None,
) -> np.ndarray:
    """Transfer only non-coherent nuisance statistics from a real trace.

    V1 added the template's complex residual directly.  Any unknown joint or
    defect in that trace therefore became an unlabeled synthetic peak.  V2
    uses only a smooth magnitude envelope and a residual-noise PSD, then draws
    a new independent complex realization.  Template choice can alter cable
    family/measurement texture, but cannot copy a reflection event.
    """
    if template_s11 is None or (params.template_mix_scale <= 0 and params.template_slow_scale <= 0):
        return out
    n = len(out)
    if n < 64:
        return out

    template = np.asarray(template_s11, dtype=np.complex128)
    mag = np.maximum(np.abs(template), 1e-8)
    log_mag = np.log(mag)
    envelope_window = max(301, int(n / 28))
    if envelope_window % 2 == 0:
        envelope_window += 1
    log_envelope = smooth_array(log_mag, envelope_window)
    log_envelope -= float(np.nanmedian(log_envelope))
    log_envelope = np.clip(log_envelope, -0.85, 0.85)

    # Magnitude-only transfer preserves all event phases/positions.
    shape_strength = (0.18 if is_rg58_profile(params.profile) else 0.12) * params.template_slow_scale
    shaped = out * np.exp(shape_strength * log_envelope)

    if params.template_mix_scale <= 0:
        return shaped

    complex_slow_window = max(101, int(n / 95))
    if complex_slow_window % 2 == 0:
        complex_slow_window += 1
    residual = template - smooth_array(template, complex_slow_window)
    power_window = max(81, int(n / 140))
    if power_window % 2 == 0:
        power_window += 1
    residual_rms = np.sqrt(np.maximum(smooth_array(np.abs(residual) ** 2, power_window), 0.0))
    normalizer = max(float(np.nanpercentile(residual_rms, 90)), 1e-9)
    residual_rms = np.clip(residual_rms / normalizer, 0.15, 2.2)

    # A fresh, mildly frequency-correlated realization: no template phase is
    # reused, hence no template-specific distance peak can survive.
    white = rng.normal(size=n) + 1j * rng.normal(size=n)
    corr_len = 9 if is_rg58_profile(params.profile) else 17
    x = np.arange(-corr_len, corr_len + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * np.square(x / max(corr_len * 0.38, 1.0)))
    kernel /= np.sqrt(np.sum(kernel ** 2))
    colored = np.convolve(white.real, kernel, mode="same") + 1j * np.convolve(white.imag, kernel, mode="same")
    colored /= max(float(np.sqrt(np.mean(np.abs(colored) ** 2))), 1e-9)
    f_norm = np.clip(freq_hz / max(float(freq_hz[-1]), 1.0), 0.0, 1.0)
    taper = 0.35 + 0.65 * np.exp(-params.highfreq_decay_strength * np.power(f_norm, 0.8))
    noise_amp = params.template_mix_scale * (0.16 if is_rg58_profile(params.profile) else 0.11)
    return shaped + noise_amp * residual_rms * taper * colored


def apply_rg58_measured_energy_floor(
    out: np.ndarray,
    freq_hz: np.ndarray,
    template_s11: np.ndarray | None,
) -> np.ndarray:
    """Recover RG58 broadband energy without importing fast residual phase."""
    if template_s11 is None:
        return out
    adjusted = out.copy()
    bands = [(0.0, 200e6, 0.62), (200e6, 400e6, 0.58), (400e6, 1e9, 0.52)]
    for lo, hi, target_fraction in bands:
        mask = (freq_hz >= lo) & (freq_hz <= min(hi, freq_hz[-1]))
        if int(mask.sum()) < 20:
            continue
        target = float(np.nanpercentile(np.abs(template_s11[mask]), 75)) * target_fraction
        current = float(np.nanpercentile(np.abs(adjusted[mask]), 75))
        if not np.isfinite(target) or not np.isfinite(current) or current <= 1e-10 or current >= target:
            continue
        gain = float(np.clip(target / current, 1.0, 5.5))
        adjusted[mask] *= gain
    return adjusted


def apply_length_dependent_hf_loss(
    out: np.ndarray,
    freq_hz: np.ndarray,
    cable: CableSample,
    params: DirtyParams,
) -> np.ndarray:
    f_norm = np.clip((freq_hz - freq_hz[0]) / max(freq_hz[-1] - freq_hz[0], 1.0), 0.0, 1.0)
    length_factor = (max(cable.total_length, 20.0) / 1000.0) ** (0.55 if params.profile == "field" else 0.35)
    strength = params.highfreq_decay_strength
    floor = 0.18
    if params.profile == "field" and cable.total_length < 500.0:
        strength *= 0.30
        floor = 0.62
    elif is_rg58_profile(params.profile):
        strength *= 0.28
        floor = 0.82
    loss = np.exp(-strength * length_factor * np.power(f_norm, 0.82))
    return out * (floor + (1.0 - floor) * loss)


def dirty_params_for_profile(profile: str, rng: np.random.RandomState) -> DirtyParams:
    if is_rg58_profile(profile):
        return DirtyParams(
            profile=profile,
            additive_scale=float(rng.uniform(0.4, 0.9)),
            multiplicative_scale=float(rng.uniform(0.3, 0.8)),
            ripple_scale=float(rng.uniform(0.006, 0.018)),
            phase_scale_rad=float(rng.uniform(0.008, 0.035)),
            fixture_scale=float(rng.uniform(0.020, 0.055)),
            template_slow_scale=float(rng.uniform(0.035, 0.100)),
            template_mix_scale=float(rng.uniform(0.006, 0.018)),
            highfreq_decay_strength=float(rng.uniform(0.055, 0.155)),
            event_hf_damping=float(rng.uniform(0.55, 0.75)),
        )
    return DirtyParams(
        profile=profile,
        additive_scale=float(rng.uniform(1.0, 2.2)),
        multiplicative_scale=float(rng.uniform(0.8, 1.8)),
        ripple_scale=float(rng.uniform(0.012, 0.055)),
        phase_scale_rad=float(rng.uniform(0.030, 0.120)),
        fixture_scale=float(rng.uniform(0.045, 0.135)),
        template_slow_scale=float(rng.uniform(0.160, 0.340)),
        template_mix_scale=float(rng.uniform(0.040, 0.130)),
        highfreq_decay_strength=float(rng.uniform(0.10, 0.30)),
        event_hf_damping=float(rng.uniform(0.55, 0.78)),
    )


def effective_terminal_phase_length_m(cable: CableSample) -> float:
    """Approximate terminal electrical length under long slow-velocity defects."""
    base_eps = max(float(cable.epsr), 1.1)
    effective = float(cable.total_length)
    for defect in cable.defect_info:
        defect_type = str(defect.get("type", "short"))
        if defect_type not in {"aging", "moisture_distributed"}:
            continue
        length = float(defect.get("length", 0.0))
        eps = max(float(defect.get("epsr", base_eps)), 1.1)
        if length <= 0.0 or eps <= base_eps:
            continue
        effective += length * (math.sqrt(eps / base_eps) - 1.0)
    return effective


def causal_lowpass(freq_hz: np.ndarray, corner_hz: float, order: float = 2.0) -> np.ndarray:
    """Stable minimum-phase low-pass evaluated on the positive frequency axis."""
    corner = max(float(corner_hz), 1.0)
    return np.power(1.0 + 1j * np.asarray(freq_hz, dtype=np.float64) / corner, -float(order))


def causal_resonant_mode(freq_hz: np.ndarray, resonance_hz: float, q_factor: float) -> np.ndarray:
    """Normalized causal second-order mode with a decaying sinusoidal impulse."""
    f0 = max(float(resonance_hz), 1.0)
    q = max(float(q_factor), 0.55)
    x = np.asarray(freq_hz, dtype=np.float64) / f0
    response = 1.0 / (1.0 - x * x + 1j * x / q)
    peak = max(float(np.nanmax(np.abs(response))), 1.0)
    return response / peak


def calibration_texture_factor(calibration_template: np.ndarray | None) -> float:
    """Use calibration only as a scalar fixture-complexity estimate."""
    if calibration_template is None or len(calibration_template) < 64:
        return 1.0
    arr = np.asarray(calibration_template, dtype=np.complex128)
    slow = smooth_array(arr, max(31, len(arr) // 180))
    rough = float(np.nanpercentile(np.abs(arr - slow), 90))
    level = max(float(np.nanpercentile(np.abs(arr), 75)), 1e-6)
    return float(np.clip(0.82 + 2.2 * rough / level, 0.78, 1.35))


def localized_time_kernel_to_s11(
    freq_hz: np.ndarray,
    target_impulse: np.ndarray,
    window: str = "hann",
) -> np.ndarray:
    """Approximately invert the shared Client IFFT for an additive kernel.

    The regularized window inversion is used only to synthesize a nuisance
    component; the final stored response is always recomputed from the summed
    S11.  Thus exact CSV/IFFT consistency is retained.
    """
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    df = estimate_first_step(freq_hz)
    steps = int(np.floor(float(freq_hz[-1]) / df))
    n = 2 * steps + 1
    if len(target_impulse) != n:
        raise ValueError("target_impulse length does not match Client IFFT grid")
    fft_order = np.fft.fft(np.asarray(target_impulse, dtype=np.complex128))
    centered_windowed = fft_shift(fft_order, inverse=False)
    win = apply_window(n, window_type=window)
    # Hann endpoints are exactly zero.  A floor avoids an unstable inverse;
    # the final IFFT naturally bandlimits the requested near-end texture.
    unwindowed = centered_windowed / np.maximum(win, 0.075)
    positive = unwindowed[steps:]
    f_lin = np.arange(steps + 1, dtype=np.float64) * df
    real = np.interp(freq_hz, f_lin, positive.real)
    imag = np.interp(freq_hz, f_lin, positive.imag)
    return real + 1j * imag


def build_rg58_near_end_texture_s11(
    freq_hz: np.ndarray,
    cable: CableSample,
    rng: np.random.RandomState,
    params: DirtyParams,
    window: str = "hann",
) -> np.ndarray:
    """Generate the dense, decaying RG58 port texture seen in real IFFTs."""
    df = estimate_first_step(freq_hz)
    steps = int(np.floor(float(freq_hz[-1]) / df))
    n = 2 * steps + 1
    dt = 1.0 / (df * n)
    v = 299_792_458.0 / math.sqrt(max(float(cable.epsr), 1.1))
    distance = np.arange(n, dtype=np.float64) * dt * v / 2.0
    extent = min(42.0, max(12.0, float(cable.total_length) * 0.58))
    mask = distance <= extent
    target = np.zeros(n, dtype=np.float64)
    count = int(mask.sum())
    if count < 8:
        return np.zeros_like(freq_hz, dtype=np.complex128)

    raw = rng.normal(size=count)
    # Mix sample-scale and 2-4 sample correlated components to reproduce the
    # dense high-frequency cluster rather than isolated synthetic reflectors.
    kx = np.arange(-3, 4, dtype=np.float64)
    kernel = np.exp(-0.5 * np.square(kx / rng.uniform(0.8, 1.4)))
    kernel /= np.sqrt(np.sum(kernel ** 2))
    corr = np.convolve(rng.normal(size=count), kernel, mode="same")
    texture = 0.72 * raw + 0.28 * corr
    texture /= max(float(np.std(texture)), 1e-9)
    decay_m = rng.uniform(8.0, 18.0)
    envelope = np.exp(-distance[mask] / decay_m)
    # A few larger port transients occur only in the first metres.
    early = distance[mask] < rng.uniform(1.5, 3.5)
    texture[early] *= rng.uniform(1.3, 2.2)
    target_rms = params.fixture_scale * rng.uniform(0.028, 0.060)
    target[mask] = target_rms * envelope * texture
    target[0] += rng.choice([-1.0, 1.0]) * params.fixture_scale * rng.uniform(0.10, 0.24)

    correction = localized_time_kernel_to_s11(freq_hz, target, window=window)
    # Regularized inversion can overshoot in S11; scale by the actual produced
    # near-end percentile, not by an assumed FFT normalization.
    try:
        d_test, h_test, _, _ = s11_to_responses(freq_hz, correction, epsr=cable.epsr, window=window)
        produced_mask = d_test <= extent
        produced = float(np.nanpercentile(np.abs(h_test.real[produced_mask]), 90))
        desired = float(np.nanpercentile(np.abs(target[mask]), 90))
        if produced > 1e-12 and desired > 0:
            correction *= float(np.clip(desired / produced, 0.15, 4.0))
    except Exception:
        pass
    return correction


def apply_front_end_model(
    out: np.ndarray,
    freq_hz: np.ndarray,
    cable: CableSample,
    rng: np.random.RandomState,
    params: DirtyParams,
    calibration_template: np.ndarray | None,
    window: str = "hann",
) -> np.ndarray:
    """Add a causal fixture/clamp response confined to the near end.

    The V1 model used a handful of delayed delta reflectors, hence sparse peaks.
    Here the direct fixture is a sum of damped resonant modes and minimum-phase
    echoes.  RG58 receives higher-frequency, higher-Q modes; field clamps keep
    the broader low-frequency front-end behavior already seen in measurements.
    """
    omega = 2.0 * math.pi * freq_hz
    v = 299_792_458.0 / math.sqrt(max(float(cable.epsr), 1.1))
    factor = calibration_texture_factor(calibration_template)
    result = np.asarray(out, dtype=np.complex128).copy()

    if is_rg58_profile(params.profile):
        result += build_rg58_near_end_texture_s11(freq_hz, cable, rng, params, window=window)
        direct_amp = params.fixture_scale * factor * rng.uniform(0.45, 0.90)
        direct_delay = rng.uniform(0.02, 0.22)
        result += rng.choice([-1.0, 1.0]) * direct_amp * causal_lowpass(
            freq_hz, rng.uniform(650e6, 1.45e9), rng.uniform(0.75, 1.35)
        ) * np.exp(-2j * omega * direct_delay / v)

        n_modes = int(rng.randint(3, 6))
        for idx in range(n_modes):
            f0 = rng.uniform(95e6, 620e6)
            q = rng.uniform(7.0, 32.0)
            delay = rng.uniform(0.05, 1.15)
            amp = params.fixture_scale * factor * rng.uniform(0.18, 0.48) / (idx + 1) ** 0.18
            sign = -1.0 if idx % 2 else 1.0
            result += sign * amp * causal_resonant_mode(freq_hz, f0, q) * np.exp(-2j * omega * delay / v)

        # Weak connector reverberations; dense near-end texture without isolated
        # high-amplitude reflectors at arbitrary distances.
        spacing = rng.uniform(0.10, 0.24)
        rho = rng.uniform(0.92, 0.975)
        echo_amp = params.fixture_scale * factor * rng.uniform(0.025, 0.060)
        lp = causal_lowpass(freq_hz, rng.uniform(720e6, 1.60e9), rng.uniform(0.45, 0.95))
        for idx in range(1, int(rng.randint(28, 65))):
            result += rng.choice([-1.0, 1.0]) * echo_amp * (rho ** idx) * lp * np.exp(-2j * omega * spacing * idx / v)
    else:
        short_field = cable.total_length < 500.0
        direct_amp = params.fixture_scale * factor * (0.35 if short_field else 0.62) * rng.uniform(0.65, 1.15)
        result += rng.choice([-1.0, 1.0]) * direct_amp * causal_lowpass(
            freq_hz,
            rng.uniform(150e6, 420e6) if short_field else rng.uniform(70e6, 230e6),
            rng.uniform(1.25, 2.15),
        ) * np.exp(-2j * omega * rng.uniform(0.15, 2.8) / v)

        n_modes = int(rng.randint(2, 4) if short_field else rng.randint(3, 6))
        for idx in range(n_modes):
            f0 = rng.uniform(35e6, 220e6) if short_field else rng.uniform(12e6, 145e6)
            q = rng.uniform(1.4, 4.5) if short_field else rng.uniform(1.8, 7.0)
            delay = rng.uniform(0.4, 8.0) if short_field else rng.uniform(0.8, 22.0)
            amp = params.fixture_scale * factor * rng.uniform(0.08, 0.30) / (idx + 1) ** 0.25
            result += rng.choice([-1.0, 1.0]) * amp * causal_resonant_mode(freq_hz, f0, q) * np.exp(-2j * omega * delay / v)

        # Clamp lead echoes are low-pass and causal, so they stay at the front
        # rather than creating symmetric ringing around every remote event.
        echo_count = int(rng.randint(3, 7))
        spacing = rng.uniform(1.2, 5.5)
        rho = rng.uniform(0.48, 0.72)
        lp = causal_lowpass(freq_hz, rng.uniform(45e6, 160e6), rng.uniform(1.6, 2.8))
        echo_amp = params.fixture_scale * factor * rng.uniform(0.06, 0.15)
        for idx in range(1, echo_count + 1):
            result += rng.choice([-1.0, 1.0]) * echo_amp * (rho ** idx) * lp * np.exp(-2j * omega * spacing * idx / v)
    return result


def _step_delta_near(distance: np.ndarray, step: np.ndarray, center_m: float, width_m: float) -> float | None:
    pre = (distance > center_m - 2.5 * width_m) & (distance < center_m - 0.65 * width_m)
    post = (distance > center_m + 0.45 * width_m) & (distance < center_m + 2.4 * width_m)
    if not pre.any() or not post.any():
        return None
    return float(np.nanmedian(step[post]) - np.nanmedian(step[pre]))


def ensure_terminal_visibility_s11(
    out: np.ndarray,
    freq_hz: np.ndarray,
    cable: CableSample,
    profile: str,
    rng: np.random.RandomState,
    window: str = "hann",
) -> np.ndarray:
    """Enforce a measured-like terminal step by adding a causal S11 kernel.

    This is an S11-domain correction.  The exported impulse and step responses
    are still obtained exclusively by the shared Client IFFT and therefore are
    exactly reproducible from the CSV S11 columns.
    """
    if len(freq_hz) < 128:
        return out
    termination = str(getattr(cable, "termination", "open"))
    sign = -1.0 if termination == "short" else 1.0
    terminal_m = effective_terminal_phase_length_m(cable)
    length = float(cable.total_length)

    if is_rg58_profile(profile):
        corner = rng.uniform(260e6, 720e6)
        order = rng.uniform(1.25, 2.15)
        target_abs = 2.0e-10 if termination != "short" else 1.6e-10
        width = 12.0
    else:
        if length < 500.0:
            corner = rng.uniform(35e6, 105e6)
            target_abs = 1.55e-10
            width = 34.0
        elif length < 1500.0:
            corner = rng.uniform(13e6, 42e6)
            target_abs = 1.20e-10
            width = 70.0
        else:
            corner = rng.uniform(7e6, 26e6)
            target_abs = 8.5e-11
            width = 100.0
        order = rng.uniform(2.0, 3.4)
        if termination == "weak_open":
            target_abs *= 0.48
        elif termination == "short":
            target_abs *= 0.90

    omega = 2.0 * math.pi * freq_hz
    v = 299_792_458.0 / math.sqrt(max(float(cable.epsr), 1.1))
    unit_kernel = sign * causal_lowpass(freq_hz, corner, order) * np.exp(-2j * omega * terminal_m / v)

    try:
        distance, _, step, _ = s11_to_responses(freq_hz, out, epsr=cable.epsr, window=window)
        _, _, unit_step, _ = s11_to_responses(freq_hz, unit_kernel, epsr=cable.epsr, window=window)
    except Exception:
        return out
    current_delta = _step_delta_near(distance, step, terminal_m, width)
    unit_delta = _step_delta_near(distance, unit_step, terminal_m, width)
    if current_delta is None or unit_delta is None:
        return out
    current_signed = sign * current_delta
    unit_signed = sign * unit_delta
    if not np.isfinite(current_signed) or not np.isfinite(unit_signed) or unit_signed <= 1e-16:
        return out
    needed = target_abs - current_signed
    if needed <= 0:
        return out
    coefficient = float(np.clip(needed / unit_signed, 0.0, 0.55 if is_rg58_profile(profile) else 0.42))
    return out + coefficient * unit_kernel


def apply_dirty_model(
    s11: np.ndarray,
    freq_hz: np.ndarray,
    cable: CableSample,
    rng: np.random.RandomState,
    params: DirtyParams,
    calibration_template: np.ndarray | None = None,
    measured_template: np.ndarray | None = None,
    window: str = "hann",
) -> np.ndarray:
    out = s11.astype(np.complex128).copy()

    amp_ripple = smooth_random_curve(rng, len(freq_hz), params.ripple_scale)
    phase_ripple = smooth_random_curve(rng, len(freq_hz), params.phase_scale_rad)
    low_freq_bias = 1.0 + (0.018 if params.profile == "field" else 0.004) * np.exp(-freq_hz / 45e6)
    out *= low_freq_bias * (1.0 + amp_ripple) * np.exp(1j * phase_ripple)
    out = apply_length_dependent_hf_loss(out, freq_hz, cable, params)

    omega = 2.0 * math.pi * freq_hz
    v = 299_792_458.0 / math.sqrt(max(cable.epsr, 1.1))

    # Physical/declared joints only.  Minimum-phase envelopes yield a causal
    # one-sided event rather than the symmetric high-frequency burst in V1.
    for pos in getattr(cable, "joint_positions", []):
        attenuation = math.exp(-float(pos) / max(cable.total_length * 0.85, 80.0))
        if is_rg58_profile(params.profile):
            gamma = rng.uniform(0.003, 0.016) * attenuation
            corner = rng.uniform(420e6, 1.10e9)
            order = rng.uniform(1.2, 2.0)
        else:
            gamma = rng.uniform(0.0025, 0.013) * attenuation
            corner = rng.uniform(55e6, 240e6)
            order = rng.uniform(1.8, 3.0)
        gamma *= rng.choice([-1.0, 1.0])
        out += gamma * np.exp(-2j * omega * float(pos) / v) * causal_lowpass(freq_hz, corner, order)

    out = apply_front_end_model(out, freq_hz, cable, rng, params, calibration_template, window=window)
    out = apply_measured_template_shape(out, freq_hz, rng, params, measured_template)
    if is_rg58_profile(params.profile):
        out = apply_rg58_measured_energy_floor(out, freq_hz, measured_template)
    out = constrain_fast_real_energy(out, freq_hz, params.profile, rng, params, measured_template)
    out = ensure_terminal_visibility_s11(out, freq_hz, cable, params.profile, rng, window=window)

    sigma_add = 2.5e-4 * (0.55 + 1.85 * np.exp(-freq_hz / 120e6)) * params.additive_scale
    noise_taper = 0.75 + 0.25 * np.exp(-freq_hz / 400e6)
    if params.profile == "field":
        noise_taper = 0.25 + 0.75 * np.exp(-freq_hz / 260e6)
    sigma_mult = rng.uniform(0.003, 0.018) * params.multiplicative_scale * np.abs(out) * noise_taper
    sigma = np.sqrt(sigma_add ** 2 + sigma_mult ** 2)
    out += rng.normal(0.0, sigma) + 1j * rng.normal(0.0, sigma)

    mag = np.abs(out)
    too_large = mag > 1.25
    out[too_large] *= 1.25 / mag[too_large]
    return out


def params_for_sweep(params: DirtyParams, sweep: SweepConfig) -> DirtyParams:
    """Use lower dirty strength for the 200 MHz long-range sweep."""
    if sweep.stop_hz > 250e6:
        return params
    if params.profile == "field":
        return replace(
            params,
            additive_scale=params.additive_scale * 0.65,
            multiplicative_scale=params.multiplicative_scale * 0.55,
            ripple_scale=params.ripple_scale * 0.45,
            phase_scale_rad=params.phase_scale_rad * 0.45,
            fixture_scale=params.fixture_scale * 0.12,
            template_slow_scale=params.template_slow_scale * 0.75,
            template_mix_scale=params.template_mix_scale * 0.22,
            highfreq_decay_strength=params.highfreq_decay_strength * 0.45,
            event_hf_damping=min(params.event_hf_damping + 0.14, 0.92),
        )
    return replace(
        params,
        additive_scale=params.additive_scale * 0.75,
        multiplicative_scale=params.multiplicative_scale * 0.65,
        fixture_scale=params.fixture_scale * 0.50,
        template_slow_scale=params.template_slow_scale * 0.85,
        template_mix_scale=params.template_mix_scale * 0.70,
        highfreq_decay_strength=params.highfreq_decay_strength * 0.70,
        event_hf_damping=min(params.event_hf_damping + 0.05, 0.85),
    )


def client_csv_coverage(distance: np.ndarray, cable_length: float) -> dict:
    cutoff = max(5.0, cable_length * 1.2)
    keep = np.where(distance <= cutoff)[0]
    n_keep = int(keep[-1] + 1) if len(keep) else min(len(distance), 128)
    d_out = distance[:n_keep]
    max_saved_distance = float(d_out[-1]) if len(d_out) else 0.0
    target_distance = float(cable_length * 1.2)
    distance_step = float(np.median(np.diff(distance[:min(len(distance), 32)]))) if len(distance) > 2 else 0.0
    return {
        "distance_rows": int(len(d_out)),
        "max_distance_m": max_saved_distance,
        "target_distance_m": target_distance,
        "distance_step_m": distance_step,
        "truncated_by_ifft_range": bool(max_saved_distance + max(distance_step, 1e-9) < target_distance),
    }


def save_client_csv(
    path: Path,
    freq_hz: np.ndarray,
    s11: np.ndarray,
    distance: np.ndarray,
    impulse: np.ndarray,
    step: np.ndarray,
    cable_length: float,
) -> dict:
    coverage = client_csv_coverage(distance, cable_length)
    cutoff = max(5.0, cable_length * 1.2)
    keep = np.where(distance <= cutoff)[0]
    n_keep = int(keep[-1] + 1) if len(keep) else min(len(distance), 128)
    d_out = distance[:n_keep]
    imp_out = np.real(impulse[:n_keep])
    step_out = step[:n_keep]

    path.parent.mkdir(parents=True, exist_ok=True)
    max_rows = max(len(freq_hz), len(d_out))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for i in range(max_rows):
            if i < len(freq_hz):
                row = [f"{freq_hz[i]:.17g}", f"{s11[i].real:.17g}", f"{s11[i].imag:.17g}"]
            else:
                row = ["", "", ""]
            if i < len(d_out):
                row.extend([f"{d_out[i]:.17g}", f"{imp_out[i]:.17g}", f"{step_out[i]:.17g}"])
            else:
                row.extend(["", "", ""])
            writer.writerow(row)
    return {"rows": int(max_rows), **coverage}


def sample_metadata(
    sample_id: str,
    split: str,
    profile: str,
    cable: CableSample,
    seed: int,
    dirty_params: DirtyParams,
    band_coverage: dict,
) -> dict:
    defects = cable.defect_info
    return {
        "sample_id": sample_id,
        "profile": profile,
        "csv_1ghz": f"{sample_id}_1GHz.csv",
        "csv_200mhz": f"{sample_id}_200MHz.csv",
        "total_length_m": float(round(cable.total_length, 4)),
        "epsr": float(cable.epsr),
        "seed": int(seed),
        "split": split,
        "sweep_1ghz": {"start_hz": 9e3, "stop_hz": 1e9, "n_points": 50000},
        "sweep_200mhz": {"start_hz": 9e3, "stop_hz": 200e6, "n_points": 5000},
        "label_grid": {"d_max_m": D_MAX, "dd_m": DD, "n_points": N_GRID},
        "n_segments": len(cable.segments),
        "defects": [
            {
                "type": str(d.get("type", "short")),
                "start_m": float(round(d.get("start", d["position"] - d["length"] / 2.0), 4)),
                "end_m": float(round(d.get("end", d["position"] + d["length"] / 2.0), 4)),
                "center_m": float(round(d["position"], 4)),
                "position_m": float(round(d["position"], 4)),
                "length_m": float(round(d["length"], 4)),
                "z0_ohm": float(round(d["z0"], 4)),
                "epsr": float(round(d["epsr"], 4)),
                "alpha_db_per_m_100mhz": float(round(d.get("alpha", 0.0), 6)),
                "severity": float(round(d["severity"], 4)),
            }
            for d in defects
        ],
        "joint_positions_m": [float(round(p, 4)) for p in getattr(cable, "joint_positions", [])],
        "end_position_m": float(round(cable.total_length, 4)),
        "termination": getattr(cable, "termination", "open"),
        "z_load_ohm": float(getattr(cable, "z_load_open", 1e13)),
        "defect_count_policy": getattr(cable, "defect_count_policy", defect_count_policy(profile, cable.total_length)),
        "defect_type_policy": getattr(cable, "defect_type_policy", defect_type_policy(profile, cable.total_length)),
        "dirty_params": dirty_params.__dict__,
        "band_distance_coverage": band_coverage,
    }


def generate_band(
    cable: CableSample,
    sweep: SweepConfig,
    rng: np.random.RandomState,
    profile: str,
    dirty_params: DirtyParams,
    calibration_path: Path | None,
    measured_template_path: Path | None = None,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    freq_hz, clean = generate_s11(cable, sweep, rng=rng, add_noise=False, inject_joints=False)
    cal_template = load_calibration_template(calibration_path, freq_hz)
    measured_template = load_measured_template(measured_template_path, freq_hz)
    band_params = params_for_sweep(dirty_params, sweep)
    dirty = apply_dirty_model(
        clean, freq_hz, cable, rng, band_params, cal_template, measured_template, window=window
    )
    # V2 invariant: these are the only response calculations.  No distance-
    # domain edits are permitted after this point.
    distance, impulse, step, _ = s11_to_responses(
        freq_hz, dirty, epsr=cable.epsr, window=window
    )
    return freq_hz, dirty, distance, impulse, step


def generate_dual_bands(
    cable: CableSample,
    rng: np.random.RandomState,
    profile: str,
    dirty_params: DirtyParams,
    calibration_path: Path | None,
    measured_template_path: Path | None = None,
    window: str = "hann",
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """Generate both sweeps from one S11 realization with exact consistency."""
    f1, s1, d1, i1, st1 = generate_band(
        cable, SWEEP_1GHZ, rng, profile, dirty_params, calibration_path,
        measured_template_path, window,
    )
    f2 = SWEEP_200MHZ.frequencies()
    s2 = interpolate_s11(f1, s1, f2)
    d2, i2, st2, _ = s11_to_responses(f2, s2, epsr=cable.epsr, window=window)
    return (f1, s1, d1, i1, st1), (f2, s2, d2, i2, st2)


def _normalize_window_name(window: str | None) -> str:
    text = str(window or "hann").strip().lower()
    aliases = {
        "hann": "hann",
        "hanning": "hann",
        "汉宁": "hann",
        "hamming": "hamming",
        "blackman": "blackman",
        "rectangular": "rectangular",
        "none": "rectangular",
        "rect": "rectangular",
    }
    return aliases.get(text, "hann")


def _normalize_band_name(band: str | None) -> str:
    text = str(band or "1GHz").strip().lower().replace(" ", "")
    if text in {"200m", "200mhz", "0.2ghz"}:
        return "200MHz"
    return "1GHz"


def template_length_hint(path: Path) -> float | None:
    """Infer a physical length from a template path without reading the CSV."""
    try:
        value = infer_length_from_path(path, default_m=float("nan"), clip=False)
    except Exception:
        return None
    return float(value) if np.isfinite(value) and value > 0.0 else None


def weighted_template_choice(
    paths: list[Path],
    profile: str,
    rng: np.random.RandomState,
    minimum_score: float = 0.38,
    target_length_m: float | None = None,
) -> Path | None:
    """Choose a good template, preferring a similar physical cable length.

    V1 treated all field traces as interchangeable.  Even after coherent phase
    transfer is removed, a 100 m and a 2400 m trace have very different
    broadband attenuation statistics.  V2 therefore performs a cheap
    path-name length preselection before reading candidate CSVs.
    """
    if not paths:
        return None

    candidates = list(paths)
    target = float(target_length_m) if target_length_m is not None else float("nan")
    length_by_path: dict[Path, float | None] = {path: template_length_hint(path) for path in candidates}
    if np.isfinite(target) and target > 0.0:
        known = [(path, length) for path, length in length_by_path.items() if length is not None]
        if len(known) >= 5:
            known.sort(key=lambda item: abs(math.log(max(item[1], 1.0) / target)))
            # A moderate shortlist avoids both gross length mismatch and
            # overfitting to one single trace.
            candidates = [path for path, _ in known[: min(48, max(12, len(known) // 3))]]

    # Inspect only a few candidates.  Reading every 50k-point CSV in every
    # multiprocessing worker would dominate generation time.
    n_try = min(len(candidates), 9)
    indices = (
        rng.choice(len(candidates), size=n_try, replace=False)
        if len(candidates) > n_try
        else np.arange(len(candidates))
    )
    accepted: list[tuple[Path, float, float]] = []
    sigma_log = 0.34 if is_rg58_profile(profile) else 0.62
    for idx in indices:
        path = candidates[int(idx)]
        score = assess_template_quality(path, profile)
        if score < minimum_score:
            continue
        length = length_by_path.get(path)
        if np.isfinite(target) and target > 0.0 and length is not None:
            log_distance = math.log(max(length, 1.0) / target)
            length_weight = math.exp(-0.5 * (log_distance / sigma_log) ** 2)
        else:
            length_weight = 0.55
        accepted.append((path, score, length_weight))
        if len(accepted) >= 4:
            break
    if not accepted:
        return None
    weights = np.asarray(
        [score ** 4 * (0.25 + 0.75 * length_weight) for _, score, length_weight in accepted],
        dtype=np.float64,
    )
    weights /= weights.sum()
    return accepted[int(rng.choice(len(accepted), p=weights))][0]


def _choose_template_paths(
    profile: str, real_data_root: Path, rng: np.random.RandomState, target_length_m: float
) -> tuple[Path | None, Path | None]:
    try:
        rg58_paths, field_paths, calibration_paths = discover_real_files(real_data_root)
    except Exception:
        return None, None

    if is_rg58_profile(profile):
        measured = weighted_template_choice(rg58_paths, "rg58", rng, target_length_m=target_length_m)
        return measured, None

    measured = weighted_template_choice(field_paths, "field", rng, target_length_m=target_length_m)
    calibration: Path | None = None
    if calibration_paths:
        if measured is not None:
            same_dir = [p for p in calibration_paths if p.parent == measured.parent]
            calibration = weighted_template_choice(same_dir, "calibration", rng, minimum_score=0.25)
        if calibration is None:
            calibration = weighted_template_choice(calibration_paths, "calibration", rng, minimum_score=0.25)
    return measured, calibration


def _interactive_sample_seed(seed_value: object | None) -> int:
    if seed_value is None or str(seed_value).strip() == "":
        return int(np.random.randint(0, 2**31 - 1))
    return int(seed_value)


def generate_interactive_sample(config: dict) -> dict:
    """
    Generate one GUI-oriented sample while reusing the DirtyGenerator pipeline.

    This API is intentionally thin: it prepares GUI defaults/overrides and then
    calls the same cable, dirty, dual-band, metadata, and CSV helper functions
    used by the dataset generator.
    """
    warnings: list[str] = []
    requested_profile = str(config.get("profile", "field")).strip().lower()
    if requested_profile in {"rg58", "rg58_random"}:
        profile = "rg58_random"
    elif requested_profile == "field":
        profile = "field"
    else:
        profile = choose_profile("mixed", np.random.RandomState(_interactive_sample_seed(config.get("seed"))))
        warnings.append(f"Unknown profile '{requested_profile}', sampled mixed profile as {profile}.")

    seed = _interactive_sample_seed(config.get("seed"))
    rng = np.random.RandomState(seed)
    length_value = config.get("length_m")
    length_used_default = length_value is None or str(length_value).strip() == ""
    total_length = None if length_used_default else float(length_value)

    epsr_value = config.get("epsr")
    epsr_used_default = epsr_value is None or str(epsr_value).strip() == ""
    epsr = 2.23 if epsr_used_default else float(epsr_value)

    n_defects_value = config.get("n_defects")
    n_defects_used_default = n_defects_value is None or str(n_defects_value).strip() == ""
    n_defects_override = None if n_defects_used_default else int(n_defects_value)
    if n_defects_override is not None:
        max_count = 5 if profile == "field" else 2
        clipped = int(np.clip(n_defects_override, 0, max_count))
        if clipped != n_defects_override:
            warnings.append(f"Defect count clipped from {n_defects_override} to {clipped} for {profile}.")
        n_defects_override = clipped

    allowed_input = config.get("allowed_defect_types")
    allowed_used_default = allowed_input is None or allowed_input == "" or allowed_input == []
    allowed_types = normalize_allowed_defect_types(profile, allowed_input)
    if (not allowed_used_default) and profile != "field" and any(item != "short" for item in normalize_allowed_defect_types("field", allowed_input)):
        warnings.append("RG58 profile only supports short defects; long defect selections were downgraded to short.")

    if profile == "field":
        cable = make_field_cable(
            rng,
            total_length=total_length,
            epsr=epsr,
            n_defects_override=n_defects_override,
            allowed_defect_types=None if allowed_used_default else allowed_types,
        )
    else:
        cable = make_random_rg58_cable(
            rng,
            total_length=total_length,
            epsr=epsr,
            n_defects_override=n_defects_override,
        )

    dirty_params = dirty_params_for_profile(profile, rng)
    real_data_root = Path(config.get("real_data_root") or r"E:\FDR案例-csv")
    use_templates = bool(config.get("use_templates", True))
    if use_templates:
        measured_template_path, calibration_path = _choose_template_paths(
            profile, real_data_root, rng, cable.total_length
        )
    else:
        measured_template_path, calibration_path = None, None
    window = _normalize_window_name(config.get("window"))
    band_1ghz, band_200mhz = generate_dual_bands(
        cable,
        rng,
        profile,
        dirty_params,
        calibration_path,
        measured_template_path,
        window=window,
    )

    coverage_1ghz = client_csv_coverage(band_1ghz[2], cable.total_length)
    coverage_200mhz = client_csv_coverage(band_200mhz[2], cable.total_length)
    band_coverage = {"1GHz": coverage_1ghz, "200MHz": coverage_200mhz}
    metadata = sample_metadata("dg_gui_sample", "interactive", profile, cable, seed, dirty_params, band_coverage)
    metadata["n_defects"] = len(cable.defect_info)
    metadata["window"] = window
    metadata["requested_profile"] = requested_profile
    metadata["allowed_defect_types"] = allowed_types
    metadata["actual_defect_types"] = [str(d.get("type", "short")) for d in cable.defect_info]
    if calibration_path is not None:
        metadata["calibration_source"] = str(calibration_path)
    if measured_template_path is not None:
        metadata["measured_template_source"] = str(measured_template_path)
        metadata["measured_template_quality"] = assess_template_quality(measured_template_path, profile)
        metadata["measured_template_length_hint_m"] = template_length_hint(measured_template_path)
    metadata["response_consistency"] = "strict_s11_ifft_only"
    metadata["template_mode"] = "auto" if use_templates else "disabled"

    selected_band = _normalize_band_name(config.get("band"))
    selected_tuple = band_200mhz if selected_band == "200MHz" else band_1ghz
    f_sel, s_sel, d_sel, imp_sel, step_sel = selected_tuple
    return {
        "profile": profile,
        "requested_profile": requested_profile,
        "selected_band": selected_band,
        "seed": seed,
        "cable": cable,
        "metadata": metadata,
        "dirty_params": dirty_params,
        "input_defaults": {
            "length_used_default": length_used_default,
            "n_defects_used_default": n_defects_used_default,
            "allowed_types_used_default": allowed_used_default,
            "epsr_used_default": epsr_used_default,
        },
        "warnings": warnings,
        "bands": {
            "1GHz": {"freq_hz": band_1ghz[0], "s11": band_1ghz[1], "distance": band_1ghz[2], "impulse": band_1ghz[3], "step": band_1ghz[4]},
            "200MHz": {"freq_hz": band_200mhz[0], "s11": band_200mhz[1], "distance": band_200mhz[2], "impulse": band_200mhz[3], "step": band_200mhz[4]},
        },
        "band": {"freq_hz": f_sel, "s11": s_sel, "distance": d_sel, "impulse": imp_sel, "step": step_sel},
    }


def build_label(cable: CableSample) -> np.ndarray:
    defects = cable.defect_info
    short_defects = [d for d in defects if d.get("type", "short") == "short"]
    label = build_label_vector(
        [d["position"] for d in short_defects],
        [d["severity"] for d in short_defects],
        cable.total_length,
        LABEL_GRID,
        sigma_defect=0.8,
        sigma_end=1.0,
        joint_positions=getattr(cable, "joint_positions", []),
        sigma_joint=0.45,
        joint_amplitude=0.55,
    )
    for defect in defects:
        defect_type = defect.get("type", "short")
        if defect_type == "moisture":
            defect_type = "moisture_local"
        if defect_type not in {"aging", "moisture_local", "moisture_distributed"}:
            continue
        start_m = float(defect.get("start", defect["position"] - defect["length"] / 2.0))
        end_m = float(defect.get("end", defect["position"] + defect["length"] / 2.0))
        amplitude = float(defect["severity"])
        edge_width = float(np.clip((end_m - start_m) * 0.12, 3.0, 18.0))
        left = 1.0 / (1.0 + np.exp(-(LABEL_GRID - start_m) / edge_width))
        right = 1.0 / (1.0 + np.exp((LABEL_GRID - end_m) / edge_width))
        interval = amplitude * left * right
        label = np.maximum(label, interval)
    return label.astype(np.float32)


def generate_one_sample_job(job: dict) -> dict:
    sample_id = job["sample_id"]
    split = job["split"]
    seed = int(job["seed"])
    output_dir = Path(job["output_dir"])
    requested_profile = job["profile"]
    calibration_paths = [Path(p) for p in job.get("calibration_paths", [])]
    rg58_template_paths = [Path(p) for p in job.get("rg58_template_paths", [])]
    field_template_paths = [Path(p) for p in job.get("field_template_paths", [])]

    rng = np.random.RandomState(seed)
    profile = choose_profile(requested_profile, rng)
    cable = make_cable_for_profile(profile, rng)
    dirty_params = dirty_params_for_profile(profile, rng)
    measured_template_path: Path | None = None
    calibration_path: Path | None = None
    if is_rg58_profile(profile):
        measured_template_path = weighted_template_choice(
            rg58_template_paths, "rg58", rng, target_length_m=cable.total_length
        )
    else:
        measured_template_path = weighted_template_choice(
            field_template_paths, "field", rng, target_length_m=cable.total_length
        )
        if measured_template_path is not None:
            same_dir = [p for p in calibration_paths if p.parent == measured_template_path.parent]
            calibration_path = weighted_template_choice(same_dir, "calibration", rng, minimum_score=0.25)
        if calibration_path is None:
            calibration_path = weighted_template_choice(calibration_paths, "calibration", rng, minimum_score=0.25)

    raw_dir = output_dir / "raw" / split
    label_dir = output_dir / "labels" / split
    raw_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    band_1ghz, band_200mhz = generate_dual_bands(cable, rng, profile, dirty_params, calibration_path, measured_template_path)
    f1, s1, d1, i1, st1 = band_1ghz
    coverage_1ghz = save_client_csv(raw_dir / f"{sample_id}_1GHz.csv", f1, s1, d1, i1, st1, cable.total_length)

    f2, s2, d2, i2, st2 = band_200mhz
    coverage_200mhz = save_client_csv(raw_dir / f"{sample_id}_200MHz.csv", f2, s2, d2, i2, st2, cable.total_length)

    band_coverage = {
        "1GHz": coverage_1ghz,
        "200MHz": coverage_200mhz,
    }
    meta = sample_metadata(sample_id, split, profile, cable, seed, dirty_params, band_coverage)
    if calibration_path is not None:
        meta["calibration_source"] = str(calibration_path)
    if measured_template_path is not None:
        meta["measured_template_source"] = str(measured_template_path)
        meta["measured_template_quality"] = assess_template_quality(measured_template_path, profile)
        meta["measured_template_length_hint_m"] = template_length_hint(measured_template_path)
    meta["response_consistency"] = "strict_s11_ifft_only"
    with (raw_dir / f"{sample_id}.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, allow_unicode=True, sort_keys=False)

    label = build_label(cable)
    np.save(label_dir / f"{sample_id}.npy", label)

    return {
        "sample_id": sample_id,
        "split": split,
        "profile": profile,
        "total_length_m": float(round(cable.total_length, 2)),
        "epsr": float(round(cable.epsr, 4)),
        "n_defects": len(cable.defect_info),
        "n_joints": len(getattr(cable, "joint_positions", [])),
        "seed": seed,
        "csv_1ghz": f"raw/{split}/{sample_id}_1GHz.csv",
        "csv_200mhz": f"raw/{split}/{sample_id}_200MHz.csv",
        "yaml": f"raw/{split}/{sample_id}.yaml",
        "label": f"labels/{split}/{sample_id}.npy",
        "band_distance_coverage": band_coverage,
    }


def split_for_index(i: int, train_count: int, val_count: int, shuffled_indices: np.ndarray) -> dict[int, str]:
    split_map: dict[int, str] = {}
    for idx in shuffled_indices[:train_count]:
        split_map[int(idx)] = "train"
    for idx in shuffled_indices[train_count:train_count + val_count]:
        split_map[int(idx)] = "val"
    for idx in shuffled_indices[train_count + val_count:]:
        split_map[int(idx)] = "test"
    return split_map


def write_manifest(output_dir: Path, args: argparse.Namespace, entries: list[dict]) -> None:
    profile_counts: dict[str, int] = {}
    for entry in entries:
        profile_counts[entry["profile"]] = profile_counts.get(entry["profile"], 0) + 1
    manifest = {
        "name": "DG_max2p5km",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_total": len(entries),
        "n_train": sum(1 for e in entries if e["split"] == "train"),
        "n_val": sum(1 for e in entries if e["split"] == "val"),
        "n_test": sum(1 for e in entries if e["split"] == "test"),
        "seed": int(args.seed),
        "profile_request": args.profile,
        "profile_counts": profile_counts,
        "defect_count_policy": {
            "field": {
                "<80m": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "80-250m": [0.45, 0.35, 0.17, 0.03, 0.0, 0.0],
                "250-500m": [0.28, 0.34, 0.24, 0.10, 0.03, 0.01],
                "500-800m": [0.22, 0.31, 0.27, 0.13, 0.05, 0.02],
                "800-1500m": [0.16, 0.28, 0.29, 0.17, 0.07, 0.03],
                "1500-2500m": [0.12, 0.24, 0.29, 0.21, 0.10, 0.04],
            },
            "rg58_random": {
                "10-50m": [0.70, 0.25, 0.05],
                "50-120m": [0.58, 0.32, 0.10],
                "120-200m": [0.50, 0.35, 0.15],
            },
            "rg58": "known_template_no_extra_random_body_defects",
        },
        "defect_type_policy": {
            "field": {
                "long_probability_per_defect": {
                    "<250m": 0.0,
                    "250-500m": 0.03,
                    "500-800m": 0.06,
                    "800-1500m": 0.10,
                    "1500-2500m": 0.16,
                },
                "max_long_defects": {"<800m": 1, ">=800m": 2},
                "long_type_probabilities": {"aging": 0.60, "moisture_local": 0.25, "moisture_distributed": 0.15},
                "aging_length_model": "clip(total_length * U(0.05, 0.16), 20m, 320m)",
                "moisture_local_length_model": "clip(total_length * U(0.03, 0.10), 15m, 220m)",
                "moisture_distributed_length_model": "clip(total_length * U(0.18, 0.55), 120m, max(130m, 0.72*total_length))",
                "moisture_segment_model": "6-18 smooth subsegments; moisture alias maps to moisture_local",
            },
            "rg58": {"allowed_types": ["short"]},
            "rg58_random": {"allowed_types": ["short"]},
        },
        "label_grid": {"d_max_m": D_MAX, "dd_m": DD, "n_points": N_GRID},
        "sweeps": {
            "1GHz": {"start_hz": 9e3, "stop_hz": 1e9, "n_points": 50000},
            "200MHz": {"start_hz": 9e3, "stop_hz": 200e6, "n_points": 5000},
        },
        "window": "hann",
        "distance_coverage_note": (
            "CSV distance columns use the Client/IFFT unambiguous range for each sweep. "
            "Distance columns use the Client/IFFT unambiguous range for each sweep; "
            "truncated_by_ifft_range=true marks any sample whose saved distance axis "
            "does not reach 1.2x cable length."
        ),
        "samples": entries,
    }
    with (output_dir / "manifest.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, allow_unicode=True, sort_keys=False)


def run_dataset_generation(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    real_data_root = Path(args.real_data_root)
    rg58_paths, field_paths, calibration_paths = discover_real_files(real_data_root)

    n_total = int(args.n_total)
    n_train = int(n_total * args.train_ratio)
    n_val = int(n_total * args.val_ratio)
    master_rng = np.random.RandomState(int(args.seed))
    seeds = master_rng.randint(0, 2**31 - 1, size=n_total)
    shuffled = master_rng.permutation(n_total)
    split_map = split_for_index(n_total, n_train, n_val, shuffled)

    jobs = []
    for i in range(n_total):
        jobs.append({
            "sample_id": f"dg{i:06d}",
            "split": split_map[i],
            "seed": int(seeds[i]),
            "output_dir": str(output_dir),
            "profile": args.profile,
            "calibration_paths": [str(p) for p in calibration_paths],
            "rg58_template_paths": [str(p) for p in rg58_paths],
            "field_template_paths": [str(p) for p in field_paths],
        })

    print("DG dataset generation:")
    print(f"  output_dir={output_dir}")
    print(f"  n_total={n_total} train={n_train} val={n_val} test={n_total - n_train - n_val}")
    print(f"  profile={args.profile} label_grid={N_GRID} points @ {DD} m")
    print(f"  calibration_templates={len(calibration_paths)}")
    print(f"  measured_templates=rg58:{len(rg58_paths)} field:{len(field_paths)}")

    entries: list[dict] = []
    t0 = time.time()
    workers = max(1, int(args.workers))
    if workers == 1:
        for i, job in enumerate(jobs, 1):
            entry = generate_one_sample_job(job)
            entries.append(entry)
            print_progress(i, n_total, t0, entry)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(generate_one_sample_job, job) for job in jobs]
            for i, future in enumerate(as_completed(futures), 1):
                entry = future.result()
                entries.append(entry)
                print_progress(i, n_total, t0, entry)

    entries.sort(key=lambda item: item["sample_id"])
    write_manifest(output_dir, args, entries)
    elapsed = time.time() - t0
    print(f"Done: {len(entries)} samples in {elapsed:.1f}s ({len(entries)/max(elapsed, 1e-6):.2f}/s)")
    print(f"Manifest: {output_dir / 'manifest.yaml'}")


def print_progress(done: int, total: int, t0: float, entry: dict) -> None:
    if done == 1 or done == total or done % max(1, min(25, total // 10 or 1)) == 0:
        elapsed = time.time() - t0
        rate = done / max(elapsed, 1e-6)
        eta = (total - done) / max(rate, 1e-6)
        print(
            f"  [{done:4d}/{total}] {entry['sample_id']} {entry['profile']} "
                f"L={entry['total_length_m']:.1f}m defects={entry['n_defects']} "
                f"rate={rate:.2f}/s ETA={eta:.0f}s"
        )
        if entry.get("band_distance_coverage", {}).get("200MHz", {}).get("truncated_by_ifft_range"):
            print("    note: 200MHz distance column is limited by its coarser frequency step; use the 1GHz file for the longer unambiguous range.")


def make_preview_cable(path: Path, profile: str, rng: np.random.RandomState) -> CableSample:
    if profile == "rg58":
        text = str(path)
        if "RG58-74M" in text:
            return make_known_rg58_cable("rg58_74m", rng, epsr=2.25)
        if "LineC" in path.name and "LineA" not in path.name and "LineB" not in path.name:
            return make_cable_for_profile("rg58", rng, total_length=25.0)
        if "LineC" in path.name:
            return make_known_rg58_cable("rg58_3lines_long", rng, epsr=2.25)
        return make_known_rg58_cable("rg58_3lines_ab", rng, epsr=2.25)
    return make_field_cable(rng, total_length=infer_length_from_path(path))


def moving_average(values: np.ndarray, n: int) -> np.ndarray:
    if len(values) < n:
        return values
    kernel = np.ones(n, dtype=np.float64) / n
    return np.convolve(values, kernel, mode="same")


def estimate_measured_end_from_s11(
    freq_hz: np.ndarray,
    s11: np.ndarray,
    approx_length_m: float,
    epsr: float = 2.3,
) -> tuple[float, float]:
    """Estimate field cable end from measured IFFT near the directory length."""
    distance, impulse, step, _ = s11_to_responses(freq_hz, s11, epsr=epsr, window="hann")
    step_s = moving_average(np.real(step), 101)
    grad = np.abs(np.gradient(step_s, distance))
    imp_abs = moving_average(np.abs(np.real(impulse)), 21)

    windows = [(0.70, 1.25), (0.55, 1.35), (0.35, 1.45)]
    for lo_mul, hi_mul in windows:
        lo = max(30.0, approx_length_m * lo_mul)
        hi = min(float(distance[-1]), approx_length_m * hi_mul)
        mask = (distance >= lo) & (distance <= hi)
        if int(mask.sum()) < 30:
            continue
        grad_part = grad[mask]
        imp_part = imp_abs[mask]
        ds = distance[mask]
        score = 0.55 * grad_part / max(float(np.nanpercentile(grad_part, 99)), 1e-12)
        score += 0.45 * imp_part / max(float(np.nanpercentile(imp_part, 99)), 1e-12)
        score *= 0.8 + 0.2 * (ds - ds.min()) / max(float(ds.max() - ds.min()), 1.0)
        best = int(np.nanargmax(score))
        return float(ds[best]), float(score[best])
    return float(np.clip(approx_length_m, 30.0, 2500.0)), 0.0


def infer_termination_from_measured(
    path: Path,
    distance: np.ndarray,
    step: np.ndarray,
    end_m: float,
) -> str:
    name = path.name + " " + path.parent.name
    if "短路" in name:
        return "short"
    if "开路" in name:
        return "open"
    d = np.asarray(distance, dtype=np.float64)
    s = moving_average(np.real(step), 101)
    before = (d >= max(0.0, end_m - 80.0)) & (d <= max(0.0, end_m - 15.0))
    after = (d >= end_m + 10.0) & (d <= end_m + 80.0)
    if np.any(before) and np.any(after):
        delta = float(np.nanmedian(s[after]) - np.nanmedian(s[before]))
        if delta < 0:
            return "short"
        if delta > 0:
            return "open"
    return "weak_open"


def window_peak_p99(distance: np.ndarray, values: np.ndarray, lo_m: float, hi_m: float) -> float:
    d = np.asarray(distance, dtype=np.float64)
    y = np.asarray(values)
    mask = (d >= lo_m) & (d <= hi_m)
    if int(mask.sum()) < 2:
        return float("nan")
    return float(np.nanpercentile(np.abs(np.real(y[mask])), 99))


def fast_real_p95(freq_hz: np.ndarray, s11: np.ndarray, lo_hz: float, hi_hz: float) -> float:
    f = np.asarray(freq_hz, dtype=np.float64)
    y = np.asarray(s11).real
    mask = (f >= lo_hz) & (f <= min(hi_hz, float(f[-1])))
    if int(mask.sum()) < 20:
        return float("nan")
    part = y[mask]
    window = max(51, int(len(part) / 8))
    if window % 2 == 0:
        window += 1
    fast = part - smooth_array(part, window)
    return float(np.nanpercentile(np.abs(fast), 95))


def band_real_p95(freq_hz: np.ndarray, s11: np.ndarray, lo_hz: float, hi_hz: float) -> float:
    f = np.asarray(freq_hz, dtype=np.float64)
    y = np.asarray(s11).real
    mask = (f >= lo_hz) & (f <= min(hi_hz, float(f[-1])))
    if int(mask.sum()) < 20:
        return float("nan")
    return float(np.nanpercentile(np.abs(y[mask]), 95))


def finite_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) <= 1e-12:
        return float("nan")
    return float(numerator / denominator)


def band_mask(freq_hz: np.ndarray, lo_hz: float, hi_hz: float) -> np.ndarray:
    f = np.asarray(freq_hz, dtype=np.float64)
    return (f >= lo_hz) & (f <= min(hi_hz, float(f[-1])))


def band_magnitude_diff_db(
    freq_ref: np.ndarray,
    s_ref: np.ndarray,
    freq_dg: np.ndarray,
    s_dg: np.ndarray,
    lo_hz: float,
    hi_hz: float,
) -> tuple[float, float]:
    mask = band_mask(freq_ref, lo_hz, hi_hz)
    if int(mask.sum()) < 20:
        return float("nan"), float("nan")
    dg_interp = interpolate_s11(freq_dg, s_dg, np.asarray(freq_ref)[mask])
    diff = s11_magnitude_db(dg_interp) - s11_magnitude_db(np.asarray(s_ref)[mask])
    return float(np.nanmedian(diff)), float(np.nanpercentile(np.abs(diff), 95))


def phase_trend_stats(
    freq_ref: np.ndarray,
    s_ref: np.ndarray,
    freq_dg: np.ndarray,
    s_dg: np.ndarray,
    lo_hz: float,
    hi_hz: float,
) -> tuple[float, float]:
    mask = band_mask(freq_ref, lo_hz, hi_hz)
    if int(mask.sum()) < 50:
        return float("nan"), float("nan")
    f_part = np.asarray(freq_ref, dtype=np.float64)[mask]
    dg_interp = interpolate_s11(freq_dg, s_dg, f_part)
    phase_ref = s11_relative_unwrapped_phase_rad(np.asarray(s_ref)[mask])
    phase_dg = s11_relative_unwrapped_phase_rad(dg_interp)
    x = (f_part - f_part[0]) / max(float(f_part[-1] - f_part[0]), 1.0)
    ref_fit = np.polyfit(x, phase_ref, 1)
    dg_fit = np.polyfit(x, phase_dg, 1)
    ref_residual = phase_ref - np.polyval(ref_fit, x)
    dg_residual = phase_dg - np.polyval(dg_fit, x)
    slope_ratio = finite_ratio(float(dg_fit[0]), float(ref_fit[0]))
    residual_p95 = float(np.nanpercentile(np.abs(dg_residual - ref_residual), 95))
    return slope_ratio, residual_p95


def step_end_delta(distance: np.ndarray, step: np.ndarray, center_m: float) -> float:
    d = np.asarray(distance, dtype=np.float64)
    y = moving_average(np.real(step), 101)
    before = (d >= max(0.0, center_m - 80.0)) & (d <= max(0.0, center_m - 15.0))
    after = (d >= center_m + 10.0) & (d <= center_m + 80.0)
    if int(before.sum()) < 5 or int(after.sum()) < 5:
        return float("nan")
    return float(np.nanmedian(y[after]) - np.nanmedian(y[before]))


def build_preview_diagnostics(row: dict) -> dict:
    profile = row["profile"]
    cable = row["cable"]
    metrics: dict[str, float | str | int | None] = {
        "profile": profile,
        "path": str(row["real_path"]),
        "folder": row["real_path"].parent.name,
        "file": row["real_path"].name,
        "length_m": float(cable.total_length),
    }
    if row.get("folder_length_m") is not None:
        metrics["folder_length_m"] = float(row["folder_length_m"])
        metrics["estimated_end_1g_m"] = float(row["estimated_end_m"])
        metrics["estimated_end_200m_m"] = float(row["estimated_end_200m_m"])
        metrics["epsr_if_folder_length"] = float(row["epsr_if_folder_length"])
        metrics["termination"] = row.get("termination")

    for suffix, lo_hz, hi_hz in [
        ("0_200", 0.0, 200e6),
        ("200_400", 200e6, 400e6),
        ("400_1000", 400e6, 1e9),
    ]:
        measured_re = band_real_p95(row["freq_real"], row["s_real"], lo_hz, hi_hz)
        dg_re = band_real_p95(row["f1"], row["s1"], lo_hz, hi_hz)
        metrics[f"re_p95_{suffix}_measured"] = measured_re
        metrics[f"re_p95_{suffix}_dg1"] = dg_re
        metrics[f"re_p95_ratio_{suffix}"] = finite_ratio(dg_re, measured_re)
        metrics[f"hf_fast_ratio_{suffix}"] = finite_ratio(
            fast_real_p95(row["f1"], row["s1"], lo_hz, hi_hz),
            fast_real_p95(row["freq_real"], row["s_real"], lo_hz, hi_hz),
        )
        mag_median, mag_p95 = band_magnitude_diff_db(
            row["freq_real"], row["s_real"], row["f1"], row["s1"], lo_hz, hi_hz
        )
        metrics[f"mag_median_diff_db_{suffix}"] = mag_median
        metrics[f"mag_p95_absdiff_db_{suffix}"] = mag_p95
        slope_ratio, phase_residual = phase_trend_stats(
            row["freq_real"], row["s_real"], row["f1"], row["s1"], lo_hz, hi_hz
        )
        metrics[f"phase_slope_ratio_{suffix}"] = slope_ratio
        metrics[f"phase_residual_p95_rad_{suffix}"] = phase_residual

    end_m = float(cable.total_length)
    measured_step_1g = step_end_delta(row["d_real_1g"], row["step_real_1g"], end_m)
    dg_step_1g = step_end_delta(row["d1"], row["step1"], end_m)
    measured_step_200m = step_end_delta(row["d_real_200m"], row["step_real_200m"], end_m)
    dg_step_200m = step_end_delta(row["d2"], row["step2"], end_m)
    metrics["step_end_delta_measured_1g"] = measured_step_1g
    metrics["step_end_delta_dg1"] = dg_step_1g
    metrics["step_end_delta_ratio_1g"] = finite_ratio(abs(dg_step_1g), abs(measured_step_1g))
    metrics["step_end_delta_measured_200m"] = measured_step_200m
    metrics["step_end_delta_dg200"] = dg_step_200m
    metrics["step_end_delta_ratio_200m"] = finite_ratio(abs(dg_step_200m), abs(measured_step_200m))

    if profile == "rg58":
        near_measured = window_peak_p99(row["d_real_1g"], row["imp_real_1g"], 0.0, 10.0)
        near_dg = window_peak_p99(row["d1"], row["imp1"], 0.0, 10.0)
        end_measured = window_peak_p99(row["d_real_1g"], row["imp_real_1g"], end_m - 2.0, end_m + 3.0)
        end_dg = window_peak_p99(row["d1"], row["imp1"], end_m - 2.0, end_m + 3.0)
        joint_ratios = []
        for pos in getattr(cable, "joint_positions", []):
            measured_joint = window_peak_p99(row["d_real_1g"], row["imp_real_1g"], pos - 2.0, pos + 2.0)
            dg_joint = window_peak_p99(row["d1"], row["imp1"], pos - 2.0, pos + 2.0)
            ratio = finite_ratio(dg_joint, measured_joint)
            if np.isfinite(ratio):
                joint_ratios.append(ratio)
        metrics["near_peak_ratio"] = finite_ratio(near_dg, near_measured)
        metrics["joint_peak_ratio_median"] = float(np.nanmedian(joint_ratios)) if joint_ratios else float("nan")
        metrics["end_peak_ratio"] = finite_ratio(end_dg, end_measured)
        metrics["end_to_near_measured"] = finite_ratio(end_measured, near_measured)
        metrics["end_to_near_dg"] = finite_ratio(end_dg, near_dg)
    else:
        base_lo = max(20.0, end_m * 0.08)
        base_hi = max(30.0, end_m * 0.85)
        base_1g_measured = window_peak_p99(row["d_real_1g"], row["imp_real_1g"], base_lo, base_hi)
        base_1g_dg = window_peak_p99(row["d1"], row["imp1"], base_lo, base_hi)
        base_200m_measured = window_peak_p99(row["d_real_200m"], row["imp_real_200m"], base_lo, base_hi)
        base_200m_dg = window_peak_p99(row["d2"], row["imp2"], base_lo, base_hi)
        end_1g_measured = window_peak_p99(row["d_real_1g"], row["imp_real_1g"], end_m - 25.0, end_m + 35.0)
        end_1g_dg = window_peak_p99(row["d1"], row["imp1"], end_m - 25.0, end_m + 35.0)
        end_200m_measured = window_peak_p99(row["d_real_200m"], row["imp_real_200m"], end_m - 25.0, end_m + 35.0)
        end_200m_dg = window_peak_p99(row["d2"], row["imp2"], end_m - 25.0, end_m + 35.0)
        metrics["field_base_peak_ratio_1g"] = finite_ratio(base_1g_dg, base_1g_measured)
        metrics["field_base_peak_ratio_200m"] = finite_ratio(base_200m_dg, base_200m_measured)
        metrics["field_end_peak_ratio_1g"] = finite_ratio(end_1g_dg, end_1g_measured)
        metrics["field_end_peak_ratio_200m"] = finite_ratio(end_200m_dg, end_200m_measured)
    return metrics


def diagnostics_payload(rows: list[dict], seed: int) -> dict:
    row_metrics = [build_preview_diagnostics(row) for row in rows]
    grouped: dict[str, dict] = {}
    for profile in ["rg58", "field"]:
        subset = [row for row in row_metrics if row["profile"] == profile]
        grouped[profile] = {}
        for key in [
            "hf_fast_ratio_400_1000",
            "mag_p95_absdiff_db_400_1000",
            "phase_residual_p95_rad_400_1000",
            "step_end_delta_ratio_1g",
            "step_end_delta_ratio_200m",
        ]:
            values = [float(row[key]) for row in subset if key in row and np.isfinite(float(row[key]))]
            grouped[profile][key] = {
                "median": float(np.nanmedian(values)) if values else None,
                "max": float(np.nanmax(values)) if values else None,
            }
    return {
        "seed": int(seed),
        "row_count": len(row_metrics),
        "summary": grouped,
        "rows": row_metrics,
    }


def print_preview_metrics(row: dict) -> None:
    profile = row["profile"]
    cable = row["cable"]
    f_real = row["freq_real"]
    s_real = row["s_real"]
    print(
        "    Low-band Re P95 <=200MHz measured/DG1/DG200="
        f"{band_real_p95(f_real, s_real, 0.0, 200e6):.4g}/"
        f"{band_real_p95(row['f1'], row['s1'], 0.0, 200e6):.4g}/"
        f"{band_real_p95(row['f2'], row['s2'], 0.0, 200e6):.4g}"
    )
    print(
        "    HF fast Re P95 200-400MHz measured/DG="
        f"{fast_real_p95(f_real, s_real, 200e6, 400e6):.4g}/"
        f"{fast_real_p95(row['f1'], row['s1'], 200e6, 400e6):.4g}; "
        "400-1000MHz measured/DG="
        f"{fast_real_p95(f_real, s_real, 400e6, 1e9):.4g}/"
        f"{fast_real_p95(row['f1'], row['s1'], 400e6, 1e9):.4g}"
    )
    if profile == "rg58":
        joints = getattr(cable, "joint_positions", [])
        measured_joint = [window_peak_p99(row["d_real_1g"], row["imp_real_1g"], p - 2.0, p + 2.0) for p in joints]
        dg_joint = [window_peak_p99(row["d1"], row["imp1"], p - 2.0, p + 2.0) for p in joints]
        print(
            "    RG58 impulse P99 near/joint/end measured="
            f"{window_peak_p99(row['d_real_1g'], row['imp_real_1g'], 0.0, 10.0):.4g}/"
            f"{[round(v, 5) for v in measured_joint]}/"
            f"{window_peak_p99(row['d_real_1g'], row['imp_real_1g'], cable.total_length - 2.0, cable.total_length + 3.0):.4g}; "
            "DG1="
            f"{window_peak_p99(row['d1'], row['imp1'], 0.0, 10.0):.4g}/"
            f"{[round(v, 5) for v in dg_joint]}/"
            f"{window_peak_p99(row['d1'], row['imp1'], cable.total_length - 2.0, cable.total_length + 3.0):.4g}"
        )
    else:
        total = cable.total_length
        base_lo = max(20.0, total * 0.08)
        base_hi = max(30.0, total * 0.85)
        print(
            "    Field impulse base P99 measured1/DG1/measured200/DG200="
            f"{window_peak_p99(row['d_real_1g'], row['imp_real_1g'], base_lo, base_hi):.4g}/"
            f"{window_peak_p99(row['d1'], row['imp1'], base_lo, base_hi):.4g}/"
            f"{window_peak_p99(row['d_real_200m'], row['imp_real_200m'], base_lo, base_hi):.4g}/"
            f"{window_peak_p99(row['d2'], row['imp2'], base_lo, base_hi):.4g}; "
            "end P99 measured1/DG1/measured200/DG200="
            f"{window_peak_p99(row['d_real_1g'], row['imp_real_1g'], total - 25.0, total + 35.0):.4g}/"
            f"{window_peak_p99(row['d1'], row['imp1'], total - 25.0, total + 35.0):.4g}/"
            f"{window_peak_p99(row['d_real_200m'], row['imp_real_200m'], total - 25.0, total + 35.0):.4g}/"
            f"{window_peak_p99(row['d2'], row['imp2'], total - 25.0, total + 35.0):.4g}"
        )


def s11_to_responses_for_stop(
    freq_hz: np.ndarray,
    s11: np.ndarray,
    stop_hz: float,
    epsr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = freq_hz <= stop_hz
    if int(mask.sum()) < 32:
        mask = np.ones_like(freq_hz, dtype=bool)
    distance, impulse, step, _ = s11_to_responses(freq_hz[mask], s11[mask], epsr=epsr, window="hann")
    return distance, impulse, step


def select_preview_files(real_data_root: Path, seed: int) -> tuple[list[Path], list[Path], list[Path]]:
    rg58_files, field_files, calibration_files = discover_real_files(real_data_root)
    if len(rg58_files) < 4:
        raise RuntimeError(f"Need at least 4 RG58 csv files under {real_data_root}")
    if len(field_files) < 4:
        raise RuntimeError(f"Need at least 4 field csv files under {real_data_root}")
    rng = np.random.RandomState(seed)
    rg58_selected = [rg58_files[i] for i in rng.choice(len(rg58_files), 4, replace=False)]
    field_candidates = [p for p in field_files if infer_length_from_path(p, clip=False) <= 2500.0]
    if len(field_candidates) < 4:
        field_candidates = field_files
    field_selected = [field_candidates[i] for i in rng.choice(len(field_candidates), 4, replace=False)]
    return rg58_selected, field_selected, calibration_files


def run_preview(args: argparse.Namespace) -> None:
    real_data_root = Path(args.real_data_root)
    rg58_files, field_files, calibration_files = select_preview_files(real_data_root, int(args.seed))
    selected = [(p, "rg58") for p in rg58_files] + [(p, "field") for p in field_files]
    image_dir = PROJECT_ROOT / "Image"
    image_dir.mkdir(parents=True, exist_ok=True)
    realpart_path = image_dir / "DG_max2p5km_realpart_comparison.png"
    impulse_path = image_dir / "DG_max2p5km_impulse_comparison.png"
    step_path = image_dir / "DG_max2p5km_step_comparison.png"
    magnitude_path = image_dir / "DG_max2p5km_magnitude_comparison.png"
    phase_path = image_dir / "DG_max2p5km_phase_comparison.png"

    plt.rcParams["font.family"] = ["Times New Roman", "SimHei"]
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"

    preview_rows: list[dict] = []
    for idx, (real_path, profile) in enumerate(selected):
        rng = np.random.RandomState(int(args.seed) + 1000 + idx)
        freq_real, s_real = read_s11_csv_compatible(real_path)
        folder_length = None
        estimated_end = None
        if profile == "field":
            folder_length = infer_length_from_path(real_path, clip=False)
            estimated_end, estimate_score = estimate_measured_end_from_s11(
                freq_real, s_real, approx_length_m=folder_length, epsr=2.3
            )
            estimated_end_200m, estimate_score_200m = estimate_measured_end_from_s11(
                freq_real[freq_real <= 200e6],
                s_real[freq_real <= 200e6],
                approx_length_m=folder_length,
                epsr=2.3,
            )
            epsr_if_folder = float(2.3 * (estimated_end / folder_length) ** 2) if folder_length > 0 else None
            d_for_term, _, step_for_term = s11_to_responses_for_stop(freq_real, s_real, 1e9, 2.3)
            termination = infer_termination_from_measured(real_path, d_for_term, step_for_term, estimated_end)
            cable = make_field_cable(
                rng,
                total_length=float(np.clip(estimated_end, 30.0, 2500.0)),
                epsr=2.3,
                termination=termination,
                n_defects_override=0,
            )
        else:
            estimate_score = None
            estimated_end_200m = None
            estimate_score_200m = None
            epsr_if_folder = None
            termination = None
            cable = make_preview_cable(real_path, profile, rng)
        params = dirty_params_for_profile(profile, rng)
        if profile == "field":
            if cable.total_length < 500.0:
                params = replace(
                    params,
                    template_slow_scale=max(params.template_slow_scale, 0.72),
                    template_mix_scale=min(max(params.template_mix_scale, 0.004), 0.012),
                    fixture_scale=params.fixture_scale * 0.16,
                    highfreq_decay_strength=params.highfreq_decay_strength * 0.45,
                    event_hf_damping=max(params.event_hf_damping, 0.82),
                )
            else:
                params = replace(
                    params,
                    template_slow_scale=max(params.template_slow_scale, 0.62),
                    template_mix_scale=max(params.template_mix_scale, 0.08),
                    fixture_scale=params.fixture_scale * 0.75,
                    event_hf_damping=max(params.event_hf_damping, 0.72),
                )
        else:
            params = replace(
                params,
                template_slow_scale=max(params.template_slow_scale, 0.38),
                template_mix_scale=0.0,
                fixture_scale=max(params.fixture_scale, 0.026),
                event_hf_damping=max(params.event_hf_damping, 0.68),
            )
        calibration_path = None
        if profile == "field" and calibration_files:
            same_dir_cals = [p for p in calibration_files if p.parent == real_path.parent]
            calibration_path = same_dir_cals[0] if same_dir_cals else calibration_files[int(rng.randint(0, len(calibration_files)))]

        band_1ghz, band_200mhz = generate_dual_bands(cable, rng, profile, params, calibration_path, real_path)
        f1, s1, d1, imp1, step1 = band_1ghz
        f2, s2, d2, imp2, step2 = band_200mhz
        real_epsr = 2.3 if profile == "field" else cable.epsr
        d_real_1g, imp_real_1g, step_real_1g = s11_to_responses_for_stop(freq_real, s_real, 1e9, real_epsr)
        d_real_200m, imp_real_200m, step_real_200m = s11_to_responses_for_stop(freq_real, s_real, 200e6, real_epsr)

        preview_rows.append({
            "real_path": real_path,
            "profile": profile,
            "cable": cable,
            "freq_real": freq_real,
            "s_real": s_real,
            "f1": f1,
            "s1": s1,
            "f2": f2,
            "s2": s2,
            "d_real_1g": d_real_1g,
            "imp_real_1g": imp_real_1g,
            "step_real_1g": step_real_1g,
            "d_real_200m": d_real_200m,
            "imp_real_200m": imp_real_200m,
            "step_real_200m": step_real_200m,
            "d1": d1,
            "imp1": imp1,
            "step1": step1,
            "d2": d2,
            "imp2": imp2,
            "step2": step2,
            "folder_length_m": folder_length,
            "estimated_end_m": estimated_end,
            "estimated_end_200m_m": estimated_end_200m,
            "estimate_score": estimate_score,
            "estimate_score_200m": estimate_score_200m,
            "epsr_if_folder_length": epsr_if_folder,
            "termination": termination,
        })
        if estimated_end is None:
            print(f"  preview {idx + 1}/8: {real_path}")
        else:
            print(
                f"  preview {idx + 1}/8: {real_path} "
                f"(dir={folder_length:.1f}m, estimated_end_1g={estimated_end:.1f}m, "
                f"estimated_end_200m={estimated_end_200m:.1f}m, epsr_if_dir={epsr_if_folder:.3f}, "
                f"termination={termination})"
            )
        print_preview_metrics(preview_rows[-1])

    if getattr(args, "diagnose_preview", False):
        payload = diagnostics_payload(preview_rows, int(args.seed))
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    plot_realpart_preview(preview_rows, realpart_path)
    plot_response_preview(preview_rows, impulse_path, response="impulse")
    plot_response_preview(preview_rows, step_path, response="step")
    plot_magnitude_preview(preview_rows, magnitude_path)
    plot_phase_preview(preview_rows, phase_path)
    print(f"Preview figure saved: {realpart_path}")
    print(f"Preview figure saved: {impulse_path}")
    print(f"Preview figure saved: {step_path}")
    print(f"Preview figure saved: {magnitude_path}")
    print(f"Preview figure saved: {phase_path}")


def build_defect_case_specs() -> list[dict]:
    return [
        {"name": "RG58 known compact BNC", "profile": "rg58", "length_m": 88.0, "defects": [
            {"type": "short", "start_m": 27.0, "length_m": 1.4, "z0_mult": 1.020, "epsr_delta": 0.022, "alpha_mult": 1.10},
            {"type": "short", "start_m": 61.0, "length_m": 2.6, "z0_mult": 0.978, "epsr_delta": 0.028, "alpha_mult": 1.14},
        ]},
        {"name": "RG58 random mid healthy", "profile": "rg58_random", "length_m": 96.0, "defects": []},
        {"name": "RG58 random far short", "profile": "rg58_random", "length_m": 144.0, "defects": [
            {"type": "short", "start_m": 101.0, "length_m": 2.2, "z0_mult": 1.070, "epsr_delta": 0.052, "alpha_mult": 1.42}
        ]},
        {"name": "RG58 random asymmetric shorts", "profile": "rg58_random", "length_m": 172.0, "defects": [
            {"type": "short", "start_m": 46.0, "length_m": 2.4, "z0_mult": 0.930, "epsr_delta": 0.060, "alpha_mult": 1.44},
            {"type": "short", "start_m": 132.0, "length_m": 1.7, "z0_mult": 1.075, "epsr_delta": 0.050, "alpha_mult": 1.36},
        ]},
        {"name": "Field medium healthy", "profile": "field", "length_m": 760.0, "defects": []},
        {"name": "Field early short", "profile": "field", "length_m": 540.0, "defects": [
            {"type": "short", "start_m": 92.0, "length_m": 3.5, "z0_mult": 1.125, "epsr_delta": 0.09, "alpha_mult": 1.58}
        ]},
        {"name": "Field late short", "profile": "field", "length_m": 1460.0, "defects": [
            {"type": "short", "start_m": 1188.0, "length_m": 7.0, "z0_mult": 0.885, "epsr_delta": 0.14, "alpha_mult": 1.82}
        ]},
        {"name": "Field aging long", "profile": "field", "length_m": 1380.0, "defects": [
            {"type": "aging", "start_m": 420.0, "length_m": 175.0, "z0_mult": 0.955, "epsr_delta": 0.22, "alpha_mult": 2.55, "label_amplitude": 0.59}
        ]},
        {"name": "Field moisture_local early", "profile": "field", "length_m": 1040.0, "defects": [
            {"type": "moisture_local", "start_m": 225.0, "length_m": 70.0, "z0_mult": 0.915, "epsr_delta": 0.48, "alpha_mult": 3.65, "label_amplitude": 0.65}
        ]},
        {"name": "Field moisture_distributed central", "profile": "field", "length_m": 1760.0, "defects": [
            {"type": "moisture_distributed", "start_m": 540.0, "length_m": 1040.0, "z0_mult": 0.365, "epsr_delta": 0.54, "alpha_mult": 2.45, "label_amplitude": 0.67}
        ]},
        {"name": "Field aging+short", "profile": "field", "length_m": 2050.0, "defects": [
            {"type": "aging", "start_m": 350.0, "length_m": 130.0, "z0_mult": 1.050, "epsr_delta": 0.19, "alpha_mult": 2.30, "label_amplitude": 0.57},
            {"type": "short", "start_m": 1515.0, "length_m": 5.8, "z0_mult": 1.150, "epsr_delta": 0.12, "alpha_mult": 1.74},
        ]},
        {"name": "Field aging+moisture_distributed", "profile": "field", "length_m": 2480.0, "defects": [
            {"type": "aging", "start_m": 580.0, "length_m": 160.0, "z0_mult": 1.040, "epsr_delta": 0.20, "alpha_mult": 2.40, "label_amplitude": 0.58},
            {"type": "moisture_distributed", "start_m": 1540.0, "length_m": 780.0, "z0_mult": 0.360, "epsr_delta": 0.58, "alpha_mult": 2.65, "label_amplitude": 0.70},
        ]},
    ]


def cable_from_defect_case(case: dict) -> CableSample:
    profile = str(case["profile"])
    total_length = float(case["length_m"])
    if is_rg58_profile(profile):
        eps = 2.25
        base_z0 = 50.0
        alpha = 0.085
    else:
        eps = 2.35
        base_z0 = 56.0
        alpha = 0.012

    defects = sorted(case.get("defects", []), key=lambda d: float(d["start_m"]))
    segments: list[SegmentParams] = []
    distributed_moisture_regions: list[dict] = []
    distributed_long_regions: list[dict] = []
    cursor = 0.0
    for defect in defects:
        start_m = float(defect["start_m"])
        length_m = float(defect["length_m"])
        if start_m > cursor:
            segments.append(_segment(start_m - cursor, base_z0, eps, alpha))
        defect_type = str(defect.get("type", "short"))
        if defect_type == "moisture":
            defect_type = "moisture_local"
        label_amplitude = defect.get("label_amplitude")
        region = append_gradual_defect_segments(
            segments,
            length_m,
            base_z0,
            eps,
            alpha,
            base_z0 * float(defect.get("z0_mult", 1.0)),
            eps + float(defect.get("epsr_delta", 0.0)),
            alpha * float(defect.get("alpha_mult", 1.0)),
            defect_type,
            None if label_amplitude is None else float(label_amplitude),
            f"manual-{len(segments)}",
        )
        if region is not None:
            if region.get("type") == "moisture_distributed":
                distributed_moisture_regions.append(region)
            else:
                distributed_long_regions.append(region)
        cursor = start_m + length_m
    if cursor < total_length:
        segments.append(_segment(total_length - cursor, base_z0, eps, alpha))

    cable = CableSample(segments=segments, epsr=eps, seed=0)
    cable.distributed_moisture_regions = distributed_moisture_regions
    cable.distributed_long_regions = distributed_long_regions
    cable.has_joint_reflections = False
    cable.joint_positions = cumulative_internal_positions(cable) if profile == "rg58" and defects else []
    cable.z_load_open = 900.0 if is_rg58_profile(profile) else 1e13
    cable.termination = "finite_open" if is_rg58_profile(profile) else "open"
    cable.preview_profile = profile
    cable.defect_count_policy = {"profile": profile, "source": "manual_defect_case_preview", "sampled_count": len(cable.defect_info)}
    cable.defect_type_policy = {"profile": profile, "source": "manual_defect_case_preview", "sampled_types": [d["type"] for d in cable.defect_info]}
    return cable


def generate_clean_case_bands(cable: CableSample) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    rng = np.random.RandomState(int(getattr(cable, "seed", 0)))
    f1, s1 = generate_s11(cable, SWEEP_1GHZ, rng=rng, add_noise=False, inject_joints=False)
    d1, i1, st1, _ = s11_to_responses(f1, s1, epsr=cable.epsr, window="hann")
    f2 = SWEEP_200MHZ.frequencies()
    s2 = interpolate_s11(f1, s1, f2)
    d2, i2, st2, _ = s11_to_responses(f2, s2, epsr=cable.epsr, window="hann")
    return (f1, s1, d1, i1, st1), (f2, s2, d2, i2, st2)


def select_defect_case_backgrounds(real_data_root: Path, seed: int) -> dict[str, list[Path]]:
    try:
        rg58_files, field_files, calibration_files = discover_real_files(real_data_root)
    except Exception:
        return {"rg58": [], "field": [], "calibration": []}

    rng = np.random.RandomState(seed)
    field_candidates = [p for p in field_files if infer_length_from_path(p, clip=False) <= 2500.0]
    if not field_candidates:
        field_candidates = list(field_files)

    def shuffled(paths: list[Path]) -> list[Path]:
        if not paths:
            return []
        order = rng.permutation(len(paths))
        return [paths[int(i)] for i in order]

    return {
        "rg58": shuffled(list(rg58_files)),
        "field": shuffled(field_candidates),
        "calibration": list(calibration_files),
    }


def defect_case_template_for_profile(profile: str, case_index: int, backgrounds: dict[str, list[Path]]) -> Path | None:
    paths = backgrounds.get("field" if profile == "field" else "rg58", [])
    if not paths:
        return None
    return paths[case_index % len(paths)]


def calibration_for_template(template_path: Path | None, backgrounds: dict[str, list[Path]], rng: np.random.RandomState) -> Path | None:
    if template_path is None:
        return None
    calibration_paths = backgrounds.get("calibration", [])
    if not calibration_paths:
        return None
    same_dir = [p for p in calibration_paths if p.parent == template_path.parent]
    if same_dir:
        return same_dir[0]
    return calibration_paths[int(rng.randint(0, len(calibration_paths)))]


def dirty_params_for_defect_case(profile: str, cable: CableSample, rng: np.random.RandomState) -> DirtyParams:
    params = dirty_params_for_profile(profile, rng)
    defect_types = {d.get("type", "short") for d in cable.defect_info}
    if profile == "field":
        if "moisture_distributed" in defect_types:
            return replace(
                params,
                additive_scale=params.additive_scale * 0.15,
                multiplicative_scale=params.multiplicative_scale * 0.15,
                ripple_scale=0.0,
                phase_scale_rad=0.0,
                template_slow_scale=0.0,
                template_mix_scale=0.0,
                fixture_scale=0.0,
                highfreq_decay_strength=params.highfreq_decay_strength * 0.25,
                event_hf_damping=max(params.event_hf_damping, 0.78),
            )
        if cable.total_length < 500.0:
            return replace(
                params,
                template_slow_scale=max(params.template_slow_scale, 0.70),
                template_mix_scale=min(max(params.template_mix_scale, 0.010), 0.030),
                fixture_scale=params.fixture_scale * 0.22,
                highfreq_decay_strength=params.highfreq_decay_strength * 0.45,
                event_hf_damping=max(params.event_hf_damping, 0.82),
            )
        return replace(
            params,
            template_slow_scale=max(params.template_slow_scale, 0.74),
            template_mix_scale=max(params.template_mix_scale, 0.115),
            fixture_scale=params.fixture_scale * 0.85,
            event_hf_damping=max(params.event_hf_damping, 0.74),
        )

    return replace(
        params,
        template_slow_scale=max(params.template_slow_scale, 0.42),
        template_mix_scale=max(params.template_mix_scale, 0.010),
        fixture_scale=max(params.fixture_scale, 0.028),
        event_hf_damping=max(params.event_hf_damping, 0.68),
    )


def generate_defect_case_bands(
    cable: CableSample,
    profile: str,
    rng: np.random.RandomState,
    use_dirty: bool,
    calibration_path: Path | None,
    measured_template_path: Path | None,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    if not use_dirty:
        return generate_clean_case_bands(cable)
    params = dirty_params_for_defect_case(profile, cable, rng)
    return generate_dual_bands(cable, rng, profile, params, calibration_path, measured_template_path)


def generate_defect_case_rows(
    seed: int = 20260623,
    real_data_root: Path | str | None = None,
    use_dirty: bool = True,
) -> list[dict]:
    real_data_root = Path(real_data_root) if real_data_root is not None else Path(r"E:\FDR案例-csv")
    backgrounds = select_defect_case_backgrounds(real_data_root, seed) if use_dirty else {"rg58": [], "field": [], "calibration": []}
    rows: list[dict] = []
    for idx, case in enumerate(build_defect_case_specs()):
        rng = np.random.RandomState(int(seed) + 5000 + idx)
        cable = cable_from_defect_case(case)
        profile = str(case["profile"])
        measured_template_path = defect_case_template_for_profile(profile, idx, backgrounds)
        if profile == "rg58_random":
            measured_template_path = None
        if profile == "field" and cable.defect_info:
            measured_template_path = None
        calibration_path = calibration_for_template(measured_template_path if profile == "field" else None, backgrounds, rng)
        band_1ghz, band_200mhz = generate_defect_case_bands(
            cable,
            profile,
            rng,
            use_dirty,
            calibration_path,
            measured_template_path,
        )
        f1, s1, d1, imp1, step1 = band_1ghz
        f2, s2, d2, imp2, step2 = band_200mhz
        rows.append({
            "case_name": case["name"],
            "profile": case["profile"],
            "cable": cable,
            "preview_mode": "dirty" if use_dirty else "clean",
            "measured_template_path": str(measured_template_path) if measured_template_path is not None else None,
            "calibration_template_path": str(calibration_path) if calibration_path is not None else None,
            "f1": f1,
            "s1": s1,
            "f2": f2,
            "s2": s2,
            "d1": d1,
            "imp1": imp1,
            "step1": step1,
            "d2": d2,
            "imp2": imp2,
            "step2": step2,
            "seed": int(seed) + idx,
        })
    return rows


def defect_case_title(row: dict) -> str:
    cable = row["cable"]
    types = [d["type"] for d in cable.defect_info]
    type_text = ",".join(types) if types else "healthy"
    return f"{row['case_name']}\nL={cable.total_length:.0f}m, {type_text}"


def plot_defect_case_frequency(rows: list[dict], out_path: Path, mode: str) -> None:
    n_cols = 5
    n_rows = int(math.ceil(len(rows) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 4.2 * n_rows), constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(n_rows, n_cols)
    for idx, row in enumerate(rows):
        ax = axes_arr[idx // n_cols, idx % n_cols]
        if mode == "realpart":
            y1 = row["s1"].real
            y2 = row["s2"].real
            ylabel = "Re(S11)"
            title = "S11 Real Part"
        elif mode == "magnitude":
            y1 = s11_magnitude_db(row["s1"])
            y2 = s11_magnitude_db(row["s2"])
            ylabel = "20log10(|S11|) (dB)"
            title = "Magnitude"
        elif mode == "phase":
            y1 = s11_wrapped_phase_deg(row["s1"])
            y2 = s11_wrapped_phase_deg(row["s2"])
            ylabel = "Wrapped phase (degree)"
            title = "Wrapped Phase"
            ax.set_ylim(-190, 190)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        ax.plot(row["f1"] / 1e6, y1, color="#d62728", linewidth=0.55, alpha=0.85, label="1GHz")
        ax.plot(row["f2"] / 1e6, y2, color="#2ca02c", linewidth=0.85, alpha=0.85, label="200MHz")
        ax.set_title(defect_case_title(row), fontsize=9)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 1000)
        ax.legend(fontsize=7, loc="best")
        style_preview_axis(ax)
    for idx in range(len(rows), n_rows * n_cols):
        axes_arr[idx // n_cols, idx % n_cols].axis("off")
    fig.suptitle(f"DG Defect Case Preview - {title}", fontsize=14)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_defect_case_response(rows: list[dict], out_path: Path, response: str) -> None:
    n_cols = 5
    n_rows = int(math.ceil(len(rows) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 4.2 * n_rows), constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(n_rows, n_cols)
    for idx, row in enumerate(rows):
        ax = axes_arr[idx // n_cols, idx % n_cols]
        if response == "impulse":
            y1 = np.real(row["imp1"])
            y2 = np.real(row["imp2"])
            ylabel = "Impulse response"
            title = "IFFT Impulse Response"
        elif response == "step":
            y1 = np.real(row["step1"])
            y2 = np.real(row["step2"])
            ylabel = "Step response"
            title = "IFFT Step Response"
        else:
            raise ValueError(f"Unsupported response: {response}")
        ax.plot(row["d1"], y1, color="#d62728", linewidth=0.65, alpha=0.85, label="1GHz")
        ax.plot(row["d2"], y2, color="#2ca02c", linewidth=0.90, alpha=0.85, label="200MHz")
        for defect in row["cable"].defect_info:
            ax.axvspan(float(defect["start"]), float(defect["end"]), color="#9467bd", alpha=0.16)
        nominal_end = float(row["cable"].total_length)
        effective_end = float(effective_terminal_phase_length_m(row["cable"]))
        ax.axvline(nominal_end, color="#444444", linestyle="--", linewidth=0.8, alpha=0.55)
        if effective_end > nominal_end + 5.0:
            ax.axvline(effective_end, color="#111111", linestyle=":", linewidth=0.9, alpha=0.70)
        ax.set_title(defect_case_title(row), fontsize=9)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, min(max(effective_end * 1.16, nominal_end * 1.2, 30.0), 3000.0))
        ax.legend(fontsize=7, loc="best")
        style_preview_axis(ax)
    for idx in range(len(rows), n_rows * n_cols):
        axes_arr[idx // n_cols, idx % n_cols].axis("off")
    fig.suptitle(f"DG Defect Case Preview - {title}", fontsize=14)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_defect_case_preview(args: argparse.Namespace) -> None:
    image_dir = PROJECT_ROOT / "Image"
    image_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = ["Times New Roman", "SimHei"]
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"

    rows = generate_defect_case_rows(seed=int(args.seed), real_data_root=Path(args.real_data_root), use_dirty=True)
    outputs = {
        "realpart": image_dir / "DG_defect_cases_realpart.png",
        "magnitude": image_dir / "DG_defect_cases_magnitude.png",
        "phase": image_dir / "DG_defect_cases_phase.png",
        "impulse": image_dir / "DG_defect_cases_impulse.png",
        "step": image_dir / "DG_defect_cases_step.png",
    }
    plot_defect_case_frequency(rows, outputs["realpart"], "realpart")
    plot_defect_case_frequency(rows, outputs["magnitude"], "magnitude")
    plot_defect_case_frequency(rows, outputs["phase"], "phase")
    plot_defect_case_response(rows, outputs["impulse"], "impulse")
    plot_defect_case_response(rows, outputs["step"], "step")
    print("Defect case preview generated with dirty measurement backgrounds:")
    for row in rows:
        template = row.get("measured_template_path")
        template_name = f"template-{abs(hash(template)) % 100000:05d}" if template else "none"
        print(
            f"  {row['case_name']}: profile={row['profile']} "
            f"L={row['cable'].total_length:.1f}m "
            f"defects={[d['type'] for d in row['cable'].defect_info]} "
            f"template={template_name}"
        )
    for path in outputs.values():
        print(f"Preview figure saved: {path}")


def style_preview_axis(ax) -> None:
    ax.grid(True, alpha=0.25)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def preview_title(row: dict) -> str:
    cable = row["cable"]
    if row.get("estimated_end_m") is not None:
        return (
            f"{row['profile'].upper()} L~{cable.total_length:.0f}m "
            f"(dir {row['folder_length_m']:.0f}m, 1G {row['estimated_end_m']:.0f}m, "
            f"200M {row['estimated_end_200m_m']:.0f}m, eps {row['epsr_if_folder_length']:.2f})\n"
            f"{row['real_path'].parent.name}"
        )
    return f"{row['profile'].upper()} L~{cable.total_length:.0f}m\n{row['real_path'].parent.name}"


def plot_realpart_preview(rows: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    for idx, row in enumerate(rows):
        ax = axes[idx // 4, idx % 4]
        ax.plot(row["freq_real"] / 1e6, row["s_real"].real, color="#1f77b4", linewidth=0.55, alpha=0.85, label="Measured")
        ax.plot(row["f1"] / 1e6, row["s1"].real, color="#d62728", linewidth=0.55, alpha=0.75, label="DG 1GHz")
        ax.plot(row["f2"] / 1e6, row["s2"].real, color="#2ca02c", linewidth=0.9, alpha=0.85, label="DG 200MHz")
        ax.set_title(preview_title(row), fontsize=9)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Re(S11)")
        ax.set_xlim(0, min(max(row["freq_real"].max(), 1e9) / 1e6, 1000))
        ax.legend(fontsize=7, loc="best")
        style_preview_axis(ax)
    fig.suptitle("DG DirtyGenerator Real-Part Shape Comparison", fontsize=14)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def s11_magnitude_db(s11: np.ndarray) -> np.ndarray:
    mag = np.maximum(np.abs(np.asarray(s11, dtype=np.complex128)), 1e-8)
    return 20.0 * np.log10(mag)


def s11_unwrapped_phase_rad(s11: np.ndarray) -> np.ndarray:
    return np.unwrap(np.angle(np.asarray(s11, dtype=np.complex128)))


def s11_relative_unwrapped_phase_rad(s11: np.ndarray) -> np.ndarray:
    phase = s11_unwrapped_phase_rad(s11)
    return phase - phase[0]


def s11_wrapped_phase_deg(s11: np.ndarray) -> np.ndarray:
    return np.angle(np.asarray(s11, dtype=np.complex128), deg=True)


def plot_magnitude_preview(rows: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    for idx, row in enumerate(rows):
        ax = axes[idx // 4, idx % 4]
        ax.plot(
            row["freq_real"] / 1e6,
            s11_magnitude_db(row["s_real"]),
            color="#1f77b4",
            linewidth=0.55,
            alpha=0.85,
            label="Measured",
        )
        ax.plot(
            row["f1"] / 1e6,
            s11_magnitude_db(row["s1"]),
            color="#d62728",
            linewidth=0.55,
            alpha=0.75,
            label="DG 1GHz",
        )
        ax.plot(
            row["f2"] / 1e6,
            s11_magnitude_db(row["s2"]),
            color="#2ca02c",
            linewidth=0.9,
            alpha=0.85,
            label="DG 200MHz",
        )
        ax.set_title(preview_title(row), fontsize=9)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("|S11| (dB)")
        ax.set_xlim(0, min(max(row["freq_real"].max(), 1e9) / 1e6, 1000))
        ax.legend(fontsize=7, loc="best")
        style_preview_axis(ax)
    fig.suptitle("DG DirtyGenerator Magnitude-Frequency Comparison", fontsize=14)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_phase_preview(rows: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    for idx, row in enumerate(rows):
        ax = axes[idx // 4, idx % 4]
        ax.plot(
            row["freq_real"] / 1e6,
            s11_wrapped_phase_deg(row["s_real"]),
            color="#1f77b4",
            linewidth=0.55,
            alpha=0.85,
            label="Measured",
        )
        ax.plot(
            row["f1"] / 1e6,
            s11_wrapped_phase_deg(row["s1"]),
            color="#d62728",
            linewidth=0.55,
            alpha=0.75,
            label="DG 1GHz",
        )
        ax.plot(
            row["f2"] / 1e6,
            s11_wrapped_phase_deg(row["s2"]),
            color="#2ca02c",
            linewidth=0.9,
            alpha=0.85,
            label="DG 200MHz",
        )
        ax.set_title(preview_title(row), fontsize=9)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Wrapped phase (degree)")
        ax.set_xlim(0, min(max(row["freq_real"].max(), 1e9) / 1e6, 1000))
        ax.set_ylim(-190, 190)
        ax.legend(fontsize=7, loc="best")
        style_preview_axis(ax)
    fig.suptitle("DG DirtyGenerator Wrapped Phase-Frequency Comparison", fontsize=14)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def normalize_three_for_plot(
    x_arrays: list[np.ndarray],
    y_arrays: list[np.ndarray],
    x_max: float,
) -> list[np.ndarray]:
    visible_parts = []
    for x, y in zip(x_arrays, y_arrays):
        arr = np.asarray(y, dtype=np.float64)
        mask = np.asarray(x) <= x_max
        if np.any(mask):
            visible_parts.append(arr[mask])
    if visible_parts:
        joined = np.concatenate(visible_parts)
    else:
        joined = np.concatenate([np.asarray(y, dtype=np.float64) for y in y_arrays])
    scale = max(float(np.nanpercentile(np.abs(joined), 99)), 1e-12)
    return [np.clip(np.asarray(y, dtype=np.float64) / scale, -1.5, 1.5) for y in y_arrays]


def subtract_initial_baseline(distance: np.ndarray, values: np.ndarray, x_max: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    d = np.asarray(distance, dtype=np.float64)
    baseline_limit = max(5.0, x_max * 0.05)
    mask = d <= baseline_limit
    if not np.any(mask):
        mask = np.arange(len(arr)) < max(1, int(len(arr) * 0.05))
    return arr - float(np.nanmedian(arr[mask]))


def plot_response_preview(rows: list[dict], out_path: Path, response: str) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    for idx, row in enumerate(rows):
        ax = axes[idx // 4, idx % 4]
        cable = row["cable"]
        x_max = min(max(cable.total_length * 1.2, 30.0), 3000.0)
        if response == "impulse":
            y_real_1g_raw = np.real(row["imp_real_1g"])
            y_real_200m_raw = np.real(row["imp_real_200m"])
            y_1g_raw = np.real(row["imp1"])
            y_200m_raw = np.real(row["imp2"])
            y_label = "Impulse real (norm.)"
            title = "DG DirtyGenerator IFFT Impulse Comparison"
        elif response == "step":
            y_real_1g_raw = subtract_initial_baseline(row["d_real_1g"], row["step_real_1g"], x_max)
            y_real_200m_raw = subtract_initial_baseline(row["d_real_200m"], row["step_real_200m"], x_max)
            y_1g_raw = subtract_initial_baseline(row["d1"], row["step1"], x_max)
            y_200m_raw = subtract_initial_baseline(row["d2"], row["step2"], x_max)
            y_label = "Step response (norm.)"
            title = "DG DirtyGenerator IFFT Step Comparison"
        else:
            raise ValueError(f"Unknown response: {response}")

        y_real_1g, y_1g, y_real_200m, y_200m = normalize_three_for_plot(
            [row["d_real_1g"], row["d1"], row["d_real_200m"], row["d2"]],
            [y_real_1g_raw, y_1g_raw, y_real_200m_raw, y_200m_raw],
            x_max,
        )

        ax.plot(row["d_real_1g"], y_real_1g, color="#1f77b4", linewidth=0.75, alpha=0.85, label="Measured <=1GHz")
        ax.plot(row["d1"], y_1g, color="#d62728", linewidth=0.65, alpha=0.75, label="DG 1GHz")
        ax.plot(row["d_real_200m"], y_real_200m, color="#17becf", linewidth=0.75, alpha=0.75, label="Measured <=200MHz")
        ax.plot(row["d2"], y_200m, color="#2ca02c", linewidth=0.85, alpha=0.85, label="DG 200MHz")
        ax.axvline(cable.total_length, color="#444444", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_title(preview_title(row), fontsize=9)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel(y_label)
        ax.set_xlim(0, x_max)
        ax.legend(fontsize=7, loc="best")
        style_preview_axis(ax)
    fig.suptitle(title, fontsize=14)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DirtyGenerator dataset for <=2.5 km cable S11.")
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "DataSet" / "DG_max2p5km"))
    parser.add_argument("--n_total", type=positive_int, default=3000)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.133)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--workers", type=positive_int, default=max((os.cpu_count() or 2) - 1, 1))
    parser.add_argument("--profile", choices=["mixed", "rg58", "rg58_random", "field"], default="mixed")
    parser.add_argument("--real_data_root", default=r"E:\FDR案例-csv")
    parser.add_argument("--preview_only", action="store_true")
    parser.add_argument("--diagnose_preview", action="store_true")
    parser.add_argument("--defect_case_preview", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise SystemExit("train_ratio and val_ratio must leave a positive test split")
    if args.defect_case_preview:
        run_defect_case_preview(args)
    elif args.preview_only or args.diagnose_preview:
        run_preview(args)
    else:
        run_dataset_generation(args)


if __name__ == "__main__":
    main()
