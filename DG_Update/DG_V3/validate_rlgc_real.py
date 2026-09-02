"""Validate the DG V3 RLGC kernel against measured RG58 data.

This is deliberately an independent, read-only validation program.  It does
not use fixture/noise/template models and never edits IFFT results.  Results
are written only when ``main`` is executed; importing this module has no side
effects.

Compared models
---------------
* ADS V1 reference RLGC kernel (generic segment cascade built from REF code)
* DG V2.7 physical RLGC kernel (no noise and no injected joints)
* frozen legacy DG V3 empirical alpha/beta/Zc equations
* current DG V3 RLGC kernel through a small adapter in ``new_v3_rlgc_s11``

The adapter is intentionally isolated because the DG V3 RLGC API is being
implemented in parallel.  It accepts the canonical CableSegment fields and
uses topology_abcd/network_s11; if that public API changes, only the adapter
needs adjustment.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DG_V3_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(r"E:\FDR案例-csv")
DEFAULT_OUTPUT = ROOT / "AgentsStorage" / "DG_V3_RLGC_validation"
ADS_PATH = ROOT / "REF" / "[ADS_V1]v3.3_74m_s11_generator.py"
V27_PATH = ROOT / "DG_V2.7" / "core" / "s11_generator.py"

C0 = 299_792_458.0
Z_REF = 50.0
Z_OPEN = 1.0e13
TAN_DELTA_V27 = 2.5e-4


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ADS = _load_module("dg_validation_ads_ref", ADS_PATH)
V27 = _load_module("dg_validation_v27_core", V27_PATH)
sys.path.insert(0, str(DG_V3_DIR))


@dataclass(frozen=True)
class SegmentSpec:
    length_m: float
    z0_ohm: float
    epsr: float
    alpha_db_per_m_at_100mhz: float
    tan_delta_at_100mhz: float
    region: str = "healthy"


@dataclass(frozen=True)
class MeasurementCase:
    path: Path
    conductor: str
    family: str
    topology_name: str
    repeat: int
    split: str
    nominal_length_m: float
    segments: tuple[SegmentSpec, ...]


def _ads_targets() -> tuple[Any, Any]:
    cfg = ADS.default_config()
    return cfg.healthy, cfg.aged


ADS_HEALTHY, ADS_AGED = _ads_targets()
HEALTHY = SegmentSpec(
    0.0,
    float(ADS_HEALTHY.z0_target_ohm),
    float(1.0 / ADS_HEALTHY.vf_target**2),
    float(ADS_HEALTHY.alpha_target_db_per_m_at_fref),
    TAN_DELTA_V27,
    "healthy",
)
AGED = SegmentSpec(
    0.0,
    float(ADS_AGED.z0_target_ohm),
    float(1.0 / ADS_AGED.vf_target**2),
    float(ADS_AGED.alpha_target_db_per_m_at_fref),
    TAN_DELTA_V27,
    "aging",
)
VF_REFERENCE = float(ADS_HEALTHY.vf_target)


def _seg(length: float, template: SegmentSpec) -> SegmentSpec:
    return SegmentSpec(length, template.z0_ohm, template.epsr,
                       template.alpha_db_per_m_at_100mhz,
                       template.tan_delta_at_100mhz, template.region)


TOPOLOGIES: dict[str, tuple[SegmentSpec, ...]] = {
    "74m_40+4+30": (_seg(40.0, HEALTHY), _seg(4.0, AGED), _seg(30.0, HEALTHY)),
    "71m_30+1+40": (_seg(30.0, HEALTHY), _seg(1.0, AGED), _seg(40.0, HEALTHY)),
    "96m_40+1+25+30": (
        _seg(40.0, HEALTHY), _seg(1.0, AGED),
        _seg(25.0, HEALTHY), _seg(30.0, HEALTHY),
    ),
}


def discover_cases(data_root: Path) -> list[MeasurementCase]:
    """Discover Core/Shield repeats and attach only documented topologies."""
    roots = (
        (data_root / "RG58-74M(40+4+30)", "rg58_74m"),
        (data_root / "RG58-3Lines", "rg58_3lines"),
    )
    cases: list[MeasurementCase] = []
    repeated = re.compile(r"-(\d+)\.csv$", re.IGNORECASE)
    for folder, family in roots:
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.csv")):
            if not path.name.startswith(("Core-", "Shield-")):
                continue
            match = repeated.search(path.name)
            if match is None:
                continue
            repeat = int(match.group(1))
            if family == "rg58_74m":
                topology_name = "74m_40+4+30"
            elif "LineA+CUT1+LineC+LineB" in path.name:
                topology_name = "96m_40+1+25+30"
            elif "LineB+CUT1+LineA" in path.name:
                topology_name = "71m_30+1+40"
            else:
                continue
            segments = TOPOLOGIES[topology_name]
            cases.append(MeasurementCase(
                path=path,
                conductor="core" if path.name.startswith("Core-") else "shield",
                family=family,
                topology_name=topology_name,
                repeat=repeat,
                split="parameter_check" if repeat == 1 else "holdout",
                nominal_length_m=float(sum(item.length_m for item in segments)),
                segments=segments,
            ))
    return cases


def load_s11(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", skip_header=1, usecols=(0, 1, 2))
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError(f"CSV不是Frequency/Real/Imag三列格式: {path}")
    data = data[np.isfinite(data).all(axis=1)]
    data = data[data[:, 0] > 0.0]
    order = np.argsort(data[:, 0])
    data = data[order]
    unique = np.concatenate(([True], np.diff(data[:, 0]) > 0.0))
    data = data[unique]
    if len(data) < 4:
        raise ValueError(f"有效频点不足: {path}")
    return data[:, 0], data[:, 1] + 1j * data[:, 2]


def _recursive_s11(freq_hz: np.ndarray, line_parameters: list[tuple[np.ndarray, np.ndarray, float]]) -> np.ndarray:
    z_load = np.full(freq_hz.shape, Z_OPEN, dtype=np.complex128)
    for zc, gamma, length_m in reversed(line_parameters):
        exp_term = np.exp(-2.0 * gamma * length_m)
        reflection = (z_load - zc) / (z_load + zc)
        z_load = zc * (1.0 + reflection * exp_term) / (1.0 - reflection * exp_term)
    return (z_load - Z_REF) / (z_load + Z_REF)


def ads_s11(freq_hz: np.ndarray, segments: tuple[SegmentSpec, ...]) -> np.ndarray:
    params = []
    for segment in segments:
        target = ADS.SegmentTargetConfig(
            z0_target_ohm=segment.z0_ohm,
            vf_target=1.0 / math.sqrt(segment.epsr),
            alpha_target_db_per_m_at_fref=segment.alpha_db_per_m_at_100mhz,
            f_ref_hz=100e6,
            sigma_cu_ref_s_per_m=float(ADS_HEALTHY.sigma_cu_ref_s_per_m),
            sigma_dielectric_s_per_m=float(ADS_HEALTHY.sigma_dielectric_s_per_m),
        )
        geometry = ADS.target_to_effective_geometry(target)
        r, l, g, c = ADS.calc_primary_params(freq_hz, geometry, target)
        zc, gamma = ADS.calc_z0_gamma(r, l, g, c, freq_hz)
        params.append((zc, gamma, segment.length_m))
    return _recursive_s11(freq_hz, params)


def v27_s11(freq_hz: np.ndarray, segments: tuple[SegmentSpec, ...]) -> np.ndarray:
    cable_segments = [
        V27.SegmentParams(
            length_m=item.length_m,
            z0_ohm=item.z0_ohm,
            epsr=item.epsr,
            alpha_db_per_m_100mhz=item.alpha_db_per_m_at_100mhz,
            is_defect=item.region != "healthy",
            tan_delta_100mhz=item.tan_delta_at_100mhz,
            debye_delta_epsr=0.0,
        )
        for item in segments
    ]
    cable = V27.CableSample(
        segments=cable_segments, epsr=segments[0].epsr,
        z_ref=Z_REF, z_load_open=Z_OPEN, has_joint_reflections=False,
    )
    return V27._compute_s11_for_cable(freq_hz, cable)


def legacy_v3_s11(freq_hz: np.ndarray, segments: tuple[SegmentSpec, ...]) -> np.ndarray:
    """Frozen pre-RLGC DG V3 equations; independent of current physics.py."""
    f = np.asarray(freq_hz, dtype=np.float64)
    params = []
    for segment in segments:
        ratio = np.maximum(f, 1.0) / 100e6
        alpha_db = segment.alpha_db_per_m_at_100mhz * (0.35 * np.sqrt(ratio) + 0.65 * ratio)
        gamma = alpha_db / 8.685889638 + 1j * 2.0 * np.pi * f * math.sqrt(segment.epsr) / C0
        # The clean legacy comparison deliberately disables the old artificial dispersion.
        zc = np.full(f.shape, segment.z0_ohm, dtype=np.complex128)
        params.append((zc, gamma, segment.length_m))
    return _recursive_s11(f, params)


def new_v3_rlgc_s11(freq_hz: np.ndarray, segments: tuple[SegmentSpec, ...]) -> np.ndarray:
    """Adapter for the current public DG V3 RLGC topology API."""
    from dg_v3.physics import network_s11, topology_abcd
    from dg_v3.topology import CableSegment, CableTopology, derive_material

    segment_objects = []
    cursor = 0.0
    for index, item in enumerate(segments):
        kwargs = dict(
            start_m=cursor,
            end_m=cursor + item.length_m,
            z0_ohm=item.z0_ohm,
            epsr=item.epsr,
            alpha_db_per_m_at_100mhz=item.alpha_db_per_m_at_100mhz,
            tan_delta_at_100mhz=item.tan_delta_at_100mhz,
            region=item.region,
            defect_id=index if item.region != "healthy" else None,
        )
        bare_segment = CableSegment(**kwargs)
        material = derive_material(bare_segment, {
            "model": "coax_rlgc",
            "conductor_conductivity_s_per_m": float(ADS_HEALTHY.sigma_cu_ref_s_per_m),
            "dielectric_conductivity_s_per_m": float(ADS_HEALTHY.sigma_dielectric_s_per_m),
        })
        segment_objects.append(CableSegment(**kwargs, material=material))
        cursor += item.length_m

    # The new topology keeps dispersion_fraction only as a legacy serialized
    # field in some intermediate revisions.  It must be zero and is not used by
    # the RLGC kernel.
    topology = CableTopology(
        profile="rg58", length_m=cursor, z_ref_ohm=Z_REF,
        base_z0_ohm=segments[0].z0_ohm, base_epsr=segments[0].epsr,
        base_alpha_db_per_m_at_100mhz=segments[0].alpha_db_per_m_at_100mhz,
        base_tan_delta_at_100mhz=segments[0].tan_delta_at_100mhz,
        segments=segment_objects, joints=[],
        termination="open", z_load_ohm=Z_OPEN, defect_regions=[],
    )
    return network_s11(topology_abcd(freq_hz, topology), Z_OPEN, Z_REF)


MODEL_FUNCTIONS: dict[str, Callable[[np.ndarray, tuple[SegmentSpec, ...]], np.ndarray]] = {
    "ads": ads_s11,
    "dg_v2.7": v27_s11,
    "dg_v3_legacy": legacy_v3_s11,
    "dg_v3_rlgc": new_v3_rlgc_s11,
}


def _equally_spaced_spectrum(freq_hz: np.ndarray, s11: np.ndarray) -> tuple[np.ndarray, float]:
    steps_raw = np.diff(freq_hz)
    steps_raw = steps_raw[steps_raw > 0]
    df = float(np.percentile(steps_raw, 5))
    if df < float(np.mean(steps_raw)) / 10.0:
        df = float(np.mean(steps_raw))
    steps = int(np.floor(freq_hz[-1] / df))
    positive_f = np.arange(steps + 1, dtype=float) * df
    positive_s = np.interp(positive_f, freq_hz, s11.real) + 1j * np.interp(positive_f, freq_hz, s11.imag)
    spectrum = np.zeros(2 * steps + 1, dtype=np.complex128)
    spectrum[steps + 1:] = positive_s[1:]
    spectrum[:steps] = np.conj(positive_s[1:][::-1])
    abs_dc = 2.0 * abs(spectrum[steps + 1]) - abs(spectrum[steps + 2])
    phase_dc = 2.0 * np.angle(spectrum[steps + 1]) - np.angle(spectrum[steps + 2])
    spectrum[steps] = abs_dc * np.exp(1j * phase_dc)
    return spectrum, df


def s11_to_responses(freq_hz: np.ndarray, s11: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spectrum, df = _equally_spaced_spectrum(freq_hz, s11)
    shifted = np.concatenate((spectrum[len(spectrum) // 2:], spectrum[:len(spectrum) // 2]))
    td = np.fft.ifft(shifted)
    dt = 1.0 / (df * len(spectrum))
    distance = C0 * VF_REFERENCE * np.arange(len(spectrum)) * dt / 2.0
    return distance, np.real(td) / dt, np.real(np.cumsum(td))


def _crossing(x: np.ndarray, y: np.ndarray, level: float) -> float | None:
    for index in range(len(y) - 1):
        a, b = y[index], y[index + 1]
        if (a - level) * (b - level) <= 0.0 and a != b:
            weight = (level - a) / (b - a)
            return float(x[index] + weight * (x[index + 1] - x[index]))
    return None


def terminal_metrics(distance: np.ndarray, impulse: np.ndarray, step: np.ndarray,
                     nominal_length_m: float) -> dict[str, float | None]:
    search = (distance >= max(2.0, nominal_length_m - 15.0)) & (distance <= nominal_length_m + 25.0)
    if not np.any(search):
        raise ValueError("距离轴未覆盖末端搜索区")
    local_indices = np.flatnonzero(search)
    peak_index = int(local_indices[np.argmax(np.abs(impulse[search]))])
    peak_pos = float(distance[peak_index])
    peak = float(impulse[peak_index])
    sign = 1.0 if peak >= 0.0 else -1.0

    shape = (distance >= peak_pos - 10.0) & (distance <= peak_pos + 40.0)
    x = distance[shape]
    y = sign * impulse[shape]
    local_peak = int(np.argmax(y))
    peak_abs = float(y[local_peak])
    left_x = x[:local_peak + 1]
    left_y = y[:local_peak + 1] / peak_abs
    right_x = x[local_peak:]
    right_y = y[local_peak:] / peak_abs
    x10_left = _crossing(left_x, left_y, 0.1)
    x90_left = _crossing(left_x, left_y, 0.9)
    x90_right = _crossing(right_x, right_y, 0.9)
    x10_right = _crossing(right_x, right_y, 0.1)
    rise = x90_left - x10_left if x10_left is not None and x90_left is not None else None
    fall = x10_right - x90_right if x90_right is not None and x10_right is not None else None

    before = (distance >= peak_pos - 30.0) & (distance <= peak_pos - 10.0)
    after = (distance >= peak_pos + 5.0) & (distance <= peak_pos + 30.0)
    step_window = (distance >= peak_pos - 15.0) & (distance <= peak_pos + 15.0)
    width = None
    if np.any(before) and np.any(after) and np.any(step_window):
        baseline = float(np.median(step[before]))
        plateau = float(np.median(step[after]))
        scale = plateau - baseline
        if abs(scale) > 1e-15:
            sx = distance[step_window]
            sy = (step[step_window] - baseline) / scale
            c10 = _crossing(sx, sy, 0.1)
            if c10 is not None:
                tail = sx >= c10
                c90 = _crossing(sx[tail], sy[tail], 0.9)
                if c90 is not None:
                    width = float(c90 - c10)
    return {
        "peak_position_m": peak_pos,
        "peak_amplitude": peak,
        "rise_10_90_m": rise,
        "fall_90_10_m": fall,
        "asymmetry_fall_over_rise": fall / rise if rise and fall else None,
        "step_10_90_width_m": width,
    }


def _visible_mask(freq_hz: np.ndarray, measured: np.ndarray) -> tuple[np.ndarray, float]:
    magnitude_db = 20.0 * np.log10(np.maximum(np.abs(measured), 1e-15))
    high_band = freq_hz >= np.quantile(freq_hz, 0.75)
    floor_db = float(np.median(magnitude_db[high_band]))
    mask = magnitude_db >= floor_db + 6.0
    return mask, floor_db


def aligned_residual(freq_hz: np.ndarray, measured: np.ndarray, model: np.ndarray,
                     real_terminal_m: float, model_terminal_m: float) -> dict[str, Any]:
    delay_s = 2.0 * (real_terminal_m - model_terminal_m) / (C0 * VF_REFERENCE)
    aligned = model * np.exp(-1j * 2.0 * np.pi * freq_hz * delay_s)
    visible, floor_db = _visible_mask(freq_hz, measured)
    visible_count = int(np.count_nonzero(visible))
    if visible_count == 0:
        return {
            "delay_shift_s": float(delay_s),
            "noise_floor_db": floor_db,
            "visible_threshold_db": floor_db + 6.0,
            "visible_point_count": 0,
            "visible_fraction": 0.0,
            "aligned_complex_rms": None,
        }
    difference = aligned[visible] - measured[visible]
    return {
        "delay_shift_s": float(delay_s),
        "noise_floor_db": floor_db,
        "visible_threshold_db": floor_db + 6.0,
        "visible_point_count": visible_count,
        "visible_fraction": float(np.mean(visible)),
        "aligned_complex_rms": float(np.sqrt(np.mean(np.abs(difference) ** 2))),
    }


def evaluate_case(case: MeasurementCase) -> tuple[dict[str, Any], dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    freq_hz, measured = load_s11(case.path)
    responses: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    distance, impulse, step = s11_to_responses(freq_hz, measured)
    responses["measured"] = (distance, impulse, step)
    real_metrics = terminal_metrics(distance, impulse, step, case.nominal_length_m)
    model_metrics: dict[str, Any] = {}
    for name, function in MODEL_FUNCTIONS.items():
        model = function(freq_hz, case.segments)
        md, mi, ms = s11_to_responses(freq_hz, model)
        responses[name] = (md, mi, ms)
        metrics = terminal_metrics(md, mi, ms, case.nominal_length_m)
        metrics.update(aligned_residual(
            freq_hz, measured, model,
            float(real_metrics["peak_position_m"]), float(metrics["peak_position_m"]),
        ))
        metrics["terminal_position_abs_error_m"] = abs(
            float(metrics["peak_position_m"]) - float(real_metrics["peak_position_m"])
        )
        if metrics["asymmetry_fall_over_rise"] and real_metrics["asymmetry_fall_over_rise"]:
            metrics["log_asymmetry_abs_error"] = abs(math.log(
                float(metrics["asymmetry_fall_over_rise"]) /
                float(real_metrics["asymmetry_fall_over_rise"])
            ))
        else:
            metrics["log_asymmetry_abs_error"] = None
        if metrics["step_10_90_width_m"] and real_metrics["step_10_90_width_m"]:
            metrics["step_width_relative_error"] = abs(
                float(metrics["step_10_90_width_m"]) /
                float(real_metrics["step_10_90_width_m"]) - 1.0
            )
        else:
            metrics["step_width_relative_error"] = None
        model_metrics[name] = metrics
    return {
        "file": str(case.path),
        "conductor": case.conductor,
        "family": case.family,
        "topology": case.topology_name,
        "repeat": case.repeat,
        "split": case.split,
        "nominal_length_m": case.nominal_length_m,
        "frequency": {
            "start_hz": float(freq_hz[0]), "stop_hz": float(freq_hz[-1]),
            "point_count": int(len(freq_hz)),
        },
        "measured": real_metrics,
        "models": model_metrics,
    }, responses


ERROR_FIELDS = (
    "terminal_position_abs_error_m",
    "log_asymmetry_abs_error",
    "step_width_relative_error",
    "aligned_complex_rms",
)


def aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for scope, selected in (
        ("core_parameter_check", [c for c in cases if c["conductor"] == "core" and c["split"] == "parameter_check"]),
        ("core_holdout", [c for c in cases if c["conductor"] == "core" and c["split"] == "holdout"]),
        ("shield_auxiliary", [c for c in cases if c["conductor"] == "shield"]),
    ):
        models: dict[str, Any] = {}
        for model in MODEL_FUNCTIONS:
            fields: dict[str, Any] = {}
            for field in ERROR_FIELDS:
                values = [c["models"][model][field] for c in selected if c["models"][model][field] is not None]
                fields[field] = {
                    "count": len(values),
                    "median": float(np.median(values)) if values else None,
                    "mean": float(np.mean(values)) if values else None,
                }
            models[model] = fields
        result[scope] = {"case_count": len(selected), "models": models}

    holdout = [c for c in cases if c["conductor"] == "core" and c["split"] == "holdout"]
    wins = 0
    comparable = 0
    for case in holdout:
        old = case["models"]["dg_v3_legacy"]
        new = case["models"]["dg_v3_rlgc"]
        ratios = []
        for field in ERROR_FIELDS:
            if old[field] is not None and new[field] is not None:
                ratios.append(float(new[field]) / max(float(old[field]), 1e-15))
        if ratios:
            comparable += 1
            wins += float(np.mean(ratios)) < 1.0
    old_summary = result["core_holdout"]["models"]["dg_v3_legacy"]
    new_summary = result["core_holdout"]["models"]["dg_v3_rlgc"]
    median_improvements = {
        field: (
            old_summary[field]["median"] is not None
            and new_summary[field]["median"] is not None
            and float(new_summary[field]["median"]) < float(old_summary[field]["median"])
        )
        for field in ERROR_FIELDS
    }
    result["acceptance"] = {
        "core_holdout_comparable_cases": comparable,
        "dg_v3_rlgc_win_count": wins,
        "dg_v3_rlgc_win_fraction": wins / comparable if comparable else None,
        "required_win_fraction": 0.75,
        "median_improvements_over_legacy": median_improvements,
        "passed": bool(
            comparable and wins / comparable >= 0.75
            and all(median_improvements.values())
        ),
    }
    return result


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    aggregate_result = payload["aggregate"]
    lines = [
        "# DG V3 RLGC实测独立验证", "",
        "本报告仅比较纯传输线内核；未使用实测模板、夹具、测量链、噪声或IFFT反投影。", "",
        "- Core：硬验收；Shield：仅辅助观察。",
        "- `-1`：参数检查；`-2/-3/...`：留出验收。",
        "- 可见频点：实测高频四分位幅值本底中位数以上6 dB。",
        "- 温度只用于样本分组；本轮没有把未知温度系数反向拟合到模型。", "",
        "## 留出集汇总", "",
        "| 模型 | 末端位置中位误差/m | log不对称比中位误差 | 阶跃宽度中位相对误差 | 对齐复数RMS中位数 |",
        "|---|---:|---:|---:|---:|",
    ]
    holdout = aggregate_result["core_holdout"]["models"]
    for model in MODEL_FUNCTIONS:
        row = holdout[model]
        values = [row[field]["median"] for field in ERROR_FIELDS]
        fmt = ["—" if value is None else f"{value:.6g}" for value in values]
        lines.append(f"| {model} | " + " | ".join(fmt) + " |")
    acceptance = aggregate_result["acceptance"]
    lines.extend([
        "", "## 验收", "",
        f"- 新RLGC综合优于旧V3：{acceptance['dg_v3_rlgc_win_count']}/{acceptance['core_holdout_comparable_cases']}条Core留出曲线。",
        "- 四项留出集中位误差均降低：" + ("是" if all(acceptance["median_improvements_over_legacy"].values()) else "否") + "。",
        f"- 75%门槛：{'通过' if acceptance['passed'] else '未通过'}。",
        "- Shield结果不计入上述判定，详见JSON。", "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def configure_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    for font_path in (r"C:\Windows\Fonts\times.ttf", r"C:\Windows\Fonts\simhei.ttf"):
        if os.path.exists(font_path):
            fm.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = ["Times New Roman", "SimHei"]
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["savefig.dpi"] = 200
    return plt


def write_summary_plot(path: Path, payload: dict[str, Any]) -> None:
    plt = configure_matplotlib()
    metrics = payload["aggregate"]["core_holdout"]["models"]
    fields = ERROR_FIELDS
    titles = ("末端位置绝对误差 (m)", "log不对称比绝对误差", "阶跃宽度相对误差", "延时对齐复数RMS")
    colors = ("tab:blue", "tab:orange", "tab:red", "tab:green")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    names = list(MODEL_FUNCTIONS)
    for ax, field, title in zip(axes.flat, fields, titles):
        values = [metrics[name][field]["median"] for name in names]
        shown = [np.nan if value is None else value for value in values]
        ax.bar(np.arange(len(names)), shown, color=colors)
        ax.set_xticks(np.arange(len(names)), names, rotation=18, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        for spine in ax.spines.values():
            spine.set_visible(True)
    fig.suptitle("RG58 Core留出集：纯物理内核实测误差")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path)
    plt.close(fig)


def write_anchor_plot(path: Path, case_result: dict[str, Any],
                      responses: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
    plt = configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {"measured": "black", "ads": "tab:blue", "dg_v2.7": "tab:orange",
              "dg_v3_legacy": "tab:red", "dg_v3_rlgc": "tab:green"}
    nominal = float(case_result["nominal_length_m"])
    real_norm = max(float(np.max(np.abs(responses["measured"][1]))), 1e-15)
    for name, (distance, impulse, step) in responses.items():
        label = "实测" if name == "measured" else name
        style = "--" if name == "dg_v3_legacy" else "-"
        mask = (distance >= nominal - 12.0) & (distance <= nominal + 28.0)
        axes[0].plot(distance[mask], impulse[mask] / real_norm, style, lw=0.9,
                     color=colors[name], label=label)
        step_scale = max(float(np.max(np.abs(step[mask]))), 1e-15)
        axes[1].plot(distance[mask], step[mask] / step_scale, style, lw=0.9,
                     color=colors[name], label=label)
    axes[0].set_title("末端脉冲响应")
    axes[0].set_ylabel("按实测全局峰归一")
    axes[1].set_title("末端阶跃响应（各自归一）")
    for ax in axes:
        ax.set_xlabel("Distance (m)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        for spine in ax.spines.values():
            spine.set_visible(True)
    fig.suptitle(Path(case_result["file"]).name)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None,
                        help="仅调试时限制样本数；正式验收不要设置")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    cases = discover_cases(args.data_root)
    if args.limit is not None:
        cases = cases[:max(args.limit, 0)]
    if not cases:
        raise FileNotFoundError(f"没有发现可验证的RG58 CSV: {args.data_root}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    case_results: list[dict[str, Any]] = []
    anchor: tuple[dict[str, Any], dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] | None = None
    for index, case in enumerate(cases, start=1):
        print(f"[{index:03d}/{len(cases):03d}] {case.path.name}", flush=True)
        result, responses = evaluate_case(case)
        case_results.append(result)
        if anchor is None and case.conductor == "core" and case.family == "rg58_74m" and case.repeat == 1:
            anchor = result, responses

    payload = {
        "protocol": {
            "data_root": str(args.data_root),
            "models": list(MODEL_FUNCTIONS),
            "core_is_hard_acceptance": True,
            "shield_is_auxiliary": True,
            "parameter_check_repeats": [1],
            "holdout_repeats": ">=2",
            "visible_band_rule": "measured magnitude >= median(top frequency quartile) + 6 dB",
            "measurement_chain_enabled": False,
            "templates_enabled": False,
            "ifft_projection_enabled": False,
            "temperature_parameter_fitting_enabled": False,
            "canonical_segments": {
                name: [asdict(segment) for segment in segments]
                for name, segments in TOPOLOGIES.items()
            },
        },
        "cases": case_results,
    }
    payload["aggregate"] = aggregate(case_results)
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(output_dir / "report.md", payload)
    if not args.no_plots:
        write_summary_plot(output_dir / "core_holdout_summary.png", payload)
        if anchor is not None:
            write_anchor_plot(output_dir / "anchor_74m_40+4+30.png", *anchor)
    print(f"完成: {output_dir}")


if __name__ == "__main__":
    main()
