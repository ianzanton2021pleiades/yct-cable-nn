"""Frequency-domain physical kernels for the three Core-16 routes.

Segments are mappings with start_m/end_m and per-meter RLGC/Debye fields.
Joints are placed at position_m and use lumped series R-L followed by shunt
G-C. An optional fixture is a lossy delayed line followed by the same lumped
parasitic pair.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


ROUTES = ("dg_v3_rlgc", "cst_v1_continuous", "cst_v1_ladder_0p1m")
LADDER_CELL_M = 0.1
_LN10_OVER_20 = np.log(10.0) / 20.0


def _identity(frequency_count: int) -> np.ndarray:
    matrix = np.zeros((frequency_count, 2, 2), dtype=np.complex128)
    matrix[:, 0, 0] = 1.0
    matrix[:, 1, 1] = 1.0
    return matrix


def _normalize(matrix: np.ndarray, zref: float) -> np.ndarray:
    """Divide each frequency's ABCD matrix by one common scalar."""
    scale = np.maximum.reduce(
        (
            np.abs(matrix[:, 0, 0]),
            np.abs(matrix[:, 0, 1]) / zref,
            np.abs(matrix[:, 1, 0]) * zref,
            np.abs(matrix[:, 1, 1]),
        )
    )
    return matrix / scale[:, None, None]


def _cascade(left: np.ndarray, right: np.ndarray, zref: float) -> np.ndarray:
    return _normalize(np.matmul(left, right), zref)


def _series(impedance_ohm: np.ndarray, zref: float) -> np.ndarray:
    matrix = _identity(impedance_ohm.size)
    matrix[:, 0, 1] = impedance_ohm
    return _normalize(matrix, zref)


def _shunt(admittance_s: np.ndarray, zref: float) -> np.ndarray:
    matrix = _identity(admittance_s.size)
    matrix[:, 1, 0] = admittance_s
    return _normalize(matrix, zref)


def _segment_zy(
    frequency_hz: np.ndarray,
    segment: Mapping[str, float],
    reference_hz: float,
    route: str,
) -> tuple[np.ndarray, np.ndarray]:
    omega = 2.0 * np.pi * frequency_hz
    frequency_ratio = frequency_hz / reference_hz
    skin = (1.0 + 1.0j) * float(segment["r_skin_ref_ohm_per_m"]) * np.sqrt(
        frequency_ratio
    )
    impedance = (
        float(segment["r_dc_ohm_per_m"])
        + skin
        + 1.0j * omega * float(segment["l_external_h_per_m"])
    )
    capacitance = float(segment["c_inf_f_per_m"])
    debye = 1.0 + float(segment["debye_strength"]) / (
        1.0 + 1.0j * frequency_hz / float(segment["debye_corner_hz"])
    )
    admittance = (
        float(segment["g_dc_s_per_m"])
        + 1.0j * omega * capacitance * debye
        + omega * capacitance * float(segment["tan_delta"])
    )

    if route == "dg_v3_rlgc":
        return impedance, admittance

    omega_ref = 2.0 * np.pi * reference_hz
    ref_frequency = np.asarray([reference_hz], dtype=np.float64)
    ref_omega = np.asarray([omega_ref], dtype=np.float64)
    ref_ratio = ref_frequency / reference_hz
    ref_skin = (1.0 + 1.0j) * float(segment["r_skin_ref_ohm_per_m"]) * np.sqrt(
        ref_ratio
    )
    ref_impedance = (
        float(segment["r_dc_ohm_per_m"])
        + ref_skin
        + 1.0j * ref_omega * float(segment["l_external_h_per_m"])
    )
    ref_debye = 1.0 + float(segment["debye_strength"]) / (
        1.0 + 1.0j * ref_frequency / float(segment["debye_corner_hz"])
    )
    ref_admittance = (
        float(segment["g_dc_s_per_m"])
        + 1.0j
        * ref_omega
        * capacitance
        * ref_debye
        + ref_omega * capacitance * float(segment["tan_delta"])
    )
    resistance = float(np.real(ref_impedance[0]))
    inductance = float(np.imag(ref_impedance[0]) / omega_ref)
    conductance = float(np.real(ref_admittance[0]))
    capacitance_frozen = float(np.imag(ref_admittance[0]) / omega_ref)
    return (
        resistance + 1.0j * omega * inductance,
        conductance + 1.0j * omega * capacitance_frozen,
    )


def _line(impedance_per_m: np.ndarray, admittance_per_m: np.ndarray,
          length_m: float, zref: float) -> np.ndarray:
    propagation = np.sqrt(impedance_per_m * admittance_per_m)
    characteristic = np.sqrt(impedance_per_m / admittance_per_m)
    # 被动线Re(gamma)>=0；负指数形式避免复tanh内部的大正指数溢出。
    two_way = np.exp(-2.0 * propagation * length_m)
    normalized_sinh = (1.0 - two_way) / (1.0 + two_way)
    matrix = _identity(impedance_per_m.size)
    matrix[:, 0, 1] = characteristic * normalized_sinh
    matrix[:, 1, 0] = normalized_sinh / characteristic
    return _normalize(matrix, zref)


def _cell(impedance_per_m: np.ndarray, admittance_per_m: np.ndarray,
          length_m: float, zref: float) -> np.ndarray:
    # CST V1 cell order: parallel G-C first, then series R-L.
    shunt = _shunt(admittance_per_m * length_m, zref)
    series = _series(impedance_per_m * length_m, zref)
    return _cascade(shunt, series, zref)


def _matrix_power(matrix: np.ndarray, exponent: int, zref: float) -> np.ndarray:
    result = _identity(matrix.shape[0])
    factor = _normalize(matrix, zref)
    while exponent:
        if exponent & 1:
            result = _cascade(result, factor, zref)
        exponent >>= 1
        if exponent:
            factor = _cascade(factor, factor, zref)
    return result


def _ladder_line(impedance_per_m: np.ndarray, admittance_per_m: np.ndarray,
                 length_m: float, zref: float) -> np.ndarray:
    full_cells = int(np.floor((length_m + 1.0e-12) / LADDER_CELL_M))
    remainder_m = length_m - full_cells * LADDER_CELL_M
    if abs(remainder_m) < 1.0e-12:
        remainder_m = 0.0
    elif remainder_m < 0.0:
        full_cells -= 1
        remainder_m = length_m - full_cells * LADDER_CELL_M

    cell = _cell(impedance_per_m, admittance_per_m, LADDER_CELL_M, zref)
    matrix = _matrix_power(cell, full_cells, zref)
    if remainder_m > 0.0:
        partial = _cell(impedance_per_m, admittance_per_m, remainder_m, zref)
        matrix = _cascade(matrix, partial, zref)
    return matrix


def _joint(joint: Mapping[str, float], frequency_hz: np.ndarray,
           zref: float) -> np.ndarray:
    omega = 2.0 * np.pi * frequency_hz
    series = _series(
        float(joint["series_r_ohm"])
        + 1.0j * omega * float(joint["series_l_h"]),
        zref,
    )
    shunt = _shunt(
        float(joint["shunt_g_s"])
        + 1.0j * omega * float(joint["shunt_c_f"]),
        zref,
    )
    return _cascade(series, shunt, zref)


def _fixture(frequency_hz: np.ndarray, fixture: Mapping[str, float] | None,
             reference_hz: float, zref: float) -> np.ndarray:
    if fixture is None:
        return _identity(frequency_hz.size)

    omega = 2.0 * np.pi * frequency_hz
    attenuation_np_per_line = (
        _LN10_OVER_20
        * float(fixture["loss_db_ref"])
        * np.sqrt(frequency_hz / reference_hz)
    )
    propagation_length = attenuation_np_per_line + 1.0j * omega * float(
        fixture["delay_s"]
    )
    tanh_gl = np.tanh(propagation_length)
    line = _identity(frequency_hz.size)
    z0 = float(fixture["z0_ohm"])
    line[:, 0, 1] = z0 * tanh_gl
    line[:, 1, 0] = tanh_gl / z0
    matrix = _normalize(line, zref)

    series_r = float(fixture.get("series_r_ohm", 0.0))
    series_l = float(fixture.get("series_l_h", 0.0))
    shunt_g = float(fixture.get("shunt_g_s", 0.0))
    shunt_c = float(fixture.get("shunt_c_f", 0.0))
    matrix = _cascade(
        matrix,
        _series(series_r + 1.0j * omega * series_l, zref),
        zref,
    )
    return _cascade(
        matrix,
        _shunt(shunt_g + 1.0j * omega * shunt_c, zref),
        zref,
    )


def _to_s11(matrix: np.ndarray, zref: float,
             load: str | complex | float) -> np.ndarray:
    a, b = matrix[:, 0, 0], matrix[:, 0, 1]
    c, d = matrix[:, 1, 0], matrix[:, 1, 1]
    if isinstance(load, str):
        if load == "open":
            return (a - zref * c) / (a + zref * c)
        if load == "short":
            return (b - zref * d) / (b + zref * d)
        raise ValueError("load must be 'open', 'short', or a complex impedance")

    load_impedance = complex(load)
    input_numerator = a * load_impedance + b
    input_denominator = c * load_impedance + d
    return (input_numerator - zref * input_denominator) / (
        input_numerator + zref * input_denominator
    )


def simulate(
    frequency_hz: np.ndarray,
    segments: Sequence[Mapping[str, float]],
    joints: Sequence[Mapping[str, float]] | None = None,
    fixture: Mapping[str, float] | None = None,
    route: str = "dg_v3_rlgc",
    reference_hz: float = 1.0e6,
    load: str | complex | float = "open",
    zref: float = 50.0,
) -> np.ndarray:
    """Return one-port S11 for one physical route.

    Segment spans are in meters and their RLGC/Debye fields are per meter.
    Joints may fall inside a segment; the material block is split at that point.
    The fixture is placed before the cable and has delay_s, z0_ohm and
    one-way loss_db_ref; optional lumped R-L and G-C fields default to zero.
    The open/short load strings select ideal terminations; a number selects a
    finite load impedance.
    """
    if route not in ROUTES:
        raise ValueError(f"unknown route: {route!r}")

    frequency_hz = np.asarray(frequency_hz, dtype=np.float64)
    matrix = _fixture(frequency_hz, fixture, reference_hz, zref)
    ordered_segments = sorted(segments, key=lambda item: float(item["start_m"]))
    ordered_joints = sorted(joints or (), key=lambda item: float(item["position_m"]))
    joint_index = 0

    for segment in ordered_segments:
        start_m = float(segment["start_m"])
        end_m = float(segment["end_m"])
        impedance, admittance = _segment_zy(
            frequency_hz, segment, reference_hz, route
        )
        cursor_m = start_m

        while joint_index < len(ordered_joints):
            position_m = float(ordered_joints[joint_index]["position_m"])
            if position_m > end_m + 1.0e-9:
                break
            if position_m > cursor_m:
                length_m = position_m - cursor_m
                block = (
                    _line(impedance, admittance, length_m, zref)
                    if route != "cst_v1_ladder_0p1m"
                    else _ladder_line(impedance, admittance, length_m, zref)
                )
                matrix = _cascade(matrix, block, zref)
            matrix = _cascade(
                matrix, _joint(ordered_joints[joint_index], frequency_hz, zref), zref
            )
            cursor_m = position_m
            joint_index += 1

        if end_m > cursor_m:
            length_m = end_m - cursor_m
            block = (
                _line(impedance, admittance, length_m, zref)
                if route != "cst_v1_ladder_0p1m"
                else _ladder_line(impedance, admittance, length_m, zref)
            )
            matrix = _cascade(matrix, block, zref)

    while joint_index < len(ordered_joints):
        matrix = _cascade(
            matrix, _joint(ordered_joints[joint_index], frequency_hz, zref), zref
        )
        joint_index += 1

    return _to_s11(matrix, zref, load)


__all__ = ["LADDER_CELL_M", "ROUTES", "simulate"]
