"""Frequency-domain network calculation for the DG V3 RLGC cable model."""
from __future__ import annotations

import numpy as np

from .topology import CableSegment, CableTopology, Joint, segment_propagation


def _cascade(left: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
             right: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
    a, b, c, d = left
    e, f, g, h = right
    return a * e + b * g, a * f + b * h, c * e + d * g, c * f + d * h


def _identity(size: int):
    ones = np.ones(size, dtype=np.complex128)
    zeros = np.zeros(size, dtype=np.complex128)
    return ones, zeros, zeros, ones


def segment_abcd(frequencies_hz: np.ndarray, segment: CableSegment,
                 dispersion_fraction: float | None = None) -> tuple[np.ndarray, ...]:
    """ABCD matrix derived from one segment's frequency-dependent RLGC.

    ``dispersion_fraction`` remains an ignored compatibility argument for
    callers of the old V3 function. Dispersion now comes only from RLGC.
    """
    gamma, zc = segment_propagation(frequencies_hz, segment)
    gl = gamma * segment.length_m
    cosh_gl = np.cosh(gl)
    sinh_gl = np.sinh(gl)
    return cosh_gl, zc * sinh_gl, sinh_gl / zc, cosh_gl


def joint_abcd(frequencies_hz: np.ndarray, joint: Joint):
    f = np.asarray(frequencies_hz, dtype=np.float64)
    series = joint.series_ohm * np.ones_like(f, dtype=np.complex128)
    shunt = 1j * 2.0 * np.pi * f * joint.shunt_pf * 1.0e-12
    return 1.0 + series * shunt, series, shunt, np.ones_like(series)


def topology_abcd(frequencies_hz: np.ndarray, topology: CableTopology):
    f = np.asarray(frequencies_hz, dtype=np.float64)
    result = _identity(f.size)
    joints_by_position = {round(j.position_m, 9): j for j in topology.joints}
    for segment in topology.segments:
        result = _cascade(result, segment_abcd(f, segment))
        joint = joints_by_position.get(round(segment.end_m, 9))
        if joint is not None and segment.end_m < topology.length_m:
            result = _cascade(result, joint_abcd(f, joint))
    return result


def fixture_abcd(frequencies_hz: np.ndarray, z_ref_ohm: float,
                 series_ohm: float, shunt_pf: float, delay_ns: float,
                 loss_db_at_100mhz: float):
    """Two-port fixture: lossy delay line followed by series/shunt parasitics."""
    f = np.asarray(frequencies_hz, dtype=np.float64)
    theta = 2.0 * np.pi * f * delay_ns * 1.0e-9
    attenuation = np.log(10.0) / 20.0 * loss_db_at_100mhz * f / 100.0e6
    gamma_l = attenuation + 1j * theta
    delay = np.cosh(gamma_l), z_ref_ohm * np.sinh(gamma_l), \
        np.sinh(gamma_l) / z_ref_ohm, np.cosh(gamma_l)
    series = series_ohm * np.ones_like(f, dtype=np.complex128)
    shunt = 1j * 2.0 * np.pi * f * shunt_pf * 1.0e-12
    parasitic = (1.0 + series * shunt, series, shunt, np.ones_like(series))
    return _cascade(delay, parasitic)


def apply_two_port_chain(network: tuple[np.ndarray, ...], near_fixture: tuple[np.ndarray, ...]):
    return _cascade(near_fixture, network)


def network_s11(network: tuple[np.ndarray, ...], z_load_ohm: complex, z_ref_ohm: float):
    a, b, c, d = network
    zin = (a * z_load_ohm + b) / (c * z_load_ohm + d)
    return (zin - z_ref_ohm) / (zin + z_ref_ohm)
