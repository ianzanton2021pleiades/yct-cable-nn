"""Frequency-domain ABCD network calculation (DG V3 core, retained after its
RG58 validation: aligned complex rms 0.068, end position error 0.05 m, 30/30
held-out wins).  Segments cascade as ABCD; joints and localized two-port
defects are series/shunt lumps.
"""
from __future__ import annotations

import numpy as np

from .materials import SegmentMaterial, SegmentSpec, segment_propagation


def segment_abcd(f: np.ndarray, spec: SegmentSpec, material: SegmentMaterial):
    gamma, zc = segment_propagation(f, spec, material)
    gl = gamma * spec.length_m
    ch = np.cosh(gl)
    sh = np.sinh(gl)
    return ch, zc * sh, sh / zc, ch


def lump_abcd(f: np.ndarray, series_ohm: float | np.ndarray,
              shunt_pf: float | np.ndarray):
    """Series impedance + shunt capacitance two-port (joint / local defect)."""
    series = np.asarray(series_ohm, dtype=np.complex128) * np.ones_like(f)
    shunt = 1j * 2.0 * np.pi * f * np.asarray(shunt_pf, dtype=np.float64) * 1.0e-12
    return 1.0 + series * shunt, series, shunt, np.ones_like(f)


def cascade(left, right):
    a, b, c, d = left
    e, f2, g, h = right
    return a * e + b * g, a * f2 + b * h, c * e + d * g, c * f2 + d * h


def network_s11(network, z_load_ohm: complex, z_ref_ohm: float):
    a, b, c, d = network
    zin = (a * z_load_ohm + b) / (c * z_load_ohm + d)
    return (zin - z_ref_ohm) / (zin + z_ref_ohm)
