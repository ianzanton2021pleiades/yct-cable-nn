"""Shared core API for DG V2.5.

The implementation of the Client-compatible IFFT lives only in
:mod:`core.tdr_signal`.  This module intentionally re-exports it instead of
keeping a second copy, so generator, loader, GUI, and inference cannot silently
diverge.
"""
from .tdr_signal import (
    SPEED_OF_LIGHT,
    Z0_REF,
    apply_window,
    build_equally_spaced_spectrum,
    compute_impedance,
    compute_step_response,
    estimate_first_step,
    fft_shift,
    read_s11_csv,
    s11_to_responses,
    spectrum_to_time,
    to_fixed_distance_grid,
)

__all__ = [
    "SPEED_OF_LIGHT",
    "Z0_REF",
    "apply_window",
    "build_equally_spaced_spectrum",
    "compute_impedance",
    "compute_step_response",
    "estimate_first_step",
    "fft_shift",
    "read_s11_csv",
    "s11_to_responses",
    "spectrum_to_time",
    "to_fixed_distance_grid",
]
