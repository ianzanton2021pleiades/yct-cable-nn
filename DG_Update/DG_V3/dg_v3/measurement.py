"""Measurement-chain model: two-port fixtures, one-port VNA errors, correlated noise."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import GeneratorConfig
from .physics import fixture_abcd, network_s11, topology_abcd, apply_two_port_chain
from .topology import CableTopology


@dataclass(frozen=True)
class FixtureParams:
    series_ohm: float
    shunt_pf: float
    delay_ns: float
    loss_db_at_100mhz: float

    def as_dict(self) -> dict[str, float]:
        return {
            "series_ohm": self.series_ohm,
            "shunt_pf": self.shunt_pf,
            "delay_ns": self.delay_ns,
            "loss_db_at_100mhz": self.loss_db_at_100mhz,
        }


@dataclass(frozen=True)
class VnaErrorParams:
    directivity: float
    source_match: float
    tracking: float
    directivity_phase_rad: float
    source_match_phase_rad: float
    tracking_phase_rad: float

    def as_dict(self) -> dict[str, float]:
        return {
            "directivity": self.directivity,
            "source_match": self.source_match,
            "tracking": self.tracking,
            "directivity_phase_rad": self.directivity_phase_rad,
            "source_match_phase_rad": self.source_match_phase_rad,
            "tracking_phase_rad": self.tracking_phase_rad,
        }


@dataclass(frozen=True)
class MeasurementChain:
    near_fixture: FixtureParams
    vna: VnaErrorParams
    noise_sigma: float
    noise_correlation: float
    low_frequency_factor: float

    @classmethod
    def sample(cls, profile: str, seed: int, config: GeneratorConfig) -> "MeasurementChain":
        rng = np.random.default_rng(seed)
        p = config.parameters
        fixture = p["fixture"]

        def draw_fixture() -> FixtureParams:
            return FixtureParams(
                float(rng.uniform(fixture["series_ohm"]["min"], fixture["series_ohm"]["max"])),
                float(rng.uniform(fixture["shunt_pf"]["min"], fixture["shunt_pf"]["max"])),
                float(rng.uniform(fixture["delay_ns"]["min"], fixture["delay_ns"]["max"])),
                float(rng.uniform(fixture["loss_db_at_100mhz"]["min"], fixture["loss_db_at_100mhz"]["max"])),
            )
        noise = p["noise"]
        sigma_key = "rg58_sigma" if profile == "rg58" else "field_sigma"
        return cls(
            draw_fixture(),
            VnaErrorParams(
                float(rng.uniform(fixture["vna_directivity"]["min"], fixture["vna_directivity"]["max"])),
                float(rng.uniform(fixture["vna_source_match"]["min"], fixture["vna_source_match"]["max"])),
                float(rng.uniform(fixture["vna_tracking_error"]["min"], fixture["vna_tracking_error"]["max"])),
                float(rng.uniform(-np.pi, np.pi)),
                float(rng.uniform(-np.pi, np.pi)),
                float(rng.uniform(
                    fixture["vna_tracking_phase_rad"]["min"],
                    fixture["vna_tracking_phase_rad"]["max"],
                )),
            ),
            float(rng.uniform(noise[sigma_key]["min"], noise[sigma_key]["max"])),
            float(rng.uniform(noise["correlation"]["min"], noise["correlation"]["max"])),
            float(rng.uniform(noise["low_frequency_factor"]["min"], noise["low_frequency_factor"]["max"])),
        )

    def as_dict(self) -> dict:
        return {
            "near_fixture": self.near_fixture.as_dict(),
            "vna": self.vna.as_dict(),
            "noise_sigma": self.noise_sigma,
            "noise_correlation": self.noise_correlation,
            "low_frequency_factor": self.low_frequency_factor,
        }

    def _correlated_noise(self, size: int, rng: np.random.Generator, scale: float):
        white = (rng.standard_normal(size) + 1j * rng.standard_normal(size)) / np.sqrt(2.0)
        output = np.empty(size, dtype=np.complex128)
        output[0] = white[0]
        innovation = np.sqrt(1.0 - self.noise_correlation ** 2)
        for index in range(1, size):
            output[index] = self.noise_correlation * output[index - 1] + innovation * white[index]
        return scale * output

    def evaluate(self, topology: CableTopology, frequencies_hz: np.ndarray,
                 noise_seed: int, low_band: bool = False) -> np.ndarray:
        frequencies = np.asarray(frequencies_hz, dtype=np.float64)
        physical = topology_abcd(frequencies, topology)
        near = fixture_abcd(frequencies, topology.z_ref_ohm, self.near_fixture.series_ohm,
                            self.near_fixture.shunt_pf, self.near_fixture.delay_ns,
                            self.near_fixture.loss_db_at_100mhz)
        measured_network = apply_two_port_chain(physical, near)
        gamma = network_s11(measured_network, topology.z_load_ohm, topology.z_ref_ohm)
        f_norm = frequencies / max(float(frequencies[-1]), 1.0)
        vna = self.vna
        e00 = vna.directivity * np.exp(1j * vna.directivity_phase_rad) * (1.0 + 0.10 * f_norm)
        e11 = vna.source_match * np.exp(1j * vna.source_match_phase_rad) * (1.0 + 0.06 * f_norm)
        tracking = vna.tracking * np.exp(1j * vna.tracking_phase_rad) * np.exp(-0.015 * f_norm)
        corrected = e00 + tracking * gamma / (1.0 - e11 * gamma)
        noise_rng = np.random.default_rng(noise_seed)
        # A bounded low-frequency rise models calibration drift without
        # turning the 9 kHz endpoint into an unphysical multi-percent error.
        frequency_scale = 1.0 + 0.35 / (1.0 + np.sqrt(np.maximum(frequencies, 1.0) / 100.0e6))
        if low_band:
            frequency_scale *= self.low_frequency_factor
        return corrected + self._correlated_noise(frequencies.size, noise_rng,
                                                   self.noise_sigma * frequency_scale)
