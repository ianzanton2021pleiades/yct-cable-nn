"""Versioned configuration and public data types for DG V3.

The V3 generator deliberately has no implicit V2 compatibility path.  A
configuration is a complete description of the sweep, topology priors and
measurement chain and is required at construction time.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class BandConfig:
    name: str
    start_hz: float
    stop_hz: float
    points: int

    def frequencies(self):
        import numpy as np
        return np.linspace(self.start_hz, self.stop_hz, self.points, dtype=np.float64)


@dataclass(frozen=True)
class GeneratorConfig:
    schema_version: str
    generator_version: str
    parameter_profile: str
    bands: Mapping[str, BandConfig]
    parameters: Mapping[str, Any]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GeneratorConfig":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
        required = {"schema_version", "generator_version", "parameter_profile", "bands", "parameters"}
        missing = required.difference(raw)
        if missing:
            raise ValueError(f"DG V3 config missing keys: {sorted(missing)}")
        if raw["schema_version"] != "dg-v3":
            raise ValueError(f"unsupported DG V3 schema: {raw['schema_version']!r}")
        bands = {}
        for name in ("1ghz", "200mhz"):
            value = raw["bands"].get(name)
            if value is None or not {"start_hz", "stop_hz", "points"}.issubset(value):
                raise ValueError(f"DG V3 config missing complete {name} band")
            bands[name] = BandConfig(name, float(value["start_hz"]),
                                     float(value["stop_hz"]), int(value["points"]))
        parameters = raw["parameters"]
        for key in ("profiles", "defects", "joints", "fixture", "noise"):
            if key not in parameters:
                raise ValueError(f"DG V3 config missing parameters.{key}")
        profiles = parameters["profiles"]
        expected_models = {"rg58": "coax_rlgc", "field": "effective_rlgc"}
        for profile, model in expected_models.items():
            spec = profiles.get(profile)
            if spec is None:
                raise ValueError(f"DG V3 config missing parameters.profiles.{profile}")
            material = spec.get("material")
            if not isinstance(material, Mapping) or material.get("model") != model:
                raise ValueError(
                    f"DG V3 {profile} profile requires material.model={model!r}"
                )
        for mechanism in ("aging", "moisture_local", "moisture_distributed"):
            defect = parameters["defects"].get(mechanism)
            if not isinstance(defect, Mapping) or not isinstance(defect.get("debye"), Mapping):
                raise ValueError(f"DG V3 defect {mechanism!r} requires a debye block")
        return cls(
            str(raw["schema_version"]),
            str(raw["generator_version"]),
            str(raw["parameter_profile"]),
            bands,
            parameters,
        )


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "provisional_rlgc_v1.yaml"


def load_config(path: str | Path | None = None) -> GeneratorConfig:
    """Load the checked-in, versioned configuration (or an explicit path)."""
    return GeneratorConfig.from_yaml(default_config_path() if path is None else path)
