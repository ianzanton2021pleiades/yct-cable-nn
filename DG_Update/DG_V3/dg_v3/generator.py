"""Public DG V3 generation API."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .config import GeneratorConfig, load_config
from .measurement import MeasurementChain
from .topology import CableTopology, build_topology


@dataclass(frozen=True)
class BandData:
    frequency_hz: np.ndarray
    s11: np.ndarray


@dataclass(frozen=True)
class GeneratedSample:
    sample_id: int
    profile: str
    topology: CableTopology
    bands: dict[str, BandData]
    truth: list[dict]
    generation: dict


def choose_profile(seed: int, requested: str, config: GeneratorConfig) -> str:
    if requested in {"rg58", "field"}:
        return requested
    if requested != "mixed":
        raise ValueError("profile must be mixed, rg58 or field")
    ratio = float(config.parameters["profiles"]["mixed_rg58_ratio"])
    return "rg58" if np.random.default_rng(seed).random() < ratio else "field"


def generate_sample(seed: int, profile: str = "mixed",
                    config: GeneratorConfig | None = None,
                    overrides: dict | None = None) -> GeneratedSample:
    """Generate one physically shared two-band sample.

    Both bands use exactly the same topology and measurement-chain parameters;
    only their correlated measurement-noise streams have independent seeds.
    """
    cfg = load_config() if config is None else config
    selected = choose_profile(seed, profile, cfg)
    topology = build_topology(selected, seed, cfg, overrides=overrides)
    chain = MeasurementChain.sample(selected, seed + 17_003, cfg)
    high = cfg.bands["1ghz"].frequencies()
    low = cfg.bands["200mhz"].frequencies()
    bands = {
        "1ghz": BandData(high, chain.evaluate(topology, high, seed + 31_001, low_band=False)),
        "200mhz": BandData(low, chain.evaluate(topology, low, seed + 47_003, low_band=True)),
    }
    generation = {
        "generator_version": cfg.generator_version,
        "parameter_profile": cfg.parameter_profile,
        "truth_delay_reference_hz": topology.delay_reference_hz,
        "measurement": chain.as_dict(),
        "topology": topology.as_dict(),
    }
    return GeneratedSample(
        seed,
        selected,
        topology,
        bands,
        [record.as_dict() for record in topology.truth_records()],
        generation,
    )


def iter_samples(count: int, seed: int = 0, profile: str = "mixed",
                 config: GeneratorConfig | None = None, workers: int = 1) -> Iterator[GeneratedSample]:
    """Yield deterministic samples, optionally using a small worker pool."""
    if count < 1:
        raise ValueError("count must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")
    cfg = load_config() if config is None else config
    sample_seeds = [seed + index for index in range(count)]
    if workers == 1:
        for sample_seed in sample_seeds:
            yield generate_sample(sample_seed, profile, cfg)
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        yield from executor.map(lambda item: generate_sample(item, profile, cfg), sample_seeds)
