"""DG V3: direct complex-S11 physical data generator."""
from .config import BandConfig, GeneratorConfig, default_config_path, load_config
from .generator import BandData, GeneratedSample, generate_sample, iter_samples
from .topology import (
    CableTopology, DefectRegion, Joint, RLGCMaterial, TruthRecord,
    build_topology, derive_material,
)

__all__ = [
    "BandConfig", "GeneratorConfig", "default_config_path", "load_config",
    "BandData", "GeneratedSample", "generate_sample", "iter_samples",
    "CableTopology", "DefectRegion", "Joint", "RLGCMaterial", "TruthRecord",
    "build_topology", "derive_material",
]
