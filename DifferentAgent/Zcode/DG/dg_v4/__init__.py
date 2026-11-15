"""DG V4: field-realistic S11 dataset generator.

Upgrades over DG V3 (validated RLGC/ABCD core retained):
- native instrument log-frequency grids (9 kHz-1 GHz 50000/10000 pt, 9 kHz-200 MHz)
- field measurement chain re-scaled to real clamp-fixture + wideband analyzer levels
- five defect mechanisms (moisture local/distributed, aging, low-resistance short,
  series-resistance/shield damage, void) with enforced mechanism quotas
- rejection sampling instead of hard failures
- response-derived labels with per-event observability
"""
from .generate import generate_sample, iter_samples

__version__ = "4.0.0"
