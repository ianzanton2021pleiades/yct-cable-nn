from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "Src" / "DG_dataset_max2.5km.py"


def load_dg_module():
    spec = importlib.util.spec_from_file_location("dg_dataset_max2p5km", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_field_long_defect_policy_is_length_dependent():
    dg = load_dg_module()
    expected = [
        (200.0, 0.0, 1),
        (400.0, 0.03, 1),
        (650.0, 0.06, 1),
        (1000.0, 0.10, 2),
        (1800.0, 0.16, 2),
    ]
    for length_m, probability, max_long in expected:
        policy = dg.defect_type_policy("field", length_m)
        assert policy["long_probability_per_defect"] == probability
        assert policy["max_long_defects"] == max_long


def test_rg58_profiles_do_not_generate_long_defects():
    dg = load_dg_module()
    for profile in ["rg58", "rg58_random"]:
        policy = dg.defect_type_policy(profile, 160.0)
        assert policy["long_probability_per_defect"] == 0.0
        assert policy["allowed_types"] == ["short"]


def test_field_defect_type_sampler_limits_long_defects():
    dg = load_dg_module()
    rng = np.random.RandomState(20260623)
    for _ in range(200):
        types = dg.sample_defect_types("field", 1800.0, 5, rng)
        assert len(types) == 5
        assert sum(t in {"aging", "moisture_local", "moisture_distributed"} for t in types) <= 2
        assert set(types).issubset({"short", "aging", "moisture_local", "moisture_distributed"})


def test_metadata_contains_type_and_interval_fields():
    dg = load_dg_module()
    found_long = None
    for seed in range(200):
        cable = dg.make_field_cable(np.random.RandomState(seed), total_length=2200.0, n_defects_override=5)
        long_defects = [d for d in cable.defect_info if d["type"] in {"aging", "moisture_local", "moisture_distributed"}]
        if long_defects:
            found_long = long_defects[0]
            break
    assert found_long is not None
    for key in ["type", "start", "end", "position", "length", "z0", "epsr", "alpha", "severity"]:
        assert key in found_long
    assert found_long["end"] > found_long["start"]
    assert found_long["length"] >= 15.0


def test_moisture_variants_are_explicit_and_distributed_avoids_many_hard_boundaries():
    dg = load_dg_module()
    policy = dg.defect_type_policy("field", 1800.0)
    assert "moisture_local" in policy["allowed_types"]
    assert "moisture_distributed" in policy["allowed_types"]

    case = {
        "name": "test distributed",
        "profile": "field",
        "length_m": 1800.0,
        "defects": [
            {
                "type": "moisture_distributed",
                "start_m": 420.0,
                "length_m": 900.0,
                "z0_mult": 0.90,
                "epsr_delta": 0.60,
                "alpha_mult": 4.8,
                "label_amplitude": 0.70,
            }
        ],
    }
    cable = dg.cable_from_defect_case(case)
    defect_segments = [s for s in cable.segments if s.is_defect]
    assert len(defect_segments) == 0
    assert len(getattr(cable, "distributed_moisture_regions", [])) == 1
    assert len(cable.defect_info) == 1
    assert cable.defect_info[0]["type"] == "moisture_distributed"


def test_long_defect_label_covers_interval_not_single_peak():
    dg = load_dg_module()
    segs = [
        dg._segment(100.0, 52.0, 2.3, 0.01, False),
        dg._segment(80.0, 48.0, 2.8, 0.05, True),
        dg._segment(220.0, 52.0, 2.3, 0.01, False),
    ]
    segs[1].defect_type = "moisture"
    segs[1].label_amplitude = 0.65
    cable = dg.CableSample(segments=segs, epsr=2.3)
    label = dg.build_label(cable)
    covered = int(np.sum(label[400:720] > 0.40))
    assert covered > 120


def test_long_field_phase_preview_retains_low_band_wrapped_periodicity():
    dg = load_dg_module()
    rows = dg.generate_defect_case_rows(seed=20260623)
    field_rows = [row for row in rows if row["profile"] == "field" and row["cable"].total_length >= 900.0]
    assert field_rows
    for row in field_rows:
        mask = row["f1"] <= 500e6
        phase = dg.s11_wrapped_phase_deg(row["s1"][mask])
        assert float(np.nanpercentile(phase, 95) - np.nanpercentile(phase, 5)) > 35.0


def test_field_phase_retention_does_not_dominate_realpart():
    dg = load_dg_module()
    case = next(c for c in dg.build_defect_case_specs() if c["name"] == "Field medium healthy")
    cable = dg.cable_from_defect_case(case)
    freq_hz, clean = dg.generate_s11(
        cable,
        dg.SWEEP_1GHZ,
        rng=np.random.RandomState(0),
        add_noise=False,
        inject_joints=False,
    )
    adjusted = dg.apply_field_lowfreq_coherent_phase(clean, freq_hz, cable, np.random.RandomState(11))
    mask = freq_hz <= 500e6
    clean_real_span = float(np.nanpercentile(clean[mask].real, 95) - np.nanpercentile(clean[mask].real, 5))
    adjusted_real_span = float(np.nanpercentile(adjusted[mask].real, 95) - np.nanpercentile(adjusted[mask].real, 5))
    clean_mag_p95 = float(np.nanpercentile(np.abs(clean[mask]), 95))
    adjusted_mag_p95 = float(np.nanpercentile(np.abs(adjusted[mask]), 95))
    assert adjusted_real_span <= clean_real_span * 2.0
    assert adjusted_mag_p95 <= clean_mag_p95 * 1.8


if __name__ == "__main__":
    test_field_long_defect_policy_is_length_dependent()
    test_rg58_profiles_do_not_generate_long_defects()
    test_field_defect_type_sampler_limits_long_defects()
    test_metadata_contains_type_and_interval_fields()
    test_moisture_variants_are_explicit_and_distributed_avoids_many_hard_boundaries()
    test_long_defect_label_covers_interval_not_single_peak()
    test_long_field_phase_preview_retains_low_band_wrapped_periodicity()
    test_field_phase_retention_does_not_dominate_realpart()
    print("defect type policy tests passed")
