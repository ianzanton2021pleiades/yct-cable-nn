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


def test_field_defect_policy_probabilities():
    dg = load_dg_module()
    expected = [
        (50.0, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (120.0, [0.45, 0.35, 0.17, 0.03, 0.0, 0.0]),
        (300.0, [0.28, 0.34, 0.24, 0.10, 0.03, 0.01]),
        (650.0, [0.22, 0.31, 0.27, 0.13, 0.05, 0.02]),
        (1000.0, [0.16, 0.28, 0.29, 0.17, 0.07, 0.03]),
        (2000.0, [0.12, 0.24, 0.29, 0.21, 0.10, 0.04]),
    ]
    for length_m, probabilities in expected:
        assert dg.defect_count_policy("field", length_m)["probabilities"] == probabilities


def test_rg58_random_policy_stays_limited_to_two_defects():
    dg = load_dg_module()
    rng = np.random.RandomState(20260623)
    counts = [dg.sample_defect_count("rg58_random", 160.0, rng) for _ in range(2000)]
    assert min(counts) >= 0
    assert max(counts) <= 2
    assert set(counts) == {0, 1, 2}


def test_generators_attach_defect_count_policy_metadata():
    dg = load_dg_module()
    field = dg.make_field_cable(np.random.RandomState(1), total_length=1600.0)
    rg58_random = dg.make_random_rg58_cable(np.random.RandomState(2), total_length=160.0)
    known_rg58 = dg.make_known_rg58_cable("rg58_74m", np.random.RandomState(3))

    assert field.defect_count_policy["profile"] == "field"
    assert len(field.defect_count_policy["probabilities"]) == 6
    assert 0 <= len(field.defect_info) <= 5

    assert rg58_random.defect_count_policy["profile"] == "rg58_random"
    assert 0 <= len(rg58_random.defect_info) <= 2

    assert known_rg58.defect_count_policy["profile"] == "rg58"
    assert known_rg58.defect_count_policy["source"] == "known_template"


def test_field_preview_override_keeps_defects_disabled():
    dg = load_dg_module()
    cable = dg.make_field_cable(np.random.RandomState(4), total_length=1800.0, n_defects_override=0)
    assert len(cable.defect_info) == 0
    assert cable.defect_count_policy["source"] == "override"
    assert cable.defect_count_policy["probabilities"] == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]


if __name__ == "__main__":
    test_field_defect_policy_probabilities()
    test_rg58_random_policy_stays_limited_to_two_defects()
    test_generators_attach_defect_count_policy_metadata()
    test_field_preview_override_keeps_defects_disabled()
    print("defect policy tests passed")
