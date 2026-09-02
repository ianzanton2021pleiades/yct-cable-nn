from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dg_v3 import generate_sample, iter_samples, load_config
from dg_v3.topology import build_topology
from response import s11_to_responses


class SynthesisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(ROOT / "configs" / "provisional_rlgc_v1.yaml")

    def test_bands_profiles_and_determinism(self) -> None:
        first = generate_sample(1234, "field", self.config)
        second = generate_sample(1234, "field", self.config)
        self.assertEqual(len(first.bands["1ghz"].frequency_hz), 50_000)
        self.assertEqual(len(first.bands["200mhz"].frequency_hz), 6_250)
        np.testing.assert_array_equal(first.bands["1ghz"].s11, second.bands["1ghz"].s11)
        parallel = list(iter_samples(2, 2000, "field", self.config, workers=2))
        serial = list(iter_samples(2, 2000, "field", self.config, workers=1))
        for left, right in zip(parallel, serial):
            np.testing.assert_array_equal(left.bands["200mhz"].s11, right.bands["200mhz"].s11)
        rg58 = generate_sample(7, "rg58", self.config)
        self.assertGreaterEqual(rg58.topology.length_m, 10.0)
        self.assertLessEqual(rg58.topology.length_m, 200.0)
        self.assertTrue(np.isfinite(first.bands["1ghz"].s11).all())

    def test_sparse_truth_for_four_defect_mechanisms(self) -> None:
        for mechanism in ("short", "aging", "moisture_local", "moisture_distributed"):
            sample = generate_sample(
                42, "field", self.config,
                {"length_m": 1200.0, "epsr": 2.4, "defect_count": 1, "defect_type": mechanism, "termination": "open"},
            )
            defects = [event for event in sample.truth if event["role"] == "defect"]
            self.assertEqual(len(defects), 1)
            event = defects[0]
            self.assertEqual(event["mechanism"], mechanism)
            self.assertLessEqual(event["delay_start_s"], event["delay_center_s"])
            self.assertLessEqual(event["delay_center_s"], event["delay_end_s"])
            self.assertIn("z0_inside_ohm", event["electrical_change"])
            self.assertEqual(sum(item["role"] == "terminal" for item in sample.truth), 1)

    def test_200mhz_nominal_2500m_coverage_at_epsr_3p2(self) -> None:
        frequency = self.config.bands["200mhz"].frequencies()
        values = np.zeros_like(frequency, dtype=np.complex128)
        coverage = s11_to_responses(
            frequency, values, epsr=3.2, distance_step_m=0.25, target_distance_max_m=2500.0,
        )[-1]
        self.assertGreaterEqual(coverage["valid_distance_max_m"], 2499.75)
        self.assertFalse(coverage["truncated_by_ifft_range"])

    def test_length_stratified_topology_sampling_has_valid_placement(self) -> None:
        for seed in range(500):
            profile = "rg58" if seed % 5 == 0 else "field"
            topology = build_topology(profile, seed, self.config)
            self.assertTrue(all(segment.length_m > 0 for segment in topology.segments))
            self.assertTrue(all(0 < joint.position_m < topology.length_m for joint in topology.joints))


if __name__ == "__main__":
    unittest.main()
