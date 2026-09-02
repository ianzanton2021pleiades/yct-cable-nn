from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from build_responses import build_responses
from dataset import CableDataset
from storage import DatasetProtocolError, make_dataset_metadata, write_annotations, write_dataset_metadata, write_frequency
from validate_dataset import validate_dataset


def terminal_event(length_m: float, epsr: float) -> dict:
    delay = 2.0 * length_m * np.sqrt(epsr) / 299_792_458.0
    return {
        "event_id": "terminal_0", "role": "terminal", "geometry": "point", "mechanism": "open",
        "physical_start_m": length_m, "physical_center_m": length_m, "physical_end_m": length_m,
        "delay_start_s": delay, "delay_center_s": delay, "delay_end_s": delay,
        "severity": 1.0, "electrical_change": {"load_ohm": 1000.0},
    }


def make_dataset(root: Path) -> None:
    metadata = make_dataset_metadata(point_counts={"1ghz": 5, "200mhz": 5}, requested_samples=3)
    metadata["split_counts"] = {"train": 1, "val": 1, "test": 1}
    write_dataset_metadata(root, metadata)
    for split in ("train", "val", "test"):
        sample_id = f"sample_{split}"
        record = {
            "sample_id": sample_id, "split": split, "profile": "field", "seed": 1,
            "physical_length_m": 10.0, "reference_epsr": 2.3,
            "reference_impedance_ohm": 50.0, "termination": "open",
            "frequency_files": {
                "1ghz": f"frequency/{split}/1ghz/{sample_id}.npz",
                "200mhz": f"frequency/{split}/200mhz/{sample_id}.npz",
            },
            "generation": {"generator_version": "3.0.0", "parameter_profile": "test"},
            "events": [terminal_event(10.0, 2.3)],
        }
        write_annotations(root, split, [record])
        frequency = np.arange(9_000.0, 45_001.0, 9_000.0)
        signal = 0.1 * np.exp(1j * frequency / 50_000.0)
        for band in ("1ghz", "200mhz"):
            write_frequency(root, split, band, sample_id, frequency, signal.real, signal.imag)


class ProtocolTests(unittest.TestCase):
    def test_round_trip_response_and_numpy_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_dataset(root)
            self.assertEqual(build_responses(root), 6)
            report = validate_dataset(root)
            self.assertEqual(report["version"], "3.0")
            record = CableDataset(root, "train")[0]
            self.assertEqual(record["frequency"]["1ghz"]["frequency_hz"].dtype, np.float64)
            self.assertEqual(record["responses"]["1ghz"]["distance_m"].dtype, np.float64)
            self.assertTrue(np.allclose(np.diff(record["responses"]["1ghz"]["distance_m"]), 0.25))
            self.assertLessEqual(record["responses"]["1ghz"]["distance_m"][-1], 12.0)

    def test_schema_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_dataset(root)
            metadata_path = root / "dataset.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["version"] = "2.7"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaises(DatasetProtocolError):
                validate_dataset(root)


if __name__ == "__main__":
    unittest.main()
