"""Small, local tests for DG V3 calibration (no E: corpus access)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from calibrate_real_data import (  # noqa: E402
    calibrate,
    classify_file,
    detect_isolated_first_frequency,
    read_s11_csv,
)


def write_csv(path: Path, header: str = "Frequency,Real,Imag", offset: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        "1000000,0.10,0.01",
        "2000000,0.11,0.01",
        "3000000,0.12,0.02",
        "4000000,0.13,0.02",
        "5000000,0.14,0.03",
        "6000000,0.15,0.03",
    ]
    if offset:
        rows = [f"{int(float(row.split(',')[0]))},{float(row.split(',')[1]) + offset}, {row.split(',')[2]}" for row in rows]
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


class CalibrationTests(unittest.TestCase):
    def test_observed_headers_and_rg58_first_grid_point(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "RG58-demo.csv"
            path.write_text(
                "Frequency_Hz,Real,Imag\n"
                "100000,0.1,0.0\n"
                "1000000,0.2,0.0\n"
                "1100000,0.2,0.0\n"
                "1200000,0.2,0.0\n"
                "1300000,0.2,0.0\n",
                encoding="utf-8",
            )
            freq, value = read_s11_csv(path, category="rg58")
            self.assertTrue(detect_isolated_first_frequency(np.array([100000, 1000000, 1100000, 1200000, 1300000], dtype=float)))
            self.assertEqual(len(freq), 4)
            self.assertEqual(freq[0], 1_000_000)
            self.assertAlmostEqual(value[0].real, 0.2)

    def test_classification_and_scope_exclusion(self) -> None:
        self.assertEqual(classify_file(Path("RG58-3Lines") / "line.csv"), "rg58")
        self.assertEqual(classify_file(Path("1500m接头浸水") / "line.csv"), "wet_1500m")
        self.assertEqual(classify_file(Path("普通Field") / "line-校正数据.csv"), "correction")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "corpus"
            out = Path(temp) / "out"
            write_csv(root / "RG58-3Lines" / "line_1.csv")
            write_csv(root / "RG58-3Lines" / "line_2.csv", offset=0.001)
            write_csv(root / "普通Field" / "500m" / "field_1.csv")
            write_csv(root / "普通Field" / "3000m" / "too_long.csv")
            write_csv(root / "1500m接头浸水" / "wet.csv")
            write_csv(root / "普通Field" / "field-校正数据.csv")
            write_csv(root / "IFFT现场数据汇总" / "must_not_read.csv")
            summary = calibrate(root, out, output_format="both", progress_every=2)
            self.assertEqual(summary["scan"]["excluded_ifft_csv"], 1)
            self.assertEqual(summary["scan"]["excluded_field_over_2500m"], 1)
            self.assertEqual(summary["scan"]["categories"]["rg58"], 2)
            self.assertEqual(summary["scan"]["categories"]["field"], 1)
            self.assertEqual(summary["scan"]["categories"]["wet_1500m"], 1)
            self.assertEqual(summary["scan"]["categories"]["correction"], 1)
            self.assertEqual(summary["repeat_noise_statistics"]["repeat_groups"], 1)
            self.assertTrue((out / "calibration_summary.json").is_file())
            self.assertTrue((out / "calibration_summary.yaml").is_file())
            loaded = json.loads((out / "calibration_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(loaded["schema_version"], "dg-v3-calibration-1")
            # Output must not carry per-curve names or row-level residuals.
            output_text = (out / "calibration_summary.json").read_text(encoding="utf-8")
            self.assertNotIn("line_1", output_text)
            self.assertNotIn("residual", output_text.lower())


if __name__ == "__main__":
    unittest.main()
