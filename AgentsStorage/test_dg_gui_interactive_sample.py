from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "Src" / "DG_dataset_max2.5km.py"


def load_dg_module():
    spec = importlib.util.spec_from_file_location("dg_dataset_max2p5km", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_interactive_sample_fills_random_defaults_and_keeps_selected_band():
    dg = load_dg_module()

    result = dg.generate_interactive_sample({
        "profile": "field",
        "band": "200MHz",
        "length_m": None,
        "n_defects": None,
        "allowed_defect_types": [],
        "epsr": None,
        "seed": 20260701,
        "real_data_root": str(ROOT / "missing-real-data-root"),
    })

    assert result["selected_band"] == "200MHz"
    assert result["seed"] == 20260701
    assert result["input_defaults"]["epsr_used_default"] is True
    assert result["metadata"]["epsr"] == 2.23
    assert result["metadata"]["total_length_m"] > 0
    assert result["metadata"]["n_defects"] == len(result["metadata"]["defects"])
    assert len(result["band"]["freq_hz"]) == 5000
    assert len(result["band"]["s11"]) == 5000
    assert len(result["band"]["distance"]) == len(result["band"]["impulse"])
    assert len(result["band"]["distance"]) == len(result["band"]["step"])


def test_interactive_sample_honors_allowed_type_collection_for_field():
    dg = load_dg_module()

    result = dg.generate_interactive_sample({
        "profile": "field",
        "band": "1GHz",
        "length_m": 1200.0,
        "n_defects": 3,
        "allowed_defect_types": ["short", "moisture_local"],
        "epsr": 2.45,
        "seed": 20260702,
        "real_data_root": str(ROOT / "missing-real-data-root"),
    })

    defect_types = [defect["type"] for defect in result["metadata"]["defects"]]
    assert result["metadata"]["epsr"] == 2.45
    assert result["metadata"]["n_defects"] == 3
    assert set(defect_types).issubset({"short", "moisture_local"})
    assert len(result["band"]["freq_hz"]) == 50000


def test_interactive_rg58_downgrades_long_defect_types_to_short():
    dg = load_dg_module()

    result = dg.generate_interactive_sample({
        "profile": "rg58",
        "band": "1GHz",
        "length_m": 100.0,
        "n_defects": 2,
        "allowed_defect_types": ["aging", "moisture_distributed"],
        "epsr": 2.23,
        "seed": 20260703,
        "real_data_root": str(ROOT / "missing-real-data-root"),
    })

    assert result["profile"] == "rg58_random"
    assert result["warnings"]
    assert result["metadata"]["n_defects"] == 2
    assert {defect["type"] for defect in result["metadata"]["defects"]} == {"short"}


if __name__ == "__main__":
    test_interactive_sample_fills_random_defaults_and_keeps_selected_band()
    test_interactive_sample_honors_allowed_type_collection_for_field()
    test_interactive_rg58_downgrades_long_defect_types_to_short()
    print("interactive sample smoke tests passed")
