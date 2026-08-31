from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.dataset import CableDefectDataset
from core.tdr_signal import s11_to_responses


def _load_generator_module():
    path = PROJECT_ROOT / "[V2.1]DG_dataset_max2.5km.py"
    spec = importlib.util.spec_from_file_location("dg_v21_test_module", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DG = _load_generator_module()


@pytest.fixture(scope="module")
def representative_samples() -> dict[str, dict]:
    common = {
        "band": "1GHz",
        "window": "hann",
        "epsr": 2.23,
        # Keep invariants independent of the optional local reference corpus.
        "real_data_root": "/path/that/does/not/exist",
    }
    return {
        "rg58": DG.generate_interactive_sample(
            {
                **common,
                "profile": "rg58",
                "length_m": 50.0,
                "n_defects": 2,
                "allowed_defect_types": ["short"],
                "seed": 2015977662,
            }
        ),
        "field_short": DG.generate_interactive_sample(
            {
                **common,
                "profile": "field",
                "length_m": 1400.0,
                "n_defects": 1,
                "allowed_defect_types": ["short"],
                "seed": 1868816610,
            }
        ),
        "field_healthy": DG.generate_interactive_sample(
            {
                **common,
                "profile": "field",
                "length_m": 1400.0,
                "n_defects": 0,
                "allowed_defect_types": ["short"],
                "seed": 1868816610,
            }
        ),
        "field_moisture": DG.generate_interactive_sample(
            {
                **common,
                "profile": "field",
                "length_m": 1400.0,
                "n_defects": 1,
                "allowed_defect_types": ["moisture_distributed"],
                "seed": 440840743,
            }
        ),
    }


def test_every_saved_response_is_recomputed_only_from_s11(representative_samples):
    """The central V2.1 invariant must hold for both frequency grids."""
    for sample in representative_samples.values():
        epsr = float(sample["metadata"]["epsr"])
        for band in ("1GHz", "200MHz"):
            result = sample["bands"][band]
            distance, impulse, step, _ = s11_to_responses(
                result["freq_hz"], result["s11"], epsr=epsr, window="hann"
            )
            np.testing.assert_array_equal(distance, result["distance"])
            np.testing.assert_array_equal(impulse, result["impulse"])
            np.testing.assert_array_equal(step, result["step"])


def test_csv_round_trip_preserves_exact_ifft_inputs(tmp_path, representative_samples):
    """17-digit CSV output must round-trip without changing any response sample."""
    sample = representative_samples["field_short"]
    epsr = float(sample["metadata"]["epsr"])
    cable_length = float(sample["cable"].total_length)

    for band in ("1GHz", "200MHz"):
        result = sample["bands"][band]
        csv_path = tmp_path / f"roundtrip_{band}.csv"
        DG.save_client_csv(
            csv_path,
            result["freq_hz"],
            result["s11"],
            result["distance"],
            result["impulse"],
            result["step"],
            cable_length,
        )

        freq: list[float] = []
        s11: list[complex] = []
        distance_saved: list[float] = []
        impulse_saved: list[float] = []
        step_saved: list[float] = []
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["Frequency"]:
                    freq.append(float(row["Frequency"]))
                    s11.append(float(row["S11_Real"]) + 1j * float(row["S11_Imaginary"]))
                if row["Distance"]:
                    distance_saved.append(float(row["Distance"]))
                    impulse_saved.append(float(row["ImpulseResponse"]))
                    step_saved.append(float(row["StepResponse"]))

        distance, impulse, step, _ = s11_to_responses(
            np.asarray(freq), np.asarray(s11), epsr=epsr, window="hann"
        )
        n = len(distance_saved)
        np.testing.assert_array_equal(distance[:n], np.asarray(distance_saved))
        np.testing.assert_array_equal(impulse.real[:n], np.asarray(impulse_saved))
        np.testing.assert_array_equal(step[:n], np.asarray(step_saved))


def test_rg58_front_end_is_dense_and_localized(representative_samples):
    """RG58 front-end texture should be a dense decaying cluster, not 3-5 sparse peaks."""
    result = representative_samples["rg58"]["bands"]["1GHz"]
    distance = result["distance"]
    impulse = result["impulse"].real

    near = impulse[(distance >= 0.0) & (distance <= 10.0)]
    peak = float(np.max(np.abs(near)))
    a, b = near[:-1], near[1:]
    meaningful = (np.abs(a) > 0.01 * peak) | (np.abs(b) > 0.01 * peak)
    zero_crossings = int(np.count_nonzero((a * b < 0.0) & meaningful))
    assert zero_crossings >= 40

    # The texture must decay rather than remain broadband over the whole cable.
    early_rms = float(np.sqrt(np.mean(impulse[(distance >= 0.0) & (distance <= 5.0)] ** 2)))
    middle_rms = float(np.sqrt(np.mean(impulse[(distance >= 12.0) & (distance <= 17.0)] ** 2)))
    assert early_rms > 2.0 * middle_rms


def test_field_remote_defect_has_no_large_high_frequency_burst(representative_samples):
    """Only tiny noise may oscillate around a remote short-defect event."""
    sample = representative_samples["field_short"]
    result = sample["bands"]["1GHz"]
    distance = result["distance"]
    impulse = result["impulse"].real
    center = float(sample["metadata"]["defects"][0]["center_m"])
    local = impulse[(distance >= center - 30.0) & (distance <= center + 30.0)]
    peak = float(np.max(np.abs(local)))
    a, b = local[:-1], local[1:]
    substantial = (np.abs(a) > 0.003 * peak) | (np.abs(b) > 0.003 * peak)
    substantial_crossings = int(np.count_nonzero((a * b < 0.0) & substantial))
    assert substantial_crossings <= 3


def test_field_open_terminal_has_a_visible_signed_step(representative_samples):
    sample = representative_samples["field_healthy"]
    assert sample["metadata"]["termination"] == "open"
    result = sample["bands"]["1GHz"]
    distance = result["distance"]
    step = result["step"]
    terminal = float(sample["cable"].total_length)
    delta = DG._step_delta_near(distance, step, terminal, 70.0)
    assert delta is not None
    assert delta > 7.0e-11



def test_distributed_moisture_is_physical_and_has_effective_terminal(representative_samples):
    sample = representative_samples["field_moisture"]
    result = sample["bands"]["1GHz"]
    distance = result["distance"]
    step = result["step"]
    defect = sample["metadata"]["defects"][0]
    start = float(defect["start_m"])
    end = float(defect["end_m"])

    pre = (distance >= start - 100.0) & (distance <= start - 20.0)
    interior = (distance >= start + 0.25 * (end - start)) & (distance <= start + 0.75 * (end - start))
    assert float(np.median(step[interior])) < float(np.median(step[pre])) - 1.0e-11

    effective_end = DG.effective_terminal_phase_length_m(sample["cable"])
    assert effective_end > float(sample["cable"].total_length)
    delta = DG._step_delta_near(distance, step, effective_end, 70.0)
    assert delta is not None
    assert delta > 7.0e-11


def test_template_phase_cannot_copy_a_coherent_reflection_event():
    """Changing only a template reflection delay must not move an event into DG output."""
    freq_hz = np.linspace(9.0e3, 1.0e9, 5000, dtype=np.float64)
    omega = 2.0 * np.pi * freq_hz
    velocity = 299_792_458.0 / np.sqrt(2.23)
    template_a = 0.2 + 0.8 * np.exp(-2j * omega * 300.0 / velocity)
    template_b = 0.2 + 0.8 * np.exp(-2j * omega * 900.0 / velocity)
    base = np.full_like(template_a, 0.1 + 0.0j)
    params = DG.DirtyParams(
        profile="field",
        additive_scale=0.0,
        multiplicative_scale=0.0,
        ripple_scale=0.0,
        phase_scale_rad=0.0,
        fixture_scale=0.0,
        template_slow_scale=0.25,
        template_mix_scale=0.10,
        highfreq_decay_strength=0.20,
        event_hf_damping=0.60,
    )
    output_a = DG.apply_measured_template_shape(
        base, freq_hz, np.random.RandomState(123), params, template_a
    )
    output_b = DG.apply_measured_template_shape(
        base, freq_hz, np.random.RandomState(123), params, template_b
    )
    relative_rms = float(
        np.sqrt(np.mean(np.abs(output_a - output_b) ** 2))
        / np.sqrt(np.mean(np.abs(output_a) ** 2))
    )
    correlation = float(
        np.abs(np.vdot(output_a, output_b))
        / (np.linalg.norm(output_a) * np.linalg.norm(output_b))
    )
    assert relative_rms < 0.002
    assert correlation > 0.999

def test_dataset_loader_uses_manifest_paths_and_aligned_grid(tmp_path, representative_samples):
    sample = representative_samples["field_short"]
    sample_id = "dgtest000"
    raw_dir = tmp_path / "raw" / "train"
    label_dir = tmp_path / "labels" / "train"
    raw_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    band = sample["bands"]["1GHz"]
    csv_path = raw_dir / f"{sample_id}_1GHz.csv"
    DG.save_client_csv(
        csv_path,
        band["freq_hz"],
        band["s11"],
        band["distance"],
        band["impulse"],
        band["step"],
        float(sample["cable"].total_length),
    )
    metadata = {
        "sample_id": sample_id,
        "epsr": float(sample["metadata"]["epsr"]),
    }
    with (raw_dir / f"{sample_id}.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, allow_unicode=True)
    np.save(label_dir / f"{sample_id}.npy", DG.build_label(sample["cable"]))

    manifest = {
        "samples": [
            {
                "sample_id": sample_id,
                "split": "train",
                "epsr": float(sample["metadata"]["epsr"]),
                "csv_1ghz": f"raw/train/{sample_id}_1GHz.csv",
                "label": f"labels/train/{sample_id}.npy",
            }
        ]
    }
    manifest_path = tmp_path / "manifest.yaml"
    with manifest_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, allow_unicode=True)

    dataset = CableDefectDataset(str(manifest_path), split="train", band="1GHz")
    inputs, labels = dataset[0]
    assert tuple(inputs.shape) == (4, 10000)
    assert tuple(labels.shape) == (10000,)


def _interval_rms(distance: np.ndarray, values: np.ndarray, lo_m: float, hi_m: float) -> float:
    mask = (distance >= lo_m) & (distance < hi_m)
    arr = np.asarray(values, dtype=np.float64)[mask]
    return float(np.sqrt(np.mean(arr ** 2)))


def test_rg58_zero_m_boost_decays_through_hybrid_near_end_span(representative_samples):
    """The strongest RG58 burst is at the port and decays through ~8.5 m."""
    result = representative_samples["rg58"]["bands"]["1GHz"]
    distance = result["distance"]
    impulse = result["impulse"].real
    rms_0_05 = _interval_rms(distance, impulse, 0.0, 0.5)
    rms_5_85 = _interval_rms(distance, impulse, 5.0, 8.5)
    rms_85_12 = _interval_rms(distance, impulse, 8.5, 12.0)
    assert rms_0_05 > 3.0 * rms_5_85
    assert rms_5_85 > 1.35 * rms_85_12


def test_field_clamp_burst_is_strong_dense_and_spectrally_bounded(representative_samples):
    """Field front-end non-ideality must be visible without relying on |S11| clipping."""
    result = representative_samples["field_healthy"]["bands"]["1GHz"]
    distance = result["distance"]
    impulse = result["impulse"].real
    magnitude = np.abs(result["s11"])

    rms_0_5 = _interval_rms(distance, impulse, 0.0, 5.0)
    rms_5_15 = _interval_rms(distance, impulse, 5.0, 15.0)
    rms_15_60 = _interval_rms(distance, impulse, 15.0, 60.0)
    assert rms_0_5 > 8.0e-3
    assert rms_5_15 > 1.5e-3
    assert rms_15_60 < 0.35 * rms_5_15
    assert float(np.mean(magnitude >= 1.249999)) < 1.0e-4
    assert float(np.max(magnitude)) < 1.25


def test_moisture_has_steep_wet_slope_persistent_drop_and_flat_post_region(
    representative_samples,
):
    """Wet entry/section/post-section morphology follows the measured three-zone rule."""
    sample = representative_samples["field_moisture"]
    result = sample["bands"]["1GHz"]
    distance = result["distance"]
    step = result["step"]
    defect = sample["metadata"]["defects"][0]
    start = float(defect["start_m"])
    end = float(defect["end_m"])
    length = end - start

    dd = float(np.median(np.diff(distance[:2048])))
    smooth_n = int(np.clip(round(24.0 / dd), 31, 501))
    if smooth_n % 2 == 0:
        smooth_n += 1
    smooth_step = DG.smooth_array(step, smooth_n)

    pre = (distance >= start - max(70.0, 0.22 * length)) & (
        distance <= start - max(25.0, 0.10 * length)
    )
    wet = (distance >= start + 0.20 * length) & (
        distance <= end - 0.20 * length
    )
    post = (distance >= end + max(25.0, 0.08 * length)) & (
        distance <= min(0.86 * sample["cable"].total_length, end + max(180.0, 0.75 * length))
    )

    pre_slope = float(np.polyfit(distance[pre], smooth_step[pre], 1)[0])
    wet_slope = float(np.polyfit(distance[wet], smooth_step[wet], 1)[0])
    post_slope = float(np.polyfit(distance[post], smooth_step[post], 1)[0])
    assert wet_slope < -1.5e-13
    assert abs(post_slope) < 0.45 * abs(wet_slope)
    assert float(np.median(smooth_step[post])) < float(np.median(smooth_step[pre])) - 4.0e-11
