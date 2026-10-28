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
    path = PROJECT_ROOT / "[V2.3]DG_dataset_max2.5km.py"
    spec = importlib.util.spec_from_file_location("dg_v23_test_module", path)
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
                "length_m": 120.0,
                "n_defects": 2,
                "allowed_defect_types": ["short"],
                "seed": 1422844967,
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
        "field_joint": DG.generate_interactive_sample(
            {
                **common,
                "profile": "field",
                "length_m": 600.0,
                "n_defects": 2,
                "allowed_defect_types": ["short"],
                "termination": "open",
                "seed": 2024053301,
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
    """The central V2.3 invariant must hold for both frequency grids."""
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


def test_field_remote_defect_does_not_create_a_local_hf_burst(representative_samples):
    """A defect may sit on the requested weak baseline, but must not excite it."""
    sample = representative_samples["field_short"]
    result = sample["bands"]["1GHz"]
    distance = result["distance"]
    impulse = result["impulse"].real
    defect = sample["metadata"]["defects"][0]
    center = float(defect["center_m"])
    half_defect = 0.5 * float(defect["length_m"])
    dx = float(np.median(np.diff(distance[:2048])))
    smooth_n = max(9, int(round(2.4 / dx)))
    if smooth_n % 2 == 0:
        smooth_n += 1
    high_pass = impulse - DG.smooth_array(impulse, smooth_n)

    local_background = (np.abs(distance - center) >= half_defect + 4.0) & (
        np.abs(distance - center) <= 28.0
    )
    flanks = ((distance >= center - 70.0) & (distance <= center - 38.0)) | (
        (distance >= center + 38.0) & (distance <= center + 70.0)
    )
    local_rms = float(np.sqrt(np.mean(high_pass[local_background] ** 2)))
    flank_rms = float(np.sqrt(np.mean(high_pass[flanks] ** 2)))
    assert local_rms < 2.4 * max(flank_rms, 1.0e-8)


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
    assert rms_0_05 > 2.0 * rms_5_85
    assert rms_5_85 > 1.15 * rms_85_12


def test_field_clamp_burst_is_compact_and_spectrally_bounded(representative_samples):
    """Field fixture energy stays compact and does not depend on hard clipping."""
    result = representative_samples["field_healthy"]["bands"]["1GHz"]
    distance = result["distance"]
    impulse = result["impulse"].real
    magnitude = np.abs(result["s11"])

    rms_0_5 = _interval_rms(distance, impulse, 0.0, 5.0)
    rms_5_15 = _interval_rms(distance, impulse, 5.0, 15.0)
    rms_15_60 = _interval_rms(distance, impulse, 15.0, 60.0)
    assert rms_0_5 > 3.0e-3
    assert rms_5_15 > 7.0e-4
    assert rms_15_60 < 0.55 * rms_5_15
    assert float(np.mean(magnitude >= 1.199999)) < 2.0e-4
    assert float(np.max(magnitude)) <= 1.200001


def test_moisture_has_a_persistent_downward_step_without_recovery(
    representative_samples,
):
    """The wet section lowers the step and does not rebound to the pre-wet level."""
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
    wet = (distance >= start + 0.25 * length) & (
        distance <= end - 0.20 * length
    )
    post = (distance >= end + max(25.0, 0.08 * length)) & (
        distance <= min(0.86 * sample["cable"].total_length, end + max(180.0, 0.75 * length))
    )

    pre_level = float(np.median(smooth_step[pre]))
    wet_level = float(np.median(smooth_step[wet]))
    post_level = float(np.median(smooth_step[post]))
    assert wet_level < pre_level - 5.0e-11
    assert post_level < pre_level - 7.0e-11
    # A diffuse lossy exit may continue downward, but it must not recover most
    # of the wet-region drop as an ideal lossless finite section would.
    assert post_level < wet_level + 0.30 * abs(pre_level - wet_level)



def test_rg58_joint_count_is_length_aware_and_spaced():
    for length in (20.0, 50.0, 80.0, 100.0, 120.0, 160.0, 200.0):
        cap = 4 if length <= 100.0 else 5
        for seed in range(40):
            cable = DG.make_random_rg58_cable(
                np.random.RandomState(10000 + seed),
                total_length=length,
                n_defects_override=2,
            )
            joints = np.asarray(cable.joint_positions, dtype=np.float64)
            assert len(joints) <= cap
            if len(joints) > 1:
                assert float(np.min(np.diff(joints))) >= 5.0 - 1e-9
            for joint in joints:
                for defect in cable.defect_info:
                    assert not (
                        float(defect["start"]) - 2.5
                        <= joint
                        <= float(defect["end"]) + 2.5
                    )


def test_rg58_port_floor_and_full_length_baseline_are_continuous(representative_samples):
    sample = representative_samples["rg58"]
    result = sample["bands"]["1GHz"]
    distance = result["distance"]
    impulse = result["impulse"].real
    length = float(sample["cable"].total_length)

    port = float(np.max(np.abs(impulse[(distance >= 0.0) & (distance <= 2.2)])))
    terminal = float(
        np.max(np.abs(impulse[(distance >= length - 3.0) & (distance <= length + 4.0)]))
    )
    assert port >= 0.34 * terminal

    early = _interval_rms(distance, impulse, 20.0, 40.0)
    middle = _interval_rms(distance, impulse, 55.0, 75.0)
    late = _interval_rms(distance, impulse, 90.0, 112.0)
    assert middle > 0.12 * early
    assert late > 0.12 * early
    assert late > 2.0e-5


def _assert_bipolar_event(distance, impulse, center, half_width, minimum_balance):
    mask = (distance >= center - half_width) & (distance <= center + half_width)
    local = np.asarray(impulse, dtype=np.float64)[mask]
    assert local.size >= 8
    positive = float(np.max(local))
    negative = float(abs(np.min(local)))
    assert positive > 0.0 and negative > 0.0
    assert min(positive, negative) >= minimum_balance * max(positive, negative)


def test_rg58_and_field_joints_are_bipolar_and_visible(representative_samples):
    rg58 = representative_samples["rg58"]
    r = rg58["bands"]["1GHz"]
    assert rg58["metadata"]["joint_positions_m"]
    for position in rg58["metadata"]["joint_positions_m"]:
        _assert_bipolar_event(r["distance"], r["impulse"].real, position, 2.5, 0.22)

    field = representative_samples["field_joint"]
    f = field["bands"]["1GHz"]
    assert field["metadata"]["joint_positions_m"]
    for position in field["metadata"]["joint_positions_m"]:
        _assert_bipolar_event(f["distance"], f["impulse"].real, position, 32.0, 0.30)


def test_field_short_front_end_has_no_sparse_one_sided_far_tail(representative_samples):
    sample = representative_samples["field_short"]
    result = sample["bands"]["1GHz"]
    distance = result["distance"]
    impulse = result["impulse"].real
    transition = _interval_rms(distance, impulse, 12.0, 24.0)
    far_tail = _interval_rms(distance, impulse, 28.0, 60.0)
    # V2.3 deliberately has a weak continuous body baseline; the old near-end
    # sparse tail is gone when the far interval remains well below transition.
    assert far_tail < 0.38 * transition


def _local_lobe_metrics(
    distance: np.ndarray,
    impulse: np.ndarray,
    position_m: float,
    cap_m: float,
) -> dict[str, float]:
    """Measure the first positive/negative connector pair, ignoring later baseline."""
    values = np.asarray(impulse, dtype=np.float64)
    pos_zone = (distance >= position_m - 0.15 * cap_m) & (
        distance <= position_m + 0.18 * cap_m
    )
    pos_indices = np.flatnonzero(pos_zone)
    assert pos_indices.size >= 4
    positive_index = int(pos_indices[np.argmax(values[pos_indices])])

    # The connector's negative lobe is the first broad lobe after the positive
    # peak.  Restricting the search prevents unrelated full-length baseline
    # texture near the far edge of the 1.5%-wide admissible window from being
    # mistaken for part of the connector.
    neg_zone = (distance >= distance[positive_index] + 0.05) & (
        distance <= position_m + 0.42 * cap_m
    )
    neg_indices = np.flatnonzero(neg_zone)
    assert neg_indices.size >= 4
    negative_index = int(neg_indices[np.argmin(values[neg_indices])])

    def width_at_half(index: int, sign: float, lo: float, hi: float) -> float:
        signed = sign * values
        peak = float(signed[index])
        if peak <= 0.0:
            return 0.0
        half = 0.5 * peak
        left = index
        while left > 0 and distance[left] > lo and signed[left] >= half:
            left -= 1
        right = index
        while right < len(signed) - 1 and distance[right] < hi and signed[right] >= half:
            right += 1
        return float(distance[right] - distance[left])

    lo = position_m - 0.15 * cap_m
    hi = position_m + cap_m
    positive = float(values[positive_index])
    negative = float(abs(values[negative_index]))
    return {
        "positive": positive,
        "negative": negative,
        "ratio": negative / max(positive, 1.0e-12),
        "positive_width": width_at_half(positive_index, 1.0, lo, hi),
        "negative_width": width_at_half(negative_index, -1.0, lo, hi),
        "peak_separation": float(distance[negative_index] - distance[positive_index]),
    }


def test_rg58_joint_pairs_are_submetre_and_clear_of_the_baseline(representative_samples):
    sample = representative_samples["rg58"]
    result = sample["bands"]["1GHz"]
    distance = result["distance"]
    impulse = result["impulse"].real
    for position in sample["metadata"]["joint_positions_m"]:
        metrics = _local_lobe_metrics(distance, impulse, float(position), 1.0)
        side = (np.abs(distance - float(position)) >= 1.2) & (
            np.abs(distance - float(position)) <= 3.8
        )
        baseline = impulse[side] - float(np.median(impulse[side]))
        baseline_rms = float(np.sqrt(np.mean(baseline ** 2)))
        assert metrics["positive"] > 2.4 * max(baseline_rms, 1.0e-8)
        assert metrics["negative"] > 1.3 * max(baseline_rms, 1.0e-8)
        assert metrics["peak_separation"] < 0.80
        assert metrics["positive_width"] < 0.55
        assert metrics["negative_width"] < 0.80


def test_field_normal_joint_is_asymmetric_and_within_length_cap(representative_samples):
    sample = representative_samples["field_joint"]
    result = sample["bands"]["1GHz"]
    distance = result["distance"]
    impulse = result["impulse"].real
    length = float(sample["cable"].total_length)
    cap = 0.015 * length
    for position in sample["metadata"]["joint_positions_m"]:
        metrics = _local_lobe_metrics(distance, impulse, float(position), cap)
        assert 0.18 <= metrics["ratio"] <= 0.90
        assert metrics["negative_width"] >= 1.20 * metrics["positive_width"]
        assert metrics["peak_separation"] < 0.65 * cap
        assert metrics["negative_width"] < cap


def test_field_baseline_is_continuous_with_a_nonzero_remote_floor(representative_samples):
    sample = representative_samples["field_healthy"]
    result = sample["bands"]["1GHz"]
    distance = result["distance"]
    impulse = result["impulse"].real
    length = float(sample["cable"].total_length)

    excluded = np.zeros_like(distance, dtype=bool)
    for position in sample["metadata"]["joint_positions_m"]:
        excluded |= np.abs(distance - float(position)) < 20.0
    early = (distance >= 70.0) & (distance <= 0.28 * length) & ~excluded
    middle = (distance >= 0.42 * length) & (distance <= 0.62 * length) & ~excluded
    late = (distance >= 0.76 * length) & (distance <= 0.89 * length) & ~excluded
    rms_early = float(np.sqrt(np.mean(impulse[early] ** 2)))
    rms_middle = float(np.sqrt(np.mean(impulse[middle] ** 2)))
    rms_late = float(np.sqrt(np.mean(impulse[late] ** 2)))
    assert rms_early > rms_middle > 0.0
    assert rms_late > 2.0e-5
    assert rms_late > 0.10 * rms_early


def test_field_s11_is_a_coherent_decaying_carrier(representative_samples):
    result = representative_samples["field_healthy"]["bands"]["1GHz"]
    freq_mhz = result["freq_hz"] / 1.0e6
    s11 = result["s11"]
    low = (freq_mhz >= 20.0) & (freq_mhz < 100.0)
    mid = (freq_mhz >= 300.0) & (freq_mhz < 600.0)
    high = (freq_mhz >= 600.0) & (freq_mhz <= 1000.0)
    assert float(np.median(np.abs(s11[low]))) > 1.7 * float(np.median(np.abs(s11[high])))

    # Real and imaginary trajectories should be strongly correlated from one
    # frequency bin to the next, unlike V2.2's broad random walk.
    for values in (s11.real[mid], s11.imag[mid], s11.real[high], s11.imag[high]):
        adjacent_correlation = float(np.corrcoef(values[:-1], values[1:])[0, 1])
        assert adjacent_correlation > 0.97
