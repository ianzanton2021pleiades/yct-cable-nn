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


def test_defect_case_preview_rows_cover_supported_cases_without_real_data():
    dg = load_dg_module()
    cases = dg.build_defect_case_specs()
    names = [case["name"] for case in cases]
    expected = {
        "RG58 known compact BNC",
        "RG58 random mid healthy",
        "RG58 random far short",
        "RG58 random asymmetric shorts",
        "Field medium healthy",
        "Field early short",
        "Field late short",
        "Field aging long",
        "Field moisture_local early",
        "Field moisture_distributed central",
        "Field aging+short",
        "Field aging+moisture_distributed",
    }
    assert set(names) == expected

    rows = dg.generate_defect_case_rows(seed=20260623)
    assert len(rows) == len(expected)
    assert all("freq_real" not in row and "s_real" not in row for row in rows)
    assert {row["profile"] for row in rows} == {"rg58", "rg58_random", "field"}
    assert any("aging" in row["case_name"] for row in rows)
    assert any("moisture" in row["case_name"] for row in rows)
    assert all(len(row["f1"]) == 50000 for row in rows)
    assert all(len(row["f2"]) == 5000 for row in rows)


def test_defect_case_preview_uses_dirty_measurement_background_by_default():
    dg = load_dg_module()
    rows = dg.generate_defect_case_rows(seed=20260623)
    assert all(row["preview_mode"] == "dirty" for row in rows)
    assert any(row.get("measured_template_path") for row in rows if row["profile"] == "field")
    assert all(
        row.get("measured_template_path") is None
        for row in rows
        if row["profile"] == "field" and row["cable"].defect_info
    )

    clean_rows = dg.generate_defect_case_rows(seed=20260623, use_dirty=False)
    dirty_field = next(row for row in rows if row["case_name"] == "Field medium healthy")
    clean_field = next(row for row in clean_rows if row["case_name"] == "Field medium healthy")
    mask = dirty_field["f1"] <= 500e6
    dirty_real_span = float(np.nanpercentile(dirty_field["s1"][mask].real, 95) - np.nanpercentile(dirty_field["s1"][mask].real, 5))
    clean_real_span = float(np.nanpercentile(clean_field["s1"][mask].real, 95) - np.nanpercentile(clean_field["s1"][mask].real, 5))
    dirty_clean_delta = float(np.sqrt(np.nanmean(np.abs(dirty_field["s1"][mask] - clean_field["s1"][mask]) ** 2)))
    assert dirty_real_span > clean_real_span * 1.1
    assert dirty_clean_delta > 0.025


def test_long_field_defect_case_keeps_terminal_pulse_visible():
    dg = load_dg_module()
    rows = dg.generate_defect_case_rows(seed=20260623)
    field_rows = [
        row for row in rows
        if row["profile"] == "field"
        and row["cable"].total_length >= 1500.0
        and not any(d["type"] == "moisture_distributed" for d in row["cable"].defect_info)
    ]
    assert field_rows
    for row in field_rows:
        distance = row["d1"]
        impulse = row["imp1"].real
        length = row["cable"].total_length
        end_mask = (distance > length - 35.0) & (distance < length + 35.0)
        mid_mask = (distance > max(80.0, length * 0.2)) & (distance < length - 80.0)
        assert end_mask.any()
        assert mid_mask.any()
        end_peak = float(np.nanmax(np.abs(impulse[end_mask])))
        mid_p95 = float(np.nanpercentile(np.abs(impulse[mid_mask]), 95))
        assert end_peak > max(mid_p95 * 8.0, 8e-4)


def test_moisture_distributed_step_depresses_within_interval():
    dg = load_dg_module()
    row = next(row for row in dg.generate_defect_case_rows(seed=20260623) if row["case_name"] == "Field moisture_distributed central")
    defect = row["cable"].defect_info[0]
    distance = row["d2"]
    step = row["step2"].real
    pre_mask = (distance > defect["start"] - 180.0) & (distance < defect["start"] - 40.0)
    in_mask = (distance > defect["end"] - 280.0) & (distance < defect["end"] - 80.0)
    assert pre_mask.any()
    assert in_mask.any()
    pre_level = float(np.nanmedian(step[pre_mask]))
    inside_level = float(np.nanmedian(step[in_mask]))
    assert inside_level < pre_level - 4e-11


def test_moisture_distributed_200mhz_step_declines_across_wet_interval():
    dg = load_dg_module()
    row = next(row for row in dg.generate_defect_case_rows(seed=20260623) if row["case_name"] == "Field moisture_distributed central")
    defect = row["cable"].defect_info[0]
    distance = row["d2"]
    step = row["step2"].real
    pre_mask = (distance > defect["start"] - 180.0) & (distance < defect["start"] - 40.0)
    early_mask = (distance > defect["start"] + 80.0) & (distance < defect["start"] + 260.0)
    mid_mask = (distance > defect["start"] + 450.0) & (distance < defect["end"] - 450.0)
    late_mask = (distance > defect["end"] - 280.0) & (distance < defect["end"] - 80.0)
    assert pre_mask.any()
    assert early_mask.any()
    assert mid_mask.any()
    assert late_mask.any()
    pre_level = float(np.nanmedian(step[pre_mask]))
    early_level = float(np.nanmedian(step[early_mask]))
    mid_level = float(np.nanmedian(step[mid_mask]))
    late_level = float(np.nanmedian(step[late_mask]))
    total_drop = pre_level - late_level
    early_drop = pre_level - early_level
    assert early_level <= pre_level
    assert mid_level <= pre_level
    assert late_level < mid_level - 5e-11
    assert total_drop > 8e-11
    assert early_drop < total_drop * 0.38


def test_field_short_preview_terminal_is_not_hidden_by_template_events():
    dg = load_dg_module()
    row = next(row for row in dg.generate_defect_case_rows(seed=20260623) if row["case_name"] == "Field late short")
    distance = row["d1"]
    impulse_abs = np.abs(row["imp1"].real)
    length = row["cable"].total_length
    end_mask = (distance > length - 45.0) & (distance < length + 45.0)
    pre_end_mask = (distance > length - 170.0) & (distance < length - 60.0)
    assert end_mask.any()
    assert pre_end_mask.any()
    terminal_peak = float(np.nanmax(impulse_abs[end_mask]))
    pre_end_p95 = float(np.nanpercentile(impulse_abs[pre_end_mask], 95))
    assert terminal_peak > 7.0 * pre_end_p95


def test_field_medium_healthy_keeps_terminal_event_visible_with_template_background():
    dg = load_dg_module()
    row = next(row for row in dg.generate_defect_case_rows(seed=20260623) if row["case_name"] == "Field medium healthy")
    length = float(row["cable"].total_length)
    for distance, impulse, step in [(row["d1"], row["imp1"], row["step1"]), (row["d2"], row["imp2"], row["step2"])]:
        impulse_abs = np.abs(impulse.real)
        terminal_mask = (distance > length - 42.0) & (distance < length + 42.0)
        pre_end_mask = (distance > length - 170.0) & (distance < length - 70.0)
        post_step_mask = (distance > length + 18.0) & (distance < length + 80.0)
        pre_step_mask = (distance > length - 120.0) & (distance < length - 35.0)
        assert terminal_mask.any()
        assert pre_end_mask.any()
        assert post_step_mask.any()
        assert pre_step_mask.any()
        terminal_peak = float(np.nanmax(impulse_abs[terminal_mask]))
        pre_end_p95 = float(np.nanpercentile(impulse_abs[pre_end_mask], 95))
        pre_end_peak = float(np.nanmax(impulse_abs[pre_end_mask]))
        assert terminal_peak > max(pre_end_p95 * 2.2, 0.010)
        assert pre_end_p95 < max(terminal_peak * 0.45, 0.010)
        assert pre_end_peak < max(terminal_peak * 0.72, 0.016)
        step_rise = float(np.nanmedian(step.real[post_step_mask]) - np.nanmedian(step.real[pre_step_mask]))
        assert step_rise > 1.2e-10


def robust_slope(distance: np.ndarray, values: np.ndarray, mask: np.ndarray) -> float:
    assert mask.any()
    x = distance[mask].astype(np.float64)
    y = values[mask].astype(np.float64)
    if len(x) > 80:
        keep = np.linspace(0, len(x) - 1, 80).astype(int)
        x = x[keep]
        y = y[keep]
    slope, _ = np.polyfit(x - float(np.nanmedian(x)), y, 1)
    return float(slope)


def test_moisture_distributed_step_slope_recovers_after_wet_section():
    dg = load_dg_module()
    row = next(row for row in dg.generate_defect_case_rows(seed=20260623) if row["case_name"] == "Field moisture_distributed central")
    defect = next(d for d in row["cable"].defect_info if d["type"] == "moisture_distributed")
    distance = row["d2"]
    step = row["step2"].real
    terminal_m = float(dg.effective_terminal_phase_length_m(row["cable"]))
    pre_mask = (distance > defect["start"] - 260.0) & (distance < defect["start"] - 80.0)
    wet_mask = (distance > defect["start"] + 360.0) & (distance < defect["end"] - 220.0)
    post_mask = (distance > defect["end"] + 80.0) & (distance < terminal_m - 140.0)
    assert pre_mask.any()
    assert wet_mask.any()
    assert post_mask.any()
    pre_slope = robust_slope(distance, step, pre_mask)
    wet_slope = robust_slope(distance, step, wet_mask)
    post_slope = robust_slope(distance, step, post_mask)
    assert wet_slope < pre_slope - 1.5e-13
    assert abs(post_slope - pre_slope) < abs(wet_slope - pre_slope) * 0.45


def test_moisture_local_step_slope_recovers_after_wet_section():
    dg = load_dg_module()
    row = next(row for row in dg.generate_defect_case_rows(seed=20260623) if row["case_name"] == "Field moisture_local early")
    defect = next(d for d in row["cable"].defect_info if d["type"] == "moisture_local")
    distance = row["d2"]
    step = row["step2"].real
    length = float(defect["end"] - defect["start"])
    pre_mask = (distance > max(20.0, defect["start"] - max(95.0, length * 1.25))) & (distance < defect["start"] - max(18.0, length * 0.45))
    wet_margin = min(max(length * 0.18, 10.0), 180.0)
    wet_mask = (distance > defect["start"] + wet_margin) & (distance < defect["end"] - wet_margin)
    post_mask = (distance > defect["end"] + 55.0) & (distance < min(row["cable"].total_length - 140.0, defect["end"] + max(220.0, length * 4.6)))
    assert pre_mask.any()
    assert wet_mask.any()
    assert post_mask.any()
    pre_slope = robust_slope(distance, step, pre_mask)
    wet_slope = robust_slope(distance, step, wet_mask)
    post_slope = robust_slope(distance, step, post_mask)
    assert abs(post_slope - pre_slope) < abs(wet_slope - pre_slope) * 0.55


def test_moisture_distributed_combo_keeps_weak_terminal_step_rise():
    dg = load_dg_module()
    row = next(row for row in dg.generate_defect_case_rows(seed=20260623) if row["case_name"] == "Field aging+moisture_distributed")
    terminal_m = float(dg.effective_terminal_phase_length_m(row["cable"]))
    distance = row["d1"]
    step = row["step1"].real
    before_mask = (distance > terminal_m - 150.0) & (distance < terminal_m - 55.0)
    after_mask = (distance > terminal_m + 18.0) & (distance < terminal_m + 95.0)
    assert before_mask.any()
    assert after_mask.any()
    terminal_rise = float(np.nanmedian(step[after_mask]) - np.nanmedian(step[before_mask]))
    assert terminal_rise > 5.5e-11


def test_long_field_defect_preview_keeps_measured_like_low_frequency_realpart():
    dg = load_dg_module()
    rows = dg.generate_defect_case_rows(seed=20260623)
    long_defect_rows = [
        row for row in rows
        if row["profile"] == "field"
        and row["cable"].total_length > 1500.0
        and row["cable"].defect_info
    ]
    assert long_defect_rows
    for row in long_defect_rows:
        freq_hz = row["f1"]
        real_abs = np.abs(row["s1"].real)
        low_mask = (freq_hz >= 0.0) & (freq_hz <= 20e6)
        mid_mask = (freq_hz > 20e6) & (freq_hz <= 80e6)
        high_mask = (freq_hz > 80e6) & (freq_hz <= 200e6)
        assert low_mask.any()
        assert mid_mask.any()
        assert high_mask.any()
        low_p95 = float(np.nanpercentile(real_abs[low_mask], 95))
        mid_p95 = float(np.nanpercentile(real_abs[mid_mask], 95))
        high_p95 = float(np.nanpercentile(real_abs[high_mask], 95))
        assert low_p95 >= 0.62
        assert mid_p95 >= 0.65
        assert high_p95 <= max(mid_p95 * 1.08, 0.78)


def test_long_defects_do_not_create_defect_level_local_ringing():
    dg = load_dg_module()
    rows = dg.generate_defect_case_rows(seed=20260623)
    checked = 0
    for row in rows:
        if row["profile"] != "field":
            continue
        for distance, impulse in [(row["d1"], row["imp1"]), (row["d2"], row["imp2"])]:
            impulse_abs = np.abs(impulse.real)
            terminal_m = dg.effective_terminal_phase_length_m(row["cable"])
            terminal_mask = (distance > terminal_m - 55.0) & (distance < terminal_m + 55.0)
            terminal_peak = float(np.nanmax(impulse_abs[terminal_mask])) if terminal_mask.any() else 0.0
            for defect in row["cable"].defect_info:
                if defect["type"] not in {"aging", "moisture_local"}:
                    continue
                checked += 1
                start = float(defect["start"])
                end = float(defect["end"])
                interval_mask = (distance > start - 5.0) & (distance < end + 5.0)
                side_mask = (
                    ((distance > start - 80.0) & (distance < start - 8.0))
                    | ((distance > end + 8.0) & (distance < end + 80.0))
                )
                if not interval_mask.any() or not side_mask.any():
                    continue
                interval_peak = float(np.nanmax(impulse_abs[interval_mask]))
                interval_p95 = float(np.nanpercentile(impulse_abs[interval_mask], 95))
                side_peak = float(np.nanmax(impulse_abs[side_mask]))
                assert interval_peak <= max(terminal_peak * 0.28, 0.004)
                assert side_peak <= max(interval_p95 * 3.0, terminal_peak * 0.34, 0.004)
    assert checked > 0


def test_distributed_moisture_boundaries_do_not_create_short_like_peaks():
    dg = load_dg_module()
    rows = dg.generate_defect_case_rows(seed=20260623)
    checked = 0
    for row in rows:
        defects = [d for d in row["cable"].defect_info if d["type"] == "moisture_distributed"]
        if not defects:
            continue
        for distance, impulse in [(row["d1"], row["imp1"]), (row["d2"], row["imp2"])]:
            impulse_abs = np.abs(impulse.real)
            terminal_m = dg.effective_terminal_phase_length_m(row["cable"])
            terminal_mask = (distance > terminal_m - 55.0) & (distance < terminal_m + 55.0)
            terminal_peak = float(np.nanmax(impulse_abs[terminal_mask])) if terminal_mask.any() else 0.0
            for defect in defects:
                checked += 1
                start = float(defect["start"])
                end = float(defect["end"])
                boundary_mask = (
                    ((distance > start - 25.0) & (distance < start + 45.0))
                    | ((distance > end - 45.0) & (distance < end + 25.0))
                )
                assert boundary_mask.any()
                boundary_peak = float(np.nanmax(impulse_abs[boundary_mask]))
                assert boundary_peak <= max(terminal_peak * 0.30, 0.006)
    assert checked > 0


def test_rg58_random_preview_has_no_defect_level_unlabeled_peaks():
    dg = load_dg_module()
    rows = dg.generate_defect_case_rows(seed=20260623)
    for case_name in ["RG58 random mid healthy", "RG58 random far short", "RG58 random asymmetric shorts"]:
        row = next(row for row in rows if row["case_name"] == case_name)
        distance = row["d1"]
        impulse_abs = np.abs(row["imp1"].real)
        length = row["cable"].total_length
        defects = row["cable"].defect_info
        healthy_mask = (distance > 14.0) & (distance < length - 8.0)
        for defect in defects:
            healthy_mask &= ~((distance > defect["start"] - 6.0) & (distance < defect["end"] + 6.0))
        assert healthy_mask.any()
        healthy_peak = float(np.nanmax(impulse_abs[healthy_mask]))
        end_mask = (distance > length - 5.0) & (distance < length + 8.0)
        assert end_mask.any()
        reference_peak = float(np.nanmax(impulse_abs[end_mask]))
        for defect in defects:
            defect_mask = (distance > defect["start"] - 4.0) & (distance < defect["end"] + 4.0)
            if defect_mask.any():
                reference_peak = max(reference_peak, float(np.nanmax(impulse_abs[defect_mask])))
        assert healthy_peak < max(reference_peak * 0.22, 0.010)


def count_prominent_local_peaks(values: np.ndarray, threshold: float) -> int:
    if len(values) < 3:
        return 0
    center = values[1:-1]
    peaks = (center > values[:-2]) & (center >= values[2:]) & (center > threshold)
    return int(np.sum(peaks))


def test_moisture_distributed_impulse_has_no_repeated_internal_peaks_without_joints():
    dg = load_dg_module()
    rows = dg.generate_defect_case_rows(seed=20260623)
    moisture_rows = [row for row in rows if "moisture_distributed" in row["case_name"]]
    assert moisture_rows
    for row in moisture_rows:
        defects = [d for d in row["cable"].defect_info if d["type"] == "moisture_distributed"]
        assert defects
        defect = defects[0]
        for distance, impulse in [(row["d1"], row["imp1"]), (row["d2"], row["imp2"])]:
            impulse_abs = np.abs(impulse.real)
            interior_mask = (distance > defect["start"] + 80.0) & (distance < defect["end"] - 80.0)
            terminal_m = dg.effective_terminal_phase_length_m(row["cable"])
            end_mask = (distance > terminal_m - 55.0) & (distance < terminal_m + 55.0)
            assert interior_mask.any()
            interior = impulse_abs[interior_mask]
            terminal_peak = float(np.nanmax(impulse_abs[end_mask])) if end_mask.any() else 0.0
            background = float(np.nanpercentile(interior, 90))
            threshold = max(terminal_peak * 0.10, background * 2.8, 2e-5)
            assert count_prominent_local_peaks(interior, threshold) <= 2
            assert float(np.nanmax(interior)) < max(terminal_peak * 0.28, background * 4.0, 0.0015)


def test_moisture_distributed_terminal_shifts_later_than_nominal_length():
    dg = load_dg_module()
    rows = dg.generate_defect_case_rows(seed=20260623)
    moisture_rows = [row for row in rows if "moisture_distributed" in row["case_name"]]
    assert moisture_rows
    for row in moisture_rows:
        nominal = float(row["cable"].total_length)
        effective = float(dg.effective_terminal_phase_length_m(row["cable"]))
        assert effective > nominal + 35.0
        distance = row["d1"]
        impulse_abs = np.abs(row["imp1"].real)
        delayed_mask = (distance > effective - 65.0) & (distance < effective + 65.0)
        nominal_mask = (distance > nominal - 55.0) & (distance < nominal + 55.0)
        assert delayed_mask.any()
        assert nominal_mask.any()
        delayed_peak = float(np.nanmax(impulse_abs[delayed_mask]))
        nominal_peak = float(np.nanmax(impulse_abs[nominal_mask]))
        assert delayed_peak > nominal_peak * 1.25


if __name__ == "__main__":
    test_defect_case_preview_rows_cover_supported_cases_without_real_data()
    test_defect_case_preview_uses_dirty_measurement_background_by_default()
    test_long_field_defect_case_keeps_terminal_pulse_visible()
    test_moisture_distributed_step_depresses_within_interval()
    test_moisture_distributed_200mhz_step_declines_across_wet_interval()
    test_field_short_preview_terminal_is_not_hidden_by_template_events()
    test_field_medium_healthy_keeps_terminal_event_visible_with_template_background()
    test_moisture_distributed_step_slope_recovers_after_wet_section()
    test_moisture_local_step_slope_recovers_after_wet_section()
    test_moisture_distributed_combo_keeps_weak_terminal_step_rise()
    test_long_field_defect_preview_keeps_measured_like_low_frequency_realpart()
    test_long_defects_do_not_create_defect_level_local_ringing()
    test_distributed_moisture_boundaries_do_not_create_short_like_peaks()
    test_rg58_random_preview_has_no_defect_level_unlabeled_peaks()
    test_moisture_distributed_impulse_has_no_repeated_internal_peaks_without_joints()
    test_moisture_distributed_terminal_shifts_later_than_nominal_length()
    print("defect case preview tests passed")
