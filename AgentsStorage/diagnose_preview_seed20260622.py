from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from dataclasses import replace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Src" / "DG_dataset_max2.5km.py"
spec = importlib.util.spec_from_file_location("dg_preview_diag", SRC)
dg = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dg
assert spec.loader is not None
spec.loader.exec_module(dg)


def p99_window(distance, values, lo, hi):
    d = np.asarray(distance)
    y = np.asarray(values).real
    mask = (d >= lo) & (d <= hi)
    if not np.any(mask):
        return float("nan")
    return float(np.nanpercentile(np.abs(y[mask]), 99))


def end_event_stats(distance, values, center, half_width=35.0):
    d = np.asarray(distance)
    y = np.asarray(values).real
    mask = (d >= center - half_width) & (d <= center + half_width)
    if int(mask.sum()) < 3:
        return float("nan"), float("nan"), float("nan")
    xw = d[mask]
    yw = np.abs(y[mask])
    peak = float(np.nanmax(yw))
    area = float(np.trapezoid(yw, xw))
    threshold = peak * 0.35
    wide = xw[yw >= threshold]
    width = float(wide[-1] - wide[0]) if len(wide) >= 2 else 0.0
    return peak, area, width


def phase_slope(freq_hz, s11, lo_hz, hi_hz):
    f = np.asarray(freq_hz)
    y = np.unwrap(np.angle(np.asarray(s11)))
    mask = (f >= lo_hz) & (f <= hi_hz)
    if int(mask.sum()) < 20:
        return float("nan"), float("nan")
    coef = np.polyfit(f[mask] / 1e6, y[mask] - y[mask][0], 1)
    residual = y[mask] - y[mask][0] - np.polyval(coef, f[mask] / 1e6)
    return float(coef[0]), float(np.nanpercentile(np.abs(residual), 95))


def mag_stats(freq_hz, s11, lo_hz, hi_hz):
    f = np.asarray(freq_hz)
    mag = 20 * np.log10(np.maximum(np.abs(s11), 1e-8))
    mask = (f >= lo_hz) & (f <= hi_hz)
    if int(mask.sum()) < 20:
        return float("nan"), float("nan")
    return float(np.nanmedian(mag[mask])), float(np.nanpercentile(mag[mask], 95) - np.nanpercentile(mag[mask], 5))


def main() -> None:
    seed = 20260622
    real_root = Path(r"E:\FDR案例-csv")
    rg58_files, field_files, calibration_files = dg.select_preview_files(real_root, seed)
    selected = [(p, "rg58") for p in rg58_files] + [(p, "field") for p in field_files]
    print(f"seed={seed}")
    for idx, (real_path, profile) in enumerate(selected):
        rng = np.random.RandomState(seed + 1000 + idx)
        freq_real, s_real = dg.read_s11_csv_compatible(real_path)
        if profile != "field":
            continue
        folder_length = dg.infer_length_from_path(real_path, clip=False)
        estimated_end, _ = dg.estimate_measured_end_from_s11(freq_real, s_real, approx_length_m=folder_length, epsr=2.3)
        estimated_end_200m, _ = dg.estimate_measured_end_from_s11(
            freq_real[freq_real <= 200e6],
            s_real[freq_real <= 200e6],
            approx_length_m=folder_length,
            epsr=2.3,
        )
        epsr_if_folder = 2.3 * (estimated_end / folder_length) ** 2
        d_for_term, _, step_for_term = dg.s11_to_responses_for_stop(freq_real, s_real, 1e9, 2.3)
        termination = dg.infer_termination_from_measured(real_path, d_for_term, step_for_term, estimated_end)
        cable = dg.make_field_cable(
            rng,
            total_length=float(np.clip(estimated_end, 30.0, 2500.0)),
            epsr=2.3,
            termination=termination,
            n_defects_override=0,
        )
        params = dg.dirty_params_for_profile(profile, rng)
        if cable.total_length < 500.0:
            params = replace(
                params,
                template_slow_scale=max(params.template_slow_scale, 0.72),
                template_mix_scale=min(max(params.template_mix_scale, 0.004), 0.012),
                fixture_scale=params.fixture_scale * 0.16,
                dispersion_strength=params.dispersion_strength * 0.16,
                highfreq_decay_strength=params.highfreq_decay_strength * 0.45,
                event_hf_damping=max(params.event_hf_damping, 0.82),
            )
        else:
            params = replace(
                params,
                template_slow_scale=max(params.template_slow_scale, 0.62),
                template_mix_scale=max(params.template_mix_scale, 0.08),
                fixture_scale=params.fixture_scale * 0.75,
                event_hf_damping=max(params.event_hf_damping, 0.72),
            )
        same_dir_cals = [p for p in calibration_files if p.parent == real_path.parent]
        calibration_path = same_dir_cals[0] if same_dir_cals else calibration_files[int(rng.randint(0, len(calibration_files)))]
        band_1ghz, band_200mhz = dg.generate_dual_bands(cable, rng, profile, params, calibration_path, real_path)
        f1, s1, d1, imp1, step1 = band_1ghz
        f2, s2, d2, imp2, step2 = band_200mhz
        d_real_1g, imp_real_1g, step_real_1g = dg.s11_to_responses_for_stop(freq_real, s_real, 1e9, 2.3)
        d_real_200m, imp_real_200m, step_real_200m = dg.s11_to_responses_for_stop(freq_real, s_real, 200e6, 2.3)

        name = real_path.parent.name + "/" + real_path.name
        print("\nFIELD", idx + 1, name)
        print(f"  folder={folder_length:.1f} estimated1g={estimated_end:.1f} estimated200={estimated_end_200m:.1f} epsr_if_folder={epsr_if_folder:.3f} cable_z0={cable.segments[0].z0_ohm:.2f} z_load={cable.z_load_open:.2f}")
        for lo, hi in [(0, 200e6), (200e6, 400e6), (400e6, 1e9)]:
            r_med, r_span = mag_stats(freq_real, s_real, lo, hi)
            g_med, g_span = mag_stats(f1, s1, lo, hi)
            rs, rr = phase_slope(freq_real, s_real, lo, hi)
            gs, gr = phase_slope(f1, s1, lo, hi)
            print(
                f"  band {lo/1e6:.0f}-{hi/1e6:.0f}MHz "
                f"mag_med(real/dg)={r_med:.2f}/{g_med:.2f}dB "
                f"mag_span(real/dg)={r_span:.2f}/{g_span:.2f}dB "
                f"phase_slope(real/dg)={rs:.3f}/{gs:.3f} rad/MHz "
                f"phase_resid95(real/dg)={rr:.2f}/{gr:.2f}"
            )
        for lo, hi in [(0, 80), (80, 170), (170, 270), (270, 330), (330, 430)]:
            print(
                f"  impulse P99 {lo:>3}-{hi:<3}m "
                f"real1/dg1/real200/dg200="
                f"{p99_window(d_real_1g, imp_real_1g, lo, hi):.4g}/"
                f"{p99_window(d1, imp1, lo, hi):.4g}/"
                f"{p99_window(d_real_200m, imp_real_200m, lo, hi):.4g}/"
                f"{p99_window(d2, imp2, lo, hi):.4g}"
            )
        r_peak, r_area, r_width = end_event_stats(d_real_1g, imp_real_1g, estimated_end, 35.0)
        g_peak, g_area, g_width = end_event_stats(d1, imp1, estimated_end, 35.0)
        r200_peak, r200_area, r200_width = end_event_stats(d_real_200m, imp_real_200m, estimated_end_200m, 35.0)
        g200_peak, g200_area, g200_width = end_event_stats(d2, imp2, estimated_end_200m, 35.0)
        print(
            "  end event peak/area/width real1/dg1="
            f"{r_peak:.4g}/{r_area:.4g}/{r_width:.2f}m / "
            f"{g_peak:.4g}/{g_area:.4g}/{g_width:.2f}m; "
            "real200/dg200="
            f"{r200_peak:.4g}/{r200_area:.4g}/{r200_width:.2f}m / "
            f"{g200_peak:.4g}/{g200_area:.4g}/{g200_width:.2f}m"
        )


if __name__ == "__main__":
    main()
