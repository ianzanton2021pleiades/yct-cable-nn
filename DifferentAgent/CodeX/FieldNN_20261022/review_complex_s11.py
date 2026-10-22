from __future__ import annotations

import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
CORE_DIR = PROJECT / "DifferentAgent" / "CodeX" / "Core16_Field"
V3_DIR = PROJECT / "DG_Update" / "DG_V3"
OUT = HERE / "output" / "complex_s11_review"
sys.path.insert(0, str(CORE_DIR))
sys.path.insert(0, str(V3_DIR))
import field_physics as core  # noqa: E402
from field_cases import build_cases  # noqa: E402
from dg_v3.topology import C0, CableSegment, derive_material, segment_rlgc  # noqa: E402

SOURCES = [
    ("wuxi_1100_nominal", "Wuxi nominal 1100 m, single test", 1100.0,
     "2023-12-26-无锡蓝安+藕光/蓝安431-1100米.db8.fdr", 1,
     "00001_蓝安431-1100米.db8_s1.npz"),
    ("field_637m_A", "637 m first segment, A phase, open end", 637.0,
     "2026-7-20 10.46.01深圳某地两段电缆/第一段（从柜子测）/2026-7-20 10.46.01.fdr", 1,
     "00067_2026-7-20_10.46.01_s1.npz"),
    ("field_2430m_A", "2430 m from joint start, A phase", 2430.0,
     "2026-9-14 11.14.34深圳220KV清水河站10kV布吉环保二线F12北控环保污水厂#1智能柜(从接头开始测)/2026-9-14 11.14.34.fdr", 1,
     "00093_2026-9-14_11.14.34_s1.npz"),
]
BANDS = ((9e3, 1e5, "9kHz-100kHz"), (1e5, 1e6, "100kHz-1MHz"),
         (1e6, 5e6, "1-5MHz"), (5e6, 2e8, "5-200MHz"))
PROBES = (1e5, 1e6, 1e8)
ZREF = 50.0
ALPHA_DB = 20.0 / np.log(10.0)
MODE_LABELS = {
    "current_c00": ("Current C00: 5 fixed joints + background + Core fixture", "#d62728"),
    "core_uniform_fixture": ("Core uniform, no joints + Core fixture", "#2ca02c"),
    "v3_loss_budget_fixture": ("V3 effective RLGC loss budget + Core fixture", "#1f77b4"),
}


def core_zy(frequency_hz: np.ndarray, body: dict) -> tuple[np.ndarray, np.ndarray]:
    segment = {
        "r_dc_ohm_per_m": body["r_dc_ohm_per_m"],
        "r_skin_ref_ohm_per_m": body["r_skin_ref_ohm_per_m"],
        "l_external_h_per_m": body["z0_ohm"] / body["velocity_m_s"],
        "c_inf_f_per_m": 1.0 / (body["z0_ohm"] * body["velocity_m_s"]),
        "g_dc_s_per_m": body["g_dc_s_per_m"], "tan_delta": body["tan_delta"],
        "debye_strength": 0.0, "debye_corner_hz": body["debye_corner_hz"],
    }
    return core._segment_zy(frequency_hz, segment, 1e6, "dg_v3_rlgc")


def core_uniform(frequency_hz: np.ndarray, length_m: float, body: dict,
                 fixture: dict | None) -> np.ndarray:
    z, y = core_zy(frequency_hz, body)
    line = core._line(z, y, length_m, ZREF)
    near = core._fixture(frequency_hz, fixture, 1e6, ZREF)
    return core._to_s11(core._cascade(near, line, ZREF), ZREF, "open")


def v3_uniform(frequency_hz: np.ndarray, segment: CableSegment, fixture: dict) -> np.ndarray:
    r, l, g, c = segment_rlgc(frequency_hz, segment)
    omega = 2.0 * np.pi * frequency_hz
    line = core._line(r + 1j * omega * l, g + 1j * omega * c, segment.length_m, ZREF)
    near = core._fixture(frequency_hz, fixture, 1e6, ZREF)
    return core._to_s11(core._cascade(near, line, ZREF), ZREF, "open")


def material_values(frequency_hz: float, kind: str, body: dict,
                    segment: CableSegment | None = None) -> dict:
    def rlgc(f):
        if kind == "core":
            z, y = core_zy(f, body)
            w = 2.0 * np.pi * f
            return z, y, z.real, z.imag / w, y.real, y.imag / w
        r, l, g, c = segment_rlgc(f, segment)
        w = 2.0 * np.pi * f
        return r + 1j * w * l, g + 1j * w * c, r, l, g, c

    f = float(frequency_hz)
    nearby = np.asarray([f * (1.0 - 1e-4), f, f * (1.0 + 1e-4)])
    z, y, r, l, g, c = rlgc(nearby)
    gamma = np.sqrt(z * y)
    zc = np.sqrt(z / y)
    gd = (gamma[2].imag - gamma[0].imag) / (2.0 * np.pi * (nearby[2] - nearby[0]))
    return {
        "frequency_hz": f, "R_ohm_per_m": float(r[1]), "L_h_per_m": float(l[1]),
        "G_s_per_m": float(g[1]), "C_f_per_m": float(c[1].real),
        "alpha_db_per_m": float(gamma[1].real * ALPHA_DB),
        "beta_rad_per_m": float(gamma[1].imag),
        "characteristic_impedance_ohm": [float(zc[1].real), float(zc[1].imag)],
        "phase_velocity_m_s": float(2.0 * np.pi * f / gamma[1].imag),
        "group_delay_ns_per_m": float(gd * 1e9),
    }


def phase_rms_deg(sim: np.ndarray, obs: np.ndarray, mask: np.ndarray) -> float | None:
    if not np.any(mask):
        return None
    error = np.angle(sim[mask] * np.conj(obs[mask]))
    return float(np.sqrt(np.mean(error**2)) * 180.0 / np.pi)


def zero_crossings(values: np.ndarray) -> int:
    nonzero = values[values != 0.0]
    return int(np.count_nonzero(np.signbit(nonzero[1:]) != np.signbit(nonzero[:-1])))


def band_metrics(f: np.ndarray, obs: np.ndarray, sim: np.ndarray) -> dict:
    phase_obs, phase_sim = np.unwrap(np.angle(obs)), np.unwrap(np.angle(sim))
    result = {}
    for low, high, name in BANDS:
        mask = (f >= low) & ((f <= high) if high == BANDS[-1][1] else (f < high))
        o, s = obs[mask], sim[mask]
        if not len(o):
            continue
        gate_o, gate_s = np.abs(o) > 0.05, np.abs(s) > 0.05
        result[name] = {
            "sample_count": int(len(o)),
            "complex_rmse": float(np.sqrt(np.mean(np.abs(s - o) ** 2))),
            "real_rmse": float(np.sqrt(np.mean((s.real - o.real) ** 2))),
            "imag_rmse": float(np.sqrt(np.mean((s.imag - o.imag) ** 2))),
            "phase_circular_rms_deg": phase_rms_deg(s, o, np.ones(len(o), bool)),
            "phase_rms_deg_obs_abs_gt_0p05": phase_rms_deg(s, o, gate_o),
            "phase_rms_deg_sim_abs_gt_0p05": phase_rms_deg(s, o, gate_s),
            "phase_rms_deg_both_abs_gt_0p05": phase_rms_deg(s, o, gate_o & gate_s),
            "phase_gate_counts": {"obs": int(gate_o.sum()), "sim": int(gate_s.sum()), "both": int((gate_o & gate_s).sum())},
            "gamma_abs_obs_q05_q50_q95": [float(x) for x in np.quantile(np.abs(o), [0.05, 0.5, 0.95])],
            "gamma_abs_sim_q05_q50_q95": [float(x) for x in np.quantile(np.abs(s), [0.05, 0.5, 0.95])],
            "unwrapped_phase_turns_obs": float((phase_obs[mask][-1] - phase_obs[mask][0]) / (2 * np.pi)),
            "unwrapped_phase_turns_sim": float((phase_sim[mask][-1] - phase_sim[mask][0]) / (2 * np.pi)),
            "real_zero_crossings_obs": zero_crossings(o.real),
            "real_zero_crossings_sim": zero_crossings(s.real),
        }
    return result


def main() -> None:
    cfg = json.loads((CORE_DIR / "field_config.json").read_text(encoding="utf-8"))
    v3_yaml = yaml.safe_load((V3_DIR / "configs" / "provisional_rlgc_v1.yaml").read_text(encoding="utf-8"))
    material_spec = v3_yaml["parameters"]["profiles"]["field"]["material"]
    body, fixture = cfg["body"], cfg["fixture"]
    segment_base = CableSegment(
        start_m=0.0, end_m=1.0, z0_ohm=body["z0_ohm"],
        epsr=(C0 / body["velocity_m_s"]) ** 2,
        alpha_db_per_m_at_100mhz=0.02, tan_delta_at_100mhz=0.001,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    arrays, results, plot_data = {}, [], []
    plt.rcParams.update({"font.family": ["Times New Roman", "SimHei"], "font.sans-serif": ["SimHei"],
                         "axes.unicode_minus": False, "xtick.direction": "in", "ytick.direction": "in",
                         "figure.dpi": 200, "savefig.dpi": 200})
    meta_rows = list(csv.DictReader((CORE_DIR / "output" / "scope_measurements.csv").open(encoding="utf-8-sig", newline="")))
    cal_rows = list(csv.DictReader((CORE_DIR / "output" / "calibrations.csv").open(encoding="utf-8-sig", newline="")))

    for case_id, title, length, source, rowid, filename in SOURCES:
        npz = CORE_DIR / "output" / "signals" / filename
        with np.load(npz, allow_pickle=False) as data:
            full_f, full_obs = data["frequency_hz"], data["s11_complex"]
            real, imag = data["s11_real"], data["s11_imag"]
            source_in_npz, sweep_name = str(data["source_relative_path"].item()), str(data["sweep_name"].item())
        if source_in_npz != source:
            raise ValueError(f"source path mismatch for {filename}")
        crop = (full_f >= 9e3) & (full_f <= 2e8)
        f, obs = full_f[crop], full_obs[crop]
        c00 = next(c for c in build_cases(cfg, length) if c["id"] == "C00")
        c00_s11 = core.simulate(f, c00["segments"], c00["joints"], fixture=fixture,
                                route="dg_v3_rlgc", reference_hz=cfg["reference_hz"], load="open", zref=ZREF)
        core_with_fixture = core_uniform(f, length, body, fixture)
        core_no_fixture = core_uniform(f, length, body, None)
        no_delay_fixture = dict(fixture)
        no_delay_fixture["delay_s"] = 0.0
        core_zero_delay = core_uniform(f, length, body, no_delay_fixture)
        segment = replace(segment_base, end_m=length,
                           material=derive_material(segment_base, material_spec))
        v3_s11 = v3_uniform(f, segment, fixture)
        models = {"current_c00": c00_s11, "core_uniform_fixture": core_with_fixture,
                  "core_uniform_no_fixture": core_no_fixture, "core_uniform_zero_delay": core_zero_delay,
                  "v3_loss_budget_fixture": v3_s11}
        arrays[f"frequency_{case_id}"], arrays[f"observed_{case_id}"] = f, obs
        for key, value in models.items():
            arrays[f"{key}_{case_id}"] = value
        metrics = {key: band_metrics(f, obs, value) for key, value in models.items()}

        core_probes, v3_probes = [], []
        for probe in PROBES:
            core_item = material_values(probe, "core", body)
            v3_item = material_values(probe, "v3", body, segment)
            core_item["group_delay_ns_over_L"] = core_item["group_delay_ns_per_m"] * length
            v3_item["group_delay_ns_over_L"] = v3_item["group_delay_ns_per_m"] * length
            core_probes.append(core_item)
            v3_probes.append(v3_item)

        row = next(item for item in meta_rows if item["source_relative_path"] == source and item["sweep_rowid"] == str(rowid))
        calibration = next(item for item in cal_rows if item["source_relative_path"] == source)
        grid = "linear" if row["frequency_linear_grid"].lower() == "true" else "logarithmic"
        results.append({
            "id": case_id, "title": title, "length_m_used": length,
            "source_relative_path": source, "sweep_rowid": rowid, "sweep_name": sweep_name,
            "npz": str(npz),
            "npz_gamma_equals_real_plus_j_imag_max_abs_error": float(np.max(np.abs(full_obs - (real + 1j * imag)))),
            "native_frequency": {"min_hz": float(full_f[0]), "max_hz": float(full_f[-1]),
                                 "points": int(full_f.size), "grid": grid,
                                 "review_crop_hz": [float(f[0]), float(f[-1])], "review_points": int(f.size)},
            "wave_velocity_raw_m_per_us": float(row["wave_velocity_raw"]),
            "display_axis_metadata_only": {"x_limit_min_raw": float(row["x_limit_min_raw"]),
                                            "x_limit_max_raw": float(row["x_limit_max_raw"]),
                                            "x_axis_offset_raw": float(row["x_axis_offset_raw"])},
            "calibration_data_summary": {"same_fdr_source": source, "row_count": int(calibration["row_count"]),
                                         "frequency_first_hz": float(calibration["frequency_first_hz"]),
                                         "frequency_last_hz": float(calibration["frequency_last_hz"]),
                                         "grid": "linear" if calibration["frequency_linear_grid"].lower() == "true" else "logarithmic",
                                         "raw_complex_points_exported": False},
            "models": metrics,
            "material_propagation_probes": {"Core_current_uniform": core_probes, "V3_effective_rlgc": v3_probes},
            "c00_joint_count": len(c00["joints"]),
        })
        plot_data.append((title, f, obs, models))

    for name, limits in (("s11_complex_low_9k_1M.png", (9e3, 1e6)),
                         ("s11_complex_high_5_200M.png", (5e6, 2e8))):
        fig, axes = plt.subplots(3, 3, figsize=(16, 11), sharex="col")
        for r, (title, f, obs, models) in enumerate(plot_data):
            visible = (f >= limits[0]) & (f <= limits[1])
            series = [("Observed S11", obs, "#111111"),
                      ("Current C00 (5 joints + background + fixture)", models["current_c00"], "#d62728"),
                      ("Core uniform no-joint + fixture", models["core_uniform_fixture"], "#2ca02c"),
                      ("V3 effective RLGC loss budget + Core fixture", models["v3_loss_budget_fixture"], "#1f77b4")]
            for c, component in enumerate(("real", "imag", "phase")):
                ax = axes[r, c]
                for label, gamma, color in series:
                    values = gamma.real if component == "real" else gamma.imag if component == "imag" else np.angle(gamma)
                    ax.plot(f[visible], values[visible], color=color, linewidth=0.65, label=label)
                ax.set_xscale("log")
                ax.set_xlim(*limits)
                ax.set_ylabel({"real": "Real", "imag": "Imaginary", "phase": "Wrapped phase (rad)"}[component])
                ax.set_title(title if c == 0 else component.capitalize())
                ax.grid(True, alpha=0.25)
                ax.tick_params(top=True, right=True, direction="in")
        for ax in axes[-1]:
            ax.set_xlabel("Frequency (Hz)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2, fontsize=8)
        title = "Native complex S11, 9 kHz–1 MHz" if limits[0] < 1e6 else "Native complex S11, 5–200 MHz"
        fig.suptitle(title, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        fig.savefig(OUT / name, dpi=200)
        plt.close(fig)

    report = {
        "scope": "Three existing signal NPZ files only; CPU; no source rewrite, no fitting, no phase alignment.",
        "complex_sign_evidence": "Core analyze_corpus.py selects Freq, Real, Image and writes s11_complex = Real + 1j*Image.",
        "phase_turns_note": "Unwrapped phase turns describe a frequency phase trajectory, not event count or physical one-way delay; low-|Gamma| phase is separately gated at 0.05.",
        "native_grid_weighting_note": "RMSE is pointwise unweighted on each native grid. Wuxi is linear; 637m/2430m are logarithmic, so do not rank cases by cross-line RMSE.",
        "v3_ablation_note": "V3 effective_rlgc loss-budget ablation only, not the complete old generator or a field fit. It uses the V3 field material shunt G0, equal L/Z0/epsr, no joints, and the same Core fixture as the Core uniform comparator.",
        "uniform_fixture_variants": "Core uniform body is also evaluated with fixture removed and with only delay_s=0; the latter retains fixture loss and lumped parasitics.",
        "c00_note": "Current C00 includes five fixed joints and seeded background, so its direct comparison with uniform V3 is not single-factor; interpret causality from Core uniform versus V3 uniform with the same fixture.",
        "v3_parameters": {"z0_ohm": float(body["z0_ohm"]), "epsr": float((C0 / body["velocity_m_s"])**2),
                          "tan_delta_at_100mhz": 0.001, "alpha_db_per_m_at_100mhz": 0.02,
                          "material_spec": material_spec},
        "cases": results,
        "npz": str(OUT / "complex_s11_review.npz"),
        "plots": [str(OUT / "s11_complex_low_9k_1M.png"), str(OUT / "s11_complex_high_5_200M.png")],
    }
    np.savez_compressed(OUT / "complex_s11_review.npz", **arrays)
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "plots": report["plots"], "cases": [
        {"id": item["id"], "high_band_c00": item["models"]["current_c00"]["5-200MHz"],
         "high_band_v3": item["models"]["v3_loss_budget_fixture"]["5-200MHz"]}
        for item in results
    ], "material_probe_alpha_db_per_m": [
        {"frequency_hz": item["frequency_hz"], "Core": item["alpha_db_per_m"], "V3": v3["alpha_db_per_m"]}
        for item, v3 in zip(results[0]["material_propagation_probes"]["Core_current_uniform"],
                            results[0]["material_propagation_probes"]["V3_effective_rlgc"])
    ]}, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
