from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.tdr_signal import s11_to_responses


def load_generator():
    path = PROJECT_ROOT / "[V2.6]DG_dataset_max2.5km.py"
    spec = importlib.util.spec_from_file_location("dg_v26_validation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def meaningful_zero_crossings(values: np.ndarray, threshold_fraction: float) -> int:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return 0
    peak = max(float(np.max(np.abs(values))), 1e-30)
    left, right = values[:-1], values[1:]
    meaningful = (np.abs(left) > threshold_fraction * peak) | (
        np.abs(right) > threshold_fraction * peak
    )
    return int(np.count_nonzero((left * right < 0.0) & meaningful))


def plot_response(
    output_path: Path,
    distance: np.ndarray,
    values: np.ndarray,
    title: str,
    ylabel: str,
    cable_length: float,
    defects: list[dict],
    effective_end: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(distance, values)
    for defect in defects:
        ax.axvspan(float(defect["start_m"]), float(defect["end_m"]), alpha=0.12)
    ax.axvline(cable_length, linestyle="--", linewidth=1.0, label="Nominal end")
    if effective_end is not None and abs(effective_end - cable_length) > 0.5:
        ax.axvline(effective_end, linestyle=":", linewidth=1.0, label="Effective end")
    ax.set_xlim(0.0, min(float(distance[-1]), max(cable_length * 1.2, cable_length + 60.0)))
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def validate_sample(dg, name: str, config: dict, output_dir: Path) -> dict:
    sample = dg.generate_interactive_sample(config)
    metrics: dict = {
        "profile": sample["profile"],
        "seed": int(sample["seed"]),
        "length_m": float(sample["cable"].total_length),
        "termination": sample["metadata"]["termination"],
        "defects": sample["metadata"]["defects"],
        "measured_template_source": sample["metadata"].get("measured_template_source"),
        "measured_template_quality": sample["metadata"].get("measured_template_quality"),
        "bands": {},
    }
    epsr = float(sample["metadata"]["epsr"])
    for band_name, band in sample["bands"].items():
        distance, impulse, step, _ = s11_to_responses(
            band["freq_hz"], band["s11"], epsr=epsr, window="hann"
        )
        metrics["bands"][band_name] = {
            "max_distance_error": float(np.max(np.abs(distance - band["distance"]))),
            "max_impulse_error": float(np.max(np.abs(impulse - band["impulse"]))),
            "max_step_error": float(np.max(np.abs(step - band["step"]))),
        }

    band = sample["bands"]["1GHz"]
    distance = band["distance"]
    impulse = band["impulse"].real
    step = band["step"]
    s11_magnitude = np.abs(band["s11"])
    length = float(sample["cable"].total_length)

    def interval_rms(lo_m: float, hi_m: float) -> float:
        mask = (distance >= lo_m) & (distance < hi_m)
        return float(np.sqrt(np.mean(impulse[mask] ** 2))) if int(mask.sum()) else 0.0

    metrics["s11_magnitude_p99"] = float(np.nanpercentile(s11_magnitude, 99.0))
    metrics["s11_magnitude_p995"] = float(np.nanpercentile(s11_magnitude, 99.5))
    metrics["s11_magnitude_max"] = float(np.nanmax(s11_magnitude))
    metrics["s11_clip_fraction"] = float(np.mean(s11_magnitude >= 1.199999))
    metrics["impulse_rms_0_5m"] = interval_rms(0.0, 5.0)
    metrics["impulse_rms_5_15m"] = interval_rms(5.0, 15.0)
    metrics["impulse_rms_15_60m"] = interval_rms(15.0, 60.0)
    near = (distance >= 0.0) & (distance <= min(10.0, 0.25 * length))
    metrics["near_end_meaningful_zero_crossings"] = meaningful_zero_crossings(
        impulse[near], 0.01
    )

    effective_end = float(dg.effective_terminal_phase_length_m(sample["cable"]))
    width = 12.0 if sample["profile"].startswith("rg58") else 70.0
    terminal_delta = dg._step_delta_near(distance, step, effective_end, width)
    front_mask = (distance >= 0.0) & (distance <= (10.0 if sample["profile"].startswith("rg58") else 24.0))
    terminal_mask = (distance >= effective_end - width) & (distance <= effective_end + 1.7 * width)
    front_peak = float(np.max(np.abs(impulse[front_mask]))) if int(front_mask.sum()) else 0.0
    terminal_peak = float(np.max(np.abs(impulse[terminal_mask]))) if int(terminal_mask.sum()) else 0.0
    metrics["effective_end_m"] = effective_end
    metrics["front_impulse_peak"] = front_peak
    metrics["terminal_impulse_peak"] = terminal_peak
    metrics["terminal_to_front_peak_ratio"] = terminal_peak / max(front_peak, 1.0e-30)
    metrics["terminal_step_delta"] = None if terminal_delta is None else float(terminal_delta)

    # Structural terminal checks: one significant lobe, narrow rise and broad decay.
    terminal_sign = -1.0 if sample["metadata"]["termination"] == "short" else 1.0
    shape_mask = (distance >= effective_end - 15.0) & (distance <= effective_end + max(100.0, 0.08 * length))
    shape_x = distance[shape_mask]
    shape_y = terminal_sign * impulse[shape_mask]
    if len(shape_y) >= 5:
        peak_index = int(np.argmax(shape_y))
        signed_peak = float(shape_y[peak_index])
        maxima = np.flatnonzero(
            (shape_y[1:-1] > shape_y[:-2]) & (shape_y[1:-1] >= shape_y[2:])
        ) + 1
        metrics["terminal_significant_peak_count"] = int(
            np.count_nonzero(shape_y[maxima] > 0.30 * max(signed_peak, 1.0e-30))
        )
        left = peak_index
        while left > 0 and shape_y[left] >= 0.20 * signed_peak:
            left -= 1
        right = peak_index
        while right < len(shape_y) - 1 and shape_y[right] >= 0.20 * signed_peak:
            right += 1
        metrics["terminal_rise_20pct_m"] = float(shape_x[peak_index] - shape_x[left])
        metrics["terminal_fall_20pct_m"] = float(shape_x[right] - shape_x[peak_index])

    if sample["metadata"]["defects"] and sample["profile"] == "field":
        first = sample["metadata"]["defects"][0]
        center = float(first["center_m"])
        local = (distance >= center - 30.0) & (distance <= center + 30.0)
        metrics["first_defect_significant_zero_crossings"] = meaningful_zero_crossings(
            impulse[local], 0.003
        )
        if first["type"] in {"moisture_local", "moisture_distributed"}:
            start = float(first["start_m"])
            end = float(first["end_m"])
            defect_length = end - start
            dd = float(np.median(np.diff(distance[:2048])))
            smooth_n = int(np.clip(round(24.0 / dd), 31, 501))
            if smooth_n % 2 == 0:
                smooth_n += 1
            smooth_step = dg.smooth_array(step, smooth_n)
            masks = {
                "pre": (distance >= start - max(70.0, 0.22 * defect_length))
                       & (distance <= start - max(25.0, 0.10 * defect_length)),
                "wet": (distance >= start + 0.20 * defect_length)
                       & (distance <= end - 0.20 * defect_length),
                "post": (distance >= end + max(25.0, 0.08 * defect_length))
                        & (distance <= min(0.86 * length, end + max(180.0, 0.75 * defect_length))),
            }
            for region_name, region_mask in masks.items():
                if int(region_mask.sum()) >= 12:
                    metrics[f"moisture_{region_name}_slope"] = float(
                        np.polyfit(distance[region_mask], smooth_step[region_mask], 1)[0]
                    )
                    metrics[f"moisture_{region_name}_median"] = float(
                        np.median(smooth_step[region_mask])
                    )

    plot_response(
        output_dir / f"{name}_impulse.png",
        distance,
        impulse,
        f"{name} - IFFT impulse response",
        "Impulse response",
        length,
        sample["metadata"]["defects"],
        effective_end,
    )
    plot_response(
        output_dir / f"{name}_step.png",
        distance,
        step,
        f"{name} - IFFT step response",
        "Step response",
        length,
        sample["metadata"]["defects"],
        effective_end,
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DG V2.6 deterministic regression cases")
    parser.add_argument("--output_dir", default="validation_output")
    parser.add_argument("--real_data_root", default="")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    real_root = args.real_data_root or str(output_dir / "__no_reference_data__")
    common = {
        "band": "1GHz",
        "window": "hann",
        "epsr": 2.23,
        "real_data_root": real_root,
    }
    cases = {
        "rg58_short": {
            **common,
            "profile": "rg58",
            "length_m": 50.0,
            "n_defects": 2,
            "allowed_defect_types": ["short"],
            "seed": 2015977662,
        },
        "field_short": {
            **common,
            "profile": "field",
            "length_m": 1400.0,
            "n_defects": 1,
            "allowed_defect_types": ["short"],
            "seed": 1868816610,
        },
        "field_healthy": {
            **common,
            "profile": "field",
            "length_m": 1400.0,
            "n_defects": 0,
            "allowed_defect_types": ["short"],
            "seed": 1868816610,
        },
        "field_distributed_moisture": {
            **common,
            "profile": "field",
            "length_m": 1400.0,
            "n_defects": 1,
            "allowed_defect_types": ["moisture_distributed"],
            "seed": 440840743,
        },
        "field_local_wet_joint": {
            **common,
            "profile": "field",
            "length_m": 1500.0,
            "n_defects": 1,
            "allowed_defect_types": ["moisture_local"],
            "seed": 182,
        },
        "field_aging_user_seed": {
            **common,
            "profile": "field",
            "length_m": 1200.0,
            "n_defects": 1,
            "allowed_defect_types": ["aging"],
            "seed": 81785599,
        },
        "field_distributed_moisture_200_user_seed": {
            **common,
            "profile": "field",
            "length_m": 200.0,
            "n_defects": 1,
            "allowed_defect_types": ["moisture_distributed"],
            "seed": 1135813813,
        },
        "field_distributed_moisture_1700_user_seed": {
            **common,
            "profile": "field",
            "length_m": 1700.0,
            "n_defects": 1,
            "allowed_defect_types": ["moisture_distributed"],
            "seed": 79147719,
        },
        "field_distributed_moisture_2000_user_seed": {
            **common,
            "profile": "field",
            "length_m": 2000.0,
            "n_defects": 1,
            "allowed_defect_types": ["moisture_distributed"],
            "seed": 414169655,
        },
    }

    dg = load_generator()
    report = {
        name: validate_sample(dg, name, config, output_dir)
        for name, config in cases.items()
    }
    report_path = output_dir / "validation_metrics.yaml"
    with report_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(report, handle, allow_unicode=True, sort_keys=False)
    print(f"Validation written to {output_dir}")
    print(f"Metrics: {report_path}")


if __name__ == "__main__":
    main()
