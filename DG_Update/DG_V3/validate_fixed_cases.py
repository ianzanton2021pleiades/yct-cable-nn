"""Generate the small, fixed DG V3 morphology review set."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.dont_write_bytecode = True

from dg_v3 import generate_sample, load_config
from response import s11_to_responses

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "SimHei"],
    "font.sans-serif": ["SimHei"],
    "axes.unicode_minus": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
})


CASES = (
    ("rg58_74m", "rg58", 101, {"length_m": 74.0, "defect_count": 0, "joint_positions_m": [40.0, 44.0], "termination": "open"}),
    ("rg58_96m", "rg58", 102, {"length_m": 96.0, "defect_count": 0, "joint_positions_m": [40.0, 41.0, 66.0], "termination": "open"}),
    ("field_healthy_open", "field", 201, {"length_m": 600.0, "defect_count": 0, "termination": "open"}),
    ("field_healthy_short", "field", 202, {"length_m": 600.0, "defect_count": 0, "termination": "short"}),
    ("field_short", "field", 301, {"length_m": 1200.0, "defect_count": 1, "defect_type": "short", "termination": "open"}),
    ("field_aging", "field", 302, {"length_m": 1200.0, "defect_count": 1, "defect_type": "aging", "termination": "open"}),
    ("field_moisture_local", "field", 303, {"length_m": 1200.0, "defect_count": 1, "defect_type": "moisture_local", "termination": "open"}),
    ("field_moisture_distributed", "field", 304, {"length_m": 1600.0, "defect_count": 1, "defect_type": "moisture_distributed", "termination": "open"}),
)


def run(output_dir: Path, config_path: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    figure, axes = plt.subplots(len(CASES), 2, figsize=(14, 3.1 * len(CASES)), dpi=200)
    metrics: dict[str, dict] = {}
    for row, (name, profile, seed, overrides) in enumerate(CASES):
        sample = generate_sample(seed, profile, config, overrides)
        band = sample.bands["1ghz"]
        distance, impulse_real, _, step, coverage = s11_to_responses(
            band.frequency_hz,
            band.s11,
            epsr=sample.topology.base_epsr,
            distance_step_m=0.25,
            target_distance_max_m=1.2 * sample.topology.length_m,
        )
        terminal = next(event for event in sample.truth if event["role"] == "terminal")
        apparent_end = terminal["delay_center_s"] * 299_792_458.0 / (2.0 * np.sqrt(sample.topology.base_epsr))
        terminal_mask = np.abs(distance - apparent_end) <= max(4.0, 0.01 * sample.topology.length_m)
        near_mask = distance <= min(15.0, 0.1 * sample.topology.length_m)
        metrics[name] = {
            "profile": profile,
            "physical_length_m": sample.topology.length_m,
            "events": sample.truth,
            "s11_magnitude_p99": float(np.percentile(np.abs(band.s11), 99.0)),
            "near_impulse_p99": float(np.percentile(np.abs(impulse_real[near_mask]), 99.0)),
            "terminal_impulse_peak": float(np.max(np.abs(impulse_real[terminal_mask]))),
            "terminal_apparent_distance_m": float(apparent_end),
            "coverage": coverage,
        }
        axes[row, 0].plot(band.frequency_hz / 1e6, band.s11.real, linewidth=0.55)
        axes[row, 0].set(title=name, xlabel="Frequency (MHz)", ylabel="S11 Real")
        axes[row, 1].plot(
            distance,
            impulse_real / max(np.max(np.abs(impulse_real)), 1e-30),
            linewidth=0.65,
            label="Impulse normalized",
        )
        axes[row, 1].plot(distance, step / max(np.max(np.abs(step)), 1e-30), linewidth=0.65, label="Step normalized")
        axes[row, 1].axvline(apparent_end, color="red", linestyle="--", linewidth=0.8)
        axes[row, 1].set(xlabel="Distance (m)", ylabel="Response")
        axes[row, 1].legend(fontsize=7)
        for axis in axes[row]:
            axis.grid(True, alpha=0.22)
    figure.tight_layout()
    figure.savefig(output_dir / "dg_v3_fixed_cases.png", dpi=200)
    plt.close(figure)
    payload = {"generator_version": config.generator_version, "parameter_profile": config.parameter_profile, "cases": metrics}
    (output_dir / "dg_v3_fixed_cases.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DG V3 fixed validation cases")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "configs" / "provisional_rlgc_v1.yaml")
    args = parser.parse_args()
    payload = run(args.output, args.config)
    print(f"generated {len(payload['cases'])} fixed cases in {args.output}")


if __name__ == "__main__":
    main()
