"""Compare native DG V3 distributions with the current aggregate real-data evidence."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "DG_Update/DG_V3"))

from dg_v3.config import load_config  # noqa: E402
from dg_v3.generator import generate_sample  # noqa: E402


def sample_features(frequency_hz: np.ndarray, s11: np.ndarray) -> dict[str, float]:
    frequency_hz = np.asarray(frequency_hz, dtype=np.float64)
    s11 = np.asarray(s11, dtype=np.complex128)
    magnitude = np.abs(s11)
    keep = np.linspace(0, len(s11) - 1, min(len(s11), 2048), dtype=int)
    f = frequency_hz[keep]
    values = s11[keep]
    mag = np.maximum(np.abs(values), 1e-12)
    x = f / 1.0e9
    magnitude_slope = float(np.polyfit(x, np.log(mag), 1)[0])
    phase_slope = float(np.polyfit(x, np.unwrap(np.angle(values)), 1)[0])
    residual = values - (
        np.interp(np.arange(len(values)), np.linspace(0, len(values) - 1, 65), values.real[np.linspace(0, len(values) - 1, 65, dtype=int)])
        + 1j * np.interp(np.arange(len(values)), np.linspace(0, len(values) - 1, 65), values.imag[np.linspace(0, len(values) - 1, 65, dtype=int)])
    )
    centered = residual - np.mean(residual)
    lag1 = float(np.real(np.vdot(centered[:-1], centered[1:])) / max(float(np.vdot(centered, centered).real), 1e-30))
    return {
        "magnitude_q05": float(np.quantile(magnitude, 0.05)),
        "magnitude_q50": float(np.quantile(magnitude, 0.50)),
        "magnitude_q95": float(np.quantile(magnitude, 0.95)),
        "magnitude_log_slope_per_GHz": magnitude_slope,
        "unwrapped_phase_slope_rad_per_GHz": phase_slope,
        "residual_rms": float(np.sqrt(np.mean(np.abs(residual) ** 2))),
        "residual_lag1": lag1,
    }


def real_reference(summary: dict, profile: str) -> dict[str, dict[str, float]]:
    stats = summary["signal_statistics"][profile]
    return {
        "magnitude_q05": {"q05": stats["magnitude"]["q01"], "q50": stats["magnitude"]["q05"], "q95": stats["magnitude"]["q50"]},
        "magnitude_q50": {"q05": stats["magnitude"]["q05"], "q50": stats["magnitude"]["q50"], "q95": stats["magnitude"]["q95"]},
        "magnitude_q95": {"q05": stats["magnitude"]["q50"], "q50": stats["magnitude"]["q95"], "q95": stats["magnitude"]["q99"]},
        "magnitude_log_slope_per_GHz": {key: stats["magnitude_log_slope_per_GHz"][key] for key in ("q05", "q50", "q95")},
        "unwrapped_phase_slope_rad_per_GHz": {key: stats["unwrapped_phase_slope_rad_per_GHz"][key] for key in ("q05", "q50", "q95")},
    }


def evaluate_profile(profile: str, count: int, seed: int, config, real_summary: dict):
    features = []
    failures = []
    attempt = 0
    while len(features) < count and attempt < count * 4:
        current_seed = seed + attempt
        attempt += 1
        print(f"[{profile}] success={len(features)}/{count}, attempt={attempt}", flush=True)
        try:
            sample = generate_sample(current_seed, profile, config)
        except ValueError as exc:
            failures.append({"seed": current_seed, "error": str(exc)})
            continue
        band = sample.bands["1ghz"]
        features.append(sample_features(band.frequency_hz, band.s11))
    if len(features) < max(8, count // 2):
        raise RuntimeError(f"{profile}有效样本不足：{len(features)}/{count}")
    reference = real_reference(real_summary, profile)
    rows = []
    for feature, bounds in reference.items():
        synthetic = float(np.median([item[feature] for item in features]))
        passed = bool(bounds["q05"] <= synthetic <= bounds["q95"])
        rows.append({"feature": feature, "synthetic_median": synthetic, "real_q05": bounds["q05"], "real_q50": bounds["q50"], "real_q95": bounds["q95"], "passed": passed})
    return {
        "profile": profile,
        "requested_sample_count": count,
        "sample_count": len(features),
        "parameter_profile": config.parameter_profile,
        "attempt_count": attempt,
        "generation_failures": failures,
        "generation_failure_fraction": len(failures) / attempt,
        "feature_rows": rows,
        "passed_count": sum(row["passed"] for row in rows),
        "required_count": 4,
        "gate_pass": sum(row["passed"] for row in rows) >= 4 and not failures,
        "samples": features,
    }


def plot(path: Path, results: list[dict]):
    plt.rcParams.update({"font.sans-serif": ["SimHei", "Microsoft YaHei"], "axes.unicode_minus": False, "xtick.direction": "in", "ytick.direction": "in"})
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), dpi=200)
    for axis, result in zip(axes, results):
        rows = result["feature_rows"]
        labels = [row["feature"].replace("magnitude_", "mag_").replace("unwrapped_phase_", "phase_") for row in rows]
        x = np.arange(len(rows))
        real = np.asarray([row["real_q50"] for row in rows])
        synth = np.asarray([row["synthetic_median"] for row in rows])
        scale = np.maximum(np.asarray([row["real_q95"] - row["real_q05"] for row in rows]), 1e-12)
        delta = (synth - real) / scale
        shown = np.clip(delta, -3.0, 3.0)
        bars = axis.bar(x, shown, 0.58, color=["#1f77b4" if row["passed"] else "#d97706" for row in rows])
        for bar, actual in zip(bars, delta):
            axis.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + (0.10 if bar.get_height() >= 0 else -0.10), f"{actual:.2g}", ha="center", va="bottom" if bar.get_height() >= 0 else "top", fontsize=8)
        axis.axhline(0.0, color="#334155", lw=0.8)
        axis.set_ylim(-3.35, 3.35)
        axis.set_xticks(x, labels, rotation=18, ha="right")
        axis.set_title(f"{result['profile']}：DG V3原生分布 vs 当前实测聚合（n={result['sample_count']}）")
        axis.set_ylabel("相对实测中位偏差 / (q95-q05)；显示截断到±3")
        axis.grid(True, axis="y", color="#d1d5db", lw=0.45)
        axis.tick_params(direction="in", top=True, right=True)
    axes[0].legend(handles=[Patch(color="#1f77b4", label="进入实测q05-q95"), Patch(color="#d97706", label="未进入实测q05-q95")], loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-summary", type=Path, default=HERE / "real_data_calibration/calibration_summary.json")
    parser.add_argument("--output", type=Path, default=HERE / "output/distribution_gate.json")
    parser.add_argument("--figure", type=Path, default=HERE / "assets/distribution_gate.png")
    parser.add_argument("--samples-per-profile", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    real_summary = json.loads(args.real_summary.read_text(encoding="utf-8"))
    config = load_config(ROOT / "DG_Update/DG_V3/configs/provisional_rlgc_v1.yaml")
    results = [evaluate_profile(profile, args.samples_per_profile, args.seed + offset, config, real_summary) for profile, offset in (("rg58", 0), ("field", 10000))]
    payload = {
        "rule": "5项核心特征中至少4项的合成中位数落入对应实测q05-q95",
        "candidate": "dg_v3_rlgc_native",
        "parameter_profile": config.parameter_profile,
        "results": results,
        "overall_gate_pass": all(item["gate_pass"] for item in results),
        "cst_candidates": "not_evaluable_as_native_training_generators: no RG58/Field random profile or measurement distribution",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    plot(args.figure, results)
    print(json.dumps({profile["profile"]: {"passed": profile["passed_count"], "gate": profile["gate_pass"]} for profile in results}, ensure_ascii=False))


if __name__ == "__main__":
    main()
