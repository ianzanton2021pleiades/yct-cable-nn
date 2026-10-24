"""Build the field-anchored hybrid dataset.

Pipeline per measured sweep:
  measured complex S11 -> inject known reflections/intervals in the S11 domain
  -> Algorithm 1 distance response -> 8-channel features.

Labels are positive-only for measured traces: injected events are exact,
high-confidence engineer positions are weak positives, and every other cable
location is unknown (never a negative).  Fixture-short CalibrationData supplies
true no-cable negatives at d>=30 m.  Synthetic replay remains fully labelled.

The original, uninjected field features are saved separately for evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
ITER3 = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_Iter3_20261022"
ITER4 = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_Iter4_20261022"
ITER5 = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_Iter5_20261022"
ITER8 = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_Iter8_20261022"
CORE16 = PROJECT_ROOT / "DifferentAgent" / "CodeX" / "Core16_Field"
for path in (ITER3, ITER4, CORE16):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import field_eval  # noqa: E402
from features import distance_features, preprocess  # noqa: E402
from field_responses import algorithm1  # noqa: E402
from length_view import detect_endpoint_from_ifft  # noqa: E402


SOURCE_ROOT = Path(r"E:\20261010截至的全部fdr文件")
MANIFEST_PATH = (
    PROJECT_ROOT
    / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_20261022"
    / "output"
    / "real_manifest.jsonl"
)
EVIDENCE_PATH = ITER5 / "field_evidence.json"
CALIBRATION_DATA_ROOT = Path(r"E:\CableNNDataSet\CodeX\FieldNN_Iter8_20261022")
CALIBRATION_RECORDS = CALIBRATION_DATA_ROOT / "calibration_records.jsonl"
WET_CASE = "2026-6某地进行的1500m不同位置电缆接头浸水试验"
WET_SOURCE = (
    "2026-6某地进行的1500m不同位置电缆接头浸水试验/1500m电缆单个接头浸水试验.fdr"
)
WET_SWEEPS = {int(rowid): float(position) for rowid, _, position in field_eval.KNOWN_LAB_SWEEPS}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def norm_path(value: str) -> str:
    return str(value).replace("\\", "/").strip("/").casefold()


def merge_positions(values: list[float], tolerance: float = 5.0) -> list[float]:
    result: list[float] = []
    for value in sorted(float(item) for item in values):
        if not result or value - result[-1] > tolerance:
            result.append(value)
    return result


def load_known_positions(
    evidence: dict[str, Any], manifest_rows: list[dict[str, Any]]
) -> dict[tuple[str, int], list[float]]:
    source_rows: dict[str, list[int]] = defaultdict(list)
    for row in manifest_rows:
        rowid = row.get("sweep_rowid")
        if rowid is not None:
            source_rows[norm_path(str(row["source_relative_path"]))].append(int(rowid))

    raw: dict[tuple[str, int], list[float]] = defaultdict(list)
    for item in evidence.get("records", []):
        if item.get("evidence_type") != "engineer_note":
            continue
        if item.get("confidence") != "high":
            continue
        if item.get("claim") not in {"joint_position", "defect_position"}:
            continue
        positions = [
            float(value)
            for value in (item.get("positions_m") or [])
            if 30.0 <= float(value) <= 2500.0
        ]
        if not positions:
            continue
        source = norm_path(str(item.get("source_relative_path") or ""))
        rowids = [int(value) for value in (item.get("sweep_rowids") or [])]
        if not rowids:
            rowids = source_rows.get(source, [])
        for rowid in rowids:
            raw[(source, rowid)].extend(positions)

    for rowid, position in WET_SWEEPS.items():
        raw[(norm_path(WET_SOURCE), rowid)].append(position)

    return {
        key: merge_positions(values)
        for key, values in raw.items()
        if values
    }


def load_known_lengths(evidence: dict[str, Any]) -> dict[str, float]:
    lengths: dict[str, list[float]] = defaultdict(list)
    for item in evidence.get("records", []):
        if item.get("claim") != "length":
            continue
        if item.get("evidence_type") != "engineer_note":
            continue
        if item.get("confidence") != "high":
            continue
        value = item.get("length_m")
        if value is None:
            continue
        value = float(value)
        if 50.0 < value <= 3000.0:
            lengths[str(item.get("case_group") or "")].append(value)
    result = {case: float(np.median(values)) for case, values in lengths.items() if values}
    result.setdefault(WET_CASE, 1500.0)
    return result


def build_split(cases: list[str], strict_cases: set[str]) -> tuple[dict[str, str], dict[str, str]]:
    rng = random.Random(2026102207)
    forced: dict[str, str] = {WET_CASE: "test"}
    reason: dict[str, str] = {WET_CASE: "known_lab_wet_joint_forced_test"}
    for case in strict_cases:
        if case in cases:
            forced[case] = "test"
            reason[case] = "strict_unseen_cable_holdout"
    pool = sorted(case for case in cases if case not in forced)
    rng.shuffle(pool)
    n_train = int(round(0.70 * len(pool)))
    n_val = int(round(0.15 * len(pool)))
    split: dict[str, str] = dict(forced)
    for index, case in enumerate(pool):
        split[case] = "train" if index < n_train else ("val" if index < n_train + n_val else "test")
    for case in cases:
        split.setdefault(case, "train")
    return split, reason


def band_resample(
    frequency_hz: np.ndarray, s11: np.ndarray, cfg: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    f = np.asarray(frequency_hz, dtype=float)
    g = np.asarray(s11, dtype=complex)
    band = cfg["frequency"]
    if f[0] > band["start_hz"] * (1 + 1e-9) or f[-1] < band["stop_hz"] * (1 - 1e-9):
        raise ValueError("Measured S11 does not cover the configured band")
    keep = (f >= band["start_hz"]) & (f <= band["stop_hz"])
    fi = np.unique(np.r_[band["start_hz"], f[keep], band["stop_hz"]])
    gi = np.interp(fi, f, g.real) + 1j * np.interp(fi, f, g.imag)
    return fi, gi


def peak_per_amp(sigma_d_m: float) -> float:
    """Empirical Algorithm-1/Hann peak per unit S11 reflection amplitude."""
    sigma = np.asarray([3.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0])
    peak = np.asarray([0.030, 0.018, 0.010, 0.006, 0.0045, 0.0020, 0.0012])
    return float(np.interp(float(sigma_d_m), sigma, peak))


def gaussian_reflection(
    fi: np.ndarray, distance_m: float, sigma_d_m: float, amplitude: float,
    velocity_m_s: float, sign: float,
) -> np.ndarray:
    sigma_t = 2.0 * float(sigma_d_m) / float(velocity_m_s)
    envelope = np.exp(-2.0 * np.pi**2 * fi**2 * sigma_t**2)
    return (
        float(sign)
        * float(amplitude)
        * envelope
        * np.exp(-1j * 4.0 * np.pi * fi * float(distance_m) / float(velocity_m_s))
    )


def add_gaussian(
    target: np.ndarray, weight: np.ndarray, distance: np.ndarray,
    center_m: float, sigma_m: float, value_weight: float,
) -> None:
    sigma = max(float(sigma_m), 1.0)
    lo = max(0, int(np.searchsorted(distance, center_m - 4.0 * sigma)))
    hi = min(distance.size, int(np.searchsorted(distance, center_m + 4.0 * sigma)) + 1)
    if hi - lo < 3:
        return
    gaussian = np.exp(-0.5 * ((distance[lo:hi] - center_m) / sigma) ** 2).astype(np.float32)
    existing = target[lo:hi]
    target[lo:hi] = np.where(existing < 0.0, gaussian, np.maximum(existing, gaussian))
    weight[lo:hi] = np.maximum(weight[lo:hi], value_weight)


def add_interval(
    target: np.ndarray, weight: np.ndarray, distance: np.ndarray,
    start_m: float, end_m: float, value_weight: float,
) -> None:
    lo = max(0, int(np.searchsorted(distance, start_m - 10.0)))
    hi = min(distance.size, int(np.searchsorted(distance, end_m + 10.0)) + 1)
    if hi - lo < 3:
        return
    values = distance[lo:hi]
    left = np.clip((values - (start_m - 5.0)) / 5.0, 0.0, 1.0)
    right = np.clip(((end_m + 5.0) - values) / 5.0, 0.0, 1.0)
    box = (left * right).astype(np.float32)
    existing = target[lo:hi]
    target[lo:hi] = np.where(existing < 0.0, box, np.maximum(existing, box))
    weight[lo:hi] = np.maximum(weight[lo:hi], value_weight)


def background_at(distance_full: np.ndarray, background_full: np.ndarray, distance_m: float) -> float:
    return float(np.interp(float(distance_m), distance_full, background_full))


def sample_points(
    rng: np.random.Generator,
    end_m: float,
    count: int,
    known_positions: list[float],
    existing: list[float],
    distance_full: np.ndarray,
    background_full: np.ndarray,
    global_rms: float,
    min_m: float,
    min_separation_m: float,
    exclusion_m: float,
) -> list[float]:
    positions: list[float] = []
    if end_m <= min_m:
        return positions
    for _ in range(300):
        value = float(rng.uniform(min_m, end_m))
        if any(abs(value - item) < min_separation_m for item in positions + existing):
            continue
        if any(abs(value - item) < exclusion_m for item in known_positions):
            continue
        if background_at(distance_full, background_full, value) < 0.002 * global_rms:
            continue
        positions.append(value)
        if len(positions) >= count:
            return positions
    return positions


def process_trace(
    index: int,
    row: dict[str, Any],
    cfg: dict[str, Any],
    split: str,
    strict: bool,
    known_positions: list[float],
    known_length: float | None,
    variants: int,
    seed: int,
) -> dict[str, Any]:
    hybrid = cfg["hybrid"]
    source = SOURCE_ROOT / str(row["source_relative_path"])
    frequency, s11, source_velocity_m_per_us = field_eval._read_source_s11(
        source, row.get("sweep_rowid"), None
    )
    fi, gi = band_resample(frequency, s11, cfg)
    nominal_velocity = float(cfg["response"]["velocity_m_s"])
    source_velocity = (
        float(source_velocity_m_per_us) * 1e6
        if source_velocity_m_per_us is not None
        else nominal_velocity
    )
    scale = source_velocity / nominal_velocity

    distance_max = float(cfg["response"]["distance_max_m"])
    base = algorithm1(fi, gi, velocity_m_s=nominal_velocity, distance_max_m=distance_max)
    distance_full = np.asarray(base["distance_m"], dtype=float)
    impulse_full = np.asarray(base["impulse_raw"], dtype=float)
    step_full = np.asarray(base["step_rho"], dtype=float)

    dx = float(distance_full[1] - distance_full[0])
    width = max(3, int(round(30.0 / dx)))
    background_full = np.sqrt(
        uniform_filter1d(impulse_full * impulse_full, width, mode="nearest")
    )
    cable_mask = distance_full >= 30.0
    global_rms = max(float(np.sqrt(np.mean(impulse_full[cable_mask] ** 2))), 1e-12)

    if known_length is not None and 100.0 < known_length <= 3000.0:
        end_estimate = 0.85 * float(known_length)
    else:
        candidate = detect_endpoint_from_ifft(distance_full, impulse_full, step_full)
        if candidate is not None and 100.0 < float(candidate) < 2600.0:
            end_estimate = 0.85 * float(candidate)
        else:
            end_estimate = float(hybrid["fallback_end_m"])
    end_estimate = min(
        max(end_estimate, 100.0),
        float(hybrid["inject_max_m"]),
        distance_max * 0.95,
    )

    known_nominal = [
        float(value) / scale
        for value in known_positions
        if 30.0 <= float(value) <= 2500.0
    ]
    known_nominal = merge_positions(known_nominal, tolerance=8.0)

    field_feature = preprocess(fi, gi, cfg)
    field_record = {
        "index": int(index),
        "source_relative_path": str(row["source_relative_path"]),
        "sweep_rowid": int(row["sweep_rowid"]),
        "sweep_name": str(row.get("sweep_name") or ""),
        "phase": str(row.get("phase") or "unknown"),
        "case_group": str(row.get("source_case_group") or ""),
        "cable_group": str(row.get("cable_group") or ""),
        "development_status": str(row.get("development_status") or ""),
        "split": split,
        "strict": bool(strict),
        "source_wave_velocity_m_per_us": source_velocity_m_per_us,
        "source_scale_ratio": float(scale),
        "known_positions_source_m": [float(value) for value in known_positions],
        "known_positions_nominal_m": known_nominal,
        "known_length_m": None if known_length is None else float(known_length),
        "end_estimate_m": float(end_estimate),
        "label_scope": "known positive positions only; other locations are not negatives",
    }

    samples: list[tuple[np.ndarray, np.ndarray, dict[str, Any]]] = []
    d_model = np.linspace(0.0, distance_max, int(cfg["response"]["points"]))
    for variant in range(variants):
        rng = np.random.default_rng(seed + index * 10_007 + variant)
        injected: list[dict[str, Any]] = []
        intervals: list[dict[str, Any]] = []
        gi_variant = gi.copy()
        label_events: list[tuple[float, float, float]] = []

        mode = rng.random()
        if mode < float(hybrid["interval_probability"]):
            points = sample_points(
                rng, end_estimate, 1, known_nominal, [],
                distance_full, background_full, global_rms,
                float(hybrid["inject_min_m"]),
                float(hybrid["min_event_separation_m"]),
                float(hybrid["known_event_exclusion_m"]),
            )
            if points:
                start = points[0]
                width_m = float(rng.uniform(*hybrid["interval_width_m"]))
                end = min(start + width_m, end_estimate)
                if end - start >= 15.0:
                    sigma = float(rng.uniform(5.0, 18.0))
                    ratio = float(np.exp(rng.uniform(np.log(1.5), np.log(8.0))))
                    sign = 1.0 if rng.random() < 0.5 else -1.0
                    for position, event_sign, kind in (
                        (start, sign, "interval_start"),
                        (end, -sign, "interval_end"),
                    ):
                        bg = max(
                            background_at(distance_full, background_full, position),
                            0.01 * global_rms,
                        )
                        amplitude = float(
                            np.clip(
                                ratio * bg / peak_per_amp(sigma),
                                2e-4,
                                0.12,
                            )
                        )
                        gi_variant += gaussian_reflection(
                            fi, position, sigma, amplitude, nominal_velocity, event_sign
                        )
                        injected.append({
                            "position_m": float(position),
                            "sigma_m": float(sigma),
                            "amplitude": float(amplitude),
                            "sign": float(event_sign),
                            "kind": kind,
                            "target_ratio": float(ratio),
                        })
                        label_events.append((float(position), float(sigma), 1.0))
                    intervals.append({"start_m": float(start), "end_m": float(end)})
        else:
            count = 2 if mode < float(hybrid["interval_probability"]) + float(
                hybrid["two_event_probability"]
            ) else 1
            points = sample_points(
                rng, end_estimate, count, known_nominal, [],
                distance_full, background_full, global_rms,
                float(hybrid["inject_min_m"]),
                float(hybrid["min_event_separation_m"]),
                float(hybrid["known_event_exclusion_m"]),
            )
            for position in points:
                sigma = float(rng.uniform(*hybrid["event_sigma_m"]))
                ratio = float(
                    np.exp(rng.uniform(
                        np.log(float(hybrid["event_target_snr"][0])),
                        np.log(float(hybrid["event_target_snr"][1])),
                    ))
                )
                sign = 1.0 if rng.random() < 0.5 else -1.0
                bg = max(
                    background_at(distance_full, background_full, position),
                    0.01 * global_rms,
                )
                amplitude = float(
                    np.clip(ratio * bg / peak_per_amp(sigma), 2e-4, 0.12)
                )
                gi_variant += gaussian_reflection(
                    fi, position, sigma, amplitude, nominal_velocity, sign
                )
                injected.append({
                    "position_m": float(position),
                    "sigma_m": float(sigma),
                    "amplitude": float(amplitude),
                    "sign": float(sign),
                    "kind": "point",
                    "target_ratio": float(ratio),
                })
                label_events.append((float(position), float(sigma), 1.0))

        if not injected:
            continue

        # Whole-trace calibration drift: smooth complex ripple, small time shift,
        # and measurement noise.  Labels follow the time shift.
        shift_m = float(rng.uniform(-float(hybrid["time_shift_m"]), float(hybrid["time_shift_m"])))
        u = (np.log(fi) - np.log(fi[0])) / (np.log(fi[-1]) - np.log(fi[0]))
        gain = np.ones_like(u)
        phase = np.zeros_like(u)
        for harmonic in (1.0, 2.0, 3.0):
            amplitude = float(rng.uniform(-1.0, 1.0))
            offset = float(rng.uniform(0.0, 2.0 * np.pi))
            gain += float(hybrid["gain_ripple"]) * amplitude * np.sin(
                2.0 * np.pi * harmonic * u + offset
            ) / harmonic
            phase += float(hybrid["phase_ripple_rad"]) * amplitude * np.sin(
                2.0 * np.pi * harmonic * u + offset
            ) / harmonic
        gi_variant *= gain * np.exp(1j * phase)
        gi_variant *= np.exp(-1j * 4.0 * np.pi * fi * shift_m / nominal_velocity)
        noise_sigma = float(rng.uniform(*hybrid["noise_fraction"])) * max(
            float(np.median(np.abs(gi))), 1e-9
        )
        gi_variant += noise_sigma * (
            rng.normal(size=fi.size) + 1j * rng.normal(size=fi.size)
        )

        result = algorithm1(
            fi, gi_variant, velocity_m_s=nominal_velocity, distance_max_m=distance_max
        )
        impulse = np.interp(d_model, result["distance_m"], result["impulse_raw"])
        step = np.interp(d_model, result["distance_m"], result["step_rho"])
        features = distance_features(impulse, step, d_model)

        targets = np.full((5, d_model.size), -1.0, dtype=np.float32)
        targets[3] = 0.0
        targets[4] = 0.0

        shifted_events: list[dict[str, Any]] = []
        for event in injected:
            position = float(event["position_m"]) + shift_m
            event = {**event, "position_m": position}
            shifted_events.append(event)
            sigma_label = max(
                float(hybrid["label_sigma_min_m"]),
                float(hybrid["label_sigma_fraction"]) * float(event["sigma_m"]),
            )
            add_gaussian(targets[0], targets[3], d_model, position, sigma_label, 1.0)
        for interval in intervals:
            start = float(interval["start_m"]) + shift_m
            end = float(interval["end_m"]) + shift_m
            add_interval(targets[1], targets[3], d_model, start, end, 1.0)

        used_known: list[float] = []
        for position in known_nominal:
            shifted = position + shift_m
            if any(abs(shifted - float(item["position_m"])) < float(
                hybrid["known_event_exclusion_m"]
            ) for item in shifted_events):
                continue
            add_gaussian(
                targets[0], targets[3], d_model, shifted,
                float(hybrid["known_event_sigma_m"]),
                float(hybrid["known_event_weight"]),
            )
            used_known.append(float(position))

        targets[4] = (
            (targets[3] <= 0.0) & (d_model >= 30.0)
        ).astype(np.float32)

        record = {
            "index": int(index),
            "variant": int(variant),
            "source_relative_path": str(row["source_relative_path"]),
            "sweep_rowid": int(row["sweep_rowid"]),
            "sweep_name": str(row.get("sweep_name") or ""),
            "case_group": str(row.get("source_case_group") or ""),
            "cable_group": str(row.get("cable_group") or ""),
            "split": split,
            "strict": bool(strict),
            "source_scale_ratio": float(scale),
            "end_estimate_m": float(end_estimate),
            "known_length_m": None if known_length is None else float(known_length),
            "shift_m": float(shift_m),
            "noise_sigma": float(noise_sigma),
            "injected_events": shifted_events,
            "known_events_nominal_m": used_known,
            "intervals": [
                {"start_m": float(item["start_m"]) + shift_m,
                 "end_m": float(item["end_m"]) + shift_m}
                for item in intervals
            ],
            "label_scope": "injected events exact; engineer positions weak; rest unknown",
        }
        samples.append((features.astype(np.float32), targets, record))

    return {
        "index": int(index),
        "status": "ok" if samples else "no_variant",
        "field_feature": field_feature.astype(np.float32),
        "field_record": field_record,
        "samples": samples,
        "end_estimate_m": float(end_estimate),
        "source_scale_ratio": float(scale),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-cases", type=int, default=2)
    parser.add_argument("--smoke-variants", type=int, default=2)
    parser.add_argument("--variants-train", type=int)
    parser.add_argument("--variants-val", type=int)
    parser.add_argument("--variants-test", type=int)
    args = parser.parse_args()

    cfg = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    for split_name, value in (
        ("train", args.variants_train),
        ("val", args.variants_val),
        ("test", args.variants_test),
    ):
        if value is not None:
            cfg["hybrid"]["variants_per_trace"][split_name] = int(value)
    manifest_rows = read_jsonl(MANIFEST_PATH)
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    strict_cases = set(cfg.get("strict_cases", []))
    cases = sorted({str(row.get("source_case_group") or "") for row in manifest_rows})
    split, forced_reason = build_split(cases, strict_cases)
    known_positions = load_known_positions(evidence, manifest_rows)
    known_lengths = load_known_lengths(evidence)

    if args.smoke:
        chosen: set[str] = set()
        for name in ("train", "val", "test"):
            group = [case for case in cases if split[case] == name]
            chosen.update(group[: args.smoke_cases])
        chosen.add(WET_CASE)
        manifest_rows = [
            row for row in manifest_rows
            if str(row.get("source_case_group") or "") in chosen
        ]
        variants = int(args.smoke_variants)
    else:
        variants = int(cfg["hybrid"]["variants_per_trace"]["train"])

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {}
        for index, row in enumerate(manifest_rows):
            case = str(row.get("source_case_group") or "")
            case_split = split.get(case, "train")
            count = int(cfg["hybrid"]["variants_per_trace"][case_split])
            if args.smoke:
                count = variants
            futures[executor.submit(
                process_trace,
                index,
                row,
                cfg,
                case_split,
                bool(case in strict_cases),
                known_positions.get((norm_path(str(row["source_relative_path"])), int(row["sweep_rowid"])), []),
                known_lengths.get(case),
                count,
                int(cfg["seed"]),
            )] = index
        for future in tqdm(as_completed(futures), total=len(futures), desc="field-injected hybrids"):
            try:
                results.append(future.result())
            except Exception as exc:  # keep the trace and continue
                index = futures[future]
                row = manifest_rows[index]
                results.append({
                    "index": index,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "field_record": {
                        "index": index,
                        "source_relative_path": str(row.get("source_relative_path") or ""),
                        "sweep_rowid": int(row.get("sweep_rowid") or -1),
                        "case_group": str(row.get("source_case_group") or ""),
                        "split": split.get(str(row.get("source_case_group") or ""), "train"),
                    },
                    "samples": [],
                })

    results.sort(key=lambda item: item["index"])
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    field_records: list[dict[str, Any]] = []
    field_features: list[np.ndarray] = []
    split_samples: dict[str, list[tuple[np.ndarray, np.ndarray, dict[str, Any]]]] = {
        "train": [], "val": [], "test": []
    }
    for item in results:
        if item.get("status") == "error":
            continue
        field_records.append(item["field_record"])
        field_features.append(item["field_feature"])
        for sample in item["samples"]:
            split_samples[sample[2]["split"]].append(sample)

    for name in ("train", "val", "test"):
        samples = split_samples[name]
        if samples:
            np.save(
                output / f"hybrid_features_{name}.npy",
                np.stack([item[0] for item in samples]).astype(np.float32, copy=False),
            )
            np.save(
                output / f"hybrid_targets_{name}.npy",
                np.stack([item[1] for item in samples]).astype(np.float32, copy=False),
            )
            with (output / f"hybrid_records_{name}.jsonl").open(
                "w", encoding="utf-8", newline="\n"
            ) as stream:
                for _, _, record in samples:
                    stream.write(
                        json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                        + "\n"
                    )
        else:
            np.save(output / f"hybrid_features_{name}.npy", np.zeros((0, 8, int(cfg["response"]["points"])), dtype=np.float32))
            np.save(output / f"hybrid_targets_{name}.npy", np.zeros((0, 5, int(cfg["response"]["points"])), dtype=np.float32))
            (output / f"hybrid_records_{name}.jsonl").write_text("", encoding="utf-8")

    np.save(
        output / "field_features.npy",
        np.stack(field_features).astype(np.float32, copy=False),
    )
    with (output / "field_records.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for record in field_records:
            stream.write(
                json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                + "\n"
            )

    calibration_split: dict[str, str] = {}
    if CALIBRATION_RECORDS.is_file():
        for record in read_jsonl(CALIBRATION_RECORDS):
            case = str(record.get("source_case_group") or "")
            calibration_split[str(record["index"])] = split.get(case, "train")
        (output / "calibration_split.json").write_text(
            json.dumps(calibration_split, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for name in (
            "calibration_raw_imp.npy",
            "calibration_raw_step.npy",
            "calibration_records.jsonl",
        ):
            source_file = CALIBRATION_DATA_ROOT / name
            if source_file.is_file():
                shutil.copy2(source_file, output / name)

    injected_counts = Counter()
    target_ratios: list[float] = []
    end_estimates: list[float] = []
    for samples in split_samples.values():
        for _, _, record in samples:
            injected_counts[len(record["injected_events"])] += 1
            target_ratios.extend(float(item["target_ratio"]) for item in record["injected_events"])
            end_estimates.append(float(record["end_estimate_m"]))

    summary = {
        "schema": "fieldnn-hybrid-dataset-1",
        "output": str(output),
        "seed": int(cfg["seed"]),
        "manifest_rows": len(manifest_rows),
        "field_records": len(field_records),
        "splits": {
            name: {
                "cases": sum(1 for case in cases if split.get(case) == name),
                "field_traces": sum(1 for record in field_records if record.get("split") == name),
                "hybrid_samples": len(split_samples[name]),
            }
            for name in ("train", "val", "test")
        },
        "strict_cases": sorted(case for case in strict_cases if case in cases),
        "wet_case": WET_CASE,
        "forced_split_reasons": forced_reason,
        "errors": [
            {"index": item["index"], "error": item.get("error")}
            for item in results if item.get("status") == "error"
        ],
        "injected_events_per_sample": dict(sorted(injected_counts.items())),
        "target_ratio_range": [
            min(target_ratios) if target_ratios else None,
            max(target_ratios) if target_ratios else None,
        ],
        "end_estimate_m_range": [
            min(end_estimates) if end_estimates else None,
            max(end_estimates) if end_estimates else None,
        ],
        "known_positive_policy": (
            "Only high-confidence engineer_note joint/defect positions and the "
            "1500 m wet-joint sweep names are weak positives.  All other measured "
            "positions are unknown, never negatives."
        ),
        "negative_policy": (
            "CalibrationData is a fixture short with no tested cable; event and "
            "interval negatives apply only at d>=30 m.  It is not a healthy cable."
        ),
        "split_policy": (
            "Case-level random split (same cable on different dates may cross "
            "splits, per user decision); 1500 m wet joint and five whole strict "
            "holdout cases are forced to test."
        ),
        "label_policy": (
            "Injected events are exact in the nominal 172 m/us axis; engineer "
            "positions are mapped by source Wave_Velocity/172; unknown regions "
            "carry no negative label.  Terminal is not supervised on field traces."
        ),
    }
    (output / "split.json").write_text(
        json.dumps(
            {
                "case_to_split": {case: split.get(case, "train") for case in cases},
                "strict_cases": sorted(strict_cases),
                "wet_case": WET_CASE,
                "forced_reason": forced_reason,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
