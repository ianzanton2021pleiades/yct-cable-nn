"""DG V3 实测统计校准工具。

本模块只读取 CSV 并输出聚合统计。结果中不保存文件名、单条曲线、模板或
可回注残差；这些约束是刻意的，因为校准结果将被 DG 生成器作为分布参考，
而不是作为某条实测曲线的复制品。

示例（全量运行由用户自行决定）：

    python calibrate_real_data.py --input-root E:\\FDR案例-csv \
        --output-dir D:\\DG-V3-calibration --format both
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


SCHEMA_VERSION = "dg-v3-calibration-1"
EXCLUDED_DIR_NAME = "ifft现场数据汇总"
MAX_FIELD_LENGTH_M = 2500.0
DEFAULT_BANDS_HZ: tuple[tuple[str, float, float], ...] = (
    ("0_10MHz", 0.0, 10e6),
    ("10_100MHz", 10e6, 100e6),
    ("100_500MHz", 100e6, 500e6),
    ("500MHz_1GHz", 500e6, 1e9),
)
CATEGORIES = ("rg58", "field", "correction", "wet_1500m")


def _norm_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower().replace("\ufeff", ""))


def classify_file(path: Path) -> str:
    """Return one of ``rg58``, ``field``, ``correction`` or ``wet_1500m``.

    Correction suffix has priority, so a calibration capture inside an RG58
    folder remains visible as the correction population rather than being
    mixed into RG58 operating measurements.
    """
    name = path.name.lower()
    text = "\\".join(part.lower() for part in path.parts)
    if path.name.endswith("-校正数据.csv") or "calibration" in name:
        return "correction"
    if "1500m" in text and "接头" in text and "浸水" in text:
        return "wet_1500m"
    if any(part.lower().startswith("rg58") for part in path.parts):
        return "rg58"
    return "field"


def infer_nominal_length_m(path: Path) -> float | None:
    """Infer a stated length from path text; return ``None`` when absent."""
    text = " ".join(path.parts)
    if "RG58-74M" in text.upper():
        return 74.0
    if "RG58-3LINES" in text.upper():
        # The exact line length is not needed for calibration filtering; this
        # marks known RG58 captures as short and keeps them in the RG58 pool.
        return 100.0
    matches = re.findall(r"(?<![0-9])([0-9]+(?:\.[0-9]+)?)\s*(?:m|米)(?![a-z])", text, flags=re.I)
    if not matches:
        return None
    return float(matches[-1])


def _is_field_like(category: str) -> bool:
    return category in {"field", "wet_1500m", "correction"}


def should_include(path: Path, category: str | None = None) -> bool:
    """Apply the V3 scope rule: field-like captures over 2500 m are omitted."""
    category = category or classify_file(path)
    length_m = infer_nominal_length_m(path)
    return not (_is_field_like(category) and length_m is not None and length_m > MAX_FIELD_LENGTH_M)


def _float(value: str) -> float:
    return float(str(value).strip().replace("\ufeff", ""))


def _select_column(fieldnames: Sequence[str], role: str) -> str:
    aliases = {
        "frequency": ("frequencyhz", "frequency", "freqhz", "freq"),
        "real": ("real", "s11real", "realpart"),
        "imag": ("imag", "imaginary", "s11imag", "s11imaginary", "imagpart"),
    }
    normalized = {_norm_name(name): name for name in fieldnames if name is not None}
    for alias in aliases[role]:
        if alias in normalized:
            return normalized[alias]
    # The supplied real corpus also contains longer descriptive column names.
    needle = {"frequency": "freq", "real": "real", "imag": "imag"}[role]
    for key, original in normalized.items():
        if needle in key:
            return original
    raise ValueError(f"missing {role} column; expected Frequency/Frequency_Hz, Real and Imag")


def _frequency_to_hz(values: np.ndarray, header: str) -> np.ndarray:
    key = _norm_name(header)
    if "ghz" in key:
        return values * 1e9
    if "mhz" in key:
        return values * 1e6
    if "khz" in key:
        return values * 1e3
    # Existing FDR exports use Frequency as Hz; Frequency_Hz is explicit Hz.
    return values


def detect_isolated_first_frequency(freq_hz: np.ndarray, ratio_limit: float = 2.5) -> bool:
    """Detect a first RG58 point off the regular frequency grid.

    The decision uses the median interior spacing and requires a materially
    different first spacing. Thus a normal first point is retained even when
    the grid has modest jitter.
    """
    if freq_hz.size < 4:
        return False
    diffs = np.diff(freq_hz)
    interior = diffs[1:]
    positive = interior[interior > 0]
    if positive.size < 2 or diffs[0] <= 0:
        return False
    reference = float(np.median(positive))
    ratio = float(diffs[0] / reference)
    return ratio > ratio_limit or ratio < 1.0 / ratio_limit


def read_s11_csv(path: str | Path, *, category: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Read compatible S11 columns and return sorted frequency (Hz), complex S11."""
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(handle, dialect=dialect)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {source}")
        freq_col = _select_column(reader.fieldnames, "frequency")
        real_col = _select_column(reader.fieldnames, "real")
        imag_col = _select_column(reader.fieldnames, "imag")
        freq: list[float] = []
        real: list[float] = []
        imag: list[float] = []
        for row in reader:
            try:
                f, r, i = _float(row.get(freq_col, "")), _float(row.get(real_col, "")), _float(row.get(imag_col, ""))
            except (TypeError, ValueError):
                continue
            if np.isfinite(f) and f > 0 and np.isfinite(r) and np.isfinite(i):
                freq.append(f)
                real.append(r)
                imag.append(i)
    if len(freq) < 4:
        raise ValueError(f"too few valid S11 rows: {source}")
    freq_arr = _frequency_to_hz(np.asarray(freq, dtype=np.float64), freq_col)
    s11_arr = np.asarray(real, dtype=np.float64) + 1j * np.asarray(imag, dtype=np.float64)
    order = np.argsort(freq_arr, kind="stable")
    freq_arr, s11_arr = freq_arr[order], s11_arr[order]
    # Duplicate frequency rows are not a useful repeat; collapse them locally.
    unique, inverse = np.unique(freq_arr, return_inverse=True)
    if unique.size != freq_arr.size:
        sums = np.zeros(unique.size, dtype=np.complex128)
        counts = np.bincount(inverse)
        np.add.at(sums, inverse, s11_arr)
        s11_arr = sums / counts
        freq_arr = unique
    category = category or classify_file(source)
    if category == "rg58" and detect_isolated_first_frequency(freq_arr):
        freq_arr, s11_arr = freq_arr[1:], s11_arr[1:]
    return freq_arr, s11_arr


def _compact(freq: np.ndarray, value: np.ndarray, max_points: int = 512) -> tuple[np.ndarray, np.ndarray]:
    if freq.size <= max_points:
        return freq, value
    grid = np.linspace(float(freq[0]), float(freq[-1]), max_points)
    return grid, np.interp(grid, freq, value.real) + 1j * np.interp(grid, freq, value.imag)


@dataclass
class Record:
    category: str
    freq_hz: np.ndarray
    s11: np.ndarray
    path: Path
    nominal_length_m: float | None
    grid_first_removed: bool


class Reservoir:
    """Aggregate values after each input trace has already been compacted."""

    def __init__(self) -> None:
        self.values: list[float] = []
        self.total = 0

    def add(self, values: Iterable[float]) -> None:
        array = np.asarray(list(values), dtype=np.float64)
        array = array[np.isfinite(array)]
        if array.size == 0:
            return
        self.total += int(array.size)
        self.values.extend(array.tolist())

    def summary(self) -> dict[str, float | int | None]:
        if not self.values:
            return {"count": 0, "total_observations": self.total, "q01": None, "q05": None, "q50": None, "q95": None, "q99": None}
        q = np.quantile(np.asarray(self.values), [0.01, 0.05, 0.50, 0.95, 0.99])
        return {"count": len(self.values), "total_observations": self.total, "q01": float(q[0]), "q05": float(q[1]), "q50": float(q[2]), "q95": float(q[3]), "q99": float(q[4])}


class SignalStats:
    def __init__(self) -> None:
        self.points = 0
        self.measurements = 0
        self.mag, self.real, self.imag, self.phase = (Reservoir() for _ in range(4))
        self.magnitude_slopes = Reservoir()
        self.phase_slopes = Reservoir()
        self.bands: dict[str, dict[str, Reservoir]] = defaultdict(lambda: {"magnitude": Reservoir(), "phase_rad": Reservoir()})

    def add(self, freq: np.ndarray, s11: np.ndarray) -> None:
        self.measurements += 1
        self.points += int(freq.size)
        self.mag.add(np.abs(s11))
        self.real.add(s11.real)
        self.imag.add(s11.imag)
        self.phase.add(np.angle(s11))
        x = (freq - freq[0]) / 1.0e9
        if len(freq) >= 8 and x[-1] > 0:
            self.magnitude_slopes.add([
                np.polyfit(x, np.log(np.maximum(np.abs(s11), 1.0e-9)), 1)[0]
            ])
            self.phase_slopes.add([np.polyfit(x, np.unwrap(np.angle(s11)), 1)[0]])
        for name, low, high in DEFAULT_BANDS_HZ:
            mask = (freq >= low) & (freq < high)
            if np.any(mask):
                self.bands[name]["magnitude"].add(np.abs(s11[mask]))
                self.bands[name]["phase_rad"].add(np.angle(s11[mask]))

    def summary(self) -> dict:
        return {
            "measurements": self.measurements,
            "points": self.points,
            "magnitude": self.mag.summary(),
            "real": self.real.summary(),
            "imag": self.imag.summary(),
            "phase_rad_wrapped": self.phase.summary(),
            "magnitude_log_slope_per_GHz": self.magnitude_slopes.summary(),
            "unwrapped_phase_slope_rad_per_GHz": self.phase_slopes.summary(),
            "bands": {
                name: {
                    metric: reservoir.summary()
                    for metric, reservoir in self.bands.get(name, {"magnitude": Reservoir(), "phase_rad": Reservoir()}).items()
                }
                for name, _, _ in DEFAULT_BANDS_HZ
            },
        }


class NoiseStats:
    def __init__(self) -> None:
        self.groups = 0
        self.comparisons = 0
        self.additive_mag, self.additive_real, self.additive_imag = (Reservoir() for _ in range(3))
        self.multiplicative_mag, self.multiplicative_amp, self.multiplicative_phase = (Reservoir() for _ in range(3))
        self.amp_slopes, self.phase_slopes = Reservoir(), Reservoir()
        self.lag1_correlation = Reservoir()
        self.correlation_length_bins = Reservoir()
        self.bands: dict[str, dict[str, Reservoir]] = defaultdict(lambda: {"additive_magnitude": Reservoir(), "multiplicative_magnitude": Reservoir()})

    def summary(self) -> dict:
        scale_by_band = {
            name: {"relative_magnitude": self.bands.get(name, {"multiplicative_magnitude": Reservoir()})["multiplicative_magnitude"].summary()}
            for name, _, _ in DEFAULT_BANDS_HZ
        }
        return {
            "repeat_groups": self.groups,
            "pairwise_comparisons": self.comparisons,
            "additive": {"magnitude": self.additive_mag.summary(), "real": self.additive_real.summary(), "imag": self.additive_imag.summary()},
            "multiplicative": {"magnitude": self.multiplicative_mag.summary(), "amplitude_relative": self.multiplicative_amp.summary(), "phase_rad": self.multiplicative_phase.summary()},
            "frequency_dependent_scale": scale_by_band,
            "slow_drift": {"amplitude_log_slope_per_GHz": self.amp_slopes.summary(), "phase_slope_rad_per_GHz": self.phase_slopes.summary()},
            "frequency_correlation": {
                "lag1": self.lag1_correlation.summary(),
                "length_bins": self.correlation_length_bins.summary(),
            },
            "bands": {
                name: {
                    metric: reservoir.summary()
                    for metric, reservoir in self.bands.get(name, {"additive_magnitude": Reservoir(), "multiplicative_magnitude": Reservoir()}).items()
                }
                for name, _, _ in DEFAULT_BANDS_HZ
            },
        }


def _repeat_key(path: Path, root: Path) -> str:
    stem = path.stem
    stem = re.sub(r"(?:[_ -]+(?:repeat|rep|run|sample|measurement|测量|测试|第\d+次)[_ -]*\d*)$", "", stem, flags=re.I)
    stem = re.sub(r"[_ -]+\d+$", "", stem)
    # Key is internal only and is never emitted.
    return f"{path.parent.relative_to(root).as_posix()}::{stem.lower()}"


def _estimate_noise(group: Sequence[Record], noise: NoiseStats) -> None:
    if len(group) < 2:
        return
    low = max(float(record.freq_hz[0]) for record in group)
    high = min(float(record.freq_hz[-1]) for record in group)
    if high <= low:
        return
    grid = np.linspace(low, high, min(256, max(16, min(record.freq_hz.size for record in group))))
    values = np.asarray([np.interp(grid, record.freq_hz, record.s11.real) + 1j * np.interp(grid, record.freq_hz, record.s11.imag) for record in group])
    baseline = np.median(values, axis=0)
    denominator = np.maximum(np.abs(baseline), 1e-3)
    noise.groups += 1
    for row in values:
        additive = row - baseline
        ratio = row / np.where(np.abs(baseline) > 1e-3, baseline, denominator)
        mult = ratio - 1.0
        noise.comparisons += 1
        noise.additive_mag.add(np.abs(additive))
        noise.additive_real.add(additive.real)
        noise.additive_imag.add(additive.imag)
        noise.multiplicative_mag.add(np.abs(mult))
        noise.multiplicative_amp.add(np.abs(ratio) - 1.0)
        phase = np.unwrap(np.angle(ratio))
        noise.multiplicative_phase.add(np.angle(ratio))
        x = (grid - low) / 1e9
        if x[-1] > x[0]:
            amp_delta = np.log(np.maximum(np.abs(row), 1e-6)) - np.log(np.maximum(np.abs(baseline), 1e-6))
            noise.amp_slopes.add([np.polyfit(x, amp_delta, 1)[0]])
            noise.phase_slopes.add([np.polyfit(x, phase, 1)[0]])
        if len(additive) >= 8:
            a = additive.real[:-1] - float(np.mean(additive.real[:-1]))
            b = additive.real[1:] - float(np.mean(additive.real[1:]))
            scale = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
            if scale > 0:
                rho = float(np.clip(np.sum(a * b) / scale, 0.0, 0.999999))
                noise.lag1_correlation.add([rho])
                if rho > 0:
                    noise.correlation_length_bins.add([-1.0 / math.log(rho)])
        for name, band_low, band_high in DEFAULT_BANDS_HZ:
            mask = (grid >= band_low) & (grid < band_high)
            if np.any(mask):
                noise.bands[name]["additive_magnitude"].add(np.abs(additive[mask]))
                noise.bands[name]["multiplicative_magnitude"].add(np.abs(mult[mask]))


def _empty_counts() -> dict[str, int]:
    return {name: 0 for name in CATEGORIES}


def calibrate(input_root: str | Path, output_dir: str | Path, *, output_format: str = "json", progress_every: int = 25) -> dict:
    """Scan a corpus and write aggregate calibration output."""
    root = Path(input_root)
    if not root.is_dir():
        raise NotADirectoryError(root)
    paths = sorted(path for path in root.rglob("*.csv") if EXCLUDED_DIR_NAME not in {part.lower() for part in path.relative_to(root).parts})
    print(f"[DG-V3] scan: {len(paths)} CSV candidates; excluded directory={EXCLUDED_DIR_NAME}", flush=True)
    category_counts = _empty_counts()
    excluded_long = 0
    excluded_ifft = sum(1 for path in root.rglob("*.csv") if EXCLUDED_DIR_NAME in {part.lower() for part in path.relative_to(root).parts})
    errors = 0
    records: list[Record] = []
    groups: dict[str, list[Record]] = defaultdict(list)
    stats: dict[str, SignalStats] = {"overall": SignalStats(), "operational": SignalStats(), **{name: SignalStats() for name in CATEGORIES}}
    grid_removed = 0
    for index, path in enumerate(paths, 1):
        category = classify_file(path)
        if not should_include(path, category):
            excluded_long += 1
            continue
        try:
            # Read without the RG58 policy first so the decision is observable
            # and counted once in the aggregate scan below.
            freq, s11 = read_s11_csv(path, category="field")
        except (OSError, ValueError, csv.Error):
            errors += 1
            continue
        removed = category == "rg58" and detect_isolated_first_frequency(freq)
        if removed:
            freq, s11 = freq[1:], s11[1:]
            grid_removed += 1
        freq, s11 = _compact(freq, s11)
        record = Record(category, freq, s11, path, infer_nominal_length_m(path), removed)
        records.append(record)
        category_counts[category] += 1
        stats["overall"].add(freq, s11)
        stats[category].add(freq, s11)
        if category != "correction":
            stats["operational"].add(freq, s11)
        groups[_repeat_key(path, root)].append(record)
        if index == 1 or index % max(1, progress_every) == 0 or index == len(paths):
            print(f"[DG-V3] read: {index}/{len(paths)} candidates; included={len(records)}", flush=True)
    noise = NoiseStats()
    print(f"[DG-V3] noise: estimating repeated-measurement statistics for {sum(len(v) > 1 for v in groups.values())} groups", flush=True)
    for group in groups.values():
        _estimate_noise(group, noise)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "scope": {"input_root": str(root), "excluded_directory": EXCLUDED_DIR_NAME, "field_length_limit_m": MAX_FIELD_LENGTH_M, "field_over_limit_policy": "exclude", "output_policy": "aggregate_only"},
        "scan": {"csv_candidates": len(paths) + excluded_ifft, "excluded_ifft_csv": excluded_ifft, "excluded_field_over_2500m": excluded_long, "included_measurements": len(records), "read_errors": errors, "categories": category_counts, "rg58_isolated_first_frequency_removed": grid_removed},
        "frequency_grid": {"unit": "Hz", "bands_hz": [{"name": name, "low": low, "high": None if math.isinf(high) else high} for name, low, high in DEFAULT_BANDS_HZ]},
        "signal_statistics": {name: tracker.summary() for name, tracker in stats.items()},
        "repeat_noise_statistics": noise.summary(),
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    formats = {output_format} if output_format != "both" else {"json", "yaml"}
    for fmt in formats:
        target = destination / f"calibration_summary.{fmt}"
        if fmt == "json":
            target.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        elif fmt == "yaml":
            import yaml

            target.write_text(
                yaml.safe_dump(summary, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
        else:
            raise ValueError(f"unsupported output format: {fmt}")
        print(f"[DG-V3] wrote aggregate {target}", flush=True)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate DG V3 calibration statistics from real S11 CSV files")
    parser.add_argument("--input-root", required=True, help="read-only corpus root, e.g. E:\\FDR案例-csv")
    parser.add_argument("--output-dir", required=True, help="directory for aggregate JSON/YAML output")
    parser.add_argument("--format", choices=("json", "yaml", "both"), default="json", dest="output_format")
    parser.add_argument("--progress-every", type=int, default=25, help="print progress every N candidates")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    calibrate(args.input_root, args.output_dir, output_format=args.output_format, progress_every=args.progress_every)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
