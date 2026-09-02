"""DG V3 on-disk protocol.

This module is deliberately independent of the historical DG implementations.
It owns the file names, metadata contract, and small readers/writers used by
the response builder and the NumPy dataset loader.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

SCHEMA = "cable_nn_dg_v3"
VERSION = "3.0"
SPLITS = ("train", "val", "test")
BANDS = ("1ghz", "200mhz")
PROFILE = "client_hann_v1"
REQUIRED_FREQUENCY_KEYS = ("frequency_hz", "s11_real", "s11_imag")
REQUIRED_RESPONSE_KEYS = (
    "distance_m",
    "impulse_real",
    "impulse_imag",
    "step",
    "coverage_json",
)
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class DatasetProtocolError(ValueError):
    """Raised when a DG V3 file or metadata record violates the protocol."""


def canonical_band(band: str) -> str:
    value = str(band).strip().lower().replace(" ", "")
    if value not in BANDS:
        raise DatasetProtocolError(f"unsupported band: {band!r}")
    return value


def validate_split(split: str) -> str:
    value = str(split)
    if value not in SPLITS:
        raise DatasetProtocolError(f"invalid split {split!r}; expected one of {SPLITS}")
    return value


def validate_sample_id(sample_id: str) -> str:
    value = str(sample_id)
    if not value or not _SAFE_NAME.fullmatch(value) or value in {".", ".."}:
        raise DatasetProtocolError(f"invalid sample_id: {sample_id!r}")
    return value


def _finite_number(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DatasetProtocolError(f"{field} must be a finite number") from exc
    if not math.isfinite(number):
        raise DatasetProtocolError(f"{field} must be finite")
    return number


def make_dataset_metadata(
    dataset_name: str = "DG_V3",
    *,
    point_counts: Mapping[str, int] | None = None,
    profiles: Iterable[str] = (PROFILE,),
    distance_step_m: float = 0.25,
    generator_version: str = "3.0.0",
    parameter_profile: str = "provisional_rlgc_v1",
    seed: int = 0,
    requested_samples: int = 0,
) -> dict[str, Any]:
    """Return a complete, serialisable ``dataset.json`` object.

    The production defaults are 50,000 points for 1 GHz and 6,250 points for
    200 MHz.  Tests and small development datasets may explicitly provide
    smaller counts; readers always enforce the counts recorded here.
    """
    counts = {"1ghz": 50000, "200mhz": 6250}
    if point_counts is not None:
        for key, value in point_counts.items():
            band_key = canonical_band(key)
            if int(value) < 2:
                raise DatasetProtocolError(f"point count for {band_key} must be >= 2")
            counts[band_key] = int(value)
    profile_list = [str(item) for item in profiles]
    if not profile_list or any(not _SAFE_NAME.fullmatch(item) for item in profile_list):
        raise DatasetProtocolError("profiles must contain safe non-empty names")
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "schema_version": "dg-v3",
        "layout": {
            "annotations": "annotations/{split}.jsonl",
            "frequency": "frequency/{split}/{band}/{sample}.npz",
            "responses": "responses/{profile}/{split}/{band}/{sample}.npz",
        },
        "dataset_name": str(dataset_name),
        "generator_version": str(generator_version),
        "parameter_profile": str(parameter_profile),
        "seed": int(seed),
        "requested_samples": int(requested_samples),
        "split_counts": {split: 0 for split in SPLITS},
        "splits": list(SPLITS),
        "bands": {
            "1ghz": {
                "directory": "frequency/{split}/1ghz",
                "point_count": counts["1ghz"],
                "frequency_min_hz": 9000.0,
                "frequency_max_hz": 1.0e9,
            },
            "200mhz": {
                "directory": "frequency/{split}/200mhz",
                "point_count": counts["200mhz"],
                "frequency_min_hz": 9000.0,
                "frequency_max_hz": 2.0e8,
            },
        },
        "response_profiles": {
            profile: {
                "directory": f"responses/{profile}/{{split}}/{{band}}",
                "algorithm": "client_hann_v1",
                "window": "hann",
                "dc": "client_extrapolation",
                "distance_step_m": _finite_number(distance_step_m, "distance_step_m"),
            }
            for profile in profile_list
        },
    }


def validate_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return metadata; malformed schema/version fails hard."""
    if not isinstance(metadata, Mapping):
        raise DatasetProtocolError("dataset.json root must be an object")
    if metadata.get("schema") != SCHEMA:
        raise DatasetProtocolError(f"unsupported schema: {metadata.get('schema')!r}")
    if metadata.get("version") != VERSION:
        raise DatasetProtocolError(f"unsupported dataset version: {metadata.get('version')!r}")
    if metadata.get("schema_version") != "dg-v3":
        raise DatasetProtocolError(f"unsupported schema_version: {metadata.get('schema_version')!r}")
    for key in ("generator_version", "parameter_profile"):
        if not isinstance(metadata.get(key), str) or not metadata[key]:
            raise DatasetProtocolError(f"dataset.json requires {key}")
    if not isinstance(metadata.get("seed"), int) or not isinstance(metadata.get("requested_samples"), int):
        raise DatasetProtocolError("dataset.json seed/requested_samples must be integers")
    split_counts = metadata.get("split_counts")
    if not isinstance(split_counts, Mapping) or set(split_counts) != set(SPLITS):
        raise DatasetProtocolError("dataset.json split_counts must declare train/val/test")
    layout = metadata.get("layout")
    expected_layout = {
        "annotations": "annotations/{split}.jsonl",
        "frequency": "frequency/{split}/{band}/{sample}.npz",
        "responses": "responses/{profile}/{split}/{band}/{sample}.npz",
    }
    if layout != expected_layout:
        raise DatasetProtocolError("dataset.json layout does not match DG V3")
    if metadata.get("splits") != list(SPLITS):
        raise DatasetProtocolError("dataset.json splits must be train, val, test")
    bands = metadata.get("bands")
    if not isinstance(bands, Mapping) or set(bands) != set(BANDS):
        raise DatasetProtocolError("dataset.json must declare exactly 1ghz and 200mhz bands")
    for band in BANDS:
        spec = bands[band]
        if not isinstance(spec, Mapping):
            raise DatasetProtocolError(f"band spec {band} must be an object")
        if spec.get("directory") != expected_layout["frequency"].replace("{band}", band).replace("{sample}", "")[:-1]:
            # Keep this check explicit but independent of platform separators.
            expected = f"frequency/{{split}}/{band}"
            if spec.get("directory") != expected:
                raise DatasetProtocolError(f"band directory mismatch for {band}")
        count = spec.get("point_count")
        if isinstance(count, bool) or not isinstance(count, (int, float)) or int(count) != count or int(count) < 2:
            raise DatasetProtocolError(f"invalid point_count for {band}")
        _finite_number(spec.get("frequency_min_hz"), f"{band}.frequency_min_hz")
        _finite_number(spec.get("frequency_max_hz"), f"{band}.frequency_max_hz")
    profiles = metadata.get("response_profiles")
    if not isinstance(profiles, Mapping) or not profiles:
        raise DatasetProtocolError("dataset.json must declare response_profiles")
    for profile, spec in profiles.items():
        if not _SAFE_NAME.fullmatch(str(profile)) or not isinstance(spec, Mapping):
            raise DatasetProtocolError(f"invalid response profile {profile!r}")
        expected = f"responses/{profile}/{{split}}/{{band}}"
        if spec.get("directory") != expected or spec.get("algorithm") != "client_hann_v1" or spec.get("window") != "hann" or spec.get("dc") != "client_extrapolation":
            raise DatasetProtocolError(f"response profile {profile!r} is not client_hann_v1")
        if _finite_number(spec.get("distance_step_m"), f"{profile}.distance_step_m") != 0.25:
            raise DatasetProtocolError(f"{profile}.distance_step_m must be 0.25")
    return dict(metadata)


def dataset_json_path(root: str | Path) -> Path:
    return Path(root) / "dataset.json"


def write_dataset_metadata(root: str | Path, metadata: Mapping[str, Any]) -> Path:
    checked = validate_metadata(metadata)
    path = dataset_json_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(checked, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Materialise the declared directories even for an empty split.  This
    # keeps a freshly generated dataset structurally valid before any samples
    # or derived responses are written.
    for split in SPLITS:
        for band in BANDS:
            (path.parent / "frequency" / split / band).mkdir(parents=True, exist_ok=True)
            for profile in checked["response_profiles"]:
                (path.parent / "responses" / profile / split / band).mkdir(parents=True, exist_ok=True)
        (path.parent / "annotations").mkdir(parents=True, exist_ok=True)
    return path


def load_dataset_metadata(root: str | Path) -> dict[str, Any]:
    path = dataset_json_path(root)
    if not path.is_file():
        raise DatasetProtocolError(f"missing dataset.json: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetProtocolError(f"cannot read dataset.json: {path}") from exc
    return validate_metadata(value)


def frequency_path(root: str | Path, split: str, band: str, sample_id: str) -> Path:
    return Path(root) / "frequency" / validate_split(split) / canonical_band(band) / f"{validate_sample_id(sample_id)}.npz"


def response_path(root: str | Path, profile: str, split: str, band: str, sample_id: str) -> Path:
    if not _SAFE_NAME.fullmatch(str(profile)):
        raise DatasetProtocolError(f"invalid response profile: {profile!r}")
    return Path(root) / "responses" / str(profile) / validate_split(split) / canonical_band(band) / f"{validate_sample_id(sample_id)}.npz"


def write_frequency(root: str | Path, split: str, band: str, sample_id: str, frequency_hz: np.ndarray, s11_real: np.ndarray, s11_imag: np.ndarray) -> Path:
    path = frequency_path(root, split, band, sample_id)
    f = np.asarray(frequency_hz, dtype=np.float64)
    re = np.asarray(s11_real, dtype=np.float64)
    im = np.asarray(s11_imag, dtype=np.float64)
    if f.ndim != 1 or re.ndim != 1 or im.ndim != 1 or not (len(f) == len(re) == len(im)) or len(f) < 2:
        raise DatasetProtocolError("frequency and S11 arrays must be one-dimensional and aligned")
    if not (np.all(np.isfinite(f)) and np.all(np.isfinite(re)) and np.all(np.isfinite(im))):
        raise DatasetProtocolError("frequency and S11 arrays must be finite")
    if np.any(np.diff(f) <= 0) or np.any(f <= 0):
        raise DatasetProtocolError("frequency_hz must be strictly increasing and positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, frequency_hz=f, s11_real=re, s11_imag=im)
    return path


def _load_npz(path: Path, required: tuple[str, ...]) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise DatasetProtocolError(f"missing required file: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            names = tuple(archive.files)
            missing = [key for key in required if key not in names]
            if missing:
                raise DatasetProtocolError(f"{path} missing arrays: {missing}")
            return {key: np.asarray(archive[key]) for key in names}
    except (OSError, ValueError) as exc:
        raise DatasetProtocolError(f"cannot read NPZ: {path}") from exc


def read_frequency(root: str | Path, split: str, band: str, sample_id: str) -> dict[str, np.ndarray]:
    arrays = _load_npz(frequency_path(root, split, band, sample_id), REQUIRED_FREQUENCY_KEYS)
    f, re, im = (arrays[key] for key in REQUIRED_FREQUENCY_KEYS)
    if any(array.dtype != np.float64 for array in (f, re, im)):
        raise DatasetProtocolError("frequency NPZ arrays must be float64")
    if any(array.ndim != 1 for array in (f, re, im)) or not (len(f) == len(re) == len(im)):
        raise DatasetProtocolError("frequency NPZ arrays are not aligned one-dimensional arrays")
    if not (np.all(np.isfinite(f)) and np.all(np.isfinite(re)) and np.all(np.isfinite(im))) or np.any(np.diff(f) <= 0) or np.any(f <= 0):
        raise DatasetProtocolError("invalid frequency NPZ values")
    return {key: arrays[key] for key in REQUIRED_FREQUENCY_KEYS}


def write_annotations(root: str | Path, split: str, records: Iterable[Mapping[str, Any]]) -> Path:
    split_key = validate_split(split)
    checked = []
    seen: set[str] = set()
    for record in records:
        item = validate_annotation(record, expected_split=split_key)
        if item["sample_id"] in seen:
            raise DatasetProtocolError(f"duplicate annotation sample_id: {item['sample_id']}")
        seen.add(item["sample_id"])
        checked.append(item)
    path = Path(root) / "annotations" / f"{split_key}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in checked:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    return path


def validate_annotation(record: Mapping[str, Any], *, expected_split: str | None = None) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise DatasetProtocolError("annotation line must be an object")
    sample_id = validate_sample_id(record.get("sample_id", ""))
    split = validate_split(record.get("split", ""))
    if expected_split is not None and split != expected_split:
        raise DatasetProtocolError(f"annotation {sample_id} split mismatch: {split} != {expected_split}")
    required_record = {
        "profile", "seed", "physical_length_m", "reference_epsr", "termination",
        "frequency_files", "generation", "events",
    }
    missing_record = sorted(required_record.difference(record))
    if missing_record:
        raise DatasetProtocolError(f"annotation {sample_id} missing fields: {missing_record}")
    if record["profile"] not in {"rg58", "field"}:
        raise DatasetProtocolError(f"annotation {sample_id} has invalid profile")
    if record["termination"] not in {"open", "short"}:
        raise DatasetProtocolError(f"annotation {sample_id} has invalid termination")
    physical_length = _finite_number(record["physical_length_m"], f"{sample_id}.physical_length_m")
    reference_epsr = _finite_number(record["reference_epsr"], f"{sample_id}.reference_epsr")
    if physical_length <= 0 or reference_epsr <= 0:
        raise DatasetProtocolError(f"annotation {sample_id} has invalid length/epsr")
    frequency_files = record["frequency_files"]
    if not isinstance(frequency_files, Mapping) or set(frequency_files) != set(BANDS):
        raise DatasetProtocolError(f"annotation {sample_id} requires both frequency files")
    expected_files = {
        band: f"frequency/{split}/{band}/{sample_id}.npz"
        for band in BANDS
    }
    if dict(frequency_files) != expected_files:
        raise DatasetProtocolError(f"annotation {sample_id} frequency file paths do not match protocol")
    if not isinstance(record["seed"], int) or not isinstance(record["generation"], Mapping):
        raise DatasetProtocolError(f"annotation {sample_id} requires integer seed and generation object")
    events = record.get("events")
    if not isinstance(events, list):
        raise DatasetProtocolError(f"annotation {sample_id} requires an events list")
    normalised_events = []
    for index, event in enumerate(events):
        required_event = {
            "event_id", "role", "geometry", "mechanism", "physical_start_m",
            "physical_center_m", "physical_end_m", "delay_start_s", "delay_center_s",
            "delay_end_s", "severity", "electrical_change",
        }
        if not isinstance(event, Mapping) or required_event.difference(event):
            raise DatasetProtocolError(
                f"annotation {sample_id} event {index} missing fields: {sorted(required_event.difference(event or {}))}"
            )
        if event["role"] not in {"terminal", "joint", "defect"}:
            raise DatasetProtocolError(f"annotation {sample_id} event {index} has invalid role")
        if event["geometry"] not in {"point", "interval"}:
            raise DatasetProtocolError(f"annotation {sample_id} event {index} has invalid geometry")
        if event["role"] == "defect" and event["mechanism"] not in {
            "short", "aging", "moisture_local", "moisture_distributed",
        }:
            raise DatasetProtocolError(f"annotation {sample_id} event {index} has invalid defect mechanism")
        if event["role"] == "joint" and event["mechanism"] != "joint":
            raise DatasetProtocolError(f"annotation {sample_id} joint mechanism must be joint")
        start = _finite_number(event["physical_start_m"], f"{sample_id}.events[{index}].physical_start_m")
        center = _finite_number(event["physical_center_m"], f"{sample_id}.events[{index}].physical_center_m")
        end = _finite_number(event["physical_end_m"], f"{sample_id}.events[{index}].physical_end_m")
        delays = [
            _finite_number(event[name], f"{sample_id}.events[{index}].{name}")
            for name in ("delay_start_s", "delay_center_s", "delay_end_s")
        ]
        if not (0 <= start <= center <= end <= physical_length):
            raise DatasetProtocolError(f"annotation {sample_id} event {index} has invalid physical interval")
        if not (0 <= delays[0] <= delays[1] <= delays[2]):
            raise DatasetProtocolError(f"annotation {sample_id} event {index} has invalid delays")
        severity = _finite_number(event["severity"], f"{sample_id}.events[{index}].severity")
        if not 0.0 <= severity <= 1.0 or not isinstance(event["electrical_change"], Mapping):
            raise DatasetProtocolError(f"annotation {sample_id} event {index} has invalid severity/electrical_change")
        normal = dict(event)
        normalised_events.append(normal)
    result = dict(record)
    result.update(sample_id=sample_id, split=split, events=normalised_events)
    result["physical_length_m"] = physical_length
    result["reference_epsr"] = reference_epsr
    if sum(event["role"] == "terminal" for event in normalised_events) != 1:
        raise DatasetProtocolError(f"annotation {sample_id} requires exactly one terminal event")
    return result


def read_annotations(root: str | Path, split: str) -> list[dict[str, Any]]:
    split_key = validate_split(split)
    path = Path(root) / "annotations" / f"{split_key}.jsonl"
    if not path.is_file():
        raise DatasetProtocolError(f"missing annotation file: {path}")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise DatasetProtocolError(f"cannot read annotation file: {path}") from exc
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            raise DatasetProtocolError(f"blank line in {path}:{line_no}")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetProtocolError(f"invalid JSON in {path}:{line_no}") from exc
        item = validate_annotation(record, expected_split=split_key)
        if item["sample_id"] in seen:
            raise DatasetProtocolError(f"duplicate sample_id in {path}: {item['sample_id']}")
        seen.add(item["sample_id"])
        records.append(item)
    return records


def write_response(root: str | Path, profile: str, split: str, band: str, sample_id: str, *, distance_m: np.ndarray, impulse_real: np.ndarray, impulse_imag: np.ndarray, step: np.ndarray, coverage: Mapping[str, Any]) -> Path:
    path = response_path(root, profile, split, band, sample_id)
    arrays = [np.asarray(value, dtype=np.float64) for value in (distance_m, impulse_real, impulse_imag, step)]
    if any(value.ndim != 1 for value in arrays) or len({len(value) for value in arrays}) != 1 or len(arrays[0]) < 2:
        raise DatasetProtocolError("response arrays must be aligned one-dimensional arrays")
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise DatasetProtocolError("response arrays must be finite")
    coverage_obj = dict(coverage)
    json_text = json.dumps(coverage_obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, distance_m=arrays[0], impulse_real=arrays[1], impulse_imag=arrays[2], step=arrays[3], coverage_json=np.asarray(json_text))
    return path


def read_response(root: str | Path, profile: str, split: str, band: str, sample_id: str) -> dict[str, Any]:
    arrays = _load_npz(response_path(root, profile, split, band, sample_id), REQUIRED_RESPONSE_KEYS)
    names = ("distance_m", "impulse_real", "impulse_imag", "step")
    if any(arrays[name].dtype != np.float64 for name in names):
        raise DatasetProtocolError("response arrays must be float64")
    if any(arrays[name].ndim != 1 for name in names) or len({len(arrays[name]) for name in names}) != 1:
        raise DatasetProtocolError("response arrays are not aligned")
    if not all(np.all(np.isfinite(arrays[name])) for name in names):
        raise DatasetProtocolError("response arrays must be finite")
    distance = arrays["distance_m"]
    if np.any(np.diff(distance) <= 0) or np.any(distance < 0):
        raise DatasetProtocolError("distance_m must be strictly increasing and non-negative")
    raw_coverage = arrays["coverage_json"]
    if raw_coverage.ndim != 0:
        raise DatasetProtocolError("coverage_json must be a scalar string")
    try:
        coverage = json.loads(str(raw_coverage.item()))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DatasetProtocolError("invalid response coverage_json") from exc
    if not isinstance(coverage, dict):
        raise DatasetProtocolError("response coverage metadata must be an object")
    return {name: arrays[name] for name in names} | {"coverage": coverage}


class DatasetStorage:
    """Writer consumed by ``generate_dataset.py``.

    The generator calls ``write`` with a ``GeneratedSample`` carrying the exact
    protocol band names ``1ghz`` and ``200mhz``.
    """

    def __init__(self, root: str | Path, metadata: Mapping[str, Any]) -> None:
        self.root = Path(root)
        self.metadata = validate_metadata(metadata)
        write_dataset_metadata(self.root, self.metadata)
        self._counts = {"train": 0, "val": 0, "test": 0}
        self._profile_counts = {"rg58": 0, "field": 0}
        for split in SPLITS:
            path = self.root / "annotations" / f"{split}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
            for band in BANDS:
                (self.root / "frequency" / split / band).mkdir(parents=True, exist_ok=True)

    @classmethod
    def open(cls, output: str | Path, *, config: Any, seed: int, requested_samples: int) -> "DatasetStorage":
        """Create a writer from the complete DG V3 generator configuration."""
        bands = getattr(config, "bands", None)
        if bands is None or set(bands) != set(BANDS):
            raise DatasetProtocolError("generator config must declare 1ghz and 200mhz bands")
        metadata = make_dataset_metadata(
            dataset_name="DG_V3",
            point_counts={band: int(bands[band].points) for band in BANDS},
            generator_version=str(config.generator_version),
            parameter_profile=str(config.parameter_profile),
            seed=int(seed),
            requested_samples=int(requested_samples),
        )
        return cls(output, metadata)

    def _split_for_profile(self, profile: str) -> str:
        # Keep the 80/10/10 allocation independently inside both profile families.
        remainder = self._profile_counts[profile] % 10
        self._profile_counts[profile] += 1
        return "test" if remainder == 9 else ("val" if remainder == 8 else "train")

    def write(self, sample: Any) -> None:
        split = self._split_for_profile(str(sample.profile))
        sample_id = validate_sample_id(f"dg_{int(sample.sample_id):010d}")
        bands = sample.bands
        for band, data in bands.items():
            values = np.asarray(data.s11, dtype=np.complex128)
            write_frequency(self.root, split, band, sample_id, data.frequency_hz, values.real, values.imag)
        events = []
        for event in sample.truth:
            item = dict(event)
            events.append(item)
        topology = sample.topology
        annotation = {
            "sample_id": sample_id,
            "split": split,
            "profile": str(sample.profile),
            "seed": int(sample.sample_id),
            "physical_length_m": float(topology.length_m),
            "reference_epsr": float(topology.base_epsr),
            "reference_impedance_ohm": float(topology.z_ref_ohm),
            "events": events,
            "termination": str(topology.termination),
            "frequency_files": {
                band: str(frequency_path(self.root, split, band, sample_id).relative_to(self.root)).replace("\\", "/")
                for band in BANDS
            },
            "generation": dict(sample.generation),
        }
        checked = validate_annotation(annotation, expected_split=split)
        path = self.root / "annotations" / f"{split}.jsonl"
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(checked, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._counts[split] += 1

    def close(self) -> None:
        """Close the logical writer (files are written synchronously)."""
        for split in SPLITS:
            for band in BANDS:
                (self.root / "frequency" / split / band).mkdir(parents=True, exist_ok=True)
        self.metadata["split_counts"] = dict(self._counts)
        write_dataset_metadata(self.root, self.metadata)
