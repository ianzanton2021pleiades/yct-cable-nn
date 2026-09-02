"""Strict structural validator for DG V3 datasets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from storage import BANDS, DatasetProtocolError, load_dataset_metadata, read_annotations, read_frequency, read_response


def _npz_stems(directory: Path) -> set[str]:
    if not directory.is_dir():
        raise DatasetProtocolError(f"missing protocol directory: {directory}")
    files = list(directory.iterdir())
    unexpected = [path.name for path in files if path.is_file() and path.suffix != ".npz"]
    if unexpected:
        raise DatasetProtocolError(f"unexpected files in {directory}: {unexpected}")
    return {path.stem for path in files if path.is_file() and path.suffix == ".npz"}


def validate_dataset(root: str | Path, *, profile: str | None = None) -> dict[str, Any]:
    root_path = Path(root)
    metadata = load_dataset_metadata(root_path)
    profiles = [profile] if profile is not None else list(metadata["response_profiles"])
    for name in profiles:
        if name not in metadata["response_profiles"]:
            raise DatasetProtocolError(f"response profile is not declared: {name}")
    report: dict[str, Any] = {"schema": metadata["schema"], "version": metadata["version"], "splits": {}, "profiles": profiles}
    all_sample_ids: set[str] = set()
    observed_split_counts: dict[str, int] = {}
    for split in metadata["splits"]:
        records = read_annotations(root_path, split)
        sample_ids = {record["sample_id"] for record in records}
        if len(sample_ids) != len(records):
            raise DatasetProtocolError(f"duplicate samples in split {split}")
        overlap = all_sample_ids.intersection(sample_ids)
        if overlap:
            raise DatasetProtocolError(f"sample ids appear in multiple splits: {sorted(overlap)}")
        all_sample_ids.update(sample_ids)
        observed_split_counts[split] = len(records)
        split_report = {
            "samples": len(records),
            "profiles": {
                name: sum(record["profile"] == name for record in records)
                for name in ("rg58", "field")
            },
            "terminations": {
                name: sum(record["termination"] == name for record in records)
                for name in ("open", "short")
            },
            "bands": {},
        }
        for band in BANDS:
            directory = root_path / "frequency" / split / band
            actual = _npz_stems(directory)
            if actual != sample_ids:
                missing, extra = sorted(sample_ids - actual), sorted(actual - sample_ids)
                raise DatasetProtocolError(f"frequency/annotation pairing mismatch for {split}/{band}; missing={missing}, extra={extra}")
            expected_count = int(metadata["bands"][band]["point_count"])
            for record in records:
                sample_id = record["sample_id"]
                arrays = read_frequency(root_path, split, band, sample_id)
                if len(arrays["frequency_hz"]) != expected_count:
                    raise DatasetProtocolError(f"{split}/{band}/{sample_id}: point count mismatch")
                frequency = arrays["frequency_hz"]
                minimum = float(metadata["bands"][band]["frequency_min_hz"])
                maximum = float(metadata["bands"][band]["frequency_max_hz"])
                if float(frequency[0]) < minimum or float(frequency[-1]) > maximum:
                    raise DatasetProtocolError(f"{split}/{band}/{sample_id}: frequency range outside dataset.json")
            split_report["bands"][band] = {"samples": len(records), "point_count": expected_count}
        report["splits"][split] = split_report

    if observed_split_counts != metadata["split_counts"]:
        raise DatasetProtocolError(
            f"dataset.json split_counts mismatch: {metadata['split_counts']} != {observed_split_counts}"
        )
    if sum(observed_split_counts.values()) != int(metadata["requested_samples"]):
        raise DatasetProtocolError("dataset sample count does not match requested_samples")

    for profile_name in profiles:
        profile_report: dict[str, Any] = {"samples": sum(observed_split_counts.values()), "bands": {}}
        for split in metadata["splits"]:
            sample_ids = {record["sample_id"] for record in read_annotations(root_path, split)}
            for band in BANDS:
                directory = root_path / "responses" / profile_name / split / band
                actual = _npz_stems(directory)
                if actual != sample_ids:
                    missing, extra = sorted(sample_ids - actual), sorted(actual - sample_ids)
                    raise DatasetProtocolError(f"response/annotation pairing mismatch for {profile_name}/{split}/{band}; missing={missing}, extra={extra}")
                for sample_id in sorted(sample_ids):
                    response = read_response(root_path, profile_name, split, band, sample_id)
                    coverage = response["coverage"]
                    expected = {"sample_id": sample_id, "split": split, "band": band, "profile": profile_name}
                    for key, value in expected.items():
                        if coverage.get(key) != value:
                            raise DatasetProtocolError(f"{profile_name}/{split}/{band}/{sample_id}: coverage {key} mismatch")
                    required = (
                        "source_frequency_min_hz", "source_frequency_max_hz", "valid_distance_max_m",
                        "target_distance_max_m", "distance_step_m", "point_count", "algorithm", "window",
                        "dc_mode", "valid_distance_extrapolated", "truncated_by_ifft_range",
                        "analysis_epsr", "terminal_apparent_distance_m", "terminal_observable",
                    )
                    if any(key not in coverage for key in required):
                        missing = [key for key in required if key not in coverage]
                        raise DatasetProtocolError(f"{profile_name}/{split}/{band}/{sample_id}: coverage missing {missing}")
                    if coverage["algorithm"] != "client_hann_v1" or coverage["window"] != "hann" or coverage["dc_mode"] != "client_extrapolation" or coverage["valid_distance_extrapolated"] is not False:
                        raise DatasetProtocolError(f"{profile_name}/{split}/{band}/{sample_id}: incompatible coverage metadata")
                    distance = response["distance_m"]
                    if int(coverage["point_count"]) != len(distance) or abs(float(coverage["valid_distance_max_m"]) - float(distance[-1])) > 1e-9:
                        raise DatasetProtocolError(f"{profile_name}/{split}/{band}/{sample_id}: coverage length mismatch")
                    if abs(float(coverage["distance_step_m"]) - 0.25) > 1e-12:
                        raise DatasetProtocolError(f"{profile_name}/{split}/{band}/{sample_id}: distance step must be 0.25")
                    if len(distance) > 1 and not np.allclose(np.diff(distance), 0.25, rtol=0.0, atol=1e-10):
                        raise DatasetProtocolError(f"{profile_name}/{split}/{band}/{sample_id}: distance grid is not 0.25 m")
                profile_report["bands"].setdefault(band, 0)
                profile_report["bands"][band] += len(sample_ids)
        report.setdefault("responses", {})[profile_name] = profile_report
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a DG V3 dataset")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    try:
        report = validate_dataset(args.dataset_root, profile=args.profile)
    except DatasetProtocolError as exc:
        parser.error(str(exc))
    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        total = sum(int(info["samples"]) for info in report["splits"].values())
        print(f"DG V3 validation passed: {total} annotated samples; profiles={','.join(report['profiles'])}")


if __name__ == "__main__":
    main()
