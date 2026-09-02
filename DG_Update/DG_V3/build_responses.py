"""Build DG V3 distance responses from stored frequency-domain NPZ files."""
from __future__ import annotations

import argparse
from pathlib import Path

from response import s11_to_responses
from storage import (
    BANDS,
    PROFILE,
    DatasetProtocolError,
    load_dataset_metadata,
    read_annotations,
    read_frequency,
    write_response,
)


def build_responses(root: str | Path, *, profile: str = PROFILE, split: str | None = None) -> int:
    """Build all requested response files and return the number written."""
    root_path = Path(root)
    metadata = load_dataset_metadata(root_path)
    if profile not in metadata["response_profiles"]:
        raise DatasetProtocolError(f"response profile is not declared: {profile}")
    profile_spec = metadata["response_profiles"][profile]
    distance_step_m = float(profile_spec["distance_step_m"])
    split_names = [str(split)] if split is not None else list(metadata["splits"])
    for split_name in split_names:
        if split_name not in metadata["splits"]:
            raise DatasetProtocolError(f"invalid split: {split_name!r}")
        for band in BANDS:
            (root_path / "responses" / profile / split_name / band).mkdir(parents=True, exist_ok=True)
    written = 0
    total = sum(len(read_annotations(root_path, name)) for name in split_names) * len(BANDS)
    for split_name in split_names:
        records = read_annotations(root_path, split_name)
        for record in records:
            sample_id = record["sample_id"]
            epsr = float(record["reference_epsr"])
            target_distance = 1.2 * float(record["physical_length_m"])
            terminal = next(event for event in record["events"] if event["role"] == "terminal")
            terminal_apparent = (
                float(terminal["delay_center_s"]) * 299_792_458.0 / (2.0 * epsr ** 0.5)
            )
            for band in BANDS:
                arrays = read_frequency(root_path, split_name, band, sample_id)
                expected_count = int(metadata["bands"][band]["point_count"])
                if len(arrays["frequency_hz"]) != expected_count:
                    raise DatasetProtocolError(
                        f"{sample_id} {split_name}/{band}: point count {len(arrays['frequency_hz'])} != {expected_count}"
                    )
                result = s11_to_responses(
                    arrays["frequency_hz"],
                    arrays["s11_real"] + 1j * arrays["s11_imag"],
                    epsr=epsr,
                    distance_step_m=distance_step_m,
                    target_distance_max_m=target_distance,
                )
                distance, impulse_real, impulse_imag, step, coverage = result
                coverage.update({
                    "sample_id": sample_id,
                    "split": split_name,
                    "band": band,
                    "profile": profile,
                    "analysis_epsr": epsr,
                    "terminal_apparent_distance_m": terminal_apparent,
                    "terminal_observable": terminal_apparent <= float(coverage["valid_distance_max_m"]),
                })
                write_response(
                    root_path,
                    profile,
                    split_name,
                    band,
                    sample_id,
                    distance_m=distance,
                    impulse_real=impulse_real,
                    impulse_imag=impulse_imag,
                    step=step,
                    coverage=coverage,
                )
                written += 1
                if written == 1 or written == total or written % max(1, min(50, total // 10 or 1)) == 0:
                    print(f"responses [{written}/{total}] {split_name}/{band}/{sample_id}", flush=True)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DG V3 Client-compatible responses")
    parser.add_argument("dataset_root", type=Path, help="path containing dataset.json")
    parser.add_argument("--profile", default=PROFILE)
    parser.add_argument("--split", choices=("train", "val", "test"), default=None)
    args = parser.parse_args()
    count = build_responses(args.dataset_root, profile=args.profile, split=args.split)
    print(f"built {count} response files under {args.dataset_root / 'responses' / args.profile}")


if __name__ == "__main__":
    main()
