"""Framework-agnostic DG V3 dataset reader returning NumPy records."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np

from storage import (
    BANDS,
    PROFILE,
    DatasetProtocolError,
    load_dataset_metadata,
    read_annotations,
    read_frequency,
    read_response,
    response_path,
)


class CableDataset:
    """Index one DG V3 split (or the complete dataset).

    ``__getitem__`` returns a plain dictionary containing float64 NumPy arrays
    and sparse JSON-compatible event dictionaries.  No normalization,
    padding, Tensor conversion, or framework dependency is performed.
    """

    def __init__(self, root: str | Path, split: str | None = None, *, profile: str = PROFILE, load_responses: bool = True) -> None:
        self.root = Path(root)
        self.metadata = load_dataset_metadata(self.root)
        self.profile = str(profile)
        if self.profile not in self.metadata["response_profiles"]:
            raise DatasetProtocolError(f"response profile is not declared: {self.profile}")
        if split is None:
            selected_splits = tuple(self.metadata["splits"])
        else:
            selected_splits = (str(split),)
            if selected_splits[0] not in self.metadata["splits"]:
                raise DatasetProtocolError(f"invalid split: {split!r}")
        self.load_responses = bool(load_responses)
        self._items: list[dict[str, Any]] = []
        for split_name in selected_splits:
            self._items.extend(read_annotations(self.root, split_name))
        # A dataset record is only usable when every declared frequency file is
        # present.  Fail during construction so a training loop cannot silently
        # skip a pair.
        for item in self._items:
            sample_id, split_name = item["sample_id"], item["split"]
            for band in BANDS:
                path = self.root / "frequency" / split_name / band / f"{sample_id}.npz"
                if not path.is_file():
                    raise DatasetProtocolError(f"missing frequency pair for {sample_id}: {path}")
                if self.load_responses:
                    path = response_path(self.root, self.profile, split_name, band, sample_id)
                    if not path.is_file():
                        raise DatasetProtocolError(f"missing response pair for {sample_id}: {path}")

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self._items[index]
        sample_id, split_name = item["sample_id"], item["split"]
        frequency: dict[str, dict[str, np.ndarray]] = {}
        responses: dict[str, dict[str, Any]] = {}
        for band in BANDS:
            frequency[band] = read_frequency(self.root, split_name, band, sample_id)
            if self.load_responses:
                responses[band] = read_response(self.root, self.profile, split_name, band, sample_id)
        metadata = {key: value for key, value in item.items() if key not in {"sample_id", "split", "events"}}
        return {
            "sample_id": sample_id,
            "split": split_name,
            "frequency": frequency,
            "responses": responses,
            "events": item["events"],
            "metadata": metadata,
        }
