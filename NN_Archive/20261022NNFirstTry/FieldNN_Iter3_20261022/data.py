"""Lazy, worker-local reader for manifest-indexed FieldNN shards."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class ShardDataset(Dataset):
    """Read inputs and targets from ``shards/<split>/<id>`` NumPy files.

    Shard arrays and their JSONL metadata are opened only when needed. Each
    process keeps at most four open array shards; ``__getstate__`` leaves the
    Windows-spawned DataLoader workers with an empty cache.
    """

    def __init__(self, root: str | Path, split: str, cache_shards: int = 4) -> None:
        self.root = Path(root)
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.shards = [item for item in manifest["shards"] if item["split"] == split]
        self.shards.sort(key=lambda item: int(item.get("shard_index", 0)))
        self.ends = np.cumsum([int(item["count"]) for item in self.shards], dtype=np.int64)
        self.cache_shards = cache_shards
        self._array_cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._metadata: list[dict[str, Any]] | None = None

    def __len__(self) -> int:
        return int(self.ends[-1]) if len(self.ends) else 0

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_array_cache"] = OrderedDict()
        state["_metadata"] = None
        return state

    @property
    def metadata(self) -> list[dict[str, Any]]:
        """Load the split's metadata once, on demand, for evaluation callers."""
        if self._metadata is None:
            rows: list[dict[str, Any]] = []
            for shard in self.shards:
                path = self.root / shard["path"] / "metadata.jsonl"
                rows.extend(
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line
                )
            if len(rows) != len(self):
                raise ValueError(f"{len(rows)} metadata rows for {len(self)} {self.shards[0]['split']} samples")
            self._metadata = rows
        return self._metadata

    def _load_arrays(self, shard_index: int) -> tuple[np.ndarray, np.ndarray]:
        cached = self._array_cache.pop(shard_index, None)
        if cached is not None:
            self._array_cache[shard_index] = cached
            return cached

        folder = self.root / self.shards[shard_index]["path"]
        arrays = (
            np.load(folder / "inputs.npy", mmap_mode="r"),
            np.load(folder / "targets.npy", mmap_mode="r"),
        )
        self._array_cache[shard_index] = arrays
        if len(self._array_cache) > self.cache_shards:
            _, evicted = self._array_cache.popitem(last=False)
            for array in evicted:
                mmap = getattr(array, "_mmap", None)
                if mmap is not None:
                    mmap.close()
        return arrays

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, int]:
        shard_index = int(np.searchsorted(self.ends, index, side="right"))
        shard_start = int(self.ends[shard_index - 1]) if shard_index else 0
        local_index = int(index) - shard_start
        inputs, targets = self._load_arrays(shard_index)
        return inputs[local_index].copy(), targets[local_index].copy(), int(index)
