"""Datasets for synthetic replay, injected measured-field traces, and calibration shorts."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
ITER3 = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_Iter3_20261022"
ITER5 = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_Iter5_20261022"
if str(ITER3) not in sys.path:
    sys.path.append(str(ITER3))
from features import distance_features


_spec = importlib.util.spec_from_file_location("hybrid_iter5_data", ITER5 / "data.py")
_iter5_data = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _iter5_data
_spec.loader.exec_module(_iter5_data)
_iter5_data.distance_features = distance_features


def _stack_targets(targets: np.ndarray) -> np.ndarray:
    targets = np.asarray(targets, dtype=np.float32)
    weight = np.ones((1, targets.shape[-1]), dtype=np.float32)
    unknown = np.zeros_like(weight)
    return np.concatenate((targets, weight, unknown), axis=0)


class SyntheticWrapper(Dataset):
    """Iter5/Iter4 synthetic raw cache with a supervision-weight channel."""

    def __init__(self, root: str | Path, split: str, train: bool, view: str = "mixed") -> None:
        os.environ["FIELDNN_EVAL_VIEW"] = view
        self.base = _iter5_data.ShardDataset(root, split, train=train)

    def __len__(self) -> int:
        return len(self.base)

    @property
    def metadata(self):
        return self.base.metadata

    def __getitem__(self, index: int):
        features, targets, item = self.base[index]
        return features, _stack_targets(targets), item, 0


class HybridDataset(Dataset):
    """Injected measured-field traces with weighted partial labels."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        augment: bool = False,
        mixup_prob: float = 0.0,
        distance_max_m: float = 3250.0,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.augment = augment
        self.mixup_prob = float(mixup_prob)
        self.features = np.load(self.root / f"hybrid_features_{split}.npy", mmap_mode="r")
        self.targets = np.load(self.root / f"hybrid_targets_{split}.npy", mmap_mode="r")
        self.records = [
            json.loads(line)
            for line in (self.root / f"hybrid_records_{split}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        if len(self.records) != self.features.shape[0]:
            raise ValueError("hybrid record count does not match feature count")
        self.seed = 2026102207
        self.distance = np.linspace(
            0.0, float(distance_max_m), int(self.features.shape[-1]), dtype=np.float64
        )

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int):
        features = np.asarray(self.features[index], dtype=np.float32).copy()
        if self.augment:
            rng = np.random.default_rng(self.seed + int(index))
            if rng.random() < 0.30:
                sigma = float(rng.uniform(0.005, 0.02))
                features += rng.normal(0.0, sigma, size=features.shape).astype(np.float32)
            if rng.random() < 0.30:
                gain = float(rng.uniform(0.97, 1.03))
                features[0] *= gain
                features[1] *= gain
        targets = np.asarray(self.targets[index], dtype=np.float32).copy()
        if self.augment and self.mixup_prob > 0.0:
            rng = np.random.default_rng(self.seed + int(index) + 1_000_003)
            if rng.random() < self.mixup_prob:
                other = int(rng.integers(0, self.features.shape[0]))
                other_features = np.asarray(self.features[other], dtype=np.float32)
                other_targets = np.asarray(self.targets[other], dtype=np.float32)
                lam = float(rng.uniform(0.35, 0.65))
                features = (lam * features + (1.0 - lam) * other_features).astype(np.float32)
                event = np.maximum(targets[0], other_targets[0])
                interval = np.maximum(targets[1], other_targets[1])
                terminal = np.maximum(targets[2], other_targets[2])
                weight = np.maximum(targets[3], other_targets[3])
                unknown = ((weight <= 0.0) & (self.distance >= 30.0)).astype(np.float32)
                targets = np.stack((event, interval, terminal, weight, unknown))
        return features, targets, int(index), 1


class FieldPositiveDataset(Dataset):
    """Original measured traces with engineer positions as weak positives.

    This stream teaches the real event shapes without injected reflections.  It
    contains only traces that have at least one high-confidence engineer
    position; every other location remains unknown.
    """

    def __init__(self, root: str | Path, split: str, cfg: dict, augment: bool = False) -> None:
        self.root = Path(root)
        self.augment = augment
        self.features = np.load(self.root / "field_features.npy", mmap_mode="r")
        records = [
            json.loads(line)
            for line in (self.root / "field_records.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        self.records = [
            record for record in records
            if record.get("split") == split and record.get("known_positions_nominal_m")
        ]
        self.distance = np.linspace(
            0.0,
            float(cfg["response"]["distance_max_m"]),
            int(self.features.shape[-1]),
            dtype=np.float64,
        )
        self.seed = 2026102207

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, item: int):
        record = self.records[item]
        index = int(record["index"])
        features = np.asarray(self.features[index], dtype=np.float32).copy()
        if self.augment:
            rng = np.random.default_rng(self.seed + index)
            if rng.random() < 0.30:
                sigma = float(rng.uniform(0.005, 0.02))
                features += rng.normal(0.0, sigma, size=features.shape).astype(np.float32)
            if rng.random() < 0.30:
                gain = float(rng.uniform(0.97, 1.03))
                features[0] *= gain
                features[1] *= gain

        targets = np.full((5, self.distance.size), -1.0, dtype=np.float32)
        targets[3] = 0.0
        targets[4] = 0.0
        for position in record["known_positions_nominal_m"]:
            center = float(position)
            sigma = 15.0
            lo = max(0, int(np.searchsorted(self.distance, center - 4.0 * sigma)))
            hi = min(self.distance.size, int(np.searchsorted(self.distance, center + 4.0 * sigma)) + 1)
            if hi - lo < 3:
                continue
            gaussian = np.exp(
                -0.5 * ((self.distance[lo:hi] - center) / sigma) ** 2
            ).astype(np.float32)
            existing = targets[0, lo:hi]
            targets[0, lo:hi] = np.where(existing < 0.0, gaussian, np.maximum(existing, gaussian))
            targets[3, lo:hi] = np.maximum(targets[3, lo:hi], 0.75)
        targets[4] = ((targets[3] <= 0.0) & (self.distance >= 30.0)).astype(np.float32)
        return features, targets, index, 2


class CalibrationDataset(Dataset):
    """Fixture-short/no-cable references: event/interval negatives at d>=30 m.

    These are not healthy cables and carry no terminal truth.
    """

    def __init__(self, root: str | Path, split: str, cfg: dict) -> None:
        self.root = Path(root)
        self.imp = np.load(self.root / "calibration_raw_imp.npy", mmap_mode="r")
        self.step = np.load(self.root / "calibration_raw_step.npy", mmap_mode="r")
        self.records = [
            json.loads(line)
            for line in (self.root / "calibration_records.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        split_map_path = self.root / "calibration_split.json"
        if split_map_path.is_file():
            split_map = json.loads(split_map_path.read_text(encoding="utf-8"))
        else:
            split_map = {str(row["index"]): row.get("split", "train") for row in self.records}
        self.indices = [
            int(row["index"])
            for row in self.records
            if split_map.get(str(row["index"]), "train") == split
        ]
        self.distance = np.linspace(
            0.0,
            float(cfg["response"]["distance_max_m"]),
            int(self.imp.shape[-1]),
            dtype=np.float64,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = self.indices[item]
        impulse = np.asarray(self.imp[index], dtype=np.float32)
        step = np.asarray(self.step[index], dtype=np.float32)
        features = distance_features(impulse, step, self.distance)
        targets = np.full((5, self.distance.size), -1.0, dtype=np.float32)
        targets[3] = 0.0
        targets[4] = 0.0
        valid = self.distance >= 30.0
        targets[0, valid] = 0.0
        targets[1, valid] = 0.0
        targets[3, valid] = 1.0
        return features, targets, int(index), 3


class FieldDataset(Dataset):
    """Original measured-field features for evaluation (no injection)."""

    def __init__(self, root: str | Path, split: str | None = None) -> None:
        self.root = Path(root)
        self.features = np.load(self.root / "field_features.npy", mmap_mode="r")
        self.records = [
            json.loads(line)
            for line in (self.root / "field_records.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        if len(self.records) != self.features.shape[0]:
            raise ValueError("field record count does not match feature count")
        if split is None:
            self.indices = list(range(len(self.records)))
        else:
            self.indices = [i for i, record in enumerate(self.records) if record["split"] == split]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = self.indices[item]
        return np.asarray(self.features[index], dtype=np.float32).copy(), int(index)
