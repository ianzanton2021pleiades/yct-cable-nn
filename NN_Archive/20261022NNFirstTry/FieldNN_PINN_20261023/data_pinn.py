"""Training-time domain randomisation on top of the frozen Iter4/Iter5 cache.

The base view is imported unchanged from the CMDC Iter5 reader, so the clean /
stressed mixture, the fixture bank injection and the target widening are exactly
the ones the baseline used.  Two extra transforms are applied here, both with a
physical meaning:

distance warp
    a TDR distance axis is only as accurate as the assumed propagation velocity;
    field crews routinely disagree by 1-2 %.  Resampling features and targets by
    the same factor teaches invariance without inventing label noise.

mixup
    the cable corpus is small, so convex combinations of two traces (with union
    of the labelled events) enlarge the effective sample count and flatten the
    over-confident logits that produced the 0.33 train/validation gap at 100
    epochs.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
ITER5 = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_Iter5_20261022"
ITER3 = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_Iter3_20261022"
for path in (ITER5, ITER3):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from data import ShardDataset  # noqa: E402


NOISE_CHANNELS = (0, 1, 4, 5)


def stack_targets(targets: np.ndarray) -> np.ndarray:
    targets = np.asarray(targets, dtype=np.float32)
    weight = np.ones((1, targets.shape[-1]), dtype=np.float32)
    unknown = np.zeros_like(weight)
    return np.concatenate((targets, weight, unknown), axis=0)


def _resample(array: np.ndarray, scale: float) -> np.ndarray:
    length = array.shape[-1]
    source = np.clip(np.arange(length, dtype=np.float64) / scale, 0.0, length - 1)
    lower = np.floor(source).astype(np.int32)
    upper = np.minimum(lower + 1, length - 1)
    fraction = (source - lower).astype(np.float32)
    return array[:, lower] * (1.0 - fraction) + array[:, upper] * fraction


class PinnDataset(Dataset):
    """Base Iter5 view plus distance warp, feature noise and mixup."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        train: bool,
        view: str = "mixed",
        warp_prob: float = 0.5,
        warp_max: float = 0.012,
        mixup_prob: float = 0.25,
        noise_prob: float = 0.30,
        noise_sigma: float = 0.02,
        seed: int = 2026102301,
    ) -> None:
        os.environ["FIELDNN_EVAL_VIEW"] = view
        self.base = ShardDataset(root, split, train=train)
        self.train = train
        self.warp_prob = float(warp_prob)
        self.warp_max = float(warp_max)
        self.mixup_prob = float(mixup_prob)
        self.noise_prob = float(noise_prob)
        self.noise_sigma = float(noise_sigma)
        self.seed = int(seed)
        os.environ["FIELDNN_EVAL_VIEW"] = "mixed"

    def __len__(self) -> int:
        return len(self.base)

    @property
    def metadata(self):
        return self.base.metadata

    def _rng(self, index: int, tag: int) -> np.random.Generator:
        worker = torch.initial_seed() % (2**31)
        return np.random.default_rng(self.seed + tag + 7919 * worker + int(index))

    def __getitem__(self, index: int):
        features, targets, item = self.base[int(index)]
        features = np.asarray(features, dtype=np.float32).copy()
        targets = stack_targets(targets)

        if not self.train:
            return features, targets, int(item), 0

        rng = self._rng(int(index), 0)

        if rng.random() < self.mixup_prob:
            other = int(rng.integers(0, len(self.base)))
            other_features, other_targets, _ = self.base[other]
            other_targets = stack_targets(other_targets)
            lam = float(rng.uniform(0.45, 0.55))
            features = (lam * features + (1.0 - lam) * np.asarray(other_features, np.float32)).astype(np.float32)
            merged = np.empty_like(targets)
            merged[:3] = np.maximum(targets[:3], other_targets[:3])
            merged[3] = np.maximum(targets[3], other_targets[3])
            merged[4] = np.minimum(targets[4], other_targets[4])
            targets = merged

        if rng.random() < self.warp_prob:
            scale = float(rng.uniform(1.0 - self.warp_max, 1.0 + self.warp_max))
            features = _resample(features, scale).astype(np.float32)
            targets = _resample(targets, scale).astype(np.float32)
            features[7] = np.arange(features.shape[-1], dtype=np.float32) / float(features.shape[-1] - 1)

        if rng.random() < self.noise_prob:
            sigma = float(rng.uniform(0.3, 1.0)) * self.noise_sigma
            noise = rng.normal(0.0, sigma, size=(len(NOISE_CHANNELS), features.shape[-1])).astype(np.float32)
            features[list(NOISE_CHANNELS)] += noise

        return features, targets, int(item), 0


def load_hybrid_metadata(root: str | Path, split: str) -> tuple[list[dict], np.ndarray]:
    root = Path(root)
    records = [
        json.loads(line)
        for line in (root / f"hybrid_records_{split}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return records, np.load(root / f"hybrid_features_{split}.npy", mmap_mode="r")
