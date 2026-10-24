"""Datasets for PINN-v2.

Synthetic shards keep the frozen Iter5 augmentation (fixture bank injection,
width-coupled labels, clean replay) and swap only the feature stack.  The
measured-field tensors are cached as eight channels, so they are extended in
place with the invertible reconstruction described in ``features_v2``.

The training transform adds two domain randomisations on top of the Iter5 set:
a distance-axis warp (velocity uncertainty) and mixup over trace pairs, both
inherited unchanged from the PINN-v1 experiment so the comparison stays
controlled.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
CMDC = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_Hybrid_20261022"
ITER5 = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_Iter5_20261022"
PINN1 = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_PINN_20261023"
for path in (HERE, CMDC, ITER5, PINN1):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data import ShardDataset  # noqa: E402  (Iter5 reader)
from data_pinn import PinnDataset  # noqa: E402
from features_v2 import distance_features_v2, extend_features  # noqa: E402
from hybrid_data import (  # noqa: E402
    CalibrationDataset,
    FieldDataset,
    FieldPositiveDataset,
    HybridDataset,
)


class ShardV2(ShardDataset):
    """Iter5 shard reader with the eleven-channel front end."""

    def __getitem__(self, index: int):
        self._open_arrays()
        impulse = np.asarray(self._raw_imp[index], dtype=np.float32).copy()
        step = np.asarray(self._raw_step[index], dtype=np.float32).copy()
        targets = np.asarray(self._targets[index], dtype=np.float32).copy()

        if self.train:
            rng = self._train_rng()
            augmented = rng.random() >= 0.35
        elif self.view == "clean":
            augmented, rng = False, None
        elif self.view == "stress":
            augmented, rng = True, self._index_rng(index)
        else:
            augmented = int(index) % 2 == 1
            rng = self._index_rng(index) if augmented else None

        if augmented:
            impulse, step, sigma_m = self._augment(
                impulse, step, rng, float(self._min_center_separation[index])
            )
            targets = self._widen_targets(targets, index, sigma_m)
        features = distance_features_v2(impulse, step, self.distance)
        return features, targets, int(index)


class PinnV2Dataset(PinnDataset):
    """Iter5 view with the eleven-channel stack, warp, mixup and feature noise."""

    def __init__(
        self,
        root,
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
        self.base = ShardV2(root, split, train=train)
        self.train = train
        self.warp_prob = float(warp_prob)
        self.warp_max = float(warp_max)
        self.mixup_prob = float(mixup_prob)
        self.noise_prob = float(noise_prob)
        self.noise_sigma = float(noise_sigma)
        self.seed = int(seed)
        os.environ["FIELDNN_EVAL_VIEW"] = "mixed"


class HybridV2(HybridDataset):
    def __getitem__(self, index: int):
        features, targets, item, kind = super().__getitem__(index)
        return extend_features(features, self.distance), targets, item, kind


class FieldPositiveV2(FieldPositiveDataset):
    def __getitem__(self, index: int):
        features, targets, item, kind = super().__getitem__(index)
        return extend_features(features, self.distance), targets, item, kind


class FieldV2(FieldDataset):
    def __init__(self, root, split=None, distance_max_m: float = 3250.0) -> None:
        super().__init__(root, split)
        self.distance = np.linspace(
            0.0, float(distance_max_m), int(self.features.shape[-1]), dtype=np.float64
        )

    def __getitem__(self, item: int):
        features, index = super().__getitem__(item)
        return extend_features(features, self.distance), index


class CalibrationV2(CalibrationDataset):
    def __getitem__(self, item: int):
        features, targets, index, kind = super().__getitem__(item)
        return extend_features(features, self.distance), targets, index, kind
