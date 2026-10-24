"""Iter5 raw-cache reader: width-aware targets and stronger fixture adaptation.

The raw impulse/step/target cache is reused read-only from Iter4. The only
training-distribution changes are:
  * augmented curves receive Gaussian targets whose width follows the input blur,
  * the real CalibrationData fixture bank is injected at a stronger scale.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import Dataset


HERE = Path(__file__).resolve().parent
ITER3 = HERE.parent / "FieldNN_Iter3_20261022"
if str(ITER3) not in sys.path:
    sys.path.append(str(ITER3))
from features import distance_features


EVENT_BASE_SIGMA_M = 5.0
TERMINAL_BASE_SIGMA_M = 5.0
WIDTH_COUPLING = 0.6
EVENT_LABEL_SIGMA_CAP_M = 20.0
TERMINAL_LABEL_SIGMA_CAP_M = 22.0
BANK_SCALE_RANGE = (1.0, 6.0)
CLEAN_REPLAY_FRACTION = 0.35


class ShardDataset(Dataset):
    def __init__(self, root: str | Path, split: str, train: bool = False) -> None:
        self.root = Path(root)
        self.split = split
        self.train = train
        self.cfg = json.loads((self.root / "config.json").read_text(encoding="utf-8"))
        response = self.cfg["response"]
        self.distance = np.linspace(
            0.0, float(response["distance_max_m"]), int(response["points"]), dtype=np.float64
        )
        self.spacing_m = float(self.distance[1] - self.distance[0])
        self.seed = int(self.cfg.get("seed", 20261022))
        self.view = "train" if train else os.environ.get("FIELDNN_EVAL_VIEW", "mixed").strip().lower()
        if not train and self.view not in {"clean", "stress", "mixed"}:
            raise ValueError("FIELDNN_EVAL_VIEW must be clean, stress, or mixed")

        arrays = self.root / "arrays"
        self._imp_path = arrays / f"{split}_raw_imp.npy"
        self._step_path = arrays / f"{split}_raw_step.npy"
        self._target_path = arrays / f"{split}_targets.npy"
        shape_probe = np.load(self._imp_path, mmap_mode="r")
        self._count = int(shape_probe.shape[0])
        if getattr(shape_probe, "_mmap", None) is not None:
            shape_probe._mmap.close()
        self._event_centers, self._terminal_centers, self._min_center_separation = (
            self._read_centers()
        )

        self._raw_imp: np.ndarray | None = None
        self._raw_step: np.ndarray | None = None
        self._targets: np.ndarray | None = None
        self._metadata: list[dict[str, Any]] | None = None
        self._bank_imp: np.ndarray | None = None
        self._bank_step: np.ndarray | None = None
        self._rng: np.random.Generator | None = None

    def __len__(self) -> int:
        return self._count

    def _read_centers(self) -> tuple[list[list[float]], list[list[float]], np.ndarray]:
        path = self.root / f"metadata_{self.split}.jsonl"
        events: list[list[float]] = []
        terminals: list[list[float]] = []
        minimums = np.full(self._count, np.inf, dtype=np.float32)
        with path.open("r", encoding="utf-8") as stream:
            index = 0
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                centers = [float(item) for item in row.get("event_positions_m", [])]
                terminal = row.get("terminal_position_m")
                events.append(centers)
                terminals.append([] if terminal is None else [float(terminal)])
                all_centers = np.unique(np.round(np.asarray(centers + terminals[-1], dtype=np.float64), 6))
                if all_centers.size > 1:
                    minimums[index] = float(np.diff(all_centers).min())
                index += 1
        if index != self._count:
            raise ValueError(f"metadata count {index} does not match {self.split} array count {self._count}")
        return events, terminals, minimums

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        for key in (
            "_raw_imp", "_raw_step", "_targets", "_metadata", "_bank_imp", "_bank_step", "_rng"
        ):
            state[key] = None
        return state

    @property
    def metadata(self) -> list[dict[str, Any]]:
        if self._metadata is None:
            path = self.root / f"metadata_{self.split}.jsonl"
            with path.open("r", encoding="utf-8") as stream:
                self._metadata = [json.loads(line) for line in stream if line.strip()]
            if len(self._metadata) != self._count:
                raise ValueError(f"metadata count does not match {self.split} array count")
        return self._metadata

    def _open_arrays(self) -> None:
        if self._raw_imp is None:
            self._raw_imp = np.load(self._imp_path, mmap_mode="r")
            self._raw_step = np.load(self._step_path, mmap_mode="r")
            self._targets = np.load(self._target_path, mmap_mode="r")

    def _open_calibration_bank(self) -> None:
        if self._bank_imp is None:
            bank_name = self.cfg.get("calibration_bank", {}).get("array_file", "calibration_bank_v2.npz")
            with np.load(self.root / bank_name, allow_pickle=False) as bank:
                self._bank_imp = np.asarray(bank["impulse"], dtype=np.float32).copy()
                self._bank_step = np.asarray(bank["step"], dtype=np.float32).copy()

    def _train_rng(self) -> np.random.Generator:
        if self._rng is None:
            self._rng = np.random.default_rng(torch.initial_seed() % (2**32))
        return self._rng

    def _index_rng(self, index: int) -> np.random.Generator:
        split_offset = {"train": 0, "val": 1_000_003, "test": 2_000_033}.get(self.split, 3_000_091)
        return np.random.default_rng(self.seed + split_offset + int(index))

    def _augment(
        self,
        impulse: np.ndarray,
        step: np.ndarray,
        rng: np.random.Generator,
        min_center_separation_m: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        impulse = impulse.copy()
        step = step.copy()

        sigma_max_m = 40.0
        if np.isfinite(min_center_separation_m):
            sigma_max_m = min(sigma_max_m, float(min_center_separation_m) / 3.0)
        sigma_min_m = min(2.0, sigma_max_m)
        sigma_m = (
            float(rng.uniform(sigma_min_m, sigma_max_m))
            if sigma_max_m > sigma_min_m
            else sigma_max_m
        )
        sigma_points = sigma_m / self.spacing_m
        impulse = gaussian_filter1d(impulse, sigma=sigma_points, mode="nearest")
        step = gaussian_filter1d(step, sigma=sigma_points, mode="nearest")

        end_gain = float(rng.uniform(0.3, 3.0))
        distance_gain = np.exp(np.linspace(0.0, np.log(end_gain), impulse.size)).astype(np.float32)
        impulse *= distance_gain
        step_delta = np.diff(step, prepend=step[0])
        step = step[0] + np.cumsum(step_delta * distance_gain)

        near = self.distance <= 200.0
        cable_near_rms = float(np.sqrt(np.mean(impulse[near] ** 2) + 1e-12))
        self._open_calibration_bank()
        bank_index = int(rng.integers(0, self._bank_imp.shape[0]))
        bank_imp = self._bank_imp[bank_index]
        bank_step = self._bank_step[bank_index]
        bank_near_rms = float(np.sqrt(np.mean(bank_imp[near] ** 2) + 1e-12))
        calibration_scale = (
            cable_near_rms * float(rng.uniform(*BANK_SCALE_RANGE)) / max(bank_near_rms, 1e-12)
        )
        impulse += calibration_scale * bank_imp
        step += calibration_scale * bank_step

        common_gain = float(np.exp(rng.uniform(np.log(0.05), np.log(20.0))))
        impulse *= common_gain
        step *= common_gain
        return (
            impulse.astype(np.float32, copy=False),
            step.astype(np.float32, copy=False),
            sigma_m,
        )

    def _widen_targets(self, targets: np.ndarray, index: int, sigma_m: float) -> np.ndarray:
        if sigma_m <= 0.0:
            return targets
        for channel, centers, base_sigma, cap in (
            (0, self._event_centers[index], EVENT_BASE_SIGMA_M, EVENT_LABEL_SIGMA_CAP_M),
            (2, self._terminal_centers[index], TERMINAL_BASE_SIGMA_M, TERMINAL_LABEL_SIGMA_CAP_M),
        ):
            if not centers:
                continue
            sigma = min(cap, float(np.hypot(base_sigma, WIDTH_COUPLING * sigma_m)))
            if sigma <= base_sigma * 1.05:
                continue
            target = targets[channel]
            valid = target >= 0.0
            for center in centers:
                radius = 4.0 * sigma
                low = max(0, int(np.searchsorted(self.distance, center - radius)))
                high = min(target.size, int(np.searchsorted(self.distance, center + radius)) + 1)
                if high - low < 3:
                    continue
                window = self.distance[low:high]
                gaussian = np.exp(-0.5 * ((window - center) / sigma) ** 2).astype(np.float32)
                target[low:high] = np.where(
                    valid[low:high], np.maximum(target[low:high], gaussian), target[low:high]
                )
        return targets

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, int]:
        self._open_arrays()
        impulse = np.asarray(self._raw_imp[index], dtype=np.float32).copy()
        step = np.asarray(self._raw_step[index], dtype=np.float32).copy()
        targets = np.asarray(self._targets[index], dtype=np.float32).copy()

        sigma_m = 0.0
        if self.train:
            rng = self._train_rng()
            augmented = rng.random() >= CLEAN_REPLAY_FRACTION
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
        features = distance_features(impulse, step, self.distance)
        return features, targets, int(index)
