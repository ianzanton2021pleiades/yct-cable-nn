"""Memmap-backed v3 dataset with CPU-side distance-domain augmentation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from features_v3 import distance_features

ROOT = Path(r"E:\CableNNDataSet\Deepseek_CMDC\FieldNN_Iter2")
DISTANCE = np.linspace(0.0, 3250.0, 8192, dtype=np.float64)


_WORKER_RNG: np.random.Generator | None = None


def _rng(index: int, epoch: int) -> np.random.Generator:
    global _WORKER_RNG
    if _WORKER_RNG is None:
        _WORKER_RNG = np.random.default_rng(0x5EED2026)
    return _WORKER_RNG


def augment(imp: np.ndarray, step: np.ndarray, d: np.ndarray, rng: np.random.Generator):
    imp = imp.copy()
    step = step.copy()
    gain_tau = float(rng.uniform(120.0, 350.0))
    local_width = float(rng.uniform(35.0, 90.0))

    loss_length = float(rng.uniform(1200.0, 8000.0))
    attenuation = np.exp(-d / loss_length)
    imp *= attenuation
    step *= attenuation

    imp_rms = float(np.sqrt(np.mean(imp * imp)) + 1e-12)
    step_rms = float(np.sqrt(np.mean(step * step)) + 1e-12)
    imp += rng.normal(0.0, rng.uniform(0.0, 0.04) * imp_rms, imp.shape).astype(np.float32)
    step += rng.normal(0.0, rng.uniform(0.0, 0.02) * step_rms, step.shape).astype(np.float32)

    if rng.random() < 0.7:
        decay = float(rng.uniform(20.0, 250.0))
        freq = float(rng.uniform(1.0 / 60.0, 1.0 / 7.0))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        amp = float(rng.uniform(0.0, 0.12)) * imp_rms
        ring = amp * np.exp(-d / decay) * np.sin(2.0 * np.pi * freq * d + phase)
        imp += ring.astype(np.float32)
        step += (ring * float(rng.uniform(0.2, 0.8))).astype(np.float32)

    if rng.random() < 0.5:
        period = float(rng.uniform(200.0, 1200.0))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        amp = float(rng.uniform(0.0, 0.04)) * step_rms
        step += (amp * np.sin(2.0 * np.pi * d / period + phase)).astype(np.float32)

    return imp, step, gain_tau, local_width


class V3Dataset(Dataset):
    def __init__(self, root: Path = ROOT, split: str = "train", train: bool = False,
                 augment_data: bool = True, epoch: int = 0, distance: np.ndarray | None = None):
        self.root = Path(root)
        self.split = split
        self.train = train
        self.augment_data = augment_data
        self.epoch = epoch
        self.distance = DISTANCE if distance is None else distance
        self.imp_path = self.root / "arrays" / f"{split}_raw_imp.npy"
        self.step_path = self.root / "arrays" / f"{split}_raw_step.npy"
        self.target_path = self.root / "arrays" / f"{split}_targets.npy"
        self.metadata = [json.loads(line) for line in (self.root / f"metadata_{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        self.feature_path = self.root / "arrays" / f"{split}_features.npy"
        self._imp = None
        self._step = None
        self._target = None
        self._features = None
        self._count = len(self.metadata)
        shape = np.load(self.imp_path, mmap_mode="r").shape
        if shape[0] != self._count:
            raise ValueError(f"{split} metadata count {self._count} != raw_imp count {shape[0]}")

    def _open(self):
        if self._imp is None:
            self._imp = np.load(self.imp_path, mmap_mode="r")
            self._step = np.load(self.step_path, mmap_mode="r")
            self._target = np.load(self.target_path, mmap_mode="r")
            if not self.train and self.feature_path.exists():
                self._features = np.load(self.feature_path, mmap_mode="r")

    def __len__(self):
        return self._count

    def __getitem__(self, index: int):
        self._open()
        target = np.asarray(self._target[index], dtype=np.float32).copy()
        if self._features is not None:
            return np.asarray(self._features[index], dtype=np.float32), target, int(index)
        imp = np.asarray(self._imp[index], dtype=np.float32)
        step = np.asarray(self._step[index], dtype=np.float32)
        if self.train and self.augment_data:
            imp, step, gain_tau, local_width = augment(imp, step, self.distance, _rng(index, self.epoch))
        else:
            gain_tau, local_width = 200.0, 50.0
        x = distance_features(imp, step, self.distance, gain_tau, local_width)
        return x, target, int(index)


def worker_init(worker_id: int):
    global _WORKER_RNG
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    _WORKER_RNG = np.random.default_rng(seed + worker_id * 7919)
