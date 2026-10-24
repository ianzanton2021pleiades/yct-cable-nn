"""Iter8 synthetic and calibration datasets using the shared 12-channel features."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from torch.utils.data import Dataset


HERE = Path(__file__).resolve().parent
ITER3 = HERE.parent / "FieldNN_Iter3_20261022"
ITER5 = HERE.parent / "FieldNN_Iter5_20261022"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ITER3) not in sys.path:
    sys.path.append(str(ITER3))

from features import distance_features


_iter5_spec = importlib.util.spec_from_file_location(
    "iter8_iter5_data", ITER5 / "data.py"
)
_iter5_data = importlib.util.module_from_spec(_iter5_spec)
sys.modules[_iter5_spec.name] = _iter5_data
_iter5_spec.loader.exec_module(_iter5_data)
_iter5_data.distance_features = distance_features


class SyntheticDataset(_iter5_data.ShardDataset):
    """Iter5's raw-cache augmentation and targets with Iter8's 12-channel features."""


ShardDataset = SyntheticDataset


class CalibrationDataset(Dataset):
    """Partial labels for calibration/no-cable responses; not healthy cable samples."""

    def __init__(self, root: str | Path, split: str) -> None:
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
        self.indices = [
            int(record["index"])
            for record in self.records
            if record["split"] == split
        ]
        response = json.loads((ITER3 / "config.json").read_text(encoding="utf-8"))["response"]
        self.distance = np.linspace(
            0.0, float(response["distance_max_m"]), int(self.imp.shape[-1]), dtype=np.float64
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[np.ndarray, np.ndarray, int]:
        index = self.indices[item]
        impulse = np.asarray(self.imp[index], dtype=np.float32)
        step = np.asarray(self.step[index], dtype=np.float32)
        features = distance_features(impulse, step, self.distance)

        targets = np.zeros((3, self.distance.size), dtype=np.float32)
        near = self.distance < 30.0
        targets[0, near] = -1.0
        targets[1, near] = -1.0
        targets[2, :] = -1.0
        return features, targets, index
