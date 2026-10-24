"""Precompute deterministic v3 features for validation/test splits."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from data_v3 import ROOT, V3Dataset


def main():
    for split in ("val", "test"):
        dataset = V3Dataset(ROOT, split, train=False)
        features = np.empty((len(dataset), 8, 8192), dtype=np.float32)
        for index in range(len(dataset)):
            features[index] = dataset[index][0]
            if index % 200 == 0:
                print(split, index, flush=True)
        np.save(Path(ROOT) / "arrays" / f"{split}_features.npy", features)
        print(split, features.shape, flush=True)


if __name__ == "__main__":
    main()
