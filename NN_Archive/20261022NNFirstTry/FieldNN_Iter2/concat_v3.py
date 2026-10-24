"""Concatenate v3 shards into per-split memmaps for worker-friendly loading."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TARGET = Path(r"E:\CableNNDataSet\Deepseek_CMDC\FieldNN_Iter2")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=TARGET)
    args = parser.parse_args()
    root = args.root
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    arrays = root / "arrays"
    arrays.mkdir(exist_ok=True)
    for split in ("train", "val", "test"):
        shards = [s for s in manifest["shards"] if s["split"] == split]
        imp, step, targets, metadata = [], [], [], []
        for shard in shards:
            folder = root / shard["path"]
            imp.append(np.load(folder / "raw_imp.npy"))
            step.append(np.load(folder / "raw_step.npy"))
            targets.append(np.load(folder / "targets.npy"))
            metadata.extend(json.loads(line) for line in (folder / "metadata.jsonl").read_text(encoding="utf-8").splitlines())
        imp = np.concatenate(imp).astype(np.float32)
        step = np.concatenate(step).astype(np.float32)
        targets = np.concatenate(targets).astype(np.float32)
        np.save(arrays / f"{split}_raw_imp.npy", imp)
        np.save(arrays / f"{split}_raw_step.npy", step)
        np.save(arrays / f"{split}_targets.npy", targets)
        (root / f"metadata_{split}.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in metadata) + "\n", encoding="utf-8")
        print(split, imp.shape, step.shape, targets.shape, len(metadata), flush=True)


if __name__ == "__main__":
    main()
