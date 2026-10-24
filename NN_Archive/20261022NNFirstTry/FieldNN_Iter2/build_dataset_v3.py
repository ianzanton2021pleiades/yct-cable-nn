"""Derive a compact v3 dataset (raw impulse/step + targets) from the frozen 10k set."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

SOURCE = Path(r"E:\CableNNDataSet\CodeX\FieldNN_20261022")
TARGET = Path(r"E:\CableNNDataSet\Deepseek_CMDC\FieldNN_Iter2")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--target", type=Path, default=TARGET)
    args = parser.parse_args()
    source, target = args.source, args.target
    target.mkdir(parents=True, exist_ok=True)
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    config["model"] = {"input_channels": 8, "width": 48, "depth": 3, "state_size": 8}
    config["training"] = {
        "epochs": 60, "batch_size": 32, "learning_rate": 0.001, "weight_decay": 0.01,
        "patience": 20, "num_workers": 7, "amp": True,
    }
    config["status"] = "v3 TDR features; raw impulse/step stored, features computed with augmentation"
    (target / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(source / "manifest.json", target / "manifest.json")
    shutil.copy2(source / "frequency.npy", target / "frequency.npy")
    shutil.copy2(source / "distance.npy", target / "distance.npy")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    for number, shard in enumerate(manifest["shards"]):
        in_dir = source / shard["path"]
        out_dir = target / shard["path"]
        out_dir.mkdir(parents=True, exist_ok=True)
        if not (out_dir / "raw_imp.npy").exists() or not (out_dir / "raw_step.npy").exists():
            x = np.load(in_dir / "inputs.npy", mmap_mode="r")
            if x.shape[1] != 4:
                raise ValueError(f"expected 4 input channels, got {x.shape}")
            imp = np.asarray(x[:, 0] * (10.0 ** x[:, 2]), dtype=np.float32)
            step = np.asarray(x[:, 1] * (10.0 ** x[:, 3]), dtype=np.float32)
            np.save(out_dir / "raw_imp.npy", imp)
            np.save(out_dir / "raw_step.npy", step)
        for name in ("targets.npy", "metadata.jsonl"):
            dst = out_dir / name
            if not dst.exists():
                shutil.copy2(in_dir / name, dst)
        if number % 10 == 0 or number + 1 == len(manifest["shards"]):
            print(f"{number + 1}/{len(manifest['shards'])} {shard['path']}", flush=True)
    (target / "dataset_info.json").write_text(json.dumps({
        "source": str(source), "schema": "fieldnn-v3-raw", "samples": 10000,
        "note": "raw_imp/raw_step recovered exactly from the frozen v1 4-channel inputs; no S11 regeneration",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("done:", target)


if __name__ == "__main__":
    main()
