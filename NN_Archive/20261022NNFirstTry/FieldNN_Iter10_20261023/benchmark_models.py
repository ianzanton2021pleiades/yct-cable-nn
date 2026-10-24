"""Measure batch-1 neural-network forward latency from preloaded Iter8 field features."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
ITER3 = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_Iter3_20261022"
ITER8 = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_Iter8_20261022"
DEFAULT_DATA = Path(r"E:\CableNNDataSet\CodeX\FieldNN_Iter8_20261022")


def load_models(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load model factory: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint_spec(value: str) -> tuple[str, Path]:
    label, path = value.split("=", 1)
    return label, Path(path).expanduser().resolve()


def build_checkpoint_model(torch, checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model_config = config["model"]
    channels = int(model_config["input_channels"])
    name = str(checkpoint["model_name"]).lower()
    width = int(model_config.get("width", 32))
    depth = int(model_config.get("depth", 3))
    state_size = int(model_config.get("state_size", 8))
    if channels == 8:
        factory = load_models(ITER3 / "models.py", f"benchmark_iter3_{name}")
    elif channels == 12:
        factory = load_models(ITER8 / "models.py", f"benchmark_iter8_{name}")
    else:
        raise ValueError(f"No frozen Iter3/Iter8 model factory for {channels} input channels")
    model = factory.build_model(
        name,
        input_channels=channels,
        width=width,
        depth=depth,
        state_size=state_size,
    )
    model.load_state_dict(checkpoint["model"])
    weight_tensor_bytes = sum(
        int(value.numel()) * int(value.element_size())
        for value in model.state_dict().values()
    )
    parameters = sum(int(parameter.numel()) for parameter in model.parameters())
    del checkpoint
    return model, name, channels, parameters, weight_tensor_bytes


def synchronize(torch, device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", "--output-json", "--outputJSON", dest="output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    import torch

    device = torch.device(args.device)
    features = np.load(args.data / "field_features.npy", mmap_mode="r")
    results = []
    for specification in args.checkpoint:
        label, checkpoint_path = checkpoint_spec(specification)
        model, model_name, channels, parameters, weight_tensor_bytes = build_checkpoint_model(torch, checkpoint_path)
        model = model.to(device).eval()
        x = torch.from_numpy(np.asarray(features[0, :channels], dtype=np.float32)).unsqueeze(0).to(device)
        amp = device.type == "cuda"

        with torch.inference_mode():
            for _ in range(20):
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    model(x)
            synchronize(torch, device)
            elapsed_ms = []
            for _ in range(100):
                synchronize(torch, device)
                start = time.perf_counter()
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    model(x)
                synchronize(torch, device)
                elapsed_ms.append((time.perf_counter() - start) * 1000.0)

        results.append(
            {
                "label": label,
                "checkpoint": str(checkpoint_path),
                "model_name": model_name,
                "input_shape": [1, channels, int(features.shape[-1])],
                "parameters": parameters,
                "weight_tensor_bytes": weight_tensor_bytes,
                "inference_package_bytes": None,
                "checkpoint_bytes": checkpoint_path.stat().st_size,
                "batch_size": 1,
                "amp": "float16" if amp else "disabled",
                "warmup_iterations": 20,
                "measured_iterations": 100,
                "median_ms": statistics.median(elapsed_ms),
                "p95_ms": float(np.percentile(elapsed_ms, 95)),
                "raw_ms": elapsed_ms,
                "scope": "NN forward only; input preloaded and resident on device before timing; excludes H2D, FDR read, FFT/Algorithm 1, preprocessing and postprocessing.",
            }
        )
        del model, x

    hardware = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        hardware["gpu_total_memory_bytes"] = int(properties.total_memory)
        hardware["gpu_compute_capability"] = f"{properties.major}.{properties.minor}"
    payload = {
        "schema": "fieldnn-nn-benchmark-1",
        "hardware": hardware,
        "data": str(args.data.resolve()),
        "models": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "hardware": hardware,
                      "results": [{key: row[key] for key in ("label", "median_ms", "p95_ms", "parameters", "weight_tensor_bytes", "checkpoint_bytes")} for row in results]}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
