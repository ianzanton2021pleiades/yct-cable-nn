"""Forward latency and parameter count for a checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
CMDC = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_Hybrid_20261022"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(CMDC) not in sys.path:
    sys.path.insert(0, str(CMDC))

from models_pinn import build_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    model = build_model(state["model_name"], input_channels=8, width=32, depth=3,
                        dropout=config["model"].get("dropout", 0.15)).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    parameters = sum(p.numel() for p in model.parameters())
    result = {
        "checkpoint": str(args.checkpoint),
        "model": state["model_name"],
        "parameters": parameters,
        "size_mb_fp32": parameters * 4 / 2**20,
    }
    for batch in (1, 16, 64):
        inputs = torch.randn(batch, 8, 8192, device=device)
        with torch.inference_mode():
            for _ in range(10):
                model(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            begun = time.perf_counter()
            repeats = 30 if batch == 1 else 20
            for _ in range(repeats):
                model(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - begun) / repeats
        result[f"forward_ms_batch{batch}"] = elapsed * 1000.0
        result[f"per_curve_ms_batch{batch}"] = elapsed * 1000.0 / batch
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
