"""Shape and speed sanity check for the PINN pipeline (not a test suite)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data_pinn import PinnDataset  # noqa: E402
from losses_pinn import PhysicsResidual, PhysicsWeights, compute_losses, total_loss  # noqa: E402
from models_pinn import build_model  # noqa: E402


def main() -> int:
    root = Path(r"E:\CableNNDataSet\CodeX\FieldNN_Iter4_20261022")
    train = PinnDataset(root, "train", train=True, warp_prob=0.5, mixup_prob=0.25)
    val = PinnDataset(root, "val", train=False, view="mixed")
    print("lengths", len(train), len(val))

    begun = time.perf_counter()
    features, targets, item, kind = train[0]
    print("sample", features.shape, features.dtype, targets.shape, "item", item, "kind", kind,
          "seconds", round(time.perf_counter() - begun, 3))
    print("feature ranges", [(float(features[c].min()), float(features[c].max())) for c in range(8)])
    print("target ranges", [(float(targets[c].min()), float(targets[c].max())) for c in range(5)])

    begun = time.perf_counter()
    for index in range(20):
        train[index]
    print("load 20 samples", round(time.perf_counter() - begun, 3), "s")

    device = torch.device("cuda")
    model = build_model("unet", input_channels=8, width=32, depth=3, dropout=0.15).to(device)
    physics = PhysicsResidual(weights=PhysicsWeights()).to(device)
    print("parameters", f"{sum(p.numel() for p in model.parameters()):,}")

    batch_x = torch.stack([torch.from_numpy(train[i][0]) for i in range(8)]).to(device)
    batch_y = torch.stack([torch.from_numpy(train[i][1]) for i in range(8)]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(6):
        optimizer.zero_grad(set_to_none=True)
        output = model(batch_x)
        components = compute_losses(output[:, :3], batch_y)
        supervision = total_loss(components, 0.0)
        physics_value, detail = physics(output.float(), batch_x.float())
        loss = supervision + physics_value
        loss.backward()
        optimizer.step()
        print(
            f"step {step} loss {float(loss):.4f} sup {float(supervision):.4f} "
            f"event {float(components['event']):.4f} interval {float(components['interval']):.4f} "
            f"term {float(components['terminal']):.4f} phys {float(physics_value):.4f} "
            + " ".join(f"{name}={float(value):.4f}" for name, value in detail.items())
        )
    print("peak_gpu_gib", round(torch.cuda.max_memory_allocated(device) / 2**30, 2))

    model.eval()
    for name in ("unet", "transformer"):
        probe = build_model(name, input_channels=8, width=32, depth=3, dropout=0.15).to(device)
        probe.eval()
        with torch.no_grad():
            for _ in range(2):
                probe(batch_x)
            torch.cuda.synchronize()
            begun = time.perf_counter()
            for _ in range(5):
                probe(batch_x)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - begun) / 5.0
        print(name, "batch8 forward", round(elapsed * 1000, 1), "ms",
              f"-> {8 / elapsed:.0f} samples/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
