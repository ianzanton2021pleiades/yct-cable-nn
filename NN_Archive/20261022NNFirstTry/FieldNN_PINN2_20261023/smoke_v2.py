"""Two checks that must pass before a long run: feature extension fidelity and a
finite forward/backward through the physics module."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
for path in (
    HERE,
    PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_Hybrid_20261022",
    PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_Iter5_20261022",
    PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_20261022",
    PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_PINN_20261023",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data_v2 import PinnV2Dataset  # noqa: E402
from features_v2 import extend_features  # noqa: E402
from losses_v2 import PhysicsResidual, PhysicsWeights, compute_losses, supervision_only  # noqa: E402
from models_pinn import build_model  # noqa: E402

ROOT = Path(r"E:\CableNNDataSet\CodeX\FieldNN_Iter4_20261022")


def check_feature_extension() -> None:
    dataset = PinnV2Dataset(ROOT, "val", train=False, view="clean")
    worst = []
    for index in range(0, len(dataset), max(1, len(dataset) // 40)):
        features, _, _, _ = dataset[index]
        rebuilt = extend_features(features[:8], dataset.base.distance)
        difference = np.abs(rebuilt[8:] - features[8:])
        worst.append(float(difference.max()))
    print(
        f"feature extension: curves={len(worst)} max_abs_error={max(worst):.4f} "
        f"median={float(np.median(worst)):.4f}",
        flush=True,
    )
    if max(worst) > 0.6:
        raise SystemExit("extension error too large; the inversion assumption is broken")
    truth = dataset[0][0]
    print(
        f"channels={truth.shape[0]} length={truth.shape[-1]} finite={np.isfinite(truth).all()}",
        flush=True,
    )


def check_physics_step() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model("unet", input_channels=11, width=32, depth=3, dropout=0.1).to(device)
    physics = PhysicsResidual(weights=PhysicsWeights()).to(device)
    dataset = PinnV2Dataset(ROOT, "val", train=False, view="clean")
    batch = torch.stack([torch.from_numpy(dataset[i][0]) for i in range(4)]).to(device)
    targets = torch.stack([torch.from_numpy(dataset[i][1]) for i in range(4)]).to(device)
    output = model(batch)
    components = compute_losses(output[:, :3], targets)
    physics_value, parts = physics(output.float(), batch.float())
    loss = supervision_only(components) + physics_value
    loss.backward()
    print(
        "forward: loss={:.4f} sup={:.4f} physics={:.4f} parts={}".format(
            float(loss), float(supervision_only(components)), float(physics_value),
            {name: round(float(value), 4) for name, value in parts.items()},
        ),
        flush=True,
    )
    gradient = sum(
        float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None
    )
    print(f"backward: grad_abs_sum={gradient:.4f}", flush=True)
    if not np.isfinite(gradient) or gradient == 0.0:
        raise SystemExit("gradient is not finite")


if __name__ == "__main__":
    check_feature_extension()
    check_physics_step()
    print("smoke ok", flush=True)
