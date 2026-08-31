"""
test_train_pipeline.py - smoke test for training pipeline
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Src"))

import numpy as np
import torch
from models.unet1d import UNet1D, count_parameters
from losses import CombinedLoss
from metrics import compute_sample_metrics


def test_pipeline():
    print("=" * 50)
    print("Training pipeline smoke test")
    print("=" * 50)

    # 1. Build models (3 widths)
    for name, C in [("narrow", 16), ("medium", 32), ("wide", 64)]:
        model = UNet1D(base_ch=C)
        n = count_parameters(model)
        print(f"  {name} (C={C}): {n:,} params")

    model = UNet1D(base_ch=16)

    # 2. Fake data
    inputs = torch.randn(4, 2, 2400)
    labels = torch.zeros(4, 2400)
    labels[:, 100] = 1.0
    labels[:, 500] = 0.6
    labels[:, 2000] = 1.0

    # 3. Forward
    logits, probs = model(inputs)
    assert probs.shape == (4, 2400), f"Expected (4,2400), got {probs.shape}"
    assert probs.min() >= 0 and probs.max() <= 1
    print(f"  forward: output={tuple(probs.shape)}, range=[{probs.min():.4f}, {probs.max():.4f}] OK")

    # 4. Loss
    loss_fn = CombinedLoss(focal_gamma=2.0, alpha=1.0, beta=0.3)
    loss = loss_fn(logits, probs, labels)
    assert loss.item() > 0
    assert torch.isfinite(loss)
    print(f"  loss: {loss.item():.4f} OK")

    # 5. Backward
    loss.backward()
    has_grad = all(p.grad is not None for p in model.parameters())
    assert has_grad
    print(f"  backward: all grads OK")

    # 6. Optimizer step
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.step()
    print(f"  optimizer step OK")

    # 7. Metrics test
    m = compute_sample_metrics(
        probs[0].detach().numpy(), labels[0].numpy(),
        np.linspace(0, 1200, 2400, endpoint=False) + 0.25
    )
    print(f"  metrics: recall={m.recall:.2f}, precision={m.precision:.2f}, "
          f"label_peaks={m.n_label_peaks}, pred_peaks={m.n_pred_peaks}")

    print("\n=== All smoke tests PASSED ===")


if __name__ == "__main__":
    test_pipeline()
