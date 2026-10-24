"""Short CUDA-only throughput probe for the existing Iter3 smoke shard."""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW

from data import ShardDataset
from losses import compute_losses, total_loss
from models import build_model


HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE / "smoke_data_v2"
OUTPUT_PATH = HERE / "output" / "batch_probe.json"
MODELS = ("unet", "transformer")
BATCH_SIZES = (16, 32, 64)
WARMUP_STEPS = 2
TIMED_STEPS = 3
SCALER_INIT_SCALE = 1024.0


def load_first_batch() -> tuple[torch.Tensor, torch.Tensor]:
    dataset = ShardDataset(DATA_ROOT, "train")
    source_count = min(16, len(dataset))
    rows = [dataset[index] for index in range(source_count)]
    inputs = torch.from_numpy(np.stack([row[0] for row in rows]))
    targets = torch.from_numpy(np.stack([row[1] for row in rows]))
    return inputs, targets


def make_batch(base_inputs: torch.Tensor, base_targets: torch.Tensor, size: int, device: torch.device):
    repeats = math.ceil(size / len(base_inputs))
    inputs = base_inputs.repeat(repeats, 1, 1)[:size].contiguous().to(device)
    targets = base_targets.repeat(repeats, 1, 1)[:size].contiguous().to(device)
    return inputs, targets, repeats


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> bool:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(inputs)
    loss = total_loss(compute_losses(logits, targets))
    previous_scale = scaler.get_scale()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    return scaler.get_scale() < previous_scale


def benchmark_one(model_name: str, batch_size: int, base_inputs: torch.Tensor,
                  base_targets: torch.Tensor, device: torch.device) -> dict:
    torch.manual_seed(20261022)
    inputs, targets, repeats = make_batch(base_inputs, base_targets, batch_size, device)
    model = build_model(
        model_name, input_channels=8, width=32, depth=3, state_size=8
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=3.0e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", init_scale=SCALER_INIT_SCALE, enabled=True)

    model.train()
    warmup_skips = sum(
        int(train_step(model, optimizer, scaler, inputs, targets))
        for _ in range(WARMUP_STEPS)
    )
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    elapsed = []
    timed_skips = 0
    for _ in range(TIMED_STEPS):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        timed_skips += int(train_step(model, optimizer, scaler, inputs, targets))
        torch.cuda.synchronize(device)
        elapsed.append(time.perf_counter() - started)

    total_seconds = sum(elapsed)
    peak_allocated_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
    result = {
        "model": model_name,
        "batch_size": batch_size,
        "source_samples": int(len(base_inputs)),
        "source_repeat_factor": int(repeats),
        "input_shape": list(inputs.shape),
        "target_shape": list(targets.shape),
        "warmup_steps": WARMUP_STEPS,
        "timed_steps": TIMED_STEPS,
        "mean_step_seconds": total_seconds / TIMED_STEPS,
        "samples_per_second": batch_size * TIMED_STEPS / total_seconds,
        "peak_allocated_gib": peak_allocated_gib,
        "amp_init_scale": SCALER_INIT_SCALE,
        "amp_skipped_steps_warmup": warmup_skips,
        "amp_skipped_steps_timed": timed_skips,
        "amp_skipped_steps_total": warmup_skips + timed_skips,
    }
    del inputs, targets, model, optimizer, scaler
    torch.cuda.empty_cache()
    return result


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this probe is CUDA-only")

    device = torch.device("cuda")
    base_inputs, base_targets = load_first_batch()
    results = []
    for model_name in MODELS:
        for batch_size in BATCH_SIZES:
            row = benchmark_one(model_name, batch_size, base_inputs, base_targets, device)
            results.append(row)
            print(
                f"{model_name:11s} batch={batch_size:2d} "
                f"{row['samples_per_second']:.1f} samples/s "
                f"peak={row['peak_allocated_gib']:.2f} GiB "
                f"AMP skips={row['amp_skipped_steps_total']}",
                flush=True,
            )

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "device_only": True,
        "data_root": str(DATA_ROOT),
        "source_split": "train",
        "source_samples_used": int(len(base_inputs)),
        "source_input_shape": list(base_inputs.shape),
        "source_target_shape": list(base_targets.shape),
        "models": list(MODELS),
        "batch_sizes": list(BATCH_SIZES),
        "warmup_steps": WARMUP_STEPS,
        "timed_steps": TIMED_STEPS,
        "amp_dtype": "float16",
        "amp_init_scale": SCALER_INIT_SCALE,
        "optimizer": "AdamW(lr=3e-4, weight_decay=0.01)",
        "gradient_clip_norm": 1.0,
        "torch_threads": 1,
        "memory_metric_note": "peak_allocated_gib is torch.cuda.max_memory_allocated: peak PyTorch tensor allocation for this process, not device-wide usage or nvidia-smi total memory use.",
        "measurement_note": "Inputs and targets are tiled from the first 16 smoke train samples and copied to CUDA before timing. Timing covers forward, loss, backward, clipping, and AdamW update with CUDA synchronization; it excludes DataLoader and host-to-device transfer and is not a real training throughput estimate.",
        "timing_caveat": "Only two warmup steps were used as requested. U-Net batch 16 was the first timed setting, so first-use setup or clock ramp may affect that row; treat its throughput as less stable than the later batch-size comparisons.",
        "results": results,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
