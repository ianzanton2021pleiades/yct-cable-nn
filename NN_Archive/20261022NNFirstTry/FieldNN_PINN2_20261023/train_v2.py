"""PINN-v2 synthetic pretraining: one continuous 100-epoch cosine, no restarts.

The PINN-v1 curve was assembled from three separate runs whose learning-rate
restarts produced the visible jumps at epochs 51 and 93, and the second restart
re-energised fitting just as the validation loss had flattened.  v2 keeps a
single schedule: three warmup epochs, then cosine from 3e-4 to 1e-6 over the
remaining 97.  Validation loss is measured every epoch on the fixed mixed view;
event F1 is measured every fifth epoch because decoding 1200 curves is the
expensive part.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
CMDC = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_Hybrid_20261022"
CODEX = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_20261022"
PINN1 = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_PINN_20261023"
for path in (HERE, CMDC, CODEX, PINN1):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data_v2 import PinnV2Dataset  # noqa: E402
from decode import decode_predictions  # noqa: E402
from losses_v2 import (  # noqa: E402
    PhysicsResidual,
    PhysicsWeights,
    compute_losses,
    supervision_only,
    total_loss,
)
from metrics import evaluate_predictions  # noqa: E402
from models_pinn import build_model  # noqa: E402
from train_pinn import EMA, append_rows, synthetic_truths, write_json  # noqa: E402

DEFAULT_SYNTHETIC = Path(r"E:\CableNNDataSet\CodeX\FieldNN_Iter4_20261022")
FEATURE_CHANNELS = 11
DECODE_THRESHOLDS = {"event": 0.5, "interval": 0.95, "terminal": 0.5}


def worker_init(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


@torch.inference_mode()
def predict(model, dataset, device, batch_size=64):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    probabilities, indices = [], []
    for batch in loader:
        inputs, item = batch[0], batch[2]
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            output = model(inputs.to(device, non_blocking=device.type == "cuda"))
        probabilities.extend(output[:, :3].float().sigmoid().cpu().numpy())
        indices.extend(int(value) for value in item.tolist())
    return probabilities, indices


@torch.inference_mode()
def evaluate_loss(model, loader, device, amp, physics, bg_weight):
    model.eval()
    totals: dict[str, float] = {}
    seen = 0
    for batch in loader:
        inputs, targets = batch[0], batch[1]
        inputs = inputs.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")
        with torch.autocast(device_type=device.type, enabled=amp):
            output = model(inputs)
        components = compute_losses(output[:, :3], targets)
        supervision = supervision_only(components)
        physics_value = torch.zeros((), device=device)
        if physics is not None:
            physics_value, _ = physics(output.float(), inputs.float())
        values = {
            "supervision": supervision,
            "event": components["event"],
            "interval": components["interval"],
            "terminal": components["terminal"],
            "background": components["background"],
            "physics": physics_value,
            "total": supervision + bg_weight * components["background"] + physics_value,
        }
        count = int(inputs.shape[0])
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + float(value) * count
        seen += count
    return {name: value / max(seen, 1) for name, value in totals.items()}


def train_epoch(model, loader, optimizer, scaler, ema, device, amp, epoch, physics, bg_weight):
    model.train()
    totals: dict[str, float] = {}
    steps = []
    seen = 0
    skipped = 0
    grad_max = 0.0
    physics_parts: dict[str, float] = {}
    progress = tqdm(loader, desc=f"epoch {epoch + 1}", leave=False, unit="batch")
    for step, batch in enumerate(progress):
        inputs, targets = batch[0], batch[1]
        inputs = inputs.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            output = model(inputs)
        components = compute_losses(output[:, :3], targets)
        supervision = supervision_only(components)
        objective = total_loss(components, bg_weight)
        physics_value = torch.zeros((), device=device)
        parts: dict[str, torch.Tensor] = {}
        if physics is not None:
            physics_value, parts = physics(output.float(), inputs.float())
        loss = objective + physics_value

        old_scale = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad_max = max(grad_max, float(norm.detach()))
        scaler.step(optimizer)
        scaler.update()
        skipped += int(amp and scaler.get_scale() < old_scale)
        ema.update(model)

        count = int(inputs.shape[0])
        values = {
            "total": loss,
            "objective": objective,
            "supervision": supervision,
            "event": components["event"],
            "interval": components["interval"],
            "terminal": components["terminal"],
            "background": components["background"],
            "physics": physics_value,
        }
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + float(value) * count
        for name, value in parts.items():
            physics_parts[name] = physics_parts.get(name, 0.0) + float(value) * count
        seen += count
        row = {
            "step": epoch * len(loader) + step + 1,
            "epoch": epoch + 1,
            "loss": float(loss.detach()),
            "supervision": float(supervision.detach()),
            "event": float(components["event"].detach()),
            "background": float(components["background"].detach()),
            "physics": float(physics_value),
            "reconstruction": float(parts.get("reconstruction", torch.zeros(()))),
            "mutual": float(parts.get("mutual", torch.zeros(()))),
        }
        steps.append(row)
        progress.set_postfix(loss=f"{row['loss']:.4f}", phys=f"{row['physics']:.3f}")
    result = {name: value / max(seen, 1) for name, value in totals.items()}
    result.update({
        "steps": steps,
        "physics_parts": {name: value / max(seen, 1) for name, value in physics_parts.items()},
        "grad_norm_max": grad_max,
        "amp_skipped": float(skipped),
        "samples": float(seen),
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", type=Path, default=DEFAULT_SYNTHETIC)
    parser.add_argument("--output", type=Path, default=HERE / "output" / "v2")
    parser.add_argument("--model", choices=("unet", "transformer"), default="unet")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--physics", type=int, default=1)
    parser.add_argument("--warp", type=float, default=0.02)
    parser.add_argument("--mixup", type=float, default=0.30)
    parser.add_argument("--noise-prob", type=float, default=0.35)
    parser.add_argument("--w-bg", type=float, default=0.10)
    parser.add_argument("--w-recon", type=float, default=0.10)
    parser.add_argument("--w-step", type=float, default=0.03)
    parser.add_argument("--w-envelope", type=float, default=0.04)
    parser.add_argument("--w-mutual", type=float, default=0.05)
    parser.add_argument("--w-align", type=float, default=0.02)
    parser.add_argument("--w-order", type=float, default=0.02)
    parser.add_argument("--w-sparse", type=float, default=0.002)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--eval-f1-every", type=int, default=5)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--epoch-snapshots", type=str, default="25,50,75,100")
    parser.add_argument("--init-from", type=Path, default=None)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    seed = 2026102303
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = torch.device(args.device)
    amp = device.type == "cuda"

    train_set = PinnV2Dataset(
        args.synthetic, "train", train=True,
        warp_prob=0.5 if args.warp > 0 else 0.0, warp_max=args.warp,
        mixup_prob=args.mixup, noise_prob=args.noise_prob,
    )
    val_set = PinnV2Dataset(args.synthetic, "val", train=False, view="mixed")
    if args.limit_train:
        from torch.utils.data import Subset
        step = max(1, len(train_set) // args.limit_train)
        train_set = Subset(train_set, list(range(0, len(train_set), step))[: args.limit_train])
    if args.limit_val:
        from torch.utils.data import Subset
        step = max(1, len(val_set) // args.limit_val)
        val_set = Subset(val_set, list(range(0, len(val_set), step))[: args.limit_val])

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=True, drop_last=True,
        persistent_workers=args.workers > 0, worker_init_fn=worker_init,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    physics = None
    if args.physics:
        physics = PhysicsResidual(weights=PhysicsWeights(
            reconstruction=args.w_recon, step=args.w_step, envelope=args.w_envelope,
            mutual=args.w_mutual, align=args.w_align, ordering=args.w_order,
            sparse=args.w_sparse,
        )).to(device)

    model = build_model(
        args.model, input_channels=FEATURE_CHANNELS, width=32, depth=3, dropout=args.dropout
    ).to(device)
    if args.init_from:
        initial = torch.load(args.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(initial["model"], strict=True)
        print(f"initialised from {args.init_from} epoch={initial.get('epoch')}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - args.warmup), eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0, enabled=amp)
    ema = EMA(model, 0.999)
    ema_model = copy.deepcopy(model).to(device)

    output = args.output.expanduser().resolve() / args.model
    output.mkdir(parents=True, exist_ok=True)
    last_path = output / "last.pt"
    best_path = output / "best.pt"
    if last_path.exists():
        raise FileExistsError(f"Run exists; choose another output: {last_path}")

    config = {
        "model": {
            "input_channels": FEATURE_CHANNELS, "width": 32, "depth": 3,
            "state_size": 8, "dropout": args.dropout,
        },
        "response": {"distance_max_m": 3250.0, "points": 8192, "velocity_m_s": 172000000.0},
        "evaluation": {
            "event_tolerances_m": [2.0, 5.0, 10.0],
            "event_min_separation_m": 3.0,
            "interval_min_width_m": 3.0,
            "interval_merge_gap_m": 2.0,
            "terminal_tolerance_m": 10.0,
            "near_exclusion_m": 30.0,
            "event_prominence": 0.10,
        },
        "training": {
            "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.lr,
            "weight_decay": args.weight_decay, "scheduler": "cosine", "warmup": args.warmup,
            "amp": amp, "physics": bool(args.physics), "warp": args.warp,
            "mixup": args.mixup, "background_weight": args.w_bg,
        },
    }
    write_json(output / "config.json", config)
    write_json(output / "run_info.json", {
        "model": args.model,
        "parameters": sum(p.numel() for p in model.parameters()),
        "synthetic": str(args.synthetic.resolve()),
        "features": {"channels": FEATURE_CHANNELS, "sharp_window_m": 8.0, "broad_window_m": 100.0},
        "physics_weights": physics.weights.as_dict() if physics else None,
        "training": config["training"],
        "tag": args.tag,
    })

    history_path = output / "history.csv"
    fields = [
        "epoch", "lr", "train_total", "train_supervision", "train_event", "train_background",
        "train_physics", "val_total", "val_supervision", "val_event", "val_background",
        "val_physics", "val_f1", "val_f1_2m", "val_terminal_f1", "ema_val_supervision",
        "best_f1", "best_score", "seconds", "samples_per_second", "peak_gpu_gib",
    ]
    snapshots = {int(item) for item in args.epoch_snapshots.split(",") if item.strip()}
    print(
        f"{args.model}: params={sum(p.numel() for p in model.parameters()):,} "
        f"train={len(train_set)} val={len(val_set)} batch={args.batch_size} "
        f"physics={bool(args.physics)} warp={args.warp} mixup={args.mixup}",
        flush=True,
    )

    best_score = float("inf")
    best_f1 = -1.0
    for epoch in range(args.epochs):
        if epoch < args.warmup:
            scale = 0.1 + 0.9 * epoch / max(args.warmup, 1)
            for group in optimizer.param_groups:
                group["lr"] = args.lr * scale
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        begun = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_result = train_epoch(
            model, train_loader, optimizer, scaler, ema, device, amp, epoch, physics, args.w_bg
        )
        append_rows(output / "steps.csv", train_result["steps"], [
            "step", "epoch", "loss", "supervision", "event", "background", "physics",
            "reconstruction", "mutual",
        ])

        val = evaluate_loss(model, val_loader, device, amp, physics, args.w_bg)
        ema.copy_to(ema_model)
        ema_val = evaluate_loss(ema_model, val_loader, device, amp, physics, args.w_bg)

        val_f1 = val_f1_2 = val_terminal_f1 = None
        if (epoch + 1) % args.eval_f1_every == 0 or epoch == args.epochs - 1:
            val_probs, val_indices = predict(model, val_set, device, args.batch_size)
            report = evaluate_predictions(
                [decode_predictions(item, config, DECODE_THRESHOLDS) for item in val_probs],
                synthetic_truths(val_set, val_indices), config,
            )
            val_f1 = float(report["event_metrics_by_tolerance_m"]["5"].get("f1") or 0.0)
            val_f1_2 = float(report["event_metrics_by_tolerance_m"]["2"].get("f1") or 0.0)
            val_terminal_f1 = float(report["terminal_metrics"].get("f1") or 0.0)
            ema_probs, ema_indices = predict(ema_model, val_set, device, args.batch_size)
            ema_report = evaluate_predictions(
                [decode_predictions(item, config, DECODE_THRESHOLDS) for item in ema_probs],
                synthetic_truths(val_set, ema_indices), config,
            )
            ema_f1 = float(ema_report["event_metrics_by_tolerance_m"]["5"].get("f1") or 0.0)
            raw_score = (1.0 - val_f1) + 0.1 * val["supervision"]
            ema_score = (1.0 - ema_f1) + 0.1 * ema_val["supervision"]
            ema_selected = ema_score < raw_score
            selected_f1 = ema_f1 if ema_selected else val_f1
            selected_score = ema_score if ema_selected else raw_score
            selected_model = ema_model if ema_selected else model
            if selected_score < best_score:
                best_score = selected_score
                best_f1 = selected_f1
                torch.save({
                    "model_name": args.model,
                    "config": config,
                    "model": selected_model.state_dict(),
                    "epoch": epoch,
                    "val_f1": selected_f1,
                    "score": selected_score,
                    "ema": bool(ema_selected),
                }, best_path)
            if (epoch + 1) in snapshots:
                torch.save({
                    "model_name": args.model,
                    "config": config,
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_f1": val_f1,
                    "score": raw_score,
                    "ema": False,
                }, output / f"epoch{epoch + 1:03d}.pt")
        scheduler.step()

        elapsed = time.perf_counter() - begun
        row = {
            "epoch": epoch + 1,
            "lr": epoch_lr,
            "train_total": train_result["total"],
            "train_supervision": train_result["supervision"],
            "train_event": train_result["event"],
            "train_background": train_result["background"],
            "train_physics": train_result["physics"],
            "val_total": val["total"],
            "val_supervision": val["supervision"],
            "val_event": val["event"],
            "val_background": val["background"],
            "val_physics": val["physics"],
            "val_f1": val_f1,
            "val_f1_2m": val_f1_2,
            "val_terminal_f1": val_terminal_f1,
            "ema_val_supervision": ema_val["supervision"],
            "best_f1": best_f1,
            "best_score": best_score,
            "seconds": elapsed,
            "samples_per_second": train_result["samples"] / max(elapsed, 1e-9),
            "peak_gpu_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0,
        }
        state = {
            "model_name": args.model,
            "config": config,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "ema": ema.shadow,
            "epoch": epoch,
            "best_f1": best_f1,
            "best_score": best_score,
        }
        torch.save(state, output / "last.tmp")
        (output / "last.tmp").replace(last_path)
        append_rows(history_path, [row], fields)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
