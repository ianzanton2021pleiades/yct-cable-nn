"""Physics-regularised synthetic pretraining with step-level and epoch-level logging.

Designed as a drop-in comparison against the CMDC 50/100-epoch curves: same
Iter4/Iter5 cache, same U-Net backbone, same validation view, same selection
score.  The differences are the signed-reflectivity head with its forward-model
residuals, the distance-warp / mixup randomisation, and a stronger weight decay.
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
for path in (HERE, CMDC, CODEX):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data_pinn import PinnDataset  # noqa: E402
from decode import decode_predictions  # noqa: E402
from losses_pinn import (  # noqa: E402
    PhysicsResidual,
    PhysicsWeights,
    compute_losses,
    total_loss,
)
from metrics import evaluate_predictions  # noqa: E402
from models_pinn import build_model  # noqa: E402


DEFAULT_SYNTHETIC = Path(r"E:\CableNNDataSet\CodeX\FieldNN_Iter4_20261022")
DECODE_THRESHOLDS = {"event": 0.5, "interval": 0.95, "terminal": 0.5}


def worker_init(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {key: value.detach().clone() for key, value in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for key, value in model.state_dict().items():
            target = self.shadow[key]
            if value.dtype.is_floating_point:
                target.mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                target.copy_(value.detach())

    def copy_to(self, model: torch.nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def append_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    if not rows:
        return
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _metadata_of(dataset):
    if hasattr(dataset, "metadata"):
        return dataset.metadata
    if hasattr(dataset, "dataset"):
        return _metadata_of(dataset.dataset)
    raise AttributeError("dataset does not expose metadata")


def synthetic_truths(dataset, indices: list[int]) -> list[dict[str, Any]]:
    metadata = _metadata_of(dataset)
    rows = []
    for index in indices:
        row = metadata[index]
        rows.append({
            "event_positions_m": row.get("event_positions_m", []),
            "defect_intervals_m": row.get("defect_intervals_m", []),
            "terminal_position_m": row.get("terminal_position_m"),
            "sample_kind": row.get("sample_kind"),
        })
    return rows


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
def evaluate_loss(model, loader, device, amp, physics: PhysicsResidual | None) -> dict[str, float]:
    model.eval()
    names = ["total", "supervision", "event", "interval", "terminal", "sparsity", "physics"]
    totals = {name: 0.0 for name in names}
    seen = 0
    for batch in loader:
        inputs, targets = batch[0], batch[1]
        inputs = inputs.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")
        with torch.autocast(device_type=device.type, enabled=amp):
            output = model(inputs)
        components = compute_losses(output[:, :3], targets)
        supervision = total_loss(components, 0.0)
        physics_value = torch.zeros((), device=device)
        if physics is not None:
            physics_value, _ = physics(output.float(), inputs.float())
        total = supervision + physics_value
        count = int(inputs.shape[0])
        totals["total"] += float(total) * count
        totals["supervision"] += float(supervision) * count
        totals["physics"] += float(physics_value) * count
        for name in ("event", "interval", "terminal", "sparsity"):
            totals[name] += float(components[name]) * count
        seen += count
    return {name: value / max(seen, 1) for name, value in totals.items()}


def train_epoch(model, loader, optimizer, scaler, ema, device, amp, epoch, physics, sparse_weight):
    model.train()
    names = ["total", "supervision", "event", "interval", "terminal", "physics"]
    totals = {name: 0.0 for name in names}
    steps = []
    seen = 0
    skipped = 0
    grad_sum = 0.0
    grad_max = 0.0
    grad_count = 0
    progress = tqdm(loader, desc=f"epoch {epoch + 1}", leave=False, unit="batch")
    for step, batch in enumerate(progress):
        inputs, targets = batch[0], batch[1]
        inputs = inputs.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            output = model(inputs)
        components = compute_losses(output[:, :3], targets)
        supervision = total_loss(components, sparse_weight)
        if physics is not None:
            physics_value, _ = physics(output.float(), inputs.float())
            loss = supervision + physics_value
        else:
            physics_value = torch.zeros((), device=device)
            loss = supervision
        old_scale = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad = float(norm.detach())
        grad_sum += grad
        grad_max = max(grad_max, grad)
        grad_count += 1
        scaler.step(optimizer)
        scaler.update()
        skipped += int(amp and scaler.get_scale() < old_scale)
        ema.update(model)

        count = int(inputs.shape[0])
        totals["total"] += float(loss.detach()) * count
        totals["supervision"] += float(supervision.detach()) * count
        totals["physics"] += float(physics_value) * count
        for name in ("event", "interval", "terminal"):
            totals[name] += float(components[name].detach()) * count
        seen += count
        row = {
            "step": epoch * len(loader) + step + 1,
            "epoch": epoch + 1,
            "loss": float(loss.detach()),
            "supervision": float(supervision.detach()),
            "event": float(components["event"].detach()),
            "interval": float(components["interval"].detach()),
            "terminal": float(components["terminal"].detach()),
            "physics": float(physics_value),
        }
        steps.append(row)
        progress.set_postfix(loss=f"{row['loss']:.4f}", phys=f"{row['physics']:.3f}")
    result = {name: value / max(seen, 1) for name, value in totals.items()}
    result.update({
        "steps": steps,
        "grad_norm_mean": grad_sum / max(grad_count, 1),
        "grad_norm_max": grad_max,
        "amp_skipped": float(skipped),
        "samples": float(seen),
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", type=Path, default=DEFAULT_SYNTHETIC)
    parser.add_argument("--output", type=Path, default=HERE / "output" / "pinn100")
    parser.add_argument("--model", choices=("unet", "transformer"), default="unet")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--physics", type=int, default=1)
    parser.add_argument("--warp", type=float, default=0.012)
    parser.add_argument("--mixup", type=float, default=0.25)
    parser.add_argument("--noise-prob", type=float, default=0.30)
    parser.add_argument("--w-recon", type=float, default=0.10)
    parser.add_argument("--w-step", type=float, default=0.03)
    parser.add_argument("--w-support", type=float, default=0.05)
    parser.add_argument("--w-order", type=float, default=0.02)
    parser.add_argument("--w-sparse", type=float, default=0.002)
    parser.add_argument("--w-consist", type=float, default=0.05)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--epoch-snapshots", type=str, default="50")
    parser.add_argument("--phase2-epoch", type=int, default=50)
    parser.add_argument("--phase2-lr", type=float, default=1e-4)
    parser.add_argument("--init-from", type=Path, default=None)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    seed = 2026102301
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

    train_set = PinnDataset(
        args.synthetic, "train", train=True,
        warp_prob=0.5 if args.warp > 0 else 0.0, warp_max=args.warp,
        mixup_prob=args.mixup, noise_prob=args.noise_prob,
    )
    val_set = PinnDataset(args.synthetic, "val", train=False, view="mixed")
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
            reconstruction=args.w_recon, step=args.w_step, support=args.w_support,
            ordering=args.w_order, sparse=args.w_sparse, consistency=args.w_consist,
        )).to(device)

    model = build_model(
        args.model, input_channels=8, width=32, depth=3, dropout=args.dropout
    ).to(device)
    if args.init_from:
        initial = torch.load(args.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(initial["model"], strict=True)
        print(f"initialised from {args.init_from} epoch={initial.get('epoch')}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.phase2_epoch - 1), eta_min=1e-6
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
        "model": {"input_channels": 8, "width": 32, "depth": 3, "state_size": 8, "dropout": args.dropout},
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
            "weight_decay": args.weight_decay, "scheduler": "cosine", "amp": amp,
            "physics": bool(args.physics), "warp": args.warp, "mixup": args.mixup,
        },
    }
    write_json(output / "config.json", config)
    write_json(output / "run_info.json", {
        "model": args.model,
        "parameters": sum(p.numel() for p in model.parameters()),
        "synthetic": str(args.synthetic.resolve()),
        "physics_weights": physics.weights.as_dict() if physics else None,
        "training": config["training"],
        "tag": args.tag,
    })

    history_path = output / "history.csv"
    fields = [
        "epoch", "train_loss", "train_supervision", "train_physics", "train_event",
        "train_interval", "train_terminal", "val_loss", "val_supervision", "val_physics",
        "val_event", "val_interval", "val_terminal", "val_f1", "val_f1_2m", "val_terminal_f1",
        "ema_selected", "best_f1", "best_score", "lr", "seconds", "samples_per_second",
        "peak_gpu_gib",
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
        if epoch == 0 and not args.no_warmup:
            for group in optimizer.param_groups:
                group["lr"] = args.lr * 0.1
        if epoch == args.phase2_epoch and args.phase2_epoch < args.epochs:
            for group in optimizer.param_groups:
                group["lr"] = args.phase2_lr
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, args.epochs - args.phase2_epoch), eta_min=1e-6
            )
            print(f"phase 2 restart at epoch {epoch + 1}: lr={args.phase2_lr:g}", flush=True)
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        begun = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_result = train_epoch(
            model, train_loader, optimizer, scaler, ema, device, amp, epoch, physics, 0.0
        )
        append_rows(output / "steps.csv", train_result["steps"], [
            "step", "epoch", "loss", "supervision", "event", "interval", "terminal", "physics",
        ])

        val_loss = evaluate_loss(model, val_loader, device, amp, physics)
        val_probs, val_indices = predict(model, val_set, device, args.batch_size)
        val_report = evaluate_predictions(
            [decode_predictions(item, config, DECODE_THRESHOLDS) for item in val_probs],
            synthetic_truths(val_set, val_indices), config,
        )
        val_f1 = float(val_report["event_metrics_by_tolerance_m"]["5"].get("f1") or 0.0)
        val_f1_2 = float(val_report["event_metrics_by_tolerance_m"]["2"].get("f1") or 0.0)
        val_terminal_f1 = float(val_report["terminal_metrics"].get("f1") or 0.0)

        ema.copy_to(ema_model)
        ema_value = evaluate_loss(ema_model, val_loader, device, amp, physics)
        ema_probs, ema_indices = predict(ema_model, val_set, device, args.batch_size)
        ema_report = evaluate_predictions(
            [decode_predictions(item, config, DECODE_THRESHOLDS) for item in ema_probs],
            synthetic_truths(val_set, ema_indices), config,
        )
        ema_f1 = float(ema_report["event_metrics_by_tolerance_m"]["5"].get("f1") or 0.0)

        raw_score = (1.0 - val_f1) + 0.1 * val_loss["supervision"]
        ema_score = (1.0 - ema_f1) + 0.1 * ema_value["supervision"]
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
                "model": selected_model.state_dict(),
                "epoch": epoch,
                "val_f1": selected_f1,
                "score": selected_score,
                "ema": bool(ema_selected),
            }, output / f"epoch{epoch + 1:03d}.pt")
        scheduler.step()

        elapsed = time.perf_counter() - begun
        row = {
            "epoch": epoch + 1,
            "train_loss": train_result["total"],
            "train_supervision": train_result["supervision"],
            "train_physics": train_result["physics"],
            "train_event": train_result["event"],
            "train_interval": train_result["interval"],
            "train_terminal": train_result["terminal"],
            "val_loss": val_loss["total"],
            "val_supervision": val_loss["supervision"],
            "val_physics": val_loss["physics"],
            "val_event": val_loss["event"],
            "val_interval": val_loss["interval"],
            "val_terminal": val_loss["terminal"],
            "val_f1": val_f1,
            "val_f1_2m": val_f1_2,
            "val_terminal_f1": val_terminal_f1,
            "ema_selected": int(ema_selected),
            "best_f1": best_f1,
            "best_score": best_score,
            "lr": epoch_lr,
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
