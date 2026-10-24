"""Synthetic-only pretraining with step/epoch loss and F1 logging.

This is the clean long curve: complete synthetic truth, no field adaptation.
The resulting checkpoint can initialize the measured-field fine-tune.
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
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
LEGACY = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_20261022"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(LEGACY) not in sys.path:
    sys.path.append(str(LEGACY))

from decode import decode_predictions  # noqa: E402
from hybrid_data import SyntheticWrapper  # noqa: E402
from losses import compute_losses, total_loss  # noqa: E402
from metrics import evaluate_predictions  # noqa: E402
from models import build_model  # noqa: E402


DEFAULT_SYNTHETIC = Path(r"E:\CableNNDataSet\CodeX\FieldNN_Iter4_20261022")


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


def append_steps(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = ["step", "epoch", "loss", "event", "interval", "terminal"]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def synthetic_truths(dataset, indices: list[int]) -> list[dict[str, Any]]:
    rows = []
    for index in indices:
        metadata = dataset.metadata[index]
        rows.append({
            "event_positions_m": metadata.get("event_positions_m", []),
            "defect_intervals_m": metadata.get("defect_intervals_m", []),
            "terminal_position_m": metadata.get("terminal_position_m"),
            "sample_kind": metadata.get("sample_kind"),
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
            logits = model(inputs.to(device, non_blocking=device.type == "cuda"))
        probabilities.extend(logits.float().sigmoid().cpu().numpy())
        indices.extend(int(value) for value in item.tolist())
    return probabilities, indices


@torch.inference_mode()
def evaluate_loss(model, loader, device, amp) -> dict[str, float]:
    model.eval()
    totals = {"total": 0.0, "event": 0.0, "interval": 0.0, "terminal": 0.0, "sparsity": 0.0}
    seen = 0
    for batch in loader:
        inputs, targets = batch[0], batch[1]
        inputs = inputs.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(inputs)
        components = compute_losses(logits, targets)
        loss = total_loss(components, 0.0)
        count = int(inputs.shape[0])
        totals["total"] += float(loss) * count
        for name in ("event", "interval", "terminal", "sparsity"):
            totals[name] += float(components[name]) * count
        seen += count
    return {name: value / max(seen, 1) for name, value in totals.items()}


def train_epoch(model, loader, optimizer, scaler, ema, device, amp, epoch):
    model.train()
    totals = {"total": 0.0, "event": 0.0, "interval": 0.0, "terminal": 0.0, "sparsity": 0.0}
    steps = []
    seen = 0
    skipped = 0
    grad_sum = 0.0
    grad_max = 0.0
    grad_count = 0
    progress = tqdm(loader, desc="pretrain", leave=False, unit="batch")
    for step, batch in enumerate(progress):
        inputs, targets = batch[0], batch[1]
        inputs = inputs.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(inputs)
        components = compute_losses(logits, targets)
        loss = total_loss(components, 0.0)
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
        total_value = float(loss.detach())
        totals["total"] += total_value * count
        for name in ("event", "interval", "terminal", "sparsity"):
            totals[name] += float(components[name].detach()) * count
        seen += count
        steps.append({
            "step": epoch * len(loader) + step + 1,
            "epoch": epoch + 1,
            "loss": total_value,
            "event": float(components["event"].detach()),
            "interval": float(components["interval"].detach()),
            "terminal": float(components["terminal"].detach()),
        })
        progress.set_postfix(loss=f"{total_value:.4f}")
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
    parser.add_argument("--output", type=Path, default=HERE / "output" / "pretrain50")
    parser.add_argument("--model", choices=("unet", "transformer"), default="unet")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--restart-scheduler", action="store_true")
    args = parser.parse_args()

    seed = 2026102207
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    amp = device.type == "cuda"

    train_set = SyntheticWrapper(args.synthetic, "train", train=True, view="mixed")
    val_set = SyntheticWrapper(args.synthetic, "val", train=False, view="mixed")
    probe_base = SyntheticWrapper(args.synthetic, "train", train=False, view="clean")
    stride = max(1, len(probe_base) // 300)
    probe = Subset(probe_base, list(range(0, len(probe_base), stride))[:300])
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0, drop_last=True,
        worker_init_fn=worker_init,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(args.model, input_channels=8, width=32, depth=3, dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - 1), eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0, enabled=amp)
    ema = EMA(model, 0.999)
    ema_model = copy.deepcopy(model).to(device)

    output = args.output.expanduser().resolve() / args.model
    output.mkdir(parents=True, exist_ok=True)
    last_path = output / "last.pt"
    best_path = output / "best.pt"
    start_epoch = 0
    best_f1 = -1.0
    best_score = float("inf")
    if args.resume:
        if not last_path.is_file():
            raise FileNotFoundError(f"Cannot resume without {last_path}")
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"]) + 1
        if args.restart_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, args.epochs - start_epoch),
                eta_min=1e-6,
            )
            for group in optimizer.param_groups:
                group["lr"] = args.lr
        else:
            scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        ema.shadow = {key: value.to(device) for key, value in state["ema"].items()}
        best_f1 = float(state["best_f1"])
        best_score = float(state["best_score"])
    else:
        if last_path.exists():
            raise FileExistsError(f"Run exists; use --resume or another output: {last_path}")
        ema = EMA(model, 0.999)

    config = {
        "model": {"input_channels": 8, "width": 32, "depth": 3, "state_size": 8, "dropout": 0.1},
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
        "training": {"epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.lr,
                     "scheduler": "cosine", "amp": amp},
    }
    write_json(output / "config.json", config)
    write_json(output / "run_info.json", {
        "model": args.model,
        "parameters": sum(p.numel() for p in model.parameters()),
        "synthetic": str(args.synthetic.resolve()),
        "training": config["training"],
    })

    history_path = output / "history.csv"
    fields = [
        "epoch", "train_loss", "train_event", "train_interval", "train_terminal",
        "val_loss", "val_event", "val_interval", "val_terminal",
        "train_f1", "val_f1", "val_terminal_f1", "ema_selected",
        "best_f1", "best_score", "lr", "seconds", "samples_per_second", "peak_gpu_gib",
    ]
    print(f"{args.model}: params={sum(p.numel() for p in model.parameters()):,} "
          f"train={len(train_set)} val={len(val_set)} batch={args.batch_size}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        if epoch == 0:
            for group in optimizer.param_groups:
                group["lr"] = args.lr * 0.1
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        begun = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_result = train_epoch(model, train_loader, optimizer, scaler, ema, device, amp, epoch)
        append_steps(output / "steps.csv", train_result["steps"])

        val_loss = evaluate_loss(model, val_loader, device, amp)
        val_probs, val_indices = predict(model, val_set, device, args.batch_size)
        val_truth = synthetic_truths(val_set, val_indices)
        val_report = evaluate_predictions(
            [decode_predictions(p, config, {"event": 0.5, "interval": 0.95, "terminal": 0.5})
             for p in val_probs],
            val_truth, config,
        )
        val_f1 = float(val_report["event_metrics_by_tolerance_m"]["5"].get("f1") or 0.0)
        val_terminal_f1 = float(val_report["terminal_metrics"].get("f1") or 0.0)

        probe_probs, probe_indices = predict(model, probe, device, args.batch_size)
        probe_truth = synthetic_truths(probe_base, probe_indices)
        probe_report = evaluate_predictions(
            [decode_predictions(p, config, {"event": 0.5, "interval": 0.95, "terminal": 0.5})
             for p in probe_probs],
            probe_truth, config,
        )
        train_f1 = float(probe_report["event_metrics_by_tolerance_m"]["5"].get("f1") or 0.0)

        ema.copy_to(ema_model)
        ema_value = evaluate_loss(ema_model, val_loader, device, amp)
        ema_probs, ema_indices = predict(ema_model, val_set, device, args.batch_size)
        ema_report = evaluate_predictions(
            [decode_predictions(p, config, {"event": 0.5, "interval": 0.95, "terminal": 0.5})
             for p in ema_probs],
            synthetic_truths(val_set, ema_indices), config,
        )
        ema_f1 = float(ema_report["event_metrics_by_tolerance_m"]["5"].get("f1") or 0.0)

        raw_score = (1.0 - val_f1) + 0.1 * val_loss["total"]
        ema_score = (1.0 - ema_f1) + 0.1 * ema_value["total"]
        ema_selected = ema_score < raw_score
        selected_f1 = ema_f1 if ema_selected else val_f1
        selected_score = ema_score if ema_selected else raw_score
        if selected_score < best_score:
            best_score = selected_score
            best_f1 = selected_f1
            torch.save({
                "model_name": args.model,
                "config": config,
                "model": (ema_model if ema_selected else model).state_dict(),
                "epoch": epoch,
                "val_f1": selected_f1,
                "score": selected_score,
                "ema": bool(ema_selected),
            }, best_path)
        scheduler.step()

        elapsed = time.perf_counter() - begun
        row = {
            "epoch": epoch + 1,
            "train_loss": train_result["total"],
            "train_event": train_result["event"],
            "train_interval": train_result["interval"],
            "train_terminal": train_result["terminal"],
            "val_loss": val_loss["total"],
            "val_event": val_loss["event"],
            "val_interval": val_loss["interval"],
            "val_terminal": val_loss["terminal"],
            "train_f1": train_f1,
            "val_f1": val_f1,
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
        exists = history_path.exists()
        with history_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
