"""Train v3 Transformer/U-Net with TDR features, 7-worker augmentation and cosine LR."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_v3 import ROOT, V3Dataset, worker_init
from models import build_model

HERE = Path(__file__).resolve().parent


def task_loss(logits, targets):
    targets = targets.clone()
    targets[:, 0] = torch.where(targets[:, 0] >= 0.135, targets[:, 0], 0.0)
    end_index = targets[:, 2].argmax(dim=-1)
    positions = torch.arange(targets.shape[-1], device=targets.device)[None, :]
    mask = (positions <= end_index[:, None] + 25).to(targets.dtype)
    weights = logits.new_tensor([10.0, 4.0, 60.0])[None, :, None]
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=weights, reduction="none")
    bce = (bce * mask.unsqueeze(1)).sum(dim=(1, 2)) / (mask.sum(dim=1) * 3)
    probability = logits.float().sigmoid() * mask.unsqueeze(1)
    truth = targets * mask.unsqueeze(1)
    dice = 1 - (2 * (probability * truth).sum(-1) + 1) / (probability.sum(-1) + truth.sum(-1) + 1)
    dice_weights = logits.new_tensor([0.5, 1.0, 0.5])
    return bce.mean() + (dice * dice_weights).mean()


def checkpoint_load(path, device="cpu"):
    return torch.load(path, map_location=device, weights_only=False)


def model_from_checkpoint(path, device):
    checkpoint = checkpoint_load(path, device)
    model = build_model(checkpoint["model_name"], **checkpoint["config"]["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def lr_factor(epoch: int, epochs: int, warmup: int = 3) -> float:
    if epoch < warmup:
        return (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, epochs - warmup)
    return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))


def train_one(name: str, args):
    root = Path(args.data)
    cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
    settings = dict(cfg["training"])
    if args.epochs is not None:
        settings["epochs"] = args.epochs
    if args.batch_size is not None:
        settings["batch_size"] = args.batch_size
    if args.lr is not None:
        settings["learning_rate"] = args.lr
    if args.workers is not None:
        settings["num_workers"] = args.workers
    if args.patience is not None:
        settings["patience"] = args.patience
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    train_data = V3Dataset(root, "train", train=True, augment_data=not args.no_augment)
    val_data = V3Dataset(root, "val", train=False)
    workers = int(settings["num_workers"])
    loader_kwargs = {"batch_size": settings["batch_size"], "num_workers": workers,
                     "pin_memory": device.type == "cuda", "worker_init_fn": worker_init}
    if workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 4})
    train_loader = DataLoader(train_data, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_data, shuffle=False, **loader_kwargs)
    model = build_model(name, **cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    amp = bool(settings.get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    run = Path(args.output) / name
    run.mkdir(parents=True, exist_ok=True)
    start_epoch, best, stale = 0, float("inf"), 0
    if args.resume and (run / "last.pt").exists():
        state = checkpoint_load(run / "last.pt", device)
        if state["model_name"] != name or state["config"]["model"] != cfg["model"]:
            raise ValueError("Resume model/config does not match")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, best, stale = state["epoch"] + 1, state["best_val_loss"], state["stale_epochs"]
        torch.set_rng_state(state["torch_rng"].cpu())
        if device.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(state["cuda_rng"].cpu())
    elif (run / "last.pt").exists():
        raise FileExistsError(f"Run exists: {run}; use --resume or a new output")
    total_parameters = sum(p.numel() for p in model.parameters())
    (run / "run_info.json").write_text(json.dumps({
        "model": name, "parameters": total_parameters, "data_root": str(root.resolve()),
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__, "training": settings, "augment": not args.no_augment,
        "features": "v3 8-channel TDR gain/detrend/local/log/distance",
        "loss": {"event_tail_cut": 0.135, "pos_weight": [10.0, 4.0, 60.0], "dice_weight": [0.5, 1.0, 0.5]},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log_path = run / "history.csv"
    fields = ["epoch", "train_loss", "val_loss", "lr", "seconds", "samples_per_second", "peak_gpu_gib"]
    print(f"{name}: {total_parameters:,} params; train={len(train_data)} val={len(val_data)} "
          f"batch={settings['batch_size']} workers={workers} augment={not args.no_augment} device={device}", flush=True)
    for epoch in range(start_epoch, settings["epochs"]):
        for group in optimizer.param_groups:
            group["lr"] = settings["learning_rate"] * lr_factor(epoch, settings["epochs"])
        begun = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        total, seen = 0.0, 0
        for step, (x, y, _) in enumerate(tqdm(train_loader, desc=f"{name} {epoch + 1}/{settings['epochs']} train")):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = task_loss(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * len(x)
            seen += len(x)
            if args.max_train_batches and step + 1 >= args.max_train_batches:
                break
        model.eval()
        val_total, val_seen = 0.0, 0
        with torch.inference_mode():
            for step, (x, y, _) in enumerate(tqdm(val_loader, desc=f"{name} val", leave=False)):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=amp):
                    loss = task_loss(model(x), y)
                val_total += float(loss) * len(x)
                val_seen += len(x)
                if args.max_val_batches and step + 1 >= args.max_val_batches:
                    break
        validation = val_total / max(1, val_seen)
        improved = validation < best
        best, stale = (validation, 0) if improved else (best, stale + 1)
        elapsed = time.perf_counter() - begun
        row = {"epoch": epoch + 1, "train_loss": total / max(1, seen), "val_loss": validation,
               "lr": optimizer.param_groups[0]["lr"], "seconds": elapsed,
               "samples_per_second": (seen + val_seen) / elapsed,
               "peak_gpu_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0}
        state = {"model_name": name, "config": cfg, "data_root": str(root.resolve()),
                 "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "epoch": epoch, "best_val_loss": best, "stale_epochs": stale,
                 "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}
        torch.save(state, run / "last.tmp")
        (run / "last.tmp").replace(run / "last.pt")
        if improved:
            torch.save(state, run / "best.pt")
        exists = log_path.exists()
        with log_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if stale >= settings["patience"]:
            print(f"Early stopping: {stale} stale epochs", flush=True)
            break
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=HERE / "output")
    parser.add_argument("--model", choices=["transformer", "unet", "all"], default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    args = parser.parse_args()
    for name in (["transformer", "unet"] if args.model == "all" else [args.model]):
        train_one(name, args)


if __name__ == "__main__":
    main()
