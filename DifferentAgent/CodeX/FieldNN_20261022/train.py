"""Train one or all three baselines; long runs are launched by the user."""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import DEFAULT_DATA, HERE, ShardDataset, load_config, write_json
from models import build_model


def task_loss(logits, targets):
    weights = logits.new_tensor([30., 2., 80.])[None, :, None]
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=weights)
    probability = logits[:, 1].sigmoid()
    truth = targets[:, 1]
    dice = 1 - (2 * (probability * truth).sum(-1) + 1) / (probability.sum(-1) + truth.sum(-1) + 1)
    return bce + .5 * dice.mean()


def checkpoint_load(path, device="cpu"):
    return torch.load(path, map_location=device, weights_only=False)


def model_from_checkpoint(path, device):
    checkpoint = checkpoint_load(path, device)
    model = build_model(checkpoint["model_name"], **checkpoint["config"]["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def train_one(name, args):
    root = Path(args.data)
    cfg = load_config(root / "config.json")
    settings = cfg["training"]
    if args.epochs is not None:
        settings["epochs"] = args.epochs
    if args.batch_size is not None:
        settings["batch_size"] = args.batch_size
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select --device cpu explicitly for a CPU check")
    train_data, val_data = ShardDataset(root, "train"), ShardDataset(root, "val")
    if not len(train_data) or not len(val_data):
        raise ValueError("Training requires nonempty train and validation splits")
    loader_kwargs = {"batch_size": settings["batch_size"], "num_workers": settings["num_workers"], "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_data, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_data, shuffle=False, **loader_kwargs)
    model = build_model(name, **cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5, patience=4)
    amp = bool(settings["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    run = Path(args.output) / name
    run.mkdir(parents=True, exist_ok=True)
    start_epoch, best, stale = 0, float("inf"), 0
    if args.resume and (run / "last.pt").exists():
        state = checkpoint_load(run / "last.pt", device)
        if state["model_name"] != name or state["config"]["model"] != cfg["model"]:
            raise ValueError("Resume model/config does not match this run")
        if state["data_root"] != str(root.resolve()):
            raise ValueError("Resume dataset differs from the checkpoint dataset")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best, stale = state["epoch"] + 1, state["best_val_loss"], state["stale_epochs"]
        torch.set_rng_state(state["torch_rng"].cpu())
        if device.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(state["cuda_rng"].cpu())
    elif (run / "last.pt").exists():
        raise FileExistsError(f"Run already exists: {run}; use --resume or choose a new --output")
    write_json(run / "config.json", cfg)
    write_json(run / "run_info.json", {"model": name, "parameters": sum(p.numel() for p in model.parameters()), "data_root": str(root.resolve()), "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "torch": torch.__version__, "synthetic_supervision": "geometric positions and changed-parameter intervals; not measured field ground truth"})
    log_path = run / "history.csv"
    fields = ["epoch", "train_loss", "val_loss", "lr", "seconds", "samples_per_second", "peak_gpu_gib"]
    print(f"{name}: {sum(p.numel() for p in model.parameters()):,} parameters; train={len(train_data)} val={len(val_data)} device={device}", flush=True)
    for epoch in range(start_epoch, settings["epochs"]):
        begun = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        total, seen = 0., 0
        for step, (x, y, _) in enumerate(tqdm(train_loader, desc=f"{name} {epoch+1}/{settings['epochs']} train")):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = task_loss(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss; inspect the saved configuration and input data")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * len(x)
            seen += len(x)
            if args.max_train_batches and step + 1 >= args.max_train_batches:
                break
        model.eval()
        val_total, val_seen = 0., 0
        with torch.inference_mode():
            for step, (x, y, _) in enumerate(tqdm(val_loader, desc=f"{name} validation", leave=False)):
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device.type, enabled=amp):
                    loss = task_loss(model(x), y)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite validation loss")
                val_total += float(loss) * len(x)
                val_seen += len(x)
                if args.max_val_batches and step + 1 >= args.max_val_batches:
                    break
        validation = val_total / val_seen
        improved = validation < best
        best, stale = (validation, 0) if improved else (best, stale + 1)
        elapsed = time.perf_counter() - begun
        row = {"epoch": epoch + 1, "train_loss": total / seen, "val_loss": validation, "lr": optimizer.param_groups[0]["lr"], "seconds": elapsed, "samples_per_second": (seen + val_seen) / elapsed, "peak_gpu_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0}
        scheduler.step(validation)
        state = {"model_name": name, "config": cfg, "data_root": str(root.resolve()), "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch, "best_val_loss": best, "stale_epochs": stale, "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}
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
            print(f"Early stopping: no validation improvement for {stale} epochs", flush=True)
            break
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", choices=["mamba", "transformer", "unet", "all"], default="all")
    parser.add_argument("--output", type=Path, default=HERE / "output" / "pilot_v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0, help="Only for a small pipeline check; leave 0 for real training")
    parser.add_argument("--max-val-batches", type=int, default=0)
    args = parser.parse_args()
    for name in (["mamba", "transformer", "unet"] if args.model == "all" else [args.model]):
        train_one(name, args)


if __name__ == "__main__":
    main()
