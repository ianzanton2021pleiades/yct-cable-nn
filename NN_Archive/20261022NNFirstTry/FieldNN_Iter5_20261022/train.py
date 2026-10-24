"""Train Iter5 models with width-aware labels and field fixture adaptation."""

from __future__ import annotations

import argparse
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
ITER3 = HERE.parent / "FieldNN_Iter3_20261022"
for path in (HERE, ITER3):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from data import (
    BANK_SCALE_RANGE,
    CLEAN_REPLAY_FRACTION,
    EVENT_LABEL_SIGMA_CAP_M,
    ShardDataset,
    TERMINAL_LABEL_SIGMA_CAP_M,
    WIDTH_COUPLING,
)
from losses import compute_losses, total_loss
from models import build_model


DEFAULT_DATA = Path(r"E:\CableNNDataSet\CodeX\FieldNN_Iter4_20261022")
MODEL_DEFAULTS = {"input_channels": 8, "width": 32, "depth": 3, "state_size": 8}
TRAINING_DEFAULTS = {
    "epochs": 20,
    "batch_size": 32,
    "learning_rate": 1.0e-4,
    "weight_decay": 0.01,
    "num_workers": 7,
    "patience": 6,
    "warmup_epochs": 1,
    "plateau_factor": 0.5,
    "plateau_patience": 2,
    "min_learning_rate": 1.0e-6,
    "amp": True,
    "amp_init_scale": 1024.0,
}
NEAR_END_SUPPRESSION_M = 30.0
LOSS_DESCRIPTION = {
    "event_terminal": "Iter3 FP32 Gaussian focal with ignored target<0 windows",
    "interval": "Iter3 full-grid BCE pos_weight=2 plus 0.5 Dice on positive curves",
    "terminal_truth_mask": False,
    "label_widening": (
        f"augmented samples widen event/terminal targets by "
        f"sqrt(5^2 + ({WIDTH_COUPLING}*input_sigma)^2), caps "
        f"{EVENT_LABEL_SIGMA_CAP_M:.0f} m event / {TERMINAL_LABEL_SIGMA_CAP_M:.0f} m terminal"
    ),
    "clean_replay_fraction": CLEAN_REPLAY_FRACTION,
    "calibration_bank_scale_range": list(BANK_SCALE_RANGE),
    "near_end_suppression_m": NEAR_END_SUPPRESSION_M,
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def worker_init(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


def load_run_config(data_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    cfg = json.loads((data_root / "config.json").read_text(encoding="utf-8"))
    model_cfg = {**MODEL_DEFAULTS, **cfg.get("model", {})}
    if int(model_cfg["input_channels"]) != 8:
        raise ValueError(
            f"Iter5 feature input must have 8 channels; config requests {model_cfg['input_channels']}"
        )
    cfg["model"] = model_cfg

    settings = {**cfg.get("training", {}), **TRAINING_DEFAULTS}
    if args.epochs is not None:
        settings["epochs"] = args.epochs
    if args.batch_size is not None:
        settings["batch_size"] = args.batch_size
    if args.workers is not None:
        settings["num_workers"] = args.workers
    cfg["training"] = settings
    cfg["loss"] = LOSS_DESCRIPTION
    cfg.setdefault("evaluation", {})["near_exclusion_m"] = NEAR_END_SUPPRESSION_M
    cfg["evaluation"]["event_prominence"] = 0.10
    return cfg


def _rng_state(device: torch.device) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def _restore_rng(state: dict[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda" and state.get("cuda") is not None:
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def _batch_pass(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    max_batches: int = 0,
    description: str = "",
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"event": 0.0, "interval": 0.0, "terminal": 0.0}
    seen = 0
    grad_norm_sum = 0.0
    grad_norm_max = 0.0
    grad_norm_count = 0
    skipped_steps = 0
    progress = tqdm(loader, desc=description, leave=False, unit="batch")
    for step, (inputs, targets, _) in enumerate(progress):
        inputs = inputs.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(inputs)
            components = compute_losses(logits, targets)
            loss = total_loss(components)
            if training:
                assert optimizer is not None and scaler is not None
                old_scale = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                grad_norm = float(norm.detach())
                grad_norm_sum += grad_norm
                grad_norm_max = max(grad_norm_max, grad_norm)
                grad_norm_count += 1
                scaler.step(optimizer)
                scaler.update()
                skipped_steps += int(amp and scaler.get_scale() < old_scale)

        count = int(inputs.shape[0])
        for name, component in components.items():
            totals[name] += float(component.detach()) * count
        seen += count
        progress.set_postfix(
            loss=f"{float(loss.detach()):.4f}",
            event=f"{float(components['event'].detach()):.4f}",
            interval=f"{float(components['interval'].detach()):.4f}",
            terminal=f"{float(components['terminal'].detach()):.4f}",
        )
        if max_batches and step + 1 >= max_batches:
            break

    result = {name: value / seen for name, value in totals.items()}
    result["total"] = sum(result.values())
    result["grad_norm_mean"] = grad_norm_sum / grad_norm_count if grad_norm_count else 0.0
    result["grad_norm_max"] = grad_norm_max
    result["amp_skipped_steps"] = float(skipped_steps)
    result["samples"] = float(seen)
    return result


def train_one(name: str, args: argparse.Namespace) -> None:
    root = Path(args.data).expanduser().resolve()
    cfg = load_run_config(root, args)
    settings = cfg["training"]
    seed = int(cfg.get("seed", 20261022))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select --device cpu for a CPU check")
    amp = bool(settings["amp"] and device.type == "cuda")

    train_data = ShardDataset(root, "train", train=True)
    val_data = ShardDataset(root, "val", train=False)
    if not len(train_data) or not len(val_data):
        raise ValueError("Training requires nonempty train and validation splits")

    workers = int(settings["num_workers"])
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_data,
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
        worker_init_fn=worker_init,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=int(settings["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    model_cfg = cfg["model"]
    model = build_model(name, **model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(settings["plateau_factor"]),
        patience=int(settings["plateau_patience"]),
        min_lr=float(settings["min_learning_rate"]),
    )
    scaler = torch.amp.GradScaler(
        "cuda", init_scale=float(settings["amp_init_scale"]), enabled=amp
    )

    run = Path(args.output).expanduser() / name
    run.mkdir(parents=True, exist_ok=True)
    last_path = run / "last.pt"
    start_epoch, best_val_loss, stale_epochs = 0, float("inf"), 0
    if not args.resume and last_path.exists():
        raise FileExistsError(
            f"Run already has a checkpoint; use --resume or choose another output: {last_path}"
        )
    if args.resume:
        if not last_path.is_file():
            raise FileNotFoundError(f"Cannot resume without this run's checkpoint: {last_path}")
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state["model_name"] != name or state["config"]["model"] != model_cfg:
            raise ValueError("Resume checkpoint model/config does not match this run")
        if state["data_root"] != str(root):
            raise ValueError("Resume checkpoint was created from another data root")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
        best_val_loss = float(state["best_val_loss"])
        stale_epochs = int(state["stale_epochs"])
        _restore_rng(state["rng"], device)
    elif args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint["model_name"] != name or checkpoint["config"]["model"] != model_cfg:
            raise ValueError("Init checkpoint model/config does not match this model")
        model.load_state_dict(checkpoint["model"])

    write_json(run / "config.json", cfg)
    write_json(
        run / "run_info.json",
        {
            "model": name,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "data_root": str(root),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch": torch.__version__,
            "training": settings,
            "loss": LOSS_DESCRIPTION,
            "fieldnn_eval_view": val_data.view,
            "init_checkpoint_weights_only": (
                str(Path(args.init_checkpoint).resolve()) if args.init_checkpoint else None
            ),
        },
    )

    history_path = run / "history.csv"
    fields = [
        "epoch", "train_loss", "train_event_loss", "train_interval_loss", "train_terminal_loss",
        "val_loss", "val_event_loss", "val_interval_loss", "val_terminal_loss",
        "grad_norm_mean", "grad_norm_max", "amp_skipped_steps", "lr", "seconds",
        "samples_per_second", "peak_gpu_gib",
    ]
    print(
        f"{name}: params={sum(p.numel() for p in model.parameters()):,} "
        f"train={len(train_data)} val={len(val_data)} batch={settings['batch_size']} "
        f"workers={workers}/0 device={device} amp={amp} val_view={val_data.view}",
        flush=True,
    )

    for epoch in range(start_epoch, int(settings["epochs"])):
        if epoch < int(settings["warmup_epochs"]):
            warmup_lr = float(settings["learning_rate"]) * (epoch + 1) / int(settings["warmup_epochs"])
            for group in optimizer.param_groups:
                group["lr"] = warmup_lr
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        begun = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        train_result = _batch_pass(
            model, train_loader, device, amp, optimizer, scaler,
            max_batches=args.max_train_batches,
            description=f"{name} train {epoch + 1}/{settings['epochs']}",
        )
        val_result = _batch_pass(
            model, val_loader, device, amp,
            max_batches=args.max_val_batches,
            description=f"{name} val {epoch + 1}/{settings['epochs']}",
        )
        improved = val_result["total"] < best_val_loss
        best_val_loss, stale_epochs = (
            (val_result["total"], 0) if improved else (best_val_loss, stale_epochs + 1)
        )
        if epoch >= int(settings["warmup_epochs"]) - 1:
            scheduler.step(val_result["total"])

        elapsed = time.perf_counter() - begun
        row = {
            "epoch": epoch + 1,
            "train_loss": train_result["total"],
            "train_event_loss": train_result["event"],
            "train_interval_loss": train_result["interval"],
            "train_terminal_loss": train_result["terminal"],
            "val_loss": val_result["total"],
            "val_event_loss": val_result["event"],
            "val_interval_loss": val_result["interval"],
            "val_terminal_loss": val_result["terminal"],
            "grad_norm_mean": train_result["grad_norm_mean"],
            "grad_norm_max": train_result["grad_norm_max"],
            "amp_skipped_steps": int(train_result["amp_skipped_steps"]),
            "lr": epoch_lr,
            "seconds": elapsed,
            "samples_per_second": (train_result["samples"] + val_result["samples"]) / elapsed,
            "peak_gpu_gib": (
                torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
            ),
        }
        state = {
            "model_name": name,
            "config": cfg,
            "data_root": str(root),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "stale_epochs": stale_epochs,
            "rng": _rng_state(device),
        }
        torch.save(state, run / "last.tmp")
        (run / "last.tmp").replace(last_path)
        if improved:
            torch.save(state, run / "best.pt")

        exists = history_path.exists()
        with history_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if stale_epochs >= int(settings["patience"]):
            print(
                f"Early stopping after {stale_epochs} epochs without validation improvement",
                flush=True,
            )
            break

    del train_loader, val_loader, train_data, val_data, model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", choices=("unet", "transformer", "all"), default="all")
    parser.add_argument("--output", type=Path, default=HERE / "output" / "iter5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--resume", action="store_true")
    modes.add_argument("--init-checkpoint", type=Path)
    args = parser.parse_args()
    if args.model == "all" and args.init_checkpoint is not None:
        parser.error("--init-checkpoint applies to one model; select --model unet or transformer")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model_names = ("unet", "transformer") if args.model == "all" else (args.model,)
    for model_name in model_names:
        train_one(model_name, args)


if __name__ == "__main__":
    main()
