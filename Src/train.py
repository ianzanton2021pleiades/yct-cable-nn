"""
train.py — 电缆缺陷检测训练主循环

用法:
    python Src/train.py --config Src/configs/default.yaml
    python Src/train.py --config Src/configs/default.yaml --model.base_ch 16
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# 路径
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR
sys.path.insert(0, str(SRC_DIR))

from config import load_config, config_to_dict, Config
from models.unet1d import UNet1D
from models.cableformer import CableFormer
from models.unet1d_v2 import UNet1DV2
from losses import CombinedLoss, V2CombinedLoss, V3CombinedLoss
from metrics import compute_sample_metrics, aggregate_metrics
from core.dataset import CableDefectDataset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(cfg: Config) -> torch.device:
    if cfg.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    print("⚠️  CUDA 不可用，回退到 CPU")
    return torch.device("cpu")


def build_model(cfg: Config) -> nn.Module:
    if cfg.model.name == "cableformer":
        return CableFormer(
            base_ch=cfg.model.base_ch,
            in_channels=cfg.model.in_channels,
            out_channels=cfg.model.out_channels,
            transformer_dim=cfg.model.transformer_dim,
            n_transformer_blocks=cfg.model.n_transformer_blocks,
            n_heads=cfg.model.n_heads,
            use_cbam=cfg.model.use_cbam,
        )
    elif cfg.model.name == "unet1d_v2":
        return UNet1DV2(
            base_ch=cfg.model.base_ch,
            in_channels=cfg.model.in_channels,
            out_channels=cfg.model.out_channels,
            use_cbam=cfg.model.use_cbam,
        )
    return UNet1D(
        base_ch=cfg.model.base_ch,
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        kernel_size=cfg.model.kernel_size,
    )


def build_optimizer(model: nn.Module, cfg: Config):
    if cfg.train.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )
    elif cfg.train.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )
    raise ValueError(f"Unknown optimizer: {cfg.train.optimizer}")


def build_scheduler(optimizer, cfg: Config):
    if cfg.train.scheduler == "cosine_warm":
        # Linear warmup + cosine annealing
        warmup = cfg.train.warmup_epochs
        total = cfg.train.epochs

        def lr_lambda(epoch):
            if epoch < warmup:
                return (epoch + 1) / warmup
            progress = (epoch - warmup) / max(total - warmup, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif cfg.train.scheduler == "one_cycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg.train.lr,
            epochs=cfg.train.epochs, steps_per_epoch=100
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train.epochs
    )


def build_loss(cfg: Config):
    if cfg.loss.version == "v3":
        return V3CombinedLoss(
            focal_gamma=cfg.loss.focal_gamma,
            alpha=cfg.loss.alpha,
            beta=cfg.loss.peak_weight,
            smooth_weight=cfg.loss.smooth_weight,
        )
    if cfg.loss.version == "v2":
        return V2CombinedLoss(
            focal_gamma=cfg.loss.focal_gamma,
            dice_weight=cfg.loss.dice_weight,
            focal_weight=cfg.loss.focal_weight,
            smooth_weight=cfg.loss.smooth_weight,
        )
    return CombinedLoss(
        focal_gamma=cfg.loss.focal_gamma,
        alpha=cfg.loss.alpha,
        beta=cfg.loss.peak_weight,
    )


def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    cfg: Config,
    device: torch.device,
) -> dict:
    """评估一个 epoch，返回指标 dict"""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_metrics = []
    grid = np.linspace(0, 1200, 2400, endpoint=False) + 0.25

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits, probs = model(inputs)
            loss = loss_fn(logits, probs, labels)
            total_loss += loss.item()
            n_batches += 1

            # 逐样本指标
            for i in range(probs.shape[0]):
                pred_np = probs[i].cpu().numpy()
                label_np = labels[i].cpu().numpy()
                m = compute_sample_metrics(
                    pred_np, label_np, grid,
                    peak_height=cfg.eval.find_peaks_height,
                    peak_distance=cfg.eval.find_peaks_distance,
                    peak_prominence=cfg.eval.find_peaks_prominence,
                    tolerance_m=cfg.eval.peak_tolerance_m,
                )
                all_metrics.append(m)

    agg = aggregate_metrics(all_metrics)
    return {
        "loss": total_loss / max(n_batches, 1),
        "recall": agg.recall,
        "precision": agg.precision,
        "far": agg.far,
        "loc_error_mean": agg.loc_error_mean,
    }


def save_checkpoint(model: nn.Module, path: Path, cfg: Config):
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config_to_dict(cfg),
    }, path)


def main():
    # 解析参数
    parser_args = sys.argv[1:]
    config_path = None
    cli_overrides = []

    i = 0
    while i < len(parser_args):
        if parser_args[i] == "--config" and i + 1 < len(parser_args):
            config_path = parser_args[i + 1]
            i += 2
        else:
            cli_overrides.append(parser_args[i])
            i += 1

    cfg = load_config(config_path, cli_overrides)
    set_seed(cfg.experiment.seed)
    device = get_device(cfg)

    # 项目根目录 (AI_TEST/)，用于解析相对路径
    PROJECT_ROOT = SCRIPT_DIR.parent
    # 解析相对路径
    if not Path(cfg.data.manifest).is_absolute():
        cfg.data.manifest = str(PROJECT_ROOT / cfg.data.manifest)
    if not Path(cfg.experiment.exp_dir).is_absolute():
        cfg.experiment.exp_dir = str(PROJECT_ROOT / cfg.experiment.exp_dir)

    # 实验目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(cfg.experiment.exp_dir) / f"exp_{timestamp}_{cfg.experiment.name}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置
    import yaml
    with open(exp_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config_to_dict(cfg), f, allow_unicode=True, default_flow_style=False)

    print(f"实验目录: {exp_dir}")
    print(f"设备: {device}")

    # 构建组件
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {n_params:,}")

    loss_fn = build_loss(cfg)

    # 数据集
    train_ds = CableDefectDataset(
        manifest_path=cfg.data.manifest,
        split="train",
        channels=cfg.data.channels,
        augment=cfg.data.augment,
    )
    val_ds = CableDefectDataset(
        manifest_path=cfg.data.manifest,
        split="val",
        channels=cfg.data.channels,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size,
        shuffle=True, num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size,
        shuffle=False, num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # AMP
    use_amp = cfg.train.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # 训练日志
    log_path = exp_dir / "train_log.csv"
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "train_loss", "val_loss", "val_recall",
                         "val_precision", "val_far", "val_loc_error", "lr"])

    best_recall = 0.0
    no_improve = 0
    t_start = time.time()

    for epoch in range(1, cfg.train.epochs + 1):
        # ─── Train ───
        model.train()
        train_loss = 0.0
        n_train = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.train.epochs}",
                     leave=False)
        for inputs, labels in pbar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, probs = model(inputs)
                loss = loss_fn(logits, probs, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            n_train += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= max(n_train, 1)
        scheduler.step()

        # ─── Val ───
        val_m = evaluate_epoch(model, val_loader, loss_fn, cfg, device)
        lr = optimizer.param_groups[0]["lr"]

        log_writer.writerow([
            epoch, f"{train_loss:.6f}", f"{val_m['loss']:.6f}",
            f"{val_m['recall']:.4f}", f"{val_m['precision']:.4f}",
            f"{val_m['far']:.2f}", f"{val_m['loc_error_mean']:.4f}",
            f"{lr:.6f}",
        ])
        log_file.flush()

        elapsed = time.time() - t_start
        print(f"  Epoch {epoch:3d} | "
              f"train_loss={train_loss:.4f} | "
              f"val_loss={val_m['loss']:.4f} | "
              f"recall={val_m['recall']:.4f} | "
              f"prec={val_m['precision']:.4f} | "
              f"FAR={val_m['far']:.2f} | "
              f"loc_err={val_m['loc_error_mean']:.2f}m | "
              f"lr={lr:.6f} | {elapsed:.0f}s")

        # ─── Save best ───
        if val_m["recall"] > best_recall:
            best_recall = val_m["recall"]
            save_checkpoint(model, exp_dir / "best.pt", cfg)
            no_improve = 0
            print(f"  ★ 新最优 recall={best_recall:.4f}")
        else:
            no_improve += 1

        save_checkpoint(model, exp_dir / "last.pt", cfg)

        # 早停
        if no_improve >= cfg.train.early_stop_patience:
            print(f"\n早停: val_recall 连续 {no_improve} epoch 无提升")
            break

    log_file.close()

    # 最终指标
    print(f"\n训练完成! best_recall={best_recall:.4f}, 耗时={time.time()-t_start:.0f}s")

    final_metrics = {
        "best_recall": best_recall,
        "epochs_run": epoch,
        "time_seconds": time.time() - t_start,
        "n_params": n_params,
    }
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    print(f"实验输出: {exp_dir}")


if __name__ == "__main__":
    main()
