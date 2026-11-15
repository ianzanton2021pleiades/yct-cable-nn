from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import FieldDataset, collate_groups
from losses import masked_detection_loss
from model import PhaseDetector


ROOT = Path(__file__).resolve().parent
TASKS = ("joint", "terminal", "anomaly_point", "anomaly_range")
COUNT_FIELDS = ("positive", "negative", "supervised")


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def int_stat(stats: dict, path: tuple[str, ...]) -> int:
    value = stats
    for key in path:
        if not isinstance(value, dict):
            return 0
        value = value.get(key)
    return int(value or 0)


def float_or_none(value) -> float | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().float().item()
    return float(value)


class EpochStats:
    def __init__(self) -> None:
        self.losses: list[float] = []
        self.task_losses: dict[str, list[float]] = {task: [] for task in TASKS}
        self.boundary_losses: list[float] = []
        self.counts: dict[str, dict[str, int]] = {
            task: {name: 0 for name in COUNT_FIELDS} for task in TASKS
        }
        self.total_counts = {name: 0 for name in COUNT_FIELDS}
        self.boundary_count = 0
        self.batches_seen = 0
        self.batches_scored = 0
        self.optimizer_updates = 0

    def add(self, loss, stats: dict, scored: bool, optimizer_stepped: bool) -> None:
        self.batches_seen += 1
        self.batches_scored += int(scored)
        self.optimizer_updates += int(optimizer_stepped)
        if scored:
            number = float_or_none(loss)
            if number is not None:
                self.losses.append(number)

        for task in TASKS:
            task_stats = stats.get(task, {})
            task_loss = float_or_none(task_stats.get("loss")) if isinstance(task_stats, dict) else None
            if scored and task_loss is not None:
                self.task_losses[task].append(task_loss)
            for field in COUNT_FIELDS:
                self.counts[task][field] += int_stat(stats, (task, field))
                self.total_counts[field] += int_stat(stats, ("total", field)) if task == TASKS[0] else 0

        boundary_count = int(stats.get("boundary_count", 0) or 0)
        self.boundary_count += boundary_count
        boundary_loss = float_or_none(stats.get("boundary_loss"))
        if scored and boundary_count > 0 and boundary_loss is not None:
            self.boundary_losses.append(boundary_loss)

    @staticmethod
    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    def summary(self) -> dict:
        return {
            "loss": self.mean(self.losses),
            "task_losses": {task: self.mean(self.task_losses[task]) for task in TASKS},
            "boundary_loss": self.mean(self.boundary_losses),
            "counts": self.counts,
            "total_counts": self.total_counts,
            "boundary_count": self.boundary_count,
            "batches_seen": self.batches_seen,
            "batches_scored": self.batches_scored,
            "optimizer_updates": self.optimizer_updates,
        }


def move_batch(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key in ("x", "valid", "phase_mask", "distance_m", "target", "supervision_mask"):
        value = batch.get(key)
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=device.type == "cuda")
    moved["meta"] = batch.get("meta")
    moved["boundary_constraints"] = batch.get("boundary_constraints")
    return moved


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    mixed_precision: bool,
    desc: str,
) -> dict:
    training = optimizer is not None
    model.train(training)
    result = EpochStats()
    iterator = tqdm(loader, desc=desc, leave=False)

    for raw_batch in iterator:
        batch = move_batch(raw_batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        autocast_enabled = mixed_precision and device.type == "cuda"
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
            if autocast_enabled
            else nullcontext()
        )
        with torch.set_grad_enabled(training):
            with autocast:
                logits = model(batch["x"], batch["valid"], batch["phase_mask"])
                loss, stats = masked_detection_loss(
                    logits.float(),
                    batch["target"],
                    batch["supervision_mask"],
                    boundary_constraints=batch.get("boundary_constraints"),
                )

            supervised = int_stat(stats, ("total", "supervised"))
            boundary_count = int(stats.get("boundary_count", 0) or 0)
            has_objective = supervised > 0 or boundary_count > 0
            optimizer_stepped = False
            if training and has_objective:
                if scaler.is_enabled():
                    scale_before = scaler.get_scale()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer_stepped = scaler.get_scale() >= scale_before
                else:
                    loss.backward()
                    optimizer.step()
                    optimizer_stepped = True

        result.add(loss, stats, has_objective, optimizer_stepped)
        iterator.set_postfix(
            loss=f"{float_or_none(loss):.4g}" if has_objective else "skip",
            supervised=supervised,
            boundary=boundary_count,
        )

    return result.summary()


def history_columns() -> list[str]:
    columns = [
        "epoch",
        "learning_rate",
        "train_loss",
        "val_loss",
        "train_boundary_loss",
        "val_boundary_loss",
        "train_boundary_count",
        "val_boundary_count",
        "train_batches_scored",
        "train_optimizer_updates",
        "actual_optimizer_updates",
        "epochs_completed",
        "val_batches_scored",
    ]
    for task in TASKS:
        columns.extend((f"train_{task}_loss", f"val_{task}_loss"))
        for field in COUNT_FIELDS:
            columns.extend((f"train_{task}_{field}", f"val_{task}_{field}"))
    for field in COUNT_FIELDS:
        columns.extend((f"train_total_{field}", f"val_total_{field}"))
    return columns


def history_row(
    epoch: int,
    learning_rate: float,
    train: dict,
    val: dict,
    actual_optimizer_updates: int,
    epochs_completed: int,
) -> dict:
    row = {
        "epoch": epoch,
        "learning_rate": learning_rate,
        "train_loss": train["loss"],
        "val_loss": val["loss"],
        "train_boundary_loss": train["boundary_loss"],
        "val_boundary_loss": val["boundary_loss"],
        "train_boundary_count": train["boundary_count"],
        "val_boundary_count": val["boundary_count"],
        "train_batches_scored": train["batches_scored"],
        "train_optimizer_updates": train["optimizer_updates"],
        "actual_optimizer_updates": actual_optimizer_updates,
        "epochs_completed": epochs_completed,
        "val_batches_scored": val["batches_scored"],
    }
    for task in TASKS:
        row[f"train_{task}_loss"] = train["task_losses"][task]
        row[f"val_{task}_loss"] = val["task_losses"][task]
        for field in COUNT_FIELDS:
            row[f"train_{task}_{field}"] = train["counts"][task][field]
            row[f"val_{task}_{field}"] = val["counts"][task][field]
    for field in COUNT_FIELDS:
        row[f"train_total_{field}"] = train["total_counts"][field]
        row[f"val_total_{field}"] = val["total_counts"][field]
    return row


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    actual_optimizer_updates: int,
    val_loss: float | None,
    config: dict,
    split: dict,
    arm: str,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "epochs_completed": epoch,
            "actual_optimizer_updates": actual_optimizer_updates,
            "val_loss": val_loss,
            "val_metric": "masked_validation_loss",
            "arm": arm,
            "config": config,
            "split": split,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the field FDR detector; the user runs this command.")
    parser.add_argument("--arm", choices=("A", "B"), required=True)
    parser.add_argument("--data-dir", default="output/field_cache")
    parser.add_argument("--split", default="config/split_template.json")
    parser.add_argument("--config", default="config/first_round.json")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--encoder-init", default=None)
    args = parser.parse_args()

    config_path = project_path(args.config)
    split_path = project_path(args.split)
    data_dir = project_path(args.data_dir)
    run_dir = project_path(args.run_dir)
    config = load_json(config_path)
    split = load_json(split_path)
    train_groups = list(split.get("train", []))
    val_groups = list(split.get("val", []))
    if not train_groups or not val_groups:
        parser.error("正式训练前必须由用户在split文件中显式填写非空train和val case_group_id；模板不会自动划分。")
    overlap = sorted(set(train_groups).intersection(val_groups))
    if overlap:
        parser.error(f"train和val包含相同case_group_id，存在物理案例跨集：{overlap}")
    if args.arm == "B" and not args.encoder_init:
        parser.error("B臂需要--encoder-init指定兼容的共享单相encoder_state_dict；不会自动加载历史模型。")
    if args.arm == "A" and args.encoder_init:
        parser.error("A臂从随机初始化开始；移除--encoder-init，或改用B臂。")
    encoder_path = project_path(args.encoder_init) if args.encoder_init else None
    if encoder_path is not None and not encoder_path.is_file():
        parser.error(f"encoder初始化文件不存在：{encoder_path}")
    if not data_dir.is_dir():
        parser.error(f"数据目录不存在：{data_dir}；先由用户执行run_prepare.ps1。")
    if run_dir.exists() and any(run_dir.iterdir()):
        parser.error(f"输出目录非空：{run_dir}；为保留已有训练记录，请选择新run目录。")
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / "config.json")
    shutil.copy2(split_path, run_dir / "split.json")

    seed_everything(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = config["model"]
    model = PhaseDetector(
        in_channels=int(model_cfg.get("in_channels", 5)),
        width=int(model_cfg.get("width", 16)),
    )
    if args.arm == "B":
        checkpoint = torch.load(encoder_path, map_location="cpu")
        encoder_state = checkpoint["encoder_state_dict"]
        model.load_encoder_state_dict(encoder_state)
    model.to(device)
    run_info = {
        "arm": args.arm,
        "encoder_init": str(encoder_path) if encoder_path else None,
        "device": str(device),
        "seed": int(config["seed"]),
        "train_case_groups": train_groups,
        "val_case_groups": val_groups,
        "best_checkpoint_rule": "lowest masked validation loss",
    }
    with (run_dir / "run_info.json").open("w", encoding="utf-8") as stream:
        json.dump(run_info, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    train_cfg = config["training"]
    train_set = FieldDataset(
        data_dir,
        train_groups,
        training=True,
        single_phase_probability=float(train_cfg["single_phase_probability"]),
    )
    val_set = FieldDataset(data_dir, val_groups, training=False, single_phase_probability=0.0)
    batch_size = int(train_cfg["batch_size"])
    workers = int(train_cfg["num_workers"])
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate_groups,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_groups,
        pin_memory=device.type == "cuda",
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg["learning_rate"]))
    use_amp = bool(train_cfg.get("mixed_precision", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    max_epochs = int(train_cfg["epochs"])
    patience = int(train_cfg["patience"])
    history_path = run_dir / "history.csv"
    best_val = float("inf")
    epochs_without_improvement = 0
    total_optimizer_updates = 0
    epochs_completed = 0
    columns = history_columns()

    with history_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for epoch in range(1, max_epochs + 1):
            train_summary = run_epoch(
                model,
                train_loader,
                device,
                optimizer,
                scaler,
                use_amp,
                f"train {epoch}/{max_epochs}",
            )
            total_optimizer_updates += train_summary["optimizer_updates"]
            val_summary = run_epoch(
                model,
                val_loader,
                device,
                None,
                scaler,
                use_amp,
                f"val {epoch}/{max_epochs}",
            )
            if val_summary["loss"] is None:
                raise RuntimeError("验证集没有像素监督或起点约束，无法按同口径验证loss选模。")
            epochs_completed = epoch
            learning_rate = float(optimizer.param_groups[0]["lr"])
            writer.writerow(
                history_row(
                    epoch,
                    learning_rate,
                    train_summary,
                    val_summary,
                    total_optimizer_updates,
                    epochs_completed,
                )
            )
            stream.flush()

            save_checkpoint(
                run_dir / "last.pt",
                model,
                optimizer,
                epoch,
                total_optimizer_updates,
                val_summary["loss"],
                config,
                split,
                args.arm,
            )
            if val_summary["loss"] < best_val:
                best_val = float(val_summary["loss"])
                epochs_without_improvement = 0
                save_checkpoint(
                    run_dir / "best.pt",
                    model,
                    optimizer,
                    epoch,
                    total_optimizer_updates,
                    best_val,
                    config,
                    split,
                    args.arm,
                )
            else:
                epochs_without_improvement += 1

            train_loss_value = train_summary["loss"]
            train_loss_text = f"{train_loss_value:.6g}" if train_loss_value is not None else "n/a(skip)"
            print(
                f"epoch={epoch}/{max_epochs} arm={args.arm} "
                f"train_loss={train_loss_text} "
                f"val_loss={val_summary['loss']:.6g} "
                f"best_val_loss={best_val:.6g} "
                f"train_batches_scored={train_summary['batches_scored']} "
                f"train_optimizer_updates={train_summary['optimizer_updates']} "
                f"actual_optimizer_updates={total_optimizer_updates} "
                f"train_supervised={train_summary['total_counts']['supervised']} "
                f"train_boundary={train_summary['boundary_count']}"
            )
            if epochs_without_improvement >= patience:
                print(f"early_stop: validation loss did not improve for {patience} epochs")
                break

    run_info["epochs_completed"] = epochs_completed
    run_info["actual_optimizer_updates"] = total_optimizer_updates
    with (run_dir / "run_info.json").open("w", encoding="utf-8") as stream:
        json.dump(run_info, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"training complete: {run_dir}")
    print("best.pt is selected by lowest masked validation loss; it is not an F1-optimized checkpoint.")


if __name__ == "__main__":
    main()
