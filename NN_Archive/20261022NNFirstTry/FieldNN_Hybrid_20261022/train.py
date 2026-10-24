"""Train U-Net/Transformer on synthetic replay + injected measured-field hybrids."""

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
from torch.utils.data import ConcatDataset, DataLoader, Sampler, Subset
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
ITER3 = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_Iter3_20261022"
LEGACY = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_20261022"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(LEGACY) not in sys.path:
    sys.path.append(str(LEGACY))

from decode import decode_predictions  # noqa: E402
from eval_utils import calibration_metrics, field_metrics, hybrid_metrics  # noqa: E402
from losses import compute_losses, total_loss  # noqa: E402
from models import build_model  # noqa: E402
from hybrid_data import (  # noqa: E402
    CalibrationDataset,
    FieldDataset,
    FieldPositiveDataset,
    HybridDataset,
    SyntheticWrapper,
)
from metrics import evaluate_predictions  # noqa: E402


DEFAULT_DATA = Path(r"E:\CableNNDataSet\DeepSeek_CMDC\FieldNN_Hybrid_20261022")
DEFAULT_SYNTHETIC = Path(r"E:\CableNNDataSet\CodeX\FieldNN_Iter4_20261022")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def append_steps(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = ["step", "epoch", "domain", "loss", "event", "interval", "terminal"]
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
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


def worker_init(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


class DomainBatchSampler(Sampler[list[int]]):
    """Yield single-domain batches so each batch loss is attributable to one stream."""

    def __init__(
        self,
        lengths: list[int],
        fractions: list[float],
        batch_size: int,
        num_batches: int,
        seed: int,
    ) -> None:
        self.lengths = [int(value) for value in lengths]
        probabilities = np.asarray(fractions, dtype=np.float64)
        self.probabilities = probabilities / probabilities.sum()
        self.batch_size = int(batch_size)
        self.num_batches = int(num_batches)
        self.seed = int(seed)
        self.offsets = np.cumsum([0] + self.lengths[:-1]).tolist()

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        for _ in range(self.num_batches):
            domain = int(rng.choice(len(self.lengths), p=self.probabilities))
            local = rng.integers(0, self.lengths[domain], size=self.batch_size)
            yield (self.offsets[domain] + local).tolist()


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }

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


def build_datasets(args: argparse.Namespace, cfg: dict[str, Any]):
    synthetic = Path(args.synthetic)
    data = Path(args.data)
    workers = int(args.workers)
    tcfg = cfg["training"]

    syn_train = SyntheticWrapper(synthetic, "train", train=True, view="mixed")
    hyb_train = HybridDataset(
        data,
        "train",
        augment=True,
        mixup_prob=float(cfg["hybrid"].get("mixup_prob", 0.0)),
        distance_max_m=float(cfg["response"]["distance_max_m"]),
    )
    field_train = FieldPositiveDataset(data, "train", cfg, augment=True)
    cal_train = CalibrationDataset(data, "train", cfg)
    if not len(syn_train) or not len(hyb_train) or not len(cal_train):
        raise ValueError(
            f"Empty training set: synthetic={len(syn_train)} hybrid={len(hyb_train)} "
            f"field={len(field_train)} calibration={len(cal_train)}"
        )

    n_syn, n_hyb, n_field, n_cal = (
        len(syn_train), len(hyb_train), len(field_train), len(cal_train)
    )
    w_syn = float(tcfg["synthetic_fraction"])
    w_hyb = float(tcfg["hybrid_fraction"])
    w_field = float(tcfg.get("field_fraction", 0.0))
    w_cal = float(tcfg["calibration_fraction"])
    datasets = [syn_train, hyb_train]
    lengths = [n_syn, n_hyb]
    fractions = [w_syn, w_hyb]
    if n_field:
        datasets.append(field_train)
        lengths.append(n_field)
        fractions.append(w_field)
    datasets.append(cal_train)
    lengths.append(n_cal)
    fractions.append(w_cal)
    combined = ConcatDataset(datasets)
    batch_size = int(tcfg["batch_size"])
    sampler = DomainBatchSampler(
        lengths,
        fractions,
        batch_size,
        max(1, n_syn // batch_size),
        int(cfg.get("seed", 2026102207)),
    )
    train_loader = DataLoader(
        combined,
        batch_sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        worker_init_fn=worker_init,
    )

    syn_val = SyntheticWrapper(synthetic, "val", train=False, view="mixed")
    hyb_val = HybridDataset(
        data,
        "val",
        augment=False,
        distance_max_m=float(cfg["response"]["distance_max_m"]),
    )
    cal_val = CalibrationDataset(data, "val", cfg)
    field_val_loss = FieldPositiveDataset(data, "val", cfg, augment=False)
    field_eval_dataset = FieldDataset(data, "val")
    train_probe_base = SyntheticWrapper(synthetic, "train", train=False, view="clean")
    probe_stride = max(1, len(train_probe_base) // 300)
    train_probe = Subset(
        train_probe_base,
        list(range(0, len(train_probe_base), probe_stride))[:300],
    )
    val_loaders = {
        "synthetic": DataLoader(syn_val, batch_size=batch_size, shuffle=False, num_workers=0),
        "hybrid": DataLoader(hyb_val, batch_size=batch_size, shuffle=False, num_workers=0),
        "calibration": DataLoader(cal_val, batch_size=batch_size, shuffle=False, num_workers=0),
        "field": DataLoader(field_val_loss, batch_size=batch_size, shuffle=False, num_workers=0),
    }
    counts = {
        "synthetic": len(syn_train),
        "hybrid": len(hyb_train),
        "field": len(field_train),
        "calibration": len(cal_train),
        "train_probe": len(train_probe),
    }
    return train_loader, val_loaders, field_eval_dataset, train_probe, counts


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


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    ema: EMA,
    device: torch.device,
    amp: bool,
    sparsity_weight: float,
    epoch: int,
    step_offset: int,
    max_batches: int,
) -> dict[str, Any]:
    model.train()
    totals = {"total": 0.0, "event": 0.0, "interval": 0.0, "terminal": 0.0, "sparsity": 0.0}
    domain_totals: dict[int, dict[str, float]] = {}
    domain_counts: dict[int, int] = {}
    step_rows: list[dict[str, Any]] = []
    seen = 0
    skipped = 0
    grad_norm_sum = 0.0
    grad_norm_max = 0.0
    grad_steps = 0
    progress = tqdm(loader, desc="train", leave=False, unit="batch")
    for step, batch in enumerate(progress):
        inputs, targets = batch[0], batch[1]
        domain_id = int(batch[3][0])
        inputs = inputs.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(inputs)
        components = compute_losses(logits, targets)
        loss = total_loss(components, sparsity_weight)
        old_scale = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad_norm = float(norm.detach())
        grad_norm_sum += grad_norm
        grad_norm_max = max(grad_norm_max, grad_norm)
        grad_steps += 1
        scaler.step(optimizer)
        scaler.update()
        skipped += int(amp and scaler.get_scale() < old_scale)
        ema.update(model)

        count = int(inputs.shape[0])
        total_value = float(loss.detach())
        totals["total"] += total_value * count
        entry = domain_totals.setdefault(
            domain_id,
            {"total": 0.0, "event": 0.0, "interval": 0.0, "terminal": 0.0, "sparsity": 0.0},
        )
        entry["total"] += total_value * count
        for name in ("event", "interval", "terminal", "sparsity"):
            value = float(components[name].detach())
            totals[name] += value * count
            entry[name] += value * count
        domain_counts[domain_id] = domain_counts.get(domain_id, 0) + count
        seen += count
        step_rows.append({
            "step": step_offset + step + 1,
            "epoch": epoch + 1,
            "domain": domain_id,
            "loss": total_value,
            "event": float(components["event"].detach()),
            "interval": float(components["interval"].detach()),
            "terminal": float(components["terminal"].detach()),
        })
        progress.set_postfix(loss=f"{total_value:.4f}", event=f"{float(components['event']):.4f}")
        if max_batches and step + 1 >= max_batches:
            break

    result = {name: value / max(seen, 1) for name, value in totals.items()}
    result["domains"] = {
        domain: {
            name: value / max(domain_counts.get(domain, 0), 1)
            for name, value in entry.items()
        }
        for domain, entry in domain_totals.items()
    }
    result["steps"] = step_rows
    result["grad_norm_mean"] = grad_norm_sum / max(grad_steps, 1)
    result["grad_norm_max"] = grad_norm_max
    result["amp_skipped_steps"] = float(skipped)
    result["samples"] = float(seen)
    return result


@torch.inference_mode()
def evaluate_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    sparsity_weight: float,
    max_batches: int = 0,
) -> dict[str, float]:
    model.eval()
    totals = {"total": 0.0, "event": 0.0, "interval": 0.0, "terminal": 0.0, "sparsity": 0.0}
    seen = 0
    for step, batch in enumerate(loader):
        inputs, targets = batch[0], batch[1]
        inputs = inputs.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(inputs)
        components = compute_losses(logits, targets)
        loss = total_loss(components, sparsity_weight)
        count = int(inputs.shape[0])
        totals["total"] += float(loss) * count
        for name in ("event", "interval", "terminal", "sparsity"):
            totals[name] += float(components[name]) * count
        seen += count
        if max_batches and step + 1 >= max_batches:
            break
    return {name: value / max(seen, 1) for name, value in totals.items()}


@torch.inference_mode()
def predict(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    amp: bool,
    batch_size: int,
) -> tuple[list[np.ndarray], list[int]]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    probabilities: list[np.ndarray] = []
    indices: list[int] = []
    for batch in loader:
        if len(batch) == 2:
            inputs, item = batch
        elif len(batch) == 3:
            inputs, _, item = batch
        else:
            inputs, _, item, _ = batch
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(inputs.to(device, non_blocking=device.type == "cuda"))
        probabilities.extend(logits.float().sigmoid().cpu().numpy())
        indices.extend(int(value) for value in item.tolist())
    return probabilities, indices


def validate(
    model: torch.nn.Module,
    val_loaders: dict[str, DataLoader],
    syn_val_dataset: SyntheticWrapper,
    hyb_val_dataset: HybridDataset,
    cal_val_dataset: CalibrationDataset,
    field_val_dataset: FieldDataset,
    train_probe: Subset,
    device: torch.device,
    amp: bool,
    sparsity_weight: float,
    batch_size: int,
    max_val_batches: int,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    synthetic = evaluate_loss(model, val_loaders["synthetic"], device, amp, sparsity_weight, max_val_batches)
    hybrid = evaluate_loss(model, val_loaders["hybrid"], device, amp, sparsity_weight, max_val_batches)
    calibration = evaluate_loss(model, val_loaders["calibration"], device, amp, sparsity_weight, max_val_batches)
    field_loss = evaluate_loss(model, val_loaders["field"], device, amp, sparsity_weight, max_val_batches)

    probabilities, indices = predict(model, hyb_val_dataset, device, amp, batch_size)
    records = [hyb_val_dataset.records[index] for index in indices]
    hyb = hybrid_metrics(probabilities, records, cfg, {"event": 0.5, "interval": 0.99, "terminal": 0.4})
    cal_probs, _ = predict(model, cal_val_dataset, device, amp, batch_size)
    cal = calibration_metrics(cal_probs, cfg, {"event": 0.5, "interval": 0.99, "terminal": 0.4})
    field_probs, field_indices = predict(model, field_val_dataset, device, amp, batch_size)
    field_records = [field_val_dataset.records[index] for index in field_indices]
    field = field_metrics(field_probs, field_records, cfg, {"event": 0.5, "interval": 0.99, "terminal": 0.4})

    decode_thresholds = {"event": 0.5, "interval": 0.95, "terminal": 0.5}
    syn_probs, syn_indices = predict(model, syn_val_dataset, device, amp, batch_size)
    syn_truth = synthetic_truths(syn_val_dataset, syn_indices)
    syn_report = evaluate_predictions(
        [decode_predictions(probability, cfg, decode_thresholds) for probability in syn_probs],
        syn_truth,
        cfg,
    )
    syn_event = syn_report["event_metrics_by_tolerance_m"]["5"]
    syn_f1 = float(syn_event.get("f1") or 0.0)
    syn_terminal_f1 = float(syn_report["terminal_metrics"].get("f1") or 0.0)

    probe_probs, probe_indices = predict(model, train_probe, device, amp, batch_size)
    probe_truth = synthetic_truths(train_probe.dataset, probe_indices)
    probe_report = evaluate_predictions(
        [decode_predictions(probability, cfg, decode_thresholds) for probability in probe_probs],
        probe_truth,
        cfg,
    )
    train_syn_f1 = float(probe_report["event_metrics_by_tolerance_m"]["5"].get("f1") or 0.0)

    hit = float(hyb.get("hit_rate_20") or 0.0)
    candidates = float(hyb.get("candidates_per_curve") or 0.0)
    field_hit = float(field.get("hit_rate_20") or 0.0)
    field_candidates = float(field.get("candidates_per_curve") or 0.0)
    score = (
        (1.0 - syn_f1)
        + 0.5 * float(synthetic["total"])
        + 0.5 * float(hybrid["total"])
        + 0.1 * float(calibration["total"])
        + 0.05 * float(field_loss["total"])
        + 0.5 * (1.0 - hit)
        + 0.5 * (1.0 - field_hit)
        + 0.05 * candidates
        + 0.05 * field_candidates
    )
    return {
        "synthetic": synthetic,
        "hybrid": hybrid,
        "calibration": calibration,
        "field_loss": field_loss,
        "hybrid_metrics": hyb,
        "calibration_metrics": cal,
        "field_metrics": field,
        "syn_f1": syn_f1,
        "syn_terminal_f1": syn_terminal_f1,
        "train_syn_f1": train_syn_f1,
        "score": float(score),
    }


def train_one(name: str, args: argparse.Namespace) -> None:
    config_path = Path(args.config) if args.config else HERE / "config.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    tcfg = cfg["training"]
    if args.epochs is not None:
        tcfg["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        tcfg["batch_size"] = int(args.batch_size)
    if args.lr is not None:
        tcfg["learning_rate"] = float(args.lr)
    if args.workers is not None:
        tcfg["num_workers"] = int(args.workers)

    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu")
    amp = bool(tcfg["amp"] and device.type == "cuda")
    sparsity_weight = float(cfg["hybrid"]["sparsity_weight"])

    train_loader, val_loaders, field_val_dataset, train_probe, counts = build_datasets(args, cfg)
    syn_val_dataset = val_loaders["synthetic"].dataset
    hyb_val_dataset = val_loaders["hybrid"].dataset
    cal_val_dataset = val_loaders["calibration"].dataset
    data_cfg = {
        "input_channels": int(cfg["model"]["input_channels"]),
        "width": int(cfg["model"]["width"]),
        "depth": int(cfg["model"]["depth"]),
        "state_size": int(cfg["model"]["state_size"]),
        "dropout": float(cfg["model"]["dropout"]),
    }
    model = build_model(name, **data_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(tcfg["learning_rate"]), weight_decay=float(tcfg["weight_decay"])
    )
    scheduler_type = str(tcfg.get("scheduler", "plateau")).lower()
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(tcfg["epochs"]) - int(tcfg["warmup_epochs"])),
            eta_min=float(tcfg["min_learning_rate"]),
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(tcfg["plateau_factor"]),
            patience=int(tcfg["plateau_patience"]),
            min_lr=float(tcfg["min_learning_rate"]),
        )
    scaler = torch.amp.GradScaler("cuda", init_scale=float(tcfg["amp_init_scale"]), enabled=amp)
    ema = EMA(model, float(tcfg["ema_decay"]))
    ema_model = copy.deepcopy(model).to(device)

    run = Path(args.output).expanduser() / name
    run.mkdir(parents=True, exist_ok=True)
    last_path = run / "last.pt"
    best_path = run / "best.pt"
    start_epoch = 0
    best_score = float("inf")
    stale_epochs = 0
    if args.resume:
        if not last_path.is_file():
            raise FileNotFoundError(f"Cannot resume without {last_path}")
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state["model_name"] != name:
            raise ValueError("Resume checkpoint model name does not match")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        ema.shadow = {key: value.to(device) for key, value in state["ema"].items()}
        start_epoch = int(state["epoch"]) + 1
        best_score = float(state["best_score"])
        stale_epochs = int(state["stale_epochs"])
        _restore_rng(state["rng"], device)
    else:
        if last_path.exists():
            raise FileExistsError(f"Run already exists; use --resume or another output: {last_path}")
        if args.init_checkpoint:
            checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
            if checkpoint["model_name"] != name:
                raise ValueError("Initial checkpoint model name does not match")
            model.load_state_dict(checkpoint["model"], strict=True)
            print(f"{name}: initialized from {args.init_checkpoint}", flush=True)
        ema = EMA(model, float(tcfg["ema_decay"]))

    if args.freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(False)

    write_json(run / "config.json", cfg)
    write_json(
        run / "run_info.json",
        {
            "model": name,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "frozen_encoder": bool(args.freeze_encoder),
            "data": str(Path(args.data).resolve()),
            "synthetic": str(Path(args.synthetic).resolve()),
            "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
            "counts": counts,
            "training": tcfg,
            "hybrid": cfg["hybrid"],
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
    )

    history_path = run / "history.csv"
    fields = [
        "epoch", "train_loss", "train_event", "train_interval", "train_terminal",
        "train_sparsity", "train_syn_loss", "train_hyb_loss", "train_field_loss",
        "train_cal_loss", "val_syn", "val_hyb", "val_field", "val_cal",
        "syn_f1", "train_syn_f1", "syn_terminal_f1",
        "hyb_hit20", "hyb_cand", "field_hit20", "field_cand", "cal_false_events",
        "score_raw", "score_ema", "best_score", "ema_selected",
        "grad_norm_mean", "grad_norm_max", "amp_skipped", "lr", "seconds",
        "samples_per_second", "peak_gpu_gib",
    ]
    print(
        f"{name}: params={sum(p.numel() for p in model.parameters()):,} counts={counts} "
        f"batch={tcfg['batch_size']} workers={tcfg['num_workers']} amp={amp}",
        flush=True,
    )

    for epoch in range(start_epoch, int(tcfg["epochs"])):
        if epoch < int(tcfg["warmup_epochs"]):
            warmup_lr = float(tcfg["learning_rate"]) * (epoch + 1) / int(tcfg["warmup_epochs"])
            for group in optimizer.param_groups:
                group["lr"] = warmup_lr
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        begun = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        train_result = train_epoch(
            model, train_loader, optimizer, scaler, ema, device, amp,
            sparsity_weight, epoch, epoch * len(train_loader), args.max_train_batches,
        )
        append_steps(run / "steps.csv", train_result["steps"])
        raw = validate(
            model, val_loaders, syn_val_dataset, hyb_val_dataset, cal_val_dataset,
            field_val_dataset, train_probe, device, amp, sparsity_weight,
            int(tcfg["batch_size"]), args.max_val_batches, cfg,
        )
        ema.copy_to(ema_model)
        ema_result = validate(
            ema_model, val_loaders, syn_val_dataset, hyb_val_dataset, cal_val_dataset,
            field_val_dataset, train_probe, device, amp, sparsity_weight,
            int(tcfg["batch_size"]), args.max_val_batches, cfg,
        )
        ema_selected = ema_result["score"] < raw["score"]
        selected = ema_result if ema_selected else raw
        improved = selected["score"] < best_score
        if improved:
            best_score = selected["score"]
            stale_epochs = 0
            state_model = ema_model if ema_selected else model
            torch.save(
                {
                    "model_name": name,
                    "config": cfg,
                    "model": state_model.state_dict(),
                    "epoch": epoch,
                    "score": float(best_score),
                    "ema": bool(ema_selected),
                },
                best_path,
            )
        else:
            stale_epochs += 1
        if epoch >= int(tcfg["warmup_epochs"]) - 1:
            if scheduler_type == "cosine":
                scheduler.step()
            else:
                scheduler.step(selected["score"])

        elapsed = time.perf_counter() - begun
        train_domains = train_result.get("domains", {})
        row = {
            "epoch": epoch + 1,
            "train_loss": train_result["total"],
            "train_event": train_result["event"],
            "train_interval": train_result["interval"],
            "train_terminal": train_result["terminal"],
            "train_sparsity": train_result["sparsity"],
            "train_syn_loss": train_domains.get(0, {}).get("total"),
            "train_hyb_loss": train_domains.get(1, {}).get("total"),
            "train_field_loss": train_domains.get(2, {}).get("total"),
            "train_cal_loss": train_domains.get(3, {}).get("total"),
            "val_syn": raw["synthetic"]["total"],
            "val_hyb": raw["hybrid"]["total"],
            "val_field": raw["field_loss"]["total"],
            "val_cal": raw["calibration"]["total"],
            "syn_f1": raw["syn_f1"],
            "train_syn_f1": raw["train_syn_f1"],
            "syn_terminal_f1": raw["syn_terminal_f1"],
            "hyb_hit20": raw["hybrid_metrics"].get("hit_rate_20"),
            "hyb_cand": raw["hybrid_metrics"].get("candidates_per_curve"),
            "field_hit20": raw["field_metrics"].get("hit_rate_20"),
            "field_cand": raw["field_metrics"].get("candidates_per_curve"),
            "cal_false_events": raw["calibration_metrics"]["false_event_count"],
            "score_raw": raw["score"],
            "score_ema": ema_result["score"],
            "best_score": best_score,
            "ema_selected": int(ema_selected),
            "grad_norm_mean": train_result["grad_norm_mean"],
            "grad_norm_max": train_result["grad_norm_max"],
            "amp_skipped": int(train_result["amp_skipped_steps"]),
            "lr": epoch_lr,
            "seconds": elapsed,
            "samples_per_second": train_result["samples"] / max(elapsed, 1e-9),
            "peak_gpu_gib": (
                torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
            ),
        }
        state = {
            "model_name": name,
            "config": cfg,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "ema": ema.shadow,
            "epoch": epoch,
            "best_score": best_score,
            "stale_epochs": stale_epochs,
            "rng": _rng_state(device),
        }
        torch.save(state, run / "last.tmp")
        (run / "last.tmp").replace(last_path)
        exists = history_path.exists()
        with history_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if stale_epochs >= int(tcfg["patience"]):
            print(f"{name}: early stopping after {stale_epochs} epochs", flush=True)
            break

    del train_loader, val_loaders, model, ema_model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--synthetic", type=Path, default=DEFAULT_SYNTHETIC)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model", choices=("unet", "transformer", "all"), default="all")
    parser.add_argument("--output", type=Path, default=HERE / "output" / "hybrid")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--resume", action="store_true")
    modes.add_argument("--init-checkpoint", type=Path)
    args = parser.parse_args()
    if args.model == "all" and args.init_checkpoint is not None:
        parser.error("--init-checkpoint applies to one model; select unet or transformer")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    names = ("unet", "transformer") if args.model == "all" else (args.model,)
    for name in names:
        train_one(name, args)


if __name__ == "__main__":
    main()
