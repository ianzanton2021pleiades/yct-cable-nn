"""Measured-field adaptation of the physics-regularised pretrained backbone.

Reuses the frozen CMDC injected-field corpus (read only): measured S11 as the
background with known synthetic reflections injected at exactly known
distances.  A synthetic replay slice keeps the clean-domain skill alive while
the physics residuals keep the trace explanation honest on real backgrounds.
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
from torch.utils.data import ConcatDataset, DataLoader, Subset, WeightedRandomSampler

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
CMDC = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_Hybrid_20261022"
CODEX = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_20261022"
for path in (HERE, CMDC, CODEX):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data_pinn import PinnDataset  # noqa: E402
from decode import decode_predictions  # noqa: E402
from eval_utils import field_metrics, hybrid_metrics  # noqa: E402
from hybrid_data import FieldPositiveDataset, HybridDataset  # noqa: E402
from losses_pinn import PhysicsResidual, PhysicsWeights, compute_losses, total_loss  # noqa: E402
from metrics import evaluate_predictions  # noqa: E402
from models_pinn import build_model  # noqa: E402
from train_pinn import EMA, append_rows, evaluate_loss, predict, synthetic_truths, write_json  # noqa: E402


FIELD_ROOT = Path(r"E:\CableNNDataSet\Deepseek_CMDC\FieldNN_Hybrid_Long_20261022")
SYNTHETIC = Path(r"E:\CableNNDataSet\CodeX\FieldNN_Iter4_20261022")
THRESHOLDS = {"event": 0.5, "interval": 0.95, "terminal": 0.6}


def collect(model, dataset, device, batch_size=64):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    probabilities, indices = [], []
    with torch.inference_mode():
        for batch in loader:
            inputs, item = batch[0], batch[2]
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                output = model(inputs.to(device, non_blocking=device.type == "cuda"))
            probabilities.extend(output[:, :3].float().sigmoid().cpu().numpy())
            indices.extend(int(value) for value in item.tolist())
    return probabilities, indices


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "output" / "field_ft")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=8e-6)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--synthetic-weight", type=float, default=0.35)
    parser.add_argument("--field-weight", type=float, default=0.15)
    parser.add_argument("--physics", type=int, default=1)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    seed = 2026102302
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    amp = device.type == "cuda"

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    model = build_model(
        state["model_name"], input_channels=8, width=32, depth=3,
        dropout=config["model"].get("dropout", 0.15),
    ).to(device)
    model.load_state_dict(state["model"], strict=True)
    print(f"loaded {args.checkpoint} epoch={state.get('epoch')} val_f1={state.get('val_f1'):.4f}",
          flush=True)
    if args.freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in model.bottleneck.parameters():
            parameter.requires_grad_(False)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"frozen encoder and bottleneck, trainable parameters {trainable:,}", flush=True)

    physics = None
    if args.physics:
        physics = PhysicsResidual(weights=PhysicsWeights()).to(device)

    hybrid_train = HybridDataset(FIELD_ROOT, "train", augment=True, mixup_prob=0.20)
    hybrid_val = HybridDataset(FIELD_ROOT, "val", augment=False)
    synthetic = PinnDataset(SYNTHETIC, "train", train=True, warp_prob=0.5, mixup_prob=0.15)
    stride = max(1, len(synthetic) // 2200)
    synthetic = Subset(synthetic, list(range(0, len(synthetic), stride))[:2200])
    field_positive = FieldPositiveDataset(FIELD_ROOT, "train", config, augment=True)

    combined = ConcatDataset([hybrid_train, synthetic, field_positive])
    sizes = [len(hybrid_train), len(synthetic), len(field_positive)]
    weights = torch.tensor(
        [1.0] * sizes[0] + [args.synthetic_weight] * sizes[1] + [args.field_weight] * sizes[2],
        dtype=torch.double,
    )
    sampler = WeightedRandomSampler(weights, num_samples=len(hybrid_train), replacement=True)
    train_loader = DataLoader(
        combined, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
        pin_memory=True, drop_last=True, persistent_workers=args.workers > 0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - 1), eta_min=1e-7
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0, enabled=amp)
    ema = EMA(model, 0.999)

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    best_path = output / "best.pt"
    history_path = output / "history.csv"
    fields = [
        "epoch", "train_loss", "train_supervision", "train_physics", "hyb_hit20", "hyb_cand",
        "field_hit20", "field_cand", "syn_f1", "score", "lr", "seconds",
    ]
    print(f"hybrid train={len(hybrid_train)} synthetic={len(synthetic)} "
          f"field_pos={len(field_positive)} val={len(hybrid_val)}", flush=True)

    synthetic_val = PinnDataset(SYNTHETIC, "val", train=False, view="mixed")
    probe_root = PinnDataset(SYNTHETIC, "val", train=False, view="clean")
    probe = Subset(probe_root, list(range(0, len(probe_root), max(1, len(probe_root) // 300)))[:300])

    best_score = float("inf")
    for epoch in range(args.epochs):
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        begun = time.perf_counter()
        model.train()
        seen = 0
        total_sum = sup_sum = phys_sum = 0.0
        for batch in train_loader:
            inputs, targets = batch[0], batch[1]
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                output = model(inputs)
            components = compute_losses(output[:, :3], targets)
            supervision = total_loss(components, 0.0)
            if physics is not None:
                physics_value, _ = physics(output.float(), inputs.float())
                loss = supervision + physics_value
            else:
                physics_value = torch.zeros((), device=device)
                loss = supervision
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            count = int(inputs.shape[0])
            total_sum += float(loss) * count
            sup_sum += float(supervision) * count
            phys_sum += float(physics_value) * count
            seen += count

        candidates = [model]
        ema_model = None
        chosen = model
        ema_model = copy.deepcopy(model).to(device)
        ema.copy_to(ema_model)
        candidates.append(ema_model)

        best_epoch_report = None
        for candidate_index, candidate in enumerate(candidates):
            hybrid_probs, hybrid_indices = collect(candidate, hybrid_val, device)
            hybrid_records = [hybrid_val.records[i] for i in hybrid_indices]
            hybrid_report = hybrid_metrics(hybrid_probs, hybrid_records, config, THRESHOLDS)["non_wet"]
            probe_probs, probe_indices = collect(candidate, probe, device)
            probe_report = evaluate_predictions(
                [decode_predictions(item, config, THRESHOLDS) for item in probe_probs],
                synthetic_truths(probe_root, probe_indices), config,
            )
            syn_f1 = float(probe_report["event_metrics_by_tolerance_m"]["5"].get("f1") or 0.0)
            hit = float(hybrid_report.get("hit_rate_20") or 0.0)
            cand = float(hybrid_report.get("candidates_per_curve") or 0.0)
            score = 0.5 * (1.0 - hit) + 0.5 * (1.0 - syn_f1) + 0.02 * cand
            if best_epoch_report is None or score < best_epoch_report["score"]:
                best_epoch_report = {
                    "score": score, "hit": hit, "cand": cand, "syn_f1": syn_f1,
                    "candidate": candidate_index,
                }
                chosen = candidate
        if best_epoch_report["score"] < best_score:
            best_score = best_epoch_report["score"]
            torch.save({
                "model_name": state["model_name"],
                "config": config,
                "model": chosen.state_dict(),
                "epoch": epoch,
                "hybrid_hit20": best_epoch_report["hit"],
                "syn_f1": best_epoch_report["syn_f1"],
                "score": best_score,
            }, best_path)
        scheduler.step()

        elapsed = time.perf_counter() - begun
        row = {
            "epoch": epoch + 1,
            "train_loss": total_sum / max(seen, 1),
            "train_supervision": sup_sum / max(seen, 1),
            "train_physics": phys_sum / max(seen, 1),
            "hyb_hit20": best_epoch_report["hit"],
            "hyb_cand": best_epoch_report["cand"],
            "field_hit20": None,
            "field_cand": None,
            "syn_f1": best_epoch_report["syn_f1"],
            "score": best_epoch_report["score"],
            "lr": epoch_lr,
            "seconds": elapsed,
        }
        append_rows(history_path, [row], fields)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    write_json(output / "run_info.json", {
        "checkpoint": str(args.checkpoint), "epochs": args.epochs, "lr": args.lr,
        "tag": args.tag, "best_score": best_score,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
