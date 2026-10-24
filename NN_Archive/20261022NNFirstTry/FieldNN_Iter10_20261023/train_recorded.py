"""Recorded replay of an Iter8 field-adaptation route."""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
ITER8_DIR = HERE.parent / "FieldNN_Iter8_20261022"
ITER6_DIR = HERE.parent / "FieldNN_Iter6_20261022"
ITER5_DIR = HERE.parent / "FieldNN_Iter5_20261022"
# The selection density budget is deliberately shared by both architectures.
ITER5_FIELD_BASELINE = ITER5_DIR / "output" / "iter5" / "unet" / "field_e06" / "predictions.jsonl"
ITER5_INITIAL_CHECKPOINTS = {
    model_name: ITER5_DIR / "output" / "iter5" / model_name / "best.pt"
    for model_name in ("unet", "transformer")
}
ITER8_REFERENCE_CHECKPOINTS = {
    model_name: ITER8_DIR / "output" / "iter8" / model_name / "best.pt"
    for model_name in ("unet", "transformer")
}

SEED = 2026102301
INPUT_CHANNELS = 12
WEAK_WEIGHT = 0.1
CALIBRATION_WEIGHT = 0.5
FIELD_EVERY_STEPS = 2
CALIBRATION_BATCH_SIZE = 8
FIELD_EVENT_FIXED = 0.94
FIELD_EVENT_DIAGNOSTIC = 0.60
FIELD_THRESHOLD_GRID = tuple(index / 100.0 for index in range(60, 101))
MIN_EVENT_DECODE_THRESHOLD = 0.60
EVAL_FP32 = False
FIELD_TOLERANCE_M = 20.0
PATIENCE = 2
MIN_LR = 2e-6
BACKBONE_LR = 1e-5
STEM_LR = 1e-4


if str(ITER8_DIR) not in sys.path:
    sys.path.insert(0, str(ITER8_DIR))
for _path in (ITER5_DIR, ITER6_DIR):
    if str(_path) not in sys.path:
        sys.path.append(str(_path))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


i8_train = load_module(ITER8_DIR / "train.py", "iter10_i8_train_replay")
field_evaluator = load_module(ITER6_DIR / "evaluate_field.py", "iter10_field_evaluator")
density_tools = load_module(
    ITER6_DIR / "calibrate_field_density.py", "iter10_density_tools"
)
decoder = load_module(ITER5_DIR / "decode.py", "iter10_i5_decoder")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


@dataclass
class FieldCache:
    """Cached Iter8 features plus their source/report-axis records."""

    features: np.ndarray
    records: list[dict[str, Any]]

    @classmethod
    def from_root(cls, root: str | Path) -> "FieldCache":
        root = Path(root)
        return cls(
            np.load(root / "field_features.npy", mmap_mode="r"),
            read_jsonl(root / "field_records.jsonl"),
        )

    @property
    def records_by_index(self) -> dict[int, dict[str, Any]]:
        return {int(row["index"]): row for row in self.records}


def _source_axis_prediction(
    probability: np.ndarray,
    record: dict[str, Any],
    cfg: dict[str, Any],
    threshold: float,
) -> dict[str, Any]:
    prediction = decoder.decode_predictions(
        probability,
        cfg,
        {"event": float(threshold), "interval": 0.99, "terminal": 0.4},
    )
    scale = float(record.get("source_scale_ratio") or 1.0)
    for event in prediction["events"]:
        event["distance_m"] = float(event["distance_m"]) * scale
    for interval in prediction["intervals"]:
        interval["start_m"] = float(interval["start_m"]) * scale
        interval["end_m"] = float(interval["end_m"]) * scale
    if prediction["terminal"] is not None:
        prediction["terminal"]["distance_m"] = (
            float(prediction["terminal"]["distance_m"]) * scale
        )
    return prediction


@torch.inference_mode()
def _infer_logits(
    model: torch.nn.Module,
    field: FieldCache,
    indices: Iterable[int],
    device: torch.device,
    batch_size: int,
) -> dict[int, np.ndarray]:
    model.eval()
    amp = device.type == "cuda" and not EVAL_FP32
    selected = list(indices)
    outputs: dict[int, np.ndarray] = {}
    for start in range(0, len(selected), batch_size):
        current = selected[start : start + batch_size]
        features = np.array(field.features[current], dtype=np.float32, copy=True)
        inputs = torch.from_numpy(features).to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(inputs).float()
        for index, value in zip(current, logits.cpu().numpy()):
            outputs[int(index)] = value
    return outputs


@torch.inference_mode()
def infer_fields(
    model: torch.nn.Module,
    field: FieldCache | str | Path,
    cfg: dict[str, Any],
    threshold: float = 0.6,
    batch_size: int = 8,
) -> dict[int, dict[str, Any]]:
    """Infer cached field curves and return Iter6-compatible index rows.

    The returned mapping is keyed by continuous ``field_records.index``. Each
    value has ``record/status/prediction`` and event distances use the source /
    report axis, ready for ``Iter6.evaluate_field``.
    """
    if isinstance(field, (str, Path)):
        field = FieldCache.from_root(field)
    device = next(model.parameters()).device
    records_by_index = field.records_by_index
    logits = _infer_logits(
        model, field, sorted(records_by_index), device, batch_size
    )
    result = {}
    for index, record in records_by_index.items():
        probability = torch.sigmoid(torch.from_numpy(logits[index])).numpy()
        result[index] = {
            "record": record,
            "status": "predicted",
            "prediction": _source_axis_prediction(probability, record, cfg, threshold),
            "distance_axis": "source_report",
        }
    return result


class FixedOrderSampler(Sampler[int]):
    def __init__(self, indices: list[int]) -> None:
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


class EpochRoute:
    """Deterministic field-bag and calibration draw sequence for resume."""

    def __init__(self, epoch: int, bag_count: int, calibration_count: int) -> None:
        self.bag_rng = np.random.default_rng(SEED + 2000 + epoch)
        self.calibration_rng = np.random.default_rng(SEED + 3000 + epoch)
        self.bag_count = bag_count
        self.calibration_count = calibration_count
        self.bag_order: list[int] = []
        self.calibration_order = self.calibration_rng.permutation(calibration_count).tolist()
        self.calibration_cursor = 0

    def next_bag(self) -> int:
        if not self.bag_order:
            self.bag_order = self.bag_rng.permutation(self.bag_count).tolist()
        return int(self.bag_order.pop())

    def next_calibration_indices(self, batch_size: int) -> list[int]:
        if self.calibration_cursor >= self.calibration_count:
            self.calibration_order = self.calibration_rng.permutation(
                self.calibration_count
            ).tolist()
            self.calibration_cursor = 0
        end = min(self.calibration_cursor + batch_size, self.calibration_count)
        selected = self.calibration_order[self.calibration_cursor : end]
        self.calibration_cursor = end
        return [int(index) for index in selected]

    def state_dict(self) -> dict[str, Any]:
        return {
            "bag_rng": self.bag_rng.bit_generator.state,
            "calibration_rng": self.calibration_rng.bit_generator.state,
            "bag_order": list(self.bag_order),
            "calibration_order": list(self.calibration_order),
            "calibration_cursor": self.calibration_cursor,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.bag_rng.bit_generator.state = state["bag_rng"]
        self.calibration_rng.bit_generator.state = state["calibration_rng"]
        self.bag_order = [int(index) for index in state["bag_order"]]
        self.calibration_order = [int(index) for index in state["calibration_order"]]
        self.calibration_cursor = int(state["calibration_cursor"])


def train_epoch_order(length: int, epoch: int) -> tuple[list[int], torch.Generator]:
    order_generator = torch.Generator().manual_seed(SEED + 1000 + epoch)
    order = torch.randperm(length, generator=order_generator).tolist()
    worker_generator = torch.Generator().manual_seed(SEED + 4000 + epoch)
    return order, worker_generator


def field_mil_loss(
    logits_by_index: dict[int, np.ndarray],
    field_bags,
    cfg: dict[str, Any],
) -> float:
    losses = []
    for bag in field_bags.bags:
        indices = [int(index) for index in bag["indices"]]
        logits = torch.from_numpy(np.stack([logits_by_index[index] for index in indices]))
        windows = field_bags.windows(
            bag, logits.shape[-1], float(cfg["response"]["distance_max_m"])
        )
        losses.append(float(i8_train.field_objective(logits, windows)))
    return float(np.mean(losses)) if losses else 0.0


def positive_metrics(
    bags: list[dict[str, Any]],
    records_by_index: dict[int, dict[str, Any]],
    predictions_by_index: dict[int, dict[str, Any]],
    allowed_indices: set[int],
) -> dict[str, Any]:
    report = field_evaluator.positive_summary(
        bags, records_by_index, predictions_by_index, allowed_indices
    )
    macro = report["independent_group_macro"][str(int(FIELD_TOLERANCE_M))]
    return {
        "group_macro_hit20": float(macro["macro_hit_rate"] or 0.0),
        "micro_hit20": float(macro["micro_hit_rate"] or 0.0),
        "matched_positions": int(macro["matched_positions"]),
        "unique_positions": int(macro["unique_positions"]),
    }


def validation_rank(validation: dict[str, Any]) -> tuple[float, int, float, float]:
    return (
        round(float(validation["val_macro_hit20_selected"]), 12),
        -int(validation["val_candidates_total_selected"]),
        -float(validation["val_field_loss"]),
        float(validation["val_threshold"]),
    )


class FieldMetricContext:
    def __init__(self, data_root: Path, field: FieldCache, field_train, field_val) -> None:
        self.data_root = data_root
        self.field = field
        self.field_train = field_train
        self.field_val = field_val
        self.records = field.records
        self.records_by_index = field.records_by_index
        bag_data = json.loads((data_root / "field_bags.json").read_text(encoding="utf-8"))
        self.all_bags = bag_data["bags"] if isinstance(bag_data, dict) else bag_data
        self.train_bags = field_evaluator.bag_rows(
            "train", self.all_bags, self.records_by_index
        )
        self.val_bags = field_evaluator.bag_rows(
            "val", self.all_bags, self.records_by_index
        )
        self.train_indices = {int(index) for bag in self.train_bags for index in bag["indices"]}
        self.val_indices = {int(index) for bag in self.val_bags for index in bag["indices"]}
        self.annotated_indices = sorted(self.train_indices | self.val_indices)

        baseline_keyed = field_evaluator.load_prediction_files(
            [f"iter5_unet={ITER5_FIELD_BASELINE}"]
        )["iter5_unet"]
        baseline_val_records = [
            row for row in self.records if row["split"] == "val"
        ]
        self.baseline_by_index = field_evaluator.prediction_by_index(
            baseline_val_records, baseline_keyed
        )
        common_val = {
            index for index in self.val_indices
            if index in self.baseline_by_index
            and self.baseline_by_index[index].get("status") == "predicted"
        }
        baseline_density = density_tools.density_summary(
            common_val, self.baseline_by_index, field_evaluator
        )
        self.density_indices = common_val
        self.density_budget = (
            baseline_density["event_candidates_unverified"]
            / baseline_density["applicable_curves"]
        )
        self.baseline_density = baseline_density

    def evaluate(
        self,
        model: torch.nn.Module,
        cfg: dict[str, Any],
        device: torch.device,
        batch_size: int,
    ) -> dict[str, Any]:
        logits_by_index = _infer_logits(
            model, self.field, self.annotated_indices, device, batch_size
        )
        source_predictions: dict[int, dict[str, Any]] = {}
        for index in self.annotated_indices:
            record = self.records_by_index[index]
            probability = torch.sigmoid(torch.from_numpy(logits_by_index[index])).numpy()
            prediction = _source_axis_prediction(
                probability, record, cfg, MIN_EVENT_DECODE_THRESHOLD
            )
            source_predictions[index] = {
                "record": record,
                "status": "predicted",
                "prediction": prediction,
                "distance_axis": "source_report",
            }

        train_fixed = density_tools.filter_by_confidence(
            source_predictions, FIELD_EVENT_FIXED
        )
        val_fixed = density_tools.filter_by_confidence(
            source_predictions, FIELD_EVENT_FIXED
        )
        train_diag = density_tools.filter_by_confidence(
            source_predictions, FIELD_EVENT_DIAGNOSTIC
        )
        val_diag = density_tools.filter_by_confidence(
            source_predictions, FIELD_EVENT_DIAGNOSTIC
        )
        train_fixed_metric = positive_metrics(
            self.train_bags, self.records_by_index, train_fixed, self.train_indices
        )
        val_fixed_metric = positive_metrics(
            self.val_bags, self.records_by_index, val_fixed, self.val_indices
        )
        train_diag_metric = positive_metrics(
            self.train_bags, self.records_by_index, train_diag, self.train_indices
        )
        val_diag_metric = positive_metrics(
            self.val_bags, self.records_by_index, val_diag, self.val_indices
        )

        feasible = []
        for threshold in FIELD_THRESHOLD_GRID:
            filtered = density_tools.filter_by_confidence(source_predictions, threshold)
            density = density_tools.density_summary(
                self.density_indices, filtered, field_evaluator
            )
            candidates = int(density["event_candidates_unverified"])
            curves = max(int(density["applicable_curves"]), 1)
            candidates_per_curve = candidates / curves
            if candidates_per_curve <= self.density_budget + 1e-12:
                hit = positive_metrics(
                    self.val_bags, self.records_by_index, filtered, self.val_indices
                )
                feasible.append((
                    hit["group_macro_hit20"], -candidates, threshold,
                    hit, candidates, candidates_per_curve,
                ))

        if feasible:
            selected = max(feasible, key=lambda row: (round(row[0], 12), row[1], row[2]))
            threshold = float(selected[2])
            selected_hit = selected[3]
            selected_candidates = int(selected[4])
            selected_density = float(selected[5])
            fallback = False
        else:
            threshold = 1.0
            suppressed = density_tools.filter_by_confidence(source_predictions, None)
            selected_hit = positive_metrics(
                self.val_bags, self.records_by_index, suppressed, self.val_indices
            )
            selected_candidates = 0
            selected_density = 0.0
            fallback = True

        train_loss = field_mil_loss(logits_by_index, self.field_train, cfg)
        val_loss = field_mil_loss(logits_by_index, self.field_val, cfg)
        return {
            "train_field_loss": train_loss,
            "val_field_loss": val_loss,
            "train_hit20_fixed": train_fixed_metric["group_macro_hit20"],
            "val_hit20_fixed": val_fixed_metric["group_macro_hit20"],
            "train_hit20_fixed_micro": train_fixed_metric["micro_hit20"],
            "val_hit20_fixed_micro": val_fixed_metric["micro_hit20"],
            "train_hit20_fixed_060": train_diag_metric["group_macro_hit20"],
            "val_hit20_fixed_060": val_diag_metric["group_macro_hit20"],
            "val_macro_hit20_selected": selected_hit["group_macro_hit20"],
            "val_micro_hit20_selected": selected_hit["micro_hit20"],
            "val_threshold": threshold,
            "val_candidates_total_selected": selected_candidates,
            "val_candidates_per_curve_selected": selected_density,
            "val_density_budget_per_curve": self.density_budget,
            "val_density_fallback": fallback,
            "fixed_event_threshold": FIELD_EVENT_FIXED,
            "train_candidates_per_curve_fixed": _density_per_curve(
                train_fixed, self.train_indices, field_evaluator
            ),
            "val_candidates_per_curve_fixed": _density_per_curve(
                val_fixed, self.val_indices, field_evaluator
            ),
            "train_fixed_candidates_per_curve_094": (
                _density_per_curve(train_fixed, self.train_indices, field_evaluator)
                if abs(FIELD_EVENT_FIXED - 0.94) < 1e-9 else None
            ),
            "val_fixed_candidates_per_curve_094": (
                _density_per_curve(val_fixed, self.val_indices, field_evaluator)
                if abs(FIELD_EVENT_FIXED - 0.94) < 1e-9 else None
            ),
        }


def _density_per_curve(predictions, indices: set[int], evaluator) -> float:
    density = density_tools.density_summary(indices, predictions, evaluator)
    count = int(density["applicable_curves"])
    return float(density["event_candidates_unverified"] / count) if count else 0.0


@torch.inference_mode()
def validate_nonfield(
    model: torch.nn.Module,
    synthetic_loader,
    calibration_loader,
    device: torch.device,
    amp: bool,
) -> tuple[float, float, float]:
    model.eval()
    synthetic_total = synthetic_count = 0.0
    for x, y, _ in synthetic_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(x)
        loss = i8_train.total_loss(i8_train.compute_losses(logits, y))
        synthetic_total += float(loss) * len(x)
        synthetic_count += len(x)

    calibration_total = calibration_count = high_count = valid_count = 0.0
    for x, targets, _ in calibration_loader:
        x = x.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(x)
        loss, high, valid = i8_train.calibration_objective(logits, targets)
        calibration_total += float(loss) * len(x)
        calibration_count += len(x)
        high_count += int(high)
        valid_count += int(valid)
    return (
        synthetic_total / max(synthetic_count, 1.0),
        calibration_total / max(calibration_count, 1.0),
        high_count / max(valid_count, 1.0),
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def optimizer_learning_rates(
    optimizer: torch.optim.Optimizer, stem_only: bool,
) -> tuple[float, float]:
    if stem_only:
        return 0.0, float(optimizer.param_groups[0]["lr"])
    return float(optimizer.param_groups[0]["lr"]), float(optimizer.param_groups[1]["lr"])


HISTORY_FIELDS = [
    "optimizer_step",
    "train_field_loss",
    "val_field_loss",
    "train_hit20_fixed",
    "val_hit20_fixed",
    "val_macro_hit20_selected",
    "val_candidates_per_curve_selected",
    "val_threshold",
    "amp_skipped_steps",
    "train_hit20_fixed_micro",
    "val_hit20_fixed_micro",
    "train_hit20_fixed_060",
    "val_hit20_fixed_060",
    "val_micro_hit20_selected",
    "val_candidates_total_selected",
    "val_density_budget_per_curve",
    "val_density_fallback",
    "fixed_event_threshold",
    "train_candidates_per_curve_fixed",
    "val_candidates_per_curve_fixed",
    "train_fixed_candidates_per_curve_094",
    "val_fixed_candidates_per_curve_094",
    "synthetic_val_loss",
    "calibration_val_loss",
    "calibration_val_high_bin_fraction",
    "lr_backbone",
    "lr_stem",
]

STEP_FIELDS = [
    "optimizer_step",
    "attempt_step",
    "epoch",
    "step_in_epoch",
    "synthetic_loss",
    "field_positive_loss",
    "calibration_loss",
    "weighted_total_loss",
    "field_update",
    "calibration_update",
    "grad_norm",
    "amp_skipped",
    "amp_scale",
    "lr_backbone",
    "lr_stem",
    "seconds_since_start",
]


def checkpoint_payload(
    model: torch.nn.Module,
    model_name: str,
    cfg: dict[str, Any],
    validation: dict[str, Any],
    source: str,
    step: int | None,
) -> dict[str, Any]:
    return {
        "schema": f"fieldnn-iter10-{model_name}-checkpoint-1",
        "model_name": model_name,
        "config": cfg,
        "model": model.state_dict(),
        "epoch": -1 if step == 0 else (int(step) - 1) // int(cfg["training"]["steps_per_epoch"]),
        "step": step,
        "validation": jsonable(validation),
        "threshold": float(validation["val_threshold"]),
        "selected_source": source,
        "optimizer_step": step,
    }


def select_checkpoint(
    run: Path,
    model_name: str,
    reference_checkpoint: dict[str, Any],
    reference_validation: dict[str, Any],
    dense_best_validation: dict[str, Any],
    dense_best_step: int,
) -> dict[str, Any]:
    if dense_best_validation is None or validation_rank(reference_validation) >= validation_rank(dense_best_validation):
        selected_source = "iter8_reference"
        selected_step = None
        selected_validation = reference_validation
        payload = copy.deepcopy(reference_checkpoint)
        payload.update({
            "schema": f"fieldnn-iter10-{model_name}-checkpoint-1",
            "model_name": model_name,
            "validation": jsonable(reference_validation),
            "threshold": float(reference_validation["val_threshold"]),
            "selected_source": selected_source,
            "step": None,
            "optimizer_step": None,
        })
    else:
        selected_source = "iter10_dense"
        selected_step = int(dense_best_step)
        selected_validation = dense_best_validation
        payload = torch.load(run / "best_dense.pt", map_location="cpu", weights_only=False)
        payload["selected_source"] = selected_source

    torch.save(payload, run / "best.pt")
    selection = {
        "selected_source": selected_source,
        "threshold": float(selected_validation["val_threshold"]),
        "step": selected_step,
        "validation": jsonable(selected_validation),
        "dense_best_step": int(dense_best_step),
        "reference_validation": jsonable(reference_validation),
    }
    write_json(run / "selection.json", selection)
    return selection


def save_last(
    path: Path,
    model: torch.nn.Module,
    model_name: str,
    cfg: dict[str, Any],
    run_config: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    step: int,
    attempt_step: int,
    epoch: int,
    batch_offset: int,
    route: EpochRoute,
    history_rows: list[dict[str, Any]],
    optimizer_rows: list[dict[str, Any]],
    dense_best_rank: tuple[float, int, float, float],
    dense_best_validation: dict[str, Any],
    dense_best_step: int,
    reference_validation: dict[str, Any],
    initial_validation: dict[str, Any],
    device: torch.device,
) -> None:
    torch.save({
        "schema": "fieldnn-iter10-last-1",
        "model_name": model_name,
        "config": cfg,
        "run_config": run_config,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "scheduler": scheduler.state_dict(),
        "optimizer_step": step,
        "attempt_step": attempt_step,
        "epoch": epoch,
        "batch_offset": batch_offset,
        "route": route.state_dict(),
        "history_rows": history_rows,
        "optimizer_rows": optimizer_rows,
        "dense_best_rank": dense_best_rank,
        "dense_best_validation": dense_best_validation,
        "dense_best_step": dense_best_step,
        "reference_validation": reference_validation,
        "initial_validation": initial_validation,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        },
    }, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("unet", "transformer"), default="unet")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="Run root; writes OUT/<model>/")
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Optional full synthetic data epochs; overrides --steps")
    parser.add_argument("--record-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--init-iter8", action="store_true",
                        help="Initialize the selected 12-channel model from its Iter8 best.pt")
    parser.add_argument("--stem-only", action="store_true",
                        help="Freeze the selected backbone and train only its 12-to-8 input stem")
    parser.add_argument("--weak-weight", type=float, default=WEAK_WEIGHT)
    parser.add_argument("--stem-lr", type=float, default=STEM_LR)
    parser.add_argument("--event-threshold-min", type=float, default=0.6)
    parser.add_argument("--fixed-event-threshold", type=float, default=0.94)
    parser.add_argument("--eval-fp32", action="store_true",
                        help="Disable autocast for field validation inference")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    global FIELD_EVENT_FIXED, FIELD_THRESHOLD_GRID, MIN_EVENT_DECODE_THRESHOLD, EVAL_FP32
    MIN_EVENT_DECODE_THRESHOLD = args.event_threshold_min
    FIELD_EVENT_FIXED = args.fixed_event_threshold
    FIELD_THRESHOLD_GRID = tuple(
        index / 100.0
        for index in range(int(round(args.event_threshold_min * 100)), 101)
    )
    EVAL_FP32 = bool(args.eval_fp32)

    data_root = args.data.expanduser().resolve()
    synthetic_root = args.synthetic.expanduser().resolve()
    model_name = args.model
    run = args.output.expanduser().resolve() / model_name
    run.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    amp = device.type == "cuda"
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if amp:
        torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    initialization_checkpoint = (
        ITER8_REFERENCE_CHECKPOINTS[model_name]
        if args.init_iter8 else ITER5_INITIAL_CHECKPOINTS[model_name]
    )
    initial = torch.load(initialization_checkpoint, map_location="cpu", weights_only=False)
    initialization_version = 'Iter8' if args.init_iter8 else 'Iter5'
    cfg = copy.deepcopy(initial["config"])
    cfg["seed"] = SEED
    cfg["model"]["input_channels"] = INPUT_CHANNELS
    cfg["training"].update(
        epochs=0,
        batch_size=args.batch_size,
        num_workers=args.workers,
        learning_rate=0.0 if args.stem_only else BACKBONE_LR,
        stem_learning_rate=args.stem_lr,
        patience=PATIENCE,
    )
    cfg["adaptation10"] = {
        "route": f"{initialization_version} {model_name} input-stem-only adaptation" if args.stem_only
        else f"replay Iter8 {model_name} from Iter8 initialization" if args.init_iter8
        else f"replay Iter8 {model_name} from Iter5 initialization",
        "source_checkpoint": str(initialization_checkpoint),
        "init_iter8": bool(args.init_iter8),
        "stem_only": bool(args.stem_only),
        "trainable_parameters": 96 if args.stem_only else None,
        "field_supervision": "same positive-only cross-sweep bag MIL as Iter8",
        "weak_weight": args.weak_weight,
        "calibration_weight": CALIBRATION_WEIGHT,
        "stem_learning_rate": args.stem_lr,
        "field_every_steps": FIELD_EVERY_STEPS,
        "calibration_batch_size": CALIBRATION_BATCH_SIZE,
        "calibration_semantics": "confirmed no-test-cable fixture references; event/interval negatives only from the Iter8 CalibrationDataset masks; terminal and near-end are ignored",
        "selection": "group-macro best-single-sweep hit@20 on field val under the shared Iter5 U-Net validation candidate-density cap; fewer candidates then bag-average val MIL loss",
        "reference_checkpoint": str(ITER8_REFERENCE_CHECKPOINTS[model_name]),
        "data": str(data_root),
        "synthetic": str(synthetic_root),
        "joint_change_note": (
            f"Only the 12-to-8 input stem is trainable; initialize from {initialization_version} {model_name}; Iter8 remains an independent validation reference."
            if args.stem_only else
            f"This route replays the Iter8 {model_name} from the selected initialization checkpoint."
        ),
    }
    run_config = {
        "schema": "fieldnn-iter10-recorded-config-1",
        "model": model_name,
        "data": str(data_root),
        "synthetic": str(synthetic_root),
        "model_config": {key: cfg["model"][key] for key in cfg["model"]},
        "steps": args.steps,
        "record_every": args.record_every,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "device": str(device),
        "amp": amp,
        "event_threshold_min": args.event_threshold_min,
        "fixed_event_threshold": args.fixed_event_threshold,
        "eval_fp32": bool(args.eval_fp32),
        "amp_init_scale": 1024.0,
        "weight_decay": 0.01,
        "gradient_clip_norm": 1.0,
        "backbone_learning_rate": 0.0 if args.stem_only else BACKBONE_LR,
        "stem_learning_rate": args.stem_lr,
        "weak_weight": args.weak_weight,
        "calibration_weight": CALIBRATION_WEIGHT,
        "field_every_steps": FIELD_EVERY_STEPS,
        "calibration_batch_size": CALIBRATION_BATCH_SIZE,
        "seed": SEED,
        "selection": cfg["adaptation10"]["selection"],
        "initialization_checkpoint": str(initialization_checkpoint),
        "init_iter8": bool(args.init_iter8),
        "stem_only": bool(args.stem_only),
    }

    field_cache = FieldCache.from_root(data_root)
    field_train = i8_train.FieldBags(data_root, "train", device)
    field_val = i8_train.FieldBags(data_root, "val", device)
    if len(field_train.bags) != 21 or len(field_val.bags) != 10:
        raise ValueError(
            f"Iter10 expects the frozen Iter8 field split (21 train and 10 val bags); "
            f"got {len(field_train.bags)} and {len(field_val.bags)}"
        )
    metric_context = FieldMetricContext(data_root, field_cache, field_train, field_val)

    train_data = i8_train.ShardDataset(synthetic_root, "train", train=True)
    val_data = i8_train.ShardDataset(synthetic_root, "val")
    steps_per_epoch = math.ceil(len(train_data) / args.batch_size)
    if args.epochs is not None:
        args.steps = args.epochs * steps_per_epoch
    route_epochs = math.ceil(args.steps / steps_per_epoch)
    cfg["training"]["epochs"] = route_epochs
    cfg["training"]["steps_per_epoch"] = steps_per_epoch
    run_config["steps_per_epoch"] = steps_per_epoch
    run_config["epochs_equivalent"] = route_epochs
    run_config["requested_epochs"] = args.epochs
    run_config["steps"] = args.steps
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, num_workers=0,
        pin_memory=amp,
    )
    calibration_train = i8_train.CalibrationDataset(data_root, "train")
    calibration_val = i8_train.CalibrationDataset(data_root, "val")
    calibration_val_loader = DataLoader(
        calibration_val, batch_size=CALIBRATION_BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=amp,
    )

    model = i8_train.build_model(model_name, **cfg["model"]).to(device)
    if args.init_iter8:
        model.load_state_dict(initial["model"])
    else:
        model.backbone.load_state_dict(initial["model"])
    if args.stem_only:
        model.backbone.requires_grad_(False)
        parameter_groups = [{"params": model.input_stem.parameters(), "lr": args.stem_lr}]
    else:
        parameter_groups = [
            {"params": model.backbone.parameters(), "lr": BACKBONE_LR},
            {"params": model.input_stem.parameters(), "lr": args.stem_lr},
        ]
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0, enabled=amp)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=PATIENCE, factor=0.5, min_lr=MIN_LR
    )

    reference_checkpoint_path = ITER8_REFERENCE_CHECKPOINTS[model_name]
    reference = torch.load(reference_checkpoint_path, map_location="cpu", weights_only=False)
    last_path = run / "last.pt"
    best_dense_path = run / "best_dense.pt"
    best_path = run / "best.pt"
    if args.resume:
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if (
            state.get("model_name") != model_name
            or state["run_config"].get("model") != model_name
            or state["run_config"]["data"] != str(data_root)
            or state["run_config"]["synthetic"] != str(synthetic_root)
            or int(state["run_config"]["batch_size"]) != args.batch_size
            or int(state["run_config"]["steps_per_epoch"]) != steps_per_epoch
            or bool(state["run_config"].get("init_iter8", False)) != bool(args.init_iter8)
            or bool(state["run_config"].get("stem_only", False)) != bool(args.stem_only)
            or float(state["run_config"].get("weak_weight", WEAK_WEIGHT)) != float(args.weak_weight)
            or float(state["run_config"].get("stem_learning_rate", STEM_LR)) != float(args.stem_lr)
        ):
            raise ValueError("Resume configuration differs from the recorded run")
        reference_validation = state["reference_validation"]
        initial_validation = state["initial_validation"]
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        scheduler.load_state_dict(state["scheduler"])
        global_step = int(state["optimizer_step"])
        attempt_step = int(state["attempt_step"])
        epoch_index = attempt_step // steps_per_epoch
        batch_offset = attempt_step % steps_per_epoch
        history_rows = list(state["history_rows"])
        optimizer_rows = list(state["optimizer_rows"])
        dense_best_rank = validation_rank(state["dense_best_validation"])
        dense_best_validation = state["dense_best_validation"]
        dense_best_step = int(state["dense_best_step"])
        reference_validation = state["reference_validation"]
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["torch"])
        if amp:
            torch.cuda.set_rng_state(state["rng"]["cuda"], device)
        route = EpochRoute(epoch_index, len(field_train.bags), len(calibration_train))
        route.load_state_dict(state["route"])
        if batch_offset == 0:
            route = EpochRoute(epoch_index, len(field_train.bags), len(calibration_train))
        write_csv(run / "history_steps.csv", history_rows, HISTORY_FIELDS)
        write_csv(run / "optimizer_steps.csv", optimizer_rows, STEP_FIELDS)
        write_json(run / "config.json", run_config)
    else:
        if last_path.exists() or best_dense_path.exists() or best_path.exists():
            raise FileExistsError(f"Run already contains Iter10 checkpoints: {run}")
        global_step = 0
        attempt_step = 0
        epoch_index = 0
        batch_offset = 0
        route = EpochRoute(0, len(field_train.bags), len(calibration_train))
        history_rows = []
        optimizer_rows = []
        reference_model = i8_train.build_model(
            model_name, **reference["config"]["model"]
        ).to(device)
        reference_model.load_state_dict(reference["model"])
        reference_validation = metric_context.evaluate(
            reference_model, reference["config"], device, args.batch_size
        )
        reference_synthetic, reference_calibration, reference_calibration_high = validate_nonfield(
            reference_model, val_loader, calibration_val_loader, device, amp
        )
        reference_validation.update({
            "synthetic_val_loss": reference_synthetic,
            "calibration_val_loss": reference_calibration,
            "calibration_val_high_bin_fraction": reference_calibration_high,
            "candidate_source": "iter8_reference",
            "source_checkpoint": str(reference_checkpoint_path),
        })
        del reference_model
        initial_validation = metric_context.evaluate(model, cfg, device, args.batch_size)
        initial_synthetic, initial_calibration, initial_calibration_high = validate_nonfield(
            model, val_loader, calibration_val_loader, device, amp
        )
        initial_validation.update({
            "synthetic_val_loss": initial_synthetic,
            "calibration_val_loss": initial_calibration,
            "calibration_val_high_bin_fraction": initial_calibration_high,
            "candidate_source": "iter10_dense",
        })
        dense_best_validation = initial_validation
        dense_best_rank = validation_rank(initial_validation)
        dense_best_step = 0
        write_json(run / "config.json", run_config)
        write_json(run / "reference_validation.json", reference_validation)
        write_json(run / "initial_validation.json", initial_validation)
        write_csv(run / "history_steps.csv", history_rows, HISTORY_FIELDS)
        write_csv(run / "optimizer_steps.csv", optimizer_rows, STEP_FIELDS)
        torch.save(
            checkpoint_payload(
                model, model_name, cfg, initial_validation, "iter10_dense", 0
            ),
            best_dense_path,
        )
        select_checkpoint(
            run, model_name, reference,
            reference_validation=reference_validation,
            dense_best_validation=dense_best_validation,
            dense_best_step=dense_best_step,
        )

    def write_metric_point(
        step: int,
        validation: dict[str, Any],
        amp_skipped_steps: int,
        include_nonfield: bool,
    ) -> dict[str, Any]:
        if include_nonfield:
            synthetic_val, calibration_val, calibration_high = validate_nonfield(
                model, val_loader, calibration_val_loader, device, amp
            )
        else:
            synthetic_val = calibration_val = calibration_high = None
        lr_backbone, lr_stem = optimizer_learning_rates(optimizer, args.stem_only)
        row = {
            "optimizer_step": step,
            **{key: value for key, value in validation.items() if key in HISTORY_FIELDS},
            "amp_skipped_steps": amp_skipped_steps,
            "synthetic_val_loss": synthetic_val,
            "calibration_val_loss": calibration_val,
            "calibration_val_high_bin_fraction": calibration_high,
            "lr_backbone": lr_backbone,
            "lr_stem": lr_stem,
        }
        return jsonable(row)

    if not args.resume:
        initial_row = write_metric_point(0, initial_validation, 0, include_nonfield=True)
        history_rows.append(initial_row)
        write_csv(run / "history_steps.csv", history_rows, HISTORY_FIELDS)
        selection = select_checkpoint(
            run, model_name, reference,
            reference_validation=reference_validation,
            dense_best_validation=dense_best_validation,
            dense_best_step=dense_best_step,
        )
        save_last(
            last_path, model, model_name, cfg, run_config, optimizer, scaler, scheduler,
            0, 0, 0, 0, route, history_rows, optimizer_rows, dense_best_rank,
            dense_best_validation, dense_best_step, reference_validation,
            initial_validation, device,
        )

    start_time = time.perf_counter()
    amp_skipped_total = sum(int(row["amp_skipped"]) for row in optimizer_rows)
    while global_step < args.steps:
        epoch_index = attempt_step // steps_per_epoch
        start_offset = attempt_step % steps_per_epoch
        order, worker_generator = train_epoch_order(len(train_data), epoch_index)
        train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            sampler=FixedOrderSampler(order),
            num_workers=args.workers,
            pin_memory=amp,
            persistent_workers=False,
            worker_init_fn=i8_train.worker_init,
            generator=worker_generator,
        )
        model.train()
        progress = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            initial=start_offset,
            desc=f"{model_name} {epoch_index + 1}/{route_epochs}",
            unit="batch",
        )
        for batch_index, (x, y, _) in progress:
            if batch_index < start_offset:
                continue
            if global_step >= args.steps:
                break
            optimizer.zero_grad(set_to_none=True)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(x)
            synthetic_loss = i8_train.total_loss(i8_train.compute_losses(logits, y))
            scaler.scale(synthetic_loss).backward()
            synthetic_value = float(synthetic_loss.detach())
            del logits, synthetic_loss, x, y

            step_in_epoch = batch_index
            field_loss_value = None
            calibration_loss_value = None
            is_field_update = (step_in_epoch % FIELD_EVERY_STEPS) == 0
            is_calibration_update = is_field_update
            if is_field_update:
                bag = field_train.bags[route.next_bag()]
                field_x = field_train.batch(bag)
                with torch.autocast(device_type=device.type, enabled=amp):
                    field_logits = model(field_x)
                windows = field_train.windows(
                    bag, field_logits.shape[-1], cfg["response"]["distance_max_m"]
                )
                field_loss = i8_train.field_objective(field_logits, windows)
                scaler.scale(args.weak_weight * field_loss).backward()
                field_loss_value = float(field_loss.detach())
                del field_logits, field_x, field_loss

                cal_indices = route.next_calibration_indices(CALIBRATION_BATCH_SIZE)
                cal_samples = [calibration_train[index] for index in cal_indices]
                cal_x = torch.from_numpy(np.stack([sample[0] for sample in cal_samples])).to(
                    device, non_blocking=True
                )
                cal_targets = torch.from_numpy(np.stack([sample[1] for sample in cal_samples])).to(
                    device, non_blocking=True
                )
                with torch.autocast(device_type=device.type, enabled=amp):
                    cal_logits = model(cal_x)
                calibration_loss, _, _ = i8_train.calibration_objective(cal_logits, cal_targets)
                scaler.scale(CALIBRATION_WEIGHT * calibration_loss).backward()
                calibration_loss_value = float(calibration_loss.detach())
                del cal_logits, cal_x, cal_targets, calibration_loss

            scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            amp_skipped = int(scaler.get_scale() < scale_before)
            amp_skipped_total += amp_skipped
            attempt_step += 1
            if not amp_skipped:
                global_step += 1
            batch_offset = batch_index + 1

            weighted_total = synthetic_value
            if field_loss_value is not None:
                weighted_total += args.weak_weight * field_loss_value
                weighted_total += CALIBRATION_WEIGHT * float(calibration_loss_value)
            lr_backbone, lr_stem = optimizer_learning_rates(optimizer, args.stem_only)
            optimizer_row = {
                "optimizer_step": global_step,
                "attempt_step": attempt_step,
                "epoch": epoch_index + 1,
                "step_in_epoch": step_in_epoch + 1,
                "synthetic_loss": synthetic_value,
                "field_positive_loss": field_loss_value,
                "calibration_loss": calibration_loss_value,
                "weighted_total_loss": weighted_total,
                "field_update": is_field_update,
                "calibration_update": is_calibration_update,
                "grad_norm": grad_norm,
                "amp_skipped": amp_skipped,
                "amp_scale": scaler.get_scale(),
                "lr_backbone": lr_backbone,
                "lr_stem": lr_stem,
                "seconds_since_start": time.perf_counter() - start_time,
            }
            optimizer_rows.append(jsonable(optimizer_row))
            progress.set_postfix(
                syn=f"{synthetic_value:.4f}",
                field="-" if field_loss_value is None else f"{field_loss_value:.4f}",
                cal="-" if calibration_loss_value is None else f"{calibration_loss_value:.4f}",
                grad=f"{grad_norm:.2f}",
            )

            epoch_boundary = (batch_index + 1) == steps_per_epoch
            record_point = (
                not amp_skipped
                and (global_step % args.record_every == 0 or global_step == args.steps)
            )
            if record_point:
                validation = metric_context.evaluate(model, cfg, device, args.batch_size)
                row = write_metric_point(
                    global_step, validation, amp_skipped_total,
                    include_nonfield=epoch_boundary,
                )
                model.train()
                history_rows.append(row)
                current_rank = validation_rank(validation)
                improved_dense = current_rank > dense_best_rank
                if improved_dense:
                    dense_best_rank = current_rank
                    dense_best_validation = row
                    dense_best_step = global_step
                    torch.save(
                        checkpoint_payload(
                            model, model_name, cfg, row, "iter10_dense", global_step
                        ),
                        best_dense_path,
                    )
                if epoch_boundary:
                    scheduler.step(float(validation["val_field_loss"]))
                write_csv(run / "history_steps.csv", history_rows, HISTORY_FIELDS)
                write_csv(run / "optimizer_steps.csv", optimizer_rows, STEP_FIELDS)
                selection = select_checkpoint(
                    run, model_name, reference,
                    reference_validation=reference_validation,
                    dense_best_validation=dense_best_validation,
                    dense_best_step=dense_best_step,
                )
                saved_epoch = epoch_index + 1 if epoch_boundary else epoch_index
                saved_offset = 0 if epoch_boundary else batch_offset
                saved_route = (
                    EpochRoute(saved_epoch, len(field_train.bags), len(calibration_train))
                    if epoch_boundary else route
                )
                save_last(
                    last_path, model, model_name, cfg, run_config, optimizer, scaler, scheduler,
                    global_step, attempt_step, saved_epoch, saved_offset, saved_route,
                    history_rows, optimizer_rows, dense_best_rank,
                    dense_best_validation, dense_best_step,
                    reference_validation, initial_validation, device,
                )
                write_json(run / "run_info.json", {
                    "status": "running",
                    "optimizer_step": global_step,
                    "epochs_equivalent": global_step / steps_per_epoch,
                    "steps_per_epoch": steps_per_epoch,
                    "history_points": len(history_rows),
                    "optimizer_rows": len(optimizer_rows),
                    "best_dense_step": dense_best_step,
                    "selected_source": selection["selected_source"],
                    "selected_step": selection["step"],
                    "selected_threshold": selection["threshold"],
                    "initialization_checkpoint": str(initialization_checkpoint),
                    "init_iter8": bool(args.init_iter8),
                    "stem_only": bool(args.stem_only),
                    "trainable_parameters": sum(
                        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
                    ),
                    "synthetic_train_patches": len(train_data),
                    "field_train_bags": len(field_train.bags),
                    "field_val_bags": len(field_val.bags),
                    "field_records_by_split": {
                        split: sum(record["split"] == split for record in field_cache.records)
                        for split in ("train", "val", "test")
                    },
                })
            elif epoch_boundary:
                # Default 900-step runs also record at each 150-step scheduler epoch.
                validation = metric_context.evaluate(model, cfg, device, args.batch_size)
                scheduler.step(float(validation["val_field_loss"]))
                model.train()
                saved_epoch = epoch_index + 1
                saved_route = EpochRoute(saved_epoch, len(field_train.bags), len(calibration_train))
                save_last(
                    last_path, model, model_name, cfg, run_config, optimizer, scaler, scheduler,
                    global_step, attempt_step, saved_epoch, 0, saved_route,
                    history_rows, optimizer_rows, dense_best_rank,
                    dense_best_validation, dense_best_step,
                    reference_validation, initial_validation, device,
                )

        del train_loader
        if global_step >= args.steps:
            break
        epoch_index += 1
        batch_offset = 0
        route = EpochRoute(epoch_index, len(field_train.bags), len(calibration_train))

    selection = select_checkpoint(
        run, model_name, reference,
        reference_validation=reference_validation,
        dense_best_validation=dense_best_validation,
        dense_best_step=dense_best_step,
    )
    write_json(run / "run_info.json", {
        "status": "complete",
        "optimizer_steps": global_step,
        "epochs_equivalent": global_step / steps_per_epoch,
        "steps_per_epoch": steps_per_epoch,
        "record_points": len(history_rows),
        "optimizer_rows": len(optimizer_rows),
        "dense_best_step": dense_best_step,
        "selected_source": selection["selected_source"],
        "selected_step": selection["step"],
        "selected_threshold": selection["threshold"],
        "initialization_checkpoint": str(initialization_checkpoint),
        "init_iter8": bool(args.init_iter8),
        "stem_only": bool(args.stem_only),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "field_records_by_split": {
            split: sum(record["split"] == split for record in field_cache.records)
            for split in ("train", "val", "test")
        },
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "model_weight_tensor_bytes": sum(
            tensor.numel() * tensor.element_size() for tensor in model.state_dict().values()
        ),
        "best_checkpoint_bytes": best_path.stat().st_size,
        "best_dense_checkpoint_bytes": best_dense_path.stat().st_size,
        "last_checkpoint_bytes": last_path.stat().st_size,
        "full_route_interpretation": (
            f"Iter8 {model_name} route replayed for the requested actual optimizer updates; "
            f"Iter8 {model_name} reference is an independent val-only fallback, "
            "not a trajectory point"
        ),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
