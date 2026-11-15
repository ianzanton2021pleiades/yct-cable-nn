"""Group-wise dataset and labels for field cable response detection."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


TASKS = ("joint", "terminal", "anomaly_point", "anomaly_range")
TASK_INDEX = {name: index for index, name in enumerate(TASKS)}
POINT_SIGMA_M = 2.0
BOUNDARY_SIDE_M = 2.0


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _meta_from_npz(value: np.ndarray) -> dict:
    raw = value.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(str(raw))


def _required_ids(owner_id: str, record: dict) -> list[str]:
    values = record.get("required_measurement_ids")
    if values is None:
        return [owner_id]
    if isinstance(values, str):
        return [values]
    return [str(value) for value in values]


def _context_mask(
    owner_id: str,
    record: dict,
    measurement_ids: list[str | None],
    valid: np.ndarray,
    phase_mask: np.ndarray,
) -> np.ndarray | None:
    """Return locations where the owner and required comparison inputs are valid."""
    slot_by_id = {str(value): index for index, value in enumerate(measurement_ids) if value}
    required_ids = _required_ids(owner_id, record)
    if owner_id not in slot_by_id or not phase_mask[slot_by_id[owner_id]]:
        return None
    required_slots = []
    for measurement_id in required_ids:
        slot = slot_by_id.get(measurement_id)
        if slot is None or not phase_mask[slot]:
            return None
        required_slots.append(slot)

    minimum = int(record.get("min_context", len(required_slots)))
    available_count = (valid & phase_mask[:, None]).sum(axis=0)
    context = valid[required_slots].all(axis=0)
    context &= available_count >= minimum
    return context


def _add_positive(
    target: torch.Tensor,
    supervision_mask: torch.Tensor,
    phase_index: int,
    task_index: int,
    event: dict,
    distance_m: np.ndarray,
    context: np.ndarray,
) -> None:
    shape = event.get("shape")
    if shape == "start_boundary":
        return
    if "position_m" in event:
        center = float(event["position_m"])
        delta = distance_m - center
        support = np.abs(delta) <= 3.0 * POINT_SIGMA_M
        locations = np.flatnonzero(support & context)
        if locations.size == 0:
            return
        indices = torch.as_tensor(locations, dtype=torch.long)
        values = torch.as_tensor(
            np.exp(-0.5 * np.square(delta[locations] / POINT_SIGMA_M)),
            dtype=target.dtype,
        )
        current = target[phase_index, task_index, indices]
        target[phase_index, task_index, indices] = torch.maximum(current, values)
        supervision_mask[phase_index, task_index, indices] = True
        return

    if "start_m" in event and "end_m" in event:
        lo, hi = sorted((float(event["start_m"]), float(event["end_m"])))
        locations = np.flatnonzero((distance_m >= lo) & (distance_m <= hi) & context)
        if locations.size:
            indices = torch.as_tensor(locations, dtype=torch.long)
            target[phase_index, task_index, indices] = 1.0
            supervision_mask[phase_index, task_index, indices] = True


def _boundary_constraint(
    phase_index: int,
    center_m: float,
    distance_m: np.ndarray,
    context: np.ndarray,
) -> dict | None:
    center_index = int(np.argmin(np.abs(distance_m - center_m)))
    center = float(distance_m[center_index])
    left_index = int(np.searchsorted(distance_m, center - BOUNDARY_SIDE_M, side="left"))
    right_index = int(np.searchsorted(distance_m, center + BOUNDARY_SIDE_M, side="left"))
    if left_index >= center_index or right_index <= center_index:
        return None
    if not context[left_index:right_index].all():
        return None
    return {
        "phase_index": phase_index,
        "center_index": center_index,
        "left_index": left_index,
        "right_index": right_index,
    }


class FieldDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        case_groups: Iterable[str],
        training: bool = False,
        single_phase_probability: float = 0.5,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.training = training
        self.single_phase_probability = single_phase_probability
        manifest = _load_json(self.data_dir / "manifest.json")
        selected_case_groups = set(case_groups)
        self.groups = [
            group for group in manifest.get("groups", [])
            if group.get("case_group_id") in selected_case_groups
        ]
        labels_path = self.data_dir / "review" / "accepted_labels.json"
        labels = _load_json(labels_path)
        self.labels_by_measurement = {
            item["measurement_id"]: item
            for item in labels.get("measurements", [])
        }

    def __len__(self) -> int:
        return len(self.groups)

    def _drop_to_one_phase(
        self, x: np.ndarray, valid: np.ndarray, phase_mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.training or random.random() >= self.single_phase_probability:
            return x, valid, phase_mask
        available = np.flatnonzero(phase_mask & valid.any(axis=1))
        if available.size <= 1:
            return x, valid, phase_mask
        keep = int(random.choice(available.tolist()))
        selected = np.zeros_like(phase_mask, dtype=bool)
        selected[keep] = True
        phase_mask = selected
        valid = valid & phase_mask[:, None]
        x = x * phase_mask[:, None, None]
        return x, valid, phase_mask

    def __getitem__(self, index: int) -> dict:
        group = self.groups[index]
        cache_path = self.data_dir / group["file"]
        with np.load(cache_path, allow_pickle=False) as cache:
            x = cache["x"].astype(np.float32, copy=True)
            valid = cache["valid"].astype(bool, copy=True)
            phase_mask = cache["phase_mask"].astype(bool, copy=True)
            distance_m = cache["distance_m"].astype(np.float32, copy=True)
            meta = _meta_from_npz(cache["meta"])

        measurement_ids = list(meta["measurement_ids"])
        valid &= phase_mask[:, None]
        phase_mask &= valid.any(axis=1)
        valid &= phase_mask[:, None]
        x, valid, phase_mask = self._drop_to_one_phase(x, valid, phase_mask)
        x *= valid[:, None, :]

        phase_count, _, length = x.shape
        target = torch.zeros((phase_count, len(TASKS), length), dtype=torch.float32)
        supervision_mask = torch.zeros((phase_count, len(TASKS), length), dtype=torch.bool)
        boundary_constraints = []
        slot_by_id = {
            str(measurement_id): slot
            for slot, measurement_id in enumerate(measurement_ids)
            if measurement_id
        }

        for measurement_id, phase_index in slot_by_id.items():
            label = self.labels_by_measurement.get(measurement_id)
            if label is None or not phase_mask[phase_index]:
                continue
            label_context = _context_mask(
                measurement_id, label, measurement_ids, valid, phase_mask
            )
            if label_context is None:
                continue
            for task, ranges in (label.get("negative_ranges") or {}).items():
                task_index = TASK_INDEX.get(task)
                if task_index is None:
                    continue
                for lo, hi in ranges:
                    lo, hi = sorted((float(lo), float(hi)))
                    locations = np.flatnonzero(
                        (distance_m >= lo) & (distance_m <= hi) & label_context
                    )
                    indices = torch.as_tensor(locations, dtype=torch.long)
                    supervision_mask[phase_index, task_index, indices] = True

            for event in label.get("events") or []:
                task_index = TASK_INDEX.get(event.get("task"))
                if task_index is None:
                    continue
                context = _context_mask(
                    measurement_id, event, measurement_ids, valid, phase_mask
                )
                if context is None:
                    continue
                if event.get("shape") == "start_boundary":
                    center_m = event.get("start_m", event.get("position_m"))
                    if center_m is None or event.get("task") != "anomaly_range":
                        continue
                    constraint = _boundary_constraint(
                        phase_index,
                        float(center_m),
                        distance_m,
                        context & valid[phase_index],
                    )
                    if constraint is not None:
                        boundary_constraints.append(constraint)
                    continue
                _add_positive(
                    target,
                    supervision_mask,
                    phase_index,
                    task_index,
                    event,
                    distance_m,
                    context & valid[phase_index],
                )

        meta = dict(meta)
        meta.setdefault("comparison_group_id", group.get("comparison_group_id"))
        meta.setdefault("case_group_id", group.get("case_group_id"))
        events = []
        negative_ranges = {}
        for measurement_id in measurement_ids:
            if not measurement_id:
                continue
            label = self.labels_by_measurement.get(str(measurement_id))
            if label is None:
                continue
            negative_ranges[str(measurement_id)] = label.get("negative_ranges") or {}
            events.extend(
                {**event, "measurement_id": str(measurement_id)}
                for event in label.get("events") or []
            )
        meta["events"] = events
        meta["negative_ranges"] = negative_ranges
        meta["boundary_constraints"] = [
            {**constraint, "measurement_id": measurement_ids[constraint["phase_index"]]}
            for constraint in boundary_constraints
        ]

        return {
            "x": torch.from_numpy(x),
            "valid": torch.from_numpy(valid),
            "phase_mask": torch.from_numpy(phase_mask),
            "distance_m": torch.from_numpy(distance_m),
            "meta": meta,
            "target": target,
            "supervision_mask": supervision_mask,
            "boundary_constraints": boundary_constraints,
        }


def collate_groups(items: list[dict]) -> dict:
    batch_size = len(items)
    phase_count = max(item["x"].shape[0] for item in items)
    channels = items[0]["x"].shape[1]
    max_length = max(item["x"].shape[-1] for item in items)
    dtype = items[0]["x"].dtype

    x = torch.zeros((batch_size, phase_count, channels, max_length), dtype=dtype)
    valid = torch.zeros((batch_size, phase_count, max_length), dtype=torch.bool)
    phase_mask = torch.zeros((batch_size, phase_count), dtype=torch.bool)
    distance_m = torch.zeros((batch_size, max_length), dtype=items[0]["distance_m"].dtype)
    target = torch.zeros(
        (batch_size, phase_count, len(TASKS), max_length), dtype=items[0]["target"].dtype
    )
    supervision_mask = torch.zeros(
        (batch_size, phase_count, len(TASKS), max_length), dtype=torch.bool
    )
    boundary_constraints = []

    for batch_index, item in enumerate(items):
        phases = item["x"].shape[0]
        length = item["x"].shape[-1]
        x[batch_index, :phases, :, :length] = item["x"]
        valid[batch_index, :phases, :length] = item["valid"]
        phase_mask[batch_index, :phases] = item["phase_mask"]
        distance_m[batch_index, :length] = item["distance_m"]
        target[batch_index, :phases, :, :length] = item["target"]
        supervision_mask[batch_index, :phases, :, :length] = item["supervision_mask"]
        boundary_constraints.extend(
            {**constraint, "batch_index": batch_index}
            for constraint in item["boundary_constraints"]
        )

    return {
        "x": x,
        "valid": valid,
        "phase_mask": phase_mask,
        "distance_m": distance_m,
        "meta": [item["meta"] for item in items],
        "target": target,
        "supervision_mask": supervision_mask,
        "boundary_constraints": boundary_constraints,
    }
