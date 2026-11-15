"""Masked, task-balanced detection loss with optional start-boundary ordering."""
from __future__ import annotations

import torch
from torch.nn import functional as F


TASKS = ("joint", "terminal", "anomaly_point", "anomaly_range")
ANOMALY_RANGE_INDEX = TASKS.index("anomaly_range")


def masked_detection_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    supervision_mask: torch.Tensor,
    boundary_constraints: list[dict] | None = None,
) -> tuple[torch.Tensor, dict]:
    """Return task-balanced BCE loss and per-head supervision statistics.

    Soft targets are split into positive and negative BCE mass before balancing.
    Positions outside ``supervision_mask`` contribute to neither term.
    """
    target = target.to(dtype=logits.dtype)
    mask = supervision_mask.bool()
    log_positive = F.logsigmoid(logits)
    log_negative = F.logsigmoid(-logits)
    weighted_mask = mask.to(dtype=logits.dtype)

    positive_mass = (target * weighted_mask).sum(dim=(0, 1, 3))
    negative_mass = ((1.0 - target) * weighted_mask).sum(dim=(0, 1, 3))
    positive_term = -(target * log_positive * weighted_mask).sum(dim=(0, 1, 3))
    negative_term = -((1.0 - target) * log_negative * weighted_mask).sum(dim=(0, 1, 3))

    positive_loss = positive_term / positive_mass.clamp_min(1.0)
    negative_loss = negative_term / negative_mass.clamp_min(1.0)
    class_count = (positive_mass > 0).to(logits.dtype) + (negative_mass > 0).to(logits.dtype)
    task_losses = (positive_loss + negative_loss) / class_count.clamp_min(1.0)

    positive_counts = ((target >= 0.5) & mask).sum(dim=(0, 1, 3))
    negative_counts = ((target < 0.5) & mask).sum(dim=(0, 1, 3))
    supervised_counts = mask.sum(dim=(0, 1, 3))
    active_tasks = supervised_counts > 0

    boundary_constraints = boundary_constraints or []
    boundary_terms = []
    for constraint in boundary_constraints:
        batch_index = constraint["batch_index"]
        phase_index = constraint["phase_index"]
        center_index = constraint["center_index"]
        left_index = constraint["left_index"]
        right_index = constraint["right_index"]
        range_logits = logits[batch_index, phase_index, ANOMALY_RANGE_INDEX]
        left_mean = range_logits[left_index:center_index].mean()
        right_mean = range_logits[center_index:right_index].mean()
        boundary_terms.append(F.softplus(-(right_mean - left_mean)))

    loss_terms = [task_losses[index] for index in range(len(TASKS)) if active_tasks[index]]
    boundary_loss = torch.stack(boundary_terms).mean() if boundary_terms else None
    if boundary_loss is not None:
        loss_terms.append(boundary_loss)
    total_loss = torch.stack(loss_terms).mean() if loss_terms else logits.sum() * 0.0

    counts = torch.stack((positive_counts, negative_counts, supervised_counts)).detach().cpu().tolist()
    task_loss_values = task_losses.detach().cpu().tolist()
    active_values = active_tasks.detach().cpu().tolist()
    stats = {}
    for index, name in enumerate(TASKS):
        stats[name] = {
            "loss": float(task_loss_values[index]) if active_values[index] else None,
            "positive": int(counts[0][index]),
            "negative": int(counts[1][index]),
            "supervised": int(counts[2][index]),
        }

    boundary_loss_value = (
        float(boundary_loss.detach().cpu()) if boundary_loss is not None else None
    )
    boundary_count = len(boundary_terms)
    stats["boundary_loss"] = boundary_loss_value
    stats["boundary_count"] = boundary_count
    stats["total"] = {
        "loss": float(total_loss.detach().cpu()),
        "positive": int(sum(counts[0])),
        "negative": int(sum(counts[1])),
        "supervised": int(sum(counts[2])),
    }
    return total_loss, stats
