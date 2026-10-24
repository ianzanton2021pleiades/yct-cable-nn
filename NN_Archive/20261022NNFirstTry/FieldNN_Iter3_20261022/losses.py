"""False-positive-conscious losses for event, interval, and terminal tracks."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gaussian_focal_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """CenterNet-style focal loss normalized by each sample's positive-center count."""
    logits = logits.float()
    targets = targets.float()
    valid = targets >= 0.0
    targets = targets.clamp(0.0, 1.0)
    probability = logits.sigmoid()
    positive = (targets >= 0.99) & valid
    negative_weight = (1.0 - targets).pow(4) * valid

    positive_loss = -F.logsigmoid(logits) * (1.0 - probability).pow(2) * positive
    negative_loss = -F.logsigmoid(-logits) * probability.pow(2) * negative_weight

    center_count = positive.sum(dim=-1).clamp_min(1)
    per_sample = (positive_loss.sum(dim=-1) + negative_loss.sum(dim=-1)) / center_count
    return per_sample.mean()


def interval_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Full-grid weighted BCE plus Dice on curves containing an interval."""
    logits = logits.float()
    targets = targets.float()
    valid = targets >= 0.0
    targets = targets.clamp(0.0, 1.0)
    positive_weight = logits.new_tensor(2.0)
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=positive_weight, reduction="none"
    )
    valid_count = valid.sum(dim=-1).clamp_min(1)
    bce_by_sample = (bce * valid).sum(dim=-1) / valid_count

    valid_targets = targets * valid
    has_interval = valid_targets.sum(dim=-1) > 0
    if has_interval.any():
        probability = logits.sigmoid() * valid
        intersection = (probability * valid_targets).sum(dim=-1)
        denominator = probability.sum(dim=-1) + valid_targets.sum(dim=-1)
        dice_by_sample = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
        bce_by_sample = bce_by_sample + 0.5 * dice_by_sample * has_interval
    return bce_by_sample.mean()


def compute_losses(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return one scalar loss per output head; logits/targets are ``[B, 3, L]``."""
    if logits.shape != targets.shape or logits.ndim != 3 or logits.shape[1] != 3:
        raise ValueError("logits and targets must have matching [batch, 3, length] shapes")
    return {
        "event": gaussian_focal_loss(logits[:, 0], targets[:, 0]),
        "interval": interval_loss(logits[:, 1], targets[:, 1]),
        "terminal": gaussian_focal_loss(logits[:, 2], targets[:, 2]),
    }


def total_loss(components: dict[str, torch.Tensor]) -> torch.Tensor:
    return components["event"] + components["interval"] + components["terminal"]
