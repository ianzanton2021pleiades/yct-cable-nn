"""Weighted losses for synthetic, injected-field, and calibration-short samples.

Target layout is [batch, 5, length]:
  0 event target, 1 interval target, 2 terminal target,
  3 per-point supervision weight, 4 unknown-region mask for sparsity.
Targets below zero are ignored.  Synthetic samples use weight 1 wherever the
target is defined; injected-field samples use 1 for injected labels, 0.5 for
weak engineer labels, and 0 elsewhere.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _prepare(targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if targets.ndim != 3:
        raise ValueError("targets must have shape [batch, channels, length]")
    labels = targets[:, :3].float()
    if targets.shape[1] >= 5:
        weight = targets[:, 3].float()
        unknown = targets[:, 4].float()
    else:
        weight = (labels >= 0.0).float()
        unknown = torch.zeros_like(weight)
    return labels, weight, unknown


def weighted_gaussian_focal_loss(
    logits: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """CenterNet-style focal loss with per-point supervision weights."""
    logits = logits.float()
    valid = targets >= 0.0
    targets = targets.clamp(0.0, 1.0)
    probability = logits.sigmoid()
    positive = (targets >= 0.99) & valid
    positive_weight = positive.float() * weight
    negative_weight = (1.0 - targets).pow(4) * valid.float() * weight

    positive_loss = -F.logsigmoid(logits) * (1.0 - probability).pow(2) * positive_weight
    negative_loss = -F.logsigmoid(-logits) * probability.pow(2) * negative_weight

    center_count = positive_weight.sum(dim=-1).clamp_min(1.0)
    per_sample = (positive_loss.sum(dim=-1) + negative_loss.sum(dim=-1)) / center_count
    return per_sample.mean()


def weighted_interval_loss(
    logits: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Weighted BCE plus weighted Dice on curves containing an interval."""
    logits = logits.float()
    valid = targets >= 0.0
    targets = targets.clamp(0.0, 1.0)
    point_weight = valid.float() * weight
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=logits.new_tensor(2.0), reduction="none"
    )
    bce_by_sample = (bce * point_weight).sum(dim=-1) / point_weight.sum(dim=-1).clamp_min(1.0)

    weighted_targets = targets * point_weight
    has_interval = weighted_targets.sum(dim=-1) > 0
    if has_interval.any():
        probability = logits.sigmoid() * point_weight
        intersection = (probability * weighted_targets).sum(dim=-1)
        denominator = probability.sum(dim=-1) + weighted_targets.sum(dim=-1)
        dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
        bce_by_sample = bce_by_sample + 0.5 * dice * has_interval
    return bce_by_sample.mean()


def topk_unknown_sparsity(
    logits: torch.Tensor, unknown: torch.Tensor, k: int = 64
) -> torch.Tensor:
    """Penalize the strongest event probabilities in unknown regions.

    This is a weak positive-unlabelled prior: injected and engineer labels are
    known positives, but most of a measured trace is merely unknown.  Only the
    top-k responses are penalized so weak unlabelled events are not forced to
    zero everywhere.
    """
    probability = logits.float().sigmoid() * unknown
    if probability.shape[-1] <= k:
        return probability.mean()
    return torch.topk(probability, k, dim=-1).values.mean()


def compute_losses(
    logits: torch.Tensor, targets: torch.Tensor
) -> dict[str, torch.Tensor]:
    if logits.ndim != 3 or logits.shape[1] != 3:
        raise ValueError("logits must have shape [batch, 3, length]")
    labels, weight, unknown = _prepare(targets)
    if logits.shape[-1] != labels.shape[-1]:
        raise ValueError("logits and targets must share the same length")
    return {
        "event": weighted_gaussian_focal_loss(logits[:, 0], labels[:, 0], weight),
        "interval": weighted_interval_loss(logits[:, 1], labels[:, 1], weight),
        "terminal": weighted_gaussian_focal_loss(logits[:, 2], labels[:, 2], weight),
        "sparsity": topk_unknown_sparsity(logits[:, 0], unknown, k=64),
    }


def total_loss(components: dict[str, torch.Tensor], sparsity_weight: float = 0.02) -> torch.Tensor:
    return (
        components["event"]
        + components["interval"]
        + components["terminal"]
        + float(sparsity_weight) * components["sparsity"]
    )
