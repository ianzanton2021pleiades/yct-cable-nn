"""Supervision losses (identical maths to the CMDC baseline) plus TDR physics residuals.

Supervision
    The three heatmap terms are byte-for-byte the baseline definitions so that a
    validation loss from this code is directly comparable with the published
    50/100-epoch curves.

Physics
    FDR measures the complex reflection coefficient at the cable head.  After the
    standard IFFT the observed impulse response is, to first order,

        h(d) = a(d) * g(d)          g = measurement pulse, width set by the band
        s(d) = integral_0^d a(x) dx  step response, i.e. the running sum of a

    with ``a(d)`` the signed reflection-coefficient profile of the cable.  The
    model therefore emits ``a`` explicitly and four residuals are evaluated
    against the observed trace:

    reconstruction  cosine between a*g and the observed normalised impulse
    support         predicted event mass must sit where the observed envelope is
    ordering        the terminal cannot precede the last reflection
    sparsity        reflectivity is a sparse spike train, so |a| is L1-penalised

    All residuals are scale-free because the trace normalisation inside the
    feature builder is not invertible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
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
    """Penalise the strongest event probabilities in unlabelled regions."""
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


@dataclass
class PhysicsWeights:
    reconstruction: float = 0.10
    step: float = 0.03
    support: float = 0.05
    ordering: float = 0.02
    sparse: float = 0.002
    consistency: float = 0.05

    def as_dict(self) -> dict[str, float]:
        return {
            "reconstruction": self.reconstruction,
            "step": self.step,
            "support": self.support,
            "ordering": self.ordering,
            "sparse": self.sparse,
            "consistency": self.consistency,
        }

    @property
    def active(self) -> bool:
        return any(value > 0.0 for value in self.as_dict().values())


class PhysicsResidual(nn.Module):
    """Forward-model residuals between the predicted reflectivity and the trace."""

    def __init__(
        self,
        distance_max_m: float = 3250.0,
        points: int = 8192,
        pulse_sigma_m: float = 5.0,
        near_exclusion_m: float = 30.0,
        terminal_margin_m: float = 25.0,
        weights: PhysicsWeights | None = None,
    ) -> None:
        super().__init__()
        self.weights = weights or PhysicsWeights()
        spacing = float(distance_max_m) / max(int(points) - 1, 1)
        self.spacing_m = spacing
        self.terminal_margin_m = float(terminal_margin_m)
        half = max(1, int(round(4.0 * pulse_sigma_m / spacing)))
        offsets = torch.arange(-half, half + 1, dtype=torch.float32) * spacing
        kernel = torch.exp(-0.5 * (offsets / float(pulse_sigma_m)) ** 2)
        kernel = kernel / kernel.max()
        self.register_buffer("kernel", kernel.view(1, 1, -1))
        self.kernel_half = half
        distance = torch.linspace(0.0, float(distance_max_m), int(points))
        self.register_buffer("distance", distance)
        self.register_buffer("valid", (distance >= near_exclusion_m).float().view(1, 1, -1))
        window = max(3, int(round(30.0 / spacing)))
        self.register_buffer("env_window", torch.ones(1, 1, window) / float(window))
        self.window = window

    def _smooth(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.window // 2
        return F.conv1d(F.pad(x, (pad, pad - 1 + self.window % 2), mode="replicate"), self.env_window)

    def _cosine(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(dim=-1, keepdim=True)
        y = y - y.mean(dim=-1, keepdim=True)
        numerator = (x * y).sum(dim=-1)
        return numerator / (x.norm(dim=-1) * y.norm(dim=-1) + 1e-6)

    def forward(
        self, output: torch.Tensor, features: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        event_logit = output[:, 0:1]
        terminal_logit = output[:, 2:3]
        amplitude = 16.0 * torch.tanh(output[:, 3:4])

        padded = F.pad(amplitude, (self.kernel_half, self.kernel_half), mode="replicate")
        impulse_hat = F.conv1d(padded, self.kernel)
        observed = 4.0 * features[:, 0:1]

        valid = self.valid
        weight_valid = valid.expand_as(observed)
        count = weight_valid.sum().clamp_min(1.0)

        reconstruction = 1.0 - self._cosine(impulse_hat * valid, observed * valid).mean()

        state_hat = torch.cumsum(
            (amplitude - amplitude.mean(dim=-1, keepdim=True)) * self.spacing_m, dim=-1
        )
        state_hat = state_hat - self._smooth(state_hat)
        observed_step = 4.0 * features[:, 1:2]
        observed_step = observed_step - self._smooth(observed_step)
        step_residual = 1.0 - self._cosine(state_hat * valid, observed_step * valid).mean()

        envelope = torch.sqrt(self._smooth(observed.pow(2)).clamp_min(1e-8))
        envelope = envelope / envelope.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        event_probability = event_logit.sigmoid() * valid
        ratio = (event_probability * envelope).sum(dim=-1) / event_probability.sum(dim=-1).clamp_min(1e-6)
        support = (1.0 - ratio / (1.0 + ratio)).mean()

        temperature = 0.1
        weights = torch.softmax((event_logit * valid).flatten(1) / temperature, dim=-1)
        last_event = (weights * self.distance.view(1, -1)).sum(dim=-1, keepdim=True)
        terminal_probability = terminal_logit.sigmoid() * valid
        terminal_position = (terminal_probability.flatten(1) * self.distance.view(1, -1)).sum(
            dim=-1, keepdim=True
        ) / terminal_probability.flatten(1).sum(dim=-1, keepdim=True).clamp_min(1e-6)
        ordering = F.relu(last_event + self.terminal_margin_m - terminal_position).mean() / 1000.0

        sparse = amplitude.abs().mean()

        magnitude = (amplitude.abs() * valid).flatten(1)
        magnitude = magnitude / magnitude.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        probability = event_probability.flatten(1)
        probability = probability / probability.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        consistency = 1.0 - (magnitude * probability).sum(dim=-1).mean()

        components = {
            "reconstruction": reconstruction,
            "step": step_residual,
            "support": support,
            "ordering": ordering,
            "sparse": sparse,
            "consistency": consistency,
        }
        weights_map = self.weights.as_dict()
        total = sum(weights_map[name] * value for name, value in components.items())
        return total, components
