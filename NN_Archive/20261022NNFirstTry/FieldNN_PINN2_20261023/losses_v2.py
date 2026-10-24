"""Supervision terms (baseline maths) plus the PINN-v2 forward-model residuals.

What changed against PINN-v1
----------------------------
v1 explained the trace with a single 5 m pulse and a single 30 m envelope, so
the reconstruction residual could be satisfied by broad lumps that are not
reflections.  v2 evaluates the same forward model at three pulse widths
(3/8/20 m) and the step at two detrend widths (30/120 m); a reflectivity
profile only scores well if it is consistent with the observed trace at every
scale, which is what a real reflection coefficient profile is.

Two residuals are new.  ``mutual`` forces the event head and |a| to occupy the
same support in both directions, and ``align`` penalises the distance between
their soft peak positions.  Together they close the loophole where the event
head and the physics head each explain the trace in their own way.

``background`` is not a physics term.  The baseline focal loss divides by the
number of positive centres, so a uniform 3 % background probability already
costs ~0.5 and the cheapest way for the network to buy recall is to raise that
floor.  Taking the 128 strongest off-label probabilities and penalising them
directly is the hard-negative counterpart, and it is also what reduces the
candidate count on measured traces.
"""

from __future__ import annotations

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


def background_ohem(
    logits: torch.Tensor, targets: torch.Tensor, k: int = 128, margin: float = 0.05
) -> torch.Tensor:
    """Mean of the strongest event probabilities on points that are known quiet."""
    labels, weight, _ = _prepare(targets)
    quiet = (labels[:, 0] < margin) & (labels[:, 1] < 0.5) & (weight > 0.0)
    probability = logits[:, 0].float().sigmoid() * quiet.float()
    k = min(int(k), probability.shape[-1])
    return torch.topk(probability, k, dim=-1).values.mean()


def compute_losses(
    logits: torch.Tensor, targets: torch.Tensor
) -> dict[str, torch.Tensor]:
    if logits.ndim != 3 or logits.shape[1] != 3:
        raise ValueError("logits must have shape [batch, 3, length]")
    labels, weight, _ = _prepare(targets)
    if logits.shape[-1] != labels.shape[-1]:
        raise ValueError("logits and targets must share the same length")
    return {
        "event": weighted_gaussian_focal_loss(logits[:, 0], labels[:, 0], weight),
        "interval": weighted_interval_loss(logits[:, 1], labels[:, 1], weight),
        "terminal": weighted_gaussian_focal_loss(logits[:, 2], labels[:, 2], weight),
        "background": background_ohem(logits, targets),
    }


def total_loss(components: dict[str, torch.Tensor], background_weight: float = 0.10) -> torch.Tensor:
    return (
        components["event"]
        + components["interval"]
        + components["terminal"]
        + float(background_weight) * components["background"]
    )


def supervision_only(components: dict[str, torch.Tensor]) -> torch.Tensor:
    """Exactly the PINN-v1 three-term definition, for a like-for-like loss curve."""
    return components["event"] + components["interval"] + components["terminal"]


@dataclass
class PhysicsWeights:
    reconstruction: float = 0.10
    step: float = 0.03
    envelope: float = 0.04
    mutual: float = 0.05
    align: float = 0.02
    ordering: float = 0.02
    sparse: float = 0.002

    def as_dict(self) -> dict[str, float]:
        return {
            "reconstruction": self.reconstruction,
            "step": self.step,
            "envelope": self.envelope,
            "mutual": self.mutual,
            "align": self.align,
            "ordering": self.ordering,
            "sparse": self.sparse,
        }

    @property
    def active(self) -> bool:
        return any(value > 0.0 for value in self.as_dict().values())


class PhysicsResidual(nn.Module):
    """Multi-scale forward-model residuals between reflectivity and trace."""

    RECONSTRUCTION_SIGMA_M = (3.0, 8.0, 20.0)
    STEP_SMOOTH_M = (30.0, 120.0)

    def __init__(
        self,
        distance_max_m: float = 3250.0,
        points: int = 8192,
        near_exclusion_m: float = 30.0,
        terminal_margin_m: float = 25.0,
        weights: PhysicsWeights | None = None,
    ) -> None:
        super().__init__()
        self.weights = weights or PhysicsWeights()
        spacing = float(distance_max_m) / max(int(points) - 1, 1)
        self.spacing_m = spacing
        self.terminal_margin_m = float(terminal_margin_m)

        for index, sigma in enumerate(self.RECONSTRUCTION_SIGMA_M):
            half = max(1, int(round(4.0 * sigma / spacing)))
            offsets = torch.arange(-half, half + 1, dtype=torch.float32) * spacing
            kernel = torch.exp(-0.5 * (offsets / float(sigma)) ** 2)
            kernel = kernel / kernel.max()
            self.register_buffer(f"kernel_{index}", kernel.view(1, 1, -1))

        distance = torch.linspace(0.0, float(distance_max_m), int(points))
        self.register_buffer("distance", distance)
        self.register_buffer("valid", (distance >= near_exclusion_m).float().view(1, 1, -1))
        for index, width_m in enumerate(self.STEP_SMOOTH_M):
            window = max(3, int(round(width_m / spacing)))
            window += 1 - window % 2
            self.register_buffer(
                f"window_{index}", torch.ones(1, 1, window) / float(window)
            )
            setattr(self, f"window_size_{index}", window)

    def _smooth(self, x: torch.Tensor, index: int) -> torch.Tensor:
        window = getattr(self, f"window_size_{index}")
        pad = window // 2
        return F.conv1d(F.pad(x, (pad, pad), mode="replicate"), getattr(self, f"window_{index}"))

    @staticmethod
    def _cosine(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(dim=-1, keepdim=True)
        y = y - y.mean(dim=-1, keepdim=True)
        numerator = (x * y).sum(dim=-1)
        return numerator / (x.norm(dim=-1) * y.norm(dim=-1) + 1e-6)

    @staticmethod
    def _soft_position(weight_map: torch.Tensor, distance: torch.Tensor, temperature: float) -> torch.Tensor:
        distribution = torch.softmax(weight_map.flatten(1) / temperature, dim=-1)
        return (distribution * distance.view(1, -1)).sum(dim=-1, keepdim=True)

    def forward(
        self, output: torch.Tensor, features: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        event_logit = output[:, 0:1]
        terminal_logit = output[:, 2:3]
        amplitude = 16.0 * torch.tanh(output[:, 3:4])

        observed = 4.0 * features[:, 0:1]
        valid = self.valid

        reconstruction = observed.new_zeros(())
        for index in range(len(self.RECONSTRUCTION_SIGMA_M)):
            kernel = getattr(self, f"kernel_{index}")
            half = kernel.shape[-1] // 2
            impulse_hat = F.conv1d(F.pad(amplitude, (half, half), mode="replicate"), kernel)
            reconstruction = reconstruction + (
                1.0 - self._cosine(impulse_hat * valid, observed * valid).mean()
            ) / len(self.RECONSTRUCTION_SIGMA_M)

        centred = amplitude - amplitude.mean(dim=-1, keepdim=True)
        state_hat = torch.cumsum(centred * self.spacing_m, dim=-1)
        observed_step = 4.0 * features[:, 1:2]
        step_residual = observed.new_zeros(())
        for index in range(len(self.STEP_SMOOTH_M)):
            step_residual = step_residual + (
                1.0
                - self._cosine(
                    (state_hat - self._smooth(state_hat, index)) * valid,
                    (observed_step - self._smooth(observed_step, index)) * valid,
                ).mean()
            ) / len(self.STEP_SMOOTH_M)

        event_probability = event_logit.sigmoid() * valid
        magnitude = (amplitude.abs() * valid)
        magnitude_normalised = magnitude / magnitude.amax(dim=-1, keepdim=True).clamp_min(1e-6)

        envelope = torch.sqrt(self._smooth(observed.pow(2), 0).clamp_min(1e-8))
        envelope = envelope / envelope.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        ratio = (event_probability * envelope).sum(dim=-1) / event_probability.sum(dim=-1).clamp_min(1e-6)
        envelope_residual = (1.0 - ratio / (1.0 + ratio)).mean()

        event_mass = event_probability.sum(dim=-1).clamp_min(1e-6)
        magnitude_mass = magnitude.sum(dim=-1).clamp_min(1e-6)
        event_on_reflectivity = (event_probability * magnitude_normalised).sum(dim=-1) / event_mass
        reflectivity_on_event = (magnitude_normalised * event_probability).sum(dim=-1) / magnitude_mass
        mutual = (1.0 - 0.5 * (event_on_reflectivity + reflectivity_on_event)).mean()

        event_position = self._soft_position(event_probability, self.distance, 0.05)
        reflectivity_position = self._soft_position(magnitude_normalised, self.distance, 0.02)
        align = (event_position - reflectivity_position).abs().mean() / 1000.0

        last_event = self._soft_position(event_probability, self.distance, 0.1)
        terminal_probability = terminal_logit.sigmoid() * valid
        terminal_position = (
            (terminal_probability.flatten(1) * self.distance.view(1, -1)).sum(dim=-1, keepdim=True)
            / terminal_probability.flatten(1).sum(dim=-1, keepdim=True).clamp_min(1e-6)
        )
        ordering = F.relu(last_event + self.terminal_margin_m - terminal_position).mean() / 1000.0

        sparse = amplitude.abs().mean()

        components = {
            "reconstruction": reconstruction,
            "step": step_residual,
            "envelope": envelope_residual,
            "mutual": mutual,
            "align": align,
            "ordering": ordering,
            "sparse": sparse,
        }
        weights_map = self.weights.as_dict()
        total = sum(weights_map[name] * value for name, value in components.items())
        return total, components
