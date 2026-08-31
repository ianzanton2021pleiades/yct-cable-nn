"""
losses.py — 电缆缺陷检测损失函数

组合损失: FocalBCEWithLogits + PeakLocalization
处理极度稀疏正样本（2400 点中仅 ~5-15 个高值）。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_bce_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Focal BCE with logits loss.

    focal_weight = labels * (1-p)^γ + (1-labels) * p^γ
    loss = (focal_weight * BCE).mean()
    """
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    focal_weight = labels * (1 - p) ** gamma + (1 - labels) * p ** gamma
    return (focal_weight * ce).mean()


def peak_localization_loss(
    probs: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    峰定位损失：对高标签值区域施加更强的 L1 惩罚。
    weight = labels^2，使峰位误差主导损失。
    """
    weight = labels ** 2
    w_sum = weight.sum().clamp(min=1.0)
    return (weight * (probs - labels).abs()).sum() / w_sum


class CombinedLoss(torch.nn.Module):
    """
    Loss = α · FocalBCE(logits, labels) + β · PeakLoc(probs, labels)
    """

    def __init__(self, focal_gamma: float = 2.0,
                 alpha: float = 1.0, beta: float = 0.3):
        super().__init__()
        self.gamma = focal_gamma
        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        logits: torch.Tensor,
        probs: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        l_focal = focal_bce_with_logits(logits, labels, self.gamma)
        l_peak = peak_localization_loss(probs, labels)
        return self.alpha * l_focal + self.beta * l_peak


# ═══════════════════════════════════════════════
# V2 损失函数
# ═══════════════════════════════════════════════


def dice_loss(
    probs: torch.Tensor,
    labels: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Soft Dice Loss -- 直接优化预测与标签的重叠度。
    天然适合稀疏正样本场景。
    """
    numerator = 2.0 * (probs * labels).sum(dim=-1) + smooth
    denominator = (probs * probs).sum(dim=-1) + (labels * labels).sum(dim=-1) + smooth
    return 1.0 - (numerator / denominator).mean()


def smoothness_loss(probs: torch.Tensor) -> torch.Tensor:
    """
    平滑度正则 -- 惩罚预测曲线的一阶差分。
    抑制预测中的随机毛刺，降低 FAR。
    """
    diff = probs[:, 1:] - probs[:, :-1]
    return diff.abs().mean()


class V3CombinedLoss(torch.nn.Module):
    """
    V3 组合损失 (CableFormer 最终版):
    Loss = α · FocalBCE(γ=2.0) + β · PeakLoc + γ · Smoothness

    设计思路:
    - FocalBCE(γ=2) 是 V1 高召回率的核心驱动力
    - PeakLoc 强化峰位精度
    - 轻量 Smoothness (0.05) 保留 V2 的低 FAR 优势
    - 去掉 Dice Loss (过度抑制小峰 → 漏检)
    """

    def __init__(
        self,
        focal_gamma: float = 2.0,
        alpha: float = 1.0,
        beta: float = 0.3,
        smooth_weight: float = 0.05,
    ):
        super().__init__()
        self.gamma = focal_gamma
        self.alpha = alpha
        self.beta = beta
        self.w_smooth = smooth_weight

    def forward(
        self,
        logits: torch.Tensor,
        probs: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        l_focal = focal_bce_with_logits(logits, labels, self.gamma)
        l_peak = peak_localization_loss(probs, labels)
        l_smooth = smoothness_loss(probs)
        return self.alpha * l_focal + self.beta * l_peak + self.w_smooth * l_smooth


class V2CombinedLoss(torch.nn.Module):
    """
    V2 组合损失:
    Loss = w_dice * DiceLoss + w_focal * FocalBCE(gamma) + w_smooth * Smoothness
    """

    def __init__(
        self,
        focal_gamma: float = 1.0,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        smooth_weight: float = 0.1,
    ):
        super().__init__()
        self.gamma = focal_gamma
        self.w_dice = dice_weight
        self.w_focal = focal_weight
        self.w_smooth = smooth_weight

    def forward(
        self,
        logits: torch.Tensor,
        probs: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        l_dice = dice_loss(probs, labels)
        l_focal = focal_bce_with_logits(logits, labels, self.gamma)
        l_smooth = smoothness_loss(probs)
        return self.w_dice * l_dice + self.w_focal * l_focal + self.w_smooth * l_smooth
