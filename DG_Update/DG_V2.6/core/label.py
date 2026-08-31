"""
label.py — 距离域置信度标签构造

在固定距离网格上生成缺陷/末端的置信度标签向量。
模型学习预测这条曲线，置信度高的地方 = 缺陷或末端。

用法:
    from core.label import build_label_vector
    label = build_label_vector(defect_positions=[43.0], severities=[0.6],
                               end_pos=74.0, grid=grid, sigma=1.0)
"""
from __future__ import annotations

import numpy as np
from typing import List, Optional


def build_label_vector(
    defect_positions: List[float],
    severities: List[float],
    end_pos: float,
    grid: np.ndarray,
    sigma_defect: float = 1.2,
    sigma_end: float = 1.5,
    end_amplitude: float = 1.0,
    defect_scale: float = 0.8,
    joint_positions: Optional[List[float]] = None,
    sigma_joint: float = 0.8,
    joint_amplitude: float = 0.6,
) -> np.ndarray:
    """
    在固定距离网格上生成距离域置信度标签。

    每个缺陷/末端/接头在其精确位置处放一个高斯峰。
    末端峰幅值高（始终为正事件），缺陷峰幅值 ∝ 严重程度，
    接头峰为窄高斯（点状反射）。

    Args:
        defect_positions: 缺陷位置列表 (m)，可为空列表（无缺陷电缆）
        severities: 缺陷严重度列表 (0~1)，与 defect_positions 一一对应
        end_pos: 末端（开路）位置 (m)
        grid: 固定距离网格 (m)
        sigma_defect: 缺陷高斯峰的标准差 (m)，控制空间分辨率
        sigma_end: 末端高斯峰的标准差 (m)，末端峰稍宽以反映开路反射特征
        end_amplitude: 末端峰值高度（归一化到1.0）
        defect_scale: 缺陷峰值缩放因子，最终幅值 = severity × defect_scale
        joint_positions: 接头/BNC连接器位置列表 (m)，可为 None
        sigma_joint: 接头高斯峰标准差 (m)，更窄以反映点状反射
        joint_amplitude: 接头峰值高度

    Returns:
        label: 形状 [n_points] 的置信度向量，值域 [0, ~1]
    """
    label = np.zeros(len(grid), dtype=np.float64)

    # 末端高斯峰（开路反射 — 始终存在）
    end_peak = end_amplitude * np.exp(-0.5 * ((grid - end_pos) / sigma_end) ** 2)
    label = np.maximum(label, end_peak)

    # 缺陷高斯峰（每个缺陷一个）
    for pos, sev in zip(defect_positions, severities):
        amp = sev * defect_scale
        peak = amp * np.exp(-0.5 * ((grid - pos) / sigma_defect) ** 2)
        label = np.maximum(label, peak)

    # 接头/BNC高斯峰（点状反射，更窄的sigma）
    if joint_positions:
        for pos in joint_positions:
            peak = joint_amplitude * np.exp(-0.5 * ((grid - pos) / sigma_joint) ** 2)
            label = np.maximum(label, peak)

    return label
