"""
metrics.py — 电缆缺陷检测评估指标

基于峰匹配的指标:
- 峰定位误差 (m)
- 召回率 / 精度
- 虚警率 (FAR)
- 逐点 AUC
- 末端召回率
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy.signal import find_peaks


@dataclass
class SampleMetrics:
    """单样本指标"""
    recall: float = 0.0
    precision: float = 0.0
    far: float = 0.0          # 虚警峰数
    loc_error_mean: float = 0.0
    loc_error_max: float = 0.0
    n_label_peaks: int = 0
    n_pred_peaks: int = 0
    end_recalled: bool = False


@dataclass
class AggregateMetrics:
    """汇总指标"""
    recall: float = 0.0
    precision: float = 0.0
    far: float = 0.0
    loc_error_mean: float = 0.0
    loc_error_median: float = 0.0
    loc_error_max: float = 0.0
    auc: float = 0.0
    end_recall: float = 0.0


def compute_sample_metrics(
    pred: np.ndarray,
    label: np.ndarray,
    grid: np.ndarray,
    peak_height: float = 0.2,
    peak_distance: int = 4,
    peak_prominence: float = 0.1,
    tolerance_m: float = 2.0,
    end_pos: float | None = None,
) -> SampleMetrics:
    """
    计算单样本的峰匹配指标。

    Args:
        pred: 预测置信度 [n_points]
        label: 标签 [n_points]
        grid: 距离网格 [n_points] (m)
        peak_height: find_peaks 高度阈值
        peak_distance: 最小峰距 (格)
        peak_prominence: find_peaks 突出度
        tolerance_m: 匹配容差 (m)
        end_pos: 末端位置 (m)，用于末端召回
    """
    dd = grid[1] - grid[0] if len(grid) > 1 else 0.5
    tol_grid = int(round(tolerance_m / dd))

    # 找峰
    label_peaks, _ = find_peaks(label, height=peak_height,
                                distance=peak_distance,
                                prominence=peak_prominence)
    pred_peaks, _ = find_peaks(pred, height=peak_height,
                               distance=peak_distance,
                               prominence=peak_prominence)

    m = SampleMetrics()
    m.n_label_peaks = len(label_peaks)
    m.n_pred_peaks = len(pred_peaks)

    if len(label_peaks) == 0 and len(pred_peaks) == 0:
        m.recall = 1.0
        m.precision = 1.0
        return m

    if len(label_peaks) == 0:
        m.precision = 0.0
        m.far = len(pred_peaks)
        return m

    if len(pred_peaks) == 0:
        m.recall = 0.0
        return m

    # 峰匹配：对每个 label 峰找最近的 pred 峰
    matched_label = set()
    matched_pred = set()
    errors = []

    for li, lp in enumerate(label_peaks):
        dists = np.abs(pred_peaks - lp)
        closest = np.argmin(dists)
        if dists[closest] <= tol_grid:
            matched_label.add(li)
            matched_pred.add(closest)
            errors.append(abs(grid[pred_peaks[closest]] - grid[lp]))

    m.recall = len(matched_label) / len(label_peaks)
    m.precision = len(matched_pred) / len(pred_peaks) if len(pred_peaks) > 0 else 0.0
    m.far = len(pred_peaks) - len(matched_pred)

    if errors:
        m.loc_error_mean = float(np.mean(errors))
        m.loc_error_max = float(np.max(errors))

    # 末端召回
    if end_pos is not None:
        end_idx = np.argmin(np.abs(grid - end_pos))
        # 检查末端附近是否有 pred 峰
        for pp in pred_peaks:
            if abs(pp - end_idx) <= tol_grid:
                m.end_recalled = True
                break

    return m


def compute_auc(pred: np.ndarray, label: np.ndarray) -> float:
    """逐点 ROC-AUC（简化版）"""
    from sklearn.metrics import roc_auc_score
    # 二值化标签 (阈值 0.3)
    label_bin = (label > 0.3).astype(int)
    if label_bin.sum() == 0 or label_bin.sum() == len(label_bin):
        return 0.5
    try:
        return float(roc_auc_score(label_bin, pred))
    except ValueError:
        return 0.5


def aggregate_metrics(
    all_metrics: List[SampleMetrics],
    all_preds: List[np.ndarray] | None = None,
    all_labels: List[np.ndarray] | None = None,
    end_positions: List[float] | None = None,
    grid: np.ndarray | None = None,
) -> AggregateMetrics:
    """汇总多样本指标"""
    agg = AggregateMetrics()
    if not all_metrics:
        return agg

    recalls = [m.recall for m in all_metrics]
    precisions = [m.precision for m in all_metrics]
    fars = [m.far for m in all_metrics]

    agg.recall = float(np.mean(recalls))
    agg.precision = float(np.mean(precisions))
    agg.far = float(np.mean(fars))

    # 定位误差（只从有误差的样本取）
    errors = [m.loc_error_mean for m in all_metrics if m.loc_error_mean > 0]
    if errors:
        agg.loc_error_mean = float(np.mean(errors))
        agg.loc_error_median = float(np.median(errors))

    max_errors = [m.loc_error_max for m in all_metrics if m.loc_error_max > 0]
    if max_errors:
        agg.loc_error_max = float(np.max(max_errors))

    # AUC
    if all_preds and all_labels:
        preds_flat = np.concatenate(all_preds)
        labels_flat = np.concatenate(all_labels)
        agg.auc = compute_auc(preds_flat, labels_flat)

    # 末端召回
    if end_positions is not None and grid is not None:
        end_recalled_count = sum(1 for m in all_metrics if m.end_recalled)
        agg.end_recall = end_recalled_count / len(all_metrics)

    return agg
