"""Core-16 的两种原有变换；显式设置波速、距离和物理平滑宽度。

公式来源：CST_Reproduction/V1/cst_fdr_reproduction.py 的
algorithm1_ifft / algorithm2_step_integral。没有新增第三种响应算法。
"""
from __future__ import annotations

import math

import numpy as np


def moving_mean(values: np.ndarray, points: int) -> np.ndarray:
    """MATLAB movmean 的 shrink 端点及偶数窗定义。"""
    points = max(1, int(points))
    left, right = points // 2, (points - 1) // 2
    index = np.arange(values.size)
    lo = np.maximum(0, index - left)
    hi = np.minimum(values.size, index + right + 1)
    cumulative = np.r_[0.0, np.cumsum(values)]
    return (cumulative[hi] - cumulative[lo]) / (hi - lo)


def algorithm1(
    frequency_hz: np.ndarray,
    s11: np.ndarray,
    *,
    velocity_m_s: float,
    distance_max_m: float,
    fft_step_hz: float | None = None,
) -> dict[str, np.ndarray | float]:
    """复数 S11、DC 外推、共轭镜像、Hann、IFFT、积分阶跃。

    对数原始频率的最小间隔不能铺满 1 GHz。这里按分析距离指定
    均匀重采样步进；重采样不增加原测量信息，原始大间隙另行报告。
    step_rho 是带限累计响应，step_raw 保留旧版 dt 因子口径。
    """
    f = np.asarray(frequency_hz, dtype=float)[1:]
    gamma = np.asarray(s11, dtype=complex)[1:]
    df = float(fft_step_hz or (velocity_m_s / (4.0 * distance_max_m)))
    steps = int(np.floor(f[-1] / df))
    n = 2 * steps + 1
    grid = np.arange(steps + 1) * df
    uniform = np.interp(grid, f, gamma.real) + 1j * np.interp(grid, f, gamma.imag)
    spectrum = np.zeros(n, dtype=complex)
    spectrum[steps + 1:] = uniform[1:]
    spectrum[:steps] = np.conj(uniform[1:][::-1])
    dc_abs = 2.0 * abs(uniform[1]) - abs(uniform[2])
    dc_phase = 2.0 * np.angle(uniform[1]) - np.angle(uniform[2])
    spectrum[steps] = dc_abs * np.exp(1j * dc_phase)
    spectrum *= np.hanning(n)
    impulse = np.fft.ifft(np.r_[spectrum[steps:], spectrum[:steps]]).real
    dt = 1.0 / (df * n)
    distance = np.arange(n) * dt * velocity_m_s / 2.0
    step_rho = np.cumsum(impulse)
    keep = distance <= distance_max_m
    return {
        "distance_m": distance[keep],
        "impulse_raw": impulse[keep],
        "step_raw": (step_rho * dt)[keep],
        "step_rho": step_rho[keep],
        "fft_step_hz": df,
        "period_distance_m": velocity_m_s / (2.0 * df),
        "native_max_gap_hz": float(np.max(np.diff(f))),
        "dt_s": dt,
    }


def algorithm2(
    frequency_hz: np.ndarray,
    s11: np.ndarray,
    *,
    velocity_m_s: float,
    distance_max_m: float,
    distance_step_m: float,
    step_smoothing_m: float,
    impulse_smoothing_m: float,
    chunk_size: int = 64,
) -> dict[str, np.ndarray | float]:
    """原 Algorithm 2：只取实部，去均值，余弦积分，阶跃差分。

    保留 10 V 和 c*6.5 的原始幅值口径。没有末端重构或衰减补偿；
    平滑窗按米定义，避免长电缆把固定 50 点变成数十米平滑。
    """
    f = np.asarray(frequency_hz, dtype=float)[1:]
    real = np.asarray(s11, dtype=complex).real[1:]
    points = int(math.ceil(distance_max_m / distance_step_m)) + 1
    time_s = np.linspace(1.0e-9, 2.0 * distance_max_m / velocity_m_s, points)
    distance = time_s * velocity_m_s / 2.0
    omega = 2.0 * np.pi * f
    loss = (real - np.mean(real)) / omega
    trapezoid = (loss[:-1] + loss[1:]) / 2.0
    step = np.empty(points)
    for start in range(0, points, chunk_size):
        times = time_s[start:start + chunk_size]
        delta = np.cos(omega[:-1, None] * times) - np.cos(omega[1:, None] * times)
        step[start:start + chunk_size] = np.sum(
            (20.0 / np.pi) * trapezoid[:, None] * delta / times, axis=0
        )
    impulse = np.diff(step) / (np.diff(time_s) * 3.0e8 * 6.5)
    dx = float(distance[1] - distance[0])
    return {
        "distance_m": distance,
        "distance_impulse_m": distance[1:],
        "step_raw": step,
        "impulse_raw": impulse,
        "step_smoothed": moving_mean(step, round(step_smoothing_m / dx)),
        "impulse_smoothed": moving_mean(impulse, round(impulse_smoothing_m / dx)),
        "distance_step_m": dx,
        "step_smoothing_points": max(1, round(step_smoothing_m / dx)),
        "impulse_smoothing_points": max(1, round(impulse_smoothing_m / dx)),
    }
