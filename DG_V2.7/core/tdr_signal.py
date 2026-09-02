"""
tdr_signal.py — TDR 信号处理共享库

严格移植自 Client 的 yct-TDR-2-GUI-multisamples-FreqSel_causal_fixed.pyw，
保证合成S11、真实S11、推理时Client传入的S11 三者走同一套IFFT代码。

移植规格详见: AgentsStorage/research/01_client_ifft_spec.md

用法:
    from core.tdr_signal import s11_to_responses, to_fixed_distance_grid
    distance, impulse, step, Z = s11_to_responses(freqs, S, epsr=2.23, window='hann')
    grid, imp_grid, step_grid = to_fixed_distance_grid(distance, impulse, step)
"""
from __future__ import annotations

import numpy as np
from typing import Tuple

# ═══════════════════════════════════════════════
# 全局常量（与 Client 一致）
# ═══════════════════════════════════════════════
SPEED_OF_LIGHT = 299_792_458.0  # m/s
Z0_REF = 50.0                   # 参考阻抗 Ω


# ═══════════════════════════════════════════════
# 核心函数（逐个移植自 Client）
# ═══════════════════════════════════════════════

def fft_shift(arr: np.ndarray, inverse: bool = False) -> np.ndarray:
    """
    自定义 FFT 顺序转换（半交换）。
    注意：对奇数长度数组，行为与 np.fft.ifftshift / fftshift 不同，
    不可用 numpy 内置函数替代。
    """
    n = len(arr)
    k = n // 2
    if inverse:
        return np.concatenate((arr[k:], arr[:k]))
    else:
        return np.concatenate((arr[-k:], arr[:-k]))


def apply_window(N: int, window_type: str = "rectangular") -> np.ndarray:
    """
    生成窗函数。窗作用于整个双边频谱（含DC和负频半部）。
    """
    if window_type == "hann":
        return np.hanning(N)
    elif window_type == "hamming":
        return np.hamming(N)
    elif window_type == "blackman":
        return np.blackman(N)
    elif window_type in ("rectangular", "none"):
        return np.ones(N)
    else:
        raise ValueError(f"Unknown window type: {window_type}")


def estimate_first_step(freqs: np.ndarray, quantile: int = 5) -> float:
    """
    鲁棒频步估计器：取去重排序后正频差的第5百分位数。
    含退化保护：若 firstStep < mean_df/10 则用 mean_df。
    """
    f = np.sort(np.unique(np.asarray(freqs, float)))
    if len(f) < 2:
        raise ValueError("频率点过少，无法估计步进")
    df = np.diff(f)
    df = df[df > 0]
    if len(df) < 2:
        return float((f[-1] - f[0]) / max(len(f) - 1, 1))
    first_step = float(np.percentile(df, quantile))
    mean_df = float(np.mean(df))
    if first_step < mean_df / 10:
        first_step = mean_df
    return first_step


def build_equally_spaced_spectrum(
    freqs: np.ndarray,
    S: np.ndarray,
    automatic_dc: bool = True,
    window_type: str = "rectangular",
    padding_bins: int = 0,
    df_quantile: int = 5,
    max_bins: int = 2_000_001,
) -> Tuple[np.ndarray, float, int, int]:
    """
    将（可能非均匀的）频域S11数据构建为等间距双边Hermitian频谱。

    Returns:
        spectrum: 复数频谱 [N], centered布局 [-fmax...0...+fmax]
        firstStep: 频率步长 (Hz)
        steps: 正频半轴 bin 数
        N: 总 bin 数（奇数）
    """
    freqs = np.asarray(freqs, float)
    S = np.asarray(S, complex)
    order = np.argsort(freqs)
    freqs_sorted = freqs[order]
    S_sorted = S[order]

    firstStep = estimate_first_step(freqs_sorted, quantile=df_quantile)
    fmax = freqs_sorted[-1]
    steps = int(np.floor(fmax / firstStep))
    N = 2 * steps + 1

    # max_bins 上限
    if N > max_bins:
        scale = N / max_bins
        firstStep *= scale
        steps = int(np.floor(fmax / firstStep))
        N = 2 * steps + 1

    # 正频半网格插值（实虚分别线性插值，端点钳位）
    f_lin = np.arange(0, steps + 1, dtype=float) * firstStep
    S_lin = (np.interp(f_lin, freqs_sorted, S_sorted.real)
             + 1j * np.interp(f_lin, freqs_sorted, S_sorted.imag))

    # Hermitian 装配
    spectrum = np.zeros(N, dtype=np.complex128)
    for i in range(1, steps + 1):
        spectrum[steps + i] = S_lin[i]
        spectrum[steps - i] = np.conj(S_lin[i])

    # DC 外推（仅 steps > 2 时）
    if automatic_dc and steps > 2:
        abs_dc = 2 * np.abs(spectrum[steps + 1]) - np.abs(spectrum[steps + 2])
        pha_dc = 2 * np.angle(spectrum[steps + 1]) - np.angle(spectrum[steps + 2])
        spectrum[steps] = abs_dc * np.exp(1j * pha_dc)
    else:
        spectrum[steps] = S_lin[0]

    # 窗函数（作用于整个双边谱）
    spectrum *= apply_window(N, window_type=window_type)

    # 零填充（对称插入DC两侧）
    if padding_bins > 0:
        left = spectrum[:steps]
        dc = spectrum[steps:steps + 1]
        right = spectrum[steps + 1:]
        pad = np.zeros(padding_bins, dtype=np.complex128)
        spectrum = np.concatenate([left, pad, dc, pad, right])
        N = spectrum.size

    return spectrum, firstStep, steps, N


def spectrum_to_time(
    spectrum: np.ndarray,
    firstStep: float,
    N: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    双边频谱 → 因果时域响应（IFFT）。
    np.fft.ifft 自带 1/N 归一化，不再额外除 N。

    Returns:
        td: 复数时域响应 [N]
        t: 时间轴 [N], 因果 t >= 0
    """
    spec = fft_shift(spectrum, inverse=True)  # centered → FFT order
    td = np.fft.ifft(spec)
    dt = 1.0 / (firstStep * N)
    t = np.arange(N) * dt
    return td, t


def compute_step_response(td: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    阶跃响应 = Re(cumsum(复数td) * dt)
    注意：先对复数 cumsum 再取实，不可交换。
    """
    dt = t[1] - t[0]
    return np.real(np.cumsum(td) * dt)


def compute_impedance(step_real: np.ndarray, t: np.ndarray, Z0: float = Z0_REF) -> np.ndarray:
    """
    从阶跃响应计算阻抗分布。
    因果轴下 baseline = 前 5% 采样点均值。
    """
    if np.any(t < 0):
        baseline = np.mean(step_real[t < 0])
    else:
        baseline = np.mean(step_real[:max(1, int(0.05 * len(t)))])
    gamma = np.clip(step_real - baseline, -0.9999, 0.9999)
    Z = Z0 * (1.0 + gamma) / (1.0 - gamma)
    return Z


# ═══════════════════════════════════════════════
# 便捷一站函数
# ═══════════════════════════════════════════════

def s11_to_responses(
    freqs: np.ndarray,
    S: np.ndarray,
    epsr: float = 2.23,
    window: str = "hann",
    automatic_dc: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    S11 频域数据 → 距离域脉冲响应 + 阶跃响应。

    Args:
        freqs: 频率轴 (Hz), 一维数组
        S: S11 复数数组，与 freqs 同长
        epsr: 相对介电常数（用于距离计算）
        window: 窗函数类型 ("hann"/"rectangular"/"hamming"/"blackman")
        automatic_dc: 是否自动外推 DC 值

    Returns:
        distance: 距离轴 (m), 单程 d = t * c / (2*sqrt(epsr))
        impulse: 脉冲响应（复数 IFFT 输出）
        step: 阶跃响应（Re(cumsum(impulse)*dt)）
        Z: 阻抗分布 (Ω)
    """
    spectrum, firstStep, steps, N = build_equally_spaced_spectrum(
        freqs, S, automatic_dc=automatic_dc, window_type=window
    )
    td, t = spectrum_to_time(spectrum, firstStep, N)
    impulse = td
    step = compute_step_response(td, t)
    Z = compute_impedance(step, t, Z0=Z0_REF)

    v = SPEED_OF_LIGHT / np.sqrt(epsr)
    distance = t * v / 2.0
    return distance, impulse, step, Z


def to_fixed_distance_grid(
    distance: np.ndarray,
    impulse: np.ndarray,
    step: np.ndarray,
    d_max: float = 1200.0,
    dd: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将距离域响应重采样到固定网格。

    Args:
        distance: 原始距离轴 (m)
        impulse: 原始脉冲响应（复数或实数）
        step: 原始阶跃响应
        d_max: 最大距离 (m)
        dd: 距离步长 (m)

    Returns:
        grid: 固定距离网格 [n_points]
        impulse_grid: 重采样脉冲响应（实部）
        step_grid: 重采样阶跃响应
    """
    n_points = int(round(d_max / dd))
    grid = np.linspace(0, d_max, n_points, endpoint=False) + dd / 2  # 网格中心对齐

    impulse_arr = np.asarray(impulse, dtype=np.complex128)
    step_arr = np.asarray(step, dtype=np.float64)

    # 用线性插值重采样
    impulse_grid = np.interp(grid, distance, impulse_arr.real).astype(np.float64)
    step_grid = np.interp(grid, distance, step_arr)

    return grid, impulse_grid, step_grid


def read_s11_csv(path: str, skip_first: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取 S11 CSV 文件（兼容 Client 格式）。
    列名大小写不敏感匹配 "freq"/"real"/"imag"。

    Args:
        path: CSV 文件路径
        skip_first: 是否跳过第一个数据行（Client 默认 True）

    Returns:
        freqs: 频率轴 (Hz)
        S: S11 复数数组
    """
    import pandas as pd
    df = pd.read_csv(path, header=0, comment='#')
    freq_col = [c for c in df.columns if "freq" in c.lower()][0]
    real_col = [c for c in df.columns if "real" in c.lower()][0]
    im_cols = [c for c in df.columns if "imag" in c.lower()]

    freqs = df[freq_col].astype(float).values
    re = df[real_col].astype(float).values
    if im_cols:
        im = df[im_cols[0]].astype(float).values
        S = re + 1j * im
    else:
        S = re.astype(np.complex128)

    if skip_first and len(freqs) > 1:
        freqs = freqs[1:]
        S = S[1:]

    return freqs, S
