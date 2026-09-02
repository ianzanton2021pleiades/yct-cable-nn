"""Core calculations for the legacy MATLAB FDR cable-response algorithm.

The GUI deliberately calls this module instead of embedding numerical code in
Tk callbacks.  Only the real part of S11 is used because that is what
``FDR_Response_Calculation_SingleCalSaveFig4RG58.m`` consumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in incomplete runtimes
    NUMBA_AVAILABLE = False

    def njit(*_args, **_kwargs):
        def decorator(function):
            return function

        return decorator

    prange = range


LIGHT_SPEED_M_S = 3.0e8
START_TIME_S = 1.0e-9
MAX_END_TIME_S = 1.0e-4
#DEFAULT_VELOCITY_FACTOR = 173.0 / 300.0 / 0.834
DEFAULT_VELOCITY_FACTOR = 0.6715

@dataclass(frozen=True)
class AnalysisParameters:
    """Validated inputs shared by every selected CSV file."""

    cable_length_m: float = 95.0
    velocity_factor: float = DEFAULT_VELOCITY_FACTOR
    step_smoothing_points: int = 5
    impulse_smoothing_points: int = 50
    time_points: int = 10_000
    line_offset_m: float = 0.0
    step_offset: float = 0.0
    impulse_normalization_factor: float = 6.5
    test_voltage_v: float = 10.0
    reference_impedance_ohm: float = 50.0
    frequency_min_hz: Optional[float] = None
    frequency_max_hz: Optional[float] = None
    downsample_divisor: Optional[int] = None
    skip_first_data_point: bool = True

    def validate(self) -> None:
        finite_fields = {
            "电缆长度": self.cable_length_m,
            "波速度系数": self.velocity_factor,
            "测试线长度校正": self.line_offset_m,
            "阶跃偏置": self.step_offset,
            "脉冲归一化系数": self.impulse_normalization_factor,
            "测试电压": self.test_voltage_v,
            "参考阻抗": self.reference_impedance_ohm,
        }
        for label, value in finite_fields.items():
            if not np.isfinite(value):
                raise ValueError(f"{label}必须是有限数值")

        if self.cable_length_m <= 0:
            raise ValueError("电缆长度必须大于 0")
        if self.velocity_factor <= 0:
            raise ValueError("波速度系数必须大于 0")
        if self.test_voltage_v <= 0:
            raise ValueError("测试电压必须大于 0")
        if self.reference_impedance_ohm <= 0:
            raise ValueError("参考阻抗必须大于 0")
        if self.impulse_normalization_factor == 0:
            raise ValueError("脉冲归一化系数不能为 0")

        self._validate_integer("阶跃平滑点数", self.step_smoothing_points, 1)
        self._validate_integer("脉冲平滑点数", self.impulse_smoothing_points, 1)
        self._validate_integer("时间点数", self.time_points, 2)

        if self.downsample_divisor is not None:
            self._validate_integer("下采样除数", self.downsample_divisor, 1)

        for label, value in (
            ("频率下限", self.frequency_min_hz),
            ("频率上限", self.frequency_max_hz),
        ):
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{label}必须为空或为大于 0 的有限数值")
        if (
            self.frequency_min_hz is not None
            and self.frequency_max_hz is not None
            and self.frequency_min_hz >= self.frequency_max_hz
        ):
            raise ValueError("频率下限必须小于频率上限")

        end_time = min(
            3.0
            * self.cable_length_m
            / (LIGHT_SPEED_M_S * self.velocity_factor),
            MAX_END_TIME_S,
        )
        if end_time <= START_TIME_S:
            raise ValueError("电缆长度过短，无法在当前时间起点后建立计算窗口")

    @staticmethod
    def _validate_integer(label: str, value: int, minimum: int) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{label}必须是整数")
        if int(value) < minimum:
            raise ValueError(f"{label}必须不小于 {minimum}")


@dataclass(frozen=True)
class FrequencyData:
    """A filtered, downsampled LibreVNA trace ready for calculation."""

    source_path: Path
    frequency_hz: np.ndarray
    s11_real: np.ndarray

    @property
    def name(self) -> str:
        return self.source_path.name


@dataclass(frozen=True)
class AnalysisResult:
    """All raw and displayed arrays for one source file."""

    source_path: Path
    parameters: AnalysisParameters
    frequency_hz: np.ndarray
    s11_real: np.ndarray
    time_s: np.ndarray
    distance_uncorrected_m: np.ndarray
    distance_m: np.ndarray
    step_raw: np.ndarray
    step_smoothed: np.ndarray
    impulse_raw: np.ndarray
    impulse_smoothed: np.ndarray
    impulse_compensation: float
    impulse_magnitude_final_db: np.ndarray
    impulse_magnitude_double_smoothed_db: np.ndarray
    impedance_smoothed_ohm: np.ndarray

    @property
    def name(self) -> str:
        return self.source_path.name

    @property
    def point_count(self) -> int:
        return int(self.frequency_hz.size)

    @property
    def frequency_min_hz(self) -> float:
        return float(self.frequency_hz[0])

    @property
    def frequency_max_hz(self) -> float:
        return float(self.frequency_hz[-1])


def _normalized_column_name(name: object) -> str:
    return "".join(character for character in str(name).lower() if character.isalnum())


def _find_column(columns: list[object], kind: str) -> object:
    normalized = [(column, _normalized_column_name(column)) for column in columns]
    if kind == "frequency":
        exact = {"frequency", "freq", "frequencyhz", "freqhz"}
        for column, name in normalized:
            if name in exact:
                return column
        for column, name in normalized:
            if "frequency" in name or name.startswith("freq"):
                return column
    elif kind == "real":
        exact = {"s11real", "real", "s11re"}
        for column, name in normalized:
            if name in exact:
                return column
        for column, name in normalized:
            if "real" in name and ("s11" in name or name == "real"):
                return column
    raise ValueError(
        "CSV 缺少可识别的 Frequency 或 S11_Real 列；"
        f"当前表头为: {', '.join(map(str, columns))}"
    )


def read_librevna_csv(
    path: str | Path,
    parameters: AnalysisParameters,
) -> FrequencyData:
    """Read and preprocess one LibreVNA CSV without using its imaginary column."""

    parameters.validate()
    source_path = Path(path)
    if not source_path.is_file():
        raise ValueError(f"CSV 文件不存在: {source_path}")

    try:
        frame = pd.read_csv(source_path, header=0, comment="#")
    except Exception as exc:
        raise ValueError(f"无法读取 CSV: {exc}") from exc

    if frame.empty:
        raise ValueError(f"{source_path.name} 中没有数据")

    frequency_column = _find_column(list(frame.columns), "frequency")
    real_column = _find_column(list(frame.columns), "real")
    try:
        frequency = pd.to_numeric(frame[frequency_column], errors="raise").to_numpy(
            dtype=np.float64
        )
        s11_real = pd.to_numeric(frame[real_column], errors="raise").to_numpy(
            dtype=np.float64
        )
    except Exception as exc:
        raise ValueError(f"{source_path.name} 的频率或 S11 实部包含非数值内容") from exc

    if parameters.skip_first_data_point:
        if frequency.size <= 1:
            raise ValueError(f"{source_path.name} 跳过首个测量点后没有足够数据")
        frequency = frequency[1:]
        s11_real = s11_real[1:]

    if not np.all(np.isfinite(frequency)) or not np.all(np.isfinite(s11_real)):
        raise ValueError(f"{source_path.name} 包含 NaN 或无穷值")
    if np.any(frequency <= 0):
        raise ValueError(f"{source_path.name} 的频率必须全部大于 0")
    if np.any(np.diff(frequency) <= 0):
        raise ValueError(f"{source_path.name} 的频率必须按原文件严格递增且不能重复")

    mask = np.ones(frequency.size, dtype=bool)
    if parameters.frequency_min_hz is not None:
        mask &= frequency >= parameters.frequency_min_hz
    if parameters.frequency_max_hz is not None:
        mask &= frequency <= parameters.frequency_max_hz
    frequency = frequency[mask]
    s11_real = s11_real[mask]

    divisor = parameters.downsample_divisor
    if divisor is not None and divisor > 1:
        frequency = frequency[::divisor]
        s11_real = s11_real[::divisor]

    if frequency.size < 2:
        raise ValueError(f"{source_path.name} 筛选或下采样后少于 2 个频率点")

    return FrequencyData(
        source_path=source_path,
        frequency_hz=np.ascontiguousarray(frequency, dtype=np.float64),
        s11_real=np.ascontiguousarray(s11_real, dtype=np.float64),
    )


@njit(parallel=True, fastmath=False, cache=True)
def _compute_step_numba(
    frequency_hz: np.ndarray,
    s11_real: np.ndarray,
    time_s: np.ndarray,
    test_voltage_v: float,
) -> np.ndarray:
    """MATLAB loop translated without cross-frequency parallel reduction."""

    frequency_count = frequency_hz.size
    omega = 2.0 * np.pi * frequency_hz

    mean_real = 0.0
    for index in range(frequency_count):
        mean_real += s11_real[index]
    mean_real /= frequency_count

    cable_loss = np.empty(frequency_count, dtype=np.float64)
    for index in range(frequency_count):
        cable_loss[index] = (s11_real[index] - mean_real) / omega[index]

    response = np.zeros(time_s.size, dtype=np.float64)
    scale = test_voltage_v * 2.0 / np.pi
    for time_index in prange(time_s.size):
        current_time = time_s[time_index]
        accumulator = 0.0
        for frequency_index in range(frequency_count - 1):
            accumulator += (
                scale
                * (cable_loss[frequency_index] + cable_loss[frequency_index + 1])
                / 2.0
                * (
                    np.cos(omega[frequency_index] * current_time)
                    - np.cos(omega[frequency_index + 1] * current_time)
                )
                / current_time
            )
        response[time_index] = accumulator
    return response


def _compute_step_scalar_reference(
    frequency_hz: np.ndarray,
    s11_real: np.ndarray,
    time_s: np.ndarray,
    test_voltage_v: float,
) -> np.ndarray:
    """Slow independent reference used by tests on deliberately small arrays."""

    omega = 2.0 * np.pi * frequency_hz
    cable_loss = (s11_real - np.mean(s11_real)) / omega
    response = np.zeros(time_s.size, dtype=np.float64)
    scale = test_voltage_v * 2.0 / np.pi
    for time_index, current_time in enumerate(time_s):
        for frequency_index in range(cable_loss.size - 1):
            response[time_index] += (
                scale
                * (cable_loss[frequency_index] + cable_loss[frequency_index + 1])
                / 2.0
                * (
                    np.cos(omega[frequency_index] * current_time)
                    - np.cos(omega[frequency_index + 1] * current_time)
                )
                / current_time
            )
    return response


def matlab_movmean(values: np.ndarray, window_points: int) -> np.ndarray:
    """Match MATLAB movmean shrink endpoints, including even-window alignment."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("movmean 只接受一维数组")
    if isinstance(window_points, bool) or not isinstance(
        window_points, (int, np.integer)
    ):
        raise ValueError("平滑点数必须是整数")
    window_points = int(window_points)
    if window_points < 1:
        raise ValueError("平滑点数必须不小于 1")
    if window_points == 1 or array.size == 0:
        return array.copy()

    if window_points % 2:
        left_count = right_count = window_points // 2
    else:
        # MATLAB centers an even window on the current and previous sample.
        left_count = window_points // 2
        right_count = window_points // 2 - 1

    positions = np.arange(array.size)
    lower = np.maximum(positions - left_count, 0)
    upper = np.minimum(positions + right_count + 1, array.size)

    # A cumulative sum is both fast and MATLAB-compatible for finite arrays.
    # For local infinities (for example log10(0)), cumulative subtraction would
    # incorrectly contaminate every later window, so use explicit local means.
    if not np.all(np.isfinite(array)):
        result = np.empty(array.size, dtype=np.float64)
        with np.errstate(invalid="ignore"):
            for index in range(array.size):
                result[index] = np.mean(array[lower[index] : upper[index]])
        return result

    cumulative = np.concatenate(
        (np.array([0.0], dtype=np.float64), np.cumsum(array, dtype=np.float64))
    )
    return (cumulative[upper] - cumulative[lower]) / (upper - lower)


def warm_up_accelerator() -> None:
    """Compile the Numba kernel before the first full calculation."""

    frequency = np.linspace(1.0e6, 8.0e6, 8, dtype=np.float64)
    real = np.linspace(-0.5, 0.5, 8, dtype=np.float64)
    time_axis = np.linspace(1.0e-9, 8.0e-9, 8, dtype=np.float64)
    _compute_step_numba(frequency, real, time_axis, 10.0)


def compute_response(
    data: FrequencyData,
    parameters: AnalysisParameters,
) -> AnalysisResult:
    """Compute all six GUI views for one preprocessed trace."""

    parameters.validate()
    frequency_hz = np.asarray(data.frequency_hz, dtype=np.float64)
    s11_real = np.asarray(data.s11_real, dtype=np.float64)
    if frequency_hz.ndim != 1 or s11_real.ndim != 1:
        raise ValueError("频率和 S11 实部必须是一维数组")
    if frequency_hz.size != s11_real.size or frequency_hz.size < 2:
        raise ValueError("频率和 S11 实部必须等长且至少包含 2 点")

    end_time_s = min(
        3.0
        * parameters.cable_length_m
        / (LIGHT_SPEED_M_S * parameters.velocity_factor),
        MAX_END_TIME_S,
    )
    time_s = np.linspace(
        START_TIME_S,
        end_time_s,
        parameters.time_points,
        dtype=np.float64,
    )
    distance_uncorrected_m = (
        time_s * LIGHT_SPEED_M_S * parameters.velocity_factor / 2.0
    )
    distance_m = distance_uncorrected_m - parameters.line_offset_m

    step_raw = _compute_step_numba(
        np.ascontiguousarray(frequency_hz),
        np.ascontiguousarray(s11_real),
        np.ascontiguousarray(time_s),
        parameters.test_voltage_v,
    )
    step_raw = step_raw + parameters.step_offset
    step_smoothed = matlab_movmean(step_raw, parameters.step_smoothing_points)

    impulse_raw = np.diff(step_raw) / (
        np.diff(time_s)
        * LIGHT_SPEED_M_S
        * parameters.impulse_normalization_factor
    )
    impulse_smoothed = matlab_movmean(
        impulse_raw, parameters.impulse_smoothing_points
    )

    distance_impulse_uncorrected = distance_uncorrected_m[1:]
    after_half = np.flatnonzero(
        distance_impulse_uncorrected > parameters.cable_length_m / 2.0
    )
    if after_half.size == 0:
        raise ValueError("时间窗口未覆盖半电缆长度，无法计算脉冲补偿")
    maximum_relative_index = int(np.argmax(impulse_raw[after_half]))
    end_index = int(after_half[0] + maximum_relative_index)
    detected_end_distance = float(distance_impulse_uncorrected[end_index])

    one_fifth_candidates = np.flatnonzero(
        distance_impulse_uncorrected > detected_end_distance / 5.0
    )
    four_fifth_candidates = np.flatnonzero(
        distance_impulse_uncorrected > 4.0 * detected_end_distance / 5.0
    )
    if one_fifth_candidates.size == 0 or four_fifth_candidates.size == 0:
        raise ValueError("无法建立脉冲补偿区间")
    compensation_start = int(one_fifth_candidates[0])
    compensation_end = int(four_fifth_candidates[0])
    impulse_compensation = float(
        np.mean(impulse_raw[compensation_start : compensation_end + 1])
    )

    compensated_impulse = impulse_raw - impulse_compensation
    with np.errstate(divide="ignore", invalid="ignore"):
        magnitude_final_raw_db = 20.0 * np.log10(np.abs(compensated_impulse))
        inner_smoothed = matlab_movmean(
            compensated_impulse, parameters.step_smoothing_points
        )
        magnitude_double_raw_db = 20.0 * np.log10(np.abs(inner_smoothed))
        impedance_raw = np.abs(
            (1.0 + step_raw)
            / (1.0 - step_raw)
            * parameters.reference_impedance_ohm
        )

    impulse_magnitude_final_db = matlab_movmean(
        magnitude_final_raw_db, parameters.impulse_smoothing_points
    )
    impulse_magnitude_double_smoothed_db = matlab_movmean(
        magnitude_double_raw_db, parameters.impulse_smoothing_points
    )
    impedance_smoothed_ohm = matlab_movmean(
        impedance_raw, parameters.step_smoothing_points
    )

    return AnalysisResult(
        source_path=data.source_path,
        parameters=parameters,
        frequency_hz=frequency_hz,
        s11_real=s11_real,
        time_s=time_s,
        distance_uncorrected_m=distance_uncorrected_m,
        distance_m=distance_m,
        step_raw=step_raw,
        step_smoothed=step_smoothed,
        impulse_raw=impulse_raw,
        impulse_smoothed=impulse_smoothed,
        impulse_compensation=impulse_compensation,
        impulse_magnitude_final_db=impulse_magnitude_final_db,
        impulse_magnitude_double_smoothed_db=(
            impulse_magnitude_double_smoothed_db
        ),
        impedance_smoothed_ohm=impedance_smoothed_ohm,
    )


def analyze_csv(
    path: str | Path,
    parameters: AnalysisParameters,
) -> AnalysisResult:
    """Convenience API used by GUI workers and scripts."""

    return compute_response(read_librevna_csv(path, parameters), parameters)


__all__ = [
    "AnalysisParameters",
    "AnalysisResult",
    "DEFAULT_VELOCITY_FACTOR",
    "FrequencyData",
    "LIGHT_SPEED_M_S",
    "NUMBA_AVAILABLE",
    "analyze_csv",
    "compute_response",
    "matlab_movmean",
    "read_librevna_csv",
    "warm_up_accelerator",
]
