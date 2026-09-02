"""Tkinter GUI for the legacy MATLAB FDR response calculation."""

from __future__ import annotations

import ctypes
import queue
import sys
import time
import tkinter as tk
import tkinter.font as tkfont
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")
matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Microsoft YaHei UI"]
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Microsoft YaHei UI"]
matplotlib.rcParams["font.size"] = 9
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["mathtext.fontset"] = "stix"

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from fdr_response_core import (
    AnalysisParameters,
    AnalysisResult,
    DEFAULT_VELOCITY_FACTOR,
    NUMBA_AVAILABLE,
    analyze_csv,
    warm_up_accelerator,
)


VIEW_RAW_S11 = "raw_s11"
VIEW_IMPULSE = "impulse"
VIEW_STEP = "step"
VIEW_IMPEDANCE = "impedance"
VIEW_MAGNITUDE_FINAL = "magnitude_final"
VIEW_MAGNITUDE_DOUBLE = "magnitude_double"


def enable_windows_dpi_awareness() -> None:
    """Prevent Windows from bitmap-scaling the whole Tk process."""
    if sys.platform != "win32":
        return

    try:
        set_dpi_context = ctypes.windll.user32.SetProcessDpiAwarenessContext
        set_dpi_context.argtypes = (ctypes.c_void_p,)
        set_dpi_context.restype = ctypes.c_bool
        if set_dpi_context(ctypes.c_void_p(-4)):  # Per-monitor DPI aware V2
            return
    except (AttributeError, OSError):
        pass

    try:
        if ctypes.windll.shcore.SetProcessDpiAwareness(1) == 0:
            return
    except (AttributeError, OSError):
        pass

    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass


class FDRResponseApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("电缆缺陷诊断（旧 FDR MATLAB 算法 Python 移植）")
        self.root.geometry("1720x940")
        self.root.minsize(1180, 720)
        self._configure_fonts()

        self.filepaths: list[Path] = []
        self.results: list[AnalysisResult] = []
        self.current_mode = VIEW_IMPULSE
        self._future: Future | None = None
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="fdr-calculation"
        )
        self._messages: queue.Queue[str] = queue.Queue()
        self._closing = False

        self._build_variables()
        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_fonts(self) -> None:
        font_sizes = {
            "TkDefaultFont": 9,
            "TkTextFont": 9,
            "TkMenuFont": 9,
            "TkHeadingFont": 9,
            "TkCaptionFont": 9,
            "TkSmallCaptionFont": 8,
            "TkIconFont": 9,
            "TkTooltipFont": 9,
        }
        for font_name, size in font_sizes.items():
            try:
                tkfont.nametofont(font_name).configure(
                    family="Microsoft YaHei UI", size=size
                )
            except tk.TclError:
                continue
        self.root.option_add("*Font", ("Microsoft YaHei UI", 9))

    def _build_variables(self) -> None:
        self.cable_length_var = tk.StringVar(value="95")
        self.frequency_min_var = tk.StringVar(value="")
        self.frequency_max_var = tk.StringVar(value="")
        self.downsample_var = tk.StringVar(value="")
        self.skip_first_var = tk.BooleanVar(value=True)

        self.velocity_factor_var = tk.StringVar(
            value=f"{DEFAULT_VELOCITY_FACTOR:.15g}"
        )
        self.step_smoothing_var = tk.StringVar(value="5")
        self.impulse_smoothing_var = tk.StringVar(value="50")
        self.time_points_var = tk.StringVar(value="10000")
        self.line_offset_var = tk.StringVar(value="0.0")
        self.step_offset_var = tk.StringVar(value="0")
        self.impulse_normalization_var = tk.StringVar(value="6.5")
        self.test_voltage_var = tk.StringVar(value="10")
        self.reference_impedance_var = tk.StringVar(value="50")
        self.status_var = tk.StringVar(value="请选择一个或多个 LibreVNA CSV 文件")

    def _build_layout(self) -> None:
        self.root.grid_rowconfigure(4, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        first_row = ttk.Frame(self.root, padding=(8, 6, 8, 2))
        first_row.grid(row=0, column=0, sticky="ew")

        self.file_button = ttk.Button(
            first_row, text="选择文件", command=self.select_files, width=11
        )
        self.file_button.grid(row=0, column=0, padx=(0, 4))
        self.run_button = ttk.Button(
            first_row, text="开始计算", command=self.start_calculation, width=11
        )
        self.run_button.grid(row=0, column=1, padx=4)

        column = 2
        column = self._add_entry(
            first_row, column, "电缆长度(m)", self.cable_length_var, 8
        )
        column = self._add_entry(
            first_row, column, "频率下限(kHz)", self.frequency_min_var, 9
        )
        column = self._add_entry(
            first_row, column, "频率上限(MHz)", self.frequency_max_var, 9
        )
        column = self._add_entry(
            first_row, column, "下采样除数", self.downsample_var, 7
        )
        ttk.Checkbutton(
            first_row,
            text="忽略首个测量点",
            variable=self.skip_first_var,
        ).grid(row=0, column=column, padx=6)
        column += 1
        column = self._add_entry(
            first_row, column, "波速度系数", self.velocity_factor_var, 12
        )
        self.export_button = ttk.Button(
            first_row,
            text="导出当前图幅数据",
            command=self.export_current_view,
            state=tk.DISABLED,
            width=17,
        )
        self.export_button.grid(row=0, column=column, padx=(4, 0))

        second_row = ttk.Frame(self.root, padding=(8, 2, 8, 3))
        second_row.grid(row=1, column=0, sticky="ew")
        column = 0
        second_entries = (
            ("阶跃平滑点数", self.step_smoothing_var, 6),
            ("脉冲平滑点数", self.impulse_smoothing_var, 6),
            ("时间点数", self.time_points_var, 8),
            ("测试线校正(m)", self.line_offset_var, 7),
            ("阶跃偏置", self.step_offset_var, 7),
            ("脉冲归一化系数", self.impulse_normalization_var, 7),
            ("测试电压(V)", self.test_voltage_var, 6),
            ("参考阻抗(Ω)", self.reference_impedance_var, 6),
        )
        for label, variable, width in second_entries:
            column = self._add_entry(second_row, column, label, variable, width)

        status_frame = ttk.Frame(self.root, padding=(8, 2, 8, 2))
        status_frame.grid(row=2, column=0, sticky="ew")
        status_frame.grid_columnconfigure(0, weight=1)
        self.status_label = ttk.Label(
            status_frame,
            textvariable=self.status_var,
            anchor="w",
            justify="left",
        )
        self.status_label.grid(row=0, column=0, sticky="ew")
        self.root.bind("<Configure>", self._update_status_wraplength, add="+")

        button_row = ttk.Frame(self.root, padding=(8, 3, 8, 5))
        button_row.grid(row=3, column=0)
        button_specs = (
            ("原始 S11", VIEW_RAW_S11),
            ("脉冲响应结果", VIEW_IMPULSE),
            ("阶跃响应结果", VIEW_STEP),
            ("阻抗响应结果", VIEW_IMPEDANCE),
            ("脉冲幅值结果（最终平滑）", VIEW_MAGNITUDE_FINAL),
            ("脉冲幅值结果（2次平滑）", VIEW_MAGNITUDE_DOUBLE),
        )
        self.view_buttons: list[ttk.Button] = []
        for index, (label, mode) in enumerate(button_specs):
            button = ttk.Button(
                button_row,
                text=label,
                command=lambda selected=mode: self.show_view(selected),
                state=tk.DISABLED,
            )
            button.grid(row=0, column=index, padx=4)
            self.view_buttons.append(button)

        plot_frame = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        plot_frame.grid(row=4, column=0, sticky="nsew")
        plot_frame.grid_rowconfigure(0, weight=1)
        plot_frame.grid_columnconfigure(0, weight=1)

        self.figure = Figure(figsize=(9, 5.2), dpi=100, constrained_layout=True)
        self.axis = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.toolbar = NavigationToolbar2Tk(
            self.canvas, plot_frame, pack_toolbar=False
        )
        self.toolbar.update()
        self.toolbar.grid(row=1, column=0, sticky="ew")
        self._style_axis()
        self.canvas.draw_idle()

    @staticmethod
    def _add_entry(
        parent: ttk.Frame,
        column: int,
        label: str,
        variable: tk.StringVar,
        width: int,
    ) -> int:
        ttk.Label(parent, text=f"{label}:").grid(
            row=0, column=column, padx=(7, 2), sticky="e"
        )
        ttk.Entry(parent, textvariable=variable, width=width).grid(
            row=0, column=column + 1, padx=(0, 2), sticky="w"
        )
        return column + 2

    def _update_status_wraplength(self, _event: tk.Event) -> None:
        self.status_label.configure(wraplength=max(self.root.winfo_width() - 32, 500))

    def select_files(self) -> None:
        selected = filedialog.askopenfilenames(
            title="选择 LibreVNA CSV 文件",
            filetypes=(("CSV 文件", "*.csv"), ("所有文件", "*.*")),
        )
        if not selected:
            return
        self.filepaths = [Path(path) for path in selected]
        self.results.clear()
        self._set_result_controls(False)
        names = "、".join(path.name for path in self.filepaths)
        self.status_var.set(f"已选择 {len(self.filepaths)} 个文件：{names}")

    @staticmethod
    def _required_float(variable: tk.StringVar, label: str) -> float:
        text = variable.get().strip()
        if not text:
            raise ValueError(f"{label}不能为空")
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"{label}必须是数值") from exc

    @staticmethod
    def _required_int(variable: tk.StringVar, label: str) -> int:
        text = variable.get().strip()
        if not text:
            raise ValueError(f"{label}不能为空")
        try:
            return int(text)
        except ValueError as exc:
            raise ValueError(f"{label}必须是整数") from exc

    @staticmethod
    def _optional_float(variable: tk.StringVar, label: str) -> float | None:
        text = variable.get().strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"{label}必须为空或为数值") from exc

    @staticmethod
    def _optional_int(variable: tk.StringVar, label: str) -> int | None:
        text = variable.get().strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError as exc:
            raise ValueError(f"{label}必须为空或为整数") from exc

    def _read_parameters(self) -> AnalysisParameters:
        frequency_min_khz = self._optional_float(
            self.frequency_min_var, "频率下限"
        )
        frequency_max_mhz = self._optional_float(
            self.frequency_max_var, "频率上限"
        )
        parameters = AnalysisParameters(
            cable_length_m=self._required_float(
                self.cable_length_var, "电缆长度"
            ),
            velocity_factor=self._required_float(
                self.velocity_factor_var, "波速度系数"
            ),
            step_smoothing_points=self._required_int(
                self.step_smoothing_var, "阶跃平滑点数"
            ),
            impulse_smoothing_points=self._required_int(
                self.impulse_smoothing_var, "脉冲平滑点数"
            ),
            time_points=self._required_int(self.time_points_var, "时间点数"),
            line_offset_m=self._required_float(
                self.line_offset_var, "测试线长度校正"
            ),
            step_offset=self._required_float(self.step_offset_var, "阶跃偏置"),
            impulse_normalization_factor=self._required_float(
                self.impulse_normalization_var, "脉冲归一化系数"
            ),
            test_voltage_v=self._required_float(
                self.test_voltage_var, "测试电压"
            ),
            reference_impedance_ohm=self._required_float(
                self.reference_impedance_var, "参考阻抗"
            ),
            frequency_min_hz=(
                frequency_min_khz * 1.0e3
                if frequency_min_khz is not None
                else None
            ),
            frequency_max_hz=(
                frequency_max_mhz * 1.0e6
                if frequency_max_mhz is not None
                else None
            ),
            downsample_divisor=self._optional_int(
                self.downsample_var, "下采样除数"
            ),
            skip_first_data_point=self.skip_first_var.get(),
        )
        parameters.validate()
        return parameters

    def start_calculation(self) -> None:
        if self._future is not None and not self._future.done():
            messagebox.showinfo("正在计算", "当前计算尚未完成")
            return
        if not self.filepaths:
            messagebox.showerror("错误", "请先选择一个或多个 CSV 文件")
            return
        try:
            parameters = self._read_parameters()
        except ValueError as exc:
            messagebox.showerror("参数错误", str(exc))
            return

        self.results.clear()
        self._set_result_controls(False)
        self.run_button.configure(state=tk.DISABLED)
        self.file_button.configure(state=tk.DISABLED)
        self.status_var.set(
            "正在准备并行计算内核……"
            if NUMBA_AVAILABLE
            else "未安装 Numba，将使用兼容计算路径……"
        )
        self._future = self._executor.submit(
            self._calculate_files, tuple(self.filepaths), parameters
        )
        self.root.after(80, self._poll_calculation)

    def _calculate_files(
        self,
        filepaths: tuple[Path, ...],
        parameters: AnalysisParameters,
    ) -> tuple[list[AnalysisResult], float]:
        if not NUMBA_AVAILABLE:
            raise RuntimeError(
                "当前 Python 环境未安装 Numba，无法启用计划要求的多线程加速"
            )
        started = time.perf_counter()
        warm_up_accelerator()
        calculated: list[AnalysisResult] = []
        for index, filepath in enumerate(filepaths, start=1):
            self._messages.put(
                f"正在计算 {index}/{len(filepaths)}：{filepath.name}"
            )
            calculated.append(analyze_csv(filepath, parameters))
        return calculated, time.perf_counter() - started

    def _poll_calculation(self) -> None:
        if self._closing:
            return
        while True:
            try:
                self.status_var.set(self._messages.get_nowait())
            except queue.Empty:
                break

        if self._future is None:
            return
        if not self._future.done():
            self.root.after(80, self._poll_calculation)
            return

        self.run_button.configure(state=tk.NORMAL)
        self.file_button.configure(state=tk.NORMAL)
        try:
            self.results, elapsed = self._future.result()
        except Exception as exc:
            self.results.clear()
            self._set_result_controls(False)
            self.status_var.set("计算失败")
            messagebox.showerror("计算失败", str(exc))
        else:
            self._set_result_controls(True)
            details = []
            for result in self.results:
                details.append(
                    f"{result.name}: "
                    f"{result.frequency_min_hz / 1.0e3:.9g} kHz–"
                    f"{result.frequency_max_hz / 1.0e6:.9g} MHz, "
                    f"{result.point_count} 点"
                )
            self.status_var.set(
                f"完成 {len(self.results)} 个文件，耗时 {elapsed:.3f} s | "
                + " | ".join(details)
            )
            self.show_view(VIEW_IMPULSE)
        finally:
            self._future = None

    def _set_result_controls(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        self.export_button.configure(state=state)
        for button in self.view_buttons:
            button.configure(state=state)

    @staticmethod
    def _view_data(
        result: AnalysisResult, mode: str
    ) -> tuple[np.ndarray, np.ndarray]:
        if mode == VIEW_RAW_S11:
            return result.frequency_hz, result.s11_real
        if mode == VIEW_IMPULSE:
            return result.distance_m[1:], result.impulse_smoothed
        if mode == VIEW_STEP:
            return result.distance_m, result.step_smoothed
        if mode == VIEW_IMPEDANCE:
            return result.distance_m, result.impedance_smoothed_ohm
        if mode == VIEW_MAGNITUDE_FINAL:
            return result.distance_m[1:], result.impulse_magnitude_final_db
        if mode == VIEW_MAGNITUDE_DOUBLE:
            return (
                result.distance_m[1:],
                result.impulse_magnitude_double_smoothed_db,
            )
        raise ValueError(f"未知图形模式: {mode}")

    def show_view(self, mode: str) -> None:
        if not self.results:
            return
        self.current_mode = mode
        self.axis.clear()

        for result in self.results:
            x_values, y_values = self._view_data(result, mode)
            if mode == VIEW_RAW_S11:
                self.axis.semilogx(
                    x_values, y_values, linewidth=0.8, label=result.name
                )
            else:
                self.axis.plot(
                    x_values, y_values, linewidth=0.8, label=result.name
                )

        titles = {
            VIEW_RAW_S11: ("原始 S11 实部", "Frequency (Hz)", "S11 Real"),
            VIEW_IMPULSE: ("脉冲响应", "Distance (m)", "RHO"),
            VIEW_STEP: ("阶跃响应", "Distance (m)", "RHO"),
            VIEW_IMPEDANCE: ("阻抗响应", "Distance (m)", "Impedance (Ω)"),
            VIEW_MAGNITUDE_FINAL: (
                "脉冲幅值（最终平滑）",
                "Distance (m)",
                "Magnitude (dB)",
            ),
            VIEW_MAGNITUDE_DOUBLE: (
                "脉冲幅值（2次平滑）",
                "Distance (m)",
                "Magnitude (dB)",
            ),
        }
        title, x_label, y_label = titles[mode]
        self.axis.set_title(title, fontsize=10)
        self.axis.set_xlabel(x_label, fontsize=9)
        self.axis.set_ylabel(y_label, fontsize=9)

        if mode != VIEW_RAW_S11:
            parameters = self.results[0].parameters
            self.axis.set_xlim(-5.0, parameters.cable_length_m * 1.2)
        self._autoscale_y_axis()
        self._style_axis()
        self.axis.legend(loc="best", fontsize=8)
        self.toolbar.update()
        self.canvas.draw_idle()

    def _autoscale_y_axis(self) -> None:
        self.axis.relim(visible_only=True)
        self.axis.autoscale_view(scalex=False, scaley=True)
        self.axis.margins(y=0.08)

    def _style_axis(self) -> None:
        for spine in self.axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.7)
        self.axis.tick_params(
            axis="both",
            which="both",
            direction="in",
            top=True,
            right=True,
            width=0.7,
            labelsize=9,
        )
        self.axis.grid(True, which="major", linewidth=0.3, alpha=0.25)

    def export_current_view(self) -> None:
        if not self.results:
            messagebox.showerror("错误", "当前没有可导出的图幅数据")
            return

        mode_labels = {
            VIEW_RAW_S11: "原始S11",
            VIEW_IMPULSE: "脉冲响应",
            VIEW_STEP: "阶跃响应",
            VIEW_IMPEDANCE: "阻抗响应",
            VIEW_MAGNITUDE_FINAL: "脉冲幅值_最终平滑",
            VIEW_MAGNITUDE_DOUBLE: "脉冲幅值_2次平滑",
        }
        first_x, _ = self._view_data(self.results[0], self.current_mode)
        x_header = (
            "Frequency(Hz)"
            if self.current_mode == VIEW_RAW_S11
            else "Distance(m)"
        )
        columns: list[np.ndarray] = [np.asarray(first_x, dtype=np.float64)]
        headers = [x_header]

        for result in self.results:
            x_values, y_values = self._view_data(result, self.current_mode)
            x_values = np.asarray(x_values, dtype=np.float64)
            y_values = np.asarray(y_values, dtype=np.float64)
            if np.array_equal(x_values, first_x):
                exported_y = y_values
            else:
                exported_y = np.interp(
                    first_x,
                    x_values,
                    y_values,
                    left=np.nan,
                    right=np.nan,
                )
            columns.append(exported_y)
            headers.append(result.name)

        initial_name = f"FDR_{mode_labels[self.current_mode]}.csv"
        output_path = filedialog.asksaveasfilename(
            title="导出当前图幅数据",
            defaultextension=".csv",
            filetypes=(("CSV 文件", "*.csv"), ("所有文件", "*.*")),
            initialfile=initial_name,
        )
        if not output_path:
            return

        try:
            output = np.column_stack(columns)
            pd.DataFrame(output, columns=headers).to_csv(
                output_path,
                index=False,
                encoding="utf-8-sig",
                float_format="%.15g",
            )
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc))
            return
        self.status_var.set(f"当前图幅数据已导出：{output_path}")
        messagebox.showinfo("导出成功", f"数据已保存到：\n{output_path}")

    def _on_close(self) -> None:
        self._closing = True
        if self._future is not None:
            self._future.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def main() -> None:
    enable_windows_dpi_awareness()
    root = tk.Tk()
    FDRResponseApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
