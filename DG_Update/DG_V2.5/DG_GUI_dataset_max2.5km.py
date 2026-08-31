from __future__ import annotations

import importlib.util
import sys
import threading
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox
import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DG_MODULE_PATH = SCRIPT_DIR / "[V2.5]DG_dataset_max2.5km.py"
DEFAULT_EXPORT_DIR = PROJECT_ROOT / "DataSet" / "DG_GUI_max2p5km"
DEFAULT_REAL_DATA_ROOT = r"E:\FDR案例-csv"


def load_dg_module():
    spec = importlib.util.spec_from_file_location("dg_dataset_max2p5km", DG_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Cannot load DG module: {DG_MODULE_PATH}")
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


dg = load_dg_module()


DEFECT_TYPE_OPTIONS = [
    "",
    "All",
    "short",
    "aging",
    "moisture_local",
    "moisture_distributed",
    "capacitance_high",
    "capacitance_low",
    "loss_local",
    "resistance_high",
    "short+aging",
    "short+moisture_local",
    "short+moisture_distributed",
    "capacitance_high+capacitance_low",
    "loss_local+resistance_high",
    "aging+moisture_local+moisture_distributed",
    "capacitance_high+capacitance_low+loss_local+resistance_high",
]


class DGSingleSampleGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("DG 2.5km V2.5 Single Sample Analyzer")
        self.root.geometry("1320x820")
        self.current_sample: dict | None = None
        self.current_plot_mode = "s11_real"

        self.profile_var = tk.StringVar(value="Field")
        self.length_var = tk.StringVar(value="")
        self.defect_count_var = tk.StringVar(value="")
        self.defect_types_var = tk.StringVar(value="All")
        self.band_var = tk.StringVar(value="1GHz")
        self.window_var = tk.StringVar(value="Hann")
        self.epsr_var = tk.StringVar(value="2.23")
        self.seed_var = tk.StringVar(value="")
        self.use_templates_var = tk.BooleanVar(value=True)
        self.real_data_root_var = tk.StringVar(value=DEFAULT_REAL_DATA_ROOT)
        self.status_var = tk.StringVar(value="未生成样本")

        self._build_layout()
        self._configure_matplotlib()
        self._draw_empty_plot()

    def _build_layout(self) -> None:
        controls = ttk.Frame(self.root, padding=(8, 6))
        controls.pack(side=tk.TOP, fill=tk.X)

        self._label(controls, "电缆类型").pack(side=tk.LEFT)
        ttk.Combobox(controls, textvariable=self.profile_var, values=["RG58", "Field"], width=8, state="readonly").pack(side=tk.LEFT, padx=(2, 8))

        self._label(controls, "电缆长度(m)").pack(side=tk.LEFT)
        ttk.Entry(controls, textvariable=self.length_var, width=10).pack(side=tk.LEFT, padx=(2, 8))

        self._label(controls, "缺陷数量").pack(side=tk.LEFT)
        ttk.Entry(controls, textvariable=self.defect_count_var, width=7).pack(side=tk.LEFT, padx=(2, 8))

        self._label(controls, "缺陷类型").pack(side=tk.LEFT)
        ttk.Combobox(controls, textvariable=self.defect_types_var, values=DEFECT_TYPE_OPTIONS, width=38).pack(side=tk.LEFT, padx=(2, 8))

        self._label(controls, "频带").pack(side=tk.LEFT)
        ttk.Combobox(controls, textvariable=self.band_var, values=["1GHz", "200MHz"], width=8, state="readonly").pack(side=tk.LEFT, padx=(2, 8))

        self._label(controls, "窗函数").pack(side=tk.LEFT)
        ttk.Combobox(controls, textvariable=self.window_var, values=["Hann", "Hamming", "Blackman", "Rectangular"], width=12, state="readonly").pack(side=tk.LEFT, padx=(2, 8))

        self._label(controls, "相对介电常数").pack(side=tk.LEFT)
        ttk.Entry(controls, textvariable=self.epsr_var, width=8).pack(side=tk.LEFT, padx=(2, 8))

        self.compute_button = ttk.Button(controls, text="开始计算", command=self.start_compute)
        self.compute_button.pack(side=tk.LEFT, padx=(4, 6))
        self.export_button = ttk.Button(controls, text="导出数据", command=self.export_current_sample)
        self.export_button.pack(side=tk.LEFT)

        options = ttk.Frame(self.root, padding=(8, 2))
        options.pack(side=tk.TOP, fill=tk.X)
        self._label(options, "随机种子").pack(side=tk.LEFT)
        ttk.Entry(options, textvariable=self.seed_var, width=13).pack(side=tk.LEFT, padx=(2, 10))
        ttk.Checkbutton(options, text="使用实测模板", variable=self.use_templates_var).pack(side=tk.LEFT, padx=(0, 10))
        self._label(options, "实测数据目录").pack(side=tk.LEFT)
        ttk.Entry(options, textvariable=self.real_data_root_var, width=72).pack(
            side=tk.LEFT, padx=(2, 6), fill=tk.X, expand=True
        )

        info_frame = ttk.Frame(self.root, padding=(8, 2))
        info_frame.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(info_frame, textvariable=self.status_var, anchor=tk.W, justify=tk.LEFT, wraplength=1280).pack(side=tk.LEFT, fill=tk.X, expand=True)

        plot_buttons = ttk.Frame(self.root, padding=(8, 4))
        plot_buttons.pack(side=tk.TOP, fill=tk.X)
        button_specs = [
            ("S11实部", "s11_real"),
            ("S11虚部", "s11_imag"),
            ("S11幅值", "s11_magnitude"),
            ("S11相位", "s11_phase"),
            ("IFFT脉冲响应", "ifft_impulse"),
            ("IFFT阶跃响应", "ifft_step"),
        ]
        for text, mode in button_specs:
            ttk.Button(plot_buttons, text=text, command=lambda m=mode: self.set_plot_mode(m)).pack(side=tk.LEFT, padx=3)

        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(11.5, 6.5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, canvas_frame)
        self.toolbar.update()

    @staticmethod
    def _label(parent: ttk.Frame, text: str) -> ttk.Label:
        return ttk.Label(parent, text=text)

    @staticmethod
    def _configure_matplotlib() -> None:
        matplotlib.rcParams["font.family"] = ["Times New Roman", "SimHei"]
        matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        matplotlib.rcParams["xtick.direction"] = "in"
        matplotlib.rcParams["ytick.direction"] = "in"

    def _draw_empty_plot(self) -> None:
        self.ax.clear()
        self.ax.set_title("等待生成样本")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.grid(True, alpha=0.25)
        self.canvas.draw_idle()

    def parse_config(self) -> dict:
        length_text = self.length_var.get().strip()
        count_text = self.defect_count_var.get().strip()
        epsr_text = self.epsr_var.get().strip()
        type_text = self.defect_types_var.get().strip()
        seed_text = self.seed_var.get().strip()
        return {
            "profile": self.profile_var.get().strip(),
            "band": self.band_var.get().strip(),
            "window": self.window_var.get().strip(),
            "length_m": None if not length_text else float(length_text),
            "n_defects": None if not count_text else int(count_text),
            "allowed_defect_types": [] if not type_text or type_text == "All" else type_text.split("+"),
            "epsr": None if not epsr_text else float(epsr_text),
            "seed": int(np.random.randint(0, 2**31 - 1)) if not seed_text else int(seed_text),
            "use_templates": bool(self.use_templates_var.get()),
            "real_data_root": self.real_data_root_var.get().strip() or DEFAULT_REAL_DATA_ROOT,
        }

    def start_compute(self) -> None:
        try:
            config = self.parse_config()
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc), parent=self.root)
            return
        self.compute_button.configure(state=tk.DISABLED)
        self.status_var.set("正在生成单条样本...")
        worker = threading.Thread(target=self._compute_worker, args=(config,), daemon=True)
        worker.start()

    def _compute_worker(self, config: dict) -> None:
        try:
            sample = dg.generate_interactive_sample(config)
        except Exception as exc:
            self.root.after(0, lambda err=exc: self._compute_failed(err))
            return
        self.root.after(0, lambda: self._compute_finished(sample))

    def _compute_failed(self, exc: Exception) -> None:
        self.compute_button.configure(state=tk.NORMAL)
        self.status_var.set("生成失败")
        messagebox.showerror("生成失败", str(exc), parent=self.root)

    def _compute_finished(self, sample: dict) -> None:
        self.current_sample = sample
        self.compute_button.configure(state=tk.NORMAL)
        self._feedback_inputs(sample)
        self._update_status(sample)
        self.redraw_plot()

    def _feedback_inputs(self, sample: dict) -> None:
        metadata = sample["metadata"]
        self.length_var.set(f"{metadata['total_length_m']:.4g}")
        self.defect_count_var.set(str(metadata["n_defects"]))
        self.epsr_var.set(f"{metadata['epsr']:.4g}")
        actual_types = metadata.get("actual_defect_types", [])
        unique_types = [item for item in dg.SUPPORTED_DEFECT_TYPES if item in set(actual_types)]
        combo_text = "+".join(unique_types) if unique_types else "All"
        self.defect_types_var.set(combo_text)

    def _update_status(self, sample: dict) -> None:
        metadata = sample["metadata"]
        defects = metadata.get("defects", [])
        defect_text = "无缺陷" if not defects else "; ".join(
            f"{d['type']}[{d['start_m']:.1f}-{d['end_m']:.1f}m]" for d in defects
        )
        joints = metadata.get("joint_positions_m", [])
        joint_text = "无" if not joints else ", ".join(f"{p:.1f}m" for p in joints)
        sources = []
        if metadata.get("measured_template_source"):
            sources.append(f"模板={Path(metadata['measured_template_source']).name}")
        if metadata.get("calibration_source"):
            sources.append(f"校准={Path(metadata['calibration_source']).name}")
        if metadata.get("template_mode") == "disabled":
            source_text = "实测模板已禁用"
        else:
            source_text = "；".join(sources) if sources else "未找到可用模板/校准源"
        warnings = "；".join(sample.get("warnings", []))
        if warnings:
            warnings = f"；警告：{warnings}"
        self.status_var.set(
            f"seed={sample['seed']}；profile={sample['profile']}；band={sample['selected_band']}；"
            f"L={metadata['total_length_m']:.2f}m；epsr={metadata['epsr']:.4g}；"
            f"termination={metadata.get('termination', 'unknown')}；缺陷={defect_text}；"
            f"接头={joint_text}；{source_text}{warnings}"
        )

    def set_plot_mode(self, mode: str) -> None:
        self.current_plot_mode = mode
        self.redraw_plot()

    def redraw_plot(self) -> None:
        if self.current_sample is None:
            self._draw_empty_plot()
            return
        self.ax.clear()
        band = self.current_sample["band"]
        mode = self.current_plot_mode
        if mode.startswith("s11"):
            self._plot_frequency(mode, band)
        else:
            self._plot_distance(mode, band)
        self.ax.grid(True, alpha=0.25)
        dg.style_preview_axis(self.ax)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _plot_frequency(self, mode: str, band: dict) -> None:
        freq_mhz = band["freq_hz"] / 1e6
        s11 = band["s11"]
        if mode == "s11_real":
            y = np.real(s11)
            ylabel = "Re(S11)"
            title = "S11 Real Part"
        elif mode == "s11_imag":
            y = np.imag(s11)
            ylabel = "Im(S11)"
            title = "S11 Imaginary Part"
        elif mode == "s11_magnitude":
            y = 20.0 * np.log10(np.maximum(np.abs(s11), 1e-12))
            ylabel = "20log10(|S11|) (dB)"
            title = "S11 Magnitude"
        else:
            y = np.degrees(np.angle(s11))
            ylabel = "Wrapped phase (degree)"
            title = "S11 Wrapped Phase"
            self.ax.set_ylim(-190, 190)
        self.ax.plot(freq_mhz, y, color="#1f77b4", linewidth=0.8)
        self.ax.set_xlim(0, float(np.nanmax(freq_mhz)))
        self.ax.set_xlabel("Frequency (MHz)")
        self.ax.set_ylabel(ylabel)
        self.ax.set_title(f"{title} - {self.current_sample['selected_band']}")

    def _plot_distance(self, mode: str, band: dict) -> None:
        distance = band["distance"]
        if mode == "ifft_impulse":
            y = np.real(band["impulse"])
            ylabel = "Impulse response"
            title = "IFFT Impulse Response"
        else:
            y = np.real(band["step"])
            ylabel = "Step response"
            title = "IFFT Step Response"
        self.ax.plot(distance, y, color="#1f77b4", linewidth=0.8)
        cable = self.current_sample["cable"]
        metadata = self.current_sample["metadata"]
        for defect in metadata.get("defects", []):
            self.ax.axvspan(float(defect["start_m"]), float(defect["end_m"]), color="#808080", alpha=0.18)
        nominal_end = float(metadata["total_length_m"])
        effective_end = float(dg.effective_terminal_phase_length_m(cable))
        self.ax.axvline(nominal_end, color="#666666", linestyle="--", linewidth=0.9, alpha=0.65, label="Nominal end")
        if effective_end > nominal_end + 5.0:
            self.ax.axvline(effective_end, color="#222222", linestyle=":", linewidth=1.0, alpha=0.75, label="Effective end")
        self.ax.set_xlim(0, min(max(effective_end * 1.16, nominal_end * 1.2, 30.0), 3000.0))
        self.ax.set_xlabel("Distance (m)")
        self.ax.set_ylabel(ylabel)
        self.ax.set_title(f"{title} - {self.current_sample['selected_band']}")
        self.ax.legend(fontsize=8, loc="best")

    def export_current_sample(self) -> None:
        if self.current_sample is None:
            messagebox.showinfo("没有数据", "请先点击开始计算生成一条样本。", parent=self.root)
            return
        DEFAULT_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        metadata = self.current_sample["metadata"]
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = (
            f"dg_gui_{stamp}_{self.current_sample['profile']}_"
            f"{self.current_sample['selected_band']}_seed{self.current_sample['seed']}"
        )
        default_path = DEFAULT_EXPORT_DIR / f"{base_name}.csv"
        path_str = filedialog.asksaveasfilename(
            parent=self.root,
            title="导出当前频带 CSV",
            initialdir=str(DEFAULT_EXPORT_DIR),
            initialfile=default_path.name,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path_str:
            return
        csv_path = Path(path_str)
        band = self.current_sample["band"]
        coverage = dg.save_client_csv(
            csv_path,
            band["freq_hz"],
            band["s11"],
            band["distance"],
            band["impulse"],
            band["step"],
            metadata["total_length_m"],
        )
        export_meta = dict(metadata)
        export_meta["selected_band"] = self.current_sample["selected_band"]
        export_meta["export_csv"] = str(csv_path)
        export_meta["exported_at"] = datetime.now().isoformat(timespec="seconds")
        export_meta["band_distance_coverage"] = {self.current_sample["selected_band"]: coverage}
        with csv_path.with_suffix(".yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(export_meta, f, allow_unicode=True, sort_keys=False)
        messagebox.showinfo("导出完成", f"已导出：\n{csv_path}\n{csv_path.with_suffix('.yaml')}", parent=self.root)


def main() -> None:
    root = tk.Tk()
    DGSingleSampleGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
