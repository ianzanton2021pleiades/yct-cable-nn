import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path

# ============ matplotlib 全局样式 ============
from matplotlib import font_manager as fm

# Register fonts explicitly to bypass Windows font cache issues
for _fpath in [r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\times.ttf"]:
    if os.path.isfile(_fpath):
        try:
            fm.fontManager.addfont(_fpath)
        except Exception:
            pass

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["SimHei", "Times New Roman", "Arial"],
    "axes.unicode_minus": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "figure.dpi": 100,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_libraries.common import discover_libraries

VISIBLE_AI_MODEL_KEYS = (
    "short_bncnet_v2_ref_ft",
    "unet1d_medium_baseline",
    "unet1d_baseline_full",
)

# ============ 工具函数 ============
def fft_shift(arr, inverse=False):
    n = len(arr); k = n // 2
    return np.concatenate((arr[k:], arr[:k])) if inverse else np.concatenate((arr[-k:], arr[:-k]))

def apply_window(N, window_type="rectangular"):
    if window_type == "hann":
        return np.hanning(N)
    elif window_type == "hamming":
        return np.hamming(N)
    elif window_type == "blackman":
        return np.blackman(N)
    elif window_type == "rectangular" or window_type == "none":
        return np.ones(N)
    else:
        raise ValueError("Unknown window type")

def estimate_first_step(freqs, quantile=5):
    f = np.sort(np.unique(np.asarray(freqs, float)))
    if len(f) < 2:
        raise ValueError("频率点过少，无法估计步进")
    df = np.diff(f)
    df = df[df > 0]
    if len(df) < 2:
        return (f[-1] - f[0]) / max(len(f) - 1, 1)
    firstStep = np.percentile(df, quantile)
    mean_df = np.mean(df)
    if firstStep < mean_df / 10:
        firstStep = mean_df
    return float(firstStep)

def build_equally_spaced_spectrum(freqs, S, automatic_dc=True, window_type="rectangular",
                                  padding_bins=0, df_quantile=5, max_bins=2_000_001):
    freqs = np.asarray(freqs, float)
    S = np.asarray(S, complex)
    order = np.argsort(freqs)
    freqs_sorted = freqs[order]
    S_sorted = S[order]

    firstStep = estimate_first_step(freqs_sorted, quantile=df_quantile)
    fmax = freqs_sorted[-1]
    steps = int(np.floor(fmax / firstStep))
    N = 2 * steps + 1

    if N > max_bins:
        scale = N / max_bins
        firstStep *= scale
        steps = int(np.floor(fmax / firstStep))
        N = 2 * steps + 1

    f_lin = np.arange(0, steps + 1, dtype=float) * firstStep
    S_lin = np.interp(f_lin, freqs_sorted, S_sorted.real) + 1j*np.interp(f_lin, freqs_sorted, S_sorted.imag)

    spectrum = np.zeros(N, dtype=np.complex128)
    for i in range(1, steps + 1):
        spectrum[steps + i] = S_lin[i]
        spectrum[steps - i] = np.conj(S_lin[i])

    if automatic_dc and steps > 2:
        abs_dc = 2 * np.abs(spectrum[steps + 1]) - np.abs(spectrum[steps + 2])
        pha_dc = 2 * np.angle(spectrum[steps + 1]) - np.angle(spectrum[steps + 2])
        spectrum[steps] = abs_dc * np.exp(1j * pha_dc)
    else:
        spectrum[steps] = S_lin[0]

    spectrum *= apply_window(N, window_type=window_type)

    if padding_bins > 0:
        left = spectrum[:steps]
        dc = spectrum[steps:steps+1]
        right = spectrum[steps+1:]
        pad = np.zeros(padding_bins, dtype=np.complex128)
        spectrum = np.concatenate([left, pad, dc, pad, right])
        N = spectrum.size

    return spectrum, firstStep, steps, N

def spectrum_to_time(spectrum, firstStep, N):
    """
    将以 [-fmax ... 0 ... +fmax] 为中心排列的双边频谱转换为因果时域响应。

    原版本在 IFFT 后再次 fft_shift，将时间轴显示为 -T/2 ~ +T/2，随后只保留
    distance >= 0，等价于只显示一半正向无歧义距离。这里改为常规 TDR 的
    因果显示方式：t = 0 ~ T，对应 distance = 0 ~ v/(2*df)。

    注意：np.fft.ifft 已经自带 1/N 归一化，因此这里不再额外除以 N。
    """
    spec = fft_shift(spectrum, inverse=True)  # centered -> FFT order: [DC, +f, ..., -f]
    td = np.fft.ifft(spec)
    dt = 1.0 / (firstStep * N)
    t = np.arange(N) * dt
    return td, t

def postprocess_time_domain(time_domain, t, Z0=50.0):
    dt = t[1] - t[0]
    impulse_real = np.real(time_domain)
    impulse_imag = np.imag(time_domain)
    impulse_mag = np.abs(time_domain)
    step_real = np.real(np.cumsum(time_domain) * dt)

    baseline = np.mean(step_real[t < 0]) if np.any(t < 0) else np.mean(step_real[:max(1,int(0.05*len(t)))])
    gamma = step_real - baseline
    gamma = np.clip(gamma, -0.9999, 0.9999)
    Z = Z0 * (1.0 + gamma) / (1.0 - gamma)

    max_step = np.max(np.abs(step_real))
    step_plot = step_real / max_step if max_step > 0 else step_real

    max_impulse_real = np.max(np.abs(impulse_real))
    max_impulse_imag = np.max(np.abs(impulse_imag))
    max_impulse_mag = np.max(np.abs(impulse_mag))
    impulse_rplt = impulse_real / max_impulse_real if max_impulse_real > 0 else impulse_real
    impulse_iplt = impulse_imag / max_impulse_imag if max_impulse_imag > 0 else impulse_imag
    impulse_mplt = impulse_mag / max_impulse_mag if max_impulse_mag > 0 else impulse_mag
    return impulse_rplt, impulse_iplt, impulse_mplt, step_plot, gamma, Z

def smooth_signal(signal, factor):
    """滑动平均平滑。factor<=1 返回原信号；factor>=2 取整数窗口长度进行滑动均值（same 模式）。"""
    try:
        f = int(round(float(factor)))
    except Exception:
        return signal
    if f <= 1:
        return signal
    kernel = np.ones(f) / f
    return np.convolve(signal, kernel, mode='same')

# ============ CSV 读取 ============
def robust_read_csv(path):
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
    return freqs, S

# ============ GUI ============
class TDRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("电缆缺陷定位 (PC端分析程序)")
        self.filepaths = []
        self.data_list = []
        self.ai_libraries = discover_libraries()
        self.ai_display_to_key = {}

        top_frame = tk.Frame(root)
        top_frame.pack(fill="x", padx=8, pady=6)

        self.btn_file = tk.Button(top_frame, text="选择文件", width=12, command=self.load_files)
        self.btn_file.pack(side="left", padx=4)

        self.btn_run = tk.Button(top_frame, text="开始计算", width=12, command=self.run_analysis)
        self.btn_run.pack(side="left", padx=4)

        # 频率范围输入
        tk.Label(top_frame, text="频率下限(kHz):").pack(side="left", padx=(10,2))
        self.fmin_entry = tk.Entry(top_frame, width=8)
        self.fmin_entry.insert(0, "")
        self.fmin_entry.pack(side="left", padx=2)

        tk.Label(top_frame, text="频率上限(MHz):").pack(side="left", padx=(10,2))
        self.fmax_entry = tk.Entry(top_frame, width=8)
        self.fmax_entry.insert(0, "")
        self.fmax_entry.pack(side="left", padx=2)

        # 平滑系数输入
        tk.Label(top_frame, text="平滑系数:").pack(side="left", padx=(10,2))
        self.smooth_entry = tk.Entry(top_frame, width=6)
        self.smooth_entry.insert(0, "")  # 空表示不平滑
        self.smooth_entry.pack(side="left", padx=2)

        tk.Label(top_frame, text="窗函数:").pack(side="left", padx=(10,2))
        self.window_var = tk.StringVar(value="rectangular")
        self.combo_window = ttk.Combobox(
            top_frame, textvariable=self.window_var,
            values=["rectangular", "hann", "hamming", "blackman", "none"],
            state="readonly", width=10
        )
        self.combo_window.pack(side="left", padx=4)

        tk.Label(top_frame, text="相对介电常数 εr:").pack(side="left", padx=(10,2))
        self.er = tk.Entry(top_frame, width=6)
        self.er.insert(0, "2.23")
        self.er.pack(side="left", padx=2)

        self.skip_first = tk.BooleanVar(value=True)
        tk.Checkbutton(top_frame, text="忽略 CSV 第一行", variable=self.skip_first).pack(side="left", padx=6)

        tk.Label(top_frame, text="脉冲显示:").pack(side="left", padx=(10,2))
        self.impulse_type = tk.StringVar(value="Real")
        self.combo_impulse = ttk.Combobox(
            top_frame, textvariable=self.impulse_type,
            values=["Real", "Imag", "Magnitude", "Phase"],
            state="readonly", width=10
        )
        self.combo_impulse.pack(side="left", padx=4)

        self.btn_export = tk.Button(top_frame, text="导出当前图幅数据", width=18, command=self.export_data)
        self.btn_export.pack(side="left", padx=4)

        self.param_label = tk.Label(root, text="参数信息将在这里显示", anchor="w", justify="right", font=("Helvetica", 10))
        self.param_label.pack(fill="x", padx=8, pady=(0,6))

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)
        self.btn_impulse = tk.Button(btn_frame, text="脉冲响应结果", command=self.show_impulse, state="disabled")
        self.btn_step = tk.Button(btn_frame, text="阶跃响应结果", command=self.show_step, state="disabled")
        self.btn_impedance = tk.Button(btn_frame, text="阻抗响应结果", command=self.show_impedance, state="disabled")
        self.btn_impulse.pack(side="left", padx=5)
        self.btn_step.pack(side="left", padx=5)
        self.btn_impedance.pack(side="left", padx=5)

        tk.Label(btn_frame, text="神经网络库:").pack(side="left", padx=(14, 2))
        self.ai_model_var = tk.StringVar(value="")
        ai_values = self._build_ai_model_values()
        self.combo_ai_model = ttk.Combobox(
            btn_frame,
            textvariable=self.ai_model_var,
            values=ai_values,
            state="readonly",
            width=28,
        )
        if ai_values:
            self.combo_ai_model.current(0)
        self.combo_ai_model.pack(side="left", padx=4)

        # 距离范围输入框
        self.dist_min_entry = tk.Entry(btn_frame, width=6)
        self.dist_min_entry.pack(side="left", padx=(8, 2))
        tk.Label(btn_frame, text="-").pack(side="left")
        self.dist_max_entry = tk.Entry(btn_frame, width=6)
        self.dist_max_entry.pack(side="left", padx=(2, 8))

        self.btn_ai = tk.Button(btn_frame, text="神经网络分析", command=self.run_ai_analysis, state="disabled")
        self.btn_ai.pack(side="left", padx=5)

        self.btn_histogram = tk.Button(btn_frame, text="柱状图分析", command=self.run_histogram_analysis, state="disabled")
        self.btn_histogram.pack(side="left", padx=5)

        tk.Label(btn_frame, text="α:").pack(side="left", padx=(4, 2))
        self.alpha_entry = tk.Entry(btn_frame, width=8)
        self.alpha_entry.pack(side="left", padx=2)

        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, root)
        self.toolbar.update()
        self.canvas._tkcanvas.pack(fill="both", expand=True)

    def _build_ai_model_values(self):
        values = []
        keys = [key for key in VISIBLE_AI_MODEL_KEYS if key in self.ai_libraries]
        for key in keys:
            lib = self.ai_libraries.get(key)
            if not lib:
                continue
            info = lib["info"]
            name = info.get("name", key)
            if lib.get("analyze") is None:
                display = f"{name} (不可用)"
            else:
                display = str(name)
            self.ai_display_to_key[display] = key
            values.append(display)
        return values

    def load_files(self):
        filepaths = filedialog.askopenfilenames(filetypes=[("CSV files","*.csv")])
        if filepaths:
            self.filepaths = list(filepaths)

    def run_analysis(self):
        if not self.filepaths:
            messagebox.showerror("错误", "请先选择 CSV 文件")
            return

        try:
            er_value = float(self.er.get())
            if er_value <= 0:
                raise ValueError("εr必须大于0")
        except Exception as e:
            messagebox.showerror("输入错误", f"介电常数 εr 无效: {e}")
            return

        # 解析频率范围
        try:
            fmin_text = self.fmin_entry.get().strip()
            fmax_text = self.fmax_entry.get().strip()

            fmin = float(fmin_text) * 1e3 if fmin_text != "" else None    # kHz -> Hz
            fmax = float(fmax_text) * 1e6 if fmax_text != "" else None    # MHz -> Hz

            if fmin is not None and fmax is not None and fmin >= fmax:
                raise ValueError("频率下限必须小于上限")
        except Exception as e:
            messagebox.showerror("输入错误", f"频率范围无效: {e}")
            return

        # 解析平滑系数
        try:
            smooth_text = self.smooth_entry.get().strip()
            smooth_factor = float(smooth_text) if smooth_text != "" else 0.0
            if smooth_factor < 0:
                raise ValueError("平滑系数必须为非负数")
        except Exception as e:
            messagebox.showerror("输入错误", f"平滑系数无效: {e}")
            return

        self.data_list.clear()
        Z0 = 50.0
        c = 299792458.0
        v = c / np.sqrt(er_value)
        save_range = 1.2 * 5000.0
        win_type = self.window_var.get()

        try:
            for filepath in self.filepaths:
                freqs, S = robust_read_csv(filepath)
                if self.skip_first.get() and len(freqs) > 1:
                    freqs, S = freqs[1:], S[1:]

                # 应用频率筛选
                if fmin is not None:
                    mask = freqs >= fmin
                    freqs, S = freqs[mask], S[mask]

                if fmax is not None:
                    mask = freqs <= fmax
                    freqs, S = freqs[mask], S[mask]

                if len(freqs) < 10:
                    raise ValueError(f"{os.path.basename(filepath)} 的频率点过少，无法执行分析")

                spectrum, firstStep, steps, N = build_equally_spaced_spectrum(
                    freqs, S, df_quantile=5, window_type=win_type
                )
                time_domain, t = spectrum_to_time(spectrum, firstStep, N)

                # 应用平滑（若设置）。为了保证 Real/Imag/Magnitude/Phase 来自同一个复数时域响应，
                # 这里先对复数响应的实部和虚部分别平滑，再统一计算后处理量。
                if smooth_factor and float(smooth_factor) > 1.0:
                    time_domain = (
                        smooth_signal(np.real(time_domain), smooth_factor)
                        + 1j * smooth_signal(np.imag(time_domain), smooth_factor)
                    )

                imp_r, imp_i, imp_m, step_n, gamma, Z = postprocess_time_domain(time_domain, t, Z0=Z0)

                distance = t * v / 2.0
                mask = (distance >= 0) & (distance <= save_range)

                self.data_list.append({
                    "name": os.path.basename(filepath),
                    "freqs": freqs,
                    "S": S,
                    "distance": distance,
                    "mask": mask,
                    "td": time_domain,
                    "imp_r": imp_r,
                    "imp_i": imp_i,
                    "imp_m": imp_m,
                    "step_n": step_n,
                    "gamma": gamma,
                    "Z": Z
                })

            info_parts = [f"共处理 {len(self.data_list)} 个文件"]
            if fmin is not None or fmax is not None:
                rmin = fmin if fmin is not None else "min"
                rmax = fmax if fmax is not None else "max"
                info_parts.append(f"频率区间: {rmin}Hz - {rmax}Hz")
            if smooth_factor and float(smooth_factor) > 1.0:
                info_parts.append(f"平滑: 窗口={int(round(float(smooth_factor)))}")
            self.param_label.config(text=" | ".join(info_parts))

            self.btn_impulse.config(state="normal")
            self.btn_step.config(state="normal")
            self.btn_impedance.config(state="normal")
            self.btn_histogram.config(state="normal")
            self.btn_ai.config(state="normal" if self.ai_libraries else "disabled")
            self.show_impulse()
        except Exception as e:
            messagebox.showerror("计算失败", f"处理数据时出错：{e}")

    def show_impulse(self):
        self.ax.clear()
        dtype = self.impulse_type.get()
        for data in self.data_list:
            if dtype == "Real":
                ydata = data["imp_r"]
            elif dtype == "Imag":
                ydata = data["imp_i"]
            elif dtype == "Magnitude":
                ydata = data["imp_m"]
            elif dtype == "Phase":
                ydata = np.angle(data["td"])
            self.ax.plot(data["distance"][data["mask"]], ydata[data["mask"]], label=data["name"])
        self.ax.set_title(f"Impulse Response ({dtype})")
        self.ax.set_xlabel("Distance (m)")
        self.ax.legend()
        self.canvas.draw()

    def show_step(self):
        self.ax.clear()
        for data in self.data_list:
            self.ax.plot(data["distance"][data["mask"]], data["step_n"][data["mask"]], label=data["name"])
        self.ax.set_title("Step Response")
        self.ax.set_xlabel("Distance (m)")
        self.ax.legend()
        self.canvas.draw()

    def show_impedance(self):
        self.ax.clear()
        for data in self.data_list:
            self.ax.plot(data["distance"][data["mask"]], data["Z"][data["mask"]], label=data["name"])
        self.ax.set_title("Impedance Response")
        self.ax.set_xlabel("Distance (m)")
        self.ax.set_ylabel("Ohm")
        self.ax.legend()
        self.canvas.draw()

    def run_ai_analysis(self):
        if not self.data_list:
            messagebox.showerror("Error", "Please run analysis first")
            return

        display = self.ai_model_var.get()
        key = self.ai_display_to_key.get(display)
        if not key:
            messagebox.showerror("Error", "Please select a neural network library")
            return
        lib = self.ai_libraries.get(key)
        if not lib or lib.get("analyze") is None:
            err = "" if not lib else lib["info"].get("load_error", "")
            messagebox.showerror("Error", f"Library unavailable: {err}")
            return

        # 解析距离范围输入
        dist_min_text = self.dist_min_entry.get().strip()
        dist_max_text = self.dist_max_entry.get().strip()
        try:
            if dist_min_text and dist_max_text:
                d_min = float(dist_min_text)
                d_max = float(dist_max_text)
                if d_min >= d_max:
                    raise ValueError("距离下限必须小于上限")
                visible_range = (d_min, d_max)
            elif dist_min_text or dist_max_text:
                raise ValueError("请同时输入距离下限和上限，或都留空")
            else:
                visible_range = self.ax.get_xlim()
        except ValueError as e:
            messagebox.showerror("输入错误", f"距离范围无效: {e}")
            return

        self.ax.clear()
        summaries = []
        try:
            for idx, data in enumerate(self.data_list):
                result = lib["analyze"](
                    s11={"frequency_hz": data["freqs"], "s11": data["S"]},
                    distance=data["distance"],
                    impulse_response=data["td"],
                    step_response=data["step_n"],
                    visible_distance_range=visible_range,
                )
                x = np.asarray(result["distance"], dtype=float)
                y = np.asarray(result["confidence"], dtype=float)
                if len(x) < 2:
                    continue

                # fill_between for smooth confidence envelope
                self.ax.fill_between(x, 0, y, alpha=0.25, label=f'{data["name"]} - {result["model_name"]}')
                self.ax.plot(x, y, linewidth=1.0, alpha=0.8)

                end_pos = result.get("end_position_m")
                end_conf = result.get("end_confidence")
                if end_pos is not None:
                    self.ax.axvline(end_pos, color="red", linestyle="--", linewidth=1.2)
                    # Offset label to the right of the dashed line
                    x_range = visible_range[1] - visible_range[0] if visible_range else 100
                    label_offset = x_range * 0.003
                    self.ax.text(end_pos + label_offset, 1.02, "End",
                                 color="red", rotation=90, va="bottom", ha="left",
                                 fontsize=13)
                    summaries.append(f'{data["name"]}: End {end_pos:.2f}m, conf={end_conf:.3f}')
                else:
                    summaries.append(f'{data["name"]}: No end event above threshold')

                defect_positions = result.get("defect_positions_m", [])
                defect_confidences = result.get("defect_confidences", [])
                for pos, conf in zip(defect_positions, defect_confidences):
                    self.ax.axvline(pos, color="orange", linestyle=":", linewidth=1.0)
                    self.ax.plot(pos, conf, "o", color="orange", markersize=6)
                    summaries.append(f'  Defect {pos:.2f}m, conf={conf:.3f}')

            self.ax.set_title("Neural Network Defect Confidence")
            self.ax.set_xlabel("Distance (m)")
            self.ax.set_ylabel("Defect confidence")
            self.ax.set_ylim(-0.02, 1.1)
            self.ax.legend(loc="upper right")
            self.canvas.draw()
            self.param_label.config(text=" | ".join(summaries[:6]) if summaries else "NN analysis complete, no results to show")
        except Exception as e:
            messagebox.showerror("NN Analysis Failed", str(e))

    def run_histogram_analysis(self):
        """
        柱状图分析：基于脉冲响应与衰减基准线的差值，反映各位置阻抗变化的极性和严重程度。
        衰减基准线 = 平均幅值 × exp(-α × distance)
        α输入框为空时自动估计并回填，有值时使用用户指定的α。
        """
        if not self.data_list:
            messagebox.showerror("错误", "请先执行分析")
            return

        # 解析 α 输入
        alpha_text = self.alpha_entry.get().strip()
        user_alpha = None
        if alpha_text:
            try:
                user_alpha = float(alpha_text)
            except ValueError:
                messagebox.showerror("输入错误", f"衰减系数 α 无效: {alpha_text}")
                return

        self.ax.clear()
        try:
            for data in self.data_list:
                dist = data["distance"][data["mask"]]
                imp_r = data["imp_r"][data["mask"]]

                if len(dist) < 2:
                    continue

                # 计算平均幅值作为基准
                avg_amp = np.mean(np.abs(imp_r))

                # 确定 α
                if user_alpha is not None:
                    alpha = user_alpha
                else:
                    # 自动估计：取前10%和后10%的幅值比
                    n = len(imp_r)
                    front_section = imp_r[:max(n // 10, 10)]
                    rear_section = imp_r[-max(n // 10, 10):]

                    front_mean = np.mean(np.abs(front_section))
                    rear_mean = np.mean(np.abs(rear_section))

                    if front_mean > 1e-12 and rear_mean > 1e-12:
                        dist_span = dist[-1] - dist[0]
                        if dist_span > 0:
                            alpha = -np.log(rear_mean / front_mean) / dist_span
                            alpha = max(alpha, 0)
                        else:
                            alpha = 0.001
                    else:
                        alpha = 0.001

                    # 回填到输入框（只取第一个文件的 α）
                    self.alpha_entry.delete(0, tk.END)
                    self.alpha_entry.insert(0, f"{alpha:.6f}")

                # 生成衰减基准线
                baseline = avg_amp * np.exp(-alpha * dist)

                # 计算差值（柱状图数据）
                diff = imp_r - baseline

                # 绘制柱状图，正值为红色（阻抗升高），负值为蓝色（阻抗降低）
                colors = ['red' if v >= 0 else 'blue' for v in diff]

                # 为了可视化效果，对密集数据进行降采样
                if len(dist) > 500:
                    step = len(dist) // 500
                    dist_ds = dist[::step]
                    diff_ds = diff[::step]
                    colors_ds = colors[::step]
                else:
                    dist_ds = dist
                    diff_ds = diff
                    colors_ds = colors

                width = (dist_ds[1] - dist_ds[0]) * 0.8 if len(dist_ds) > 1 else 1.0
                self.ax.bar(dist_ds, diff_ds, width=width, color=colors_ds, alpha=0.7,
                           label=data["name"])

                # 绘制衰减基准线（参考）
                self.ax.plot(dist, baseline, 'k--', linewidth=0.8, alpha=0.5, label='Decay baseline')

            self.ax.set_title("Histogram Analysis (Impedance Change)")
            self.ax.set_xlabel("Distance (m)")
            self.ax.set_ylabel("Normalized Amplitude Difference")
            self.ax.axhline(0, color='gray', linewidth=0.5, linestyle='-')
            self.ax.legend(loc="upper right")
            self.canvas.draw()
            self.param_label.config(text="柱状图分析完成：红色=阻抗升高，蓝色=阻抗降低")
        except Exception as e:
            messagebox.showerror("柱状图分析失败", str(e))

    def export_data(self):
        if not self.data_list:
            messagebox.showerror("错误", "当前没有数据可导出")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                 filetypes=[("CSV files","*.csv")])
        if not file_path:
            return

        max_len = max(len(d["distance"][d["mask"]]) for d in self.data_list)
        distance = self.data_list[0]["distance"][self.data_list[0]["mask"]][:max_len]

        dtype = self.impulse_type.get()
        all_ydata = []
        headers = ["Distance(m)"]
        for data in self.data_list:
            if dtype == "Real":
                ydata = data["imp_r"]
            elif dtype == "Imag":
                ydata = data["imp_i"]
            elif dtype == "Magnitude":
                ydata = data["imp_m"]
            elif dtype == "Phase":
                ydata = np.angle(data["td"])
            ydata = ydata[data["mask"]][:max_len]
            all_ydata.append(ydata)
            headers.append(data["name"])

        out_arr = np.column_stack([distance] + all_ydata)
        pd.DataFrame(out_arr, columns=headers).to_csv(file_path, index=False)
        messagebox.showinfo("导出成功", f"数据已保存到 {file_path}")

# ============ 启动 ============
if __name__ == "__main__":
    root = tk.Tk()
    app = TDRApp(root)
    root.mainloop()
