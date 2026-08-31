import tkinter as tk 
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import pandas as pd
import os

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
    impulse_mag = np.abs(time_domain)
    step_real = np.real(np.cumsum(time_domain) * dt)

    baseline = np.mean(step_real[t < 0]) if np.any(t < 0) else np.mean(step_real[:max(1,int(0.05*len(t)))])
    gamma = step_real - baseline
    gamma = np.clip(gamma, -0.9999, 0.9999)
    Z = Z0 * (1.0 + gamma) / (1.0 - gamma)

    max_step = np.max(np.abs(step_real))
    if max_step > 0:
        step_plot = step_real / max_step
        impulse_rplt = impulse_real / max_step
        impulse_mplt = impulse_mag / max_step
    else:
        step_plot, impulse_rplt, impulse_mplt = step_real, impulse_real, impulse_mag
    return impulse_rplt, impulse_mplt, step_plot, gamma, Z

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


def _local_maxima_indices(y):
    """返回一维数组中局部极大值的索引；只依赖 numpy，避免额外依赖 scipy。"""
    y = np.asarray(y, float)
    if y.size < 3:
        return np.array([], dtype=int)
    return np.where((y[1:-1] >= y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1

def _median_step_jump(distance, step_response, center_m, dmax):
    """计算候选点前后 Step Response 的稳健跳变量。"""
    d = np.asarray(distance, float)
    s = np.asarray(step_response, float)
    # 窗宽随总距离窗口自适应；过小容易受噪声影响，过大会吞掉短距离结构。
    w = max(10.0, min(80.0, 0.018 * float(dmax)))
    pre = (d >= max(0.0, center_m - 2.0 * w)) & (d <= max(0.0, center_m - 0.75 * w))
    post = (d >= center_m + 0.75 * w) & (d <= min(float(dmax), center_m + 2.0 * w))
    if np.count_nonzero(pre) < 3 or np.count_nonzero(post) < 3:
        return 0.0
    return float(abs(np.median(s[post]) - np.median(s[pre])))

def detect_endpoint_from_ifft(distance, impulse_real, step_response, mask=None):
    """
    纯算法末端识别：不使用标称长度先验，只基于 IFFT 后的 Impulse Real 与 Step Response。

    第二修订版：
    - 极近端 3~5m 仍硬排除，避免把端口/夹具的 0m 主峰当末端；
    - 近端响应区不再硬排除，但近端候选只作为“短电缆兜底候选”；
    - 只要中远距离存在可信事件，就优先在中远距离中选择末端候选；
    - 候选评分同时使用 Impulse 包络、Step 前后变化、位置靠后偏好和右边界惩罚；
    - 右边界最后 5% 不直接作为候选，仍用于风险抑制。

    这样可以避免长电缆被 5m 左右的近端夹具振荡误判，同时仍允许几十米短电缆被识别。
    """
    d = np.asarray(distance, float)
    y = np.asarray(impulse_real, float)
    step = np.asarray(step_response, float)
    if mask is None:
        m = np.isfinite(d) & np.isfinite(y) & np.isfinite(step)
    else:
        m = np.asarray(mask, bool) & np.isfinite(d) & np.isfinite(y) & np.isfinite(step)

    d = d[m]
    y = y[m]
    step = step[m]
    if d.size < 50:
        return None

    order = np.argsort(d)
    d = d[order]
    y = y[order]
    step = step[order]

    pos = d >= 0
    d, y, step = d[pos], y[pos], step[pos]
    if d.size < 50:
        return None

    dmax = float(np.nanmax(d))
    if not np.isfinite(dmax) or dmax <= 0:
        return None

    dd = float(np.nanmedian(np.diff(d))) if d.size > 2 else 1.0
    if not np.isfinite(dd) or dd <= 0:
        dd = 1.0

    env_raw = np.abs(y)

    # 对 |Impulse Real| 做米级平滑，降低单点毛刺和高频振铃对候选峰的影响。
    smooth_m = max(3.0, min(10.0, 0.0025 * dmax))
    win = int(round(smooth_m / dd))
    win = max(5, min(win, 401))
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win, dtype=float) / win
    env = np.convolve(env_raw, kernel, mode='same')

    hard_near_min = max(3.0, min(5.0, 0.002 * dmax))
    edge_start = 0.95 * dmax

    # 背景水平避开极近端和右边界，从中间距离段估计。
    bg_region = (d >= 0.10 * dmax) & (d <= 0.80 * dmax)
    if np.count_nonzero(bg_region) < 20:
        bg_region = (d >= hard_near_min) & (d < edge_start)
    if np.count_nonzero(bg_region) < 20:
        return None

    bg_med = float(np.nanmedian(env[bg_region]))
    bg90 = float(np.nanpercentile(env[bg_region], 90))
    bg = max(bg90, bg_med * 2.0, 1e-30)

    # 自适应估计近端夹具响应大致衰减到哪里。这个值不用于硬删除，只用于近端软惩罚和兜底判断。
    near_window = d <= min(30.0, 0.02 * dmax)
    if np.count_nonzero(near_window) >= 3:
        near_peak = float(np.nanmax(env[near_window]))
    else:
        near_peak = float(np.nanmax(env[:max(3, min(20, env.size))]))
    threshold = max(2.2 * bg90, 0.10 * near_peak, 1e-30)
    continuous_m = max(8.0, min(25.0, 0.006 * dmax))
    continuous_pts = max(3, int(round(continuous_m / dd)))
    search_end = min(0.12 * dmax, 250.0)
    search = np.flatnonzero((d >= hard_near_min) & (d <= search_end))
    near_settle = min(0.02 * dmax, 80.0)
    if search.size > continuous_pts:
        below = env < threshold
        for idx in search:
            j2 = min(idx + continuous_pts, below.size)
            if j2 - idx >= continuous_pts and np.all(below[idx:j2]):
                near_settle = float(d[idx])
                break
    near_settle = float(max(hard_near_min, min(near_settle, 0.03 * dmax, 100.0)))

    # 近端候选区上界：近端候选不删除，但不能和中远距离候选同等竞争。
    # 对短电缆：如果没有可信中远距离事件，近端候选仍可作为兜底结果。
    # 近端候选兜底区上界：不能只覆盖 near_settle，否则 50~100m 的夹具拖尾仍可能压过长电缆远端。
    # 这里使用较宽的“近端竞争抑制区”：约 8% Dmax，但上限 450m。
    # 对几十米短电缆，如果后方没有可信事件，仍会通过兜底逻辑被选中。
    near_candidate_limit = max(25.0, min(450.0, max(2.0 * near_settle, 0.08 * dmax)))

    valid = (d > hard_near_min) & (d < edge_start)
    if np.count_nonzero(valid) < 20:
        return None

    local = _local_maxima_indices(env)
    local = np.array([i for i in local if valid[i]], dtype=int)
    if local.size == 0:
        idxs = np.flatnonzero(valid)
        local = np.array([idxs[np.argmax(env[idxs])]], dtype=int)

    # 候选最小间距，避免一串振铃被当作多个独立事件。
    min_sep_m = max(10.0, min(45.0, 0.010 * dmax))
    min_sep_pts = max(1, int(round(min_sep_m / dd)))
    order = local[np.argsort(env[local])[::-1]]
    candidates = []
    for idx in order:
        if all(abs(int(idx) - int(j)) >= min_sep_pts for j in candidates):
            candidates.append(int(idx))
        if len(candidates) >= 80:
            break
    if not candidates:
        return None

    step_jumps = np.array([_median_step_jump(d, step, float(d[i]), dmax) for i in candidates], dtype=float)
    step_scale = max(float(np.nanpercentile(step_jumps, 75)) if step_jumps.size else 0.0, 1e-12)

    edge_region = d >= edge_start
    edge_peak = float(np.nanmax(env_raw[edge_region])) if np.any(edge_region) else 0.0

    # 近端软惩罚：仅作为近端候选内部排序的一个因素；最终会优先考虑可信中远距离事件。
    p_min = 0.20
    tau = max(5.0, min(20.0, 0.20 * near_settle))

    infos = []
    for idx, jump in zip(candidates, step_jumps):
        dist_i = float(d[idx])
        frac = dist_i / dmax
        amp = float(env[idx])
        raw_amp = float(env_raw[idx])
        impulse_score = amp / bg
        step_score = 1.0 + min(jump / step_scale, 3.0)

        # 较弱的靠后偏好：末端通常偏靠后，但不能压死短电缆。
        terminal_pref = 0.85 + 0.25 * frac
        # 用于中远距离候选之间排序的较强靠后偏好。
        terminal_pref_late = 0.55 + 1.65 * (frac ** 1.15)

        near_penalty = p_min + (1.0 - p_min) / (1.0 + np.exp(-(dist_i - near_settle) / tau))

        if frac > 0.93:
            edge_penalty = 0.30
        elif frac > 0.88:
            edge_penalty = 0.60
        elif frac > 0.80:
            edge_penalty = 0.85
        else:
            edge_penalty = 1.0

        edge_ratio = edge_peak / max(raw_amp, 1e-30)
        edge_ratio_penalty = 1.0 / (1.0 + 0.22 * max(edge_ratio - 1.0, 0.0))

        evidence_score = impulse_score * step_score * edge_penalty * edge_ratio_penalty
        overall_score = evidence_score * terminal_pref * near_penalty
        late_score = evidence_score * terminal_pref_late

        infos.append({
            "idx": int(idx),
            "dist": dist_i,
            "frac": frac,
            "impulse_score": float(impulse_score),
            "step_score": float(step_score),
            "evidence_score": float(evidence_score),
            "overall_score": float(overall_score),
            "late_score": float(late_score),
            "is_near": bool(dist_i <= near_candidate_limit),
            "edge_ratio": float(edge_ratio),
        })

    if not infos:
        return None

    # 第一优先级：中远距离可信事件。这样长电缆不会被 5m 左右的夹具振荡抢走。
    # 条件故意较宽：弱远端也可以进入，但纯噪声通常进不来。
    late_infos = [r for r in infos if (not r["is_near"]) and (r["frac"] < 0.95)]
    credible_late = [
        r for r in late_infos
        if (r["impulse_score"] >= 0.45 or r["step_score"] >= 1.25 or r["evidence_score"] >= 0.60)
    ]

    if credible_late:
        # 对于末端识别，多个可信事件同时存在时，允许“偏后”的可信事件胜出。
        # 但右边界惩罚已经包含在 late_score 中，避免直接选到周期边界伪峰。
        best = max(credible_late, key=lambda r: r["late_score"])
    else:
        # 兜底：没有可信中远距离事件时，才允许近端候选胜出，适合几十米短电缆。
        best = max(infos, key=lambda r: r["overall_score"])

    if best is None or not np.isfinite(best["dist"]):
        return None
    return float(best["dist"])

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
        self.root.title("电缆缺陷定位 (S11 → TDR)")
        self.filepaths = []
        self.data_list = []

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

        # 第二行：下采样除数。空白表示不下采样；填写正整数 d 后，
        # 每隔 d 个频点取一个样本，例如 d=2 时约保留原始点数的一半。
        point_frame = tk.Frame(root)
        point_frame.pack(fill="x", padx=8, pady=(0, 2))
        tk.Label(point_frame, text="下采样除数:").pack(side="left", padx=(244, 2))
        self.downsample_divisor_entry = tk.Entry(point_frame, width=8)
        self.downsample_divisor_entry.insert(0, "")
        self.downsample_divisor_entry.pack(side="left", padx=2)

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

        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, root)
        self.toolbar.update()
        self.canvas._tkcanvas.pack(fill="both", expand=True)

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

        # 解析下采样除数。空白表示不下采样；填写 d 后取 freqs[::d]。
        # d=1 等价于不下采样；d<1 无意义，因此报错。
        try:
            downsample_text = self.downsample_divisor_entry.get().strip()
            downsample_divisor = int(downsample_text) if downsample_text != "" else None
            if downsample_divisor is not None and downsample_divisor < 1:
                raise ValueError("下采样除数必须为空或不小于 1 的整数")
        except Exception as e:
            messagebox.showerror("输入错误", f"下采样除数无效: {e}")
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

                # 下采样除数：直接在当前筛选后的频点上按固定间隔抽取样本。
                # 这样不会插值虚构新点；若原数据为线性采样，抽取后仍保持等间隔。
                # 不强行补入最后一个频点，因为补点会破坏严格等间隔。
                if downsample_divisor is not None and downsample_divisor > 1:
                    order = np.argsort(freqs)
                    freqs_sorted = np.asarray(freqs, dtype=float)[order]
                    S_sorted = np.asarray(S, dtype=complex)[order]
                    freqs = freqs_sorted[::downsample_divisor]
                    S = S_sorted[::downsample_divisor]

                    if len(freqs) < 10:
                        raise ValueError(
                            f"{os.path.basename(filepath)} 下采样后频率点过少，无法执行分析"
                        )

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

                imp_r, imp_m, step_n, gamma, Z = postprocess_time_domain(time_domain, t, Z0=Z0)

                distance = t * v / 2.0
                mask = (distance >= 0) & (distance <= save_range)

                self.data_list.append({
                    "name": os.path.basename(filepath),
                    "distance": distance,
                    "mask": mask,
                    "td": time_domain,
                    "imp_r": imp_r,
                    "imp_m": imp_m,
                    "step_n": step_n,
                    "gamma": gamma,
                    "Z": Z,
                    # 实际进入 IFFT 的频率数据。用于单文件输入时显示真实频率范围与点数，
                    # 可反映频率筛选和下采样除数共同作用后的结果。
                    "ifft_freq_min_hz": float(np.min(freqs)),
                    "ifft_freq_max_hz": float(np.max(freqs)),
                    "ifft_point_count": int(len(freqs)),
                    "ifft_step_hz": float(firstStep)
                })

            # 单文件输入时执行纯算法末端识别；多文件对比时不启用，避免图上多条线混淆。
            if len(self.data_list) == 1:
                d0 = self.data_list[0]
                endpoint_m = detect_endpoint_from_ifft(
                    d0["distance"], d0["imp_r"], d0["step_n"], mask=d0["mask"]
                )
                d0["endpoint_m"] = endpoint_m
            else:
                for d in self.data_list:
                    d["endpoint_m"] = None

            info_parts = [f"共处理 {len(self.data_list)} 个文件"]
            # 只有单文件输入时才显示实际进入 IFFT 的频率范围与点数，避免多文件频率范围、
            # 原始点数或下采样结果不同导致界面信息被误读为所有文件共用参数。
            if len(self.data_list) == 1:
                d0 = self.data_list[0]
                endpoint_text = ""
                if d0.get("endpoint_m") is not None and np.isfinite(d0.get("endpoint_m")):
                    endpoint_text = f"   末端识别：{d0['endpoint_m']:.1f}m"
                info_parts.append(
                    f"实际进入IFFT: {d0['ifft_freq_min_hz'] / 1e3:.6g} kHz - "
                    f"{d0['ifft_freq_max_hz'] / 1e6:.6g} MHz, "
                    f"频率点数: {d0['ifft_point_count']}, "
                    f"step_hz={d0['ifft_step_hz']:.6g} Hz"
                    f"{endpoint_text}"
                )
            if smooth_factor and float(smooth_factor) > 1.0:
                info_parts.append(f"平滑: 窗口={int(round(float(smooth_factor)))}")
            self.param_label.config(text=" | ".join(info_parts))

            self.btn_impulse.config(state="normal")
            self.btn_step.config(state="normal")
            self.btn_impedance.config(state="normal")
            self.show_impulse()
        except Exception as e:
            messagebox.showerror("计算失败", f"处理数据时出错：{e}")

    def show_impulse(self):
        self.ax.clear()
        dtype = self.impulse_type.get()
        endpoint_drawn = False
        for data in self.data_list:
            if dtype == "Real":
                ydata = data["imp_r"]
            elif dtype == "Imag":
                ydata = np.imag(data["td"])
            elif dtype == "Magnitude":
                ydata = data["imp_m"]
            elif dtype == "Phase":
                ydata = np.angle(data["td"])
            self.ax.plot(data["distance"][data["mask"]], ydata[data["mask"]], label=data["name"])
            endpoint_m = data.get("endpoint_m")
            if endpoint_m is not None and np.isfinite(endpoint_m) and not endpoint_drawn:
                self.ax.axvline(endpoint_m, color="red", linestyle="--", linewidth=1.2, label="末端识别位置")
                endpoint_drawn = True
        self.ax.set_title(f"Impulse Response ({dtype})")
        self.ax.set_xlabel("Distance (m)")
        self.ax.legend()
        self.canvas.draw()

    def show_step(self):
        self.ax.clear()
        endpoint_drawn = False
        for data in self.data_list:
            self.ax.plot(data["distance"][data["mask"]], data["step_n"][data["mask"]], label=data["name"])
            endpoint_m = data.get("endpoint_m")
            if endpoint_m is not None and np.isfinite(endpoint_m) and not endpoint_drawn:
                self.ax.axvline(endpoint_m, color="red", linestyle="--", linewidth=1.2, label="末端识别位置")
                endpoint_drawn = True
        self.ax.set_title("Step Response")
        self.ax.set_xlabel("Distance (m)")
        self.ax.legend()
        self.canvas.draw()

    def show_impedance(self):
        self.ax.clear()
        endpoint_drawn = False
        for data in self.data_list:
            self.ax.plot(data["distance"][data["mask"]], data["Z"][data["mask"]], label=data["name"])
            endpoint_m = data.get("endpoint_m")
            if endpoint_m is not None and np.isfinite(endpoint_m) and not endpoint_drawn:
                self.ax.axvline(endpoint_m, color="red", linestyle="--", linewidth=1.2, label="末端识别位置")
                endpoint_drawn = True
        self.ax.set_title("Impedance Response")
        self.ax.set_xlabel("Distance (m)")
        self.ax.set_ylabel("Ohm")
        self.ax.legend()
        self.canvas.draw()

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
                ydata = np.imag(data["td"])
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
