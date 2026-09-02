import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib
# 关键优化：设置matplotlib高性能后端，彻底解决卡顿
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# 禁用matplotlib冗余日志
plt.set_loglevel("error")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# ===================== 修复字体警告 + 高性能配置 =====================
plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
# 加速渲染配置
plt.rcParams['path.simplify'] = True
plt.rcParams['path.simplify_threshold'] = 0.5
# ====================================================================

class SParamAnalyzer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("S11参数对比工具")
        self.geometry("1200x800")
        self.resizable(True, True)

        # 数据存储
        self.file_data = {}
        self.calc_data = {}
        self.plot_types = ["幅值", "相位", "实部", "虚部"]

        # 绘图组件（延迟初始化：启动时不创建，解决卡顿！）
        self.fig = None
        self.ax = None
        self.canvas = None
        self.toolbar = None

        # 初始化界面
        self.init_widgets()

    def init_widgets(self):
        # 第一行
        frame1 = tk.Frame(self)
        frame1.pack(pady=10, fill=tk.X, padx=10)
        self.btn_read = tk.Button(frame1, text="读取文件", command=self.read_csv_files, width=12)
        self.btn_read.pack(side=tk.LEFT, padx=5)
        self.btn_calc = tk.Button(frame1, text="开始对比", command=self.calculate_params, width=12, state=tk.DISABLED)
        self.btn_calc.pack(side=tk.LEFT, padx=5)
        self.label_status = tk.Label(frame1, text="等待操作", font=("微软雅黑", 10))
        self.label_status.pack(side=tk.LEFT, padx=20)

        # 第二行
        frame2 = tk.Frame(self)
        frame2.pack(pady=5, fill=tk.X, padx=10)
        tk.Label(frame2, text="选择对比类型：").pack(side=tk.LEFT, padx=5)
        self.plot_var = tk.StringVar()
        self.combo_plot = ttk.Combobox(frame2, textvariable=self.plot_var, values=self.plot_types, state="readonly", width=15)
        self.combo_plot.current(0)
        self.combo_plot.pack(side=tk.LEFT, padx=5)
        self.combo_plot.bind("<<ComboboxSelected>>", self.update_plot)

        # 第三行：占位框架，延迟加载画布（启动秒开核心！）
        self.plot_frame = tk.Frame(self)
        self.plot_frame.pack(pady=5, fill=tk.BOTH, expand=True)

    def _init_matplotlib(self):
        """延迟初始化画布：仅第一次绘图时创建，解决启动卡顿"""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(12, 6), dpi=100)
            # 嵌入画布
            self.canvas = FigureCanvasTkAgg(self.fig, self.plot_frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            # 工具栏
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
            self.toolbar.update()

    def read_csv_files(self):
        file_paths = filedialog.askopenfilenames(
            title="选择CSV文件",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")],
        )
        if not file_paths:
            return

        self.file_data.clear()
        self.calc_data.clear()
        self.btn_calc.config(state=tk.DISABLED)

        for path in file_paths:
            try:
                # 高速读取CSV
                df = pd.read_csv(path, skiprows=1, usecols=[0, 1, 2], header=None, dtype=np.float64)
                freq = df.iloc[:, 0]
                real = df.iloc[:, 1]
                imag = df.iloc[:, 2]
                file_name = path.split("/")[-1].split("\\")[-1]
                self.file_data[file_name] = {"freq": freq, "real": real, "imag": imag}
            except Exception as e:
                messagebox.showerror("错误", f"文件读取失败：{str(e)}")

        self.label_status.config(text=f"已读取 {len(self.file_data)} 个文件")
        if self.file_data:
            self.btn_calc.config(state=tk.NORMAL)

    def calculate_params(self):
        if not self.file_data:
            messagebox.showwarning("提示", "请先读取文件！")
            return

        self.calc_data.clear()
        for name, data in self.file_data.items():
            real = data["real"]
            imag = data["imag"]
            magnitude = np.hypot(real, imag)  # 更快的幅值计算
            phase = np.degrees(np.arctan2(imag, real))
            self.calc_data[name] = {
                "freq": data["freq"], "幅值": magnitude, "相位": phase, "实部": real, "虚部": imag
            }

        self.label_status.config(text="已计算")
        self.update_plot()

    def update_plot(self, event=None):
        if not self.calc_data:
            if self.file_data:
                messagebox.showwarning("提示", "请先计算数据！")
            return

        # 初始化画布（仅第一次执行，之后秒切）
        self._init_matplotlib()
        plot_type = self.plot_var.get()

        # 核心修复：自动适配坐标轴
        self.ax.clear()
        self.ax.cla()

        # 绘制曲线
        for name, data in self.calc_data.items():
            self.ax.plot(data["freq"], data[plot_type], label=name, linewidth=1.2)

        # 图表样式
        self.ax.set_title(f"S11参数对比 - {plot_type}", fontsize=14)
        self.ax.set_xlabel("频率 (Hz)", fontsize=12)
        self.ax.set_ylabel(plot_type, fontsize=12)
        self.ax.legend(loc="best")
        self.ax.grid(True, alpha=0.3)
        
        # ✅ 关键修复：自动重新计算坐标轴范围（解决刻度不切换问题）
        self.ax.relim()
        self.ax.autoscale(enable=True, axis='both', tight=True)
        
        # 高速刷新画布
        self.canvas.draw_idle()

if __name__ == "__main__":
    app = SParamAnalyzer()
    app.mainloop()