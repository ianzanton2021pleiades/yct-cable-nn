"""DG V3 single-sample inspector using the same generator API as batch output."""
from __future__ import annotations

import csv
import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

sys.dont_write_bytecode = True

from dg_v3 import generate_sample, load_config
from response import s11_to_responses


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "SimHei"],
    "font.sans-serif": ["SimHei"],
    "axes.unicode_minus": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "figure.dpi": 200,
})


class DGV3App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("DG V3 单样本分析")
        self.config = load_config(Path(__file__).resolve().parent / "configs" / "provisional_v1.yaml")
        self.sample = None
        self.response = None
        self.band = "1ghz"
        self.mode = "s11_real"

        controls = ttk.Frame(root)
        controls.pack(fill=tk.X, padx=8, pady=6)
        self.profile = tk.StringVar(value="field")
        self.length = tk.StringVar(value="600")
        self.epsr = tk.StringVar(value="2.3")
        self.defect_count = tk.StringVar(value="1")
        self.defect_type = tk.StringVar(value="short")
        self.termination = tk.StringVar(value="open")
        self.band_var = tk.StringVar(value="1ghz")
        self.seed = tk.StringVar(value="20260831")

        fields = (
            ("Profile", self.profile, ("field", "rg58")),
            ("长度(m)", self.length, None),
            ("epsr", self.epsr, None),
            ("缺陷数", self.defect_count, None),
            ("缺陷", self.defect_type, ("short", "aging", "moisture_local", "moisture_distributed")),
            ("末端", self.termination, ("open", "short")),
            ("频段", self.band_var, ("1ghz", "200mhz")),
            ("Seed", self.seed, None),
        )
        for label, variable, values in fields:
            ttk.Label(controls, text=label).pack(side=tk.LEFT, padx=(4, 2))
            if values is None:
                ttk.Entry(controls, textvariable=variable, width=9).pack(side=tk.LEFT)
            else:
                ttk.Combobox(controls, textvariable=variable, values=values, state="readonly", width=12).pack(side=tk.LEFT)
        ttk.Button(controls, text="生成", command=self.generate).pack(side=tk.LEFT, padx=8)
        ttk.Button(controls, text="导出", command=self.export).pack(side=tk.LEFT)

        modes = ttk.Frame(root)
        modes.pack(fill=tk.X, padx=8)
        for key, label in (
            ("s11_real", "S11 Real"), ("s11_imag", "S11 Imag"),
            ("magnitude", "Magnitude dB"), ("phase", "Phase"),
            ("impulse", "Impulse"), ("step", "Step"),
        ):
            ttk.Button(modes, text=label, command=lambda value=key: self.show(value)).pack(side=tk.LEFT, padx=2)

        self.status = tk.StringVar(value="尚未生成")
        ttk.Label(root, textvariable=self.status).pack(fill=tk.X, padx=8, pady=4)
        self.figure, self.axis = plt.subplots(figsize=(10, 5), dpi=200)
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, root).update()

    def generate(self) -> None:
        try:
            profile = self.profile.get()
            overrides = {
                "length_m": float(self.length.get()),
                "epsr": float(self.epsr.get()),
                "defect_count": int(self.defect_count.get()),
                "defect_type": self.defect_type.get(),
                "termination": self.termination.get(),
            }
            if profile == "rg58":
                overrides["defect_type"] = "short"
                overrides["termination"] = "open"
            self.sample = generate_sample(int(self.seed.get()), profile, self.config, overrides)
            self.band = self.band_var.get()
            data = self.sample.bands[self.band]
            self.response = s11_to_responses(
                data.frequency_hz,
                data.s11,
                epsr=self.sample.topology.base_epsr,
                distance_step_m=0.25,
                target_distance_max_m=1.2 * self.sample.topology.length_m,
            )
            self.status.set(
                f"{profile} | L={self.sample.topology.length_m:.2f}m | "
                f"events={len(self.sample.truth)} | {self.band} | Hann"
            )
            self.show(self.mode)
        except Exception as exc:
            messagebox.showerror("生成失败", str(exc))

    def show(self, mode: str) -> None:
        self.mode = mode
        if self.sample is None or self.response is None:
            return
        self.axis.clear()
        band = self.sample.bands[self.band]
        if mode == "s11_real":
            x, y, ylabel = band.frequency_hz / 1e6, band.s11.real, "S11 Real"
        elif mode == "s11_imag":
            x, y, ylabel = band.frequency_hz / 1e6, band.s11.imag, "S11 Imag"
        elif mode == "magnitude":
            x, y, ylabel = band.frequency_hz / 1e6, 20.0 * np.log10(np.maximum(np.abs(band.s11), 1e-12)), "|S11| (dB)"
        elif mode == "phase":
            x, y, ylabel = band.frequency_hz / 1e6, np.angle(band.s11, deg=True), "Phase (degree)"
        elif mode == "impulse":
            x, y, ylabel = self.response[0], self.response[1], "Impulse Real"
        else:
            x, y, ylabel = self.response[0], self.response[3], "Step"
        self.axis.plot(x, y, linewidth=0.8)
        self.axis.set_xlabel("Frequency (MHz)" if mode in {"s11_real", "s11_imag", "magnitude", "phase"} else "Distance (m)")
        self.axis.set_ylabel(ylabel)
        self.axis.set_title(f"DG V3 {self.band} - {ylabel}")
        self.axis.grid(True, alpha=0.25)
        self.figure.tight_layout()
        self.canvas.draw()

    def export(self) -> None:
        if self.sample is None:
            messagebox.showerror("导出失败", "请先生成样本")
            return
        directory = filedialog.askdirectory(title="选择导出目录")
        if not directory:
            return
        target = Path(directory)
        sample_id = f"dg_{int(self.sample.sample_id):010d}"
        for band_name, band in self.sample.bands.items():
            np.savez(
                target / f"{sample_id}_{band_name}.npz",
                frequency_hz=band.frequency_hz.astype(np.float64),
                s11_real=band.s11.real.astype(np.float64),
                s11_imag=band.s11.imag.astype(np.float64),
            )
        (target / f"{sample_id}.json").write_text(
            json.dumps({"sample_id": sample_id, "generation": self.sample.generation, "events": self.sample.truth}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        band = self.sample.bands[self.band]
        with (target / f"{sample_id}_{self.band}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Frequency_Hz", "S11_Real", "S11_Imag"])
            writer.writerows(zip(band.frequency_hz, band.s11.real, band.s11.imag))
        self.status.set(f"已导出到 {target}")


def main() -> None:
    root = tk.Tk()
    root.geometry("1500x850")
    DGV3App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
