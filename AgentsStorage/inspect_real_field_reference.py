from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "Src" / "DG_dataset_max2.5km.py"
spec = importlib.util.spec_from_file_location("dg_dataset_max2p5km", MODULE_PATH)
dg = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = dg
spec.loader.exec_module(dg)

files = [
    p for p in Path("E:/FDR案例-csv/无校准S11").rglob("*.csv")
    if "校正数据" not in p.name
][:6]

fig, axes = plt.subplots(4, len(files), figsize=(3.4 * len(files), 10), constrained_layout=True)
for col, path in enumerate(files):
    freq_hz, s11 = dg.read_s11_csv_compatible(path)
    distance, impulse, step, _ = dg.s11_to_responses(freq_hz, s11, epsr=2.6, window="hann")
    mask_d = distance <= min(float(distance[-1]), 2600.0)
    axes[0, col].plot(freq_hz / 1e6, s11.real, lw=0.7)
    axes[1, col].plot(freq_hz / 1e6, dg.s11_wrapped_phase_deg(s11), lw=0.7)
    axes[2, col].plot(distance[mask_d], impulse[mask_d], lw=0.7)
    axes[3, col].plot(distance[mask_d], step[mask_d], lw=0.7)
    axes[0, col].set_title(f"field real {col}", fontsize=9)
    for row in range(4):
        axes[row, col].tick_params(direction="in", labelsize=7)
        axes[row, col].grid(True, alpha=0.22)

axes[0, 0].set_ylabel("Re(S11)")
axes[1, 0].set_ylabel("Phase (deg)")
axes[2, 0].set_ylabel("Impulse")
axes[3, 0].set_ylabel("Step")
for ax in axes[0, :]:
    ax.set_xlabel("Frequency (MHz)")
for ax in axes[1, :]:
    ax.set_xlabel("Frequency (MHz)")
for ax in axes[2, :]:
    ax.set_xlabel("Distance (m)")
for ax in axes[3, :]:
    ax.set_xlabel("Distance (m)")

out = ROOT / "AgentsStorage" / "real_field_reference_preview.png"
fig.savefig(out, dpi=180)
print(out)
