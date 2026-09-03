"""Figures: how close DG V3 RLGC vs CST V1 continuous are."""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = Path(r"D:/GitRepository/Cable-NN/DifferentAgent/CodeX/DG_Route_Evaluation/output")
FIG = HERE

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["SimHei", "Microsoft YaHei", "Arial", "Times New Roman"],
    "axes.unicode_minus": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.spines.top": True,
    "axes.spines.right": True,
    "font.size": 10,
})

A = "dg_v3_rlgc"
B = "cst_v1_continuous"
REF = "cst_v1_ladder_0p1m"
LAB = {"dg_v3_rlgc": "DG V3 RLGC", "cst_v1_continuous": "CST V1 连续固定RLGC",
       "cst_v1_ladder_0p1m": "CST V1 0.1 m梯形"}
COL = {"dg_v3_rlgc": "#1f77b4", "cst_v1_continuous": "#334155", "cst_v1_ladder_0p1m": "#7c3aed"}


def load_s11(layer: str, cand: str, case: str):
    p = OUT / "s11" / layer / cand / f"s11_{case}.csv"
    d = np.genfromtxt(p, delimiter=",", names=True, dtype=float)
    return d["Frequency_Hz"], d["S11_Real"] + 1j * d["S11_Imag"]


def load_resp(layer: str, cand: str, case: str):
    return dict(np.load(OUT / "responses" / layer / cand / f"response_{case}.npz"))


# --- Figure 1: baseline S11 spectra + |A-B| residual ---
f, sA = load_s11("clean", A, "baseline")
_, sB = load_s11("clean", B, "baseline")
_, sR = load_s11("clean", REF, "baseline")
fig, axes = plt.subplots(2, 1, figsize=(11.5, 8), dpi=200, sharex=True)
ax = axes[0]
ax.semilogx(f, 20 * np.log10(np.maximum(np.abs(sA), 1e-12)), color=COL[A], lw=0.9, label=LAB[A])
ax.semilogx(f, 20 * np.log10(np.maximum(np.abs(sB), 1e-12)), color=COL[B], lw=0.9, ls="--", label=LAB[B])
ax.semilogx(f, 20 * np.log10(np.maximum(np.abs(sR), 1e-12)), color=COL[REF], lw=0.7, alpha=0.45, label=LAB[REF])
ax.set_ylabel("|S11| (dB)")
ax.set_ylim(-140, 5)
ax.legend(loc="best", fontsize=8)
ax.grid(True, color="#d1d5db", lw=0.45, alpha=0.6)
ax.set_title("clean-baseline |S11|：三条路线与 100 MHz 锚定的关系")
ax = axes[1]
ax.semilogx(f, np.abs(sA - sB), color="#dc2626", lw=0.9, label="|DG V3 − CST连续|")
ax.semilogx(f, np.abs(sB - sR), color=COL[REF], lw=0.9, alpha=0.75, label="|CST连续 − CST 0.1m梯形|")
ax.axvline(100e6, color="#999999", lw=0.8, ls=":")
ax.text(100e6, 1e-4, "100 MHz 锚定点", rotation=90, fontsize=8, color="#666666", ha="right")
ax.set_yscale("log")
ax.set_ylim(1e-6, 5)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("复数差幅值 |ΔS11|")
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, which="both", color="#d1d5db", lw=0.4, alpha=0.6)
fig.tight_layout()
fig.savefig(FIG / "fig1_baseline_s11_diff.png", bbox_inches="tight")
plt.close(fig)

# --- Figure 2: worst-case distance domain (local_R50ohm_15m) ---
case = "local_R50ohm_15m"
rA = load_resp("clean", A, case)
rB = load_resp("clean", B, case)
rR = load_resp("clean", REF, case)
fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), dpi=200, sharex=True)
d = rA["algorithm1_distance_m"]
for ax, key, name in ((axes[0, 0], "algorithm1_step", "算法1 阶跃"), (axes[0, 1], "algorithm1_impulse", "算法1 脉冲")):
    ax.plot(d, rA[key], color=COL[A], lw=0.9, label=LAB[A])
    ax.plot(d, rB[key], color=COL[B], lw=0.9, ls="--", label=LAB[B])
    ax.set_title(f"clean-{case} {name}")
    ax.set_xlim(0, 60)
    ax.grid(True, color="#d1d5db", lw=0.4, alpha=0.6)
    ax.legend(fontsize=7, loc="best")
d2 = rA["algorithm2_distance_m"]
d2i = rA["algorithm2_impulse_distance_m"]
axes[1, 0].plot(d2, rA["algorithm2_step"], color=COL[A], lw=0.9, label=LAB[A])
axes[1, 0].plot(d2, rB["algorithm2_step"], color=COL[B], lw=0.9, ls="--", label=LAB[B])
axes[1, 0].set_title("算法2 阶跃（平滑）")
axes[1, 0].set_xlim(0, 60)
axes[1, 0].grid(True, color="#d1d5db", lw=0.4, alpha=0.6)
axes[1, 0].legend(fontsize=7, loc="best")
axes[1, 1].plot(d2i, rA["algorithm2_impulse"], color=COL[A], lw=0.9, label=LAB[A])
axes[1, 1].plot(d2i, rB["algorithm2_impulse"], color=COL[B], lw=0.9, ls="--", label=LAB[B])
axes[1, 1].plot(d2i, rR["algorithm2_impulse"], color=COL[REF], lw=0.6, alpha=0.4, label=LAB[REF])
axes[1, 1].set_title("算法2 脉冲（平滑）")
axes[1, 1].set_xlim(0, 60)
axes[1, 1].grid(True, color="#d1d5db", lw=0.4, alpha=0.6)
axes[1, 1].legend(fontsize=7, loc="best")
for ax in axes.flat:
    ax.set_xlabel("Distance (m)")
fig.suptitle("差异最大的工况 clean-local_R50ohm_15m：DG V3 与 CST 连续线几乎重叠", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(FIG / "fig2_worst_case_distance.png", bbox_inches="tight")
plt.close(fig)

print("saved:", FIG / "fig1_baseline_s11_diff.png", FIG / "fig2_worst_case_distance.png")
