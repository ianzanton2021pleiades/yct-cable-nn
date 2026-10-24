"""Summary figure: same known-positive hits with far fewer field candidates."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[3] / "Image" / "20261022_FieldNN_Iter5_precision.png"

plt.rcParams["font.family"] = ["Times New Roman", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

models = ["U-Net", "Transformer"]
iter4_candidates = [4.56, 3.75]
iter5_candidates = [2.65, 2.21]
iter4_extra = [88, 41]
iter5_extra = [28, 21]
hits = ["7/11", "6/11"]

x = np.arange(len(models))
width = 0.34

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))

ax = axes[0]
b1 = ax.bar(x - width / 2, iter4_candidates, width, label="Iter4", color="#9aa7b5")
b2 = ax.bar(x + width / 2, iter5_candidates, width, label="Iter5", color="#2f6fb0")
for bars in (b1, b2):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("Events per field curve")
ax.set_title("(a) Field candidate density (213 curves)", fontsize=12)
ax.set_ylim(0, 5.4)
ax.legend(frameon=True, fontsize=9)
ax.grid(axis="y", alpha=0.25)

ax = axes[1]
b1 = ax.bar(x - width / 2, iter4_extra, width, label="Iter4", color="#9aa7b5")
b2 = ax.bar(x + width / 2, iter5_extra, width, label="Iter5", color="#2f6fb0")
for bars, values in ((b1, hits), (b2, hits)):
    for bar, hit in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{int(bar.get_height())}\n({hit})", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("Additional unverified candidates")
ax.set_title("(b) 1500 m known-positive sweeps", fontsize=12)
ax.set_ylim(0, 135)
ax.legend(frameon=True, fontsize=9, loc="upper right")
ax.grid(axis="y", alpha=0.25)

for ax in axes:
    ax.tick_params(direction="in", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_visible(True)

fig.suptitle("FieldNN Iter5: known hits unchanged, spurious candidates cut by 42-68%", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.95))
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200)
plt.close(fig)
print("saved", OUT)
