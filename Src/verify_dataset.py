"""
verify_dataset.py — 合成数据真实度验证

对比合成样本 IFFT 结果与真实数据：
  1. 合成样本的脉冲/阶跃响应 vs RG58-74M 实测
  2. 合成样本 vs FDR 现场数据
  3. 量化噪声统计对比
  4. 真实无标注 S11 → IFFT → 管线确认

输出对比图到 Image/
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from core.tdr_signal import s11_to_responses, to_fixed_distance_grid, read_s11_csv
from core.s11_generator import generate_s11, generate_random_cable, SweepConfig

# ═══════════════════════════════════════════════
# 路径
# ═══════════════════════════════════════════════
PROJ = Path(__file__).resolve().parent.parent
IMAGE_DIR = PROJ / "Image"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

RG58_DIR = r"D:\GitRepository\cable-defect-location-method-based-on-BIS\AI_TEST\REF\RG58-74M(40+4+30)"
FDR_REAL = r"D:\FDR案例-csv\无校准S11\完好RG58电缆-100m\完好电缆-健康电缆.csv"

# ═══════════════════════════════════════════════
# 图1: 合成含缺陷电缆 vs RG58-74M 实测
# ═══════════════════════════════════════════════
print("生成图1: 合成 vs RG58-74M 实测...")

# 真实数据
real_path = Path(RG58_DIR) / "Core-LineA+CUT1+LineB(20degree)-1.csv"
freq_real, S_real = read_s11_csv(str(real_path), skip_first=True)
dist_real, imp_real, step_real, Z_real = s11_to_responses(
    freq_real, S_real, epsr=2.25, window='hann'
)

# 截断到 0-90m
mask_real = (dist_real >= 0) & (dist_real <= 90)

# 合成数据（模拟 74m 含中间缺陷的电缆）
rng = np.random.RandomState(7)
from core.s11_generator import generate_random_cable, generate_s11, SweepConfig
# 手动构造类似 40+4+30 的电缆
from core.s11_generator import CableSample, SegmentParams
cable_synth = CableSample(
    segments=[
        SegmentParams(length_m=40, z0_ohm=50.0, epsr=2.25, alpha_db_per_m_100mhz=0.14),
        SegmentParams(length_m=4, z0_ohm=51.5, epsr=2.40, alpha_db_per_m_100mhz=0.16, is_defect=True),
        SegmentParams(length_m=30, z0_ohm=50.0, epsr=2.25, alpha_db_per_m_100mhz=0.14),
    ],
    epsr=2.25,
    has_joint_reflections=True,
    seed=7,
)
sweep = SweepConfig()
freq_synth, S_synth = generate_s11(cable_synth, sweep, rng=rng,
                                   add_noise=True, inject_joints=True)
dist_synth, imp_synth, step_synth, Z_synth = s11_to_responses(
    freq_synth, S_synth, epsr=2.25, window='hann'
)
mask_synth = (dist_synth >= 0) & (dist_synth <= 90)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("合成数据 vs RG58-74M 实测数据对比", fontsize=14, fontweight='bold')

# 脉冲响应 - 实部
ax = axes[0, 0]
ax.plot(dist_real[mask_real], np.real(imp_real[mask_real]),
        'b-', alpha=0.7, label='真实 (RG58 Core 20°C)', linewidth=0.8)
ax.plot(dist_synth[mask_synth], np.real(imp_synth[mask_synth]),
        'r-', alpha=0.7, label='合成 (40+4+30m)', linewidth=0.8)
ax.set_title("脉冲响应 (实部)")
ax.set_xlabel("距离 (m)")
ax.set_ylabel("幅度")
ax.legend()
ax.grid(True, alpha=0.3)
# 标注关键位置
ax.axvline(40, color='gray', linestyle=':', alpha=0.5)
ax.axvline(44, color='gray', linestyle=':', alpha=0.5)
ax.axvline(74, color='gray', linestyle=':', alpha=0.5)
ax.text(40, ax.get_ylim()[1]*0.9, '40m', ha='center', fontsize=8)
ax.text(44, ax.get_ylim()[1]*0.9, '44m', ha='center', fontsize=8)
ax.text(74, ax.get_ylim()[1]*0.9, '74m末端', ha='center', fontsize=8)

# 阶跃响应
ax = axes[0, 1]
step_real_n = step_real / max(np.abs(step_real).max(), 1e-10)
step_synth_n = step_synth / max(np.abs(step_synth).max(), 1e-10)
ax.plot(dist_real[mask_real], step_real_n[mask_real],
        'b-', alpha=0.7, label='真实', linewidth=0.8)
ax.plot(dist_synth[mask_synth], step_synth_n[mask_synth],
        'r-', alpha=0.7, label='合成', linewidth=0.8)
ax.set_title("阶跃响应 (归一化)")
ax.set_xlabel("距离 (m)")
ax.set_ylabel("归一化幅度")
ax.legend()
ax.grid(True, alpha=0.3)

# |S11| 频域
ax = axes[1, 0]
ax.plot(freq_real/1e6, np.abs(S_real), 'b-', alpha=0.7, label='真实', linewidth=0.5)
ax.plot(freq_synth/1e6, np.abs(S_synth), 'r-', alpha=0.5, label='合成', linewidth=0.5)
ax.set_title("|S11| 频域")
ax.set_xlabel("频率 (MHz)")
ax.set_ylabel("|S11|")
ax.set_xlim([0, 2000])
ax.legend()
ax.grid(True, alpha=0.3)

# 阻抗
ax = axes[1, 1]
Z_real_clip = np.clip(Z_real, -200, 300)
Z_synth_clip = np.clip(Z_synth, -200, 300)
ax.plot(dist_real[mask_real], Z_real_clip[mask_real],
        'b-', alpha=0.7, label='真实', linewidth=0.8)
ax.plot(dist_synth[mask_synth], Z_synth_clip[mask_synth],
        'r-', alpha=0.7, label='合成', linewidth=0.8)
ax.set_title("阻抗分布")
ax.set_xlabel("距离 (m)")
ax.set_ylabel("阻抗 (Ω)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
out1 = IMAGE_DIR / "01_synth_vs_real_rg58.png"
plt.savefig(out1, dpi=150, bbox_inches='tight')
plt.close()
print(f"  已保存: {out1}")

# ═══════════════════════════════════════════════
# 图2: 多个合成样本多样性展示
# ═══════════════════════════════════════════════
print("生成图2: 合成样本多样性...")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("合成电缆样本多样性（脉冲响应实部）", fontsize=14, fontweight='bold')

configs = [
    ("无缺陷 50m", CableSample(segments=[SegmentParams(length_m=50, epsr=2.23)]), 2.23),
    ("无缺陷 200m", CableSample(segments=[SegmentParams(length_m=200, epsr=2.1)]), 2.1),
    ("单缺陷 100m", CableSample(
        segments=[
            SegmentParams(length_m=45, epsr=2.23),
            SegmentParams(length_m=2, z0_ohm=55, epsr=2.4, is_defect=True),
            SegmentParams(length_m=53, epsr=2.23),
        ]), 2.23),
    ("双缺陷 300m", CableSample(
        segments=[
            SegmentParams(length_m=100, epsr=2.2),
            SegmentParams(length_m=3, z0_ohm=53, epsr=2.5, is_defect=True),
            SegmentParams(length_m=100, epsr=2.2),
            SegmentParams(length_m=2, z0_ohm=48, epsr=2.0, is_defect=True),
            SegmentParams(length_m=95, epsr=2.2),
        ]), 2.2),
    ("三缺陷 500m", CableSample(
        segments=[
            SegmentParams(length_m=120, epsr=2.25),
            SegmentParams(length_m=4, z0_ohm=56, epsr=2.6, is_defect=True),
            SegmentParams(length_m=150, epsr=2.25),
            SegmentParams(length_m=3, z0_ohm=47, epsr=1.9, is_defect=True),
            SegmentParams(length_m=180, epsr=2.25),
            SegmentParams(length_m=2, z0_ohm=54, epsr=2.4, is_defect=True),
            SegmentParams(length_m=41, epsr=2.25),
        ]), 2.25),
    ("短电缆 30m", CableSample(segments=[SegmentParams(length_m=30, epsr=2.3)]), 2.3),
]

for idx, (name, cable, epsr) in enumerate(configs):
    ax = axes[idx // 3, idx % 3]
    rng = np.random.RandomState(100 + idx)
    cable.has_joint_reflections = True
    f, s = generate_s11(cable, sweep, rng=rng, add_noise=True, inject_joints=True)
    d, imp, st, Z = s11_to_responses(f, s, epsr=epsr, window='hann')
    cutoff = cable.total_length * 1.3
    mask = (d >= 0) & (d <= cutoff)
    ax.plot(d[mask], np.real(imp[mask]), 'b-', linewidth=0.7)
    ax.set_title(f"{name} (L={cable.total_length:.0f}m)")
    ax.set_xlabel("距离 (m)")
    ax.set_ylabel("脉冲实部")
    ax.grid(True, alpha=0.3)
    # 标注末端
    ax.axvline(cable.total_length, color='r', linestyle='--', alpha=0.5, label='末端')
    ax.legend(fontsize=8)

plt.tight_layout()
out2 = IMAGE_DIR / "02_synth_diversity.png"
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.close()
print(f"  已保存: {out2}")

# ═══════════════════════════════════════════════
# 图3: 噪声统计量化对比
# ═══════════════════════════════════════════════
print("生成图3: 噪声统计量化对比...")

# 合成样本噪声分析（无缺陷电缆做重复）
n_reps = 5
synth_noises = []
synth_freq = None
base_cable = CableSample(segments=[SegmentParams(length_m=100, epsr=2.23)])
for rep in range(n_reps):
    rng = np.random.RandomState(1000 + rep)
    f_s, s = generate_s11(base_cable, sweep, rng=rng, add_noise=True, inject_joints=False)
    synth_noises.append(s)
    if synth_freq is None:
        synth_freq = f_s

# 合成噪声 = 各重复与均值的差
synth_mean = np.mean(synth_noises, axis=0)
synth_diffs = [np.abs(s - synth_mean) for s in synth_noises]
synth_avg_diff = np.mean(synth_diffs, axis=0)

# 真实噪声（RG58 重复测量）
real_noises = []
freq_real_local = None
for rep in [1, 2]:
    p = Path(RG58_DIR) / f"Core-LineA+CUT1+LineB(20degree)-{rep}.csv"
    f_r, s = read_s11_csv(str(p), skip_first=True)
    real_noises.append(s)
    if freq_real_local is None:
        freq_real_local = f_r
real_mean = np.mean(real_noises, axis=0)
real_diff = np.abs(real_noises[0] - real_noises[1])

# 频段统计
bands = [(0, 0.1e9), (0.1e9, 0.5e9), (0.5e9, 1e9), (1e9, 2e9)]
band_names = ["0-100MHz", "100-500MHz", "0.5-1GHz", "1-2GHz"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Noise Statistics Comparison (Synthetic vs Real)", fontsize=14, fontweight='bold')

# 频域噪声曲线（用各自对应的频率轴）
ax = axes[0]
ax.plot(synth_freq/1e6, synth_avg_diff, 'r-', alpha=0.6, label='Synthetic |dS|', linewidth=0.5)
ax.plot(freq_real_local/1e6, real_diff, 'b-', alpha=0.6, label='Real |dS|', linewidth=0.5)
ax.set_xlabel("Frequency (MHz)")
ax.set_ylabel("|dS|")
ax.set_title("Frequency-domain Noise Amplitude")
ax.set_xlim([0, 2000])
ax.set_ylim([0, 0.05])
ax.legend()
ax.grid(True, alpha=0.3)

# 频段柱状图
ax = axes[1]
synth_band_means = []
real_band_means = []
for (flo, fhi) in bands:
    # 合成
    mask_s = (synth_freq >= flo) & (synth_freq < fhi)
    synth_band_means.append(np.mean(synth_avg_diff[mask_s]) if mask_s.sum() > 0 else 0)
    # 真实
    mask_r = (freq_real_local >= flo) & (freq_real_local < fhi)
    real_band_means.append(np.mean(real_diff[mask_r]) if mask_r.sum() > 0 else 0)

x = np.arange(len(bands))
width = 0.35
ax.bar(x - width/2, synth_band_means, width, label='Synthetic', color='r', alpha=0.7)
ax.bar(x + width/2, real_band_means, width, label='Real', color='b', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(band_names)
ax.set_ylabel("Mean |dS|")
ax.set_title("Per-band Mean Noise")
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
out3 = IMAGE_DIR / "03_noise_comparison.png"
plt.savefig(out3, dpi=150, bbox_inches='tight')
plt.close()
print(f"  已保存: {out3}")

# ═══════════════════════════════════════════════
# 图4: 真实 FDR 现场 S11 → IFFT 管线确认
# ═══════════════════════════════════════════════
print("生成图4: FDR 现场数据 IFFT 管线确认...")

if Path(FDR_REAL).exists():
    freq_fdr, S_fdr = read_s11_csv(FDR_REAL, skip_first=True)
    dist_fdr, imp_fdr, step_fdr, Z_fdr = s11_to_responses(
        freq_fdr, S_fdr, epsr=2.25, window='hann'
    )
    mask_fdr = (dist_fdr >= 0) & (dist_fdr <= 150)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"FDR 现场数据 IFFT 管线确认\n{Path(FDR_REAL).name}", fontsize=12)

    ax = axes[0]
    ax.plot(dist_fdr[mask_fdr], np.real(imp_fdr[mask_fdr]), 'b-', linewidth=0.7)
    ax.set_title("脉冲响应 (实部)")
    ax.set_xlabel("距离 (m)")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    step_fdr_n = step_fdr / max(np.abs(step_fdr).max(), 1e-10)
    ax.plot(dist_fdr[mask_fdr], step_fdr_n[mask_fdr], 'g-', linewidth=0.7)
    ax.set_title("阶跃响应 (归一化)")
    ax.set_xlabel("距离 (m)")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    Z_fdr_clip = np.clip(Z_fdr, -100, 200)
    ax.plot(dist_fdr[mask_fdr], Z_fdr_clip[mask_fdr], 'm-', linewidth=0.7)
    ax.set_title("阻抗分布")
    ax.set_xlabel("距离 (m)")
    ax.set_ylabel("Ω")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out4 = IMAGE_DIR / "04_fdr_pipeline_check.png"
    plt.savefig(out4, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {out4}")
else:
    print(f"  跳过：FDR 文件不存在 {FDR_REAL}")

# ═══════════════════════════════════════════════
# 打印量化结论
# ═══════════════════════════════════════════════
print("\n" + "=" * 60)
print("量化对比结论")
print("=" * 60)
print(f"{'频段':>12s} {'合成噪声':>12s} {'真实噪声':>12s} {'比值':>8s}")
for (flo, fhi), bname, sm, rm in zip(bands, band_names, synth_band_means, real_band_means):
    ratio = sm / rm if rm > 0 else float('inf')
    print(f"{bname:>12s} {sm:>12.6f} {rm:>12.6f} {ratio:>8.2f}")

print("\n✅ 验证完成，对比图已保存到 Image/")
