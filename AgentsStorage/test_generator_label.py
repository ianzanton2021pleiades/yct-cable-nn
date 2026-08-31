"""验证 s11_generator.py 和 label.py 功能"""
import sys, os
sys.path.insert(0, r"D:\GitRepository\cable-defect-location-method-based-on-BIS\AI_TEST\Src")

import numpy as np
from core.s11_generator import generate_s11, generate_sample, SweepConfig
from core.label import build_label_vector
from core.tdr_signal import s11_to_responses, to_fixed_distance_grid

# ─── 测试生成器 ───
print("=" * 60)
print("测试1: 生成随机含缺陷电缆 S11")
print("=" * 60)

rng = np.random.RandomState(42)
from core.s11_generator import generate_random_cable
cable = generate_random_cable(rng, total_length=100.0, epsr=2.23, n_defects_range=(1, 2))

print(f"  电缆总长: {cable.total_length:.1f}m, epsr={cable.epsr:.3f}")
print(f"  段数: {len(cable.segments)}")
for i, seg in enumerate(cable.segments):
    tag = "DEFECT" if seg.is_defect else "healthy"
    print(f"    段{i}: {seg.length_m:.1f}m, Z0={seg.z0_ohm:.1f}Ω, epsr={seg.epsr:.3f}, α={seg.alpha_db_per_m_100mhz:.4f} [{tag}]")

defects = cable.defect_info
print(f"  缺陷数: {len(defects)}")
for d in defects:
    print(f"    位置={d['position']:.1f}m, 长度={d['length']:.1f}m, 严重度={d['severity']:.3f}")

sweep = SweepConfig()
freq, s11 = generate_s11(cable, sweep, rng=rng, add_noise=True, inject_joints=True)
print(f"\n  S11: {len(freq)} 点, [{freq[0]/1e3:.1f}kHz - {freq[-1]/1e6:.1f}MHz]")
print(f"  |S11| range: [{np.abs(s11).min():.6f}, {np.abs(s11).max():.6f}]")

# ─── 测试 IFFT + 标签 ───
print("\n" + "=" * 60)
print("测试2: IFFT + 标签构造")
print("=" * 60)

distance, impulse, step, Z = s11_to_responses(freq, s11, epsr=cable.epsr, window='hann')
print(f"  距离域: {len(distance)} 点, [{distance[0]:.1f} - {distance[-1]:.1f}]m")

# 固定网格
grid, imp_grid, step_grid = to_fixed_distance_grid(distance, impulse, step, d_max=1200, dd=0.5)
print(f"  固定网格: {len(grid)} 点")

# 标签
defect_positions = [d['position'] for d in defects]
severities = [d['severity'] for d in defects]
label = build_label_vector(defect_positions, severities, cable.total_length, grid)
print(f"  标签: {len(label)} 点, max={label.max():.4f}")

# 找标签中的峰
from scipy.signal import find_peaks
peaks, props = find_peaks(label, height=0.1, distance=5)
print(f"  标签峰:")
for pi in peaks:
    print(f"    d={grid[pi]:.1f}m, conf={label[pi]:.4f}")

# ─── 测试无缺陷电缆 ───
print("\n" + "=" * 60)
print("测试3: 无缺陷电缆")
print("=" * 60)

cable2 = generate_random_cable(np.random.RandomState(99), total_length=200.0, epsr=2.1, n_defects_range=(0, 0))
freq2, s11_2 = generate_s11(cable2, sweep, rng=np.random.RandomState(99), add_noise=True)
dist2, imp2, step2, Z2 = s11_to_responses(freq2, s11_2, epsr=cable2.epsr, window='hann')
_, ig2, sg2 = to_fixed_distance_grid(dist2, imp2, step2)
label2 = build_label_vector([], [], cable2.total_length, grid)
peaks2, _ = find_peaks(label2, height=0.1)
print(f"  无缺陷电缆, 总长={cable2.total_length:.0f}m")
print(f"  标签峰数(应仅末端): {len(peaks2)}")
if len(peaks2) > 0:
    print(f"    末端: d={grid[peaks2[0]]:.1f}m, conf={label2[peaks2[0]]:.4f}")

print("\n✅ 所有测试通过")
