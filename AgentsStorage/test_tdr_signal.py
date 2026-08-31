"""验证 tdr_signal.py 信号库基本功能"""
import sys, os
sys.path.insert(0, r"D:\GitRepository\cable-defect-location-method-based-on-BIS\AI_TEST\Src")

import numpy as np
from core.tdr_signal import (
    s11_to_responses, to_fixed_distance_grid, read_s11_csv,
    build_equally_spaced_spectrum, spectrum_to_time
)

print("=" * 60)
print("测试1: 用 RG58-74M Core 20degree-1 验证")
print("=" * 60)

path = r"D:\GitRepository\cable-defect-location-method-based-on-BIS\AI_TEST\REF\RG58-74M(40+4+30)\Core-LineA+CUT1+LineB(20degree)-1.csv"
freqs, S = read_s11_csv(path, skip_first=True)
print(f"  加载: {len(freqs)} 点, freq=[{freqs[0]/1e6:.1f}, {freqs[-1]/1e6:.1f}] MHz")

distance, impulse, step, Z = s11_to_responses(freqs, S, epsr=2.25, window='hann')
print(f"  IFFT完成: {len(distance)} 点, distance=[{distance[0]:.2f}, {distance[-1]:.2f}] m")
print(f"  impulse range: [{np.real(impulse).min():.6f}, {np.real(impulse).max():.6f}]")
print(f"  step range: [{step.min():.6f}, {step.max():.6f}]")

# 找末端反射 (~74m)
imp_real = np.real(impulse)
mask_80 = (distance >= 60) & (distance <= 80)
d80 = distance[mask_80]
i80 = imp_real[mask_80]
end_idx = np.argmax(np.abs(i80))
print(f"  末端反射: d={d80[end_idx]:.2f}m, impulse_re={i80[end_idx]:.6f}")

# 固定网格重采样
grid, imp_grid, step_grid = to_fixed_distance_grid(distance, impulse, step, d_max=1200, dd=0.5)
print(f"  固定网格: {len(grid)} 点, d=[{grid[0]:.2f}, {grid[-1]:.2f}] m")
print(f"  imp_grid range: [{imp_grid.min():.6f}, {imp_grid.max():.6f}]")

# 在网格上找末端
mask_end = (grid >= 70) & (grid <= 80)
g_end = grid[mask_end]
i_end = imp_grid[mask_end]
ei = np.argmax(np.abs(i_end))
print(f"  末端(固定网格): d={g_end[ei]:.2f}m, impulse={i_end[ei]:.6f}")

print("\n✅ tdr_signal.py 基本验证通过")
