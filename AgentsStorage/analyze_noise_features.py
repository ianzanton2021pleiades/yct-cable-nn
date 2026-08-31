"""
噪声模型校准 + 接头/缺陷特征提取分析
读取 RG58-74M 实验室数据（只读），输出结论到控制台。
"""
import numpy as np
import pandas as pd
import os

# ─── 路径 ───
RG58_DIR = r"D:\GitRepository\cable-defect-location-method-based-on-BIS\AI_TEST\REF\RG58-74M(40+4+30)"
FDR_DIR = r"D:\GitRepository\cable-defect-location-method-based-on-BIS\AI_TEST\REF\RG58-74M(40+4+30)"  # placeholder
FDR_REAL = r"D:\FDR案例-csv\无校准S11\完好RG58电缆-100m\完好电缆-健康电缆.csv"

# ─── 辅助函数 ───
def load_s11(path):
    """加载 CSV, 返回 (freq, S_complex)"""
    df = pd.read_csv(path, header=0)
    # 列名匹配
    cols = [c for c in df.columns]
    freq_col = None
    real_col = None
    imag_col = None
    for c in cols:
        cl = c.lower().strip()
        if 'freq' in cl and freq_col is None:
            freq_col = c
        if 'real' in cl and real_col is None:
            real_col = c
        if 'imag' in cl and imag_col is None:
            imag_col = c
    freq = df[freq_col].astype(float).values
    re = df[real_col].astype(float).values
    if imag_col:
        im = df[imag_col].astype(float).values
        S = re + 1j * im
    else:
        S = re.astype(np.complex128)
    return freq, S

def quick_ifft(freq, S, epsr=2.25):
    """简化版IFFT用于分析"""
    f_sorted_idx = np.argsort(freq)
    freqs = freq[f_sorted_idx]
    S_sorted = S[f_sorted_idx]
    # 估计频步
    df_all = np.diff(freqs)
    df_all = df_all[df_all > 0]
    firstStep = np.percentile(df_all, 5)
    mean_df = np.mean(df_all)
    if firstStep < mean_df / 10:
        firstStep = mean_df
    fmax = freqs[-1]
    steps = int(np.floor(fmax / firstStep))
    N = 2 * steps + 1
    # 插值
    f_lin = np.arange(0, steps + 1) * firstStep
    S_lin = np.interp(f_lin, freqs, S_sorted.real) + 1j * np.interp(f_lin, freqs, S_sorted.imag)
    # Hermitian
    spectrum = np.zeros(N, dtype=np.complex128)
    for i in range(1, steps + 1):
        spectrum[steps + i] = S_lin[i]
        spectrum[steps - i] = np.conj(S_lin[i])
    # DC外推
    if steps > 2:
        abs_dc = 2 * np.abs(spectrum[steps + 1]) - np.abs(spectrum[steps + 2])
        pha_dc = 2 * np.angle(spectrum[steps + 1]) - np.angle(spectrum[steps + 2])
        spectrum[steps] = abs_dc * np.exp(1j * pha_dc)
    else:
        spectrum[steps] = S_lin[0]
    # 汉宁窗
    spectrum *= np.hanning(N)
    # FFT顺序转换
    k = N // 2
    spec_fft = np.concatenate((spectrum[k:], spectrum[:k]))
    # IFFT
    td = np.fft.ifft(spec_fft)
    dt = 1.0 / (firstStep * N)
    t = np.arange(N) * dt
    impulse = td
    step = np.real(np.cumsum(td) * dt)
    c = 299792458.0
    distance = t * c / (2.0 * np.sqrt(epsr))
    return distance, impulse, step

# ─────────────────────────────────────────────
# PART 1a: 噪声模型校准（重复测量残差分析）
# ─────────────────────────────────────────────
print("=" * 70)
print("PART 1a: 噪声模型校准 — 重复测量残差分析")
print("=" * 70)

# 选取 Core 20degree 重复测量对
pairs = [
    ("Core CUT1 20°C", 
     os.path.join(RG58_DIR, "Core-LineA+CUT1+LineB(20degree)-1.csv"),
     os.path.join(RG58_DIR, "Core-LineA+CUT1+LineB(20degree)-2.csv")),
    ("Core CUT1 70°C",
     os.path.join(RG58_DIR, "Core-LineA+CUT1+LineB(70degree)-1.csv"),
     os.path.join(RG58_DIR, "Core-LineA+CUT1+LineB(70degree)-2.csv")),
    ("Shield CUT2 5.5circle",
     os.path.join(RG58_DIR, "Shield-LineA+CUT2+LineB(5.5circle)-1.csv"),
     os.path.join(RG58_DIR, "Shield-LineA+CUT2+LineB(5.5circle)-2.csv")),
]

bands = [(0, 0.1e9), (0.1e9, 0.5e9), (0.5e9, 1e9), (1e9, 2e9)]
band_names = ["0-100MHz", "100-500MHz", "0.5-1GHz", "1-2GHz"]

all_noise_results = []

for label, p1, p2 in pairs:
    print(f"\n--- {label} ---")
    f1, S1 = load_s11(p1)
    f2, S2 = load_s11(p2)
    dS = S1 - S2
    abs_dS = np.abs(dS)
    abs_S = 0.5 * (np.abs(S1) + np.abs(S2))
    
    print(f"  Points: {len(f1)}, Freq range: {f1[0]/1e6:.1f} - {f1[-1]/1e6:.1f} MHz")
    print(f"  Overall |ΔS|: mean={np.mean(abs_dS):.6f}, std={np.std(abs_dS):.6f}, max={np.max(abs_dS):.6f}")
    print(f"  Overall |S11|: mean={np.mean(abs_S):.4f}")
    
    # Re/Im 独立性
    corr = np.corrcoef(np.real(dS), np.imag(dS))[0, 1]
    print(f"  Correlation(Re(ΔS), Im(ΔS)): {corr:.4f}")
    
    print(f"  {'Band':>12s} {'|ΔS| mean':>12s} {'|ΔS| std':>12s} {'|S11| mean':>12s} {'Rel noise':>12s}")
    for (flo, fhi), bname in zip(bands, band_names):
        mask = (f1 >= flo) & (f1 < fhi)
        if np.sum(mask) < 10:
            continue
        d = abs_dS[mask]
        s = abs_S[mask]
        rel = np.mean(d) / max(np.mean(s), 1e-10)
        print(f"  {bname:>12s} {np.mean(d):>12.6f} {np.std(d):>12.6f} {np.mean(s):>12.6f} {rel:>12.4f}")
        all_noise_results.append((label, bname, np.mean(d), np.std(d), np.mean(s), rel))

# 汇总
print("\n--- 汇总（所有对平均）---")
print(f"  {'Band':>12s} {'avg |ΔS| mean':>14s} {'avg |ΔS| std':>14s} {'avg |S11|':>14s} {'avg Rel':>10s}")
for _, bname, _, _, _, _ in [(None, bn, 0, 0, 0, 0) for bn in band_names]:
    vals = [r for r in all_noise_results if r[1] == bname]
    if vals:
        avg_d = np.mean([v[2] for v in vals])
        avg_s = np.mean([v[3] for v in vals])
        avg_sig = np.mean([v[4] for v in vals])
        avg_rel = np.mean([v[5] for v in vals])
        print(f"  {bname:>12s} {avg_d:>14.6f} {avg_s:>14.6f} {avg_sig:>14.6f} {avg_rel:>10.4f}")

# ─────────────────────────────────────────────
# PART 1a-extra: FDR 现场数据高频抖动
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("PART 1a-extra: FDR 现场数据高频抖动")
print("=" * 70)
if os.path.exists(FDR_REAL):
    ff, fS = load_s11(FDR_REAL)
    jitter = np.abs(np.diff(fS))
    print(f"  Points: {len(ff)}, Freq: {ff[0]/1e6:.1f} - {ff[-1]/1e6:.1f} MHz")
    bands_fdr = [(0, 0.1e9), (0.1e9, 0.5e9), (0.5e9, 1e9)]
    band_names_fdr = ["0-100MHz", "100-500MHz", "0.5-1GHz"]
    for (flo, fhi), bname in zip(bands_fdr, band_names_fdr):
        mid = (ff[:-1] + ff[1:]) / 2
        mask = (mid >= flo) & (mid < fhi)
        if np.sum(mask) < 10:
            continue
        j = jitter[mask]
        sig = 0.5 * (np.abs(fS[:-1][mask]) + np.abs(fS[1:][mask]))
        rel = np.mean(j) / max(np.mean(sig), 1e-10)
        print(f"  {bname:>12s}: jitter mean={np.mean(j):.6f}, std={np.std(j):.6f}, |S11|={np.mean(sig):.6f}, rel={rel:.4f}")

# ─────────────────────────────────────────────
# PART 1b: 接头反射特征 + CUT 段扰动分析
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("PART 1b: 接头反射特征 + CUT 段扰动分析")
print("=" * 70)

# 参考文件
ref_file = os.path.join(RG58_DIR, "Core-LineA+CUT1+LineB(20degree)-1.csv")
freq, S_ref = load_s11(ref_file)
dist, imp, step = quick_ifft(freq, S_ref, epsr=2.25)

imp_mag = np.abs(imp)
imp_real = np.real(imp)

# 端到截断到0-80m（RG58-74m电缆）
mask_80 = (dist >= 0) & (dist <= 80)
d80 = dist[mask_80]
im80 = imp_mag[mask_80]
ir80 = imp_real[mask_80]
st80 = step[mask_80]

# 找 top 峰
print(f"\n--- 参考: Core 20degree-1, 距离域 0-80m ---")
print(f"  距离点数: {len(d80)}, 步长: {d80[1]-d80[0]:.4f} m")

# 找峰（局部最大值，阈值=最大值*0.05）
from scipy.signal import find_peaks
peaks_idx, peaks_props = find_peaks(im80, height=np.max(im80)*0.05, distance=20)
print(f"  检测到 {len(peaks_idx)} 个峰:")
# 按 distance 排序
peak_data = [(d80[i], im80[i], ir80[i]) for i in peaks_idx]
peak_data.sort(key=lambda x: x[0])
for d_p, mag_p, real_p in peak_data[:10]:
    print(f"    d={d_p:.2f}m, |impulse|={mag_p:.6f}, Re(impulse)={real_p:.6f}")

# 末端反射（~74m）作为基准
end_peak = [p for p in peak_data if 70 < p[0] < 80]
if end_peak:
    end_amp = max(p[1] for p in end_peak)
    end_d = [p for p in end_peak if p[1] == end_amp][0][0]
    print(f"\n  末端反射基准: d={end_d:.2f}m, |imp|={end_amp:.6f}")
    
    # 接头特征分析
    joint_peaks = [p for p in peak_data if (38 < p[0] < 46)]
    print(f"\n  40-44m 区域接头峰:")
    for d_p, mag_p, real_p in joint_peaks:
        ratio = mag_p / end_amp if end_amp > 0 else 0
        sign = "+" if real_p > 0 else "-"
        # FWHM 近似
        half = mag_p / 2
        above = im80 >= half
        # 找连续区间
        # ...简化：打印
        print(f"    d={d_p:.2f}m, |imp|={mag_p:.6f}, Re={real_p:.6f} ({sign}), 相对末端={ratio:.4f}")
        
        # 粗略 FWHM
        # 从峰值向两侧找半高点
        pk_idx_local = np.argmin(np.abs(d80 - d_p))
        # 左侧
        left_idx = pk_idx_local
        while left_idx > 0 and im80[left_idx] > half:
            left_idx -= 1
        # 右侧
        right_idx = pk_idx_local
        while right_idx < len(im80) - 1 and im80[right_idx] > half:
            right_idx += 1
        fwhm = d80[right_idx] - d80[left_idx]
        print(f"      FWHM ≈ {fwhm:.2f} m")

# ─────────────────────────────────────────────
# CUT 扰动分析：温度效应
# ─────────────────────────────────────────────
print("\n\n--- CUT 温度扰动 (20°C vs 70°C) ---")
f20, S20 = load_s11(os.path.join(RG58_DIR, "Core-LineA+CUT1+LineB(20degree)-1.csv"))
f70, S70 = load_s11(os.path.join(RG58_DIR, "Core-LineA+CUT1+LineB(70degree)-1.csv"))
_, imp20, step20 = quick_ifft(f20, S20, epsr=2.25)
_, imp70, step70 = quick_ifft(f70, S70, epsr=2.25)

# 38-46m 区域
mask_cut = (dist >= 35) & (dist <= 50)
d_cut = dist[mask_cut]
imp_diff_mag = np.abs(np.abs(imp20[mask_cut]) - np.abs(imp70[mask_cut]))
imp_diff_real = np.abs(np.real(imp20[mask_cut]) - np.real(imp70[mask_cut]))
step_diff = np.abs(step20[mask_cut] - step70[mask_cut])

max_idx = np.argmax(imp_diff_mag)
print(f"  脉冲响应差(幅值): max={imp_diff_mag[max_idx]:.6f} at d={d_cut[max_idx]:.2f}m")
print(f"  脉冲响应差(实部): max={imp_diff_real.max():.6f}")
print(f"  阶跃响应差: peak-to-peak={step_diff.max():.6f}")

# ─────────────────────────────────────────────
# CUT 扰动分析：绕圈效应
# ─────────────────────────────────────────────
print("\n--- CUT 绕圈扰动 (0.5circle vs 14.5circle) ---")
f05, S05 = load_s11(os.path.join(RG58_DIR, "Core-LineA+CUT2+LineB(0.5circle)-1.csv"))
f145, S145 = load_s11(os.path.join(RG58_DIR, "Core-LineA+CUT2+LineB(14.5circle)-1.csv"))
_, imp05, step05 = quick_ifft(f05, S05, epsr=2.25)
_, imp145, step145 = quick_ifft(f145, S145, epsr=2.25)

imp_diff2_mag = np.abs(np.abs(imp05[mask_cut]) - np.abs(imp145[mask_cut]))
imp_diff2_real = np.abs(np.real(imp05[mask_cut]) - np.real(imp145[mask_cut]))
step_diff2 = np.abs(step05[mask_cut] - step145[mask_cut])

max_idx2 = np.argmax(imp_diff2_mag)
print(f"  脉冲响应差(幅值): max={imp_diff2_mag[max_idx2]:.6f} at d={d_cut[max_idx2]:.2f}m")
print(f"  脉冲响应差(实部): max={imp_diff2_real.max():.6f}")
print(f"  阶跃响应差: peak-to-peak={step_diff2.max():.6f}")

print("\n" + "=" * 70)
print("分析完毕")
print("=" * 70)
