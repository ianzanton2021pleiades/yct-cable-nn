# 三方对比：RG58-74M(40+4+30) 实测 vs ADS V1 vs DG V3(纯净)
# 实测数据: E:\FDR案例-csv\RG58-74M(40+4+30)\Core-LineA+CUT1+LineB(20degree)-1.csv（只读）
# 回答两个问题:
#   1) 哪个模型的底层实现更接近实测？
#   2) 末端脉冲/阶跃前肩形状差异（快上升+慢拖尾）的定量证据与成因
# 复用 compare_ads_vs_dgv3.py 中的模型与 IFFT 函数。

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import compare_ads_vs_dgv3 as base  # noqa: E402  (模块导入不会执行其 main)

REAL_CSV = Path(r"E:\FDR案例-csv\RG58-74M(40+4+30)\Core-LineA+CUT1+LineB(20degree)-1.csv")
REAL_CSV_2 = Path(r"E:\FDR案例-csv\RG58-74M(40+4+30)\Core-LineA+CUT1+LineB(20degree)-2.csv")
VF = base.VF_H

import matplotlib.pyplot as plt  # noqa: E402


def load_real(path: Path):
    data = np.genfromtxt(path, delimiter=",", skip_header=1, usecols=(0, 1, 2))
    data = data[np.isfinite(data).all(axis=1)]
    return data[:, 0], data[:, 1] + 1j * data[:, 2]


def interp_model(s_model_f, f_model, f_target):
    re = np.interp(f_target, f_model, s_model_f.real)
    im = np.interp(f_target, f_model, s_model_f.imag)
    return re + 1j * im


def end_peak(dist, imp, lo, hi):
    m = (dist >= lo) & (dist <= hi)
    i = np.argmax(np.abs(imp[m]))
    return float(dist[m][i]), float(imp[m][i])


def impulse_shape(dist, imp, pos):
    """末端峰形状：前肩(10%->90%上升距离)、后沿(90%->10%回落距离)、不对称比。"""
    def crossing(x, y, level, side):
        if side == "left":
            seg_x, seg_y = x[:pos_idx + 1][::-1], y[:pos_idx + 1][::-1]
        else:
            seg_x, seg_y = x[pos_idx:], y[pos_idx:]
        seg_y = seg_y / peak
        for k in range(len(seg_y) - 1):
            if (seg_y[k] - level) * (seg_y[k + 1] - level) <= 0 and seg_y[k] != seg_y[k + 1]:
                w = (level - seg_y[k]) / (seg_y[k + 1] - seg_y[k])
                return float(seg_x[k] + w * (seg_x[k + 1] - seg_x[k]))
        return None

    m = (dist >= pos - 10.0) & (dist <= pos + 40.0)
    x, y = dist[m], imp[m]
    pos_idx = int(np.argmax(np.abs(y)))
    peak = y[pos_idx]
    x10l, x90l = crossing(x, y, 0.1, "left"), crossing(x, y, 0.9, "left")
    x90r, x10r = crossing(x, y, 0.9, "right"), crossing(x, y, 0.1, "right")
    rise = (x90l - x10l) if (x90l and x10l) else None
    fall = (x10r - x90r) if (x90r and x10r) else None
    return {
        "peak_pos_m": float(x[pos_idx]), "peak_amp": float(peak),
        "rise_10_90_m": rise, "fall_90_10_m": fall,
        "asymmetry_fall_over_rise": (fall / rise) if (rise and fall) else None,
    }


def step_rise_width(dist, step, pos):
    """阶跃响应在末端的 10%~90% 上升宽度（米）。"""
    m = (dist >= pos + 5.0) & (dist <= pos + 30.0)
    plateau = float(np.median(step[m]))
    base0 = float(np.median(step[(dist >= pos - 30.0) & (dist <= pos - 10.0)]))
    seg = (dist >= pos - 15.0) & (dist <= pos + 15.0)
    x, y = dist[seg], (step[seg] - base0) / (plateau - base0)
    c10 = c90 = None
    for k in range(len(y) - 1):
        if c10 is None and (y[k] - 0.1) * (y[k + 1] - 0.1) <= 0 and y[k] != y[k + 1]:
            w = (0.1 - y[k]) / (y[k + 1] - y[k])
            c10 = x[k] + w * (x[k + 1] - x[k])
        if c10 is not None and c90 is None and (y[k] - 0.9) * (y[k + 1] - 0.9) <= 0 and y[k] != y[k + 1]:
            w = (0.9 - y[k]) / (y[k + 1] - y[k])
            c90 = x[k] + w * (x[k + 1] - x[k])
    return {
        "plateau": plateau,
        "step_10_90_width_m": float(c90 - c10) if (c10 and c90) else None,
    }


def align_delay(s_model, f, delay_s):
    """给模型加纯时延相位斜坡，把末端峰对齐到实测到达时刻。"""
    return s_model * np.exp(-1j * 2.0 * np.pi * f * delay_s)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    out = {}

    f_r, s_r = load_real(REAL_CSV)
    f_r2, s_r2 = load_real(REAL_CSV_2)
    rep = np.sqrt(np.mean(np.abs(s_r - interp_model(s_r2, f_r2, f_r)) ** 2))
    out["real_repeat_rmse_complex"] = float(rep)

    keep = f_r >= 100e3
    f, s_real = f_r[keep], s_r[keep]

    # 模型在密集同网格上计算后插值到实测网格（避免插值误差进入相位）
    f_dense = np.arange(f[0], f[-1] + 1.0, 10e3)
    s_ads_d = base.ads_s11(f_dense, 40.0, 4.0, 30.0)
    s_dg_d = base.dgv3_s11(f_dense, [
        (0.0, 40.0, 50.0, base.EPSR_H, base.ALPHA_H),
        (40.0, 44.0, 51.0, base.EPSR_A, base.ALPHA_A),
        (44.0, 74.0, 50.0, base.EPSR_H, base.ALPHA_H),
    ])
    s_ads = interp_model(s_ads_d, f_dense, f)
    s_dg = interp_model(s_dg_d, f_dense, f)

    # ---------- IFFT ----------
    d_r, imp_r, step_r = base.s11_to_impulse_step(f, s_real, VF)
    d_a, imp_a, step_a = base.s11_to_impulse_step(f, s_ads, VF)
    d_g, imp_g, step_g = base.s11_to_impulse_step(f, s_dg, VF)

    ev = {}
    for name, d, imp, step in (("real", d_r, imp_r, step_r),
                               ("ads", d_a, imp_a, step_a),
                               ("dgv3", d_g, imp_g, step_g)):
        end_pos, end_amp = end_peak(d, imp, 60.0, 95.0)
        j1_pos, j1_amp = end_peak(d, imp, 36.0, 42.5)
        j2_pos, j2_amp = end_peak(d, imp, 42.5, 50.0)
        ev[name] = {
            "end": {"pos_m": end_pos, "amp": end_amp,
                    **impulse_shape(d, imp, end_pos),
                    **step_rise_width(d, step, end_pos)},
            "joint_40m": {"pos_m": float(j1_pos), "amp": float(j1_amp)},
            "joint_44m": {"pos_m": float(j2_pos), "amp": float(j2_amp)},
        }
    out["ifft_events"] = ev

    # ---------- 时延对齐后的 S11 残差（去掉 VF 绝对偏差，比较"结构"一致性） ----------
    tau_a = 2.0 * (ev["real"]["end"]["pos_m"] - ev["ads"]["end"]["pos_m"]) / (base.C0 * VF)
    tau_g = 2.0 * (ev["real"]["end"]["pos_m"] - ev["dgv3"]["end"]["pos_m"]) / (base.C0 * VF)
    s_ads_al = align_delay(s_ads, f, tau_a)
    s_dg_al = align_delay(s_dg, f, tau_g)
    out["delay_shift_us"] = {"ads": float(tau_a * 1e6), "dgv3": float(tau_g * 1e6)}

    band = f <= 500e6  # 实测高频段被仪器误差本底主导，结构比较限在 500 MHz 以内
    res = {}
    for name, s_m in (("ads", s_ads_al), ("dgv3", s_dg_al)):
        d_amp = 20 * np.log10(np.abs(s_m[band])) - 20 * np.log10(np.abs(s_real[band]))
        d_ph = np.angle(s_m[band] / s_real[band]) * 180 / np.pi
        res[name] = {
            "band_hz": [float(f[band][0]), float(f[band][-1])],
            "amp_rms_db": float(np.sqrt(np.mean(d_amp ** 2))),
            "amp_max_db": float(np.abs(d_amp).max()),
            "phase_rms_deg": float(np.sqrt(np.mean(d_ph ** 2))),
            "complex_rms": float(np.sqrt(np.mean(np.abs(s_m[band] - s_real[band]) ** 2))),
        }
    out["aligned_residual_500mhz"] = res

    floor = float(np.median(20 * np.log10(np.abs(s_real[f > 1.5e9]))))
    end_level = {
        "real_floor_above_1.5ghz_db": floor,
        "ads_end_reflection_at_2ghz_db": float(20 * np.log10(np.abs(s_ads_d[-1]))),
        "dgv3_end_reflection_at_2ghz_db": float(20 * np.log10(np.abs(s_dg_d[-1]))),
    }
    out["high_freq_floor"] = end_level

    with open(base.RES_DIR / "real_metrics.json", "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)

    # ---------- 图 1：幅值三方对比 ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(f_r / 1e6, 20 * np.log10(np.abs(s_r)), lw=0.8, color="black", label="实测")
    ax.plot(f / 1e6, 20 * np.log10(np.abs(s_ads)), lw=0.9, color="tab:blue", label="ADS")
    ax.plot(f / 1e6, 20 * np.log10(np.abs(s_dg)), lw=0.9, ls="--", color="tab:red", label="DG V3 clean")
    ax.axhline(floor, color="grey", lw=0.7, ls=":")
    ax.annotate(f"实测误差本底≈{floor:.1f} dB", (1200, floor + 3), fontsize=8, color="grey")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("|S11| (dB)")
    ax.set_title("(a) 幅值：实测 / ADS / DG V3")
    ax.set_xlim(0, 2000)
    ax.legend(loc="upper right", fontsize=8)
    ax = axes[1]
    ax.plot(f[band] / 1e6, 20 * np.log10(np.abs(s_ads_al[band])) - 20 * np.log10(np.abs(s_real[band])),
            lw=0.9, color="tab:blue", label="ADS−实测")
    ax.plot(f[band] / 1e6, 20 * np.log10(np.abs(s_dg_al[band])) - 20 * np.log10(np.abs(s_real[band])),
            lw=0.9, ls="--", color="tab:red", label="DG−实测")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Δ|S11| (dB，时延对齐后)")
    ax.set_title("(b) 对齐后幅值残差（≤500 MHz）")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(base.FIG_DIR / "real_s11_compare.png")
    plt.close(fig)

    # ---------- 图 2：IFFT 三方对比 ----------
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    norm = float(np.abs(imp_r).max())
    ax = axes[0, 0]
    ax.plot(d_r, imp_r / norm, lw=1.0, color="black", label="实测")
    ax.plot(d_a, imp_a / norm, lw=0.9, color="tab:blue", label="ADS")
    ax.plot(d_g, imp_g / norm, lw=0.9, ls="--", color="tab:red", label="DG V3 clean")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("脉冲响应（按实测峰值归一）")
    ax.set_title("(a) 脉冲响应（含近端区）")
    ax.set_xlim(0, 90)
    ax.legend(loc="upper right", fontsize=8)
    ax = axes[0, 1]
    ax.plot(d_r, imp_r / norm, lw=1.0, color="black", label="实测")
    ax.plot(d_a, imp_a / norm, lw=0.9, color="tab:blue", label="ADS")
    ax.plot(d_g, imp_g / norm, lw=0.9, ls="--", color="tab:red", label="DG V3 clean")
    ax.set_xlabel("Distance (m)")
    ax.set_title("(b) 末端峰放大（前肩形状比较）")
    ax.set_xlim(64, 86)
    ax.legend(loc="upper right", fontsize=8)
    ax = axes[1, 0]
    ax.plot(d_r, step_r, lw=1.0, color="black", label="实测")
    ax.plot(d_a, step_a, lw=0.9, color="tab:blue", label="ADS")
    ax.plot(d_g, step_g, lw=0.9, ls="--", color="tab:red", label="DG V3 clean")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("阶跃响应")
    ax.set_title("(c) 阶跃响应")
    ax.set_xlim(0, 90)
    ax.legend(loc="lower right", fontsize=8)
    ax = axes[1, 1]
    ax.plot(d_r, step_r, lw=1.0, color="black", label="实测")
    ax.plot(d_a, step_a, lw=0.9, color="tab:blue", label="ADS")
    ax.plot(d_g, step_g, lw=0.9, ls="--", color="tab:red", label="DG V3 clean")
    ax.set_xlabel("Distance (m)")
    ax.set_title("(d) 末端阶跃放大（上升沿比较）")
    ax.set_xlim(64, 86)
    ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("RG58-74M(40+4+30) 实测 vs ADS vs DG V3（纯净）—— IFFT 三方对比", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(base.FIG_DIR / "real_ifft_compare.png")
    plt.close(fig)

    # ---------- 打印 ----------
    print("=" * 76)
    print(f"实测文件: {REAL_CSV.name}（重复测试 RMS 差异 {rep:.4f}）")
    print(f"实测扫频: {f_r[0]/1e3:.1f} kHz ~ {f_r[-1]/1e9:.2f} GHz, df≈{(f_r[1]-f_r[0])/1e3:.0f} kHz")
    for name in ("real", "ads", "dgv3"):
        e = ev[name]["end"]
        j1, j2 = ev[name]["joint_40m"], ev[name]["joint_44m"]
        print(f"\n[{name}] 末端峰 {e['pos_m']:.2f} m, 幅值 {e['peak_amp']:.3e}")
        print(f"   前肩10-90%={e['rise_10_90_m']} m, 后沿90-10%={e['fall_90_10_m']} m, "
              f"不对称比={e['asymmetry_fall_over_rise']}")
        print(f"   阶跃平台={e['plateau']:.3f}, 阶跃10-90%宽度={e['step_10_90_width_m']} m")
        print(f"   接头峰: 40m区 {j1['pos_m']:.2f} m ({j1['amp']:.2e}), 44m区 {j2['pos_m']:.2f} m ({j2['amp']:.2e})")
    print("\n时延对齐量(相对实测): ADS %.3f us, DG %.3f us" %
          (out["delay_shift_us"]["ads"], out["delay_shift_us"]["dgv3"]))
    for name in ("ads", "dgv3"):
        r = res[name]
        print(f"[{name}] 对齐后残差(≤500MHz): 幅值RMS={r['amp_rms_db']:.2f} dB, "
              f"相位RMS={r['phase_rms_deg']:.2f} deg, 复数RMS={r['complex_rms']:.4f}")
    print("\n高频本底: 实测>1.5GHz中位 |S11| = %.1f dB; 模型末端反射@2GHz: ADS %.1f dB, DG %.1f dB" %
          (end_level["real_floor_above_1.5ghz_db"],
           end_level["ads_end_reflection_at_2ghz_db"],
           end_level["dgv3_end_reflection_at_2ghz_db"]))
    print("\n结果: results/real_metrics.json, 图: figures/real_*.png")


if __name__ == "__main__":
    main()
