# ADS V1 参考仿真程序 与 DG V3 纯净物理模型 的对比试验
# 环境: E:\Anaconda\envs\gpushare_cu124\python.exe (Python 3.11)
# 用法: python compare_ads_vs_dgv3.py
# 输出: results/*.csv, results/metrics.json, figures/*.png

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
FIG_DIR = SCRIPT_DIR / "figures"
RES_DIR = SCRIPT_DIR / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

# ---------- 导入 ADS 参考程序（REF，不修改原文件） ----------
_ADS_PATH = ROOT / "REF" / "[ADS_V1]v3.3_74m_s11_generator.py"
_spec = importlib.util.spec_from_file_location("ads_ref", _ADS_PATH)
ads = importlib.util.module_from_spec(_spec)
sys.modules["ads_ref"] = ads
_spec.loader.exec_module(ads)

# ---------- 导入 DG V3（DG_Update/DG_V3，不修改原文件） ----------
sys.path.insert(0, str(ROOT / "DG_Update" / "DG_V3"))
from dg_v3.topology import CableSegment, CableTopology          # noqa: E402
from dg_v3.physics import topology_abcd, network_s11             # noqa: E402

# ---------- 作图设置（Times New Roman + 黑体，刻度朝内，DPI 200） ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

for _p in (r"C:\Windows\Fonts\times.ttf", r"C:\Windows\Fonts\simhei.ttf"):
    if os.path.exists(_p):
        fm.fontManager.addfont(_p)
plt.rcParams["font.family"] = ["Times New Roman", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["font.size"] = 9
plt.rcParams["axes.linewidth"] = 0.8

import warnings                                                   # noqa: E402
warnings.filterwarnings("ignore", message="Glyph.*missing from")

C0 = 299_792_458.0
Z_REF = 50.0
Z_OPEN = 1e13

# ADS 默认两段材料参数（healthy / aged），直接沿用参考程序的目标值
ADS_DEFAULT = ads.default_config()
HEALTHY_CFG = ADS_DEFAULT.healthy
AGED_CFG = ADS_DEFAULT.aged
VF_H = HEALTHY_CFG.vf_target                     # 0.67055
EPSR_H = 1.0 / VF_H ** 2
ALPHA_H = HEALTHY_CFG.alpha_target_db_per_m_at_fref
VF_A = AGED_CFG.vf_target
EPSR_A = 1.0 / VF_A ** 2
ALPHA_A = AGED_CFG.alpha_target_db_per_m_at_fref

TAN_DELTA_CLEAN = 1e-12          # DG V3 纯净模式：介电损耗近似为零
DISPERSION_CLEAN = 0.0           # DG V3 纯净模式：不加色散扰动


# ================= ADS 侧 =================
def ads_s11(freq_hz: np.ndarray, len1_m: float, len_age_m: float, len3_m: float) -> np.ndarray:
    """直接调用 ADS 参考程序的三段递推（长度置 0 即退化为均匀线）。"""
    cfg = ads.Cable74mConfig(
        healthy=HEALTHY_CFG, aged=AGED_CFG,
        z_ref_ohm=Z_REF, z_load_open_ohm=Z_OPEN,
        len1_m=len1_m, len_age_m=len_age_m, len3_m=len3_m,
    )
    return ads.compute_s11_74m(freq_hz, cfg)


# ================= DG V3 侧 =================
def dgv3_s11(freq_hz: np.ndarray, segments: list[tuple[float, float, float, float, float]]) -> np.ndarray:
    """DG V3 纯净物理链：topology_abcd + network_s11，不加夹具 / VNA误差 / 噪声。

    segments: [(start_m, end_m, z0_ohm, epsr, alpha_db_per_m_at_100mhz), ...]
    """
    seg_objs = [
        CableSegment(start, end, z0, epsr, alpha, TAN_DELTA_CLEAN)
        for start, end, z0, epsr, alpha in segments
    ]
    total_len = seg_objs[-1].end_m
    topo = CableTopology(
        profile="rg58", length_m=total_len, z_ref_ohm=Z_REF,
        base_z0_ohm=seg_objs[0].z0_ohm, base_epsr=seg_objs[0].epsr,
        base_alpha_db_per_m_at_100mhz=seg_objs[0].alpha_db_per_m_at_100mhz,
        base_tan_delta_at_100mhz=TAN_DELTA_CLEAN,
        dispersion_fraction=DISPERSION_CLEAN,
        segments=seg_objs, joints=[], termination="open",
        z_load_ohm=Z_OPEN, defect_regions=[],
    )
    abcd = topology_abcd(freq_hz, topo)
    return network_s11(abcd, Z_OPEN, Z_REF)


# ================= IFFT（照抄 REF 参考程序的算法，矩形窗、无 padding、DC 外推） ==========
def _fft_shift(arr, inverse=False):
    n = len(arr)
    k = n // 2
    return np.concatenate((arr[k:], arr[:k])) if inverse else np.concatenate((arr[-k:], arr[:-k]))


def build_equally_spaced_spectrum(freqs, S, automatic_dc=True):
    freqs = np.asarray(freqs, float)
    S = np.asarray(S, complex)
    order = np.argsort(freqs)
    freqs_sorted = freqs[order]
    S_sorted = S[order]
    df = np.diff(freqs_sorted)
    df = df[df > 0]
    firstStep = float(np.percentile(df, 5))
    if firstStep < np.mean(df) / 10:
        firstStep = float(np.mean(df))
    fmax = freqs_sorted[-1]
    steps = int(np.floor(fmax / firstStep))
    N = 2 * steps + 1
    f_lin = np.arange(0, steps + 1, dtype=float) * firstStep
    S_lin = np.interp(f_lin, freqs_sorted, S_sorted.real) + 1j * np.interp(f_lin, freqs_sorted, S_sorted.imag)
    spectrum = np.zeros(N, dtype=np.complex128)
    for i in range(1, steps + 1):
        spectrum[steps + i] = S_lin[i]
        spectrum[steps - i] = np.conj(S_lin[i])
    if automatic_dc and steps > 2:
        abs_dc = 2 * np.abs(spectrum[steps + 1]) - np.abs(spectrum[steps + 2])
        pha_dc = 2 * np.angle(spectrum[steps + 1]) - np.angle(spectrum[steps + 2])
        spectrum[steps] = abs_dc * np.exp(1j * pha_dc)
    else:
        spectrum[steps] = S_lin[0]
    return spectrum, firstStep, N


def s11_to_impulse_step(freqs, S, vf):
    """返回 (distance_m, impulse, step)，均为真实物理标定。

    np.fft.ifft 输出 td = s(t)·dt，故脉冲响应 = Re(td)/dt，
    阶跃响应 = Re(cumsum(td))（REF GUI 里的 *dt 因子被它自身的
    最大幅值归一化抵消，这里不做归一化，需显式去掉）。
    """
    spectrum, firstStep, N = build_equally_spaced_spectrum(freqs, S)
    spec = _fft_shift(spectrum, inverse=True)
    td = np.fft.ifft(spec)
    dt = 1.0 / (firstStep * N)
    t = np.arange(N) * dt
    distance = C0 * vf * t / 2.0
    impulse = np.real(td) / dt
    step = np.real(np.cumsum(td))
    return distance, impulse, step


def step_to_impedance(step, z_ref=Z_REF):
    """按 REF 的方式把阶跃响应换算成阻抗剖面。"""
    baseline = np.mean(step[: max(1, int(0.05 * len(step)))])
    gamma = np.clip(step - baseline, -0.9999, 0.9999)
    return z_ref * (1.0 + gamma) / (1.0 - gamma), baseline


# ================= 测试案例 =================
CASES = [
    {
        "name": "A_74m_uniform",
        "title": "案例A：74 m 均匀完好线",
        "length": 74.0, "df": 100e3,
        "ads_lens": (0.0, 0.0, 74.0),
        "dgv3_segs": [(0.0, 74.0, 50.0, EPSR_H, ALPHA_H)],
        "events": [(74.0, "末端")],
    },
    {
        "name": "B_74m_40p4p30",
        "title": "案例B：74 m = 40 m完好 + 4 m老化 + 30 m完好（ADS默认拓扑）",
        "length": 74.0, "df": 100e3,
        "ads_lens": (40.0, 4.0, 30.0),
        "dgv3_segs": [
            (0.0, 40.0, 50.0, EPSR_H, ALPHA_H),
            (40.0, 44.0, 51.0, EPSR_A, ALPHA_A),
            (44.0, 74.0, 50.0, EPSR_H, ALPHA_H),
        ],
        "events": [(40.0, "老化段前端"), (44.0, "老化段后端"), (74.0, "末端")],
    },
    {
        "name": "C_500m_uniform",
        "title": "案例C：500 m 均匀完好线",
        "length": 500.0, "df": 100e3,
        "ads_lens": (0.0, 0.0, 500.0),
        "dgv3_segs": [(0.0, 500.0, 50.0, EPSR_H, ALPHA_H)],
        "events": [(500.0, "末端")],
    },
    {
        "name": "D_1500m_uniform",
        "title": "案例D：1500 m 均匀完好线",
        "length": 1500.0, "df": 50e3,
        "ads_lens": (0.0, 0.0, 1500.0),
        "dgv3_segs": [(0.0, 1500.0, 50.0, EPSR_H, ALPHA_H)],
        "events": [(1500.0, "末端")],
    },
    {
        "name": "E_2400m_uniform",
        "title": "案例E：2400 m 均匀完好线",
        "length": 2400.0, "df": 30e3,
        "ads_lens": (0.0, 0.0, 2400.0),
        "dgv3_segs": [(0.0, 2400.0, 50.0, EPSR_H, ALPHA_H)],
        "events": [(2400.0, "末端")],
    },
]


# ================= 指标 =================
def wrapped_phase_diff_deg(s_dg, s_ads):
    return np.angle(s_dg / s_ads) * 180.0 / np.pi


def event_peak(distance, impulse, pos_m, win_m=3.0):
    mask = (distance >= pos_m - win_m) & (distance <= pos_m + win_m)
    seg = np.abs(impulse[mask])
    idx = np.argmax(seg)
    d = distance[mask][idx]
    return float(d), float(impulse[mask][idx])


# ================= 单案例处理 =================
def run_case(case):
    f = np.arange(case["df"], 1e9 + case["df"] / 2, case["df"])
    len1, len_a, len3 = case["ads_lens"]
    s_ads = ads_s11(f, len1, len_a, len3)
    s_dg = dgv3_s11(f, case["dgv3_segs"])

    amp_ads_db = 20.0 * np.log10(np.abs(s_ads))
    amp_dg_db = 20.0 * np.log10(np.abs(s_dg))
    ph_ads = np.angle(s_ads, deg=True)
    ph_dg = np.angle(s_dg, deg=True)
    d_amp_db = amp_dg_db - amp_ads_db
    d_ph = wrapped_phase_diff_deg(s_dg, s_ads)
    d_re = s_dg.real - s_ads.real
    d_abs = np.abs(s_dg - s_ads)

    metrics = {
        "title": case["title"],
        "length_m": case["length"],
        "freq_points": int(f.size),
        "s11": {
            "max_abs_diff": float(d_abs.max()),
            "rms_abs_diff": float(np.sqrt(np.mean(d_abs ** 2))),
            "max_real_diff_fullband": float(np.abs(d_re).max()),
            "amp_diff_at_100mhz_db": float(d_amp_db[np.argmin(np.abs(f - 100e6))]),
            "phase_diff_at_100mhz_deg": float(d_ph[np.argmin(np.abs(f - 100e6))]),
        },
    }

    # 公共可见频带：两个模型的反射都还没有被各自衰减模型"埋掉"（|S11| > -60 dB）
    common = (amp_ads_db > -60.0) & (amp_dg_db > -60.0)
    if np.any(common):
        fc = f[common]
        idx_30 = np.nonzero(np.abs(d_ph) > 30.0)[0]
        metrics["s11"]["common_band"] = {
            "start_hz": float(fc[0]),
            "stop_hz": float(fc[-1]),
            "max_amp_diff_db": float(np.abs(d_amp_db[common]).max()),
            "rms_amp_diff_db": float(np.sqrt(np.mean(d_amp_db[common] ** 2))),
            "max_phase_diff_deg": float(np.abs(d_ph[common]).max()),
            "rms_phase_diff_deg": float(np.sqrt(np.mean(d_ph[common] ** 2))),
            "phase_30deg_hz": float(f[idx_30[0]]) if idx_30.size else None,
            "max_real_diff": float(np.abs(d_re[common]).max()),
            "rms_abs_diff": float(np.sqrt(np.mean(d_abs[common] ** 2))),
        }
    else:
        metrics["s11"]["common_band"] = None

    np.savetxt(
        RES_DIR / f"{case['name']}_s11.csv",
        np.column_stack([f, s_ads.real, s_ads.imag, s_dg.real, s_dg.imag]),
        delimiter=",",
        header="Frequency_Hz,ADS_Re,ADS_Im,DGV3_Re,DGV3_Im",
        comments="",
        fmt="%.10g",
    )

    # ---------- S11 对比图 ----------
    fmhz = f / 1e6
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    ax = axes[0, 0]
    ax.plot(fmhz, s_ads.real, lw=1.0, color="tab:blue", label="ADS")
    ax.plot(fmhz, s_dg.real, lw=0.8, ls="--", color="tab:red", label="DG V3 clean")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("S11 实部")
    ax.set_title("(a) S11 实部")
    ax.legend(loc="best", fontsize=8)
    ax = axes[0, 1]
    ax.plot(fmhz, amp_ads_db, lw=1.0, color="tab:blue", label="ADS")
    ax.plot(fmhz, amp_dg_db, lw=0.8, ls="--", color="tab:red", label="DG V3 clean")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("|S11| (dB)")
    ax.set_title("(b) S11 幅值")
    ax.legend(loc="best", fontsize=8)
    ax = axes[1, 0]
    ax.plot(fmhz, ph_ads, lw=1.0, color="tab:blue", label="ADS")
    ax.plot(fmhz, ph_dg, lw=0.8, ls="--", color="tab:red", label="DG V3 clean")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("相位 (deg)")
    ax.set_title("(c) S11 相位（卷绕）")
    ax.legend(loc="best", fontsize=8)
    ax = axes[1, 1]
    ax.plot(fmhz, d_amp_db, lw=0.9, color="tab:purple", label="Δ幅值 (DG−ADS)")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Δ|S11| (dB)", color="tab:purple")
    ax.tick_params(axis="y", labelcolor="tab:purple")
    ax.set_title("(d) 差异 (DG V3 − ADS)")
    ax2 = ax.twinx()
    ax2.plot(fmhz, d_ph, lw=0.9, color="tab:green", label="Δ相位")
    ax2.set_ylabel("Δ相位 (deg)", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)
    fig.suptitle(case["title"] + " —— S11 对比", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG_DIR / f"{case['name']}_s11.png")
    plt.close(fig)

    # ---------- IFFT ----------
    dist_a, imp_a, step_a = s11_to_impulse_step(f, s_ads, VF_H)
    dist_d, imp_d, step_d = s11_to_impulse_step(f, s_dg, VF_H)
    n = min(len(dist_a), len(dist_d))
    dist, imp_a, imp_d = dist_a[:n], imp_a[:n], imp_d[:n]
    step_a, step_d = step_a[:n], step_d[:n]
    z_a, _ = step_to_impedance(step_a)
    z_d, _ = step_to_impedance(step_d)

    ev_metrics = []
    for pos, label in case["events"]:
        pa, va = event_peak(dist, imp_a, pos)
        pd_, vd = event_peak(dist, imp_d, pos)
        ev_metrics.append({
            "event": label, "true_pos_m": pos,
            "ads_peak_pos_m": pa, "ads_peak_amp": va,
            "dg_peak_pos_m": pd_, "dg_peak_amp": vd,
            "pos_diff_m": pd_ - pa,
            "amp_ratio_dg_over_ads": (vd / va) if va != 0 else None,
        })
    metrics["ifft"] = {
        "max_impulse_abs_diff": float(np.abs(imp_d - imp_a).max()),
        "max_step_abs_diff": float(np.abs(step_d - step_a).max()),
    }
    # 阶跃在末端"之后"才升到平台，取末端后 3 m 处的值评估平台幅值
    step_eval_pos = min(case["length"] + 3.0, float(dist[-1]) - 0.1)
    idx_eval = int(np.argmin(np.abs(dist - step_eval_pos)))
    z_a, _ = step_to_impedance(step_a)
    z_d, _ = step_to_impedance(step_d)
    metrics["ifft"].update({
        "ads_end_step_gamma": float(step_a[idx_eval]),
        "dg_end_step_gamma": float(step_d[idx_eval]),
        "ads_end_impedance_ohm": float(z_a[idx_eval]),
        "dg_end_impedance_ohm": float(z_d[idx_eval]),
        "events": ev_metrics,
    })

    np.savetxt(
        RES_DIR / f"{case['name']}_ifft.csv",
        np.column_stack([dist, imp_a, imp_d, step_a, step_d]),
        delimiter=",",
        header="Distance_m,ADS_impulse,DGV3_impulse,ADS_step,DGV3_step",
        comments="",
        fmt="%.10g",
    )

    # ---------- IFFT 对比图 ----------
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    imp_norm = float(np.abs(imp_a).max()) or 1.0
    ax = axes[0, 0]
    ax.plot(dist, imp_a / imp_norm, lw=1.0, color="tab:blue", label="ADS")
    ax.plot(dist, imp_d / imp_norm, lw=0.8, ls="--", color="tab:red", label="DG V3 clean")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("脉冲响应（按ADS峰值归一）")
    ax.set_title("(a) IFFT 脉冲响应")
    ax.legend(loc="best", fontsize=8)
    ax = axes[0, 1]
    ax.plot(dist, step_a, lw=1.0, color="tab:blue", label="ADS")
    ax.plot(dist, step_d, lw=0.8, ls="--", color="tab:red", label="DG V3 clean")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("阶跃响应 Re")
    ax.set_title("(b) IFFT 阶跃响应")
    ax.legend(loc="best", fontsize=8)
    ax = axes[1, 0]
    ax.plot(dist, imp_d - imp_a, lw=0.9, color="tab:purple")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Δ脉冲响应")
    ax.set_title("(c) 脉冲响应差异 (DG−ADS)")
    ax = axes[1, 1]
    ax.plot(dist, step_d - step_a, lw=0.9, color="tab:purple")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Δ阶跃响应")
    ax.set_title("(d) 阶跃响应差异 (DG−ADS)")
    xlim = min(float(dist[-1]), case["length"] * 1.6)
    for ax in axes.flat:
        ax.set_xlim(0.0, xlim)
    fig.suptitle(case["title"] + " —— IFFT 对比（矩形窗，距离按 VF=0.67055 换算）", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG_DIR / f"{case['name']}_ifft.png")
    plt.close(fig)

    return metrics


# ================= 汇总图 =================
def summary_plot(all_metrics):
    fq = np.linspace(1e6, 1e9, 400)
    geom_h = ads.target_to_effective_geometry(HEALTHY_CFG)
    R, L, G, C = ads.calc_primary_params(fq, geom_h, HEALTHY_CFG)
    z0_ads, gamma_ads = ads.calc_z0_gamma(R, L, G, C, fq)
    alpha_ads = np.real(gamma_ads) * 8.686
    vf_ads = 2.0 * np.pi * fq / (np.imag(gamma_ads) * C0)
    ratio = fq / 100e6
    alpha_dg = ALPHA_H * (0.35 * np.sqrt(ratio) + 0.65 * ratio)

    names = [m["title"].split("：")[0] for m in all_metrics]
    common = [m["s11"]["common_band"] for m in all_metrics]
    max_amp = [c["max_amp_diff_db"] if c else 0.0 for c in common]
    max_ph = [c["max_phase_diff_deg"] if c else 0.0 for c in common]
    max_imp = [m["ifft"]["max_impulse_abs_diff"] for m in all_metrics]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    ax = axes[0, 0]
    ax.plot(fq / 1e6, alpha_ads, lw=1.2, color="tab:blue", label="ADS（RLGC 完整模型）")
    ax.plot(fq / 1e6, alpha_dg, lw=1.2, ls="--", color="tab:red", label="DG V3 经验公式")
    ax.axvline(100.0, color="grey", lw=0.7, ls=":")
    ax.annotate("100 MHz 对齐点", (100, ax.get_ylim()[1] * 0.85), fontsize=8, color="grey")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("衰减系数 (dB/m)")
    ax.set_title("(a) 完好段衰减系数 α(f) 模型对比")
    ax.legend(loc="best", fontsize=8)
    ax = axes[0, 1]
    ax.plot(fq / 1e6, vf_ads, lw=1.2, color="tab:blue", label="ADS（β 反算）")
    ax.axhline(VF_H, color="tab:red", lw=1.2, ls="--", label=f"DG V3 恒定 VF={VF_H:.5f}")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("速度比 VF")
    ax.set_title("(b) 相速度（色散）对比")
    ax.legend(loc="best", fontsize=8)
    ax = axes[1, 0]
    ax.bar(names, max_amp, color="tab:purple", alpha=0.8)
    ax.set_ylabel("max |Δ|S11|| (dB)")
    ax.set_title("(c) 公共可见频带内 S11 幅值最大偏差")
    ax = axes[1, 1]
    ax.bar(names, max_ph, color="tab:green", alpha=0.8)
    ax.set_ylabel("max |Δ相位| (deg)")
    ax.set_title("(d) 公共可见频带内 S11 相位最大偏差")
    for axi, vals in ((axes[1, 0], max_amp), (axes[1, 1], max_ph)):
        for tick, v in zip(axi.get_xticklabels(), vals):
            tick.set_fontsize(8)
    imp_txt = ", ".join(f"{nm}:{v:.2e}" for nm, v in zip(names, max_imp))
    fig.suptitle(f"ADS vs DG V3（纯净）汇总 —— 脉冲响应最大偏差  {imp_txt}", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_DIR / "summary.png")
    plt.close(fig)


def model_form_table():
    """两模型在关键频点的衰减系数与相速度（VF），用于结论分析。"""
    fq = np.array([1e5, 1e6, 1e7, 1e8, 3e8, 1e9])
    geom_h = ads.target_to_effective_geometry(HEALTHY_CFG)
    R, L, G, C = ads.calc_primary_params(fq, geom_h, HEALTHY_CFG)
    _, gamma_ads = ads.calc_z0_gamma(R, L, G, C, fq)
    alpha_ads = (np.real(gamma_ads) * 8.686).tolist()
    vf_ads = (2.0 * np.pi * fq / (np.imag(gamma_ads) * C0)).tolist()
    ratio = fq / 100e6
    alpha_dg = (ALPHA_H * (0.35 * np.sqrt(ratio) + 0.65 * ratio)).tolist()
    return {
        "freq_hz": fq.tolist(),
        "ads_alpha_db_per_m": alpha_ads,
        "dgv3_alpha_db_per_m": alpha_dg,
        "ads_vf": vf_ads,
        "dgv3_vf": [VF_H] * len(fq),
        "ads_z0_at_100mhz_ohm": float(ads.diagnostics_at(100e6, HEALTHY_CFG)["z0_ohm"]),
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    all_metrics = []
    print("=" * 78)
    print("ADS V1 vs DG V3(纯净) 对比试验")
    print(f"ADS 完好段目标: Z0=50 Ω, VF={VF_H:.5f}, alpha={ALPHA_H} dB/m@100MHz")
    print(f"ADS 老化段目标: Z0=51 Ω, VF={VF_A:.5f}, alpha={ALPHA_A:.4f} dB/m@100MHz")
    dh = ads.diagnostics_at(100e6, HEALTHY_CFG)
    da = ads.diagnostics_at(100e6, AGED_CFG)
    print(f"ADS 反求几何(完好): rc={dh['rc_mm']:.4f} mm, rs={dh['rs_mm']:.4f} mm, "
          f"epsr={dh['epsr_eff']:.6f}, Z0={dh['z0_ohm']:.4f} Ω, VF={dh['vf']:.6f}")
    print(f"ADS 反求几何(老化): rc={da['rc_mm']:.4f} mm, rs={da['rs_mm']:.4f} mm, "
          f"epsr={da['epsr_eff']:.6f}, Z0={da['z0_ohm']:.4f} Ω, VF={da['vf']:.6f}")
    mf = model_form_table()
    with open(RES_DIR / "model_form.json", "w", encoding="utf-8") as fp:
        json.dump(mf, fp, ensure_ascii=False, indent=2)
    print("\n模型形式参考（完好段）: ")
    for i, fq in enumerate(mf["freq_hz"]):
        print(f"  f={fq:9.0f} Hz  alpha: ADS={mf['ads_alpha_db_per_m'][i]:.5f} / "
              f"DG={mf['dgv3_alpha_db_per_m'][i]:.5f} dB/m   "
              f"VF: ADS={mf['ads_vf'][i]:.6f} / DG={VF_H:.6f}")
    print("=" * 78)
    for case in CASES:
        print(f"\n>>> {case['title']} (df={case['df']/1e3:.0f} kHz)")
        m = run_case(case)
        all_metrics.append(m)
        s = m["s11"]
        cb = s["common_band"]
        print(f"    S11全带: max|Δ复数|={s['max_abs_diff']:.3e}  max|ΔRe|={s['max_real_diff_fullband']:.3e}")
        print(f"      100MHz处: Δ幅值={s['amp_diff_at_100mhz_db']:.4f} dB, "
              f"Δ相位={s['phase_diff_at_100mhz_deg']:.2f} deg")
        if cb:
            p30 = cb["phase_30deg_hz"]
            p30_txt = f"{p30/1e6:.2f} MHz" if p30 else ">频带上限"
            print(f"      公共可见带 {cb['start_hz']/1e6:.2f}~{cb['stop_hz']/1e6:.1f} MHz: "
                  f"max|Δ幅值|={cb['max_amp_diff_db']:.3f} dB, max|Δ相位|={cb['max_phase_diff_deg']:.2f} deg, "
                  f"max|ΔRe|={cb['max_real_diff']:.3e}")
            print(f"      Δ相位首次超30°的频率: {p30_txt}")
        i = m["ifft"]
        print(f"    IFFT: max|Δ脉冲|={i['max_impulse_abs_diff']:.3e}  max|Δ阶跃|={i['max_step_abs_diff']:.3e}")
        print(f"      末端阶跃平台: ADS={i['ads_end_step_gamma']:.4f}, DG={i['dg_end_step_gamma']:.4f} "
              f"(换算阻抗 {i['ads_end_impedance_ohm']:.1f} vs {i['dg_end_impedance_ohm']:.1f} Ω)")
        for ev in i["events"]:
            print(f"      事件[{ev['event']}] 位置 ADS={ev['ads_peak_pos_m']:.2f} m / DG={ev['dg_peak_pos_m']:.2f} m, "
                  f"峰幅比 DG/ADS={ev['amp_ratio_dg_over_ads']:.4f}")
    with open(RES_DIR / "metrics.json", "w", encoding="utf-8") as fp:
        json.dump(all_metrics, fp, ensure_ascii=False, indent=2)
    summary_plot(all_metrics)
    print("\n完成：结果在 results/，图在 figures/")


if __name__ == "__main__":
    main()
