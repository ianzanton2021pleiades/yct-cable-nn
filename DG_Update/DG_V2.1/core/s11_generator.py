"""Broadband cascaded-transmission-line S11 generator for DG V2.1.

The core model uses frequency-dependent RLGC parameters, conductor skin loss,
dielectric loss tangent, and optional Debye dielectric dispersion.  Fixture,
template-statistics, and terminal visibility effects are added by the dataset
generator in S11 domain; this module remains the physical cable backbone.

The standard full sweep is 9 kHz–1 GHz with 50,000 linearly spaced points.
"""
from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

# ═══════════════════════════════════════════════
# 物理常数（与参考 ADS 生成器一致）
# ═══════════════════════════════════════════════
PI = math.pi
MU0 = 4.0e-7 * PI
EPS0 = 8.854_187_817e-12
C0 = 299_792_458.0


# ═══════════════════════════════════════════════
# 配置数据类
# ═══════════════════════════════════════════════

@dataclass
class SweepConfig:
    """频率扫频配置"""
    start_hz: float = 9e3
    stop_hz: float = 1e9
    n_points: int = 50000  # DG V2.1 full-sweep standard

    def frequencies(self) -> np.ndarray:
        return np.linspace(self.start_hz, self.stop_hz, self.n_points)


@dataclass
class SegmentParams:
    """单段电缆参数"""
    length_m: float
    z0_ohm: float = 50.0
    epsr: float = 2.23
    alpha_db_per_m_100mhz: float = 0.14
    is_defect: bool = False
    sigma_cu: float = 5.89107e7
    sigma_dielectric: float = 1e-12
    # Broadband dielectric-loss / dispersion terms.  These are kept explicit
    # so moisture and aging are represented in S11 itself rather than by
    # post-editing the IFFT curves.
    tan_delta_100mhz: float = 2.5e-4
    debye_delta_epsr: float = 0.0
    debye_corner_hz: float = 80e6
    debye_exponent: float = 1.0


@dataclass
class CableSample:
    """单个合成电缆样本的完整描述"""
    segments: List[SegmentParams]
    epsr: float = 2.23
    z_ref: float = 50.0
    z_load_open: float = 1e13
    has_joint_reflections: bool = True
    seed: int = 0
    joint_positions: List[float] = field(default_factory=list)

    @property
    def total_length(self) -> float:
        return sum(s.length_m for s in self.segments)

    @property
    def defect_info(self) -> List[Dict]:
        """返回缺陷信息列表 [{position, length, z0, severity}]"""
        defects = []
        pos = 0.0
        active = None

        def flush_active() -> None:
            nonlocal active
            if active is None:
                return
            length = active["length"]
            z0 = active["z0_sum"] / max(length, 1e-12)
            epsr = active["epsr_sum"] / max(length, 1e-12)
            alpha = active["alpha_sum"] / max(length, 1e-12)
            z0_dev = abs(z0 - self.z_ref) / self.z_ref
            severity = min(z0_dev * 5, 1.0)
            severity = max(severity, active["label_amplitude"])
            defects.append({
                "type": active["type"],
                "start": active["start"],
                "end": active["end"],
                "position": active["start"] + length / 2,
                "length": length,
                "z0": z0,
                "epsr": epsr,
                "alpha": alpha,
                "severity": severity,
            })
            active = None

        for idx, seg in enumerate(self.segments):
            if seg.is_defect:
                group = getattr(seg, "defect_group", f"segment-{idx}")
                defect_type = str(getattr(seg, "defect_type", "short"))
                label_amplitude = float(getattr(seg, "label_amplitude", 0.0))
                if active is None or active["group"] != group or active["type"] != defect_type:
                    flush_active()
                    active = {
                        "group": group,
                        "type": defect_type,
                        "start": pos,
                        "end": pos + seg.length_m,
                        "length": 0.0,
                        "z0_sum": 0.0,
                        "epsr_sum": 0.0,
                        "alpha_sum": 0.0,
                        "label_amplitude": 0.0,
                    }
                active["end"] = pos + seg.length_m
                active["length"] += seg.length_m
                active["z0_sum"] += seg.z0_ohm * seg.length_m
                active["epsr_sum"] += seg.epsr * seg.length_m
                active["alpha_sum"] += seg.alpha_db_per_m_100mhz * seg.length_m
                active["label_amplitude"] = max(active["label_amplitude"], label_amplitude)
            else:
                flush_active()
            pos += seg.length_m
        flush_active()
        distributed_regions = []
        distributed_regions.extend(getattr(self, "distributed_moisture_regions", []))
        distributed_regions.extend(getattr(self, "distributed_long_regions", []))
        seen_regions = set()
        for region in distributed_regions:
            key = (
                str(region.get("type", "moisture_distributed")),
                round(float(region.get("start", 0.0)), 6),
                round(float(region.get("length", 0.0)), 6),
            )
            if key in seen_regions:
                continue
            seen_regions.add(key)
            start = float(region["start"])
            length = float(region["length"])
            defects.append({
                "type": str(region.get("type", "moisture_distributed")),
                "start": start,
                "end": float(region.get("end", start + length)),
                "position": float(region.get("position", start + length / 2.0)),
                "length": length,
                "z0": float(region.get("z0", self.z_ref)),
                "epsr": float(region.get("epsr", self.epsr)),
                "alpha": float(region.get("alpha", 0.0)),
                "severity": float(region.get("severity", 0.62)),
            })
        defects.sort(key=lambda d: float(d["start"]))
        return defects


# ═══════════════════════════════════════════════
# 传输线计算核心（基于参考 ADS 生成器）
# ═══════════════════════════════════════════════

def _target_to_geometry(z0_target: float, epsr: float, alpha_target: float) -> Tuple[float, float]:
    """
    反求有效同轴几何参数 (rc, rs)，匹配 Z0、epsr、alpha@100MHz。
    简化版：直接用 bisection 搜索 rc。

    Returns: (rc_m, rs_m)
    """
    vf = 1.0 / math.sqrt(epsr)
    log_term = z0_target * math.sqrt(epsr) / 60.0
    ratio = math.exp(log_term)
    omega_ref = 2.0 * PI * 100e6
    sigma_cu = 5.89107e7
    sigma_dielectric = 1e-12

    def alpha_db_for_rc(rc_m: float) -> float:
        rs_m = rc_m * ratio
        C = 2.0 * PI * epsr * EPS0 / log_term
        G = 2.0 * PI * sigma_dielectric / log_term
        skin_depth = math.sqrt(2.0 / (omega_ref * MU0 * sigma_cu))
        R = (1.0 / (2.0 * PI * rc_m * skin_depth * sigma_cu)
             + 1.0 / (2.0 * PI * rs_m * skin_depth * sigma_cu))
        L = (MU0 / (2.0 * PI) * log_term
             + MU0 / (4.0 * PI) * (1.0 / rc_m + 1.0 / rs_m) * skin_depth)
        Z = complex(R, omega_ref * L)
        Y = complex(G, omega_ref * C)
        gamma = complex(np.sqrt(Z * Y))
        return float(gamma.real * 8.686)

    target = alpha_target
    lo, hi = 1e-5, 5e-3
    # 确保范围有效
    try:
        alo, ahi = alpha_db_for_rc(lo), alpha_db_for_rc(hi)
    except Exception:
        alo, ahi = 10.0, 0.001

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        try:
            am = alpha_db_for_rc(mid)
        except Exception:
            am = target + 1
        if am > target:
            lo = mid
        else:
            hi = mid
    rc_m = 0.5 * (lo + hi)
    rs_m = rc_m * ratio
    return rc_m, rs_m


def _compute_s11_for_cable(
    freq_hz: np.ndarray,
    cable: CableSample,
) -> np.ndarray:
    """
    Compute the cascaded cable S11 with broadband RLGC parameters.

    V2.1 keeps all visible effects in frequency domain.  In addition to skin
    effect, each segment may carry a dielectric loss tangent and a causal
    Debye relaxation term.  This gives wet/aged sections their slower wave
    speed and extra attenuation without modifying the IFFT result afterwards.
    """
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    omega = 2.0 * PI * freq_hz
    n_freq = len(freq_hz)
    z_load = np.full(n_freq, cable.z_load_open, dtype=np.complex128)

    for seg in reversed(cable.segments):
        epsr_ref = max(float(seg.epsr), 1.05)
        log_term = max(float(seg.z0_ohm) * math.sqrt(epsr_ref) / 60.0, 1e-6)
        ratio = math.exp(log_term)

        tan_delta = max(float(getattr(seg, "tan_delta_100mhz", 0.0)), 0.0)
        debye_delta = max(float(getattr(seg, "debye_delta_epsr", 0.0)), 0.0)
        debye_corner = max(float(getattr(seg, "debye_corner_hz", 80e6)), 1.0)
        debye_exp = float(np.clip(getattr(seg, "debye_exponent", 1.0), 0.55, 1.35))

        # Reserve the requested 100 MHz attenuation budget for dielectric loss
        # before solving the equivalent conductor geometry.
        x_ref = (100e6 / debye_corner) ** debye_exp
        eps_debye_ref = debye_delta / complex(1.0, x_ref)
        effective_tan_ref = tan_delta
        if eps_debye_ref.real + epsr_ref > 0:
            effective_tan_ref += max(-eps_debye_ref.imag / (epsr_ref + eps_debye_ref.real), 0.0)
        beta_ref = 2.0 * PI * 100e6 * math.sqrt(epsr_ref + max(eps_debye_ref.real, 0.0)) / C0
        alpha_dielectric_db = 8.686 * 0.5 * beta_ref * effective_tan_ref
        conductor_alpha_target = max(float(seg.alpha_db_per_m_100mhz) - alpha_dielectric_db, 2.0e-4)

        rc_m, rs_m = _target_to_geometry(seg.z0_ohm, epsr_ref, conductor_alpha_target)
        skin_depth = np.sqrt(2.0 / np.maximum(omega * MU0 * seg.sigma_cu, 1e-30))

        # Causal Debye permittivity: eps* = eps_inf + Δeps/(1 + j(f/fc)^p).
        # With the exp(+jwt) convention, Im(eps*)<0 produces positive loss.
        debye_den = 1.0 + 1j * np.power(np.maximum(freq_hz, 0.0) / debye_corner, debye_exp)
        eps_complex = epsr_ref + debye_delta / debye_den
        c_complex = 2.0 * PI * EPS0 * eps_complex / log_term
        c_real = 2.0 * PI * EPS0 * np.maximum(eps_complex.real, 1.01) / log_term
        g_conductive = 2.0 * PI * float(seg.sigma_dielectric) / log_term
        g_loss = omega * c_real * tan_delta

        R = (1.0 / (2.0 * PI * rc_m * skin_depth * seg.sigma_cu)
             + 1.0 / (2.0 * PI * rs_m * skin_depth * seg.sigma_cu))
        L = (MU0 / (2.0 * PI) * log_term
             + MU0 / (4.0 * PI) * (1.0 / rc_m + 1.0 / rs_m) * skin_depth)

        Z = R + 1j * omega * L
        Y = g_conductive + g_loss + 1j * omega * c_complex
        z0 = np.sqrt(Z / Y)
        gamma = np.sqrt(Z * Y)

        exp_term = np.exp(-2.0 * gamma * float(seg.length_m))
        reflection = (z_load - z0) / (z_load + z0)
        denominator = 1.0 - reflection * exp_term
        tiny = np.abs(denominator) < 1e-14
        if np.any(tiny):
            denominator = denominator.copy()
            denominator[tiny] += 1e-14
        z_load = z0 * (1.0 + reflection * exp_term) / denominator

    return (z_load - cable.z_ref) / (z_load + cable.z_ref)


# ═══════════════════════════════════════════════
# 噪声模型（校准自 RG58-74M 实测数据）
# ═══════════════════════════════════════════════

def _apply_noise(
    s11: np.ndarray,
    freq_hz: np.ndarray,
    rng: np.random.RandomState,
    additive_scale: float = 1.0,
    multiplicative_scale: float = 1.0,
) -> np.ndarray:
    """
    校准化真实噪声模型。

    基于 RG58-74M 重复测量残差和 FDR 现场数据分析：
    - 加性噪声 σ_add(f) = 5e-4 * (1 + 2*exp(-f/100MHz)) * additive_scale
      → 低频稍大(~0.001-0.002), 中高频稳定(~0.0005)
    - 乘性噪声 σ_mult = α * |S11(f)|, α ∈ [0.005, 0.05]
      → 覆盖干净(Core)到嘈杂(现场)的范围
    - Re/Im 独立高斯（保守假设）
    """
    f_ghz = freq_hz / 1e9  # 方便计算

    # 加性噪声地板（频率相关）
    sigma_add = 5e-4 * (1.0 + 2.0 * np.exp(-f_ghz / 0.1)) * additive_scale
    # 乘性噪声系数
    alpha_mult = rng.uniform(0.005, 0.05) * multiplicative_scale
    sigma_mult = alpha_mult * np.abs(s11)

    # 总噪声
    sigma_total = np.sqrt(sigma_add ** 2 + sigma_mult ** 2)

    # Re/Im 独立高斯
    noise_re = rng.normal(0, sigma_total)
    noise_im = rng.normal(0, sigma_total)

    s11_noisy = s11 + noise_re + 1j * noise_im
    return s11_noisy


# ═══════════════════════════════════════════════
# 接头反射注入（模拟BNC连接器）
# ═══════════════════════════════════════════════

def _inject_joint_reflections(
    s11: np.ndarray,
    freq_hz: np.ndarray,
    cable: CableSample,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    在段边界处注入BNC风格接头反射特征。

    基于实测数据：
    - 幅度：反射系数 0.03-0.15（覆盖BNC到N型接头）
    - FWHM: ~0.1-0.3m（空间域极窄）
    - 频域表现为周期性振荡叠加

    简化实现：在 S11 上叠加一个小的频率无关反射系数增量，
    对应一个空间极窄的阻抗突变。
    """
    if not cable.has_joint_reflections:
        return s11

    # 在内部段边界注入（跳过首尾边界）
    positions_m = []
    pos = 0.0
    for i, seg in enumerate(cable.segments[:-1]):
        pos += seg.length_m
        if rng.random() < 0.8:  # 80% 概率有接头反射（真实电缆几乎每个接头都有）
            positions_m.append(pos)

    # 记录注入的接头位置到 cable 对象
    cable.joint_positions = list(positions_m)

    if not positions_m:
        return s11

    omega = 2.0 * PI * freq_hz
    epsr = cable.epsr
    v = C0 / math.sqrt(epsr)

    s11_modified = s11.copy()
    for joint_pos in positions_m:
        # 接头反射系数（0.03-0.15，覆盖BNC到N型接头的阻抗失配范围）
        gamma_joint = rng.uniform(0.03, 0.15) * rng.choice([-1, 1])
        # 考虑往返传播衰减
        delta_s11 = gamma_joint * np.exp(-2j * omega * joint_pos / v)
        s11_modified += delta_s11

    return s11_modified


# ═══════════════════════════════════════════════
# 随机电缆拓扑生成
# ═══════════════════════════════════════════════

def generate_random_cable(
    rng: np.random.RandomState,
    total_length: Optional[float] = None,
    n_defects_range: Tuple[int, int] = (0, 3),
    epsr: Optional[float] = None,
    length_range: Tuple[float, float] = (30.0, 500.0),
) -> CableSample:
    """
    生成随机电缆拓扑。

    Args:
        rng: 随机数生成器
        total_length: 固定总长 (m)，None 则随机
        n_defects_range: 缺陷数量范围 (min, max)
        epsr: 介电常数，None 则随机 [2.0, 2.4]
        length_range: 随机总长范围 (m)，偏重 30-500m

    Returns:
        CableSample 实例
    """
    if epsr is None:
        epsr = rng.uniform(2.0, 2.4)
    if total_length is None:
        # 偏重中等长度（30-500m），偶尔更长
        total_length = rng.lognormal(np.log(150), 0.8)
        total_length = np.clip(total_length, length_range[0], length_range[1])

    n_defects = rng.randint(n_defects_range[0], n_defects_range[1] + 1)

    # 健康段基准参数
    healthy_z0 = 50.0
    healthy_alpha = rng.uniform(0.10, 0.20)  # dB/m @ 100MHz

    # 构建段列表
    segments = []

    if n_defects == 0:
        # 无缺陷：单段电缆
        segments.append(SegmentParams(
            length_m=total_length,
            z0_ohm=healthy_z0,
            epsr=epsr,
            alpha_db_per_m_100mhz=healthy_alpha,
            is_defect=False,
        ))
    else:
        # 将电缆分成 (n_defects + 1) 个健康段 + n_defects 个缺陷段
        # 缺陷段位置随机，长度 0.5-5m
        defect_lengths = [rng.uniform(0.5, 5.0) for _ in range(n_defects)]
        total_defect = sum(defect_lengths)

        if total_defect >= total_length * 0.8:
            # 缺陷段太长，缩减
            defect_lengths = [min(l, total_length * 0.1) for l in defect_lengths]
            total_defect = sum(defect_lengths)

        healthy_total = total_length - total_defect

        # 随机分配健康段长度
        healthy_lengths = [healthy_total / (n_defects + 1) for _ in range(n_defects + 1)]
        # 加随机扰动
        perturbation = rng.normal(0, 1, n_defects + 1) * healthy_total * 0.2
        healthy_lengths = [max(5.0, l + p) for l, p in zip(healthy_lengths, perturbation)]
        # 归一化到 healthy_total
        s = sum(healthy_lengths)
        healthy_lengths = [l * healthy_total / s for l in healthy_lengths]

        # 交替排列：健康段 → 缺陷段 → 健康段 → ...
        positions = sorted(rng.choice(range(n_defects + 1), n_defects, replace=False))
        defect_idx = 0
        for i in range(n_defects + 1):
            if i in positions:
                # 先放健康段
                seg_epsr = epsr + rng.normal(0, 0.05)
                seg_epsr = max(1.5, min(3.0, seg_epsr))
                segments.append(SegmentParams(
                    length_m=healthy_lengths[i],
                    z0_ohm=healthy_z0,
                    epsr=seg_epsr,
                    alpha_db_per_m_100mhz=healthy_alpha,
                    is_defect=False,
                ))
                # 再放缺陷段
                if defect_idx < n_defects:
                    # 物理正确：特性阻抗始终为正，允许 ±15% 失配（高阻或低阻缺陷）
                    mismatch_sign = rng.choice([-1, 1])            # -1=低阻缺陷, +1=高阻缺陷
                    mismatch_ratio = rng.uniform(0.01, 0.15)       # 失配比例 1%~15%
                    defect_z0 = healthy_z0 * (1.0 + mismatch_sign * mismatch_ratio)
                    defect_epsr = epsr + rng.normal(0, 0.2)
                    defect_epsr = max(1.5, min(4.0, defect_epsr))
                    defect_alpha = healthy_alpha * rng.uniform(0.8, 3.0)
                    segments.append(SegmentParams(
                        length_m=defect_lengths[defect_idx],
                        z0_ohm=defect_z0,
                        epsr=defect_epsr,
                        alpha_db_per_m_100mhz=defect_alpha,
                        is_defect=True,
                    ))
                    defect_idx += 1
            else:
                seg_epsr = epsr + rng.normal(0, 0.05)
                seg_epsr = max(1.5, min(3.0, seg_epsr))
                segments.append(SegmentParams(
                    length_m=healthy_lengths[i],
                    z0_ohm=healthy_z0,
                    epsr=seg_epsr,
                    alpha_db_per_m_100mhz=healthy_alpha,
                    is_defect=False,
                ))

    return CableSample(
        segments=segments,
        epsr=epsr,
        seed=rng.randint(0, 2**31),
    )


# ═══════════════════════════════════════════════
# 主生成函数
# ═══════════════════════════════════════════════

def generate_s11(
    cable: CableSample,
    sweep: Optional[SweepConfig] = None,
    rng: Optional[np.random.RandomState] = None,
    add_noise: bool = True,
    additive_scale: float = 1.0,
    multiplicative_scale: float = 1.0,
    inject_joints: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成合成 S11 数据。

    Args:
        cable: 电缆拓扑配置
        sweep: 频率扫频配置（默认 9kHz-1GHz 10000点）
        rng: 随机数生成器
        add_noise: 是否加噪声
        additive_scale: 加性噪声缩放
        multiplicative_scale: 乘性噪声缩放
        inject_joints: 是否注入接头反射

    Returns:
        freq_hz: 频率轴 (Hz), shape [n_points]
        s11: S11 复数数组, shape [n_points]
    """
    if sweep is None:
        sweep = SweepConfig()
    if rng is None:
        rng = np.random.RandomState(cable.seed)

    freq_hz = sweep.frequencies()

    # 1. 纯物理 S11（多段级联传输线模型）
    s11 = _compute_s11_for_cable(freq_hz, cable)

    # 2. 接头反射注入
    if inject_joints:
        s11 = _inject_joint_reflections(s11, freq_hz, cable, rng)

    # 3. 真实噪声注入
    if add_noise:
        s11 = _apply_noise(s11, freq_hz, rng,
                           additive_scale=additive_scale,
                           multiplicative_scale=multiplicative_scale)

    return freq_hz, s11


def generate_sample(
    seed: int,
    total_length: Optional[float] = None,
    epsr: Optional[float] = None,
    add_noise: bool = True,
    inject_joints: bool = True,
) -> CableSample:
    """
    快速生成一个随机电缆样本（不含频率计算，仅拓扑）。

    Args:
        seed: 随机种子（用于复现）
        total_length: 固定总长
        epsr: 介电常数
        add_noise: 是否加噪声（记录到 sample 上）
        inject_joints: 是否注入接头

    Returns:
        CableSample
    """
    rng = np.random.RandomState(seed)
    cable = generate_random_cable(rng, total_length=total_length, epsr=epsr)
    cable.has_joint_reflections = inject_joints
    return cable
