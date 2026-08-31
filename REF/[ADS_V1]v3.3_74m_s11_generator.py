# 这个程序能够完美地还原出ADS中进行的V1版本的电路的S参数，不管是S参数还是IFFT都很完美
# 这个版本加入40 m完好段 + 4 m缺陷段 + 30 m完好段的三段递推模型；healthy段继续按目标Z0/VF/100 MHz衰减反求有效几何，缺陷段则按独立目标Z0/VF/100 MHz衰减反求自己的有效几何。

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
import numpy as np

PI = math.pi
MU0 = 4.0e-7 * PI
EPS0 = 8.854_187_817e-12
C0 = 299_792_458.0


@dataclass
class SweepConfig:
    start_hz: float = 9e3
    stop_hz: float = 1e9
    step_hz: float = 100e3

    def frequencies(self) -> np.ndarray:
        if self.start_hz <= 0 or self.stop_hz <= self.start_hz or self.step_hz <= 0:
            raise ValueError("Invalid sweep settings.")
        n = int(math.floor((self.stop_hz - self.start_hz) / self.step_hz + 0.5)) + 1
        f = self.start_hz + np.arange(n, dtype=float) * self.step_hz
        if abs(f[-1] - self.stop_hz) > 1e-9:
            f = np.append(f, self.stop_hz)
        else:
            f[-1] = self.stop_hz
        return f


@dataclass
class SegmentTargetConfig:
    z0_target_ohm: float
    vf_target: float
    alpha_target_db_per_m_at_fref: float
    f_ref_hz: float = 100e6
    sigma_cu_ref_s_per_m: float = 5.89107e7
    sigma_dielectric_s_per_m: float = 1e-12


@dataclass
class Cable74mConfig:
    healthy: SegmentTargetConfig
    aged: SegmentTargetConfig
    z_ref_ohm: float = 50.0
    z_load_open_ohm: float = 1e13
    len1_m: float = 40.0
    len_age_m: float = 4.0
    len3_m: float = 30.0


@dataclass
class EffectiveGeometry:
    rc_m: float
    rs_m: float
    epsr_eff: float

    @property
    def log_term(self) -> float:
        return math.log(self.rs_m / self.rc_m)



def target_to_effective_geometry(cfg: SegmentTargetConfig) -> EffectiveGeometry:
    """
    Build an effective segment geometry that matches three targets simultaneously:
      1) Z0 target
      2) VF target
      3) alpha(f_ref) target

    Strategy:
      - epsr_eff from VF
      - log(rs/rc) from low-loss coax Z0 relation
      - absolute rc scale fitted from alpha at f_ref using the same RLGC kernel
    """
    epsr_eff = 1.0 / (cfg.vf_target ** 2)
    log_term = cfg.z0_target_ohm * math.sqrt(epsr_eff) / 60.0
    ratio = math.exp(log_term)

    omega_ref = 2.0 * PI * cfg.f_ref_hz

    def alpha_db_for_rc(rc_m: float) -> float:
        rs_m = rc_m * ratio
        C = 2.0 * PI * epsr_eff * EPS0 / log_term
        G = 2.0 * PI * cfg.sigma_dielectric_s_per_m / log_term
        skin_depth = math.sqrt(2.0 / (omega_ref * MU0 * cfg.sigma_cu_ref_s_per_m))
        R = (
            1.0 / (2.0 * PI * rc_m * skin_depth * cfg.sigma_cu_ref_s_per_m)
            + 1.0 / (2.0 * PI * rs_m * skin_depth * cfg.sigma_cu_ref_s_per_m)
        )
        L = (
            MU0 / (2.0 * PI) * log_term
            + MU0 / (4.0 * PI) * (1.0 / rc_m + 1.0 / rs_m) * skin_depth
        )
        Z = complex(R, omega_ref * L)
        Y = complex(G, omega_ref * C)
        gamma = complex(np.sqrt(Z * Y))
        return float(gamma.real * 8.686)

    target = cfg.alpha_target_db_per_m_at_fref
    lo, hi = 1e-5, 5e-3
    alo, ahi = alpha_db_for_rc(lo), alpha_db_for_rc(hi)
    if not (alo >= target >= ahi):
        raise ValueError(
            f"Target alpha {target} dB/m at {cfg.f_ref_hz/1e6:.1f} MHz is outside achievable range "
            f"for current material assumptions: [{ahi:.6f}, {alo:.6f}] dB/m."
        )

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        am = alpha_db_for_rc(mid)
        if am > target:
            lo = mid
        else:
            hi = mid
    rc_m = 0.5 * (lo + hi)
    rs_m = rc_m * ratio
    return EffectiveGeometry(rc_m=rc_m, rs_m=rs_m, epsr_eff=epsr_eff)



def calc_primary_params(freq_hz: np.ndarray, geom: EffectiveGeometry, cfg: SegmentTargetConfig):
    omega = 2.0 * PI * freq_hz
    log_term = geom.log_term

    C = np.full_like(freq_hz, 2.0 * PI * geom.epsr_eff * EPS0 / log_term, dtype=float)
    G = np.full_like(freq_hz, 2.0 * PI * cfg.sigma_dielectric_s_per_m / log_term, dtype=float)

    skin_depth = np.sqrt(2.0 / (omega * MU0 * cfg.sigma_cu_ref_s_per_m))
    R = (
        1.0 / (2.0 * PI * geom.rc_m * skin_depth * cfg.sigma_cu_ref_s_per_m)
        + 1.0 / (2.0 * PI * geom.rs_m * skin_depth * cfg.sigma_cu_ref_s_per_m)
    )
    L = (
        MU0 / (2.0 * PI) * log_term
        + MU0 / (4.0 * PI) * (1.0 / geom.rc_m + 1.0 / geom.rs_m) * skin_depth
    )
    return R, L, G, C



def calc_z0_gamma(R: np.ndarray, L: np.ndarray, G: np.ndarray, C: np.ndarray, freq_hz: np.ndarray):
    omega = 2.0 * PI * freq_hz
    Z = R + 1j * omega * L
    Y = G + 1j * omega * C
    z0 = np.sqrt(Z / Y)
    gamma = np.sqrt(Z * Y)
    return z0, gamma



def transform_uniform_line(z_load: np.ndarray, z0: np.ndarray, gamma: np.ndarray, seg_len_m: float) -> np.ndarray:
    exp_term = np.exp(-2.0 * gamma * seg_len_m)
    T = (z_load - z0) / (z_load + z0)
    z_in = z0 * (1.0 + T * exp_term) / (1.0 - T * exp_term)
    return z_in



def compute_s11_74m(freq_hz: np.ndarray, cfg: Cable74mConfig) -> np.ndarray:
    if np.any(freq_hz <= 0):
        raise ValueError("Frequencies must be positive.")

    geom_h = target_to_effective_geometry(cfg.healthy)
    geom_a = target_to_effective_geometry(cfg.aged)

    Rh, Lh, Gh, Ch = calc_primary_params(freq_hz, geom_h, cfg.healthy)
    z0_h, gamma_h = calc_z0_gamma(Rh, Lh, Gh, Ch, freq_hz)

    Ra, La, Ga, Ca = calc_primary_params(freq_hz, geom_a, cfg.aged)
    z0_a, gamma_a = calc_z0_gamma(Ra, La, Ga, Ca, freq_hz)

    # End -> source recursion: 30 m healthy, 4 m aged, 40 m healthy
    z_load = np.full_like(freq_hz, cfg.z_load_open_ohm, dtype=complex)
    z_after_len3 = transform_uniform_line(z_load, z0_h, gamma_h, cfg.len3_m)
    z_after_age = transform_uniform_line(z_after_len3, z0_a, gamma_a, cfg.len_age_m)
    z_in = transform_uniform_line(z_after_age, z0_h, gamma_h, cfg.len1_m)

    s11 = (z_in - cfg.z_ref_ohm) / (z_in + cfg.z_ref_ohm)
    return s11



def diagnostics_at(freq_hz: float, seg_cfg: SegmentTargetConfig):
    geom = target_to_effective_geometry(seg_cfg)
    f = np.array([freq_hz], dtype=float)
    R, L, G, C = calc_primary_params(f, geom, seg_cfg)
    z0, gamma = calc_z0_gamma(R, L, G, C, f)
    beta = float(np.imag(gamma[0]))
    vp = 2.0 * PI * freq_hz / beta if beta != 0 else float("inf")
    vf = vp / C0
    alpha_db_m = float(np.real(gamma[0]) * 8.686)
    return {
        "rc_mm": geom.rc_m * 1e3,
        "rs_mm": geom.rs_m * 1e3,
        "epsr_eff": geom.epsr_eff,
        "z0_ohm": float(np.real(z0[0])),
        "vf": vf,
        "alpha_db_m": alpha_db_m,
    }



def save_csv(path: str | Path, freq_hz: np.ndarray, s11: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Frequency_Hz", "S11_Real", "S11_Imag"])
        for fi, si in zip(freq_hz, s11):
            writer.writerow([f"{int(round(fi))}", f"{si.real:.10f}", f"{si.imag:.10f}"])



def default_config() -> Cable74mConfig:
    healthy = SegmentTargetConfig(
        z0_target_ohm=50.0,
        vf_target=0.67055,
        alpha_target_db_per_m_at_fref=0.14,
        f_ref_hz=100e6,
        sigma_cu_ref_s_per_m=5.89107e7,
        sigma_dielectric_s_per_m=1e-12,
    )
    aged = SegmentTargetConfig(
        z0_target_ohm=51.0,
        vf_target=1.0 / math.sqrt(2.40),
        alpha_target_db_per_m_at_fref=1.10 * 0.14,
        f_ref_hz=100e6,
        sigma_cu_ref_s_per_m=5.89107e7,
        sigma_dielectric_s_per_m=1e-12,
    )
    return Cable74mConfig(
        healthy=healthy,
        aged=aged,
        z_ref_ohm=50.0,
        z_load_open_ohm=1e13,
        len1_m=40.0,
        len_age_m=4.0,
        len3_m=30.0,
    )



def main() -> None:
    sweep = SweepConfig(start_hz=9e3, stop_hz=1000e6, step_hz=5e6)
    cfg = default_config()

    freq_hz = sweep.frequencies()
    s11 = compute_s11_74m(freq_hz, cfg)

    script_dir = Path(__file__).resolve().parent
    out_path = script_dir / "data" / "v3_3_74m_s11.csv"
    save_csv(out_path, freq_hz, s11)

    dh = diagnostics_at(100e6, cfg.healthy)
    da = diagnostics_at(100e6, cfg.aged)
    print(f"Saved: {out_path}")
    print(
        "Healthy @100 MHz: "
        f"rc:{dh['rc_mm']:.4f} mm, rs:{dh['rs_mm']:.4f} mm, epsr:{dh['epsr_eff']:.6f}, "
        f"Z0:{dh['z0_ohm']:.4f} ohm, VF:{dh['vf']:.6f}, alpha:{dh['alpha_db_m']:.6f} dB/m"
    )
    print(
        "Aged @100 MHz: "
        f"rc:{da['rc_mm']:.4f} mm, rs:{da['rs_mm']:.4f} mm, epsr:{da['epsr_eff']:.6f}, "
        f"Z0:{da['z0_ohm']:.4f} ohm, VF:{da['vf']:.6f}, alpha:{da['alpha_db_m']:.6f} dB/m"
    )
    print("Example rows:")
    for i in range(min(3, len(freq_hz))):
        print(f"{int(round(freq_hz[i]))},{s11[i].real:.10f},{s11[i].imag:.10f}")


if __name__ == "__main__":
    main()
