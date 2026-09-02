# 探测：1) 字体回退是否生效 2) 低频大偏差的物理来源
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]

# ---------- 字体探测 ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

for p in (r"C:\Windows\Fonts\times.ttf", r"C:\Windows\Fonts\simhei.ttf"):
    fm.fontManager.addfont(p)

plt.rcParams["font.family"] = ["Times New Roman", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
fig, ax = plt.subplots(figsize=(5, 2))
ax.set_title("中文黑体 English TNR: S11 实部 相位")
ax.set_xlabel("Frequency (MHz) 频率")
fig.savefig(SCRIPT_DIR / "_probe_font.png", dpi=120)
plt.close(fig)
print("font probe saved")

# ---------- 低频物理探测 ----------
_ADS_PATH = ROOT / "REF" / "[ADS_V1]v3.3_74m_s11_generator.py"
_spec = importlib.util.spec_from_file_location("ads_ref", _ADS_PATH)
ads = importlib.util.module_from_spec(_spec)
sys.modules["ads_ref"] = ads
_spec.loader.exec_module(ads)

sys.path.insert(0, str(ROOT / "DG_Update" / "DG_V3"))
from dg_v3.topology import CableSegment, CableTopology          # noqa: E402
from dg_v3.physics import topology_abcd, network_s11             # noqa: E402

CFG = ads.default_config()
VF_H = CFG.healthy.vf_target
EPSR_H = 1.0 / VF_H ** 2
ALPHA_H = CFG.healthy.alpha_target_db_per_m_at_fref
L = 74.0

dh = ads.diagnostics_at(100e6, CFG.healthy)
print("ADS healthy geom @100MHz:", {k: round(v, 6) for k, v in dh.items()})
rc = ads.target_to_effective_geometry(CFG.healthy).rc_m
f_delta_eq_rc = 2.0 / ((2 * np.pi * 100e6) * 4e-7 * np.pi * 5.89107e7 * rc ** 2)
print(f"rc = {rc*1e3:.4f} mm, skin_depth == rc at f = {f_delta_eq_rc/1e3:.1f} kHz")

f = np.array([1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8, 3e8, 1e9])
cfg = ads.Cable74mConfig(healthy=CFG.healthy, aged=CFG.aged,
                         z_ref_ohm=50.0, z_load_open_ohm=1e13,
                         len1_m=0.0, len_age_m=0.0, len3_m=L)
s_ads = ads.compute_s11_74m(f, cfg)
seg = CableSegment(0.0, L, 50.0, EPSR_H, ALPHA_H, 1e-12)
topo = CableTopology(profile="rg58", length_m=L, z_ref_ohm=50.0,
                     base_z0_ohm=50.0, base_epsr=EPSR_H,
                     base_alpha_db_per_m_at_100mhz=ALPHA_H,
                     base_tan_delta_at_100mhz=1e-12,
                     dispersion_fraction=0.0, segments=[seg], joints=[],
                     termination="open", z_load_ohm=1e13, defect_regions=[])
s_dg = network_s11(topology_abcd(f, topo), 1e13, 50.0)

geom = ads.target_to_effective_geometry(CFG.healthy)
R, Ll, G, C = ads.calc_primary_params(f, geom, CFG.healthy)
_, gamma_ads = ads.calc_z0_gamma(R, Ll, G, C, f)
alpha_ads = np.real(gamma_ads) * 8.686
vf_ads = 2 * np.pi * f / (np.imag(gamma_ads) * 299792458.0)
ratio = f / 100e6
alpha_dg = ALPHA_H * (0.35 * np.sqrt(ratio) + 0.65 * ratio)

print(f"\n{'f':>12} {'ADS_a_dB/m':>11} {'DG_a_dB/m':>11} {'ADS_VF':>9} "
      f"{'|S11|ADS_dB':>12} {'|S11|DG_dB':>11} {'dphase_deg':>10}")
for i in range(len(f)):
    rt_ads = 20 * np.log10(abs(s_ads[i]))
    rt_dg = 20 * np.log10(abs(s_dg[i]))
    dp = np.angle(s_dg[i] / s_ads[i]) * 180 / np.pi
    print(f"{f[i]:12.4g} {alpha_ads[i]:11.5f} {alpha_dg[i]:11.5f} {vf_ads[i]:9.6f} "
          f"{rt_ads:12.4f} {rt_dg:11.4f} {dp:10.3f}")
