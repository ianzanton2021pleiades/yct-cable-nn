"""Preprocess the field .fdr corpus into per-phase npz + manifest.

Outputs to E:\\CableNNDataSet\\ZCode\\field_data\\:
  phases/<stem>__<phase>.npz   freq, s11 (complex64), calib (freq/re/im),
                               vendor impulse/step tables (dist/rho), meta json
  manifest.csv                 one row per phase: location stats + labels + split

Annotations: curated from the engineer txt files (survey 2026-10-29).  The
1500 m controlled soak trial's labels live in its sweep names.  Sealed blind
folders are marked split=blind and must never enter training or tuning.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dg_v4.fdr_io import read_fdr, phase_arrays

FIELD_ROOT = Path(r"E:\20261010截至的全部fdr文件")
OUT = Path(r"E:\CableNNDataSet\ZCode\field_data")
SKIP_FOLDERS = ("RG58", "核级")

# folder-name keyword -> engineer annotations (length in m; joints from txt)
CURATED = {
    "三院": dict(length_m=770, joints_m=[55, 91, 175, 638, 682],
                notes="健康样本+接头真值;低压脉冲55-93-174-633-684-765(末端)"),
    "平湖站10kv辅城坳线F17平湖山厦": dict(length_m=598, joints_m=[222, 240, 255, 285, 355, 421],
                notes="222阻抗突升/255绝缘下降/225接头解开绝缘恢复=受潮接头"),
    "白塔寺": dict(length_m=1100, joints_m=[240],
                notes="耐压前怀疑240m阻抗上升,耐压后240m击穿;两份fdr=耐压前后"),
    "石杨线": dict(length_m=400, joints_m=[], notes="5号至6号进水"),
    "浸水）": dict(length_m=1930, joints_m=[], defect_intervals_m=[(364, 476)],
                notes="深圳龙岗局部浸水,阶跃波阻抗显著降低,脉冲先负后正"),
    "邦润线": dict(length_m=640, joints_m=[], notes="270m B相阻抗下降,420m三相下降,浸水"),
    "逸翠园": dict(length_m=3870, joints_m=[100, 355, 446, 826, 985, 1200, 1482, 1626, 2062, 2350, 2449, 2862, 3055, 3370, 3592],
                notes="接头人工标注(2350/2862疑似)"),
    "喀纳斯": dict(length_m=5244, joints_m=[180, 450, 890, 1392, 1642, 1850, 2400, 2925, 3419, 3870, 4443, 4864],
                notes="末端1号分支箱5244m;另有全长9852/11156版本"),
    "两段电缆": dict(length_m=None, joints_m=None,
                notes="第一段接头53/183/332/412/514,330浸水最严重;第二段206中间头;分段文件按子目录处理"),
    "清水河": dict(length_m=4714, joints_m=[396, 667, 1004, 1367, 1383],
                notes="20-40m第一接头受潮击穿;9-14接头解开后重测"),
    "金嘉一线": dict(length_m=301, joints_m=[], notes="健康"),
    "金富线": dict(length_m=1724, joints_m=[154], notes="154m短路"),
    "杨凌": dict(length_m=2764, joints_m=[], notes="定位偏差2000m失败案例;快速+标准双模式"),
    "达悦三联水库": dict(length_m=965, joints_m=[96, 463, 671, 935],
                notes="95m附近外屏蔽锈蚀,绝缘没问题"),
    "鼎盛": dict(length_m=945, joints_m=[213, 338, 644, 945], notes="338接头老化"),
    "香园": dict(length_m=2750, joints_m=[], notes="快速/标准双模式对照"),
}
# permanently sealed blind evaluation set
BLIND_KEYWORDS = ("1500m不同位置", "两段电缆", "三院")


def sweep_name_labels(sweeps) -> dict[str, dict]:
    """Parse the controlled soak trial: names carry the soak position."""
    out = {}
    for r in sweeps:
        name = str(r[0])
        m = re.search(r"(\d+)\s*m", name)
        if "正常" in name or "正常" in name:
            out[name] = dict(soak_position_m=None)
        elif m:
            out[name] = dict(soak_position_m=float(m.group(1)))
    return out


def phase_stats(s11: np.ndarray) -> dict:
    mag = np.abs(s11)
    n = len(mag)
    return dict(s11_mean=round(float(mag.mean()), 4),
                s11_max=round(float(mag.max()), 4),
                s11_low5=round(float(mag[:max(n // 20, 1)].mean()), 4),
                s11_high5=round(float(mag[-max(n // 20, 1):].mean()), 4),
                frac_over_1p05=round(float((mag > 1.05).mean()), 5))


def valid_phase(stats: dict) -> bool:
    return stats["frac_over_1p05"] < 0.05 and stats["s11_max"] < 1.6


def main() -> None:
    (OUT / "phases").mkdir(parents=True, exist_ok=True)
    fdr_files = []
    for folder in sorted(FIELD_ROOT.iterdir()):
        if not folder.is_dir() or any(k in folder.name for k in SKIP_FOLDERS):
            continue
        fdr_files += sorted(folder.glob("*.fdr")) + sorted(folder.glob("*.db8"))
        for sub in folder.iterdir():
            if sub.is_dir():
                fdr_files += sorted(sub.glob("*.fdr"))
    rows = []
    for path in fdr_files:
        folder = path.parent.name
        blind = any(k in folder for k in BLIND_KEYWORDS)
        try:
            data = read_fdr(str(path))
        except Exception as exc:  # noqa: BLE001
            rows.append(dict(file=str(path), error=str(exc)[:120]))
            print(f"FAIL {path.name}: {exc}")
            continue
        curated = None
        for key, lab in CURATED.items():
            if key in folder:
                curated = lab
                break
        soak = sweep_name_labels(data.get("sweeps", []))
        for idx, ph in enumerate(data.get("phases", [])):
            try:
                arr = phase_arrays(ph)
            except Exception as exc:  # noqa: BLE001
                print(f"  phase parse fail {ph['name']}: {exc}")
                continue
            stats = phase_stats(arr["s11"])
            stem = f"{path.stem}__{ph['name'].strip()}"
            meta = dict(file=str(path), folder=folder, phase=ph["name"],
                        fstart=float(ph["fstart"]), fstop=float(ph["fstop"]),
                        points=int(ph["points"]), wave_velocity=float(ph["wave_velocity"]),
                        x_max_m=float(ph.get("x_max_m") or 0),
                        valid=bool(valid_phase(stats)), stats=stats,
                        curated=curated, soak_labels=soak if soak else None,
                        split="blind" if blind else "field")
            np.savez(OUT / "phases" / f"{stem}.npz",
                     freq=arr["freq"].astype(np.float64),
                     s11=arr["s11"].astype(np.complex64),
                     impulse_dist=arr.get("impulse_dist", np.zeros(0)),
                     impulse_rho=arr.get("impulse_rho", np.zeros(0)),
                     step_dist=arr.get("step_dist", np.zeros(0)),
                     step_rho=arr.get("step_rho", np.zeros(0)),
                     meta=np.array(json.dumps(meta, ensure_ascii=False)))
            rows.append(dict(file=path.name, folder=folder, phase=ph["name"],
                             stem=stem, **stats, points=ph["points"],
                             wave_velocity=ph["wave_velocity"], x_max_m=ph.get("x_max_m"),
                             valid=meta["valid"], split=meta["split"],
                             length_m=(curated or {}).get("length_m"),
                             joints_m=json.dumps((curated or {}).get("joints_m"), ensure_ascii=False)
                             if curated else "",
                             notes=(curated or {}).get("notes", "") if curated else ""))
        print(f"ok {path.name}: {len(data.get('phases', []))} phases")
    with (OUT / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"done: {len(rows)} phase rows -> {OUT}")


if __name__ == "__main__":
    main()
