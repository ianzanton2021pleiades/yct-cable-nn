"""T2 组级划分 + 现场样本构建 v2（不覆盖旧 prep_field_train.py / field_train）。

改进点（对应第一轮三个问题）：
  1. 划分泄漏：组 = .fdr 文件（同缆同次测试的全部相/复测），固定 rng=np.random.default_rng(20261030)，
     组名排序后组级 shuffle 85/15 -> field_train / field_val；另抽约 10% 组（长度分散）为 blind。
     两个同数据副本文件合并为同一 split 单元（8-18 主/子文件、7-15 原件/（之前）副本）。
  2. 标注：curated_v2.json（T1 产物）按 .fdr 文件提供 joints/defects/terminal_full_length_m，
     事件带 weight（engineer_fdr=1.0, field_test=0.3）与 mechanism（可 null，未硬编码）。
  3. 超栅格污染：txt 全长（缺则 manifest length_m，再缺则 vendor x_max）> 2500m 的组
     -> split="excluded_long"，不进训练/验证，计数留痕。
  4. 数组截断：valid_len = min(11008, (末端锚点*1.15+10)/0.25)，下限 400；
     npz 只存 impulse/step(float32, valid_len 长) + meta(json)，无 role/mechanism/z0_dev/s11 字段。

    python prep_field_train_v2.py
  -> E:\\CableNNDataSet\\ZCode\\field_train_v2\\phases\\*.npz + manifest.csv
     E:\\CableNNDataSet\\ZCode\\field_data\\blind_groups_v2.json
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dg_v4.labeling import SNR_OBSERVABLE

FIELD_DATA = Path(r"E:\CableNNDataSet\ZCode\field_data")
OUT = Path(r"E:\CableNNDataSet\ZCode\field_train_v2")
GRID_BINS = 11008
DIST_STEP = 0.25
GRID_M = GRID_BINS * DIST_STEP
N_VALID_MIN = 400
SEED = 20261030
LONG_LIMIT_M = 2500.0

# 同一次测试被拆成/复制成多个 fdr 文件 -> 同一 split 单元（数据重叠，防止近似样本跨集）
SPLIT_UNIT_MERGE = {
    "2026-8-18 15.46.10(1)": "2026-8-18 15.46.10",          # 主文件含(1)的3条数据+新增1条
    "2026-7-15 16.19.30（之前）": "2026-7-15 16.19.30",      # 逐字节相同的副本
}


def grid_distance() -> np.ndarray:
    return np.arange(GRID_BINS, dtype=np.float64) * DIST_STEP


def resample_table(dist: np.ndarray, rho: np.ndarray, dgrid: np.ndarray) -> np.ndarray | None:
    """vendor 表 -> 0.25m 栅格；空表返回 None（该相无内嵌脉冲/阶跃表，跳过并计数）。"""
    if dist.size < 2:
        return None
    order = np.argsort(dist)
    return np.interp(dgrid, dist[order], rho[order]).astype(np.float32)


def anchor_near(impulse: np.ndarray, dgrid: np.ndarray, pos_m: float) -> tuple[float, float]:
    """峰锚点与 SNR（复用 v1 口径）。"""
    w = max(8.0, 0.04 * pos_m)
    lo = int(max(0, (pos_m - w) / DIST_STEP))
    hi = int(min(GRID_BINS - 1, (pos_m + w) / DIST_STEP))
    if hi <= lo:
        return pos_m, 0.0
    k = lo + int(np.argmax(np.abs(impulse[lo:hi + 1])))
    f_lo = min(GRID_BINS - 1, int((pos_m + 2 * w) / DIST_STEP))
    f_hi = min(GRID_BINS - 1, int((pos_m + 6 * w) / DIST_STEP))
    if f_hi > f_lo:
        floor = float(np.sqrt(np.mean(impulse[f_lo:f_hi + 1] ** 2))) + 1e-9
    else:
        floor = float(np.sqrt(np.mean(impulse ** 2))) + 1e-9
    return float(dgrid[k]), float(abs(impulse[k]) / floor)


def assign_splits() -> tuple[dict[str, str], dict[str, float], list[str]]:
    """组级划分：excluded_long -> blind(长度分散~10%) -> 85/15。返回 {组: split}, {组: 全长}, excluded。"""
    curated = json.loads((FIELD_DATA / "curated_v2.json").read_text(encoding="utf-8"))["files"]
    by_key = {f"{v['folder']}|{v['file']}": v for v in curated.values()}
    by_alias = {(v.get("manifest_folder") or v["folder"], v["file"]): v for v in curated.values()}

    groups: dict[str, dict] = {}
    stems: dict[str, str] = {}
    with (FIELD_DATA / "manifest.csv").open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if not row.get("valid") or row["valid"] in ("False", "false", "0"):
                continue
            g = row["group"]
            rec = groups.setdefault(g, dict(tfl=None, len_m=None))
            stems.setdefault(g, row["stem"])
            ce = by_key.get(f"{row['folder']}|{row['file']}") or by_alias.get((row["folder"], row["file"]))
            if ce is not None and ce.get("terminal_full_length_m") is not None and rec["tfl"] is None:
                rec["tfl"] = float(ce["terminal_full_length_m"])
            if row.get("length_m") and rec["len_m"] is None:
                rec["len_m"] = float(row["length_m"])

    lengths = {}
    for g, rec in groups.items():
        if rec["tfl"] is not None:
            lengths[g] = rec["tfl"]
        elif rec["len_m"]:
            lengths[g] = rec["len_m"]
        else:
            with np.load(FIELD_DATA / "phases" / f"{stems[g]}.npz", allow_pickle=False) as z:
                lengths[g] = float(json.loads(str(z["meta"])).get("x_max_m") or 0)

    excluded = sorted(g for g, L in lengths.items() if L > LONG_LIMIT_M)
    pool = sorted(g for g, L in lengths.items() if L <= LONG_LIMIT_M)

    # blind：按全长排序（同名按组名）后均匀抽取 ~10% 组（长度分散）
    n_blind = max(1, int(round(0.10 * len(pool))))
    by_len = sorted(pool, key=lambda g: (lengths[g], g))
    blind_idx = ({int(round(i * (len(pool) - 1) / (n_blind - 1))) for i in range(n_blind)}
                 if n_blind > 1 else {0})
    blind = [by_len[i] for i in sorted(blind_idx)]

    rest = sorted(g for g in pool if g not in set(blind))
    rng2 = np.random.default_rng(SEED)
    perm = rng2.permutation(len(rest))
    rest_shuffled = [rest[i] for i in perm]
    n_val = int(round(0.15 * len(rest_shuffled)))
    val = rest_shuffled[:n_val]
    train = rest_shuffled[n_val:]

    # split 单元合并：副本文件与原件必须同 split
    split: dict[str, str] = {}
    for g in excluded:
        split[g] = "excluded_long"
    for g in blind:
        split[g] = "blind"
    for g in train:
        split[g] = "field_train"
    for g in val:
        split[g] = "field_val"
    for g, unit in SPLIT_UNIT_MERGE.items():
        if g in split and unit in split:
            split[g] = split[unit]
    return split, lengths, excluded


def main() -> None:
    (OUT / "phases").mkdir(parents=True, exist_ok=True)
    curated = json.loads((FIELD_DATA / "curated_v2.json").read_text(encoding="utf-8"))["files"]
    split, lengths, excluded = assign_splits()
    dgrid = grid_distance()

    out_rows = []
    ev_total = dict(joint=0, defect=0, terminal=0)
    w_total = {1.0: 0, 0.3: 0}
    mech_total: dict[str, int] = {}
    n_phase = 0
    no_table = []

    with (FIELD_DATA / "manifest.csv").open(encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("valid") and r["valid"] not in ("False", "false", "0")]

    for row in rows:
        g = row["group"]
        sp = split[g]
        if sp == "excluded_long":
            continue
        ce = curated.get(f"{row['folder']}|{row['file']}")
        if ce is None:
            ce = next((v for v in curated.values()
                       if v.get("manifest_folder") == row["folder"] and v["file"] == row["file"]), {})
        with np.load(FIELD_DATA / "phases" / f"{row['stem']}.npz", allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            if not meta.get("valid"):
                continue
            imp = resample_table(z["impulse_dist"], z["impulse_rho"], dgrid)
            stp = resample_table(z["step_dist"], z["step_rho"], dgrid)
        if imp is None or stp is None:
            no_table.append(f"{row['stem']}")
            continue

        over = (ce.get("phase_overrides") or {}).get(row["phase"]) or {}
        j_src = over.get("joints") if over else (ce.get("joints") or [])
        d_src = over.get("defects") if over else (ce.get("defects") or [])

        events = []
        for i, jj in enumerate(j_src):
            p = float(jj["pos_m"])
            if not (0 < p < GRID_M):
                continue
            anchor, snr = anchor_near(imp, dgrid, p)
            events.append(dict(event_id=f"joint_{i:02d}", role="joint", mechanism="joint",
                               physical_center_m=p, axis_pos_m=p, anchor_m=anchor, snr=snr,
                               observable=bool(snr >= SNR_OBSERVABLE), severity=0.5,
                               weight=float(jj["weight"]), reliability=jj["reliability"]))
        for i, dd in enumerate(d_src):
            c = 0.5 * (float(dd["start_m"]) + float(dd["end_m"]))
            if not (0 < c < GRID_M):
                continue
            anchor, snr = anchor_near(imp, dgrid, c)
            events.append(dict(event_id=f"defect_{i:02d}", role="defect",
                               mechanism=dd["mechanism"],
                               physical_start_m=float(dd["start_m"]), physical_end_m=float(dd["end_m"]),
                               physical_center_m=c, axis_pos_m=c, anchor_m=anchor, snr=snr,
                               observable=bool(snr >= SNR_OBSERVABLE),
                               severity=float(dd["severity"]) if dd.get("severity") is not None else 0.7,
                               weight=float(dd["weight"]), type_raw=dd["type_raw"]))
        # 末端：响应锚定（主口径），txt 全长作并行监控口径
        tfl = lengths[g]
        terminal_full_length_m = ce.get("terminal_full_length_m")
        end_anchor = 0.0
        if tfl > 0:
            lo = int(0.6 * tfl / DIST_STEP)
            hi = int(min(GRID_M, 1.35 * tfl) / DIST_STEP)
            if hi > lo:
                k = lo + int(np.argmax(np.abs(imp[lo:hi])))
                end_anchor = float(dgrid[k])
                fz = imp[int(0.7 * tfl / DIST_STEP):int(0.95 * tfl / DIST_STEP)]
                snr_t = float(abs(imp[k]) / (np.sqrt(np.mean(fz ** 2)) + 1e-9)) if fz.size else 0.0
                events.append(dict(event_id="terminal", role="terminal", mechanism="open",
                                   physical_center_m=tfl, axis_pos_m=tfl, anchor_m=end_anchor,
                                   snr=snr_t, observable=bool(snr_t >= SNR_OBSERVABLE),
                                   severity=1.0, weight=1.0, plate_m=tfl))
        length_resp = end_anchor if end_anchor > 0 else min(tfl, GRID_M)

        valid_len = int(min(GRID_BINS, (length_resp * 1.15 + 10) / DIST_STEP)) if length_resp else GRID_BINS
        valid_len = max(valid_len, N_VALID_MIN)
        imp_v = imp[:valid_len].astype(np.float32)
        stp_v = stp[:valid_len].astype(np.float32)

        meta_out = dict(valid_len=valid_len, profile="field", split=sp,
                        length_m=length_resp,
                        terminal_full_length_m=terminal_full_length_m,
                        events=events, source=row["file"], folder=row["folder"],
                        phase=row["phase"], group=g)
        np.savez(OUT / "phases" / f"{row['stem']}.npz",
                 impulse=imp_v, step=stp_v,
                 meta=np.array(json.dumps(meta_out, ensure_ascii=False)))

        for e in events:
            ev_total[e["role"]] = ev_total.get(e["role"], 0) + 1
            if e["role"] != "terminal":
                w_total[1.0 if e["weight"] == 1.0 else 0.3] = w_total.get(1.0 if e["weight"] == 1.0 else 0.3, 0) + 1
                m = e["mechanism"] or "null"
                mech_total[m] = mech_total.get(m, 0) + 1
        out_rows.append(dict(file=f"{row['stem']}.npz", split=sp, profile="field",
                             length_m=length_resp,
                             n_events=len(events),
                             n_joints=sum(1 for e in events if e["role"] == "joint"),
                             n_defects=sum(1 for e in events if e["role"] == "defect"),
                             group=g, source=row["file"]))
        n_phase += 1

    with (OUT / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    # ---- 统计与泄漏自检 ----
    cnt_split: dict[str, int] = {}
    for r in out_rows:
        cnt_split[r["split"]] = cnt_split.get(r["split"], 0) + 1
    g_split: dict[str, set] = {}
    for r in out_rows:
        g_split.setdefault(r["group"], set()).add(r["split"])
    leak_groups = {g: s for g, s in g_split.items() if len(s) > 1}
    print(f"phases written: {n_phase} -> {OUT}")
    if no_table:
        print(f"skipped (vendor 表为空, {len(no_table)}): {no_table}")
    print(f"split counts (phases): {dict(sorted(cnt_split.items()))}")
    gcnt: dict[str, int] = {}
    for g, s in g_split.items():
        key = next(iter(s))
        gcnt[key] = gcnt.get(key, 0) + 1
    print(f"split counts (groups): {dict(sorted(gcnt.items()))}")
    print(f"excluded_long groups (>2500m): {len(excluded)}")
    print(f"events: joints={ev_total.get('joint', 0)} defects={ev_total.get('defect', 0)} terminals={ev_total.get('terminal', 0)}")
    print(f"event weights: engineer(1.0)={w_total.get(1.0, 0)} field(0.3)={w_total.get(0.3, 0)}")
    print(f"mechanisms: {dict(sorted(mech_total.items(), key=lambda kv: -kv[1]))}")
    print(f"LEAK CHECK: fdr 组跨 split 数量 = {len(leak_groups)} (必须为 0)")
    if leak_groups:
        print(leak_groups)
    hist, edges = np.histogram([lengths[g] for g in g_split], bins=[0, 200, 400, 600, 800, 1000, 1500, 2000, 2500])
    print("length hist (groups): " + ", ".join(f"{int(edges[i])}-{int(edges[i+1])}m:{int(c)}" for i, c in enumerate(hist)))

    blind_groups = sorted(g for g, s in g_split.items() if "blind" in s)
    blind_doc = dict(
        generated=datetime.now().isoformat(timespec="seconds"), generated_by="ZCode", seed=SEED,
        note=("封存：blind 组禁止用于任何训练/调参/特征选择；仅供最终一次性评估。"
              "旧 blind（v1 prep_field_train.py 的 12 个浸水试验相）自本文件起作废。"),
        old_blind_invalid="v1 blind = 1500m浸水试验.fdr 的12相（2026-10-30 作废）",
        selection="先剔除 excluded_long(>2500m)，按全长排序均匀抽 ~10% 组（长度分散），再对剩余组 85/15",
        n_groups=len(blind_groups), groups=blind_groups)
    (FIELD_DATA / "blind_groups_v2.json").write_text(
        json.dumps(blind_doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"blind_groups_v2.json: {len(blind_groups)} 组 -> {FIELD_DATA / 'blind_groups_v2.json'}")
    print("excluded_long:", *excluded, sep="\n  ")


if __name__ == "__main__":
    main()
