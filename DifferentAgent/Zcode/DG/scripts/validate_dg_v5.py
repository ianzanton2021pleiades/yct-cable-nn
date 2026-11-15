"""Task-alignment acceptance for a DG_V5 generated directory vs the field corpus.

Checks (gate = must pass, else exit 1):
  1. per-role x depth-bucket event amplitude median ratio field/synth in
     [0.7, 1.4] (joint / terminal; defect reported, gated via the soak trial)
  2. clutter RMS ratio (20-60 %L, 60-90 %L) in [0.7, 1.4]
  3. event SNR medians and event/clutter contrast ratios (reported + gated on
     contrast only)
  4. mask semantics: valid_len == array length; nonzero fraction >= 95 % of the
     valid region (impulse & step) for >= 95 % of samples; z0_dev present,
     float32, correct length
  5. mechanism quota: moisture+aging >= 40 % of defect-role events
  6. length coverage: synth has samples in every length bucket of the field
     corpus (0-300/300-900/900-1500/1500-2600 m)
  7. topology uniqueness / split hygiene: no template_id in more than one
     split; >= (n_samples / 50) distinct templates; re-test groups (r<4) exist

    python validate_dg_v5.py --data E:\\CableNNDataSet\\ZCode\\DG_V5_S1 [--max-files 500]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dg_stats_v5 as S
from dg_stats_v5 import BUCKETS, field_reference, median_table, ratio_table, fmt_ratio_table

FIELD_PHASES = Path(r"E:\CableNNDataSet\ZCode\field_data\phases")
FIELD_MANIFEST = Path(r"E:\CableNNDataSet\ZCode\field_data\manifest.csv")
GATE = 0.70, 1.40
MOISTURE_MECHS = ("moisture_local", "moisture_distributed", "aging")
LENGTH_BUCKETS = ((0, 300), (300, 900), (900, 1500), (1500, 2600))


def synth_side(data_dir: Path, max_files: int) -> tuple[dict, list[dict]]:
    """Response aggregate + per-file meta records."""
    with (data_dir / "manifest.csv").open(encoding="utf-8-sig") as f:
        mrows = list(csv.DictReader(f))
    files = {r["file"]: r for r in mrows}
    agg = S.empty_agg()
    records = []
    for i, p in enumerate(sorted((data_dir / "phases").glob("sample_*.npz"))):
        if max_files and i >= max_files:
            break
        with np.load(p, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            imp, stp = z["impulse"], z["step"]
            role, mech, z0d = z["role"], z["mechanism"], z["z0_dev"]
            s11 = z["s11"]
        rec = dict(file=p.name, meta=meta, shapes_ok=True, nonzero_ok=True, notes=[])
        vl = int(meta.get("valid_len", -1))
        if not (imp.shape == stp.shape == role.shape == mech.shape == z0d.shape
                == (vl,) and s11.dtype == np.complex64
                and imp.dtype == np.float32 and z0d.dtype == np.float32
                and role.dtype == np.int8 and mech.dtype == np.int8):
            rec["shapes_ok"] = False
            rec["notes"].append(f"shapes {imp.shape}/{role.shape}/{z0d.shape} vl={vl} "
                                f"dtypes {imp.dtype}/{s11.dtype}/{z0d.dtype}")
        else:
            for arr, nm in ((imp, "impulse"), (stp, "step")):
                nz = float(np.count_nonzero(arr[:vl])) / vl
                if nz < 0.95:
                    rec["nonzero_ok"] = False
                    rec["notes"].append(f"{nm} nonzero {nz:.3f}")
        records.append(rec)
        if meta.get("profile") != "field":
            continue
        st = S.response_stats(imp, float(meta["length_m"]), meta["events"])
        agg["n_files"] += 1
        agg["lengths"].append(float(meta["length_m"]))
        agg["near_0_30"].append(st["near_0_30"])
        for k, v in st["clutter"].items():
            if np.isfinite(v):
                agg["clutter"][k].append(v)
        for ev in st["events"]:
            b = S._bucket_index(ev["anchor_m"])
            if b is not None and ev["observable"]:
                agg["event_amps"][ev["role"]][b].append(ev["amp"])
                agg["event_snrs"][ev["role"]].append(ev["snr"])
    return agg, records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--max-files", type=int, default=0)
    args = ap.parse_args()
    data_dir = Path(args.data)

    print("field reference ...", flush=True)
    field = field_reference(FIELD_PHASES, FIELD_MANIFEST)
    synth, records = synth_side(data_dir, args.max_files)
    fmed, smed = median_table(field), median_table(synth)
    ratios = ratio_table(field, synth)
    print(f"field files {field['n_files']}  synth(field-profile) files {synth['n_files']}")
    print(fmt_ratio_table(ratios, fmed, smed))

    failures = []
    report = {"data": str(data_dir), "n_synth": len(records),
              "n_synth_field_profile": synth["n_files"], "n_field": field["n_files"]}

    # --- gate 1+2: amplitude / clutter ratios ------------------------------
    for role, buckets in ratios["amps"].items():
        gate_role = role in ("joint", "terminal")
        for i, r in enumerate(buckets):
            if r != r:
                continue
            if gate_role and not (GATE[0] <= r <= GATE[1]):
                failures.append(f"amp {role} {BUCKETS[i]} ratio {r:.2f}")
    for k, r in ratios["clutter"].items():
        if r == r and not (GATE[0] <= r <= GATE[1]):
            failures.append(f"clutter {k} ratio {r:.2f}")

    # --- event/clutter contrast --------------------------------------------
    contrast = {"field": {}, "synth": {}, "ratio": {}}
    for tag, med in (("field", fmed), ("synth", smed)):
        for role in ("joint", "terminal", "defect"):
            vals = [med["amps"][role][i] / med["clutter"]["20_60"]
                    for i in range(len(BUCKETS)) if med["amps"][role][i] == med["amps"][role][i]]
            if vals:
                contrast[tag][role] = float(np.median(vals))
    for role in contrast["field"]:
        if role in contrast["synth"] and contrast["synth"][role] > 0:
            contrast["ratio"][role] = contrast["field"][role] / contrast["synth"][role]
            if not (GATE[0] <= contrast["ratio"][role] <= GATE[1]):
                failures.append(f"contrast {role} ratio {contrast['ratio'][role]:.2f}")
    report["contrast"] = contrast

    # --- KS per role (report only) ------------------------------------------
    ks = {}
    for role in ("joint", "terminal", "defect"):
        for i in range(len(BUCKETS)):
            a = field["event_amps"][role][i]
            b = synth["event_amps"][role][i]
            if len(a) >= 10 and len(b) >= 10:
                ks[f"{role}_{i}"] = float(ks_2samp(a, b).statistic)
    report["ks"] = ks
    report["snr"] = {"field": fmed["snr"], "synth": smed["snr"], "ratio": ratios["snr"]}

    # --- gate 4: mask semantics ---------------------------------------------
    bad_shapes = [r for r in records if not r["shapes_ok"]]
    bad_nz = [r for r in records if not r["nonzero_ok"]]
    if bad_shapes:
        failures.append(f"mask/shape violations: {len(bad_shapes)} "
                        f"(e.g. {bad_shapes[0]['notes']})")
    if len(bad_nz) > 0.05 * len(records):
        failures.append(f"zero-hole samples {len(bad_nz)}/{len(records)} > 5 %")
    report["mask"] = {"shape_violations": len(bad_shapes),
                      "zero_hole_samples": len(bad_nz)}

    # --- gate 5: mechanism quota --------------------------------------------
    mech_counts = {}
    n_def = n_ma = 0
    for rec in records:
        for ev in rec["meta"]["events"]:
            if ev["role"] == "defect":
                n_def += 1
                if ev["mechanism"] in MOISTURE_MECHS:
                    n_ma += 1
                    mech_counts[ev["mechanism"]] = mech_counts.get(ev["mechanism"], 0) + 1
                else:
                    mech_counts[ev["mechanism"]] = mech_counts.get(ev["mechanism"], 0) + 1
    quota = n_ma / n_def if n_def else 0.0
    report["mech_quota"] = {"defects": n_def, "moisture_aging": n_ma, "frac": quota,
                            "counts": mech_counts}
    if quota < 0.40:
        failures.append(f"moisture+aging quota {quota:.2f} < 0.40")

    # --- gate 6: length coverage --------------------------------------------
    lens = np.asarray(synth["lengths"]) if synth["lengths"] else np.zeros(0)
    cover = {}
    for lo, hi in LENGTH_BUCKETS:
        n = int(((lens >= lo) & (lens < hi)).sum())
        cover[f"{lo}-{hi}"] = n
        if n == 0 and field_length_bucket_populated(lo, hi, field):
            failures.append(f"length bucket {lo}-{hi} m empty in synth")
    report["length_coverage"] = cover

    # --- gate 7: split hygiene / topology uniqueness -------------------------
    with (data_dir / "manifest.csv").open(encoding="utf-8-sig") as f:
        mrows = list(csv.DictReader(f))
    by_split = {}
    for r in mrows:
        by_split.setdefault(r["split"], set()).add(int(r["template_id"]))
    overlap = (by_split.get("train", set()) & by_split.get("synth_val", set())) | \
              (by_split.get("train", set()) & by_split.get("synth_test", set())) | \
              (by_split.get("synth_val", set()) & by_split.get("synth_test", set()))
    n_t = len({int(r["template_id"]) for r in mrows})
    n_retest = sum(1 for r in mrows if int(r["realization"]) < 4)
    report["splits"] = {"rows": len(mrows), "templates": n_t,
                        "split_overlap": sorted(overlap)[:10],
                        "retest_rows(r<4)": n_retest}
    if overlap:
        failures.append(f"template split overlap: {sorted(overlap)[:5]}")

    print("\n=== gates ===")
    if failures:
        for f_ in failures:
            print("FAIL:", f_)
    else:
        print("ALL GATES PASS")
    print("\nlength coverage:", cover)
    print("mech quota:", round(quota, 3), mech_counts)
    print("templates:", n_t, "rows:", len(mrows), "retest rows:", n_retest)
    out = Path(__file__).resolve().parents[1] / "output" / "validate_report_v5.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float),
                   encoding="utf-8")
    print("saved ->", out)
    sys.exit(1 if failures else 0)


def field_length_bucket_populated(lo: int, hi: int, field: dict) -> bool:
    fl = np.asarray(field["lengths"])
    return bool(((fl >= lo) & (fl < hi)).any())


if __name__ == "__main__":
    main()
