"""Diagnose field_val F1 collapse: FP composition, terminal head behaviour.
Read-only vs training data; writes a JSON report next to the run.
  python diag_field_val.py --weights runs/unet_v2_main/best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import DGDataset, GRID_M, collate, load_manifest
from evaluate import pick_candidates
from model_unet import UNet1DV2

CLASS_OF_ROLE = {"joint": 1, "terminal": 2, "defect": 3}


def run() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_manifest(args.data, args.split)
    ck = torch.load(args.weights, map_location=device, weights_only=False)
    cargs = ck.get("args", {}) or {}
    model = UNet1DV2(width=int(cargs.get("width", 48)), dsconv=bool(cargs.get("dsconv", False))).to(device)
    model.load_state_dict(ck.get("model", ck))
    model.eval()

    ds = DGDataset(rows, args.data, augment=False)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=4, collate_fn=collate, num_workers=0)

    fp_role = Counter()
    fp_prob_bins = Counter()
    fp_dist = Counter()
    fp_per_curve = []
    tp_role = Counter()
    fn_role = Counter()
    term_pred_m, term_gt_m, term_prob = [], [], []
    imp_fp_prob, imp_tp_prob = [], []
    ev_count = Counter()

    with torch.no_grad():
        for batch in loader:
            out = model(batch["x"].to(device))
            probs = torch.sigmoid(out["role"].float()).cpu().numpy()
            term = out["term"].float().cpu().numpy()
            x = batch["x"].cpu().numpy()
            for b in range(probs.shape[0]):
                events = [e for e in batch["events"][b] if e.get("observable", True)]
                ev_count.update(e["role"] for e in events)
                cands = []
                for c in (1, 2, 3):
                    for p, k in pick_candidates(probs[b, c], 0.5):
                        cands.append((p, c, k))
                cands.sort(reverse=True)
                cands = cands[:3]
                used = set()
                for ev in events:
                    c = CLASS_OF_ROLE[ev["role"]]
                    pos = float(ev["anchor_m"])
                    best, bi = None, -1
                    tol = max(5.0, 0.02 * pos)
                    for i, (p, pc, pk) in enumerate(cands):
                        if i in used or pc != c:
                            continue
                        dm = abs(pk * 0.25 - pos)
                        if dm <= tol and (best is None or dm < best):
                            best, bi = dm, i
                    if bi >= 0:
                        used.add(bi)
                        tp_role[ev["role"]] += 1
                        imp_tp_prob.append(float(np.abs(x[b, 0, cands[bi][2]])))
                    else:
                        fn_role[ev["role"]] += 1
                fp_per_curve.append(len(cands) - len(used))
                for i, (p, c, k) in enumerate(cands):
                    if i in used:
                        continue
                    fp_role[c] += 1
                    fp_prob_bins[round(p, 1)] += 1
                    fp_dist[int(k * 0.25 // 250) * 250] += 1
                    imp_fp_prob.append(float(np.abs(x[b, 0, k])))
                vis = [e for e in events if e["role"] == "terminal"]
                if vis:
                    pv = float(torch.sigmoid(torch.tensor(term[b, 1])).item())
                    term_prob.append(pv)
                    if pv > 0.5:
                        term_pred_m.append(float(term[b, 0]) * GRID_M)
                        term_gt_m.append(float(vis[0]["anchor_m"]))

    n = len(fp_per_curve)
    report = dict(
        weights=args.weights, n_samples=n,
        fp_per_curve_mean=float(np.mean(fp_per_curve)),
        curves_with_fp=int(sum(v > 0 for v in fp_per_curve)),
        fp_by_role={"joint": fp_role[1], "terminal": fp_role[2], "defect": fp_role[3]},
        fp_prob_hist={f"{k:.1f}": v for k, v in sorted(fp_prob_bins.items())},
        fp_by_distance_250m={str(k): v for k, v in sorted(fp_dist.items())},
        tp_by_role=dict(tp_role), fn_by_role=dict(fn_role), events_by_role=dict(ev_count),
        fp_impulse_amp_median=float(np.median(imp_fp_prob)) if imp_fp_prob else None,
        tp_impulse_amp_median=float(np.median(imp_tp_prob)) if imp_tp_prob else None,
        term_pred_rate=float(np.mean([p > 0.5 for p in term_prob])),
        term_pred_m_median=float(np.median(term_pred_m)) if term_pred_m else None,
        term_gt_m_median=float(np.median(term_gt_m)) if term_gt_m else None,
        term_signed_err_median=float(np.median(np.array(term_pred_m) - np.array(term_gt_m))) if term_pred_m else None,
        term_gt_values=sorted(term_gt_m),
        term_pred_values=sorted(term_pred_m),
    )
    out = Path(args.weights).parent / (Path(args.weights).stem + "_diag_" + args.split + ".json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"E:\CableNNDataSet\ZCode\field_train")
    ap.add_argument("--split", default="field_val")
    ap.add_argument("--weights", default=r"runs\unet_v2_main\best.pt")
    args = ap.parse_args()
    run()
