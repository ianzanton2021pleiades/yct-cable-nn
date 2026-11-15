r"""Evaluation v3 under the frozen protocol (v2 amendment): NN decode with
terminal-channel quota, dual-terminal criteria (response-anchored anchor_m
main / txt full-length parallel), rule-based peak-detection baseline, and
group-level bootstrap 95% CI for F1@20 (role-matched).

    python evaluate.py --data E:\CableNNDataSet\ZCode\field_train_v2 \
        --split field_val --weights runs\v3_mixed\best.pt --budget 3.0 \
        --out runs\v3_mixed\eval_field_val.json

Workpoints: main budget 3.0 + event-density budget ceil(1.2 x events/curve).
Matching: same-role one-to-one (role-matched) and role-agnostic, each under
adaptive tol max(5 m, 2% d) and fixed 20 m.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import DGDataset, GRID_M, collate, load_manifest
from model_unet import UNet1D, UNet1DV2

CLASS_OF_ROLE = {"joint": 1, "terminal": 2, "defect": 3}


def pick_candidates(probs: np.ndarray, threshold: float, min_gap_m: float = 25.0):
    """Local maxima above threshold, non-max suppressed by distance."""
    cands = []
    gap = int(min_gap_m / 0.25)
    for k in range(1, probs.size - 1):
        if probs[k] >= threshold and probs[k] >= probs[k - 1] and probs[k] >= probs[k + 1]:
            if cands and k - cands[-1][1] < gap:
                if probs[k] > cands[-1][0]:
                    cands[-1] = (probs[k], k)
            else:
                cands.append((probs[k], k))
    return sorted(cands, reverse=True)


def decode_cands(probs_row: np.ndarray, threshold: float, budget: float,
                 valid_len: int) -> list[tuple]:
    """Role channels -> candidates [(prob, channel, bin)].

    Terminal channel (2) contributes at most its single highest local maximum
    (protocol v2 9.4: one end per cable).  Candidates beyond the valid length
    are dropped.  Cross-channel confidence sort, top-budget keep."""
    cands = []
    for c in (1, 2, 3):
        peaks = pick_candidates(probs_row[c], threshold)
        if c == 2:
            peaks = peaks[:1]
        for p, k in peaks:
            if k < valid_len:
                cands.append((float(p), c, int(k)))
    cands.sort(reverse=True)
    return cands[:max(1, int(round(budget)))]


def match_events(events: list[tuple], cands: list[tuple], tol_fn,
                 role_matched: bool = True) -> tuple[int, int, int]:
    """One-to-one greedy matching (nearest within tol, per event in order).
    events: [(role, pos_m)]; cands: [(prob, channel, bin)].
    Returns (tp, fp, fn)."""
    used: set[int] = set()
    tp = fn = 0
    for role, pos in events:
        best, bi = None, -1
        for i, (p, pc, pk) in enumerate(cands):
            if i in used or (role_matched and pc != CLASS_OF_ROLE[role]):
                continue
            dm = abs(pk * 0.25 - pos)
            if dm <= tol_fn(pos) and (best is None or dm < best):
                best, bi = dm, i
        if bi >= 0:
            used.add(bi)
            tp += 1
        else:
            fn += 1
    return tp, len(cands) - len(used), fn


def decode_f1(role_logits: torch.Tensor, term_logits: torch.Tensor,
              batch: dict, tol_m: float = 20.0, threshold: float = 0.5,
              budget: float = 3.0) -> tuple[float, float]:
    """F1@tol over role classes joint/terminal/defect with the v3 decode
    (terminal quota 1, valid-length restriction) + terminal mean error.
    Used by train.py for val/monitor splits."""
    probs = torch.sigmoid(role_logits).detach().cpu().numpy()
    tp = fp = fn = 0
    term_err = []
    for b in range(probs.shape[0]):
        cands = decode_cands(probs[b], threshold, budget, int(batch["valid_len"][b]))
        gt = [(ev["role"], float(ev["anchor_m"]))
              for ev in batch["events"][b] if ev.get("observable", True)]
        t, f, n = match_events(gt, cands, lambda pos: tol_m, True)
        tp += t
        fp += f
        fn += n
        vis = [ev for ev in batch["events"][b]
               if ev["role"] == "terminal" and ev.get("observable", True)]
        if vis:
            pv = torch.sigmoid(torch.tensor(term_logits[b, 1].item())).item()
            if pv > 0.5:
                pred_m = float(term_logits[b, 0].item()) * GRID_M
                term_err.append(abs(pred_m - vis[0]["anchor_m"]))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    term_mae = float(np.mean(term_err)) if term_err else float("nan")
    return f1, term_mae


def baseline_cands(imp_row: np.ndarray, valid_len: int, budget: int,
                   l_full_m: float) -> list[tuple]:
    """Rule baseline (protocol v2 9.2): |impulse| local maxima inside the
    valid region, amplitude sort, NMS 25 m, top-budget peaks; the strongest
    peak inside [0.6, 1.35] x l_full becomes the terminal candidate, the
    rest are naive 'joint' event candidates."""
    n = min(valid_len, imp_row.size)
    amp = np.abs(imp_row[:n].astype(np.float64))
    peaks = pick_candidates(amp, 1e-9)[:max(1, budget)]
    lo, hi = 0.6 * l_full_m / 0.25, 1.35 * l_full_m / 0.25
    term_k = None
    for p, k in peaks:  # peaks are amplitude-sorted: first hit = strongest in window
        if lo <= k <= hi:
            term_k = k
            break
    cands = [(float(p), 2 if k == term_k else 1, int(k)) for p, k in peaks]
    cands.sort(reverse=True)
    return cands


def workpoint_metrics(recs: list[dict], cands_key: str, budget: float,
                      role_matched: bool) -> dict:
    counts = {"adaptive": [0, 0, 0], "fixed20": [0, 0, 0]}  # tp/fp/fn
    buckets: dict[int, list[int]] = {}
    for r in recs:
        cands = sorted(r[cands_key], reverse=True)[:max(1, int(round(budget)))]
        for tol_mode in counts:
            tp, fp, fn = match_events(r["events"], cands,
                                      lambda pos: max(5.0, 0.02 * pos) if tol_mode == "adaptive" else 20.0,
                                      role_matched)
            counts[tol_mode][0] += tp
            counts[tol_mode][1] += fp
            counts[tol_mode][2] += fn
            if tol_mode == "fixed20" and role_matched:
                for role, pos in r["events"]:
                    bk = int(pos // 250)
                    buckets.setdefault(bk, [0, 0])
                    buckets[bk][1] += 1
                hit_pos = set()
                for role, pos in r["events"]:
                    best, bi = None, -1
                    for i, (p, pc, pk) in enumerate(cands):
                        if i in hit_pos or pc != CLASS_OF_ROLE[role]:
                            continue
                        dm = abs(pk * 0.25 - pos)
                        if dm <= 20.0 and (best is None or dm < best):
                            best, bi = dm, i
                    if bi >= 0:
                        hit_pos.add(bi)
                        buckets[int(pos // 250)][0] += 1

    def prf(c):
        tp, fp, fn = c
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        return prec, rec, 2 * prec * rec / max(prec + rec, 1e-9)

    pa, ra, fa = prf(counts["adaptive"])
    p2, r2, f2 = prf(counts["fixed20"])
    return dict(budget=budget, role_matched=role_matched,
                f1=round(fa, 4), precision=round(pa, 4), recall=round(ra, 4),
                f1_20=round(f2, 4), precision_20=round(p2, 4), recall_20=round(r2, 4),
                tp=counts["adaptive"][0], fp=counts["adaptive"][1], fn=counts["adaptive"][2],
                tp_20=counts["fixed20"][0], fp_20=counts["fixed20"][1], fn_20=counts["fixed20"][2],
                hit_by_distance_bucket_20m={f"{k * 250}-{(k + 1) * 250}m": f"{v[0]}/{v[1]}"
                                            for k, v in sorted(buckets.items())})


def terminal_stats(recs: list[dict], mode: str, source: str = "nn") -> dict:
    """Full-denominator terminal coverage.  mode 'anchor' = response-anchored
    (main), 'full' = txt full length (samples without it are skipped)."""
    tot = hit = 0
    errs = []
    for r in recs:
        if not r["term_visible"]:
            continue
        if mode == "full":
            if math.isnan(r["l_full"]):
                continue
            target = r["l_full"]
        else:
            target = r["term_anchor"]
        tot += 1
        pred = r["term_pred_nn"] if source == "nn" else r["term_pred_base"]
        if pred is not None:
            err = abs(pred - target)
            errs.append(err)
            if err <= 50:
                hit += 1
    return dict(denominator=tot, output_rate=round(sum(
        1 for r in recs if r["term_visible"] and
        (r["term_pred_nn"] if source == "nn" else r["term_pred_base"]) is not None) / max(tot, 1), 3),
        hit50=hit, mae_m=round(float(np.mean(errs)), 1) if errs else None)


def bootstrap_f1_ci(recs: list[dict], budget: float, b: int = 1000,
                    seed: int = 0) -> dict | None:
    """Group-level bootstrap 95% CI of F1@20 (role-matched, fixed tol 20 m)
    for NN vs baseline and their paired difference."""
    groups = sorted({r["group"] for r in recs if r["group"]})
    if not groups:
        return None
    gidx = {g: [i for i, r in enumerate(recs) if r["group"] == g] for g in groups}

    def per_rec_counts(key):
        out = {}
        for i, r in enumerate(recs):
            cands = sorted(r[key], reverse=True)[:max(1, int(round(budget)))]
            out[i] = match_events(r["events"], cands, lambda pos: 20.0, True)
        return out

    nn_c = per_rec_counts("nn_cands")
    ba_c = per_rec_counts("base_cands")
    rng = np.random.default_rng(seed)
    f_nn, f_ba, f_df = [], [], []
    for _ in range(b):
        pick = rng.choice(len(groups), size=len(groups), replace=True)
        cnt = {"nn": [0, 0, 0], "ba": [0, 0, 0]}
        for g in pick:
            for i in gidx[groups[int(g)]]:
                for tag, c in (("nn", nn_c[i]), ("ba", ba_c[i])):
                    for j in range(3):
                        cnt[tag][j] += c[j]

        def f1(c):
            prec = c[0] / max(c[0] + c[1], 1)
            rec = c[0] / max(c[0] + c[2], 1)
            return 2 * prec * rec / max(prec + rec, 1e-9)

        a, b2 = f1(cnt["nn"]), f1(cnt["ba"])
        f_nn.append(a)
        f_ba.append(b2)
        f_df.append(a - b2)

    def ci(x):
        return [round(float(np.percentile(x, 2.5)), 4), round(float(np.percentile(x, 97.5)), 4)]

    return dict(b=b, n_groups=len(groups), f1_20_nn_ci=ci(f_nn),
                f1_20_baseline_ci=ci(f_ba), diff_nn_minus_baseline_ci=ci(f_df))


def collect_records(model, rows, data_dir, device, budget: float,
                    threshold: float) -> list[dict]:
    """NN candidates are kept untruncated (workpoints slice later); baseline
    peaks are generated per sample with a floor of the main budget and the
    max event-density budget, since its terminal rule depends on the cut."""
    ds = DGDataset(rows, data_dir, augment=False)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=4, collate_fn=collate, num_workers=2)
    sid2row = {r["file"]: r for r in rows}
    recs = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            out = model(x)
            probs = torch.sigmoid(out["role"].float()).cpu().numpy()
            term = out["term"].float().cpu().numpy()
            xnp = batch["x"].cpu().numpy()
            for b in range(probs.shape[0]):
                sid = batch["sample_id"][b]
                row = sid2row[sid]
                valid_len = int(batch["valid_len"][b])
                events = [(e["role"], float(e["anchor_m"]))
                          for e in batch["events"][b] if e.get("observable", True)]
                l_full = float(batch["l_full"][b])
                l_full_eff = l_full if not math.isnan(l_full) else float(row["length_m"] or 0)
                cands = decode_cands(probs[b], threshold, 1e9, valid_len)
                base_budget = max(int(round(budget)), math.ceil(1.2 * len(events)))
                base = baseline_cands(xnp[b, 0], valid_len, base_budget, l_full_eff)
                tvis = False
                tanchor = float("nan")
                for e in batch["events"][b]:
                    if e["role"] == "terminal" and e.get("observable", True):
                        tvis = True
                        tanchor = float(e["anchor_m"])
                pred_nn = None
                if torch.sigmoid(torch.tensor(term[b, 1])).item() > 0.5:
                    pred_nn = float(term[b, 0]) * GRID_M
                pred_base = None
                tc = [c for c in base if c[1] == 2]
                if tc:
                    pred_base = tc[0][2] * 0.25
                recs.append(dict(group=row.get("group", ""), events=events,
                                 nn_cands=cands, base_cands=base,
                                 term_visible=tvis, term_anchor=tanchor,
                                 term_pred_nn=pred_nn, term_pred_base=pred_base,
                                 l_full=l_full, length_m=float(row["length_m"] or 0)))
    return recs


def evaluate_split(model, rows, data_dir, device, budget: float,
                   threshold: float, b_boot: int = 1000) -> dict:
    recs = collect_records(model, rows, data_dir, device, budget, threshold)
    n_ev = float(np.mean([len(r["events"]) for r in recs])) if recs else 0.0
    b_density = max(1, math.ceil(1.2 * n_ev))
    budget_max = max(budget, b_density)

    def block(key):
        return {"main": workpoint_metrics(recs, key, budget, True),
                "main_role_agnostic": workpoint_metrics(recs, key, budget, False),
                "density": workpoint_metrics(recs, key, b_density, True),
                "density_role_agnostic": workpoint_metrics(recs, key, b_density, False)}

    nn_block = block("nn_cands")
    base_block = block("base_cands")
    return dict(n_samples=len(recs),
                mean_events_per_curve=round(n_ev, 2),
                threshold=threshold,
                nn=dict(role_matched=nn_block["main"],
                        role_matched_density=nn_block["density"],
                        role_agnostic=nn_block["main_role_agnostic"],
                        role_agnostic_density=nn_block["density_role_agnostic"],
                        terminal_anchor=terminal_stats(recs, "anchor", "nn"),
                        terminal_full=terminal_stats(recs, "full", "nn")),
                baseline=dict(role_matched=base_block["main"],
                              role_matched_density=base_block["density"],
                              role_agnostic=base_block["main_role_agnostic"],
                              role_agnostic_density=base_block["density_role_agnostic"],
                              terminal_anchor=terminal_stats(recs, "anchor", "base"),
                              terminal_full=terminal_stats(recs, "full", "base")),
                bootstrap_ci=bootstrap_f1_ci(recs, budget, b_boot))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="blind")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--budget", type=float, default=3.0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--b", type=int, default=1000, help="bootstrap resamples")
    ap.add_argument("--width", type=int, default=0,
                    help="0 = take width/arch from the checkpoint args")
    ap.add_argument("--arch", default="", choices=("", "v1", "v2"),
                    help="empty = auto-detect from checkpoint args")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_manifest(args.data, args.split)
    if args.limit:
        rows = rows[:args.limit]
    ck = torch.load(args.weights, map_location=device, weights_only=False)
    cargs = ck.get("args", {}) or {}
    arch = args.arch or cargs.get("arch", "v1")
    width = args.width or int(cargs.get("width", 32))
    extra_log = bool(cargs.get("extra_log", False))
    if arch == "v1":
        model = UNet1D(width=width, cin=5 if extra_log else 4).to(device)
    else:
        model = UNet1DV2(width=width, dsconv=bool(cargs.get("dsconv", False)),
                         cin=5 if extra_log else 4).to(device)
    model.load_state_dict(ck.get("model", ck))
    model.eval()
    res = evaluate_split(model, rows, args.data, device, args.budget,
                         args.threshold, args.b)
    res = dict(split=args.split, weights=args.weights, **res)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                  encoding="utf-8")


if __name__ == "__main__":
    main()
