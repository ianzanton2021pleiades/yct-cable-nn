"""Evaluate v3 checkpoints or a physical step-detrend detector on real labeled cases."""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import load_config
from features_v3 import preprocess_s11_v3
from metrics import _local_peaks
from real_cases import all_cases, read_csv
from train_v3 import model_from_checkpoint


def decode(score, d, threshold, min_sep, near_min, top_k, max_m):
    idx = _local_peaks(score, threshold)
    idx.sort(key=lambda i: -float(score[i]))
    selected = []
    for i in idx:
        if d[i] < near_min or d[i] > max_m:
            continue
        if all(abs(d[i] - d[j]) >= min_sep for j in selected):
            selected.append(i)
        if len(selected) >= top_k:
            break
    return sorted([(float(d[i]), float(score[i])) for i in selected])


def match(pred, truth, tol):
    if not pred or not truth:
        return 0
    cost = np.abs(np.asarray(truth)[:, None] - np.asarray(pred)[None, :])
    compatible = cost <= tol
    if not compatible.any():
        return 0
    big = (len(truth) + len(pred) + 1) * 1e6
    rows, cols = linear_sum_assignment(np.where(compatible, cost, big))
    return int(sum(compatible[r, c] for r, c in zip(rows, cols)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--detector", choices=("none", "step_local", "step_detrend", "step_diff", "imp_gain"),
                        default="none")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-sep", type=float, default=5.0)
    parser.add_argument("--near-min", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--tolerances", type=float, nargs="+", default=[5, 10, 20, 30])
    parser.add_argument("--groups", nargs="+", default=["wet_1500", "rg58_74", "rg58_3line"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.detector == "none" and args.checkpoint is None:
        parser.error("provide --checkpoint or --detector")

    device = torch.device(args.device)
    if args.checkpoint:
        model, state = model_from_checkpoint(args.checkpoint, device)
        cfg = state["config"]
    else:
        cfg = load_config(HERE / "config.json")
    d = np.linspace(0.0, cfg["response"]["distance_max_m"], cfg["response"]["points"])
    cases = [c for c in all_cases() if c.group in args.groups]
    groups = defaultdict(lambda: {"n": 0, "tp": defaultdict(int), "fp": defaultdict(int),
                                  "fn": defaultdict(int), "term_hits": 0, "term_err": []})
    for number, case in enumerate(cases):
        f, g = read_csv(case.path)
        x = preprocess_s11_v3(f, g, cfg)
        if args.detector == "none":
            with torch.inference_mode():
                p = model(torch.from_numpy(x[None]).to(device)).sigmoid()[0].float().cpu().numpy()
            event_score, terminal_score = p[0], p[2]
        else:
            channel = {"step_local": 0, "step_detrend": 2, "step_diff": 3, "imp_gain": 4}[args.detector]
            event_score = np.abs(x[channel])
            terminal_score = np.abs(x[2]) if args.detector != "step_diff" else np.abs(x[3])
        pred = decode(event_score, d, args.threshold, args.min_sep, args.near_min, args.top_k, case.end_m + 50)
        positions = [a for a, _ in pred]
        s = groups[case.group]
        s["n"] += 1
        for tol in args.tolerances:
            m = match(positions, case.events_m, tol)
            s["tp"][tol] += m
            s["fp"][tol] += len(positions) - m
            s["fn"][tol] += len(case.events_m) - m
        term = decode(terminal_score, d, args.threshold, 1.0, 10.0, 1, d[-1] - 15)
        if term:
            err = abs(term[0][0] - case.end_m)
            s["term_err"].append(err)
            s["term_hits"] += err <= 10.0
        if args.verbose:
            print(f"{number:3d} {case.group:12s} {case.path.name[:55]:55s} "
                  f"truth={[round(v,1) for v in case.events_m]} pred={[round(v,1) for v in positions[:8]]} "
                  f"term={None if not term else round(term[0][0],1)}")
    print("\n=== summary ===")
    for group in args.groups:
        if group not in groups:
            continue
        s = groups[group]
        parts = [f"{group}: n={s['n']}"]
        for tol in args.tolerances:
            tp, fp, fn = s["tp"][tol], s["fp"][tol], s["fn"][tol]
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * p * r / (p + r) if p + r else 0.0
            parts.append(f"@{tol:g}m P={p:.3f} R={r:.3f} F1={f1:.3f}")
        err = float(np.mean(s["term_err"])) if s["term_err"] else None
        parts.append(f"term {s['term_hits']}/{s['n']} err={err:.1f}" if err is not None else "term n/a")
        print(" | ".join(parts))


if __name__ == "__main__":
    main()
