"""Compare two threshold sweeps at matched candidate budgets.

Field hit rate depends on how many candidates the operator is willing to look
at, so comparing two models at their default threshold is misleading when their
precision differs.  This interpolates the hit-rate curve of each sweep onto a
common candidates-per-curve grid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> tuple[list[float], list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    curve = sorted(payload["curve"], key=lambda row: row["candidates_per_curve"])
    return ([row["candidates_per_curve"] for row in curve],
            [row["hit_rate_20"] for row in curve])


def interpolate(candidates: list[float], hits: list[float], target: float) -> float | None:
    if target < candidates[0] or target > candidates[-1]:
        return None
    for index in range(1, len(candidates)):
        if candidates[index] >= target:
            left, right = candidates[index - 1], candidates[index]
            span = right - left
            weight = 0.0 if span <= 0 else (target - left) / span
            return hits[index - 1] + weight * (hits[index] - hits[index - 1])
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True, help="sweep json of the reference model")
    parser.add_argument("--b", type=Path, required=True, help="sweep json of the new model")
    parser.add_argument("--label-a", default="v1")
    parser.add_argument("--label-b", default="v2")
    parser.add_argument("--budgets", default="1.5,2.0,2.5,3.0,4.0,5.0,6.0")
    args = parser.parse_args()

    candidates_a, hits_a = load(args.a)
    candidates_b, hits_b = load(args.b)
    print(f"| candidates/curve | {args.label_a} hit@20 | {args.label_b} hit@20 | delta |")
    print("|---|---|---|---|")
    for budget in [float(item) for item in args.budgets.split(",")]:
        left = interpolate(candidates_a, hits_a, budget)
        right = interpolate(candidates_b, hits_b, budget)
        text = lambda value: "—" if value is None else f"{value:.3f}"
        delta = "—" if (left is None or right is None) else f"{right - left:+.3f}"
        print(f"| {budget:.1f} | {text(left)} | {text(right)} | {delta} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
