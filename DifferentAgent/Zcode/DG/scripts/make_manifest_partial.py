"""Write a manifest from whatever sample files exist (for early smoke runs).
The full dataset generation rewrites manifest.csv when it finishes.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

DATA = Path(r"E:\CableNNDataSet\ZCode\DG_V4_S1")


def split_of(template_id: int) -> str:
    if template_id % 10 == 0:
        return "synth_val"
    if template_id % 20 == 10:
        return "synth_test"
    return "train"


def main() -> None:
    rows = []
    for p in sorted((DATA / "phases").glob("sample_*.npz")):
        with np.load(p, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
        rows.append(dict(file=p.name, seed=meta["seed"], template_id=meta["template_id"],
                         realization=meta["realization"], profile=meta["profile"],
                         band=meta["band"], split=split_of(meta["template_id"]),
                         length_m=round(meta["length_m"], 2), n_events=len(meta["events"]),
                         events=json.dumps([{k: e[k] for k in ("event_id", "role", "mechanism",
                                                               "axis_pos_m", "anchor_m", "snr",
                                                               "observable", "severity")}
                                            for e in meta["events"]], ensure_ascii=False)))
    with (DATA / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"partial manifest: {len(rows)} samples")


if __name__ == "__main__":
    main()
