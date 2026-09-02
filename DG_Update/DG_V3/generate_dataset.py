"""DG V3 dataset CLI.

Storage is intentionally supplied by the follow-up storage module.  This CLI
uses its function contract (``make_dataset_metadata``,
``write_dataset_metadata``, ``write_frequency`` and ``write_annotations``);
the generator owns no file format.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

from dg_v3 import iter_samples, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate direct physical DG V3 complex-S11 samples")
    parser.add_argument("--n-total", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--profile", choices=("mixed", "rg58", "field"), default="mixed")
    parser.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 1, 1))
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "configs" / "provisional_v1.yaml"),
        help="complete versioned dg-v3 YAML configuration",
    )
    parser.add_argument("--output", required=True, help="storage destination passed to DatasetStorage")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    # Deliberately a required integration point: storage owns the file format.
    from storage import DatasetStorage

    storage = DatasetStorage.open(
        args.output,
        config=config,
        seed=args.seed,
        requested_samples=args.n_total,
    )
    started = time.time()
    for index, sample in enumerate(
        iter_samples(args.n_total, args.seed, args.profile, config, args.workers),
        1,
    ):
        storage.write(sample)
        if index == 1 or index == args.n_total or index % max(1, min(25, args.n_total // 10 or 1)) == 0:
            elapsed = time.time() - started
            rate = index / max(elapsed, 1e-9)
            eta = (args.n_total - index) / max(rate, 1e-9)
            print(
                f"[{index}/{args.n_total}] profile={sample.profile} "
                f"length={sample.topology.length_m:.1f}m rate={rate:.2f}/s ETA={eta:.0f}s",
                flush=True,
            )
    storage.close()
    print(f"dataset={args.output} elapsed={time.time() - started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
