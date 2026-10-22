"""Inference starts from measured S11; field labels remain unknown."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from common import HERE, SOURCE_ROOT, distance_grid, preprocess_s11, read_s11, write_json
from metrics import decode_predictions
from train import model_from_checkpoint


def terminal_consistency(prediction, inputs, cfg):
    """Expose the physical evidence for a candidate rather than forcing an end."""
    candidate = prediction["terminal"]
    if candidate is None:
        return {"status": "no_candidate"}
    d = distance_grid(cfg)
    p = candidate["distance_m"]
    impulse, step = inputs[:2]
    near = max(10., .02 * p)
    peak = np.abs(impulse[np.abs(d - p) <= 5])
    background = impulse[(np.abs(d - p) >= 15) & (np.abs(d - p) <= 40)]
    left = step[(d >= p - 15) & (d <= p - 5)]
    right = step[(d >= p + 5) & (d <= p + 15)]
    change = float(np.median(right) - np.median(left)) if len(left) and len(right) else None
    rms = float(np.sqrt(np.mean(background**2))) if len(background) else 0.
    pulse_ratio = float(peak.max() / max(rms, 1e-12)) if len(peak) else 0.
    later = [x["distance_m"] for x in prediction["events"] if x["confidence"] >= .8]
    checks = {"outside_near_end": p > near,
              "away_from_right_boundary": p < d[-1] - 15.,
              "after_high_score_events": not later or p >= max(later) - 10.,
              "local_impulse_present": pulse_ratio >= 2.,
              "open_end_step_rises": change is not None and change > 0.}
    return {"status": "consistent_candidate" if all(checks.values()) else "uncertain_candidate",
            "checks": checks, "local_peak_to_background_rms": pulse_ratio,
            "normalized_step_change": change,
            "note": "Heuristic consistency for the open-end training scope; not an independent field length measurement"}


def predict_one(model, f, s11, cfg, device, thresholds):
    inputs = preprocess_s11(f, s11, cfg)
    with torch.inference_mode():
        probabilities = model(torch.from_numpy(inputs[None]).to(device)).sigmoid()[0].float().cpu().numpy()
    result = decode_predictions(probabilities, cfg, thresholds)
    result["terminal_consistency"] = terminal_consistency(result, inputs, cfg)
    return result, inputs, probabilities


def plot_prediction(path, inputs, probabilities, result, cfg, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": ["Times New Roman", "SimHei"], "axes.unicode_minus": False,
                         "xtick.direction": "in", "ytick.direction": "in", "figure.dpi": 200})
    d = distance_grid(cfg)
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(d, inputs[0], linewidth=.6)
    axes[0].set_ylabel("脉冲 / RMS")
    axes[1].plot(d, inputs[1], linewidth=.6)
    axes[1].set_ylabel("累计阶跃 / RMS")
    for k, name in enumerate(["反射事件得分", "参数异常区间得分", "末端候选得分"]):
        axes[2].plot(d, probabilities[k], linewidth=.7, label=name)
    for ax in axes:
        for event in result["events"]:
            ax.axvline(event["distance_m"], color="tab:orange", alpha=.5, linewidth=.6)
        for interval in result["intervals"]:
            ax.axvspan(interval["start_m"], interval["end_m"], color="tab:red", alpha=.12)
        if result["terminal"] is not None:
            ax.axvline(result["terminal"]["distance_m"], color="tab:green", linestyle="--", linewidth=.8)
        ax.tick_params(top=True, right=True)
        ax.grid(alpha=.15)
    axes[2].legend(fontsize=8)
    axes[2].set_ylim(0, 1.03)
    axes[2].set_xlabel(f"距离 (m)，换算波速 {cfg['response']['velocity_m_s']/1e6:g} m/μs")
    fig.suptitle(title[:100])
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path, help="Three-column S11 CSV, NPZ, or FDR")
    group.add_argument("--manifest", type=Path, help="Read-only field source manifest")
    parser.add_argument("--sweep", type=int)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--subset", choices=["development", "unconfirmed"], default="development")
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plots", type=int, default=4)
    parser.add_argument("--limit", type=int, help="Limit only for small execution checks")
    args = parser.parse_args()
    device = torch.device(args.device)
    model, state = model_from_checkpoint(args.checkpoint, device)
    cfg = state["config"]
    threshold_path = args.thresholds or args.checkpoint.parent / "thresholds.json"
    calibration = json.loads(threshold_path.read_text(encoding="utf-8"))
    if calibration.get("calibration_split") != "val":
        raise ValueError("Inference requires thresholds selected on validation data")
    if calibration.get("checkpoint_epoch") != state["epoch"] or calibration.get("model_name") != state["model_name"]:
        raise ValueError("Thresholds belong to a different model/checkpoint epoch; calibrate this checkpoint on validation data")
    thresholds = calibration["thresholds"]
    if args.input:
        records = [{"input_path": str(args.input), "sweep_rowid": args.sweep, "label_status": "unknown"}]
        default_folder = f"infer_{args.checkpoint.stem}"
    else:
        records = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines()]
        status = "development_only" if args.subset == "development" else "independent_identity_unconfirmed"
        records = [row for row in records if row["development_status"] == status]
        default_folder = f"field_{args.subset}"
    if args.limit:
        records = records[:args.limit]
    output = args.output or args.checkpoint.parent / default_folder
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    successes = errors = plots = 0
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for number, record in enumerate(tqdm(records, desc="S11 -> field predictions")):
            path = Path(record["input_path"]) if "input_path" in record else args.source_root / record["source_relative_path"]
            try:
                f, s11 = read_s11(path, record.get("sweep_rowid"))
                result, inputs, probabilities = predict_one(model, f, s11, cfg, device, thresholds)
                entry = {"record": record, "status": "predicted", "prediction": result,
                         "distance_velocity_m_s": cfg["response"]["velocity_m_s"],
                         "source_in_band_frequency_max_gap_hz": float(np.max(np.diff(f[(f >= cfg["frequency"]["start_hz"]) & (f <= cfg["frequency"]["stop_hz"])]))),
                         "calibration": "raw measured S11; no division by fixture short measurement"}
                if plots < args.plots:
                    plot_prediction(output / f"sample_{number:04d}.png", inputs, probabilities, result, cfg, path.name)
                    plots += 1
                successes += 1
            except (ValueError, KeyError, IndexError, OSError) as exc:
                if args.input:
                    raise
                errors += 1
                entry = {"record": record, "status": "not_processed", "reason": f"{type(exc).__name__}: {exc}"}
                print(entry["reason"], flush=True)
            stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
    write_json(output / "summary.json", {"checkpoint": str(args.checkpoint.resolve()), "model": state["model_name"],
               "thresholds_path": str(threshold_path), "processed": successes, "not_processed": errors,
               "seconds": time.perf_counter() - started, "field_metrics": None,
               "evidence_boundary": "Field truth is unknown: model scores/plots are exploratory; no field precision, recall or location accuracy is claimed",
               "input_source": "S11", "input_true_length_used": False, "scope": "single-phase, open-end-dominant, <=2500m; unconfirmed identities are not a verified blind test set"})
    print(output, flush=True)


if __name__ == "__main__":
    main()
