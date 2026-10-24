"""Run full-grid exploratory inference on measured FDR curves.

Field-manifest labels remain unknown. The optional 1500 m experiment contributes
one documented positive wet-joint location per sweep; every other candidate is
reported as unverified rather than as a false positive.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
CODEX_EXPERIMENT = HERE.parent / "FieldNN_20261022"
SOURCE_ROOT = Path(r"E:\20261010截至的全部fdr文件")
DEFAULT_MANIFEST = CODEX_EXPERIMENT / "output" / "real_manifest.jsonl"
KNOWN_LAB_RELATIVE = Path("2026-6某地进行的1500m不同位置电缆接头浸水试验") / "1500m电缆单个接头浸水试验.fdr"
KNOWN_LAB_EVIDENCE = KNOWN_LAB_RELATIVE.parent / "其他信息.txt"
KNOWN_LAB_SWEEPS = (
    (2, "100m接头浸水", 100),
    (3, "200m接头浸水", 200),
    (4, "300m接头浸水", 300),
    (5, "400m接头浸水", 400),
    (6, "500m接头浸水", 500),
    (7, "600m接头浸水", 600),
    (8, "700m接头浸水", 700),
    (9, "900m接头浸水", 900),
    (10, "1100m接头浸水", 1100),
    (11, "1300m接头浸水", 1300),
    (12, "1400m接头浸水", 1400),
)
KNOWN_LAB_TOLERANCES_M = (10, 20, 30)
CORE16_FIELD = PROJECT_ROOT / "DifferentAgent" / "CodeX" / "Core16_Field"
ITER4_EXPERIMENT = HERE.parent / "FieldNN_Iter4_20261022"

sys.path.insert(0, str(HERE))
from decode import decode_predictions
from features import preprocess
from models import build_model
sys.path.insert(0, str(ITER4_EXPERIMENT))
from length_view import analyze_length_view


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _legacy_modules() -> tuple[Any, Any, Any]:
    common = _load_module(CODEX_EXPERIMENT / "common.py", "field_eval_legacy_common")
    models = _load_module(CODEX_EXPERIMENT / "models.py", "field_eval_legacy_models")
    metrics = _load_module(CODEX_EXPERIMENT / "metrics.py", "field_eval_legacy_metrics")
    return common, models, metrics


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_source_s11(path: Path, sweep_rowid: int | str | None,
                     fallback_reader: Any) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Read a sweep and its report-axis velocity from the same in-memory FDR DB."""
    if path.suffix.lower() not in {".fdr", ".db8"}:
        frequency, s11 = fallback_reader(path, sweep_rowid)
        return frequency, s11, None
    if sweep_rowid is None:
        raise ValueError("FDR input requires a manifest sweep_rowid")

    if str(CORE16_FIELD) not in sys.path:
        sys.path.insert(0, str(CORE16_FIELD))
    from analyze_corpus import load_reader_module

    rowid = int(sweep_rowid)
    plaintext, _ = load_reader_module().FdrCrypto().decrypt(path.read_bytes())
    with sqlite3.connect(":memory:") as connection:
        connection.deserialize(plaintext)
        tables = {str(row[0]) for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        if "Sweeps" not in tables:
            raise ValueError("FDR database has no Sweeps table")
        cursor = connection.execute('SELECT rowid, * FROM "Sweeps" WHERE rowid = ?', (rowid,))
        columns = [item[0] for item in cursor.description]
        row = cursor.fetchone()
        if row is None:
            raise ValueError(f"FDR sweep rowid {rowid} was not found")
        sweep = dict(zip(columns, row))
        try:
            source_velocity = float(sweep.get("Wave_Velocity"))
        except (TypeError, ValueError):
            source_velocity = None
        if source_velocity is not None and (not np.isfinite(source_velocity) or source_velocity <= 0):
            source_velocity = None

        table_name = f"MeasureData_{rowid}"
        if table_name not in tables:
            raise ValueError(f"FDR database has no {table_name} table")
        rows = connection.execute(
            f'SELECT Freq, Real, Image FROM "{table_name}" ORDER BY Freq'
        ).fetchall()
        data = np.asarray(rows, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 3 or len(data) == 0:
        raise ValueError(f"FDR sweep {rowid} has no usable S11 rows")
    return data[:, 0], data[:, 1] + 1j * data[:, 2], source_velocity


def _axis_metadata(source_velocity_m_per_us: float | None, nominal_velocity_m_per_us: float,
                   requested_axis: str) -> dict[str, Any]:
    source_ratio = (source_velocity_m_per_us / nominal_velocity_m_per_us
                    if source_velocity_m_per_us is not None else None)
    if requested_axis == "source" and source_ratio is not None:
        selected_ratio = source_ratio
        selected_velocity = source_velocity_m_per_us
        selected_axis = "source_report"
    elif requested_axis == "source":
        selected_ratio = 1.0
        selected_velocity = nominal_velocity_m_per_us
        selected_axis = "nominal_fallback"
    else:
        selected_ratio = 1.0
        selected_velocity = nominal_velocity_m_per_us
        selected_axis = "nominal"
    return {
        "requested_distance_axis": requested_axis,
        "selected_distance_axis": selected_axis,
        "source_wave_velocity_m_per_us": source_velocity_m_per_us,
        "nominal_velocity_m_per_us": nominal_velocity_m_per_us,
        "source_to_nominal_scale_ratio": source_ratio,
        "selected_scale_ratio": selected_ratio,
        "selected_velocity_m_per_us": selected_velocity,
        "status": "source_sweep_velocity" if source_velocity_m_per_us is not None else "nominal_fallback",
        "coordinate_note": "Wave_Velocity is the FDR report-axis setting; it is not an independent propagation-speed measurement.",
    }


def _map_prediction_distance(prediction: dict[str, Any], scale: float) -> dict[str, Any]:
    mapped = copy.deepcopy(prediction)
    for event in mapped.get("events", []):
        event["distance_m"] = float(event["distance_m"]) * scale
    for interval in mapped.get("intervals", []):
        interval["start_m"] = float(interval["start_m"]) * scale
        interval["end_m"] = float(interval["end_m"]) * scale
    if mapped.get("terminal") is not None:
        mapped["terminal"]["distance_m"] = float(mapped["terminal"]["distance_m"]) * scale
    return mapped


def _known_lab_records() -> list[dict[str, Any]]:
    records = []
    for rowid, sweep_name, position_m in KNOWN_LAB_SWEEPS:
        records.append({
            "record_id": f"wet_1500_sweep_{rowid}",
            "source_relative_path": KNOWN_LAB_RELATIVE.as_posix(),
            "sweep_rowid": rowid,
            "sweep_name": sweep_name,
            "phase": "unknown",
            "source_case_group": "2026-6某地进行的1500m不同位置电缆接头浸水试验",
            "cable_group": "development_experiment:1500m_wet_joint",
            "development_status": "development_only",
            "label_status": "partial_known_positive_event",
            "known_event_position_m": float(position_m),
            "known_event_kind": "wet_joint",
            "evidence_source_relative_path": KNOWN_LAB_EVIDENCE.as_posix(),
            "evidence_summary": "TXT states that the wet-joint locations are marked in sweep names (100 m to 1400 m). Only this positive location is known; other events and intervals are unlabelled.",
            "label_scope": "known positive event only; no complete event, interval, negative, or terminal truth",
        })
    return records


def _deduplicate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for index, record in enumerate(records):
        source = str(record.get("source_relative_path") or record.get("input_path") or "").replace("\\", "/").casefold()
        rowid = record.get("sweep_rowid")
        key = (source, str(rowid)) if source and rowid is not None else ("row", str(index))
        current = selected.get(key)
        if current is None or (record.get("label_status") == "partial_known_positive_event"
                               and current.get("label_status") != "partial_known_positive_event"):
            selected[key] = record
    return list(selected.values())


def _load_thresholds(checkpoint_path: Path, checkpoint: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    threshold_path = checkpoint_path.parent / "thresholds.json"
    if not threshold_path.is_file():
        raise FileNotFoundError(f"Validation-calibrated thresholds were not found: {threshold_path}")
    calibration = json.loads(threshold_path.read_text(encoding="utf-8"))
    if calibration.get("calibration_split") != "val":
        raise ValueError("Field inference requires thresholds calibrated on the validation split")
    if calibration.get("model_name") != checkpoint.get("model_name"):
        raise ValueError("Threshold model_name does not match checkpoint")
    if int(calibration.get("checkpoint_epoch", -1)) != int(checkpoint.get("epoch", -2)):
        raise ValueError("Threshold checkpoint_epoch does not match checkpoint")
    thresholds = calibration.get("thresholds")
    required = {"event", "interval", "terminal"}
    if not isinstance(thresholds, dict) or not required.issubset(thresholds):
        raise ValueError("Threshold file must contain event, interval, and terminal values")
    return {name: float(thresholds[name]) for name in required}, {
        "path": str(threshold_path.resolve()),
        "calibration_split": calibration["calibration_split"],
        "checkpoint_epoch": int(calibration["checkpoint_epoch"]),
        "model_name": calibration["model_name"],
    }


def _load_predictor(checkpoint_path: Path, device: torch.device, legacy: bool):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {"config", "model_name", "model", "epoch"}
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"Checkpoint is missing required fields: {missing}")
    cfg = checkpoint["config"]
    if legacy:
        common, old_models, old_metrics = _legacy_modules()
        model = old_models.build_model(checkpoint["model_name"], **cfg["model"])
        preprocess_fn = common.preprocess_s11
        read_s11_fn = common.read_s11
        decode_fn = old_metrics.decode_predictions
        input_channels = 4
    else:
        model = build_model(checkpoint["model_name"], **cfg["model"])
        preprocess_fn = preprocess
        source_common = _load_module(CODEX_EXPERIMENT / "common.py", "field_eval_source_common")
        read_s11_fn = source_common.read_s11
        decode_fn = decode_predictions
        input_channels = int(cfg["model"]["input_channels"])
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    thresholds, calibration = _load_thresholds(checkpoint_path, checkpoint)
    return model, checkpoint, cfg, preprocess_fn, read_s11_fn, decode_fn, input_channels, thresholds, calibration


def _resolve_source(record: dict[str, Any], source_root: Path) -> Path:
    raw = record.get("input_path") or record.get("source_relative_path")
    if not raw:
        raise ValueError("Manifest record has neither input_path nor source_relative_path")
    path = Path(raw)
    return path if path.is_absolute() else source_root / path


def _terminal_consistency(prediction: dict[str, Any], inputs: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
    candidate = prediction.get("terminal")
    if candidate is None:
        return {"status": "no_candidate"}
    response = cfg["response"]
    distances = np.linspace(0.0, float(response["distance_max_m"]), int(response["points"]))
    position = float(candidate["distance_m"])
    impulse, step = inputs[0], inputs[1]
    peak_mask = np.abs(distances - position) <= 5.0
    background_mask = (np.abs(distances - position) >= 15.0) & (np.abs(distances - position) <= 40.0)
    left_mask = (distances >= position - 15.0) & (distances <= position - 5.0)
    right_mask = (distances >= position + 5.0) & (distances <= position + 15.0)
    change = float(np.median(step[right_mask]) - np.median(step[left_mask])) if left_mask.any() and right_mask.any() else None
    background_rms = float(np.sqrt(np.mean(impulse[background_mask]**2))) if background_mask.any() else 0.0
    pulse_ratio = float(np.max(np.abs(impulse[peak_mask])) / max(background_rms, 1e-12)) if peak_mask.any() else 0.0
    later_events = [float(item["distance_m"]) for item in prediction.get("events", [])
                    if float(item.get("confidence", 0.0)) >= 0.8]
    checks = {
        "outside_near_end": bool(position > 10.0),
        "away_from_right_boundary": bool(position < distances[-1] - 15.0),
        "after_high_score_events": bool(not later_events or position >= max(later_events) - 10.0),
        "local_impulse_present": bool(pulse_ratio >= 2.0),
        "step_rises_locally": bool(change is not None and change > 0.0),
    }
    return {
        "status": "consistent_candidate" if all(checks.values()) else "uncertain_candidate",
        "checks": checks,
        "local_peak_to_background_rms": pulse_ratio,
        "normalized_step_change": change,
        "note": "Heuristic only; does not remove candidates and is not an independent length measurement.",
    }


def _known_event_result(events: list[dict[str, Any]], truth_m: float,
                        distance_axis: str) -> dict[str, Any]:
    closest = min(events, key=lambda item: abs(float(item["distance_m"]) - truth_m), default=None)
    by_tolerance = {}
    for tolerance in KNOWN_LAB_TOLERANCES_M:
        hit = closest is not None and abs(float(closest["distance_m"]) - truth_m) <= tolerance
        by_tolerance[str(tolerance)] = {
            "tolerance_m": tolerance,
            "hit": bool(hit),
            "closest_candidate_distance_m": None if closest is None else float(closest["distance_m"]),
            "closest_candidate_confidence": None if closest is None else float(closest["confidence"]),
            "absolute_error_m": None if closest is None else float(abs(float(closest["distance_m"]) - truth_m)),
            "additional_unverified_candidate_count": max(0, len(events) - int(hit)),
        }
    return {
        "known_event_position_m": float(truth_m),
        "distance_axis": distance_axis,
        "by_tolerance": by_tolerance,
    }


def _plot_prediction(path: Path, inputs: np.ndarray, probabilities: np.ndarray,
                     prediction: dict[str, Any], cfg: dict[str, Any], title: str,
                     known_event_position_m: float | None = None,
                     distance_axis_label: str = "nominal model axis",
                     length_plot_responses: dict[str, Any] | None = None,
                     plot_window_m: float | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": ["Times New Roman", "SimHei"],
        "axes.unicode_minus": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.dpi": 200,
    })
    response = cfg["response"]
    distances = np.linspace(0.0, float(response["distance_max_m"]), int(response["points"]))
    fig, axes = plt.subplots(5, 1, figsize=(11, 9), sharex=True)
    if length_plot_responses is None:
        for axis, values, ylabel in zip(axes[:2], inputs[:2], ("Impulse feature", "Step feature")):
            axis.plot(distances, values, linewidth=0.65)
            axis.set_ylabel(ylabel)
    else:
        for track, axis, ylabel in (("impulse_real", axes[0], "Impulse response"),
                                    ("step_response", axes[1], "Step response")):
            full = length_plot_responses["full_grid"]
            axis.plot(full["distance_m"], full[track], color="0.45", alpha=0.65,
                      linewidth=0.65, label="Algorithm 1 full grid")
            recalculated = length_plot_responses.get("recomputed")
            if recalculated is not None:
                axis.plot(recalculated["distance_m"], recalculated[track], color="tab:blue",
                          linewidth=0.7, label="Algorithm 1 recomputed")
            axis.set_ylabel(ylabel)
            axis.legend(fontsize=7, loc="best")
        estimate = length_plot_responses.get("candidate_m")
        recomputed_candidate = length_plot_responses.get("recomputed_candidate_m")
        if estimate is not None:
            for axis in axes[:2]:
                axis.axvline(estimate, color="tab:orange", linestyle="--", linewidth=0.7,
                             label="full-grid endpoint candidate")
        if recomputed_candidate is not None:
            for axis in axes[:2]:
                axis.axvline(recomputed_candidate, color="tab:green", linestyle=":", linewidth=0.8,
                             label="recomputed endpoint candidate")
        for axis in axes[:2]:
            if estimate is not None or recomputed_candidate is not None:
                axis.legend(fontsize=7, loc="best")
    for axis, values, ylabel in zip(axes[2:], probabilities, ("Event score", "Interval score", "Terminal score")):
        axis.plot(distances, values, linewidth=0.7)
        axis.set_ylabel(ylabel)
        axis.set_ylim(0.0, 1.03)
    for event in prediction.get("events", []):
        axes[2].axvline(event["distance_m"], color="tab:orange", alpha=0.55, linewidth=0.6)
    for interval in prediction.get("intervals", []):
        axes[3].axvspan(interval["start_m"], interval["end_m"], color="tab:red", alpha=0.13)
    if prediction.get("terminal") is not None:
        axes[4].axvline(prediction["terminal"]["distance_m"], color="tab:green", linestyle="--", linewidth=0.8)
    if known_event_position_m is not None:
        axes[2].axvline(known_event_position_m, color="tab:blue", linestyle=":", linewidth=0.9,
                        label="Known positive event")
        axes[2].legend(fontsize=8)
    for axis in axes:
        axis.tick_params(top=True, right=True)
        axis.grid(alpha=0.15)
    axes[-1].set_xlabel(
        f"{distance_axis_label} (m), velocity={response['velocity_m_s'] / 1e6:g} m/μs"
    )
    if plot_window_m is not None:
        axes[-1].set_xlim(0.0, float(plot_window_m))
    fig.suptitle(title[:120])
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _field_count_summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        if entry.get("status") == "predicted" and entry.get("record", {}).get("label_status") == "unknown":
            record = entry["record"]
            groups[record.get("cable_group", "unknown_group")].append(entry)
    result = {}
    for group, values in sorted(groups.items()):
        event_counts = [len(value["prediction"]["events"]) for value in values]
        interval_counts = [len(value["prediction"]["intervals"]) for value in values]
        result[group] = {
            "record_count": len(values),
            "source_case_groups": sorted({value["record"].get("source_case_group", "") for value in values}),
            "events_per_curve_median": float(np.median(event_counts)),
            "events_per_curve_range": [min(event_counts), max(event_counts)],
            "intervals_per_curve_median": float(np.median(interval_counts)),
            "intervals_per_curve_range": [min(interval_counts), max(interval_counts)],
            "terminal_candidates": sum(value["prediction"]["terminal"] is not None for value in values),
            "terminal_consistency_counts": dict(Counter(value["terminal_consistency"]["status"] for value in values)),
        }
    return result


def _known_lab_summary(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    rows = [entry for entry in entries if entry.get("record", {}).get("label_status") == "partial_known_positive_event"
            and entry.get("status") == "predicted"]
    if not rows:
        return None
    metrics = {}
    for tolerance in KNOWN_LAB_TOLERANCES_M:
        hits = sum(bool(row["known_event_evaluation"]["by_tolerance"][str(tolerance)]["hit"]) for row in rows)
        additional = sum(int(row["known_event_evaluation"]["by_tolerance"][str(tolerance)]["additional_unverified_candidate_count"])
                         for row in rows)
        metrics[str(tolerance)] = {
            "tolerance_m": tolerance,
            "known_positive_sweeps_evaluated": len(rows),
            "known_event_hits": hits,
            "known_event_hit_rate": hits / len(rows),
            "additional_unverified_candidate_count": additional,
        }
    return {
        "group": "one 1500 m wet-joint experiment; not an independent-cable test set",
        "metric_scope": "known-positive event hit only; no precision/recall, full truth, or negative labels",
        "distance_axis": "source/report axis when Wave_Velocity is available; nominal_fallback otherwise",
        "normal_sweep_used": False,
        "sweeps_evaluated": len(rows),
        "by_tolerance": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, help="Process only the first N selected rows")
    parser.add_argument("--subset", choices=("development", "all"), default="development")
    parser.add_argument("--distance-axis", choices=("source", "nominal"), default="source")
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--known-lab", action="store_true", help="Include the 11 documented 1500 m wet-joint positive sweeps")
    length_group = parser.add_mutually_exclusive_group()
    length_group.add_argument("--length-view", dest="length_view", action="store_true",
                              help="Estimate endpoint and recompute Algorithm 1 to about 1.2L (default)")
    length_group.add_argument("--no-length-view", dest="length_view", action="store_false",
                              help="Disable the independent endpoint estimate and response recomputation")
    parser.add_argument("--legacy", action="store_true", help="Load an older four-channel CodeX checkpoint and its matching preprocessing")
    parser.add_argument("--plots", type=int, default=4)
    parser.set_defaults(length_view=True)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.plots < 0:
        parser.error("--plots must be nonnegative")
    if not args.manifest.is_file():
        parser.error(f"Manifest not found: {args.manifest}")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")

    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu")
    model, checkpoint, cfg, preprocess_fn, read_s11_fn, decode_fn, input_channels, thresholds, calibration = _load_predictor(
        args.checkpoint.resolve(), device, args.legacy
    )

    records = _read_jsonl(args.manifest)
    if args.subset == "development":
        records = [row for row in records if row.get("development_status") == "development_only"]
    if args.known_lab:
        records.extend(_known_lab_records())
        records = _deduplicate_records(records)
    if args.limit is not None:
        records = records[:args.limit]

    output = args.output or HERE / "output" / f"field_eval_{args.checkpoint.stem}_{args.distance_axis}"
    output = output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty; choose a new path to preserve existing results: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    successes = errors = plots = 0
    plotted_groups = set()
    entries: list[dict[str, Any]] = []
    predictions_path = output / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8", newline="\n") as stream:
        for number, record in enumerate(tqdm(records, desc="S11 -> full-grid field inference")):
            source = _resolve_source(record, args.source_root)
            try:
                frequency, s11, source_velocity_m_per_us = _read_source_s11(
                    source, record.get("sweep_rowid"), read_s11_fn
                )
                inputs = preprocess_fn(frequency, s11, cfg)
            except (ValueError, KeyError, IndexError, OSError, sqlite3.Error) as exc:
                entry = {"record": record, "status": "not_processed", "reason": f"{type(exc).__name__}: {exc}"}
                errors += 1
                entries.append(entry)
                stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                continue

            if inputs.ndim != 2 or inputs.shape[0] != input_channels:
                raise ValueError(f"Preprocessor returned {inputs.shape}; checkpoint expects {input_channels} channels")
            if inputs.shape[1] != int(cfg["response"]["points"]):
                raise ValueError(f"Preprocessor returned {inputs.shape[1]} points; config requires {cfg['response']['points']}")
            with torch.inference_mode():
                logits = model(torch.from_numpy(inputs[None]).to(device))
                probabilities = logits.sigmoid()[0].float().cpu().numpy()
            prediction_nominal = decode_fn(probabilities, cfg, thresholds)
            terminal_consistency = _terminal_consistency(prediction_nominal, inputs, cfg)
            nominal_velocity_m_per_us = float(cfg["response"]["velocity_m_s"]) / 1e6
            axis_metadata = _axis_metadata(
                source_velocity_m_per_us, nominal_velocity_m_per_us, args.distance_axis
            )
            source_scale = axis_metadata["source_to_nominal_scale_ratio"] or 1.0
            prediction_source = _map_prediction_distance(prediction_nominal, float(source_scale))
            prediction = prediction_source if args.distance_axis == "source" else prediction_nominal
            length_result = None
            length_public = {}
            if args.length_view:
                length_velocity_m_per_us = (
                    source_velocity_m_per_us
                    if source_velocity_m_per_us is not None else nominal_velocity_m_per_us
                )
                length_scale = float(length_velocity_m_per_us) / nominal_velocity_m_per_us
                length_result = analyze_length_view(
                    frequency,
                    s11,
                    cfg,
                    float(length_velocity_m_per_us) * 1e6,
                    float(cfg["response"]["distance_max_m"]) * length_scale,
                )
                length_public = {key: value for key, value in length_result.items() if not key.startswith("_")}
                length_axis_basis = (
                    "source_report" if source_velocity_m_per_us is not None else "nominal_fallback"
                )
                length_public["length_estimate"]["coordinate_axis"] = length_axis_basis
                length_public["response_recomputed"]["velocity_basis"] = length_axis_basis
            known_event = record.get("known_event_position_m")
            known_axis = "source_report" if source_velocity_m_per_us is not None else "nominal_fallback"
            known_evaluation = (
                _known_event_result(prediction_source.get("events", []), float(known_event), known_axis)
                if known_event is not None else None
            )
            band = cfg["frequency"]
            in_band = frequency[(frequency >= float(band["start_hz"])) & (frequency <= float(band["stop_hz"]))]
            entry = {
                "record": record,
                "status": "predicted",
                "prediction": prediction,
                "prediction_nominal": prediction_nominal,
                "source_axis_metadata": axis_metadata,
                "terminal_consistency": terminal_consistency,
                "terminal_consistency_axis": "nominal",
                "known_event_evaluation": known_evaluation,
                "length_view_enabled": bool(args.length_view),
                **length_public,
                "checkpoint": {"model_name": checkpoint["model_name"], "epoch": int(checkpoint["epoch"]),
                               "legacy_four_channel": bool(args.legacy)},
                "calibration": calibration,
                "source_frequency": {
                    "point_count": int(len(frequency)),
                    "in_band_point_count": int(len(in_band)),
                    "in_band_min_hz": float(in_band[0]) if len(in_band) else None,
                    "in_band_max_hz": float(in_band[-1]) if len(in_band) else None,
                    "in_band_max_gap_hz": float(np.max(np.diff(in_band))) if len(in_band) > 1 else None,
                },
                "distance_axis_policy": "prediction uses the selected distance axis; known-positive checks use the FDR source/report axis when Wave_Velocity exists; no true-length crop, length-based near-end ratio, or top-k event limit",
            }
            plot_group = record.get("cable_group") or record.get("source_case_group") or str(source)
            if plots < args.plots and plot_group not in plotted_groups:
                plot_path = output / f"sample_{number:04d}.png"
                # Length-view plots use the source/report axis; disabling it keeps
                # the selected distance-axis behavior used by the original plot.
                plot_axis = "source" if args.length_view else args.distance_axis
                plot_scale = float(source_scale if plot_axis == "source" else 1.0)
                plot_cfg = copy.deepcopy(cfg)
                plot_cfg["response"]["distance_max_m"] *= plot_scale
                plot_velocity = (float(length_velocity_m_per_us) if args.length_view else
                                 float(axis_metadata["selected_velocity_m_per_us"]))
                plot_cfg["response"]["velocity_m_s"] = plot_velocity * 1e6
                if known_event is None:
                    plot_known_event = None
                elif plot_axis == "source":
                    plot_known_event = float(known_event)
                else:
                    plot_known_event = float(known_event) / float(source_scale)
                if plot_axis == "source":
                    plot_axis_label = (
                        "source_report" if source_velocity_m_per_us is not None else "nominal_fallback"
                    ) if args.length_view else axis_metadata["selected_distance_axis"]
                else:
                    plot_axis_label = "nominal model axis"
                plot_prediction = prediction_source if plot_axis == "source" else prediction_nominal
                length_plot_responses = None
                plot_window_m = None
                if length_result is not None:
                    length_plot_responses = {
                        **length_result["_plot_responses"],
                        "candidate_m": length_result["length_estimate"]["full_grid_candidate_m"],
                        "recomputed_candidate_m": length_result["length_estimate"]["recomputed_candidate_m"],
                    }
                    plot_window_m = float(length_result["plot_window_m"])
                _plot_prediction(plot_path, inputs, probabilities, plot_prediction, plot_cfg,
                                 f"{record.get('sweep_name', source.name)} | {record.get('source_case_group', '')}",
                                 plot_known_event, plot_axis_label, length_plot_responses, plot_window_m)
                entry["plot"] = plot_path.name
                plots += 1
                plotted_groups.add(plot_group)
            successes += 1
            entries.append(entry)
            stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()

    label_counts = Counter(str(row.get("label_status", "missing")) for row in records)
    status_counts = Counter(entry["status"] for entry in entries)
    selected_ratios = [float(entry["source_axis_metadata"]["selected_scale_ratio"])
                       for entry in entries if entry.get("status") == "predicted"]
    selected_grid_endpoints = [float(cfg["response"]["distance_max_m"]) * ratio for ratio in selected_ratios]
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "model_name": checkpoint["model_name"],
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "legacy_four_channel": bool(args.legacy),
        "manifest": str(args.manifest.resolve()),
        "subset": args.subset,
        "distance_axis": args.distance_axis,
        "known_lab_metric_axis": "source/report axis when Sweep.Wave_Velocity exists; nominal_fallback otherwise",
        "all_subset_note": "all rows are read-only coverage and are not an independent test set" if args.subset == "all" else None,
        "known_lab_included": bool(args.known_lab),
        "length_view_enabled": bool(args.length_view),
        "selected_rows": len(records),
        "processed": successes,
        "not_processed": errors,
        "record_status_counts": dict(status_counts),
        "label_status_counts": dict(label_counts),
        "thresholds": thresholds,
        "threshold_calibration": calibration,
        "seconds": time.perf_counter() - started,
        "plot_count": plots,
        "full_grid_m": [0.0, float(cfg["response"]["distance_max_m"])],
        "nominal_grid_m": [0.0, float(cfg["response"]["distance_max_m"])],
        "selected_grid_endpoint_range_m": (
            [min(selected_grid_endpoints), max(selected_grid_endpoints)]
            if selected_grid_endpoints else [0.0, float(cfg["response"]["distance_max_m"])]
        ),
        "selected_scale_ratio_range": [min(selected_ratios), max(selected_ratios)] if selected_ratios else [1.0, 1.0],
        "configured_points": int(cfg["response"]["points"]),
        "field_metrics": None,
        "unlabeled_detection_counts_by_cable_group": _field_count_summary(entries),
        "known_lab_positive_event_summary": _known_lab_summary(entries),
        "evidence_policy": "Manifest field labels remain unknown. The optional wet-joint group reports only known-positive hits at 10/20/30 m and counts other candidates as unverified; it has no negative or complete truth labels.",
        "length_policy": "No physical length is used to crop inputs or outputs, set a near-end ratio, or suppress predictions.",
        "prediction_file": str(predictions_path.resolve()),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "processed": successes, "not_processed": errors,
                      "known_lab_positive_event_summary": summary["known_lab_positive_event_summary"]},
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

