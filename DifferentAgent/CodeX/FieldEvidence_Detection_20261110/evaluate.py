from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import FieldDataset, collate_groups
from model import PhaseDetector


ROOT = Path(__file__).resolve().parent
TASKS = ("joint", "terminal", "anomaly_point", "anomaly_range")
POINT_TASKS = ("joint", "terminal", "anomaly_point")
TASK_INDEX = {name: index for index, name in enumerate(TASKS)}


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def as_list(value) -> list:
    if value is None:
        return []
    return list(value)


def to_numpy(value) -> list[float]:
    return value.detach().float().cpu().tolist()


def contiguous_runs(mask, distance_m, scores=None) -> list[dict]:
    runs = []
    start = None
    last = len(mask) - 1
    for index, active in enumerate(mask):
        if active and start is None:
            start = index
        if start is not None and (not active or index == last):
            end = index - 1 if not active else index
            peak_index = (
                max(range(start, end + 1), key=lambda score_index: scores[score_index])
                if scores is not None
                else (start + end) // 2
            )
            runs.append(
                {
                    "start_m": float(distance_m[start]),
                    "end_m": float(distance_m[end]),
                    "center_m": float((distance_m[start] + distance_m[end]) / 2.0),
                    "position_m": float(distance_m[peak_index]),
                    "start_index": start,
                    "end_index": end,
                }
            )
            start = None
    return runs


def event_point_distance(prediction_m: float, event: dict) -> float:
    allowed = event.get("allowed_m")
    if isinstance(allowed, (list, tuple)) and len(allowed) == 2:
        low, high = sorted((float(allowed[0]), float(allowed[1])))
        if low <= prediction_m <= high:
            return 0.0
        return min(abs(prediction_m - low), abs(prediction_m - high))
    position = event.get("position_m")
    return abs(prediction_m - float(position)) if position is not None else float("inf")


def match_points(events: list[dict], predictions: list[dict], tolerance_m: float) -> tuple[set[int], set[int]]:
    pairs = []
    for event_index, event in enumerate(events):
        for prediction_index, prediction in enumerate(predictions):
            distance = event_point_distance(prediction["position_m"], event)
            if distance <= tolerance_m:
                pairs.append((distance, event_index, prediction_index))
    used_events: set[int] = set()
    used_predictions: set[int] = set()
    for _, event_index, prediction_index in sorted(pairs):
        if event_index not in used_events and prediction_index not in used_predictions:
            used_events.add(event_index)
            used_predictions.add(prediction_index)
    return used_events, used_predictions


def interval_overlap(prediction: dict, event: dict) -> tuple[float, float]:
    start = float(event["start_m"])
    end = float(event["end_m"])
    if end < start:
        start, end = end, start
    predicted_start = prediction["start_m"]
    predicted_end = prediction["end_m"]
    intersection = max(0.0, min(end, predicted_end) - max(start, predicted_start))
    union = max(end, predicted_end) - min(start, predicted_start)
    iou = intersection / union if union > 0 else 0.0
    coverage = intersection / (end - start) if end > start else 0.0
    return iou, coverage


def match_intervals(events: list[dict], predictions: list[dict]) -> tuple[dict[int, tuple[float, float]], set[int]]:
    pairs = []
    for event_index, event in enumerate(events):
        for prediction_index, prediction in enumerate(predictions):
            iou, coverage = interval_overlap(prediction, event)
            if iou > 0:
                pairs.append((iou, event_index, prediction_index, coverage))
    event_scores: dict[int, tuple[float, float]] = {}
    used_predictions: set[int] = set()
    for iou, event_index, prediction_index, coverage in sorted(pairs, reverse=True):
        if event_index not in event_scores and prediction_index not in used_predictions:
            event_scores[event_index] = (iou, coverage)
            used_predictions.add(prediction_index)
    for event_index in range(len(events)):
        event_scores.setdefault(event_index, (0.0, 0.0))
    return event_scores, used_predictions


def in_ranges(position_m: float, ranges) -> bool:
    for bounds in ranges or []:
        if len(bounds) != 2:
            continue
        low, high = sorted((float(bounds[0]), float(bounds[1])))
        if low <= position_m <= high:
            return True
    return False


def run_intersects_ranges(prediction: dict, ranges) -> bool:
    for bounds in ranges or []:
        if len(bounds) != 2:
            continue
        low, high = sorted((float(bounds[0]), float(bounds[1])))
        if prediction["end_m"] >= low and prediction["start_m"] <= high:
            return True
    return False


def point_interval_distance(position_m: float, event: dict) -> float:
    if "start_m" not in event or "end_m" not in event:
        return float("inf")
    low, high = sorted((float(event["start_m"]), float(event["end_m"])))
    if low <= position_m <= high:
        return 0.0
    return min(abs(position_m - low), abs(position_m - high))


class TaskMetrics:
    def __init__(self) -> None:
        self.positive_events = 0
        self.scoreable_positive_events = 0
        self.not_scoreable_events = 0
        self.matched_positive_events = 0
        self.known_background_false_positives = 0
        self.unknown_candidate_count = 0
        self.scoreable_intervals = 0
        self.interval_iou: list[float] = []
        self.interval_coverage: list[float] = []
        self.scoreable_start_boundaries = 0
        self.matched_start_boundaries = 0
        self.start_errors_m: list[float] = []
        self.complete_background = True
        self.background_checks = 0
        self.groups_seen = 0

    def result(self, task: str) -> dict:
        recall = (
            self.matched_positive_events / self.scoreable_positive_events
            if task in POINT_TASKS and self.scoreable_positive_events
            else None
        )
        precision = None
        f1 = None
        if (
            task in POINT_TASKS
            and self.complete_background
            and self.background_checks > 0
            and self.scoreable_positive_events > 0
            and self.not_scoreable_events == 0
            and self.unknown_candidate_count == 0
        ):
            denominator = self.matched_positive_events + self.known_background_false_positives
            precision = self.matched_positive_events / denominator if denominator else None
            if precision is not None and recall is not None and precision + recall:
                f1 = 2.0 * precision * recall / (precision + recall)

        has_scoreable_labels = (
            self.scoreable_positive_events > 0
            or self.scoreable_intervals > 0
            or self.scoreable_start_boundaries > 0
        )
        status = "partial" if has_scoreable_labels else "not_scored"
        result = {
            "status": status,
            "positive_events": self.positive_events,
            "scoreable_positive_events": self.scoreable_positive_events,
            "not_scoreable_events": self.not_scoreable_events,
            "label_coverage": (
                self.scoreable_positive_events / self.positive_events
                if self.positive_events
                else None
            ),
            "groups_seen": self.groups_seen,
        }
        if task in POINT_TASKS:
            if precision is not None and recall is not None:
                result["status"] = "scored"
            result.update(
                {
                    "known_positive_recall": recall,
                    "known_background_false_positive_count": self.known_background_false_positives,
                    "unknown_candidate_count": self.unknown_candidate_count,
                    "precision": precision,
                    "f1": f1,
                    "background_coverage_complete": bool(
                        self.complete_background and self.background_checks > 0
                    ),
                    "note": (
                        "Precision/F1 are omitted unless the task has explicit complete positive and negative coverage."
                        if precision is None
                        else "Thresholds are development starting values, not acceptance criteria."
                    ),
                }
            )
        else:
            result.update(
                {
                    "interval_count": self.scoreable_intervals,
                    "mean_interval_iou": (sum(self.interval_iou) / len(self.interval_iou)) if self.interval_iou else None,
                    "mean_known_interval_coverage": (sum(self.interval_coverage) / len(self.interval_coverage)) if self.interval_coverage else None,
                    "known_background_false_positive_count": self.known_background_false_positives,
                    "start_boundary_count": self.scoreable_start_boundaries,
                    "start_boundary_matched": self.matched_start_boundaries,
                    "start_boundary_recall": (
                        self.matched_start_boundaries / self.scoreable_start_boundaries
                        if self.scoreable_start_boundaries
                        else None
                    ),
                    "start_error_mae_m": (
                        sum(self.start_errors_m) / len(self.start_errors_m)
                        if self.start_errors_m
                        else None
                    ),
                    "note": "Range task reports interval IoU/coverage and start-boundary error; precision/F1 are not defined.",
                }
            )
        return result


def measurement_is_scoreable(
    measurement_id: str,
    event: dict,
    active_ids: set[str],
    distance_m,
    valid_by_id: dict[str, list[bool]],
) -> bool:
    required = event.get("required_measurement_ids") or [measurement_id]
    required_ids = [str(value) for value in required]
    if measurement_id not in active_ids or any(value not in active_ids for value in required_ids):
        return False
    if not required_ids:
        return False

    shape = event.get("shape")
    distances = distance_m
    if shape == "start_boundary":
        center = event.get("start_m", event.get("position_m"))
        if center is None:
            return False
        if float(center) - 2.0 < distances[0] or float(center) + 2.0 > distances[-1]:
            return False
        sample_indices = [
            index for index, value in enumerate(distances)
            if float(center) - 2.0 <= value <= float(center) + 2.0
        ]
    elif "position_m" in event:
        center = float(event["position_m"])
        allowed = event.get("allowed_m")
        low, high = (center, center)
        if isinstance(allowed, (list, tuple)) and len(allowed) == 2:
            low, high = sorted((float(allowed[0]), float(allowed[1])))
        sample_indices = [
            index for index, value in enumerate(distances)
            if low <= value <= high
        ]
        if not sample_indices:
            nearest = min(range(len(distances)), key=lambda index: abs(distances[index] - center))
            spacing = min(
                (distances[index + 1] - distances[index] for index in range(len(distances) - 1)),
                default=0.0,
            )
            if abs(distances[nearest] - center) > spacing / 2.0 + 1e-6:
                return False
            sample_indices = [nearest]
    elif "start_m" in event and "end_m" in event:
        low, high = sorted((float(event["start_m"]), float(event["end_m"])))
        if low < distances[0] or high > distances[-1]:
            return False
        sample_indices = [
            index for index, value in enumerate(distances)
            if low <= value <= high
        ]
    else:
        return False
    if not sample_indices:
        return False

    for required_id in required_ids:
        valid = valid_by_id.get(required_id)
        if valid is None or any(not valid[index] for index in sample_indices):
            return False
    return True


def make_views(batch: dict, p_count: int) -> list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]]:
    x = batch["x"].clone()
    valid = batch["valid"].clone()
    phase_mask = batch["phase_mask"].clone()
    views = [("full_context", x, valid, phase_mask)]
    for phase_index in range(p_count):
        if not bool(phase_mask[0, phase_index]):
            continue
        single_x = x.clone()
        single_valid = valid.clone()
        single_mask = torch.zeros_like(phase_mask)
        single_mask[0, phase_index] = True
        for other in range(p_count):
            if other != phase_index:
                single_x[0, other].zero_()
                single_valid[0, other].zero_()
        views.append((f"single_phase_slot_{phase_index}", single_x, single_valid, single_mask))
    return views


def evaluate_view(
    logits: torch.Tensor,
    valid: torch.Tensor,
    phase_mask: torch.Tensor,
    distance_m: list[float],
    meta: dict,
    thresholds: dict,
    tolerance_m: float,
    label_records: dict[str, dict],
    task_metrics: dict[str, TaskMetrics],
    phase_names: list[str | None],
    target_measurement_ids: set[str] | None = None,
) -> None:
    measurement_ids = list(meta.get("measurement_ids") or [])
    if len(measurement_ids) < logits.shape[1]:
        measurement_ids.extend([None] * (logits.shape[1] - len(measurement_ids)))
    measurement_ids = measurement_ids[: logits.shape[1]]
    active_ids = {
        str(measurement_ids[index])
        for index in range(logits.shape[1])
        if measurement_ids[index] and bool(phase_mask[0, index])
    }
    valid_by_id = {
        str(measurement_ids[index]): valid[0, index].detach().cpu().bool().tolist()
        for index in range(logits.shape[1])
        if measurement_ids[index]
    }
    phase_by_id = {
        str(measurement_ids[index]): index
        for index in range(logits.shape[1])
        if measurement_ids[index]
    }
    events = list(meta.get("events") or [])
    negative_by_id = meta.get("negative_ranges") or {}

    for task in TASKS:
        task_metrics[task].groups_seen += 1

    predictions: dict[str, dict[str, list[dict]]] = {}
    for measurement_id in active_ids:
        slot = phase_by_id[measurement_id]
        predictions[measurement_id] = {}
        valid_mask = valid[0, slot].detach().cpu().bool().tolist()
        for task in TASKS:
            probability = torch.sigmoid(logits[0, slot, TASK_INDEX[task]]).detach().float().cpu().tolist()
            binary = [
                bool(valid_mask[index]) and probability[index] >= float(thresholds.get(task, 0.5))
                for index in range(len(probability))
            ]
            predictions[measurement_id][task] = contiguous_runs(binary, distance_m, probability)

    by_measurement: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        owner = event.get("measurement_id")
        if owner:
            owner_id = str(owner)
            if target_measurement_ids is not None and owner_id not in target_measurement_ids:
                continue
            by_measurement[owner_id].append(event)

    for measurement_id, measurement_events in by_measurement.items():
        if measurement_id not in active_ids:
            for event in measurement_events:
                task = event.get("task")
                if task in task_metrics:
                    task_metrics[task].positive_events += 1
                    task_metrics[task].not_scoreable_events += 1
            continue

        for task in TASKS:
            relevant = [event for event in measurement_events if event.get("task") == task]
            scoreable = [
                event for event in relevant
                if measurement_is_scoreable(measurement_id, event, active_ids, distance_m, valid_by_id)
            ]
            not_scoreable = [event for event in relevant if event not in scoreable]
            metrics = task_metrics[task]
            metrics.positive_events += len(relevant)
            metrics.scoreable_positive_events += len(scoreable)
            metrics.not_scoreable_events += len(not_scoreable)

            pred_runs = predictions[measurement_id][task]
            negative_ranges = (negative_by_id.get(measurement_id) or {}).get(task) or []
            label_record = label_records.get(measurement_id, {})
            complete_map = (
                label_record.get("negative_ranges_complete")
                or label_record.get("coverage_complete_by_task")
                or label_record.get("coverage_complete")
                or {}
            )
            complete = bool(complete_map.get(task, False))
            metrics.complete_background = metrics.complete_background and complete
            metrics.background_checks += int(complete)

            if task in POINT_TASKS:
                matched_events, matched_predictions = match_points(scoreable, pred_runs, tolerance_m)
                metrics.matched_positive_events += len(matched_events)
                non_scoreable_events = not_scoreable
                for prediction_index, prediction in enumerate(pred_runs):
                    if prediction_index in matched_predictions:
                        continue
                    center = prediction["position_m"]
                    near_unscoreable = any(
                        event_point_distance(center, event) <= tolerance_m
                        or point_interval_distance(center, event) <= tolerance_m
                        for event in non_scoreable_events
                    )
                    if near_unscoreable:
                        continue
                    if in_ranges(center, negative_ranges):
                        metrics.known_background_false_positives += 1
                    else:
                        metrics.unknown_candidate_count += 1
                continue

            full_intervals = [
                event for event in scoreable
                if event.get("shape") == "interval"
                and event.get("start_m") is not None
                and event.get("end_m") is not None
            ]
            start_boundaries = [
                event for event in scoreable
                if event.get("shape") == "start_boundary"
            ]
            metrics.scoreable_intervals += len(full_intervals)
            metrics.known_background_false_positives += sum(
                run_intersects_ranges(prediction, negative_ranges)
                for prediction in pred_runs
            )
            interval_scores, used_interval_predictions = match_intervals(full_intervals, pred_runs)
            for event_index in range(len(full_intervals)):
                iou, coverage = interval_scores[event_index]
                metrics.interval_iou.append(iou)
                metrics.interval_coverage.append(coverage)

            metrics.scoreable_start_boundaries += len(start_boundaries)
            used_boundary_predictions: set[int] = set()
            for event in start_boundaries:
                boundary_m = event.get("start_m", event.get("position_m"))
                if boundary_m is None:
                    continue
                candidates = [
                    (abs(prediction["start_m"] - float(boundary_m)), index)
                    for index, prediction in enumerate(pred_runs)
                    if index not in used_interval_predictions
                    and index not in used_boundary_predictions
                    and abs(prediction["start_m"] - float(boundary_m)) <= tolerance_m
                ]
                if candidates:
                    error, prediction_index = min(candidates)
                    used_boundary_predictions.add(prediction_index)
                    metrics.matched_start_boundaries += 1
                    metrics.start_errors_m.append(error)

            used_predictions = used_interval_predictions | used_boundary_predictions
            for prediction_index, prediction in enumerate(pred_runs):
                if prediction_index in used_predictions:
                    continue
                if not run_intersects_ranges(prediction, negative_ranges) and not any(
                    point_interval_distance(prediction["center_m"], event) <= tolerance_m
                    for event in not_scoreable
                ):
                    metrics.unknown_candidate_count += 1

    for measurement_id in active_ids - set(by_measurement):
        negative_ranges_by_task = negative_by_id.get(measurement_id) or {}
        label_record = label_records.get(measurement_id, {})
        complete_map = (
            label_record.get("negative_ranges_complete")
            or label_record.get("coverage_complete_by_task")
            or label_record.get("coverage_complete")
            or {}
        )
        for task in TASKS:
            metrics = task_metrics[task]
            complete = bool(complete_map.get(task, False))
            metrics.complete_background = metrics.complete_background and complete
            metrics.background_checks += int(complete)
            for prediction in predictions[measurement_id][task]:
                negative_ranges = negative_ranges_by_task.get(task) or []
                is_background = (
                    run_intersects_ranges(prediction, negative_ranges)
                    if task == "anomaly_range"
                    else in_ranges(prediction["position_m"], negative_ranges)
                )
                if is_background:
                    metrics.known_background_false_positives += 1
                else:
                    metrics.unknown_candidate_count += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate field checkpoints; the user runs this command.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="output/field_cache")
    parser.add_argument("--labels", default=None)
    parser.add_argument("--split", default=None, help="Defaults to the split saved in the checkpoint.")
    parser.add_argument("--partition", choices=("val", "train"), default="val")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    checkpoint_path = project_path(args.checkpoint)
    data_dir = project_path(args.data_dir)
    if not checkpoint_path.is_file():
        parser.error(f"checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    split = load_json(project_path(args.split)) if args.split else checkpoint["split"]
    case_groups = list(split.get(args.partition, []))
    if not case_groups:
        parser.error(f"split partition {args.partition!r} is empty; fill the split file before evaluation.")
    if not data_dir.is_dir():
        parser.error(f"data directory not found: {data_dir}")
    labels_path = project_path(args.labels) if args.labels else data_dir / "review" / "accepted_labels.json"
    labels_root = load_json(labels_path)
    label_records = {
        item["measurement_id"]: item
        for item in labels_root.get("measurements", [])
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = config["model"]
    model = PhaseDetector(
        in_channels=int(model_cfg.get("in_channels", 5)),
        width=int(model_cfg.get("width", 16)),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    dataset = FieldDataset(data_dir, case_groups, training=False, single_phase_probability=0.0)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_groups)
    evaluation_cfg = config.get("evaluation", {})
    thresholds = evaluation_cfg.get("probability_thresholds", {})
    tolerance_m = float(evaluation_cfg.get("event_match_tolerance_m", 20.0))
    view_results: dict[str, dict[str, TaskMetrics]] = {}
    view_group_counts: dict[str, int] = defaultdict(int)

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"evaluate {args.partition}"):
            meta = batch["meta"][0]
            x = batch["x"].to(device)
            valid = batch["valid"].to(device)
            phase_mask = batch["phase_mask"].to(device)
            distance_m = to_numpy(batch["distance_m"][0])
            phase_names = list(meta.get("phase_names") or [])
            p_count = x.shape[1]

            for view_name, view_x, view_valid, view_mask in make_views(batch, p_count):
                logits = model(
                    view_x.to(device),
                    view_valid.to(device),
                    view_mask.to(device),
                )
                if view_name.startswith("single_phase_slot_"):
                    slot = int(view_name.rsplit("_", 1)[1])
                    name = phase_names[slot] if slot < len(phase_names) and phase_names[slot] else f"slot_{slot}"
                    result_key = f"single_phase:{name}"
                    measurement_ids = list(meta.get("measurement_ids") or [])
                    target_measurement_ids = (
                        {str(measurement_ids[slot])}
                        if slot < len(measurement_ids) and measurement_ids[slot]
                        else set()
                    )
                else:
                    result_key = view_name
                    target_measurement_ids = None
                if result_key not in view_results:
                    view_results[result_key] = {task: TaskMetrics() for task in TASKS}
                view_group_counts[result_key] += 1
                evaluate_view(
                    logits,
                    view_valid,
                    view_mask,
                    distance_m,
                    meta,
                    thresholds,
                    tolerance_m,
                    label_records,
                    view_results[result_key],
                    phase_names,
                    target_measurement_ids,
                )

    payload = {
        "checkpoint": str(checkpoint_path),
        "partition": args.partition,
        "case_groups": case_groups,
        "probability_thresholds": {task: float(thresholds.get(task, 0.5)) for task in TASKS},
        "event_match_tolerance_m": tolerance_m,
        "selection_note": "0.5 thresholds are development starting values; best.pt is selected by masked validation loss, not现场F1.",
        "views": {
            view: {
                "groups_seen": view_group_counts[view],
                "tasks": {task: metrics[task].result(task) for task in TASKS},
            }
            for view, metrics in view_results.items()
        },
    }
    output_path = project_path(args.output) if args.output else checkpoint_path.parent / "evaluation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"evaluation written to {output_path}")


if __name__ == "__main__":
    main()
