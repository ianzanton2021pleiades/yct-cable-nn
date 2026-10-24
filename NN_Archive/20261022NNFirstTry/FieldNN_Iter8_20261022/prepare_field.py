"""Iter6's frozen evidence/splits, recomputed with Iter8's 12-channel features."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm
import features  # Keep this experiment's representation when the legacy reader imports it.


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
ITER3_DIR = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_Iter3_20261022"
ITER5_DIR = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_Iter5_20261022"
MANIFEST_PATH = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_20261022" / "output" / "real_manifest.jsonl"
EVIDENCE_PATH = ITER5_DIR / "field_evidence.json"
CONFIG_PATH = ITER5_DIR / "output" / "iter5" / "unet" / "config.json"
CALIBRATION_SOURCES_PATH = Path(
    r"E:\CableNNDataSet\CodeX\FieldNN_Iter4_20261022\calibration_sources_v2.jsonl"
)
SOURCE_ROOT = Path(r"E:\20261010截至的全部fdr文件")
SEED = 2026102206
MAX_SOURCE_POSITION_M = 2500.0
MIN_NOMINAL_POSITION_M = 30.0
MAX_NOMINAL_POSITION_M = 3240.0
DEFAULT_POSITION_RADIUS_M = 20.0
WET_POSITION_RADIUS_M = 10.0
TDR_POSITION_RADIUS_M = 40.0
WET_CASE = "2026-6某地进行的1500m不同位置电缆接头浸水试验"

# Confirmed repeats and same-site/circuit aliases are kept in one split.
CASE_ALIASES = [
    [
        "2026-7-15 16.19.30深圳220kV马坳站10KV住宅局线F52嘉欣园断路器柜",
        "2026-7-20 10.46.01深圳某地两段电缆",
        "2026-8-22 15.06.16深圳220kV马坳站10kV住宅局线F52嘉欣园断路器柜",
    ],
    [
        "2026-5-25 10.48.04深圳110kV平湖站10kv辅城坳线F17平湖山厦外国语学校智能柜",
        "2026-5-28 20.18.53深圳110k平湖站10kV辅城坳线F17平湖山厦外国语学校智能柜",
    ],
    [
        "2026-5-30 12.13.22深圳110kV坂田变电站10kV四季线F65坂田南学校#1公用智能柜",
        "2026-5-31 10.42.00深圳110kV坂田变电站10KV四季线F65坂田南学校#1公用智能柜",
    ],
    [
        "2026-6-2 11.31.03深圳大芬变电站10kV浦良线F02丹竹头岭背#3智能柜",
        "2026-6-23 10.41.51深圳大芬变电站10kV浦良线F02丹竹头岭背#3智能柜",
    ],
]

# Coordinates in these cases have an explicit source conflict; keep their FDR
# curves out of weak-label bags until the coordinate mapping is resolved.
CONFLICT_CASES = {
    "2026-6-23 10.41.51深圳大芬变电站10kV浦良线F02丹竹头岭背#3智能柜":
        "engineer positions disagree systematically with the FDR-measured connector positions",
    "2026-6-24 11.51.24深圳龙岗某地（之前坪地介损击穿了）":
        "A-phase 735 m claim conflicts with the 437/743 m FDR connector annotation",
}


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, value: str) -> None:
        self.parent.setdefault(value, value)

    def find(self, value: str) -> str:
        self.add(value)
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            low, high = sorted((a, b))
            self.parent[high] = low


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def norm_path(value: str) -> str:
    return str(value).replace("\\", "/").strip("/").casefold()


def add_group_aliases(uf: UnionFind, case_nodes: dict[str, str], rules_path: Path) -> None:
    rules = json.loads(rules_path.read_text(encoding="utf-8"))
    for section in ("development_groups", "linked_cable_groups"):
        for group in rules.get(section, []):
            cases = [case_nodes[name] for name in group.get("case_groups", []) if name in case_nodes]
            for case in cases[1:]:
                uf.union(cases[0], case)
    for aliases in CASE_ALIASES:
        present = [case_nodes[name] for name in aliases if name in case_nodes]
        for case in present[1:]:
            uf.union(present[0], case)


def build_groups(rows: list[dict[str, Any]], rules_path: Path) -> tuple[dict[str, str], dict[str, list[int]]]:
    uf = UnionFind()
    case_nodes: dict[str, str] = {}
    row_tokens: list[tuple[str, str, str]] = []
    for index, row in enumerate(rows):
        case = str(row.get("source_case_group") or row.get("case_group") or "")
        cable = str(row.get("cable_group") or f"unconfirmed_case:{case}")
        source = norm_path(str(row["source_relative_path"]))
        case_node, cable_node, source_node = f"case:{case}", f"cable:{cable}", f"source:{source}"
        uf.union(case_node, cable_node)
        uf.union(case_node, source_node)
        case_nodes[case] = case_node
        row_tokens.append((case_node, cable_node, source_node))

    add_group_aliases(uf, case_nodes, rules_path)

    components: dict[str, list[int]] = defaultdict(list)
    component_cables: dict[str, set[str]] = defaultdict(set)
    for index, (case_node, cable_node, _) in enumerate(row_tokens):
        root = uf.find(case_node)
        components[root].append(index)
        component_cables[root].add(cable_node.removeprefix("cable:"))

    row_group: dict[str, str] = {}
    group_rows: dict[str, list[int]] = {}
    for root, indices in components.items():
        cable_groups = sorted(component_cables[root])
        explicit = [name for name in cable_groups if name.startswith(("development_", "linked_"))]
        group = explicit[0] if explicit else (cable_groups[0] if cable_groups else root.removeprefix("case:"))
        group_rows[group] = indices
        for index in indices:
            case = str(rows[index].get("source_case_group") or rows[index].get("case_group") or "")
            row_group[case] = group
    return row_group, group_rows


def assign_splits(
    rows: list[dict[str, Any]], row_group: dict[str, str], group_rows: dict[str, list[int]],
    calibration_records: list[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, str]]:
    known_lab_group = "development_experiment:1500m_wet_joint"
    exposed_cases = {str(row.get("source_case_group") or "") for row in calibration_records}
    exposed_paths = {norm_path(str(row.get("source_relative_path") or "")) for row in calibration_records}

    forced: dict[str, str] = {}
    forced_reason: dict[str, str] = {}
    for group, indices in group_rows.items():
        group_rows_data = [rows[index] for index in indices]
        is_wet = group == known_lab_group or any(
            str(row.get("source_case_group") or "") == WET_CASE for row in group_rows_data
        )
        if is_wet:
            forced[group] = "test"
            forced_reason[group] = "known_lab_forced_test; excluded from Iter6 training"
            continue
        has_exposure = any(
            str(row.get("source_case_group") or "") in exposed_cases
            or norm_path(str(row.get("source_relative_path") or "")) in exposed_paths
            or row.get("development_status") == "development_only"
            for row in group_rows_data
        )
        if has_exposure:
            forced[group] = "train"
            forced_reason[group] = "prior_development_or_calibration_exposure_forced_train"

    total_groups = len(group_rows)
    targets = {
        "train": round(total_groups * 0.60),
        "val": round(total_groups * 0.20),
    }
    targets["test"] = total_groups - targets["train"] - targets["val"]
    counts = Counter(forced.values())
    pool = [group for group in group_rows if group not in forced]
    random.Random(SEED).shuffle(pool)
    split_for_group = dict(forced)
    for group in pool:
        remaining = {split: targets[split] - counts[split] for split in ("train", "val", "test")}
        split = max(("train", "val", "test"), key=lambda name: (remaining[name], -("train", "val", "test").index(name)))
        split_for_group[group] = split
        counts[split] += 1

    if len(split_for_group) != total_groups:
        raise RuntimeError("Group split did not assign every group")
    return split_for_group, forced_reason


def known_length_status(case_group: str, evidence_records: list[dict[str, Any]]) -> tuple[str, float | None]:
    lengths = sorted({
        float(item["length_m"])
        for item in evidence_records
        if item.get("case_group") == case_group
        and item.get("claim") == "length"
        and item.get("evidence_type") == "engineer_note"
        and item.get("confidence") == "high"
        and item.get("length_m") is not None
    })
    if not lengths:
        return "unknown_length", None
    if any(value > MAX_SOURCE_POSITION_M for value in lengths):
        return "known_over_2500m", max(lengths)
    if len(lengths) == 1:
        return "known_within_2500m", lengths[0]
    return "unknown_length", None


def label_radius(case_group: str, known_lab: bool) -> float:
    if known_lab:
        return WET_POSITION_RADIUS_M
    if "2025-5-15 15.23.15深圳龙岗某地（浸水）" in case_group:
        return TDR_POSITION_RADIUS_M
    return DEFAULT_POSITION_RADIUS_M


def read_feature(index: int, row: dict[str, Any], cfg: dict[str, Any]) -> tuple[int, np.ndarray, float | None]:
    from field_eval import _read_source_s11
    from features import preprocess

    source = SOURCE_ROOT / str(row["source_relative_path"])
    frequency, s11, source_velocity = _read_source_s11(source, row.get("sweep_rowid"), None)
    feature = preprocess(frequency, s11, cfg)
    return index, feature, source_velocity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path,
                        help="Exact output directory, for example E:/CableNNDataSet/CodeX/FieldNN_Iter6_20261022")
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--smoke", action="store_true", help="Limit output to at most two bags per split")
    parser.add_argument("--limit", type=int, help="Bag limit per split for --smoke only (1 or 2)")
    args = parser.parse_args()
    if args.limit is not None and (not args.smoke or args.limit < 1 or args.limit > 2):
        parser.error("--limit is only valid with --smoke and must be 1 or 2")

    sys.path.insert(0, str(ITER3_DIR))
    # Importing Iter3's reader and transform does not initialize or use CUDA.
    import field_eval

    rows = read_jsonl(MANIFEST_PATH)
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    evidence_records = evidence["records"]
    calibration_records = read_jsonl(CALIBRATION_SOURCES_PATH)
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg['model']['input_channels'] = 12
    rules_path = PROJECT_ROOT / "NN_Archive" / "20261022NNFirstTry" / "FieldNN_20261022" / "real_group_rules.json"

    case_to_group, group_rows = build_groups(rows, rules_path)
    split_for_group, forced_reason = assign_splits(rows, case_to_group, group_rows, calibration_records)
    case_lengths = {
        case: known_length_status(case, evidence_records)
        for case in {str(row.get("source_case_group") or "") for row in rows}
    }
    conflict_by_case = {key.casefold(): value for key, value in CONFLICT_CASES.items()}
    wet_positions = {
        int(rowid): float(position)
        for rowid, _, position in field_eval.KNOWN_LAB_SWEEPS
    }
    wet_source = norm_path(field_eval.KNOWN_LAB_RELATIVE.as_posix())

    row_by_key: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        row_by_key[(norm_path(str(row["source_relative_path"])), str(row.get("sweep_rowid")))].append(index)

    # Candidates are keyed by one source FDR and exactly the sweeps to which
    # that evidence applies. Different evidence scopes never share a bag.
    candidates: dict[tuple[str, str, str, tuple[int, ...]], dict[str, Any]] = {}
    exclusions: Counter[str] = Counter()
    eligible_evidence_records = 0
    unmatched_evidence_sweeps = 0
    for item in evidence_records:
        claim = item.get("claim")
        positions = [float(value) for value in (item.get("positions_m") or [])]
        if not positions or claim not in {"defect_position", "joint_position"}:
            continue
        case = str(item.get("case_group") or "")
        if case == WET_CASE:
            exclusions["wet_summary_replaced_by_sweep_specific_map"] += 1
            continue
        if item.get("evidence_type") != "engineer_note" or item.get("confidence") != "high":
            exclusions["not_high_confidence_engineer_note"] += 1
            continue
        if case.casefold() in conflict_by_case:
            exclusions["explicit_coordinate_conflict"] += 1
            continue
        length_status, _ = case_lengths.get(case, ("unknown_length", None))
        if length_status == "known_over_2500m":
            exclusions["known_length_over_2500m"] += 1
            continue

        source = norm_path(str(item.get("source_relative_path") or ""))
        rowids = [str(value) for value in (item.get("sweep_rowids") or [])]
        radius = label_radius(case, known_lab=False)
        source_positions = [value for value in positions if 0.0 < value <= MAX_SOURCE_POSITION_M]
        exclusions["position_outside_0_2500m"] += len(positions) - len(source_positions)
        if not source_positions:
            continue
        applicable = [
            index for rowid in rowids
            for index in row_by_key.get((source, rowid), [])
        ]
        if not applicable:
            unmatched_evidence_sweeps += len(rowids)
            exclusions["no_matching_manifest_sweep"] += 1
            continue

        eligible_evidence_records += 1
        group = case_to_group.get(case, str(item.get("cable_group") or case))
        applicable_key = tuple(sorted(applicable))
        key = (group, case, source, applicable_key)
        bag = candidates.setdefault(key, {
            "split": split_for_group[group],
            "group": group,
            "cable_group": str(item.get("cable_group") or group),
            "case_group": case,
            "source_relative_path": str(item["source_relative_path"]),
            "original_indices": set(),
            "positions_radius": {},
            "known_lab": False,
            "annotation_claims": set(),
            "evidence_files": set(),
            "unknown_length": length_status == "unknown_length",
            "known_length_m": case_lengths.get(case, ("unknown_length", None))[1],
        })
        bag["original_indices"].update(applicable)
        bag["annotation_claims"].add(str(claim))
        if item.get("file"):
            bag["evidence_files"].add(str(item["file"]))
        for position in source_positions:
            bag["positions_radius"][position] = max(radius, bag["positions_radius"].get(position, 0.0))

    # Use the frozen Iter3 sweep-to-label mapping: each curve gets only its own
    # single known wet-joint location; the general 11-location summary is ignored.
    for row_index, row in enumerate(rows):
        source = norm_path(str(row["source_relative_path"]))
        try:
            rowid = int(row.get("sweep_rowid"))
        except (TypeError, ValueError):
            continue
        if source != wet_source or rowid not in wet_positions:
            continue
        case = str(row.get("source_case_group") or WET_CASE)
        group = case_to_group[case]
        position = wet_positions[rowid]
        key = (group, case, f"{source}#rowid={rowid}")
        candidates[key] = {
            "split": "test",
            "group": group,
            "cable_group": str(row.get("cable_group") or group),
            "case_group": case,
            "source_relative_path": str(row["source_relative_path"]),
            "original_indices": {row_index},
            "positions_radius": {position: WET_POSITION_RADIUS_M},
            "known_lab": True,
            "annotation_claims": {"known_lab_wet_joint"},
            "evidence_files": [field_eval.KNOWN_LAB_EVIDENCE.as_posix()],
            "unknown_length": False,
            "known_length_m": 1500.0,
        }

    # Smoke selection is applied after the complete group split and label audit.
    ordered_candidates = sorted(
        candidates.values(), key=lambda item: (item["split"], item["group"], item["source_relative_path"], item["known_lab"])
    )
    if args.smoke:
        cap = args.limit or 2
        chosen: list[dict[str, Any]] = []
        for split in ("train", "val", "test"):
            split_bags = [bag for bag in ordered_candidates if bag["split"] == split]
            if split == "test":
                wet_bags = [bag for bag in split_bags if bag["known_lab"]]
                regular_bags = [bag for bag in split_bags if not bag["known_lab"]]
                split_bags = wet_bags[:1] + regular_bags
            chosen.extend(split_bags[:cap])
        ordered_candidates = chosen

    original_indices = sorted({index for bag in ordered_candidates for index in bag["original_indices"]})
    if not original_indices:
        raise RuntimeError("No labeled bag curves remain after the evidence and split filters")
    new_index = {old: new for new, old in enumerate(original_indices)}
    cfg["response"]["points"] = int(cfg["response"]["points"])
    feature_results: dict[int, tuple[np.ndarray, float | None]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(read_feature, index, rows[index], cfg): index
            for index in original_indices
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="FDR -> 12-channel field features"):
            index, feature, source_velocity = future.result()
            feature_results[index] = (feature, source_velocity)

    feature_rows: list[np.ndarray] = []
    record_rows: list[dict[str, Any]] = []
    original_record: dict[int, dict[str, Any]] = {}
    nominal_velocity_m_per_us = float(cfg["response"]["velocity_m_s"]) / 1e6
    for old_index in original_indices:
        row = rows[old_index]
        feature, source_velocity = feature_results[old_index]
        if feature.shape != (12, int(cfg["response"]["points"])):
            raise ValueError(f"Unexpected field feature shape {feature.shape} for manifest row {old_index}")
        ratio = source_velocity / nominal_velocity_m_per_us if source_velocity is not None else 1.0
        source_path = norm_path(str(row["source_relative_path"]))
        case = str(row.get("source_case_group") or row.get("case_group") or "")
        record = {
            "index": new_index[old_index],
            "source_relative_path": str(row["source_relative_path"]),
            "sweep_rowid": int(row["sweep_rowid"]),
            "sweep_name": str(row.get("sweep_name") or ""),
            "phase": str(row.get("phase") or "unknown"),
            "cable_group": str(row.get("cable_group") or ""),
            "case_group": case,
            "group": case_to_group[case],
            "split": split_for_group[case_to_group[case]],
            "source_wave_velocity_m_per_us": source_velocity,
            "source_scale_ratio": ratio,
            "source_axis_status": "source_report" if source_velocity is not None else "nominal_fallback",
            "positions_source_m": [],
            "positions_nominal_m": [],
            "position_windows_source_m": [],
            "position_windows_nominal_m": [],
            "label_claims": [],
            "known_lab": False,
            "unknown_length": case_lengths.get(case, ("unknown_length", None))[0] == "unknown_length",
            "known_length_m": case_lengths.get(case, ("unknown_length", None))[1],
            "label_scope": "positive-only bag supervision; unannotated locations are not negatives",
        }
        feature_rows.append(feature)
        record_rows.append(record)
        original_record[old_index] = record

    bag_rows: list[dict[str, Any]] = []
    bag_index = 0
    for candidate in ordered_candidates:
        accepted_positions: set[float] = set()
        accepted_indices: list[int] = []
        for old_index in sorted(candidate["original_indices"]):
            record = original_record[old_index]
            radius_by_position = candidate["positions_radius"]
            valid: list[float] = []
            for position in sorted(radius_by_position):
                nominal = position / float(record["source_scale_ratio"])
                if not (0.0 < position <= MAX_SOURCE_POSITION_M):
                    exclusions["position_outside_0_2500m"] += 1
                    continue
                if not (MIN_NOMINAL_POSITION_M <= nominal <= MAX_NOMINAL_POSITION_M):
                    exclusions["position_outside_decodable_nominal_axis"] += 1
                    continue
                valid.append(position)
            if not valid:
                continue
            record["positions_source_m"] = valid
            record["positions_nominal_m"] = [position / float(record["source_scale_ratio"]) for position in valid]
            record["position_windows_source_m"] = [
                [max(0.0, position - radius_by_position[position]), position + radius_by_position[position]]
                for position in valid
            ]
            record["position_windows_nominal_m"] = [
                [window[0] / float(record["source_scale_ratio"]), window[1] / float(record["source_scale_ratio"])]
                for window in record["position_windows_source_m"]
            ]
            record["label_claims"] = sorted(candidate["annotation_claims"])
            record["known_lab"] = bool(candidate["known_lab"])
            accepted_positions.update(valid)
            accepted_indices.append(new_index[old_index])
        if not accepted_indices:
            exclusions["bag_has_no_decodable_positive_position"] += 1
            continue
        candidate_windows = [
            [max(0.0, position - candidate["positions_radius"][position]),
             position + candidate["positions_radius"][position]]
            for position in sorted(accepted_positions)
        ]
        bag_rows.append({
            "bag_id": bag_index,
            "split": candidate["split"],
            "group": candidate["group"],
            "cable_group": candidate["cable_group"],
            "case_group": candidate["case_group"],
            "source_relative_path": candidate["source_relative_path"],
            "indices": accepted_indices,
            "positions_source_m": sorted(accepted_positions),
            "position_windows_source_m": candidate_windows,
            "known_lab": bool(candidate["known_lab"]),
            "unknown_length": bool(candidate["unknown_length"]),
            "known_length_m": candidate["known_length_m"],
            "annotation_claims": sorted(candidate["annotation_claims"]),
            "evidence_files": sorted(candidate["evidence_files"]),
            "phase_policy": "MIL over only the sweeps to which this source-specific evidence applies",
        })
        bag_index += 1

    # Keep only records that survived axis/range filtering into at least one bag.
    used_indices = sorted({index for bag in bag_rows for index in bag["indices"]})
    if len(used_indices) != len(record_rows):
        keep = set(used_indices)
        selected_record_indices = [index for index, record in enumerate(record_rows) if record["index"] in keep]
        old_new = {
            record_rows[old_index]["index"]: new_index
            for new_index, old_index in enumerate(selected_record_indices)
        }
        feature_rows = [feature_rows[index] for index in selected_record_indices]
        record_rows = [record_rows[index] for index in selected_record_indices]
        for new, record in enumerate(record_rows):
            record["index"] = new
        for bag in bag_rows:
            bag["indices"] = [old_new[index] for index in bag["indices"] if index in old_new]

    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "field_features.npy", np.stack(feature_rows).astype(np.float32, copy=False))
    with (output_dir / "field_records.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for record in record_rows:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
    (output_dir / "field_bags.json").write_text(
        json.dumps({"bags": bag_rows}, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    group_counts = Counter(split_for_group.values())
    full_record_counts = Counter(split_for_group[case_to_group[str(row.get("source_case_group") or row.get("case_group") or "")]] for row in rows)
    feature_record_counts = Counter(record["split"] for record in record_rows)
    bag_counts = Counter(bag["split"] for bag in bag_rows)
    summary = {
        "schema": "fieldnn-iter8-weak-field-cache-1",
        "output": str(output_dir),
        "manifest": str(MANIFEST_PATH),
        "manifest_rows_full_scope": len(rows),
        "source_root_read_only": str(SOURCE_ROOT),
        "feature_shape": list(np.stack(feature_rows[:1]).shape[1:]) if feature_rows else [12, int(cfg["response"]["points"])],
        "features_are_bag_curves_only": True,
        "split_seed": SEED,
        "split_target_group_fraction": {"train": 0.60, "val": 0.20, "test": 0.20},
        "split_group_basis": "connected case_group/cable_group/source_file aliases; all sweeps from one source stay together",
        "groups_by_split": dict(group_counts),
        "manifest_rows_by_split_before_smoke": dict(full_record_counts),
        "feature_records_by_split": dict(feature_record_counts),
        "bags_by_split": dict(bag_counts),
        "known_lab_policy": "1500 m wet-joint group forced to test and excluded from training; one Iter3 KNOWN_LAB_SWEEPS position per matching rowid",
        "calibration_exposure": {
            "source_list": str(CALIBRATION_SOURCES_PATH),
            "source_files": len(calibration_records),
            "case_groups": sorted({str(item.get("source_case_group") or "") for item in calibration_records}),
            "policy": "all connected groups with prior development_only/calibration exposure are forced to train",
        },
        "length_policy": "known high-confidence engineer lengths above 2500 m are excluded; missing or conflicting length evidence is marked unknown_length",
        "position_policy": {
            "source_axis_range_m": [0.0, MAX_SOURCE_POSITION_M],
            "nominal_decode_axis_range_m": [MIN_NOMINAL_POSITION_M, MAX_NOMINAL_POSITION_M],
            "engineer_point_window_radius_m": DEFAULT_POSITION_RADIUS_M,
            "known_lab_window_radius_m": WET_POSITION_RADIUS_M,
            "2025_05_15_tdr_window_radius_m": TDR_POSITION_RADIUS_M,
            "unannotated_locations_are_negative": False,
            "phase_unknown": "bag-MIL; no per-phase negative labels",
        },
        "explicit_conflict_exclusions": CONFLICT_CASES,
        "forced_group_reasons": forced_reason,
        "exclusion_counts": dict(exclusions),
        "eligible_evidence_records": eligible_evidence_records,
        "unmatched_evidence_sweep_count": unmatched_evidence_sweeps,
        "smoke": bool(args.smoke),
        "smoke_bag_limit_per_split": (args.limit or 2) if args.smoke else None,
        "heldout_scope_note": "This is label-isolated for Iter6 only. The same 213 field sweeps were already used in several prior development/evaluation rounds; the split is not an independent blind test.",
        "terminal_truth_included": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
