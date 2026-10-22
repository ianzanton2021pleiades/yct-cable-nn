from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[2]
DEFAULT_SCOPE_CSV = PROJECT_ROOT / "DifferentAgent" / "CodeX" / "Core16_Field" / "output" / "scope_measurements.csv"
DEFAULT_EVIDENCE_JSON = PROJECT_ROOT / "DifferentAgent" / "CodeX" / "Core16_Field" / "output" / "evidence_records.json"
DEFAULT_RULES_JSON = EXPERIMENT_DIR / "real_group_rules.json"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "output" / "real_manifest.jsonl"


def as_bool(value: Any) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes", "y"}


def normalize_relative_directory(value: Any) -> str:
    return str(value or "").replace("\\", "/").strip("/").casefold()


def phase_from_sweep_name(value: Any) -> tuple[str, list[str]]:
    name = str(value or "").strip()
    match = re.search(r"(?<![A-Za-z])([ABC]{1,3})\s*相", name, flags=re.IGNORECASE)
    if match:
        candidates = sorted(set(match.group(1).upper()), key="ABC".index)
        if len(candidates) == 1:
            return candidates[0], candidates
        return "multiple_or_ambiguous", candidates
    if name.casefold() in {"a", "b", "c"}:
        phase = name.upper()
        return phase, [phase]
    return "unknown", []


def build_rule_lookup(rules: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    development: dict[str, dict[str, Any]] = {}
    linked: dict[str, dict[str, Any]] = {}
    for group in rules.get("development_groups", []):
        for case_group in group.get("case_groups", []):
            development[case_group] = group
    for group in rules.get("linked_cable_groups", []):
        for case_group in group.get("case_groups", []):
            linked[case_group] = group
    return development, linked


def assign_group(case_group: str, development: dict[str, dict[str, Any]],
                 linked: dict[str, dict[str, Any]], rules: dict[str, Any]) -> dict[str, str]:
    if case_group in development:
        group = development[case_group]
        return {
            "cable_group": str(group["cable_group"]),
            "development_status": "development_only",
            "group_rule_id": str(group["group_id"]),
            "group_note": str(group["note"]),
        }
    if case_group in linked:
        group = linked[case_group]
        return {
            "cable_group": str(group["cable_group"]),
            "development_status": "independent_identity_unconfirmed",
            "group_rule_id": str(group["group_id"]),
            "group_note": str(group["note"]),
        }
    fallback = rules["fallback_grouping"]
    return {
        "cable_group": str(fallback["cable_group_format"]).format(source_case_group=case_group),
        "development_status": str(fallback["development_status"]),
        "group_rule_id": "fallback_case_group_unconfirmed",
        "group_note": str(fallback["note"]),
    }


def build_evidence_index(evidence_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = evidence_data.get("records", [])
    return {
        normalize_relative_directory(record.get("source_directory_relative_path")): record
        for record in records
        if record.get("source_directory_relative_path")
    }


def evidence_for_row(row: dict[str, str], evidence_index: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    record = evidence_index.get(normalize_relative_directory(row.get("matched_evidence_directory")))
    if not record:
        return None
    fields = record.get("evidence_fields", {})
    return {
        "source_relative_path": record.get("source_relative_path"),
        "related_fdr_filenames": record.get("related_fdr_filenames", []),
        "explicit_length_claims": fields.get("explicit_length_claims", []),
        "joint_and_endpoint_claim_lines": fields.get("joint_and_endpoint_claim_lines", []),
        "connection_mode_or_contact_claim_lines": fields.get("connection_mode_or_contact_claim_lines", []),
        "engineer_interpretation_lines": fields.get("engineer_interpretation_lines", []),
        "field_or_other_test_observation_lines": fields.get("field_or_other_test_observation_lines", []),
        "repair_dissection_or_replacement_lines": fields.get("repair_dissection_or_replacement_lines", []),
        "repeat_and_topology_lines": fields.get("repeat_and_topology_lines", []),
        "acquisition_signal_or_frequency_quality_lines": fields.get("acquisition_signal_or_frequency_quality_lines", []),
        "evidence_boundary": record.get("evidence_boundary"),
    }


def manifest_record(row: dict[str, str], evidence_index: dict[str, dict[str, Any]],
                    development: dict[str, dict[str, Any]], linked: dict[str, dict[str, Any]],
                    rules: dict[str, Any]) -> dict[str, Any]:
    case_group = str(row.get("case_group") or row.get("top_level_case") or "")
    phase, phase_candidates = phase_from_sweep_name(row.get("sweep_name"))
    joined_evidence = evidence_for_row(row, evidence_index)
    try:
        sweep_rowid: int | str = int(row.get("sweep_rowid", ""))
    except (TypeError, ValueError):
        sweep_rowid = str(row.get("sweep_rowid") or "")

    length_scope = {
        "scope_bucket": row.get("scope_bucket") or None,
        "scope_basis": row.get("scope_basis") or None,
        "matched_evidence_directory": row.get("matched_evidence_directory") or None,
        "matched_evidence_depth": int(row["matched_evidence_depth"])
        if str(row.get("matched_evidence_depth", "")).isdigit() else None,
        "evidence_assessment_bucket": row.get("evidence_assessment_bucket") or None,
        "evidence_bucket_basis": row.get("evidence_bucket_basis") or None,
        "manual_doc_source": row.get("manual_doc_source") or None,
        "text_evidence": joined_evidence,
    }
    return {
        "source_relative_path": row.get("source_relative_path"),
        "sweep_rowid": sweep_rowid,
        "sweep_name": row.get("sweep_name") or "",
        "phase": phase,
        "phase_candidates": phase_candidates,
        "source_case_group": case_group,
        **assign_group(case_group, development, linked, rules),
        "label_status": "unknown",
        "length_scope": length_scope,
    }


def prepare_manifest(scope_csv: Path, evidence_json: Path, rules_json: Path,
                     output_path: Path) -> dict[str, Any]:
    rules = json.loads(rules_json.read_text(encoding="utf-8"))
    evidence_data = json.loads(evidence_json.read_text(encoding="utf-8"))
    evidence_index = build_evidence_index(evidence_data)
    development, linked = build_rule_lookup(rules)

    with scope_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))

    scope_counts: Counter[str] = Counter()
    development_counts_by_scope: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    for row in rows:
        is_valid = as_bool(row.get("valid_curve"))
        bucket = str(row.get("scope_bucket") or "missing_scope")
        scope_key = bucket if is_valid else "invalid_curve"
        scope_counts[scope_key] += 1
        case_group = str(row.get("case_group") or row.get("top_level_case") or "")
        assignment = assign_group(case_group, development, linked, rules)
        development_counts_by_scope[f"{scope_key}:{assignment['development_status']}"] += 1
        if is_valid and bucket == "up_to_2500m":
            selected.append(manifest_record(row, evidence_index, development, linked, rules))

    selected.sort(key=lambda item: (str(item["source_relative_path"]).casefold(), str(item["sweep_rowid"])))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for item in selected:
            stream.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")

    return {
        "scope_csv": str(scope_csv),
        "evidence_json": str(evidence_json),
        "rules_json": str(rules_json),
        "output_jsonl": str(output_path),
        "input_sweep_rows": len(rows),
        "valid_curve_scope_counts": dict(sorted(scope_counts.items())),
        "development_status_by_scope": dict(sorted(development_counts_by_scope.items())),
        "manifest_rows": len(selected),
        "manifest_development_status_counts": dict(sorted(Counter(item["development_status"] for item in selected).items())),
        "manifest_phase_counts": dict(sorted(Counter(item["phase"] for item in selected).items())),
        "manifest_label_status_counts": dict(sorted(Counter(item["label_status"] for item in selected).items())),
        "evidence_directory_match_count": sum(item["length_scope"]["text_evidence"] is not None for item in selected),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="从已有Core16现场清单生成保守的短线现场JSONL清单")
    parser.add_argument("--scope-csv", type=Path, default=DEFAULT_SCOPE_CSV)
    parser.add_argument("--evidence-json", type=Path, default=DEFAULT_EVIDENCE_JSON)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES_JSON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    stats = prepare_manifest(args.scope_csv, args.evidence_json, args.rules, args.output)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
