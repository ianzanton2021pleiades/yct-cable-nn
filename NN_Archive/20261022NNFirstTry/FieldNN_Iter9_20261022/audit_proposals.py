"""Report label-independent physical candidate coverage without model/GPU use."""
import argparse
import json
from pathlib import Path
from evaluate import read_jsonl, field_reports, proposal_prediction_rows, load_evaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = read_jsonl(args.data / 'field_records.jsonl')
    candidates = read_jsonl(args.data / 'field_candidates.jsonl')
    bags = json.loads((args.data / 'field_bags.json').read_text(encoding='utf-8'))['bags']
    _, report = field_reports(records, bags, candidates,
                              proposal_prediction_rows(records, candidates), 0., load_evaluator())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    val = report['by_split']['val']['positive_labels']['independent_group_macro']['20']
    print(json.dumps({'scope': 'Physical proposals only, not trained detections',
                      'validation_known_position_group_coverage20': val['macro_hit_rate'],
                      'report': str(args.output)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
