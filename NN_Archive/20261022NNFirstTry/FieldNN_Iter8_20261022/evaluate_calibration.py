"""Count false cable-event peaks on confirmed short/no-cable references."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import CalibrationDataset
from models import build_model

HERE = Path(__file__).resolve().parent
ITER5 = HERE.parent / "FieldNN_Iter5_20261022"
spec = importlib.util.spec_from_file_location('iter8_calibration_decoder', ITER5 / 'decode.py')
decoder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(decoder)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--limit', type=int)
    args = ap.parse_args()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = state['config']
    baseline_eight = int(cfg['model']['input_channels']) == 8
    if baseline_eight:
        old_spec = importlib.util.spec_from_file_location('iter8_calibration_baseline_model', HERE.parent / 'FieldNN_Iter3_20261022' / 'models.py')
        old_models = importlib.util.module_from_spec(old_spec)
        old_spec.loader.exec_module(old_models)
        model = old_models.build_model(state['model_name'], **cfg['model']).to(device).eval()
    else:
        model = build_model(state['model_name'], **cfg['model']).to(device).eval()
    model.load_state_dict(state['model'])
    thresholds = {'event': .6, 'interval': .99, 'terminal': .4}
    result = {'checkpoint': str(args.checkpoint), 'event_threshold': .6,
              'semantics': 'User-confirmed short/no-test-cable references; no cable-event truth at >=30 nominal m. Not healthy cables; terminal unscored.',
              'split_basis': 'Same connected source/case groups as Iter6 field data', 'splits': {}}
    for split in ('train', 'val', 'test'):
        dataset = CalibrationDataset(args.data, split)
        entries = []
        with torch.inference_mode():
            for step, (x, _, indices) in enumerate(tqdm(DataLoader(dataset, batch_size=8), desc=f'calibration {split}')):
                with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                    inputs = x[:, :8] if baseline_eight else x
                    probs = model(inputs.to(device)).float().sigmoid().cpu().numpy()
                for probability, index in zip(probs, indices.tolist()):
                    prediction = decoder.decode_predictions(probability, cfg, thresholds)
                    entries.append({'index': index, 'false_cable_events': prediction['events'],
                                    'false_cable_intervals': prediction['intervals']})
                if args.limit and len(entries) >= args.limit:
                    entries = entries[:args.limit]
                    break
        counts = [len(row['false_cable_events']) for row in entries]
        result['splits'][split] = {'references': len(entries), 'false_event_count': sum(counts),
                                  'false_events_per_reference': float(np.mean(counts)) if counts else None,
                                  'fraction_with_false_event': float(np.mean(np.asarray(counts) > 0)) if counts else None,
                                  'entries': entries}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: {a:b for a,b in v.items() if a != 'entries'} for k,v in result['splits'].items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
