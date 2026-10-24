"""Selected checkpoint: full-corpus FDR inference and scoped NN timing.

This is a user-run GPU stage. It never reselects weights or thresholds on test.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
I8 = HERE.parent / 'FieldNN_Iter8_20261022'
I5 = HERE.parent / "FieldNN_Iter5_20261022"
sys.path.insert(0, str(I8))
from models import build_model
import features as iter8_features


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--run', type=Path, required=True, help='Training output/unet')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch-size', type=int, default=64, help='Batch size used for training-validation inference; full FDR inference uses batch 1')
    args = ap.parse_args()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    amp = device.type == 'cuda'
    # A recoverable checkpoint was captured when identical rational hit rates
    # differed by 1e-16. Apply the declared candidate/loss tie-break before test.
    tie_path = args.run / 'tie_candidate.pt'
    if tie_path.is_file():
        training = load_module(HERE / 'train_recorded.py', 'iter10_numeric_selection')
        candidate = torch.load(tie_path, map_location='cpu', weights_only=False)
        dense = torch.load(args.run / 'best_dense.pt', map_location='cpu', weights_only=False)
        step = int(candidate['optimizer_step'])
        candidate_model = build_model('unet', **candidate['config']['model']).to(device)
        candidate_model.load_state_dict(candidate['model'])
        field = training.FieldCache.from_root(args.data)
        field_train = training.i8_train.FieldBags(args.data, 'train', device)
        field_val = training.i8_train.FieldBags(args.data, 'val', device)
        context = training.FieldMetricContext(args.data, field, field_train, field_val)
        validation = context.evaluate(candidate_model, candidate['config'], device, args.batch_size)
        write_json(args.run / 'recovered_checkpoint_validation.json', {
            'optimizer_step': step, 'validation': validation,
            'note': 'Re-evaluated the saved weights on train/val only; no historical metric was substituted.'})
        del candidate_model, context, field_train, field_val, field
        promoted = False
        if training.validation_rank(validation) > training.validation_rank(dense['validation']):
            promoted = True
            dense = {'model_name': 'unet', 'config': candidate['config'], 'model': candidate['model'],
                     'epoch': (step - 1) // int(candidate['run_config']['steps_per_epoch']),
                     'step': step, 'optimizer_step': step, 'validation': validation,
                     'selected_source': 'iter10_dense', 'threshold': validation['val_threshold']}
            torch.save(dense, args.run / 'best_dense.pt')
        reference = torch.load(I8 / 'output' / 'iter8' / 'unet' / 'best.pt', map_location='cpu', weights_only=False)
        reference_validation = json.loads((args.run / 'reference_validation.json').read_text(encoding='utf-8'))
        selected = training.select_checkpoint(args.run, 'unet', reference, reference_validation,
                                              dense['validation'], int(dense['optimizer_step']))
        last = torch.load(args.run / 'last.pt', map_location='cpu', weights_only=False)
        last.update(dense_best_validation=dense['validation'], dense_best_step=int(dense['optimizer_step']),
                    dense_best_rank=training.validation_rank(dense['validation']))
        if promoted:
            row = {key: value for key, value in validation.items() if key in training.HISTORY_FIELDS}
            row.update(optimizer_step=step,
                       amp_skipped_steps=sum(int(r['amp_skipped']) for r in candidate['optimizer_rows']),
                       synthetic_val_loss=None, calibration_val_loss=None,
                       calibration_val_high_bin_fraction=None,
                       lr_backbone=candidate['optimizer']['param_groups'][0]['lr'],
                       lr_stem=candidate['optimizer']['param_groups'][-1]['lr'])
            history = [r for r in last['history_rows'] if int(r['optimizer_step']) != step]
            history.append(row)
            last['history_rows'] = sorted(history, key=lambda r: int(r['optimizer_step']))
            training.write_csv(args.run / 'history_steps.csv', last['history_rows'], training.HISTORY_FIELDS)
        torch.save(last, args.run / 'last.pt')
        info_path = args.run / 'run_info.json'
        info = json.loads(info_path.read_text(encoding='utf-8'))
        info['dense_best_step'] = int(dense['optimizer_step'])
        info.update(selected_source=selected['selected_source'], selected_step=selected['step'],
                    selected_threshold=selected['threshold'])
        info['record_points'] = len(last['history_rows'])
        info['numeric_tie_correction'] = 'Compared saved dense best and recoverable checkpoint; hit rounded to 12 decimals, then candidate count and MIL loss; no test used.'
        write_json(info_path, info)
        del candidate, dense, reference, last
    saved = torch.load(args.run / 'best.pt', map_location='cpu', weights_only=False)
    choice = json.loads((args.run / 'selection.json').read_text(encoding='utf-8'))
    cfg = saved['config']
    model = build_model(saved['model_name'], **cfg['model']).to(device).eval()
    model.load_state_dict(saved['model'])
    decoder = load_module(I5 / 'decode.py', 'iter10_frozen_decoder')
    evaluation = load_module(HERE.parent / 'FieldNN_Iter6_20261022' / 'evaluate_field.py', 'iter10_field_evaluation')
    features = np.load(args.data / 'field_features.npy', mmap_mode='r')
    records = evaluation.read_jsonl(args.data / 'field_records.jsonl')
    bags = json.loads((args.data / 'field_bags.json').read_text(encoding='utf-8'))['bags']
    output = args.run / 'evaluation'
    output.mkdir(exist_ok=True)
    thresholds = {'event': float(choice['threshold']), 'interval': .99, 'terminal': .4}
    inference_thresholds = dict(thresholds)
    if choice['validation'].get('val_density_fallback', False):
        inference_thresholds['event'] = 1.000001  # Explicit reject-all, as in validation.
    field_operating_point = {
        'model_name': saved['model_name'], 'checkpoint_epoch': saved['epoch'],
        'calibration_split': 'val', 'thresholds': inference_thresholds,
        'policy': 'Field validation only; synthetic calibration is separate.'}
    write_json(args.run / 'field_thresholds.json', field_operating_point)
    write_json(args.run / 'thresholds.json', field_operating_point)
    # The training cache contains only 138 annotated sweeps. Use the established
    # FDR pipeline to retain the full 213-sweep corpus, including unlabelled rows.
    pipeline = load_module(HERE.parent / 'FieldNN_Iter3_20261022' / 'field_eval.py', 'iter10_full_fdr')
    pipeline.build_model = build_model
    pipeline.preprocess = iter8_features.preprocess
    pipeline.decode_predictions = decoder.decode_predictions
    original_argv = sys.argv
    sys.argv = [str(I8 / 'field_infer.py'), '--checkpoint', str(args.run / 'best.pt'),
                '--output', str(output), '--device', args.device, '--subset', 'all',
                '--distance-axis', 'source', '--known-lab', '--plots', '0', '--no-length-view']
    pipeline.main()
    sys.argv = original_argv
    keyed = evaluation.load_prediction_files([f'selected={output / "predictions.jsonl"}'])['selected']
    predictions = evaluation.prediction_by_index(records, keyed)
    coverage = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    record_map = {int(row['index']): row for row in records}
    report = {'model': 'validation_selected_unet', 'selection': choice,
              'cache': str(args.data), 'thresholds': thresholds,
              'validation_inference_batch_size': args.batch_size, 'full_fdr_inference_batch_size': 1,
              'full_corpus_coverage': {key: coverage[key] for key in ('selected_rows', 'processed', 'not_processed')},
              'note': 'Event evaluation only. Full FDR corpus inference; 138 annotated cached sweeps for grouped scoring. '
                      'Iter8 preprocessing; source/report coordinates. '
                      'Unknown locations remain unknown; retained sets have been used repeatedly for development.',
              'splits': {split: evaluation.split_report(split, records, bags, record_map, predictions, 0)
                         for split in ('train', 'val', 'test')}}
    write_json(output / 'field_report.json', report)

    # Batch-1 forward only: fixed preloaded field tensor, warmups excluded.
    x = torch.from_numpy(np.array(features[:1], copy=True)).to(device)
    def forward():
        with torch.autocast(device_type=device.type, enabled=amp):
            return model(x)
    def sync():
        if amp:
            torch.cuda.synchronize(device)
    for _ in range(20):
        forward()
    sync()
    times = []
    for _ in range(100):
        sync()
        begun = time.perf_counter()
        forward()
        sync()
        times.append((time.perf_counter() - begun) * 1000.)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    package = args.run / 'inference.pt'
    torch.save({'model_name': saved['model_name'], 'config': cfg, 'model': state,
                'epoch': saved.get('epoch'), 'selection': choice}, package)
    write_json(args.run / 'nn_benchmark.json', {
        'scope': 'NN forward only; input already on device; excludes FDR/FFT/features/H2D/sigmoid/decoding',
        'batch_size': 1, 'input_shape': list(x.shape), 'warmups': 20, 'repetitions': 100,
        'device': torch.cuda.get_device_name(device) if amp else str(device),
        'torch': torch.__version__, 'autocast': 'float16' if amp else 'disabled/float32',
        'median_ms': float(np.median(times)), 'p95_ms': float(np.percentile(times, 95)),
        'mean_ms': float(np.mean(times)), 'raw_ms': times,
        'parameters': sum(p.numel() for p in model.parameters()),
        'weight_tensor_bytes': sum(t.numel() * t.element_size() for t in state.values()),
        'inference_package_bytes': package.stat().st_size,
        'checkpoint_bytes': (args.run / 'best.pt').stat().st_size,
        'timing_limit': 'One run; concurrent GPU work and WDDM scheduling may affect these measurements.'})
    print(f'Selected model field report: {output / "field_report.json"}', flush=True)


if __name__ == '__main__':
    main()
