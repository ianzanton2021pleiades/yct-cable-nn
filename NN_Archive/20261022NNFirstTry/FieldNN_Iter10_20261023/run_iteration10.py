"""User-scheduled bounded refinement, true curves, and meeting artifacts."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
I8 = HERE.parent / 'FieldNN_Iter8_20261022'


def stage(name, command, output, env):
    log = output / 'logs' / f'{name}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    begun = time.perf_counter()
    print(f'\n=== {name} ===\n{subprocess.list2cmdline(command)}', flush=True)
    with log.open('a', encoding='utf-8') as stream:
        stream.write('\n' + subprocess.list2cmdline(command) + '\n')
        with subprocess.Popen(command, cwd=HERE, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                              errors='replace', bufsize=1) as process:
            for line in process.stdout:
                print(line, end='', flush=True)
                stream.write(line)
                stream.flush()
            code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)
    return {'stage': name, 'seconds': time.perf_counter() - begun, 'log': str(log)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, default=Path(r'E:\CableNNDataSet\CodeX\FieldNN_Iter8_20261022'))
    ap.add_argument('--synthetic', type=Path, default=Path(r'E:\CableNNDataSet\CodeX\FieldNN_Iter4_20261022'))
    ap.add_argument('--output', type=Path, default=HERE / 'output' / 'iter10_50ep')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--steps', type=int, help='Optional explicit update budget instead of epochs')
    ap.add_argument('--record-every', type=int, default=100)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--label', default='Iter10-U-Net')
    ap.add_argument('--model', choices=['unet', 'transformer'], default='unet')
    ap.add_argument('--init-iter8', action='store_true')
    ap.add_argument('--stem-only', action='store_true')
    ap.add_argument('--weak-weight', type=float)
    ap.add_argument('--stem-lr', type=float)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    affinity_spec = importlib.util.spec_from_file_location(
        'iter10_cpu_budget', HERE.parent / 'FieldNN_Iter3_20261022' / 'run_iteration.py')
    affinity_module = importlib.util.module_from_spec(affinity_spec)
    affinity_spec.loader.exec_module(affinity_module)
    cpu = affinity_module.reserve_one_core()
    env = os.environ.copy()
    env.update(PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env['MPLCONFIGDIR'] = str(args.output.resolve() / 'matplotlib_cache')
    env['MPLBACKEND'] = 'Agg'
    prefix = [sys.executable, '-X', 'utf8', '-B']
    stages = []
    run = args.output / args.model
    summary = {'status': 'running', 'cpu': cpu, 'data': str(args.data),
               'synthetic': str(args.synthetic), 'output': str(args.output), 'stages': stages}
    summary_path = args.output / 'iteration10_summary.json'
    def save():
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    def execute(name, cmd, custom_env=None):
        stages.append(stage(name, prefix + cmd, args.output, custom_env or env))
        save()
    save()
    execute('train_recorded', [str(HERE / 'train_recorded.py'), '--data', str(args.data),
            '--synthetic', str(args.synthetic), '--output', str(args.output), '--model', args.model,
            *(['--steps', str(args.steps)] if args.steps is not None else ['--epochs', str(args.epochs)]),
            '--record-every', str(args.record_every),
            '--batch-size', str(args.batch_size), '--workers', str(args.workers),
            '--device', args.device, *(['--resume'] if args.resume else []),
            *(['--init-iter8'] if args.init_iter8 else []),
            *(['--stem-only'] if args.stem_only else []),
            *(['--weak-weight', str(args.weak_weight)] if args.weak_weight is not None else []),
            *(['--stem-lr', str(args.stem_lr)] if args.stem_lr is not None else [])])
    execute('selected_field_and_nn_timing', [str(HERE / 'postprocess.py'), '--data', str(args.data),
            '--run', str(run), '--device', args.device, '--batch-size', str(args.batch_size)])
    execute('true_training_curves', [str(HERE / 'plot_curves.py'), '--run', str(run)])
    execute('short_reference', [str(I8 / 'evaluate_calibration.py'), '--data', str(args.data),
            '--checkpoint', str(run / 'best.pt'), '--device', args.device,
            '--output', str(run / 'calibration_iter8.json')])
    # Synthetic thresholds are a separate operating point, never a field threshold.
    execute('synthetic_val_calibrate', [str(I8 / 'evaluate.py'), '--data', str(args.synthetic),
            '--checkpoint', str(run / 'best.pt'), '--split', 'val', '--calibrate',
            '--precision-target', '.9', '--batch-size', str(args.batch_size), '--device', args.device])
    for view in ('mixed', 'clean'):
        view_env = env.copy()
        view_env['FIELDNN_EVAL_VIEW'] = view
        execute(f'synthetic_test_{view}', [str(I8 / 'evaluate.py'), '--data', str(args.synthetic),
                '--checkpoint', str(run / 'best.pt'), '--split', 'test',
                '--batch-size', str(args.batch_size), '--device', args.device,
                '--output', str(run / f'evaluation_test_{view}')], view_env)
    selection = json.loads((run / 'selection.json').read_text(encoding='utf-8'))
    execute('selected_model_artifacts', [str(HERE / 'export_existing.py'), '--data', str(args.data),
            '--predictions', str(run / 'evaluation' / 'predictions.jsonl'),
            '--checkpoint', str(run / 'best.pt'), '--threshold', str(selection['threshold']),
            '--label', args.label,
            '--benchmark', str(run / 'nn_benchmark.json'), '--output', str(args.output / 'deliverables')])
    execute('comprehensive_table_figure', [str(HERE / 'plot_table.py'),
            '--input', str(args.output / 'deliverables' / 'performance.csv'),
            '--output', str(args.output / 'deliverables' / 'performance_table.png')])
    summary.update(status='complete', selection=selection,
                   curves=str(run / 'figures'), deliverables=str(args.output / 'deliverables'),
                   benchmark=str(run / 'nn_benchmark.json'))
    save()
    print(f'Complete: {summary_path}', flush=True)


if __name__ == '__main__':
    main()
