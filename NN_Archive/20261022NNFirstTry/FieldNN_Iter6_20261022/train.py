"""Iter6: synthetic replay plus group-isolated, positive-only field MIL adaptation."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ITER3 = HERE.parent / 'FieldNN_Iter3_20261022'
ITER5 = HERE.parent / "FieldNN_Iter5_20261022"
sys.path.extend([str(ITER5), str(ITER3)])
from data import ShardDataset
from losses import compute_losses, total_loss
from models import build_model


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def worker_init(_):
    torch.set_num_threads(1)
    np.random.seed(torch.initial_seed() % 2**32)


class FieldBags:
    def __init__(self, root, split, device):
        root = Path(root)
        self.features = np.load(root / 'field_features.npy', mmap_mode='r')
        self.records = [json.loads(line) for line in (root / 'field_records.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        self.records = {int(row['index']): row for row in self.records}
        self.bags = [b for b in json.loads((root / 'field_bags.json').read_text(encoding='utf-8'))['bags'] if b['split'] == split and not b.get('known_lab', False)]
        self.device = device

    def batch(self, bag):
        indices = bag['indices']
        x = torch.from_numpy(np.array(self.features[indices], copy=True)).to(self.device)
        return x

    def windows(self, bag, length, max_m):
        d = np.linspace(0, max_m, length)
        windows = []
        for position, source_window in zip(bag['positions_source_m'], bag['position_windows_source_m']):
            spans = []
            for row, index in enumerate(bag['indices']):
                ratio = float(self.records[index]['source_scale_ratio'])
                low = int(np.searchsorted(d, float(source_window[0]) / ratio))
                high = min(length, int(np.searchsorted(d, float(source_window[1]) / ratio, side='right')))
                if high > low:
                    spans.append((row, low, high))
            if spans:
                windows.append(spans)
        return windows


def field_objective(logits, teacher_logits, windows):
    """One known positive per bag-position; all other positions stay unlabelled.

    A position can occur in any applicable phase/sweep and within the documented
    coordinate tolerance. Max pooling does not declare every phase positive.
    The frozen Iter5 teacher is a soft preservation target, never negative truth.
    """
    z = logits.float()
    losses = []
    unknown = torch.ones_like(z, dtype=torch.bool)
    for spans in windows:
        local = torch.cat([z[row, 0, low:high] for row, low, high in spans])
        losses.append(F.softplus(-local.max()))
        for row, low, high in spans:
            pad = high - low
            unknown[row, 0, max(0, low - pad):min(z.shape[-1], high + pad)] = False
    positive = torch.stack(losses).mean() if losses else z.sum() * 0
    if teacher_logits is None:
        return positive, z.sum() * 0
    q = teacher_logits.float().sigmoid().detach()
    # KL to the teacher preserves unlabelled responses; upweight existing peaks
    # so 8192 background bins do not drown out sparse candidate preservation.
    kl = F.binary_cross_entropy_with_logits(z, q, reduction='none') - F.binary_cross_entropy_with_logits(teacher_logits.float(), q, reduction='none')
    weight = (1 + 20 * q) * unknown
    preservation = (kl * weight).sum() / weight.sum().clamp_min(1)
    return positive, preservation


@torch.no_grad()
def validate(model, loader, field, device, amp, cfg, max_batches=0):
    model.eval()
    total, count = 0., 0
    for step, (x, y, _) in enumerate(tqdm(loader, desc='synthetic val', leave=False)):
        with torch.autocast(device_type=device.type, enabled=amp):
            out = model(x.to(device, non_blocking=True))
        loss = total_loss(compute_losses(out, y.to(device, non_blocking=True)))
        total += float(loss) * len(x)
        count += len(x)
        if max_batches and step + 1 >= max_batches:
            break
    weak = []
    bags = field.bags[:2] if max_batches else field.bags
    for bag in bags:
        x = field.batch(bag)
        with torch.autocast(device_type=device.type, enabled=amp):
            out = model(x)
        windows = field.windows(bag, out.shape[-1], cfg['response']['distance_max_m'])
        positive, _ = field_objective(out, None, windows)
        weak.append(float(positive))
    return total / count, float(np.mean(weak))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True, help='Iter6 field cache')
    ap.add_argument('--synthetic', type=Path, default=Path(r'E:\CableNNDataSet\CodeX\FieldNN_Iter4_20261022'))
    ap.add_argument('--model', choices=['unet', 'transformer'], required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--epochs', type=int, default=18)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--max-train-batches', type=int, default=0)
    ap.add_argument('--max-val-batches', type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    seed = 2026102206
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(args.device)
    amp = device.type == 'cuda'
    source = ITER5 / 'output' / 'iter5' / args.model / 'best.pt'
    initial = torch.load(source, map_location='cpu', weights_only=False)
    cfg = copy.deepcopy(initial['config'])
    cfg['seed'] = seed
    cfg['training'].update(epochs=args.epochs, batch_size=args.batch_size, num_workers=args.workers, learning_rate=3e-5)
    cfg['adaptation6'] = {'weak_weight': 0.5, 'teacher_weight': 0.5, 'field_every_steps': 2,
                          'positive_window_source_m': 'per-bag evidence: engineer 20m, TDR-derived 40m, wet-lab 10m',
                          'field_supervision': 'positive-only cross-sweep MIL',
                          'selection': 'normalized synthetic val loss + 0.5 * normalized field-val MIL loss',
                          'source_checkpoint': str(source), 'data': str(args.data.resolve())}
    run = args.output / args.model
    run.mkdir(parents=True, exist_ok=True)
    if (run / 'last.pt').exists() and not args.resume:
        raise FileExistsError(f'Use --resume or a new output: {run}')
    write_json(run / 'config.json', cfg)
    model = build_model(args.model, **cfg['model']).to(device)
    model.load_state_dict(initial['model'])
    teacher = copy.deepcopy(model).eval()
    teacher.requires_grad_(False)
    train_data = ShardDataset(args.synthetic, 'train', train=True)
    val_data = ShardDataset(args.synthetic, 'val')
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=amp,
                              persistent_workers=args.workers > 0, worker_init_fn=worker_init)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=0, pin_memory=amp)
    field_train = FieldBags(args.data, 'train', device)
    field_val = FieldBags(args.data, 'val', device)
    if not field_train.bags or not field_val.bags:
        raise ValueError('Need nonempty, group-isolated field train and validation bags')
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024., enabled=amp)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=.5, min_lr=2e-6)
    start, best, stale = 0, float('inf'), 0
    base_synth, base_field = validate(model, val_loader, field_val, device, amp, cfg, args.max_val_batches)
    normalization = (max(base_synth, 1e-6), max(base_field, 1e-6))
    def state(epoch):
        return {'model_name': args.model, 'config': cfg, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(), 'scheduler': scheduler.state_dict(),
                'epoch': epoch, 'best_score': best, 'stale': stale, 'normalization': normalization,
                'rng': {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
                        'cuda': torch.cuda.get_rng_state() if amp else None}}
    if args.resume:
        saved = torch.load(run / 'last.pt', map_location='cpu', weights_only=False)
        if saved['config']['adaptation6']['data'] != str(args.data.resolve()):
            raise ValueError('Resume data differs from the previous run')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler']); scheduler.load_state_dict(saved['scheduler'])
        start, best, stale = saved['epoch'] + 1, saved['best_score'], saved['stale']
        normalization = saved['normalization']
        random.setstate(saved['rng']['python']); np.random.set_state(saved['rng']['numpy'])
        torch.set_rng_state(saved['rng']['torch'])
        if amp:
            torch.cuda.set_rng_state(saved['rng']['cuda'])
    else:
        best = 1.5
        torch.save(state(-1), run / 'best.pt')
        write_json(run / 'initial_validation.json', {'synthetic': base_synth, 'field_positive': base_field,
                   'selection_score': best, 'policy': 'Initial Iter5 weights remain eligible; no forced promotion'})
    write_json(run / 'run_info.json', {'model': args.model, 'parameters': sum(p.numel() for p in model.parameters()),
               'field_train_bags': len(field_train.bags), 'field_val_bags': len(field_val.bags),
               'synthetic_train': len(train_data), 'training': cfg['training'], 'adaptation': cfg['adaptation6'],
               'gpu': torch.cuda.get_device_name() if amp else None, 'torch': torch.__version__})
    for epoch in range(start, args.epochs):
        begun = time.perf_counter(); model.train()
        sums = dict(synthetic=0., positive=0., preservation=0.)
        count = field_count = skipped = 0
        if amp:
            torch.cuda.reset_peak_memory_stats()
        bag_order = np.random.permutation(len(field_train.bags)).tolist()
        progress = tqdm(train_loader, desc=f'{args.model} {epoch+1}/{args.epochs}', unit='batch')
        for step, (x, y, _) in enumerate(progress):
            optimizer.zero_grad(set_to_none=True)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(x)
            synthetic_loss = total_loss(compute_losses(logits, y))
            scaler.scale(synthetic_loss).backward()
            sums['synthetic'] += float(synthetic_loss.detach()) * len(x); count += len(x)
            if step % 2 == 0:
                if not bag_order:
                    bag_order = np.random.permutation(len(field_train.bags)).tolist()
                bag = field_train.bags[bag_order.pop()]
                field_x = field_train.batch(bag)
                with torch.no_grad(), torch.autocast(device_type=device.type, enabled=amp):
                    teacher_out = teacher(field_x)
                with torch.autocast(device_type=device.type, enabled=amp):
                    field_out = model(field_x)
                windows = field_train.windows(bag, field_out.shape[-1], cfg['response']['distance_max_m'])
                positive, preservation = field_objective(field_out, teacher_out, windows)
                scaler.scale(.5 * positive + .5 * preservation).backward()
                sums['positive'] += float(positive.detach()); sums['preservation'] += float(preservation.detach()); field_count += 1
            scaler.unscale_(optimizer)
            grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
            old_scale = scaler.get_scale(); scaler.step(optimizer); scaler.update()
            skipped += int(scaler.get_scale() < old_scale)
            progress.set_postfix(syn=f'{float(synthetic_loss.detach()):.3f}', weak=f'{sums["positive"]/max(field_count,1):.3f}')
            if args.max_train_batches and step + 1 >= args.max_train_batches:
                break
        val_synth, val_field = validate(model, val_loader, field_val, device, amp, cfg, args.max_val_batches)
        score = val_synth / normalization[0] + .5 * val_field / normalization[1]
        improved = score < best
        best, stale = (score, 0) if improved else (best, stale + 1)
        scheduler.step(score)
        row = {'epoch': epoch+1, 'train_synthetic': sums['synthetic']/count,
               'train_field_positive': sums['positive']/max(field_count,1),
               'train_teacher_kl': sums['preservation']/max(field_count,1),
               'val_synthetic': val_synth, 'val_field_positive': val_field, 'selection_score': score,
               'best_score': best, 'grad_norm_last': grad, 'amp_skipped_steps': skipped,
               'lr': optimizer.param_groups[0]['lr'], 'seconds': time.perf_counter()-begun,
               'synthetic_samples_per_second': count/(time.perf_counter()-begun),
               'peak_gpu_gib': torch.cuda.max_memory_allocated()/2**30 if amp else 0.}
        torch.save(state(epoch), run / 'last.pt')
        if improved:
            torch.save(state(epoch), run / 'best.pt')
        history = run / 'history.csv'
        exists = history.exists()
        with history.open('a', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=row)
            if not exists: writer.writeheader()
            writer.writerow(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if stale >= 5:
            print('Early stop: 5 epochs without validation improvement', flush=True)
            break


if __name__ == '__main__':
    main()
