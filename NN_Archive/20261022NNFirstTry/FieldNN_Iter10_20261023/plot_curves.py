"""Plot genuine checkpoint observations; no smoothing or interpolated samples."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / 'output' / 'matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    args = ap.parse_args()
    rows = list(csv.DictReader((args.run / 'history_steps.csv').open(encoding='utf-8')))
    choice = json.loads((args.run / 'selection.json').read_text(encoding='utf-8'))
    config = json.loads((args.run / 'config.json').read_text(encoding='utf-8'))
    architecture = 'Transformer' if config.get('model') == 'transformer' else 'U-Net'
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'],
                         'font.sans-serif': ['SimHei'], 'axes.unicode_minus': False,
                         'xtick.direction': 'in', 'ytick.direction': 'in',
                         'savefig.dpi': 200, 'figure.dpi': 200, 'axes.spines.top': True,
                         'axes.spines.right': True, 'font.size': 11})
    x = [int(r['optimizer_step']) for r in rows]
    target = args.run / 'figures'
    target.mkdir(exist_ok=True)
    for kind, ylabel, train_key, val_key, name in (
        ('loss', 'Known-positive MIL loss', 'train_field_loss', 'val_field_loss', 'field_loss'),
        ('hit', 'Group-mean known-position hit@20 m', 'train_hit20_fixed', 'val_hit20_fixed', 'field_hit20'),
    ):
        fig, ax = plt.subplots(figsize=(7.5, 4.7), layout='constrained')
        for key, label, color in ((train_key, 'Train', '#2368a2'), (val_key, 'Validation', '#c64e35')):
            ax.plot(x, [float(r[key]) for r in rows], label=label, color=color,
                    lw=1.5, marker='o', ms=2.1)
        ax.axvline(choice['dense_best_step'], color='#59636c', ls='--', lw=1.0,
                   label='Best checkpoint in this trajectory')
        ax.set(xlabel='Optimizer updates', ylabel=ylabel,
               title=f'Field {architecture}: positive-only supervision' if kind == 'loss'
               else f'{architecture} field localization (fixed threshold = 0.94)')
        if kind == 'hit':
            ax.set_ylim(-.025, 1.025)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        else:
            ax.set_ylim(bottom=0)
        ax.tick_params(top=True, right=True)
        ax.grid(alpha=.16, lw=.6)
        ax.legend(frameon=False, fontsize=9)
        fig.savefig(target / f'{name}.png')
        fig.savefig(target / f'{name}.pdf')
        plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.7), sharex=True, layout='constrained')
    axes[0].plot(x, [float(r['val_macro_hit20_selected']) for r in rows], color='#2368a2', marker='o', ms=2)
    axes[0].set(ylabel='Validation group hit@20 m', ylim=(-.025, 1.025))
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].plot(x, [float(r['val_candidates_per_curve_selected']) for r in rows], color='#c64e35', marker='o', ms=2)
    axes[1].set(xlabel='Optimizer updates', ylabel='Validation candidates / curve', ylim=(0, None))
    for ax in axes:
        ax.axvline(choice['dense_best_step'], color='#59636c', ls='--', lw=1)
        ax.tick_params(top=True, right=True)
        ax.grid(alpha=.16)
    axes[0].set_title('Validation-calibrated operating point under candidate budget')
    fig.savefig(target / 'validation_operating_point.png')
    plt.close(fig)
    (target / 'CURVE_NOTES.md').write_text(
        f'# 曲线口径\n\n共{len(rows)}个实际观测点，横轴为成功执行的optimizer updates。'
        '保留全部原始记录，无平滑、插值或删点；连线只连接相邻观测。\n\n'
        'Loss为固定现场train/val正例袋同定义MIL loss（eval模式），不是完整分类交叉熵，也不是训练混合总Loss。'
        'Hit为固定0.94阈值、20m容差下按独立组平均的最佳单扫频一对一已知位置命中率；未知候选不算负例。'
        '曲线不含1500m浸水实验或RG58。\n\n'
        '主图阈值不随记录点变化；补图另显示每点只使用val校准的候选预算操作点。'
        f'本轮轨迹最优点为{choice["dense_best_step"]}步。最终选中来源：{choice["selected_source"]}。'
        '如果最终保留Iter8参考权重，本轮曲线仅代表Iter10探索过程，不能称为该Iter8权重的历史训练轨迹。\n', encoding='utf-8')
    print(f'Wrote {target} ({len(rows)} genuine observations)')


if __name__ == '__main__':
    main()
