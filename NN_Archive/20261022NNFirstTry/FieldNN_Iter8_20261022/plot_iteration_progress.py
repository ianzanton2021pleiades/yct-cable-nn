"""Generate an inspectable figure of existing field results, without inference."""
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
os.environ['MPLCONFIGDIR'] = str(HERE / 'output' / 'progress_plot_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    files = [
        HERE.parent / 'FieldNN_Iter6_20261022' / 'output' / 'density_analysis' / 'field_density_analysis.json',
        HERE.parent / 'FieldNN_Iter7_20261022' / 'output' / 'iter7' / 'density_analysis.json',
        HERE / 'output' / 'iter8' / 'density_analysis.json',
    ]
    reports = [read(p) for p in files]
    values = {}
    for model in ('unet', 'transformer'):
        base = reports[-1]['test_after_selection'][model]['populations']['other_test_groups']['baseline_at_locked_threshold']
        rows = [base] + [r['test_after_selection'][model]['populations']['other_test_groups']['candidate_validation_selected'] for r in reports]
        values[model] = {'hit': [r['positive_labels']['group_macro_hit_rate'] * 100 for r in rows],
                         'density': [r['density']['mean_candidates_per_predicted_curve'] for r in rows]}
    plt.rcParams.update({'font.family': ['Times New Roman', 'SimHei'], 'axes.unicode_minus': False,
                         'xtick.direction': 'in', 'ytick.direction': 'in', 'font.size': 11})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), dpi=200)
    x = np.arange(4)
    colors = {'unet': '#19677E', 'transformer': '#B66A36'}
    for model, label in (('unet', 'U-Net'), ('transformer', 'Transformer')):
        for ax, metric in zip(axes, ('hit', 'density')):
            ax.plot(x, values[model][metric], 'o-', lw=1.8, color=colors[model], label=label)
            for i, value in enumerate(values[model][metric]):
                ax.annotate(f'{value:.1f}' if metric == 'hit' else f'{value:.2f}',
                            (i, value), xytext=(0, 8 if model == 'unet' else -16),
                            textcoords='offset points', ha='center', fontsize=9, color=colors[model])
    axes[0].set_ylabel('Known-position hit @20 m (%)\nMacro mean across 6 cable groups')
    axes[0].set_ylim(0, 65)
    axes[1].set_ylabel('Unverified event candidates per curve\nSame 31 annotated test sweeps')
    axes[1].set_ylim(0, 3.8)
    for ax in axes:
        ax.set_xticks(x, ['Iter5', 'Iter6', 'Iter7', 'Iter8'])
        ax.set_xlim(-.3, 3.3)
        ax.grid(axis='y', alpha=.2)
        ax.legend(frameon=False, loc='lower left')
        ax.tick_params(top=True, right=True)
    fig.suptitle('FieldNN: held-out positive evidence and candidate burden', fontsize=14)
    fig.text(.5, .025, 'Thresholds selected on validation only under each Iter5 candidate budget.\n'
             'Partial engineer annotations; these are not field precision/recall. Historical development corpus.',
             ha='center', va='bottom', fontsize=9)
    fig.tight_layout(rect=[0, .14, 1, .93])
    output = ROOT / 'Image' / '20261022_FieldNN_Iter8_progress.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
