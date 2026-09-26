"""Publication figure entry point: main-text panel integration, September 2026.

Reuse original numerical builders and cached out-of-fold results. No fitting,
selection of observations, or change to the fixed material-pair definitions.
"""
import argparse
import numpy as np
import pandas as pd
import plot_full24_revision as base


def capture(function):
    records = []
    previous = base.render
    def record(n, stem, builders, rows, cols, size, tables, **kwargs):
        records.append(dict(n=n, stem=stem, builders=builders, rows=rows,
                            cols=cols, size=size, tables=tables, kwargs=kwargs))
    base.render = record
    try:
        function()
    finally:
        base.render = previous
    assert len(records) == 1
    return records[0]


def fig1():
    original, descriptors = capture(base.fig1), capture(base.figs2)
    base.render('Fig01', 'data_and_comparisons',
                original['builders'] + descriptors['builders'][2:], 3, 2,
                (11.4, 14.3), {**original['tables'],
                              'material_profiles': descriptors['tables']['profiles']})


def fig6():
    original, pairs = capture(base.fig6), capture(base.figs4)
    # Two strong modification contrasts and the near-equivalent P8 comparison.
    # Remaining pairs are retained in Fig. S3; all eight remain in Fig. 6a,b.
    def compact(builder):
        def draw(ax):
            builder(ax)
            labels = {'Removal advantage of material a (pp)': 'Removal advantage (pp)',
                      'Contrast MAE minus history (pp)': 'Contrast MAE\nminus History (pp)',
                      'Selection loss minus history (pp)': 'Selection loss\nminus History (pp)'}
            if ax.get_xlabel() in labels:
                ax.set_xlabel(labels[ax.get_xlabel()])
            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_fontsize(12.5)
            if ax.images:
                for tick in ax.get_xticklabels():
                    tick.set_fontsize(11.5)
            if ax.get_xlabel().startswith('Selection loss\nminus'):
                ax.set_xticks([0, .01, .02], ['0.00', '0.01', '0.02'])
            for child in ax.child_axes:
                if child.get_xlabel() == 'Observed material advantage (pp)':
                    child.set_xlabel('Removal advantage (pp)')
                elif child.get_xlabel() == 'Joint descriptor change / range':
                    child.set_xlabel('Descriptor change / range')
        return draw
    base.render('Fig06', 'material_comparisons',
                [compact(b) for b in original['builders'] + [pairs['builders'][i] for i in [3, 4, 7]]],
                3, 3, (14.4, 13.2),
                {**original['tables'], 'paired_predictions': pairs['tables']['paired_predictions']})


def figs1():
    original, descriptors = capture(base.figs1), capture(base.figs2)
    base.render('FigS01', 'data_checks',
                original['builders'] + descriptors['builders'][:2], 3, 2,
                (11.4, 14.0), {**original['tables'],
                              'rank_diagnostic': descriptors['tables']['rank_diagnostic']})


def figs2():
    original = capture(base.figs3)
    base.render('FigS02', 'complete_model_diagnostics', original['builders'],
                original['rows'], original['cols'], original['size'],
                original['tables'], **original['kwargs'])


def figs3():
    original = capture(base.figs4)
    summaries = base.read('material_pair_summary.csv')
    a = summaries[(summaries.track == 'broad') &
                  summaries.strategy.isin(['History', 'TOPSIS'])]
    differences = a.pivot(index=['pair', 'seed'], columns='strategy', values='error_pp')
    differences['TOPSIS_minus_History_pp'] = differences.TOPSIS - differences.History
    differences = differences.reset_index()
    def errors(ax):
        g = differences.groupby('pair').TOPSIS_minus_History_pp.agg(['min', 'median', 'max']).reindex(base.PAIR)
        ax.axvline(0, color=base.INK, lw=.7, ls='--')
        ax.hlines(np.arange(8), g['min'], g['max'], color=base.TEAL, lw=1.8)
        ax.scatter(g['median'], np.arange(8), c=base.TEAL, s=42,
                   edgecolors=base.INK, lw=.45)
        ax.set_yticks(range(8), [f'P{i+1}' for i in range(8)])
        ax.invert_yaxis()
        base.axis(ax, 'TOPSIS − History MAE (pp)', ticks=11)
        ax.set_title('All eight material pairs', fontsize=12, pad=12)
        ax.xaxis.set_major_locator(base.MaxNLocator(4))
    base.render('FigS03', 'remaining_material_pair_predictions',
                [original['builders'][i] for i in [0, 1, 2, 5, 6]] + [errors],
                3, 2, (11.4, 13.3),
                {**original['tables'], 'paired_error_differences': differences})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', default='1,2,3,4,5,6,8,s1,s2,s3')
    args = parser.parse_args()
    for key in args.only.split(','):
        if key in ['1', '6', 's1', 's2', 's3']:
            globals()['fig' + key]()
        elif key == '5':
            from plot_topsis_surfaces import main as plot_five
            plot_five()
        else:
            getattr(base, 'fig' + key)()
