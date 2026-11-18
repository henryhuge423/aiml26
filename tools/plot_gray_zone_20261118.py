"""Render bilingual figures from the development-only gray-zone audit."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
import numpy as np
import pandas as pd
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'docs/progress/2026-11-18'
LOCAL = ROOT / 'runs/gray_zone_20261118'
TEAL, NAVY, ORANGE, RED, GREY = '#007d76', '#293d6d', '#bd622c', '#aa375b', '#8a9aaa'
METHODS = ['joint', 'a_uncertainty', 'b_uncertainty']
COLORS = [TEAL, NAVY, ORANGE]
COHORTS = ['pooled', 'airogs']


def labels(zh):
    if zh:
        return (['合并开发集', 'AIROGS 开发集'], ['A + 质量灰区', 'A 不确定性', 'B 不确定性'],
                ['正确分类', '漏诊', '误报'])
    return (['Pooled development', 'AIROGS development'], ['A + quality', 'A uncertainty', 'B uncertainty'],
            ['Correct', 'Missed positive', 'False positive'])


def save(fig, name, lang):
    fig.savefig(REPORT / f'{name}.{lang}.png', dpi=200, facecolor='white')
    plt.close(fig)


def plot_boundary(results, lang):
    zh = lang == 'zh'
    cohort_labels, _, outcomes = labels(zh)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=True, layout='constrained')
    for cohort, name, ax in zip(COHORTS, cohort_labels, axes):
        data = pd.read_csv(LOCAL / f'{cohort}-development.csv')
        model = results[cohort]
        threshold = model['validation']['p_a']['threshold']
        cut = model['main']['quality_rule_refit_cut']
        q = np.linspace(0, 1, 400)
        halfwidth = 1 - q - cut
        active = halfwidth >= 0
        lo = expit(logit(threshold) - halfwidth)
        hi = expit(logit(threshold) + halfwidth)
        ax.fill_betweenx(q, lo, hi, where=active, color=TEAL, alpha=.17,
                         label='全开发集拟合的灰区' if zh else 'Gray zone fitted on full cohort')
        ax.plot(lo[active], q[active], color=TEAL, lw=1.25)
        ax.plot(hi[active], q[active], color=TEAL, lw=1.25)
        for outcome, label, color, size, alpha in zip(
                ['correct', 'FN', 'FP'], outcomes, [GREY, RED, ORANGE], [7, 16, 16], [.31, .83, .83]):
            subset = data[data.a_result.eq(outcome)]
            ax.scatter(subset.p_a, subset.q_clipped, color=color, s=size, alpha=alpha,
                       label=f'{label}（{len(subset)}）' if zh else f'{label} ({len(subset)})',
                       linewidths=0)
        ax.axvline(threshold, linestyle=(0, (3, 3)), color='#263443', linewidth=1)
        ax.set(title=f'{name} · n={len(data):,} · t={threshold:.3f}', xlim=(-.025, 1.025), ylim=(-.025, 1.025),
               xlabel='疾病分数 p' if zh else 'Disease score p')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(.2))
        ax.legend(loc='lower left', fontsize=8.4, ncol=1, framealpha=.93)
    axes[0].set_ylabel('预测全图质量 q' if zh else 'Predicted whole-image quality q')
    save(fig, 'gray-zone-map', lang)


def plot_swaps(results, lang):
    zh = lang == 'zh'
    cohort_labels, _, outcomes = labels(zh)
    fig, axes = plt.subplots(1, 2, figsize=(8.9, 2.75), sharex=True, layout='constrained')
    max_images = max(results[key]['main']['swaps']['quality_only']['images'] for key in COHORTS)
    for cohort, name, ax in zip(COHORTS, cohort_labels, axes):
        main = results[cohort]['main']
        swaps = main['swaps']
        left = np.zeros(2)
        for metric, outcome, color in zip(['correct', 'fn', 'fp'], outcomes, [GREY, RED, ORANGE]):
            widths = [swaps[key][metric] for key in ('quality_only', 'uncertainty_only')]
            ax.barh([1, 0], widths, left=left, height=.49, color=color, label=outcome)
            for y, base, width in zip([1, 0], left, widths):
                if width:
                    ax.text(base + width / 2, y, str(width), color='white' if color != GREY else '#162839',
                            fontsize=9, ha='center', va='center')
            left += widths
        ax.set(title=f"{name} · k={main['counts']['joint']['reviewed']}",
               yticks=[1, 0], yticklabels=['质量换入', '质量换出'] if zh else ['Added by quality', 'Displaced by quality'],
               xlim=(0, max_images + 2), xlabel='复核名单中变动的照片数' if zh else 'Images changing review assignment')
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))
    handles, legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend, loc='outside lower center', ncol=3, frameon=False)
    save(fig, 'review-swaps', lang)


def plot_workloads(results, lang):
    zh = lang == 'zh'
    cohort_labels, method_labels, _ = labels(zh)
    fig, axes = plt.subplots(2, 2, figsize=(8.9, 5), sharex=True, layout='constrained')
    for column, (cohort, name) in enumerate(zip(COHORTS, cohort_labels)):
        data = results[cohort]['workloads']
        x = [point['actual'] * 100 for point in data]
        for row, metric in enumerate(('automatic_error_rate', 'residual_miss_rate')):
            ax = axes[row, column]
            ax.axvline(x[2], color=GREY, ls=':', lw=1)
            for method, label, color, marker, style in zip(METHODS, method_labels, COLORS, ['o', 's', '^'], ['-', '--', '-.']):
                y = [point['counts'][method][metric] * 100 for point in data]
                ax.plot(x, y, label=label, color=color, marker=marker, linestyle=style,
                        linewidth=1.65, markersize=4)
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
            ax.xaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
            ax.set_xticks([5, 10, 15, 20])
            ax.grid(axis='y', alpha=.22)
        axes[0, column].set_title(name)
        axes[1, column].set_xlabel('实际复核比例' if zh else 'Observed review share')
    axes[0, 0].set_ylabel('自动分支错误率' if zh else 'Automated error rate')
    axes[1, 0].set_ylabel('仍漏诊 / 全部阳性' if zh else 'Residual misses / all positives')
    handles, legend = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend, loc='outside lower center', ncol=3, frameon=False)
    save(fig, 'review-workload', lang)


def plot_sources(results, lang):
    zh = lang == 'zh'
    main = results['pooled']['main']
    source = main['sources']
    major = {'airogs-light-v2': 'AIROGS', 'DATASET_101/ODIR-5K': 'ODIR-5K',
             'Kaggle-Eye_diseases_classification': 'Kaggle', 'lag_part1': 'LAG', 'keh': 'KEH'}
    names = [*major.values(), '其他来源' if zh else 'Other sources']
    n = [source[k]['joint']['n'] for k in major]
    n.append(main['counts']['joint']['n'] - sum(n))
    contrasts = []
    for key in METHODS[1:]:
        d = [source[k]['joint']['errors'] - source[k][key]['errors'] for k in major]
        d.append(main['counts']['joint']['errors'] - main['counts'][key]['errors'] - sum(d))
        contrasts.append(d)
    fig, ax = plt.subplots(figsize=(8.9, 3.45), layout='constrained')
    y = np.arange(len(names))
    axes_labels = ['相对 A 不确定性', '相对 B 不确定性'] if zh else ['vs A uncertainty', 'vs B uncertainty']
    for difference, offset, color, label in zip(contrasts, [-.17, .17], [TEAL, ORANGE], axes_labels):
        bars = ax.barh(y + offset, difference, height=.29, label=label, color=color)
        ax.bar_label(bars, labels=[f'{v:+d}' if v else '0' for v in difference], padding=4, fontsize=8.6)
    ax.axvline(0, color='#42546c', linewidth=1, zorder=0)
    ax.set(yticks=y, yticklabels=[f'{name} (n={count})' for name, count in zip(names, n)],
           xlim=(min(min(d) for d in contrasts) - 8, max(max(d) for d in contrasts) + 9),
           xlabel='质量灰区改变的剩余错误数（负数表示减少）' if zh else
                  'Difference in residual errors with quality review (negative = fewer)')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=.17)
    ax.legend(loc='lower right', frameon=False)
    save(fig, 'review-by-source', lang)


def main():
    results = json.loads((REPORT / 'gray-zone-results.json').read_text(encoding='utf-8'))['cohorts']
    font = Path('C:/Windows/Fonts/msyh.ttc')
    font_manager.fontManager.addfont(font)
    chinese = font_manager.FontProperties(fname=font).get_name()
    for lang in ('en', 'zh'):
        with plt.rc_context({'font.family': chinese if lang == 'zh' else 'DejaVu Sans',
                             'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False}):
            plot_boundary(results, lang)
            plot_swaps(results, lang)
            plot_workloads(results, lang)
            plot_sources(results, lang)


if __name__ == '__main__':
    main()
