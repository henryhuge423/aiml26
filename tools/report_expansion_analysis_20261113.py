"""Render the deeper paired analysis as bilingual Markdown, LaTeX and PDF."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.analyze_expansion_results_20261113 import ARMS, SOURCES, OUT, RUN, read_csv, read_json
BUILD=ROOT/'.build/progress-20261113-analysis'
LABELS={'zh':['原数据','扩展数据','30%退化','50%退化'],
        'en':['Original','Expanded','30% degradation','50% degradation']}
COLORS=['#34495e','#2874a6','#229954','#d68910']


def fmt_interval(value,percent=False):
    scale=100 if percent else 1
    digits=2 if percent else 4
    point=value['difference']*scale;lo,hi=[x*scale for x in value['interval_95']]
    return f'{point:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]'


def charts(r):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import roc_curve

    final = r['checkpoints']['budget_end']
    quality = final['quality']
    for lang in ('zh', 'en'):
        zh = lang == 'zh'
        plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                             'axes.spines.right': False,
                             'font.family': 'Microsoft YaHei' if zh else 'Arial',
                             'axes.unicode_minus': False})
        names = LABELS[lang]
        def save(fig, stem):
            fig.tight_layout()
            fig.savefig(OUT / f'{stem}.{lang}.png', dpi=200)
            plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.4))
        titles = ['扩展数据－原数据', '50%退化－扩展无退化'] if zh else [
            'Expanded minus original data', '50% degradation minus expanded clean']
        for ax, contrast, title in zip(axes, [
                'expanded_clean minus original_clean',
                'expanded_degradation50 minus expanded_clean'], titles):
            values = [final['sources'][s]['contrasts'][contrast]['auc'] for s in SOURCES]
            points = np.array([v['difference'] for v in values])
            ci = np.array([v['interval_95'] for v in values])
            y = np.arange(len(SOURCES))[::-1]
            ax.errorbar(points, y, xerr=[points-ci[:, 0], ci[:, 1]-points],
                        fmt='o', capsize=3, color=COLORS[1])
            ax.set_yticks(y, list(SOURCES.values()))
            ax.axvline(0, color='gray', ls='--', lw=1)
            ax.set(xlabel='开发集 AUROC 差' if zh else 'Development AUROC difference',
                   title=title)
            ax.grid(axis='x', alpha=.2)
        axes[0].set_xlim(-.15, .39)
        axes[1].set_xlim(-.09, .13)
        save(fig, 'source-differences')

        fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.4))
        exposure = r['training_exposure']
        orig = [exposure[a]['strata']['origin/formal_v3']['draws'] for a in ARMS]
        added = [exposure[a]['strata'].get('origin/expansion_20261113', {}).get('draws', 0)
                 for a in ARMS]
        x = np.arange(4)
        axes[0].bar(x, orig, label='原数据图像' if zh else 'Original images', color=COLORS[0])
        axes[0].bar(x, added, bottom=orig, label='新增图像' if zh else 'Added images', color=COLORS[1])
        axes[0].set_xticks(x, ['原数据', '扩展', '30%退化', '50%退化'] if zh else
                           ['Original', 'Expanded', '30%', '50%'])
        axes[0].set(ylabel='图像采样次数' if zh else 'Image presentations',
                    title='1,200 次更新内的采样次数' if zh else 'Presentations over 1,200 updates',
                    ylim=(0, 22500))
        for i, (o, a) in enumerate(zip(orig, added)):
            axes[0].text(i, o/2, f'{o:,}', ha='center', va='center', color='white')
            if a:
                axes[0].text(i, o+a/2, f'{a:,}', ha='center', va='center', color='white')
        axes[0].legend(fontsize=8, loc='upper left', ncol=2)
        sel = r['selection']['expanded_clean']['views']
        labels = ['本轮 11 来源均值', '验证图像至少 20 张的 7 来源', '原 3 来源均值'] if zh else [
            'Current 11-source mean', 'Seven sources with at least 20 images', 'Original three-source mean']
        for name, label, color in zip(['declared_11', 'sources_at_least_20', 'original_3'],
                                      labels, [COLORS[0], COLORS[1], COLORS[3]]):
            curve = sel[name]['curve']
            axes[1].plot([v['update'] for v in curve], [v['value'] for v in curve],
                         marker='o', label=label, color=color)
        axes[1].set(xlabel='有效优化更新数' if zh else 'Successful optimizer updates',
                    ylabel='验证集来源 AUROC 均值' if zh else 'Mean source validation AUROC',
                    title='扩展无退化：验证来源选择的影响' if zh else 'Expanded clean: source selection matters',
                    ylim=(.89, .97))
        axes[1].set_xticks([300, 600, 900, 1200])
        axes[1].grid(alpha=.2)
        axes[1].legend(fontsize=7.5, loc='lower left')
        save(fig, 'exposure-selection')

        fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6))
        for a, label, color in zip(ARMS, names, COLORS):
            rows = [v for v in read_csv(RUN/a/'last_development_predictions.csv')
                    if v['condition'] == 'defocus_2_at512']
            fpr, tpr, _ = roc_curve([int(v['label']) for v in rows],
                                    [float(v['prediction']) for v in rows])
            auc = final['stress']['conditions']['defocus_2_at512']['points'][a]['auc']
            axes[0].plot(fpr, tpr, color=color, label=f'{label}: {auc:.3f}')
        axes[0].set(xlabel='误报率' if zh else 'False-positive rate',
                    ylabel='敏感度' if zh else 'Sensitivity',
                    title='同一 384 张开发原图的失焦变体' if zh else 'Defocus of the same 384 development images',
                    xlim=(0, .35), ylim=(.2, 1.01))
        axes[0].axvline(.05, color='gray', ls='--', lw=1)
        axes[0].legend(fontsize=8, loc='lower right')
        axes[0].grid(alpha=.2)
        width = .32
        labels = ['阴性图像（238 张）', '阳性图像（146 张）'] if zh else [
            'Negative images (n=238)', 'Positive images (n=146)']
        for i, label in enumerate(labels):
            values = [final['stress']['score_shifts']['defocus_2_at512'][a][str(i)] for a in ARMS]
            means = np.array([v['mean_change'] for v in values])
            ci = np.array([v['interval_95'] for v in values])
            axes[1].bar(x+(i-.5)*width, means, width, label=label,
                        color=[COLORS[1], COLORS[3]][i],
                        yerr=[means-ci[:, 0], ci[:, 1]-means], capsize=3)
        axes[1].set_xticks(x, ['原数据', '扩展', '30%退化', '50%退化'] if zh else
                           ['Original', 'Expanded', '30%', '50%'])
        axes[1].axhline(0, color='gray', lw=1)
        axes[1].set(ylabel='平均预测概率差：失焦－原图' if zh else 'Mean probability change: defocus minus original',
                    title='阳性与阴性图像的分数变化' if zh else 'Probability changes by disease label',
                    ylim=(-.23, .33))
        axes[1].legend(fontsize=8, loc='upper right')
        axes[1].grid(axis='y', alpha=.2)
        save(fig, 'defocus-ranking-shifts')

        fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7))
        strata = quality['manifest_quality_strata']['train']
        categories = [('fives', 'poor'), ('fives', 'other'), ('mshf', 'poor'), ('mshf', 'other')]
        positive = np.array([strata[s][q]['positive'] for s, q in categories])
        negative = np.array([strata[s][q]['negative'] for s, q in categories])
        ticks = ['FIVES\n低质量', 'FIVES\n其他质量', 'MSHF\n低质量', 'MSHF\n其他质量'] if zh else [
            'FIVES\nPoor', 'FIVES\nOther', 'MSHF\nPoor', 'MSHF\nOther']
        axes[0].bar(x, positive, color=COLORS[3], label='阳性' if zh else 'Positive')
        axes[0].bar(x, negative, bottom=positive, color=COLORS[1], label='阴性' if zh else 'Negative')
        for i, (p, n) in enumerate(zip(positive, negative)):
            axes[0].text(i, p+n+2, f'{p}+{n}', ha='center', fontsize=8)
        axes[0].set_xticks(x, ticks)
        axes[0].set(ylabel='训练图像数' if zh else 'Training images', ylim=(0, 155),
                    title='真实质量与疾病标签的分布' if zh else 'Real quality and disease labels')
        axes[0].legend(fontsize=8, loc='upper right')
        axes[0].grid(axis='y', alpha=.2)
        for a, label, color in zip(ARMS, names, COLORS):
            values = [quality['development_sensitivity'][a][s] for s in ('fives', 'mshf')]
            points = np.array([v['sensitivity'] for v in values]) * 100
            ci = np.array([v['exact_binomial_95'] for v in values]) * 100
            offset = (ARMS.index(a)-1.5)*.13
            axes[1].errorbar(points, np.array([1, 0])+offset,
                             xerr=[points-ci[:, 0], ci[:, 1]-points],
                             fmt='o', capsize=3, color=color, label=label)
        axes[1].set_yticks([1, 0], ['FIVES (n=7)', 'MSHF (n=5)'])
        axes[1].set(xlabel='检出比例（%），阈值 0.5' if zh else 'Detected positives (%), threshold 0.5',
                    title='真实低质量开发阳性：95% 区间' if zh else 'Real poor-quality positives: 95% intervals',
                    xlim=(0, 106), ylim=(-.4, 1.6))
        axes[1].legend(fontsize=7.5, loc='upper left')
        axes[1].grid(axis='x', alpha=.2)
        save(fig, 'real-quality-labels')
    for stem in ('source-differences', 'exposure-selection', 'defocus-ranking-shifts'):
        old = OUT / f'{stem}.png'
        if old.is_file():
            old.unlink()
    (OUT / 'figure-data.json').write_text(json.dumps({
        'analysis': 'analysis.json', 'checkpoint': 'budget_end',
        'real_quality_training_counts': quality['manifest_quality_strata']['train'],
        'real_quality_development_sensitivity': quality['development_sensitivity'],
        'figures': [f'{stem}.{lang}.png' for stem in ('source-differences', 'exposure-selection',
                   'defocus-ranking-shifts', 'real-quality-labels') for lang in ('zh', 'en')]
    }, indent=2), encoding='utf-8')


def tables(r,lang):
    final=r['checkpoints']['budget_end'];names=LABELS[lang]
    source=['| 来源：图像／正例／关联组 | 扩展－原数据：AUROC差及95%区间 | 50%退化－扩展：AUROC差及95%区间 |','|---|---:|---:|'] if lang=='zh' else ['| Source: images / positives / groups | Expansion − original: AUROC difference [95% interval] | 50% − expanded: AUROC difference [95% interval] |','|---|---:|---:|']
    for s,label in SOURCES.items():
        v=final['sources'][s];p=v['points'][ARMS[0]]
        source.append(f'| {label}: {p["n"]}/{p["positives"]}/{p["linked_groups"]} | {fmt_interval(v["contrasts"]["expanded_clean minus original_clean"]["auc"])} | {fmt_interval(v["contrasts"]["expanded_degradation50 minus expanded_clean"]["auc"])} |')
    stress=['| 条件／指标 | 原数据 | 扩展数据 | 30%退化 | 50%退化 |','|--------------------|--------:|--------:|--------:|--------:|'] if lang=='zh' else ['| Condition / measure | Original | Expanded | 30% | 50% |','|--------------------|--------:|--------:|--------:|--------:|']
    for condition,label in [('original','原图' if lang=='zh' else 'Original'),('defocus_2_at512','失焦' if lang=='zh' else 'Defocus'),('exposure_070','曝光0.70' if lang=='zh' else 'Exposure 0.70')]:
        v=final['stress']['conditions'][condition]
        stress.append(f'| {label}: AUROC | '+' | '.join(f'{v["points"][a]["auc"]:.4f}' for a in ARMS)+' |')
        stress.append(f'| {label}: '+('漏诊／误报' if lang=='zh' else 'misses / false positives')+' | '+' | '.join(f'{v["points"][a]["false_negatives"]}/{v["points"][a]["false_positives"]}' for a in ARMS)+' |')
    exposure=['| 方案 | 原数据采样次数／不同原图 | 新增数据采样次数／不同原图 |','|---|---:|---:|'] if lang=='zh' else ['| Arm | Original presentations / distinct images | Added presentations / distinct images |','|---|---:|---:|']
    for name,a in zip(names,ARMS):
        v=r['training_exposure'][a]['strata'];o=v['origin/formal_v3'];new=v.get('origin/expansion_20261113')
        exposure.append(f'| {name} | {o["draws"]:,}/{o["unique_images"]:,} | '+(f'{new["draws"]:,}/{new["unique_images"]:,}' if new else '0/0')+' |')
    diagnostic=['| 第1,200次更新，扩展－原数据 | AUPRC差及95%区间 | Brier差及95%区间 |','|---|---:|---:|'] if lang=='zh' else ['| Update 1,200: expansion − original | AUPRC difference [95% interval] | Brier difference [95% interval] |','|---|---:|---:|']
    for s in ('airogs-light-v2','DATASET_101/ODIR-5K','lag_part1','keh'):
        v=final['sources'][s]['contrasts']['expanded_clean minus original_clean']
        diagnostic.append(f'| {SOURCES[s]} | {fmt_interval(v["auprc"])} | {fmt_interval(v["brier"])} |')
    operating=['| 条件：验证阈值固定 | 原数据 | 扩展数据 | 30%退化 | 50%退化 |','|--------------------|--------:|--------:|--------:|--------:|'] if lang=='zh' else ['| Condition: fixed validation thresholds | Original | Expanded | 30% | 50% |','|--------------------|--------:|--------:|--------:|--------:|']
    for condition,label in [('original','原图' if lang=='zh' else 'Original'),('defocus_2_at512','失焦' if lang=='zh' else 'Defocus'),('exposure_070','曝光0.70' if lang=='zh' else 'Exposure 0.70')]:
        cells=[]
        for a in ARMS:
            v=final['stress']['validation_threshold_operating'][condition][a]
            cells.append(f'{sum(x["false_negatives"] for x in v.values())}/{sum(x["false_positives"] for x in v.values())}')
        operating.append('| '+label+' | '+' | '.join(cells)+' |')
    return ['\n'.join(t) for t in (source,diagnostic,stress,exposure,operating)]


def reports(r):
    for lang in ('zh', 'en'):
        source, diagnostic, stress, exposure, operating = tables(r, lang)
        if lang == 'zh':
            text = f'''# 数据扩展与退化增强：实验结果分析

2026年11月13日 · [English](experimental-analysis.en.md) · [中文 PDF](experimental-analysis.zh.pdf)

扩展数据提高了 KEH 和 LAG 来源的青光眼图像排序表现，AIROGS 和 ODIR 来源的表现下降。50% 退化增强提高了合成失焦图像的排序表现，并减少误报，同时增加部分阳性漏诊。12 张真实低质量开发阳性图像在三个扩展方案下均被检出，退化增强在这组图像上没有额外检出。

本文承接同日的[数据与方法审查](../data-review/data-and-degradation-review.zh.md)和[训练结果报告](../training-results/training-results.zh.md)，进一步分析来源差异、实际采样次数、权重选择以及真实低质量图像的标签分布。这些分析为下一轮控制实验提供依据。

## 数据、模型与比较方法

四个方案均使用 EfficientNet-B0、相同初始化和随机种子 20261113，各完成 1,200 次有效优化更新。原数据方案使用 10,338 张训练图像；三个扩展方案使用 15,480 张。30% 和 50% 表示采样图像应用混合退化增强的概率。阳性保留各来源的定义，包括青光眼诊断和 AIROGS 的青光眼转诊标签，因此结果按来源分析。模型验证集含 2,046 张图像，其中阳性 823 张；规则开发集含 2,053 张，其中阳性 814 张。模型验证集用于选择权重和设定来源阈值，规则开发集用于拟定人工复核规则及本文的结果分析。最终测试将在模型和复核规则确定后执行。

训练报告记录的是验证指标最高时选中的权重：原数据、30% 退化和 50% 退化均选中第 1,200 次更新，扩展无退化选中第 300 次更新。本文主要使用四组第 1,200 次更新权重，使比较具有相同更新预算；选中权重的结果另存于[分析数据](analysis.json)。一轮训练表示完整遍历训练集，优化更新表示处理一个批次后调整模型参数，本轮预算按更新次数设定。

95% 区间通过 4,000 次配对自助重采样计算：每次抽取完整关联组，同一批组同时用于比较两个模型。关联组将已知双眼、同眼重拍或其他关联图像放在一起，以保留图像间的依赖关系；指标按图像计算。区间反映开发图像抽样的不确定性，本轮单个训练种子的随机性需由后续多种子实验评估。区间均为单项百分位区间，多来源比较采用逐项解释。[来源配对差异表](paired-source-contrasts.csv)保留完整结果。

## 数据扩展带来的来源差异

AUROC 是受试者工作特征曲线下面积（area under the receiver operating characteristic curve）。直观地说，随机抽取一张阳性和一张阴性图像时，它表示模型给阳性更高分的概率，分数并列计一半。例如 AUROC 为 0.93，表示约 93% 的阳性—阴性配对能够正确排序；AUROC 为 0.5 相当于随机排序，为 1 表示完全区分。改变分类阈值，会得到不同的敏感度和误报率，这些点组成 ROC 曲线；AUROC 汇总整条曲线的排序能力，实际检出和误报数量则由所选阈值决定。下表按来源比较同一批开发原图，正值表示前一个方案更好，例如 +0.027 表示 AUROC 增加 0.027。

{source}

KEH 的 AUROC 从 0.6627 升至 0.9333，LAG 从 0.9465 升至 0.9890，两项差异的 95% 区间均高于零。KEH 以视盘裁剪图为主，LAG 为完整眼底图，改善覆盖两种成像视野。AIROGS 和 ODIR 分别下降 0.0136 和 0.0397，两项区间均低于零。Kaggle、BEH、FIVES 的区间较宽，覆盖正负两种变化。

扩展无退化选中第 300 次更新权重时，AIROGS 相对原数据下降 0.0312；到第 1,200 次更新，降幅缩小至 0.0136。ODIR 的两项降幅分别为 0.0389 和 0.0397。因此，权重选择时间解释了部分 AIROGS 差异，ODIR 的下降在训练预算结束时仍存在。

![第1,200次更新权重的来源AUROC配对差异](source-differences.zh.png)

图1左侧比较扩展无退化与原数据，右侧比较 50% 退化与扩展无退化。圆点是 AUROC 差，横线是关联组重采样的 95% 区间，虚线是零差异。左右横轴范围不同，以便看清各自的变化。50% 退化在 LAG 原图上下降 0.0058，区间为 [−0.0098, −0.0025]；AIROGS、ODIR、KEH 的额外变化区间均覆盖零。

AUPRC 是精确率—召回率曲线下面积，关注阳性检出的精确率和覆盖程度，随来源的阳性比例变化，适合在同来源内比较模型。Brier 分数是预测概率与 0/1 标签之间的平均平方误差，越低表示概率预测越准确。

{diagnostic}

KEH 和 LAG 的 AUPRC 提高、Brier 下降，AIROGS 的 AUPRC 下降、Brier 上升，与 AUROC 的方向一致。ODIR 的 Brier 上升，但 AUPRC 差异区间覆盖零。ODIR 的 305 张开发图像只有 193 个关联组，其中阳性组 19 个，这使阳性检出指标的区间较宽。

## 新增来源如何改变采样和权重选择

每次优化更新使用一批采样图像；同一张图像可在训练过程中出现多次。现有采样器为每张图像分配其“来源与类别组合”图像数的平方根倒数权重。一个组合内图像越多，其总采样权重越大，但增长慢于图像数。加入新来源后，旧来源分得的采样比例降低。

{exposure}

表中“不同原图”表示实际至少采样过一次的图像。原数据对照共采样 19,186 次，覆盖 8,339/10,338 张训练图像（80.7%）。扩展方案中，原数据采样 9,484 次，占 49.4%，覆盖 5,929 张原图（57.4%）；新增数据采样 9,708 次，占 50.6%，覆盖 3,901/5,142 张新增训练图像（75.9%）。三个扩展方案逐图采样次数完全相同，适合比较退化增强的影响。末尾不满批次使原数据与扩展方案的总采样次数相差 6 次。

新增图像最多重复 30 次，原数据对照最多重复 14 次。数据扩展同时改变了来源组成、原数据采样次数和重复频率；下一轮控制原数据采样配额，可以进一步检验这些因素对原来源表现的影响。

![实际图像采样次数与验证指标变化](exposure-selection.zh.png)

图2左侧显示四组的原数据和新增数据采样次数；右侧显示扩展无退化方案在三种验证来源组合下的 AUROC 等权均值。本轮选择权重时使用 11 个同时包含阳性和阴性的验证来源。HRF、CRFO、Fundus-AVSeg、LES-AV 分别只有 4、6、3、2 张验证图像，合计 15 张，却占这个均值 4/11，即 36.4% 的权重。

扩展无退化从第 300 到 1,200 次更新，HRF AUROC 从 1.0 降至 0.5，使 11 来源均值下降 0.04545；该均值的总降幅为 0.04807。CRFO 另贡献 −0.01136，其他来源的改善抵消了部分下降。原数据对照中，CRFO 从 0.5 升至 1.0，贡献 +0.04545，而均值总升幅为 0.03241。少量验证图像对权重选择有较大影响。

若分析至少有 20 张验证图像的七个来源，扩展无退化的均值从 0.9235 升至 0.9373，最高点位于第 900 次更新；原三个来源的均值也在第 900 次更新最高。这是对已有验证日志的敏感性分析。第 600 和 900 次更新的完整权重未保存，因此开发集结果仍使用已保存的第 300 和 1,200 次权重。

## 退化增强在合成失焦下的效果

合成退化评估使用同一组 384 张开发原图，包括阳性 146 张、阴性 238 张，按图像标识的哈希固定选取。失焦半径以 512 像素眼底视野为参照取 2，并随实际视野缩放；曝光变体将亮度乘以 0.70。每张图像的原图与退化版本配对比较。下表使用第 1,200 次更新权重，以预测概率大于 0.5 判为阳性；该阈值用于本轮结果展示。

{stress}

失焦条件下，50% 退化相对扩展无退化的 AUROC 增加 0.0555 [0.0308, 0.0817]，AUPRC 增加 0.1019 [0.0511, 0.1592]，Brier 下降 0.0684 [0.0445, 0.0938]。误报从 68/238 降至 19/238，误报率下降 20.59 个百分点 [15.25, 26.03]；漏诊从 18/146 升至 28/146，漏诊率增加 6.85 个百分点 [0.71, 13.33]。排序改善与固定阈值下的漏诊—误报变化需要一起评价。

![合成失焦下的ROC曲线与预测概率变化](defocus-ranking-shifts.zh.png)

图3左侧的 ROC 曲线展示不同阈值对应的敏感度与误报率，图例同时标出 AUROC；右侧按真实标签显示失焦分数减去原图分数的平均差及 95% 区间。扩展无退化的阴性图像分数平均增加 0.2295，50% 退化只增加 0.0622。原数据对照的阳性分数平均下降 0.1523，50% 退化下降 0.0273。退化增强减轻了失焦对阳性和阴性分数的影响。

将同一退化子集上的特异度匹配到 95.4% 后，四组失焦敏感度分别为 65.8%、38.4%、65.1%、72.6%。这项探索性比较为每组、每种条件重新计算阈值，用来解释排序表现。来源内比较也显示改善：50% 退化在失焦 AIROGS 上的 AUROC 增加 0.0691 [0.0258, 0.1149]，在 LAG 上增加 0.0835 [0.0341, 0.1455]。

另一项比较固定每组从原始验证阴性图像设定的来源阈值，再用于原图与退化版本。验证集同时包含两类标签的七个来源贡献 356 张压力评估图像，其中阳性 139 张、阴性 217 张；其余 28 张图像所在的小来源缺少两类验证标签。下表依次列出漏诊数／误报数。

{operating}

固定验证阈值后，50% 退化相对扩展无退化的失焦误报从 47 降至 10，漏诊从 28 升至 40。曝光乘以 0.70 时，AUROC 差为 +0.0028 [−0.0078, 0.0138]，漏诊从 25 升至 35。当前增强收益主要集中在失焦；下一轮单独比较失焦增强与混合增强，可以确定有效的退化类型和强度。

## 真实低质量图像的检出与标签分布

真实低质量开发图像来自 FIVES 7 张、MSHF 5 张，均为阳性，属于 12 个不同关联组。在阈值 0.5 下，原数据对照检出 7/12，三个扩展方案均检出 12/12。12/12 对应的精确二项 95% 区间为 73.5%–100%；FIVES 和 MSHF 分别为 59.0%–100% 和 47.8%–100%。新增检出为 FIVES 2 张、MSHF 3 张，无新增漏诊；比较同图检出变化的 McNemar 精确检验双侧 p=0.0625。区间说明这组图像的检出比例仍存在较大的抽样不确定性。

![真实质量与疾病标签的训练分布以及开发阳性检出比例](real-quality-labels.zh.png)

图4左侧按来源和原始质量标注显示训练图像数，柱顶数字依次为阳性数＋阴性数；右侧显示同来源真实低质量开发阳性的检出比例与精确二项 95% 区间。FIVES 以至少一个质量维度标为差定义低质量，MSHF 使用总体质量标注。“其他质量”包含该来源剩余质量档次，两个来源保留各自的标注标准。

FIVES 的 47 张低质量训练图像中，阳性 44 张、阴性 3 张；其他 129 张中，阳性 46 张、阴性 83 张。MSHF 的 28 张低质量和 2 张其他质量训练图像全部为阳性。实际训练采样中，这两个来源的真实低质量阳性出现 413 次，阴性仅 10 次。质量与疾病标签高度关联，因此下一轮应加入同来源真实低质量阴性，比较质量平衡采样，区分来源、质量线索与疾病特征的作用。

固定验证来源阈值后，扩展无退化和 50% 退化均检出 FIVES 的 7/7 张低质量阳性。对比这两个方案，FIVES 全部 13 张开发阳性的敏感度从 76.9% 升至 92.3%，14 张阴性的特异度从 100% 降至 78.6%。阈值来自 14 张验证阴性，并取其最高分数，后续需要更多阴性图像来稳定阈值。MSHF 当前仅有阳性图像，其来源内特异度需在补充阴性后评估。

## 下一轮控制实验建议

建议设置一个原数据对照，以及四个扩展方案，分别组合两种采样方法（现有方法、原数据 65% 采样配额）和两种增强设置（无退化、50% 混合退化）。每组使用三个共同种子、相同初始化条件和 2,400 次有效更新。按每批 16 张计算，65% 配额预计提供 24,960 次原数据采样，高于本轮原数据对照的 19,186 次；来源内继续使用平方根倒数权重。在相同预算下比较采样与增强，并保存第 1,200 和 2,400 次更新权重，评估继续训练的影响。

预先指定 AIROGS、ODIR、Kaggle、LAG、KEH 五来源 AUROC 的等权均值作为主要权重选择指标，同时报告原三个来源均值、各来源敏感度和特异度。BEH、FIVES 和更小来源作为补充评估。每 300 次更新保存完整权重与预测，分别报告开发图像重采样区间和跨训练种子的变化。

随后在相同应用概率、采样序列和更新预算下，比较单独失焦与混合退化增强。评估固定验证阈值下的漏诊、误报、真实低质量阴性表现，以及原图 AUROC 和 Brier。依据预先约定的原来源表现、质量分层结果和复核工作量选择模型与阈值，再执行最终测试。上述方案是下一轮建议，尚未运行。

## 分析产物与复现

[训练结果](../training-results/training-results.zh.md) · [数据审计](../training-results/data-audit.zh.md) · [退化审计](../training-results/degradation-audit.zh.md) · [图像采样记录](training-exposure.json) · [图表数据](figure-data.json) · [分析程序](../../../../tools/analyze_expansion_results_20261113.py) · [报告生成程序](../../../../tools/report_expansion_analysis_20261113.py)

在训练环境运行分析程序的 `--extract-draws` 选项，提取检查点内的逐图采样计数；随后在文档环境运行分析程序和报告生成程序。分析核对四组的图像标识、标签、来源、关联组与冻结清单，并记录输入文件的 SHA256。统计测试覆盖并列分数处理、关联组重采样和同图配对检查。
'''
        else:
            text = f'''# Data expansion and degradation augmentation: experimental analysis

November 13, 2026 · [中文](experimental-analysis.zh.md) · [English PDF](experimental-analysis.en.pdf)

Data expansion improves glaucoma-image ranking in KEH and LAG, while performance declines in AIROGS and ODIR. Adding 50% degradation improves ranking under synthetic defocus and reduces false positives, with an increase in missed positives. All three expanded arms detect the same 12 real poor-quality development positives, so degradation adds no detections in that group.

This report follows the same-day [data and methods review](../data-review/data-and-degradation-review.en.md) and [training report](../training-results/training-results.en.md). It examines source differences, actual image presentations, checkpoint selection, and the association between real image quality and disease labels to inform the next controlled experiment.

## Data, models, and comparison method

All four arms use EfficientNet-B0, the same initialization, seed 20261113, and 1,200 successful optimizer updates. The original-data arm contains 10,338 training images; the expanded arms contain 15,480. The 30% and 50% settings are probabilities of applying mixed degradation to sampled images. Positive labels retain source definitions, including glaucoma diagnosis and AIROGS glaucoma-referral labels, so results are analyzed by source. Model validation contains 2,046 images, including 823 positives, and rule development contains 2,053 images, including 814 positives. Validation selects checkpoints and supplies source thresholds; rule development supports manual-review rules and supplies the results below. Final testing follows model and review-rule selection.

The training report uses checkpoints with the highest validation criterion: update 1,200 for original data, 30% degradation, and 50% degradation, and update 300 for expanded clean. This analysis mainly compares all update-1,200 checkpoints at an equal update budget. Selected-checkpoint results are retained in [analysis data](analysis.json). An epoch is a complete pass through a training set; an optimizer update adjusts model parameters after a batch. This round sets its budget in updates.

The 95% intervals use 4,000 paired bootstrap resamples of whole linked groups. The same groups are drawn for each model comparison. Groups retain known pairs of eyes, repeat photographs, or other linked images together, preserving dependence; metrics are image weighted. Intervals describe development-image sampling uncertainty. Variation from the single training seed will be assessed in subsequent multiseed experiments. Each interval is an individual percentile interval, and source contrasts are interpreted individually. [Paired source contrasts](paired-source-contrasts.csv) retain the full results.

## Source differences after data expansion

AUROC is the area under the receiver operating characteristic curve. For a randomly chosen positive and negative image, it is the probability that the model gives the positive a higher score, with ties counting half. An AUROC of 0.93 means about 93% of positive–negative pairs are ranked correctly; 0.5 corresponds to random ranking and 1 to perfect separation. Varying the classification threshold gives different sensitivity and false-positive rates, forming the ROC curve. AUROC summarizes ranking over that curve, while the chosen threshold determines actual detections and false positives. The table compares the same development originals within each source. Positive differences favor the first arm; +0.027 means an AUROC increase of 0.027.

{source}

KEH AUROC rises from 0.6627 to 0.9333, and LAG from 0.9465 to 0.9890, with both difference intervals above zero. KEH mainly contains optic-disc crops and LAG full fundus photographs, so gains span both fields of view. AIROGS and ODIR decline by 0.0136 and 0.0397, respectively, with both intervals below zero. The wider Kaggle, BEH, and FIVES intervals include changes in both directions.

At the selected update-300 expanded-clean checkpoint, AIROGS falls by 0.0312 relative to original training; the decline narrows to 0.0136 by update 1,200. ODIR falls by 0.0389 and 0.0397 in the two comparisons. Checkpoint timing therefore accounts for part of the AIROGS difference, while the ODIR decline persists at the end of the budget.

![Paired source AUROC differences at update 1,200](source-differences.en.png)

Figure 1 compares expanded clean with original data on the left and 50% degradation with expanded clean on the right. Points show AUROC differences, horizontal bars linked-group 95% intervals, and dashed lines zero difference. The panels use different horizontal ranges to show their respective changes clearly. Adding 50% degradation reduces LAG original-image AUROC by 0.0058 [−0.0098, −0.0025]; the AIROGS, ODIR, and KEH additional-change intervals include zero.

AUPRC is the area under the precision–recall curve. It describes precision and coverage of positive detections, varies with source prevalence, and supports comparisons within a source. The Brier score is the mean squared difference between predicted probabilities and 0/1 labels; lower values indicate more accurate probability predictions.

{diagnostic}

KEH and LAG improve in AUPRC and Brier, while AIROGS worsens in both, consistent with their AUROC directions. ODIR Brier increases, but its AUPRC difference interval includes zero. The 305 ODIR development images represent only 193 linked groups, including 19 positive groups, contributing to wider positive-detection intervals.

## How added sources affect sampling and checkpoint selection

Each optimizer update processes a sampled batch; an image can appear several times during training. The current sampler assigns each image the inverse square root of the number of images in its source-and-class combination. Larger combinations receive more total sampling weight, but the weight grows more slowly than image count. Adding sources reduces the sampling share allocated to original sources.

{exposure}

Distinct images in the table are images presented at least once. Original training makes 19,186 presentations, covering 8,339/10,338 training images (80.7%). Expanded training makes 9,484 original-image presentations, or 49.4%, covering 5,929 original images (57.4%). Added images contribute 9,708 presentations, or 50.6%, covering 3,901/5,142 added training images (75.9%). All three expanded arms have identical per-image presentation counts, supporting the augmentation comparison. Partial final batches cause the six-presentation difference in totals.

A single added image appears up to 30 times, versus 14 for original training. Expansion changes source composition, original-image presentations, and repetition frequency together. Controlling the original-image quota in the next round will help assess their effects on original-source performance.

![Actual image presentations and changes in validation means](exposure-selection.en.png)

Figure 2 shows original and added image presentations on the left and expanded-clean validation AUROC means over three source combinations on the right. This round selects checkpoints using an equal mean over 11 validation sources containing both classes. HRF, CRFO, Fundus-AVSeg, and LES-AV have just 4, 6, 3, and 2 validation images: 15 images account for 4/11, or 36.4%, of that mean's weight.

Between updates 300 and 1,200, expanded-clean HRF AUROC falls from 1.0 to 0.5, reducing the 11-source mean by 0.04545; the total decline is 0.04807. CRFO contributes another −0.01136, partly offset by other-source gains. In original training, CRFO rises from 0.5 to 1.0 and contributes +0.04545, compared with a total mean gain of 0.03241. A few validation images consequently have substantial influence on checkpoint selection.

Considering only the seven sources with at least 20 validation images, the expanded-clean mean rises from 0.9235 to 0.9373 and peaks at update 900. The original three-source mean also peaks at 900. This sensitivity analysis uses existing validation logs. Full update-600 and update-900 checkpoints were not saved, so development results use the saved update-300 and update-1,200 weights.

## Effects of augmentation under synthetic defocus

Synthetic degradation evaluation uses the same 384 development originals: 146 positives and 238 negatives, selected deterministically by image-identifier hashes. Defocus uses radius 2 for a 512-pixel fundus field, scaled to the actual field. Exposure variants multiply brightness by 0.70. Originals and degraded versions are paired. The table uses update-1,200 weights and labels an image positive when its probability exceeds 0.5, the display threshold for this round.

{stress}

Under defocus, 50% degradation improves AUROC by 0.0555 [0.0308, 0.0817] and AUPRC by 0.1019 [0.0511, 0.1592], and reduces Brier by 0.0684 [0.0445, 0.0938], relative to expanded clean. False positives fall from 68/238 to 19/238: a 20.59-percentage-point reduction [15.25, 26.03]. Misses rise from 18/146 to 28/146: a 6.85-point increase [0.71, 13.33]. Ranking gains and the missed-positive/false-positive balance at a fixed threshold should be assessed together.

![ROC curves and probability changes under synthetic defocus](defocus-ranking-shifts.en.png)

Figure 3 shows sensitivity versus false-positive rate over thresholds on the left, with AUROC in the legend. The right panel shows mean defocus-minus-original probability changes by disease label with 95% intervals. Expanded-clean negative scores rise by 0.2295 on average, versus 0.0622 with 50% degradation. Original-training positive scores fall by 0.1523, versus 0.0273 with 50% degradation. Augmentation reduces the effect of defocus on both classes' scores.

Matching specificity at 95.4% on the same degraded subset gives defocus sensitivities of 65.8%, 38.4%, 65.1%, and 72.6%. This exploratory comparison recalculates thresholds for each arm and condition to explain ranking. Within-source comparisons also show gains: adding 50% degradation improves defocused AIROGS AUROC by 0.0691 [0.0258, 0.1149] and LAG by 0.0835 [0.0341, 0.1455].

A separate comparison fixes each arm's source thresholds from original validation negatives and applies them to original and degraded images. Seven sources with both validation classes contribute 356 stress-evaluation images: 139 positives and 217 negatives. The other 28 images come from small sources missing one validation class. Entries below are misses / false positives.

{operating}

At fixed validation thresholds, adding 50% degradation reduces defocus false positives from 47 to 10 and increases misses from 28 to 40. With exposure multiplied by 0.70, the AUROC difference is +0.0028 [−0.0078, 0.0138] and misses increase from 25 to 35. Current augmentation gains concentrate on defocus. Separating defocus-only from mixed augmentation in the next round will help identify useful degradation types and strengths.

## Real poor-quality detections and label distribution

Real poor-quality development images comprise seven FIVES and five MSHF positives in 12 distinct linked groups. At threshold 0.5, original training detects 7/12, and all three expanded arms detect 12/12. The exact binomial 95% interval for 12/12 is 73.5%–100%; FIVES and MSHF intervals are 59.0%–100% and 47.8%–100%, respectively. Five additional detections comprise two FIVES and three MSHF images, with no additional misses. The exact McNemar test of changed decisions on the same images gives two-sided p=0.0625. The intervals indicate substantial sampling uncertainty for this small set.

![Training distribution of real quality and disease labels, and development detections](real-quality-labels.en.png)

Figure 4 shows training-image counts by source and original quality annotation on the left; labels above bars are positive + negative counts. The right panel shows detected proportions and exact binomial 95% intervals for each source's real poor-quality development positives. FIVES defines poor quality as at least one poor quality dimension, while MSHF uses overall quality. Other quality covers each source's remaining grades; source-specific annotation standards are retained.

Of 47 poor-quality FIVES training images, 44 are positive and three negative. Its other 129 images contain 46 positives and 83 negatives. All 28 poor-quality and two other-quality MSHF training images are positive. Actual training includes 413 poor-quality positive presentations from these sources and only ten negative presentations. Given this association between quality and disease labels, the next round should add same-source poor-quality negatives and compare quality-balanced sampling to distinguish source and quality cues from disease features.

At fixed source thresholds from validation, expanded clean and 50% degradation both detect all seven poor-quality FIVES positives. Comparing these arms, sensitivity over all 13 FIVES development positives rises from 76.9% to 92.3%, while specificity on 14 negatives falls from 100% to 78.6%. Thresholds come from just 14 validation negatives and select their maximum score; additional negatives are needed to stabilize threshold estimates. MSHF currently contains positives only, so source-specific specificity requires added negatives.

## Proposed next controlled experiment

Use one original-data control and four expanded arms combining two sampling methods, current sampling and a 65% original-image quota, with two augmentation settings, no degradation and 50% mixed degradation. Use three common seeds, equal initialization conditions, and 2,400 successful updates per arm. At 16 images per batch, the 65% quota provides about 24,960 original-image presentations, versus 19,186 in this round's control. Retain inverse-square-root weights within sources. Compare sampling and augmentation at equal budgets, and save update-1,200 and update-2,400 checkpoints to assess continued training.

Predeclare the equal AUROC mean over AIROGS, ODIR, Kaggle, LAG, and KEH as the primary checkpoint-selection criterion. Report the original three-source mean and source-specific sensitivity and specificity as well. Treat BEH, FIVES, and smaller sources as supplementary evaluations. Save full weights and predictions every 300 updates, and report development-image bootstrap intervals separately from variation across seeds.

Then compare defocus-only with mixed augmentation using the same application probability, sampling sequence, and update budget. Evaluate misses, false positives, real poor-quality negatives, and original-image AUROC and Brier at fixed validation thresholds. Use predeclared original-source performance, quality-stratified results, and review workload to select models and thresholds before final testing. These are proposed experiments and have not yet run.

## Analysis artifacts and reproduction

[Training results](../training-results/training-results.en.md) · [Data audit](../training-results/data-audit.en.md) · [Degradation audit](../training-results/degradation-audit.en.md) · [Image presentation counts](training-exposure.json) · [Figure data](figure-data.json) · [Analysis program](../../../../tools/analyze_expansion_results_20261113.py) · [Report generator](../../../../tools/report_expansion_analysis_20261113.py)

Run the analysis program with `--extract-draws` in the training environment to extract per-image presentation counts from checkpoints. Then run the analysis and report programs in the document environment. The analysis checks image identifiers, labels, sources, linked groups, and frozen-manifest metadata for all four arms, retaining input SHA256 hashes. Statistical tests cover tied scores, linked-group resampling, and same-image pairing.
'''
        (OUT / f'experimental-analysis.{lang}.md').write_text(text, encoding='utf-8')


def compile_reports():
    BUILD.mkdir(parents=True,exist_ok=True)
    result={}
    for lang in ('zh','en'):
        stem=f'experimental-analysis.{lang}';tex=OUT/f'{stem}.tex'
        subprocess.run(['pandoc',str(OUT/f'{stem}.md'),'-s','-t','latex','-o',str(tex),
            '-V','mainfont=Arial','-V','CJKmainfont=Microsoft YaHei','-V','monofont=Consolas',
            '-V','fontsize=10pt','-V','geometry:margin=0.65in','-V','colorlinks=true','-V','urlcolor=blue'],cwd=OUT,check=True,capture_output=True)
        text=tex.read_text(encoding='utf-8').replace('\\begin{figure}','\\begin{figure}[H]')
        text=text.replace('\\begin{longtable}','\\Needspace{12\\baselineskip}\n\\begin{longtable}')
        prefix='\\usepackage{float}\n\\usepackage{needspace}\n\\setCJKmonofont{Microsoft YaHei}\n\\Urlmuskip=0mu plus 2mu\\relax\n'+('\\renewcommand{\\figurename}{图}\n' if lang=='zh' else '')
        tex.write_text(text.replace('\\begin{document}',prefix+'\\begin{document}'),encoding='utf-8')
        for index in range(2):
            process=subprocess.run(['xelatex','-interaction=nonstopmode','-halt-on-error',f'-output-directory={BUILD}',str(tex)],cwd=OUT,capture_output=True)
            (BUILD/f'{stem}-build-{index}.log').write_bytes(process.stdout+process.stderr);process.check_returncode()
        shutil.copyfile(BUILD/f'{stem}.pdf',OUT/f'{stem}.pdf')
        # Only remove prior renderer products in this analysis-specific directory.
        for pattern in (f'page-{lang}-*.png',f'contact-{lang}-*.jpg'):
            for path in BUILD.glob(pattern):
                if path.resolve().parent!=BUILD.resolve():raise ValueError('renderer target outside build')
                path.unlink()
        subprocess.run(['pdftoppm','-scale-to','1500','-png',str(OUT/f'{stem}.pdf'),str(BUILD/f'page-{lang}')],check=True,capture_output=True)
        subprocess.run(['pdftotext','-layout',str(OUT/f'{stem}.pdf'),str(BUILD/f'{stem}.txt')],check=True,capture_output=True)
        result[lang]=dict(pages=(BUILD/f'{stem}.txt').read_text(encoding='utf-8').count('\f'),latex_retained=True,pdf_bytes=(OUT/f'{stem}.pdf').stat().st_size)
    (OUT/'render-verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


def main():
    OUT.mkdir(parents=True,exist_ok=True);r=read_json(OUT/'analysis.json')
    charts(r);reports(r);compile_reports()


if __name__=='__main__':main()
