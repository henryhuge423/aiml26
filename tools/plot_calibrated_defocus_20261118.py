"""Bilingual figures for the calibrated defocus and clinician scenario results."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np

REPORT = Path(__file__).resolve().parents[1]/'docs/progress/2026-11-18'
BLUE = '#246484'
AMBER = '#bb5126'
INK = '#243d4a'
GRAY = '#657681'


def read(name):
    return json.loads((REPORT/name).read_text(encoding='utf-8'))


def style(axis, grid='y'):
    axis.grid(axis=grid,color='#dae4e8')
    axis.set_axisbelow(True)
    axis.spines[['top','right']].set_visible(False)
    axis.tick_params(length=3,colors=INK)


def output(figure,name,lang):
    figure.savefig(REPORT/f'{name}.{lang}.png',dpi=200,facecolor='white')
    plt.close(figure)


def calibration(data,lang,labels):
    methods = ['current','cofe_100','cofe_111','arc','calibrated']
    metrics = data['metrics']
    scores = data['model_scores']['methods']
    index = np.arange(len(methods))
    colors = [BLUE,GRAY,GRAY,GRAY,AMBER]
    figure,(left,right) = plt.subplots(1,2,figsize=(8.9,3.05),sharey=True,layout='constrained',
                                       gridspec_kw={'width_ratios':[1,1]})
    truth = metrics['real_low']['green_laplacian_variance']['median']
    level = np.array([metrics[k]['green_laplacian_variance']['median'] for k in methods])
    bars = left.barh(index,100*level,color=colors,height=.58)
    left.bar_label(bars,labels=[f'{v:.0%}' for v in level],padding=4,fontsize=8)
    left.axvline(100*truth,color=INK,linestyle='--',linewidth=1.2,
                 label=f'{labels[7]} {truth:.0%}')
    left.set(xlabel=labels[5],yticks=index,yticklabels=labels[:5],xlim=(0,90))
    left.invert_yaxis()
    left.legend(loc='lower right',fontsize=8,frameon=False)
    realized = scores['real_low']['absolute_change_over_005']
    changes = [scores[key]['absolute_change_over_005'] for key in methods]
    bars = right.barh(index,changes,color=colors,height=.58)
    right.bar_label(bars,padding=4,fontsize=8)
    right.axvline(realized,color=INK,linestyle='--',linewidth=1.2,label=f'{labels[7]} {realized}')
    right.set(xlabel=labels[6],xlim=(0,max(changes)+3))
    right.legend(loc='lower right',fontsize=8,frameon=False)
    for axis in (left,right):
        style(axis,'x')
    output(figure,'calibrated-image-benchmark',lang)


def expert_scenarios(data,lang,labels):
    arm_keys = ['quality_gray_zone','second_model_negative_reranking']
    scenarios = ['model_only','lower_sensitivity_assumption','intermediate_assumption',
                 'selected_eyepacs_graders','perfect_reference_ceiling']
    index = np.arange(len(scenarios))
    figure,(left,right) = plt.subplots(1,2,figsize=(8.9,3.15),sharey=True,layout='constrained')
    for arm,offset,color,label in zip(arm_keys,(-.16,.16),(BLUE,AMBER),labels[5:7]):
        points = data['policies'][arm]['scenarios']
        gain = [0,*[points[key]['net_errors_prevented'] for key in scenarios[1:]]]
        misses = [data['model_only_fn'],*[points[key]['remaining_fn'] for key in scenarios[1:]]]
        left.scatter(gain,index+offset,color=color,s=29,label=label,zorder=3)
        right.scatter(misses,index+offset,color=color,s=29,zorder=3)
    right.axvline(data['model_only_fn'],color=GRAY,linestyle='--',linewidth=1)
    left.set(xlabel=labels[7],yticks=index,yticklabels=labels[:5],xlim=(-1.5,33))
    left.invert_yaxis()
    right.set(xlabel=labels[8],xlim=(0,5.8))
    left.legend(loc='upper right',fontsize=8,frameon=False)
    for axis in (left,right):
        style(axis,'x')
    output(figure,'calibrated-expert-scenarios',lang)


def learning(data,lang,labels):
    seeds = data['seeds']
    figure,axes = plt.subplots(1,2,figsize=(8.9,2.9),layout='constrained')
    for axis,metric,heading in zip(axes,('validation_airogs_auc','validation_five_source_auc'),labels[:2]):
        for arm,color,label in zip(data['arms'],(BLUE,AMBER),labels[4:6]):
            traces = [data['training'][str(seed)][arm][metric] for seed in seeds]
            updates = data['training'][str(seeds[0])][arm]['update_checkpoints']
            for trace in traces:
                axis.plot(updates,trace,color=color,alpha=.23,linewidth=1)
            axis.plot(updates,np.mean(traces,axis=0),color=color,marker='o',markersize=4,
                      linewidth=2,label=label)
        axis.set(title=heading,xlabel=labels[2],xticks=updates)
        style(axis)
    axes[0].set_ylabel(labels[3])
    axes[0].legend(fontsize=8,loc='lower right',frameon=False)
    output(figure,'calibrated-validation',lang)


def paired_auc(data,lang,labels):
    figure,axis = plt.subplots(figsize=(8.55,3.05),layout='constrained')
    axis.axvline(0,color=GRAY,linestyle='--',linewidth=1)
    for index,item in enumerate(data['conditions'].values()):
        lo,hi = item['interval_95']
        points = [b['calibrated_defocus']-b['original_mix'] for b in item['by_seed'].values()]
        axis.scatter(points,np.full(len(points),index+.16),s=18,facecolors='none',edgecolors=BLUE,zorder=2)
        axis.hlines(index-.11,lo,hi,color=AMBER,linewidth=2,label=labels[7] if index==0 else None)
        axis.scatter(item['difference'],index-.11,s=33,color=AMBER,zorder=3)
    axis.scatter([],[],facecolors='none',edgecolors=BLUE,s=18,label=labels[6])
    axis.set(yticks=range(5),yticklabels=labels[:5],xlabel=labels[5],ylim=(4.55,-.55))
    style(axis,'x')
    axis.legend(loc='best',fontsize=8,frameon=False)
    output(figure,'calibrated-development-auroc',lang)


def routing(data,lang,labels):
    seeds = data['seeds']
    values = data['screening_scenario']['by_seed']
    figure,(misses,alarms) = plt.subplots(1,2,figsize=(8.9,2.8),layout='constrained')
    for axis,keys,heading in [
        (misses,('project_model_only_fn','remaining_fn_per_1000','clinician_scenario_fn_per_1000'),labels[3]),
        (alarms,('project_model_only_fp','remaining_fp_per_1000'),labels[4]),
    ]:
        for arm,color,name,offset in zip(data['arms'],(BLUE,AMBER),labels[5:7],(-.08,.08)):
            observations = [[values[str(seed)][arm][key] for seed in seeds] for key in keys]
            points = np.arange(len(keys))+offset
            for i, seen in enumerate(observations):
                axis.scatter(np.repeat(points[i],len(seen)),seen,color=color,s=17,alpha=.45)
            axis.plot(points,np.mean(observations,axis=1),color=color,marker='D',linewidth=1.5,
                      markersize=4,label=name)
        axis.set(title=heading,xticks=range(len(keys)),xticklabels=labels[:len(keys)])
        axis.set_ylim(bottom=0)
        style(axis)
    alarms.legend(loc='upper right',fontsize=8,frameon=False)
    output(figure,'calibrated-review-routing',lang)


def main():
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = [font_manager.FontProperties(fname=font).get_name(),'DejaVu Sans']
    plt.rcParams.update({'font.size':10,'axes.labelcolor':INK,'text.color':INK})
    prior = read('review-degradation-benchmark.json')
    for lang,values in {
        'zh': (
            ['现有混合','CoFe 失焦','CoFe 综合','ArcNet 白内障','同眼标定',
             '相对清晰图的细节指标，中位数 (%)','疾病分数变动幅度超过 0.05 的照片数','真实'],
            ['模型单独判读','60% / 88%','75% / 90%','86% / 96.4%','全部正确',
             '原质量灰区','双模型重排','每千张净减少的错误','每千张仍漏掉的转诊'],
            ['AIROGS 模型验证集','五来源验证均值','参数更新次数','AUROC','原混合','同眼标定失焦'],
            ['混合开发原图','AIROGS 开发原图','失焦对照原图','模拟轻失焦','模拟强失焦',
             '同眼标定失焦减原混合的 AUROC','单种子差值','三种子均值与 95% 区间'],
            ['模型原判','复核名单外','模拟阅片后','每千张漏掉的转诊','每千张误报','原混合','同眼标定失焦'],
        ),
        'en': (
            ['Original mix','CoFe blur','CoFe combined','ArcNet cataract','Pair-calibrated',
             'Median detail relative to the clearer image (%)','Photos with an absolute score change > 0.05','Real'],
            ['Model alone','60% / 88%','75% / 90%','86% / 96.4%','Perfect review',
             'Quality gray zone','Two-model ranking','Net errors avoided per 1,000','Misses left per 1,000'],
            ['AIROGS model validation','Five-source validation mean','Optimizer updates','AUROC','Original mix','Calibrated defocus'],
            ['Pooled originals','AIROGS originals','Blur-set originals','Simulated mild blur','Simulated strong blur',
             'AUROC difference, calibrated defocus minus original mix','Individual seeds','Mean and 95% interval'],
            ['Model alone','Outside review','After simulated reading','Misses per 1,000','False positives per 1,000',
             'Original mix','Calibrated defocus'],
        ),
    }.items():
        calibration(prior['degradation'],lang,values[0])
        expert_scenarios(prior['clinician_projection'],lang,values[1])
        path = REPORT/'calibrated-defocus-results.json'
        if path.exists():
            study = read(path.name)
            learning(study,lang,values[2])
            paired_auc(study,lang,values[3])
            routing(study,lang,values[4])


if __name__=='__main__':
    main()
