"""Build bilingual reports and render every page for review."""
from __future__ import annotations
import datetime as dt
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageOps,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/progress/2026-11-15'
BUILD=ROOT/'.build/quality-20261115'
RUN=ROOT/'runs/followup_20261115'
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','Arial'],'axes.unicode_minus':False,'font.size':9})


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',
        *['| '+' | '.join(map(str,row))+' |' for row in rows]])


def charts(r):
    bins=list(r['roles']['boundary']['quality_bins'].values())
    data=pd.read_csv(RUN/'error_ranking/development_join.csv')
    boundary=data[data['split'].eq('boundary')]
    for lang in ('zh','en'):
        zh=lang=='zh';x=np.arange(5)
        fig,axes=plt.subplots(1,2,figsize=(10,3.7),constrained_layout=True)
        n=np.array([b['n'] for b in bins]);errors=np.array([b['errors'] for b in bins])
        axes[0].bar(x,n-errors,color='#cddfe2',label='正确' if zh else 'Correct')
        axes[0].bar(x,errors,bottom=n-errors,color='#b9473d',label='错误' if zh else 'Errors')
        for i,b in enumerate(bins):axes[0].text(i,b['n']+12,f"{b['errors']}/{b['n']}",ha='center',fontsize=8)
        axes[0].set(ylabel='图像数' if zh else 'Images',title='错误数 / 图像数' if zh else 'Errors / images',ylim=(0,870))
        axes[0].legend(loc='upper left',fontsize=8)
        rate=np.array([b['error_rate'] for b in bins])*100
        ci=np.array([b['error_rate_group_interval95'] for b in bins])*100
        axes[1].errorbar(x,rate,yerr=[rate-ci[:,0],ci[:,1]-rate],fmt='o',capsize=4,color='#17697c',label='错判率及95%区间' if zh else 'Error rate and 95% interval')
        axes[1].axhline(100*r['roles']['boundary']['overall']['error_rate'],color='#999',ls='--',label='全体错判率' if zh else 'Overall error rate')
        axes[1].set(ylabel='错判率 (%)' if zh else 'Error rate (%)',title='按关联组重采样' if zh else 'Resampling linked groups',ylim=(0,22))
        axes[1].legend(fontsize=7.5,loc='upper right')
        for ax in axes:
            ax.set_xticks(x,['0–0.2','0.2–0.4','0.4–0.6','0.6–0.8','0.8–1'])
            ax.set_xlabel('质量模型分数 q' if zh else 'Quality-model score q');ax.spines[['top','right']].set_visible(False)
        fig.savefig(OUT/f'quality-counts-rates.{lang}.png',dpi=200);plt.close(fig)

        fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
        groups=[data[data.source.eq('fives')&data.poor_human],data[data.source.eq('fives')&~data.poor_human],data[data.source.eq('mshf')&data.poor_human]]
        labels=['FIVES 差\n(n=15)','FIVES 其他\n(n=39)','MSHF 差\n(n=10)'] if zh else ['FIVES poor\n(n=15)','FIVES other\n(n=39)','MSHF poor\n(n=10)']
        for ax,key,title in [(axes[0],'q','原质量学生' if zh else 'Original quality student'),(axes[1],'q_eyeq','EyeQ 联合质量学生' if zh else 'EyeQ joint quality student')]:
            if key not in data:ax.set_visible(False);continue
            ax.boxplot([g[key].to_numpy() for g in groups],tick_labels=labels,showfliers=False,widths=.5)
            for i,g in enumerate(groups):
                jitter=np.random.default_rng(4).uniform(-.12,.12,len(g))
                ax.scatter(i+1+jitter,g[key],s=14,alpha=.65,color='#27798e')
            ax.axhline(.4,color='#999',ls='--',lw=.8)
            ax.set(ylabel='原始回归分数' if zh else 'Raw regression score',title=title,ylim=(-.05,1.05))
            ax.spines[['top','right']].set_visible(False)
        fig.savefig(OUT/f'human-quality-scores.{lang}.png',dpi=200);plt.close(fig)

        fig,axes=plt.subplots(1,2,figsize=(10,3.7),constrained_layout=True)
        strata=r['quality_metadata']['strata']['train']
        entries=[strata['fives']['poor'],strata['fives']['other'],strata['mshf']['poor'],strata['mshf']['other']]
        neg=np.array([e['negative'] for e in entries]);pos=np.array([e['positive'] for e in entries]);x=np.arange(4)
        axes[0].bar(x,neg,color='#27798e',label='阴性' if zh else 'Negative')
        axes[0].bar(x,pos,bottom=neg,color='#c47752',label='阳性' if zh else 'Positive')
        for i,e in enumerate(entries):axes[0].text(i,e['negative']+e['positive']+2,f"{e['positive']}+{e['negative']}",ha='center')
        axes[0].set_xticks(x,['FIVES 差','FIVES 其他','MSHF 差','MSHF 其他'] if zh else ['FIVES poor','FIVES other','MSHF poor','MSHF other'],rotation=10)
        axes[0].set(ylabel='训练图像数' if zh else 'Training images',title='柱顶：阳性 + 阴性' if zh else 'Labels: positive + negative',ylim=(0,150));axes[0].legend(fontsize=8)
        official=r['quality_metadata']['original_source_composition']['fives_official']
        values=[official['Train']['N'],official['Test']['N']]
        rates=[100*v['poor']/v['n'] for v in values]
        axes[1].bar([0,1],rates,color=['#27798e','#c47752'])
        for i,v in enumerate(values):axes[1].text(i,rates[i]+1,f"{v['poor']}/{v['n']} ({rates[i]:.1f}%)",ha='center')
        axes[1].set_xticks([0,1],['官方训练部分','官方测试部分'] if zh else ['Official train partition','Official test partition'])
        axes[1].set(ylabel='低质量比例 (%)' if zh else 'Poor-quality proportion (%)',title='FIVES：原始正常眼底图' if zh else 'FIVES: original normal photographs',ylim=(0,43))
        for ax in axes:ax.spines[['top','right']].set_visible(False)
        fig.savefig(OUT/f'quality-label-composition.{lang}.png',dpi=200);plt.close(fig)

        models=r['crossfit_error_ranking']['models']
        names=['uncertainty','low_quality','p_model','pq_model']
        labels=['p 不确定性','仅低 q','p 错误模型','p、q 错误模型'] if zh else ['p uncertainty','Low q only','p error model','p,q error model']
        colors=['#27798e','#c47752','#778165','#aa3f63']
        if 'p_eyeq_model' in models:
            names.append('p_eyeq_model');labels.append('p、EyeQ q' if zh else 'p, EyeQ q');colors.append('#596db0')
        fig,axes=plt.subplots(1,2,figsize=(10,3.7),constrained_layout=True)
        for name,label,color in zip(names,labels,colors):
            entries=[models[name]['review'][str(f)] for f in (.1,.2,.3)]
            axes[0].plot([10,20,30],[e['captured_errors']/208*100 for e in entries],marker='o',label=label,color=color)
            axes[1].plot([10,20,30],[e['remaining_fn'] for e in entries],marker='o',label=label,color=color)
        axes[0].set(ylabel='选中全部错误的比例 (%)' if zh else 'Share of all errors selected (%)',ylim=(0,100))
        axes[1].set(ylabel='未复核图像中的漏诊数' if zh else 'Misses in unreviewed images',ylim=(0,125))
        for ax in axes:
            ax.set_xlabel('复核图像比例 (%)' if zh else 'Review fraction (%)');ax.set_xticks([10,20,30]);ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
        axes[0].legend(fontsize=7.5,loc='lower left',bbox_to_anchor=(0,1.02),ncol=2)
        fig.savefig(OUT/f'review-comparison.{lang}.png',dpi=200);plt.close(fig)


def snapshot():
    status=read(RUN/'pipeline/status.json')
    summaries=[];validation=[];checkpoints=[]
    for directory in sorted(RUN.glob('seed*/*')):
        if not directory.is_dir():continue
        path=directory/'run_summary.json'
        if path.exists():
            summary=read(path);summaries.append(dict(run=directory.relative_to(RUN).as_posix(),**summary))
        epochs=directory/'epochs.jsonl'
        if epochs.exists():
            records=[json.loads(line) for line in epochs.read_text().splitlines() if line.strip()]
            if records:validation.append(dict(run=directory.relative_to(RUN).as_posix(),latest=records[-1]))
        path=directory/'evaluation_summary.json'
        if path.exists():
            evaluation=read(path)
            checkpoints.append(dict(run=evaluation['run'],selected_update=evaluation['selected_update'],
                updates={update:{key:report[key] for key in ('checkpoint_sha256','prediction_sha256','fingerprint')}
                    for update,report in evaluation['checkpoints'].items()}))
    result=dict(captured_at=dt.datetime.now(dt.timezone.utc).isoformat(),pipeline=status,
        completed_training=summaries,latest_validation=validation,evaluated_checkpoint_evidence=checkpoints,
        preflight=read(RUN/'preflight/verification.json') if (RUN/'preflight/verification.json').exists() else None,
        plan=read(RUN/'experiment_plan.json'),comparison=read(RUN/'comparison.json') if (RUN/'comparison.json').exists() else None)
    (OUT/'training-snapshot.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def reports(r,s):
    b=r['roles']['boundary'];rank=r['crossfit_error_ranking'];models=rank['models'];qcheck=r['quality_metadata']['human_quality_score_checks']
    interval=rank['contrasts']['pq_model minus p_model']['review']['0.2']['matched_capacity_group_bootstrap_interval_95']
    review_ci=f'[{interval[0]:+.0f}, {interval[1]:+.0f}]'
    completed=[v for v in s['completed_training'] if v.get('budget_completed')]
    evaluated=s['pipeline']['completed'];current=s['pipeline'].get('current_run','preflight')
    qrows=list(b['quality_bins'].items())
    for lang in ('zh','en'):
        zh=lang=='zh'
        qtable=table(['q 区间','图像数','漏诊 / 阳性','误报 / 阴性','错判率'] if zh else ['q interval','Images','Misses / positives','False alarms / negatives','Error rate'],
            [(name,v['n'],f"{v['fn']}/{v['positive']}",f"{v['fp']}/{v['negative']}",f"{v['error_rate']:.1%}") for name,v in qrows])
        review_names=[('uncertainty','p 不确定性' if zh else 'p uncertainty'),('low_quality','仅低 q' if zh else 'Low q only'),('p_model','p 错误模型' if zh else 'p error model'),('pq_model','p、q 错误模型' if zh else 'p,q error model')]
        if 'p_eyeq_model' in models:review_names.append(('p_eyeq_model','p、EyeQ q' if zh else 'p, EyeQ q'))
        reviewtable=table(['排序方式','错误 AUROC','选中错误 / 208','选中漏诊 / 124','未复核错误 / 图像数'] if zh else ['Ranking','Error AUROC','Errors selected / 208','Misses selected / 124','Unreviewed errors / images'],
            [(label,f"{models[name]['error_auroc']:.4f}",f"{models[name]['review']['0.2']['captured_errors']}/208",f"{models[name]['review']['0.2']['captured_fn']}/124",
              f"{models[name]['review']['0.2']['remaining_errors']}/1642") for name,label in review_names])
        validationtable=table(['运行','已评价更新数','五来源验证 AUROC 均值'] if zh else ['Run','Latest evaluated update','Five-source validation AUROC mean'],
            [(v['run'],v['latest']['optimizer_updates'],f"{v['latest']['selection_value']:.4f}") for v in s['latest_validation']]) if s['latest_validation'] else ('正式训练尚未产生验证快照。' if zh else 'The new research runs have not yet produced a validation snapshot.')
        finaltable='';qualitytable='';training_interpretation=''
        comparison=s['comparison']
        if comparison and comparison['rows']:
            finaltable=table(['运行','更新数','AIROGS','ODIR','Kaggle','LAG','KEH'] if zh else ['Run','Updates','AIROGS','ODIR','Kaggle','LAG','KEH'],
                [(f"{v['seed']} / "+({'original_clean':'原数据','expanded_clean':'扩展无退化','quota65_clean':'65% 配额','expanded_mixed50':'混合退化','quota65_mixed50':'配额混合退化','quota65_defocus50':'配额仅失焦'}[v['arm']] if zh else v['arm'].replace('_',' ')),str(v['update'])+('（选中）' if zh else ' (selected)') if v['selected'] else v['update'],*[f"{v['source_auc'][source]:.4f}" for source in ('airogs-light-v2','DATASET_101/ODIR-5K','Kaggle-Eye_diseases_classification','lag_part1','keh')])
                    for v in comparison['rows'] if v['update'] in (1200,2400) or v['selected']])
            qualityrows=[v for v in comparison['rows'] if v['update']==2400 and 'quality_error_ranking' in v]
            if qualityrows:
                qualitytable=table(['运行','错误数 / 图像数','低 q 错误 AUROC','p 不确定性错误 AUROC'] if zh else ['Run','Errors / images','Low-q error AUROC','p uncertainty error AUROC'],
                    [(f"{v['seed']} / "+({'original_clean':'原数据','expanded_clean':'扩展无退化','quota65_clean':'65% 配额','expanded_mixed50':'混合退化','quota65_mixed50':'配额混合退化','quota65_defocus50':'配额仅失焦'}[v['arm']] if zh else v['arm'].replace('_',' ')),
                      f"{v['quality_error_ranking']['errors']}/{v['quality_error_ranking']['n']}",f"{v['quality_error_ranking']['lowq_error_auc']:.4f}",f"{v['quality_error_ranking']['uncertainty_error_auc']:.4f}") for v in qualityrows])
            ends={v['arm']:v for v in comparison['rows'] if v['seed']==20261115 and v['update']==2400}
            if 'original_clean' in ends and 'expanded_clean' in ends:
                delta={source:ends['expanded_clean']['source_auc'][source]-ends['original_clean']['source_auc'][source]
                    for source in ends['original_clean']['source_auc']}
                training_interpretation=(f"本轮首个种子 20261115 在相同 2,400 次更新下，扩展数据相对原数据的疾病 AUROC 变化为：AIROGS {delta['airogs-light-v2']:+.4f}、LAG {delta['lag_part1']:+.4f}、KEH {delta['keh']:+.4f}、ODIR {delta['DATASET_101/ODIR-5K']:+.4f}。ODIR 的方向与前一轮第 1,200 次更新对照不同，两轮种子与预算也不同，因此它的变化需等待本轮三个共同种子完成后判断。低质量的 12 张开发阳性在本轮原数据对照仍漏诊 5 张，扩展无退化组为零；该组仍没有阴性病例。" if zh else
                    f"For the first seed, 20261115, at the same 2,400-update budget, expansion changes disease AUROC by {delta['airogs-light-v2']:+.4f} on AIROGS, {delta['lag_part1']:+.4f} on LAG, {delta['keh']:+.4f} on KEH, and {delta['DATASET_101/ODIR-5K']:+.4f} on ODIR. ODIR differs in direction from the previous update-1,200 comparison; seeds and budgets also differ between rounds, so its effect should be assessed after all three common seeds complete. The original-data control still misses five of the 12 poor-quality development positives, while expanded clean misses none; this stratum still has no negatives.")
        alternative=''
        if 'p_eyeq_model' in models:
            auc=models['p_eyeq_model']['error_auroc'];check=qcheck['fives']['eyeq_poor_detection_auc'];extra=rank['contrasts']['p_eyeq_model minus p_model']['auc']
            alternative=(f"对同一批 4,099 张开发原图另行运行第 17 轮选中的 EyeQ 联合质量模型后，FIVES 人工质量的区分 AUROC 为 {check:.4f}；加入该质量分数的错误模型 AUROC 为 {auc:.4f}，相对 p 模型差为 {extra['difference']:+.4f}，95% 区间为 [{extra['interval_95'][0]:+.4f}, {extra['interval_95'][1]:+.4f}]。两套质量模型均输出原始回归分数，后文在相同复核量下比较错误和漏诊检出。" if zh else
                f"An additional inference pass uses the EyeQ joint quality checkpoint selected at epoch 17 on the same 4,099 development originals. Its FIVES human-quality AUROC is {check:.4f}. Adding this score gives an error-model AUROC of {auc:.4f}, a difference of {extra['difference']:+.4f} versus the p model, with a 95% interval [{extra['interval_95'][0]:+.4f}, {extra['interval_95'][1]:+.4f}]. Both quality models output raw regression scores; the following comparison examines errors and misses at matched review volume.")
        if zh:
            text=f'''# 训练推进与质量相关错判分析

2026 年 11 月 15 日 · [English](quality-and-training.en.md) · [中文 PDF](quality-and-training.zh.pdf)

当前扩大数据集上的错判分布确实没有呈现出“质量越低，错误越集中”的简单关系。低质量病例不足限制了这一关系的估计。现有证据更直接地显示：真实低质量训练图像几乎全为阳性，质量评分标准在来源之间不同，而且疾病分数本身已包含较强的错误排序信息。现有开发结果尚未支持质量分数在疾病分数之外提供稳定的复核收益。

新一轮六方案、三种子训练已启动，计划每组完成 2,400 次有效优化更新。本文同时记录本轮训练状态和针对上述现象的统计分析，承接 [11 月 13 日实验](../2026-11-13/README.md)。

此前两套疾病模型、原质量模型与 EyeQ 质量模型均已完成 20 轮训练，见 [11 月 12 日记录](../2026-11-12/训练进展报告.md)；11 月 13 日四个固定预算实验也已全部完成。复核边界及最终测试尚待模型和规则确定。

## 数据与 proposal 的关系

[原 proposal](../../../proposal-1106.pdf)提出检验青光眼分数 p 和图像质量分数 q 能否共同定义人工复核区域；它没有预先保证错误必然集中在低 q 一侧。需要检验的是：相同复核量下，加入 q 能否比只用 p 更有效地选中错误和漏诊。

主要分析使用冻结版本 expanded_20261113 的 2,053 张规则开发原图，包括 814 张阳性、1,239 张阴性和 1,917 个关联组。关联组将已确认的双眼、同眼重拍或同图变体共同处理，用于划分和统计抽样。疾病分数来自 50% 混合退化方案第 1,200 次更新的预算结束权重；q 来自此前选中的 FTHNet 蒸馏质量学生：该模型学习既有质量模型的连续评分，数值越大表示预测质量越好。另以 2,046 张模型验证原图检查分布，并在相同 4,099 张原图上评价 EyeQ 质量模型。模型验证集已用于选模，规则开发集用于本次探索，最终保留测试尚未推理。

展示分类以 p 大于 0.5 判阳性。此阈值下，规则开发集有 124/814 张阳性漏诊、84/1,239 张阴性误报。各来源保留自己的疾病定义，例如 AIROGS 的转诊标签与其他来源的诊断标签；来源内结果用于解释差异。

## 分布变化：数量、比例和疾病类别

{qtable}

![质量区间内的图像数量与错判率](quality-counts-rates.zh.png)

图 1 左侧标注每个 q 区间的错误数 / 图像数；右侧为错判率及按关联组进行 2,000 次重采样的 95% 区间。q 小于 0.2 只有 49 张图像，占开发集 2.4%，其中错误 4 张。q 不低于 0.4 的图像占 82.8%，容纳 82.7% 的错误。中高 q 区域错误多，主要与该区域原图多一致；按区间计算的错判率约为 8%–10%，没有持续随 q 增大而下降。

低 q 对阴性误报和阳性漏诊的排序方向也不同：错误排序 AUROC 分别为 0.5950 和 0.4082。高 q 不能排除漏诊。例如 q 在 0.8–1 的区间有 37 张阳性，其中漏诊 10 张。开发集另有 50 个错误的 p 位于两端，即 p 不大于 0.1 或不小于 0.9；其中 41 个的 q 不低于 0.4。质量分数无法单独覆盖这些病例。

这类 AUROC 评价错误识别，含义是随机抽取一张错误图和一张正确图时，排序分数把错误图排在更前面的概率；0.5 表示随机排序。它与疾病分类 AUROC 是两个不同指标。仅按低 q 排序的开发 AUROC 为 0.5092 [0.4703, 0.5476]，模型验证集为 0.5238；原数据对照为 0.5683，扩展无退化为 0.5111，30% 和 50% 退化分别为 0.5154 和 0.5092。变化在扩展无退化中已经出现，不能单独归因于退化增强。

## 真实质量标注与 q 的尺度

![人工质量分层下两套质量模型的原始分数](human-quality-scores.zh.png)

图 2 使用模型验证集和规则开发集合计的 FIVES 54 张及 MSHF 10 张原图。FIVES 以至少一个质量维度为差定义低质量；MSHF 使用官方总体质量。各点是一张原图，箱体为四分位范围，中线为中位数；0.4 虚线只是方便比较数值，不是人工质量等级边界。

FIVES 的 15 张人工低质量图 q 中位数为 0.5261，39 张其他质量图为 0.7008；只有 1/15 张低质量图的 q 小于 0.4。MSHF 的 10 张总体低质量图中位数为 0.2453，9/10 张低于 0.4。FIVES 内部以低 q 区分人工差质量的 AUROC 为 0.9026，因此这些结果更支持“评分尺度和质量定义存在来源差异”，不能把它们概括为质量模型完全失效。全图质量也不直接等于视盘诊断信息的可见程度。

{alternative}

本轮还核对两套质量训练清单与这批疾病开发图像的文件哈希，精确文件交集均为零；这项检查说明没有字节完全相同的训练图，其他已确认关联沿用冻结数据审计。

## 样本不足首先表现为低质量阴性缺失

![质量与疾病标签组成，以及 FIVES 官方划分的差异](quality-label-composition.zh.png)

图 3 左侧使用本次冻结训练清单，柱顶为阳性数 + 阴性数。FIVES 低质量训练图为 44 阳性、3 阴性；其他质量为 46 阳性、83 阴性。MSHF 低质量训练图为 28 阳性、0 阴性，其他质量也只有 2 张阳性。合计低质量训练图 75 张，72 张为阳性。FIVES 内低质量与阳性标签关联的优势比为 26.46：以阳性数除以阴性数得到的阳性优势，在低质量组是其他质量组的约 26 倍。这种组成让模型可能利用质量或来源线索，现有检出结果不能确定它只依据疾病特征。重复采样提高的是既有图像出现次数；此前实测低质量阳性出现 413 次、阴性仅 10 次。

图 3 右侧核对 FIVES 原始官方质量表。正常眼底图的低质量比例在官方训练部分为 5/150，即 3.3%；官方测试部分为 17/50，即 34.0%。这解释了开发数据缺少低质量阴性的部分原因。冻结后的规则开发集有 FIVES 7 张和 MSHF 5 张低质量阳性，没有低质量阴性；模型验证集仅有 1 张 FIVES 低质量阴性。因此，12/12 阳性被检出不能评价这一组的误报，也不足以证明质量鲁棒性已经稳定。

若 12 个独立阳性病例均无漏诊，精确二项分布的单侧 95% 漏诊率上界仍为 22.1%。在无错误的理想条件下，要把该上界压到 5% 以下，至少需 59 个独立病例；阴性误报率需要另有至少 59 个独立阴性病例。若错误率约 10%，希望 95% 区间半宽约 5 个百分点，常规近似需要约 139 个病例。这些数是每个拟评价层的规划参照，真实阳性和阴性应分别计数；关联图像不能当作新增独立病例。

本地 MSHF Healthy 工作表的 26 张图总体质量全部为良好。已有审计中，20 张为新来源内重复，另 6 张同时存在同图标签冲突，全部被排除。因此，本地 Healthy 文件无法直接补充新的低质量阴性。现有 FIVES 测试用途图像应继续用于最后评价。下一轮补样优先寻找同来源、同采集条件、有明确疾病阴性标签且带真实质量标注的病例；其他眼病图不能仅因文件夹未写青光眼就改标为阴性。

## 加入 q 后，复核效果是否改善

为减少在同一批图上拟合和评价边界的偏差，本轮使用五折关联组交叉验证：同一关联组的图像始终在同一折。用其余四折训练低复杂度逻辑回归错误模型，学习分数与错判之间的关系，再为留出折生成错误风险分数。p 模型使用 p 的对数几率、绝对对数几率及平方项，以表示分数朝向阳性还是阴性，以及离 0.5 多远；p、q 模型在此基础上增加 q 和两项交互。标准化只在每折训练部分拟合。疾病模型本身已经固定，这项交叉验证评价的是错误排序模型。

下面在相同 411/2,053 张图，即约 20% 的复核量下比较。仅 p 不确定性按 p 接近 0.5 排序，仅低 q 按 q 从低到高排序；其余为折外预测。并列分数按图像标识固定处理。

{reviewtable}

![不同复核量下选中的错误及未复核漏诊](review-comparison.zh.png)

图 4 分别展示选中的错误比例和留在自动结果中的漏诊数。加入 q 相对同结构 p 模型的错误 AUROC 差仅 +0.00029，按关联组配对重采样的 95% 区间为 [−0.00158, +0.00225]；AUPRC 汇总错误检出的精确率与召回率，分别表示选中图像有多少确实错误，以及全部错误中有多少被选中。其差为 +0.0067 [−0.0142, +0.0277]。在 20% 复核量下多选中 1 个错误；每次病例重采样都重新按分数排序，并为两种方法选中相同复核张数，所得 95% 区间为 {review_ci}。加入来源指标后的增益同样很小，另两个分折种子也未显示明显增益。

EyeQ 分数对应的错误模型在约 20% 复核量下选中 145 个错误，其中 90 个为漏诊。与 p 不确定性相比，总错误数相同，但多选中 4 个漏诊，未复核漏诊由 38 降至 34。这项取舍值得后续检验，当前只是一次开发数据上的结果，不能由它推断稳定优势。

这些结果支持以 p 不确定性作为必要对照，继续检验 q 的额外信息。本轮容量匹配是在规则开发数据上做的探索，并未锁定部署阈值；自助区间固定了已经生成的折外分数，反映病例抽样，不包含重新拟合边界的全部不确定性。表中只计算哪些错误被送去复核，未假定真实专家一定纠正它们。

## 新训练正在检验什么

计划包含原数据无退化、扩展无退化、65% 原来源配额无退化、当前采样加 50% 混合退化、65% 配额加 50% 混合退化、65% 配额加 50% 仅失焦，共六个方案；随机种子为 20261115、20261116、20261117。所有方案均从相同 ImageNet 权重初始化 EfficientNet-B0，采用 320 像素输入、批量最多 16、AdamW 学习率 0.0001，各自训练 2,400 次有效参数更新，不是 20 个完成轮次。一次更新根据一批采样图像的误差调整参数；本实验的一轮按与训练池图像数相同的采样次数计，同一图像可重复出现。

原数据对照使用 10,338 张训练图，其余使用 15,480 张。每 300 次更新，在同一 2,046 张模型验证图上计算结果并保存完整权重和逐图预测。权重按 AIROGS、ODIR、Kaggle、LAG、KEH 的等权 AUROC 均值选择，避免此前四个极小来源占选模均值 36.4% 的情况。保留第 1,200 和 2,400 次更新，区分预算结束与选中权重。

65% 配额在原数据和新增数据内部继续使用来源与类别计数的平方根倒数采样。配额分散到连续的采样位置，截断训练预算也保持约 65% 原来源占比。上一轮原来源图像的出现次数由 19,186 降至 9,484。该设计可以检验恢复原来源出现次数是否改善 AIROGS 和 ODIR；它没有增加真实低质量病例数。仅失焦方案与混合方案使用相同应用概率和采样序列，以区分退化机制。

状态快照时间为 {s['captured_at']}。已完成训练预算 {len(completed)}/18 组，已完成开发评价 {len(evaluated)}/18 组，当前为 `{current}`，驱动状态为 `{s['pipeline']['status']}`。训练使用隐藏后台进程，结束当前对话后继续执行；断点恢复核对输入指纹。

{validationtable}

下表为已完成方案在规则开发原图上的疾病分类 AUROC：AIROGS 773 张（380 阳性）、ODIR 305 张（28 阳性）、Kaggle 142 张（95 阳性）、LAG 481 张（169 阳性）、KEH 153 张（75 阳性）。标为“选中”的权重由模型验证集决定。

{finaltable}

{training_interpretation}

另在第 2,400 次更新的疾病权重上，用同一批规则开发图像及原质量学生分数复查错误排序，结果如下。这里比较低 q 和 p 不确定性两个单独排序，未重新拟合 p、q 边界。

{qualitytable}

这里的最新验证行表示相应更新时的结果；已有开发 AUROC 表按来源、按保存的更新权重报告。三种子全部完成后，才汇总种子间差异并判断采样及退化收益。最终保留测试仍等待模型、分类阈值和复核规则确定。

## 复现与证据

[统计数据](quality-error-analysis.json) · [训练快照](training-snapshot.json) · [分析程序](../../../tools/analyze_quality_errors_20261115.py) · [报告程序](../../../tools/report_quality_training_20261115.py) · [训练入口](../../../tools/start_followup_experiments_20261115.ps1) · [实时执行状态](../../../runs/followup_20261115/pipeline/status.json)

逐图连接与折外风险分数保存在本地 `runs/followup_20261115/error_ranking/`；质量分数推理、计划、每次验证权重和训练日志保存在新运行目录。报告数据记录输入文件 SHA256。中英文 Markdown、PDF、LaTeX 与图表放在同一日期目录，历史实验保持原样。
'''
        else:
            text=f'''# Training continuation and quality-related errors

November 15, 2026 · [中文](quality-and-training.zh.md) · [English PDF](quality-and-training.en.pdf)

Errors on the expanded dataset do not follow a simple pattern of increasing concentration at lower quality. Too few real poor-quality cases contribute to the uncertainty, but stronger direct evidence concerns their disease-label imbalance, differences between source quality standards, and substantial error-ranking information already present in disease scores. Current development results do not establish a stable additional review benefit from quality scores.

Six training conditions with three common seeds are now running, with 2,400 successful optimizer updates planned per run. This report records their status and the analysis of quality-related errors, following the [November 13 experiments](../2026-11-13/README.en.md).

Both earlier disease models, the original quality model, and the EyeQ quality model completed 20 epochs, as recorded in the [November 12 report](../2026-11-12/training-progress.en.md). All four November 13 fixed-budget experiments are also complete. Review boundaries and final testing await model and rule selection.

## Data and the proposal

The [original proposal](../../../proposal-1106.pdf) asks whether glaucoma score p and quality score q can jointly define a review region. It does not guarantee that errors will cluster at low q. The relevant test is whether adding q selects more errors and misses than p alone at the same review volume.

The primary analysis uses 2,053 rule-development originals from frozen release expanded_20261113: 814 positives, 1,239 negatives, and 1,917 linked groups. Linked groups keep confirmed paired eyes, repeat photographs, and related image versions together for split assignment and statistical resampling. Disease scores come from the update-1,200 budget-end checkpoint of the 50% mixed-degradation arm. Quality scores come from the selected FTHNet-distilled student checkpoint. This student is trained to reproduce an existing quality model’s continuous scores, with larger values indicating better predicted quality. A separate 2,046-image model-validation set checks the distribution; the EyeQ quality model is also evaluated on these same 4,099 originals. Model validation has already informed checkpoint selection, and rule development supports this exploratory analysis. Held-out final tests have not been scored.

The display decision calls an image positive when p exceeds 0.5. This yields 124/814 missed positives and 84/1,239 false alarms on rule development. Sources retain their own disease definitions, including AIROGS referral labels and other sources' diagnosis labels; within-source results explain differences.

## Counts, rates, and disease classes

{qtable}

![Image counts and error rates within quality-score intervals](quality-counts-rates.en.png)

Figure 1 labels errors / images on the left and shows error rates with 95% intervals from 2,000 linked-group bootstrap samples on the right. Only 49 images, 2.4% of rule development, have q below 0.2; four are errors. Images at q of at least 0.4 account for 82.8% of photographs and 82.7% of errors. More errors in the middle and upper quality ranges largely follow their greater image counts. Interval-specific error rates are about 8%–10%, without a sustained decline as q rises.

Low q also ranks false alarms and misses in different directions: their error-ranking AUROCs are 0.5950 and 0.4082, respectively. High q does not exclude a miss; the q interval 0.8–1 contains 37 positives, with ten missed. Fifty development errors have p at either extreme, defined as p at most 0.1 or at least 0.9; 41 have q of at least 0.4. Quality alone cannot cover these cases.

Error-ranking AUROC means the probability of assigning higher risk to a randomly selected error than to a randomly selected correct prediction; 0.5 means random ranking. It differs from disease-classification AUROC. Ranking by low q gives development AUROC 0.5092 [0.4703, 0.5476] and model-validation AUROC 0.5238. The original-data control gives 0.5683, expanded clean 0.5111, 30% degradation 0.5154, and 50% degradation 0.5092. The change already appears without augmentation, so degradation alone does not explain it.

## Human quality grades and score scales

![Raw scores from two quality models, grouped by source human grades](human-quality-scores.en.png)

Figure 2 uses 54 FIVES and ten MSHF photographs across model validation and rule development. FIVES defines poor quality as at least one poor dimension; MSHF uses official overall quality. Dots are photographs, boxes span quartiles, and the central line is the median. The dashed 0.4 line aids numerical comparison and is not a human-grade threshold.

The original student's median q is 0.5261 for 15 poor-quality FIVES images and 0.7008 for 39 other-quality images. Only 1/15 poor images fall below 0.4. The ten poor-overall MSHF images have median 0.2453, with 9/10 below 0.4. Within FIVES, low q distinguishes poor human quality with AUROC 0.9026. These findings support differences in source score scales and quality definitions rather than complete quality-model failure. Whole-image quality also differs from visibility of the optic-disc information needed for diagnosis.

{alternative}

File hashes of both quality training cohorts were compared with these disease development photographs: both exact-file intersections are zero. This excludes byte-identical training photographs; other confirmed relationships follow the frozen association audit.

## The shortage primarily concerns poor-quality negatives

![Disease-label composition by quality and the original FIVES partition difference](quality-label-composition.en.png)

Figure 3 uses the frozen training manifest on the left; labels above bars are positive + negative counts. Poor-quality FIVES training contains 44 positives and three negatives, while other quality contains 46 positives and 83 negatives. MSHF contributes 28 poor-quality positives and no poor-quality negatives; its two other-quality images are also positive. Of 75 poor-quality training photographs, 72 are positive. The FIVES odds ratio associating poor quality with a positive label is 26.46: positive-to-negative odds are about 26 times larger among poor than other-quality training photographs. This composition may let the model use source or quality cues; the current detections do not establish recognition based solely on disease features. Repeated sampling increases presentations of existing photographs: the previous run presented real poor-quality positives 413 times and negatives only ten times.

The right panel checks the original official FIVES quality workbook. Among normal photographs, poor-quality prevalence is 5/150, or 3.3%, in its official training partition, versus 17/50, or 34.0%, in its test partition. This partly explains the shortage of poor-quality negatives in development. The frozen rule-development set has seven poor FIVES positives and five poor MSHF positives, with no poor negatives; model validation contains only one poor FIVES negative. Detecting 12/12 positives therefore cannot estimate false alarms in this stratum or establish stable quality robustness.

If 12 independent positive cases have zero misses, the exact one-sided 95% upper bound on miss rate remains 22.1%. Under the ideal zero-error condition, at least 59 independent cases are needed to bring that bound below 5%; false-alarm precision requires a separate set of at least 59 independent negatives. At an error rate around 10%, the usual approximation requires about 139 cases for a 95% interval half-width near five percentage points. These are planning references for each evaluated stratum, with positives and negatives counted separately; linked photographs do not add independent cases.

All 26 photographs in the local MSHF Healthy worksheet have good overall quality. The earlier audit excluded 20 as within-acquisition duplicates and six as duplicates with conflicting identical-image labels. They cannot directly provide new poor-quality negatives. Existing FIVES test-role images remain reserved for final evaluation. Additional cases should prioritize the same sources and acquisition conditions, reliable negative disease labels, and real human quality annotations. Other-disease folders do not establish glaucoma-negative status.

## Does q improve review selection?

To reduce fitting and assessing a boundary on the same images, this analysis uses five-fold linked-group cross-validation. All images from a linked group stay in one fold. A low-complexity logistic error model is trained on four folds and generates error-risk predictions for the held-out fold. The p model uses log-odds of p, absolute log-odds, and a squared term. The p,q model adds q and two interactions. Scaling is fitted within each training fold. The disease model is already fixed; this cross-validation evaluates the error-ranking model.

The following comparison selects the same 411/2,053 images, about 20%, for review. p uncertainty prioritizes scores nearest 0.5; low q prioritizes the smallest q. The fitted models use out-of-fold predictions. Image identifiers deterministically resolve tied scores.

{reviewtable}

![Errors selected and misses left unreviewed at different review volumes](review-comparison.en.png)

Figure 4 shows the share of errors selected and misses remaining among automatic outputs. Adding q changes error AUROC by only +0.00029 versus the corresponding p model, with a paired linked-group 95% interval [−0.00158, +0.00225]. AUPRC summarizes error-selection precision and recall: how many selected images are errors, and how many of all errors are selected. It changes by +0.0067 [−0.0142, +0.0277]. At 20% review, q selects one additional error. Each case bootstrap reranks the scores and selects the same number of images with both methods, giving a 95% interval {review_ci}. Adding source indicators also yields a small change, and two other fold-assignment seeds show no material improvement.

The error model using EyeQ scores selects 145 errors at about 20% review, including 90 misses. Compared with p uncertainty, it selects the same number of total errors but four additional misses, reducing unreviewed misses from 38 to 34. This tradeoff warrants further testing; it is a development-data result and does not yet establish a stable advantage.

These results establish p uncertainty as a necessary comparator while testing q's additional information. Capacity matching is exploratory on rule development and does not lock a deployment threshold. Bootstrap intervals hold the generated out-of-fold predictions fixed, reflecting case sampling rather than all uncertainty from refitting a boundary. The table counts errors sent for review without assuming that real reviewers always correct them.

## What the new training tests

The six conditions are original-data clean, expanded clean, 65% original-origin quota clean, current sampling with 50% mixed degradation, 65% quota with 50% mixed degradation, and 65% quota with 50% defocus-only degradation. Common seeds are 20261115, 20261116, and 20261117. Each EfficientNet-B0 starts from the same ImageNet weights and uses 320-pixel input, batches up to 16, AdamW learning rate 0.0001, and 2,400 successful parameter updates. This budget does not mean 20 completed epochs. One successful update adjusts parameters using one sampled batch. An epoch draws as many images as the training pool contains, with repeats permitted by the sampler.

The original-data control uses 10,338 training photographs; the other conditions use 15,480. Every 300 updates, the same 2,046 model-validation photographs are evaluated and full checkpoints plus image predictions are saved. Selection uses the equal AUROC mean over AIROGS, ODIR, Kaggle, LAG, and KEH, reducing the earlier influence of four tiny sources carrying 36.4% of the selection mean. Update-1,200 and update-2,400 weights remain available separately from the selected checkpoint.

Within original and added origin pools, the 65% quota retains inverse-square-root source-class sampling. The quota is distributed through consecutive draw positions, so truncated update budgets retain approximately 65% original-image presentations. The previous round reduced original-source presentations from 19,186 to 9,484. This tests whether restoring original-source presentations improves AIROGS and ODIR; it adds no real poor-quality cases. Defocus-only and mixed augmentation share application probability and image sampling sequence to separate mechanisms.

The status snapshot was captured at {s['captured_at']}. Training budgets are complete for {len(completed)}/18 runs and development evaluation for {len(evaluated)}/18. The current run is `{current}`, and the driver state is `{s['pipeline']['status']}`. A hidden background process continues after this conversation; checkpoint recovery verifies input fingerprints.

{validationtable}

The following disease-classification AUROCs use the same rule-development originals: AIROGS 773 (380 positives), ODIR 305 (28), Kaggle 142 (95), LAG 481 (169), and KEH 153 (75). The checkpoint marked selected was chosen on model validation.

{finaltable}

{training_interpretation}

A further check uses the update-2,400 disease weights, the same rule-development photographs, and the original quality student scores to rank errors. The following compares low q and p uncertainty separately without refitting a p,q boundary.

{qualitytable}

Latest validation rows report their corresponding saved update. Available development AUROCs are shown by source and update checkpoint. Variation across training seeds and sampling or degradation conclusions will be summarized after all three seeds finish. Final held-out testing waits until models, classification thresholds, and review rules are fixed.

## Reproduction and evidence

[Analysis data](quality-error-analysis.json) · [Training snapshot](training-snapshot.json) · [Analysis program](../../../tools/analyze_quality_errors_20261115.py) · [Report generator](../../../tools/report_quality_training_20261115.py) · [Training launcher](../../../tools/start_followup_experiments_20261115.ps1) · [Live execution status](../../../runs/followup_20261115/pipeline/status.json)

Image-level joins and out-of-fold risks are stored locally under `runs/followup_20261115/error_ranking/`. The new run directory holds quality inference evidence, the plan, every validation checkpoint, and training logs. Report data retain input SHA256 hashes. Chinese and English Markdown, PDF, LaTeX, and figures share this dated directory; historical experiments remain intact.
'''
        (OUT/f'quality-and-training.{lang}.md').write_text(text,encoding='utf-8')


def compile_and_render():
    BUILD.mkdir(parents=True,exist_ok=True);evidence={}
    for lang in ('zh','en'):
        stem=f'quality-and-training.{lang}';tex=OUT/f'{stem}.tex'
        subprocess.run(['pandoc',str(OUT/f'{stem}.md'),'-s','-t','latex','-o',str(tex),
            '-V','mainfont=Arial','-V','CJKmainfont=Microsoft YaHei','-V','monofont=Consolas',
            '-V','fontsize=10pt','-V','geometry:margin=0.7in','-V','colorlinks=true','-V','urlcolor=blue'],cwd=OUT,check=True,capture_output=True)
        text=tex.read_text(encoding='utf-8').replace('\\begin{figure}','\\begin{figure}[H]')
        text=text.replace('\\real{0.1429}','\\real{0.14285}')
        text=text.replace('\\begin{longtable}','\\Needspace{9\\baselineskip}\n\\begin{longtable}')
        prefix='\\usepackage{float}\n\\usepackage{needspace}\n\\setCJKmonofont{Microsoft YaHei}\n\\Urlmuskip=0mu plus 2mu\\relax\n'+('\\renewcommand{\\figurename}{图}\n' if lang=='zh' else '')
        tex.write_text(text.replace('\\begin{document}',prefix+'\\begin{document}'),encoding='utf-8')
        for index in range(2):
            process=subprocess.run(['xelatex','-interaction=nonstopmode','-halt-on-error',f'-output-directory={BUILD}',str(tex)],cwd=OUT,capture_output=True)
            (BUILD/f'{stem}-build-{index}.log').write_bytes(process.stdout+process.stderr);process.check_returncode()
        shutil.copyfile(BUILD/f'{stem}.pdf',OUT/f'{stem}.pdf')
        import pymupdf
        doc=pymupdf.open(OUT/f'{stem}.pdf');pages=[]
        for index,page in enumerate(doc):
            path=BUILD/f'page-{lang}-{index+1:02d}.png';page.get_pixmap(matrix=pymupdf.Matrix(1.6,1.6)).save(path);pages.append(path)
        for start in range(0,len(pages),4):
            sheet=Image.new('RGB',(1800,2460),'#dddddd');draw=ImageDraw.Draw(sheet)
            for offset,path in enumerate(pages[start:start+4]):
                page=ImageOps.contain(Image.open(path).convert('RGB'),(880,1180));x=(offset%2)*900+(900-page.width)//2;y=(offset//2)*1230+30
                sheet.paste(page,(x,y));draw.text((x,y-20),f'{lang.upper()} page {start+offset+1}',fill='black')
            sheet.save(BUILD/f'contact-{lang}-{start//4+1}.jpg',quality=95)
        evidence[lang]=dict(pages=len(doc),pdf_bytes=(OUT/f'{stem}.pdf').stat().st_size,latex_retained=True,
            all_pages_rendered=True,layout_warnings=[line for line in (BUILD/f'{stem}-build-1.log').read_text(errors='replace').splitlines() if 'Overfull' in line or 'Missing character' in line])
    (OUT/'render-verification.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8');print(json.dumps(evidence,indent=2))


def main():
    OUT.mkdir(parents=True,exist_ok=True);r=read(OUT/'quality-error-analysis.json');s=snapshot()
    charts(r);reports(r,s);compile_and_render()


if __name__=='__main__':main()
