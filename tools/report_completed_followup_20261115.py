"""Render the completed experiment report, bilingual plots, and every PDF page."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageOps
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'docs/progress/2026-11-15/completed-experiments'
BUILD = ROOT/'.build/completed-followup-20261115'
RUN = ROOT/'runs/followup_20261115'
ARMS = ('original_clean','expanded_clean','quota65_clean','expanded_mixed50','quota65_mixed50','quota65_defocus50')
SOURCES = ('airogs-light-v2','DATASET_101/ODIR-5K','Kaggle-Eye_diseases_classification','lag_part1','keh')
SOURCE_NAMES = ('AIROGS','ODIR','Kaggle','LAG','KEH')
LABELS = {
    'zh': ['原数据','扩展数据','65% 原来源','扩展＋混合退化','65% 原来源＋混合退化','65% 原来源＋仅失焦'],
    'en': ['Original data','Expanded data','65% original sources','Expanded + mixed','65% original + mixed','65% original + defocus']}
COLORS = ['#7c8185','#23788d','#4b8561','#cf8352','#965c85','#5268a9']
FIGURES = ('source-auroc','paired-source-effects','validation-curves','stress-and-training-dose',
           'misses-and-false-alarms','quality-error-rates','quality-review-effects','disease-quality-distribution',
           'real-quality-composition','review-workload')
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','Arial'],'axes.unicode_minus':False,'font.size':9})


def save(fig,name,lang):
    fig.savefig(OUT/f'{name}.{lang}.png',dpi=200,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def charts(r):
    for lang in ('zh','en'):
        zh=lang=='zh'; labels=LABELS[lang]
        fig,axes=plt.subplots(1,2,figsize=(11.1,4.15),constrained_layout=True)
        for ax,mode,title in zip(axes,('selected','budget2400'),
            ('验证集选中权重','第 2,400 次更新权重') if zh else ('Validation-selected checkpoints','Checkpoints at update 2,400')):
            matrix=np.array([[r['aggregate'][mode][a]['source_auc'][s]['mean'] for s in SOURCES] for a in ARMS])
            ax.imshow(matrix,vmin=.55,vmax=1,cmap='YlGnBu',aspect='auto')
            ax.set_xticks(range(5),SOURCE_NAMES);ax.set_yticks(range(6),labels if ax is axes[0] else ['']*6)
            ax.set_title(title)
            for i in range(6):
                for j in range(5):ax.text(j,i,f'{matrix[i,j]:.4f}',ha='center',va='center',fontsize=8,color='white' if matrix[i,j]>.85 else '#222222')
        save(fig,'source-auroc',lang)

        fig,axes=plt.subplots(1,3,figsize=(11.1,4.0),constrained_layout=True)
        names=['扩展 − 原数据','65% 原来源 − 扩展','混合退化 − 扩展','65%：混合 − 无退化','65%：仅失焦 − 混合'] if zh else ['Expansion − original','65% original − expanded','Mixed − expanded','65%: mixed − clean','65%: defocus − mixed']
        for ax,source,title in zip(axes,(SOURCES[0],SOURCES[1],SOURCES[4]),('AIROGS','ODIR','KEH')):
            for i,contrast in enumerate(r['contrasts']['selected']):
                v=contrast['source'][source];lo,hi=v['case_interval95'];mean=v['mean']
                ax.plot([lo,hi],[i,i],color='#305d75',lw=2)
                ax.scatter(v['values'],[i-.1,i,i+.1],color='#9e5961',s=20,zorder=3)
                ax.scatter([mean],[i],color='#1b4c64',s=32,marker='D',zorder=4)
            ax.axvline(0,color='#888888',ls='--',lw=1);ax.set_title(title);ax.invert_yaxis()
            ax.set_yticks(range(5),names if ax is axes[0] else ['']*5)
            ax.set_xlabel('疾病 AUROC 变化' if zh else 'Change in disease AUROC')
            ax.spines[['top','right']].set_visible(False)
        save(fig,'paired-source-effects',lang)

        curves=pd.DataFrame(r['training_curves'])
        fig,axes=plt.subplots(1,2,figsize=(11.1,3.55),constrained_layout=True)
        for ax,key,title in zip(axes,('selection_value','original_source_mean'),
            ('用于选模的五来源平均值','原三来源平均值：AIROGS、ODIR、Kaggle') if zh else ('Five-source mean used for selection','Original three sources: AIROGS, ODIR, Kaggle')):
            for arm,label,color in zip(ARMS,labels,COLORS):
                sub=curves[curves.arm.eq(arm)]; pivot=sub.pivot(index='update',columns='seed',values=key)
                for seed in pivot:ax.plot(pivot.index,pivot[seed],color=color,alpha=.18,lw=.8)
                ax.plot(pivot.index,pivot.mean(axis=1),color=color,label=label,lw=1.7)
            ax.set(title=title,xlabel='参数更新次数' if zh else 'Parameter updates',ylabel='模型验证集 AUROC 均值' if zh else 'Mean model-validation AUROC')
            ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False);ax.set_xticks([300,900,1500,2100,2400])
        fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1.14),ncol=3,fontsize=8)
        save(fig,'validation-curves',lang)

        fig,axes=plt.subplots(1,2,figsize=(11.1,4.15),constrained_layout=True)
        matrix=np.array([[r['aggregate']['selected'][a]['overall']['auc']['mean'],
            r['aggregate']['selected'][a]['stress']['defocus_2_at512']['auc']['mean'],
            r['aggregate']['selected'][a]['stress']['exposure_070']['auc']['mean']] for a in ARMS])
        axes[0].imshow(matrix,vmin=.90,vmax=.985,cmap='YlGnBu',aspect='auto')
        axes[0].set_xticks(range(3),['原图','合成失焦','亮度 ×0.70'] if zh else ['Original','Synthetic defocus','Brightness ×0.70'])
        axes[0].set_yticks(range(6),labels);axes[0].set_title('选中权重的疾病 AUROC' if zh else 'Disease AUROC: selected checkpoints')
        for i in range(6):
            for j in range(3):axes[0].text(j,i,f'{matrix[i,j]:.4f}',ha='center',va='center',color='white' if matrix[i,j]>.945 else '#222222')
        origin=np.zeros(6)
        for family,color,label in zip(('original','defocus','illumination','scatter'),('#d1d7d9','#516ca4','#d6a357','#91759b'),
            ['无退化','失焦','照明','散射'] if zh else ['Unmodified','Defocus','Illumination','Scatter']):
            values=[]
            for arm in ARMS:
                selected=[v for v in r['rows'] if v['arm']==arm and v['mode']=='budget2400']
                values.append(np.mean([v['family_draws'].get(family,0)/sum(v['family_draws'].values())*100 for v in selected]))
            axes[1].barh(range(6),values,left=origin,color=color,label=label);origin+=values
        axes[1].set_yticks(range(6),['']*6);axes[1].invert_yaxis();axes[1].set_xlim(0,100)
        axes[1].set(xlabel='训练图像出现次数占比 (%)' if zh else 'Share of training image presentations (%)',title='2,400 次更新的实际增强比例' if zh else 'Actual augmentation exposure at 2,400 updates')
        axes[1].legend(loc='upper center',bbox_to_anchor=(.5,1.27),ncol=2,fontsize=8)
        save(fig,'stress-and-training-dose',lang)

        fig,axes=plt.subplots(1,2,figsize=(11.1,4.2),constrained_layout=True)
        for ax,key,title in zip(axes,('fn','fp'),('漏诊：814 张阳性','误报：1,239 张阴性') if zh else ('Misses among 814 positives','False alarms among 1,239 negatives')):
            for mode,offset,color,label in [('selected',-.17,'#30798e','选中权重' if zh else 'Selected'),('budget2400',.17,'#c17c50','2,400 次更新' if zh else 'Update 2,400')]:
                means=[r['aggregate'][mode][a]['overall'][key]['mean'] for a in ARMS]
                ax.barh(np.arange(6)+offset,means,height=.31,color=color,label=label)
                for i,a in enumerate(ARMS):ax.scatter(r['aggregate'][mode][a]['overall'][key]['values'],[i+offset]*3,s=16,color='#333333',zorder=3)
            ax.set_title(title);ax.set_yticks(range(6),labels if ax is axes[0] else ['']*6);ax.invert_yaxis()
            ax.set_xlabel('图像数（分类阈值 p > 0.5）' if zh else 'Images (positive when p > 0.5)');ax.spines[['top','right']].set_visible(False)
        axes[0].legend(loc='lower right',fontsize=8)
        save(fig,'misses-and-false-alarms',lang)

        fig,axes=plt.subplots(1,2,figsize=(11.1,3.7),constrained_layout=True)
        for ax,key,title in zip(axes,('fn_rate','fp_rate'),('阳性图中的漏诊率','阴性图中的误报率') if zh else ('Miss rate among positives','False-alarm rate among negatives')):
            for arm,label,color in zip(ARMS,labels,COLORS):
                chosen=[v for v in r['quality_review_runs'] if v['arm']==arm]
                means=np.mean([[b[key] for b in v['quality_bins']] for v in chosen],axis=0)*100
                ax.plot(range(5),means,marker='o',color=color,label=label,lw=1.6,ms=4)
            denominators=[b['positive' if key=='fn_rate' else 'negative'] for b in chosen[0]['quality_bins']]
            ax.set_xticks(range(5),[f'{name}\nn={n}' for name,n in zip(['0–0.2','0.2–0.4','0.4–0.6','0.6–0.8','0.8–1'],denominators)],fontsize=8)
            ax.set(title=title,xlabel='质量分数 q 区间及病例数' if zh else 'Quality-score q interval and images',ylabel='比例 (%)' if zh else 'Rate (%)',ylim=(0,45))
            ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
        fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1.13),ncol=3,fontsize=8)
        save(fig,'quality-error-rates',lang)

        fig,axes=plt.subplots(1,2,figsize=(11.1,4.5),constrained_layout=True)
        for index,arm in enumerate(ARMS):
            for key,offset,color,label in [('pq minus p',-.12,'#30798e','FTHNet q'),('p_eyeq minus p',.12,'#965c85','EyeQ q')]:
                v=r['quality_review_summary'][arm]['contrasts'][key]
                for ax,metric,interval in zip(axes,('auc','capture'),('auc_case_interval95','capture_case_interval95')):
                    lo,hi=v[interval];y=index+offset
                    ax.plot([lo,hi],[y,y],color=color,lw=2)
                    ax.scatter(v[metric]['values'],[y-.025,y,y+.025],s=20,color=color,alpha=.8)
                    ax.scatter([v[metric]['mean']],[y],s=32,marker='D',color=color,label=label if index==0 else None)
        for ax in axes:
            ax.axvline(0,color='#999999',ls='--');ax.set_yticks(range(6),labels if ax is axes[0] else ['']*6);ax.invert_yaxis();ax.spines[['top','right']].set_visible(False)
        axes[0].set_xlabel('加入 q 后的错误识别 AUROC 变化' if zh else 'Change in error-ranking AUROC after adding q')
        axes[1].set_xlabel('复核 411 张图时额外选中的错误数' if zh else 'Additional errors selected at 411-image review')
        axes[0].legend(loc='upper center',bbox_to_anchor=(.5,1.2),ncol=2,fontsize=8)
        save(fig,'quality-review-effects',lang)

        candidate=max(ARMS[1:],key=lambda a:r['aggregate']['selected'][a]['validation_selection']['mean'])
        frame=pd.read_csv(RUN/'completed_analysis'/f'20261115-{candidate}-out-of-fold.csv')
        fig,axes=plt.subplots(1,2,figsize=(11.1,3.75),constrained_layout=True)
        for ax,label,title in zip(axes,(1,0),('阳性图：红色为漏诊','阴性图：红色为误报') if zh else ('Positive images: red denotes misses','Negative images: red denotes false alarms')):
            sub=frame[frame.label.eq(label)];correct=sub[~sub.error];wrong=sub[sub.error]
            ax.scatter(correct.prediction,correct.q,s=8,alpha=.24,color='#6e8c99',label='正确' if zh else 'Correct')
            ax.scatter(wrong.prediction,wrong.q,s=19,alpha=.9,color='#b74240',label='错误' if zh else 'Error')
            ax.axvline(.5,color='#888888',ls='--');ax.axhline(.4,color='#aaaaaa',ls=':',lw=.8)
            ax.set(title=title,xlabel='疾病分数 p' if zh else 'Disease score p',ylabel='质量分数 q' if zh else 'Quality score q',xlim=(-.02,1.02),ylim=(-.02,1.02))
            ax.legend(loc='upper center',bbox_to_anchor=(.5,1.26),ncol=2,fontsize=8);ax.spines[['top','right']].set_visible(False)
        save(fig,'disease-quality-distribution',lang)

        previous=json.loads((OUT.parent/'quality-error-analysis.json').read_text(encoding='utf-8'))
        fig,axes=plt.subplots(1,3,figsize=(11.1,3.8),constrained_layout=True)
        strata=previous['quality_metadata']['strata']
        names=['训练','模型验证','规则开发'] if zh else ['Training','Model validation','Rule development']
        pos=[];neg=[]
        for role in ('train','val','boundary'):
            pos.append(sum(strata[role][s]['poor']['positive'] for s in ('fives','mshf')))
            neg.append(sum(strata[role][s]['poor']['negative'] for s in ('fives','mshf')))
        axes[0].bar(range(3),pos,color='#c07b50',label='阳性' if zh else 'Positive')
        axes[0].bar(range(3),neg,bottom=pos,color='#30798e',label='阴性' if zh else 'Negative')
        for i in range(3):axes[0].text(i,pos[i]+neg[i]+1,f'{pos[i]} + {neg[i]}',ha='center',fontsize=8)
        axes[0].set_xticks(range(3),names,rotation=8);axes[0].set_title('人工低质量图：阳性 + 阴性' if zh else 'Human-rated poor: positive + negative')
        axes[0].set_ylabel('图像数' if zh else 'Images');axes[0].set_ylim(0,90);axes[0].legend(fontsize=8)
        scores=pd.read_csv(RUN/'error_ranking/development_join.csv')
        strata_frames=[scores[scores.source.eq('fives')&scores.poor_human],scores[scores.source.eq('fives')&~scores.poor_human],scores[scores.source.eq('mshf')&scores.poor_human]]
        tick=['FIVES 差\n15 张','FIVES 其他\n39 张','MSHF 差\n10 张'] if zh else ['FIVES poor\n15 images','FIVES other\n39 images','MSHF poor\n10 images']
        for ax,key,title in zip(axes[1:],('q','q_eyeq'),('FTHNet 质量分数','EyeQ 质量分数') if zh else ('FTHNet quality scores','EyeQ quality scores')):
            ax.boxplot([g[key] for g in strata_frames],tick_labels=tick,showfliers=False,widths=.5)
            for i,g in enumerate(strata_frames):ax.scatter(i+1+np.random.default_rng(4).uniform(-.12,.12,len(g)),g[key],s=12,color='#30798e',alpha=.6)
            ax.axhline(.4,color='#aaaaaa',ls=':');ax.set_title(title);ax.set_ylim(-.05,1.05)
        for ax in axes:ax.spines[['top','right']].set_visible(False)
        save(fig,'real-quality-composition',lang)

        chosen=[v for v in r['quality_review_runs'] if v['arm']==candidate]
        model_names=('uncertainty','low_q','p','pq','p_eyeq')
        names=['p 接近 0.5','仅低 q','仅 p 拟合','p 与 FTHNet q','p 与 EyeQ q'] if zh else ['p near 0.5','Low q only','Fitted p only','p and FTHNet q','p and EyeQ q']
        fig,axes=plt.subplots(1,2,figsize=(11.1,3.75),constrained_layout=True)
        for model,label,color in zip(model_names,names,('#23788d','#cf8352','#7c8185','#965c85','#5268a9')):
            for ax,key in zip(axes,('captured_errors','remaining_fn')):
                vals=np.array([[v['models'][model]['review'][str(f)][key] for f in (.1,.2,.3)] for v in chosen])
                ax.plot([10,20,30],vals.mean(axis=0),marker='o',label=label,color=color,ms=4)
                ax.fill_between([10,20,30],vals.min(axis=0),vals.max(axis=0),color=color,alpha=.08)
        axes[0].set_ylabel('选中错误的图像数' if zh else 'Images with errors selected')
        axes[1].set_ylabel('未复核阳性图中的漏诊数' if zh else 'Misses among unreviewed positives')
        for ax in axes:
            ax.set_xlabel('复核图像比例 (%)' if zh else 'Fraction of images reviewed (%)');ax.set_xticks([10,20,30]);ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
        fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1.13),ncol=3,fontsize=8)
        save(fig,'review-workload',lang)
    return candidate


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',
                      *['| '+' | '.join(map(str,row))+' |' for row in rows]])


def reports(r,candidate):
    selected=r['aggregate']['selected'];final=r['aggregate']['budget2400']
    contrast=r['contrasts']['selected']; qsummary=r['quality_review_summary']
    qdelta=qsummary[candidate]['contrasts']['pq minus p']
    best=max([v for v in r['rows'] if v['mode']=='selected' and v['arm']==candidate],key=lambda v:v['validation_selection'])
    original_draws=np.mean([v['origin_draws']['formal_v3'] for v in r['rows'] if v['mode']=='budget2400' and v['arm']=='expanded_clean'])
    budget_fn_delta=final['quota65_clean']['overall']['fn']['mean']-final['expanded_clean']['overall']['fn']['mean']
    budget_fp_delta=final['quota65_clean']['overall']['fp']['mean']-final['expanded_clean']['overall']['fp']['mean']
    short={'zh':['原数据','扩展数据','65% 原来源','扩展＋混合','65%＋混合','65%＋失焦'],
           'en':['Original','Expanded','65% original','Expanded + mixed','65% + mixed','65% + defocus']}
    for lang in ('zh','en'):
        zh=lang=='zh';labels=short[lang]
        format_value=lambda value:f"{value['mean']:.4f} ± {value['sd']:.4f}"
        selection_table=table(['方案','选中更新数（三种子）','验证均值','开发均值','漏诊','误报','Brier'] if zh else
            ['Condition','Selected updates (3 seeds)','Validation mean','Development mean','Misses','False alarms','Brier'],
            [(label,' / '.join(map(str,selected[a]['updates'])),f"{selected[a]['validation_selection']['mean']:.4f}",
              f"{selected[a]['source_mean']['mean']:.4f}",f"{selected[a]['overall']['fn']['mean']:.1f}",
              f"{selected[a]['overall']['fp']['mean']:.1f}",f"{selected[a]['overall']['brier']['mean']:.4f}") for a,label in zip(ARMS,labels)])
        end_table=table(['方案','AIROGS','ODIR','Kaggle','LAG','KEH','五来源均值'] if zh else
            ['Condition','AIROGS','ODIR','Kaggle','LAG','KEH','Five-source mean'],
            [(label,*[f"{final[a]['source_auc'][s]['mean']:.4f}" for s in SOURCES],f"{final[a]['source_mean']['mean']:.4f}") for a,label in zip(ARMS,labels)])
        review_table=table(['方案','仅低 q AUROC','p 不确定性 AUROC','加入 q：AUROC 差','加入 q：多选中错误'] if zh else
            ['Condition','Low-q AUROC','p uncertainty AUROC','Adding q: AUROC change','Adding q: extra errors'],
            [(label,f"{qsummary[a]['models']['low_q']['auc']['mean']:.4f}",f"{qsummary[a]['models']['uncertainty']['auc']['mean']:.4f}",
              f"{qsummary[a]['contrasts']['pq minus p']['auc']['mean']:+.4f}",
              ' / '.join(f'{v:+.0f}' for v in qsummary[a]['contrasts']['pq minus p']['capture']['values'])) for a,label in zip(ARMS,labels)])
        operating_table=table(['方案 / 来源','检出率','特异度'] if zh else ['Condition / source','Sensitivity','Specificity'],
            [(labels[ARMS.index(a)]+' / '+SOURCE_NAMES[SOURCES.index(s)],
              f"{np.mean([v['fixed_validation_thresholds']['original'][s]['sensitivity'] for v in r['rows'] if v['arm']==a and v['mode']=='selected']):.1%}",
              f"{np.mean([v['fixed_validation_thresholds']['original'][s]['specificity'] for v in r['rows'] if v['arm']==a and v['mode']=='selected']):.1%}")
             for a in ('expanded_clean','quota65_mixed50','quota65_defocus50') for s in (SOURCES[0],SOURCES[1])])
        models=('uncertainty','low_q','p','pq','p_eyeq')
        model_labels=['p 接近 0.5','仅低 q','仅 p 拟合','p 与 FTHNet q','p 与 EyeQ q'] if zh else ['p near 0.5','Low q only','Fitted p only','p and FTHNet q','p and EyeQ q']
        runs=[v for v in r['quality_review_runs'] if v['arm']==candidate]
        workload_table=table(['复核方式','选中错误','选中漏诊','未复核漏诊'] if zh else ['Review ranking','Errors selected','Misses selected','Unreviewed misses'],
            [(label,f"{qsummary[candidate]['models'][m]['errors_selected']['mean']:.1f}",
              f"{qsummary[candidate]['models'][m]['misses_selected']['mean']:.1f}",
              f"{np.mean([v['models'][m]['review']['0.2']['remaining_fn'] for v in runs]):.1f}") for m,label in zip(models,model_labels)])
        if zh:
            text=f'''# 三种子训练完成：数据扩展、退化增强与质量复核结果

2026 年 11 月 15 日 · [English](completed-results.en.md) · [中文 PDF](completed-results.zh.pdf)

六个方案、三个共同随机种子的 18 组训练及开发集评价已全部完成。每组完成 2,400 次参数更新，后台队列于 10:29 UTC 结束。与训练初期的单种子观察相比，完整结果明确了三点：扩展数据稳定改善 LAG 和 KEH；恢复原来源采样比例对 AIROGS 有小幅收益，对 ODIR 的方向随训练阶段变化；退化增强提高合成失焦图上的表现，原图分类和人工复核的收益需要分别评价。

质量复核方面，18 个选中模型上，仅低质量分数识别错误的 AUROC 为 0.4923–0.5627，按疾病分数接近 0.5 排序为 0.8248–0.8976。在每次复核 411 张图的比较中，加入 FTHNet 质量分数比仅使用疾病分数的拟合方法多选中 −4 至 +4 个错误。这一轮的主要收益来自疾病模型与训练数据的改进。

## 数据与实验设计

数据版本为 expanded_20261113。原数据对照使用 10,338 张训练图，扩展方案使用 15,480 张。共同的模型验证集有 2,046 张图，用来选择权重；规则开发集有 2,053 张原图，包括 814 张阳性、1,239 张阴性和 1,917 个关联组，用来分析分类错误和复核方法。已确认的双眼、重拍或同图变体按关联组共同处理。疾病分数 p 越高表示模型越倾向判阳性；质量分数 q 越高表示模型预测照片质量越好。[项目提案](../../../../proposal-1106.pdf)使用两种分数选择人工复核照片，本轮评价相同复核量下加入 q 的额外收益。最终测试安排在模型、分类阈值和复核规则固定之后。

五个主要来源的规则开发图为 AIROGS 773 张（380 阳性）、ODIR 305 张（28 阳性，19 个阳性关联组）、Kaggle 142 张（95 阳性）、LAG 481 张（169 阳性）、KEH 153 张（75 阳性）。其他来源合计 199 张，计入全体漏诊与误报。AIROGS 使用转诊标签，其他来源按各自疾病定义保留标签；来源内 AUROC 用来比较同一来源的阳性、阴性排序。

所有方案使用相同 ImageNet 初始化的 EfficientNet-B0、320 像素输入、最多 16 张一批、AdamW 学习率 0.0001。随机种子为 20261115、20261116、20261117，每 300 次更新保存权重和验证预测。一次更新是根据一批图像的误差调整参数；本次预算为 2,400 次更新。每组从保存的八次验证中，按 AIROGS、ODIR、Kaggle、LAG、KEH 的等权 AUROC 均值选择权重。

“65% 原来源”表示采样时约 65% 的图像出现次数来自原数据来源，其余来自新增池；两个池内均按来源、类别计数的平方根倒数采样。“混合”表示约一半图像随机接受照明、失焦或散射中的一种处理；“失焦”表示约一半图像接受失焦处理。无退化方案保留共同的翻转等基础训练变换。

下表用验证集选中的权重评价。验证均值和开发均值均指五来源等权 AUROC；漏诊和误报以 p > 0.5 判阳性，表中计数为三个种子的平均值。Brier 为预测分数与 0/1 疾病标签差值平方的均值，越低表示分数误差越小。

{selection_table}

## 扩展数据的收益与来源采样的取舍

![两个训练阶段的来源 AUROC](source-auroc.zh.png)

图 1 每个格子是同一来源、三个种子的平均疾病 AUROC。左侧使用验证集选中权重，右侧统一使用第 2,400 次更新权重，颜色越深表示排序越好。两个训练阶段分别回答“按既定验证标准选模后表现如何”和“相同更新预算结束时表现如何”。

选中权重下，扩展无退化使 LAG 从 {selected['original_clean']['source_auc'][SOURCES[3]]['mean']:.4f} 提升至 {selected['expanded_clean']['source_auc'][SOURCES[3]]['mean']:.4f}，KEH 从 {selected['original_clean']['source_auc'][SOURCES[4]]['mean']:.4f} 提升至 {selected['expanded_clean']['source_auc'][SOURCES[4]]['mean']:.4f}。三个种子上的 LAG 增量为 +0.0472、+0.0383、+0.0401；KEH 为 +0.2979、+0.2769、+0.3593。LAG 和 KEH 被纳入训练来源后，来源覆盖产生了明确收益。合并全部来源的开发 AUROC 从 0.9365 提高至 0.9753。

![主要来源的配对 AUROC 变化](paired-source-effects.zh.png)

图 2 红点为三个共同种子的配对变化，菱形为其均值，横线为按同一批关联组重采样 2,000 次得到的 95% 区间。每次抽样对两方案使用相同病例，并对三个固定模型的变化取平均。横线反映病例抽样；三个点单独呈现训练种子的变化。LAG 扩展增量的病例区间为 [0.0280, 0.0555]，KEH 为 [0.2326, 0.3940]。

当前扩展采样在 2,400 次更新中，原来源平均出现 {original_draws:,.0f} 次，占 49.1%；65% 方案出现 24,950 次，比当前扩展采样多约 32%。恢复这部分训练次数后，选中权重的 AIROGS AUROC 三个种子均提高，增量为 +0.0070、+0.0038、+0.0041，平均 +0.0050。同时 LAG 三个种子均下降，平均 −0.0059。这是原来源训练量与新增来源训练量之间的取舍。

ODIR 的扩展增量在三个选中模型中分别为 −0.0057、+0.0164、−0.0244；65% 采样相对当前扩展采样分别为 −0.0023、+0.0006、+0.0222。其方向变化较大。ODIR 的阳性病例只有 28 张、19 个关联组，病例抽样区间也较宽；后续应同时报告检出率、误报和训练阶段，而不是依据某个种子的 AUROC 确定采样比例。

## 为什么选中权重和训练结束结果不同

![完整验证曲线](validation-curves.zh.png)

图 3 使用 2,046 张模型验证图。粗线为三个种子的均值，细线为各个种子。左侧是用于选模的五来源 AUROC 均值，右侧只计算原三个来源。扩展方案较早提升五来源表现，后续收益逐渐变小；原数据对照的五来源均值受 KEH 影响明显。

选中更新数在 1,200–2,400 之间。以 65% 原来源加混合退化为例，三个种子选中第 2,400、1,200、2,100 次更新；在原图开发集上的漏诊数为 78、84、68。继续到统一第 2,400 次更新后，漏诊数为 78、144、86，误报数从 76、109、101 变为 76、49、65。第二个种子的误报减少 60 张，漏诊增加 60 张，反映分数和阈值表现随训练发生变化。

相同第 2,400 次更新的来源平均值如下；各个种子和标准差保存在[逐种子结果表](per-seed-results.csv)。

{end_table}

65% 采样在这一训练阶段对 AIROGS 的平均收益为 +0.0048，对 ODIR 为 −0.0039，与选中权重的 ODIR 平均 +0.0068 方向不同。后续比较将验证集选中权重作为主结果，第 2,400 次更新作为相同训练预算的补充结果。

## 退化增强：失焦收益与实际增强次数

![合成压力集表现与增强次数](stress-and-training-dose.zh.png)

图 4 左侧使用选中权重：原图有 2,053 张，两个合成条件各使用相同 384 张父图，包括 146 张阳性、238 张阴性。失焦为按可见视野直径缩放的高斯模糊，亮度条件将原图亮度乘以 0.70。右侧按全部 2,400 次更新记录的图像出现次数展示增强组成。

当前扩展采样下，混合退化相对无退化的失焦 AUROC 在三个种子均提高，平均增加 0.0197，病例抽样区间为 [0.0086, 0.0316]；65% 采样下平均增加 0.0321。65% 原来源仅失焦的失焦 AUROC 为 0.9676，高于对应混合方案的 0.9593，三个种子均提高。原图五来源均值则从 0.9642 降至 0.9627，亮度条件从 0.9756 降至 0.9720。

增强频率解释了这组比较的一部分。混合方案中失焦约占全部训练图像出现次数的 17%，仅失焦方案约占 50%；首个种子分别为 6,554 次和 19,341 次，相差 2.95 倍。因此，目前比较同时改变增强组成和失焦出现次数。进一步分解机制时，应增加约六分之一概率的仅失焦方案，与混合方案匹配失焦次数。

失焦还会推动阳性、阴性分数向彼此靠近。在扩展无退化模型中，同一父图合成失焦后，阴性 p 平均上移 0.0590，阳性下移 0.0761；混合退化后两者为 +0.0280 和 −0.0617。增强减轻了这一分数变化，尤其减少阴性分数上移。真实低质量病例的检出将在人工质量分层中单独评价。

## 漏诊、误报与分类阈值

![原图漏诊和误报](misses-and-false-alarms.zh.png)

图 5 使用相同 2,053 张规则开发原图、分类阈值 0.5。柱为三个种子的平均计数，黑点为各个种子；蓝色是选中权重，棕色是第 2,400 次更新。漏诊以 814 张阳性为分母，误报以 1,239 张阴性为分母。

选中权重下，扩展无退化平均漏诊 69.0 张、误报 97.3 张；65% 原来源加混合退化为 76.7 和 95.3；仅失焦为 64.0 和 102.0。按当前阈值，仅失焦偏向多检出阳性，代价是更多误报。第 2,400 次更新下，65% 无退化方案相比扩展无退化少 {-budget_fp_delta:.1f} 个误报、多 {budget_fn_delta:.1f} 个漏诊，总错误数差为 {budget_fn_delta+budget_fp_delta:+.1f}。总错误数需要与这两类错误分别阅读。

另以每个来源的验证阴性分数第 95 百分位确定阈值，再固定用于规则开发原图，得到下表。表中为三个种子的平均结果；规则开发集的实际特异度由表中数据给出。

{operating_table}

ODIR 在这一阈值比较下，扩展无退化检出率为 57.1%，65% 加混合退化为 48.8%，仅失焦为 46.4%。分类 AUROC 较高仍会与某个阈值下的低检出率同时出现。接下来的阈值选择应明确可接受的漏诊率，并展示对应的误报和复核工作量。

## 质量与错误的关系在完整实验中如何变化

![按质量区间分别计算漏诊率和误报率](quality-error-rates.zh.png)

图 6 使用原 FTHNet 质量分数 q 和 18 个选中疾病模型。线为每个方案三个种子的平均错误率，横轴下方 n 是每个区间的阳性或阴性图像数，同一批病例在各种子中重复评价。漏诊率和误报率分别按疾病类别计算。

低 q 区间的阳性、阴性组成不同。q < 0.2 只有 26 张阳性和 23 张阴性，而 q ≥ 0.4 有 654 张阳性、1,045 张阴性。中高 q 区域的大量错误首先与病例数量相对应。按阳性计算的漏诊率在最高 q 区间仍较高：65% 加混合退化三个种子分别漏诊 6/37、6/37、4/37。全图质量较好也可能伴随疾病判断困难。

![疾病分数与质量分数的联合分布](disease-quality-distribution.zh.png)

图 7 展示按验证标准表现最高的方案——65% 原来源加混合退化——首个种子第 2,400 次更新的规则开发结果。蓝灰点为正确分类，红点为错误；左侧仅阳性，右侧仅阴性。竖线 p=0.5 是本次分类阈值，横线 q=0.4 用于比较质量分数范围。漏诊和误报分布在不同质量水平，复核区域需要结合 p、q 及两类错误分别评价。

## 加入质量分数的复核收益

每个固定疾病模型使用五折分组交叉验证拟合错误排序：同一关联组的图像进入同一折，用其余四折拟合，再为留出折生成预测。逻辑回归学习输入分数与出错概率之间的关系；仅 p 的方法使用疾病分数的正负方向和距 0.5 的程度，加入 q 后进一步学习质量与疾病分数的共同作用。两种方法在同一模型上使用相同分折，标准化在训练折内计算。另用第 17 轮选中的 EyeQ 质量模型重复比较。

下表的 AUROC 衡量识别错误的排序能力：随机抽取一张错误图和一张正确图，分数将错误排在前面的概率，0.5 为随机水平。前三个指标为各方案三个种子的平均值；最后一列依次给出三个种子在复核 411 张图时，加入 FTHNet q 相对仅 p 拟合多选中的错误数。

{review_table}

![两套质量分数的额外复核效果](quality-review-effects.zh.png)

图 8 各方案有 FTHNet 和 EyeQ 两组结果。点为三个种子，菱形为均值，横线为 2,000 次关联组配对抽样的 95% 区间；每次抽样都重新排序，两方法复核张数保持相同。区间固定疾病模型和已生成的折外风险分数，评价病例抽样变化。

65% 加混合退化中，加入 FTHNet q 的平均错误 AUROC 差为 {qdelta['auc']['mean']:+.5f}，病例区间为 [{qdelta['auc_case_interval95'][0]:+.5f}, {qdelta['auc_case_interval95'][1]:+.5f}]；20% 复核时多选中 2、1、0 个错误，平均 +1，病例区间为 [−2.33, +2.00]。换用 EyeQ 后增量也较小。为错误模型加入来源指标后，q 的增量方向仍在不同种子间变化。

![复核工作量与选中错误、剩余漏诊](review-workload.zh.png)

图 9 使用 65% 加混合退化的三个选中模型。线为平均值，浅色范围为三种子的最小值至最大值。左侧表示送去复核的错误，右侧表示未送复核的阳性漏诊，均按照片计数。

在相同 411 张复核量下，各方法结果如下。整数计数在三个模型间取平均，因此表中可能出现小数。

{workload_table}

p 接近 0.5 的简单排序平均选中 125.3 个错误、58.7 个漏诊，未复核漏诊为 18.0；拟合 p、q 后分别为 124.7、56.0、20.7。照片质量识别与疾病错误识别是两项不同任务；本轮复核收益以疾病分数提供的信息为主。后续 proposal 比较保留简单 p 排序和同一模型的仅 p 拟合作为对照，以额外捕获错误、漏诊和复核量判断 q 的价值。

## 真实低质量样本如何影响结果

![人工质量病例组成与两套质量模型分数](real-quality-composition.zh.png)

图 10 左侧汇总 FIVES、MSHF 人工低质量图的用途和疾病类别，柱顶为阳性 + 阴性。右侧使用模型验证和规则开发合计的 FIVES 54 张、MSHF 10 张原图；点为照片，箱体为四分位范围，中线为中位数。FIVES 的差质量定义是至少一个质量维度为差，MSHF 使用官方总体质量。

训练中有 72 张人工低质量阳性、3 张阴性；规则开发中有 12 张阳性、0 张阴性，模型验证仅 1 张低质量阴性。全部 15 个扩展模型均检出这 12 张阳性，原数据对照分别漏诊 5、5、4 张。无退化扩展也达到 12/12，表明这组病例的检出提升已经随数据扩展出现。12 个病例零漏诊对应的单侧 95% 漏诊率上界为 22.1%；低质量误报则需要独立阴性病例来估计。

样本不足包含两部分：低质量病例总数少，以及低质量与疾病标签、来源强相关。重复采样增加的是同一病例的训练次数，合成退化增加的是既有病例的变换；真实低质量阴性补样能同时改善训练组成和误报评价。补样优先寻找同来源、同采集条件、有明确青光眼阴性诊断及人工质量等级的独立病例。若无错误，分别至少 59 个独立阳性和 59 个独立阴性可使单侧 95% 错误率上界低于 5%；它们是后续收集规划的最低参照。

质量等级与连续分数还需按来源解释。FIVES 的 15 张人工低质量开发图 q 中位数为 0.5261，只有 1 张 q<0.4；MSHF 的 10 张低质量图中位数为 0.2453，9 张 q<0.4。FTHNet 在 FIVES 内辨认人工低质量的 AUROC 为 0.9026，EyeQ 为 0.9641。两套模型具有质量区分能力，当前较弱的是质量分数对疾病错误的额外排序信息。下一步人工复核应同时记录全图质量和视盘诊断信息是否可见。

## 后续安排

按预先使用的五来源验证均值，65% 原来源加混合退化为后续候选方案，三种子均值为 {selected[candidate]['validation_selection']['mean']:.4f}；该方案内验证值最高的是种子 {best['seed']}、第 {best['update']:,} 次更新，验证均值 {best['validation_selection']:.4f}。扩展无退化保留为数据扩展和质量复核的主要对照。该方案的三种子平均全体开发 AUROC 为 0.9747，分类阈值 0.5 时平均漏诊 76.7、误报 95.3。

优先推进三项工作：一是补充真实低质量阴性及独立阳性，分来源建立质量评价层；二是选择分类阈值并比较相同复核量下的残余漏诊和误报，维持疾病分数对照；三是匹配失焦出现次数，分解混合增强中的失焦频率与其他处理的作用。模型、阈值和复核区域确定后，按冻结用途进行最终测试。

本轮数据与图表见[分析 JSON](completed-analysis.json)、[逐种子 CSV](per-seed-results.csv)；方法见[分析程序](../../../../tools/analyze_completed_followup_20261115.py)和[报告程序](../../../../tools/report_completed_followup_20261115.py)。[执行状态](../../../../runs/followup_20261115/pipeline/status.json)记录 18 组完成，[早期质量分析](../quality-and-training.zh.md)保留训练启动时的快照。中英文 Markdown、PDF、LaTeX 和十组双语图表保存在本目录。
'''
        else:
            text=f'''# Completed three-seed experiments: data expansion, degradation, and quality-guided review

November 15, 2026 · [中文](completed-results.zh.md) · [English PDF](completed-results.en.pdf)

Training and development evaluation are complete for all 18 runs: six conditions with three common random seeds, each trained for 2,400 parameter updates. The queue finished at 10:29 UTC. The complete results clarify the initial single-seed observations. Expansion consistently improves LAG and KEH. Restoring original-source sampling provides a small AIROGS gain, while ODIR changes direction across training stages. Degradation improves performance on synthetic defocus; classification on original photographs and manual-review gains require separate evaluation.

Across the 18 selected disease models, ranking errors by low quality scores gives AUROC 0.4923–0.5627, compared with 0.8248–0.8976 for ranking by proximity of the disease score to 0.5. When 411 images are reviewed, adding FTHNet quality scores selects between four fewer and four more errors than a fitted disease-score method. The main gains in this round come from disease-model and training-data improvements.

## Data and experiment design

The data version is expanded_20261113. The original-data control uses 10,338 training photographs; expanded conditions use 15,480. The shared model-validation set contains 2,046 photographs and selects checkpoints. The rule-development set contains 2,053 originals: 814 positives, 1,239 negatives, and 1,917 linked groups. Confirmed fellow-eye images, repeat photographs, and image variants are handled together within these groups. Higher disease scores p indicate a greater tendency to classify positive; higher quality scores q indicate better predicted photographic quality. The [project proposal](../../../../proposal-1106.pdf) uses both scores to select images for manual review; this round evaluates the additional value of q at equal review workloads. Final testing follows the selection of models, classification thresholds, and review rules.

The five principal sources contribute AIROGS 773 photographs (380 positives), ODIR 305 (28 positives in 19 positive linked groups), Kaggle 142 (95 positives), LAG 481 (169 positives), and KEH 153 (75 positives). The other sources contribute 199 photographs and enter the overall miss and false-alarm counts. AIROGS uses referral labels; the remaining sources retain their own disease definitions. Within-source AUROC compares positive and negative ranking within the same source.

All conditions use EfficientNet-B0 with the same ImageNet initialization, 320-pixel input, batches up to 16, and AdamW learning rate 0.0001. The seeds are 20261115, 20261116, and 20261117. Checkpoints and validation predictions are saved every 300 updates. One update adjusts model parameters using the errors in one sampled batch; the budget is 2,400 updates. Each run selects from eight saved validations using the equal mean AUROC over AIROGS, ODIR, Kaggle, LAG, and KEH.

“65% original” assigns approximately 65% of image presentations to original data sources and the rest to the added pool. Within each pool, sampling weights are the inverse square root of source-class counts. “Mixed” applies one randomly chosen illumination, defocus, or scattering transformation to approximately half of presentations. “Defocus” applies defocus to approximately half. Conditions without degradation retain the shared basic training transformations, including flipping.

The following table evaluates validation-selected checkpoints. Validation and development means are equal-weight five-source AUROCs. Misses and false alarms use p > 0.5 for positive classification; counts are averages across three seeds. Brier is the mean squared difference between the prediction score and the 0/1 disease label; lower values indicate smaller score errors.

{selection_table}

## Expansion gains and the source-sampling tradeoff

![Source AUROC at two training stages](source-auroc.en.png)

Figure 1 shows three-seed mean disease AUROC within each source. The left panel uses validation-selected checkpoints; the right uses update 2,400 in every run. Darker cells indicate better ranking. These panels assess performance after the specified checkpoint-selection procedure and performance at a common update budget, respectively.

With selected checkpoints, clean expansion raises LAG from {selected['original_clean']['source_auc'][SOURCES[3]]['mean']:.4f} to {selected['expanded_clean']['source_auc'][SOURCES[3]]['mean']:.4f} and KEH from {selected['original_clean']['source_auc'][SOURCES[4]]['mean']:.4f} to {selected['expanded_clean']['source_auc'][SOURCES[4]]['mean']:.4f}. The three LAG gains are +0.0472, +0.0383, and +0.0401; KEH gains are +0.2979, +0.2769, and +0.3593. Adding LAG and KEH as training sources provides a clear benefit from source coverage. Pooled development AUROC increases from 0.9365 to 0.9753.

![Paired AUROC changes in principal sources](paired-source-effects.en.png)

Figure 2 shows paired changes for the three common seeds as red points, their mean as a diamond, and a 95% interval from 2,000 resamples of the same linked groups. Each resample uses the same cases for both conditions and averages changes across three fixed models. Intervals describe case sampling; individual points show training-seed variation. The expansion interval is [0.0280, 0.0555] for LAG and [0.2326, 0.3940] for KEH.

Under current expanded sampling, original-source photographs are presented {original_draws:,.0f} times on average by update 2,400, or 49.1% of presentations. The 65% condition presents them 24,950 times, approximately 32% more often. Restoring this exposure improves selected AIROGS AUROC in all three seeds, by +0.0070, +0.0038, and +0.0041, averaging +0.0050. LAG declines in all three seeds, averaging −0.0059. This is a tradeoff between training exposure to original and newly added sources.

Selected ODIR gains from expansion are −0.0057, +0.0164, and −0.0244. The 65% condition changes ODIR relative to current expanded sampling by −0.0023, +0.0006, and +0.0222. Direction varies substantially. ODIR has 28 positive photographs in 19 linked groups, with correspondingly broad case intervals. Further sampling comparisons should report sensitivity, false alarms, and training stage alongside AUROC.

## Selected checkpoints and end-of-budget results

![Complete validation curves](validation-curves.en.png)

Figure 3 uses the 2,046 model-validation photographs. Thick lines are means across three seeds; thin lines show individual seeds. The left panel is the five-source selection mean; the right uses only the original three sources. Expanded conditions improve five-source performance early, with smaller later gains. KEH has a substantial influence on the original-data control's five-source mean.

Selected updates range from 1,200 to 2,400. For 65% original plus mixed degradation, the selected updates are 2,400, 1,200, and 2,100; development misses are 78, 84, and 68. At the common update-2,400 budget, misses become 78, 144, and 86, while false alarms change from 76, 109, and 101 to 76, 49, and 65. In the second seed, 60 fewer false alarms accompany 60 additional misses, reflecting changes in scores and threshold behavior during training.

Source means at the common update-2,400 budget follow. Individual seeds and the information needed to calculate their standard deviations are retained in the [per-seed results](per-seed-results.csv).

{end_table}

At this stage, the 65% condition improves AIROGS by an average +0.0048 and changes ODIR by −0.0039. The ODIR direction differs from the +0.0068 mean at selected checkpoints. Subsequent comparisons will use validation-selected checkpoints as the main result and update 2,400 as the common-budget supplement.

## Degradation: defocus gains and actual training exposure

![Synthetic stress performance and augmentation exposure](stress-and-training-dose.en.png)

Figure 4 uses selected checkpoints. Originals contain 2,053 photographs; each synthetic condition uses the same 384 parent images, including 146 positives and 238 negatives. Defocus applies Gaussian blur scaled to the visible field diameter; the brightness condition multiplies brightness by 0.70. The right panel summarizes the recorded composition of image presentations over all 2,400 updates.

Under current expanded sampling, mixed degradation improves defocus AUROC in all three seeds, with a mean gain of 0.0197, with case interval [0.0086, 0.0316]. Under 65% sampling, the mean gain is 0.0321. Defocus-only with 65% original sampling reaches defocus AUROC 0.9676, compared with 0.9593 for mixed degradation, improving all three seeds. Its five-source original-image mean decreases from 0.9642 to 0.9627, while brightness-condition AUROC decreases from 0.9756 to 0.9720.

Exposure frequency explains part of this comparison. Defocus accounts for approximately 17% of all presentations under mixed degradation and 50% under defocus-only. In the first seed these counts are 6,554 and 19,341, a 2.95-fold difference. The comparison changes both augmentation composition and defocus frequency. A defocus-only condition with approximately one-sixth application probability would match defocus exposure in the mixed condition and help separate these effects.

Defocus also moves positive and negative scores toward one another. In clean expanded models, applying synthetic defocus to the same parent images raises negative p by 0.0590 and lowers positive p by 0.0761 on average. With mixed degradation the changes are +0.0280 and −0.0617. Augmentation reduces these shifts, particularly the rise in negative scores. Detection in real poor-quality cases is assessed separately using human quality labels.

## Misses, false alarms, and classification thresholds

![Misses and false alarms on original photographs](misses-and-false-alarms.en.png)

Figure 5 uses the same 2,053 rule-development originals and classification threshold 0.5. Bars show three-seed mean counts, with black points for individual seeds. Blue denotes selected checkpoints and brown update 2,400. Misses are counted among 814 positives; false alarms among 1,239 negatives.

At selected checkpoints, clean expansion averages 69.0 misses and 97.3 false alarms; 65% original plus mixed degradation averages 76.7 and 95.3; defocus-only averages 64.0 and 102.0. At this threshold, defocus-only detects more positives at the cost of more false alarms. At update 2,400, clean 65% sampling yields {-budget_fp_delta:.1f} fewer false alarms and {budget_fn_delta:.1f} more misses than clean expanded sampling, changing total errors by {budget_fn_delta+budget_fp_delta:+.1f}. Total errors should be read alongside both components.

A second operating-point comparison sets each source threshold to the 95th percentile of its model-validation negative scores, then holds it fixed for rule-development originals. Results below are means across three seeds. The table reports the specificity actually achieved in rule development.

{operating_table}

For ODIR, sensitivity under this procedure is 57.1% with clean expansion, 48.8% with 65% plus mixed degradation, and 46.4% with defocus-only. High classification AUROC can coexist with low sensitivity at a particular threshold. Threshold selection should specify the acceptable miss rate and show the associated false alarms and review workload.

## Quality and errors across the complete experiment

![Miss and false-alarm rates within quality intervals](quality-error-rates.en.png)

Figure 6 uses the original FTHNet quality score q and the 18 selected disease models. Lines show each condition's three-seed mean error rates. The n values below the axis are positive or negative image counts within each interval; the same cases are evaluated in every seed. Misses and false alarms use separate disease-class denominators.

Positive and negative composition differs across q intervals. Only 26 positives and 23 negatives have q < 0.2, while 654 positives and 1,045 negatives have q ≥ 0.4. Many errors in moderate-to-high q regions first reflect the large number of cases there. Positive miss rates remain substantial in the highest-q interval: 65% plus mixed degradation misses 6/37, 6/37, and 4/37 across seeds. Good overall image quality can accompany difficult disease classification.

![Joint disease-score and quality-score distribution](disease-quality-distribution.en.png)

Figure 7 shows the first seed's update-2,400 rule-development results for the condition with the highest mean validation score: 65% original plus mixed degradation. Blue-gray points denote correct classifications and red points errors, with positives on the left and negatives on the right. The vertical p=0.5 line is the classification threshold; q=0.4 marks a quality-score range for comparison. Misses and false alarms occur at multiple quality levels, so review regions require evaluation of p, q, and both error types.

## Additional review value of quality scores

For each fixed disease model, error ranking is fitted using five-fold group cross-validation. All images in one linked group stay in the same fold. The other four folds fit a model and generate predictions for the held-out fold. Logistic regression learns how input scores relate to the probability of error. The p-only method represents the direction of the disease score and its distance from 0.5; adding q learns how quality and disease scores act together. Both methods use the same folds within each disease model; standardization is fitted within the training folds. The comparison is repeated with the EyeQ quality model selected at epoch 17.

The AUROC below measures error ranking: the probability that a randomly drawn error image ranks above a randomly drawn correct image, with 0.5 indicating chance. The first three metrics are three-seed means. The final column lists the three seeds' additional errors selected after adding FTHNet q to fitted p, when reviewing the same 411 images.

{review_table}

![Additional review effects of two quality scores](quality-review-effects.en.png)

Figure 8 includes FTHNet and EyeQ comparisons for every condition. Points denote seeds, diamonds means, and lines 95% intervals from 2,000 paired linked-group resamples. Each resample reranks images and keeps the reviewed count equal between methods. These intervals fix the disease models and generated out-of-fold risks, describing variation from case sampling.

For 65% plus mixed degradation, adding FTHNet q changes mean error AUROC by {qdelta['auc']['mean']:+.5f}, with case interval [{qdelta['auc_case_interval95'][0]:+.5f}, {qdelta['auc_case_interval95'][1]:+.5f}]. At 20% review it selects 2, 1, and 0 additional errors, averaging +1, with interval [−2.33, +2.00]. EyeQ also provides small increments. Adding source indicators to error models leaves the direction of q gains varying across seeds.

![Review workload, selected errors, and remaining misses](review-workload.en.png)

Figure 9 uses the three selected models from 65% plus mixed degradation. Lines are means; shading spans the minimum and maximum across seeds. The left panel counts errors referred for review, while the right counts positive misses left outside review. Both use photographs as the counting unit.

At the same 411-image workload, the following table summarizes the methods. Averaging integer counts across three models produces fractional values.

{workload_table}

Simple ranking by proximity of p to 0.5 selects 125.3 errors and 58.7 misses on average, leaving 18.0 unreviewed misses. Fitted p,q selects 124.7 errors and 56.0 misses, leaving 20.7. Photographic-quality recognition and disease-error recognition are different tasks; disease-score information accounts for most review gains in this round. The proposal comparison will retain simple p ranking and fitted p within the same classifier as controls, judging q by additional errors or misses captured at a specified review workload.

## How real poor-quality samples affect interpretation

![Human quality composition and two quality-score models](real-quality-composition.en.png)

Figure 10 summarizes the disease-label composition and allocation of human-rated poor-quality FIVES and MSHF photographs on the left; labels are positive + negative. The score panels use 54 FIVES and 10 MSHF originals from model validation and rule development. Points are photographs, boxes the interquartile range, and lines medians. FIVES defines poor quality as at least one poor dimension; MSHF uses its official overall quality grade.

Training contains 72 human-rated poor-quality positives and three negatives. Rule development contains 12 positives and no negatives; model validation contains one poor-quality negative. All 15 expanded models detect these 12 positives, while original-data controls miss 5, 5, and 4. Clean expansion also achieves 12/12, showing that detection improvement in this group already accompanies data expansion. Zero misses in 12 independent cases gives a one-sided 95% upper miss-rate bound of 22.1%. Poor-quality false alarms require independent negative cases for estimation.

The shortage has two components: few real poor-quality cases overall, and strong association of poor quality with disease label and source. Resampling increases presentations of the same cases; synthetic degradation adds transformations of existing cases. Real poor-quality negative acquisition can improve both training composition and false-alarm evaluation. Acquisition should prioritize independent cases with the same source and collection conditions, an explicit negative glaucoma diagnosis, and human quality labels. Under zero errors, at least 59 independent positives and separately 59 negatives would put each one-sided 95% error-rate bound below 5%; these are minimum planning references.

Human grades and continuous quality scores also require source-specific interpretation. The 15 human-rated poor FIVES development images have median q 0.5261, with only one below 0.4. The ten poor MSHF images have median 0.2453, with nine below 0.4. Within FIVES, FTHNet distinguishes poor human quality with AUROC 0.9026; EyeQ reaches 0.9641. Both models distinguish photographic quality; the weak result here is their additional ranking information for disease errors. Further human review should record overall image quality and the visibility of diagnostic optic-disc information.

## Next steps

Under the existing five-source validation criterion, 65% original plus mixed degradation is the candidate for subsequent work, with three-seed mean {selected[candidate]['validation_selection']['mean']:.4f}. Its highest validation score is seed {best['seed']} at update {best['update']:,}, with mean {best['validation_selection']:.4f}. Clean expansion remains the principal control for data expansion and quality review. The candidate condition's mean pooled development AUROC is 0.9747; at classification threshold 0.5 it averages 76.7 misses and 95.3 false alarms.

The priorities are to acquire real poor-quality negatives and independent positives with source-specific quality evaluation; select classification thresholds and compare residual misses and false alarms at equal review workloads while retaining disease-score controls; and match defocus presentations to separate defocus frequency from the other mixed transformations. Once models, thresholds, and review regions are specified, final testing will follow the frozen allocations.

Evidence is available in the [analysis JSON](completed-analysis.json) and [per-seed CSV](per-seed-results.csv); methods are implemented in the [analysis program](../../../../tools/analyze_completed_followup_20261115.py) and [report generator](../../../../tools/report_completed_followup_20261115.py). The [execution status](../../../../runs/followup_20261115/pipeline/status.json) records 18 completed runs. The [earlier quality analysis](../quality-and-training.en.md) preserves the training-launch snapshot. Chinese and English Markdown, PDF, LaTeX, and ten sets of bilingual figures share this directory.
'''
        text=text.replace('父图','原图').replace('已生成的折外风险分数','交叉验证生成的错误风险分数').replace('后续 proposal 比较','后续项目提案中的比较')
        (OUT/f'completed-results.{lang}.md').write_text(text,encoding='utf-8')


def compile_and_render():
    BUILD.mkdir(parents=True,exist_ok=True)
    records={}
    for lang in ('zh','en'):
        stem=f'completed-results.{lang}';tex=OUT/f'{stem}.tex'
        subprocess.run(['pandoc',str(OUT/f'{stem}.md'),'-s','-t','latex','-o',str(tex),
            '-V','mainfont=Arial','-V','CJKmainfont=Microsoft YaHei','-V','monofont=Consolas',
            '-V','fontsize=10pt','-V','geometry:margin=0.7in','-V','colorlinks=true','-V','urlcolor=blue'],cwd=OUT,check=True,capture_output=True)
        source=tex.read_text(encoding='utf-8').replace('\\begin{figure}','\\begin{figure}[H]')
        source=source.replace('\\real{0.1429}','\\real{0.14285}')
        source=source.replace('\\begin{longtable}','\\Needspace{18\\baselineskip}\n\\begin{longtable}',1)
        source=source.replace('\\begin{longtable}','\\Needspace{8\\baselineskip}\n\\begin{longtable}')
        packages='\\usepackage{float}\n\\usepackage{needspace}\n\\setCJKmonofont{Microsoft YaHei}\n\\Urlmuskip=0mu plus 2mu\\relax\n'+('\\renewcommand{\\figurename}{图}\n' if lang=='zh' else '')
        tex.write_text(source.replace('\\begin{document}',packages+'\\begin{document}'),encoding='utf-8')
        for index in range(2):
            proc=subprocess.run(['xelatex','-interaction=nonstopmode','-halt-on-error',f'-output-directory={BUILD}',str(tex)],cwd=OUT,capture_output=True)
            (BUILD/f'{stem}-build-{index}.log').write_bytes(proc.stdout+proc.stderr);proc.check_returncode()
        shutil.copyfile(BUILD/f'{stem}.pdf',OUT/f'{stem}.pdf')
        pdf=pymupdf.open(OUT/f'{stem}.pdf');pages=[]
        for index,page in enumerate(pdf):
            path=BUILD/f'page-{lang}-{index+1:02d}.png'
            page.get_pixmap(matrix=pymupdf.Matrix(1.6,1.6)).save(path);pages.append(path)
        for start in range(0,len(pages),4):
            sheet=Image.new('RGB',(1800,2460),'#dddddd');draw=ImageDraw.Draw(sheet)
            for offset,path in enumerate(pages[start:start+4]):
                rendered=ImageOps.contain(Image.open(path).convert('RGB'),(880,1180))
                x=(offset%2)*900+(900-rendered.width)//2;y=(offset//2)*1230+30
                sheet.paste(rendered,(x,y));draw.text((x,y-20),f'{lang.upper()} page {start+offset+1}',fill='black')
            sheet.save(BUILD/f'contact-{lang}-{start//4+1}.jpg',quality=95)
        warnings=[line for line in (BUILD/f'{stem}-build-1.log').read_text(errors='replace').splitlines() if 'Overfull' in line or 'Missing character' in line]
        records[lang]=dict(pages=len(pdf),all_pages_rendered=True,figure_count=len(FIGURES),layout_warnings=warnings,latex_retained=True)
    (OUT/'render-verification.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(json.dumps(records,indent=2))


def main():
    r=json.loads((OUT/'completed-analysis.json').read_text(encoding='utf-8'))
    candidate=charts(r)
    reports(r,candidate)
    compile_and_render()


if __name__=='__main__':main()
