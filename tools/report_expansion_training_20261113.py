"""Build bilingual Markdown/PDF reports from completed experiment artifacts."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
RUN=ROOT/'runs/expanded_20261113'
HERE=ROOT/'docs/progress/2026-11-13/training-results'
BUILD=ROOT/'.build/progress-20261113-training'
ARMS=('original_clean','expanded_clean','expanded_degradation30','expanded_degradation50')
LABELS={'zh':['原数据','扩展数据','扩展＋30%退化','扩展＋50%退化'],
        'en':['Original data','Expanded data','Expanded + 30%','Expanded + 50%']}
SOURCES={'airogs-light-v2':'AIROGS','DATASET_101/ODIR-5K':'ODIR',
    'Kaggle-Eye_diseases_classification':'Kaggle','lag_part1':'LAG','keh':'KEH','beh':'BEH','fives':'FIVES'}


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def number(value):return '—' if value is None else f'{value:.4f}'


def chart(reports):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    colors=['#34495e','#2874a6','#229954','#d68910']
    fig,axes=plt.subplots(1,2,figsize=(10.2,3.6))
    for arm,label,color in zip(ARMS,LABELS['en'],colors):
        records=[json.loads(line) for line in (RUN/arm/'epochs.jsonl').read_text().splitlines()]
        axes[0].plot([r['optimizer_updates'] for r in records],[r['selection_value'] for r in records],marker='o',label=label,color=color)
        axes[1].plot([r['optimizer_updates'] for r in records],[r['loss'] for r in records],marker='o',color=color)
    axes[0].set(xlabel='Optimizer updates',ylabel='Mean source validation AUROC',title='Checkpoint validation')
    axes[1].set(xlabel='Optimizer updates',ylabel='Validation cross-entropy',title='Probability error on validation images')
    for ax in axes:ax.grid(alpha=.2);ax.set_xticks([300,600,900,1200])
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(HERE/'training-curves.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.2,3.7))
    width=.18;x=np.arange(len(SOURCES))
    for i,(arm,color) in enumerate(zip(ARMS,colors)):
        values=[reports[arm]['sets']['boundary/original']['per_source'][s]['auc'] for s in SOURCES]
        axes[0].bar(x+(i-1.5)*width,values,width,color=color,label=LABELS['en'][i])
    axes[0].set_xticks(x,list(SOURCES.values()),rotation=35,ha='right');axes[0].set(ylabel='Development AUROC',ylim=(0,1.05),title='Selected checkpoint, original photographs')
    axes[0].legend(fontsize=7,loc='lower right');axes[0].grid(axis='y',alpha=.2)
    real=[reports[a]['sets']['boundary/original']['real_poor_positive'] for a in ARMS]
    misses=[r['positive_misses'] for r in real]
    axes[1].bar(np.arange(4),misses,color=colors);axes[1].set_xticks(np.arange(4),['Original','Expanded','30%','50%'])
    axes[1].set(ylabel='Positive images below or equal to 0.5',ylim=(0,13),title='Real poor-quality development positives (n=12)')
    for i,count in enumerate(misses):axes[1].text(i,count+.2,f'{count}/12',ha='center',fontsize=10)
    axes[1].grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(HERE/'development-results.png',dpi=180);plt.close(fig)


    cohort=read(HERE/'data-cohort-summary.json')
    fig,axes=plt.subplots(1,2,figsize=(10.2,3.7))
    roles=['train','val','boundary','test']
    names=['Training','Model\nvalidation','Rule\ndevelopment','Reserved\ntest']
    old=[cohort['origin_role_counts']['formal_v3/'+role] for role in roles]
    new=[cohort['origin_role_counts']['expansion_20261113/'+role] for role in roles]
    x=np.arange(4)
    axes[0].bar(x,old,label='Original images',color='#34495e')
    axes[0].bar(x,new,bottom=old,label='Additions',color='#2874a6')
    for i,(o,n) in enumerate(zip(old,new)):
        axes[0].text(i,o+n+160,f'{o+n:,}',ha='center',fontsize=9)
    axes[0].set_xticks(x,names);axes[0].set(ylabel='Disease photographs',title='Frozen allocation',ylim=(0,17500))
    axes[0].legend(fontsize=8)
    fives=[cohort['real_poor_positive_counts']['fives/'+role] for role in roles]
    mshf=[cohort['real_poor_positive_counts']['mshf/'+role] for role in roles]
    axes[1].bar(x,fives,label='FIVES: any poor dimension',color='#229954')
    axes[1].bar(x,mshf,bottom=fives,label='MSHF: poor overall quality',color='#d68910')
    for i,(f,m) in enumerate(zip(fives,mshf)):
        axes[1].text(i,f+m+1.5,f'{f+m}',ha='center',fontsize=10)
    axes[1].set_xticks(x,names);axes[1].set(ylabel='Real poor-quality positive images',title='146 annotated poor-quality positives',ylim=(0,90))
    axes[1].legend(fontsize=7)
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.tight_layout();fig.savefig(HERE/'cohort-and-quality.png',dpi=200);plt.close(fig)


def tables(reports,summaries,lang):
    if lang=='zh':
        summary=['| 方案 | 训练图像 | 选中更新数 | 验证来源平均 AUROC | 真实低质量正例漏诊 |','|---|---:|---:|---:|---:|']
        sources=['| 来源（开发图像／正例） | 原数据 | 扩展数据 | 30%退化 | 50%退化 |','|---|---:|---:|---:|---:|']
        stress=['| 条件 | 原数据：漏诊／误报 | 扩展数据 | 30%退化 | 50%退化 |','|---|---:|---:|---:|---:|']
        operating=['| 来源 | 原数据：敏感度／特异度 | 扩展数据 | 30%退化 | 50%退化 |','|---|---:|---:|---:|---:|']
    else:
        summary=['| Arm | Training images | Selected update | Mean source validation AUROC | Real poor-quality positive misses |','|---|---:|---:|---:|---:|']
        sources=['| Source (images / positives) | Original | Expanded | 30% degradation | 50% degradation |','|---|---:|---:|---:|---:|']
        stress=['| Condition | Original: misses / false positives | Expanded | 30% degradation | 50% degradation |','|---|---:|---:|---:|---:|']
        operating=['| Source | Original: sensitivity / specificity | Expanded | 30% degradation | 50% degradation |','|---|---:|---:|---:|---:|']
    for label,arm in zip(LABELS[lang],ARMS):
        selected=reports[arm]['selected_successful_updates']
        epochs=[json.loads(line) for line in (RUN/arm/'epochs.jsonl').read_text().splitlines()]
        value=next(r['selection_value'] for r in epochs if r['optimizer_updates']==selected)
        poor=reports[arm]['sets']['boundary/original']['real_poor_positive']
        summary.append(f'| {label} | {summaries[arm]["training_rows"]:,} | {selected} | {value:.4f} | {poor["positive_misses"]}/{poor["positives"]} |')
    for source,label in SOURCES.items():
        source_metrics=[reports[a]['sets']['boundary/original']['per_source'][source] for a in ARMS]
        sources.append(f'| {label} ({source_metrics[0]["n"]}/{source_metrics[0]["positives"]}) | '+' | '.join(number(r['auc']) for r in source_metrics)+' |')
    for source in ('DATASET_101/ODIR-5K','airogs-light-v2','fives'):
        values=[reports[a]['source_thresholds_95_validation_specificity'][source]['boundary'] for a in ARMS]
        operating.append(f'| {SOURCES[source]} | '+' | '.join(f'{r["sensitivity"]*100:.1f}% / {r["specificity"]*100:.1f}%' for r in values)+' |')
    from evaluate_expansion_20261113 import metrics,read_csv
    scores={arm:read_csv(RUN/arm/'development_predictions.csv') for arm in ARMS}
    ids={r['sample_id'] for r in scores[ARMS[0]] if r['condition']=='defocus_2_at512'}
    for condition,label in [('original','原图' if lang=='zh' else 'Original'),('defocus_2_at512','轻度失焦' if lang=='zh' else 'Mild defocus'),('exposure_070','曝光乘数0.70' if lang=='zh' else 'Exposure multiplier 0.70')]:
        values=[metrics([r for r in scores[a] if r['sample_id'] in ids and r['condition']==condition]) for a in ARMS]
        counts=[]
        for arm,metric in zip(ARMS,values):
            selected=[r for r in scores[arm] if r['sample_id'] in ids and r['condition']==condition]
            false_positives=sum(int(r['label'])==0 and float(r['prediction'])>.5 for r in selected)
            counts.append(f'{metric["positive_misses"]} / {false_positives}')
        stress.append(f'| {label} | '+' | '.join(counts)+' |')
    return '\n'.join(summary),'\n'.join(sources),'\n'.join(stress),'\n'.join(operating)


def main():
    HERE.mkdir(parents=True,exist_ok=True);BUILD.mkdir(parents=True,exist_ok=True)
    status=read(RUN/'pipeline/status.json')
    if status['status']!='completed':raise ValueError('report requires all four completed arms and evaluations')
    comparison=read(RUN/'comparison.json');reports=comparison['arms']
    budget_review=read(RUN/'budget_end_comparison.json');final_reports=budget_review['arms']
    summaries={a:read(RUN/a/'run_summary.json') for a in ARMS}
    if not all(s['budget_completed'] and s['optimizer_updates']==1200 for s in summaries.values()):raise ValueError('unequal training budgets')
    preflight=read(RUN/'preflight/verification.json');cohort=read(ROOT/'data/prepared/expanded_20261113/cohort_verification.json')
    chart(reports)
    odir='DATASET_101/ODIR-5K'
    odir_aucs=[reports[a]['sets']['boundary/original']['per_source'][odir]['auc'] for a in ARMS]
    keh_aucs=[reports[a]['sets']['boundary/original']['per_source']['keh']['auc'] for a in ARMS]
    final_keh=[final_reports[a]['sets']['boundary/original']['per_source']['keh']['auc'] for a in ARMS]
    final_odir=[final_reports[a]['sets']['boundary/original']['per_source'][odir]['auc'] for a in ARMS]
    poor_misses=[reports[a]['sets']['boundary/original']['real_poor_positive']['positive_misses'] for a in ARMS]
    ci=comparison['paired_real_poor_boundary_sensitivity_differences']
    resources='[退化算法审计](degradation-audit.zh.md)'
    for lang in ('zh','en'):
        summary_table,source_table,stress_table,operating_table=tables(reports,summaries,lang)
        final_table=('| 方案（第1,200次更新） | ODIR AUROC | AIROGS AUROC | KEH AUROC | 真实低质量漏诊 | 合成失焦误报 |\n|---|---:|---:|---:|---:|---:|\n' if lang=='zh' else
            '| Arm at update 1,200 | ODIR AUROC | AIROGS AUROC | KEH AUROC | Poor-quality misses | Defocus false positives |\n|---|---:|---:|---:|---:|---:|\n')
        from evaluate_expansion_20261113 import read_csv
        final_stress_counts={}
        for label,arm in zip(LABELS[lang],ARMS):
            final=final_reports[arm]['sets']['boundary/original'];misses=final['real_poor_positive']['positive_misses']
            stress=[r for r in read_csv(RUN/arm/'last_development_predictions.csv') if r['condition']=='defocus_2_at512']
            false_positives=sum(int(r['label'])==0 and float(r['prediction'])>.5 for r in stress)
            final_stress_counts[arm]=dict(false_positives=false_positives,positive_misses=sum(int(r['label'])==1 and float(r['prediction'])<=.5 for r in stress))
            final_table+=f'| {label} | {final["per_source"][odir]["auc"]:.4f} | {final["per_source"]["airogs-light-v2"]["auc"]:.4f} | {final["per_source"]["keh"]["auc"]:.4f} | {misses}/12 | {false_positives}/238 |\n'
        intervals=[]
        contrast_labels={
            'expanded_clean minus original_clean':('扩展数据 − 原数据','Expanded − Original'),
            'expanded_degradation30 minus expanded_clean':('30%退化 − 扩展数据','30% − Expanded'),
            'expanded_degradation50 minus expanded_clean':('50%退化 − 扩展数据','50% − Expanded')}
        for key,value in ci.items():
            lo,hi=value['group_bootstrap_95_interval']
            label=contrast_labels[key][0 if lang=='zh' else 1]
            intervals.append(f'| {label} | {value["difference"]*100:+.1f} | [{lo*100:+.1f}, {hi*100:+.1f}] |')
        interval_table=('| 对照（后者减前者） | 敏感度变化（百分点） | 关联组重抽样95%区间 |\n|---|---:|---:|\n' if lang=='zh' else
            '| Contrast (later arm minus reference) | Sensitivity change (percentage points) | Group-bootstrap 95% interval |\n|---|---:|---:|\n')+'\n'.join(intervals)
        repeats=', '.join(str(summaries[a]['unique_sample_draws']) for a in ARMS)
        applied=[sum(v for k,v in summaries[a]['degradation_family_draws'].items() if k!='original')/sum(summaries[a]['degradation_family_draws'].values()) for a in ARMS]
        if lang=='zh':
            text=f'''# 数据整合、退化算法审计与新一轮训练结果

2026年11月13日 · Eye Sentinel

[English](training-results.en.md) · [中文PDF](training-results.zh.pdf) · [当日进度总览](../progress-overview.zh.md)

## 本轮完成了什么

本报告承接同日的[数据与退化方法审查](../data-review/data-and-degradation-review.zh.md)，记录审计后的数据划分、四组训练和开发集评价；[实验分析](../experimental-analysis/experimental-analysis.zh.md)进一步解释来源差异、训练采样和选模的影响。

新增数据与原疾病数据合并为固定版本 expanded_20261113。四组 EfficientNet-B0 使用相同的 ImageNet 预训练初始化，分别比较原数据、扩展数据、扩展数据加30%退化和扩展数据加50%退化。每组完成1,200次优化更新，即1,200次实际参数调整；配置中的20轮是训练上限，本轮按更新次数结束。模型验证集用于选择权重，规则开发集用于分析表现和后续设计人工复核规则，两者用途分开。四组均使用随机种子20261113。

AUROC的全称是受试者工作特征曲线下面积。随机抽取一张阳性和一张阴性照片时，它表示模型给阳性更高分的概率，并列分数计一半。0.5对应随机排序，1对应完全区分；例如0.93表示约93%的阳性／阴性配对中阳性分数更高。具体筛查阈值下的漏诊和误报另行评价。

下面首先报告各组由模型验证集选中的权重：扩展无退化组为第300次更新，其余组为第1,200次更新。相同第1,200次更新的比较见“模型选择的补充检查”，也用于当日总览和实验分析。

在305张ODIR规则开发图像（28张正例）上，原数据与扩展数据方案的AUROC分别为 {odir_aucs[0]:.4f} 和 {odir_aucs[1]:.4f}；30%和50%退化方案分别为 {odir_aucs[2]:.4f} 和 {odir_aucs[3]:.4f}。在12张真实低质量正例上，以0.5为展示阈值，四组漏诊数依次为 {', '.join(map(str,poor_misses))}。这组结果表明，扩展数据提高了这12张低质量阳性的检出，同时降低了ODIR原图的排序表现。下文分别报告来源结果、漏诊与误报，以解释这两种变化。

在153张KEH视盘裁剪开发图像（75张正例）上，四组AUROC依次为 {', '.join(f'{v:.4f}' for v in keh_aucs)}。KEH图像主要是视盘局部裁剪，与完整眼底照片有不同的成像范围。ODIR的28张阳性照片属于19个已确认阳性关联组；一组内的双眼或关联图像在统计时共同处理。

## 数据整合和本次新增发现

统一疾病清单共有24,491张图像。文件SHA256用于核对图像内容是否变化；完整RGB解码检查确认图像可以读取。全部图像通过检查，新增9,070张与审查阶段记录的原始像素一致。按患者、同眼、同图变体及已确认跨来源关联合并分组后，关联组跨用途和跨用途相同RGB像素均为零。386个已知患者标识均分配到单一用途。缺少患者标识的来源按已确认图像关系分组，所记录的关联组数是图像分组数。

独立重建关联图发现，原疾病数据中有18张图像与官方EyeQ测试组关联。本次从新版本排除16张训练图和2张模型验证图，并落实12张原规则开发图的既有修复。这18张图的移除同时应用于原数据对照和三个扩展组。历史模型在移除前训练；涉及这些EyeQ组的后续评价使用本轮模型。

| 用途 | 原数据图像／正例 | 新增图像／正例 | 合计图像／正例 |
|---|---:|---:|---:|
| 训练 | 10,338 / 4,409 | 5,142 / 1,996 | 15,480 / 6,405 |
| 模型验证 | 1,188 / 487 | 858 / 336 | 2,046 / 823 |
| 规则开发 | 1,220 / 503 | 833 / 311 | 2,053 / 814 |
| 内部及新增保留测试 | 1,233 / 489 | 2,237 / 803 | 3,470 / 1,292 |

另保留ACRIMA的705张、396张正例和HYGD的737张、540张正例。PAPILA全来源420张（87张正例，来自47名已知阳性患者）纳入保留测试，105张官方测试图沿整组传播为测试用途。204张疑似图单独保存，疾病二分类监督使用有明确阳性或阴性标签的图像。LAG、GRAPE、FIVES、Fundus-AVSeg和MSHF按已确认关联联合分组。KEH保留视盘裁剪域说明；RG转诊标签与各来源临床诊断标签保持分列。

真实低质量正例分别保留在训练72张、模型验证12张、规则开发12张、测试50张。FIVES按至少一个官方质量维度差定义，MSHF按官方总体质量编码0定义，两者不合并成同一种质量标注。规则开发的12张是FIVES 7张和MSHF 5张。

![冻结数据划分与真实低质量阳性分布](cohort-and-quality.png)

图1左侧按用途区分原数据和新增数据，合计包含训练、验证、开发和内部及新增保留测试；ACRIMA和HYGD的外部测试单列于正文。右侧显示146张真实低质量阳性按用途和来源的分布。FIVES与MSHF采用各自的官方质量定义。

## 退化算法与作者模型

已下载CoFe-Net、ArcNet、FD3、Catintell和CUNSB-RFIE五个固定提交版本，核查压缩包完整性、代码、许可和权重。CUNSB-RFIE的G、D、E、F四份权重均可用受限权重加载方式读取，张量均为有限值；G严格匹配作者DSCNet，并完成一次320×320 CPU前向运行。这次运行核实了权重和作者代码的相容性。作者网络用于图像恢复，本轮疾病训练的退化图则由项目实现的成像模拟生成；两者是不同处理步骤。

退化增强在疾病图补边并缩放到320×320后、ImageNet标准化前施加。每次等概率选择照明变化、失焦或轻度散射中的一种；失焦可附带轻噪声或JPEG压缩。散射通过平滑亮度混合降低局部对比度。模糊大小按有效眼底视野直径换算，曝光范围0.72–1.12，512像素视野参考模糊半径0.5–2.0，散射透明度0.03–0.10。图像标识、轮次和随机种子决定参数；保存的参数可重新生成相同图像，原始黑边保持原值。退化概率与疾病标签无关。

18对HRF同眼重拍及120对Real Fundus的描述性比对显示，真实低／高质量绿通道Laplacian方差比中位数分别约0.516和0.101，而固定温和失焦例约0.805和0.719。这一方差描述边缘和细节的变化，也受噪声、图像对齐和内容影响；本次比对用于说明训练退化的强度。对质量模型的监督仍使用原图及其教师质量分数或人工质量等级，合成退化只用于疾病训练。

## 训练预算、采样和模型选择

四组共享清单、验证角色、种子20261113、320像素输入、batch 16、AdamW学习率0.0001和相同更新预算。原数据组只过滤训练来源，不改变验证集。每张图的抽样权重为其来源、类别组合图像数的平方根倒数；图像数较少的组合因此获得更高的单图抽样概率。四组使用同一规则，扩展组加入来源后，各原来源获得的训练采样次数会改变。累计抽到的不同原始图像依次为 {repeats}，这些计数与生成版本数分开保存。实际退化抽样比例为 {applied[2]*100:.1f}% 和 {applied[3]*100:.1f}%。

每300次更新验证一次，按11个同时有正负例的来源各自AUROC的平均值选权重。来源AUROC先分别计算，再等权平均。恢复训练和多进程加载的一致性核验见配套审计记录。

{summary_table}

![相同更新预算下的验证结果](training-curves.png)

图2左侧为来源AUROC平均值，右侧为全验证图像的交叉熵，即模型为错误答案给出高置信度时产生的概率误差。曲线节点是300、600、900和1,200次更新，表中权重选自这些节点，扩展无退化组选中第300次更新，其余组选中第1,200次更新。

## 规则开发图像上的实际表现

下表使用每组已选权重，在相同原图上计算来源AUROC。规则开发集用于本轮训练后的分析。完整结果还包括AUPRC（精确率与召回率曲线下面积）、Brier分数（预测概率与0／1标签的平均平方误差）及校准误差。AUPRC受阳性比例影响，适合同一来源内比较；Brier分数越低，概率误差越小。

{source_table}

各组另按本来源验证阴性分数的第95百分位确定阈值，严格高于阈值才判为阳性。下表报告这些阈值在规则开发集上的敏感度和特异度，便于区分正例检出改善与整体分数上移；95%是验证集上的目标，表中开发集特异度由新的病例分数决定。FIVES只有14张验证阴性，因此该规则在验证集选择最大阴性分数，达到100%验证特异度。

{operating_table}

MSHF入选49张图全部为正例，没有同来源阴性可估计特异度。MSHF结果因此报告阳性检出；FIVES同时报告阳性检出和阴性误报。

![原图排序和真实低质量正例漏诊](development-results.png)

图3左侧比较七个主要来源的排序表现，右侧给出12张真实低质量正例的漏诊数量。0.5用于统一展示，来源验证阈值的结果见前表。12张低质量阳性由FIVES 7张和MSHF 5张组成，合计数描述这组固定开发照片。项目分类阈值及人工复核规则将在后续开发中确定。

{interval_table}

区间按患者／图像关联组进行2,000次配对重抽样，比较同一张照片在两组模型中的阈值判断。本轮使用一个训练种子和12张正例，区间描述病例抽样变化。退化组与扩展无退化组在这12张上的判断完全一致，因此抽样差异区间为零；新病例上的差异仍需更大的真实质量样本评价。

另外按图像身份哈希选择同一批384张规则开发原图，其中146张正例、238张阴性，分别施加参考512像素视野半径2的失焦和曝光乘数0.70。表中每格是漏诊数量／误报数量，分别基于146张正例和238张阴性，以0.5为展示阈值；每张原图及其两个变体属于同一个病例观察。

{stress_table}

扩展无退化组在失焦正例上漏诊3/146张，同时误报104/238张阴性，显示模糊图分数整体上移。30%和50%退化组的误报分别降为23/238和19/238，漏诊增加为36/146和28/146。真实低质量阳性结果见前表；真实低质量阴性是下一轮的补充重点。PAPILA、ACRIMA、HYGD及其余保留测试集将在模型和复核规则确定后评价。

## 模型选择的补充检查

训练过程中，HRF仅4张验证图的一次排序变化就会改变来源AUROC平均值，使扩展数据组选中了第300次更新的权重。针对这个已观察到的问题，本次增加预算结束权重的开发评价，直接比较四组第1,200次更新，不改动原选模规则或选中权重。以下结果均来自同一规则开发集。

{final_table}

该表与前面的选中权重结果分开保存，用于判断原有来源排序的下降有多少来自较早选模。下一轮应预先以具有足够正负例的主要来源设定选择标准，增加随机种子和更新预算，保留原图与真实低质量病例的检出、特异度和概率误差，然后锁定分类阈值及质量引导复核规则。

本轮数据扩展的主要观察为：同一1,200次更新预算下，KEH开发AUROC从{final_keh[0]:.4f}提高到{final_keh[1]:.4f}，真实低质量开发正例的展示阈值漏诊从5/12减少到0/12。50%退化将扩展模型的ODIR AUROC从{final_odir[1]:.4f}提高到{final_odir[3]:.4f}，仍低于原数据对照的{final_odir[0]:.4f}；三个扩展组均检出这12张低质量阳性。下一轮需处理原有来源的排序和阈值表现，并补充真实低质量病例比较退化的额外效果。

同预算的合成失焦评价中，50%退化将扩展原图模型的阴性误报从68/238降低到19/238，正例漏诊从18/146变为28/146。该变化明确呈现了误报减少与漏诊增加的关系；下一轮将结合真实低质量阴性，评价固定筛查阈值下的表现。

## 代码、证据和复现

[数据整合审计](data-audit.zh.md) · [退化方法审计](degradation-audit.zh.md) · [冻结清单核验](../../../../data/prepared/expanded_20261113/cohort_verification.json) · [完整对照结果](../../../../runs/expanded_20261113/comparison.json) · [预算结束权重检查](../../../../runs/expanded_20261113/budget_end_comparison.json) · [四组配置](../../../../configs/expanded_20261113/) · [启动脚本](../../../../tools/start_expansion_experiments_20261113.ps1)

已完成的目录保留best.pt、last.pt、逐步损失、验证曲线、原图和压力版本的开发预测，以及退化参数记录。配置、初始化、数据清单、发布信息和训练模块共同绑定到权重指纹；恢复时检查一致性。重新运行启动脚本会核查已完成预算，保留本轮产物。更改训练预算或设计时建立新配置和输出目录。

'''
        else:
            text=f'''# Data integration, degradation audit, and new training results

November 13, 2026 · Eye Sentinel

[中文](training-results.zh.md) · [English PDF](training-results.en.pdf) · [Daily progress overview](../progress-overview.en.md)

## What this round completed

This report follows the same-day [data and degradation review](../data-review/data-and-degradation-review.en.md) and records the allocation, training, and development evaluation of the audited images. The [experimental analysis](../experimental-analysis/experimental-analysis.en.md) examines source differences, sampling, and checkpoint selection in more detail.

The fixed expanded_20261113 release combines the additions with the original disease cohort. Four EfficientNet-B0 arms start from the same ImageNet pretrained initialization and compare original data, expanded data, and expanded data with 30% or 50% degradation. Each completes 1,200 optimizer updates, or parameter adjustments, ending at that budget within the configuration's 20-epoch ceiling. Model validation selects checkpoints, while rule development supports post-training analysis and subsequent manual-review design. All arms use seed 20261113.

AUROC is the area under the receiver operating characteristic curve. It represents the probability that a randomly selected positive image scores above a randomly selected negative, with ties counting as half. A value of 0.5 corresponds to random ranking and 1 to complete separation. Misses and false positives are evaluated separately at specified screening thresholds.

The opening results use each arm's validation-selected checkpoint: update 300 for expanded clean and update 1,200 for the other arms. The matched update-1,200 comparison appears under “Additional check of checkpoint selection” and is also used in the daily overview and experimental analysis.

On the 305-image ODIR rule-development subset, with 28 positives, AUROC was {odir_aucs[0]:.4f} for original-data training and {odir_aucs[1]:.4f} for expanded-data training. The 30% and 50% degradation arms achieved {odir_aucs[2]:.4f} and {odir_aucs[3]:.4f}. At the common display threshold of 0.5, misses among 12 real poor-quality positives were {', '.join(map(str,poor_misses))}, respectively. Expansion improves detection of these 12 poor-quality positives while reducing ranking on original ODIR photographs. The source-specific results and miss/false-positive counts below explain both changes.

On the 153-image KEH optic-disc-crop development subset, with 75 positives, the four AUROCs were {', '.join(f'{v:.4f}' for v in keh_aucs)}, respectively. KEH mainly contains optic-disc crops, with a different field of view from full fundus photographs. ODIR's 28 positive images form 19 confirmed positive linked groups; paired eyes and associated images in a group are handled together in statistical resampling.

## Data integration and the additional finding

The joint disease manifest contains 24,491 images. SHA256 checks image content, and full RGB decoding confirms that each image can be read. Every image passed these checks, and all 9,070 additions match the pixel identities recorded during the source review. Patient, same-eye, photograph-variant, and confirmed cross-source associations were combined before allocation. No linked group or identical decoded RGB image crosses roles. All 386 known patient identifiers have one role. Sources without patient identifiers are grouped by confirmed image relationships; their recorded group counts describe image groups.

Rebuilding the association graph independently found 18 original disease images linked to official EyeQ test groups. The new release removes 16 training and two model-validation images, and applies the existing 12-image rule-development repair. The same exclusions apply to the original-data control and all expanded arms. Historical models trained before this removal; subsequent evaluation involving these EyeQ groups uses this round's models.

| Role | Original images / positives | Additive images / positives | Total images / positives |
|---|---:|---:|---:|
| Training | 10,338 / 4,409 | 5,142 / 1,996 | 15,480 / 6,405 |
| Model validation | 1,188 / 487 | 858 / 336 | 2,046 / 823 |
| Rule development | 1,220 / 503 | 833 / 311 | 2,053 / 814 |
| Internal and new reserved tests | 1,233 / 489 | 2,237 / 803 | 3,470 / 1,292 |

ACRIMA's 705 images/396 positives and HYGD's 737 images/540 positives also retain their external-test roles. All 420 PAPILA images are held out as a complete source; its 87 positive images represent 47 known positive patients. The 105 official test images propagate test assignment to their entire linked groups. The 204 suspect images are retained separately; binary disease supervision uses explicit positive or negative labels. LAG, GRAPE, FIVES, Fundus-AVSeg, and MSHF use joint association grouping. KEH's optic-disc crop domain and source-specific clinical versus RG referral endpoints remain explicit.

Real poor-quality positives are allocated as 72 training, 12 model-validation, 12 rule-development, and 50 test images. FIVES uses at least one poor official quality dimension; MSHF uses official overall-quality code 0. Their definitions remain distinct. The 12 rule-development positives comprise seven FIVES and five MSHF images.

![Frozen allocation and real poor-quality positives](cohort-and-quality.png)

Figure 1 separates original and additive images by role, covering training, validation, development, and internal/new reserved tests. ACRIMA and HYGD external tests are listed separately in the text. The right panel shows the allocation of 146 real poor-quality positives by source and role, retaining each source's official quality definition.

## Degradation algorithms and author models

Five author releases were verified: CoFe-Net, ArcNet, FD3, Catintell, and CUNSB-RFIE. CUNSB's four checkpoints loaded with finite tensors; its generator matched the author code and ran once on a 320×320 image. The [degradation audit](degradation-audit.en.md) records version and execution details. This round uses the project acquisition simulator for disease training; the author network supports a separate restoration experiment.

Disease images are padded and resized to 320×320 before degradation and ImageNet normalization. Each draw selects illumination changes, defocus, or mild scatter with equal probability; defocus may add light noise or JPEG compression. Scatter reduces local contrast through smooth intensity mixing. Spatial blur scales with visible fundus-field diameter. Exposure spans 0.72–1.12, reference blur radius 0.5–2.0 at a 512-pixel field, and scatter opacity 0.03–0.10. Image identity, epoch, and seed determine replayable parameters; original black borders remain unchanged. Degradation probability is independent of disease label.

A descriptive comparison uses 18 HRF and 120 Real Fundus same-eye pairs. Median green-channel Laplacian-variance ratios relative to high-quality originals are 0.516 and 0.101 for real low-quality images, versus 0.805 and 0.719 for mild synthetic defocus. This statistic describes detail changes and also responds to noise and alignment. Quality training uses original images and their original quality targets; degradation applies to disease training.

## Training budget, sampling, and checkpoint selection

All arms share the manifest, validation roles, seed 20261113, 320-pixel inputs, batch size 16, AdamW learning rate 0.0001, and update budget. The original-data arm filters training origins only. Each image receives the inverse square root of its source/class image count as its sampling weight. Smaller combinations have higher individual draw probabilities. Adding sources changes the original sources' draw counts under this common rule. Distinct original images actually sampled were {repeats}, respectively; these counts remain separate from image variants. Realized degradation proportions were {applied[2]*100:.1f}% and {applied[3]*100:.1f}%.

Validation occurs every 300 updates. Checkpoints maximize the mean of 11 source-specific AUROCs where validation contains both classes; endpoints are evaluated within source before averaging. Resume and worker-loading consistency checks are retained in the supporting audit records.

{summary_table}

![Validation at matched optimizer-update budgets](training-curves.png)

Figure 2 shows mean source AUROC on the left and cross-entropy over validation images on the right. Cross-entropy penalizes confident wrong probabilities. Nodes represent 300, 600, 900, and 1,200 updates. Expanded clean selects update 300; the other arms select update 1,200.

## Observed effects on rule-development photographs

The following AUROCs use each arm's selected checkpoint on the same rule-development originals. The complete results also retain AUPRC, the area under the precision–recall curve; Brier score, the mean squared difference between predicted probabilities and 0/1 labels; and calibration error. AUPRC depends on positive prevalence and is useful for comparisons within a source. Lower Brier scores indicate smaller probability errors.

{source_table}

Each arm also sets a source-specific threshold at the higher 95th percentile of its validation-negative scores, predicting positive only above the threshold. The table reports resulting sensitivity and specificity on rule development, distinguishing improved detection from a general score shift. The 95% target is set on validation; the development specificity in the table follows from the new case scores. With only 14 validation negatives, the FIVES rule selects the maximum negative score and achieves 100% validation specificity.

{operating_table}

All 49 admitted MSHF images are positive, so same-source specificity cannot be estimated. MSHF therefore contributes positive-detection results, while FIVES provides both positive detection and negative false positives.

![Original-image ranking and real poor-quality positive misses](development-results.png)

Figure 3 compares seven major sources on the left and misses among 12 real poor-quality positives on the right. The common 0.5 threshold displays comparable counts; source validation thresholds are evaluated in the preceding table. The 12 poor-quality positives comprise seven FIVES and five MSHF images, and the combined count describes these fixed development photographs. Disease thresholds and manual-review rules will be chosen during subsequent development.

{interval_table}

Intervals use 2,000 paired resamples of patient/linked image groups, comparing decisions on the same photographs. Only 12 positive images contribute. The experiment trains one seed, and the intervals describe case-sampling variation. Degradation and expanded clean make identical decisions on these 12 images, producing a zero difference in every resample. Differences in new cases require evaluation on a larger real-quality sample.

A fixed subset of 384 rule-development originals, comprising 146 positive and 238 negative images, was selected by identity hash. It was evaluated after defocus of radius 2 at a 512-pixel reference field and exposure multiplication by 0.70. Each cell is the count of misses / false positives, among 146 positives and 238 negatives respectively, at display threshold 0.5. Each original and its two variants belong to the same case observation.

{stress_table}

The expanded original-only arm misses just three defocused positives but produces 104/238 false-positive negatives, showing a general score shift on blurred images. The 30% and 50% degradation arms reduce these false positives to 23/238 and 19/238, with 36/146 and 28/146 misses. Under synthetic defocus, augmentation reduces false positives and increases misses. Real poor-quality positive results appear above, and real poor-quality negatives are a priority for the next evaluation. PAPILA, ACRIMA, HYGD, and the remaining reserved tests await the final model and review rules.

## Additional check of checkpoint selection

During training, a ranking change among HRF's four validation images shifted the mean source AUROC enough to select the expanded-data checkpoint at update 300. This observed issue motivated an additional development evaluation of all four budget-end checkpoints at update 1,200. The comparison retains the originally selected checkpoints and uses the same rule-development set.

{final_table}

This table is saved separately from selected-checkpoint results and helps identify how much of the original-source ranking decrease arises from earlier selection. The next experiment should predeclare primary sources with adequate positive/negative counts, increase seeds and update budgets, retain original/real poor-quality detection, specificity, and probability error, then lock disease thresholds and quality-guided review rules.

At update 1,200, expansion improves KEH AUROC from {final_keh[0]:.4f} to {final_keh[1]:.4f}. Adding 50% degradation raises expanded-model ODIR AUROC from {final_odir[1]:.4f} to {final_odir[3]:.4f}, below the original control's {final_odir[0]:.4f}. All expanded arms detect the 12 real poor-quality positives. The next comparison needs additional real-quality cases and a sampling design that maintains original-source training exposure.

At matched budgets under synthetic defocus, 50% degradation reduces false positives from 68/238 to 19/238 and increases misses from 18/146 to 28/146. The next experiment will evaluate fixed screening thresholds with real poor-quality negatives.

## Code, evidence, and reproduction

[Data audit](data-audit.en.md) · [Degradation audit](degradation-audit.en.md) · [Frozen cohort verification](../../../../data/prepared/expanded_20261113/cohort_verification.json) · [Complete comparison](../../../../runs/expanded_20261113/comparison.json) · [Budget-end checkpoint diagnosis](../../../../runs/expanded_20261113/budget_end_comparison.json) · [Four configurations](../../../../configs/expanded_20261113/) · [Launch script](../../../../tools/start_expansion_experiments_20261113.ps1)

Run directories retain best.pt, last.pt, step losses, validation curves, original and stress-condition development predictions, and degradation parameters. Weights bind configuration, initialization, manifest, release, and training-module fingerprints. Resume checks those inputs. Re-running the launch script verifies completed budgets and preserves these outputs. Use a new configuration and output directory for a changed budget or design.

'''
        stem=f'training-results.{lang}'
        (HERE/f'{stem}.md').write_text(text.rstrip()+'\n',encoding='utf-8')
        subprocess.run(['pandoc',str(HERE/f'{stem}.md'),'-s','-t','latex','-o',str(HERE/f'{stem}.tex'),
            '--pdf-engine=xelatex','-V','mainfont=Arial','-V','CJKmainfont=Microsoft YaHei','-V','monofont=Consolas',
            '-V','fontsize=10pt','-V','geometry:margin=0.65in','-V','colorlinks=true','-V','urlcolor=blue'],cwd=HERE,check=True,capture_output=True)
        tex=HERE/f'{stem}.tex';latex=tex.read_text(encoding='utf-8')
        prefix='\\usepackage{float}\n\\usepackage{needspace}\n\\Urlmuskip=0mu plus 2mu\\relax\n'+('\\renewcommand{\\figurename}{图}\n' if lang=='zh' else '')
        latex=latex.replace('\\begin{figure}', '\\begin{figure}[H]')
        latex=latex.replace('\\begin{longtable}', '\\Needspace{12\\baselineskip}\n\\begin{longtable}')
        tex.write_text(latex.replace('\\begin{document}',prefix+'\\begin{document}'),encoding='utf-8')
        for index in range(2):
            result=subprocess.run(['xelatex','-interaction=nonstopmode','-halt-on-error',f'-output-directory={BUILD}',str(tex)],cwd=HERE,capture_output=True)
            (BUILD/f'{stem}-build-{index}.log').write_bytes(result.stdout+result.stderr);result.check_returncode()
        shutil.copyfile(BUILD/f'{stem}.pdf',HERE/f'{stem}.pdf')
        for pattern in (f'page-{lang}-*.png',f'contact-{lang}-*.jpg'):
            for path in BUILD.glob(pattern):
                if path.resolve().parent!=BUILD.resolve():raise ValueError('renderer path outside build directory')
                path.unlink()
        subprocess.run(['pdftoppm','-scale-to','1500','-png',str(HERE/f'{stem}.pdf'),str(BUILD/f'page-{lang}')],check=True,capture_output=True)
        subprocess.run(['pdftotext','-layout',str(HERE/f'{stem}.pdf'),str(BUILD/f'{stem}.txt')],check=True,capture_output=True)
    write=dict(all_arms_completed=True,optimizer_updates_each=1200,preflight_passed=preflight['passed'],
        protected_test_inference=False,cohort_images=cohort['unique_paths_file_hash_and_full_RGB_verified'],
        reports={lang:dict(pages=(BUILD/f'training-results.{lang}.txt').read_text(encoding='utf-8').count('\f'),
            pdf_bytes=(HERE/f'training-results.{lang}.pdf').stat().st_size,latex_retained=True) for lang in ('zh','en')})
    (HERE/'delivery-verification.json').write_text(json.dumps(write,indent=2),encoding='utf-8')
    print(json.dumps(write,indent=2))


if __name__=='__main__':main()
