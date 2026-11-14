"""Paired development analysis of the frozen November 13 experiments.

Reads saved predictions and manifest metadata; never performs image inference.
Bootstrap units are linked groups. Point estimates remain image weighted.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta, binomtest
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/expanded_20261113'
OUT = ROOT / 'docs/progress/2026-11-13/experimental-analysis'
ARMS = ('original_clean', 'expanded_clean', 'expanded_degradation30', 'expanded_degradation50')
SOURCES = {'airogs-light-v2': 'AIROGS', 'DATASET_101/ODIR-5K': 'ODIR',
           'Kaggle-Eye_diseases_classification': 'Kaggle', 'lag_part1': 'LAG',
           'keh': 'KEH', 'beh': 'BEH', 'fives': 'FIVES'}
CONTRASTS = ((0, 1), (1, 2), (1, 3), (2, 3))
CONDITIONS = ('original', 'defocus_2_at512', 'exposure_070')
REPEATS = 4000
SEED = 20261113


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group_weights(groups, repeats=REPEATS, seed=SEED):
    """Sample the observed number of groups with replacement, preserving their images."""
    units, indices = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    multiplicities = rng.multinomial(len(units), np.full(len(units), 1 / len(units)), size=repeats)
    return multiplicities[:, indices].astype(float)


def weighted_metrics(y, p, weights):
    """Exact weighted AUROC/AP with ties; accepts a matrix of bootstrap weights."""
    y, p = np.asarray(y), np.asarray(p)
    weights = np.atleast_2d(weights)
    positive = weights @ y
    negative = weights @ (1 - y)
    total = positive + negative
    order = np.argsort(p, kind='stable')
    starts = np.r_[0, np.flatnonzero(np.diff(p[order])) + 1]
    pos = np.add.reduceat(weights[:, order] * y[order], starts, axis=1)
    neg = np.add.reduceat(weights[:, order] * (1 - y[order]), starts, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        auc = (pos * (np.cumsum(neg, axis=1) - .5 * neg)).sum(axis=1) / (positive * negative)
        pos_desc, neg_desc = pos[:, ::-1], neg[:, ::-1]
        precision = np.cumsum(pos_desc, axis=1) / np.cumsum(pos_desc + neg_desc, axis=1)
        ap = np.nansum(precision * pos_desc, axis=1) / positive
        brier = (weights @ ((p - y) ** 2)) / total
    return {'auc': auc, 'auprc': ap, 'brier': brier}


def interval(samples):
    samples = np.asarray(samples)
    valid = samples[np.isfinite(samples)]
    return dict(interval_95=np.quantile(valid, [.025, .975]).tolist() if len(valid) else None,
                valid_repeats=int(len(valid)))


def point_metrics(rows):
    y = np.array([int(r['label']) for r in rows])
    p = np.array([float(r['prediction']) for r in rows])
    both = len(set(y)) == 2
    return dict(n=len(rows), positives=int(y.sum()), linked_groups=len({r['related_group'] for r in rows}),
                positive_groups=len({r['related_group'] for r in rows if int(r['label'])}),
                auc=float(roc_auc_score(y, p)) if both else None,
                auprc=float(average_precision_score(y, p)) if both else None,
                brier=float(np.mean((p-y)**2)), false_negatives=int(((p<=.5)&(y==1)).sum()),
                false_positives=int(((p>.5)&(y==0)).sum()))


def align_rows(rows_by_arm):
    """Validate both identities and immutable metadata before paired analysis."""
    keys = ('sample_id', 'source', 'label', 'related_group', 'target_definition',
            'origin_cohort', 'split', 'condition', 'real_poor_quality')
    result = []
    reference = None
    for rows in rows_by_arm:
        ids = [(r['sample_id'], r['condition']) for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError('duplicate prediction identity/condition')
        ordered = sorted(rows, key=lambda r: (r['sample_id'], r['condition']))
        metadata = [tuple(r[k] for k in keys) for r in ordered]
        if reference is not None and reference != metadata:
            raise ValueError('paired identities or metadata differ')
        if not all(np.isfinite(float(r['prediction'])) and 0 <= float(r['prediction']) <= 1 for r in ordered):
            raise ValueError('prediction outside finite probability range')
        reference = metadata
        result.append(ordered)
    return result


def validate_complete_prediction_rows(rows, manifest):
    """Require every development original and the predeclared stress parents."""
    expected={(r['sample_id'],'original') for r in manifest if r['split'] in ('val','boundary')}
    boundary=[r for r in manifest if r['split']=='boundary']
    parents=sorted(boundary,key=lambda r:hashlib.sha256(r['sample_id'].encode()).hexdigest())[:384]
    expected.update((r['sample_id'],c) for r in parents for c in CONDITIONS[1:])
    observed=[(r['sample_id'],r['condition']) for r in rows]
    if len(observed)!=len(expected) or set(observed)!=expected:
        raise ValueError('incomplete development predictions or wrong stress parents')


def paired_source_analysis(all_rows):
    result = {}
    for source in SOURCES:
        rows = align_rows([[r for r in arm if r['split']=='boundary' and r['condition']=='original' and r['source']==source]
                           for arm in all_rows])
        y = np.array([int(r['label']) for r in rows[0]])
        groups = np.array([r['related_group'] for r in rows[0]])
        weights = group_weights(groups)
        boot = [weighted_metrics(y, [float(r['prediction']) for r in arm], weights) for arm in rows]
        points = [point_metrics(arm) for arm in rows]
        contrasts = {}
        for left, right in CONTRASTS:
            name = ARMS[right]+' minus '+ARMS[left]
            contrasts[name] = {metric: dict(difference=points[right][metric]-points[left][metric],
                                            **interval(boot[right][metric]-boot[left][metric]))
                               for metric in ('auc', 'auprc', 'brier')}
        result[source] = dict(points=dict(zip(ARMS, points)), contrasts=contrasts)
    return result


def exact_sensitivity(hits, n):
    return dict(hits=hits, n=n, sensitivity=hits/n,
                exact_binomial_95=[float(beta.ppf(.025, hits, n-hits+1)) if hits else 0.,
                                   float(beta.ppf(.975, hits+1, n-hits)) if hits<n else 1.])


def quality_analysis(all_rows, manifest):
    from evaluate_expansion_20261113 import poor
    result = {'manifest_quality_strata': {}, 'development_sensitivity': {}, 'discordant_cases': []}
    for role in ('train', 'val', 'boundary'):
        source_counts = {}
        for source in ('fives', 'mshf'):
            counts = collections.Counter(('poor' if poor(r) else 'other', r['label'])
                                         for r in manifest if r['split']==role and r['source']==source)
            source_counts[source] = {quality: {'negative': counts[(quality, '0')], 'positive': counts[(quality, '1')]}
                                     for quality in ('poor', 'other')}
        result['manifest_quality_strata'][role] = source_counts
    selected = align_rows([[r for r in arm if r['split']=='boundary' and r['condition']=='original'
                           and r['real_poor_quality']=='True' and r['label']=='1'] for arm in all_rows])
    # All 12 observed poor-quality development positives are distinct linked groups.
    if len({r['related_group'] for r in selected[0]}) != len(selected[0]):
        raise ValueError('binomial image interval needs independent group units here')
    for arm, rows in zip(ARMS, selected):
        result['development_sensitivity'][arm] = {
            source: exact_sensitivity(sum(float(r['prediction'])>.5 for r in rows if source=='combined' or r['source']==source),
                                      sum(source=='combined' or r['source']==source for r in rows))
            for source in ('combined', 'fives', 'mshf')}
    decisions = [[float(r['prediction'])>.5 for r in arm] for arm in selected]
    improved = sum(not a and b for a, b in zip(decisions[0], decisions[1]))
    worsened = sum(a and not b for a, b in zip(decisions[0], decisions[1]))
    result['paired_expansion_mcnemar'] = dict(improved=improved, worsened=worsened,
        two_sided_exact_p=float(binomtest(improved, improved+worsened, .5).pvalue) if improved+worsened else 1.)
    for index, row in enumerate(selected[0]):
        result['discordant_cases'].append(dict(sample_id=row['sample_id'], source=row['source'], related_group=row['related_group'],
            scores={arm:float(rows[index]['prediction']) for arm,rows in zip(ARMS,selected)}))
    return result


def stress_analysis(all_rows):
    ids = {r['sample_id'] for r in all_rows[0] if r['condition']=='defocus_2_at512'}
    aligned = {condition: align_rows([[r for r in arm if r['split']=='boundary' and r['condition']==condition and r['sample_id'] in ids]
                                    for arm in all_rows]) for condition in CONDITIONS}
    if any([r['sample_id'] for r in aligned[c][0]] != [r['sample_id'] for r in aligned['original'][0]] for c in CONDITIONS):
        raise ValueError('stress parents differ')
    y=np.array([int(r['label']) for r in aligned['original'][0][0:]])
    weights=group_weights([r['related_group'] for r in aligned['original'][0]])
    result = {'conditions': {}, 'score_shifts': {}, 'validation_threshold_operating': {}}
    for condition, rows in aligned.items():
        points = [point_metrics(arm) for arm in rows]
        boot = [weighted_metrics(y,[float(r['prediction']) for r in arm],weights) for arm in rows]
        contrasts = {}
        for left,right in CONTRASTS:
            contrasts[ARMS[right]+' minus '+ARMS[left]] = {metric: dict(difference=points[right][metric]-points[left][metric],
                **interval(boot[right][metric]-boot[left][metric])) for metric in ('auc','auprc','brier')}
        exploratory = {}
        for arm, armrows in zip(ARMS,rows):
            p = np.array([float(r['prediction']) for r in armrows])
            threshold = float(np.quantile(p[y==0], .95, method='higher'))
            exploratory[arm] = dict(threshold=threshold, sensitivity=float((p[y==1]>threshold).mean()),
                                   specificity=float((p[y==0]<=threshold).mean()),
                                   rule='exploratory pooled development threshold; recalculated per condition; not deployable')
        per_source={}
        for source in SOURCES:
            subsets=[[r for r in arm if r['source']==source] for arm in rows]
            source_points=[point_metrics(arm) for arm in subsets]
            item=dict(points=dict(zip(ARMS,source_points)))
            if source_points[0]['auc'] is not None:
                mask=np.array([r['source']==source for r in rows[0]])
                # Resample within this source, retaining linked-group dependence.
                source_weights=group_weights([r['related_group'] for r in subsets[0]])
                sb=[weighted_metrics(y[mask],[float(r['prediction']) for r in arm],source_weights) for arm in subsets]
                item['contrast_50_minus_clean']={metric:dict(
                    difference=source_points[3][metric]-source_points[1][metric],
                    **interval(sb[3][metric]-sb[1][metric])) for metric in ('auc','auprc','brier')}
            per_source[source]=item
        # Paired fixed-0.5 changes describe the two error types separately.
        decision=[np.array([float(r['prediction'])>.5 for r in arm]) for arm in rows]
        error_contrasts={}
        for left,right in CONTRASTS:
            delta=decision[right].astype(float)-decision[left].astype(float)
            item={}
            for label,name in ((0,'false_positive_rate'),(1,'false_negative_rate')):
                mask=y==label;sign=1 if label==0 else -1
                values=sign*delta[mask]
                samples=(weights[:,mask]@values)/weights[:,mask].sum(axis=1)
                item[name]=dict(difference=float(values.mean()),**interval(samples))
            error_contrasts[ARMS[right]+' minus '+ARMS[left]]=item
        result['conditions'][condition] = dict(points=dict(zip(ARMS, points)), contrasts=contrasts,
            error_rate_contrasts=error_contrasts,per_source=per_source, exploratory_matched_specificity=exploratory)
        result['validation_threshold_operating'][condition] = {}
        for arm_index, arm in enumerate(ARMS):
            armrows=rows[arm_index]
            val=[r for r in all_rows[arm_index] if r['split']=='val' and r['condition']=='original']
            by_source={}
            for source in SOURCES:
                negatives=[float(r['prediction']) for r in val if r['source']==source and r['label']=='0']
                positive_val=any(r['source']==source and r['label']=='1' for r in val)
                subset=[r for r in armrows if r['source']==source]
                if not negatives or not positive_val or not subset:continue
                threshold=float(np.quantile(negatives,.95,method='higher'))
                positive=[r for r in subset if r['label']=='1'];negative=[r for r in subset if r['label']=='0']
                by_source[source]=dict(n=len(subset),positives=len(positive),threshold=threshold,
                    false_negatives=sum(float(r['prediction'])<=threshold for r in positive),
                    false_positives=sum(float(r['prediction'])>threshold for r in negative))
            result['validation_threshold_operating'][condition][arm]=by_source
    for condition in CONDITIONS[1:]:
        result['score_shifts'][condition] = {}
        for arm_index,arm in enumerate(ARMS):
            clean=np.array([float(r['prediction']) for r in aligned['original'][arm_index]])
            changed=np.array([float(r['prediction']) for r in aligned[condition][arm_index]])
            shifts={}
            for label in (0,1):
                mask=y==label; delta=changed-clean
                boot=(weights[:,mask]@delta[mask])/weights[:,mask].sum(axis=1)
                shifts[str(label)]=dict(n=int(mask.sum()), mean_change=float(delta[mask].mean()),
                    median_change=float(np.median(delta[mask])), **interval(boot),
                    per_source={s:dict(n=sum(r['source']==s and int(r['label'])==label for r in aligned['original'][arm_index]),
                        mean_change=float(np.mean([delta[i] for i,r in enumerate(aligned['original'][arm_index])
                                                  if r['source']==s and int(r['label'])==label])))
                                for s in SOURCES if any(r['source']==s and int(r['label'])==label for r in aligned['original'][arm_index])})
            result['score_shifts'][condition][arm]=shifts
    return result


def selection_analysis():
    result={}
    for arm in ARMS:
        records=[json.loads(line) for line in (RUN/arm/'epochs.jsonl').read_text().splitlines()]
        views={}
        definitions={'declared_11':None,'sources_at_least_20':list(SOURCES),
                     'original_3':list(SOURCES)[:3], 'added_4':list(SOURCES)[3:]}
        for name,sources in definitions.items():
            values=[]
            for r in records:
                selected=sources if sources is not None else r['selection_sources']
                values.append(dict(update=r['optimizer_updates'],
                    value=float(np.mean([r['per_source'][s]['auc'] for s in selected])),
                    sources=selected))
            best=max(values,key=lambda r:r['value'])
            views[name]=dict(best_update=best['update'],best_value=best['value'],curve=values)
        r0,r1=records[0],records[-1]
        contributions={s:(r1['per_source'][s]['auc']-r0['per_source'][s]['auc'])/11 for s in r0['selection_sources']}
        result[arm]=dict(views=views,contribution_to_declared_macro_300_to_1200=contributions)
    return result


def extract_draws():
    import torch
    manifest={r['sample_id']:r for r in read_csv(ROOT/'data/prepared/expanded_20261113/manifest.csv')}
    result={};expanded_counts=None
    from evaluate_expansion_20261113 import poor
    for arm in ARMS:
        checkpoint=torch.load(RUN/arm/'last.pt',map_location='cpu',weights_only=True)
        counts=checkpoint['state']['sample_draw_counts']
        if arm!='original_clean':
            if expanded_counts is not None and counts!=expanded_counts:
                raise ValueError('expanded-arm sample sequences have different draw counts')
            expanded_counts=counts
        grouped=collections.defaultdict(list)
        for identity,count in counts.items():
            r=manifest[identity]
            if r['split']!='train':raise ValueError('draw outside train')
            grouped['origin/'+r['origin_cohort']].append((identity,count))
            grouped['source/'+r['source']].append((identity,count))
            if r['source'] in ('fives','mshf'):
                grouped['quality/'+r['source']+'/'+('poor' if poor(r) else 'other')+'/'+r['label']].append((identity,count))
        strata={key:dict(unique_images=len(items),draws=sum(c for _,c in items),max_repeats=max(c for _,c in items),
                         linked_groups=len({manifest[i]['related_group'] for i,_ in items})) for key,items in grouped.items()}
        result[arm]=dict(total_draws=sum(counts.values()),unique_images=len(counts),strata=strata,
                        draws_per_successful_update=sum(counts.values())/1200,
                        sample_draw_counts_sha256=hashlib.sha256(json.dumps(counts,sort_keys=True).encode()).hexdigest(),
                        checkpoint_sha256=digest(RUN/arm/'last.pt'))
    write_json(OUT/'training-exposure.json',result)
    print(json.dumps({a:{'total_draws':r['total_draws'],'origins':{k:v for k,v in r['strata'].items() if k.startswith('origin/')}} for a,r in result.items()},indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--extract-draws',action='store_true');args=parser.parse_args()
    if args.extract_draws:
        extract_draws();return
    manifest_path=ROOT/'data/prepared/expanded_20261113/manifest.csv'
    release=read_json(manifest_path.parent/'release.json')
    if digest(manifest_path)!=release['manifest_sha256']:
        raise ValueError('manifest differs from frozen release')
    manifest=read_csv(manifest_path);by_id={r['sample_id']:r for r in manifest}
    if len(by_id)!=len(manifest):raise ValueError('duplicate manifest identity')
    result=dict(method=dict(repeats=REPEATS,seed=SEED,unit='related_group',paired=True,
        point_estimand='image-weighted metric; whole linked groups resampled with replacement',
        interval='percentile 95%; per contrast; no multiplicity adjustment; seed uncertainty not included',
        protected_test_inference=False),manifest_sha256=digest(manifest_path),prediction_sha256={},checkpoints={})
    for checkpoint,prefix in (('selected',''),('budget_end','last_')):
        rows=align_rows([read_csv(RUN/a/f'{prefix}development_predictions.csv') for a in ARMS])
        for arm,armrows in zip(ARMS,rows):
            validate_complete_prediction_rows(armrows,manifest)
            for r in armrows:
                m=by_id[r['sample_id']]
                if r['split'] not in ('val','boundary'):raise ValueError('protected role in predictions')
                if any(r[k]!=m[k] for k in ('source','label','related_group','target_definition','origin_cohort','split')):
                    raise ValueError('prediction differs from frozen manifest')
            result['prediction_sha256'][checkpoint+'/'+arm]=digest(RUN/arm/f'{prefix}development_predictions.csv')
        result['checkpoints'][checkpoint]=dict(sources=paired_source_analysis(rows),quality=quality_analysis(rows,manifest),stress=stress_analysis(rows))
        print('Analyzed '+checkpoint,flush=True)
    result['selection']=selection_analysis()
    result['training_exposure']=read_json(OUT/'training-exposure.json')
    write_json(OUT/'analysis.json',result)
    table=[]
    for checkpoint,item in result['checkpoints'].items():
        for source,value in item['sources'].items():
            for contrast,metrics in value['contrasts'].items():
                for metric,v in metrics.items():
                    table.append(dict(checkpoint=checkpoint,source=source,contrast=contrast,metric=metric,
                        difference=v['difference'],lower=v['interval_95'][0],upper=v['interval_95'][1],valid_repeats=v['valid_repeats']))
    with (OUT/'paired-source-contrasts.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    print(OUT/'analysis.json')


if __name__=='__main__':main()
