"""Analyze the completed three-seed experiment using saved development predictions."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.analyze_expansion_results_20261113 import digest, group_weights, weighted_metrics, write_json
from tools.analyze_quality_errors_20261115 import crossfit, review_counts, bootstrap_review_capture

RUN = ROOT / 'runs/followup_20261115'
OUT = ROOT / 'docs/progress/2026-11-15/completed-experiments'
LOCAL = RUN / 'completed_analysis'
SEEDS = (20261115, 20261116, 20261117)
ARMS = ('original_clean', 'expanded_clean', 'quota65_clean', 'expanded_mixed50',
        'quota65_mixed50', 'quota65_defocus50')
SOURCES = ('airogs-light-v2', 'DATASET_101/ODIR-5K',
           'Kaggle-Eye_diseases_classification', 'lag_part1', 'keh')
CONDITIONS = ('original', 'defocus_2_at512', 'exposure_070')
CONTRASTS = [('original_clean', 'expanded_clean'), ('expanded_clean', 'quota65_clean'),
             ('expanded_clean', 'expanded_mixed50'), ('quota65_clean', 'quota65_mixed50'),
             ('quota65_mixed50', 'quota65_defocus50')]
REPEATS = 2000


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def describe(values):
    values = np.asarray(values, float)
    return dict(values=values.tolist(), mean=float(values.mean()), sd=float(values.std(ddof=1)),
                minimum=float(values.min()), maximum=float(values.max()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    status = read(RUN / 'pipeline/status.json')
    assert status['status'] == 'completed' and len(status['completed']) == 18
    manifest_path = ROOT / 'data/prepared/expanded_20261113/manifest.csv'
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    reference = manifest[manifest['split'].eq('boundary')].sort_values('sample_id').reset_index(drop=True)
    qpath = ROOT / 'runs/expanded_20261113/pq_quality_20261114/quality_scores.csv'
    epath = RUN / 'eyeq_development/quality_scores.csv'
    quality = pd.read_csv(qpath)
    eyeq = pd.read_csv(epath)
    quality = quality[quality['split'].eq('boundary')].merge(
        eyeq[eyeq['split'].eq('boundary')], on=['sample_id', 'source', 'split', 'label'], validate='one_to_one')
    assert len(reference) == len(quality) == 2053
    assert digest(qpath) == read(qpath.with_name('verification.json'))['scores_sha256']
    assert digest(epath) == read(epath.with_name('verification.json'))['scores_sha256']
    inputs = {p.relative_to(ROOT).as_posix(): digest(p) for p in (
        manifest_path, qpath, epath, RUN/'pipeline/status.json', RUN/'experiment_plan.json',
        OUT.parent/'quality-error-analysis.json')}
    evidence = []
    data = {}
    rows = []
    curves = []
    for seed in SEEDS:
        for arm in ARMS:
            name = f'seed{seed}/{arm}'
            directory = RUN / name
            evaluation = read(directory / 'evaluation_summary.json')
            summary = read(directory / 'run_summary.json')
            assert summary['budget_completed'] and summary['optimizer_updates'] == 2400
            selection = evaluation['selected_update']
            log = [json.loads(line) for line in (directory/'epochs.jsonl').read_text().splitlines() if line.strip()]
            assert [v['optimizer_updates'] for v in log] == list(range(300, 2401, 300))
            assert selection == max(log, key=lambda v: v['selection_value'])['optimizer_updates']
            for v in log:
                curves.append(dict(seed=seed, arm=arm, update=v['optimizer_updates'],
                                   selection_value=v['selection_value'],
                                   original_source_mean=float(np.mean([v['per_source'][s]['auc'] for s in SOURCES[:3]])),
                                   loss=v['loss']))
            for mode, update in [('selected', selection), ('budget2400', 2400)]:
                report = evaluation['checkpoints'][str(update)]
                path = directory/'development'/f'update_{update:06d}'/'predictions.csv'
                predictions = pd.read_csv(path, keep_default_na=False)
                assert digest(path) == report['prediction_sha256']
                assert set(predictions['split']) == {'boundary'}
                assert set(predictions.condition) == set(CONDITIONS)
                for condition in CONDITIONS:
                    frame = predictions[predictions.condition.eq(condition)].sort_values('sample_id').reset_index(drop=True)
                    assert frame.sample_id.is_unique
                    expected = reference if condition == 'original' else reference[reference.sample_id.isin(frame.sample_id)]
                    assert len(frame) == (2053 if condition == 'original' else 384)
                    assert frame[['sample_id', 'source', 'label', 'related_group']].reset_index(drop=True).equals(
                        expected[['sample_id', 'source', 'label', 'related_group']].reset_index(drop=True))
                    data[mode, seed, arm, condition] = frame
                original = data[mode, seed, arm, 'original']
                overall = report['conditions']['original']['overall']
                y, p = original.label.to_numpy(), original.prediction.to_numpy()
                fn, fp = int(((p <= .5) & (y == 1)).sum()), int(((p > .5) & (y == 0)).sum())
                assert fn == overall['positive_misses']
                row = dict(seed=seed, arm=arm, mode=mode, update=update,
                    validation_selection=report['validation_selection']['selection_value'],
                    source_auc={source: report['conditions']['original']['per_source'][source]['auc'] for source in SOURCES},
                    overall={**overall, 'fn': fn, 'fp': fp, 'errors': fn+fp},
                    stress={c: report['conditions'][c]['overall'] for c in CONDITIONS[1:]},
                    real_poor=report['conditions']['original']['real_poor'],
                    fixed_validation_thresholds={c: report['conditions'][c]['fixed_validation_thresholds'] for c in CONDITIONS},
                    origin_draws=report['origin_draws'], unique_draws=summary['unique_sample_draws'],
                    family_draws=summary['degradation_family_draws'])
                rows.append(row)
                if mode == 'selected':
                    inputs[path.relative_to(ROOT).as_posix()] = digest(path)
                    checkpoint = directory/'validation_checkpoints'/f'update_{update:06d}.pt'
                    assert digest(checkpoint) == report['checkpoint_sha256']
                    evidence.append(dict(run=name, selected_update=selection,
                        checkpoint_sha256=report['checkpoint_sha256'], prediction_sha256=digest(path)))
            for filename in ('evaluation_summary.json', 'run_summary.json', 'epochs.jsonl', 'config.json'):
                path = directory/filename
                inputs[path.relative_to(ROOT).as_posix()] = digest(path)
    aggregate = {}
    for mode in ('selected', 'budget2400'):
        aggregate[mode] = {}
        for arm in ARMS:
            chosen = [r for r in rows if r['mode'] == mode and r['arm'] == arm]
            aggregate[mode][arm] = dict(updates=[r['update'] for r in chosen],
                validation_selection=describe([r['validation_selection'] for r in chosen]),
                source_auc={s: describe([r['source_auc'][s] for r in chosen]) for s in SOURCES},
                source_mean=describe([np.mean(list(r['source_auc'].values())) for r in chosen]),
                overall={k: describe([r['overall'][k] for r in chosen]) for k in ('auc', 'brier', 'fn', 'fp', 'errors', 'sensitivity', 'specificity')},
                stress={c: {k: describe([r['stress'][c][k] for r in chosen]) for k in ('auc','brier','positive_misses','sensitivity','specificity')} for c in CONDITIONS[1:]},
                real_poor_misses=describe([r['real_poor']['positive_misses'] for r in chosen]))

    contrasts = {}
    for mode in ('selected', 'budget2400'):
        contrasts[mode] = []
        for left, right in CONTRASTS:
            entry = dict(left=left, right=right, source={}, stress={})
            for source in SOURCES:
                a = data[mode, SEEDS[0], left, 'original']
                mask = a.source.eq(source).to_numpy()
                y = a.label.to_numpy()[mask]
                weights = group_weights(a.related_group.to_numpy()[mask], REPEATS, 20261115)
                samples, points = [], []
                for seed in SEEDS:
                    pl = data[mode, seed, left, 'original'].prediction.to_numpy()[mask]
                    pr = data[mode, seed, right, 'original'].prediction.to_numpy()[mask]
                    points.append(roc_auc_score(y, pr) - roc_auc_score(y, pl))
                    samples.append(weighted_metrics(y, pr, weights)['auc'] - weighted_metrics(y, pl, weights)['auc'])
                entry['source'][source] = {**describe(points), 'case_interval95': np.quantile(np.mean(samples, axis=0), [.025,.975]).tolist(),
                    'n': int(mask.sum()), 'positive': int(y.sum()), 'groups': int(a.loc[mask,'related_group'].nunique())}
            for condition in CONDITIONS[1:]:
                a = data[mode, SEEDS[0], left, condition]
                y = a.label.to_numpy()
                weights = group_weights(a.related_group, REPEATS, 20261115)
                deltas, samples = [], []
                for seed in SEEDS:
                    pl = data[mode, seed, left, condition].prediction.to_numpy()
                    pr = data[mode, seed, right, condition].prediction.to_numpy()
                    deltas.append(roc_auc_score(y, pr)-roc_auc_score(y, pl))
                    samples.append(weighted_metrics(y, pr, weights)['auc']-weighted_metrics(y, pl, weights)['auc'])
                entry['stress'][condition] = {**describe(deltas), 'case_interval95': np.quantile(np.mean(samples, axis=0), [.025,.975]).tolist()}
            contrasts[mode].append(entry)
    print('Source and stress comparisons complete.', flush=True)

    qresults = []
    weights = group_weights(reference.related_group, REPEATS, 20261115)
    qsamples = {arm: [] for arm in ARMS}
    for seed in SEEDS:
        for arm in ARMS:
            frame = data['selected', seed, arm, 'original'].merge(quality,
                on=['sample_id','source','split','label'], validate='one_to_one').sort_values('sample_id').reset_index(drop=True)
            assert list(frame.sample_id) == list(reference.sample_id)
            frame['error'] = (frame.prediction > .5) != frame.label.astype(bool)
            frame['fn'] = frame.prediction.le(.5) & frame.label.eq(1)
            frame['fp'] = frame.prediction.gt(.5) & frame.label.eq(0)
            y = frame.error.astype(int).to_numpy()
            risks = {'uncertainty': -abs(frame.prediction.to_numpy()-.5), 'low_q': -frame.q.to_numpy()}
            for name, key, adjustment in [('p',None,False),('pq','q',False),('p_eyeq','q_eyeq',False),
                                         ('p_source',None,True),('pq_source','q',True)]:
                risks[name], _ = crossfit(frame, key, adjustment, seed=20261115)
            result = dict(seed=seed, arm=arm, update=next(r['update'] for r in rows if r['seed']==seed and r['arm']==arm and r['mode']=='selected'),
                          n=len(frame), errors=int(y.sum()), models={}, quality_bins=[])
            for name, risk in risks.items():
                result['models'][name] = dict(auc=float(roc_auc_score(y,risk)),
                    review={str(f): review_counts(frame,risk,f)[0] for f in (.1,.2,.3)})
            delta_samples = {}
            for name in ('p','pq','p_eyeq'):
                delta_samples[name] = dict(auc=weighted_metrics(y,risks[name],weights)['auc'],
                                          capture=bootstrap_review_capture(frame,risks[name],weights,.2))
            qsamples[arm].append(delta_samples)
            for lo,hi in zip((0,.2,.4,.6,.8),(.2,.4,.6,.8,1.00001)):
                group = frame[frame.q.ge(lo)&frame.q.lt(hi)]
                pos=int(group.label.sum());neg=len(group)-pos
                result['quality_bins'].append(dict(lo=lo,hi=min(hi,1),n=len(group),positive=pos,negative=neg,
                    fn=int(group.fn.sum()),fp=int(group.fp.sum()),error_rate=float(group.error.mean()),
                    fn_rate=float(group.fn.sum()/pos),fp_rate=float(group.fp.sum()/neg)))
            result['source_errors'] = {s: dict(n=len(g),positive=int(g.label.sum()),fn=int(g.fn.sum()),fp=int(g.fp.sum()),
                high_confidence_errors=int((g.error & (g.prediction.le(.1)|g.prediction.ge(.9))).sum()),
                lowq_error_auc=float(roc_auc_score(g.error,-g.q)) if g.error.nunique()==2 else None)
                for s,g in frame.groupby('source')}
            output=frame[['sample_id','related_group','source','label','prediction','q','q_eyeq','error','fn','fp']].copy()
            for name,risk in risks.items(): output[name]=risk
            output.to_csv(LOCAL/f'{seed}-{arm}-out-of-fold.csv',index=False)
            qresults.append(result)
            print(f'Error review complete: {seed}/{arm}', flush=True)
    qsummary = {}
    for arm in ARMS:
        chosen=[v for v in qresults if v['arm']==arm]
        qsummary[arm]=dict(models={name: dict(auc=describe([v['models'][name]['auc'] for v in chosen]),
            errors_selected=describe([v['models'][name]['review']['0.2']['captured_errors'] for v in chosen]),
            misses_selected=describe([v['models'][name]['review']['0.2']['captured_fn'] for v in chosen])) for name in chosen[0]['models']}, contrasts={})
        for name in ('pq','p_eyeq'):
            qsummary[arm]['contrasts'][name+' minus p'] = dict(
                auc=describe([v['models'][name]['auc']-v['models']['p']['auc'] for v in chosen]),
                capture=describe([v['models'][name]['review']['0.2']['captured_errors']-v['models']['p']['review']['0.2']['captured_errors'] for v in chosen]),
                auc_case_interval95=np.quantile(np.mean([v[name]['auc']-v['p']['auc'] for v in qsamples[arm]],axis=0),[.025,.975]).tolist(),
                capture_case_interval95=np.quantile(np.mean([v[name]['capture']-v['p']['capture'] for v in qsamples[arm]],axis=0),[.025,.975]).tolist())

    # The same photos and labels are used once per model. Seed repetitions do not add cases.
    stress_shifts={}
    for arm in ARMS:
        stress_shifts[arm]={}
        for condition in CONDITIONS[1:]:
            by_label={0:[],1:[]}
            for seed in SEEDS:
                stress=data['selected',seed,arm,condition]
                clean=data['selected',seed,arm,'original'].set_index('sample_id').loc[stress.sample_id]
                delta=stress.prediction.to_numpy()-clean.prediction.to_numpy()
                for label in (0,1):by_label[label].append(float(delta[stress.label.eq(label)].mean()))
            stress_shifts[arm][condition]={str(label):describe(values) for label,values in by_label.items()}
    # Isolate repeated misses from changes at one chosen seed and classification threshold.
    persistence={}
    for arm in ARMS:
        p=np.array([data['selected',s,arm,'original'].prediction.to_numpy() for s in SEEDS])
        y=reference.label.to_numpy()
        errors=(p>.5)!=y.astype(bool)
        persistence[arm]=dict(errors_all_three=int(errors.all(axis=0).sum()),
            misses_all_three=int((errors.all(axis=0)&(y==1)).sum()),
            errors_any_seed=int(errors.any(axis=0).sum()))
    result=dict(captured_at=dt.datetime.now(dt.timezone.utc).isoformat(),pipeline=status,
        method=dict(seeds=SEEDS,bootstrap_repeats=REPEATS,bootstrap_unit='related_group',
            case_intervals='paired resampling of the same image groups; average of three fixed seed models',
            seed_sd='sample standard deviation across three independently trained models',
            cross_validation='five stratified group folds; common split seed 20261115 within each classifier',
            display_threshold=.5,review_fraction=.2,review_count=411),
        data=dict(version='expanded_20261113',validation_images=2046,development_images=2053,
            development_positive=814,development_negative=1239,development_groups=1917,
            stress_images=384,stress_positive=146,stress_negative=238,real_poor_positive=12,real_poor_negative=0,
            sources={s:dict(n=int(reference.source.eq(s).sum()),positive=int(reference.loc[reference.source.eq(s),'label'].sum()),
                groups=int(reference.loc[reference.source.eq(s),'related_group'].nunique())) for s in sorted(reference.source.unique())}),
        rows=rows,aggregate=aggregate,contrasts=contrasts,training_curves=curves,
        quality_review_runs=qresults,quality_review_summary=qsummary,stress_probability_shifts=stress_shifts,
        error_persistence=persistence,checkpoint_evidence=evidence,input_sha256=inputs,
        protected_test_inference=False)
    write_json(OUT/'completed-analysis.json',result)
    pd.DataFrame([{**{k:r[k] for k in ('seed','arm','mode','update','validation_selection')},
        **r['source_auc'],**{k:r['overall'][k] for k in ('auc','brier','fn','fp','errors','sensitivity','specificity')}} for r in rows]).to_csv(OUT/'per-seed-results.csv',index=False)
    write_json(LOCAL/'analysis-completion.json',dict(passed=True,completed_runs=18,report=digest(OUT/'completed-analysis.json')))
    print(json.dumps(dict(completed_runs=18,analyzed_checkpoints=len(rows),quality_review_models=len(qresults),passed=True)))


if __name__=='__main__':
    main()
