"""Measure the effect of quality-guided review on original development images."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.rehearse_proposal_review_20261116 import (
    MANIFEST, REPORT, SOURCES, counts, data_for_analysis, margin, operating_points,
    ranked_mask, read, sha256, upper_tail, write_json,
)

RATES = (.05, .075, .10, .125, .15, .20)
WEIGHTS = (0., .5, 1., 2.)
OUTPUT = ROOT / 'docs/progress/2026-11-18'
LOCAL = ROOT / 'runs/gray_zone_20261118'


def review_fold_scores(score, folds, fraction):
    reviewed = np.zeros(len(score), dtype=bool)
    cuts = []
    for fit, held in folds:
        cut = upper_tail(score[fit], fraction)
        reviewed[held] = score[held] >= cut
        cuts.append(float(cut))
    return reviewed, cuts


def describe_swaps(y, p, threshold, left, right):
    def partition(mask):
        result = counts(y, p, threshold, ~mask)
        return dict(images=result['automatic'], fn=result['fn'], fp=result['fp'],
                    correct=result['automatic'] - result['errors'])
    return {'both': partition(left & right), 'quality_only': partition(left & ~right),
            'uncertainty_only': partition(right & ~left), 'neither': partition(~(left | right))}


def cohort(val, dev, previous):
    dev = dev.reset_index(drop=True).copy()
    thresholds, validation = operating_points(val)
    y, a, b = (dev[k].to_numpy() for k in ('label', 'p_a', 'p_b'))
    ta, tb = thresholds
    qa = dev.q_eyeq.clip(0, 1).to_numpy()
    ua, ub = margin(a, ta), margin(b, tb)
    order_a, order_b = [np.argsort(-score, kind='stable') for score in (ua, ub)]
    assert dev.sample_id.is_monotonic_increasing
    strata = dev.source.where(dev.source.isin(SOURCES), 'other') + ':' + dev.label.astype(str)
    splitter = StratifiedGroupKFold(5, shuffle=True, random_state=20261116)
    folds = list(splitter.split(dev, strata, dev.related_group))
    for fit, held in folds:
        assert not set(dev.related_group.iloc[fit]) & set(dev.related_group.iloc[held])

    def compare(fraction, weight):
        score = ua + weight * (1 - qa)
        mask, cuts = review_fold_scores(score, folds, fraction)
        k = int(mask.sum())
        methods = {
            'joint': (a, ta, mask),
            'a_uncertainty': (a, ta, ranked_mask(order_a, k)),
            'b_uncertainty': (b, tb, ranked_mask(order_b, k)),
        }
        results = {key: counts(y, p, t, m) for key, (p, t, m) in methods.items()}
        swaps = describe_swaps(y, a, ta, mask, methods['a_uncertainty'][2])
        assert all(m['reviewed'] == k for m in results.values())
        ins, outs = swaps['quality_only'], swaps['uncertainty_only']
        assert ins['images'] == outs['images']
        for endpoint in ('fn', 'fp'):
            assert results['joint'][endpoint] - results['a_uncertainty'][endpoint] == outs[endpoint] - ins[endpoint]
        return dict(target=fraction, weight=weight, cuts=cuts, actual=k / len(y),
                    counts=results, swaps=swaps), methods

    main, methods = compare(.10, 1.)
    assert main['counts'] == previous['policies'], '11/16 counts must reproduce'
    assert main['cuts'] == [fold['cut'] for fold in previous['calibration_folds']]
    main['paired_95pct_from_20261116'] = previous['contrasts']
    main['sources'] = {}
    for name, rows in dev.groupby('source'):
        ix = rows.index.to_numpy()
        main['sources'][name] = {
            key: counts(y[ix], p[ix], t, m[ix]) for key, (p, t, m) in methods.items()
        }
        main['sources'][name]['swaps'] = describe_swaps(
            y[ix], a[ix], ta, methods['joint'][2][ix], methods['a_uncertainty'][2][ix])
    for key, totals in main['counts'].items():
        for metric in ('n', 'positives', 'reviewed', 'automatic', 'fn', 'fp'):
            assert sum(row[key][metric] for row in main['sources'].values()) == totals[metric]

    cut = upper_tail(ua + (1 - qa), .10)
    main['quality_rule_refit_cut'] = cut
    result = main['counts']['joint']
    auto_positive = (~methods['joint'][2]) & (a > ta)
    main['routing'] = dict(automatic_positives=int(auto_positive.sum()),
                           additional_reviews_if_all_auto_positives_routed=int(auto_positive.sum()),
                           total_reviews_if_all_auto_positives_routed=result['reviewed'] + int(auto_positive.sum()))
    missed = (y == 1) & (a <= ta)
    main['geometry'] = dict(
        p_bounds={str(q): expit(logit(ta) + np.array([-1, 1]) * (1 - q - cut)).tolist()
                  for q in (0., .5, 1.)},
        misses_outside_even_at_q_zero=int((missed & (ua + 1 < cut)).sum()))
    no_review = {key: counts(y, p, t, np.zeros(len(y), dtype=bool))
                 for key, p, t in [('a', a, ta), ('b', b, tb)]}
    # Development cases selected to make the new case-level audit reproducible.
    dev['joint_review_oof'] = methods['joint'][2]
    dev['a_uncertainty_review'] = methods['a_uncertainty'][2]
    dev['b_uncertainty_review'] = methods['b_uncertainty'][2]
    dev['a_result'] = np.select([(a <= ta) & (y == 1), (a > ta) & (y == 0)], ['FN', 'FP'], 'correct')
    dev['q_clipped'] = qa
    dev['distance_log_odds'] = -ua
    dev['review_fold'] = -1
    for i, (_, held) in enumerate(folds):
        dev.loc[held, 'review_fold'] = i
    assert dev.review_fold.ge(0).all()
    return dict(validation_images=len(val), groups=int(dev.related_group.nunique()),
                validation=validation, no_review=no_review, main=main,
                workloads=[compare(rate, 1.)[0] for rate in RATES],
                quality_weights=[compare(.10, weight)[0] for weight in WEIGHTS]), dev


def inventory():
    """Use manifested identity metadata only; do not open reserved photographs."""
    expanded = pd.read_csv(MANIFEST, keep_default_na=False)
    inputs = {}
    for name, config_name in [('b', 'formal_v3/glaucoma_b_candidate.json'), ('joint_quality', 'eyeq_quality_v1.json')]:
        config = read(ROOT / 'configs' / config_name)
        path = ROOT / config['manifest']
        digest = sha256(path)
        assert digest == read(ROOT / config['release'])['manifest_sha256']
        frame = pd.read_csv(path, keep_default_na=False)
        frame = frame[frame.task.eq(config['task']) & frame.split.isin(['train', 'val'])]
        inputs[name] = (frame, digest)
    development = expanded[expanded.split.isin(['train', 'val', 'boundary'])]
    b = inputs['b'][0]
    q = inputs['joint_quality'][0]
    scopes = {'internal': expanded.split.eq('test'),
              'airogs_subset_of_internal': expanded.split.eq('test') & expanded.source.eq(SOURCES[0]),
              'acrima': expanded.split.eq('external_acrima'), 'hygd': expanded.split.eq('external_hygd')}
    results = {}
    for name, mask in scopes.items():
        held = expanded[mask]
        poor = ((held.source.eq('fives') & held.quality_fives_any_poor_dimension.str.lower().eq('true')) |
                (held.source.eq('mshf') & held.quality_overall_official.eq('0')))
        overlap = dict(expanded_sha256=int(held.sha256.isin(development.sha256).sum()),
                       expanded_groups=int(held.related_group.isin(development.related_group).sum()),
                       b_sha256=int(held.sha256.isin(b.sha256).sum()),
                       b_original_group=int(held.original_related_group.isin(b.related_group).sum()),
                       joint_quality_sha256=int(held.sha256.isin(q.sha256).sum()))
        results[name] = dict(images=len(held), positives=int(held.label.sum()), groups=int(held.related_group.nunique()),
                             human_poor_pos=int((poor & held.label.eq(1)).sum()),
                             human_poor_neg=int((poor & held.label.eq(0)).sum()),
                             human_poor_by_source={
                                 source: dict(positive=int(rows.label.eq(1).sum()), negative=int(rows.label.eq(0).sum()))
                                 for source, rows in held[poor].groupby('source')},
                             overlap=overlap)
    assert [r['images'] for r in results.values()] == [3470, 770, 705, 737]
    return dict(scopes=results, comparison_manifest_sha256={name: item[1] for name, item in inputs.items()},
                tests_scored=False)


def main():
    val, dev, provenance = data_for_analysis()
    previous = read(REPORT)
    analyses = {}
    for name, values in {
        'pooled': (val, dev, previous['main']['joint']),
        'airogs': (val[val.source.eq(SOURCES[0])], dev[dev.source.eq(SOURCES[0])],
                   previous['exploratory_airogs_only']['development']['joint']),
    }.items():
        v, d, archived = values
        result, cases = cohort(v, d, archived)
        analyses[name] = result
        LOCAL.mkdir(parents=True, exist_ok=True)
        columns = ['sample_id', 'source', 'label', 'related_group', 'p_a', 'p_b', 'q_clipped',
                   'distance_log_odds', 'a_result', 'review_fold', 'joint_review_oof',
                   'a_uncertainty_review', 'b_uncertainty_review']
        cases[columns].to_csv(LOCAL / f'{name}-development.csv', index=False)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / 'gray-zone-results.json', dict(
        stage='development_only', cohorts=analyses, reservations=inventory(),
        provenance={**provenance, 'prior_report_sha256': sha256(REPORT), 'analysis_sha256': sha256(Path(__file__)),
                    'case_csv_sha256': {name: sha256(LOCAL / f'{name}-development.csv') for name in analyses}}))


if __name__ == '__main__':
    main()
