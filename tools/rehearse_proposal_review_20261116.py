"""Development rehearsal of the proposal's fixed, quality-adjusted review boundary."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.analyze_expansion_results_20261113 import group_weights
from tools.evaluate_expansion_20261113 import write_json
from training.data import load_rows, sha256

MANIFEST = ROOT / 'data/prepared/expanded_20261113/manifest.csv'
RELEASE = MANIFEST.with_name('release.json')
AUDIT = ROOT / 'data/audit/session_20261110'
FOLLOWUP = ROOT / 'runs/followup_20261115'
OUT = ROOT / 'runs/proposal_review_20261116'
REPORT = ROOT / 'docs/progress/2026-11-16/proposal-review-results.json'
ID = ['sample_id', 'source', 'label', 'related_group']
SOURCES = ['airogs-light-v2', 'DATASET_101/ODIR-5K', 'Kaggle-Eye_diseases_classification', 'lag_part1', 'keh']


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def score_comparator():
    import torch
    from tools.score_repaired_boundary import predict
    from training.models import create_model

    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    name = 'glaucoma_b_candidate'
    config = read(ROOT / f'configs/formal_v3/{name}.json')
    checkpoint = ROOT / config['run_dir'] / 'best.pt'
    archived = next(model for model in read(AUDIT / 'completion_verification.json')['models'] if model['name'] == name)
    assert sha256(checkpoint) == archived['best_checkpoint_sha256']
    content = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert content['fingerprint'] == archived['fingerprint']
    model = create_model(config['architecture'], ROOT / 'models/checkpoints').to('cuda')
    model.load_state_dict(content['model'], strict=True)
    model.eval()

    formal = ROOT / config['manifest']
    train = load_rows(formal, 'glaucoma', 'train', ROOT / config['release'])
    original_val = load_rows(formal, 'glaucoma', 'val', ROOT / config['release'])
    old_val = {row['sample_id']: row for row in original_val}
    anchors_path = AUDIT / f'selected_validation/{name}/validation_predictions.csv'
    anchors = pd.read_csv(anchors_path).head(16)
    anchor_scores = predict(model, [old_val[identity] for identity in anchors.sample_id], config)
    anchor_delta = max(abs(anchors.sample_id.map(anchor_scores) - anchors.prediction))
    assert anchor_delta < 5e-5, anchor_delta

    rows = {role: load_rows(MANIFEST, 'glaucoma', role, RELEASE) for role in ('val', 'boundary')}
    formal_train_hashes = {row['sha256'] for row in train}
    formal_train_groups = {row['related_group'] for row in train}
    overlap = {}
    for role, subset in rows.items():
        overlap[role] = dict(train_file_hashes=sum(r['sha256'] in formal_train_hashes for r in subset),
                             train_original_groups=sum(r['original_related_group'] in formal_train_groups for r in subset))
        assert not any(overlap[role].values()), overlap
    assert not {r['sha256'] for r in original_val} & {r['sha256'] for r in rows['boundary']}

    records = []
    for role, subset in rows.items():
        scores = predict(model, subset, config)
        records.extend({**{key: r[key] for key in ID}, 'split': role, 'p_b': scores[r['sample_id']]} for r in subset)
        print(f'ResNet-50 scored {len(subset)} {role} images', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / 'resnet50.csv'
    with destination.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[*ID, 'split', 'p_b'])
        writer.writeheader()
        writer.writerows(records)
    write_json(OUT / 'resnet50-verification.json', dict(
        checkpoint=checkpoint.relative_to(ROOT).as_posix(), checkpoint_sha256=sha256(checkpoint),
        original_validation_anchor_max_abs_difference=float(anchor_delta),
        selected_epoch_one_based=archived['best_epoch_one_based'],
        classifier_training_rows=len(train), training_overlap_checks=overlap,
        manifest_sha256=sha256(MANIFEST), predictions_sha256=sha256(destination),
        inference_roles=list(rows), protected_test_inference=False))


def align(reference, scores, keys, column):
    joined = reference[keys].merge(scores, on=keys, validate='one_to_one', how='outer', indicator=True)
    assert joined['_merge'].eq('both').all(), column
    indexed = joined.set_index('sample_id')[column]
    assert np.isfinite(indexed).all(), column
    if column in ('p_a', 'p_b'):
        assert indexed.between(0, 1).all(), column
    return reference.sample_id.map(indexed).to_numpy()


def data_for_analysis():
    manifest_hash = sha256(MANIFEST)
    assert manifest_hash == read(RELEASE)['manifest_sha256']
    manifest = pd.read_csv(MANIFEST, keep_default_na=False)
    scores_b = OUT / 'resnet50.csv'
    b_proof = read(OUT / 'resnet50-verification.json')
    assert b_proof['predictions_sha256'] == sha256(scores_b) and b_proof['manifest_sha256'] == manifest_hash
    assert b_proof['inference_roles'] == ['val', 'boundary'] and not b_proof['protected_test_inference']
    b = pd.read_csv(scores_b)
    assert set(b['split']) == {'val', 'boundary'}

    # A follows the validation-selected mixed-degradation checkpoint named in the 11/15 plan.
    model_dir = FOLLOWUP / 'seed20261117/quota65_mixed50'
    evaluation = read(model_dir / 'evaluation_summary.json')
    update = evaluation['selected_update']
    assert update == 2100
    provenance = evaluation['checkpoints'][str(update)]
    checkpoint = model_dir / f'validation_checkpoints/update_{update:06d}.pt'
    paths_a = {'val': checkpoint.with_suffix('.csv'),
               'boundary': model_dir / f'development/update_{update:06d}/predictions.csv'}
    assert sha256(checkpoint) == provenance['checkpoint_sha256']
    assert sha256(paths_a['boundary']) == provenance['prediction_sha256'] and not provenance['protected_test_inference']

    paths_q = {'q': ROOT / 'runs/expanded_20261113/pq_quality_20261114/quality_scores.csv',
               'q_eyeq': FOLLOWUP / 'eyeq_development/quality_scores.csv'}
    quality = {}
    inputs = [MANIFEST, RELEASE, scores_b, OUT / 'resnet50-verification.json', *paths_a.values()]
    quality_checkpoints = {}
    for column, path in paths_q.items():
        proof_path = path.with_name('verification.json')
        verified = read(proof_path)
        assert sha256(path) == verified['scores_sha256'] and verified['manifest_sha256'] == manifest_hash
        assert not verified['protected_test_inference']
        quality[column] = pd.read_csv(path)
        assert set(quality[column]['split']) == {'val', 'boundary'}
        quality_checkpoints[column] = verified['checkpoint_sha256']
        inputs.extend([path, proof_path])

    datasets = {}
    for role, path in paths_a.items():
        ref = manifest[manifest['split'].eq(role)].sort_values('sample_id').reset_index(drop=True).copy()
        a = pd.read_csv(path).rename(columns={'prediction': 'p_a'})
        if role == 'boundary':
            assert set(a['split']) == {'boundary'}
            a = a[a.condition.eq('original')]
        ref['p_a'] = align(ref, a[[*ID, 'p_a']], ID, 'p_a')
        ref['p_b'] = align(ref, b[b['split'].eq(role)], [*ID, 'split'], 'p_b')
        for column, frame in quality.items():
            keys = ['sample_id', 'source', 'label', 'split']
            ref[column] = align(ref, frame.loc[frame['split'].eq(role), [*keys, column]], keys, column)
        datasets[role] = ref
    assert not set(datasets['val'].related_group) & set(datasets['boundary'].related_group)
    assert [len(datasets[k]) for k in ('val', 'boundary')] == [2046, 2053]
    checks = dict(model_a=dict(checkpoint=checkpoint.relative_to(ROOT).as_posix(), checkpoint_sha256=sha256(checkpoint),
                               validation_macro_auc=provenance['validation_selection']['selection_value']),
                  model_b=b_proof, quality_checkpoints=quality_checkpoints,
                  input_sha256={p.relative_to(ROOT).as_posix(): sha256(p) for p in inputs})
    return datasets['val'], datasets['boundary'], checks


def upper_tail(scores, fraction):
    return float(np.sort(scores)[-int(np.ceil(len(scores) * fraction))])


def margin(scores, threshold):
    return -np.abs(logit(np.clip(scores, 1e-6, 1 - 1e-6)) - logit(threshold))


def counts(y, p, threshold, reviewed):
    kept = ~reviewed
    fn = int((kept & (y == 1) & (p <= threshold)).sum())
    fp = int((kept & (y == 0) & (p > threshold)).sum())
    return dict(n=len(y), positives=int(y.sum()), reviewed=int(reviewed.sum()), automatic=int(kept.sum()),
                reviewed_positive=int(y[reviewed].sum()), fn=fn, fp=fp, errors=fn+fp,
                automatic_error_rate=(fn+fp)/kept.sum() if kept.any() else None,
                residual_miss_rate=fn/y.sum() if y.any() else None)


def ranked_mask(order, k):
    mask = np.zeros(len(order), dtype=bool)
    mask[order[:k]] = True
    return mask


def bootstrap_totals(weights, order, k, error):
    ranked = weights[:, order]
    before = ranked.cumsum(axis=1) - ranked
    taken = np.minimum(ranked, np.maximum(k[:, None] - before, 0))
    assert np.array_equal(taken.sum(axis=1), k)
    return weights @ error - taken @ error[order]


def compare(data, thresholds, fraction, with_intervals=False):
    y = data.label.to_numpy()
    p_a, p_b = data.p_a.to_numpy(), data.p_b.to_numpy()
    t_a, t_b = thresholds
    u_a, u_b = margin(p_a, t_a), margin(p_b, t_b)
    # The largest possible quality adjustment is one log-odds unit. The shape is fixed.
    # No evaluation-fold score quantile is used to place its boundary.
    risks = {name: u_a + (1 - data[column].clip(0, 1).to_numpy())
             for name, column in [('joint', 'q_eyeq'), ('distilled', 'q')]}
    rank = {key: np.argsort(-score, kind='stable') for key, score in [('a_uncertainty', u_a), ('b_uncertainty', u_b)]}
    splitter = StratifiedGroupKFold(5, shuffle=True, random_state=20261116)
    strata = data.source.where(data.source.isin(SOURCES), 'other') + ':' + data.label.astype(str)
    folds = list(splitter.split(data, strata, data.related_group))
    cases = {}
    for quality, score in risks.items():
        mask = np.zeros(len(y), dtype=bool)
        fold_results = []
        for train, evaluate in folds:
            assert not set(data.related_group.iloc[train]) & set(data.related_group.iloc[evaluate])
            cut = upper_tail(score[train], fraction)
            mask[evaluate] = score[evaluate] >= cut
            fold_results.append(dict(calibration_images=len(train), held_fold_images=len(evaluate),
                                     cut=cut, reviewed=int(mask[evaluate].sum())))
        k = int(mask.sum())
        policies = {quality: (p_a, t_a, mask), **{
            key: (p_a if key == 'a_uncertainty' else p_b, t_a if key == 'a_uncertainty' else t_b, ranked_mask(order, k))
            for key, order in rank.items()}}
        counts_by_method = {key: counts(y, p, t, m) for key, (p, t, m) in policies.items()}
        strata_masks = {name: frame.index.to_numpy() for name, frame in data.groupby('source')}
        strata_masks['human_poor'] = data.index[
            (data.source.eq('fives') & data.quality_fives_any_poor_dimension.str.lower().eq('true')) |
            (data.source.eq('mshf') & data.quality_overall_official.eq('0'))].to_numpy()
        contrasts = {}
        if with_intervals:
            weights = group_weights(data.related_group.to_numpy(), 2000, 20261116)
            k_boot = weights @ mask.astype(float)
            automatic_boot = weights.sum(axis=1) - k_boot
            positive_boot = weights @ y
            for baseline, order in rank.items():
                p, t, _ = policies[baseline]
                delta = {}
                for kind, flags in {'errors': np.ones(len(y), dtype=bool), 'fn': y == 1}.items():
                    a_error = flags & ((p_a > t_a) != y)
                    base_error = flags & ((p > t) != y)
                    difference = weights @ (a_error & ~mask) - bootstrap_totals(weights, order, k_boot, base_error)
                    den = automatic_boot if kind == 'errors' else positive_boot
                    delta[kind] = dict(count=counts_by_method[quality][kind]-counts_by_method[baseline][kind],
                                       count_interval=np.quantile(difference, [.025, .975]).tolist(),
                                       rate_interval=np.quantile(difference / den, [.025, .975]).tolist())
                contrasts[baseline] = delta
        cases[quality] = dict(reviewed=k, observed_fraction=k/len(y), calibration_folds=fold_results,
                              policies=counts_by_method, contrasts=contrasts,
                              strata={name: {key: counts(y[ix], p[ix], t, mask[ix]) for key, (p, t, mask) in policies.items()}
                                      for name, ix in strata_masks.items()})
    return cases


def operating_points(val):
    thresholds = [float(np.quantile(val.loc[val.label.eq(0), p], .95, method='higher')) for p in ('p_a', 'p_b')]
    performance = {}
    for key, threshold in zip(('p_a', 'p_b'), thresholds):
        y, p = val.label.to_numpy(), val[key].to_numpy()
        performance[key] = dict(threshold=threshold, auc=float(roc_auc_score(y, p)),
                                sensitivity=float(np.mean(p[y == 1] > threshold)), specificity=float(np.mean(p[y == 0] <= threshold)),
                                source_auc={s: float(roc_auc_score(group.label, group[key]))
                                            for s, group in val.groupby('source') if s in SOURCES})
    return thresholds, performance


def analyze():
    val, dev, provenance = data_for_analysis()
    thresholds, validation = operating_points(val)
    # Follow-up to the pooled result: recalibrate both thresholds within AIROGS only.
    airogs_val = val[val.source.eq(SOURCES[0])]
    airogs_dev = dev[dev.source.eq(SOURCES[0])].reset_index(drop=True)
    airogs_thresholds, airogs_validation = operating_points(airogs_val)
    write_json(REPORT, dict(
        stage='development_rehearsal', protected_test_inference=False, validation=validation,
        development=dict(images=len(dev), positive=int(dev.label.sum()), groups=int(dev.related_group.nunique()),
                         no_review={key: counts(dev.label.to_numpy(), dev[key].to_numpy(), t, np.zeros(len(dev), dtype=bool))
                                    for key, t in zip(('p_a', 'p_b'), thresholds)}),
        main=compare(dev, thresholds, .10, with_intervals=True),
        sensitivity_threshold_0_5=compare(dev, [.5, .5], .10),
        exploratory_airogs_only=dict(validation_images=len(airogs_val), validation=airogs_validation,
                                     development_images=len(airogs_dev),
                                     development=compare(airogs_dev, airogs_thresholds, .10, with_intervals=True)),
        quality_clipping={role: {key: dict(below_zero=int(frame[key].lt(0).sum()), above_one=int(frame[key].gt(1).sum()))
                                for key in ('q_eyeq', 'q')} for role, frame in [('val', val), ('boundary', dev)]},
        settings=dict(reference_review_fraction=.10, threshold_rule='pooled validation negative 95th percentile (higher); p>t predicts positive',
                      review_score='-|logit(p)-logit(t)| + (1-q); disease scores clipped to [1e-6,1-1e-6] for logits',
                      quality_score='raw regression score clipped to [0,1] for applying the review rule',
                      capacity_rule='quality rule cutoff from reference folds, >= cutoff in held fold; baselines rank by negative logit distance with sample_id ascending on ties',
                      crossfit_folds=5, seed=20261116, bootstrap_repeats=2000),
        provenance={**provenance, 'analysis_sha256': sha256(Path(__file__))}))
    print(REPORT.relative_to(ROOT))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['score-b', 'analyze'])
    if parser.parse_args().stage == 'score-b':
        score_comparator()
    else:
        analyze()
