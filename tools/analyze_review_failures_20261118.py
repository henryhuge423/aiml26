"""Audit missed referrals, test a second scorer, and measure paired score changes."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageOps
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.analyze_matched_backbones_20261118 import (
    AIROGS, EFFICIENTNET, ID, MANIFEST, RELEASE, REPORT, RUN, align,
    checked_checkpoint, create_model, data_for_analysis, group_weights,
    load_rows, margin, predict, read, sha256, write_json,
)
from training.data import ImageRows
from training.fundus_degradation import retinal_context

OUTPUT = REPORT.parent
LOCAL = ROOT / 'runs/review_failures_20261118'
PAIRS = ROOT / 'runs/degradation_pilot_20261111'
REAL = ROOT / 'data/external/real_fundus/Real_Fundus.zip'
PREVALENCE = 3270 / 101442
RADII = (0, 2, 5)
UPDATE = 2400


def checkpoint_model(directory, expected):
    config = read(directory / 'config.json')
    path, payload = checked_checkpoint(directory, config, UPDATE)
    assert sha256(path) == expected
    model = create_model(config['architecture'], ROOT / 'models/checkpoints').to(config['device'])
    model.load_state_dict(payload['model'], strict=True)
    return model.eval(), config


def matched_scores(val, dev, records, rows):
    validation = {}
    for name, directory in [('efficientnet', EFFICIENTNET), ('resnet50', RUN)]:
        path = directory / f'validation_checkpoints/update_{UPDATE:06d}.csv'
        assert sha256(path) == records[name]['validation_predictions_sha256']
        validation[name] = pd.read_csv(path)
    path = EFFICIENTNET / f'development/update_{UPDATE:06d}/predictions.csv'
    assert sha256(path) == read(EFFICIENTNET / 'evaluation_summary.json')['checkpoints'][str(UPDATE)]['prediction_sha256']
    e = pd.read_csv(path, float_precision='round_trip')
    e = e[e.condition.eq('original')]
    model, config = checkpoint_model(RUN, records['resnet50']['checkpoint_sha256'])
    r = pd.DataFrame(predict(model, rows, config, 'original'))
    for name, scores in [('efficientnet', e), ('resnet50', r)]:
        assert set(scores['split']) == {'boundary'}
        for cohort, predictions in [(val, validation[name]), (dev, scores)]:
            cohort[name] = align(cohort, predictions.rename(columns={'prediction': name})[[*ID, name]], ID, name)
        encoded = dev[['sample_id', name]].to_csv(index=False, lineterminator='\n').encode()
        assert hashlib.sha256(encoded).hexdigest() == records[name]['development_predictions_sha256']
    print('Both archived development prediction hashes reproduced.', flush=True)


def selection(order, weights, budget):
    ordered = weights[:, order]
    before = np.cumsum(ordered, axis=1) - ordered
    return np.minimum(ordered, np.maximum(0, budget[:, None] - before))


def report_counts(captured, total, normalizer, sample_mass):
    return {
        'captured_fn': float(captured[0] / normalizer),
        'captured_fp': float(captured[1] / normalizer),
        'capture_fn': float(captured[0] / total[0]),
        'capture_fp': float(captured[1] / total[1]),
        'capture_all': float(captured.sum() / total.sum()),
        'remaining_fn_per_1000': float(1000 * (total[0] - captured[0]) / sample_mass),
    }


def routing(val, dev, archived):
    y = dev.label.to_numpy() == 1
    scores = dev[['efficientnet', 'resnet50']].to_numpy()
    thresholds = np.quantile(val.loc[val.label.eq(0), ['efficientnet','resnet50']], .95, axis=0, method='higher')
    assert np.allclose(thresholds, [archived[key]['validation_threshold'] for key in ('efficientnet','resnet50')], atol=1e-15)
    e_call, r_call = (scores > thresholds).T
    errors = np.column_stack([y & ~e_call, ~y & e_call]).astype(float)
    assert list(errors.sum(axis=0)) == [archived['efficientnet']['no_review'][k] for k in ('fn','fp')]
    log_scores = lambda frame: logit(np.clip(frame[['efficientnet', 'resnet50']].to_numpy(), 1e-6, 1-1e-6))
    stack = LogisticRegression().fit(log_scores(val), val.label)
    priority = stack.decision_function(log_scores(dev))
    uncertainty = margin(scores[:, 0], thresholds[0])
    quality = uncertainty + 1 - dev.q_eyeq.clip(0, 1).to_numpy()
    base_order = np.argsort(-quality, kind='stable')
    u_order = np.argsort(-uncertainty, kind='stable')
    neg = np.flatnonzero(~e_call)
    pos_base = base_order[e_call[base_order]]
    neg_score = neg[np.argsort(-priority[neg], kind='stable')]
    repeats = 2000
    boot = group_weights(dev.related_group.to_numpy(), repeats, 20261118).astype(float)
    boot = np.vstack([np.ones(len(y)), boot])
    results = {}
    for scenario, prevalence in [('archived_mix', None), ('airogs_public_photo_mix', PREVALENCE)]:
        if prevalence is None:
            weights = boot
            budget = np.ceil(.1 * weights.sum(axis=1))
            normalizer = 1
        else:
            weights = boot.copy()
            for mask, mass in [(y, prevalence), (~y, 1 - prevalence)]:
                weights[:, mask] *= mass / weights[:, mask].sum(axis=1, keepdims=True)
            budget = np.full(len(boot), .1)
            normalizer = .001
        base = selection(base_order, weights, budget)
        neg_budget = base[:, ~e_call[base_order]].sum(axis=1)
        alternate_neg = selection(neg_score, weights, neg_budget)
        alternate_pos = selection(pos_base, weights, budget - neg_budget)
        assert np.allclose(alternate_neg.sum(axis=1), neg_budget)
        assert np.allclose(alternate_pos, base[:, e_call[base_order]])
        totals = weights @ errors
        captured = {
            'uncertainty': selection(u_order, weights, budget) @ errors[u_order],
            'quality_gray_zone': base @ errors[base_order],
            'second_model_negative_reranking': (
                alternate_neg @ errors[neg_score] + alternate_pos @ errors[pos_base]
            ),
        }
        if prevalence is None:
            assert list(totals[0]-captured['quality_gray_zone'][0]) == [
                archived['efficientnet']['review']['quality'][k] for k in ('fn','fp')
            ]
        observed = {key: report_counts(value[0], totals[0], normalizer, weights[0].sum()) for key, value in captured.items()}
        delta = captured['second_model_negative_reranking'] - captured['quality_gray_zone']
        results[scenario] = dict(
            review_budget=float(budget[0]/normalizer), review_negative=float(neg_budget[0]/normalizer),
            units='photos' if prevalence is None else 'per_1000_photos',
            total_fn=float(totals[0,0]/normalizer), total_fp=float(totals[0,1]/normalizer),
            policies=observed,
            extra_captured_fn_vs_quality_ci95=(np.quantile(delta[1:,0], [.025,.975])/normalizer).tolist(),
        )
        if prevalence is None:
            base_mask = np.zeros(len(y), dtype=bool)
            alt_mask = base_mask.copy()
            base_mask[base_order] = base[0] > .5
            alt_mask[neg_score] = alternate_neg[0] > .5
            alt_mask[pos_base] = alternate_pos[0] > .5
    missed = y & ~e_call
    both_missed = missed & ~r_call
    outside = missed & ~base_mask
    return dict(
        thresholds=dict(efficientnet=float(thresholds[0]), resnet50=float(thresholds[1])),
        fit_logit_coefficients=stack.coef_[0].tolist(),
        efficientnet_misses=int(missed.sum()), resnet_call_positive_among_efficientnet_misses=int((missed & r_call).sum()),
        outside_quality_review_misses=int(outside.sum()),
        resnet_call_positive_among_outside_quality_review_misses=int((outside & r_call).sum()),
        both_models_miss=int(both_missed.sum()), bootstrap_replicates=repeats,
        scenarios=results,
    ), base_mask, alt_mask


def blurred(image, radius):
    sigma = radius * retinal_context(image)['field_diameter_pixels'] / 512
    return image.filter(ImageFilter.GaussianBlur(sigma))


def score_blur(model, config, rows):
    data = ImageRows(rows, ROOT / 'data', config['image_size'], 'glaucoma')
    resize, tensor, normalize = data.transform.transforms
    def transform(image):
        resized = resize(image)
        variants = [resized, *(blurred(resized, r) for r in RADII[1:])]
        return torch.stack([normalize(tensor(variant)) for variant in variants])

    data.transform = transform
    result = {}
    with torch.inference_mode():
        for images, _, identities in torch.utils.data.DataLoader(data, batch_size=32):
            # Keep each GPU batch the same size as the archived evaluator.
            scores = [model(variant.contiguous().to(config['device'])).flatten().sigmoid().cpu()
                      for variant in images.unbind(dim=1)]
            result.update(zip(identities, torch.stack(scores, dim=1).tolist()))
    return result


def paired_summary(low, high, threshold):
    delta = low - high
    return dict(n=len(low), median_low_minus_high=float(np.median(delta)),
                iqr_low_minus_high=np.quantile(delta, [.25, .75]).tolist(),
                lower_score_on_lower_quality=int((delta < 0).sum()),
                higher_score_on_lower_quality=int((delta > 0).sum()),
                high_positive_low_negative=int(((high > threshold) & (low <= threshold)).sum()),
                high_negative_low_positive=int(((high <= threshold) & (low > threshold)).sum()))


def real_scores(model, config):
    proof = read(PAIRS / 'real_pairs_summary.json')['Real Fundus']
    archive_hash = sha256(REAL)
    assert archive_hash == proof['archive_sha256']
    pairs = pd.read_csv(PAIRS / 'real_pairs_quality.csv', dtype={'pair_id': str})
    pairs = pairs.loc[pairs.source.eq('Real Fundus')].copy().reset_index(drop=True)
    assert len(pairs) == proof['pairs'] == 120
    frozen = set(pd.read_csv(MANIFEST, usecols=['sha256']).sha256)
    transform = ImageRows([], ROOT / 'data', config['image_size'], 'glaucoma').transform
    scores = []
    with ZipFile(REAL) as archive, torch.inference_mode():
        for start in range(0, len(pairs), 8):
            images = []
            for pair in pairs.iloc[start:start+8].itertuples():
                for name in (pair.low_image, pair.high_image):
                    assert hashlib.sha256(archive.read(name)).hexdigest() not in frozen
                    with archive.open(name) as handle, Image.open(handle) as image:
                        side = max(image.size)
                        square = ImageOps.pad(image.convert('RGB'), (side, side), color=0)
                        images.append(transform(square))
            scores.extend(model(torch.stack(images).to(config['device'])).flatten().sigmoid().cpu().tolist())
    pairs[['p_low','p_high']] = np.asarray(scores).reshape(-1, 2)
    return pairs, archive_hash


def main():
    torch.set_num_threads(4)
    archived = read(REPORT)
    assert not archived['protected_test_inference']
    val, dev, _ = data_for_analysis()
    rows = load_rows(MANIFEST, 'glaucoma', 'boundary', RELEASE)
    matched_scores(val, dev, archived['provenance'], rows)
    val, dev = [frame.loc[frame.source.eq(AIROGS)].copy() for frame in (val, dev)]
    assert (len(val), len(dev), int(dev.label.sum())) == (767, 773, 380)
    review, base_mask, alt_mask = routing(val, dev, archived['airogs']['methods'])
    dev['base_review'], dev['alternate_review'] = base_mask, alt_mask
    print('Matched-capacity review evaluated.', flush=True)
    model, config = checkpoint_model(EFFICIENTNET, archived['provenance']['efficientnet']['checkpoint_sha256'])
    blur = score_blur(model, config, [r for r in rows if r['source'] == AIROGS])
    dev[['original_repeat','blur_2','blur_5']] = [blur[identity] for identity in dev.sample_id]
    anchor = float((dev.original_repeat - dev.efficientnet).abs().max())
    threshold = review['thresholds']['efficientnet']
    changed_calls = int(((dev.original_repeat > threshold) != (dev.efficientnet > threshold)).sum())
    print(f'Repeated original check: max score difference={anchor:.6f}, changed calls={changed_calls}.', flush=True)
    # The archive batches all sources together. Validate the AIROGS-only GPU pass
    # against its scores and require identical classifications at the locked threshold.
    assert anchor < .001 and changed_calls == 0, (anchor, changed_calls)
    synthetic = {
        str(label): {
            str(radius): paired_summary(group[f'blur_{radius}'].to_numpy(), group.original_repeat.to_numpy(), threshold)
            for radius in RADII[1:]
        } for label, group in dev.groupby('label')
    }
    print('Synthetic focus pairs scored.', flush=True)
    pairs, archive_hash = real_scores(model, config)
    recon = paired_summary(pairs.p_low.to_numpy(), pairs.p_high.to_numpy(), threshold)
    LOCAL.mkdir(exist_ok=True, parents=True)
    case_file = LOCAL / 'airogs-development.csv'
    dev[[*ID, 'efficientnet','resnet50','q_eyeq','base_review','alternate_review',
         'original_repeat','blur_2','blur_5']].to_csv(case_file, index=False)
    pair_file = LOCAL / 'real-fundus-development.csv'
    pairs.to_csv(pair_file, index=False)
    write_json(OUTPUT / 'review-failures-results.json', dict(
        stage='exploratory_development', protected_test_inference=False,
        validation_images=len(val), development_images=len(dev), positives=int(dev.label.sum()),
        development_groups=int(dev.related_group.nunique()), screening_photo_positive_rate=PREVALENCE,
        ranking=review, synthetic_focus=synthetic,
        real_fundus=dict(**recon, archive_sha256=archive_hash, labels_available=False, exact_manifest_matches=0),
        provenance=dict(backbone_report_sha256=sha256(REPORT), manifest_sha256=sha256(MANIFEST),
                        analysis_sha256=sha256(Path(__file__)), original_repeat_max_abs_difference=anchor,
                        airogs_case_table_sha256=sha256(case_file), real_pairs_table_sha256=sha256(pair_file)),
    ))
    print('Review, synthetic focus, and real-pair analyses saved.', flush=True)


if __name__ == '__main__':
    main()
