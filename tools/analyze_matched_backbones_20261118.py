"""Compare the two backbones on the same training protocol and development images."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.evaluate_expansion_20261113 import predict
from tools.rehearse_proposal_review_20261116 import (
    FOLLOWUP, ID, MANIFEST, RELEASE, SOURCES, align, bootstrap_totals, counts,
    data_for_analysis, group_weights, margin, ranked_mask, read, sha256, write_json,
)
from training.data import load_rows
from training.engine import fingerprint_inputs
from training.models import create_model

RUN = ROOT / 'runs/backbone_20261118/resnet50'
EFFICIENTNET = FOLLOWUP / 'seed20261117/quota65_mixed50'
REPORT = ROOT / 'docs/progress/2026-11-18/backbone-results.json'
AIROGS = SOURCES[0]
FINAL_UPDATE = 2400
BOOTSTRAP_REPEATS = 2000


def checked_checkpoint(directory, config, update):
    checkpoint = directory / 'validation_checkpoints' / f'update_{update:06d}.pt'
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if payload['fingerprint'] != fingerprint_inputs(config, ROOT):
        raise ValueError(f'Training fingerprint differs for {checkpoint}')
    if payload['state']['optimizer_updates'] != update:
        raise ValueError(f'Unexpected update count for {checkpoint}')
    return checkpoint, payload


def validation_curve(directory, update):
    curve = []
    for step in range(300, update + 1, 300):
        path = directory / 'validation_checkpoints' / f'update_{step:06d}.json'
        entry = read(path)
        curve.append(dict(update=step, airogs_auc=entry['per_source'][AIROGS]['auc'],
                          five_source_macro_auc=entry['selection_value']))
    return curve


def evaluate(val, dev):
    methods = ('efficientnet', 'resnet50', 'historical_resnet50')
    y = dev.label.to_numpy()
    positives = y == 1
    k = (len(dev) + 9) // 10
    weights = group_weights(dev.related_group.to_numpy(), BOOTSTRAP_REPEATS, 20261118)
    k_boot = (weights.sum(axis=1).astype(int) + 9) // 10
    quality = 1 - dev.q_eyeq.clip(0, 1).to_numpy()
    results, residuals = {}, {}
    for method in methods:
        p = dev[method].to_numpy()
        negatives = val.loc[val.label.eq(0), method]
        threshold = float(np.quantile(negatives, .95, method='higher'))
        result = dict(
            validation_threshold=threshold,
            validation_specificity=float((negatives <= threshold).mean()),
            auc=float(roc_auc_score(y, p)),
            no_review=counts(y, p, threshold, np.zeros(len(y), dtype=bool)),
            review={},
        )
        uncertainty = margin(p, threshold)
        policies = {'uncertainty': uncertainty}
        if method != 'historical_resnet50':
            policies['quality'] = uncertainty + quality
        for policy, score in policies.items():
            order = np.argsort(-score, kind='stable')
            reviewed = ranked_mask(order, k)
            result['review'][policy] = counts(y, p, threshold, reviewed)
            if method == 'historical_resnet50':
                continue
            miss = (p <= threshold) & positives
            false_positive = (p > threshold) & ~positives
            residuals[method, policy] = {
                key: bootstrap_totals(weights, order, k_boot, condition)
                for key, condition in {'fn': miss, 'fp': false_positive,
                                       'errors': miss | false_positive}.items()
            }
        results[method] = result

    contrasts = {}
    for left, right in [
        (('resnet50', 'uncertainty'), ('efficientnet', 'uncertainty')),
        (('efficientnet', 'quality'), ('efficientnet', 'uncertainty')),
        (('resnet50', 'quality'), ('resnet50', 'uncertainty')),
    ]:
        left_counts = results[left[0]]['review'][left[1]]
        right_counts = results[right[0]]['review'][right[1]]
        contrasts[f'{"/".join(left)} minus {"/".join(right)}'] = {
            key: dict(delta=left_counts[key] - right_counts[key],
                      paired_95pct=np.quantile(
                          residuals[left][key] - residuals[right][key], [.025, .975]).tolist())
            for key in ('fn', 'fp', 'errors')
        }
    return dict(validation_images=len(val), images=len(dev), positives=int(y.sum()),
                linked_groups=int(dev.related_group.nunique()), reviewed=k,
                methods=results, contrasts=contrasts)


def main(update):
    torch.set_num_threads(4)
    directories = {'efficientnet': EFFICIENTNET, 'resnet50': RUN}
    completed_runs = {}
    if update == FINAL_UPDATE:
        for method, directory in directories.items():
            summary = read(directory / 'run_summary.json')
            if (not summary['budget_completed'] or not summary['parameters_changed']
                    or summary['optimizer_updates'] != update):
                raise ValueError(f'Training did not complete the planned budget: {directory}')
            completed_runs[method] = {key: summary[key] for key in (
                'optimizer_updates', 'budget_completed', 'elapsed_seconds', 'peak_gpu_bytes',
            )}
    val, dev, reference = data_for_analysis()
    val['historical_resnet50'], dev['historical_resnet50'] = val.p_b, dev.p_b
    boundary = load_rows(MANIFEST, 'glaucoma', 'boundary', RELEASE)
    provenance, exposure = {}, {}
    for method, directory in directories.items():
        config = read(directory / 'config.json')
        checkpoint, payload = checked_checkpoint(directory, config, update)
        state = payload['state']
        counts_hash = hashlib.sha256(
            json.dumps(state['sample_draw_counts'], sort_keys=True).encode()).hexdigest()
        exposure[method] = dict(
            sample_draw_counts_sha256=counts_hash,
            source_class_draws=state['source_class_draws'],
            degradation_family_draws=state['degradation_family_draws'],
            degradation_stratum_draws=state['source_class_degradation_draws'],
            amp_overflows=state['amp_overflows'],
        )
        model = create_model(config['architecture'], ROOT / 'models/checkpoints').to(config['device'])
        model.load_state_dict(payload['model'], strict=True)
        model.eval()
        prediction = pd.DataFrame(predict(model, boundary, config, 'original'))
        if set(prediction['split']) != {'boundary'}:
            raise ValueError('Scoring must only use the boundary split')
        path = checkpoint.with_suffix('.csv')
        validation = pd.read_csv(path).rename(columns={'prediction': method})
        val[method] = align(val, validation[[*ID, method]], ID, method)
        dev[method] = align(dev, prediction.rename(columns={'prediction': method})[[*ID, method]], ID, method)
        provenance[method] = dict(
            checkpoint=checkpoint.relative_to(ROOT).as_posix(), checkpoint_sha256=sha256(checkpoint),
            fingerprint=payload['fingerprint'], validation_predictions_sha256=sha256(path),
            development_predictions_sha256=hashlib.sha256(
                dev[['sample_id', method]].to_csv(index=False, lineterminator='\n').encode()).hexdigest(),
            validation_curve=validation_curve(directory, update),
        )
        del model, payload

    configurations = [read(directory / 'config.json') for directory in directories.values()]
    shared = [{k: v for k, v in config.items() if k not in {'architecture', 'run_dir'}}
              for config in configurations]
    if shared[0] != shared[1]:
        raise ValueError('Expected matching training configurations')
    if exposure['efficientnet'] != exposure['resnet50']:
        raise ValueError('The runs differ in training exposure')
    results = dict(
        update=update, completed_runs=completed_runs,
        matched_training_exposure=exposure['resnet50'], shared_config=shared[0],
        protected_test_inference=False, inference_roles=['val', 'boundary'],
        settings=dict(positive_call='score > validation negative 95th percentile (higher)',
                      review_budget='ceil(0.10 * cohort size); rank ties by sample_id ascending',
                      quality_weight=1, bootstrap_repeats=BOOTSTRAP_REPEATS, bootstrap_seed=20261118),
        airogs=evaluate(val[val.source.eq(AIROGS)], dev[dev.source.eq(AIROGS)]),
        pooled=evaluate(val, dev), provenance=provenance,
        historical_comparator=reference['model_b'],
        quality_checkpoint_sha256=reference['quality_checkpoints']['q_eyeq'],
        reference_input_sha256=reference['input_sha256'],
        manifest_sha256=sha256(MANIFEST), release_sha256=sha256(RELEASE),
        analysis_sha256=sha256(Path(__file__)),
    )
    destination = REPORT if update == FINAL_UPDATE else RUN / f'checkpoint-{update}-analysis.json'
    write_json(destination, results)
    print(destination.relative_to(ROOT))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--update', type=int, choices=range(300, FINAL_UPDATE + 1, 300), default=FINAL_UPDATE)
    main(parser.parse_args().update)
