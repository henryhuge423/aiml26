"""Evaluate the paired calibrated-defocus pilot on development images only."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch

from tools.analyze_expansion_results_20261113 import group_weights, interval, weighted_metrics, write_json
from tools.analyze_review_failures_20261118 import AIROGS, ID, MANIFEST, RELEASE, align, margin, score_blur, selection
from tools.evaluate_expansion_20261113 import predict, poor
from tools.run_calibrated_defocus_20261118 import ARM, CONTROL, PROFILE, RUN, SEEDS, STATUS, UPDATE, fingerprint, read
from training.data import load_rows, sha256
from training.models import create_model

OUTPUT = ROOT/'docs/progress/2026-11-18/calibrated-defocus-results.json'
BENCHMARK = ROOT/'docs/progress/2026-11-18/review-degradation-benchmark.json'
STEM = f'update_{UPDATE:06d}'
ARMS = ('original_mix', 'calibrated_defocus')
MILESTONES = (300, 600, 900, 1200)
PREVALENCE = 3270/101442
REPEATS = 2000


def auc(y, scores):
    return float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None


def candidate_seed(seed, reference, val_reference, boundary_rows, stress_rows):
    directories = [CONTROL/f'seed{seed}'/ARM, RUN/f'seed{seed}']
    configs = [read(path/'config.json') for path in directories]
    differences = {k for k in configs[0].keys() | configs[1].keys()
                   if configs[0].get(k) != configs[1].get(k)}
    assert differences == {'run_dir', 'max_optimizer_updates', 'defocus_calibration'}
    seed_scores, evidence, draws = {}, {}, {}
    for name, directory, config in zip(ARMS, directories, configs):
        checkpoint = directory/'validation_checkpoints'/f'{STEM}.pt'
        checkpoint_hash = sha256(checkpoint)
        archived = read(directory/'evaluation_summary.json')['checkpoints'][str(UPDATE)] if name == ARMS[0] else None
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert state['fingerprint'] == fingerprint(config, ROOT)
        if name == ARMS[1]:
            completion = read(directory/'run_summary.json')
            assert completion['fingerprint'] == state['fingerprint'] and completion['budget_completed']
        record = state['state']
        assert record['optimizer_updates'] == record['last_validation_update'] == UPDATE
        draw_keys = ('sample_draw_counts', 'source_class_draws', 'degradation_family_draws',
                     'source_class_degradation_draws', 'amp_overflows', 'global_step')
        draws[name] = {key: record[key] for key in draw_keys}
        val_file = checkpoint.with_suffix('.csv')
        val = pd.read_csv(val_file, float_precision='round_trip')
        assert len(val) == len(val_reference) == 2046
        align(val_reference, val[[*ID, 'prediction']], ID, 'prediction')
        assert val.prediction.between(0, 1).all()
        milestones = [read(directory/'validation_checkpoints'/f'update_{step:06d}.json') for step in MILESTONES]
        for source, group in val.groupby('source'):
            observed_auc = auc(group.label.to_numpy(), group.prediction.to_numpy())
            expected_auc = milestones[-1]['per_source'][source]['auc']
            assert observed_auc == expected_auc or np.isclose(observed_auc, expected_auc, atol=1e-12, rtol=0)
        val_negatives = val.loc[val.source.eq(AIROGS) & val.label.eq(0), 'prediction']
        threshold = float(np.quantile(val_negatives, .95, method='higher'))
        if archived is not None:
            assert threshold == archived['validation_thresholds'][AIROGS]
        model = create_model(config['architecture'], ROOT/'models/checkpoints').to(config['device'])
        model.load_state_dict(state['model'], strict=True)
        model.eval()
        local = RUN/f'seed{seed}'/f'{name}-development.csv'
        if archived is not None:
            path = directory/'development'/STEM/'predictions.csv'
            assert not archived['protected_test_inference']
            assert archived['checkpoint_sha256'] == checkpoint_hash
            assert sha256(path) == archived['prediction_sha256']
            cached = pd.read_csv(path, float_precision='round_trip')
            original = cached.loc[cached.condition.eq('original')].copy()
        else:
            original = pd.DataFrame(predict(model,boundary_rows,config,'original'))
        assert set(original['split']) == {'boundary'}
        predictions = align(reference, original[[*ID,'prediction']], ID, 'prediction')
        blur = score_blur(model, config, stress_rows)
        stress = pd.DataFrame.from_dict(blur, orient='index', columns=['stress_anchor','blur2','blur5'])
        frames = reference[['sample_id','source','label','related_group','annotated_poor','q']].copy()
        frames['p'] = predictions
        frames = frames.join(stress, on='sample_id', validate='one_to_one')
        observed = frames.dropna(subset=['stress_anchor'])
        assert len(observed) == len(stress_rows) == 384
        error = float((observed.stress_anchor-observed.p).abs().max())
        assert error < .002, (name, seed, error)
        if archived is not None:
            mild = cached.loc[cached.condition.eq('defocus_2_at512')].set_index('sample_id').prediction
            mild_error = float((observed.set_index('sample_id').blur2-mild).abs().max())
            assert mild_error < .002, (name, seed, mild_error)
        frames.to_csv(local,index=False)
        evidence[name] = dict(
            disease_threshold=threshold, validation_negatives=len(val_negatives),
            validation_achieved_specificity=float((val_negatives<=threshold).mean()),
            update_checkpoints=list(MILESTONES), validation_five_source_auc=[x['selection_value'] for x in milestones],
            validation_airogs_auc=[x['per_source'][AIROGS]['auc'] for x in milestones],
            checkpoint_sha256=checkpoint_hash, validation_scores_sha256=sha256(val_file),
            development_scores_sha256=sha256(local), inference_anchor_max_abs_difference=error,
            source_class_draws=record['source_class_draws'], family_draws=record['degradation_family_draws'],
            amp_overflows=record['amp_overflows'], batches=record['global_step'],
        )
        seed_scores[name] = frames
        del model, state
        torch.cuda.empty_cache()
        print(f'Seed {seed}, {name}: original and both blur levels evaluated.', flush=True)
    matched = {key: draws[ARMS[0]][key] == draws[ARMS[1]][key] for key in draws[ARMS[0]]}
    if matched['amp_overflows'] and matched['global_step']:
        assert all(matched.values()), (seed, matched)
    return seed_scores, evidence, matched


def condition_results(all_scores):
    reference = all_scores[SEEDS[0]][ARMS[0]]
    conditions = [
        ('pooled_original', reference.p.notna(), 'p'),
        ('airogs_original', reference.source.eq(AIROGS), 'p'),
        ('stress_original', reference.blur2.notna(), 'stress_anchor'),
        ('mild_blur', reference.blur2.notna(), 'blur2'),
        ('strong_blur', reference.blur5.notna(), 'blur5'),
    ]
    output = {}
    observed_differences = {}
    for key, mask, score_column in conditions:
        cohort = reference.loc[mask]
        y = cohort.label.to_numpy()
        weights = group_weights(cohort.related_group, REPEATS, 20261118)
        values, deltas = {}, []
        for seed in SEEDS:
            scores = [all_scores[seed][arm].loc[mask,score_column].to_numpy() for arm in ARMS]
            values[str(seed)] = {arm: auc(y,p) for arm,p in zip(ARMS,scores)}
            bs = [weighted_metrics(y,p,weights)['auc'] for p in scores]
            deltas.append(bs[1]-bs[0])
        means = {arm: float(np.mean([v[arm] for v in values.values()])) for arm in ARMS}
        mean_bootstrap_difference = np.mean(deltas,axis=0)
        observed_differences[key] = mean_bootstrap_difference
        output[key] = dict(images=len(cohort), positives=int(y.sum()), linked_groups=int(cohort.related_group.nunique()),
                           by_seed=values, mean_auc=means, difference=means[ARMS[1]]-means[ARMS[0]],
                           **interval(mean_bootstrap_difference))
        if key in ('mild_blur', 'strong_blur'):
            output[key]['change_relative_to_stress_original'] = dict(
                difference=output[key]['difference']-output['stress_original']['difference'],
                **interval(mean_bootstrap_difference-observed_differences['stress_original']))
    return output


def operating_points(scores, evidence, weights):
    output = {}
    residual = {}
    n_samples = len(weights)
    for arm in ARMS:
        f = scores[arm]
        y = f.label.to_numpy() == 1
        p = f.p.to_numpy()
        threshold = evidence[arm]['disease_threshold']
        fn, fp = y & (p<=threshold), ~y & (p>threshold)
        error = np.column_stack([fn,fp]).astype(float)
        weighted = weights.copy()
        for mask, mass in [(y, PREVALENCE), (~y, 1-PREVALENCE)]:
            weighted[:,mask] *= mass / weighted[:,mask].sum(axis=1,keepdims=True)
        order = np.argsort(-(margin(p,threshold)+1-f.q.clip(0,1).to_numpy()),kind='stable')
        keep = selection(order, weighted, np.full(n_samples,.10))
        assert np.allclose(keep.sum(axis=1),.10)
        total = weighted @ error * 1000
        captured = keep @ error[order] * 1000
        residual[arm] = total - captured
        reviewed_positive = keep @ y[order] * 1000
        clinician_residual = residual[arm] + np.column_stack([reviewed_positive*.14,(100-reviewed_positive)*.036])
        output[arm] = dict(
            threshold=threshold, original_fn=int(fn.sum()), original_fp=int(fp.sum()),
            project_model_only_fn=float(total[0,0]), project_model_only_fp=float(total[0,1]),
            captured_fn_per_1000=float(captured[0,0]), captured_fp_per_1000=float(captured[0,1]),
            error_capture_fraction=float(captured[0].sum()/total[0].sum()),
            remaining_fn_per_1000=float(residual[arm][0,0]), remaining_fp_per_1000=float(residual[arm][0,1]),
            clinician_scenario_fn_per_1000=float(clinician_residual[0,0]),
            clinician_scenario_net_errors_prevented_per_1000=float(total[0].sum()-clinician_residual[0].sum()),
        )
    return output, residual[ARMS[1]]-residual[ARMS[0]]


def quality_inventory(all_scores):
    ref = all_scores[SEEDS[0]][ARMS[0]]
    inventory = {}
    for name, selected in [('annotated_poor',ref.annotated_poor), ('low_model_quality',ref.q.lt(.2))]:
        group = ref.loc[selected]
        targets = group.label.to_numpy() == 1
        outcomes = {}
        for seed in SEEDS:
            outcomes[str(seed)] = {}
            for arm in ARMS:
                p = all_scores[seed][arm].loc[selected,'p'].to_numpy()
                outcomes[str(seed)][arm] = dict(
                    auc=auc(targets,p), detected_positive_at_05=int(((p>.5)&targets).sum()),
                    false_positive_at_05=int(((p>.5)&~targets).sum()))
        inventory[name] = dict(images=len(group),positives=int(targets.sum()), negatives=int((~targets).sum()),
                               sources=group.groupby('source').size().to_dict(), outcomes=outcomes)
    return inventory


def main():
    torch.set_num_threads(4)
    status = read(STATUS)
    assert status['status'] == 'completed' and status['completed'] == list(SEEDS)
    profile = read(PROFILE)
    assert profile['benchmark_sha256'] == sha256(BENCHMARK)
    benchmark = read(BENCHMARK)['degradation']
    assert (profile['calibration_archive_sha256'],profile['photo_pairs'],profile['split_seed'],profile['parameters']) == (
        benchmark['archive_sha256'],benchmark['calibration_photo_pairs'],benchmark['split_seed'],benchmark['calibration_parameters'])
    rows = load_rows(MANIFEST,'glaucoma','boundary',RELEASE)
    assert len(rows) == 2053
    reference = pd.DataFrame(rows).sort_values('sample_id').reset_index(drop=True)
    reference['label'] = reference.label.astype(int)
    reference['annotated_poor'] = [poor(row) for row in reference.to_dict('records')]
    val_reference = pd.DataFrame(load_rows(MANIFEST,'glaucoma','val',RELEASE))
    val_reference['label'] = val_reference.label.astype(int)
    quality_file = CONTROL/'eyeq_development/quality_scores.csv'
    quality_proof = read(quality_file.with_name('verification.json'))
    assert quality_proof['scores_sha256'] == sha256(quality_file) and not quality_proof['protected_test_inference']
    q = pd.read_csv(quality_file)
    q = q[q['split'].eq('boundary')].rename(columns={'q_eyeq':'q'})
    quality_keys = ['sample_id','source','label']
    reference['q'] = align(reference,q[[*quality_keys,'q']],quality_keys,'q')
    stress = sorted(rows,key=lambda r:hashlib.sha256(r['sample_id'].encode()).hexdigest())[:384]
    scores, checks, matched = {}, {}, {}
    for seed in SEEDS:
        scores[seed], checks[seed], matched[seed] = candidate_seed(seed,reference,val_reference,rows,stress)
    airogs = reference.source.eq(AIROGS)
    groups = reference.loc[airogs,'related_group'].to_numpy()
    weights = np.vstack([np.ones(len(groups)), group_weights(groups, REPEATS, 20261118)])
    routing, differences = {}, []
    for seed in SEEDS:
        scored = {arm: frame.loc[airogs] for arm, frame in scores[seed].items()}
        routing[str(seed)], delta = operating_points(scored, checks[seed], weights)
        differences.append(delta)
    delta = np.mean(differences,axis=0)
    write_json(OUTPUT,dict(
        stage='paired_development_pilot', protected_test_inference=False, inference_roles=['val','boundary'],
        arms=list(ARMS), seeds=list(SEEDS), update=UPDATE, bootstrap_replicates=REPEATS,
        conditions=condition_results(scores), training=checks, training_draw_checks=matched,
        quality_strata=quality_inventory(scores),
        screening_scenario=dict(prevalence=PREVALENCE, review_fraction=.1, by_seed=routing,
                                clinician_sensitivity=.86, clinician_specificity=.964,
                                residual_fn_difference=float(delta[0,0]), residual_fp_difference=float(delta[0,1]),
                                residual_fn_difference_ci95=np.quantile(delta[1:,0],[.025,.975]).tolist(),
                                residual_fp_difference_ci95=np.quantile(delta[1:,1],[.025,.975]).tolist()),
        provenance=dict(manifest_sha256=sha256(MANIFEST), profile_sha256=sha256(PROFILE),
                        paired_image_benchmark_sha256=sha256(BENCHMARK), quality_scores_sha256=sha256(quality_file),
                        python_version=sys.version.split()[0], torch_version=str(torch.__version__),
                        runner_sha256=sha256(ROOT/'tools/run_calibrated_defocus_20261118.py'),
                        analysis_sha256=sha256(Path(__file__))),
    ))
    print(OUTPUT.relative_to(ROOT),flush=True)


if __name__ == '__main__':
    main()
