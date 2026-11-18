"""Project clinician review and benchmark acquisition simulation on paired photos."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
from scipy.special import logit
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.smoke_downloaded_fundus_models_20261113 import module, RELEASES
from tools.validate_fundus_degradation_20261113 import digest, image_statistics, pairs
from training.fundus_degradation import FundusDegradation, retinal_context

PRIOR = ROOT / 'docs/progress/2026-11-18/review-failures-results.json'
DEV = ROOT / 'runs/review_failures_20261118/airogs-development.csv'
REAL = ROOT / 'data/external/real_fundus/Real_Fundus.zip'
OUT = ROOT / 'runs/review_and_degradation_20261118'
METRICS = ('mean_luminance', 'green_contrast_std', 'green_laplacian_variance')
SEED = 20261118


def take(order, weight, budget):
    ordered = weight[order]
    allocation = np.zeros_like(weight)
    allocation[order] = np.clip(budget - (ordered.cumsum() - ordered), 0, ordered)
    return allocation


def clinician_projection(prior):
    assert digest(DEV) == prior['provenance']['airogs_case_table_sha256']
    frame = pd.read_csv(DEV)
    p = frame.efficientnet.to_numpy()
    y = frame.label.to_numpy() == 1
    t = prior['ranking']['thresholds']['efficientnet']
    positive = p > t
    pi = prior['screening_photo_positive_rate']
    weight = np.where(y, 1000 * pi / y.sum(), 1000 * (1 - pi) / (~y).sum())
    log_scores = logit(frame[['efficientnet', 'resnet50']].clip(1e-6, 1-1e-6).to_numpy())
    risk = log_scores @ prior['ranking']['fit_logit_coefficients']
    base_score = -np.abs(log_scores[:, 0] - logit(t)) + 1 - frame.q_eyeq.clip(0, 1)
    base = take(np.argsort(-base_score, kind='stable'), weight, 100)
    order = np.argsort(-risk, kind='stable')
    candidate = take(order[~positive[order]], weight, base[~positive].sum()) + base * positive
    fn = y & ~positive
    fp = ~y & positive
    baseline = np.array([weight[fn].sum(), weight[fp].sum()])
    scenarios = {
        'perfect_reference_ceiling': (1., 1.),
        'selected_eyepacs_graders': (.86, .964),
        'intermediate_assumption': (.75, .90),
        'lower_sensitivity_assumption': (.60, .88),
    }
    result = {}
    published = prior['ranking']['scenarios']['airogs_public_photo_mix']['policies']
    for name, plan in [('quality_gray_zone', base), ('second_model_negative_reranking', candidate)]:
        assert np.isclose(plan.sum(), 100)
        routed = np.array([plan[y].sum(), plan[~y].sum()])
        captured = np.array([plan[fn].sum(), plan[fp].sum()])
        assert np.allclose(captured, [published[name]['captured_fn'], published[name]['captured_fp']])
        reports = {}
        for key, (sensitivity, specificity) in scenarios.items():
            after = baseline - captured + routed * [1-sensitivity, 1-specificity]
            reports[key] = dict(sensitivity=sensitivity, specificity=specificity,
                                remaining_fn=float(after[0]), remaining_fp=float(after[1]),
                                net_errors_prevented=float(baseline.sum() - after.sum()))
        result[name] = dict(reviewed_reference_positive=float(routed[0]), reviewed_reference_negative=float(routed[1]),
                            captured_fn=float(captured[0]), captured_fp=float(captured[1]),
                            sensitivity_to_reduce_fn=float(1 - captured[0]/routed[0]), scenarios=reports)
    return dict(units='per 1000 photos', positive_rate=pi, reviews=100,
                model_only_fn=float(baseline[0]), model_only_fp=float(baseline[1]), policies=result)


def asset(archive, name):
    with archive.open(name) as stream, Image.open(stream) as image:
        rgb = image.convert('RGB')
    side = max(rgb.size)
    return ImageOps.pad(rgb, (side, side), color=0)


def statistics(image):
    result = image_statistics(image)
    return np.array([result[key] for key in METRICS])


def blur_exposure(image, sigma, gain):
    filtered = np.asarray(image.filter(ImageFilter.GaussianBlur(sigma)), dtype=np.float32)
    output = np.rint(np.clip(filtered * gain, 0, 255)).astype(np.uint8)
    output[np.asarray(image).max(axis=2) <= 2] = 0
    return Image.fromarray(output)


def fit_pair(high, low_values, high_values):
    gain = float(np.clip(low_values[0] / high_values[0], .25, 2.5))
    diameter = retinal_context(high)['field_diameter_pixels']
    candidates = [0, .5, .8, 1.2, 1.7, 2.4, 3.2, 4.5, 6.5, 9, 13]
    errors = []
    for radius in candidates:
        estimated = statistics(blur_exposure(high, radius*diameter/512, gain))[2]
        errors.append(abs(np.log(max(estimated, 1e-12) / max(low_values[2], 1e-12))))
    return candidates[int(np.argmin(errors))], gain


def author_modules():
    path = ROOT / 'docs/progress/2026-11-13/training-results/degradation-downloads.json'
    inventory = {Path(r['local_directory']).name: r for r in json.loads(path.read_text())['releases']}
    cofe_dir = 'Fundus-correction-cofe-Net-5366d53808bc'
    arc_dir = 'ArcNet-Medical-Image-Enhancement-20821fb1116b'
    files = [(cofe_dir, 'degradation_code/degrad_de.py'), (cofe_dir, 'degradation_code/utils_de.py'),
             (arc_dir, 'util/cataract_simulation.py')]
    for key, name in files:
        expected = next(entry['sha256'] for entry in inventory[key]['files'] if entry['path'] == name)
        assert digest(RELEASES/key/name) == expected
    folder = RELEASES / cofe_dir / 'degradation_code'
    sys.path.insert(0, str(folder))
    try:
        cofe = module(folder/'degrad_de.py', 'paired_cofe')
    finally:
        sys.path.remove(str(folder))
    return cofe, module(RELEASES/arc_dir/'util/cataract_simulation.py', 'paired_arc')


def cofe_versions(model, at_512, seed):
    im = np.asarray(at_512)
    mask = (im.max(axis=2) > 2)[None, :, :].astype(np.float32)
    result = {}
    for mode in ('100', '111'):
        random.seed(seed)
        np.random.seed(seed)
        output, _ = model.DE_process(at_512, mask, 512, 512, mode)
        assert output.shape == (512, 512, 3) and np.isfinite(output).all()
        result[f'cofe_{mode}'] = Image.fromarray(output)
    return result


def arc_version(model, at_512, scratch, seed):
    model.setup_seed(seed)
    model.NUM_PER_NOISE = 1
    model.IMAGE_DIR = model.OUTPUT_DIR = str(scratch)
    input_path = scratch/'photo.png'
    at_512.save(input_path)
    model.cataract_noise(input_path.name)
    with Image.open(scratch/'photo-0.png') as sheet:
        assert sheet.size == (1024, 512)
        return sheet.crop((0, 0, 512, 512)).convert('RGB')


def summary(ratios, truth):
    stats = {}
    for column, metric in enumerate(METRICS):
        values = ratios[:, column]
        targets = truth[:, column]
        stats[metric] = dict(
            median=float(np.median(values)), p10=float(np.quantile(values, .10)),
            p90=float(np.quantile(values, .90)),
            log_ratio_wasserstein=float(wasserstein_distance(np.log(values), np.log(targets))),
        )
    return stats


def contact_sheet(previews, records):
    labels = dict(high='Clearer original', real_low='Real poorer', current='Current mix',
                  cofe_100='CoFe blur', cofe_111='CoFe combined', arc='ArcNet cataract',
                  calibrated='Pair-calibrated blur + exposure')
    thumb = 230
    height = 270
    canvas = Image.new('RGB', (thumb*len(labels), height*len(previews)), 'white')
    drawing = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15)
    for row, index in enumerate(previews):
        for col, (key, label) in enumerate(labels.items()):
            x, y = col*thumb, row*height
            drawing.text((x+4, y+6), label, fill='#174055', font=font)
            canvas.paste(records[index][key].resize((thumb-4,thumb-4)), (x+2,y+35))
    canvas.save(OUT/'degradation-examples.png')


def downstream_scores(records, identities, prior):
    import torch
    from tools.analyze_review_failures_20261118 import EFFICIENTNET, REPORT, checkpoint_model, read
    from training.data import ImageRows

    torch.set_num_threads(4)
    expected = read(REPORT)['provenance']['efficientnet']['checkpoint_sha256']
    model, config = checkpoint_model(EFFICIENTNET, expected)
    transform = ImageRows([], ROOT/'data', config['image_size'], 'glaucoma').transform
    scores = {}
    with torch.inference_mode():
        for name in records[0]:
            values = []
            for start in range(0, len(records), 16):
                batch = torch.stack([transform(r[name]) for r in records[start:start+16]])
                values.extend(model(batch.to(config['device'])).flatten().sigmoid().cpu().tolist())
            scores[name] = np.asarray(values)
    archived = DEV.with_name('real-fundus-development.csv')
    assert digest(archived) == prior['provenance']['real_pairs_table_sha256']
    cached = pd.read_csv(archived, dtype={'pair_id': str}).set_index('pair_id').loc[identities]
    anchor = max(np.abs(scores['high']-cached.p_high).max(), np.abs(scores['real_low']-cached.p_low).max())
    assert anchor < .002, anchor
    real_change = scores['real_low'] - scores['high']
    threshold = prior['ranking']['thresholds']['efficientnet']
    results = {}
    for name, values in scores.items():
        if name == 'high':
            continue
        delta = values - scores['high']
        results[name] = dict(
            median_change=float(np.median(delta)),
            median_absolute_change=float(np.median(np.abs(delta))),
            absolute_change_over_005=int((np.abs(delta) > .05).sum()),
            threshold_crossings=int(((values > threshold) != (scores['high'] > threshold)).sum()),
            score_change_wasserstein=float(wasserstein_distance(delta, real_change)),
        )
    return dict(checkpoint_sha256=expected, original_score_anchor_max_difference=float(anchor), methods=results)


def benchmark(prior):
    archive_hash = digest(REAL)
    assert archive_hash == prior['real_fundus']['archive_sha256']
    cv2.setNumThreads(1)
    cofe, arc = author_modules()
    rng = np.random.default_rng(SEED)
    current = FundusDegradation(1)
    rows, images, bank = [], [], []
    at_320 = lambda image: image.resize((320, 320), Image.Resampling.BILINEAR)
    with ZipFile(REAL) as archive, TemporaryDirectory(dir=OUT) as tmp:
        identities = sorted(pairs(archive), key=lambda entry: hashlib.sha256(f'{SEED}:{entry[0]}'.encode()).digest())
        assert len(identities) == 120 and len({e[0] for e in identities}) == 120
        for identity, a, b in identities[:60]:
            high, low = [at_320(asset(archive, path)) for path in (a,b)]
            bank.append(fit_pair(high, statistics(low), statistics(high)))
        for index, (identity, a, b) in enumerate(identities[60:]):
            seed = int(rng.integers(0, 2**31))
            native = asset(archive, a)
            high, low = at_320(native), at_320(asset(archive, b))
            reference = statistics(high)
            native_512 = native.resize((512,512), Image.Resampling.BILINEAR)
            radius, gain = bank[index]
            versions = {'high': high, 'real_low': low, 'current': current(high, seed=seed)}
            versions.update({key: at_320(image) for key, image in cofe_versions(cofe, native_512, seed).items()})
            versions['arc'] = at_320(arc_version(arc, native_512, Path(tmp), seed))
            diameter = retinal_context(high)['field_diameter_pixels']
            versions['calibrated'] = blur_exposure(high, radius*diameter/512, gain)
            rows.append({key: statistics(image) / reference for key, image in versions.items()})
            images.append(versions)
    truth = np.stack([r['real_low'] for r in rows])
    models = {key: summary(np.stack([r[key] for r in rows]), truth) for key in rows[0] if key != 'high'}
    ordered = np.argsort(truth[:, 2])
    contact_sheet([ordered[15], ordered[44]], images)
    print('Image comparisons complete; scoring the same versions with EfficientNet-B0.', flush=True)
    model_scores = downstream_scores(images, [r[0] for r in identities[60:]], prior)
    return dict(archive_sha256=archive_hash, split_seed=SEED, calibration_photo_pairs=60,
                heldout_photo_pairs=60, metrics=models, calibration_parameters=bank, model_scores=model_scores,
                split='deterministic partition by photo pair; patient identifiers unavailable',
                clinical_labels_available=False,
                cofe_modes={'100':'author blur only','111':'author illumination, spots and blur'},
                arc_mode='author cataract simulation, one draw per photo; left half of saved pair')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prior = json.loads(PRIOR.read_text(encoding='utf-8'))
    reviews = clinician_projection(prior)
    print(json.dumps(reviews, indent=2), flush=True)
    degradation = benchmark(prior)
    report = dict(clinician_projection=reviews, degradation=degradation,
                  previous_review_results_sha256=digest(PRIOR),
                  analysis_sha256=digest(Path(__file__)),
                  sources=dict(eye_pacs='https://pubmed.ncbi.nlm.nih.gov/37113471/',
                               clinical_workflow='https://www.nice.org.uk/guidance/ng81/chapter/Recommendations',
                               real_fundus='https://arxiv.org/abs/2201.00466',
                               cofe='https://arxiv.org/abs/2005.05594',
                               arcnet='https://arxiv.org/abs/2203.07737'))
    (OUT/'results.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    for name, values in degradation['metrics'].items():
        print(name, [(key, round(item['median'],3), round(item['log_ratio_wasserstein'],3))
                     for key, item in values.items()])
    print(json.dumps(degradation['model_scores'], indent=2))


if __name__ == '__main__':
    main()
