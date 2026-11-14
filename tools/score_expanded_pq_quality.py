"""Score model-validation and rule-development images with the frozen quality student."""
import csv
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.data import load_rows, sha256
from training.engine import fingerprint_inputs
from training.models import create_model
from tools.score_repaired_boundary import InferenceRows, predict, read


def main():
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    config = json.loads((ROOT/'configs/formal_v3/quality_student.json').read_text())
    checkpoint = ROOT/config['run_dir']/'best.pt'
    proof = json.loads((ROOT/'data/audit/session_20261110/completion_verification.json').read_text())
    evidence = next(x for x in proof['models'] if x['name'] == 'quality_student')
    assert sha256(checkpoint) == evidence['best_checkpoint_sha256']
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert state['fingerprint'] == evidence['fingerprint']
    current_fingerprint = fingerprint_inputs(config, ROOT)
    model = create_model(config['architecture'], ROOT/'models/checkpoints').to('cuda')
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    old_val = load_rows(ROOT/config['manifest'], 'quality', 'val', ROOT/config['release'])
    by_id = {r['sample_id']: r for r in old_val}
    anchors = read(ROOT/'data/audit/session_20261110/selected_validation/quality_student/validation_predictions.csv')[:16]
    anchor_scores = predict(model, [by_id[r['sample_id']] for r in anchors], config)
    max_diff = max(abs(anchor_scores[r['sample_id']] - float(r['prediction'])) for r in anchors)
    assert max_diff < 1e-6, max_diff
    manifest = ROOT/'data/prepared/expanded_20261113/manifest.csv'
    release = ROOT/'data/prepared/expanded_20261113/release.json'
    rows = sum((load_rows(manifest, 'glaucoma', role, release) for role in ('val', 'boundary')), [])
    assert len(rows) == 4099 and len({r['sample_id'] for r in rows}) == len(rows)
    out = ROOT/'runs/expanded_20261113/pq_quality_20261114'
    out.mkdir(parents=True, exist_ok=True)
    dataset = InferenceRows(rows, ROOT/'data', config['image_size'], 'quality')
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    scores = {}
    with torch.inference_mode():
        for images, ids in loader:
            values = model(images.to('cuda')).flatten().cpu().tolist()
            for identity, value in zip(ids, values):
                assert identity not in scores and math.isfinite(value)
                scores[identity] = value
            if len(scores) % 320 == 0:
                print(f'scored {len(scores)}/{len(rows)}', flush=True)
    assert len(scores) == len(rows)
    old_boundary = read(ROOT/'data/audit/data_expansion_20261113/repaired_boundary_pq.csv')
    prior_diff = max(abs(scores[r['sample_id']] - float(r['q'])) for r in old_boundary)
    assert len(old_boundary) == 1220 and prior_diff < 5e-5, prior_diff
    path = out/'quality_scores.csv'
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['sample_id','split','source','label','q'])
        writer.writeheader()
        writer.writerows(dict(sample_id=r['sample_id'], split=r['split'], source=r['source'], label=r['label'], q=scores[r['sample_id']]) for r in rows)
    result = dict(rows=len(rows), roles={'val':2046, 'boundary':2053}, checkpoint_sha256=sha256(checkpoint),
                  recorded_training_fingerprint=state['fingerprint'], current_training_fingerprint=current_fingerprint,
                  manifest_sha256=sha256(manifest), scores_sha256=sha256(path), anchor_max_abs_difference=max_diff,
                  original_boundary_max_abs_difference=prior_diff,
                  quality_score_kind='FTHNet_distilled_student_raw_regression', protected_test_inference=False)
    (out/'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
