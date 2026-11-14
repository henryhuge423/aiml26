"""Verify current research-training inputs against the completed readiness evidence."""
from __future__ import annotations
import csv
import json
import math
import re
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.data import load_rows,sha256
from training.engine import fingerprint_inputs,load_cache
from training.models import INITIALIZATIONS
from training.teacher import TEACHER_SHA256


def main():
    directory=ROOT/'data/prepared/formal_v2';manifest=directory/'manifest.csv';release=directory/'release.json'
    metadata=json.loads(release.read_text(encoding='utf-8'))
    expected={('glaucoma','train'):9874,('glaucoma','val'):1341,('glaucoma','boundary'):1397,
        ('glaucoma','test'):1397,('glaucoma','external_acrima'):705,('glaucoma','external_hygd'):737,
        ('quality','train'):1794,('quality','val'):452}
    for (task,split),count in expected.items():
        rows=load_rows(manifest,task,split,release)
        assert len(rows)==count,(task,split,len(rows))
        assert all((ROOT/'data'/r['path']).is_file() for r in rows)
        assert not any(r['path'].endswith('1281_right.jpg') for r in rows)
    for filename,expected_sha in INITIALIZATIONS.values():
        assert sha256(ROOT/'models/checkpoints'/filename)==expected_sha
    assert sha256(ROOT/'models/checkpoints/net_g_226264S4.pth')==TEACHER_SHA256
    cache=load_cache(directory/'teacher_scores.csv',manifest,ROOT,release)
    assert len(cache)==2246 and all(math.isfinite(float(r['teacher_q'])) for r in cache.values())
    with manifest.open(encoding='utf-8-sig') as f:all_rows=list(csv.DictReader(f))
    quality={r['sample_id']:r for r in all_rows if r['task']=='quality'}
    assert all(all(cache[k][field]==quality[k][field] for field in ('sample_id','path','sha256','split')) for k in cache)
    external=json.loads((ROOT/'data/audit/deep/luna_release_review/verification_v2.json').read_text(encoding='utf-8'))
    assert external['version']=='formal_v2' and not external['issues'] and external['status'].startswith('PASS')
    assert external['manifest_rows']==len(all_rows)
    assert external['release_manifest_sha256']==sha256(manifest),'independent review is for a different manifest'
    proof_path=ROOT/'runs/preflight/formal_v2/verification.json'
    proof=json.loads(proof_path.read_text(encoding='utf-8'))
    assert proof['passed'] and len(proof['models'])==3
    purposes=[];models=[]
    for result in proof['models']:
        name=result['model']
        assert result['passed'] and result['weight_max_abs_difference']<=1e-6
        assert result['resumed_next_loss_abs_difference']<=1e-6
        config=json.loads((ROOT/'configs'/f'{name}.json').read_text(encoding='utf-8'))
        smoke=json.loads((ROOT/'runs/preflight/formal_v2'/name/'resumed.json').read_text(encoding='utf-8'))
        for key,value in config.items():
            if key not in {'purpose','epochs','run_dir','checkpoint_every'}:assert smoke[key]==value,(name,key)
        assert fingerprint_inputs(smoke,ROOT)==result['completed']['fingerprint'],name
        assert result['completed']['parameters_changed'] and result['completed']['global_step']==4
        assert math.isfinite(result['completed']['last_loss'])
        for file in ('last.pt','best.pt','validation_predictions.csv'):
            assert (ROOT/'runs/preflight/formal_v2'/name/'resumed'/file).is_file()
        purposes.append((ROOT/config['run_dir']/'last.pt').exists())
        models.append(dict(name=name,config_sha256=sha256(ROOT/'configs'/f'{name}.json'),
            real_gpu_training_validation_resume_passed=True,
            resumed_weight_difference=result['weight_max_abs_difference'],
            resumed_loss_difference=result['resumed_next_loss_abs_difference']))
    log=(ROOT/'data/audit/deep/tests_final.log').read_text(encoding='utf-8',errors='replace')
    assert re.search(r'Ran 31 tests',log) and '\nOK' in log and 'FAILED' not in log
    documents=[ROOT/'README.md',ROOT/'docs/operations/training-runbook.md',ROOT/'docs/operations/training-readiness.md',
               ROOT/'docs/operations/training-protocol-2026-11-09.md']
    missing=[(str(path),target) for path in documents for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8'))
             if not target.startswith(('http','#')) and not (path.parent/target).exists()]
    assert not missing,missing
    report=dict(research_training_ready=True,cohort_version='formal_v2',
        release_sha256=sha256(release),manifest_sha256=sha256(manifest),manifest_rows=len(all_rows),
        teacher_targets=2246,tests_passed=31,models=models,
        independent_release_review_status=external['status'],
        all_patient_independence_proven=metadata['all_patient_independence_proven'],
        quality_teacher_unseen_generalization_proven=metadata['quality_teacher_unseen_generalization_proven'],
        formal_long_training_started=any(purposes),
        readiness_scope='frozen research cohort and runnable verified training jobs; not trained-model performance',
        preflight_evidence_sha256=sha256(proof_path),tests_log_sha256=sha256(ROOT/'data/audit/deep/tests_final.log'))
    (directory/'readiness.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
