"""Verify report artifacts and frozen score inputs without test-image inference."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
import pymupdf

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/progress/2026-11-15'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    r=json.loads((OUT/'quality-error-analysis.json').read_text(encoding='utf-8'))
    for relative,expected in r['input_sha256'].items():assert sha(ROOT/relative)==expected,relative
    q=pd.read_csv(ROOT/'runs/expanded_20261113/pq_quality_20261114/quality_scores.csv')
    p=pd.read_csv(ROOT/'runs/expanded_20261113/expanded_degradation50/last_development_predictions.csv')
    p=p[p.condition.eq('original')]
    assert p.sample_id.is_unique and q.sample_id.is_unique
    joined=p.merge(q,on=['sample_id','source','label','split'],validate='one_to_one')
    assert len(joined)==len(p)==len(q)==4099 and set(joined['split'])=={'val','boundary'}
    for role in ('val','boundary'):
        d=joined[joined['split'].eq(role)]
        reference=r['roles'][role]['overall']
        assert len(d)==reference['n'] and int(d.label.sum())==reference['positive']
        fn=int(((d.prediction<=.5)&d.label.eq(1)).sum());fp=int(((d.prediction>.5)&d.label.eq(0)).sum())
        assert (fn,fp)==(reference['fn'],reference['fp'])
        assert sum(v['n'] for v in r['roles'][role]['quality_bins'].values())==len(d)
        assert sum(v['errors'] for v in r['roles'][role]['quality_bins'].values())==fn+fp
    snapshot=json.loads((OUT/'training-snapshot.json').read_text())
    assert snapshot['preflight']['passed'] and len(snapshot['plan']['jobs'])==18
    for proof in snapshot['preflight']['models']:
        assert proof['passed'] and proof['weight_max_abs_difference']==0 and proof['step_records_exact'] and proof['epoch_crossed']
    for evidence in snapshot.get('evaluated_checkpoint_evidence',[]):
        directory=ROOT/'runs/followup_20261115'/evidence['run']
        for update,proof in evidence['updates'].items():
            stem=f'update_{int(update):06d}'
            assert sha(directory/'validation_checkpoints'/f'{stem}.pt')==proof['checkpoint_sha256']
            path=directory/'development'/stem/'predictions.csv'
            assert sha(path)==proof['prediction_sha256']
            predictions=pd.read_csv(path)
            assert set(predictions['split'])=={'boundary'} and set(predictions.condition)=={'original','defocus_2_at512','exposure_070'}
            assert set(predictions[predictions.condition.eq('original')].sample_id)==set(joined[joined['split'].eq('boundary')].sample_id)
    links=[];documents=[]
    candidates=[*OUT.glob('*.md'),ROOT/'docs/progress/README.md',ROOT/'docs/progress/README.en.md',
        ROOT/'README.md',ROOT/'README.en.md',ROOT/'docs/operations/training-runbook.md',ROOT/'docs/operations/training-runbook.en.md',
        ROOT/'runs/followup_20261115/README.md',ROOT/'runs/followup_20261115/README.en.md']
    for path in candidates:
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if target.startswith(('http:','https:','mailto:','#')):continue
            target=target.split('#')[0]
            assert (path.parent/target).resolve().exists(),(path,target)
            links.append(dict(file=path.relative_to(ROOT).as_posix(),target=target))
    for lang in ('zh','en'):
        files=[OUT/f'quality-and-training.{lang}.{suffix}' for suffix in ('md','tex','pdf')]
        assert all(p.exists() and p.stat().st_size>1000 for p in files)
        pdf=pymupdf.open(files[-1]);assert all(page.get_text().strip() for page in pdf)
        text=''.join(page.get_text() for page in pdf)
        assert '\ufffd' not in text and '2053' in re.sub(r'[,\s]','',text)
        figures=list(OUT.glob(f'*.{lang}.png'));assert len(figures)==4
        documents.append(dict(language=lang,pages=len(pdf),sha256={p.name:sha(p) for p in [*files,*figures]}))
    result=dict(passed=True,score_identity_rows=len(joined),input_hashes_verified=len(r['input_sha256']),
        checked_local_links=len(links),documents=documents,protected_test_inference=False,
        training_snapshot_sha256=sha(OUT/'training-snapshot.json'))
    (OUT/'delivery-verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=True,rows=4099,checked_local_links=len(links),pages={d['language']:d['pages'] for d in documents})))


if __name__=='__main__':main()
