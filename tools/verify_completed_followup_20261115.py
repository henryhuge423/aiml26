"""Check completed experiment statistics, report inputs, links, and rendered PDFs."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import pymupdf
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.analyze_expansion_results_20261113 import digest,write_json
from tools.analyze_quality_errors_20261115 import review_counts
from tools.report_completed_followup_20261115 import OUT,BUILD,RUN,ARMS,FIGURES,SOURCES


def main():
    r=json.loads((OUT/'completed-analysis.json').read_text(encoding='utf-8'))
    assert r['pipeline']['status']=='completed' and len(r['pipeline']['completed'])==18
    for path,expected in r['input_sha256'].items():assert digest(ROOT/path)==expected,path
    assert len(r['rows'])==36 and len(r['quality_review_runs'])==18
    for row in r['rows']:
        assert row['overall']['fn']+row['overall']['fp']==row['overall']['errors']
        assert row['overall']['n']==2053 and row['overall']['positives']==814
        assert row['real_poor']['n']==12 and row['real_poor']['positives']==12
    for mode in ('selected','budget2400'):
        for arm in ARMS:
            chosen=[v for v in r['rows'] if v['mode']==mode and v['arm']==arm]
            assert len(chosen)==3
            for source in SOURCES:
                values=[v['source_auc'][source] for v in chosen]
                stats=r['aggregate'][mode][arm]['source_auc'][source]
                assert np.isclose(np.mean(values),stats['mean']) and np.isclose(np.std(values,ddof=1),stats['sd'])
    comparisons=json.loads((RUN/'comparison.json').read_text())['rows']
    for row in r['rows']:
        old=next(v for v in comparisons if v['seed']==row['seed'] and v['arm']==row['arm'] and v['update']==row['update'])
        assert old['source_auc']==row['source_auc']
        assert old['quality_error_ranking']['errors']==row['overall']['errors']
    for proof in r['checkpoint_evidence']:
        path=RUN/proof['run']/'validation_checkpoints'/f"update_{proof['selected_update']:06d}.pt"
        assert digest(path)==proof['checkpoint_sha256']
    score_hashes={}
    for run in r['quality_review_runs']:
        path=RUN/'completed_analysis'/f"{run['seed']}-{run['arm']}-out-of-fold.csv"
        frame=pd.read_csv(path)
        assert frame.sample_id.is_unique and len(frame)==2053
        assert int(frame.error.sum())==run['errors']
        for name,statistics in run['models'].items():
            assert np.isclose(roc_auc_score(frame.error,frame[name]),statistics['auc'])
            for f in (.1,.2,.3):
                assert review_counts(frame,frame[name],f)[0]==statistics['review'][str(f)]
        assert sum(v['n'] for v in run['quality_bins'])==2053
        assert sum(v['fn']+v['fp'] for v in run['quality_bins'])==run['errors']
        score_hashes[path.relative_to(ROOT).as_posix()]=digest(path)
    paths=[*OUT.glob('*.md'),OUT.parent/'README.md',OUT.parent/'README.en.md',
           ROOT/'README.md',ROOT/'README.en.md',*ROOT.glob('docs/README*.md'),
           *ROOT.glob('docs/DIRECTORY*.md'),*ROOT.glob('docs/progress/README*.md'),
           *ROOT.glob('docs/operations/next-steps*.md'),*ROOT.glob('docs/operations/training-readiness*.md'),
           *ROOT.glob('docs/operations/training-runbook*.md'),RUN/'README.md',RUN/'README.en.md']
    links=[]
    for path in paths:
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8-sig')):
            if target.startswith(('http:','https:','mailto:','#')):continue
            target=target.split('#')[0]
            assert (path.parent/target).resolve().exists(),(path,target)
            links.append(dict(file=path.relative_to(ROOT).as_posix(),target=target))
    documents=[]
    render=json.loads((OUT/'render-verification.json').read_text())
    # Both translations must retain the same numbers and table cell values.
    zh=(OUT/'completed-results.zh.md').read_text(encoding='utf-8')
    en=(OUT/'completed-results.en.md').read_text(encoding='utf-8')
    def numeric_tables(text):
        return [[re.findall(r'[+−-]?\d+(?:[,.]\d+)*%?',cell) for cell in line.split('|')[2:-1]]
                for line in text.splitlines() if line.startswith('|') and re.match(r'\s*[+−-]?\d',line.split('|')[2])]
    assert numeric_tables(zh)==numeric_tables(en)
    assert '26.0' in zh and '26.0' in en and '-2.0' in zh and '-2.0' in en
    for lang in ('zh','en'):
        markdown=(OUT/f'completed-results.{lang}.md').read_text(encoding='utf-8')
        assert len(re.findall(r'!\[',markdown))==10
        pdf=pymupdf.open(OUT/f'completed-results.{lang}.pdf')
        assert len(pdf)==render[lang]['pages'] and not render[lang]['layout_warnings']
        assert all(page.get_text().strip() for page in pdf)
        assert all('\ufffd' not in page.get_text() for page in pdf)
        for i in range(len(pdf)):assert (BUILD/f'page-{lang}-{i+1:02d}.png').exists()
        files=[OUT/f'completed-results.{lang}.{suffix}' for suffix in ('md','tex','pdf')]+[OUT/f'{name}.{lang}.png' for name in FIGURES]
        assert all(path.exists() and path.stat().st_size>1000 for path in files)
        documents.append(dict(language=lang,pages=len(pdf),figures=10,sha256={p.name:digest(p) for p in files}))
    result=dict(passed=True,verified_at=dt.datetime.now(dt.timezone.utc).isoformat(),completed_training_runs=18,
        analyzed_selected_models=18,verified_input_hashes=len(r['input_sha256']),verified_checkpoint_hashes=18,
        verified_out_of_fold_scores=score_hashes,checked_local_links=len(links),bilingual_numeric_tables_match=True,
        documents=documents,protected_test_inference=False,analysis_sha256=digest(OUT/'completed-analysis.json'))
    write_json(OUT/'delivery-verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('documents','verified_out_of_fold_scores')}))


if __name__=='__main__':main()
