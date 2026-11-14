"""Check delivered data provenance, identity-only development repair, and reports.

Image decoding is performed by tools/finalize_expansion_20261113.py in the
audited project environment. This check does not load any test predictions.
"""
from pathlib import Path
import collections
import csv
import hashlib
import json
import re
from urllib.parse import unquote

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
AUDIT=ROOT/'data/audit/data_expansion_20261113'
POOL=ROOT/'data/prepared/expansion_20261113'
REPAIR=ROOT/'data/prepared/formal_v3_boundary_repaired_20261113'
BUILD=ROOT/'.build/progress-20261113'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def js(p):return json.loads(p.read_text(encoding='utf-8'))

def verify_input(name,expected,documentation_updates):
    path=ROOT/name
    current=sha(path)
    if current==expected:return
    # This script changed only report links during documentation migration.
    # Reversing that exact substitution and Windows newline conversion must
    # reproduce the originally recorded source digest, or verification fails.
    assert path.resolve()==(ROOT/'tools/finalize_expansion_20261113.py').resolve(),name
    restored=path.read_bytes().replace(b'docs/progress/2026-11-13/data-review/',
                                     b'docs/progress/2026-11-13/')
    restored=restored.replace(b'\r\n',b'\n')
    assert hashlib.sha256(restored).hexdigest()==expected,name
    documentation_updates[name]=dict(recorded_sha256=expected,current_sha256=current,
        restored_original_sha256=hashlib.sha256(restored).hexdigest(),
        verified_changes=['report paths moved to the data-review subdirectory',
                          'LF line endings converted to CRLF on Windows'])

def main():
    sealed=js(AUDIT/'final_verification.json')
    assert sealed['passed']
    for name,digest in sealed['outputs_sha256'].items():assert sha(AUDIT/name)==digest,name
    summary=js(AUDIT/'local_summary.json')
    for name,digest in summary['inputs'].items():assert sha(ROOT/name)==digest,name
    release=js(POOL/'release.json');repair=js(REPAIR/'release.json')
    documentation_updates={}
    for item in [release,repair]:
        for name,digest in item['inputs'].items():verify_input(name,digest,documentation_updates)
    pool=read(POOL/'pool_manifest.csv');suspect=read(POOL/'suspect_manifest.csv')
    assert sha(POOL/'pool_manifest.csv')==release['manifest_sha256']
    assert sha(POOL/'suspect_manifest.csv')==release['auxiliary_manifest_sha256']
    assert len(pool)==9070 and sum(r['label']=='1' for r in pool)==3446
    assert len(suspect)==204 and all(r['label']=='suspect' for r in suspect)
    assert release['source_defined_glaucoma_positive_images']==3443
    assert release['airogs_RG_referral_positive_images']==3
    assert release['officially_poor_quality_positive_images']=={
        'FIVES_at_least_one_poor_dimension':99,'MSHF_overall_poor':47}
    assert len({r['sample_id'] for r in pool})==len(pool)
    assert len({r['photo_group'] for r in pool})==len(pool)
    assert all((ROOT/'data'/r['path']).is_file() for r in pool+suspect)
    for source,counts in release['sources'].items():
        rr=[r for r in pool if r['source']==source]
        assert len(rr)==counts['images']
        assert sum(r['label']=='1' for r in rr)==counts['positives']
    assert not release['research_training_released']
    assert summary['candidate_pairs']==sum(summary['candidate_decisions'].values())==315074
    old=read(ROOT/'data/prepared/formal_v3_boundary_repaired_20261110/manifest.csv')
    kept=read(REPAIR/'manifest.csv');excluded=read(REPAIR/'exclusions.csv')
    assert sha(REPAIR/'manifest.csv')==repair['manifest_sha256']
    assert len(old)==1230 and len(kept)==1220 and len(excluded)==10
    assert sum(r['label']=='1' for r in kept)==503
    assert {r['sample_id'] for r in old}=={r['sample_id'] for r in kept+excluded}
    conflicts=read(AUDIT/'local_existing_conflicts.csv')
    conflict_ids={r['sample_id'] for r in conflicts if r['split']=='boundary'}
    assert not ({r['sample_id'] for r in kept}&conflict_ids)
    previous={r['sample_id']:r for r in read(ROOT/'runs/formal_v3/boundary_development_repaired_20261110/boundary_pq.csv')}
    predictions=read(AUDIT/'repaired_boundary_pq.csv')
    assert sha(AUDIT/'repaired_boundary_pq.csv')==repair['predictions_sha256']
    assert {r['sample_id'] for r in predictions}=={r['sample_id'] for r in kept}
    assert all(r==previous[r['sample_id']] for r in predictions)
    analysis=js(HERE/'development_analysis.json')
    for name,digest in analysis['inputs'].items():assert sha(ROOT/name)==digest,name
    assert analysis['splits']['boundary']['positive_groups']==494
    chart=js(HERE/'admission-chart-data.json')
    for name,digest in chart['inputs_sha256'].items():assert sha(ROOT/name)==digest,name
    assert chart['candidate_images']==13176
    assert chart['disease_pool_images']+chart['auxiliary_suspects']+chart['excluded_images']==13176
    assert chart['positive_images']==3446
    documents=[HERE/f'data-and-degradation-review.{lang}.md' for lang in ['zh','en']]
    documents += [ROOT/'README.md',ROOT/'README.en.md',ROOT/'data/README.md',
        ROOT/'docs/progress/README.md',ROOT/'docs/progress/README.en.md',POOL/'README.md',POOL/'README.en.md']
    links=0
    for p in documents:
        body=p.read_text(encoding='utf-8');assert '{{' not in body,p
        for target in re.findall(r'\]\(([^)]+)\)',body):
            if '://' in target or target.startswith('#'):continue
            path=unquote(target.split('#')[0].strip('<>'))
            assert (p.parent/path).exists(),(p,path)
            links+=1
    pages=js(HERE/'pdf_build.json')
    visual=js(HERE/'visual_review.json')
    for lang in ['zh','en']:
        stem=f'data-and-degradation-review.{lang}'
        text=(BUILD/f'{stem}.txt').read_text(encoding='utf-8')
        assert '{{' not in text and '\ufffd' not in text
        assert '9,070' in text and '3,446' in text and '146' in text
        log=(BUILD/f'{stem}.xelatex-1.log').read_text(encoding='utf-8')
        assert not any(s in log for s in ['Missing character','Overfull','Undefined control sequence','! LaTeX Error'])
        assert (HERE/f'{stem}.tex').is_file() and (HERE/f'{stem}.pdf').is_file()
        count=text.count('\f');assert count==pages[lang]['pages']
        assert len(list((BUILD/'final').glob(f'page-{lang}-*.png')))==count
        assert visual['pdf_pages_reviewed'][lang]==list(range(1,count+1))
        assert visual['reviewed_sha256'][f'{stem}.pdf']==sha(HERE/f'{stem}.pdf')
    for name in ['positive_counts_and_precision.png','admission-and-sources.png']:
        assert visual['reviewed_sha256'][name]==sha(HERE/name)
    output=dict(passed=True,sealed_input_hashes=len(summary['inputs']),admitted=9070,
        positive_labels=3446,auxiliary_suspects=204,poor_quality_positives=146,
        development_images=1220,development_positives=503,identity_only_prediction_filter_verified=True,
        protected_original_manifest_hash_verified=True,final_test_predictions_loaded=False,
        checked_local_links=links,pdf_pages={lang:v['pages'] for lang,v in pages.items()},
        reviewed_figures=2,visual_review_record='visual_review.json',
        historical_input_documentation_updates_verified=documentation_updates,
        artifact_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in HERE.iterdir()
            if p.suffix in {'.md','.tex','.pdf','.png','.py','.json'} and p.name!='delivery_verification.json'})
    (HERE/'delivery_verification.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in output.items() if k!='artifact_sha256'},indent=2))

if __name__=='__main__':main()
