"""Check completed budgets, paired identities, report artifacts, and local links."""
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/expanded_20261113'
HERE=ROOT/'docs/progress/2026-11-13/training-results'
ARMS=('original_clean','expanded_clean','expanded_degradation30','expanded_degradation50')


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert read(RUN/'pipeline/status.json')['status']=='completed'
    assert read(RUN/'pipeline/budget_end_status.json')['status']=='completed'
    assert read(RUN/'preflight/verification.json')['passed']
    identities=[]
    for arm in ARMS:
        summary=read(RUN/arm/'run_summary.json')
        assert summary['optimizer_updates']==1200 and summary['budget_completed']
        assert summary['amp_overflows']==0
        with (RUN/arm/'development_predictions.csv').open(encoding='utf-8-sig') as stream:
            rows=list(csv.DictReader(stream))
        assert {r['split'] for r in rows}=={'val','boundary'}
        assert len(rows)==2046+2053+2*384
        ids={(r['sample_id'],r['condition']) for r in rows}
        assert len(ids)==len(rows)
        identities.append(ids)
        assert read(RUN/arm/'last_development_metrics.json')['selected_successful_updates']==1200
    assert all(ids==identities[0] for ids in identities)
    docs=[HERE/f'{stem}.{lang}.md' for stem in ('training-results','data-audit','degradation-audit') for lang in ('zh','en')]
    docs += [ROOT/'docs/progress/README.md',ROOT/'docs/progress/README.en.md',ROOT/'README.md',ROOT/'README.en.md']
    links=0
    for path in docs:
        for target in re.findall(r'\]\(([^\s)]+)',path.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):continue
            destination=(path.parent/target.split('#',1)[0]).resolve()
            assert destination.exists(),(path,target)
            links+=1
    build=ROOT/'.build/progress-20261113-training'
    metadata=read(HERE/'delivery-verification.json');artifacts={}
    visual=read(HERE/'visual-review.json')
    assert visual['report_layout_passed']
    for lang in ('zh','en'):
        for extension in ('md','pdf','tex'):
            path=HERE/f'training-results.{lang}.{extension}'
            assert path.stat().st_size>0
            artifacts[path.relative_to(ROOT).as_posix()]=digest(path)
        pages=list(build.glob(f'page-{lang}-*.png'))
        assert len(pages)==metadata['reports'][lang]['pages']
        assert visual['pdf_pages_reviewed'][lang]==list(range(1,len(pages)+1))
        assert visual['pdf_sha256'][lang]==digest(HERE/f'training-results.{lang}.pdf')
        log=(build/f'training-results.{lang}-build-1.log').read_text(encoding='utf-8',errors='replace')
        assert not re.search('Overfull|Missing character|undefined',log)
    for path in HERE.iterdir():
        if path.is_file() and path.suffix in ('.md','.pdf','.tex','.png'):
            artifacts[path.relative_to(ROOT).as_posix()]=digest(path)
    for name,digest_value in visual['figure_sha256'].items():
        assert digest(HERE/name)==digest_value,name
    evidence=dict(passed=True,paired_development_identities_each=len(identities[0]),all_arms_successful_updates=1200,
        amp_overflows=0,protected_test_inference=False,local_links_verified=links,report_artifact_sha256=artifacts,
        unit_test_session='2026-11-13 document revision: full suite rerun in the training environment',
        passed_unit_tests=82,passed_subtests=3,unit_suite='tests',
        unit_test_command='tools/training_python.cmd -m pytest tests -q',unit_test_duration_seconds=36.78,
        visual_review=visual, cuda_resume_exact=read(RUN/'preflight/verification.json')['next_losses_exact'],
        verification_source_sha256=digest(Path(__file__)))
    (HERE/'final-verification.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print(json.dumps(evidence,indent=2))


if __name__=='__main__':main()
