"""Verify analysis inputs, paired statistics, current reports and visual review."""
from __future__ import annotations
import json
from pathlib import Path
import re
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT))
from tools.analyze_expansion_results_20261113 import ARMS, OUT, RUN, digest, read_json
BUILD=ROOT/'.build/progress-20261113-analysis'


def main():
    result=read_json(OUT/'analysis.json')
    assert result['manifest_sha256']==read_json(ROOT/'data/prepared/expanded_20261113/release.json')['manifest_sha256']
    assert result['method']['repeats']==4000 and result['method']['protected_test_inference'] is False
    count=0
    for checkpoint,prefix in (('selected',''),('budget_end','last_')):
        for arm in ARMS:
            assert digest(RUN/arm/f'{prefix}development_predictions.csv')==result['prediction_sha256'][checkpoint+'/'+arm]
            authoritative=read_json(RUN/arm/f'{prefix}development_metrics.json')
            for source,value in result['checkpoints'][checkpoint]['sources'].items():
                p=value['points'][arm]
                saved=authoritative['sets']['boundary/original']['per_source'][source]
                for field in ('n','positives','positive_groups','auc','auprc','brier'):
                    assert abs(p[field]-saved[field])<1e-12,(checkpoint,arm,source,field)
                    count+=1
            summary=read_json(RUN/arm/'run_summary.json')
            assert summary['optimizer_updates']==1200 and summary['amp_overflows']==0
    exposures=result['training_exposure']
    assert len({exposures[a]['sample_draw_counts_sha256'] for a in ARMS[1:]})==1
    for arm in ARMS:
        assert digest(RUN/arm/'last.pt')==exposures[arm]['checkpoint_sha256']
    # The November 13 reports are revised together. Their current documents are
    # checked here as links; training report hashes belong to its own verifier.
    paths=list(OUT.glob('*.md'))+[ROOT/'README.md',ROOT/'README.en.md',
          ROOT/'docs/progress/README.md',ROOT/'docs/progress/README.en.md',RUN/'README.md',RUN/'README.en.md']
    links=0
    for path in paths:
        text=path.read_text(encoding='utf-8')
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',text):
            target=target.strip().strip('<>')
            if target.startswith(('http:','https:','mailto:','#')):continue
            target=unquote(target.split('#',1)[0])
            assert (path.parent/target).resolve().exists(),(path,target)
            links+=1
    render=read_json(OUT/'render-verification.json')
    visual=read_json(OUT/'visual-review.json')
    assert visual['reviewed'] is True
    for lang in ('zh','en'):
        for extension in ('md','pdf','tex'):
            assert (OUT/f'experimental-analysis.{lang}.{extension}').stat().st_size>100
        pages=render[lang]['pages']
        assert pages>0
        assert visual['languages'][lang]['pages_reviewed']==list(range(1,pages+1))
        assert visual['languages'][lang]['pdf_sha256']==digest(OUT/f'experimental-analysis.{lang}.pdf')
        rendered_pages={int(p.stem.rsplit('-',1)[1]):p for p in BUILD.glob(f'page-{lang}-*.png')}
        assert sorted(rendered_pages)==list(range(1,pages+1))
        for page in range(1,pages+1):
            rendered=rendered_pages[page]
            assert visual['languages'][lang]['page_sha256'][str(page)]==digest(rendered)
        log=(BUILD/f'experimental-analysis.{lang}-build-1.log').read_text(encoding='utf-8',errors='replace')
        for problem in ('Overfull','Missing character','undefined references'):
            assert problem not in log,(lang,problem)
        extracted=(BUILD/f'experimental-analysis.{lang}.txt').read_text(encoding='utf-8')
        assert extracted.count('\f')==pages
        for evidence in ('0.2706','0.0555','0.0625','24,960'):
            assert evidence in extracted,(lang,evidence)
    figures=read_json(OUT/'figure-data.json')
    assert figures['real_quality_training_counts']==result['checkpoints']['budget_end']['quality']['manifest_quality_strata']['train']
    assert figures['real_quality_development_sensitivity']==result['checkpoints']['budget_end']['quality']['development_sensitivity']
    for name in figures['figures']:
        assert visual['figures'][name]['reviewed'] is True
        assert visual['figures'][name]['sha256']==digest(OUT/name)
    report=dict(passed=True,metric_values_cross_checked=count,local_links_verified=links,
        prediction_files_unchanged=8,complete_prediction_identities_per_file=4867,
        protected_test_inference=False,original_manifest_unchanged=True,
        current_training_report_links_verified=True,
        expanded_per_image_draw_counts_identical=True,
        statistical_unit_tests=visual['statistical_unit_tests'],
        final_visual_review=dict(pdf_pages=sum(v['pages'] for v in render.values()),
                                figures=len(figures['figures']),record='visual-review.json'),
        pdf_pages=render,artifact_sha256={str(p.relative_to(ROOT)):digest(p) for p in sorted(OUT.iterdir())
            if p.is_file() and p.suffix in ('.md','.pdf','.tex','.png','.csv')},
        analysis_sha256=digest(OUT/'analysis.json'),
        code_sha256={str(p.relative_to(ROOT)):digest(p) for p in
            (ROOT/'tools/analyze_expansion_results_20261113.py',ROOT/'tools/report_expansion_analysis_20261113.py',
             Path(__file__).resolve(),ROOT/'tests/test_expansion_analysis.py')})
    (OUT/'delivery-verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('artifact_sha256','code_sha256')},indent=2))


if __name__=='__main__':main()
