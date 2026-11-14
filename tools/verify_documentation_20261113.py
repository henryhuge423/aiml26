"""Check reorganized documentation, report builds, and authoritative overview values."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/progress/2026-11-13'
BUILD = ROOT / '.build/documentation-verification-20261113'
REPORTS = [('', 'progress-overview', ROOT/'.build/progress-overview-20261113'),
           ('data-review', 'data-and-degradation-review', ROOT/'.build/progress-20261113'),
           ('training-results', 'training-results', ROOT/'.build/progress-20261113-training'),
           ('experimental-analysis', 'experimental-analysis', ROOT/'.build/progress-20261113-analysis')]
ARMS = ('original_clean', 'expanded_clean', 'expanded_degradation30', 'expanded_degradation50')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    documents = list((ROOT/'docs').rglob('*.md')) + list(ROOT.glob('*.md'))
    documents += list((ROOT/'notes').glob('*.md'))
    documents += list((ROOT/'data').glob('README*.md')) + list((ROOT/'models').glob('README*.md'))
    documents += list((ROOT/'data/prepared').glob('*20261113/README*.md'))
    documents += list((ROOT/'runs/expanded_20261113').glob('README*.md'))
    missing, links = [], 0
    for path in documents:
        body = path.read_text(encoding='utf-8-sig')
        for raw in re.findall(r'\]\(([^\n)]+)\)', body):
            if raw.startswith(('http:', 'https:', 'mailto:', '#')):
                continue
            target = unquote(raw.strip('<>').split('#', 1)[0])
            if not target:
                continue
            if not (path.parent/target).exists():
                missing.append({'document': path.relative_to(ROOT).as_posix(), 'target': target})
            links += 1
    # Older reports retain links to local references; identify every unresolved
    # link and fail rather than silently omitting historical material.
    if missing:
        (BUILD/'missing-links.json').write_text(json.dumps(missing, ensure_ascii=False, indent=2), encoding='utf-8')
        raise ValueError(f'{len(missing)} missing local links; see {BUILD / "missing-links.json"}')

    release = read(ROOT/'data/prepared/expanded_20261113/cohort_verification.json')
    assert release['passed'] and release['unique_paths_file_hash_and_full_RGB_verified'] == 24491
    assert release['counts'] == {'boundary': 2053, 'external_acrima': 705, 'external_hygd': 737,
                                 'test': 3470, 'train': 15480, 'val': 2046}
    comparison = read(ROOT/'runs/expanded_20261113/budget_end_comparison.json')
    sources = {'AIROGS': 'airogs-light-v2', 'ODIR': 'DATASET_101/ODIR-5K', 'LAG': 'lag_part1', 'KEH': 'keh'}
    values_checked = 0
    for lang in ('zh', 'en'):
        overview = (OUT/f'progress-overview.{lang}.md').read_text(encoding='utf-8')
        for label, source in sources.items():
            line = next(line for line in overview.splitlines() if line.startswith(f'| {label}'))
            observed = [cell.strip() for cell in line.split('|') if cell.strip()][-4:]
            expected = [f"{comparison['arms'][arm]['sets']['boundary/original']['per_source'][source]['auc']:.4f}" for arm in ARMS]
            assert observed == expected, (lang, label, observed, expected)
            values_checked += len(expected)
    for arm in ARMS:
        summary = read(ROOT/f'runs/expanded_20261113/{arm}/run_summary.json')
        assert summary['budget_completed'] and summary['optimizer_updates'] == 1200

    report_records = []
    for folder, stem, logs in REPORTS:
        directory = OUT/folder
        for lang in ('zh', 'en'):
            base = f'{stem}.{lang}'
            artifacts = [directory/f'{base}.{ext}' for ext in ('md','tex','pdf')]
            assert all(path.stat().st_size > 100 for path in artifacts), base
            extracted = BUILD/f'{base}.txt'
            subprocess.run(['pdftotext','-layout',str(directory/f'{base}.pdf'),str(extracted)], check=True)
            text = extracted.read_text(encoding='utf-8')
            assert '\ufffd' not in text and '{{' not in text, base
            pages = text.split('\f')
            assert all(page.strip() for page in pages[:-1]), f'blank page: {base}'
            log_files = list(logs.glob(f'{base}*1.log'))
            assert log_files, ('missing build log', base)
            for log in log_files:
                body = log.read_text(encoding='utf-8', errors='replace')
                assert not any(p in body for p in ('Overfull','Missing character','Undefined control sequence','! LaTeX Error')), (base, log)
            report_records.append({'report': (directory/f'{base}.pdf').relative_to(ROOT).as_posix(),
                                   'pages': len(pages)-1, 'sha256': digest(directory/f'{base}.pdf')})
    migrated = (ROOT/'docs/progress/2026-11-12')
    historical_pdf_hashes = {}
    for name in ('training-progress.zh.pdf','training-progress.en.pdf'):
        old = f'docs/progress/training-2026-11-12/{name}'
        original = subprocess.run(['git','show',f'HEAD:{old}'],cwd=ROOT,capture_output=True)
        # A committed rename makes HEAD use the new location on later runs.
        if original.returncode:
            original = subprocess.run(['git','show',f'HEAD:docs/progress/2026-11-12/{name}'],cwd=ROOT,capture_output=True,check=True)
        assert hashlib.sha256(original.stdout).hexdigest() == digest(migrated/name), name
        historical_pdf_hashes[name] = digest(migrated/name)
    figures = list(OUT.rglob('*.png'))
    from PIL import Image
    for path in figures:
        with Image.open(path) as im:
            im.verify()
    visual_records = {}
    for folder, stem, _ in REPORTS:
        name = ('overview-visual-review.json' if not folder else
                'visual_review.json' if folder == 'data-review' else 'visual-review.json')
        record_path = OUT/folder/name
        visual = read(record_path)
        serialized = json.dumps(visual)
        for lang in ('zh', 'en'):
            pdf = OUT/folder/f'{stem}.{lang}.pdf'
            assert digest(pdf) in serialized, ('visual review is stale', pdf, record_path)
        visual_records[folder or 'overview'] = record_path.relative_to(ROOT).as_posix()
    tests = read(OUT/'test-verification.json')
    assert tests['exit_code'] == 0 and tests['passed'] == 82 and tests['subtests_passed'] == 3
    record = {'passed': True, 'documents_checked': len(documents), 'local_links_checked': links,
              'overview_metric_values_checked': values_checked, 'reports': report_records,
              'figures_decoded': len(figures), 'historical_pdfs_unchanged': historical_pdf_hashes,
              'visual_review_records': visual_records, 'tests': tests,
              'training_updates_each': 1200, 'cohort_images': 24491,
              'artifact_sha256': {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(OUT.rglob('*'))
                  if p.is_file() and p.suffix in ('.md','.pdf','.tex','.png','.csv')},
              'source': 'tools/verify_documentation_20261113.py'}
    (OUT/'documentation-verification.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in record.items() if k!='artifact_sha256'}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
