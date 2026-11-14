"""Validate editable bilingual review sources against the frozen data audit.

Markdown is the report source of truth. Run build_pdfs.py after editing; this
check keeps the dated source catalog and counts aligned with the audit without
overwriting reviewed prose with earlier draft wording.
"""
from pathlib import Path
import json
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]

def main():
    audit = ROOT / 'data/audit/data_expansion_20261113'
    pool = json.loads((ROOT / 'data/prepared/expansion_20261113/release.json').read_text(encoding='utf-8'))
    summary = json.loads((audit / 'local_summary.json').read_text(encoding='utf-8'))
    sources = json.loads((audit / 'sources.json').read_text(encoding='utf-8'))['sources']
    assert len(sources) == 45
    assert summary['new_images'] == 13176 and summary['candidate_pairs'] == 315074
    assert pool['images'] == 9070 and pool['positive_images'] == 3446
    assert pool['auxiliary_suspect_images'] == 204
    for lang in ('zh', 'en'):
        path = HERE / f'data-and-degradation-review.{lang}.md'
        body = path.read_text(encoding='utf-8')
        assert '{{' not in body
        for number in ('13,176', '9,070', '3,446', '315,074', '146', '204'):
            assert number in body, (lang, number)
        catalog = re.search(r'\| (?:来源|Source) \| (?:本轮决定及依据|Decision and evidence) \|\n.*?(?=\n\n)', body, re.S)[0]
        assert len(re.findall(r'^\| \[', catalog, re.M)) == 45
        for source in sources:
            assert source['url'] in catalog, (lang, source['url'])
        for name, counts in pool['sources'].items():
            assert f"{counts['images']:,}" in body
            assert f"{counts['positives']:,}" in body
    print(json.dumps({'passed': True, 'languages': ['zh', 'en'], 'catalog_entries': 45,
                      'admitted_images': 9070, 'positives': 3446, 'auxiliary_suspects': 204}, indent=2))

if __name__ == '__main__': main()
