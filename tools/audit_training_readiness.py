"""Read-only source audit; writes evidence under data/audit/2026-11-09.

No automatic deletions, clinical grading, or training splits are performed.
"""
from __future__ import annotations
import collections
import concurrent.futures
import csv
import hashlib
import json
import platform
import sys
import time
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageFile
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = DATA / 'audit' / '2026-11-09'
EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp', '.gif'}
ImageFile.LOAD_TRUNCATED_IMAGES = False

def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def table(name, rows, fields=None):
    rows = list(rows)
    with (OUT / name).open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def source(path):
    p = path.relative_to(DATA).parts
    if p[0] == 'external':
        return p[1], '/'.join(p[2:-1])
    if p[0] == 'review':
        return 'quarantine', p[1]
    if p[1] == 'frontofeyedx':
        return 'FrontofEyeDx', path.parent.name
    if 'Fundoscopy' in p:
        i = p.index('Fundoscopy')
        return '/'.join(p[i+1:-2]), path.parent.name
    return 'EyeWorld-anterior-web', path.parent.name

def inspect(path):
    src, label = source(path)
    row = dict(path=path.relative_to(DATA).as_posix(), source=src, folder_label=label,
               bytes=path.stat().st_size, sha256=digest(path), width=0, height=0,
               status='full_decode_ok', error='', dhash='', gray_mean='', gray_std='',
               review_flags='')
    try:
        with Image.open(path) as im:
            im.load()  # Full pixel decode, not just Image.verify/header inspection.
            row['width'], row['height'] = im.size
            g = im.convert('L').resize((128, 128))
            a = np.asarray(g, dtype=np.float32)
            b = np.asarray(g.resize((9, 8)))
            bits = b[:, 1:] > b[:, :-1]
            row['dhash'] = f'{int.from_bytes(np.packbits(bits).tobytes(), "big"):016x}'
            row['gray_mean'], row['gray_std'] = round(float(a.mean()), 3), round(float(a.std()), 3)
            flags = []
            if min(im.size) < 224: flags.append('short_side_lt224')
            if max(im.size) / min(im.size) > 2: flags.append('extreme_aspect_ratio')
            if a.std() < 5: flags.append('near_uniform_thumbnail')
            row['review_flags'] = '|'.join(flags)
    except Exception as e:
        row['status'], row['error'] = 'decode_failed', repr(e)
    return row

def metadata_checks(rows):
    by_path = {r['path']: r for r in rows}
    result = {}
    base = DATA / 'external/hygd-1.1.0'
    labels = list(csv.DictReader((base / 'Labels.csv').open(encoding='utf-8-sig')))
    failures = []
    sums = (base / 'SHA256SUMS.txt').read_text().splitlines()
    for line in sums:
        sha, rel = line.split(maxsplit=1)
        p = base / rel.strip().lstrip('*')
        if not p.is_file() or digest(p) != sha:
            failures.append(rel)
    result['hygd'] = dict(label_rows=len(labels), patients=len({r['Patient'] for r in labels}),
        label_counts=dict(collections.Counter(r['Label'] for r in labels)),
        missing_images=[r['Image Name'] for r in labels if not (base/'Images'/r['Image Name']).exists()],
        official_checksum_entries=len(sums), official_checksum_failures=failures)
    base = DATA / 'external/airogs-light-v2/eyepac-light-v2-512-jpg'
    labels = list(csv.DictReader((base / 'metadata.csv').open(encoding='utf-8-sig')))
    expected = [f"external/airogs-light-v2/eyepac-light-v2-512-jpg/{r['folder']}/{r['label']}/{r['file_name']}" for r in labels]
    result['airogs'] = dict(label_rows=len(labels), duplicate_filenames=len(labels)-len({r['file_name'] for r in labels}),
        missing_images=[p for p in expected if p not in by_path],
        split_label_counts=dict(collections.Counter(r['folder']+'/'+r['label'] for r in labels)),
        patient_id_available=False, note='id is an image ID, not verified patient ID')
    for split in ('train', 'test'):
        p = DATA / f'external/eyeq/data/Label_EyeQ_{split}.csv'
        labels = list(csv.DictReader(p.open(encoding='utf-8-sig')))
        result['eyeq_'+split] = dict(rows=len(labels), columns=list(labels[0]), complete_images_available=False)
    p = DATA / 'external/fiqs/meta_data.xlsx'
    wb = load_workbook(p, read_only=True, data_only=True)
    ws = wb.active
    values = list(ws.values)
    result['fiqs'] = dict(rows=len(values)-1, columns=list(values[0]), sample_rows=values[1:4],
        resized_count=len(list((p.parent/'images').glob('*.PNG'))),
        original_count=len(list((p.parent/'full_size_images').glob('*.PNG'))),
        missing_original_names=sorted({x.name for x in (p.parent/'images').glob('*.PNG')} - {x.name for x in (p.parent/'full_size_images').glob('*.PNG')}))
    wb.close()
    dump('metadata_checks.json', result)

def archives():
    report = []
    for p in sorted((DATA / 'external').glob('*/source.zip')):
        print('archive check', p.parent.name, flush=True)
        with zipfile.ZipFile(p) as z:
            bad = z.testzip()
            entry_count = len(z.infolist())
        item = dict(path=p.relative_to(DATA).as_posix(), bytes=p.stat().st_size,
                    entries=entry_count, crc_bad_entry=bad)
        if p.parent.name == 'fiqs':
            item['md5'] = digest(p, 'md5')
            item['official_md5_match'] = item['md5'] == '1ea6772f3163e0e1f7df8fabd76ac659'
        report.append(item)
        dump('archive_checks.json', report)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if '--archives' in sys.argv:
        archives()
        return
    files = sorted(p for base in ('sources', 'review', 'external') for p in (DATA/base).rglob('*')
                   if p.is_file() and p.suffix.lower() in EXT and not p.name.startswith('._'))
    rows = []
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for i, row in enumerate(pool.map(inspect, files), 1):
            rows.append(row)
            if i % 2500 == 0: print(f'{i}/{len(files)} decoded in {time.time()-start:.1f}s', flush=True)
    table('image_manifest.csv', rows)
    hashes = collections.defaultdict(list)
    perceptual = collections.defaultdict(list)
    for r in rows:
        hashes[r['sha256']].append(r)
        if r['dhash']: perceptual[r['dhash']].append(r)
    dup = [dict(sha256=h, files=len(v), paths='|'.join(r['path'] for r in v),
                sources='|'.join(sorted({r['source'] for r in v}))) for h,v in hashes.items() if len(v)>1]
    table('exact_duplicates.csv', dup, ['sha256','files','paths','sources'])
    # These are candidates only: equal dHash is neither proof of identity nor exhaustive near-duplicate detection.
    near = [dict(dhash=h, files=len(v), sources='|'.join(sorted({r['source'] for r in v})),
                 paths='|'.join(r['path'] for r in v)) for h,v in perceptual.items()
            if len(v)>1 and len({r['sha256'] for r in v})>1]
    table('perceptual_duplicate_candidates.csv', near, ['dhash','files','sources','paths'])
    table('review_candidates.csv', [r for r in rows if r['status'] != 'full_decode_ok' or r['review_flags']], list(rows[0]))
    counts = collections.Counter(r['source'] for r in rows)
    summary = dict(date='2026-11-09', physical_image_files=len(rows), unique_sha256=len(hashes),
        full_decode_failures=sum(r['status']!='full_decode_ok' for r in rows),
        source_counts=dict(counts), exact_duplicate_groups=len(dup),
        duplicate_excess_files=sum(r['files']-1 for r in dup),
        equal_dhash_candidate_groups=len(near),
        review_flag_counts=dict(collections.Counter(flag for r in rows for flag in r['review_flags'].split('|') if flag)),
        note='Review flags and dHash collisions require manual adjudication. No source files changed.',
        elapsed_seconds=round(time.time()-start,1), python=sys.executable, python_version=platform.python_version())
    dump('inventory_summary.json', summary)
    metadata_checks(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

if __name__ == '__main__':
    main()
