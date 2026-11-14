"""Complete image/label/association audit for the official LAG public part 1.

Every radius-six candidate is adjudicated using retinal-interior evidence.
The output is an input to the combined cohort graph, never a new train split.
"""
from __future__ import annotations
import collections
import csv
import hashlib
import importlib.util
import json
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
import sys
import threading
import time
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/audit/data_expansion_20261113'
LAG = ROOT / 'data/external_glaucoma/lag'
sys.path.insert(0, str(ROOT / 'tools'))
import audit_external_overlap_20261113 as A

START = time.time()
DESC_DIR = OUT / 'lag_descriptor_cache'
MATCHER = None
DESCRIPTOR_LOCKS=collections.defaultdict(threading.Lock)

def status(stage, **fields):
    record = dict(stage=stage, elapsed_seconds=round(time.time()-START, 1), **fields)
    (OUT/'lag_status.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record), flush=True)

def write(name, rows, fields=None):
    A.A.save_csv(OUT/name, rows, fields or list(rows[0]))

def load_features(path):
    with path.open(encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    for r in rows:
        r['path'] = A.norm(r['path'])
    return rows

def collect():
    rows = []
    counts = {}
    for folder, binary_label in [('non_glaucoma', '0'), ('suspicious_glaucoma', '1')]:
        images = sorted((LAG/'extracted'/folder/'image').glob('*.jpg'))
        maps = sorted((LAG/'extracted'/folder/'attention_map').glob('*.jpg'))
        assert {x.name for x in images} == {x.name for x in maps}, folder
        counts[folder] = len(images)
        for p in images:
            label_path=LAG/'extracted'/folder/'label'/(p.stem+'.txt')
            assert label_path.read_text(encoding='utf-8-sig').strip()==binary_label,label_path
            rows.append(dict(path=A.norm(p), source='lag_part1',
                label=binary_label, source_binary_label=binary_label,
                original_label=folder, target_definition='source_clinical_glaucoma_diagnosis_cvpr2019_section3.1',
                patient_id='', eye='', official_split='', sha256=A.sha(p)))
    assert counts == {'non_glaucoma': 3143, 'suspicious_glaucoma': 1711}, counts
    assert len({r['path'] for r in rows}) == len(rows) == 4854
    acquisition = json.loads((LAG/'acquisition_20261113.json').read_text(encoding='utf-8'))
    assert A.sha(LAG/'LAG_database_part_1.rar') == acquisition['sha256']
    write('lag_input_images.csv', rows)
    status('collected', images=len(rows), source_binary_labels=dict(collections.Counter(r['source_binary_label'] for r in rows)))
    return rows

def feature(r):
    p = Path(r['path']); stat = p.stat()
    with A.open_image(p) as im:
        im.load(); rgb = im.convert('RGB'); width, height = im.size
        digest = hashlib.sha256()
        for y in range(0, height, 128):
            digest.update(rgb.crop((0, y, width, min(y+128, height))).tobytes())
    assert A.sha(p) == r['sha256'], p
    gray = A.thumbnail(r['path'])
    if gray.std() <= 1 or min(width, height) < 32:
        decode_status = 'decoded_low_information_exclude'
    else:
        decode_status = 'full_decode_ok'
    return dict(r, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, width=width,
        height=height, pixel_sha256=digest.hexdigest(), decode_status=decode_status,
        hashes=[A.A.phash(A.A.orient(gray, o)) for o in range(8)])

def ensure_features(rows):
    path = OUT/'lag_features.jsonl'
    cached = {r['path']:r for r in load_features(path)} if path.exists() else {}
    for r in rows:
        old = cached.get(r['path'])
        if old:
            assert old['sha256']==r['sha256'] and old['mtime_ns']==Path(r['path']).stat().st_mtime_ns
    pending = [r for r in rows if r['path'] not in cached]
    with path.open('a' if cached else 'w', encoding='utf-8') as stream, ThreadPoolExecutor(max_workers=4) as pool:
        for i, r in enumerate(pool.map(feature, pending), 1):
            cached[r['path']] = r
            stream.write(json.dumps(r)+'\n')
            if i%500==0:
                stream.flush(); status('features', processed=len(cached), images=len(rows))
    result = [cached[r['path']] for r in rows]
    assert len(result)==4854
    status('features_complete', images=len(result), decoded=dict(collections.Counter(r['decode_status'] for r in result)))
    return result

@lru_cache(maxsize=3500)
def interior_cached(path, mirrored=False):
    # lru_cache may invoke a missing key concurrently in several worker threads.
    # Serialize the file creation/read for each descriptor to satisfy Windows
    # sharing rules, while unrelated image descriptors remain parallel.
    with DESCRIPTOR_LOCKS[(path,mirrored)]:
        return build_interior(path,mirrored)

def build_interior(path,mirrored):
    gray = A.thumbnail(path)
    if mirrored:
        gray = np.fliplr(gray).copy()
    key = hashlib.sha256((path+':'+str(mirrored)).encode()).hexdigest()
    cache = DESC_DIR/(key+'.npz')
    if cache.exists():
        with np.load(cache) as stored:
            mask = stored['mask']; xy = stored['xy']; des = stored['desc'].astype(np.float32)
        return gray, mask, xy, des if len(des) else None
    binary = (gray>10).astype(np.uint8)*255
    contours,_ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(binary)
    if contours:
        cv2.drawContours(mask, [max(contours,key=cv2.contourArea)], -1, 255, -1)
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(45,45)))
    keys,des = cv2.SIFT_create(nfeatures=1000,contrastThreshold=.015).detectAndCompute(gray,mask)
    xy = np.float32([k.pt for k in keys])
    # OpenCV's SIFT descriptors are quantized integers; compact storage is exact.
    packed = np.empty((0,128), np.uint8) if des is None else des.astype(np.uint8)
    if des is not None:
        assert np.array_equal(packed.astype(np.float32), des)
    temporary = cache.with_suffix('.'+str(threading.get_ident())+'.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,mask=mask,xy=xy,desc=packed)
    temporary.replace(cache)
    return gray,mask,xy,des

def initialize_matcher():
    global MATCHER
    source = ROOT/'tools/confirm_internal_fundus_overlap_20261113.py'
    snapshot = OUT/'lag_matcher_source.py'
    if not snapshot.exists():
        snapshot.write_bytes(source.read_bytes())
    spec = importlib.util.spec_from_file_location('lag_fixed_matcher',snapshot)
    MATCHER = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(MATCHER)
    # Fixed matcher gates: do not resume old weak aperture-based decisions.
    text = snapshot.read_text(encoding='utf-8')
    assert "best['inner_inliers']>=25" in text
    assert "best['vessel_highpass_correlation']>=.70" in text
    DESC_DIR.mkdir(exist_ok=True)
    MATCHER.interior = lambda path: interior_cached(path,False)
    MATCHER.mirrored_interior = lambda path: interior_cached(path,True)
    A.load_historical_gray_cache()

def compare(r):
    a=A.A.orient(A.thumbnail(r['path_a']),int(r['orientation']))
    b=A.thumbnail(r['path_b'])
    ma,mb=float(a.mean()),float(b.mean()); va,vb=float(a.var()),float(b.var())
    covariance=float(cv2.mean(cv2.multiply(a,b,dtype=cv2.CV_32F))[0])-ma*mb
    ssim=((2*ma*mb+6.5025)*(2*covariance+58.5225))/((ma*ma+mb*mb+6.5025)*(va+vb+58.5225))
    reviewed = MATCHER.confirm(dict(r,ssim=round(ssim,8),decision='unreviewed_pHash_candidate'))
    reviewed['decision'] = reviewed['final_decision']
    return reviewed

def generate_candidates(lag, historical, local):
    target_rows = historical+local+lag
    representatives={}
    for r in target_rows:
        representatives.setdefault(r['sha256'], r)
    targets=list(representatives.values())
    index=A.A.HashIndex([r['hashes'][0] for r in targets])
    lagpaths={r['path'] for r in lag}
    exact=[]
    aliases=collections.defaultdict(list)
    for r in target_rows:
        aliases[r['sha256']].append(r['path'])
    for paths in aliases.values():
        if len(paths)>1 and lagpaths.intersection(paths):
            for p in paths[1:]:
                exact.append(dict(path_a=paths[0],path_b=p,decision='exact_file_sha256'))
    pixels=collections.defaultdict(list)
    for r in local+lag:
        pixels[(r['width'],r['height'],r['pixel_sha256'])].append(r['path'])
    for paths in pixels.values():
        if len(paths)>1 and lagpaths.intersection(paths):
            for p in paths[1:]:
                exact.append(dict(path_a=paths[0],path_b=p,decision='exact_decoded_rgb_pixels'))
    write('lag_exact_edges.csv',exact,['path_a','path_b','decision'])
    path=OUT/'lag_candidates.csv'
    binding_path=OUT/'lag_candidate_input.json'
    binding={str(p.relative_to(ROOT)):A.sha(p) for p in [OUT/'lag_features.jsonl',OUT/'local_features.jsonl',ROOT/'data/audit/session_20261110/eyeq_overlap/features.jsonl']}
    if path.exists() and binding_path.exists():
        assert json.loads(binding_path.read_text())==binding
        return A.read(path),exact
    pairs={}
    for i,r in enumerate(lag,1):
        for o,value in enumerate(r['hashes']):
            for j,d in index.query(value,6):
                t=targets[j]
                if r['path']==t['path'] or r['sha256']==t['sha256']:
                    continue
                if t['path'] in lagpaths and t['path']<r['path']:
                    continue
                key=(r['path'],t['path'])
                if key not in pairs or d<pairs[key][0]:
                    pairs[key]=(d,o)
        if i%250==0:
            status('candidate_generation',processed=i,images=len(lag),pairs=len(pairs))
    candidates=[dict(path_a=a,path_b=b,hash_distance=d,orientation=o) for (a,b),(d,o) in sorted(pairs.items())]
    write('lag_candidates.csv',candidates,['path_a','path_b','hash_distance','orientation'])
    binding_path.write_text(json.dumps(binding,indent=2))
    status('candidate_generation_complete',pairs=len(candidates),exact_edges=len(exact))
    return candidates,exact

def main():
    OUT.mkdir(exist_ok=True);cv2.setNumThreads(1);cv2.setRNGSeed(20261113)
    rows=collect();lag=ensure_features(rows)
    historical=load_features(ROOT/'data/audit/session_20261110/eyeq_overlap/features.jsonl')
    local=load_features(OUT/'local_features.jsonl')
    assert len(historical)==78047 and len(local)==7567,(len(historical),len(local))
    candidates,exact=generate_candidates(lag,historical,local)
    initialize_matcher()
    matcher_binding={str(p.relative_to(ROOT)):A.sha(p) for p in [OUT/'lag_matcher_source.py',Path(__file__),OUT/'lag_candidates.csv']}
    signature=OUT/'lag_metric_input.json'
    if signature.exists():
        assert json.loads(signature.read_text())==matcher_binding,'Matcher/input changed, cannot reuse old decisions'
    else:
        signature.write_text(json.dumps(matcher_binding,indent=2))
    dest=OUT/'lag_candidate_metrics.csv'
    reviewed={(r['path_a'],r['path_b']):r for r in A.read(dest)} if dest.exists() else {}
    pending=[r for r in candidates if (r['path_a'],r['path_b']) not in reviewed]
    decisions=collections.Counter(r['decision'] for r in reviewed.values())
    with dest.open('a' if reviewed else 'w',encoding='utf-8-sig',newline='') as stream,ThreadPoolExecutor(max_workers=6) as pool:
        writer=None
        for i,r in enumerate(pool.map(compare,pending),1):
            if writer is None:
                writer=csv.DictWriter(stream,fieldnames=list(r))
                if not reviewed:writer.writeheader()
            writer.writerow(r);reviewed[(r['path_a'],r['path_b'])]=r;decisions[r['decision']]+=1
            if i%250==0:
                stream.flush();status('confirmation',processed=len(reviewed),pairs=len(candidates),decisions=dict(decisions))
    metrics=list(reviewed.values())
    assert len(metrics)==len(candidates)
    assert all(r['decision'] in {'confirmed_same_photo','ambiguous_exclude','rejected_pHash_collision'} for r in metrics)
    confirmed=[dict(path_a=r['path_a'],path_b=r['path_b'],decision=r['decision']) for r in metrics if r['decision']=='confirmed_same_photo']
    write('lag_final_confirmed_edges.csv',confirmed,['path_a','path_b','decision'])
    excluded=[dict(path_a=r['path_a'],path_b=r['path_b'],decision=r['decision']) for r in metrics if r['decision']=='ambiguous_exclude']
    write('lag_ambiguous_excluded_pairs.csv',excluded,['path_a','path_b','decision'])
    summary=dict(date='2026-11-13',completed=True,images=len(lag),source_positive_images=1711,
        source_negative_images=3143,positive_definition='Original CVPR2019 section3.1 specialist diagnosis using IOP, visual fields and manual optic-disc assessment; original suspicious_glaucoma folder name retained',
        decoded=dict(collections.Counter(r['decode_status'] for r in lag)),historical_images=len(historical),
        other_new_images=len(local),candidate_pairs=len(candidates),exact_edges=len(exact),candidate_decisions=dict(decisions),
        method='All 8 ROI pHash orientations with exhaustive radius 6; full file and RGB decode for all LAG originals; exact file/pixel aliases; every candidate checked with retinal-interior mask eroded45, mutual SIFT homography, >=25inliers ratio>=.65 coverage>=.04 both images, aligned graycorr>=.93 and highpasscorr>=.70, reprojection<=1.5; exact normalized/near identical photo also confirms. High-evidence inconclusive candidates excluded explicitly.',
        scope='Official public part1 contains4854 originals, not5824 paper images or11760 newer README whole-database images. Attention maps excluded from image counts. Every folder label checked against per-image txt; positive diagnostic definition traced to original clinical publication. Patient IDs absent; assign source a single whole-source role. Global graph/eligible decision delegated to combined audit.',
        models_scored=False,training_manifest_changed=False,
        inputs={str(p.relative_to(ROOT)):A.sha(p) for p in [LAG/'acquisition_20261113.json',OUT/'lag_cvpr2019.pdf',OUT/'lag_input_images.csv',OUT/'lag_features.jsonl',OUT/'lag_candidates.csv',OUT/'lag_candidate_metrics.csv',OUT/'lag_exact_edges.csv',OUT/'lag_final_confirmed_edges.csv',OUT/'lag_matcher_source.py',Path(__file__)]})
    (OUT/'lag_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    status('completed',summary=summary)

if __name__=='__main__':
    main()
