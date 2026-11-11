"""CPU-only EyeQ association screening; never releases or modifies a training cohort.

Candidates: SHA256, basename, ROI pHash radius six over all eight square
orientations. Confirmation: pHash<=4, global SSIM>=.965, ORB/RANSAC>=14
inliers and ratio>=.5. Unconfirmed candidates are retained, not passed as safe.
This bounded screen cannot prove absence of arbitrary crops or unknown patients.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
ACQUISITION=ROOT/'data/audit/eyeq_acquisition_20261110'


def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def save_csv(path,rows,fields):
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)


def orient(gray,orientation):
    return np.ascontiguousarray(np.rot90(np.fliplr(gray) if orientation>=4 else gray,orientation%4))


def phash(gray):
    small=cv2.resize(gray,(64,64),interpolation=cv2.INTER_AREA).astype(np.float32)
    values=cv2.dct(small)[:8,:8].flatten()[1:]
    return int.from_bytes(np.packbits(values>np.median(values)).tobytes(),'big')


@lru_cache(maxsize=192)
def thumbnail(path):
    before=Path(path).stat()
    with Image.open(path) as image:
        image.draft('RGB',(768,768));image.thumbnail((768,768))
        gray=np.asarray(image.convert('L'))
    ys,xs=np.where(gray>8)
    if len(xs) and xs.max()-xs.min()+1>=gray.shape[1]*.35 and ys.max()-ys.min()+1>=gray.shape[0]*.35:
        gray=gray[ys.min():ys.max()+1,xs.min():xs.max()+1]
    after=Path(path).stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError(f'image changed: {path}')
    return cv2.resize(gray,(384,384),interpolation=cv2.INTER_AREA)


class HashIndex:
    """Eight disjoint 8-bit bands guarantee retrieval of every radius<=7 hit."""
    def __init__(self,hashes):
        self.hashes=list(hashes);self.bands=[collections.defaultdict(list) for _ in range(8)]
        for i,value in enumerate(self.hashes):
            for band in range(8):self.bands[band][(value>>(8*band))&255].append(i)

    def query(self,value,radius=6):
        if not 0<=radius<=7:raise ValueError('index supports radius 0..7')
        indices=set()
        for band in range(8):indices.update(self.bands[band].get((value>>(8*band))&255,()))
        return [(i,(value^self.hashes[i]).bit_count()) for i in sorted(indices)
                if (value^self.hashes[i]).bit_count()<=radius]


def compare(a,b,orientation):
    a=orient(a,orientation);af=a.astype(np.float64);bf=b.astype(np.float64)
    ma,mb=af.mean(),bf.mean();va,vb=af.var(),bf.var();cov=((af-ma)*(bf-mb)).mean()
    ssim=float(((2*ma*mb+6.5025)*(2*cov+58.5225))/((ma*ma+mb*mb+6.5025)*(va+vb+58.5225)))
    distance=(phash(a)^phash(b)).bit_count();inliers=good_n=0;ratio=0.
    if distance<=4 and ssim>=.965:
        orb=cv2.ORB_create(nfeatures=1200)
        k1,d1=orb.detectAndCompute(a,None);k2,d2=orb.detectAndCompute(b,None)
        if d1 is not None and d2 is not None and min(len(d1),len(d2))>=4:
            matches=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1,d2,k=2)
            good=[pair[0] for pair in matches if len(pair)==2 and pair[0].distance<.72*pair[1].distance]
            good_n=len(good)
            if good_n>=4:
                src=np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
                dst=np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
                _,mask=cv2.findHomography(src,dst,cv2.RANSAC,4.)
                if mask is not None:inliers=int(mask.sum());ratio=inliers/good_n
    return dict(phash_distance=distance,ssim=ssim,orb_matches=good_n,orb_inliers=inliers,
                orb_ratio=ratio,confirmed=distance<=4 and ssim>=.965 and inliers>=14 and ratio>=.5)


def component_rows(rows,active,edges,prior_edges):
    parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    patients={};groups={};sha={}
    for row in rows:
        path=row['path'];find(path)
        if row.get('eyeq_split'):
            patient=row['patient_id']
            if patient in patients:union(path,patients[patient])
            patients[patient]=path
        if row.get('sha256'):
            if row['sha256'] in sha:union(path,sha[row['sha256']])
            sha[row['sha256']]=path
    for row in active:
        key=row['related_group']
        if key in groups:union(row['path'],groups[key])
        groups[key]=row['path']
    for a,b in list(edges)+list(prior_edges):union(a,b)
    roles=collections.defaultdict(set);splits=collections.defaultdict(set)
    for row in active:roles[find(row['path'])].add(row['split'])
    for row in rows:
        if row.get('eyeq_split'):splits[find(row['path'])].add(row['eyeq_split'])
    result=[]
    for row in rows:
        if not row.get('eyeq_split'):continue
        key=find(row['path']);ar=roles[key];es=splits[key]
        protected=ar-{'train'}
        result.append(dict(row,component=hashlib.sha256(key.encode()).hexdigest()[:24],
            active_roles='|'.join(sorted(ar)),component_eyeq_splits='|'.join(sorted(es)),
            blocked_from_new_training=bool(protected or 'test' in es),
            formal_v3_development_exposure=bool(ar&{'train','val'}),
            active_role_conflict=len(ar)>1,official_split_conflict=len(es)>1,training_released=False))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'data/audit/session_20261110/eyeq_overlap')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(1);cv2.setRNGSeed(20261110)
    inputs=[ACQUISITION/'imported_originals.csv',ACQUISITION/'import_verification.json',
        ROOT/'data/audit/2026-11-09/image_manifest.csv',ROOT/'data/prepared/formal_v3/manifest.csv',
        ROOT/'data/audit/deep/release_inputs/confirmed_edges_v1.csv',Path(__file__)]
    binding={str(p.relative_to(ROOT)):digest(p) for p in inputs}
    provenance=out/'inputs.json'
    if provenance.exists() and json.loads(provenance.read_text())!=binding:raise ValueError('audit input changed; use a new output directory')
    provenance.write_text(json.dumps(binding,indent=2))
    start=time.time()
    def status(stage,**fields):
        state=dict(stage=stage,pid=os.getpid(),updated_at=time.time(),elapsed_seconds=time.time()-start,
                   training_released=False,**fields)
        temporary=out/'status.tmp';temporary.write_text(json.dumps(state,indent=2));temporary.replace(out/'status.json')
        print(json.dumps(state),flush=True)
    originals=read(inputs[0]);historical=read(inputs[2]);active=read(inputs[3])
    for row in historical:row['path']=str((ROOT/'data'/row['path']).resolve())
    for row in active:row['path']=str((ROOT/'data'/row['path']).resolve())
    # Include every historical decoded source, including excluded families, to
    # preserve bridge associations rather than only screen current train rows.
    historical=[r for r in historical if r['status']=='full_decode_ok']
    rows=originals+historical
    feature_path=out/'features.jsonl';features={}
    if feature_path.exists():
        for line in feature_path.read_text().splitlines():
            try:record=json.loads(line)
            except json.JSONDecodeError:continue
            features[record['path']]=record
    def feature(row):
        path=row['path'];stat=Path(path).stat();old=features.get(path)
        if old and old['bytes']==stat.st_size and old['mtime_ns']==stat.st_mtime_ns:return old
        image=thumbnail(path)
        return dict(path=path,sha256=row['sha256'],bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,
                    hashes=[phash(orient(image,o)) for o in range(8)])
    status('features',images=len(rows))
    with feature_path.open('a',encoding='utf-8') as stream,ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i,record in enumerate(pool.map(feature,rows),1):
            features[record['path']]=record
            stream.write(json.dumps(record)+'\n')
            if i%1000==0:stream.flush();status('features',processed=i,images=len(rows))
    thumbnail.cache_clear()
    # One representative per SHA reduces redundant candidates; exact edges
    # still reconnect every physical alias during component construction.
    representatives={}
    for row in rows:representatives.setdefault(row['sha256'],row)
    targets=list(representatives.values());index=HashIndex([features[r['path']]['hashes'][0] for r in targets])
    basenames=collections.defaultdict(list)
    for row in rows:basenames[Path(row['path']).stem.casefold()].append(row)
    candidates={}
    for i,row in enumerate(originals,1):
        path=row['path']
        for orientation,h in enumerate(features[path]['hashes']):
            for j,distance in index.query(h):
                target=targets[j];other=target['path']
                if other==path or target['sha256']==row['sha256']:continue
                if target.get('eyeq_split') and other<path:continue
                key=(path,other)
                if key not in candidates or distance<candidates[key][0]:candidates[key]=(distance,orientation,'roi_phash_radius6')
        for target in basenames[Path(path).stem.casefold()]:
            if target['path']!=path and target['sha256']!=row['sha256']:
                distances=[(h^features[target['path']]['hashes'][0]).bit_count() for h in features[path]['hashes']]
                best=min(range(8),key=lambda o:distances[o]);candidates.setdefault((path,target['path']),(distances[best],best,'same_basename'))
        if i%1000==0:status('candidates',processed=i,images=len(originals),pairs=len(candidates))
    candidate_rows=[dict(path_a=a,path_b=b,hash_distance=d,orientation=o,reason=r) for (a,b),(d,o,r) in sorted(candidates.items())]
    fields=['path_a','path_b','hash_distance','orientation','reason']
    save_csv(out/'candidates.csv',candidate_rows,fields)
    confirmed=[];uncertain=[];metric_fields=fields+['phash_distance','ssim','orb_matches','orb_inliers','orb_ratio','confirmed']
    with (out/'candidate_metrics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=metric_fields);writer.writeheader()
        for i,row in enumerate(candidate_rows,1):
            metrics=compare(thumbnail(row['path_a']),thumbnail(row['path_b']),row['orientation'])
            result=dict(row,**metrics);writer.writerow(result)
            (confirmed if metrics['confirmed'] else uncertain).append(result)
            if i%1000==0:stream.flush();status('confirmation',processed=i,pairs=len(candidate_rows),confirmed=len(confirmed))
    save_csv(out/'confirmed_edges.csv',confirmed,metric_fields)
    save_csv(out/'unresolved_candidates.csv',uncertain,metric_fields)
    prior=[(str((ROOT/'data'/r['path_a']).resolve()),str((ROOT/'data'/r['path_b']).resolve())) for r in read(inputs[4])]
    associations=component_rows(rows,active,[(r['path_a'],r['path_b']) for r in confirmed],prior)
    save_csv(out/'eyeq_components.csv',associations,list(associations[0]))
    # Unresolved candidates are conservatively blocked from automatic release.
    unresolved={r['path_a'] for r in uncertain}|{r['path_b'] for r in uncertain}
    unresolved_components={r['component'] for r in associations if r['path'] in unresolved}
    for row in associations:
        row['unresolved_component']=row['component'] in unresolved_components
        row['blocked_from_new_training']|=row['unresolved_component']
    save_csv(out/'eyeq_screening_decisions.csv',associations,list(associations[0]))
    summary=dict(eyeq_images=len(originals),historical_images=len(historical),
        unique_sha_targets=len(targets),candidate_pairs=len(candidate_rows),confirmed_pairs=len(confirmed),
        unresolved_pairs=len(uncertain),eyeq_train_blocked=sum(r['eyeq_split']=='train' and r['blocked_from_new_training'] for r in associations),
        eyeq_train_screened_unblocked=sum(r['eyeq_split']=='train' and not r['blocked_from_new_training'] for r in associations),
        eyeq_test_exposed_to_formal_v3_development=sum(r['eyeq_split']=='test' and r['formal_v3_development_exposure'] for r in associations),
        active_role_conflict_components=len({r['component'] for r in associations if r['active_role_conflict']}),
        official_split_conflict_components=len({r['component'] for r in associations if r['official_split_conflict']}),
        active_association_roles=dict(collections.Counter(r['active_roles'] or 'none' for r in associations)),
        training_released=False,coverage='All decoded historical paths and EyeQ; SHA, basename, ROI pHash radius6 in 8 orientations; strict SSIM+ORB confirmation; known patient/existing family transitive closure',
        limitations='Does not prove absence of arbitrary crop/rotation, undocumented aliases or unknown patient overlap. Unresolved candidates quarantine their complete known component. No cohort or teacher targets created.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    status('completed',summary=summary)


if __name__=='__main__':main()
