"""Resume the exact EyeQ candidate set with a persistent grayscale cache.

Same candidate identities and strict confirmation rule as audit_eyeq_overlap.
The original candidate generator must have exited before this continuation.
"""
import argparse
import collections
import csv
import json
import math
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import audit_eyeq_overlap as A


def moments(gray):
    mean,std=cv2.meanStdDev(gray)
    return float(mean[0,0]),float(std[0,0]**2)


def compare_cached(a,b,orientation,distance,stats_a,stats_b):
    ma,va=stats_a;mb,vb=stats_b
    aligned=A.orient(a,orientation)
    # Products of uint8 values are exactly representable in float32; OpenCV
    # accumulates their mean in double precision. This avoids per-pair float64
    # centered arrays, while retaining a numerical margin at the pass threshold.
    product_mean=cv2.mean(cv2.multiply(aligned,b,dtype=cv2.CV_32F))[0]
    covariance=product_mean-ma*mb
    ssim=((2*ma*mb+6.5025)*(2*covariance+58.5225))/((ma*ma+mb*mb+6.5025)*(va+vb+58.5225))
    if distance<=4 and ssim>=.965-1e-10:
        return A.compare(a,b,orientation)  # Original exact SSIM and ORB gate.
    return dict(phash_distance=distance,ssim=ssim,orb_matches=0,orb_inliers=0,orb_ratio=0.,confirmed=False)


def verify_candidate_proof(proof,count):
    if (proof.get('stage')!='confirmation' or proof.get('processed',0)<=0
            or proof.get('pairs')!=count or count<=0):
        raise ValueError('candidate generation is unsealed, incomplete or has a different pair count')


def atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)


def array_nbytes(shape,dtype):
    return math.prod(shape)*np.dtype(dtype).itemsize


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screen',type=Path,default=ROOT/'data/audit/session_20261110/eyeq_overlap')
    args=parser.parse_args();out=args.screen
    cv2.setNumThreads(1);cv2.setRNGSeed(20261110)
    source_proof_path=out/'candidate_generation_completed.json'
    source_proof=json.loads((source_proof_path if source_proof_path.exists() else out/'status.json').read_text())
    print(json.dumps(dict(stage='validating_candidates',pid=os.getpid())),flush=True)
    input_path=out/'inputs.json';bindings=json.loads(input_path.read_text())
    for relative,expected in bindings.items():
        if A.digest(ROOT/relative)!=expected:raise ValueError(f'screen input changed: {relative}')
    candidate_path=out/'candidates.csv'
    candidate_digest=A.digest(candidate_path)
    endpoints=set();pairs=0
    with candidate_path.open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):endpoints.update((row['path_a'],row['path_b']));pairs+=1
    verify_candidate_proof(source_proof,pairs)
    atomic_json(source_proof_path,source_proof)
    atomic_json(out/'status.json',dict(stage='validating_sources',pid=os.getpid(),pairs=pairs,training_released=False))
    paths=sorted(endpoints);by_path={p:i for i,p in enumerate(paths)}
    feature_records={}
    with (out/'features.jsonl').open() as stream:
        for line in stream:
            record=json.loads(line)
            if record['path'] in endpoints:feature_records[record['path']]=record
    if set(feature_records)!=endpoints:raise ValueError('candidate endpoints missing feature records')
    for path,record in feature_records.items():
        stat=Path(path).stat()
        if (stat.st_size,stat.st_mtime_ns)!=(record['bytes'],record['mtime_ns']):raise ValueError(f'source changed: {path}')
    cache=out/'confirmation_cache';cache.mkdir(exist_ok=True)
    cache_binding=dict(candidates_sha256=candidate_digest,features_sha256=A.digest(out/'features.jsonl'),
        paths=paths,shape=[len(paths),384,384],dtype='uint8')
    sidecar=cache/'binding.json'
    if sidecar.exists() and json.loads(sidecar.read_text())!=cache_binding:raise ValueError('confirmation cache input mismatch')
    atomic_json(sidecar,cache_binding)
    size=len(paths)*384*384
    if not (cache/'gray.dat').exists() and shutil.disk_usage(cache).free<size+10_000_000_000:
        raise ValueError('insufficient space for bounded grayscale cache plus 10GB reserve')
    if shutil.disk_usage(cache).free<10_000_000_000:raise ValueError('less than 10GB cache reserve remains')
    evidence_path=out/'cached_confirmation.json'
    previous_evidence=json.loads(evidence_path.read_text()) if evidence_path.exists() else None
    if previous_evidence:
        checks={'cache_sha256':cache/'gray.dat','cache_moments_sha256':cache/'moments.dat','cache_binding_sha256':sidecar}
        if previous_evidence['candidate_sha256']!=candidate_digest or any(A.digest(p)!=previous_evidence[key] for key,p in checks.items()):
            raise ValueError('existing confirmation cache integrity mismatch; refusing reuse')
    def mapped(name,dtype,shape):
        path=cache/name;expected=array_nbytes(shape,dtype)
        if path.exists() and path.stat().st_size!=expected:raise ValueError(f'incomplete cache allocation: {path}')
        return np.memmap(path,mode='r+' if path.exists() else 'w+',dtype=dtype,shape=shape)
    gray=mapped('gray.dat',np.uint8,(len(paths),384,384))
    stats=mapped('moments.dat',np.float64,(len(paths),2))
    ready=mapped('ready.dat',np.uint8,(len(paths),))
    if not previous_evidence:
        # Without a completed whole-cache checksum, reconstruct interrupted
        # rows rather than silently treating an unverifiable cache as evidence.
        ready[:]=0;ready.flush()
    started=time.time()
    def status(stage,**fields):
        state=dict(stage=stage,pid=os.getpid(),updated_at=time.time(),elapsed_seconds=time.time()-started,
                   training_released=False,continuation='cached_exact_confirmation',**fields)
        temp=out/'status.tmp';temp.write_text(json.dumps(state,indent=2));temp.replace(out/'status.json')
        print(json.dumps(state),flush=True)
    status('caching_confirmation_images',unique_paths=len(paths),pairs=pairs,cache_bytes=size)
    def fill(i):
        if not ready[i]:
            stat=Path(paths[i]).stat();record=feature_records[paths[i]]
            if (stat.st_size,stat.st_mtime_ns)!=(record['bytes'],record['mtime_ns']):
                raise ValueError(f'source changed before caching: {paths[i]}')
            image=A.thumbnail(paths[i]);gray[i]=image;stats[i]=moments(image)
        return i
    pending=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for count,index in enumerate(pool.map(fill,range(len(paths))),1):
            pending.append(index)
            if count%1000==0 or count==len(paths):
                # Commit data before its ready bitmap, so an interrupted cache
                # never marks an incomplete grayscale row as reusable.
                gray.flush();stats.flush();ready[pending]=1;ready.flush();pending=[]
                status('caching_confirmation_images',processed=count,unique_paths=len(paths),pairs=pairs)
    A.thumbnail.cache_clear()
    evidence=dict(candidate_sha256=candidate_digest,candidate_pairs=pairs,cache_sha256=A.digest(cache/'gray.dat'),
        cache_binding_sha256=A.digest(sidecar),cache_moments_sha256=A.digest(cache/'moments.dat'),
        unique_paths=len(paths),implementation_sha256=A.digest(Path(__file__)))
    atomic_json(evidence_path,evidence)
    # Both implementations remain in provenance; the source candidates and all
    # image features were generated by the unchanged original implementation.
    bindings[Path(__file__).relative_to(ROOT).as_posix()]=evidence['implementation_sha256']
    bindings[candidate_path.relative_to(ROOT).as_posix()]=candidate_digest
    bindings[source_proof_path.relative_to(ROOT).as_posix()]=A.digest(source_proof_path)
    atomic_json(input_path,bindings)
    fields=['path_a','path_b','hash_distance','orientation','reason','phash_distance','ssim','orb_matches','orb_inliers','orb_ratio','confirmed']
    confirmed=[];unresolved_paths=set();uncertain_count=0
    with candidate_path.open(encoding='utf-8-sig',newline='') as source,\
            (out/'candidate_metrics.csv').open('w',encoding='utf-8-sig',newline='') as metrics_file,\
            (out/'unresolved_candidates.csv').open('w',encoding='utf-8-sig',newline='') as unresolved_file:
        writer=csv.DictWriter(metrics_file,fieldnames=fields);writer.writeheader()
        unresolved=csv.DictWriter(unresolved_file,fieldnames=['path_a','path_b']);unresolved.writeheader()
        for count,row in enumerate(csv.DictReader(source),1):
            i,j=by_path[row['path_a']],by_path[row['path_b']]
            result=dict(row,**compare_cached(gray[i],gray[j],int(row['orientation']),int(row['hash_distance']),stats[i],stats[j]))
            writer.writerow(result)
            if result['confirmed']:confirmed.append(result)
            else:
                uncertain_count+=1;unresolved.writerow({k:row[k] for k in ('path_a','path_b')})
                unresolved_paths.update((row['path_a'],row['path_b']))
            if count%10000==0:metrics_file.flush();status('confirmation',processed=count,pairs=pairs,confirmed=len(confirmed))
    A.save_csv(out/'confirmed_edges.csv',confirmed,fields)
    if A.digest(candidate_path)!=candidate_digest:raise ValueError('candidate set changed during confirmation')
    for relative,expected in bindings.items():
        if A.digest(ROOT/relative)!=expected:raise ValueError(f'input changed during confirmation: {relative}')
    originals=A.read(A.ACQUISITION/'imported_originals.csv')
    historical=A.read(ROOT/'data/audit/2026-11-09/image_manifest.csv')
    summary=dict(eyeq_images=len(originals),historical_images=sum(r['status']=='full_decode_ok' for r in historical),
        candidate_pairs=pairs,confirmed_pairs=len(confirmed),unresolved_pairs=uncertain_count,
        training_released=False,coverage='Exact original candidate set, eight ROI pHash orientations radius6 and basename; unchanged SSIM/ORB final pass gate; persistent grayscale cache',
        limitations='No proof of absence of arbitrary transformations or unknown patient links. All unconfirmed candidates retained and quarantined by finalizer.',
        confirmation_evidence=evidence)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    status('completed',summary=summary)


if __name__=='__main__':main()
