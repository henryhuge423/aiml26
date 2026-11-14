"""Complete LAG/GRAPE cross-source audit for GRAPE acquired after LAG scan began."""
import collections
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import audit_lag_overlap_20261113 as C

def main():
    cv2.setNumThreads(1);cv2.setRNGSeed(20261113)
    lag=C.load_features(C.OUT/'lag_features.jsonl')
    grape=C.load_features(C.OUT/'grape_features.jsonl')
    assert len(lag)==4854 and len(grape)==631
    exact=[]
    by_sha=collections.defaultdict(list)
    by_pixel=collections.defaultdict(list)
    for r in lag:
        by_sha[r['sha256']].append(r['path'])
        by_pixel[(r['width'],r['height'],r['pixel_sha256'])].append(r['path'])
    for r in grape:
        for p in by_sha[r['sha256']]:
            exact.append(dict(path_a=r['path'],path_b=p,decision='exact_file_sha256'))
        for p in by_pixel[(r['width'],r['height'],r['pixel_sha256'])]:
            exact.append(dict(path_a=r['path'],path_b=p,decision='exact_decoded_rgb_pixels'))
    C.write('lag_grape_exact_edges.csv',exact,['path_a','path_b','decision'])
    representatives={}
    for r in lag:representatives.setdefault(r['sha256'],r)
    targets=list(representatives.values())
    index=C.A.A.HashIndex([r['hashes'][0] for r in targets])
    pairs={}
    for r in grape:
        for o,h in enumerate(r['hashes']):
            for j,d in index.query(h,6):
                t=targets[j]
                if r['sha256']==t['sha256']:continue
                key=(r['path'],t['path'])
                if key not in pairs or d<pairs[key][0]:pairs[key]=(d,o)
    candidates=[dict(path_a=a,path_b=b,hash_distance=d,orientation=o)
                for (a,b),(d,o) in sorted(pairs.items())]
    C.write('lag_grape_candidates.csv',candidates,['path_a','path_b','hash_distance','orientation'])
    C.DESC_DIR=C.OUT/'lag_grape_descriptor_cache'
    C.initialize_matcher()
    dest=C.OUT/'lag_grape_candidate_metrics.csv'
    old=C.A.read(dest) if dest.exists() else []
    completed={(r['path_a'],r['path_b']):r for r in old}
    pending=[r for r in candidates if (r['path_a'],r['path_b']) not in completed]
    decisions=collections.Counter(r['decision'] for r in old)
    with dest.open('a' if old else 'w',encoding='utf-8-sig',newline='') as stream,ThreadPoolExecutor(max_workers=3) as pool:
        writer=None
        for i,r in enumerate(pool.map(C.compare,pending),1):
            if writer is None:
                writer=csv.DictWriter(stream,fieldnames=list(r))
                if not old:writer.writeheader()
            writer.writerow(r);completed[(r['path_a'],r['path_b'])]=r;decisions[r['decision']]+=1
            if i%250==0:
                stream.flush(); print(json.dumps(dict(stage='lag_grape_confirmation',processed=len(completed),pairs=len(candidates),decisions=dict(decisions))),flush=True)
    metrics=list(completed.values())
    assert len(metrics)==len(candidates)
    assert all(r['decision'] in {'confirmed_same_photo','ambiguous_exclude','rejected_pHash_collision'} for r in metrics)
    confirmed=[dict(path_a=r['path_a'],path_b=r['path_b'],decision=r['decision']) for r in metrics if r['decision']=='confirmed_same_photo']
    C.write('lag_grape_final_confirmed_edges.csv',confirmed,['path_a','path_b','decision'])
    if not candidates:
        C.write('lag_grape_candidate_metrics.csv',[],['path_a','path_b','hash_distance','orientation','decision'])
    summary=dict(completed=True,lag_images=len(lag),grape_images=len(grape),candidate_pairs=len(candidates),
        exact_edges=len(exact),candidate_decisions=dict(decisions),
        method='Cross-source file and decoded-pixel identity; all eight ROI pHash orientations at radius6; every candidate adjudicated with the same fixed retinal-interior and highpass-correlation matcher as primary LAG audit',
        models_scored=False,training_manifest_changed=False,
        inputs={str(p.relative_to(C.ROOT)):C.A.sha(p) for p in [C.OUT/'lag_features.jsonl',C.OUT/'grape_features.jsonl',C.OUT/'lag_grape_exact_edges.csv',C.OUT/'lag_grape_candidates.csv',C.OUT/'lag_grape_candidate_metrics.csv',C.OUT/'lag_grape_final_confirmed_edges.csv',C.OUT/'lag_matcher_source.py',Path(C.__file__),Path(__file__)]})
    (C.OUT/'lag_grape_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
