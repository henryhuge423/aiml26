"""Full GRAPE source-label/identity-bound image association audit."""
import csv,hashlib,json,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import numpy as np
import audit_external_overlap_20261113 as A
import confirm_internal_fundus_overlap_20261113 as S

OUT=A.OUT
def read_features(p):
    with p.open(encoding='utf-8') as f:return [json.loads(line) for line in f]
def feature(r):
    p=Path(r['path']);st=p.stat()
    assert A.sha(p)==r['sha256']
    with A.open_image(p) as im:
        im.load();rgb=im if im.mode=='RGB' else im.convert('RGB');h=hashlib.sha256()
        for y in range(0,im.height,128):h.update(rgb.crop((0,y,im.width,min(y+128,im.height))).tobytes())
        wh=im.size
    g=A.thumbnail(A.norm(p))
    return dict(r,path=A.norm(p),bytes=st.st_size,mtime_ns=st.st_mtime_ns,width=wh[0],height=wh[1],pixel_sha256=h.hexdigest(),hashes=[A.A.phash(A.A.orient(g,o)) for o in range(8)])

def confirm(r):
    a=A.A.orient(A.thumbnail(r['path_a']),int(r['orientation']));b=A.thumbnail(r['path_b'])
    af=a.astype(np.float64);bf=b.astype(np.float64);ma=af.mean();mb=bf.mean();va=af.var();vb=bf.var();cov=((af-ma)*(bf-mb)).mean()
    ssim=((2*ma*mb+6.5025)*(2*cov+58.5225))/((ma*ma+mb*mb+6.5025)*(va+vb+58.5225))
    result=S.confirm(dict(r,ssim=ssim,decision='candidate'))
    result['decision']=result['final_decision'];return result

def main():
    cv2.setNumThreads(1);A.load_historical_gray_cache();t=time.time()
    rows=A.read(OUT/'grape_source_manifest.csv');fp=OUT/'grape_features.jsonl'
    if fp.exists():new=read_features(fp)
    else:
        new=[]
        with fp.open('w',encoding='utf-8') as f,ThreadPoolExecutor(max_workers=4) as pool:
            for i,r in enumerate(pool.map(feature,rows),1):new.append(r);f.write(json.dumps(r)+'\n')
    print(json.dumps(dict(stage='grape_features_completed',images=len(new),elapsed=time.time()-t)),flush=True)
    historical=read_features(A.ROOT/'data/audit/session_20261110/eyeq_overlap/features.jsonl')
    for r in historical:r['path']=A.norm(r['path'])
    local=read_features(OUT/'local_features.jsonl');others=historical+local
    if (OUT/'lag_features.jsonl').exists():others+=read_features(OUT/'lag_features.jsonl')
    allrows=others+new;bysha={};exact=[]
    for r in allrows:
        if r['sha256'] in bysha:
            if r['source']=='grape' if 'source' in r else False:exact.append(dict(path_a=r['path'],path_b=bysha[r['sha256']]['path'],decision='exact_file_sha256'))
        else:bysha[r['sha256']]=r
    A.A.save_csv(OUT/'grape_exact_edges.csv',exact,['path_a','path_b','decision'])
    cp=OUT/'grape_candidates.csv'
    if cp.exists():candidates=A.read(cp)
    else:
        targets=list(bysha.values());index=A.A.HashIndex([r['hashes'][0] for r in targets]);pairs={};newpaths={r['path'] for r in new}
        for i,r in enumerate(new,1):
            for o,h in enumerate(r['hashes']):
                for j,d in index.query(h,6):
                    target=targets[j]
                    if target['sha256']==r['sha256'] or target['path']==r['path']:continue
                    if target['path'] in newpaths and target['path']<r['path']:continue
                    key=(r['path'],target['path'])
                    if key not in pairs or d<pairs[key][0]:pairs[key]=(d,o)
        candidates=[dict(path_a=a,path_b=b,hash_distance=d,orientation=o) for (a,b),(d,o) in sorted(pairs.items())]
        A.A.save_csv(cp,candidates,['path_a','path_b','hash_distance','orientation'])
    print(json.dumps(dict(stage='grape_candidates_completed',pairs=len(candidates),elapsed=time.time()-t)),flush=True)
    mp=OUT/'grape_candidate_metrics.csv';done={}
    if mp.exists():done={(r['path_a'],r['path_b']):r for r in A.read(mp)}
    pending=[r for r in candidates if (r['path_a'],r['path_b']) not in done]
    # Keep the 631 GRAPE descriptors hot while reading each historical target
    # once, rather than repeatedly cycling through tens of thousands of targets.
    pending.sort(key=lambda r:(r['path_b'],r['path_a']))
    with mp.open('a' if done else 'w',encoding='utf-8-sig',newline='') as f,ThreadPoolExecutor(max_workers=6) as pool:
        w=None
        for i,r in enumerate(pool.map(confirm,pending),1):
            if w is None:
                w=csv.DictWriter(f,fieldnames=list(r))
                if not done:w.writeheader()
            w.writerow(r);done[(r['path_a'],r['path_b'])]=r
            if i%500==0:f.flush();print(json.dumps(dict(stage='grape_confirmation',processed=len(done),pairs=len(candidates),elapsed=time.time()-t)),flush=True)
    metrics=list(done.values());A.A.save_csv(OUT/'grape_confirmed_edges.csv',[r for r in metrics if r['decision']=='confirmed_same_photo'],list(metrics[0]))
    result=dict(completed=True,images=len(new),pairs=len(candidates),clinical_manifest_sha256=A.sha(OUT/'grape_source_manifest.csv'),features_sha256=A.sha(fp),candidate_sha256=A.sha(cp),metrics_sha256=A.sha(mp),confirmed_pairs=sum(r['decision']=='confirmed_same_photo' for r in metrics),ambiguous_pairs=sum(r['decision']=='ambiguous_exclude' for r in metrics),rejected_pairs=sum(r['decision']=='rejected_pHash_collision' for r in metrics),clinical_disease='All visits inherit baseline OAG or ACG diagnosis; progression is not a negative disease label',patients=144,eyes=263)
    (OUT/'grape_overlap_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
