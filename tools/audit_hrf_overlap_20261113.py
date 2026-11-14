"""Complete HRF45 photograph audit against the frozen project and acquired sources."""
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
 with A.open_image(p) as im:
  im.load();rgb=im.convert('RGB');h=hashlib.sha256(rgb.tobytes()).hexdigest();wh=im.size
 g=A.thumbnail(A.norm(p))
 return dict(r,path=A.norm(p),bytes=st.st_size,mtime_ns=st.st_mtime_ns,width=wh[0],height=wh[1],pixel_sha256=h,hashes=[A.A.phash(A.A.orient(g,o)) for o in range(8)])
def confirm(r):
 a=A.A.orient(A.thumbnail(r['path_a']),int(r['orientation']));b=A.thumbnail(r['path_b'])
 af=a.astype(np.float64);bf=b.astype(np.float64);ma=af.mean();mb=bf.mean();va=af.var();vb=bf.var();cov=((af-ma)*(bf-mb)).mean()
 ssim=((2*ma*mb+6.5025)*(2*cov+58.5225))/((ma*ma+mb*mb+6.5025)*(va+vb+58.5225))
 result=S.confirm(dict(r,ssim=ssim,decision='candidate'));result['decision']=result['final_decision'];return result
def main():
 cv2.setNumThreads(1);A.load_historical_gray_cache();t=time.time();rows=[]
 for p in sorted((A.ROOT/'data/external_glaucoma/hrf/extracted/images').glob('*')):
  if p.suffix.lower() not in {'.jpg','.jpeg','.png'}:continue
  d=p.stem.rsplit('_',1)[-1].lower();rows.append(dict(path=A.norm(p),source='hrf',label='1' if d=='g' else '0' if d=='h' else 'other_disease',original_label=d,patient_id='',eye='',official_split='',sha256=A.sha(p)))
 assert len(rows)==45
 A.A.save_csv(OUT/'hrf_source_manifest.csv',rows,list(rows[0]));fp=OUT/'hrf_features.jsonl'
 new=read_features(fp) if fp.exists() else list(ThreadPoolExecutor(max_workers=4).map(feature,rows))
 fp.write_text(''.join(json.dumps(r)+'\n' for r in new),encoding='utf-8')
 historical=read_features(A.ROOT/'data/audit/session_20261110/eyeq_overlap/features.jsonl')
 for r in historical:r['path']=A.norm(r['path'])
 others=historical+read_features(OUT/'local_features.jsonl')
 for file in ['lag_features.jsonl','grape_features.jsonl','crfo_features.jsonl']:
  if (OUT/file).exists():others+=read_features(OUT/file)
 bysha={};exact=[]
 for r in others+new:
  if r['sha256'] in bysha:
   if r.get('source')=='hrf':exact.append(dict(path_a=r['path'],path_b=bysha[r['sha256']]['path'],decision='exact_file_sha256'))
  else:bysha[r['sha256']]=r
 A.A.save_csv(OUT/'hrf_exact_edges.csv',exact,['path_a','path_b','decision'])
 cp=OUT/'hrf_candidates.csv'
 if cp.exists():candidates=A.read(cp)
 else:
  targets=list(bysha.values());index=A.A.HashIndex([r['hashes'][0] for r in targets]);pairs={};newpaths={r['path'] for r in new}
  for r in new:
   for o,h in enumerate(r['hashes']):
    for j,d in index.query(h,6):
     target=targets[j]
     if target['sha256']==r['sha256'] or target['path']==r['path']:continue
     if target['path'] in newpaths and target['path']<r['path']:continue
     key=(r['path'],target['path'])
     if key not in pairs or d<pairs[key][0]:pairs[key]=(d,o)
  candidates=[dict(path_a=a,path_b=b,hash_distance=d,orientation=o) for (a,b),(d,o) in sorted(pairs.items())]
  A.A.save_csv(cp,candidates,['path_a','path_b','hash_distance','orientation'])
 print(json.dumps(dict(stage='hrf_candidates',images=len(new),pairs=len(candidates),reference_images=len(others),elapsed=time.time()-t)),flush=True)
 mp=OUT/'hrf_candidate_metrics.csv';done={}
 if mp.exists():done={(r['path_a'],r['path_b']):r for r in A.read(mp)}
 pending=[r for r in candidates if (r['path_a'],r['path_b']) not in done]
 with mp.open('a' if done else 'w',encoding='utf-8-sig',newline='') as f,ThreadPoolExecutor(max_workers=6) as pool:
  w=None
  for i,r in enumerate(pool.map(confirm,pending),1):
   if w is None:
    w=csv.DictWriter(f,fieldnames=list(r))
    if not done:w.writeheader()
   w.writerow(r);done[(r['path_a'],r['path_b'])]=r
   if i%300==0:f.flush();print(json.dumps(dict(stage='hrf_confirmation',processed=len(done),pairs=len(candidates),elapsed=time.time()-t)),flush=True)
 metrics=list(done.values());A.A.save_csv(OUT/'hrf_final_confirmed_edges.csv',[r for r in metrics if r['decision']=='confirmed_same_photo'],list(metrics[0]) if metrics else ['path_a','path_b','decision'])
 result=dict(completed=True,images=len(new),reference_images=len(others),pairs=len(candidates),exact_pairs=len(exact),confirmed_pairs=sum(r['decision']=='confirmed_same_photo' for r in metrics),ambiguous_pairs=sum(r['decision']=='ambiguous_exclude' for r in metrics),rejected_pairs=sum(r['decision']=='rejected_pHash_collision' for r in metrics),manifest_sha256=A.sha(OUT/'hrf_source_manifest.csv'),features_sha256=A.sha(fp),candidate_sha256=A.sha(cp),metrics_sha256=A.sha(mp))
 (OUT/'hrf_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
