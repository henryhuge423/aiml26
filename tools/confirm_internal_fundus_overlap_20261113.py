"""Confirm anatomy within the retina, excluding camera aperture features.

Re-adjudicates every raw geometric hit and every raw ambiguous candidate.
Same-photo edges require image agreement after a transform supported by
interior retinal points. Weak aperture correspondences never bridge splits.
"""
import collections,csv,hashlib,json,sys,time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
import audit_external_overlap_20261113 as A

OUT=A.OUT

def correlation(a,b):
    """Pixel correlation without a BLAS thread pool for tiny 2-row matrices."""
    a=np.ascontiguousarray(a,dtype=np.float32);b=np.ascontiguousarray(b,dtype=np.float32)
    ma,sa=cv2.meanStdDev(a);mb,sb=cv2.meanStdDev(b)
    denominator=float(sa[0,0]*sb[0,0])
    if denominator<=1e-12:return 0.
    product=cv2.mean(cv2.multiply(a,b,dtype=cv2.CV_32F))[0]
    return float((product-ma[0,0]*mb[0,0])/denominator)

@lru_cache(maxsize=6000)
def interior(path):
    gray=A.thumbnail(path);binary=(gray>10).astype(np.uint8)*255
    contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    mask=np.zeros_like(binary)
    if contours:cv2.drawContours(mask,[max(contours,key=cv2.contourArea)],-1,255,-1)
    mask=cv2.erode(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(45,45)))
    keys,desc=cv2.SIFT_create(nfeatures=1000,contrastThreshold=.015).detectAndCompute(gray,mask)
    return gray,mask,np.float32([k.pt for k in keys]),desc

@lru_cache(maxsize=1024)
def mirrored_interior(path):
    gray,mask,_,_=interior(path);gray=np.fliplr(gray).copy();mask=np.fliplr(mask).copy()
    keys,desc=cv2.SIFT_create(nfeatures=1000,contrastThreshold=.015).detectAndCompute(gray,mask)
    return gray,mask,np.float32([k.pt for k in keys]),desc

def confirm(r):
    ga,ma,pa,da=interior(r['path_a']);gb,mb,pb,db=interior(r['path_b'])
    best=dict(inner_matches=0,inner_inliers=0,inner_ratio=0.,inner_coverage_a=0.,inner_coverage_b=0.,aligned_correlation=0.,vessel_highpass_correlation=0.,inner_reprojection=999.,inner_mirrored=False)
    if da is not None and db is not None and min(len(da),len(db))>=2:
        for mirrored in [False,True]:
            if mirrored:ga,ma,pa,da=mirrored_interior(r['path_a'])
            if da is None or len(da)<2:continue
            match=cv2.FlannBasedMatcher(dict(algorithm=1,trees=3),dict(checks=64))
            aa=match.knnMatch(da,db,k=2);bb=match.knnMatch(db,da,k=2)
            reverse={(m.queryIdx,m.trainIdx) for pair in bb if len(pair)==2 for m,n in [pair] if m.distance<.72*n.distance}
            good=[m for pair in aa if len(pair)==2 for m,n in [pair] if m.distance<.72*n.distance and (m.trainIdx,m.queryIdx) in reverse]
            if len(good)<4:continue
            x=np.float32([pa[m.queryIdx] for m in good]);y=np.float32([pb[m.trainIdx] for m in good])
            h,keep=cv2.findHomography(x,y,cv2.RANSAC,2.5,maxIters=1500,confidence=.999)
            if h is None or keep is None:continue
            keep=keep.ravel().astype(bool);n=int(keep.sum())
            if n<4:continue
            ca=float(cv2.contourArea(cv2.convexHull(x[keep])))/(384*384);cb=float(cv2.contourArea(cv2.convexHull(y[keep])))/(384*384)
            warped=cv2.warpPerspective(ga,h,(384,384));wm=cv2.warpPerspective(ma,h,(384,384),flags=cv2.INTER_NEAREST)
            common=(wm>0)&(mb>0);corr=0.;highcorr=0.
            if common.sum()>384*384*.10:
                av=warped[common].astype(float);bv=gb[common].astype(float)
                if av.std()>1 and bv.std()>1:corr=correlation(av,bv)
                higha=warped.astype(np.float32)-cv2.GaussianBlur(warped.astype(np.float32),(0,0),2.)
                highb=gb.astype(np.float32)-cv2.GaussianBlur(gb.astype(np.float32),(0,0),2.)
                ah=higha[common];bh=highb[common]
                if ah.std()>.2 and bh.std()>.2:highcorr=correlation(ah,bh)
            projected=cv2.perspectiveTransform(x[keep,None,:],h)[:,0,:]
            reproj=float(np.median(np.linalg.norm(projected-y[keep],axis=1)))
            metric=dict(inner_matches=len(good),inner_inliers=n,inner_ratio=n/len(good),inner_coverage_a=ca,inner_coverage_b=cb,aligned_correlation=corr,vessel_highpass_correlation=highcorr,inner_reprojection=reproj,inner_mirrored=mirrored)
            if corr>best['aligned_correlation'] or (corr==best['aligned_correlation'] and n>best['inner_inliers']):best=metric
    # Full-image agreement alone can confirm low-feature JPEG re-encoding,
    # provided no geometry contrary to its aligned photometric identity exists.
    original=A.thumbnail(r['path_a']);aligned=A.A.orient(original,int(r['orientation']));other=A.thumbnail(r['path_b'])
    normcorr=0.
    if aligned.std()>1 and other.std()>1:normcorr=correlation(aligned,other)
    exact_normalized=np.array_equal(aligned,other)
    full_photo=normcorr>=.998 and float(r.get('ssim',0))>=.985 and int(r['hash_distance'])<=2
    strong=best['inner_inliers']>=25 and best['inner_ratio']>=.65 and min(best['inner_coverage_a'],best['inner_coverage_b'])>=.04 and best['aligned_correlation']>=.93 and best['vessel_highpass_correlation']>=.70 and best['inner_reprojection']<=1.5
    ambiguous=not strong and not full_photo and not exact_normalized and best['inner_inliers']>=8 and best['inner_ratio']>=.5 and best['aligned_correlation']>=.80 and min(best['inner_coverage_a'],best['inner_coverage_b'])>=.015
    decision='confirmed_same_photo' if strong or full_photo or exact_normalized else 'ambiguous_exclude' if ambiguous else 'rejected_pHash_collision'
    return dict(r,raw_decision=r['decision'],**{k:round(v,6) if isinstance(v,float) else v for k,v in best.items()},full_image_correlation=normcorr,final_decision=decision)

def main():
    cv2.setNumThreads(1);A.load_historical_gray_cache();t=time.time()
    raw=A.read(OUT/'local_candidate_metrics.csv')
    pending=[r for r in raw if r['decision']!='rejected_pHash_collision' or int(r['hash_distance'])<=2 or float(r['ssim'])>=.95]
    dest=OUT/'local_internal_candidate_metrics.csv';completed={}
    if dest.exists():completed={(r['path_a'],r['path_b']):r for r in A.read(dest)}
    completed={k:v for k,v in completed.items() if 'vessel_highpass_correlation' in v}
    pending=[r for r in pending if (r['path_a'],r['path_b']) not in completed]
    with dest.open('a' if completed else 'w',encoding='utf-8-sig',newline='') as f,ThreadPoolExecutor(max_workers=6) as pool:
        writer=None
        for i,r in enumerate(pool.map(confirm,pending),1):
            if writer is None:
                writer=csv.DictWriter(f,fieldnames=list(r))
                if not completed:writer.writeheader()
            writer.writerow(r);completed[(r['path_a'],r['path_b'])]=r
            if i%500==0:f.flush();print(json.dumps(dict(stage='internal_confirmation',processed=len(completed),pending_total=len(pending),elapsed=time.time()-t,decisions=dict(collections.Counter(r['final_decision'] for r in completed.values())))),flush=True)
    total_candidates=len(A.read(OUT/'local_candidates.csv'))
    if len(raw)!=total_candidates:
        print(json.dumps(dict(stage='internal_snapshot_completed',raw_snapshot_pairs=len(raw),total_candidate_pairs=total_candidates,inner_pairs=len(completed),cohort_finalized=False)),flush=True)
        return
    final=[]
    for r in raw:
        z=completed.get((r['path_a'],r['path_b']))
        if z:r['decision']=z['final_decision']
        final.append(r)
    A.A.save_csv(OUT/'local_final_candidate_metrics.csv',final,list(final[0]))
    confirmed=[r for r in final if r['decision']=='confirmed_same_photo'];A.A.save_csv(OUT/'local_final_confirmed_edges.csv',confirmed,list(final[0]))
    A.START=time.time()
    with (OUT/'local_features.jsonl').open() as f:new=[json.loads(line) for line in f]
    with (A.ROOT/'data/audit/session_20261110/eyeq_overlap/features.jsonl').open() as f:historical=[dict(json.loads(line)) for line in f]
    for r in historical:r['path']=A.norm(r['path'])
    exact=A.read(OUT/'local_exact_edges.csv')
    A.finalize(new,historical,exact+[dict(path_a=r['path_a'],path_b=r['path_b'],decision=r['decision']) for r in confirmed],final)
    summary=dict(raw_pairs=len(raw),inner_pairs=len(completed),final_decisions=dict(collections.Counter(r['decision'] for r in final)),raw_confirmed=sum(r['decision']=='confirmed_same_photo' for r in A.read(OUT/'local_candidate_metrics.csv')),method='Retinal interior mask with 45-pixel aperture erosion; mutual SIFT correspondences; homography with >=25 inliers, >=0.65 ratio, >=0.04 area in both images, >=0.93 correlation after alignment and >=0.70 high-frequency vessel-detail correlation; full-image near identity also confirms; weaker potential same-eye cases excluded explicitly',inputs={str(p.relative_to(A.ROOT)):A.sha(p) for p in [dest,OUT/'local_final_candidate_metrics.csv',OUT/'local_final_confirmed_edges.csv',Path(__file__)]})
    (OUT/'local_internal_confirmation_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
