"""Empirical verification of a cheap rejection gate before local feature work."""
import csv,json,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2,numpy as np
import audit_external_overlap_20261113 as A
import confirm_internal_fundus_overlap_20261113 as S

OUT=A.OUT
def measure(r):
    a=A.A.orient(A.thumbnail(A.norm(r['path_a'])),int(r['orientation']));b=A.thumbnail(A.norm(r['path_b']))
    ma,sa=cv2.meanStdDev(a);mb,sb=cv2.meanStdDev(b);cov=cv2.mean(cv2.multiply(a,b,dtype=cv2.CV_32F))[0]-ma[0,0]*mb[0,0]
    va=sa[0,0]**2;vb=sb[0,0]**2
    ssim=((2*ma[0,0]*mb[0,0]+6.5025)*(2*cov+58.5225))/((ma[0,0]**2+mb[0,0]**2+6.5025)*(va+vb+58.5225))
    corr=S.correlation(a,b)
    return dict(r,ssim=float(ssim),full_image_correlation=corr,low_gate_triggered=corr<.60 and ssim<.60)

def main():
    cv2.setNumThreads(1);A.load_historical_gray_cache();t=time.time()
    legacy=[r for r in A.read(A.ROOT/'data/external/airogs_candidate_5k/positive_secondary_distance_le4.csv') if r['confirmed_strict_or_exact']=='True']
    pairs=[dict(path_a=A.norm(r['image']),path_b=A.norm(r['other']),orientation=int(r['orientation']),hash_distance=int(r['distance']),cohort='147_legacy_confirmed',expected='same_photo') for r in legacy]
    pairs+=[dict(path_a=r['path_a'],path_b=r['path_b'],orientation=int(r['orientation']),hash_distance=int(r['hash_distance']),cohort='51_matcher_controls',expected=r['expected']) for r in A.read(OUT/'local_matcher_validation.csv')]
    strict=[r for r in A.read(OUT/'local_internal_candidate_metrics.csv') if r['final_decision']=='confirmed_same_photo']
    pairs+=[dict(path_a=r['path_a'],path_b=r['path_b'],orientation=int(r['orientation']),hash_distance=int(r['hash_distance']),cohort='current_strict_confirmed',expected='same_photo') for r in strict]
    with ThreadPoolExecutor(max_workers=6) as pool:result=list(pool.map(measure,pairs))
    positive_errors=[r for r in result if r['expected']=='same_photo' and r['low_gate_triggered']]
    negatives=[];candidates=A.read(OUT/'grape_candidates.csv')
    for r in candidates[::400]:
        z=measure(dict(r,cohort='sampled_low_gate_grape',expected='different_photo'))
        if z['low_gate_triggered']:negatives.append(z)
        if len(negatives)==100:break
    if len(negatives)<100:
        for r in candidates[17::200]:
            z=measure(dict(r,cohort='sampled_low_gate_grape',expected='different_photo'))
            if z['low_gate_triggered'] and (z['path_a'],z['path_b']) not in {(n['path_a'],n['path_b']) for n in negatives}:negatives.append(z)
            if len(negatives)==100:break
    with ThreadPoolExecutor(max_workers=6) as pool:verified=list(pool.map(S.confirm,[dict(r,decision='candidate') for r in negatives]))
    negative_errors=[r for r in verified if r['final_decision']!='rejected_pHash_collision']
    for r,z in zip(negatives,verified):r['full_strict_decision']=z['final_decision']
    combined=result+negatives;fields=list(combined[0])+['full_strict_decision'];A.A.save_csv(OUT/'low_correlation_gate_validation.csv',combined,fields)
    summary=dict(completed=True,passed=not positive_errors and len(negatives)==100 and not negative_errors,legacy_positives=len(legacy),matcher_controls=51,current_strict_positives=len(strict),positive_gate_errors=len(positive_errors),low_gate_negative_controls=len(negatives),negative_full_strict_errors=len(negative_errors),elapsed_seconds=time.time()-t,gate='Aligned normalized ROI correlation <0.60 AND global SSIM <0.60',scope='The original bounded ROI/image-version audit; no claim that this detects every arbitrary crop',inputs={str(p.relative_to(A.ROOT)):A.sha(p) for p in [OUT/'local_matcher_validation.csv',A.ROOT/'data/external/airogs_candidate_5k/positive_secondary_distance_le4.csv',OUT/'grape_candidates.csv',OUT/'low_correlation_gate_validation.csv',Path(__file__)]})
    (OUT/'low_correlation_gate_validation.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
