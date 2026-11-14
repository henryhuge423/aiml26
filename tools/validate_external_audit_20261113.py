"""Empirical same-photo/cross-photo matcher checks and review contact sheets."""
import collections,csv,hashlib,json,sys,time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import audit_external_overlap_20261113 as A

OUT=A.OUT

def matcher_validation():
    cv2.setNumThreads(1)
    rows=A.read(OUT/'local_input_images.csv');selected=[]
    for ds in ['beh','keh','papila','fives','fundus_avseg','mshf']:
        selected.append(next(r for r in rows if r['source']==ds and r['label']=='1'))
    d=OUT/'matcher_validation_images';d.mkdir(exist_ok=True)
    true=[]
    originals=[]
    for i,r in enumerate(selected):
        g=A.thumbnail(r['path']);orig=d/f'{i}_{r["source"]}_original.png';Image.fromarray(g).save(orig);originals.append(orig)
        transforms={'jpeg_quality_40':g,'rotate_90':np.rot90(g),'mirror':np.fliplr(g),'crop_5pct_each_side':cv2.resize(g[19:-19,19:-19],(384,384))}
        for name,t in transforms.items():
            p=d/f'{i}_{name}.png'
            if name=='jpeg_quality_40':p=p.with_suffix('.jpg');Image.fromarray(t).save(p,quality=40)
            else:Image.fromarray(t).save(p)
            aa=A.thumbnail(str(orig));bb=A.thumbnail(str(p));dist=[(A.A.phash(A.A.orient(aa,o))^A.A.phash(bb)).bit_count() for o in range(8)];o=min(range(8),key=lambda x:dist[x])
            true.append(dict(path_a=str(orig),path_b=str(p),hash_distance=dist[o],orientation=o,expected='same_photo',transform=name,source=r['source']))
    known=[]
    legacy=A.ROOT/'data/external/airogs_candidate_5k/positive_secondary_distance_le4.csv'
    confirmed=[r for r in A.read(legacy) if r['confirmed_strict_or_exact']=='True']
    confirmed.sort(key=lambda r:float(r.get('ssim',1)))
    known=[dict(path_a=A.norm(r['image']),path_b=A.norm(r['other']),hash_distance=int(r['distance']),orientation=int(r['orientation']),expected='same_photo',transform='legacy_strict_or_exact_real_pair',source='real') for r in confirmed[:12]]
    different=[]
    for i in range(len(originals)):
        for j in range(i+1,len(originals)):
            different.append(dict(path_a=str(originals[i]),path_b=str(originals[j]),hash_distance=64,orientation=0,expected='different_photo',transform='different_sources',source='real'))
    pairs=true+known+different;original_flann=cv2.FlannBasedMatcher
    def run(use_bf):
        if use_bf:cv2.FlannBasedMatcher=lambda *args,**kwargs:cv2.BFMatcher(cv2.NORM_L2)
        else:cv2.FlannBasedMatcher=original_flann
        result=[];t=time.time()
        for r in pairs:result.append(A.compare_pair(r))
        return result,time.time()-t
    bf,tb=run(True);flann,tf=run(False);cv2.FlannBasedMatcher=original_flann
    output=[]
    for r,b,f in zip(pairs,bf,flann):
        output.append(dict(r,bf_decision=b['decision'],flann_decision=f['decision'],bf_inliers=b['sift_inliers'],flann_inliers=f['sift_inliers'],candidate_retrieved=int(r['hash_distance'])<=6,flann_expected_agreement=(f['decision']=='confirmed_same_photo')==(r['expected']=='same_photo')))
    A.A.save_csv(OUT/'local_matcher_validation.csv',output,list(output[0]))
    summary=dict(pairs=len(output),synthetic_same_photo=len(true),real_same_photo=len(known),known_different=len(different),bf_seconds=tb,flann_seconds=tf,bf_flann_decision_agreement=sum(r['bf_decision']==r['flann_decision'] for r in output),flann_expected_agreement=sum(r['flann_expected_agreement'] for r in output),synthetic_candidate_hits=sum(r['candidate_retrieved'] for r in output[:len(true)]),missed_candidates=[{k:r[k] for k in ['source','transform','hash_distance']} for r in output[:len(true)] if not r['candidate_retrieved']])
    (OUT/'local_matcher_validation.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

def sheet(pairs,name):
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
    width=1000;height=260*len(pairs);canvas=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(canvas)
    for i,(pa,pb,label) in enumerate(pairs):
        y=i*260
        for x,path in [(0,pa),(500,pb)]:
            with A.open_image(path) as im:
                im=im.convert('RGB');im.thumbnail((490,225));canvas.paste(im,(x+(490-im.width)//2,y+(225-im.height)//2))
        draw.text((5,y+228),label[:125],font=font,fill='black')
    canvas.save(OUT/name)

def contact_sheets():
    metrics=A.read(OUT/'local_internal_candidate_metrics.csv')
    confirmed=[r for r in metrics if r['final_decision']=='confirmed_same_photo']
    confirmed.sort(key=lambda r:(int(r['inner_inliers']),float(r['inner_ratio'])))
    geometric=[r for r in confirmed if int(r['inner_inliers'])>=25 and float(r['aligned_correlation'])>=.93 and float(r['vessel_highpass_correlation'])>=.70]
    near_identity=[r for r in confirmed if r not in geometric]
    confirmed=geometric[:8]+near_identity[:4]+geometric[8:]+near_identity[4:]
    selected=[];sources=collections.Counter()
    for r in confirmed:
        k=(Path(r['path_a']).parent.parent.name,Path(r['path_b']).parent.parent.name)
        if sources[k]>=2:continue
        sources[k]+=1;selected.append(r)
        if len(selected)==12:break
    if len(selected)<12:selected+=confirmed[:12-len(selected)]
    sheet([(r['path_a'],r['path_b'],f"{i+1}: pHash {r['hash_distance']}, interior inliers {r['inner_inliers']}, ratio {r['inner_ratio']}, photo/vessel corr {r['aligned_correlation']}/{r['vessel_highpass_correlation']}") for i,r in enumerate(selected)],'local_confirmed_pairs_contact_sheet.jpg')
    A.A.save_csv(OUT/'local_confirmed_pairs_contact_sheet.csv',selected,list(selected[0]))
    decisions=A.read(OUT/'local_decisions.csv');eligible=[r for r in decisions if r['decision']=='eligible_new_training_candidate'];pairs=[]
    selected_eligible=[]
    for ds in sorted({r['source'] for r in eligible}):
        for lab in ['1','0']:
            rs=[r for r in eligible if r['source']==ds and r['label']==lab][:2]
            for r in rs:selected_eligible.append(r)
            if rs:pairs.append((rs[0]['training_path'],rs[-1]['training_path'],f"{ds}, original disease label {lab}: {Path(rs[0]['path']).name} / {Path(rs[-1]['path']).name}"))
    sheet(pairs,'local_eligible_sources_contact_sheet.jpg');A.A.save_csv(OUT/'local_eligible_sources_contact_sheet.csv',selected_eligible,list(selected_eligible[0]))
    print(json.dumps(dict(confirmed_pairs=len(selected),eligible_source_rows=len(selected_eligible))),flush=True)

if __name__=='__main__':
    if '--validation' in sys.argv:matcher_validation()
    if '--sheets' in sys.argv:contact_sheets()

