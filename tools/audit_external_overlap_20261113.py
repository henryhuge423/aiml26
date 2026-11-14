"""Image association audit of local external glaucoma photographs.

Does not score models, modify frozen splits, or declare unknown patients distinct.
Every radius-six candidate is inspected with local geometric features; a pHash
collision alone is never grounds to remove a photograph.
"""
from __future__ import annotations
import collections,csv,hashlib,json,os,sys,time,zipfile
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/audit/data_expansion_20261113'
CACHE_IMAGES=None
CACHE_PATHS={}
sys.path.insert(0,str(ROOT/'tools'))
import audit_eyeq_overlap as A

def read(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return A.digest(p)
def save(name,rows):
    if rows:A.save_csv(OUT/name,rows,list(rows[0]))
def norm(p):return os.path.abspath(os.path.normpath(str(p)))
def status(stage,**v):
    r=dict(stage=stage,elapsed_seconds=round(time.time()-START,1),**v)
    (OUT/'local_status.json').write_text(json.dumps(r,indent=2))
    print(json.dumps(r),flush=True)

def open_image(path):
    # MSHF glaucoma files use .jpg names but contain LZW TIFF. The installed
    # Pillow/libtiff combination terminates its process on these original TIFFs.
    # OpenCV fully decodes the original bytes; filename extension is irrelevant.
    with Path(path).open('rb') as f:header=f.read(4)
    if header in {b'II*\x00',b'MM\x00*'}:
        array=cv2.imdecode(np.frombuffer(Path(path).read_bytes(),np.uint8),cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
        if array is None:raise ValueError(f'OpenCV TIFF decode failed: {path}')
        return Image.fromarray(cv2.cvtColor(array,cv2.COLOR_BGR2RGB))
    return Image.open(path)

@lru_cache(maxsize=192)
def thumbnail(path):
    if path in CACHE_PATHS:return np.array(CACHE_IMAGES[CACHE_PATHS[path]],copy=True)
    with Path(path).open('rb') as f:header=f.read(4)
    if header not in {b'II*\x00',b'MM\x00*'}:return A.thumbnail(path)
    with open_image(path) as image:
        image.thumbnail((768,768));gray=np.asarray(image.convert('L'))
    ys,xs=np.where(gray>8)
    if len(xs) and xs.max()-xs.min()+1>=gray.shape[1]*.35 and ys.max()-ys.min()+1>=gray.shape[0]*.35:
        gray=gray[ys.min():ys.max()+1,xs.min():xs.max()+1]
    return cv2.resize(gray,(384,384),interpolation=cv2.INTER_AREA)

def load_historical_gray_cache():
    global CACHE_IMAGES,CACHE_PATHS
    cache=ROOT/'data/audit/session_20261110/eyeq_overlap/confirmation_cache'
    binding=json.loads((cache/'binding.json').read_text())
    evidence=json.loads((cache.parent/'cached_confirmation.json').read_text())
    assert sha(cache/'binding.json')==evidence['cache_binding_sha256']
    expected_bytes=1
    for dimension in binding['shape']:expected_bytes*=dimension
    assert (cache/'gray.dat').stat().st_size==expected_bytes
    CACHE_IMAGES=np.memmap(cache/'gray.dat',mode='r',dtype='uint8',shape=tuple(binding['shape']))
    CACHE_PATHS={norm(p):i for i,p in enumerate(binding['paths'])}

def collect():
    rows=[]
    for ds in ['rim_one_dl','beh','keh']:
        zp=ROOT/'data/external_glaucoma'/ds/('processed_data.zip' if ds=='keh' else 'images.zip')
        dest=ROOT/'data/external_glaucoma'/ds/'originals_20261113';dest.mkdir(exist_ok=True)
        with zipfile.ZipFile(zp) as z:
            for m in z.infolist():
                if m.is_dir() or Path(m.filename).suffix.lower() not in {'.jpg','.jpeg','.png'} or '__MACOSX' in m.filename:continue
                if ds=='rim_one_dl' and '/partitioned_randomly/' not in m.filename:continue
                parts=Path(m.filename).parts
                label='1' if any(x in parts for x in ['glaucoma','early_glaucoma','advanced_glaucoma']) else '0'
                name='_'.join(parts[-2:]);p=(dest/name).resolve()
                assert p.is_relative_to(dest.resolve())
                raw=z.read(m)
                if not p.exists():p.write_bytes(raw)
                assert hashlib.sha256(raw).hexdigest()==sha(p)
                role='test' if '/test_set/' in m.filename else 'train' if '/training_set/' in m.filename else ''
                rows.append(dict(path=str(p),source=ds,label=label,original_label=parts[-2],patient_id='',eye='',official_split=role,sha256=sha(p)))
    for ds,p,lab in [('papila',ROOT/'data/external/papila_candidate/candidate_manifest.csv','label'),
        ('chaksu',ROOT/'data/audit/airogs_labels_20261111/chaksu_train_labels/screened_manifest.csv','label')]:
        for r in read(p):
            label=r[lab] if ds=='papila' else ('0' if r[lab]=='NORMAL' else 'suspect')
            if ds=='papila' and label=='2':label='suspect'
            rows.append(dict(path=norm(r['path']),source=ds,label=label,original_label=r[lab],patient_id=r.get('patient_id',''),eye=r.get('eye',''),official_split='train' if ds=='chaksu' else '',sha256=r['sha256']))
    for p in sorted((ROOT/'data/external/fives/extracted').glob('**/Original/*')):
        if p.suffix.lower() not in {'.jpg','.jpeg','.png'}:continue
        code=p.stem.rsplit('_',1)[-1]
        rows.append(dict(path=norm(p),source='fives',label='1' if code=='G' else '0' if code=='N' else 'other_disease',original_label=code,patient_id='',eye='',official_split='test' if '/test/' in p.as_posix() else 'train',sha256=sha(p)))
    labs={r['imageID']:r['binaryLabels'] for r in read(ROOT/'data/external_glaucoma/g1020_mirror/G1020.csv')}
    for p in sorted((ROOT/'data/external_glaucoma/g1020_mirror/images').glob('*')):
        rows.append(dict(path=norm(p),source='g1020_partial',label=labs[p.name],original_label=labs[p.name],patient_id='',eye='',official_split='',sha256=sha(p)))
    # The original AIROGS acquisition is heavily overlapping. Audit every RG
    # photograph, without spending positive-scarcity work on 4,837 more NRGs.
    for r in read(ROOT/'data/external/airogs_candidate_5k/candidate_manifest.csv'):
        if r['class']!='RG':continue
        rows.append(dict(path=norm(r['path']),source='airogs_original_rg',label='1',original_label='RG',patient_id='',eye='',official_split='train',sha256=r['sha256']))
    p=ROOT/'data/external_glaucoma/fundus_avseg/extracted/Fundus-AVSeg'
    metadata={r['image name']:r for r in pd.read_excel(p/'metadata.xlsx').to_dict('records')}
    official_test=set((p/'testing.txt').read_text().splitlines())
    for f in sorted((p/'images').glob('*.png')):
        m=metadata[f.name];d=m['disease type'];label='1' if d=='Glaucoma' else '0' if d=='Normal' else 'other_disease'
        rows.append(dict(path=norm(f),source='fundus_avseg',label=label,original_label=d,patient_id='',eye=m['eye id'],official_split='test' if f.name in official_test else 'train',sha256=sha(f)))
    p=ROOT/'data/external_glaucoma/les_av/extracted/LES-AV'
    for m in pd.read_excel(p/'Data.xlsx').to_dict('records'):
        f=p/'images'/f"{m['Identifier']}.png"
        rows.append(dict(path=norm(f),source='les_av',label='0' if m['Diagnosis5']=='normal' else '1',original_label=m['Diagnosis5'],patient_id='',eye=m['eye'],official_split='',sha256=sha(f)))
    p=ROOT/'data/external_quality/mshf/MSHF dataset 2.0/Original'
    for f in sorted(p.glob('**/*')):
        if f.suffix.lower() not in {'.jpg','.jpeg','.png'}:continue
        d=f.parent.name;label='1' if d=='Glaucoma' else '0' if d=='Healthy' else 'unlabeled_glaucoma'
        rows.append(dict(path=norm(f),source='mshf',label=label,original_label=d,patient_id='',eye='',official_split='',sha256=sha(f)))
    save('local_input_images.csv',rows)
    return rows

@lru_cache(maxsize=6000)
def desc(path):
    im=thumbnail(path)
    k,d=cv2.SIFT_create(nfeatures=650,contrastThreshold=.025).detectAndCompute(im,None)
    xy=np.float32([x.pt for x in k])
    return im,xy,d

@lru_cache(maxsize=1024)
def desc_mirror(path):
    im=desc(path)[0]
    k,d=cv2.SIFT_create(nfeatures=650,contrastThreshold=.025).detectAndCompute(np.fliplr(im).copy(),None)
    return np.float32([x.pt for x in k]),d

def compare_pair(r):
    a,ka,da=desc(r['path_a']);b,kb,db=desc(r['path_b'])
    ma,va=float(a.mean()),float(a.var());mb,vb=float(b.mean()),float(b.var())
    aa=A.orient(a,int(r['orientation']))
    cov=float(cv2.mean(cv2.multiply(aa,b,dtype=cv2.CV_32F))[0])-ma*mb
    ssim=((2*ma*mb+6.5025)*(2*cov+58.5225))/((ma*ma+mb*mb+6.5025)*(va+vb+58.5225))
    matches=[];inliers=0;ratio=coverage_a=coverage_b=0.;reprojection=999.
    if da is not None and db is not None and min(len(da),len(db))>=4:
        # Check both original and mirrored photographs. SIFT handles arbitrary
        # in-plane rotation and scale; reflected images need a second descriptor.
        for mirrored in [False,True]:
            if mirrored:
                points,des=desc_mirror(r['path_a'])
            else:points,des=ka,da
            if des is None:continue
            pairs=cv2.FlannBasedMatcher(dict(algorithm=1,trees=2),dict(checks=48)).knnMatch(des,db,k=2)
            good=[m for pair in pairs if len(pair)==2 for m,n in [pair] if m.distance<.70*n.distance]
            if len(good)<4:continue
            pa=np.float32([points[m.queryIdx] for m in good]);pb=np.float32([kb[m.trainIdx] for m in good])
            h,mask=cv2.findHomography(pa,pb,cv2.RANSAC,3.,maxIters=1000,confidence=.995)
            if h is None or mask is None:continue
            kept=mask.ravel().astype(bool);n=int(kept.sum())
            if n<=inliers:continue
            inliers=n;matches=good;ratio=n/len(good)
            if n>=3:
                coverage_a=float(cv2.contourArea(cv2.convexHull(pa[kept])))/(384*384)
                coverage_b=float(cv2.contourArea(cv2.convexHull(pb[kept])))/(384*384)
                pred=cv2.perspectiveTransform(pa[kept,None,:],h)[:,0,:]
                reprojection=float(np.median(np.linalg.norm(pred-pb[kept],axis=1)))
    confirmed=inliers>=14 and ratio>=.5 and min(coverage_a,coverage_b)>=.015 and reprojection<=2.
    # Exact normalized thumbnails are strong same-photo evidence, independently
    # of the keypoint count in smooth or very small photographs.
    thumbnail_equal=bool(np.array_equal(aa,b))
    ambiguous=not confirmed and not thumbnail_equal and ((inliers>=8 and ratio>=.45 and min(coverage_a,coverage_b)>=.01) or ssim>=.97)
    decision='confirmed_same_photo' if confirmed or thumbnail_equal else 'ambiguous_exclude' if ambiguous else 'rejected_pHash_collision'
    return dict(r,ssim=round(ssim,6),sift_matches=len(matches),sift_inliers=inliers,sift_ratio=round(ratio,5),coverage_a=round(coverage_a,5),coverage_b=round(coverage_b,5),reprojection_median=round(reprojection,5),thumbnail_equal=thumbnail_equal,decision=decision)

def main():
    OUT.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(1);cv2.setRNGSeed(20261113)
    rows=read(OUT/'local_input_images.csv') if '--reuse-input' in sys.argv else collect()
    status('collected',images=len(rows),sources=dict(collections.Counter(r['source'] for r in rows)))
    source_features=ROOT/'data/audit/session_20261110/eyeq_overlap/features.jsonl'
    status('loading_historical_features')
    features={}
    with source_features.open(encoding='utf-8') as source_stream:
        for line in source_stream:
            r=json.loads(line);r['path']=norm(r['path']);features[r['path']]=r
    historical=list(features.values())
    status('historical_features_loaded',images=len(historical))
    def feat(r):
        p=Path(r['path']);st=p.stat()
        with open_image(p) as im:
            im.load();rgb=im if im.mode=='RGB' else im.convert('RGB');h=hashlib.sha256();wh=im.size
            for y in range(0,im.height,128):h.update(rgb.crop((0,y,im.width,min(y+128,im.height))).tobytes())
            pixel=h.hexdigest()
        assert sha(p)==r['sha256'],p
        g=thumbnail(str(p));return dict(r,bytes=st.st_size,mtime_ns=st.st_mtime_ns,width=wh[0],height=wh[1],pixel_sha256=pixel,hashes=[A.phash(A.orient(g,o)) for o in range(8)])
    fp=OUT/'local_features.jsonl'
    cached={}
    if fp.exists():
        for line in fp.read_text().splitlines():
            try:r=json.loads(line);cached[r['path']]=r
            except json.JSONDecodeError:pass
    requested={r['path']:r for r in rows}
    new=[r for p,r in cached.items() if p in requested]
    pending=[r for r in rows if r['path'] not in cached]
    status('new_feature_refresh',cached=len(cached),pending=len(pending))
    if pending:
        with fp.open('a' if cached else 'w',encoding='utf-8') as stream,ThreadPoolExecutor(max_workers=1) as pool:
            for i,r in enumerate(pool.map(feat,pending),1):
                new.append(r);stream.write(json.dumps(r)+'\n')
                if i%500==0:stream.flush();status('features',processed=i,images=len(rows))
    for r in new:features[r['path']]=r
    status('features_complete',new=len(new),historical=len(historical))
    allrows=historical+new
    by_sha=collections.defaultdict(list)
    for r in allrows:by_sha[r['sha256']].append(r['path'])
    exact=[]
    newpaths={r['path'] for r in new}
    for paths in by_sha.values():
        if len(paths)>1 and newpaths.intersection(paths):
            first=paths[0]
            for p in paths[1:]:exact.append(dict(path_a=first,path_b=p,decision='exact_file_sha256'))
    bypixel=collections.defaultdict(list)
    for r in new:bypixel[(r['width'],r['height'],r['pixel_sha256'])].append(r['path'])
    for paths in bypixel.values():
        for p in paths[1:]:exact.append(dict(path_a=paths[0],path_b=p,decision='exact_decoded_rgb_pixels'))
    save('local_exact_edges.csv',exact)
    candidates_path=OUT/'local_candidates.csv'
    candidate_binding=OUT/'local_candidate_input.json'
    binding=dict(feature_sha256=sha(fp),historical_features_sha256=sha(source_features))
    if candidates_path.exists() and candidate_binding.exists() and json.loads(candidate_binding.read_text())==binding:candidates=read(candidates_path)
    else:
        representative={}
        for r in allrows:representative.setdefault(r['sha256'],r)
        targets=list(representative.values());index=A.HashIndex([r['hashes'][0] for r in targets]);pairs={}
        for i,r in enumerate(new,1):
            for o,h in enumerate(r['hashes']):
                for j,d in index.query(h,6):
                    t=targets[j]
                    if t['path']==r['path'] or t['sha256']==r['sha256']:continue
                    if t['path'] in newpaths and t['path']<r['path']:continue
                    key=(r['path'],t['path'])
                    if key not in pairs or d<pairs[key][0]:pairs[key]=(d,o)
            if i%500==0:status('candidates',processed=i,images=len(new),pairs=len(pairs))
        candidates=[dict(path_a=a,path_b=b,hash_distance=d,orientation=o) for (a,b),(d,o) in sorted(pairs.items())]
        save('local_candidates.csv',candidates)
        candidate_binding.write_text(json.dumps(binding,indent=2))
    status('candidate_generation_complete',pairs=len(candidates))
    load_historical_gray_cache()
    metrics_path=OUT/'local_candidate_metrics.csv';completed={}
    if metrics_path.exists():completed={(r['path_a'],r['path_b']):r for r in read(metrics_path)}
    candidate_keys={(r['path_a'],r['path_b']) for r in candidates}
    completed={k:v for k,v in completed.items() if k in candidate_keys}
    pending=[r for r in candidates if (r['path_a'],r['path_b']) not in completed]
    fields=list(candidates[0])+['ssim','sift_matches','sift_inliers','sift_ratio','coverage_a','coverage_b','reprojection_median','thumbnail_equal','decision']
    with metrics_path.open('a' if completed else 'w',encoding='utf-8-sig',newline='') as f,ThreadPoolExecutor(max_workers=6) as pool:
        w=csv.DictWriter(f,fieldnames=fields)
        if not completed:w.writeheader()
        for i,r in enumerate(pool.map(compare_pair,pending),1):
            w.writerow(r);completed[(r['path_a'],r['path_b'])]=r
            if i%500==0:f.flush();status('confirmation',processed=len(completed),pairs=len(candidates),decisions=dict(collections.Counter(x['decision'] for x in completed.values())))
    metrics=list(completed.values());save('local_confirmed_edges.csv',[r for r in metrics if r['decision']=='confirmed_same_photo'])
    # Decisions are definitive for the declared screen: uncertain high-evidence
    # pairs are excluded now, rather than postponed for a future release.
    edges=exact+[(dict(path_a=r['path_a'],path_b=r['path_b'],decision=r['decision'])) for r in metrics if r['decision']=='confirmed_same_photo']
    finalize(new,historical,edges,metrics)

def finalize(new,historical,edges,metrics):
    parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    for e in edges:union(e['path_a'],e['path_b'])
    legacy_source=ROOT/'data/external/airogs_candidate_5k/positive_secondary_distance_le4.csv'
    legacy_confirmed=[]
    for e in read(legacy_source):
        if e.get('confirmed_strict_or_exact')=='True':
            legacy_confirmed.append(dict(path_a=norm(e['image']),path_b=norm(e['other']),decision='legacy_strict_or_exact'))
            union(norm(e['image']),norm(e['other']))
    save('local_legacy_confirmed_edges.csv',legacy_confirmed)
    prior=read(ROOT/'data/audit/deep/release_inputs/confirmed_edges_v1.csv')
    for e in prior:union(norm(ROOT/'data'/e['path_a']),norm(ROOT/'data'/e['path_b']))
    # Known paired-eye PAPILA identity is distinct from exact photo identity.
    photo_group={r['path']:find(r['path']) for r in new}
    papila={}
    for r in new:
        if r['patient_id']:
            key=(r['source'],r['patient_id'])
            if key in papila:union(r['path'],papila[key])
            else:papila[key]=r['path']
    active=read(ROOT/'data/prepared/formal_v3/manifest.csv')
    by_active_group={}
    for r in active:
        p=norm(ROOT/'data'/r['path']);key=r['related_group']
        if key in by_active_group:union(p,by_active_group[key])
        else:by_active_group[key]=p
    eyeq=read(ROOT/'data/audit/session_20261110/eyeq_overlap/full_components.csv')
    by_eyeq_group={}
    for r in eyeq:
        p=norm(r['path']);key=r['component']
        if key in by_eyeq_group:union(p,by_eyeq_group[key])
        else:by_eyeq_group[key]=p
    for e in read(ROOT/'data/audit/session_20261110/eyeq_overlap/confirmed_edges.csv'):
        union(norm(e['path_a']),norm(e['path_b']))
    active_roles=collections.defaultdict(set);history_photos={find(r['path']) for r in historical}
    for r in active:active_roles[find(norm(ROOT/'data'/r['path']))].add(r['split'])
    for r in eyeq:
        g=find(norm(r['path']))
        if r['eyeq_split']=='test':active_roles[g].add('eyeq_official_test')
        if r.get('proposed_role')=='val' and r.get('proposal_eligible')=='True':active_roles[g].add('eyeq_quality_val')
        elif r.get('proposed_role')=='train' and r.get('proposal_eligible')=='True':active_roles[g].add('train')
    disease_roles=collections.defaultdict(set);disease_by_group=collections.defaultdict(list)
    for r in active:
        if r['task']!='glaucoma':continue
        g=find(norm(ROOT/'data'/r['path']));disease_roles[g].add(r['split']);disease_by_group[g].append(r)
    existing_conflicts=[]
    for g,rs in disease_by_group.items():
        if len(disease_roles[g])<=1:continue
        train_ids='|'.join(sorted({r['sample_id'] for r in rs if r['split']=='train'}))
        bridges=[e for e in edges+legacy_confirmed if find(e['path_a'])==g]
        for r in rs:
            existing_conflicts.append(dict(sample_id=r['sample_id'],task=r['task'],source=r['source'],split=r['split'],path=r['path'],sha256=r['sha256'],original_related_group=r['related_group'],new_component=hashlib.sha256(g.encode()).hexdigest()[:20],component_disease_roles='|'.join(sorted(disease_roles[g])),component_training_sample_ids=train_ids,confirmed_bridge_edges=len(bridges)))
    A.save_csv(OUT/'local_existing_conflicts.csv',existing_conflicts,['sample_id','task','source','split','path','sha256','original_related_group','new_component','component_disease_roles','component_training_sample_ids','confirmed_bridge_edges'])
    ambiguity={find(r[k]) for r in metrics if r['decision']=='ambiguous_exclude' for k in ['path_a','path_b']}
    # All cross-label identical photos are excluded; confirmed patients may have
    # different disease labels in each eye, which is not a label conflict.
    labels=collections.defaultdict(set)
    for r in new:labels[photo_group[r['path']]].add(r['label'])
    conflicts={g for g,ls in labels.items() if '0' in ls and '1' in ls}
    conflict_patients={find(r['path']) for r in new if photo_group[r['path']] in conflicts}
    photo_members=collections.defaultdict(list)
    for r in new:photo_members[photo_group[r['path']]].append(r['path'])
    canonical={g:min(ps) for g,ps in photo_members.items()}
    mshf_quality={r['filename']:r for r in read(ROOT/'data/external_quality/mshf_glaucoma_quality_join.csv')}
    fives_quality={(sheet.lower(),str(r['Disease']),str(int(r['Number']))):r for sheet,frame in pd.read_excel(ROOT/'data/external/fives/Quality Assessment.xlsx',sheet_name=None).items() for r in frame.to_dict('records')}
    assert len(fives_quality)==800
    decisions=[]
    for r in new:
        g=find(r['path']);pg=photo_group[r['path']];reason=[]
        roles=active_roles[g]
        if pg in conflicts:reason.append('conflicting_identical_photo_labels')
        elif g in conflict_patients:reason.append('known_patient_group_contains_conflicting_photo')
        if g in ambiguity:reason.append('ambiguous_geometric_evidence_excluded')
        if roles-{'train'}:reason.append('same_photo_or_known_patient_in_protected_existing_split')
        elif roles:reason.append('same_photo_or_known_patient_in_existing_training')
        if r['path']!=canonical[pg]:reason.append('duplicate_within_new_acquisition')
        if r['source']=='rim_one_dl':reason.append('author_terms_require_original_standalone_protocol')
        if r['source']=='g1020_partial':reason.append('original_data_permission_not_verified')
        if r['label']=='suspect':reason.append('suspect_label_separate_from_positive')
        if r['label'] in {'other_disease','unlabeled_glaucoma'}:reason.append('no_explicit_glaucoma_negative_label')
        status_='eligible_new_training_candidate' if not reason else 'excluded_from_pooled_training'
        training_path=r['path'];training_sha256=r['sha256']
        if status_=='eligible_new_training_candidate':
            with Path(r['path']).open('rb') as f:header=f.read(4)
            if header in {b'II*\x00',b'MM\x00*'}:
                dest=ROOT/'data/external_quality/mshf/lossless_png_for_training';dest.mkdir(exist_ok=True)
                png=dest/(Path(r['path']).stem+'.png')
                with open_image(r['path']) as im:im.save(png,format='PNG')
                with Image.open(png) as im:
                    im.load();assert hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()==r['pixel_sha256']
                training_path=norm(png);training_sha256=sha(png)
        mq=mshf_quality.get(Path(r['path']).name,{}) if r['source']=='mshf' else {}
        fq={}
        if r['source']=='fives':
            number,code=Path(r['path']).stem.rsplit('_',1);fq=fives_quality[(r['official_split'],code,str(int(number)))]
        majority={}
        for dimension in ['illumination','clarity','contrast','overall']:
            values=[mq.get(f'ann{i}_{dimension}') for i in [1,2,3]]
            majority[dimension]='1' if values.count('1')>=2 else '0' if values.count('0')>=2 else ''
        decisions.append(dict(path=r['path'],training_path=training_path,source=r['source'],label=r['label'],original_label=r['original_label'],target_definition=r.get('target_definition','original_source_glaucoma_diagnosis' if r['label'] in {'0','1'} else 'original_source_suspect_or_other_disease'),sha256=r['sha256'],training_sha256=training_sha256,pixel_sha256=r['pixel_sha256'],patient_id=r['patient_id'],eye=r['eye'],official_split=r['official_split'],photo_group=hashlib.sha256(pg.encode()).hexdigest()[:20],related_group=hashlib.sha256(g.encode()).hexdigest()[:20],existing_roles='|'.join(sorted(roles)),decision=status_,reasons='|'.join(reason),width=r['width'],height=r['height'],quality_overall_official=mq.get('overall',''),quality_overall_majority=majority['overall'],quality_illumination_official=mq.get('illumination',''),quality_illumination_majority=majority['illumination'],quality_clarity_official=mq.get('clarity',''),quality_clarity_majority=majority['clarity'],quality_contrast_official=mq.get('contrast',''),quality_contrast_majority=majority['contrast'],quality_dimension_disagreement=mq.get('majority_disagrees_with_final',''),quality_fives_IC=str(int(fq['IC'])) if fq else '',quality_fives_Blur=str(int(fq['Blur'])) if fq else '',quality_fives_LC=str(int(fq['LC'])) if fq else '',quality_fives_any_poor_dimension=str(any(int(fq[k])==0 for k in ['IC','Blur','LC'])) if fq else ''))
    save('local_decisions.csv',decisions);save('local_eligible_candidates.csv',[r for r in decisions if r['decision']=='eligible_new_training_candidate'])
    assert all(not(r['existing_roles'].split('|') and set(r['existing_roles'].split('|'))-{'','train'}) for r in decisions if r['decision']=='eligible_new_training_candidate')
    summary={}
    for ds in sorted({r['source'] for r in new}):
        rs=[r for r in decisions if r['source']==ds];eligible=[r for r in rs if r['decision']=='eligible_new_training_candidate']
        summary[ds]=dict(total_images=len(rs),original_positive_images=sum(r['label']=='1' for r in rs),suspect_images=sum(r['label']=='suspect' for r in rs),unique_photo_groups=len({r['photo_group'] for r in rs}),eligible_images=len(eligible),eligible_positive_images=sum(r['label']=='1' for r in eligible),eligible_related_groups=len({r['related_group'] for r in eligible}),known_patients=len({r['patient_id'] for r in rs if r['patient_id']}),exclusion_reasons=dict(collections.Counter(reason for r in rs for reason in r['reasons'].split('|') if reason)))
    result=dict(date='2026-11-13',completed=True,new_images=len(new),historical_feature_images=len(historical),candidate_pairs=len(metrics),candidate_decisions=dict(collections.Counter(r['decision'] for r in metrics)),exact_edges=len([e for e in edges if e['decision'].startswith('exact')]),sources=summary,method='Full file SHA256 and full RGB decode for new images; exact pixels within new acquisition; eight ROI pHash orientations with exhaustive radius-six candidates against 78047 historical images and every new image; SIFT reflection, rotation and scale / RANSAC geometric confirmation; entire known-patient, prior-image, formal-related-group and final EyeQ graph closure; ambiguous high-evidence pairs excluded',scope='Image association audit, not evidence of unknown-patient independence. Sources without patient IDs must be assigned one source-wide role, without random image-level division across train and evaluation. AIROGS scope is every RG from the 5000-image acquisition; other local sources complete as specified.',training_manifest_changed=False,models_scored=False,inputs={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'data/prepared/formal_v3/manifest.csv',ROOT/'data/audit/session_20261110/eyeq_overlap/full_components.csv',ROOT/'data/audit/session_20261110/eyeq_overlap/confirmed_edges.csv',OUT/'local_features.jsonl',OUT/'local_candidates.csv',OUT/'local_candidate_metrics.csv',Path(__file__)]})
    (OUT/'local_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');status('completed',summary=result)

if __name__=='__main__':
    START=time.time();main()
