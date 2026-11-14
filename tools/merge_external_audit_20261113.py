"""Merge completed source audits into one final image/patient decision graph."""
import collections,csv,hashlib,json,time
from pathlib import Path
import audit_external_overlap_20261113 as A

OUT=A.OUT
def features(p):
    with p.open(encoding='utf-8') as f:return [json.loads(line) for line in f]
def main():
    prefixes=['local','grape','lag','hrf','crfo']
    feature_files=[OUT/f'{p}_features.jsonl' for p in prefixes if (OUT/f'{p}_features.jsonl').exists()]
    inputs=[Path(__file__),A.ROOT/'tools/audit_external_overlap_20261113.py',A.ROOT/'tools/confirm_internal_fundus_overlap_20261113.py']+feature_files
    inputs+=[A.ROOT/'tools/audit_eyeq_overlap.py',A.ROOT/'data/prepared/formal_v3/manifest.csv',A.ROOT/'data/audit/session_20261110/eyeq_overlap/features.jsonl',A.ROOT/'data/audit/session_20261110/eyeq_overlap/full_components.csv',A.ROOT/'data/audit/session_20261110/eyeq_overlap/confirmed_edges.csv',A.ROOT/'data/audit/deep/release_inputs/confirmed_edges_v1.csv',A.ROOT/'data/external/airogs_candidate_5k/positive_secondary_distance_le4.csv',A.ROOT/'data/audit/session_20261110/eyeq_overlap/confirmation_cache/binding.json',A.ROOT/'data/audit/session_20261110/eyeq_overlap/cached_confirmation.json']
    inputs+=[A.ROOT/'data/external/fives/Quality Assessment.xlsx',A.ROOT/'data/external_quality/mshf_glaucoma_quality_join.csv']
    new_by_path={}
    for p in feature_files:
        for r in features(p):
            r['path']=A.norm(r['path']);r.setdefault('patient_id','');r.setdefault('eye','');r.setdefault('official_split','');r.setdefault('original_label',r.get('label',''));r.setdefault('pixel_sha256','')
            if r['path'] in new_by_path:assert new_by_path[r['path']]['sha256']==r['sha256']
            new_by_path[r['path']]=r
    new=list(new_by_path.values())
    with (OUT/'combined_features.jsonl').open('w',encoding='utf-8') as f:
        for r in new:f.write(json.dumps(r)+'\n')
    metrics_files=[OUT/'local_final_candidate_metrics.csv']
    for p in ['grape','lag','hrf','crfo']:
        fp=OUT/f'{p}_final_candidate_metrics.csv'
        if not fp.exists():fp=OUT/f'{p}_candidate_metrics.csv'
        if fp.exists():metrics_files.append(fp)
    for fp in OUT.glob('lag_grape*candidate_metrics.csv'):
        if fp not in metrics_files:metrics_files.append(fp)
    assert metrics_files[0].exists(),'Local strict confirmation has not completed'
    ranks={'rejected_pHash_collision':0,'ambiguous_exclude':1,'known_same_eye_metadata':2,'confirmed_same_photo':3}
    merged={};source_counts={};duplicates=0
    fields=['path_a','path_b','hash_distance','orientation','ssim','sift_matches','sift_inliers','sift_ratio','coverage_a','coverage_b','reprojection_median','thumbnail_equal','decision','audit_sources','identity_evidence']
    inner=['inner_matches','inner_inliers','inner_ratio','inner_coverage_a','inner_coverage_b','aligned_correlation','vessel_highpass_correlation','inner_reprojection','full_image_correlation']
    fields+=inner
    local_internal={}
    if (OUT/'local_internal_candidate_metrics.csv').exists():
        local_internal={(A.norm(r['path_a']),A.norm(r['path_b'])):r for r in A.read(OUT/'local_internal_candidate_metrics.csv')}
    for fp in metrics_files:
        rs=A.read(fp);source_counts[fp.name]=len(rs)
        for r in rs:
            r['path_a']=A.norm(r['path_a']);r['path_b']=A.norm(r['path_b']);r['decision']=r.get('final_decision') or r['decision']
            if r['decision']=='ambiguous_exclude':
                a=new_by_path.get(r['path_a']);b=new_by_path.get(r['path_b'])
                if a and b and a['source']==b['source']=='grape' and a['patient_id'] and a['patient_id']==b['patient_id'] and a['eye']==b['eye']:
                    r['decision']='known_same_eye_metadata'
                    r['identity_evidence']='Official GRAPE clinical join: same subject and eye, different visit; retain image versions within one patient group'
            if fp.name=='local_final_candidate_metrics.csv':
                proof=local_internal.get((r['path_a'],r['path_b']),{})
                for key in inner:
                    if key in proof:r[key]=proof[key]
            assert r['decision'] in ranks,(fp,r['decision'])
            key=tuple(sorted((r['path_a'],r['path_b'])))
            if key in merged:
                duplicates+=1;old=merged[key];sources=set(old['audit_sources'].split('|'))|{fp.name}
                if ranks[r['decision']]>ranks[old['decision']]:merged[key]=dict(r,audit_sources='|'.join(sorted(sources)))
                else:old['audit_sources']='|'.join(sorted(sources))
            else:merged[key]=dict(r,audit_sources=fp.name)
    final=[{k:r.get(k,'') for k in fields} for _,r in sorted(merged.items())]
    A.A.save_csv(OUT/'final_metrics.csv',final,fields)
    confirmed=[r for r in final if r['decision']=='confirmed_same_photo'];A.A.save_csv(OUT/'final_confirmed_edges.csv',confirmed,fields)
    historical=features(A.ROOT/'data/audit/session_20261110/eyeq_overlap/features.jsonl')
    for r in historical:r['path']=A.norm(r['path'])
    bysha=collections.defaultdict(list)
    for r in historical+new:bysha[r['sha256']].append(r['path'])
    newpaths=set(new_by_path);exact=[]
    for paths in bysha.values():
        paths=sorted(set(paths))
        if len(paths)<2 or not newpaths.intersection(paths):continue
        for p in paths[1:]:exact.append(dict(path_a=paths[0],path_b=p,decision='exact_file_sha256'))
    bypixel=collections.defaultdict(list)
    for r in new:
        if r.get('pixel_sha256'):bypixel[(r['width'],r['height'],r['pixel_sha256'])].append(r['path'])
    for paths in bypixel.values():
        paths=sorted(set(paths))
        for p in paths[1:]:exact.append(dict(path_a=paths[0],path_b=p,decision='exact_decoded_rgb_pixels'))
    A.A.save_csv(OUT/'final_exact_edges.csv',exact,['path_a','path_b','decision'])
    A.START=time.time();A.finalize(new,historical,exact+[dict(path_a=r['path_a'],path_b=r['path_b'],decision=r['decision']) for r in confirmed],final)
    for fp in OUT.glob('*source_manifest.csv'):inputs.append(fp)
    for pattern in ['*visual_adjudications.csv','*manual_confirmed_edges.csv','*visual_confirmed_edges.csv','*visual*.png']:
        for fp in OUT.glob(pattern):
            if fp not in inputs:inputs.append(fp)
    for fp in [OUT/'local_internal_candidate_metrics.csv',OUT/'local_gray_cache_validation.json',OUT/'local_matcher_validation.json',OUT/'local_matcher_validation.csv',OUT/'grape_clinical_join.json',OUT/'combined_features.jsonl',OUT/'final_metrics.csv',OUT/'final_confirmed_edges.csv',OUT/'final_exact_edges.csv']:
        if fp.exists():inputs.append(fp)
    inputs+=metrics_files
    for fp in A.ROOT.joinpath('tools').glob('*20261113.py'):
        if fp.name.startswith(('audit_','confirm_','merge_external_','validate_external_','validate_low_','finalize_lag_visual_','finalize_grape_cross_visual_')) and fp not in inputs:inputs.append(fp)
    for fp in OUT.glob('*.py'):
        if fp not in inputs:inputs.append(fp)
    for fp in OUT.glob('*summary.json'):
        if fp.name not in {'local_summary.json','combined_summary.json'}:inputs.append(fp)
    result=json.loads((OUT/'local_summary.json').read_text(encoding='utf-8'))
    result['scope']='Unsplit audited pool. All known patients and confirmed related-image groups must remain intact across roles. Related source families may supply training and internal image-group validation through one joint group split; without patient IDs this is not patient-independent validation. A family used for independent external evaluation remains wholly withheld. AIROGS coverage is every RG in the 5000-image acquisition; other source coverage is complete as enumerated.'
    result.update(new_images=len(new),candidate_pairs=len(final),candidate_decisions=dict(collections.Counter(r['decision'] for r in final)),raw_source_pair_counts=source_counts,duplicate_cross_audit_pairs_removed=duplicates,exact_proof_records=len(exact),exact_unordered_pairs=len({tuple(sorted((e['path_a'],e['path_b']))) for e in exact}),final_metrics='data/audit/data_expansion_20261113/final_metrics.csv',source_feature_rows={p.name:len(features(p)) for p in feature_files},method='Source-bound full decoding and file/pixel SHA checks; eight ROI pHash orientations and exhaustive radius-six retrieval; all raw geometric hits/ambiguities receive interior-retinal SIFT, transformed-image and high-frequency vessel confirmation; final strong or exact-image edges, explicitly documented visual same-eye decisions, known patient identities and legacy strict edges enter transitive closure; official same-subject/eye follow-up metadata preserves separate images within one patient group; insufficient unknown-identity matches explicitly excluded',inputs={str(p.relative_to(A.ROOT)):A.sha(p) for p in sorted(set(inputs),key=lambda p:p.as_posix())},merge_complete=True)
    (OUT/'local_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');(OUT/'combined_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(new_images=len(new),candidate_pairs=len(final),sources=result['sources'],duplicates_removed=duplicates),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
