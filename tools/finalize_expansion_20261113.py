"""Publish the audited image pool and repair development evaluation only.

No model inference, training, or modification of prior releases is performed.
"""
from pathlib import Path
import csv
import hashlib
import json
import collections
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, __version__ as PILLOW_VERSION
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/'data/audit/data_expansion_20261113'
POOL=ROOT/'data/prepared/expansion_20261113'
REPAIR=ROOT/'data/prepared/formal_v3_boundary_repaired_20261113'

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def decoded_rgb_hash(path):
    with path.open('rb') as f:header=f.read(4)
    if header in {b'II*\x00',b'MM\x00*'}:
        # Use the same original-TIFF decoder as the sealed audit, avoiding the
        # native Pillow/libtiff crash on MSHF's TIFF-with-JPG-extension files.
        array=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
        assert array is not None,path
        return hashlib.sha256(cv2.cvtColor(array,cv2.COLOR_BGR2RGB).tobytes()).hexdigest()
    with Image.open(path) as image:
        image.load()
        return hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()

def write(path,rows,fields=None):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]))
        w.writeheader();w.writerows(rows)

def main():
    # Exact decoded-pixel hashes are decoder-version dependent. The audit used
    # project Pillow 13.0.0; base Python's older JPEG decoder produces different
    # rounding for some unchanged originals. Verify in the audited environment.
    assert PILLOW_VERSION == '13.0.0', 'Run with tools/training_python.cmd (audited Pillow 13.0.0)'
    summary=json.loads((AUDIT/'local_summary.json').read_text(encoding='utf-8'))
    assert summary['completed']
    decisions=read(AUDIT/'local_decisions.csv')
    candidates=[r for r in decisions if r['decision']=='eligible_new_training_candidate']
    assert candidates and len(decisions)==summary['new_images']
    metric_path=AUDIT/'final_metrics.csv'
    if not metric_path.exists():metric_path=AUDIT/'local_candidate_metrics.csv'
    metrics=read(metric_path)
    assert len(metrics)==summary['candidate_pairs']
    assert len({(r['path_a'],r['path_b']) for r in metrics})==len(metrics)
    assert all(r['decision'] in {'confirmed_same_photo','ambiguous_exclude','rejected_pHash_collision','known_same_eye_metadata'} for r in metrics)
    for path,d in summary['inputs'].items():assert digest(ROOT/path)==d,path
    auxiliary=[r for r in decisions if r['label']=='suspect' and
        set(filter(None,r['reasons'].split('|'))) <= {'suspect_label_separate_from_positive'}]
    def verify(r):
        original=Path(r['path']).resolve()
        assert original.is_relative_to((ROOT/'data').resolve())
        assert digest(original)==r['sha256'],original
        assert decoded_rgb_hash(original)==r['pixel_sha256'],original
        p=Path(r.get('training_path') or r['path']).resolve()
        assert p.is_relative_to((ROOT/'data').resolve())
        d=r.get('training_sha256') or r['sha256']
        if p != original:
            assert digest(p)==d,p
            assert decoded_rgb_hash(p)==r['pixel_sha256'],p
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,_ in enumerate(pool.map(verify,candidates+auxiliary),1):
            if i%2000==0:print(f'verified original/pool file and RGB hashes {i}',flush=True)
    records=[];auxiliary_records=[]
    for r in candidates+auxiliary:
        p=Path(r.get('training_path') or r['path']).resolve()
        d=r.get('training_sha256') or r['sha256']
        record=dict(sample_id='exp13:'+hashlib.sha256((r['source']+':'+r['photo_group']).encode()).hexdigest()[:24],
            path=p.relative_to(ROOT/'data').as_posix(),sha256=d,source=r['source'],label=r['label'],
            original_label=r['original_label'],target_definition=r.get('target_definition') or ('source_defined_glaucoma_suspect' if r['label']=='suspect'
                else 'AIROGS_glaucoma_referral_RG' if r['source']=='airogs_original_rg' else 'source_defined_glaucoma_diagnosis'),
            source_family='LAG_GRAPE_FIVES_FundusAVSeg_MSHF' if r['source'] in {'lag','lag_part1','grape','fives','fundus_avseg','mshf'} else r['source'],
            photo_group=r['photo_group'],related_group=r['related_group'],patient_id=r['patient_id'],
            patient_id_status='KNOWN' if r['patient_id'] else 'UNKNOWN',task='glaucoma',eligible='true',
            review_status='completed_source_and_image_association_audit',
            official_split=r['official_split'],imaging_domain='optic_disc_crop' if r['source']=='keh' else 'fundus',
            original_path=Path(r['path']).resolve().relative_to(ROOT/'data').as_posix(),original_sha256=r['sha256'],
            pool_role='auxiliary_suspect_supervision_only' if r['label']=='suspect' else 'admitted_not_yet_used_by_a_model')
        record.update({k:v for k,v in r.items() if k.startswith('quality_')})
        (auxiliary_records if r['label']=='suspect' else records).append(record)
    assert len({r['sample_id'] for r in records})==len(records)
    assert len({r['sha256'] for r in records})==len(records)
    assert all(r['label'] in {'0','1'} for r in records)
    POOL.mkdir(exist_ok=True)
    write(POOL/'pool_manifest.csv',records)
    if auxiliary_records:write(POOL/'suspect_manifest.csv',auxiliary_records)
    counts={s:dict(images=len(rr:=[r for r in records if r['source']==s]),positives=sum(r['label']=='1' for r in rr),
        linked_groups=len({r['related_group'] for r in rr}),known_patient_ids=len({r['patient_id'] for r in rr if r['patient_id']}))
        for s in sorted({r['source'] for r in records})}
    release=dict(version='audited-expansion-pool-v1',date='2026-11-13',audit_complete=True,
        images=len(records),positive_images=sum(r['label']=='1' for r in records),sources=counts,
        source_defined_glaucoma_positive_images=sum(r['label']=='1' and r['source']!='airogs_original_rg' for r in records),
        airogs_RG_referral_positive_images=sum(r['label']=='1' and r['source']=='airogs_original_rg' for r in records),
        officially_poor_quality_positive_images={
            'FIVES_at_least_one_poor_dimension':sum(r['label']=='1' and r['source']=='fives' and r.get('quality_fives_any_poor_dimension')=='True' for r in records),
            'MSHF_overall_poor':sum(r['label']=='1' and r['source']=='mshf' and r.get('quality_overall_official')=='0' for r in records)},
        auxiliary_suspect_images=len(auxiliary_records),
        pixel_hash_decoder='Pillow 13.0.0 RGB; original MSHF TIFF decoded by audit OpenCV and retained as equal-pixel PNG',
        pixel_hash_decoder_versions={'pillow':PILLOW_VERSION,'opencv':cv2.__version__},
        auxiliary_suspect_by_source=dict(collections.Counter(r['source'] for r in auxiliary_records)),
        auxiliary_manifest_sha256=digest(POOL/'suspect_manifest.csv') if auxiliary_records else None,
        manifest_sha256=digest(POOL/'pool_manifest.csv'),
        inputs={str(p.relative_to(ROOT)):digest(p) for p in [AUDIT/'local_decisions.csv',AUDIT/'local_summary.json',Path(__file__)]},
        research_training_released=False,reason='Additive dataset pool; no training or source-role allocation was executed in this research turn',
        requirements='Keep known patient and confirmed image groups intact across every role. '
            'LAG, GRAPE, FIVES, Fundus-AVSeg and MSHF require a joint source-family split, not independent source-specific random splits. '
            'Internal validation may use image-association groups where patient IDs are unknown, explicitly without a patient-independence claim. '
            'Independent external evaluation requires a source-family holdout. '
            'Respect author partitions. Keep disease target definitions and imaging domains explicit.',
        joint_source_roles={'LAG_GRAPE_FIVES_FundusAVSeg_MSHF':'joint patient/image-group allocation for training and internal validation; no mutually independent external-test claims'})
    (POOL/'release.json').write_text(json.dumps(release,ensure_ascii=False,indent=2),encoding='utf-8')
    table='\n'.join(f"| {s} | {v['images']} | {v['positives']} |" for s,v in counts.items())
    for lang in ['zh','en']:
        if lang=='zh':
            body=f'''# 已审计扩展数据池\n\n2026-11-13。已纳入 {len(records)} 张图，其中 {release['positive_images']} 张阳性。\n\n| 来源 | 图像 | 阳性 |\n|---|---:|---:|\n{table}\n\n`pool_manifest.csv` 保留纳入图像及原标签、哈希、照片/患者组和作者划分；`release.json` 绑定审计输入。完整解码和文件/像素哈希再次核验通过。它是本轮新增数据池，未用于训练。原 formal_v3 及 EyeQ 划分没有覆盖。\n\n无患者ID的来源应整库固定用途；有患者ID的来源按患者/关联组划分。KEH保留视盘裁切域，不混作全眼底质量。MSHF中实际为TIFF的JPG已另存等RGB像素的PNG，原图保留，路径和哈希均记录。入选不等于新眼科复诊，疑似和无明确标签样本未混为阳性或阴性。\n\n[中文报告](../../../docs/progress/2026-11-13/data-review/data-and-degradation-review.zh.md) · [English](README.en.md)\n'''
        else:
            body=f'''# Audited expansion pool\n\nNovember 13, 2026. Admitted {len(records)} images, including {release['positive_images']} positive labels.\n\n| Source | Images | Positives |\n|---|---:|---:|\n{table}\n\n`pool_manifest.csv` retains image identities, original labels, hashes, photo/patient groups, and author partitions. `release.json` binds the audit inputs. Complete decoding and file/pixel hashes were rechecked. This is an additive data pool that has not trained a model; prior formal_v3 and EyeQ releases remain intact.\n\nAssign sources without patient identifiers to one source-wide role; split known patient/linked groups together. Keep KEH's disc-crop domain explicit. MSHF TIFF files with JPG extensions have additional PNG copies with identical decoded RGB pixels; originals, paths, and hashes remain traceable. Admission is not new clinical adjudication; suspect and unlabeled samples were not converted into positives or negatives.\n\n[English report](../../../docs/progress/2026-11-13/data-review/data-and-degradation-review.en.md) · [中文](README.md)\n'''
        body=body.replace('无患者ID的来源应整库固定用途；有患者ID的来源按患者/关联组划分。',
            '训练与内部验证按已知患者及统一图像关联组联合划分；未知患者部分明确为图像组验证，独立外测整来源家族留出。')
        body=body.replace('Assign sources without patient identifiers to one source-wide role; split known patient/linked groups together.',
            'Jointly split known patient and image-association groups for training/internal validation. State the image-group unit when patient identities are unknown; hold out whole source families for independent external testing.')
        body+='\n'+('LAG、GRAPE、FIVES、Fundus-AVSeg、MSHF 属于相互关联的来源家族，必须联合分组，不能相互冒充独立外部测试。\n'
            if lang=='zh' else 'LAG, GRAPE, FIVES, Fundus-AVSeg, and MSHF form a related source family requiring joint grouping; they are not independent external tests of one another.\n')
        if auxiliary_records:
            body += (f'\n另有 {len(auxiliary_records)} 张通过图像审计的疑似图，保留在 `suspect_manifest.csv`，只作为独立的疑似/辅助监督，不计确诊阳性。\n'
                if lang=='zh' else f'\nA separate `suspect_manifest.csv` retains {len(auxiliary_records)} image-audited suspects for auxiliary supervision; these are not counted as diagnosed positives.\n')
        (POOL/('README.md' if lang=='zh' else 'README.en.md')).write_text(body,encoding='utf-8')
    # Previously saved boundary predictions can be filtered without scoring any
    # images or letting prediction values influence exclusions.
    conflict=read(AUDIT/'local_existing_conflicts.csv')
    exclude_ids={r['sample_id'] for r in conflict if r['split']=='boundary' and
        ({'train','val'} & set(r['component_disease_roles'].split('|')))}
    old=ROOT/'data/prepared/formal_v3_boundary_repaired_20261110'
    boundary=read(old/'manifest.csv')
    excluded=[dict(r,exclusion_reason='confirmed_photo_or_known_patient_group_links_training_or_model_validation')
        for r in boundary if r['sample_id'] in exclude_ids]
    kept=[r for r in boundary if r['sample_id'] not in exclude_ids]
    assert kept
    REPAIR.mkdir(exist_ok=True)
    write(REPAIR/'manifest.csv',kept)
    write(REPAIR/'exclusions.csv',excluded,list(boundary[0])+['exclusion_reason'])
    oldpq=ROOT/'runs/formal_v3/boundary_development_repaired_20261110/boundary_pq.csv'
    pq=read(oldpq);keep_ids={r['sample_id'] for r in kept}
    retained=[r for r in pq if r['sample_id'] in keep_ids]
    assert len(retained)==len(kept) and {r['sample_id'] for r in retained}==keep_ids
    write(AUDIT/'repaired_boundary_pq.csv',retained)
    info=dict(version='evaluation-only-boundary-repair-20261113',evaluation_only=True,
        research_training_released=False,prior_boundary_images=len(boundary),excluded_images=len(excluded),
        retained_images=len(kept),retained_positives=sum(r['label']=='1' for r in kept),
        excluded_positives=sum(r['label']=='1' for r in excluded),
        manifest_sha256=digest(REPAIR/'manifest.csv'),predictions_sha256=digest(AUDIT/'repaired_boundary_pq.csv'),
        existing_conflict_roles=dict(collections.Counter(r['split'] for r in conflict)),
        inputs={str(p.relative_to(ROOT)):digest(p) for p in [old/'manifest.csv',old/'release.json',oldpq,
            AUDIT/'local_existing_conflicts.csv',AUDIT/'local_summary.json',Path(__file__)]},
        exclusions_selected_without_scores=True,models_scored=False,training_unchanged=True,
        final_test_predictions_loaded=False)
    (REPAIR/'release.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    (AUDIT/'finalization.json').write_text(json.dumps(dict(pool=release,boundary_repair=info),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(pool_images=len(records),pool_positives=release['positive_images'],sources=counts,boundary=info),indent=2))

if __name__=='__main__':main()
