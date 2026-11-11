"""Build provisional training worklists and model/environment evidence, without training."""
from __future__ import annotations
import collections
import csv
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data'
OUT = DATA/'audit/2026-11-09'

def save(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def write_csv(name, rows):
    with (OUT/name).open('w', newline='', encoding='utf-8-sig') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def model_audit():
    import torch
    import torchvision
    from torchvision.models import efficientnet_b0
    torch.set_num_threads(4)
    env=dict(python=sys.executable, python_version=platform.python_version(), torch=torch.__version__,
        torchvision=torchvision.__version__, torch_cuda_build=torch.version.cuda,
        cuda_available=torch.cuda.is_available(), dependencies={x:bool(importlib.util.find_spec(x))
            for x in ['timm','SoftPool','softpool_cuda','yaml','addict','lmdb','einops','imp']})
    env['gpu_inventory']=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
    model=efficientnet_b0(weights=None).eval()
    with torch.inference_mode():
        output=model(torch.zeros(1,3,224,224))
    env['backbone_architecture_smoke_test']=dict(weights='random; not a trained classifier', shape=list(output.shape),finite=bool(torch.isfinite(output).all()))
    save('environment.json',env)
    p=ROOT/'models/checkpoints/net_g_226264S4.pth'
    loaded=torch.load(p,map_location='cpu',weights_only=True)
    registry=dict(teacher=dict(name='FTHNet4', file=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest(),source='https://github.com/HudenJear/BasiQA',
        download='https://drive.google.com/file/d/1D1f2M6S_3qZgPwuUItBDutVBzlLgcjMU/view',
        safe_load_weights_only=True,checkpoint_keys=list(loaded),state_dict_key='params',state_dict_tensors=len(loaded['params']),
        forward_test='blocked: Python 3.12 removed imp; SoftPool/softpool_cuda unavailable; CUDA PyTorch unavailable',
        training_patient_split_provenance='unknown; do not claim independent FIQS quality test performance'),
        source_revision=json.loads((OUT/'basiqa_commit.json').read_text())['sha'],
        glaucoma_initialization=dict(name='EfficientNet-B0 ImageNet',status='missing',
        url='https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth',
        failure='PowerShell and Python download attempts both failed with TLS connection reset; random initialization is not silently substituted'),
        trained_student='absent',trained_glaucoma_classifier='absent',formal_training_ready=False)
    initialization=ROOT/'models/checkpoints/efficientnet_b0_rwightman-7f5810bc.pth'
    if initialization.exists():
        sha=hashlib.sha256(initialization.read_bytes()).hexdigest()
        if not sha.startswith('7f5810bc'):raise ValueError('EfficientNet initialization hash mismatch')
        model.load_state_dict(torch.load(initialization,map_location='cpu',weights_only=True),strict=True)
        registry['glaucoma_initialization'].update(status='downloaded_and_strict_load_verified',
            file=initialization.relative_to(ROOT).as_posix(),bytes=initialization.stat().st_size,sha256=sha)
        registry['glaucoma_initialization']['previous_failure']=registry['glaucoma_initialization'].pop('failure')
    runtime=OUT/'teacher_runtime.json'
    if runtime.exists() and json.loads(runtime.read_text())['passed']:
        registry['teacher']['forward_test']='passed; original CUDA SoftPool, strict load, real-image forward'
        registry['teacher']['runtime_evidence']='data/audit/2026-11-09/teacher_runtime.json'
    save('model_registry.json',registry)

def worklists():
    rows=list(csv.DictReader((OUT/'image_manifest.csv').open(encoding='utf-8-sig')))
    dup=collections.Counter(r['sha256'] for r in rows)
    odir_path=DATA/'sources/eyeworld/Dataset/Fundoscopy/DATASET_101/ODIR-5K/Dataset_ODIR-5K_V1.xlsx'
    wb=load_workbook(odir_path,read_only=True,data_only=True)
    vals=iter(wb.active.values); headers=next(vals)
    odir={r['filename']:r for r in (dict(zip(headers,v)) for v in vals)}; wb.close()
    hygd={r['Image Name']:r for r in csv.DictReader((DATA/'external/hygd-1.1.0/Labels.csv').open())}
    work=[]
    for r in rows:
        path=Path(r['path']); src=r['source']; raw=r['folder_label']
        label=''; task=''; group=''; role='excluded_from_glaucoma'; reasons=[]
        if src=='airogs-light-v2':
            raw=path.parent.name;label={'RG':'1','NRG':'0'}.get(raw,'');task='referable_glaucoma';role='development_candidate'
            reasons += ['patient_id_unknown','official_split_not_patient_verified','source_overlap_not_resolved']
        elif src=='hygd-1.1.0':
            meta=hygd.get(path.name,{})
            raw=meta.get('Label',''); label={'GON+':'1','GON-':'0','GON−':'0'}.get(raw,'')
            task='glaucomatous_optic_neuropathy';group='HYGD:'+meta.get('Patient','UNKNOWN');role='proposed_external_holdout'
            reasons += ['source_overlap_not_resolved']
        elif src in ('DATASET_101/ACRIMA','EyeAugmented','Kaggle-Eye_diseases_classification'):
            if raw in ('Glaucoma','Normal Retina'):
                label='1' if raw=='Glaucoma' else '0';task='folder_glaucoma_vs_normal';role='development_candidate'
                reasons += ['clinical_label_provenance_unverified','patient_id_unknown','source_overlap_not_resolved']
                if src=='EyeAugmented':reasons += ['augmentation_family_unknown']
                if src=='DATASET_101/ACRIMA':role='proposed_source_holdout'
        elif src=='DATASET_101/ODIR-5K':
            meta=odir.get(path.name,{})
            raw=str(meta.get('Diagnostic_Keywords',''))
            group='ODIR:'+str(meta.get('orig_id','UNKNOWN'))
            task='eye_level_keyword_glaucoma_vs_normal'
            if 'suspected glaucoma' in raw.lower():
                reasons += ['suspected_glaucoma_not_confirmed_positive']
            elif 'glaucoma' in raw.lower():label='1'
            elif raw.strip().lower()=='normal fundus':label='0'
            if label:
                role='development_candidate'; reasons += ['eye_keyword_mapping_requires_review','source_overlap_not_resolved']
            else:reasons += ['other_or_ambiguous_eye_diagnosis_not_mapped']
        if task:
            if dup[r['sha256']]>1:reasons+=['exact_duplicate_group']
            if r['status']!='full_decode_ok':reasons+=['decode_failure']
            if r['review_flags']:reasons+=r['review_flags'].split('|')
            reasons+=['disc_gradability_pending']
            work.append(dict(path=r['path'],source=src,original_label=raw,target=label,target_definition=task,
                patient_group=group or 'UNKNOWN',sha256=r['sha256'],proposed_role=role,split='UNASSIGNED',
                train_eligible=False,pending_reasons='|'.join(reasons)))
    # Connected split units join both recorded patients and byte-identical content.
    # This is a conservative leakage group, not a claim that patient IDs are the same person.
    parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    for r in work:
        key='sha256:'+r['sha256']
        find(key)
        if r['patient_group']!='UNKNOWN':union(key,r['patient_group'])
    for r in work:r['related_split_group']=find('sha256:'+r['sha256'])
    write_csv('glaucoma_candidate_worklist.csv',work)
    full=list(csv.DictReader((odir_path.parent/'00_Info/full_df.csv').open(encoding='utf-8-sig')))
    mismatches=[r for r in full if Path(r['path']).name!=r['filename']]
    save('worklist_summary.json',dict(rows=len(work),
        by_source_target=dict(collections.Counter(r['source']+'|'+r['target'] for r in work)),
        proposed_roles=dict(collections.Counter(r['proposed_role'] for r in work)),
        train_eligible_rows=0,locked_split=False,
        hygd_recorded_patients=len({r['patient_group'] for r in work if r['source']=='hygd-1.1.0'}),
        hygd_patient_and_exact_duplicate_components=len({r['related_split_group'] for r in work if r['source']=='hygd-1.1.0'}),
        odir_auxiliary_full_df_rows=len(full),odir_auxiliary_path_filename_mismatches=len(mismatches),
        odir_auxiliary_examples=[{k:r[k] for k in ('filename','path','orig_id','label')} for r in mismatches[:3]],
        note='Provisional worklist only. FIQS/EyeQ quality targets stay separate; no cleaned set or final labels claimed.'))

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    model_audit()
    if (OUT/'image_manifest.csv').exists():worklists()
    print('Audit evidence generated',flush=True)
