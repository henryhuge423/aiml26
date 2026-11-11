"""Freeze an audited research cohort without claiming unknown patient independence."""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.data import sha256,load_rows


class Union:
    def __init__(self):self.parent={}
    def find(self,x):
        self.parent.setdefault(x,x)
        if self.parent[x]!=x:self.parent[x]=self.find(self.parent[x])
        return self.parent[x]
    def join(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.parent[max(a,b)]=min(a,b)


def stable_fraction(group,seed):
    return int(hashlib.sha256(f'{seed}:{group}'.encode()).hexdigest()[:16],16)/2**64


def assign_cohort(work,quality,edges,decisions,seed=20261109):
    family=Union();same_hash={}
    for r in work+quality:
        family.find(r['path'])
        if r['sha256'] in same_hash:family.join(r['path'],same_hash[r['sha256']])
        same_hash[r['sha256']]=r['path']
    for a,b in edges:family.join(a,b)
    groups=Union()
    for path in family.parent:groups.join(path,family.find(path))
    patients={}
    for r in work:
        patient=r.get('patient_group','UNKNOWN')
        if patient!='UNKNOWN':
            if patient in patients:groups.join(r['path'],patients[patient])
            patients[patient]=r['path']
    possible_teacher_groups={groups.find(r['path']) for r in quality}
    role={}
    for r in work:
        group=groups.find(r['path'])
        candidate={'hygd-1.1.0':'external_hygd','DATASET_101/ACRIMA':'external_acrima'}.get(r['source'])
        if candidate and (group not in role or candidate=='external_hygd'):role[group]=candidate
    family_rows=collections.defaultdict(list)
    for r in work:
        if r['target'] in {'0','1'}:family_rows[family.find(r['path'])].append(r)
    family_decisions=collections.defaultdict(set)
    for path,decision in decisions.items():
        if path in family.parent:family_decisions[family.find(path)].add(decision)
    excluded=[];selected=[];group_split={}

    def split_for(path,quality_only=False):
        group=groups.find(path)
        if group not in group_split:
            value=stable_fraction(group,seed)
            group_split[group]=role.get(group) or (
                ('train' if value<.8 else 'val') if quality_only or group in possible_teacher_groups else
                'train' if value<.7 else 'val' if value<.8 else 'boundary' if value<.9 else 'test')
        return group_split[group]

    priority={'hygd-1.1.0':0,'DATASET_101/ACRIMA':1,'airogs-light-v2':2,
              'DATASET_101/ODIR-5K':3,'Kaggle-Eye_diseases_classification':4,'EyeAugmented':5}
    for fam,members in sorted(family_rows.items()):
        if len({r['target'] for r in members})>1:
            # Different source endpoints can disagree. Keep externally supplied
            # truth intact and remove connected development copies; never clean
            # a test label using a different training source's proxy label.
            external=[r for r in members if r['source']=='hygd-1.1.0']
            if not external:external=[r for r in members if r['source']=='DATASET_101/ACRIMA']
            if not external or len({r['target'] for r in external})>1:
                excluded += [dict(path=r['path'],reason='conflicting_image_family_labels',detail=fam) for r in members]
                continue
            excluded += [dict(path=r['path'],reason='different_endpoint_conflict_development_quarantine',detail=fam)
                         for r in members if r not in external]
            members=external
        candidates=[]
        for r in members:
            split=split_for(r['path']);action=decisions.get(r['path'],'not_sampled')
            inherited=family_decisions[fam]
            reason=None
            if r['source']=='EyeAugmented':reason='augmentation_parent_lineage_unverified_extension_only'
            elif inherited & {'quarantine_overlay','exclude_overlay','exclude_nonfundus'}:reason='technical_overlay_or_nonfundus_image_family'
            elif split.startswith('external_'):
                expected='hygd-1.1.0' if split=='external_hygd' else 'DATASET_101/ACRIMA'
                if groups.find(r['path']) in possible_teacher_groups:reason='possible_teacher_pretraining_exposure_external_quarantine'
                elif r['source']!=expected:reason='connected_to_protected_holdout'
            elif inherited & {'quarantine_severe_technical','quality_only'}:reason='severe_technical_development_quarantine'
            if reason:excluded.append(dict(path=r['path'],reason=reason,detail=action))
            else:candidates.append(r)
        if not candidates:continue
        candidates.sort(key=lambda r:(priority.get(r['source'],9),r['path']))
        r=candidates[0];split=split_for(r['path'])
        group=groups.find(r['path'])
        group_id='component:'+hashlib.sha256(group.encode()).hexdigest()[:24]
        selected.append(dict(sample_id='g:'+r['sha256'][:24],path=r['path'],sha256=r['sha256'],
            source=r['source'],label=r['target'],target_definition=r['target_definition'],
            related_group=group_id,patient_id_status='known' if r.get('patient_group','UNKNOWN')!='UNKNOWN' else 'unknown',
            split=split,task='glaucoma',eligible='true',
            review_status=decisions.get(r['path'],'publisher_label_and_technical_audit_sampling'),
            original_label=r['original_label']))
        excluded += [dict(path=other['path'],reason='duplicate_family_alias',detail=r['path']) for other in candidates[1:]]
    for r in work:
        if r['target'] not in {'0','1'}:excluded.append(dict(path=r['path'],reason='unmapped_or_suspected_label',detail=r['original_label']))
    for r in quality:
        split=split_for(r['path'],True)
        if split not in {'train','val'}:
            excluded.append(dict(path=r['path'],reason='quality_image_related_to_protected_disease_holdout',detail=split));continue
        group=groups.find(r['path'])
        selected.append(dict(sample_id='q:'+r['sha256'][:24],path=r['path'],sha256=r['sha256'],
            source='fiqs',label=r['label'],target_definition='human_MOS_0_1_and_teacher_fidelity',
            related_group='component:'+hashlib.sha256(group.encode()).hexdigest()[:24],
            patient_id_status='unknown',split=split,task='quality',eligible='true',
            review_status='human_MOS_retained_all_quality_levels',original_label=r['label']))
    return sorted(selected,key=lambda r:r['sample_id']),excluded


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def write_csv(path,rows,fields):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def main(output,edge_file,decision_file,seed):
    if (output/'release.json').exists():raise ValueError('release already frozen; choose a new version directory')
    audit=ROOT/'data/audit/2026-11-09'
    work=read_csv(audit/'glaucoma_candidate_worklist.csv')
    quality=[dict(path=r['path'],sha256=r['sha256'],label=r['human_mos_normalized'],image_id=r['image_id'])
             for r in read_csv(ROOT/'data/prepared/1109/fiqs_quality.csv')]
    edges=[(r['path_a'],r['path_b']) for r in read_csv(edge_file)]
    decisions={r['path']:r['decision'] for r in read_csv(decision_file)}
    # Match both FIQS resolutions to one image family before integrating any cross-source edges.
    for r in quality:
        edges.append((r['path'],r['path'].replace('/images/','/full_size_images/')))
    rows,excluded=assign_cohort(work,quality,edges,decisions,seed)
    if not rows:raise ValueError('empty prepared cohort')
    for r in rows:
        path=(ROOT/'data'/r['path']).resolve()
        if not path.is_relative_to((ROOT/'data').resolve()) or sha256(path)!=r['sha256']:
            raise ValueError(f'source hash changed: {r["path"]}')
    output.mkdir(parents=True,exist_ok=True)
    manifest=output/'manifest.csv';write_csv(manifest,rows,list(rows[0]))
    for task in ('glaucoma','quality'):
        for split in ('train','val'):load_rows(manifest,task,split)
    write_csv(output/'exclusions.csv',excluded,['path','reason','detail'])
    counts=collections.Counter((r['task'],r['split'],r['source'],r['label']) for r in rows)
    write_csv(output/'counts.csv',[dict(task=t,split=s,source=src,label=label,images=n)
              for (t,s,src,label),n in sorted(counts.items())],['task','split','source','label','images'])
    inputs=[audit/'glaucoma_candidate_worklist.csv',audit/'image_manifest.csv',
            ROOT/'data/prepared/1109/fiqs_quality.csv',edge_file,decision_file]
    release=dict(version='formal-research-v1',cohort_version=output.name,seed=seed,manifest_sha256=sha256(manifest),
        project_root_from_release=os.path.relpath(ROOT.resolve(),output.resolve()).replace('\\','/'),
        freezer_sha256=sha256(Path(__file__)),
        input_fingerprints={p.resolve().relative_to(ROOT.resolve()).as_posix():sha256(p) for p in inputs},
        exclusions_sha256=sha256(output/'exclusions.csv'),duplicate_edges_sha256=sha256(edge_file),
        technical_decisions_sha256=sha256(decision_file),images=len(rows),
        distinct_components=len({r['related_group'] for r in rows}),research_training_released=True,
        disease_proxy_labels_source_specific=True,all_patient_independence_proven=False,
        clinical_gradability_adjudication=False,quality_teacher_unseen_generalization_proven=False,
        known_fiqs_related_disease_components_excluded_from_final_test=True,
        splits='disease components 70/10/10/10 train/val/boundary/test; quality-only components 80/20; external sources locked',
        limitations=['AIROGS/Kaggle/ACRIMA/FIQS patient IDs unavailable',
          'near-duplicate audit is conservative and not exhaustive for all unseen transforms',
          'Luna review is technical sampling, not clinical label adjudication',
          'FIQS heldout is student-teacher fidelity, not independent teacher generalization',
          'EyeAugmented excluded pending parent lineage; mixed source proxy endpoints reported separately'])
    (output/'release.json').write_text(json.dumps(release,indent=2),encoding='utf-8')
    for task in ('glaucoma','quality'):
        for split in ('train','val'):load_rows(manifest,task,split,output/'release.json')
    print(json.dumps(release,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'data/prepared/formal')
    parser.add_argument('--edges',type=Path,required=True);parser.add_argument('--decisions',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=20261109)
    args=parser.parse_args();main(args.output,args.edges,args.decisions,args.seed)
