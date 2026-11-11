"""Freeze the reviewed 1109 proposal without changing its rows or old releases."""
import collections
import csv
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training.data import load_rows,sha256


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def main():
    proposal=ROOT/'data/audit/expansion_20261109/split_proposal'
    old=ROOT/'data/prepared/formal_v2'
    dest=ROOT/'data/prepared/formal_v3'
    if dest.exists():raise ValueError('immutable destination exists')
    info=json.loads((proposal/'proposal.json').read_text())
    assert sha256(proposal/'manifest_proposed.csv')==info['proposed_manifest_sha256']
    for path,digest in info['inputs'].items():assert sha256(ROOT/path)==digest,path
    rows=read(proposal/'manifest_proposed.csv')
    expected={'glaucoma/train':10354,'glaucoma/val':1190,'glaucoma/boundary':1232,
        'glaucoma/test':1233,'glaucoma/external_acrima':705,'glaucoma/external_hygd':737,
        'quality/train':12147,'quality/val':1642}
    assert dict(collections.Counter(r['task']+'/'+r['split'] for r in rows))==expected
    original={r['sample_id']:r for r in read(old/'manifest.csv')}
    upstream={r['file_name']:r['folder'] for r in read(ROOT/'data/external/airogs-light-v2/eyepac-light-v2-512-jpg/metadata.csv')}
    for row in rows:
        if row['task']=='glaucoma':
            for field in ('path','sha256','label','original_label','target_definition','related_group'):
                assert row[field]==original[row['sample_id']][field],(row['sample_id'],field)
            if row['source']=='airogs-light-v2':
                role=upstream[Path(row['path']).name]
                assert row['split']=={'test':'test','validation':'val'}.get(role,
                    'boundary' if original[row['sample_id']]['split']=='test' else 'train')
            else:assert row['split']==original[row['sample_id']]['split']
        elif row['source']!='fiqs':
            assert row['label']=='' and row['original_label']==''
    groups=collections.defaultdict(set)
    path_roles=collections.defaultdict(set)
    for row in rows:
        groups[row['related_group']].add(row['split'])
        path_roles[row['path']].add(row['split'])
    assert all(len(v)==1 for v in groups.values())
    edges=ROOT/'data/audit/deep/release_inputs/confirmed_edges_v1.csv'
    for edge in read(edges):
        roles=path_roles[edge['path_a']]|path_roles[edge['path_b']]
        assert len(roles)<=1,edge
    unique={r['path']:r['sha256'] for r in rows}
    def verify(item):
        relative,digest=item
        path=(ROOT/'data'/relative).resolve()
        assert path.is_relative_to((ROOT/'data').resolve())
        assert sha256(path)==digest,relative
        with Image.open(path) as image:image.load()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,_ in enumerate(pool.map(verify,unique.items()),1):
            if i%2000==0:print(f'verified hash/decode {i}/{len(unique)}',flush=True)
    release=json.loads((old/'release.json').read_text())
    release.update(cohort_version='formal_v3',manifest_sha256=info['proposed_manifest_sha256'],
        freezer_sha256=sha256(Path(__file__)),images=len(rows),
        distinct_components=len(groups),splits='AIROGS uploader test and included validation preserved; upstream train 90/10 train/boundary; other sources unchanged',
        quality_supervision='frozen FTHNet continuous teacher targets; no EyeQ originals or ordinal labels available',
        training_readiness_requires='complete bound cache, current configs and real GPU preflight; release alone is not readiness')
    additional=[old/'release.json',old/'manifest.csv',proposal/'manifest_proposed.csv',proposal/'proposal.json',
        ROOT/'data/audit/expansion_20261109/proposal_review/verification.json',edges,
        ROOT/'data/external/airogs-light-v2/eyepac-light-v2-512-jpg/metadata.csv',Path(__file__)]
    release['input_fingerprints'].update({p.relative_to(ROOT).as_posix():sha256(p) for p in additional})
    dest.mkdir()
    shutil.copyfile(proposal/'manifest_proposed.csv',dest/'manifest.csv')
    shutil.copyfile(old/'exclusions.csv',dest/'exclusions.csv')
    (dest/'release.json').write_text(json.dumps(release,indent=2),encoding='utf-8')
    for key,count in expected.items():
        task,split=key.split('/')
        assert len(load_rows(dest/'manifest.csv',task,split,dest/'release.json'))==count
    report=dict(passed=True,counts=expected,manifest_sha256=sha256(dest/'manifest.csv'),
        unique_paths_hash_and_full_decode_passed=len(unique),group_role_crossings=0,
        quality_protected_role_rows=0,upstream_test_retained=770,upstream_validation_retained=769,
        eyeq_rows=0,teacher_targets_required=13789,teacher_cache_complete=False)
    (dest/'cohort_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
