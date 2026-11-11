"""Non-destructive technical selections, explicitly separate from approved training splits."""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
from openpyxl import load_workbook


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def checked_path(data_root, relative):
    path = (data_root / relative).resolve()
    if not path.is_relative_to(data_root.resolve()) or not path.is_file():
        raise ValueError(f'invalid or missing source path: {relative}')
    return path


def verify_source(data_root, relative, inventory):
    path=checked_path(data_root,relative)
    sha=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):sha.update(block)
    if sha.hexdigest()!=inventory[relative]['sha256']:
        raise ValueError(f'source hash mismatch since audit: {relative}')


def build_manifests(data_root: Path, audit_root: Path, output: Path) -> dict:
    inventory = {r['path']: r for r in read_csv(audit_root/'image_manifest.csv')}
    work = read_csv(audit_root/'glaucoma_candidate_worklist.csv')
    groups = collections.defaultdict(list)
    for r in work:
        if r['source'] == 'hygd-1.1.0':groups[r['sha256']].append(r)
    canonical, aliases = [], []
    for sha, members in sorted(groups.items()):
        if len({r['target'] for r in members}) != 1:
            raise ValueError(f'conflicting HYGD labels for {sha}')
        if len({r['related_split_group'] for r in members}) != 1:
            raise ValueError(f'conflicting HYGD related groups for {sha}')
        members.sort(key=lambda r:r['path'])
        for r in members:
            verify_source(data_root,r['path'],inventory)
            if inventory[r['path']]['status'] != 'full_decode_ok':
                raise ValueError(f'undecodable HYGD image: {r["path"]}')
        chosen = dict(members[0])
        chosen['train_eligible']='False'
        chosen['split']='UNASSIGNED'
        canonical.append(chosen)
        if len(members)>1:
            aliases.append(dict(sha256=sha, canonical_path=chosen['path'],
                alias_paths='|'.join(r['path'] for r in members[1:]),
                recorded_patient_groups='|'.join(sorted({r['patient_group'] for r in members})),
                related_split_group=chosen['related_split_group']))
    wb=load_workbook(data_root/'external/fiqs/meta_data.xlsx',read_only=True,data_only=True)
    values=iter(wb.active.values); headers=next(values)
    quality=[]; names=set()
    try:
        for values_row in values:
            meta=dict(zip(headers,values_row)); name=str(meta['file_name'])
            if name in names:raise ValueError(f'duplicate FIQS image ID: {name}')
            names.add(name)
            rel='external/fiqs/images/'+name
            verify_source(data_root,rel,inventory)
            if inventory[rel]['status']!='full_decode_ok':
                raise ValueError(f'undecodable FIQS selected image: {rel}')
            mos=float(meta['meanOpinionScore'])
            if not 0<=mos<=100:raise ValueError(f'invalid FIQS MOS for {name}')
            quality.append(dict(image_id=Path(name).stem,path=rel,sha256=inventory[rel]['sha256'],
                human_mos=mos,human_mos_normalized=mos/100,quality_level=meta['qualityLevel'],
                label_source='FIQS human MOS',related_split_group='FIQS:'+Path(name).stem,
                split='UNASSIGNED',train_eligible='False',
                pending_reasons='patient_group_unknown|teacher_training_exposure_unknown|cross_source_overlap_pending'))
    finally:wb.close()
    excluded=[dict(path=r['path'],reason='strict_full_decode_failed',sha256=r['sha256'])
              for r in inventory.values() if r['status']!='full_decode_ok']
    output.mkdir(parents=True,exist_ok=True)
    write_csv(output/'hygd_canonical.csv',canonical,list(work[0]))
    write_csv(output/'hygd_aliases.csv',aliases,
              ['sha256','canonical_path','alias_paths','recorded_patient_groups','related_split_group'])
    write_csv(output/'fiqs_quality.csv',quality,list(quality[0]))
    write_csv(output/'technical_exclusions.csv',excluded,['path','reason','sha256'])
    summary=dict(hygd_original_rows=sum(map(len,groups.values())),hygd_unique_images=len(canonical),
        hygd_duplicate_groups=len(aliases),fiqs_unique_image_ids=len(quality),
        selected_fiqs_resolution='1024 version only',technical_decode_exclusions=len(excluded),
        original_files_modified=0,selected_source_hashes_reverified=True,
        split_frozen=False,formal_training_eligible=False,
        note='Technical selections only. Label, gradability, patient independence and teacher exposure still require review.')
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=root/'data')
    parser.add_argument('--audit-root',type=Path,default=root/'data/audit/2026-11-09')
    parser.add_argument('--output',type=Path,default=root/'data/prepared/1109')
    args=parser.parse_args()
    print(json.dumps(build_manifests(args.data_root,args.audit_root,args.output),indent=2))
