"""Cache teacher scores only for an explicitly frozen, allowed quality cohort."""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.data import load_rows,sha256
from training.teacher import FTHNetTeacher,PREPROCESS_VERSION,implementation_fingerprint
from training.cache import TeacherCacheStore,validated_reuse_rows


def main(manifest,release,output,crop_batch=8,seed=20261109,reuse_from=None,seed_only=False):
    import torch
    torch.set_num_threads(4)
    if crop_batch<1:raise ValueError('crop_batch must be positive')
    rows=load_rows(manifest,'quality','train',release)+load_rows(manifest,'quality','val',release)
    fingerprint=dict(manifest_sha256=sha256(manifest),release_sha256=sha256(release),
        teacher_sha256=sha256(ROOT/'models/checkpoints/net_g_226264S4.pth'),
        preprocessing=PREPROCESS_VERSION,seed=seed,crops=20,checkpoint_key='params',
        layout_compatibility='original_conv_outputs_made_contiguous_before_view',crop_batch=crop_batch,
        implementation=implementation_fingerprint(ROOT))
    meta_path=output.with_suffix('.json');existing={}
    journal_path=output.with_suffix('.sqlite')
    if output.exists() and not journal_path.exists():raise ValueError('CSV has no authoritative cache journal')
    journal=TeacherCacheStore(journal_path,fingerprint)
    existing=journal.records()
    lineage=None
    if meta_path.exists():lineage=json.loads(meta_path.read_text(encoding='utf-8')).get('reuse_provenance')
    if reuse_from is not None:
        old_meta=json.loads(reuse_from.with_suffix('.json').read_text(encoding='utf-8'))
        if old_meta.get('complete') is not True or sha256(reuse_from)!=old_meta['cache_sha256']:
            raise ValueError('source cache incomplete/modified')
        for key,value in fingerprint.items():
            if key not in {'manifest_sha256','release_sha256'} and old_meta.get(key)!=value:
                raise ValueError('source teacher cache computation fingerprint mismatch')
        if sha256(reuse_from.parent/'manifest.csv')!=old_meta['manifest_sha256'] or sha256(reuse_from.parent/'release.json')!=old_meta['release_sha256']:
            raise ValueError('source cache release provenance mismatch')
        with reuse_from.open(encoding='utf-8-sig') as f:old_rows=list(csv.DictReader(f))
        previous={r['sample_id']:r for r in old_rows}
        if len(previous)!=len(old_rows):raise ValueError('duplicate identity in source cache')
        reused=validated_reuse_rows(rows,previous,allow_missing=True)
        for row in reused:
            journal.put(row);existing[row['sample_id']]=row
        lineage=dict(source_cache=str(reuse_from.resolve()),source_cache_sha256=sha256(reuse_from),
                     source_manifest_sha256=old_meta['manifest_sha256'],source_release_sha256=old_meta['release_sha256'],
                     rows_rebound=len(reused),proof='identical allowed sample_id/path/image_sha256/split and teacher computation fingerprint')
    if any(not math.isfinite(float(r['teacher_q'])) for r in existing.values()):
        journal.close();raise ValueError('nonfinite journaled teacher target')
    for r in rows:
        if r['sample_id'] in existing and existing[r['sample_id']]['sha256']!=r['sha256']:
            raise ValueError('cached image identity mismatch')
    allowed={r['sample_id'] for r in rows}
    if not set(existing).issubset(allowed):raise ValueError('cache contains non-allowed image identity')
    output.parent.mkdir(parents=True,exist_ok=True)
    fields=['sample_id','path','sha256','split','teacher_q','crop_std']

    def export(complete):
        temporary=output.with_suffix('.csv.tmp')
        with temporary.open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            writer.writerows(existing[key] for key in sorted(existing))
        temporary.replace(output)
        meta=dict(fingerprint,complete=complete,rows=len(existing),expected_rows=len(rows),
            cache_sha256=sha256(output),label_type='teacher_prediction_not_human_truth')
        if lineage is not None:meta['reuse_provenance']=lineage
        temp_meta=meta_path.with_suffix('.json.tmp')
        temp_meta.write_text(json.dumps(meta,indent=2),encoding='utf-8');temp_meta.replace(meta_path)

    if len(existing)==len(rows):
        export(True);journal.close()
        print(json.dumps(dict(cached=len(existing),complete=True,reused_without_recomputation=True)),flush=True)
        return
    export(False)
    if seed_only:
        journal.close()
        print(json.dumps(dict(cached=len(existing),expected=len(rows),complete=False,seed_only=True)),flush=True)
        return
    teacher=FTHNetTeacher(ROOT);started=time.time()
    try:
        for i,r in enumerate(rows,1):
            if r['sample_id'] in existing:continue
            path=(ROOT/'data'/r['path']).resolve()
            if not path.is_relative_to((ROOT/'data').resolve()) or sha256(path)!=r['sha256']:
                raise ValueError('source image changed before teacher prediction')
            q,std=teacher.predict(path,r['sha256'],seed,crop_batch)
            row=dict(sample_id=r['sample_id'],path=r['path'],sha256=r['sha256'],split=r['split'],teacher_q=q,crop_std=std)
            journal.put(row);existing[r['sample_id']]=row
            if i%100==0:
                export(False)
                print(f'teacher cached {len(existing)}/{len(rows)} in {time.time()-started:.1f}s',flush=True)
    finally:
        export(len(existing)==len(rows));journal.close()
    print(json.dumps(dict(cached=len(existing),elapsed_seconds=time.time()-started,complete=len(existing)==len(rows))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--crop-batch',type=int,default=20)
    parser.add_argument('--reuse-from',type=Path,help='Rebind only identical allowed quality targets from a complete verified cache')
    parser.add_argument('--seed-only',action='store_true',help='Bind reusable targets and export incomplete cache without starting GPU scoring')
    args=parser.parse_args();main(args.manifest,args.release,args.output,args.crop_batch,reuse_from=args.reuse_from,seed_only=args.seed_only)
