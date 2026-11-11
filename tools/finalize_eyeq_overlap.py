"""Reconcile the completed pixel screen with ALL historical known links.

Outputs an unreleased quality-only proposal. Does not change formal_v3, copy
images, compute teacher labels or use final-test disease predictions.
"""
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from audit_eyeq_overlap import read,save_csv,digest,component_rows


def known_edges(work,quality):
    result=[];patients={}
    for row in work:
        patient=row.get('patient_group','UNKNOWN')
        if not patient or patient=='UNKNOWN':continue
        key=(row['source'],patient)
        if key in patients:result.append((row['path'],patients[key]))
        else:patients[key]=row['path']
    for row in quality:
        result.append((row['path'],row['path'].replace('/images/','/full_size_images/')))
    return result


def verify_screen_inputs(screen,root=ROOT):
    if json.loads((screen/'status.json').read_text()).get('stage')!='completed':
        raise ValueError('pixel screen not complete')
    bindings=json.loads((screen/'inputs.json').read_text())
    if not bindings:raise ValueError('pixel screen input bindings are missing')
    for relative,expected in bindings.items():
        path=(root/relative).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path)!=expected:
            raise ValueError(f'pixel screen input changed: {relative}')
    return bindings


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screen',type=Path,default=ROOT/'data/audit/session_20261110/eyeq_overlap')
    args=parser.parse_args();out=args.screen
    verify_screen_inputs(out)
    inputs=[ROOT/'data/audit/eyeq_acquisition_20261110/imported_originals.csv',
        ROOT/'data/audit/2026-11-09/image_manifest.csv',ROOT/'data/prepared/formal_v3/manifest.csv',
        ROOT/'data/audit/deep/release_inputs/confirmed_edges_v1.csv',
        ROOT/'data/audit/2026-11-09/glaucoma_candidate_worklist.csv',ROOT/'data/prepared/1109/fiqs_quality.csv',
        out/'confirmed_edges.csv',out/'unresolved_candidates.csv',Path(__file__),ROOT/'tools/audit_eyeq_overlap.py',
        out/'inputs.json',out/'summary.json',out/'status.json']
    bindings={str(p.relative_to(ROOT)):digest(p) for p in inputs}
    originals=read(inputs[0]);historical=read(inputs[1]);active=read(inputs[2]);work=read(inputs[4]);quality=read(inputs[5])
    def absolute(path):return str((ROOT/'data'/path).resolve())
    for row in historical:row['path']=absolute(row['path'])
    for row in active:row['path']=absolute(row['path'])
    prior=[(absolute(r['path_a']),absolute(r['path_b'])) for r in read(inputs[3])]
    additional=known_edges(work,quality);prior.extend((absolute(a),absolute(b)) for a,b in additional)
    confirmed=read(inputs[6]);uncertain=read(inputs[7])
    rows=component_rows(originals+historical,active,[(r['path_a'],r['path_b']) for r in confirmed],prior)
    unresolved={r['path_a'] for r in uncertain}|{r['path_b'] for r in uncertain}
    blocked_components={r['component'] for r in rows if r['path'] in unresolved}
    for row in rows:
        row['unresolved_component']=row['component'] in blocked_components
        row['blocked_from_new_training']|=row['unresolved_component']
        ar=set(filter(None,row['active_roles'].split('|')))
        if row['eyeq_split']=='test':role='test'
        elif row['unresolved_component'] or row['official_split_conflict'] or row['active_role_conflict'] or ar-{'train','val'}:
            role='quarantine'
        elif ar:role=next(iter(ar))
        else:
            fraction=int(hashlib.sha256(f"20261109:{row['component']}".encode()).hexdigest()[:16],16)/2**64
            role='train' if fraction<.9 else 'val'
        row.update(proposed_role=role,proposal_eligible=role in {'train','val'},
            label_kind='EyeQ_human_ordinal_0_Good_1_Usable_2_Reject',disease_label='',
            teacher_target='not_computed',research_training_released=False)
    verify_screen_inputs(out)
    if bindings!={str(p.relative_to(ROOT)):digest(p) for p in inputs}:
        raise ValueError('finalizer input changed while computing associations')
    save_csv(out/'full_components.csv',rows,list(rows[0]))
    save_csv(out/'eyeq_quality_split_proposal.csv',rows,list(rows[0]))
    summary=json.loads((out/'summary.json').read_text())
    summary.update(full_historical_known_patient_and_fiqs_edges=len(additional),
        full_active_association_roles=dict(collections.Counter(r['active_roles'] or 'none' for r in rows)),
        full_active_role_conflict_components=len({r['component'] for r in rows if r['active_role_conflict']}),
        full_official_split_conflict_components=len({r['component'] for r in rows if r['official_split_conflict']}),
        full_eyeq_test_development_exposure=sum(r['eyeq_split']=='test' and r['formal_v3_development_exposure'] for r in rows),
        proposed_roles=dict(collections.Counter(r['proposed_role'] for r in rows)),
        unresolved_eyeq_components=len(blocked_components),
        proposal_note='90/10 group-hash split of unassociated official train; preserve linked active train/val; protected/uncertain components quarantined. EyeQ official test retained. No teacher cache, ordinal training or research release.',
        finalizer_input_fingerprints=bindings)
    (out/'final_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
