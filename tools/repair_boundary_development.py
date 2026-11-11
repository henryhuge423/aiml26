"""Create an immutable evaluation-only boundary subset after new linkage audit.

Original training, model_val, final tests, checkpoints and formal_v3 are kept.
Only boundary families linked to training/model_val are excluded, before any
boundary performance is computed.
"""
import collections
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from audit_eyeq_overlap import read,save_csv,digest


def repair_rows(rows,conflicts):
    blocked={r['related_group'] for r in conflicts if r['task']=='glaucoma' and r['split']=='boundary'}
    boundary=[r for r in rows if r['task']=='glaucoma' and r['split']=='boundary']
    return ([dict(r) for r in boundary if r['related_group'] not in blocked],
            [dict(r) for r in boundary if r['related_group'] in blocked])


def main():
    audit=ROOT/'data/audit/session_20261110'
    source=ROOT/'data/prepared/formal_v3/manifest.csv'
    conflicts_path=audit/'boundary_conflict_active_rows.csv'
    review=audit/'boundary_conflict_review.md'
    dest=ROOT/'data/prepared/formal_v3_boundary_repaired_20261110'
    if dest.exists():raise ValueError('immutable repaired boundary already exists')
    inputs=[source,ROOT/'data/prepared/formal_v3/release.json',conflicts_path,review,
        audit/'boundary_conflict_reconstruction.json',audit/'reconstruct_boundary_conflicts.py',audit/'eyeq_overlap/final_summary.json',
        audit/'eyeq_overlap/confirmed_edges.csv',Path(__file__)]
    binding={p.relative_to(ROOT).as_posix():digest(p) for p in inputs}
    rows=read(source)
    conflicts=[dict(r,sample_id=r.get('sample_id') or r['active_sample_id'],path=r.get('path') or r['active_path'])
               for r in read(conflicts_path)]
    by_id={r['sample_id']:r for r in rows}
    for row in conflicts:
        for key in ('path','task','split','related_group','source'):
            assert by_id[row['sample_id']][key]==row[key],(row['sample_id'],key)
    clean,removed=repair_rows(rows,conflicts)
    assert clean and removed and len(clean)+len(removed)==1232
    assert len({r['related_group'] for r in removed})==2
    assert {r['label'] for r in clean}=={'0','1'}
    assert all(by_id[r['sample_id']]==r for r in clean+removed)
    reconstruction=json.loads((audit/'boundary_conflict_reconstruction.json').read_text())
    removed_ids={r['sample_id'] for r in removed}
    assert not reconstruction['all_test_cross_roles']
    for component in reconstruction['all_active_cross_role_components']:
        assert len({r['split'] for r in component['rows'] if r['sample_id'] not in removed_ids})<=1
    assert binding=={p.relative_to(ROOT).as_posix():digest(p) for p in inputs}
    dest.mkdir();save_csv(dest/'manifest.csv',clean,list(rows[0]))
    removed_report=[dict(r,reason='new_EyeQ_identity_or_patient_family_link_to_train_or_model_val') for r in removed]
    save_csv(dest/'exclusions.csv',removed_report,list(removed_report[0]))
    release=dict(version='evaluation-only-boundary-v1',cohort_version=dest.name,
        evaluation_released=True,research_training_released=False,
        original_training_cohort='formal_v3',original_boundary_rows=1232,boundary_rows=len(clean),
        excluded_boundary_rows=len(removed),excluded_boundary_groups=2,
        known_cross_role_components_remaining_after_boundary_exclusion=0,
        manifest_sha256=digest(dest/'manifest.csv'),exclusions_sha256=digest(dest/'exclusions.csv'),
        input_fingerprints=binding,training_model_val_and_final_tests_unchanged=True,
        checkpoint_retraining_required=False,
        reason='New EyeQ content/patient bridges connected two boundary groups to existing training/model_val. Exclude whole boundary groups before boundary inference; do not alter diagnosis truth or select exclusions by predictions.',
        counts_by_source=dict(collections.Counter(r['source'] for r in clean)))
    (dest/'release.json').write_text(json.dumps(release,indent=2))
    print(json.dumps(release,indent=2))


if __name__=='__main__':main()
