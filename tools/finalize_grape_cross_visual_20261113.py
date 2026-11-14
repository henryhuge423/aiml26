"""Recorded visual same-eye confirmation of three GRAPE/FIVES associations."""
import collections
import json
from pathlib import Path
import audit_lag_overlap_20261113 as C

NOTES={
    ('118_OS_4.jpg','420_G.png'):'Same superior arcade with multiple crossing/branch junctions, inferior vessel route and disc-relative temporal pattern; exposure and a small image label differ.',
    ('140_OS_2.jpg','333_G.png'):'Same disc exit branching, paired inferior temporal vessels and tertiary retinal bifurcations across the fundus; exposure and localized bright reflection differ.',
    ('46_OD_4.jpg','356_G.png'):'Same superior double vascular arch, inferior branching arcade and the disc-relative macular vessel pattern; framing and illumination differ.'}

def main():
    source=C.OUT/'grape_candidate_metrics.csv'
    original=C.A.read(source)
    assert len(original)==105499
    annotations=[];final=[]
    for r in original:
        key=(Path(r['path_a']).name,Path(r['path_b']).name)
        item=dict(r,algorithmic_decision=r['decision'],review_basis='retinal_interior_algorithm',visual_note='')
        if key in NOTES:
            assert r['decision']=='ambiguous_exclude',r
            item.update(decision='confirmed_same_photo',final_decision='confirmed_same_photo',
                review_basis='visual_retinal_vessel_identity_across_sources',visual_note=NOTES[key])
            annotations.append(dict(path_a=r['path_a'],path_b=r['path_b'],prior_decision=r['decision'],
                decision='confirmed_same_photo',final_decision='confirmed_same_photo',
                relation='same_fundus_or_eye_image_association',review_basis=item['review_basis'],note=NOTES[key]))
        final.append(item)
    assert len(annotations)==3
    C.write('grape_visual_adjudications.csv',annotations)
    C.write('grape_visual_confirmed_edges.csv',[dict(path_a=r['path_a'],path_b=r['path_b'],decision=r['decision']) for r in annotations],['path_a','path_b','decision'])
    C.write('grape_final_candidate_metrics.csv',final)
    summary=dict(completed=True,visual_cross_source_pairs_reviewed=3,visually_confirmed_same_eye_pairs=3,
        current_candidate_decisions=dict(collections.Counter(r['decision'] for r in final)),
        known_same_eye_metadata_handling='The remaining36 same-subject/same-eye visits are handled by the combined audit with explicit clinical metadata, not newly claimed identical photographs here.',
        clinical_labels_changed=False,models_scored=False,
        inputs={str(p.relative_to(C.ROOT)):C.A.sha(p) for p in [source,C.OUT/'grape_cross_identity_review.csv',C.OUT/'grape_cross_identity_visual.png',C.OUT/'grape_visual_adjudications.csv',C.OUT/'grape_visual_confirmed_edges.csv',C.OUT/'grape_final_candidate_metrics.csv',Path(__file__)]})
    (C.OUT/'grape_cross_visual_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
