"""Apply the recorded visual fundus-identity decisions to19 LAG candidates.

This confirms image/eye association, without altering clinical labels.
"""
import collections
import json
from pathlib import Path
import audit_lag_overlap_20261113 as C

REVIEWED=[
    ('0293','4826','Matching faint superior vascular arc and disc-adjacent branching despite blur; the fundus internal pattern matches.'),
    ('1525','3651','Matching inferior and superior vessel routing, temporal branching and disc-relative macular position.'),
    ('2105','5738','Matching upper arcade crossings and temporal vessel bifurcations across the fundus interior.'),
    ('2534','4902','Matching irregular disc-adjacent bright texture and several superior/inferior branching vessels.'),
    ('2946','3626','Matching vessel bifurcations around the disc and the radial internal texture despite different illumination.'),
    ('4724','4772','Matching disc vascular trunk and temporal arcade branching throughout the retina.'),
    ('0205','1677','Matching disc exit vessels and multiple temporal branching locations; a localized reflection differs.'),
    ('0296','3570','Matching disc-adjacent texture, inferior vascular routing and a dark internal retinal patch.'),
    ('0773','2993','Matching disc-relative vascular routing and temporal bifurcations despite exposure differences.'),
    ('0854','4280','Matching superior/inferior vessel exits, local curving vessels and internal texture.'),
    ('1069','4340','Matching dark temporal internal patch, neighboring vascular arc and disc-adjacent bright texture.'),
    ('1531','2325','Matching three main vessel exits and tertiary temporal branches; crop and illumination differ.'),
    ('1652','4866','Matching disc vascular crossing, superior branching and disc-adjacent bright texture.'),
    ('1913','5539','Matching disc exit pattern and the temporal vessel route; localized bright reflection differs.'),
    ('2015','2075','Matching upper arch and the inferior temporal vascular bifurcations despite uneven illumination.'),
    ('2087','3087','Matching superior/inferior vascular routing and temporal branch junctions.'),
    ('2908','4929','Matching distinctive upper branching arc, disc exit routing and inferior vessel despite blur.'),
    ('4365','5616','Matching temporal dark retinal patch, horizontal vessel route and irregular internal texture.'),
    ('4563','5358','Matching disc exit vessels and branching temporal arcade despite mild illumination differences.')]

def main():
    original=C.A.read(C.OUT/'lag_candidate_metrics.csv')
    reviewed={(a+'.jpg',b+'.jpg'):note for a,b,note in REVIEWED}
    excluded=[r for r in original if r['decision']=='ambiguous_exclude']
    assert len(excluded)==len(reviewed)==19
    annotations=[]; final=[]
    for r in original:
        item=dict(r,algorithmic_decision=r['decision'],review_basis='retinal_interior_algorithm',visual_note='')
        if r['decision']=='ambiguous_exclude':
            key=(Path(r['path_a']).name,Path(r['path_b']).name)
            assert key in reviewed,key
            assert Path(r['path_a']).parents[1].name==Path(r['path_b']).parents[1].name
            note=reviewed[key]
            item.update(decision='confirmed_same_photo',final_decision='confirmed_same_photo',
                review_basis='visual_retinal_vessel_and_local_structure_identity',visual_note=note)
            annotations.append(dict(path_a=r['path_a'],path_b=r['path_b'],
                prior_decision=r['decision'],decision='confirmed_same_photo',
                relation='same_fundus_or_eye_image_association',review_basis=item['review_basis'],note=note,
                inner_inliers=r['inner_inliers'],inner_coverage_a=r['inner_coverage_a'],
                inner_coverage_b=r['inner_coverage_b'],aligned_correlation=r['aligned_correlation'],
                vessel_highpass_correlation=r['vessel_highpass_correlation']))
        final.append(item)
    C.write('lag_visual_adjudications.csv',annotations)
    edges=[dict(path_a=r['path_a'],path_b=r['path_b'],decision='confirmed_same_photo') for r in annotations]
    C.write('lag_manual_confirmed_edges.csv',edges,['path_a','path_b','decision'])
    C.write('lag_final_candidate_metrics.csv',final)
    all_edges=[dict(path_a=r['path_a'],path_b=r['path_b'],decision='confirmed_same_photo')
        for r in final if r['decision']=='confirmed_same_photo']
    C.write('lag_all_final_confirmed_edges.csv',all_edges,['path_a','path_b','decision'])
    assert len(all_edges)==42 and not any(r['decision']=='ambiguous_exclude' for r in final)
    summary=dict(completed=True,automatic_confirmed_pairs=23,visually_confirmed_additional_pairs=19,
        final_candidate_pairs=len(final),final_candidate_decisions=dict(collections.Counter(r['decision'] for r in final)),
        visual_review='All19 automated high-evidence exclusions inspected on four local contact sheets. Internal vessel bifurcations, disc-relative vascular routing and local texture match; blur, exposure and glare reduce high-pass correlation. Camera aperture similarity alone was not used.',
        labels_changed=False,models_scored=False,
        relation_interpretation='Same fundus/eye association can include repeated captures, not necessarily byte-identical photographs. Keep the linked group together and one canonical image for independent-image counts.',
        inputs={str(p.relative_to(C.ROOT)):C.A.sha(p) for p in [C.OUT/'lag_candidate_metrics.csv',C.OUT/'lag_visual_adjudications.csv',C.OUT/'lag_final_candidate_metrics.csv',C.OUT/'lag_manual_confirmed_edges.csv',C.OUT/'lag_all_final_confirmed_edges.csv',
            *[C.OUT/f'lag_excluded_visual_{i:02d}.png' for i in range(1,5)],Path(__file__)]})
    (C.OUT/'lag_visual_confirmation_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
