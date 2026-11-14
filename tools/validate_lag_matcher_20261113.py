"""Check registration and rejection controls before using LAG audit edges."""
import json
from pathlib import Path
import cv2
from PIL import Image, ImageEnhance
import audit_lag_overlap_20261113 as C

def main():
    cv2.setNumThreads(1);cv2.setRNGSeed(20261113)
    directory=C.ROOT/'.test_tmp/lag_matcher_validation'
    directory.mkdir(parents=True,exist_ok=True)
    C.DESC_DIR=directory/'descriptors'
    C.initialize_matcher()
    originals=C.load_features(C.OUT/'lag_features.jsonl')
    source=Path(originals[0]['path'])
    with Image.open(source) as im:
        im.load();im=im.convert('RGB')
        width,height=im.size
        variants={
            'jpeg_reencoded':im.copy(),
            'mirror':im.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
            'rotate90':im.transpose(Image.Transpose.ROTATE_90),
            'resize75pct':im.resize((int(width*.75),int(height*.75))),
            'crop5pct':im.crop((int(width*.05),int(height*.05),int(width*.95),int(height*.95))),
            'brightness1p1':ImageEnhance.Brightness(im).enhance(1.1)}
    result=[]
    for name,im in variants.items():
        output=directory/(name+'.jpg');im.save(output,quality=94)
        a=C.A.thumbnail(str(source));b=C.A.thumbnail(str(output))
        distances=[(C.A.A.phash(C.A.A.orient(a,o))^C.A.A.phash(b)).bit_count() for o in range(8)]
        orientation=min(range(8),key=lambda o:distances[o])
        r=C.compare(dict(path_a=str(source),path_b=str(output),hash_distance=distances[orientation],orientation=orientation))
        result.append(dict(control=name,expected='confirmed_same_photo',candidate_radius6=distances[orientation]<=6,**r))
        assert r['decision']=='confirmed_same_photo',(name,r)
    weak=C.A.read(C.OUT/'local_weak_pairs_preview.csv')[:2]
    for i,r in enumerate(weak):
        p=C.compare({k:r[k] for k in ['path_a','path_b','hash_distance','orientation']})
        result.append(dict(control='known_aperture_false_match_'+str(i),expected='not_confirmed_same_photo',**p))
        assert p['decision']!='confirmed_same_photo',p
    summary=dict(controls=len(result),same_photo_transforms=len(variants),aperture_false_match_controls=len(weak),
        expected_decisions_passed=True,results=result,
        matcher_sha256=C.A.sha(C.OUT/'lag_matcher_source.py'),helper_sha256=C.A.sha(Path(C.__file__)),
        note='Controlled same-photo transforms and previously identified aperture false correspondences; this validation does not assert arbitrary crop retrieval or unknown patient independence.')
    (C.OUT/'lag_matcher_validation.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(dict(controls=len(result),passed=True,candidate_hits=sum(r.get('candidate_radius6',False) for r in result))),flush=True)

if __name__=='__main__':main()
