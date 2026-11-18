"""Check an alternative quality checkpoint on development photographs using CPU."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.data import load_rows,sha256
from training.eyeq_quality import QualityModel
from training.models import create_model
from tools.score_repaired_boundary import InferenceRows,read


def infer(model,rows,size):
    output={}
    dataset=InferenceRows(rows,ROOT/'data',size,'quality')
    loader=torch.utils.data.DataLoader(dataset,batch_size=16,num_workers=0)
    with torch.inference_mode():
        for x,ids in loader:
            q,logits=model(x);grade=2-(logits>0).sum(dim=1)
            assert torch.isfinite(q).all()
            for identity,value,estimate in zip(ids,q.tolist(),grade.tolist()):output[identity]=(value,estimate)
            if len(output)%320==0:print(f'scored {len(output)}/{len(rows)}',flush=True)
    return output


def main():
    torch.set_num_threads(2)
    torch.backends.mkldnn.enabled=True
    config=json.loads((ROOT/'configs/eyeq_quality_v1.json').read_text())
    directory=ROOT/config['run_dir'];checkpoint=directory/'best.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    status=json.loads((ROOT/'runs/eyeq_quality_v1/status.json').read_text())
    assert state['fingerprint']==status['result']['fingerprint']
    model=QualityModel(create_model(config['architecture'],ROOT/'models/checkpoints'))
    model.load_state_dict(state['model'],strict=True);model.eval()
    anchors=read(directory/'best_validation_predictions.csv')[:16]
    val=load_rows(ROOT/config['manifest'],'quality','val',ROOT/config['release'])
    by_id={r['sample_id']:r for r in val}
    scores=infer(model,[by_id[r['sample_id']] for r in anchors],config['image_size'])
    anchor_diff=max(abs(scores[r['sample_id']][0]-float(r['q'])) for r in anchors)
    # Confirm the adapter against the native validation path on the same device.
    # Archived predictions used CUDA; its convolution numerics differ from CPU.
    from training.engine import loader
    native={}
    anchor_rows=[dict(by_id[r['sample_id']],teacher_q=0.) for r in anchors]
    with torch.inference_mode():
        for x,_,ids in loader(anchor_rows,dict(config,device='cpu',num_workers=0),ROOT,False):
            values,_=model(x)
            native.update(zip(ids,values.tolist()))
    adapter_diff=max(abs(native[i]-scores[i][0]) for i in native)
    assert adapter_diff==0 and anchor_diff<1e-3,(adapter_diff,anchor_diff)
    assert all(not r['predicted_grade'] or scores[r['sample_id']][1]==int(r['predicted_grade']) for r in anchors)
    manifest=ROOT/'data/prepared/expanded_20261113/manifest.csv';release=manifest.with_name('release.json')
    rows=sum((load_rows(manifest,'glaucoma',role,release) for role in ('val','boundary')),[])
    scores=infer(model,rows,config['image_size']);assert len(scores)==len(rows)==4099
    out=ROOT/'runs/followup_20261115/eyeq_development';out.mkdir(parents=True,exist_ok=True)
    path=out/'quality_scores.csv'
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['sample_id','source','label','split','q_eyeq','predicted_eyeq_grade'])
        writer.writeheader();writer.writerows(dict(sample_id=r['sample_id'],source=r['source'],label=r['label'],split=r['split'],
            q_eyeq=scores[r['sample_id']][0],predicted_eyeq_grade=scores[r['sample_id']][1]) for r in rows)
    report=dict(rows=len(rows),device='cpu',checkpoint_sha256=sha256(checkpoint),training_fingerprint=state['fingerprint'],
        selected_epoch=state['state']['epoch'],anchor_max_abs_difference=anchor_diff,
        same_cpu_native_adapter_max_abs_difference=adapter_diff,archived_anchor_device='cuda',
        manifest_sha256=sha256(manifest),scores_sha256=sha256(path),protected_test_inference=False,
        score_kind='EyeQ_and_FTHNet_joint_raw_regression_not_quality_probability')
    (out/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
