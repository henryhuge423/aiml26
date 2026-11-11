"""Frozen formal_v3 best checkpoints on the repaired boundary development set.

Raw p_A, p_B and q only. No threshold/boundary selection and no final-test
inference. Original checkpoint training fingerprints remain authoritative.
"""
import csv
import json
import sys
from pathlib import Path
import torch
from PIL import Image,ImageOps

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training.data import ImageRows,sha256,load_rows
from training.engine import fingerprint_inputs
from training.models import create_model


class InferenceRows(torch.utils.data.Dataset):
    def __init__(self,rows,data_root,size,task):
        self.base=ImageRows(rows,data_root,size,task,train=False)

    def __len__(self):return len(self.base.rows)

    def __getitem__(self,index):
        row=self.base.rows[index];path=(self.base.root/row['path']).resolve()
        if not path.is_relative_to(self.base.root):raise ValueError('image outside data root')
        if sha256(path)!=row['sha256']:raise ValueError(f'image source changed: {path}')
        with Image.open(path) as image:image.load();image=image.convert('RGB')
        if self.base.task=='glaucoma':image=ImageOps.pad(image,(max(image.size),max(image.size)),color=0)
        return self.base.transform(image),row['sample_id']


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def predict(model,rows,config):
    dataset=InferenceRows(rows,ROOT/'data',config['image_size'],config['task'])
    batches=torch.utils.data.DataLoader(dataset,batch_size=16,shuffle=False,num_workers=0)
    scores={};model.eval()
    with torch.inference_mode():
        for x,ids in batches:
            values=model(x.to('cuda')).flatten()
            if config['task']=='glaucoma':values=values.sigmoid()
            for sample_id,value in zip(ids,values.cpu().tolist()):
                if sample_id in scores:raise ValueError('duplicate inference identity')
                if not __import__('math').isfinite(value):raise ValueError('nonfinite prediction')
                scores[sample_id]=value
    return scores


def main():
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    cohort=ROOT/'data/prepared/formal_v3_boundary_repaired_20261110'
    implementation_digest=sha256(Path(__file__))
    release_digest=sha256(cohort/'release.json')
    release=json.loads((cohort/'release.json').read_text())
    assert release['evaluation_released'] and not release['research_training_released']
    assert sha256(cohort/'manifest.csv')==release['manifest_sha256']
    for relative,expected in release['input_fingerprints'].items():assert sha256(ROOT/relative)==expected,relative
    rows=read(cohort/'manifest.csv')
    assert len(rows)==release['boundary_rows'] and all(r['task']=='glaucoma' and r['split']=='boundary' for r in rows)
    audit=ROOT/'data/audit/session_20261110'
    completion=json.loads((audit/'completion_verification.json').read_text())
    out=ROOT/'runs/formal_v3/boundary_development_repaired_20261110'
    if out.exists():raise ValueError('immutable boundary inference output exists')
    out.mkdir();all_scores={};proof=[]
    for name,key in (('glaucoma_a','p_a'),('glaucoma_b_candidate','p_b'),('quality_student','q')):
        config=json.loads((ROOT/f'configs/formal_v3/{name}.json').read_text())
        checkpoint=ROOT/config['run_dir']/'best.pt'
        evidence=next(m for m in completion['models'] if m['name']==name)
        assert sha256(checkpoint)==evidence['best_checkpoint_sha256']
        content=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert content['fingerprint']==fingerprint_inputs(config,ROOT)==evidence['fingerprint']
        model=create_model(config['architecture'],ROOT/'models/checkpoints').to('cuda')
        model.load_state_dict(content['model'],strict=True);del content
        # Independent real-image anchors verify this inference adapter against
        # the already recomputed best-checkpoint model_val predictions.
        anchors=read(audit/f'selected_validation/{name}/validation_predictions.csv')[:16]
        val=load_rows(ROOT/config['manifest'],config['task'],'val',ROOT/config['release'])
        by_id={r['sample_id']:r for r in val}
        anchor_scores=predict(model,[by_id[r['sample_id']] for r in anchors],config)
        difference=max(abs(anchor_scores[r['sample_id']]-float(r['prediction'])) for r in anchors)
        assert difference<1e-6,(name,difference)
        scores=predict(model,rows,config)
        assert set(scores)=={r['sample_id'] for r in rows}
        all_scores[key]=scores
        item=dict(model=name,checkpoint_sha256=evidence['best_checkpoint_sha256'],training_fingerprint=evidence['fingerprint'],
            real_validation_anchor_max_abs_difference=difference,boundary_rows=len(scores))
        proof.append(item);print(json.dumps(item),flush=True)
        del model;torch.cuda.empty_cache()
    records=[dict(row,p_a=all_scores['p_a'][row['sample_id']],p_b=all_scores['p_b'][row['sample_id']],q=all_scores['q'][row['sample_id']]) for row in rows]
    path=out/'boundary_pq.csv'
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    verification=dict(passed=True,rows=len(records),manifest_sha256=release['manifest_sha256'],
        evaluation_release_sha256=release_digest,inference_implementation_sha256=implementation_digest,
        boundary_scores_sha256=sha256(path),models=proof,
        scores_only=True,classification_threshold_selected=False,review_boundary_selected=False,
        final_test_evaluated=False,quality_score_kind='FTHNet_distilled_student_raw_regression')
    assert sha256(Path(__file__))==implementation_digest
    assert sha256(cohort/'release.json')==release_digest
    (out/'verification.json').write_text(json.dumps(verification,indent=2))
    print(json.dumps(verification),flush=True)


if __name__=='__main__':main()
