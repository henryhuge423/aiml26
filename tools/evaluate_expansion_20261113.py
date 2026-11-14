"""Evaluate selected checkpoints on development roles; never load test images."""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
RUN=ROOT/'runs/expanded_20261113'


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def poor(row):
    return (row.get('source')=='fives' and row.get('quality_fives_any_poor_dimension','').lower()=='true') or (
        row.get('source')=='mshf' and row.get('quality_overall_official','')=='0')


class FixedStress:
    def __init__(self,condition):self.condition=condition
    def __call__(self,image):
        from PIL import ImageEnhance,ImageFilter
        from training.fundus_degradation import retinal_context
        if self.condition=='defocus_2_at512':
            return image.filter(ImageFilter.GaussianBlur(2*retinal_context(image)['field_diameter_pixels']/512))
        if self.condition=='exposure_070':return ImageEnhance.Brightness(image).enhance(.70)
        return image


def predict(model,rows,config,condition):
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms as T
    from training.data import ImageRows
    dataset=ImageRows(rows,ROOT/'data',config['image_size'],'glaucoma')
    dataset.transform=T.Compose([T.Resize((config['image_size'],config['image_size'])),FixedStress(condition),
        T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
    result=[];by_id={r['sample_id']:r for r in rows}
    generator=torch.Generator().manual_seed(20261113)
    with torch.inference_mode():
        for images,targets,ids in DataLoader(dataset,batch_size=32,num_workers=2,pin_memory=True,generator=generator):
            scores=model(images.to(config['device'])).flatten().sigmoid().cpu().tolist()
            for identity,score in zip(ids,scores):
                r=by_id[identity]
                result.append(dict(sample_id=identity,source=r['source'],label=int(r['label']),
                    related_group=r['related_group'],target_definition=r['target_definition'],
                    origin_cohort=r['origin_cohort'],split=r['split'],condition=condition,
                    real_poor_quality=poor(r),prediction=score))
    return result


def metrics(rows,threshold=.5):
    import numpy as np
    from sklearn.metrics import roc_auc_score,average_precision_score
    if not rows:return None
    y=np.array([int(r['label']) for r in rows]);p=np.array([float(r['prediction']) for r in rows]);positive=y==1
    decision=p>threshold
    both=len(set(y))==2
    ece=0.
    for i in range(10):
        selected=(p>=i/10)&(p<(i+1)/10 if i<9 else p<=1)
        if selected.any():ece+=selected.mean()*abs(p[selected].mean()-y[selected].mean())
    return dict(n=len(y),positives=int(positive.sum()),positive_groups=len({r['related_group'] for r in rows if int(r['label'])==1}),
        prevalence=float(y.mean()),auc=float(roc_auc_score(y,p)) if both else None,
        auprc=float(average_precision_score(y,p)) if both else None,
        brier=float(np.mean((p-y)**2)),ece_10_bins=float(ece),
        cross_entropy=float(-np.mean(y*np.log(np.clip(p,1e-7,1))+(1-y)*np.log(np.clip(1-p,1e-7,1)))),
        display_threshold=threshold,positive_misses=int((positive&~decision).sum()),
        sensitivity=float(decision[positive].mean()) if positive.any() else None,
        specificity=float((~decision[~positive]).mean()) if (~positive).any() else None)


def evaluate_arm(arm,checkpoint_name='best'):
    import numpy as np
    import torch
    from training.data import load_rows,sha256
    from training.engine import fingerprint_inputs
    from training.models import create_model
    directory=RUN/arm;config=json.loads((directory/'config.json').read_text())
    torch.set_num_threads(4)
    checkpoint_path=directory/f'{checkpoint_name}.pt'
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    if checkpoint['fingerprint']!=fingerprint_inputs(config,ROOT):raise ValueError('selected checkpoint fingerprint differs')
    prefix='' if checkpoint_name=='best' else 'last_'
    if checkpoint_name=='last':
        selected=torch.load(directory/'best.pt',map_location='cpu',weights_only=True)
        if all(torch.equal(value,selected['model'][key]) for key,value in checkpoint['model'].items()):
            report=json.loads((directory/'development_metrics.json').read_text())
            report.update(checkpoint='last.pt',checkpoint_sha256=sha256(checkpoint_path),
                selected_successful_updates=checkpoint['state']['optimizer_updates'],
                selected_epoch_index=checkpoint['state']['epoch'],selected_next_batch=checkpoint['state']['next_batch'],
                prediction_reuse='all model state tensors exactly equal to selected checkpoint')
            (directory/'last_development_predictions.csv').write_bytes((directory/'development_predictions.csv').read_bytes())
            write_json(directory/'last_development_metrics.json',report)
            print(json.dumps(dict(arm=arm,checkpoint='last.pt',predictions_reused_after_exact_state_check=True)))
            return
    model=create_model(config['architecture'],ROOT/'models/checkpoints').to(config['device'])
    model.load_state_dict(checkpoint['model'],strict=True);model.eval()
    manifest=ROOT/config['manifest'];release=ROOT/config['release']
    predictions=[]
    for role in ('val','boundary'):
        rows=load_rows(manifest,'glaucoma',role,release)
        predictions+=predict(model,rows,config,'original')
        if role=='boundary':
            # Same 384 parent photographs in all arms, selected without labels or predictions.
            stress=sorted(rows,key=lambda r:hashlib.sha256(r['sample_id'].encode()).hexdigest())[:384]
            for condition in ('defocus_2_at512','exposure_070'):predictions+=predict(model,stress,config,condition)
    with (directory/f'{prefix}development_predictions.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(predictions[0]));writer.writeheader();writer.writerows(predictions)
    report=dict(arm=arm,checkpoint=f'{checkpoint_name}.pt',checkpoint_sha256=sha256(checkpoint_path),
        selected_successful_updates=checkpoint['state']['optimizer_updates'],
        selected_epoch_index=checkpoint['state']['epoch'],selected_next_batch=checkpoint['state']['next_batch'],
        protected_test_inference=False,sets={},source_thresholds_95_validation_specificity={})
    for role in ('val','boundary'):
        for condition in ('original','defocus_2_at512','exposure_070'):
            selected=[r for r in predictions if r['split']==role and r['condition']==condition]
            if not selected:continue
            item=dict(per_source={s:metrics([r for r in selected if r['source']==s]) for s in sorted({r['source'] for r in selected})},
                real_poor_positive_by_source={s:metrics([r for r in selected if r['source']==s and r['real_poor_quality'] and r['label']==1])
                    for s in ('fives','mshf')},
                real_poor_positive=metrics([r for r in selected if r['real_poor_quality'] and r['label']==1]))
            report['sets'][role+'/'+condition]=item
    val=[r for r in predictions if r['split']=='val' and r['condition']=='original']
    boundary=[r for r in predictions if r['split']=='boundary' and r['condition']=='original']
    for source in sorted({r['source'] for r in val}):
        sourceval=[r for r in val if r['source']==source]
        if {r['label'] for r in sourceval}!={0,1}:continue
        negatives=[r['prediction'] for r in sourceval if r['label']==0]
        threshold=float(np.quantile(negatives,.95,method='higher'))
        report['source_thresholds_95_validation_specificity'][source]=dict(threshold=threshold,
            threshold_rule='predict positive only when score > threshold; higher 95th percentile of source validation negatives',
            validation=metrics(sourceval,threshold),
            boundary=metrics([r for r in boundary if r['source']==source],threshold),
            real_poor_boundary_positive=metrics([r for r in boundary if r['source']==source and r['real_poor_quality'] and r['label']==1],threshold))
    write_json(directory/f'{prefix}development_metrics.json',report)
    print(json.dumps(dict(arm=arm,checkpoint=f'{checkpoint_name}.pt',selected_updates=report['selected_successful_updates'],development_evaluation_completed=True)))


def compare():
    import numpy as np
    from run_expansion_experiments_20261113 import ARMS
    reports={arm:json.loads((RUN/arm/'development_metrics.json').read_text()) for arm in ARMS}
    rows={arm:read_csv(RUN/arm/'development_predictions.csv') for arm in ARMS}
    result=dict(protected_test_inference=False,arms=reports,paired_real_poor_boundary_sensitivity_differences={})
    for left,right in [('original_clean','expanded_clean'),('expanded_clean','expanded_degradation30'),('expanded_clean','expanded_degradation50')]:
        a={r['sample_id']:r for r in rows[left] if r['split']=='boundary' and r['condition']=='original' and r['real_poor_quality']=='True' and r['label']=='1'}
        b={r['sample_id']:r for r in rows[right] if r['sample_id'] in a and r['split']=='boundary' and r['condition']=='original'}
        if set(a)!=set(b):raise ValueError('paired evaluation identities differ')
        groups=collections.defaultdict(list)
        for identity,r in a.items():groups[r['related_group']].append(identity)
        units=list(groups);rng=np.random.default_rng(20261113);differences=[]
        for _ in range(2000):
            if not units:break
            ids=[identity for index in rng.integers(0,len(units),len(units)) for identity in groups[units[index]]]
            differences.append(float(np.mean([(float(b[i]['prediction'])>.5)-(float(a[i]['prediction'])>.5) for i in ids])))
        point=float(np.mean([(float(b[i]['prediction'])>.5)-(float(a[i]['prediction'])>.5) for i in a])) if a else None
        result['paired_real_poor_boundary_sensitivity_differences'][right+' minus '+left]=dict(
            positives=len(a),linked_groups=len(units),display_threshold=.5,difference=point,
            group_bootstrap_95_interval=np.quantile(differences,[.025,.975]).tolist() if differences else None,
            repeats=2000,seed=20261113)
    write_json(RUN/'comparison.json',result)
    print(json.dumps(result['paired_real_poor_boundary_sensitivity_differences'],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--arm');parser.add_argument('--compare',action='store_true')
    parser.add_argument('--checkpoint',choices=('best','last'),default='best')
    args=parser.parse_args()
    if args.compare:compare()
    elif args.arm:evaluate_arm(args.arm,args.checkpoint)
    else:parser.error('provide --arm or --compare')
