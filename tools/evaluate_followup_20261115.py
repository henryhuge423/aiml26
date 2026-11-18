"""Source-specific development evaluation of saved follow-up checkpoints."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.evaluate_expansion_20261113 import read_csv,write_json,metrics,predict,poor
from tools.run_followup_experiments_20261115 import RUN,SEEDS,ARMS,SOURCES


def evaluate(name):
    import torch
    from training.data import load_rows,sha256
    from training.engine import fingerprint_inputs
    from training.models import create_model
    torch.set_num_threads(4)
    directory=(RUN/name).resolve()
    if not directory.is_relative_to(RUN.resolve()):raise ValueError('run outside experiment directory')
    config=json.loads((directory/'config.json').read_text())
    best=torch.load(directory/'best.pt',map_location='cpu',weights_only=True)
    selected_update=best['state']['optimizer_updates']
    checkpoints=sorted({1200,2400,selected_update})
    val=load_rows(ROOT/config['manifest'],'glaucoma','val',ROOT/config['release'])
    train=load_rows(ROOT/config['manifest'],'glaucoma','train',ROOT/config['release'])
    origins={r['sample_id']:r['origin_cohort'] for r in train}
    boundary=load_rows(ROOT/config['manifest'],'glaucoma','boundary',ROOT/config['release'])
    stress=sorted(boundary,key=lambda r:hashlib.sha256(r['sample_id'].encode()).hexdigest())[:384]
    summaries={}
    for update in checkpoints:
        stem=f'update_{update:06d}'
        checkpoint=directory/'validation_checkpoints'/f'{stem}.pt'
        destination=directory/'development'/stem;destination.mkdir(parents=True,exist_ok=True)
        report_path=destination/'metrics.json'
        if report_path.exists():
            report=json.loads(report_path.read_text())
            if report['checkpoint_sha256']!=sha256(checkpoint):raise ValueError('evaluation checkpoint changed')
            summaries[str(update)]=report;continue
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if state['fingerprint']!=fingerprint_inputs(config,ROOT):raise ValueError('checkpoint input fingerprint mismatch')
        model=create_model(config['architecture'],ROOT/'models/checkpoints').to(config['device'])
        model.load_state_dict(state['model'],strict=True);model.eval()
        predictions=predict(model,boundary,config,'original')
        for condition in ('defocus_2_at512','exposure_070'):predictions+=predict(model,stress,config,condition)
        import csv
        with (destination/'predictions.csv').open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(predictions[0]));writer.writeheader();writer.writerows(predictions)
        val_predictions=read_csv(directory/'validation_checkpoints'/f'{stem}.csv')
        for r in val_predictions:r.update(label=int(float(r['label'])))
        assert {r['sample_id'] for r in val_predictions}=={r['sample_id'] for r in val}
        val_reference={r['sample_id']:r for r in val}
        assert all(r['source']==val_reference[r['sample_id']]['source'] and
            r['label']==int(val_reference[r['sample_id']]['label']) and
            r['related_group']==val_reference[r['sample_id']]['related_group'] for r in val_predictions)
        original=[r for r in predictions if r['condition']=='original']
        thresholds={source:float(np.quantile([float(r['prediction']) for r in val_predictions if r['source']==source and r['label']==0],.95,method='higher'))
            for source in {r['source'] for r in val_predictions} if {r['label'] for r in val_predictions if r['source']==source}=={0,1}}
        conditions={}
        for condition in ('original','defocus_2_at512','exposure_070'):
            rows=[r for r in predictions if r['condition']==condition]
            source_results={source:metrics([r for r in rows if r['source']==source]) for source in sorted({r['source'] for r in rows})}
            fixed={source:metrics([r for r in rows if r['source']==source],thresholds[source]) for source in thresholds if any(r['source']==source for r in rows)}
            conditions[condition]=dict(overall=metrics(rows),per_source=source_results,fixed_validation_thresholds=fixed,
                real_poor=metrics([r for r in rows if r['real_poor_quality']]))
        report=dict(run=name,update=update,checkpoint_sha256=sha256(checkpoint),fingerprint=state['fingerprint'],
            selected_update=selected_update,display_threshold=.5,protected_test_inference=False,
            conditions=conditions,validation_thresholds=thresholds,
            validation_selection=json.loads((directory/'validation_checkpoints'/f'{stem}.json').read_text()),
            origin_draws={origin:sum(count for sample,count in state['state']['sample_draw_counts'].items()
                if origins[sample]==origin) for origin in ('formal_v3','expansion_20261113')},
            prediction_sha256=sha256(destination/'predictions.csv'))
        write_json(report_path,report);summaries[str(update)]=report
        del model,state
        torch.cuda.empty_cache()
    write_json(directory/'evaluation_summary.json',dict(run=name,selected_update=selected_update,checkpoints=summaries))
    print(json.dumps(dict(run=name,selected_update=selected_update,evaluated_updates=checkpoints)))


def summarize():
    from sklearn.metrics import roc_auc_score
    quality_path=ROOT/'runs/expanded_20261113/pq_quality_20261114/quality_scores.csv'
    quality_proof=json.loads(quality_path.with_name('verification.json').read_text())
    assert hashlib.sha256(quality_path.read_bytes()).hexdigest()==quality_proof['scores_sha256']
    q={r['sample_id']:float(r['q']) for r in read_csv(quality_path) if r['split']=='boundary'}
    rows=[]
    for seed in SEEDS:
        for arm in ARMS:
            directory=RUN/f'seed{seed}/{arm}'
            path=directory/'evaluation_summary.json'
            if not path.exists():continue
            evaluation=json.loads(path.read_text())
            for update,report in evaluation['checkpoints'].items():
                conditions=report['conditions']
                predictions=read_csv(directory/'development'/f'update_{int(update):06d}'/'predictions.csv')
                originals=[r for r in predictions if r['condition']=='original']
                assert {r['sample_id'] for r in originals}==set(q)
                scores=np.array([float(r['prediction']) for r in originals])
                errors=(scores>.5)!=np.array([int(r['label']) for r in originals],bool)
                quality_ranking=dict(errors=int(errors.sum()),n=len(errors),
                    lowq_error_auc=float(roc_auc_score(errors,[-q[r['sample_id']] for r in originals])) if len(set(errors))==2 else None,
                    uncertainty_error_auc=float(roc_auc_score(errors,-abs(scores-.5))) if len(set(errors))==2 else None)
                rows.append(dict(seed=seed,arm=arm,update=int(update),selected=int(update)==evaluation['selected_update'],
                    source_auc={s:conditions['original']['per_source'][s]['auc'] for s in SOURCES},
                    boundary=conditions['original']['overall'],stress={c:conditions[c]['overall'] for c in ('defocus_2_at512','exposure_070')},
                    real_poor=conditions['original']['real_poor'],origin_draws=report['origin_draws'],quality_error_ranking=quality_ranking))
    means={}
    for arm in ARMS:
        available=[r for r in rows if r['arm']==arm and r['update']==2400]
        if not available:continue
        means[arm]=dict(completed_seeds=[r['seed'] for r in available],n_seeds=len(available),
            source_auc={s:dict(mean=float(np.mean([r['source_auc'][s] for r in available])),
                values=[r['source_auc'][s] for r in available],sd=float(np.std([r['source_auc'][s] for r in available],ddof=1)) if len(available)>1 else None) for s in SOURCES})
    write_json(RUN/'comparison.json',dict(rows=rows,update2400=means,protected_test_inference=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run');parser.add_argument('--summarize',action='store_true')
    args=parser.parse_args()
    if args.summarize:summarize()
    elif args.run:evaluate(args.run)
    else:parser.error('provide --run or --summarize')
