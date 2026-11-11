"""Real-data GPU training/restart verification; produces no research accuracy claim."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.engine import train


def main(destination: Path,models,config_dir=None,train_rows=16):
    if destination.exists():raise ValueError('choose a new preflight destination; existing evidence is preserved')
    destination.mkdir(parents=True)
    torch.set_num_threads(4)
    results=[]
    for name in models:
        config=json.loads(((config_dir or ROOT/'configs')/f'{name}.json').read_text())
        config.update(purpose='preflight',epochs=2,checkpoint_every=1,
                      preflight_train_rows=train_rows,preflight_val_rows=16)
        if train_rows<3*config['batch_size']:raise ValueError('preflight needs at least three full batches for restart validation')
        if train_rows%config['batch_size']:raise ValueError('preflight train rows must form full batches')
        boundary_step=train_rows//config['batch_size']
        comparison_step=boundary_step+1
        continuous=destination/name/'continuous';resumed=destination/name/'resumed'
        continuous.parent.mkdir(parents=True)
        config['run_dir']=continuous.resolve().relative_to(ROOT.resolve()).as_posix()
        config_path=destination/name/'continuous.json';config_path.write_text(json.dumps(config,indent=2))
        direct=train(config_path,ROOT,max_steps=comparison_step)
        config['run_dir']=resumed.resolve().relative_to(ROOT.resolve()).as_posix()
        resume_config=destination/name/'resumed.json';resume_config.write_text(json.dumps(config,indent=2))
        first=train(resume_config,ROOT,max_steps=boundary_step)
        second=train(resume_config,ROOT,resumed/'last.pt',max_steps=comparison_step)
        a=torch.load(continuous/'last.pt',map_location='cpu',weights_only=True)
        b=torch.load(resumed/'last.pt',map_location='cpu',weights_only=True)
        max_difference=max((a['model'][k].float()-b['model'][k].float()).abs().max().item() for k in a['model'])
        loss_difference=abs(direct['last_loss']-second['last_loss'])
        if a['state']['epoch']!=1 or b['state']['epoch']!=1 or b['state']['next_batch']!=1:
            raise RuntimeError('resume comparison did not cross epoch boundary')
        if max_difference>1e-6 or loss_difference>1e-6:
            raise RuntimeError(f'{name} restart mismatch: weights={max_difference} loss={loss_difference}')
        if not all(r['parameters_changed'] for r in (direct,first,second)):
            raise RuntimeError(f'{name} did not update real parameters')
        # Finish this short epoch and exercise actual validation/best/last output.
        completed=train(resume_config,ROOT,resumed/'last.pt')
        if not (resumed/'validation_predictions.csv').exists() or not (resumed/'best.pt').exists():
            raise RuntimeError('validation/checkpoint outputs missing')
        result=dict(model=name,passed=True,weight_max_abs_difference=max_difference,
                    resumed_next_loss_abs_difference=loss_difference,completed=completed,
                    cross_epoch_resume_verified=True,comparison_step=comparison_step,comparison_epoch=1)
        results.append(result)
        (destination/'verification.json').write_text(json.dumps(dict(passed=False,finished=results),indent=2))
        del a,b
        if torch.cuda.is_available():torch.cuda.empty_cache()
    evidence=dict(passed=True,models=results,purpose='readiness_only_not_model_performance',
                  cross_epoch_resume_verified=True,
                  checkpoints_are_research_results=False,formal_long_training_started=False)
    (destination/'verification.json').write_text(json.dumps(evidence,indent=2))
    print(json.dumps(evidence,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'runs/preflight/formal_v1')
    parser.add_argument('--config-dir',type=Path)
    parser.add_argument('--train-rows',type=int,default=16)
    parser.add_argument('--models',nargs='+',choices=['glaucoma_a','glaucoma_b_candidate','quality_student'],
                        default=['glaucoma_a','glaucoma_b_candidate','quality_student'])
    args=parser.parse_args();main(args.output,args.models,args.config_dir,args.train_rows)
