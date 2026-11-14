"""Run four fixed-budget disease experiments, with a real CUDA resume check first."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
RUN=ROOT/'runs/expanded_20261113'
CONFIGS=ROOT/'configs/expanded_20261113'
ARMS=('original_clean','expanded_clean','expanded_degradation30','expanded_degradation50')


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    temporary.replace(path)


def make_configs():
    from training.data import load_rows
    manifest=ROOT/'data/prepared/expanded_20261113/manifest.csv'
    release=manifest.with_name('release.json')
    val=load_rows(manifest,'glaucoma','val',release)
    # Equal-weighted source-specific ranking; endpoints are never pooled for selection.
    sources=sorted(source for source in {r['source'] for r in val}
        if {r['label'] for r in val if r['source']==source}=={'0','1'})
    for arm in ARMS:
        config=dict(purpose='fixed_budget_research',task='glaucoma',architecture='efficientnet_b0',
            manifest=manifest.relative_to(ROOT).as_posix(),release=release.relative_to(ROOT).as_posix(),
            run_dir=f'runs/expanded_20261113/{arm}',image_size=320,batch_size=16,epochs=20,
            max_optimizer_updates=1200,validation_every_updates=300,learning_rate=.0001,
            weight_decay=.0001,gradient_clip=5.,seed=20261113,device='cuda',amp=True,
            checkpoint_every=100,selection_source='airogs-light-v2',selection_sources=sources,
            sampler='inverse_sqrt',num_workers=4,prefetch_factor=2,cpu_threads=4,
            record_degradation_trace=arm.endswith(('30','50')),
            degradation={'probability':.3 if arm.endswith('30') else .5 if arm.endswith('50') else 0.})
        if arm=='original_clean':config['training_origins']=['formal_v3']
        path=CONFIGS/f'{arm}.json'
        if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=config:
            raise ValueError(f'existing experiment configuration differs: {path}')
        write_json(path,config)
    write_json(RUN/'experiment_plan.json',dict(created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        arms=list(ARMS),architecture='efficientnet_b0',successful_optimizer_updates_per_arm=1200,
        validation_every_successful_updates=300,seed=20261113,sampler='inverse_sqrt',
        selection='macro mean of source-specific validation AUROCs',selection_sources=sources,
        inference_roles=['val','boundary'],protected_test_inference=False,
        interpretation='single-seed fixed-budget comparison, not 20 completed epochs or final testing'))


def child(arguments,log):
    from training.processes import ProcessJob
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a',encoding='utf-8') as output:
        if os.name=='nt':
            with ProcessJob() as job:
                process=job.popen([sys.executable,'-u',*arguments],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
                code=process.wait()
        else:code=subprocess.run([sys.executable,'-u',*arguments],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT).returncode
    if code:raise RuntimeError(f'child failed ({code}); inspect {log}')


def preflight():
    import torch
    from training.engine import fingerprint_inputs
    from training.data import sha256
    base=json.loads((CONFIGS/'expanded_degradation30.json').read_text())
    base.update(purpose='preflight',preflight_train_rows=32,epochs=5,
        max_optimizer_updates=6,validation_every_updates=3,num_workers=2)
    base.pop('selection_sources')
    configs=[]
    for name in ('uninterrupted','resumed'):
        config=dict(base,run_dir=f'runs/expanded_20261113/preflight/{name}')
        path=RUN/'preflight'/f'{name}.json';write_json(path,config);configs.append((path,config))
    report=RUN/'preflight/verification.json'
    if report.exists():
        evidence=json.loads(report.read_text())
        if evidence['fingerprint']!=fingerprint_inputs(configs[0][1],ROOT):
            raise ValueError('preflight implementation/configuration changed')
        if not evidence['passed']:raise ValueError('preflight failed')
        return
    full,partial=configs
    child(['tools/train_model.py','--config',str(full[0])],RUN/'pipeline/preflight-full.log')
    child(['tools/train_model.py','--config',str(partial[0]),'--max-steps','3'],RUN/'pipeline/preflight-partial.log')
    partial_checkpoint=ROOT/partial[1]['run_dir']/'last.pt'
    child(['tools/train_model.py','--config',str(partial[0]),'--resume',str(partial_checkpoint)],RUN/'pipeline/preflight-resume.log')
    a=torch.load(ROOT/full[1]['run_dir']/'last.pt',map_location='cpu',weights_only=True)
    b=torch.load(partial_checkpoint,map_location='cpu',weights_only=True)
    maximum=max((a['model'][name]-b['model'][name]).abs().max().item() for name in a['model'])
    losses=[]
    for _,config in configs:
        records=[json.loads(line) for line in (ROOT/config['run_dir']/'steps.jsonl').read_text().splitlines()]
        losses.append([r['loss'] for r in records])
    passed=(maximum==0 and losses[0]==losses[1] and a['state']['optimizer_updates']==b['state']['optimizer_updates']==6
        and a['state']['epoch']>=2 and a['state']['degradation_family_draws']==b['state']['degradation_family_draws'])
    write_json(report,dict(passed=passed,weight_max_abs_difference=maximum,next_losses_exact=losses[0]==losses[1],
        epoch_crossed=True,successful_updates=6,workers=2,degradation_probability=.3,
        fingerprint=fingerprint_inputs(full[1],ROOT),checkpoints_sha256=[sha256(ROOT/full[1]['run_dir']/'last.pt'),sha256(partial_checkpoint)]))
    if not passed:raise ValueError('CUDA augmentation/resume preflight mismatch')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();make_configs()
    if args.prepare_only:return
    status=dict(started_at=dt.datetime.now(dt.timezone.utc).isoformat(),pid=os.getpid(),status='preflight',completed=[])
    path=RUN/'pipeline/status.json';write_json(path,status)
    try:
        preflight()
        for arm in ARMS:
            status.update(status='training',current_arm=arm);write_json(path,status)
            config_path=CONFIGS/f'{arm}.json';directory=RUN/arm
            summary=directory/'run_summary.json'
            complete=summary.exists() and json.loads(summary.read_text()).get('budget_completed') is True
            if not complete:
                arguments=['tools/train_model.py','--config',str(config_path)]
                if (directory/'last.pt').exists():arguments+=['--resume',str(directory/'last.pt')]
                child(arguments,RUN/f'pipeline/{arm}.log')
            status.update(status='evaluating',current_arm=arm);write_json(path,status)
            child(['tools/evaluate_expansion_20261113.py','--arm',arm],RUN/f'pipeline/{arm}-evaluation.log')
            status['completed'].append(arm);write_json(path,status)
        child(['tools/evaluate_expansion_20261113.py','--compare'],RUN/'pipeline/comparison.log')
        status.update(status='completed',finished_at=dt.datetime.now(dt.timezone.utc).isoformat());write_json(path,status)
    except BaseException as error:
        status.update(status='failed',error=str(error));write_json(path,status);raise


if __name__=='__main__':main()
