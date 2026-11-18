"""Continue the frozen expanded cohort with source-quota and degradation controls."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_expansion_experiments_20261113 import child, write_json

RUN=ROOT/'runs/followup_20261115'
CONFIGS=ROOT/'configs/followup_20261115'
SEEDS=(20261115,20261116,20261117)
ARMS=('original_clean','expanded_clean','quota65_clean','expanded_mixed50','quota65_mixed50','quota65_defocus50')
SOURCES=('airogs-light-v2','DATASET_101/ODIR-5K','Kaggle-Eye_diseases_classification','lag_part1','keh')


def prepare():
    from training.data import load_rows,sha256
    manifest=ROOT/'data/prepared/expanded_20261113/manifest.csv'
    release=manifest.with_name('release.json')
    val=load_rows(manifest,'glaucoma','val',release)
    for source in SOURCES:
        assert {r['label'] for r in val if r['source']==source}=={'0','1'}
    jobs=[]
    for seed in SEEDS:
        for arm in ARMS:
            name=f'seed{seed}/{arm}'
            config=dict(purpose='fixed_budget_research',task='glaucoma',architecture='efficientnet_b0',
                manifest=manifest.relative_to(ROOT).as_posix(),release=release.relative_to(ROOT).as_posix(),
                run_dir=f'runs/followup_20261115/{name}',image_size=320,batch_size=16,epochs=20,
                max_optimizer_updates=2400,validation_every_updates=300,learning_rate=.0001,
                weight_decay=.0001,gradient_clip=5.,seed=seed,device='cuda',amp=True,checkpoint_every=100,
                selection_source=SOURCES[0],selection_sources=list(SOURCES),sampler='inverse_sqrt',
                num_workers=4,prefetch_factor=2,cpu_threads=4,save_validation_checkpoints=True,
                record_degradation_trace=False,degradation={'probability':.5 if arm.endswith('50') else 0.})
            if arm=='original_clean':config['training_origins']=['formal_v3']
            if arm.startswith('quota65'):config['original_sampling_fraction']=.65
            if arm.endswith('defocus50'):config['degradation']['families']=['defocus']
            path=CONFIGS/f'{name}.json'
            if path.exists() and json.loads(path.read_text())!=config:raise ValueError(f'configuration changed: {path}')
            write_json(path,config);jobs.append((name,path,config))
    plan=dict(created_at=dt.datetime.now(dt.timezone.utc).isoformat(),seeds=list(SEEDS),arms=list(ARMS),
        jobs=[j[0] for j in jobs],successful_updates_per_run=2400,selection_sources=list(SOURCES),
        original_quota=.65,sampler_within_origins='inverse_sqrt_source_class',
        saved_validation_updates=list(range(300,2401,300)),
        inference_roles=['val','boundary'],protected_test_inference=False,
        manifest_sha256=sha256(manifest),release_sha256=sha256(release))
    existing=RUN/'experiment_plan.json'
    if existing.exists():
        old=json.loads(existing.read_text());plan['created_at']=old['created_at']
        if old!=plan:raise ValueError('existing experiment plan changed')
    write_json(existing,plan)
    return jobs


def preflight(jobs):
    import torch
    from training.engine import fingerprint_inputs
    from training.data import sha256
    evidence=[]
    for arm in ('expanded_mixed50','quota65_mixed50','quota65_defocus50'):
        base=next(dict(config) for name,_,config in jobs if name==f'seed{SEEDS[0]}/{arm}')
        base.update(purpose='preflight',preflight_train_rows=32,epochs=5,max_optimizer_updates=6,
            validation_every_updates=3,num_workers=2)
        base.pop('selection_sources')
        configs=[]
        for kind in ('full','resumed'):
            config=dict(base,run_dir=f'runs/followup_20261115/preflight/{arm}/{kind}')
            path=RUN/f'preflight/{arm}/{kind}.json';write_json(path,config);configs.append((path,config))
        proof=RUN/f'preflight/{arm}/verification.json'
        if proof.exists():
            report=json.loads(proof.read_text())
            if not report['passed'] or report['fingerprint']!=fingerprint_inputs(configs[0][1],ROOT):
                raise ValueError('preflight evidence changed')
            evidence.append(report);continue
        full,partial=configs
        child(['tools/train_model.py','--config',str(full[0])],RUN/f'pipeline/preflight-{arm}-full.log')
        child(['tools/train_model.py','--config',str(partial[0]),'--max-steps','3'],RUN/f'pipeline/preflight-{arm}-partial.log')
        last=ROOT/partial[1]['run_dir']/'last.pt'
        child(['tools/train_model.py','--config',str(partial[0]),'--resume',str(last)],RUN/f'pipeline/preflight-{arm}-resume.log')
        checkpoints=[torch.load(ROOT/c['run_dir']/'last.pt',map_location='cpu',weights_only=True) for _,c in configs]
        a,b=checkpoints
        maximum=max((a['model'][key]-b['model'][key]).abs().max().item() for key in a['model'])
        records=[[json.loads(line) for line in (ROOT/c['run_dir']/'steps.jsonl').read_text().splitlines()] for _,c in configs]
        passed=(maximum==0 and records[0]==records[1] and a['state']==b['state'] and
            a['state']['optimizer_updates']==6 and a['state']['epoch']>=2)
        snapshots=ROOT/full[1]['run_dir']/'validation_checkpoints'
        passed=passed and all((snapshots/f'update_{update:06d}.{suffix}').exists() for update in (3,6) for suffix in ('pt','csv','json'))
        report=dict(arm=arm,passed=passed,weight_max_abs_difference=maximum,step_records_exact=records[0]==records[1],
            epoch_crossed=a['state']['epoch']>=2,successful_updates=6,workers=2,
            fingerprint=fingerprint_inputs(full[1],ROOT),checkpoints_sha256=[sha256(ROOT/c['run_dir']/'last.pt') for _,c in configs])
        write_json(proof,report);evidence.append(report)
        if not passed:raise ValueError(f'CUDA resume preflight failed: {arm}')
    write_json(RUN/'preflight/verification.json',dict(passed=all(r['passed'] for r in evidence),models=evidence))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();jobs=prepare()
    if args.prepare_only:return
    path=RUN/'pipeline/status.json'
    status=dict(started_at=dt.datetime.now(dt.timezone.utc).isoformat(),pid=os.getpid(),status='preflight',
        completed=[],total_runs=len(jobs),protected_test_inference=False)
    write_json(path,status)
    try:
        preflight(jobs)
        for name,config_path,config in jobs:
            directory=ROOT/config['run_dir'];status.update(status='training',current_run=name);write_json(path,status)
            summary=directory/'run_summary.json'
            complete=summary.exists() and json.loads(summary.read_text()).get('budget_completed') is True
            if not complete:
                arguments=['tools/train_model.py','--config',str(config_path)]
                if (directory/'last.pt').exists():arguments+=['--resume',str(directory/'last.pt')]
                child(arguments,RUN/f'pipeline/{name.replace("/","-")}.log')
            status.update(status='evaluating');write_json(path,status)
            child(['tools/evaluate_followup_20261115.py','--run',name],RUN/f'pipeline/{name.replace("/","-")}-evaluation.log')
            status['completed'].append(name);write_json(path,status)
            child(['tools/evaluate_followup_20261115.py','--summarize'],RUN/'pipeline/summary.log')
        status.update(status='completed',finished_at=dt.datetime.now(dt.timezone.utc).isoformat());write_json(path,status)
    except BaseException as error:
        status.update(status='failed',error=str(error));write_json(path,status);raise


if __name__=='__main__':main()
