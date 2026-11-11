"""Sequential, fail-stopping formal_v3 training with persistent status and logs."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training.data import sha256


def now():return datetime.now(timezone.utc).isoformat()


def verify_preflight(directory, names, config_dir):
    from training.engine import fingerprint_inputs
    proof=json.loads((directory/'verification.json').read_text())
    if proof.get('passed') is not True:raise ValueError('GPU preflight is incomplete')
    if proof.get('cross_epoch_resume_verified') is not True:raise ValueError('GPU preflight has no cross-epoch resume evidence')
    results={r['model']:r for r in proof['models']}
    for name in names:
        result=results[name]
        if result.get('cross_epoch_resume_verified') is not True or result.get('comparison_epoch')!=1:
            raise ValueError(f'GPU preflight did not verify epoch transition: {name}')
        config=json.loads((config_dir/f'{name}.json').read_text())
        smoke=json.loads((directory/name/'resumed.json').read_text())
        for key,value in config.items():
            if key not in {'purpose','epochs','run_dir','checkpoint_every'} and smoke.get(key)!=value:
                raise ValueError(f'GPU preflight configuration mismatch: {name}/{key}')
        if (not result['passed'] or result['weight_max_abs_difference']>1e-6
                or result['resumed_next_loss_abs_difference']>1e-6
                or not result['completed']['parameters_changed']
                or not math.isfinite(result['completed']['last_loss'])):
            raise ValueError(f'GPU preflight failed: {name}')
        if fingerprint_inputs(smoke,ROOT)!=result['completed']['fingerprint']:
            raise ValueError(f'GPU preflight fingerprint mismatch: {name}')
        for file in ('last.pt','best.pt','validation_predictions.csv'):
            if not (directory/name/'resumed'/file).is_file():raise ValueError(f'missing preflight output: {file}')


def ensure_quality_preflight(directory, names, config_dir, execute_stage,
                             verifier=verify_preflight):
    if directory.exists():
        try:
            verifier(directory,names,config_dir)
            return
        except Exception as error:
            backup=directory.with_name(f'{directory.name}.backup_{time.time_ns()}')
            suffix=1
            while backup.exists():
                backup=directory.with_name(f'{directory.name}.backup_{time.time_ns()}_{suffix}')
                suffix+=1
            directory.rename(backup)
            print(f'Preserved invalid quality preflight at {backup}: {error}',flush=True)
    execute_stage('quality_preflight',[ROOT/'tools/verify_training_preflight.py','--output',directory,
        '--config-dir',config_dir,'--train-rows','64','--models','quality_student'])
    verifier(directory,names,config_dir)


def main():
    import msvcrt
    import torch
    from training.engine import fingerprint_inputs,load_cache
    from training.data import load_rows,attach_teacher_targets
    from training.processes import ProcessJob
    config_dir=ROOT/'configs/formal_v3'
    output=ROOT/'runs/formal_v3/pipeline'
    output.mkdir(parents=True,exist_ok=True)
    lock=(output/'pipeline.lock').open('a+b')
    lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:raise RuntimeError('formal_v3 pipeline is already running')
    gate=json.loads((output/'review_gate.json').read_text())
    if gate.get('passed') is not True:raise ValueError('release/code review gate not passed')
    for relative,digest in gate['input_fingerprints'].items():
        if sha256(ROOT/relative)!=digest:raise ValueError(f'reviewed input changed: {relative}')
    job=ProcessJob()
    status=dict(state='running',started_at=now(),pid=os.getpid(),completed_stages=[],
                cohort_version='formal_v3',order=['glaucoma_a','glaucoma_b_candidate','teacher_cache','quality_preflight','quality_student'])

    def save(**fields):
        status.update(fields,heartbeat_at=now())
        temporary=output/'status.json.tmp'
        temporary.write_text(json.dumps(status,indent=2),encoding='utf-8')
        temporary.replace(output/'status.json')

    def execute(stage,arguments):
        log=output/f'{stage}.log'
        save(stage=stage,stage_started_at=now(),log=str(log),child_pid=None)
        print(f'{now()} starting {stage}',flush=True)
        with log.open('a',encoding='utf-8') as stream:
            stream.write(f'\n{now()} launching {arguments!r}\n');stream.flush()
            child=job.popen([sys.executable,'-u',*map(str,arguments)],cwd=ROOT,
                stdout=stream,stderr=subprocess.STDOUT)
            save(child_pid=child.pid)
            while child.poll() is None:
                time.sleep(10)
                save()
        if child.returncode:raise RuntimeError(f'{stage} exited with code {child.returncode}; inspect {log}')
        status['completed_stages'].append(stage)
        save(child_pid=None)
        print(f'{now()} completed {stage}',flush=True)

    def training(name):
        config_path=config_dir/f'{name}.json'
        config=json.loads(config_path.read_text())
        run_dir=ROOT/config['run_dir']
        checkpoint=run_dir/'last.pt'
        args=[ROOT/'tools/train_model.py','--config',config_path]
        if checkpoint.exists():
            content=torch.load(checkpoint,map_location='cpu',weights_only=True)
            if content['fingerprint']!=fingerprint_inputs(config,ROOT):
                raise ValueError(f'checkpoint fingerprint mismatch: {name}')
            if content['state']['epoch']>=config['epochs']:
                if not (run_dir/'best.pt').is_file() or not (run_dir/'epochs.jsonl').is_file():
                    raise ValueError(f'completed checkpoint lacks outputs: {name}')
                status['completed_stages'].append(name);save();return
            args.extend(['--resume',checkpoint])
        execute(name,args)
        content=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if content['state']['epoch']!=config['epochs']:raise ValueError(f'training did not finish all epochs: {name}')
        if content['fingerprint']!=fingerprint_inputs(config,ROOT):raise ValueError('completed model fingerprint mismatch')

    save(stage='checking_gates')
    try:
        verify_preflight(ROOT/'runs/preflight/formal_v3_amp_recovery',['glaucoma_a','glaucoma_b_candidate'],config_dir)
        training('glaucoma_a')
        training('glaucoma_b_candidate')
        cohort=ROOT/'data/prepared/formal_v3'
        execute('teacher_cache',[ROOT/'tools/cache_teacher.py','--manifest',cohort/'manifest.csv',
            '--release',cohort/'release.json','--output',cohort/'teacher_scores.csv','--crop-batch','20',
            '--reuse-from',ROOT/'data/prepared/formal_v2/teacher_scores.csv'])
        cache=load_cache(cohort/'teacher_scores.csv',cohort/'manifest.csv',ROOT,cohort/'release.json')
        for split in ('train','val'):
            rows=load_rows(cohort/'manifest.csv','quality',split,cohort/'release.json')
            attach_teacher_targets(rows,cache)
            for row in rows:
                if any(cache[row['sample_id']][key]!=row[key] for key in ('sample_id','sha256','path','split')):
                    raise ValueError('quality target binding mismatch')
        if len(cache)!=13789:raise ValueError('unexpected expanded teacher target count')
        quality_proof=ROOT/'runs/preflight/formal_v3_quality'
        ensure_quality_preflight(quality_proof,['quality_student'],config_dir,execute)
        training('quality_student')
        save(state='completed',stage='completed',completed_at=now(),child_pid=None)
    except BaseException as error:
        save(state='failed',error=str(error),traceback=traceback.format_exc(),failed_at=now())
        raise
    finally:
        job.close()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':main()
