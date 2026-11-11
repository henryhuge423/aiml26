"""Run the EyeQ teacher cache, real GPU preflight, and quality training in order."""
import json
import msvcrt
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.data import sha256
from training.engine import fingerprint_inputs
from training.eyeq_quality import train
from training.processes import ProcessJob

CONFIG = ROOT / 'configs/eyeq_quality_v1.json'
RUN = ROOT / 'runs/eyeq_quality_v1'


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def preflight(config, binding):
    destination = RUN / 'preflight'
    evidence = destination / 'verification.json'
    if evidence.exists():
        proof = json.loads(evidence.read_text(encoding='utf-8'))
        if proof.get('passed') is True and proof.get('binding') == binding:
            return
    if destination.exists():
        destination.rename(RUN / f'preflight_before_retry_{time.time_ns()}')
    destination.mkdir()
    smoke = dict(config, purpose='preflight', epochs=2, checkpoint_every=1)
    direct_dir, resumed_dir = destination / 'continuous', destination / 'resumed'
    direct_config, resumed_config = destination / 'continuous.json', destination / 'resumed.json'
    write_json(direct_config, dict(smoke, run_dir=direct_dir.relative_to(ROOT).as_posix()))
    write_json(resumed_config, dict(smoke, run_dir=resumed_dir.relative_to(ROOT).as_posix()))
    direct = train(direct_config, ROOT, max_steps=4)
    first = train(resumed_config, ROOT, max_steps=3)
    restarted = train(resumed_config, ROOT, resumed_dir / 'last.pt', max_steps=4)
    a, b = [torch.load(folder / 'last.pt', map_location='cpu', weights_only=True)
            for folder in (direct_dir, resumed_dir)]
    difference = max((a['model'][k].float() - b['model'][k].float()).abs().max().item()
                     for k in a['model'])
    loss_difference = abs(direct['last_loss'] - restarted['last_loss'])
    if (b['state']['epoch'], b['state']['next_batch']) != (1, 1) or difference > 1e-6 or loss_difference > 1e-6:
        raise RuntimeError(f'EyeQ cross-epoch resume mismatch: weights={difference} loss={loss_difference}')
    if first['optimizer_updates'] == 0 or restarted['optimizer_updates'] <= first['optimizer_updates']:
        raise RuntimeError('EyeQ preflight did not update parameters')
    first_steps = [json.loads(line) for line in (resumed_dir / 'steps.jsonl').read_text().splitlines()]
    if not any(r['eyeq_rows'] and r['ordinal_loss'] > 0 and r['optimizer_updated'] for r in first_steps):
        raise RuntimeError('EyeQ preflight had no ordinal-supervised optimizer update')
    completed = train(resumed_config, ROOT, resumed_dir / 'last.pt')
    if completed['epoch'] != 2 or not (resumed_dir / 'best_validation_predictions.csv').is_file():
        raise RuntimeError('EyeQ preflight did not produce validation outputs')
    write_json(evidence, dict(passed=True, binding=binding, weight_max_abs_difference=difference,
                              loss_abs_difference=loss_difference, completed=completed))


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    lock_path = RUN / 'pipeline.lock'
    with lock_path.open('a+b') as lock:
        if lock_path.stat().st_size == 0:
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        config = json.loads(CONFIG.read_text(encoding='utf-8'))
        stage = 'teacher_cache'

        def status(**fields):
            write_json(RUN / 'status.json', dict(stage=stage, pid=os.getpid(), updated_at=time.time(), **fields))

        try:
            status()
            with ProcessJob() as job, (RUN / 'teacher_cache.log').open('a', encoding='utf-8') as log:
                child = job.popen([sys.executable, '-u', str(ROOT / 'tools/cache_teacher.py'),
                                   '--manifest', config['manifest'], '--release', config['release'],
                                   '--output', config['teacher_cache'], '--reuse-from',
                                   'data/prepared/formal_v3/teacher_scores.csv'],
                                  cwd=ROOT, stdout=log, stderr=log)
                status(cache_pid=child.pid)
                if child.wait():
                    raise RuntimeError('teacher cache failed; see teacher_cache.log')
            binding = fingerprint_inputs(config, ROOT) + ':' + sha256(ROOT / 'training/eyeq_quality.py')
            stage = 'gpu_preflight'
            status()
            preflight(config, binding)
            stage = 'training'
            status()
            checkpoint = ROOT / config['run_dir'] / 'last.pt'
            result = train(CONFIG, ROOT, checkpoint if checkpoint.exists() else None)
            if result['epoch'] != config['epochs']:
                raise RuntimeError('EyeQ training did not reach configured epochs')
            stage = 'completed'
            status(result=result)
        except Exception as error:
            status(failed=True, error=str(error))
            raise


if __name__ == '__main__':
    main()
