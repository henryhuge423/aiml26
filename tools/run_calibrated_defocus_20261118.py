"""Train the paired calibrated-defocus pilot without changing the archived trainer."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from training import engine
from training.calibrated_defocus import CalibratedDefocus
from training.data import sha256
from tools.run_followup_experiments_20261115 import CONFIGS, RUN as CONTROL, SEEDS
from tools.run_expansion_experiments_20261113 import write_json

RUN = ROOT / 'runs/calibrated_defocus_20261118'
PROFILE = ROOT / 'configs/calibrated_defocus_20261118/profile.json'
ARM = 'quota65_mixed50'
UPDATE = 1200
STATUS = RUN / 'status.json'
BASE_LOADER = engine.loader
BASE_FINGERPRINT = engine.fingerprint_inputs


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def profile(config, root):
    entry = config['defocus_calibration']
    path = root / entry['profile']
    if sha256(path) != entry['sha256']:
        raise ValueError('calibrated defocus profile hash differs')
    payload = read(path)
    parameters = payload['parameters']
    if len(parameters) != payload['photo_pairs'] or not parameters:
        raise ValueError('calibrated defocus profile count differs')
    return parameters


def fingerprint(config, root):
    inherited = BASE_FINGERPRINT(config, root)
    if 'defocus_calibration' not in config:
        return inherited
    profile(config, root)
    files = [root/'training/calibrated_defocus.py', Path(__file__)]
    return hashlib.sha256(':'.join([inherited, *(sha256(f) for f in files)]).encode()).hexdigest()


def loader(rows, config, root, train, epoch=0, indices=None):
    batches = BASE_LOADER(rows, config, root, train, epoch, indices)
    if train and 'defocus_calibration' in config:
        dataset = batches.dataset if indices is None else batches.dataset.dataset
        dataset.degradation = CalibratedDefocus(profile=profile(config, root), **config['degradation'])
    return batches


def candidate_config(seed):
    config = read(CONFIGS / f'seed{seed}' / f'{ARM}.json')
    config['run_dir'] = (RUN / f'seed{seed}').relative_to(ROOT).as_posix()
    config['max_optimizer_updates'] = UPDATE
    config['defocus_calibration'] = dict(profile=PROFILE.relative_to(ROOT).as_posix(), sha256=sha256(PROFILE))
    return config


def train_seed(seed):
    config = candidate_config(seed)
    directory = RUN / f'seed{seed}'
    config_path = directory/'config.json'
    summary_path = directory/'run_summary.json'
    last = directory/'last.pt'
    if config_path.exists() and read(config_path) != config:
        raise ValueError(f'configuration changed: {config_path}')
    if summary_path.exists() and read(summary_path)['budget_completed']:
        summary = read(summary_path)
        assert summary['optimizer_updates'] == UPDATE and summary['fingerprint'] == fingerprint(config, ROOT)
        return summary
    write_json(config_path, config)
    torch.cuda.reset_peak_memory_stats()
    summary = engine.train(config_path, ROOT, resume=last if last.exists() else None)
    if not summary['budget_completed'] or summary['optimizer_updates'] != UPDATE or not summary['parameters_changed']:
        raise ValueError(f'training did not complete the intended budget for seed {seed}')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    args = parser.parse_args()
    chosen = SEEDS if args.seed is None else (args.seed,)
    engine.loader = loader
    engine.fingerprint_inputs = fingerprint
    status = dict(status='training', seeds=list(chosen), completed=[], optimizer_updates=UPDATE,
                  protected_test_inference=False)
    try:
        for seed in chosen:
            status['current_seed'] = seed
            write_json(STATUS, status)
            print(f'Training seed {seed} to {UPDATE} optimizer updates.', flush=True)
            train_seed(seed)
            status['completed'].append(seed)
        status.update(status='completed', current_seed=None)
    except BaseException as error:
        status.update(status='failed', error=str(error))
        raise
    finally:
        write_json(STATUS, status)


if __name__ == '__main__':
    main()
