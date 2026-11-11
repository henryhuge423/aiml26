"""Fit continuous teacher scores and ordered EyeQ grades on a shared q scale."""
from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch.nn import functional as F

from .data import attach_teacher_targets, epoch_indices, load_rows, sha256
from .engine import (fingerprint_inputs, load_cache, loader, optimizer_step,
                     restore_checkpoint, save_checkpoint)
from .models import create_model


def ordinal_logits(q, center, gap_parameter):
    # The first logit is for Good or Usable, the second is for Good.
    gap = F.softplus(gap_parameter)
    thresholds = center + torch.stack((-gap / 2, gap / 2))
    return 10 * (q.float()[:, None] - thresholds)


def ordinal_loss(logits, grades):
    observed = grades >= 0
    target = torch.stack((grades <= 1, grades == 0), dim=1).float()
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none').mean(dim=1)
    return (loss * observed).sum() / observed.sum().clamp_min(1)


class QualityModel(torch.nn.Module):
    def __init__(self, student):
        super().__init__()
        self.student = student
        self.center = torch.nn.Parameter(torch.tensor(.5))
        self.gap_parameter = torch.nn.Parameter(torch.tensor(-2.))

    def forward(self, x):
        q = self.student(x).flatten().float()
        return q, ordinal_logits(q, self.center, self.gap_parameter)


def grade_tensor(ids, by_id, device):
    return torch.tensor([int(by_id[s]['eyeq_grade'] or '-1') for s in ids], device=device)


def validate(model, rows, config, root):
    model.eval()
    by_id = {r['sample_id']: r for r in rows}
    predictions, truth, grades = [], [], []
    huber, mae, ordinal = 0., 0., 0.
    with torch.inference_mode():
        for x, teacher, ids in loader(rows, config, root, False):
            q, logits = model(x.to(config['device']))
            teacher = teacher.to(q.device)
            actual = grade_tensor(ids, by_id, q.device)
            estimated = 2 - (logits > 0).sum(dim=1)
            if not torch.isfinite(q).all() or not torch.isfinite(logits).all():
                raise RuntimeError('nonfinite validation output')
            huber += F.huber_loss(q, teacher, reduction='sum', delta=.1).item()
            mae += (q - teacher).abs().sum().item()
            ordinal += ordinal_loss(logits, actual).item() * (actual >= 0).sum().item()
            for sample, score, target, grade, prediction in zip(
                    ids, q.tolist(), teacher.tolist(), actual.tolist(), estimated.tolist()):
                if grade >= 0:
                    truth.append(grade)
                    grades.append(prediction)
                row = by_id[sample]
                predictions.append(dict(sample_id=sample, source=row['source'], sha256=row['sha256'],
                                        related_group=row['related_group'], q=score, teacher_q=target,
                                        eyeq_grade=grade if grade >= 0 else '',
                                        predicted_grade=prediction if grade >= 0 else ''))
    model.train()
    ordinal /= len(truth)
    return dict(n=len(rows), eyeq_n=len(truth), teacher_huber=huber / len(rows),
                teacher_mae=mae / len(rows), eyeq_ordinal_loss=ordinal,
                eyeq_macro_f1=float(f1_score(truth, grades, labels=[0, 1, 2], average='macro', zero_division=0)),
                eyeq_confusion=confusion_matrix(truth, grades, labels=[0, 1, 2]).tolist(),
                eyeq_grade_mae=sum(abs(a - b) for a, b in zip(truth, grades)) / len(truth),
                selection_loss=huber / len(rows) + config['ordinal_weight'] * ordinal), predictions


def train(config_path: Path, root: Path, resume: Path | None = None, max_steps: int | None = None):
    config = json.loads(config_path.read_text(encoding='utf-8'))
    torch.set_num_threads(4)
    device, seed = config['device'], config['seed']
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('requested CUDA unavailable')
    if config['task'] != 'quality' or not math.isfinite(config['ordinal_weight']) or config['ordinal_weight'] <= 0:
        raise ValueError('EyeQ training requires positive finite ordinal weight and the quality task')
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    manifest, release = (root / config[key] for key in ('manifest', 'release'))
    cache = load_cache(root / config['teacher_cache'], manifest, root, release)
    train_rows, val_rows = [attach_teacher_targets(load_rows(manifest, 'quality', split, release), cache)
                            for split in ('train', 'val')]
    for rows in (train_rows, val_rows):
        if any(r['eyeq_grade'] not in ({'0', '1', '2'} if r['source'] == 'EyeQ' else {''}) for r in rows):
            raise ValueError('unexpected human grade/source in release')
    if config['purpose'] == 'preflight':
        train_rows, val_rows = [
            [r for r in rows if r['source'] == 'EyeQ'][:16] +
            [r for r in rows if r['source'] != 'EyeQ'][:other]
            for rows, other in ((train_rows, 32), (val_rows, 16))]
    fingerprint = fingerprint_inputs(config, root) + ':' + sha256(Path(__file__))
    run = root / config['run_dir']
    run.mkdir(parents=True, exist_ok=True)
    last = run / 'last.pt'
    steps = run / 'steps.jsonl'
    epochs = run / 'epochs.jsonl'
    if resume is None and (last.exists() or steps.exists()):
        raise ValueError('run already has training evidence; resume or choose a fresh run directory')
    (run / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    model = QualityModel(create_model(config['architecture'], root / 'models/checkpoints')).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=config['amp'] and device == 'cuda', init_scale=1024)
    state = dict(epoch=0, next_batch=0, global_step=0, optimizer_updates=0,
                 consecutive_amp_overflows=0, best_selection=1e30)
    if resume is not None:
        state = restore_checkpoint(resume, model, optimizer, scaler, fingerprint, device)
        for log, committed in ((steps, state['global_step']), (epochs, state['epoch'])):
            if log.exists():
                lines = log.read_bytes().splitlines(keepends=True)
                if len(lines) < committed:
                    raise ValueError(f'checkpoint is ahead of the {log.name} log')
                if len(lines) > committed:
                    log.with_suffix(f'.before_resume_{len(lines)}.jsonl').write_bytes(b''.join(lines))
                    log.write_bytes(b''.join(lines[:committed]))
    by_id = {r['sample_id']: r for r in train_rows}
    batch_size = config['batch_size']
    if max_steps is not None and max_steps <= state['global_step']:
        raise ValueError('requested step limit must exceed the checkpoint step')
    last_loss = None
    model.train()
    for epoch in range(state['epoch'], config['epochs']):
        order = epoch_indices(train_rows, seed, epoch, False)
        if len(order) % batch_size == 1:
            order = order[:-1]
        for batch, (x, y, ids) in enumerate(loader(train_rows, config, root, True, epoch, order)):
            if batch < state['next_batch']:
                continue
            x, y = x.to(device), y.to(device)
            actual = grade_tensor(ids, by_id, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=scaler.is_enabled()):
                q, logits = model(x)
                distillation = F.huber_loss(q, y, delta=.1)
                ordinal = ordinal_loss(logits, actual)
                loss = distillation + config['ordinal_weight'] * ordinal
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite EyeQ training loss')
            event = optimizer_step(loss, optimizer, scaler, model.parameters(), config['gradient_clip'])
            state['optimizer_updates'] += int(event['optimizer_updated'])
            state['consecutive_amp_overflows'] = 0 if event['optimizer_updated'] else state['consecutive_amp_overflows'] + 1
            if state['consecutive_amp_overflows'] >= 8:
                raise RuntimeError('eight consecutive AMP overflows')
            last_loss = loss.item()
            state.update(epoch=epoch, next_batch=batch + 1, global_step=state['global_step'] + 1)
            record = dict(step=state['global_step'], epoch=epoch, batch=batch, loss=last_loss,
                          teacher_loss=distillation.item(), ordinal_loss=ordinal.item(),
                          eyeq_rows=int((actual >= 0).sum().item()), **event)
            with steps.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record) + '\n')
            reached_limit = max_steps is not None and state['global_step'] >= max_steps
            if state['global_step'] in (1, max_steps) or state['global_step'] % config['checkpoint_every'] == 0:
                save_checkpoint(last, model, optimizer, scaler, state, fingerprint)
            if reached_limit:
                break
        if max_steps is not None and state['global_step'] >= max_steps:
            break
        metrics, predictions = validate(model, val_rows, config, root)
        state.update(epoch=epoch + 1, next_batch=0)
        if metrics['selection_loss'] < state['best_selection']:
            state['best_selection'] = metrics['selection_loss']
            with (run / 'best_validation_predictions.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
                writer.writeheader()
                writer.writerows(predictions)
            save_checkpoint(run / 'best.pt', model, optimizer, scaler, state, fingerprint)
        with epochs.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(epoch=epoch + 1, **metrics)) + '\n')
        save_checkpoint(last, model, optimizer, scaler, state, fingerprint)
        print(json.dumps(dict(epoch=epoch + 1, **metrics)), flush=True)
    result = dict(state, last_loss=last_loss, fingerprint=fingerprint, training_rows=len(train_rows),
                  validation_rows=len(val_rows), purpose=config['purpose'])
    (run / 'run_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result
