from __future__ import annotations
import collections
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path
import torch
from sklearn.metrics import roc_auc_score
from .data import ImageRows,attach_teacher_targets,epoch_indices,load_rows,sha256
from .models import INITIALIZATIONS,create_model


def optimizer_step(loss,optimizer,scaler,parameters,gradient_clip):
    """Let GradScaler handle FP16 overflow; never clip invalid gradients.

    https://docs.pytorch.org/docs/stable/notes/amp_examples.html
    A skipped optimizer update is explicitly reported, not a successful update.
    """
    parameters=list(parameters)
    scale_before=float(scaler.get_scale())
    scaler.scale(loss).backward();scaler.unscale_(optimizer)
    checks=[torch.isfinite(p.grad).all() for p in parameters if p.grad is not None]
    if not checks:raise RuntimeError('training produced no gradients')
    finite=bool(torch.stack(checks).all())
    if not finite:
        if not scaler.is_enabled():raise RuntimeError('nonfinite gradients without AMP')
        scaler.step(optimizer);scaler.update()
        scale_after=float(scaler.get_scale())
        if not math.isfinite(scale_after) or not 0<scale_after<scale_before:
            raise RuntimeError('AMP overflow did not reduce a finite positive scale')
        return dict(optimizer_updated=False,gradient_norm=None,
                    scale_before=scale_before,scale_after=scale_after)
    norm=torch.nn.utils.clip_grad_norm_(parameters,gradient_clip,error_if_nonfinite=True)
    scaler.step(optimizer);scaler.update()
    return dict(optimizer_updated=True,gradient_norm=float(norm.item()),
                scale_before=scale_before,scale_after=float(scaler.get_scale()))


def save_checkpoint(path,model,optimizer,scaler,state,fingerprint):
    content=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
        scaler=scaler.state_dict() if scaler is not None else None,state=state,
        fingerprint=fingerprint,rng_cpu=torch.get_rng_state(),rng_python=random.getstate(),
        rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(content,temporary);temporary.replace(path)


def restore_checkpoint(path,model,optimizer,scaler,fingerprint,device):
    content=torch.load(path,map_location='cpu',weights_only=True)
    if content['fingerprint']!=fingerprint:raise ValueError('checkpoint input fingerprint mismatch')
    model.load_state_dict(content['model'],strict=True);optimizer.load_state_dict(content['optimizer'])
    for state in optimizer.state.values():
        for key,value in state.items():
            if isinstance(value,torch.Tensor) and key!='step':state[key]=value.to(device)
    if scaler is not None and content['scaler'] is not None:scaler.load_state_dict(content['scaler'])
    torch.set_rng_state(content['rng_cpu']);random.setstate(content['rng_python'])
    if device!='cpu' and content['rng_cuda']:torch.cuda.set_rng_state_all(content['rng_cuda'])
    return content['state']


def load_cache(path,manifest,root,release):
    from .teacher import implementation_fingerprint
    meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if not meta['complete'] or meta['manifest_sha256']!=sha256(manifest):
        raise ValueError('teacher cache incomplete or bound to different manifest')
    if meta['release_sha256']!=sha256(release):raise ValueError('teacher cache release hash mismatch')
    if meta['teacher_sha256']!=sha256(root/'models/checkpoints/net_g_226264S4.pth'):
        raise ValueError('teacher model hash mismatch')
    if meta['cache_sha256']!=sha256(path):raise ValueError('teacher cache file hash mismatch')
    if meta.get('implementation')!=implementation_fingerprint(root):raise ValueError('teacher implementation hash mismatch')
    with path.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    cache={r['sample_id']:r for r in rows}
    if len(cache)!=len(rows):raise ValueError('duplicate teacher cache identity')
    allowed=load_rows(manifest,'quality','train',release)+load_rows(manifest,'quality','val',release)
    if set(cache)!={r['sample_id'] for r in allowed} or len(cache)!=meta['expected_rows']:
        raise ValueError('teacher cache identity set differs from allowed cohort')
    return cache


def fingerprint_inputs(config,root):
    body={k:v for k,v in config.items() if k!='run_dir'}
    manifest=root/config['manifest'];body['manifest_digest']=sha256(manifest)
    body['release_digest']=sha256(root/config['release'])
    name=INITIALIZATIONS[config['architecture']][0]
    body['initialization_digest']=sha256(root/'models/checkpoints'/name)
    body['training_code']={n:sha256(root/'training'/n) for n in ('data.py','models.py','engine.py')}
    if config.get('degradation'):
        body['training_code']['fundus_degradation.py']=sha256(root/'training/fundus_degradation.py')
    if config['task']=='quality':
        body['teacher_cache_digest']=sha256(root/config['teacher_cache'])
        body['teacher_meta_digest']=sha256((root/config['teacher_cache']).with_suffix('.json'))
    return hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()


def loader(rows,config,root,train,epoch=0,indices=None):
    dataset=ImageRows(rows,root/'data',config['image_size'],config['task'],train,config['seed'],epoch,
        degradation=config.get('degradation'),return_metadata=train and bool(config.get('degradation')))
    if indices is not None:dataset=torch.utils.data.Subset(dataset,indices)
    generator=torch.Generator().manual_seed(config['seed']+epoch+9187)
    workers=config.get('num_workers',0)
    if not isinstance(workers,int) or workers<0:raise ValueError('num_workers must be a nonnegative integer')
    options=dict(prefetch_factor=config.get('prefetch_factor',2)) if workers else {}
    return torch.utils.data.DataLoader(dataset,batch_size=config['batch_size'],shuffle=False,
        num_workers=workers,pin_memory=config['device']=='cuda',generator=generator,**options)


def evaluate(model,rows,config,root,run_dir):
    model.eval();predictions=[];loss_sum=0.;count=0
    criterion=torch.nn.BCEWithLogitsLoss(reduction='sum') if config['task']=='glaucoma' else torch.nn.HuberLoss(delta=.1,reduction='sum')
    by_id={r['sample_id']:r for r in rows}
    with torch.inference_mode():
        for x,y,ids in loader(rows,config,root,False):
            x,y=x.to(config['device']),y.to(config['device'])
            output=model(x).flatten();loss_sum+=criterion(output,y).item();count+=len(y)
            values=output.sigmoid() if config['task']=='glaucoma' else output
            for sample,p,t in zip(ids,values.cpu().tolist(),y.cpu().tolist()):
                r=by_id[sample]
                predictions.append(dict(sample_id=sample,path=r['path'],sha256=r['sha256'],source=r['source'],
                    related_group=r['related_group'],target_definition=r['target_definition'],label=t,prediction=p,
                    label_kind='teacher_prediction' if config['task']=='quality' else 'source_disease_label',
                    human_mos_normalized=r['label'] if config['task']=='quality' else ''))
    metrics={'loss':loss_sum/count,'n':count,'per_source':{}}
    for source in sorted({r['source'] for r in rows}):
        selected=[r for r in predictions if r['source']==source]
        labels=[r['label'] for r in selected];scores=[r['prediction'] for r in selected]
        item={'n':len(selected)}
        if config['task']=='glaucoma':
            item['auc']=float(roc_auc_score(labels,scores)) if len(set(labels))==2 else None
        else:item['mae_to_teacher']=sum(abs(a-b) for a,b in zip(labels,scores))/len(labels)
        metrics['per_source'][source]=item
    if config['task']=='glaucoma':
        sources=config.get('selection_sources') or [config['selection_source']]
        values=[metrics['per_source'].get(source,{}).get('auc') for source in sources]
        if any(auc is None for auc in values):raise ValueError('validation selection source lacks both classes')
        metrics['selection_value']=sum(values)/len(values)
        metrics['selection_sources']=sources
    else:metrics['selection_value']=-metrics['loss']
    with (run_dir/'validation_predictions.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(predictions[0]));writer.writeheader();writer.writerows(predictions)
    model.train();return metrics


def train(config_path: Path,root: Path,resume: Path|None=None,max_steps: int|None=None):
    config=json.loads(config_path.read_text(encoding='utf-8'))
    torch.set_num_threads(config.get('cpu_threads',4))
    if config['device']=='cuda' and not torch.cuda.is_available():raise RuntimeError('requested CUDA unavailable')
    if config['batch_size']<2:raise ValueError('training batch must have at least two images')
    random.seed(config['seed']);torch.manual_seed(config['seed'])
    if torch.cuda.is_available():torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    manifest=root/config['manifest'];release=root/config['release'];task=config['task']
    train_rows=load_rows(manifest,task,'train',release);val_rows=load_rows(manifest,task,'val',release)
    if config.get('training_origins'):
        train_rows=[r for r in train_rows if r.get('origin_cohort') in config['training_origins']]
        if len(train_rows)<2 or {r['label'] for r in train_rows}!={'0','1'}:
            raise ValueError('training origin filter needs nonempty binary cohort')
    update_budget=config.get('max_optimizer_updates')
    validation_interval=config.get('validation_every_updates')
    for name,value in [('max_optimizer_updates',update_budget),('validation_every_updates',validation_interval)]:
        if value is not None and (not isinstance(value,int) or isinstance(value,bool) or value<1):
            raise ValueError(f'{name} must be a positive integer')
    if task=='quality':
        cache=load_cache(root/config['teacher_cache'],manifest,root,release)
        train_rows=attach_teacher_targets(train_rows,cache);val_rows=attach_teacher_targets(val_rows,cache)
    if config.get('purpose')=='preflight':
        # This is an explicitly named smoke subset, never a research performance result.
        train_rows=train_rows[:config.get('preflight_train_rows',16)]
        val_rows=val_rows[:config.get('preflight_val_rows',16)]
        if task=='glaucoma':
            val_rows=[]
            for label in ('0','1'):
                val_rows += [r for r in load_rows(manifest,task,'val',release)
                             if r['source']==config['selection_source'] and r['label']==label][:8]
    binding=fingerprint_inputs(config,root)
    run_dir=root/config['run_dir'];run_dir.mkdir(parents=True,exist_ok=True)
    if resume is None and (run_dir/'last.pt').exists():raise ValueError('run already exists; use --resume or a new run_dir')
    (run_dir/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    model=create_model(config['architecture'],root/'models/checkpoints').to(config['device'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',enabled=config.get('amp',False) and config['device']=='cuda',init_scale=1024)
    state={'epoch':0,'next_batch':0,'global_step':0,'best_selection':-1e30}
    if resume is not None:
        state=restore_checkpoint(resume,model,optimizer,scaler,binding,config['device'])
        steps_path=run_dir/'steps.jsonl'
        if steps_path.exists():
            records=[json.loads(line) for line in steps_path.read_text(encoding='utf-8').splitlines() if line.strip()]
            committed=[r for r in records if r['step']<=state['global_step']]
            if len(committed)!=len(records):
                backup=run_dir/f'steps_before_recovery_{time.time_ns()}.jsonl'
                backup.write_bytes(steps_path.read_bytes())
                steps_path.write_text(''.join(json.dumps(r)+'\n' for r in committed),encoding='utf-8')
        trace_path=run_dir/'degradation_trace.jsonl'
        if trace_path.exists():
            records=[json.loads(line) for line in trace_path.read_text(encoding='utf-8').splitlines() if line.strip()]
            committed=[r for r in records if r['step']<=state['global_step']]
            if len(committed)!=len(records):
                (run_dir/f'degradation_trace_before_recovery_{time.time_ns()}.jsonl').write_bytes(trace_path.read_bytes())
                trace_path.write_text(''.join(json.dumps(r)+'\n' for r in committed),encoding='utf-8')
        with (run_dir/'resume_events.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(checkpoint=str(resume),checkpoint_sha256=sha256(resume),
                                    committed_global_step=state['global_step']))+'\n')
    if max_steps is not None and max_steps<=state['global_step']:
        raise ValueError('max_steps must exceed the resumed global_step')
    model.train();criterion=torch.nn.BCEWithLogitsLoss() if task=='glaucoma' else torch.nn.HuberLoss(delta=.1)
    before=next(model.parameters()).detach().clone();last_loss=None;started=time.time()
    by_id={r['sample_id']:r for r in train_rows}
    def validate(epoch):
        metrics=evaluate(model,val_rows,config,root,run_dir)
        if metrics['selection_value']>state['best_selection']:
            state['best_selection']=metrics['selection_value']
            save_checkpoint(run_dir/'best.pt',model,optimizer,scaler,state,binding)
            (run_dir/'best_validation_predictions.csv').write_bytes((run_dir/'validation_predictions.csv').read_bytes())
        state['last_validation_update']=state['optimizer_updates']
        with (run_dir/'epochs.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(epoch=epoch,step=state['global_step'],optimizer_updates=state['optimizer_updates'],**metrics))+'\n')
    budget_reached=bool(update_budget and state.get('optimizer_updates',0)>=update_budget)
    if budget_reached:raise ValueError('optimizer update budget already completed')
    for epoch in range(state['epoch'],config['epochs']):
        order=epoch_indices(train_rows,config['seed'],epoch,task=='glaucoma',config.get('sampler'))
        # Drop a singleton tail because BatchNorm training may require >1 item.
        if len(order)%config['batch_size']==1:order=order[:-1]
        batches=loader(train_rows,config,root,True,epoch,order)
        skip=state['next_batch'] if epoch==state['epoch'] else 0
        for batch_index,batch in enumerate(batches):
            if batch_index<skip:continue
            x,y,ids=batch[:3]
            x,y=x.to(config['device']),y.to(config['device'])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=config['device'],enabled=scaler.is_enabled()):
                output=model(x).flatten();loss=criterion(output,y)
            if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
            event=optimizer_step(loss,optimizer,scaler,model.parameters(),config.get('gradient_clip',5.0))
            state['optimizer_updates']=state.get('optimizer_updates',state['global_step'])+int(event['optimizer_updated'])
            state['amp_overflows']=state.get('amp_overflows',0)+int(not event['optimizer_updated'])
            state['consecutive_amp_overflows']=0 if event['optimizer_updated'] else state.get('consecutive_amp_overflows',0)+1
            if state['consecutive_amp_overflows']>=8:raise RuntimeError('eight consecutive AMP overflows; refusing unstable training')
            last_loss=float(loss.item())
            state.update(epoch=epoch,next_batch=batch_index+1,global_step=state['global_step']+1)
            record=dict(step=state['global_step'],epoch=epoch,batch=batch_index,loss=last_loss,**event)
            seen=state.setdefault('sample_draw_counts',{})
            strata=state.setdefault('source_class_draws',{})
            families=state.setdefault('degradation_family_draws',{})
            for sample in ids:
                seen[sample]=seen.get(sample,0)+1
                row=by_id[sample];key=row['source']+'/'+row['label']
                strata[key]=strata.get(key,0)+1
            if len(batch)>3:
                metadata=[json.loads(value) for value in batch[3]]
                degradation_strata=state.setdefault('source_class_degradation_draws',{})
                for sample,rowmeta in zip(ids,metadata):
                    family=rowmeta.get('family','original') if rowmeta.get('applied') else 'original'
                    families[family]=families.get(family,0)+1
                    row=by_id[sample];key=row['source']+'/'+row['label']+'/'+family
                    degradation_strata[key]=degradation_strata.get(key,0)+1
                if config.get('record_degradation_trace'):
                    with (run_dir/'degradation_trace.jsonl').open('a',encoding='utf-8') as trace:
                        for sample,rowmeta in zip(ids,metadata):
                            if rowmeta.get('applied'):
                                trace.write(json.dumps(dict(step=state['global_step'],epoch=epoch,sample_id=sample,**rowmeta))+'\n')
                record['degradation_applied']=sum(bool(m.get('applied')) for m in metadata)
            if validation_interval and event['optimizer_updated'] and state['optimizer_updates']%validation_interval==0:
                validate(epoch)
            budget_reached=bool(update_budget and state['optimizer_updates']>=update_budget)
            with (run_dir/'steps.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
            if state['global_step']%config.get('checkpoint_every',100)==0 or (max_steps and state['global_step']>=max_steps) or budget_reached:
                save_checkpoint(run_dir/'last.pt',model,optimizer,scaler,state,binding)
            if (max_steps is not None and state['global_step']>=max_steps) or budget_reached:break
        if budget_reached:
            if state.get('last_validation_update')!=state['optimizer_updates']:validate(epoch)
            save_checkpoint(run_dir/'last.pt',model,optimizer,scaler,state,binding)
            break
        if max_steps is not None and state['global_step']>=max_steps:break
        if not validation_interval and state.get('last_validation_update')!=state['optimizer_updates']:validate(epoch)
        state.update(epoch=epoch+1,next_batch=0)
        save_checkpoint(run_dir/'last.pt',model,optimizer,scaler,state,binding)
    changed=bool(torch.any(next(model.parameters()).detach()!=before).item())
    result=dict(global_step=state['global_step'],last_loss=last_loss,parameters_changed=changed,
        purpose=config.get('purpose','research_training'),fingerprint=binding,elapsed_seconds=time.time()-started,
        peak_gpu_bytes=torch.cuda.max_memory_allocated() if config['device']=='cuda' else None,
        resumed_from=str(resume) if resume else None,training_rows=len(train_rows),validation_rows=len(val_rows),
        optimizer_updates=state.get('optimizer_updates',0),amp_overflows=state.get('amp_overflows',0),
        optimizer_update_budget=update_budget,budget_completed=budget_reached,
        unique_sample_draws=len(state.get('sample_draw_counts',{})),
        unique_linked_groups_drawn=len({by_id[s]['related_group'] for s in state.get('sample_draw_counts',{})}),
        max_sample_repeats=max(state.get('sample_draw_counts',{}).values(),default=0),
        source_class_draws=state.get('source_class_draws',{}),degradation_family_draws=state.get('degradation_family_draws',{}),
        source_class_degradation_draws=state.get('source_class_degradation_draws',{}))
    (run_dir/'run_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2));return result
