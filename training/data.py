from __future__ import annotations
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import torch
from PIL import Image, ImageOps
from torchvision import transforms as T

SPLITS={'train','val','boundary','test','external_acrima','external_hygd'}
FIELDS={'sample_id','path','sha256','source','label','target_definition','related_group',
        'patient_id_status','split','task','eligible','review_status'}


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def load_rows(manifest: Path, task: str, split: str, release: Path|None=None) -> list[dict]:
    if release is not None:
        if (release.parent/'SUPERSEDED_BY.json').exists():raise ValueError('data release is superseded')
        metadata=json.loads(release.read_text(encoding='utf-8'))
        if sha256(manifest)!=metadata['manifest_sha256']:raise ValueError('manifest release hash mismatch')
        if metadata.get('research_training_released') is not True or metadata.get('version')!='formal-research-v1':
            raise ValueError('research release is not frozen/recognized')
        if not metadata.get('input_fingerprints'):raise ValueError('release input provenance missing')
        project=(release.parent/metadata['project_root_from_release']).resolve()
        for relative,expected in metadata['input_fingerprints'].items():
            source=(project/relative).resolve()
            if not source.is_relative_to(project) or sha256(source)!=expected:
                raise ValueError('release source input hash mismatch')
    with manifest.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if not FIELDS.issubset(reader.fieldnames or []):raise ValueError('manifest required fields missing')
        rows=list(reader)
    ids=set(); groups=collections.defaultdict(set)
    for r in rows:
        if r['sample_id'] in ids:raise ValueError('duplicate sample_id')
        ids.add(r['sample_id'])
        if r['eligible'].lower()!='true':raise ValueError('manifest contains ineligible row')
        if r['split'] not in SPLITS:raise ValueError('unknown/unassigned split')
        if r['task'] not in {'glaucoma','quality'}:raise ValueError('unknown task')
        if not r['related_group']:raise ValueError('missing related group')
        groups[r['related_group']].add(r['split'])
        if r['task']=='quality' and r['split'] not in {'train','val'}:
            raise ValueError('quality training manifest includes protected test/boundary split')
    if any(len(s)>1 for s in groups.values()):raise ValueError('related group crosses split boundaries')
    selected=[dict(r) for r in rows if r['task']==task and r['split']==split]
    if not selected:raise ValueError(f'empty task/split: {task}/{split}')
    if split=='train' and len(selected)<2:raise ValueError('training requires at least two images')
    if task=='glaucoma':
        if any(r['label'] not in {'0','1'} for r in selected):raise ValueError('nonbinary disease label')
        if split in {'train','val'} and {r['label'] for r in selected}!={'0','1'}:
            raise ValueError('training/validation requires both disease classes')
    return selected


def attach_teacher_targets(rows: list[dict], cache: dict) -> list[dict]:
    result=[]
    for r in rows:
        target=cache.get(r['sample_id'])
        if target is None:raise ValueError(f'missing teacher target for {r["sample_id"]}')
        if target['sha256']!=r['sha256']:raise ValueError('teacher cache image hash mismatch')
        q=float(target['teacher_q'])
        if not math.isfinite(q):raise ValueError('nonfinite teacher target')
        result.append(dict(r,teacher_q=q))
    return result


def epoch_indices(rows: list[dict], seed: int, epoch: int, balance_sources: bool) -> list[int]:
    generator=torch.Generator().manual_seed(seed+epoch)
    if not balance_sources:return torch.randperm(len(rows),generator=generator).tolist()
    counts=collections.Counter((r['source'],r['label']) for r in rows)
    classes=collections.defaultdict(set)
    for src,label in counts:classes[src].add(label)
    weights=torch.tensor([1/(len(classes[r['source']])*counts[r['source'],r['label']]) for r in rows],dtype=torch.double)
    return torch.multinomial(weights,len(rows),replacement=True,generator=generator).tolist()


class ImageRows(torch.utils.data.Dataset):
    def __init__(self,rows,data_root,size,task,train=False,seed=20261109,epoch=0):
        self.rows,self.root,self.task=rows,Path(data_root).resolve(),task
        self.train,self.seed,self.epoch=train,seed,epoch
        self.transform=T.Compose([T.Resize((size,size))]+([T.RandomHorizontalFlip()] if train else [])+
            [T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index];path=(self.root/row['path']).resolve()
        if not path.is_relative_to(self.root):raise ValueError('image outside data root')
        if sha256(path)!=row['sha256']:raise ValueError(f'source changed: {row["path"]}')
        with Image.open(path) as image:
            image.load(); image=image.convert('RGB')
        if self.task=='glaucoma':
            side=max(image.size)
            image=ImageOps.pad(image,(side,side),color=0)
        # Per-image/epoch augmentation is deterministic and does not consume model/dropout RNG.
        image_seed=int(hashlib.sha256(f'{self.seed}:{self.epoch}:{row["sample_id"]}'.encode()).hexdigest()[:15],16)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(image_seed)
            tensor=self.transform(image)
        target=float(row['label'] if self.task=='glaucoma' else row['teacher_q'])
        return tensor,torch.tensor(target,dtype=torch.float32),row['sample_id']
