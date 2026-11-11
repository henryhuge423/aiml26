"""Pinned original FTHNet; deterministic crops and original CUDA SoftPool."""
from __future__ import annotations
import ast
import hashlib
import importlib.util
import sys
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms as T
from .data import sha256

PREPROCESS_VERSION='rgb_resize512_flip_crop384_imagenetnorm_mean20_sha_seed_v1'
TEACHER_SHA256='24335d54083f003a0b7fb011169ef06d1c4967aca0f807bad1ac0240aa35df03'


def implementation_fingerprint(root: Path) -> dict:
    vendor=root/'models/third_party/BasiQA-bb9b92268256c0398a9909a137d750537aa6b479/basiqa'
    softpool=Path(importlib.util.find_spec('SoftPool').origin).parent
    binary=Path(importlib.util.find_spec('softpool_cuda').origin)
    return {'adapter':sha256(Path(__file__)),
            'fthnet_arch':sha256(vendor/'archs/fthnet4_arch.py'),
            'target_network':sha256(vendor/'models/hyperiqa_model.py'),
            'softpool_python':sha256(softpool/'idea.py'),'softpool_binary':sha256(binary)}


class FTHNetTeacher:
    def __init__(self,root: Path):
        if not torch.cuda.is_available():raise RuntimeError('original teacher requires CUDA SoftPool')
        vendor=root/'models/third_party/BasiQA-bb9b92268256c0398a9909a137d750537aa6b479/basiqa'
        sys.path.insert(0,str(vendor))
        spec=importlib.util.spec_from_file_location('_training_fthnet4',vendor/'archs/fthnet4_arch.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.model=module.FTHNet4(ffa_out_ch=32,tn_in_ch=384,hyper_in_ch=384,embed_dim=64,
            depths=[2,2,6,2],num_heads=[2,4,8,16],window_size=12,
            drop_rate=0.,attn_drop_rate=0.,drop_path_rate=0.1)
        weights=root/'models/checkpoints/net_g_226264S4.pth'
        if sha256(weights)!=TEACHER_SHA256:raise ValueError('teacher checkpoint full hash mismatch')
        state=torch.load(weights,map_location='cpu',weights_only=True)['params']
        self.model.load_state_dict(state,strict=True);self.model=self.model.cuda().eval()
        # Upstream .view() assumes contiguous NCHW convolution outputs. Batched
        # hypernetwork features can be channels-last on this torch build.
        # Materialize the same values, leaving weights/operators/source untouched.
        for layer in self.model.weight_conv:
            layer.register_forward_hook(lambda module,args,output:output.contiguous())
        tree=ast.parse((vendor/'models/hyperiqa_model.py').read_text(encoding='utf-8'))
        nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in {'TargetNet','TargetFC'}]
        if len(nodes)!=2:raise RuntimeError('upstream TargetNet layout changed')
        scope={'nn':torch.nn,'F':torch.nn.functional}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'upstream_targetnet','exec'),scope)
        self.target=scope['TargetNet']
        self.preprocess=T.Compose([T.RandomHorizontalFlip(),T.RandomCrop(384),
            T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])

    @torch.inference_mode()
    def predict(self,path: Path,content_sha: str,seed: int=20261109,crop_batch: int=8) -> tuple[float,float]:
        if crop_batch<1:raise ValueError('crop_batch must be positive')
        with Image.open(path) as image:image.load();image=image.convert('RGB')
        # Resize is deterministic and consumes no RNG; doing it once yields the
        # identical twenty input crops while avoiding repeated CPU resampling.
        image=T.functional.resize(image,[512,512])
        image_seed=int(hashlib.sha256(f'{seed}:{content_sha}'.encode()).hexdigest()[:15],16)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(image_seed)
            inputs=torch.stack([self.preprocess(image) for _ in range(20)])
        values=[]
        for batch in inputs.split(crop_batch):
            parameters=self.model(batch.cuda())
            q=self.target(parameters)(parameters['target_in_vec']).reshape(-1)
            if len(q)!=len(batch) or not torch.isfinite(q).all():raise RuntimeError('invalid teacher score')
            values.append(q.cpu())
        scores=torch.cat(values)
        return float(scores.mean().item()),float(scores.std(unbiased=False).item())
