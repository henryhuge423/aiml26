"""Verify the pinned upstream teacher on a real image; this is not model evaluation."""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path


def run(image: Path, output: Path, crops: int=20, seed: int=20261109):
    import torch
    import torchvision
    from PIL import Image
    from SoftPool import SoftPool2d
    from torchvision import transforms as T
    root=Path(__file__).resolve().parents[1]
    vendor=root/'models/third_party/BasiQA-bb9b92268256c0398a9909a137d750537aa6b479/basiqa'
    weights=root/'models/checkpoints/net_g_226264S4.pth'
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required for original SoftPool verification')
    if crops<1:raise ValueError('crops must be positive')
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    probe=torch.arange(16,dtype=torch.float32,device='cuda').reshape(1,1,4,4)
    pooled=SoftPool2d(2)(probe)
    if not torch.isfinite(pooled).all():raise RuntimeError('nonfinite original SoftPool output')
    windows=torch.nn.functional.unfold(probe,kernel_size=2,stride=2).reshape(1,1,4,4)
    expected=(windows.softmax(dim=2)*windows).sum(dim=2).reshape(1,1,2,2)
    if not torch.allclose(pooled,expected,atol=1e-5,rtol=1e-5):
        raise RuntimeError('CUDA SoftPool disagrees with direct exponential-weighted pooling')
    # Load just the selected architecture, avoiding unrelated architecture auto-registration.
    sys.path.insert(0,str(vendor))
    spec=importlib.util.spec_from_file_location('_eye_fthnet4',vendor/'archs/fthnet4_arch.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    model=module.FTHNet4(ffa_out_ch=32,tn_in_ch=384,hyper_in_ch=384,embed_dim=64,
        depths=[2,2,6,2],num_heads=[2,4,8,16],window_size=12,
        drop_rate=0.,attn_drop_rate=0.,drop_path_rate=0.1)
    loaded=torch.load(weights,map_location='cpu',weights_only=True)
    model.load_state_dict(loaded['params'],strict=True)
    model=model.cuda().eval()
    # Execute the exact upstream TargetNet/TargetFC class bodies; no rewritten pooling or target network.
    target_file=vendor/'models/hyperiqa_model.py'
    tree=ast.parse(target_file.read_text(encoding='utf-8'))
    selected=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ('TargetNet','TargetFC')]
    if len(selected)!=2:raise RuntimeError('upstream TargetNet class layout changed')
    namespace={'nn':torch.nn,'F':torch.nn.functional}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(target_file),'exec'),namespace)
    preprocess=T.Compose([T.Resize((512,512)),T.RandomHorizontalFlip(),T.RandomCrop(384),T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    with Image.open(image) as im:im.load(); rgb=im.convert('RGB')
    scores=[]; started=time.time()
    with torch.inference_mode():
        for _ in range(crops):
            batch=preprocess(rgb).unsqueeze(0).cuda()
            params=model(batch)
            score=namespace['TargetNet'](params).to('cuda')(params['target_in_vec'])
            if score.numel()!=1 or not torch.isfinite(score).all():raise RuntimeError('invalid teacher scalar')
            scores.append(float(score.item()))
    torch.cuda.synchronize()
    report=dict(passed=True,test_type='real-image strict-load CUDA forward smoke; not performance evaluation',
        image=str(image.resolve()),image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
        checkpoint_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),checkpoint_key='params',strict_load=True,
        source_revision='bb9b92268256c0398a9909a137d750537aa6b479',
        python=platform.python_version(),torch=torch.__version__,torchvision=torchvision.__version__,
        cuda_build=torch.version.cuda,gpu=torch.cuda.get_device_name(),
        softpool_backend='unmodified upstream CUDA extension',softpool_probe=pooled.detach().cpu().tolist(),
        softpool_reference_max_abs_error=float((pooled-expected).abs().max().item()),
        preprocess='RGB, resize512, random-horizontal-flip, random-crop384, ImageNet normalize',
        seed=seed,crops=crops,scores_raw_normalized=scores,q_mean_raw_normalized=sum(scores)/len(scores),
        output_clipping=False,elapsed_seconds=time.time()-started,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),training_started=False)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--crops',type=int,default=20)
    parser.add_argument('--seed',type=int,default=20261109)
    args=parser.parse_args()
    run(args.image,args.output,args.crops,args.seed)
