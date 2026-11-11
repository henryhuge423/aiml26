from pathlib import Path
import torch
from torchvision.models import efficientnet_b0,resnet50
from .data import sha256

INITIALIZATIONS={
    'efficientnet_b0':('efficientnet_b0_rwightman-7f5810bc.pth','7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934'),
    'resnet50':('resnet50-11ad3fa6.pth','11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca'),
}


def create_model(architecture: str,checkpoint_root: Path) -> torch.nn.Module:
    filename,expected=INITIALIZATIONS[architecture]
    initialization=checkpoint_root/filename
    if sha256(initialization)!=expected:raise ValueError('initialization hash mismatch')
    model={'efficientnet_b0':efficientnet_b0,'resnet50':resnet50}[architecture](weights=None)
    model.load_state_dict(torch.load(initialization,map_location='cpu',weights_only=True),strict=True)
    if architecture=='efficientnet_b0':
        features=model.classifier[1].in_features
        model.classifier[1]=torch.nn.Linear(features,1)
    else:model.fc=torch.nn.Linear(model.fc.in_features,1)
    return model
