#!-*- coding:utf-8 -*-

import torch
import torch.nn as nn
from torchvision.transforms import transforms

from clusterers.dyngan_utils.base_feature_extractor import Feature_Extractor
from clusterers.dyngan_utils.simsiam_utils.simsiam import SimSiam
from torchvision.models import resnet50, resnet18
from clusterers.dyngan_utils.simsiam_utils.backbones import resnet18_cifar_variant1, resnet18_cifar_variant2
from .simsiam_utils.eval_aug import imagenet_norm

def get_backbone(backbone, castrate=True):
    backbone = eval(f"{backbone}()")

    if castrate:
        backbone.output_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()

    return backbone


def get_model(backbone, proj_layers=None):    

    model = SimSiam(get_backbone(backbone))
    if proj_layers is not None:
        model.projector.set_layers(proj_layers)
        
    return model

class SimSiamFE(Feature_Extractor):
    def __init__(self, fe_type="base", device=None, **kwargs) -> None:
        super().__init__(fe_type, device)

        simsiam_kwargs = kwargs['res18_simsiam']
        backbone = simsiam_kwargs['backbone']
        proj_layers = simsiam_kwargs['proj_layers']
        load_path = simsiam_kwargs['load_path']

        self.model = get_model(backbone, proj_layers)
        self.model = self.model.to(self.device)

        self.load(load_path)

        self.transform = transforms.Normalize(*imagenet_norm)

    def forward(self, x):
        # range of x is [-1, 1]
        x = (x + 1) / 2                # rescale to [0, 1]
        x = self.transform(x)

        return self.model.encoder(x)

    def load(self, path):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict['state_dict'])

