#!-*- coding:utf-8 -*-

import torch
import torch.nn as nn
from torchvision.models import resnet18

from clusterers.dyngan_utils.base_feature_extractor import Feature_Extractor

class Resnet18(Feature_Extractor):
    def __init__(self, fe_type="base", device=None, **kwargs):
        super(Resnet18, self).__init__(fe_type=fe_type, device=device)

        res18_sup_kwargs = kwargs['res18_sup']
        load_path = res18_sup_kwargs['load_path']

        self.model = resnet18(weights=None)
        self.model.fc = nn.Linear(512, 10)

        self.model = self.model.to(self.device)

        self.load(load_path)

    def forward(self, x):
        # range of x is [-1, 1]
        
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        return x
    
    def load(self, path):
        
        state_dict : dict = torch.load(path)
        # model_state_dict = dict()
        # for key, item in state_dict.items():
        #     model_state_dict[f"model.{key}"] = item
        # self.model.load_state_dict(model_state_dict)
        self.model.load_state_dict(state_dict)
