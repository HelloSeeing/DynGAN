#! -*-coding:utf-8 -*-

import torch
import torch.nn as nn
from copy import deepcopy

class Feature_Extractor(nn.Module):

    def __init__(self, fe_type="base", device=None) -> None:

        super(Feature_Extractor, self).__init__()
        
        self.fe_type = fe_type
        self.model = None

        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cpu")

    def forward(self, x):
        # range of x [-1, 1]
        x = x.to(self.device).to(torch.float32)
        f = self.model(x)

        return f

    def load(self, path):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)

    def update(self, model):
        self.model = deepcopy(model)
