#!-*- coding:utf-8 -*-

from clusterers.dyngan_utils.base_feature_extractor import Feature_Extractor

class Discriminator(Feature_Extractor):
    def __init__(self, fe_type="base", device=None, **kwargs):
        super().__init__(fe_type=fe_type, device=device)

        dis_kwargs = kwargs['dis']
        self.model = dis_kwargs['D']

        self.model = self.model.to(self.device)

    def forward(self, x):
        # range of x: [-1, 1]
        return self.model(x, y=None, get_features=True)

    def load(self, path):
        pass
