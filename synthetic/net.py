# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()

def param_hook(grad):
    return torch.norm(grad)

class Generator(nn.Module):
    def __init__(self, zc_dim=32, n_hidden=128, in_channels=2):
        super(Generator, self).__init__()

        self.linear1 = nn.Linear(zc_dim, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_hidden)
        self.linear3 = nn.Linear(n_hidden, in_channels)
        # self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.activation = nn.ReLU()
        # self.activation = nn.Tanh()

    def forward(self, x, output_f=False):
        x = self.activation(self.linear1(x))
        f = self.activation(self.linear2(x))
        x = self.linear3(f)

        if output_f:
            return x, f
            
        return x

class Discriminator(nn.Module):
    def __init__(self, n_hidden=128, x_depth=2, ifsigmoid=True):
        super(Discriminator, self).__init__()

        self.ifsigmoid = ifsigmoid

        self.linear1 = nn.Linear(x_depth, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_hidden)
        self.linear3 = nn.Linear(n_hidden, 1)
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        # self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.linear3(x)
        if self.ifsigmoid:
            x = torch.sigmoid(x)

        return x


class CDiscriminator(nn.Module):
    def __init__(self, n_hidden=128, x_depth=2, c_depth=1):
        super(CDiscriminator, self).__init__()

        self.linear1 = nn.Linear(x_depth + c_depth, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_hidden)
        self.linear3 = nn.Linear(n_hidden, 1)
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        # self.activation = nn.ReLU()

    def forward(self, x, c):
        x = torch.cat([x, c], dim=1)
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        feat = x
        x = torch.sigmoid(self.linear3(x))
        # x = self.linear3(x)

        return x, feat

def weights_init(m):
    """
    Initialise weights of the model.
    """
    if(type(m) == nn.Linear):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.zeros_(m.bias.data)