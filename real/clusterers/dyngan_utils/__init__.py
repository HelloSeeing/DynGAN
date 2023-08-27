from .res18_sup import Resnet18
from .discriminator import Discriminator
from .simsiam import SimSiamFE

feature_extractor_dict = {
    'res18_sup': Resnet18, 
    'discriminator': Discriminator, 
    'res18_simsiam': SimSiamFE
}