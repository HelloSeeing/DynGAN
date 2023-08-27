#! -*-coding:utf-8 -*-

import os
import numpy as np
import torch
import argparse

from tqdm import tqdm
from torch.nn.functional import adaptive_avg_pool2d
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance

class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        
        self.data = data
        self.transform = transform

    def __len__(self):

        return self.data.shape[0]

    def __getitem__(self, idx):
        img = self.data[idx].transpose(1, 2, 0)
        if self.transform is not None:
            img = self.transform(img)
        return img


def get_activations(dataset, model, max_size=None, batch_size=50, dims=2048, device='cpu',
                    num_workers=1):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(dataset):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(dataset)

    assert max_size <= len(dataset)
    if max_size is None:
        max_size = len(dataset)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    pred_arr = np.empty((len(dataset), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        try:
            pred_arr[start_idx:start_idx + pred.shape[0]] = pred
        except ValueError:
            print(pred_arr[start_idx:start_idx + pred.shape[0]].shape)
            print(pred.shape)
            input("wait")

        start_idx = start_idx + pred.shape[0]

        if start_idx >= max_size:
            break

    pred_arr = pred_arr[:max_size]

    return pred_arr

def calculate_frechet_inception_distance(real_imgs, fake_imgs, name="cifar10", dims: int=2048, device="cuda:0"):

    device = torch.device(device)

    transform = transforms.ToTensor()

    real_dset = NumpyDataset(data=real_imgs, transform=transform)
    fake_dset = NumpyDataset(data=fake_imgs, transform=transform)

    max_size = min(50000, len(fake_dset))

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    real_act_path = f"{name}_real_act.npy"
    # if not os.path.exists(real_act_path):
    if True:
        real_act = get_activations(dataset=real_dset, model=model, max_size=max_size, batch_size=50, dims=dims, device=device)
        np.save(real_act_path, real_act)
    else:
        real_act = np.load(real_act_path)

    m1 = np.mean(real_act, axis=0)
    s1 = np.cov(real_act, rowvar=False)

    fake_act = get_activations(dataset=fake_dset, model=model, max_size=max_size, batch_size=50, dims=dims, device=device)

    m2 = np.mean(fake_act, axis=0)
    s2 = np.cov(fake_act, rowvar=False)

    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value
