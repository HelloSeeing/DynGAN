import numpy as np
import torch
from torch.nn import functional as F
import os
import pickle
import argparse

from gan_training.metrics import inception_score
from gan_training.metrics import frechet_inception_divergence as fid
from gan_training.inputs import get_dataset
from gan_training.config import load_config, build_models, build_clusterer
from gan_training.distributions import get_zdist
from clusterers.dyngan_utils import feature_extractor_dict

BS = 200

def prepare(default_config_path, config_path, model_dir, it, device):

    cfg = load_config(config_path, default_path=default_config_path)
    print(cfg)

    name = cfg['data']['type']

    # load generator
    generator, discriminator = build_models(cfg)

    model_ckpt = os.path.join(model_dir, f"model_{it:08d}.pt")
    generator.load_state_dict(torch.load(model_ckpt)['generator'])

    generator = generator.to(device)
    generator = torch.nn.DataParallel(generator, device_ids=[0])

    discriminator = discriminator.to(device)
    discriminator = torch.nn.DataParallel(discriminator, device_ids=[0])

    generator.eval()
    discriminator.eval()

    # load dataloader
    dset, _ = get_dataset(name=cfg['data']['type'],
                          data_dir=cfg['data']['train_dir'],
                          size=cfg['data']['img_size'],
                          deterministic=cfg['data']['deterministic'])

    data_loader = torch.utils.data.DataLoader(
        dset,
        batch_size=BS,
        num_workers=0,
        shuffle=False,
        pin_memory=True,
        sampler=None,
        drop_last=False)
    
    # load clusterer
    x = np.load(os.path.join(model_dir, "cluster_samples.npz"))['x']
    x = torch.from_numpy(x).to(device)
    clusterer = build_clusterer(cfg, D=discriminator, x_cluster=x, y_cluster=None)
    clusterer.load(os.path.join(model_dir, f"clusterer{it:08d}.pth"))
    # with open(os.path.join(model_dir, f"clusterer{it:d}.pkl"), "rb") as f:
    #     clusterer = pickle.load(f)
    # clusterer.x = x

    # load prior
    zdist = get_zdist(cfg['z_dist']['type'], cfg['z_dist']['dim'], device=device)

    return generator, clusterer, data_loader, zdist, name

class Evaluator(object):
    def __init__(self,
                 generator,
                 zdist,
                 ydist,
                 train_loader,
                 clusterer,
                 batch_size=64,
                 device=None):
        self.generator = generator
        self.clusterer = clusterer
        self.train_loader = train_loader
        self.zdist = zdist
        self.ydist = ydist
        self.batch_size = batch_size
        self.device = device

    def sample_z(self, batch_size):
        return self.zdist.sample((batch_size, )).to(self.device)

    def get_y(self, x, y):
        return self.clusterer.get_labels(x, y).to(self.device)

    def get_fake_real_samples(self, N):
        ''' returns N fake images and N real images in pytorch form'''
        with torch.no_grad():
            self.generator.eval()
            fake_imgs = []
            real_imgs = []
            while len(fake_imgs) < N:
                for x_real, y_gt in self.train_loader:
                    x_real = x_real.cuda()
                    z = self.sample_z(x_real.size(0))
                    y = self.get_y(x_real, y_gt)
                    samples = self.generator(z, y)
                    samples = [s.data.cpu() for s in samples]
                    fake_imgs.extend(samples)
                    real_batch = [img.data.cpu() for img in x_real]
                    real_imgs.extend(real_batch)
                    assert (len(real_imgs) == len(fake_imgs))
                    print(f"[{len(fake_imgs):d} / {N:d}]")
                    if len(fake_imgs) >= N:
                        fake_imgs = fake_imgs[:N]
                        real_imgs = real_imgs[:N]
                        return fake_imgs, real_imgs

    def compute_inception_score(self, imgs):
        score, score_std = inception_score(imgs,
                                           device=self.device,
                                           resize=True,
                                           splits=1)

        return score, score_std

    def compute_fid(self, fake_imgs, real_imgs, name):
        fid_value = fid.calculate_frechet_inception_distance(fake_imgs=fake_imgs, real_imgs=real_imgs, name=name, dims=2048, device="cuda:0")

        return fid_value

    def create_samples(self, z, y=None):
        self.generator.eval()
        batch_size = z.size(0)
        # Parse y
        if y is None:
            raise NotImplementedError()
        elif isinstance(y, int):
            y = torch.full((batch_size, ),
                           y,
                           device=self.device,
                           dtype=torch.int64)
        # Sample x
        with torch.no_grad():
            x = self.generator(z, y)
        return x

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--default_config_file", type=str)
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--it", type=int)
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--is", dest="cal_is", action="store_true")
    parser.add_argument("--fid", dest="cal_fid", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    device = torch.device(args.device)

    generator, clusterer, data_loader, zdist, name = prepare(default_config_path=args.default_config_file, config_path=args.config_file, model_dir=args.model_dir, it=args.it, device=device)

    evaluator = Evaluator(generator=generator, zdist=zdist, ydist=None, train_loader=data_loader, clusterer=clusterer, batch_size=BS, device=device)

    fake_imgs, real_imgs = evaluator.get_fake_real_samples(N=args.samples)

    fake_imgs = np.stack([img.numpy() for img in fake_imgs], axis=0)
    real_imgs = np.stack([img.numpy() for img in real_imgs], axis=0)

    if args.cal_is:
        is_mean, is_std = evaluator.compute_inception_score(imgs=fake_imgs)
        print(f"is: {is_mean:.2f}")

    if args.cal_fid:
        fid = evaluator.compute_fid(fake_imgs=fake_imgs, real_imgs=real_imgs, name=name)
        print(f"fid: {fid:.2f}")

if __name__ == "__main__":
    main()
