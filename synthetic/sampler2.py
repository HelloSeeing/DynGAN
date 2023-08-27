# -*- coding:utf-8 -*-

import torch
import numpy as np
from scipy.stats import entropy

class RingSampler(object):
    def __init__(self, n_gaussian=8, radius=1.0, sigma=1e-3, n_per_mode=50):
        self.n_gaussian = n_gaussian
        self.radius = radius
        self.sigma = sigma
        self.n_per_mode = n_per_mode

        self.prob = np.ones(self.n_gaussian, dtype=np.float32) / self.n_gaussian
        self.centers = None
        self.n_mode = None
        self.n_data = None
        self.data = None

        self.build()

    def build(self):
        self.centers = np.stack([np.stack([self.radius * np.cos(2 * np.pi * i / self.n_gaussian), self.radius * np.sin(2 * np.pi * i / self.n_gaussian)], 0) for i in range(self.n_gaussian)], 0)
        self.n_mode = self.n_gaussian

        self.n_data = self.n_mode * self.n_per_mode
        self.data = np.repeat(self.centers, self.n_per_mode, axis=0) + np.random.normal(size=(self.n_data, 2)) * self.sigma

        # self.labels = np.random.randint(0, self.n_class, size=(self.n_data,), dtype=np.int64)

    @property
    def x_range(self):
        return -3, 3

    @property
    def y_range(self):
        return -3, 3

    def calc_quality(self, x, level=0):
        """
        Calculate point quality.
        """

        n_samples = x.shape[0]
        if n_samples > 0:
            x = x.reshape((-1, 1, 2))
            centers = self.centers.reshape((1, -1, 2))
            dists = np.linalg.norm(x - centers, axis=2)
            quality = np.zeros((self.n_mode, 3))

            for l in range(3):
                for n in range(self.n_mode):
                    quality[n, l] = np.count_nonzero(dists[:, n] < (l + 1) * self.sigma)

            q_l1 = np.sum(quality[:, 0]) / n_samples
            q_l2 = np.sum(quality[:, 1]) / n_samples
            q_l3 = np.sum(quality[:, 2]) / n_samples
            mode_covered = np.count_nonzero(quality[:, 0])
            covered_modes = np.where(quality[:, level] > 0)[0]
        else:
            return 0, 0, 0, 0, 0

        assign = np.argmin(dists, axis=1)
        counts = np.array([np.count_nonzero(assign == i) / n_samples for i in range(self.n_mode)])
        targets = np.ones(self.n_mode) / self.n_mode
        kl = entropy(counts, targets)

        return mode_covered, q_l1, q_l2, q_l3, kl


class GridSampler(object):
    def __init__(self, n_row=4, n_col=None, edge=1.0, sigma=0.02, n_per_mode=50):
        self.n_row = n_row
        self.n_col = n_col if n_col else n_row
        self.edge = edge
        self.sigma = sigma
        self.n_per_mode = n_per_mode

        self.centers = None
        self.n_mode = None
        self.n_data = None

        self.build()

    def build(self):
        x = np.linspace(-4, 4, self.n_row) * self.edge
        y = np.linspace(-4, 4, self.n_col) * self.edge
        X, Y = np.meshgrid(x, y)

        self.centers = np.stack((X.flatten(), Y.flatten()), 1)
        self.n_mode = self.n_row * self.n_col

        self.n_data = self.n_mode * self.n_per_mode
        self.data = np.repeat(self.centers, self.n_per_mode, axis=0) + np.random.normal(size=(self.n_data, 2)) * self.sigma
        # self.idx = np.random.choice(self.n_mode, self.n_data, p=[0.76, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        # self.data = self.centers[self.idx] + np.random.normal(size=(self.n_data, 2)) * self.sigma

        # self.labels = np.random.randint(0, self.n_class, size=(self.n_data,), dtype=np.int64)

    @property
    def x_range(self):
        return (-6, 6)

    @property
    def y_range(self):
        return (-6, 6)

    def calc_quality(self, x, level=0):
        """
        Calculate point quality.
        """

        n_samples = x.shape[0]
        if n_samples > 0:
            x = x.reshape((-1, 1, 2))
            centers = self.centers.reshape((1, -1, 2))
            dists = np.linalg.norm(x - centers, axis=2)
            quality = np.zeros((self.n_mode, 3))

            for l in range(3):
                for n in range(self.n_mode):
                    quality[n, l] = np.count_nonzero(dists[:, n] < (l + 1) * self.sigma)

            q_l1 = np.sum(quality[:, 0]) / n_samples
            q_l2 = np.sum(quality[:, 1]) / n_samples
            q_l3 = np.sum(quality[:, 2]) / n_samples
            mode_covered = np.count_nonzero(quality[:, level])
            covered_modes = np.where(quality[:, level] > 0)[0]
        else:
            return 0, 0, 0, 0, 0

        assign = np.argmin(dists, axis=1)
        counts = np.array([np.count_nonzero(assign == i) / n_samples for i in range(self.n_mode)])
        targets = np.ones(self.n_mode) / self.n_mode
        kl = entropy(counts, targets)

        return mode_covered, q_l1, q_l2, q_l3, kl


class CircleSampler(object):
    def __init__(self, p_center=0.1, radius=1.0, sigma=0.03, n_samples=1000):
        self.p_center = p_center
        self.n_theta = int(100 - p_center * 100)
        self.thetas = np.linspace(0, 2 * np.pi, num=self.n_theta)
        self.radius = radius
        self.sigma = sigma

        self.centers = None
        self.n_mode = None
        self.n_data = n_samples

        self.build()
    
    def build(self):
        self.circle_xs = self.radius * np.cos(self.thetas)
        self.circle_ys = self.radius * np.sin(self.thetas)
        self.circle_centers = np.stack((self.circle_xs, self.circle_ys), axis=1)

        self.centers = np.concatenate([self.circle_centers, np.array([[0, 0]])], axis=0)
        self.n_mode = self.n_theta + 1

        eps = np.random.rand(self.n_data)
        n_center = np.count_nonzero(eps < self.p_center)
        n_circle = self.n_data - n_center
        circle_idx = np.random.randint(low=0, high=self.n_theta, size=(n_circle,))

        center_samples = np.random.normal(size=(n_center, 2)) * self.sigma
        eps = np.random.normal(size=(n_circle, 2)) * self.sigma
        circle_samples = eps + self.circle_centers[circle_idx]

        self.data = np.concatenate((center_samples, circle_samples), axis=0).astype(np.float32)

    @property
    def x_range(self):
        return (-2 * self.radius, 2 * self.radius)

    @property
    def y_range(self):
        return (-2 * self.radius, 2 * self.radius)

    def calc_quality(self, x, level=0):
        """
        Calculate point quality.
        """

        n_samples = x.shape[0]
        if n_samples > 0:
            x = x.reshape((-1, 1, 2))
            centers = self.centers.reshape((1, -1, 2))
            dists = np.linalg.norm(x - centers, axis=2)
            quality = np.zeros((self.n_mode, 3))

            for l in range(3):
                for n in range(self.n_mode):
                    quality[n, l] = np.count_nonzero(dists[:, n] < (l + 1) * self.sigma)

            q_l1 = np.sum(quality[:, 0]) / n_samples
            q_l2 = np.sum(quality[:, 1]) / n_samples
            q_l3 = np.sum(quality[:, 2]) / n_samples
            mode_covered = np.count_nonzero(quality[:, level])
            covered_modes = np.where(quality[:, level] > 0)[0]
        else:
            return 0, 0, 0, 0, 0

        assign = np.argmin(dists, axis=1)
        counts = np.array([np.count_nonzero(assign == i) for i in range(self.n_mode)])
        counts /= n_samples
        targets = np.ones(self.n_mode) / self.n_mode
        kl = entropy(counts, targets)

        return mode_covered, q_l1, q_l2, q_l3, kl


class XSampler(object):
    def __init__(self, data_type, use_cuda=False, n_class=None, **kwargs):
        self.data_type = data_type

        n_per_mode = kwargs.get("n_per_mode", 50)

        if data_type == "grid":
            n_row = kwargs.get("n_grid", 5)
            n_col = n_row
            sampler = GridSampler(n_row=n_row, n_col=n_col, edge=1.0, sigma=0.1, n_per_mode=n_per_mode)

        elif data_type == "ring":
            n_gaussian = kwargs.get("n_ring", 8)
            sampler = RingSampler(n_gaussian=n_gaussian, radius=2.0, sigma=0.05, n_per_mode=n_per_mode)

        elif data_type == "circle":
            p_center = kwargs.get("p_center", 0.03)
            sampler = CircleSampler(p_center=p_center, radius=1.0, sigma=0.03)

        else:
            raise ValueError("%s data not supported." % data_type)

        self.sampler = sampler
        self.n_data = self.sampler.n_data
        self.use_cuda = use_cuda

        if n_class is not None:
            self.labels = np.random.randint(0, n_class, size=(self.n_data,), dtype=np.int64)
        else:
            self.labels = np.zeros(shape=(self.n_data,), dtype=np.int64)

        self.n_class = np.max(self.labels) + 1

    def __len__(self):
        return self.n_data

    def special_data_iter(self, idx : list, batch_size : int=128, shuffle : bool=True, drop_last : bool=True, onehot : bool=True):

        if shuffle:
            np.random.shuffle(idx)

        n_data = len(idx)

        s = 0

        while s < n_data:

            e = s + batch_size

            if drop_last:
                if e > n_data:
                    return
            else:
                if e > n_data:
                    e = n_data
            
            ii = idx[s : e]
            data = torch.from_numpy(self.sampler.data[ii]).float()

            if onehot:
                label = torch.from_numpy(np.eye(self.n_class)[self.labels[ii]]).float()
            else:
                label = torch.from_numpy(np.array(self.labels[ii])).long()

            if self.use_cuda:
                data = data.cuda()
                label = label.cuda()

            yield data, label, ii

            s = e

        return

    def data_iter(self, batch_size=128, shuffle=True, drop_last=True, onehot=True):
        idx = np.arange(self.n_data)

        return self.special_data_iter(idx, batch_size, shuffle, drop_last=drop_last, onehot=onehot)

    @property
    def x_range(self):
        return self.sampler.x_range

    @property
    def y_range(self):
        return self.sampler.y_range

    def calc_quality(self, x, level):
        return self.sampler.calc_quality(x, level)

    def compare(self, mod1, mod2):
        a = b = c = 0

        for mod in mod1:
            if mod not in mod2:
                a += 1
            else:
                b += 1

        for mod in mod2:
            if mod not in mod1:
                c += 1

        return a, b, c

    def sample(self, n_samples=100):
        """
        sample.
        """
        n_samples = min(n_samples, self.n_data)
        idx = np.random.permutation(np.arange(self.n_data))[:n_samples]

        data = self.sampler.data[idx]
        label = self.labels[idx]

        return data, label, idx

    def export_prob(self):
        k = np.max(self.labels) + 1

        prob = np.zeros(k)

        for i in range(k):
            prob[i] = np.where(self.labels == i)[0].shape[0] / self.n_data

        return prob

    def sample_class(self, idx, num):
        """
        Sample specific class of data.
        """
        target_idx = np.where(self.labels == idx)[0]
        np.random.shuffle(target_idx)

        data = self.sampler.data[target_idx[:num]]
        data = torch.from_numpy(data).float()

        if self.use_cuda:
            data = data.cuda()

        return data

    def __getitem__(self, idx):
        if isinstance(idx, int) or isinstance(idx, slice) or isinstance(idx, np.ndarray):
            return self.sampler.data[idx]

        else:
            raise ValueError("invalid input: {}, should be either int, slice, or np.ndarray".format(idx))


def meshgrid(x_range, y_range, x_bins=30, y_bins=30):
    xmin, xmax = x_range
    ymin, ymax = y_range

    x = np.linspace(xmin, xmax, num=x_bins, dtype=np.float32)
    y = np.linspace(ymin, ymax, num=y_bins, dtype=np.float32)

    X, Y = np.meshgrid(x, y)

    return X, Y