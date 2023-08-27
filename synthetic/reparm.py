# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import numpy as np

class Reparam(object):
    def __init__(self, z_dim=32, K=1, alpha=20, update_mu = False, update_logvar = False, update_pi = False):
        self.K = K
        self.z_dim = z_dim
        self.alpha = alpha

        mu = torch.randn(K, z_dim) if K > 1 else torch.zeros(K, z_dim)
        self.mu = nn.Parameter(mu, requires_grad = update_mu)
        logvar = 2 * torch.log(torch.std(self.mu, dim=0) / self.alpha * torch.ones_like(self.mu)) if K > 1 else torch.zeros_like(mu)
        # logvar = 2 * torch.log(torch.ones_like(self.mu) / self.alpha)
        self.logvar = nn.Parameter(logvar, requires_grad = update_logvar)
        pi = torch.zeros((K,))
        self.pi = nn.Parameter(pi, requires_grad = update_pi)

        print(self.mu)
        print(self.logvar)
        print(self.pi)

        self._parameters = list()

        self.category_sampler = torch.distributions.one_hot_categorical.OneHotCategorical(self.weights)

    def parameters(self):
        return self._parameters

    def sample(self, batch_size=128):
        categories = self.category_sampler.sample((batch_size,))
        mu = torch.matmul(categories, self.mu)
        logvar = torch.matmul(categories, self.logvar)
        std = logvar.mul(0.5).exp()
        z = torch.randn_like(mu).mul(std).add(mu)

        return z

    def equal_sample(self, n_component=32):
        categories = np.arange(self.K)
        categories = np.tile(categories, n_component)
        np.random.shuffle(categories)
        std = torch.randn(n_component * self.K, self.z_dim)
        mu = self.mu[categories]
        eps = self.logvar[categories].mul(0.5).exp()
        z = eps.mul(std).add(mu)
        # print(mu.shape, eps.shape, z.shape)

        return z, categories

    @property
    def weights(self):
        return nn.functional.softmax(self.pi, dim=0)

    def jacobian(self):
        J = -torch.matmul(self.weights.view(-1, 1), self.weights.view(1, -1))
        J += torch.eye(self.K).mul(self.weights)

        return J

    def update(self, pi_update=None, mu_update=None, logvar_update=None):
        if pi_update is not None:
            self.pi.data += pi_update
            self.category_sampler = torch.distributions.one_hot_categorical.OneHotCategorical(self.weights)

        if mu_update is not None:
            self.mu.data += mu_update

        if logvar_update is not None:
            self.logvar.data += logvar_update

class InfoGAN_NoiseSampler(object):
    """
    Sampler for Information GAN.
    """
    def __init__(self, z_size, dis_c_dim):
        super(InfoGAN_NoiseSampler, self).__init__()

        self.z_size = z_size
        self.dis_c_dim = dis_c_dim
        self.pi = torch.zeros(self.dis_c_dim)

        self.z_sampler = torch.distributions.Normal(0, 1)
        self.dis_c_sampler = torch.distributions.Categorical(self.weights)

    @property
    def weights(self):
        return nn.functional.softmax(self.pi, dim=0)

    def sample(self, n_samples=64):
        z = self.z_sampler.sample((n_samples, self.z_size))
        c = self.dis_c_sampler.sample((n_samples,))
        c_onehot = nn.functional.one_hot(c, num_classes=self.dis_c_dim)

        zc = torch.cat((z, c_onehot.float()), dim=1)

        return zc, c

    def demo_sample(self, n_samples_per_class=8):
        c_basic = torch.arange(self.dis_c_dim)
        c = c_basic.repeat(n_samples_per_class).view(-1, self.dis_c_dim).transpose(1, 0).contiguous().view(-1)
        c_onehot = torch.eye(self.dis_c_dim)[c]
        z = self.z_sampler.sample((n_samples_per_class * self.dis_c_dim, self.z_size))

        zc = torch.cat((z, c_onehot.float()), dim=1)

        return zc, c

    def sample_dis_c(self, n_samples, i):
        if i >= self.dis_c_dim or i < 0:
            raise ValueError("param `c` should be in [0, %d), however input is: %d" % (self.dis_c_dim, i))

        c = torch.LongTensor(n_samples).fill_(i)
        c_onehot = torch.eye(self.dis_c_dim)[c]
        z = self.z_sampler.sample((n_samples, self.z_size))

        zc = torch.cat((z, c_onehot.float()), dim=1)

        return zc

    def update_pi(self, pi_update=None):
        if pi_update is not None:
            self.pi.data += pi_update
            self.dis_c_sampler = torch.distributions.Categorical(self.weights)

class InfoGAN_NoiseSampler2(object):
    """
    Sampler for Information GAN.
    """
    def __init__(self, z_size, dis_c_dim, init_dis_c_dim=2):
        super(InfoGAN_NoiseSampler2, self).__init__()

        self.z_size = z_size
        self.dis_c_dim = dis_c_dim
        self.cur_dis_c_dim = init_dis_c_dim

        self.latent_dim = self.z_size + self.dis_c_dim

        assert self.cur_dis_c_dim <= self.dis_c_dim, "cur_dis_c_dim({:d}) <= dis_c_dim({:d}) is required.".format(self.cur_dis_c_dim, self.dis_c_dim)

        self.pi = torch.zeros(self.cur_dis_c_dim)

        self.z_sampler = torch.distributions.Normal(0, 1)
        self.__reset_c_sampler()

    @property
    def weights(self):
        return nn.functional.softmax(self.pi, dim=0)

    def __reset_c_sampler(self):
        self.dis_c_sampler = torch.distributions.Categorical(self.weights)

    def sample(self, n_samples=64):
        z = self.z_sampler.sample((n_samples, self.z_size))
        c = self.dis_c_sampler.sample((n_samples,))
        c_onehot = nn.functional.one_hot(c, num_classes=self.dis_c_dim)

        zc = torch.cat((z, c_onehot.float()), dim=1)

        return zc, c
        # return z, c

    def demo_sample(self, n_samples_per_class=8):
        c_basic = torch.arange(self.cur_dis_c_dim)
        c = c_basic.repeat(n_samples_per_class).view(-1, self.cur_dis_c_dim).transpose(1, 0).contiguous().view(-1)
        c_onehot = torch.eye(self.dis_c_dim)[c]
        z = self.z_sampler.sample((n_samples_per_class * self.cur_dis_c_dim, self.z_size))

        zc = torch.cat((z, c_onehot.float()), dim=1)

        return zc, c

    def sample_dis_c(self, n_samples, i):
        if i >= self.cur_dis_c_dim or i < 0:
            raise ValueError("param `c` should be in [0, %d), however input is: %d" % (self.dis_c_dim, i))

        c = torch.LongTensor(n_samples).fill_(i)
        c_onehot = torch.eye(self.dis_c_dim)[c]
        z = self.z_sampler.sample((n_samples, self.z_size))

        zc = torch.cat((z, c_onehot.float()), dim=1)

        return zc

    def sample_multi_c(self, cs, n_samples):
        if max(cs) >= self.dis_c_dim or min(cs) < 0:
            raise ValueError("param `c` should be in [0, {:d}), however input is: {}".format(self.dis_c_dim, cs))

        c = np.random.choice(cs, n_samples)
        c = torch.from_numpy(c).long()
        c_onehot = torch.eye(self.dis_c_dim)[c]
        z = self.z_sampler.sample((n_samples, self.z_size))

        zc = torch.cat((z, c_onehot.float()), dim=1)

        return zc, c

        # return z, c

    def update_pi(self, pi_update=None):
        if pi_update is not None:
            self.pi.data += pi_update
            self.__reset_c_sampler()

    
    def increase_pi(self, idx, target_p=0.1, increment=1e-3, eps=1e-8):
        if self.weights[idx] < target_p:

            p = self.weights[idx]
            pp = p + increment
            ns = torch.log(1 - p + eps) - torch.log(p + eps) + torch.log(pp + eps) - torch.log(1 - pp + eps)
            s = ns / self.cur_dis_c_dim

            for i in range(self.cur_dis_c_dim):
                if i == idx:
                    self.pi.data[i] += ns.squeeze()
                else:
                    self.pi.data[i] -= s.squeeze()

            self.__reset_c_sampler()

            return True

        else:
            return False
    

    def update_pis(self, ids, target_p, delta_ps, increment=1e-3, eps=1e-8):
        # if self.weights[ids[0]] < target_p:

        #     a = torch.sum(torch.exp(self.pi))

        #     for idx, delta in zip(ids, delta_ps):
        #         p = self.pi[idx]
        #         dk = torch.log(torch.exp(p) + a.mul(increment).mul(delta) + eps) - p
        #         self.pi[idx] += dk

        #     self.__reset_c_sampler()

        #     return True

        # else:
        #     return False

        a = torch.sum(torch.exp(self.pi))

        for idx, delta in zip(ids, delta_ps):
            p = self.pi[idx]
            dk = torch.log(torch.exp(p) + a.mul(increment).mul(delta) + eps) - p
            self.pi[idx] += dk

        self.__reset_c_sampler()


    def split_pi(self, idx):
        if self.cur_dis_c_dim < self.dis_c_dim:
            new_pi = torch.zeros(self.cur_dis_c_dim + 1)
            new_pi[:-1] = self.pi.data
            new_pi[idx] = new_pi[idx] - torch.log(torch.FloatTensor([2]))
            new_pi[-1] = new_pi[idx]

            self.cur_dis_c_dim += 1
            self.pi.data = new_pi
            self.__reset_c_sampler()

    def add_pi(self, k=1):
        k = min(self.dis_c_dim - self.cur_dis_c_dim, k)
        if k > 0:
            # future_dis_c_dim = self.cur_dis_c_dim + k
            # new_pi = torch.zeros(future_dis_c_dim)
            # t = (torch.log(torch.sum(torch.exp(self.pi))) - np.log(self.cur_dis_c_dim)) / (self.cur_dis_c_dim / k + 1)
            # new_pi[:self.cur_dis_c_dim] = self.pi.sub(t)
            # new_pi[self.cur_dis_c_dim:] = self.cur_dis_c_dim * t / k

            new_pi = torch.zeros(self.cur_dis_c_dim + k)
            for i in range(self.cur_dis_c_dim + k):
                if i < self.cur_dis_c_dim:
                    new_pi[i] = self.pi.data[i]
                else:
                    new_pi[i] = -1000
            self.pi.data = new_pi
            self.cur_dis_c_dim += k
            self.__reset_c_sampler()

            return np.arange(self.cur_dis_c_dim - k, self.cur_dis_c_dim)

    def canadd(self) -> bool:
        return self.cur_dis_c_dim < self.dis_c_dim


    def reset_pi(self):
        self.pi = torch.zeros(self.cur_dis_c_dim)
        self.__reset_c_sampler()


    def assign(self, prob, eps=1e-7):

        for i in range(self.cur_dis_c_dim):
            self.pi.data[i] = torch.log(torch.FloatTensor([prob[i]]) + eps)

        self.__reset_c_sampler()