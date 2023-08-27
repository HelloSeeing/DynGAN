#!-*- coding:utf-8 -*-

import torch
import numpy as np
import copy
import os
from sklearn.neighbors import KNeighborsClassifier

from clusterers import base_clusterer
from clusterers.dyngan_utils import feature_extractor_dict

class EMA(object):
    def __init__(self, x0, a=0.1) -> None:
        self.x = x0
        self.a = a

    def update(self, x):
        self.x = self.a * self.x + (1 - self.a) * x

    @property
    def min(self):
        return np.min(self.x)
    
    @property
    def max(self):
        return np.max(self.x)


def get_centers(kt, labels, features):
    _, f_dim = features.shape
    centers = np.zeros((kt, f_dim))
    p_valid = np.empty((kt,), dtype=np.bool)
    p_valid.fill(True)

    for p in np.arange(kt):
        idx = np.where(labels == p)[0]

        if idx.size > 0:
            centers[p, :] = np.mean(features[idx, :], axis=0)
        else:
            p_valid[p] = False

    return centers, p_valid
        
def get_discriminator_logits(discriminator, x, y):
    discriminator.eval()
    with torch.no_grad():
        return discriminator(x, y=y, get_features=False)
    
def get_batch_discriminator_logits(discriminator, cluster_x, cluster_labels, batch_size):
    '''return discriminator logits'''

    with torch.no_grad():
        outputs = []
        x = cluster_x
        y = cluster_labels
        for batch in range(x.size(0) // batch_size):
            x_batch = x[batch * batch_size:(batch + 1) * batch_size].cuda()
            y_batch = y[batch * batch_size:(batch + 1) * batch_size].cuda()
            outputs.append(get_discriminator_logits(discriminator, x_batch, y_batch).detach().cpu())
        if (x.size(0) % batch_size != 0):
            x_batch = x[x.size(0) // batch_size * batch_size:].cuda()
            y_batch = y[x.size(0) // batch_size * batch_size:].cuda()
            outputs.append(get_discriminator_logits(discriminator, x_batch, y_batch).detach().cpu())
        result = torch.cat(outputs, dim=0).numpy()
        return result

def update_partition_main(cur_partitions, features, logits, delta, max_partition, max_iteration=1000, batch_size = 100,):
    N, f_dim = features.shape
    cur_max = np.max(cur_partitions) + 1
    collapsed_idx = np.where(logits > -np.log(delta))[0]
    scores = 1 / (1 + np.exp(-logits))

    new_partitions = cur_partitions.copy()

    # if there is available conditions to extend
    if cur_max < max_partition:
            
        # add new partitions
        collapsed_partitions = cur_partitions[collapsed_idx]
        collapsed_partitions_unique = np.sort(np.unique(collapsed_partitions))
        for idx in collapsed_idx:
            partition = cur_partitions[idx]
            to_add = np.where(collapsed_partitions_unique == partition)[0]
            new_partitions[idx] = min(max_partition - 1, cur_max + to_add)

        # adaptively adjust the partitions with mode collapse scores
        i = 0
        new_max = np.max(new_partitions) + 1
        centers = np.zeros(shape=(new_max, f_dim))
        while i < max_iteration:
            break_flag = True

            centers, p_valid = get_centers(kt = new_max, labels = new_partitions, features = features)

            st = 0
            en = min(N, st + batch_size)

            while en < N:
            
                # calculate the distances
                d = np.sum(np.square(features[st : en, :].reshape(en - st, 1, f_dim) - centers[p_valid, :].reshape(1, -1, f_dim)), axis=2)

                # adaptive weight with mode collapse scores
                for w_idx in range(st, en):
                    w = np.ones(shape=(new_max,))
                    
                    # w[cur_partitions[w_idx]] = scores[w_idx]
                    partition = np.argmin(d[w_idx - st, :] * w[p_valid])
                    if partition != new_partitions[w_idx]:
                        new_partitions[w_idx] = np.arange(new_max)[p_valid][partition]
                        break_flag = False

                st = en
                en = min(N, st + batch_size)
        
            # if there is no update of partition, break the loop
            if break_flag:
                break

            i += 1
            print(f"DynGAN: Adjust for the {i:d}-th iteration")

    # otherwise
    else:
        new_max = np.max(new_partitions) + 1
        print(f"DynGAN: Number of partitions reaches the maximum {max_partition:d}")

    centers = np.zeros(shape=(new_max, f_dim))
    for p in range(new_max):
        idx = np.where(new_partitions == p)[0]
        if idx.size > 0:
            centers[p, :] = np.mean(features[idx, :])
        
    return new_partitions.copy(), centers.copy()

class Clusterer(base_clusterer.BaseClusterer):
    def __init__(self, initialization=True, matching=True, fe_type="discriminator", k0=1, delta=None, lamb=0.1, max_kmeans_iteration=1000, **kwargs):

        super().__init__(**kwargs)

        self.k0 = k0
        self.kt = self.k0
        self.delta = delta
        self.max_kmeans_iteration = max_kmeans_iteration
        self.cluster_labels = dict()
        self.knn_classifier = None

        xlabels = np.random.choice(np.arange(k0), size=(self.x.size(0),))
        self.x_labels = torch.from_numpy(xlabels).long()

        logits = np.zeros(self.x.size(0),)
        self.logits = EMA(x0=logits, a=lamb)

        if fe_type == "discriminator":
            kwargs['fe_kwargs']['dis'] = dict()
            kwargs['fe_kwargs']['dis']['D'] = self.discriminator

        self.feature_extractor = feature_extractor_dict[fe_type](fe_type=fe_type, device=self.device, **kwargs['fe_kwargs'])

    @property
    def device(self):
        return next(self.discriminator.parameters()).device

    def update_mode_collapse_status(self, discriminator, N):
        
        logits = get_batch_discriminator_logits(discriminator=discriminator, cluster_x=self.x, cluster_labels=self.x_labels, batch_size=self.batch_size)
        self.logits.update(logits)

        idx = np.where(self.logits.x >= -np.log(self.delta))[0]
        status = idx.size / self.x.size(0)

        boundaries = [- np.log(N / n - 1) for n in range(1, N)]
        boundaries.insert(0, -np.inf)
        boundaries.append(np.inf)
        nums = np.zeros(N,)

        for n in range(N):
            cond0 = np.where(self.logits.x >= boundaries[n], True, False)
            cond1 = np.where(self.logits.x < boundaries[n + 1], True, False)
            cond = np.bitwise_and(cond0, cond1)
            nums[n] = np.sum(cond) / self.x.size(0)

        return status, nums

    def update_centers(self, ):

        features = self.get_cluster_batch_features()
        labels = self.x_labels
        kt = self.kt

        f_dim = features.shape[1]
        unique_labels = np.unique(labels)
        centers = np.zeros(shape=(kt, f_dim))

        for y in unique_labels:
            idx = np.where(labels == y, True, False)
            centers[y] = np.mean(features[idx], axis=0)

        return centers

    def fit_means(self, iteration=None):
        if iteration is None:
            iteration = self.max_kmeans_iteration

        logits = get_batch_discriminator_logits(discriminator=self.discriminator, cluster_x=self.x, cluster_labels=self.x_labels, batch_size=self.batch_size)
        features = self.get_cluster_batch_features()
        
        new_labels, new_centers = update_partition_main(cur_partitions=self.x_labels.cpu().numpy(), features=features, logits=logits, delta=self.delta, max_iteration=iteration, max_partition=self.k, batch_size=100)
        self.kt = new_centers.shape[0]
        new_centers = new_centers.astype(np.float32)

        nums = np.zeros(self.kt)
        for k in range(self.kt):
            nums[k] = np.where(new_labels == k, True, False).sum()

        print(f"partition: {nums}")

        # reset logits for samples assigned in new partitions
        mask = np.where(self.x_labels.cpu().numpy() == new_labels, True, False)
        mask = np.bitwise_not(mask)
        if np.any(mask):
            self.logits.x[mask] = 0.0

        if self.knn_classifier is None:
            self.knn_classifier = KNeighborsClassifier(n_neighbors=5,)

        print("DynGAN: knn classifier fitting")
        self.knn_classifier.fit(X=features, y=new_labels)
        print("DynGAN: knn classifier fitting done")

        # recompute the fixed labels
        self.x_labels = torch.from_numpy(new_labels)

    def recluster(self, discriminator, **kwargs):
        self.discriminator = copy.deepcopy(discriminator)
        self.fit_means()

    def get_labels(self, x, y):
        np_features = self.get_features(x).detach().cpu().numpy()
        np_prediction = self.knn_classifier.predict(X=np_features)
        return torch.from_numpy(np_prediction).long().cuda()

    def get_features(self, x):
        
        return self.feature_extractor(x.to(self.device))
    
    def state_dict(self):
        return {
            "k": self.k, 
            "k0": self.k0, 
            "kt": self.kt, 
            "delta": self.delta, 
            "kmeans_max_iter": self.max_kmeans_iteration, 
            "knn_classifier": self.knn_classifier, 
            "x_labels": self.x_labels, 
            "logits": self.logits, 
            "D": self.discriminator.state_dict(), 
            "fe": self.feature_extractor.state_dict(), 
            "batch_size": self.batch_size, 
        }
    
    def load_state_dict(self, state_dict):

        self.k = state_dict["k"]
        self.k0 = state_dict["k0"]
        self.kt = state_dict["kt"]
        self.delta = state_dict["delta"]
        self.max_kmeans_iteration = state_dict["kmeans_max_iter"]
        self.knn_classifier = state_dict["knn_classifier"]
        self.x_labels = state_dict["x_labels"]
        self.logits = state_dict["logits"]
        self.discriminator.load_state_dict(state_dict["D"])
        self.feature_extractor.load_state_dict(state_dict["fe"])
        self.cluster_counts = [0] * self.k

    def save(self, path):

        torch.save(self.state_dict(), path)

    def load(self, path):

        if os.path.exists(path):

            self.load_state_dict(torch.load(path))
            return True
        
        return False

