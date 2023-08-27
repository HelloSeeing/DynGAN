# -*- coding:utf-8 -*-

import os
import time
import torch
import torch.nn as nn
import numpy as np
import argparse
import pickle
import shutil
from torch import optim
from torch.autograd import Variable
from net import Generator, CDiscriminator, weights_init
from visualization import visualize_pts, visualize_gradients, visualize_contour, visualize_grad, visualize_contour2, visualize_weights, visualize_contour3, visualize_pts2, visualize_grad2, visualize_entropy, visualize_pts3, visualize_statistics
from sampler2 import XSampler, meshgrid
from reparm import InfoGAN_NoiseSampler, InfoGAN_NoiseSampler2
from scipy.sparse import csr_matrix, coo_matrix

use_cuda = torch.cuda.is_available()

def setup(x):
    if use_cuda:
        x = x.cuda()

    return x


def cal_entropy(logits, eps=1e-7):
    """
    Args:
        logits: torch.tensor, shape in `[batch, n_class]`, before softmax.
        eps: float.

    Return:
        entropy: torch.tensor, shape in `[batch]`.
    """
    probs = torch.softmax(logits, dim=1)
    entropy = -probs.mul(torch.log(probs + eps)).sum(1)

    return entropy

D_grad = None

def hook(grad):
    global D_grad
    D_grad = grad

def evaluate_D_on_meshgrid(D, x_range, y_range, generations, n_class, idx, x_bins=30, y_bins=30, title="D values", savename="evaluate_D.png"):
    X, Y = meshgrid(x_range, y_range, x_bins=x_bins, y_bins=y_bins)
    coords = np.stack((X.flatten(), Y.flatten()), axis=1)
    coords = setup(torch.from_numpy(coords))
    y_onehot = setup(torch.eye(n_class)[np.ones(coords.shape[0]) * idx])

    D.zero_grad()

    coords = Variable(coords, requires_grad = True)
    coords.register_hook(hook)
    D_values, _ = D(coords, y_onehot)
    D_values = D_values.squeeze()
    y_real = setup(torch.ones_like(D_values))
    loss = nn.functional.binary_cross_entropy(D_values, y_real)
    loss.backward()

    Dvals = D_values.detach().cpu().numpy().reshape((x_bins, y_bins))
    Dgrads = D_grad.detach().cpu().numpy().reshape((x_bins, y_bins, 2))
    Dgrads_norm = np.linalg.norm(Dgrads, axis=2)
    Dgrads = -Dgrads / np.max(Dgrads_norm) * 0.3
    visualize_contour3(X, Y, Dvals, Dgrads, generations, x_range, y_range, grad_step = 3, savename=savename, title=title)


def evaluate_D_on_meshgrid2(Dbase, D, x_range, y_range, generations, x_bins=30, y_bins=30, title="D values", savename="evaluate_D.png"):
    X, Y = meshgrid(x_range, y_range, x_bins=x_bins, y_bins=y_bins)
    coords = np.stack((X.flatten(), Y.flatten()), axis=1)
    coords = setup(torch.from_numpy(coords))

    Dbase.zero_grad()
    D.zero_grad()

    coords = Variable(coords, requires_grad = True)
    coords.register_hook(hook)
    output = Dbase(coords)
    D_values = D(output)
    y_real = setup(torch.ones_like(D_values))
    loss = nn.functional.binary_cross_entropy(D_values, y_real)
    loss.backward()

    Dvals = D_values.detach().cpu().numpy().reshape((x_bins, y_bins))
    Dgrads = D_grad.detach().cpu().numpy().reshape((x_bins, y_bins, 2))
    Dgrads_norm = np.linalg.norm(Dgrads, axis=2)
    Dgrads = -Dgrads / np.max(Dgrads_norm) * 0.3
    visualize_contour3(X, Y, Dvals, Dgrads, generations, x_range, y_range, grad_step = 3, savename=savename, title=title)


def adjust_idx(labels, n_sp, adj):
    """
    Args:
        labels: np.ndarray, dtype=int64
        adj: scipy.sparse.coo_matrix
    """

    k = np.max(labels) + 1

    row, col = adj.row, adj.col
    new_labels = labels

    a = list()

    for r, c in zip(row, col):
        if labels[r] >= k - n_sp and labels[c] < k - n_sp:
            new_labels[c] = labels[r]
        if labels[c] >= k - n_sp and labels[r] < k - n_sp:
            new_labels[r] = labels[c]

    return new_labels


def adjust_idx_kmeans(data, labels, scores=None, eps=1e-4):
    """
    Args:
        data: np.ndarray.
        labels: np.ndarray, dtype=int64
    """

    k = np.max(labels) + 1
    N, F = data.shape

    centers = np.zeros((k, F))
    eyek = np.eye(k)

    flag = True
    new_labels = labels.copy()

    p = np.ones(shape=(N, k))

    if scores is not None:
        for i in range(data.shape[0]):
            y = int(labels[i])
            p[i, y] = scores[i]

    batch_size = 128

    count = 0

    nt = 0

    # kmeans
    while True:

        t0 = time.time()

        # update centroids
        s = 0
        _centers = np.zeros_like(centers)

        for i in range(k):
            idx = np.where(new_labels == i)[0]
            if idx.shape[0] > 0:
                _centers[i] = np.mean(data[idx], axis=0)
            else:
                print("count={:d}, i={:d}".format(count, i))
                print(idx)
                # input("what")

        if np.linalg.norm(_centers - centers) < eps:
            break

        centers = _centers.copy()

        # update labels
        s = 0
        while s < data.shape[0]:
            e = min(s + batch_size, data.shape[0])

            dists = np.linalg.norm(np.expand_dims(data[s : e, :], axis=1) - np.expand_dims(centers, axis=0), axis=2)

            dists = np.multiply(dists, p[s : e])

            new_labels[s : e] = np.argmin(dists, axis=1)

            s = e

        count += 1
        nt += (time.time() - t0)

        print("\r{:d}: {:.2f}".format(count, nt / count), end="")

    print()

    return new_labels


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_type", "-dt", 
                        type=str, 
                        default="grid", 
                        help="type of dataset.")

    parser.add_argument("--n_grid", "-ng", 
                        type = int, 
                        default = 5, 
                        help = "number of grid when data_type == `grid`.")

    parser.add_argument("--n_ring", "-nr", 
                        type = int, 
                        default = 8, 
                        help = "number of gaussian when data_type == `ring`.")

    parser.add_argument("--p_center", "-pc", 
                        type = float, 
                        default = 0.03, 
                        help = "probability of center when data_type == `circle`.")

    parser.add_argument("--zsize", type=int, default=50)

    parser.add_argument("-cdc", "--c_dim_cap", type=int, default=30)

    parser.add_argument("-cd", "--c_dim", type=int, default=1)

    parser.add_argument("-dth", "--Dthresh", type=float, default=0.2, help="threshold of discriminator value.")

    parser.add_argument("-mth", "--Mthresh", type=float, default=0.01, help="threshold of mode collapse.")

    parser.add_argument("-lamb", "--lambda", dest="lamb", type=float, default=0.1, help="lambda")

    parser.add_argument("--n_epochs", type=int, default=30000)

    parser.add_argument("--n_per_mode", type=int, default=50)

    args = parser.parse_args()

    return args


def main():
    batch_size = 128
    detect_step = 500
    partition_step = 3000
    visualize = True

    args = parse_args()

    data_type = args.data_type
    z_dim = args.zsize
    c_dim_cap = args.c_dim_cap
    c_dim = args.c_dim
    n_epochs = args.n_epochs
    discr_thre = 1 / (1 + args.Dthresh)

    data_dict = {"n_grid": args.n_grid, "n_ring": args.n_ring, "n_per_mode": args.n_per_mode, "n_class": args.c_dim_cap}

    x_sampler = XSampler(data_type, use_cuda=use_cuda, **data_dict)
    batch_per_epoch = x_sampler.sampler.n_data // batch_size

    x_range = x_sampler.x_range
    y_range = x_sampler.y_range

    zc_dim = z_dim + c_dim_cap

    n_hidden = 128

    G = Generator(zc_dim=zc_dim, n_hidden=n_hidden)
    G = setup(G)
    G.apply(weights_init)

    print(G)

    D = CDiscriminator(x_depth=2, c_depth=c_dim_cap, n_hidden=n_hidden)
    D = setup(D)
    D.apply(weights_init)

    print(D)

    y_real = setup(torch.ones((batch_size,)))
    y_fake = setup(torch.zeros((batch_size,)))

    prior_sampler = InfoGAN_NoiseSampler2(z_dim, args.c_dim_cap, args.c_dim)

    D_optimizer = optim.Adam(D.parameters(), lr=1e-3, betas=(0.5, 0.999))
    G_optimizer = optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.999))

    n_sample = 10000
    
    sample, sample_idx = prior_sampler.sample(n_sample)
    sample = setup(sample)
    sample_idx = setup(sample_idx)

    criterion = nn.BCELoss()

    log_dir = f"dyngan/{data_type}/c_cap={args.c_dim_cap:d}-c={args.c_dim:d}-delta={args.Dthresh:.2e}-tau={args.Mthresh:.2e}-lamb={args.lamb:.2e}"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    else:
        shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)

    x_sampler.labels = np.zeros(x_sampler.n_data, dtype=np.int64)

    print(prior_sampler.weights)

    ### visualize generation

    with torch.no_grad():
        generations = G(sample)
        generations = generations.detach().cpu().numpy()
        visualize_pts(generations, savename=os.path.join(log_dir, "generation_init.png"), xrange=x_range, yrange=y_range, title="Step 0")
        visualize_weights(prior_sampler.weights.detach().cpu().numpy(), title="Step 0", savename=os.path.join(log_dir, "weights_init.png"))

    data = x_sampler.sampler.data
    visualize_pts(data, savename=os.path.join(log_dir, "x.pdf"), xrange=x_range, yrange=y_range, title="Data")

    entropy_of_real = setup(torch.zeros(batch_size))

    epoch = 0
    batch_step = 0
    show_step = 500

    Gtrain_loss = 0
    Dtrain_loss = 0

    # rank_recorder = list()

    stats = {"steps" : [], "n_discrete" : [], "n_modes" : [], "q1" : [], "q2" : [], "q3" : [], "kl": []}

    flag = True
    to_train = False

    special_idx = list()
    _special_idx = list()
    special_data_iter = None
    special_batch_step = 0
    special_p_idx = None
    target_ps = None

    gan_train_time = 0
    detect_time = 0
    partition_time = 0

    data_scores = np.zeros(x_sampler.n_data)
    
    while batch_step < args.n_epochs:

        D.train()
        G.train()

        data_iter = x_sampler.data_iter(batch_size, shuffle=True)

        for x, y_onehot, ii in data_iter:

            ################################################

            # train D

            ################################################

            gan_train_start = time.time()

            D_optimizer.zero_grad()

            z, idx = prior_sampler.sample(batch_size)
            z = Variable(setup(z))
            idx_onehot = torch.eye(c_dim_cap)[idx]
            idx_onehot = Variable(setup(idx_onehot))
            x_fake = G(z).detach()

            D_real, _ = D(x, y_onehot)
            loss_real = criterion(D_real.squeeze(), y_real)

            D_fake, _ = D(x_fake, idx_onehot)
            loss_fake = criterion(D_fake.squeeze(), y_fake)

            loss = loss_real + loss_fake
            loss.backward()
            D_optimizer.step()

            Dtrain_loss = 0.9 * Dtrain_loss + 0.1 * loss.item()

            ################################################

            # train G

            ################################################

            G_optimizer.zero_grad()

            z, idx = prior_sampler.sample(batch_size)
            z = setup(z)
            idx_onehot = torch.eye(c_dim_cap)[idx]
            idx_onehot = Variable(setup(idx_onehot))
            x_fake, feat = G(z, output_f = True)

            D_fake, _ = D(x_fake, idx_onehot)
            G_loss = criterion(D_fake.squeeze(), y_real)

            G_loss.backward()
            G_optimizer.step()

            Gtrain_loss = 0.9 * Gtrain_loss + 0.1 * G_loss.item()

            gan_train_end = time.time()
            gan_train_time += gan_train_end - gan_train_start

            if to_train:
                special_batch_step += 1

            ##############################################

            batch_step += 1


            ##############################################


            if batch_step % show_step == 0:

                print("[Step %d/%d]: D_loss: %.4f, G_loss: %.4f" % (batch_step, n_epochs, Dtrain_loss, Gtrain_loss))

                ### visualize generation
        
                sample, idx = prior_sampler.sample(n_sample)
                sample = setup(sample)
                idx_onehot = torch.eye(c_dim_cap)[idx]
                idx_onehot = Variable(setup(idx_onehot))
                sample_y_real = setup(torch.ones(2500))

                G.eval()

                with torch.no_grad():
                    generations = G(sample)

                x_idx = x_sampler.labels[ii]
                
                w = prior_sampler.weights.detach().cpu().numpy()

                if visualize:

                    ## png format
                    visualize_pts3(data, generations.detach().cpu().numpy(), idx, prior_sampler.cur_dis_c_dim, savename=os.path.join(log_dir, "generation_{:d}.png".format(batch_step)), xrange=x_range, yrange=y_range, weights=None, title="Step %d" % (batch_step))

                print(prior_sampler.weights)

                ################################################

                # record statistics

                ################################################


                mode_covered, q_l1, q_l2, q_l3, kl = x_sampler.calc_quality(generations.detach().cpu().numpy(), level=0)

                stats["steps"].append(batch_step + 1)
                stats["n_discrete"].append(prior_sampler.cur_dis_c_dim)
                stats["n_modes"].append(mode_covered)
                stats["q1"].append(q_l1)
                stats["q2"].append(q_l2)
                stats["q3"].append(q_l3)
                stats["kl"].append(kl)
                print("Mode Covered: {:d}\nLevel1: {:.4f}\nLevel2: {:.4f}\nLevel3: {:.4f}\nKL: {:.4f}".format(mode_covered, q_l1, q_l2, q_l3, kl))

                print("Epoch {:d}: len(s_idx)={:d}".format(epoch, len(_special_idx)))

            ################################################

            # check whether to add new discrete component

            ################################################

            if batch_step % detect_step == 0:

                detect_start = time.time()

                data_iter = x_sampler.data_iter(batch_size, shuffle=False, drop_last=False)
                _special_idx = list()
                feats = list()

                with torch.no_grad():

                    for x, y, xi in data_iter:

                        D_real, feat = D(x, y)
                        D_real = D_real.squeeze()

                        for i, j in enumerate(xi):
                            data_scores[j] = (1 - args.lamb) * data_scores[j] + args.lamb * D_real[i].item()

                            if data_scores[j] > discr_thre:
                                _special_idx.append(j)

                        feats.append(feat.detach().cpu().numpy())

                feats = np.concatenate(feats, axis=0)

                detect_end = time.time()
                detect_time += detect_end - detect_start

                print(f"detection cost: {detect_end - detect_start:.2f}")

            #######################################################

            # when it is necessary to add new discrete component

            #######################################################

            if len(_special_idx) / len(x_sampler) >= args.Mthresh and prior_sampler.canadd() and batch_step % partition_step == 0:

                ## visualize

                if visualize:
                    special_data = x_sampler.sampler.data[_special_idx]

                    visualize_pts(special_data, xrange=x_range, yrange=y_range, savename=os.path.join(log_dir, "special_%d.pdf" % (batch_step)), title="Step: %d" % batch_step)

                partition_start = time.time()

                special_idx = _special_idx[:]

                # add new discrete component

                show_step = 100

                special_ps = dict()
                old_to_new = dict()
                new_count = 0

                for idx in special_idx:
                    label = x_sampler.labels[idx]

                    # if label not recorded in old_pi, add to old_pi and special_ps
                    if str(label) not in old_to_new.keys():

                        # if can add:
                        if prior_sampler.canadd():
                            new_pi = prior_sampler.add_pi(k=1).squeeze()
                            new_count += 1
                        else:
                            new_pi = prior_sampler.cur_dis_c_dim - 1

                        old_to_new[str(label)] = new_pi

                        special_ps[str(label)] = 0
                        special_ps[str(new_pi)] = 0

                    new_pi = old_to_new[str(label)]

                    special_ps[str(label)] -= 1
                    special_ps[str(new_pi)] += 1

                    x_sampler.labels[idx] = new_pi

                x_samples, y_samples, idx_samples = x_sampler.sample(n_samples=500)

                if visualize:
                    # pdf format
                    visualize_pts2(x_samples, y_samples, prior_sampler.cur_dis_c_dim, savename=os.path.join(log_dir, "sp_x_{:d}.pdf".format(batch_step)), xrange=x_range, yrange=y_range, weights=None, title="Step %d" % (batch_step))

                    ## png format
                    visualize_pts2(x_samples, y_samples, prior_sampler.cur_dis_c_dim, savename=os.path.join(log_dir, "sp_x_{:d}.png".format(batch_step)), xrange=x_range, yrange=y_range, weights=None, title="Step %d" % (batch_step))

                # adjust labels

                old_labels = x_sampler.labels

                new_labels = adjust_idx_kmeans(feats, old_labels, scores=data_scores)
                x_sampler.labels = new_labels

                if visualize:
                    y_samples = new_labels[idx_samples]

                    # png format
                    visualize_pts2(x_samples, y_samples, prior_sampler.cur_dis_c_dim, savename=os.path.join(log_dir, "sp_kmeans+weight_x_{:d}.png".format(batch_step)), xrange=x_range, yrange=y_range, weights=None, title="Step {:d}".format(batch_step))

                prob = x_sampler.export_prob()
                prior_sampler.assign(prob)

                partition_end = time.time()
                partition_time += partition_end - partition_start

                print(f"partitioning cost: {partition_end - partition_start:.2f}")
                
                print(prior_sampler.weights)

                if visualize:
                    if not os.path.exists(os.path.join(log_dir, "evaluate_D_{:d}".format(batch_step))):
                        os.makedirs(os.path.join(log_dir, "evaluate_D_{:d}".format(batch_step)))

                    for p in range(prior_sampler.dis_c_dim):
                        with torch.no_grad():
                            new_sample, _idx = prior_sampler.sample_multi_c([p], 500)
                            new_sample = setup(new_sample)
                            new_gens = G(new_sample)
                            # new_gens = G(new_sample, setup(_idx))

                        evaluate_D_on_meshgrid(D, x_range, y_range, new_gens.detach().cpu().numpy(), c_dim_cap, p, savename=os.path.join(log_dir, "evaluate_D_{:d}/{:d}.png".format(batch_step, p)))

        epoch += 1

    visualize_statistics(savename=os.path.join(log_dir, "stats.png"), **stats)

    with open(os.path.join(log_dir, "stats.pkl"), "wb") as f:
        pickle.dump(stats, f)

    ## save model

    if not os.path.exists("./models/info-cgan4_%s" % data_type):
        os.makedirs("./models/info-cgan4_%s" % data_type)

    torch.save(D.state_dict(), "./models/info-cgan4_%s/D.pkl" % data_type)
    torch.save(G.state_dict(), "./models/info-cgan4_%s/G.pkl" % data_type)

    with open("./models/info-cgan4_%s/prior_sampler.pkl" % data_type, "wb") as f:
        pickle.dump(prior_sampler, f)

    print(f"gan time: {gan_train_time:.2f}")
    print(f"detect time: {detect_time:.2f}")
    print(f"partition time: {partition_time:.2f}")

if __name__ == "__main__":
    main()