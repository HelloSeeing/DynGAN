# -*- coding:utf-8 -*-

import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib.gridspec import GridSpec
from sklearn import manifold

def grid_corners(n_row=4, n_col=None, edge=1.0):
    if n_col is None:
        n_col = n_row

    x = np.arange(n_row) * edge
    y = np.arange(n_col) * edge

    X, Y = np.meshgrid(x, y)
    x_coords = X.flatten().reshape(-1, 1)
    y_coords = Y.flatten().reshape(-1, 1)
    corners = np.concatenate((x_coords, y_coords), axis=1)

    return corners

def visualize_2dhist(pts, xrange=(-3.0, 3.0), yrange=(-3.0, 3.0), savename="2dhist.png", n_levels=20):
    fig, ax = plt.subplots()

    bgcolor = sns.color_palette("Greens", n_colors=256)[0]

    ax = sns.kdeplot(pts[:, 0], pts[:, 1], shade=True, cmap="Greens", n_levels=n_levels)
    ax.set_facecolor(bgcolor)
    ax.set_xlim(xmin=xrange[0], xmax=xrange[1])
    ax.set_ylim(ymin=yrange[0], ymax=yrange[1])

    fig.savefig(savename)
    plt.clf(); plt.cla(); plt.close()

def visualize_pts(pts, xrange=(-3.0, 3.0), yrange=(-3.0, 3.0), savename="pts.png", weights=None, title="points"):
    if weights is not None:
        idx = np.argsort(weights)
        pts = pts[idx]

    fig, ax = plt.subplots()

    colors = np.linspace(0.0, 1.0, num=len(pts))

    if weights is not None:
        im = ax.scatter(pts[:, 0], pts[:, 1], c=colors, alpha=0.5)
        cax = fig.add_axes([0.90, 0.3, 0.02, 0.5])
        plt.colorbar(im, cax=cax, orientation="vertical")
    else:
        # ax.scatter(pts[:, 0], pts[:, 1], c="b", alpha=0.5)
        ax.plot(pts[:, 0], pts[:, 1], "k.", alpha=0.5)
    
    if xrange is not None:
        ax.set_xlim(xmin=xrange[0], xmax=xrange[1])
    if yrange is not None:
        ax.set_ylim(ymin=yrange[0], ymax=yrange[1])
        
    ax.set_title(title, fontsize=20)

    plt.tight_layout()
    fig.savefig(savename)
    plt.clf(); plt.cla(); plt.close()

def visualize_pts_grid(pts, c, total_c=1, n_col=5, savename="pts.png"):
    n_row = np.ceil(total_c / n_col).astype(np.int)
    fig, ax = plt.subplots(nrows=n_row, ncols=n_col, figsize=(n_col * 2, n_row * 2))

    for i in set(c):
        c_ind = np.where(c == i)[0]
        pts_c = pts[c_ind]
        row = i // n_col
        col = i % n_col
        ax[row, col].scatter(pts_c[:, 0], pts_c[:, 1], c="blue", alpha=0.5)
    
    fig.savefig(savename)
    plt.clf(); plt.cla(); plt.close()

def visualize_recon(x, x_, xrange=(-3.0, 3.0), yrange=(-3.0, 3.0), savename="recon.png"):
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(x[:, 0], x[:, 1], c="b", alpha=0.3, label="input")
    ax.scatter(x_[:, 0], x_[:, 1], c="g", alpha=0.3, label="recon")
    ax.set_xlim(xmin=xrange[0], xmax=xrange[1])
    ax.set_ylim(ymin=yrange[0], ymax=yrange[1])

    ax.legend(loc="best")

    fig.savefig(savename)
    plt.clf(); plt.cla(); plt.close()

def visualize_latenthist(z, savename="latent.png"):
    fig, ax = plt.subplots()

    for i in range(z.shape[1]):
        plt.hist(z[:, i], bins="auto", density=True, histtype="step")

    plt.savefig(savename)
    plt.clf(); plt.cla(); plt.close()

def visualize_gz2d(z, gz, corners, savename="gz_2d.png"):
    assert z.shape[0] == gz.shape[0]
    assert z.shape[1] == 2
    assert corners.shape[1] == gz.shape[1]

    n_samples, n_depth = gz.shape
    n_corners = corners.shape[0]

    gz = gz.reshape((n_samples, 1, n_depth))
    corners = corners.reshape((1, n_corners, n_depth))

    dists = np.linalg.norm(gz - corners, axis=-1)
    labels = np.argmin(dists, axis=1)
    label_dists = np.array([dists[i, labels[i]] for i in range(n_samples)])
    corner_colors = np.linspace(0, 1, num=n_corners)
    colors = corner_colors[labels]

    # 2d visualize generated samples
    gz = gz.squeeze()
    xmin_, xmax_ = np.min(corners[:, :, 0]), np.max(corners[:, :, 0])
    ymin_, ymax_ = np.min(corners[:, :, 1]), np.max(corners[:, :, 1])
    xmin, xmax = xmin_ - (xmax_ - xmin_), xmax_ + (xmax_ - xmin_)
    ymin, ymax = ymin_ - (ymax_ - ymin_), ymax_ + (ymax_ - ymin_)
    fig, ax = plt.subplots()

    ax.scatter(gz[:, 0], gz[:, 1], c=colors, alpha=0.5)
    ax.set_xlim(left=xmin, right=xmax)
    ax.set_ylim(top=ymin, bottom=ymax)

    fig.savefig(savename)
    plt.show()

def visualize_latent(z, gz, corners, savename_2d="latent_2d.png", savename_3d="latent_3d.png"):
    assert z.shape[0] == gz.shape[0]
    assert z.shape[1] == 2
    assert corners.shape[1] == gz.shape[1]

    n_samples, n_depth = gz.shape
    n_corners = corners.shape[0]

    gz = gz.reshape((n_samples, 1, n_depth))
    corners = corners.reshape((1, n_corners, n_depth))

    dists = np.linalg.norm(gz - corners, axis=-1)
    labels = np.argmin(dists, axis=1)
    label_dists = np.array([dists[i, labels[i]] for i in range(n_samples)])
    corner_colors = np.linspace(0, 1, num=n_corners, endpoint=False)
    colors = corner_colors[labels]
    texts = [str(l) for l in labels]

    # 2d visualize latent
    fig, ax = plt.subplots()
    ax.scatter(z[:, 0], z[:, 1], c=colors, alpha=0.5)
    # for i in range(n_samples):
    #     ax.text(z[i, 0], z[i, 1], texts[i], fontsize=5)
    plt.show()
    plt.savefig(savename_2d)

    # 3d visualize latent
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(z[:, 0], z[:, 1], label_dists, c=colors, alpha=0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Distance to Nearest Corner")
    plt.show()
    plt.savefig(savename_3d)

    return z, label_dists, colors

def visualize_loss(loss_record, savename="loss.png"):
    fig, ax = plt.subplots()
    axt = ax.twinx()

    x = np.arange(len(loss_record["reconstruction"]))
    recon, = ax.plot(x, loss_record["reconstruction"], label="reconstruction", c="b")
    jsdiv, = axt.plot(x, loss_record["js_divergence"], label="js_divergence", c="g")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction", color="b")
    ax.set_yscale("log")
    ax.tick_params(axis='y', colors=recon.get_color())
    axt.set_ylabel("JS Divergence", color="g")
    axt.tick_params(axis='y', colors=jsdiv.get_color())

    plt.savefig(savename)
    plt.clf(); plt.cla(); plt.close()

def visualize_loss2(loss_record, savename="loss.png"):
    elms = len(loss_record.keys())
    fig, ax = plt.subplots(nrows=1, ncols=elms, figsize=(elms * 7, 5))

    for i, (key, loss) in enumerate(loss_record.items()):

        x = np.arange(len(loss))

        ax[i].plot(x, loss)
        ax[i].set_title(key.capitalize(), fontsize=20)
        ax[i].set_xlabel("Epoch")
        ax[i].set_ylabel(key)
        if key == "reconstruction":
            ax[i].set_yscale("log")

    plt.tight_layout()
    plt.savefig(savename)
    plt.clf(); plt.cla(); plt.close()

def visualize_contour(x, y, f, title="", savename="countour.png"):
    a = plt.contourf(x, y, f, cmap=plt.cm.Spectral)
    plt.colorbar(a)
    plt.title(title)

    plt.tight_layout()
    plt.savefig(savename)
    plt.clf(); plt.cla(); plt.close()

def visualize_contour2(x, y, f, pts, x_range=None, y_range=None, title="", savename="countour.png"):
    fig, ax = plt.subplots()
    ax.set_xlim(x_range)
    ax.set_xlim(y_range)
    
    a = ax.contourf(x, y, f, cmap=plt.get_cmap("Oranges"), alpha=0.4)
    cbar = fig.colorbar(a)
    cbar.ax.set_ylabel("D values")
    ax.set_title(title)

    ax.scatter(pts[:, 0], pts[:, 1], alpha=0.5, c="b")

    plt.tight_layout()
    plt.savefig(savename)
    plt.clf(); plt.cla(); plt.close()

def visualize_contour3(x, y, vals, grads, pts, x_range=None, y_range=None, grad_step=2, title="", savename="countour.png"):
    fig, ax = plt.subplots(figsize=(13, 10))
    ax.set_xlim(x_range)
    ax.set_xlim(y_range)
    
    a = ax.contourf(x, y, vals, cmap=plt.get_cmap("Oranges"), alpha=0.4)
    cbar = fig.colorbar(a)
    cbar.ax.set_ylabel("D values", fontsize=23)
    ax.set_title(title, fontsize=25)

    ax.scatter(pts[:, 0], pts[:, 1], alpha=0.5, c="b")

    grad_h, grad_w, _ = grads.shape
    for r in range(0, grad_h, grad_step):
        for c in range(0, grad_w, grad_step):
            x0, y0 = x[r, c], y[r, c]
            dx, dy = grads[r, c]
            ax.arrow(x0, y0, dx, dy, head_width=0.05, head_length=0.1, fc='k', ec='k')

    plt.tight_layout()
    plt.savefig(savename)
    plt.clf(); plt.cla(); plt.close()


def visualize_contour4(x, y, vals, grads, pts, x_range=None, y_range=None, grad_step=2, title="", savename="countour.png"):
    N = len(vals)
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8 * N, 6))

    for n in range(N):
        ax = axs[n]
        ax.set_xlim(x_range)
        ax.set_xlim(y_range)
        
        a = ax.contourf(x, y, vals[n], cmap=plt.get_cmap("Oranges"), alpha=0.4)
        cbar = fig.colorbar(a, ax=ax, shrink=0.6)
        cbar.ax.set_ylabel("D values", fontsize=23)
        ax.set_title(title, fontsize=25)

        ax.scatter(pts[n][:, 0], pts[n][:, 1], alpha=0.5, c="b")

        grad_h, grad_w, _ = grads[n].shape
        for r in range(0, grad_h, grad_step):
            for c in range(0, grad_w, grad_step):
                x0, y0 = x[r, c], y[r, c]
                dx, dy = grads[n][r, c]
                ax.arrow(x0, y0, dx, dy, head_width=0.05, head_length=0.1, fc='k', ec='k')

    plt.tight_layout()
    plt.savefig(savename)
    plt.clf(); plt.cla(); plt.close()


def visualize_grad(grad, generation, weights, Dvals, title="", savename="grad.png"):
    K = len(grad)

    n_col = 10 if K > 10 else K
    n_row = 3 + np.ceil(K / n_col).astype(np.int32)

    fig = plt.figure(figsize=(n_col * 1.5, n_row + 6))
    height_ratios = np.ones(n_row, dtype=np.int32)
    height_ratios[0] = 3
    height_ratios[1] = 3
    height_ratios[2] = 3
    gs = GridSpec(n_row, n_col, figure=fig, height_ratios=height_ratios)

    ax = fig.add_subplot(gs[0, :])
    x = list(range(K))
    ax.bar(x, grad)
    ax.set_title("grad on weights", fontsize=15)

    ax = fig.add_subplot(gs[1, :])
    x = list(range(K))
    ax.bar(x, weights)
    ax.set_title("weights", fontsize=15)

    ax = fig.add_subplot(gs[2, :])
    x = list(range(K))
    ax.bar(x, Dvals - np.mean(Dvals))
    ax.autoscale()
    yticks = ax.get_yticks()
    ax.set_yticklabels(["%.4f" % (v + np.mean(Dvals)) for v in yticks])
    ax.set_title("Dvals", fontsize=15)

    for k in range(K):
        row_idx = 3 + k // n_col
        col_idx = k % n_col
        ax = fig.add_subplot(gs[row_idx, col_idx])
        gens = generation[k, :]
        ax.scatter(gens[:, 0], gens[:, 1], alpha=0.5)
        ax.set_axis_on()
    
    plt.tight_layout()
    plt.savefig(savename)
    plt.clf(); plt.cla(); plt.close()



def visualize_weights(weights, title="", savename="weights.png"):
    K = len(weights)

    fig, ax = plt.subplots(figsize=(15, 3))

    x = list(range(K))
    ax.bar(x, weights)
    ax.set_title("weights")
    
    plt.tight_layout()
    plt.savefig(savename)
    plt.clf(); plt.cla(); plt.close()


def visualize_pts2(pts, idx, n_class, xrange=(-3.0, 3.0), yrange=(-3.0, 3.0), savename="pts.png", weights=None, title="points"):
    """
    Args:
        pts: np.ndarray, shape in `[N, 2]`.
        idx: list or 1-dim np.ndarray, shape in `[N]`.
        n_class: int, number of total classes.
        xrange: tuple, (xmin, xmax).
        yrange: tuple, (ymin, ymax).
        savename: str, name of saved file.
        title: title of the whole figure.
    """

    fig, ax = plt.subplots()

    for i in range(n_class):
        i_idx = np.where(idx == i)[0]
        # ax.scatter(pts[i_idx, 0], pts[i_idx, 1], alpha=0.5, label="%d" % i)
        ax.plot(pts[i_idx, 0], pts[i_idx, 1], ".", alpha=0.5, label="%d" % i)

    # ax.legend(ncol=math.ceil(n_class / 10), bbox_to_anchor=(1.0, 1.0))
    if xrange is not None and yrange is not None:
        ax.set_xlim(xmin=xrange[0], xmax=xrange[1])
        ax.set_ylim(ymin=yrange[0], ymax=yrange[1])

    if title is not None:
        ax.set_title(title, fontsize=20)

    plt.tight_layout()
    fig.savefig(savename)
    plt.clf(); plt.cla(); plt.close()


def visualize_pts3(pts, gens, idx, n_class, xrange=(-3.0, 3.0), yrange=(-3.0, 3.0), savename="pts.png", weights=None, title="points"):
    """
    Args:
        pts: np.ndarray, shape in `[N, 2]`.
        gens: np.ndarray, shape in `[N, 2]`.
        idx: list or 1-dim np.ndarray, shape in `[N]`.
        n_class: int, number of total classes.
        xrange: tuple, (xmin, xmax).
        yrange: tuple, (ymin, ymax).
        savename: str, name of saved file.
        title: title of the whole figure.
    """

    fig, ax = plt.subplots()

    for i in range(n_class):
        i_idx = np.where(idx == i)[0]
        # ax.scatter(pts[i_idx, 0], pts[i_idx, 1], alpha=0.5, label="%d" % i)
        ax.plot(gens[i_idx, 0], gens[i_idx, 1], ".", alpha=0.5, label="%d" % i)

    # ax.plot(gens[:, 0], gens[:, 1], "b.", alpha=0.1)

    ax.plot(pts[:, 0], pts[:, 1], "kx", alpha=0.01, label="data")
    ax.set_aspect("equal")

    # ax.legend(ncol=math.ceil(n_class / 10), bbox_to_anchor=(1.0, 1.0))
    ax.set_xlim(xmin=xrange[0], xmax=xrange[1])
    ax.set_ylim(ymin=yrange[0], ymax=yrange[1])
    ax.locator_params(nbins=4)

    # if title is not None:
    #     ax.set_title(title, fontsize=20)

    plt.tight_layout()
    fig.savefig(savename, bbox_inches="tight", dpi=100)
    plt.clf(); plt.cla(); plt.close()


def visualize_grad2(w, Dvals, grad_on_w, title=None, savename="grad2.png"):
    """
    Args:
        w: np.ndarray, shape in `[C]`, weights
        Dvals: np.ndarray, shape in `[C]`, discriminator values
        grad_on_w: np.ndarray, shape in `[C]`, gradients
        title: str, title
        savename: str, name of saved file.
    """
    ncol = w.shape[0]
    nrow = 3

    fig, ax = plt.subplots(nrows=nrow, ncols=1, figsize=(ncol * 1.5, nrow * 2))

    x = np.arange(ncol)

    ax[0].bar(x, w)
    ax[0].set_ylabel("weights", fontsize=10)
    ax[0].set_xticks([])
    
    ax[1].bar(x, Dvals)
    ax[1].set_ylabel("Dvals", fontsize=10)
    ax[1].set_xticks([])
    
    ax[2].bar(x, grad_on_w)
    ax[2].set_ylabel("gradients", fontsize=10)
    ax[2].set_xticks(x)

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()
    plt.savefig(savename)
    plt.clf(); plt.cla(); plt.close()


def visualize_entropy(entropy, n_class, title=None, savename="entropy.png"):
    """
    Args:
        entropy: np.ndarray, shape in `[batch]`.
        n_class: int, number of classes.
        title: str, title of figure.
        savename: str, filename of saved file.
    """
    sns.set(style="white", palette="muted", color_codes=True)
    fig, ax = plt.subplots(figsize=(7, 3))
    try:
        sns.distplot(entropy, hist=False, rug=True, color="g", kde_kws={"shade": True}, ax=ax)
    except np.linalg.LinAlgError as e:
        sns.barplot(entropy[:1], [1.0], ax=ax)

    if title is not None:
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(savename)
    plt.clf(); plt.cla(); plt.close()


def visualize_gradients(data, gradients, savename="gradients.png"):

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.plot(data[:, 0], data[:, 1], ".", alpha=0.5)

    xmin, xmax = ax.get_xlim()
    ratio = (xmax - xmin) / np.max(np.abs(gradients[:, 0])) * 0.05

    for i in range(0, data.shape[0], 3):
        x0, y0 = data[i]
        dx, dy = gradients[i] * ratio
        ax.arrow(x0, y0, dx, dy, head_width=0.05, head_length=0.1, fc='k', ec='k')

    plt.tight_layout()
    plt.savefig(savename)
    plt.clf(); plt.cla(); plt.close()


def visualize_tsne(X, savename="tsne.png"):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    plt.figure(figsize=(8, 8))
    plt.scatter(X_norm[:, 0], X_norm[:, 1])

    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()
    plt.savefig(savename)
    plt.clf(); plt.cla(); plt.close()


def visualize_statistics(steps, n_discrete, n_modes, q1, q2, q3, savename="stats.png", **kwargs):

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))

    line0 = ax[0].plot(steps, n_discrete, "-", label="n discrete")
    ax[0].set_xlabel("Steps")
    ax[0].set_ylabel("n discrete")

    axx = ax[0].twinx()
    line1 = axx.plot(steps, n_modes, "-r", label="n modes")
    axx.set_ylabel("n modes")

    line = line0 + line1
    ax[0].legend(line, [l.get_label() for l in line], loc="lower right")


    line0 = ax[1].plot(steps, n_discrete, "-", label="n discrete")
    ax[1].set_xlabel("Steps")
    ax[1].set_ylabel("n discrete")

    axx = ax[1].twinx()
    axx.set_ylabel("Quality")

    line1 = axx.plot(steps, q1, "-r", label="Quality 1-std")
    line2 = axx.plot(steps, q2, "-g", label="Quality 2-std")
    line3 = axx.plot(steps, q3, "-m", label="Quality 3-std")

    line = line0 + line1 + line2 + line3
    ax[1].legend(line, [l.get_label() for l in line], loc="upper left")

    plt.tight_layout()
    plt.savefig(savename)
    plt.clf(); plt.cla(); plt.close()