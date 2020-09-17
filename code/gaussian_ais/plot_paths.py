import numpy as np
import matplotlib.pylab as plt
import scipy.stats as stats
from copy import deepcopy
from tqdm import tqdm
import pdb
from matplotlib.pyplot import cm

from matplotlib.offsetbox import AnchoredText

from normal_ais import (
    Gaussian1D,
    find_sum_elements_geometric,
    find_sum_elements_moment,
    find_alpha_average_batch,
    find_average_batch,
)


def plot_dists(dists, betas, name=""):
    plt.figure(figsize=(6, 3))
    # colors = cm.rainbow(np.linspace(0, 1, len(dists)))
    colors = cm.viridis(np.linspace(0, 1, len(dists)))
    for i, dist in enumerate(dists):
        # print(name, betas[i], dist.mean, dist.variance)
        mu, sigma = dist.mean, np.sqrt(dist.variance)
        x = np.linspace(-10, 10, 500)
        plt.plot(
            x,
            stats.norm.pdf(x, mu, sigma),
            color=colors[i % len(colors)],
            label="{:.02f}".format(betas[i]),
        )
    plt.xlabel(r"$z$")
    plt.ylabel(r"$q(z)$")
    # plt.legend()
    plt.ylim([-0.02, 1.0])
    plt.savefig("/tmp/{}.pdf".format(name), bbox_inches="tight", pad_inches=0)


def plot_ti_auc(fs, betas, name="", lower_bound=True):
    # colors = cm.rainbow(np.linspace(0, 1, len(betas)))
    colors = cm.viridis(np.linspace(0, 1, len(betas)))
    fig, ax = plt.subplots(figsize=(6, 3))
    patches = ax.bar(
        betas[:-1], fs, width=betas[1:] - betas[:-1], align="edge"
    )
    for i in range(len(betas) - 1):
        patches[i].set_facecolor(colors[i])

    if lower_bound:
        text = r"$\sum_k F_k = {:.2f}$".format(np.sum(fs))
        ylabel = r"$F_{k}$"
    else:
        text = r"$\sum_k \tilde{{F}}_k = {:.2f}$".format(np.sum(fs))
        ylabel = r"$\tilde{{F}}_{k}$"
    text_box = AnchoredText(
        text, frameon=False, loc=4, pad=0, prop={"size": 12},
    )
    plt.setp(text_box.patch, facecolor="white", alpha=0.5)
    plt.gca().add_artist(text_box)

    plt.xlabel(r"$\beta$")
    plt.ylabel(ylabel)
    plt.savefig("/tmp/{}.pdf".format(name), bbox_inches="tight", pad_inches=0)


def run():
    mean1, var1 = -4.0, 1.0
    mean2, var2 = 4.0, 0.2
    # mean1, var1 = -4.0, 1.0
    # mean2, var2 = 4.0, 1.0
    # mean1, var1 = -4.0, 0.2
    # mean2, var2 = 4.0, 1.0
    # mean1, var1 = -4.0, 1.0
    # mean2, var2 = -4.0, 1.0
    K = 25
    dist1 = Gaussian1D(mean=mean1, variance=var1)
    dist2 = Gaussian1D(mean=mean2, variance=var2)
    beta_vec = np.linspace(0, 1.0, K)

    ga_dists = find_average_batch(dist1, dist2, beta_vec, "geometric")
    fname = "geometric_path_mean1_{}_mean2_{}_var1_{}_var2_{}".format(
        mean1, mean2, var1, var2,
    )
    plot_dists(ga_dists, beta_vec, fname + "_dist")

    ls, us = find_sum_elements_geometric(ga_dists, beta_vec)
    plot_ti_auc(ls, beta_vec, fname + "_lower_bound", lower_bound=True)
    plot_ti_auc(us, beta_vec, fname + "_upper_bound", lower_bound=False)

    ma_dists = find_average_batch(dist1, dist2, beta_vec, "moment")
    fname = "moment_path_mean1_{}_mean2_{}_var1_{}_var2_{}".format(
        mean1, mean2, var1, var2,
    )
    plot_dists(ma_dists, beta_vec, fname + "_dist")

    ls, us = find_sum_elements_moment(ma_dists, beta_vec)
    plot_ti_auc(ls, beta_vec, fname + "_lower_bound", lower_bound=True)
    plot_ti_auc(us, beta_vec, fname + "_upper_bound", lower_bound=False)

    alphas = [0.001, 0.01, 0.05, 0.1, 0.5]
    no_iters = 20000
    alpha_distss = find_alpha_average_batch(
        dist1, dist2, beta_vec, alphas, no_iters
    )
    for i, alpha in enumerate(alphas):
        alpha_dists = alpha_distss[i]
        fname = "alpha_path_{:.2f}_mean1_{}_mean2_{}_var1_{}_var2_{}".format(
            alpha, mean1, mean2, var1, var2,
        )
        plot_dists(alpha_dists, beta_vec, fname + "_dist")

        ls, us = find_sum_elements_moment(alpha_dists, beta_vec)
        plot_ti_auc(ls, beta_vec, fname + "_lower_bound", lower_bound=True)
        plot_ti_auc(us, beta_vec, fname + "_upper_bound", lower_bound=False)


if __name__ == "__main__":
    run()
