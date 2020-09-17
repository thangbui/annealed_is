from normal_ais import (
    run_ais,
    Gaussian1D,
    find_alpha_average_batch,
    find_average_batch,
)
import numpy as np
from tqdm import tqdm

import matplotlib.pylab as plt
from matplotlib.pyplot import cm


def compute_bounds_ais(
    mean1,
    mean2,
    var1,
    var2,
    no_betas,
    no_samples,
    seed=0,
    option="alpha",
    alpha=1.0,
    no_iters=10000,
):
    np.random.seed(seed)
    proposal = Gaussian1D(mean1, var1)
    target = Gaussian1D(mean2, var2)
    beta_vec = np.linspace(0, 1.0, no_betas)
    if option == "alpha":
        # if use alpha averaging
        beta_dists = find_alpha_average_batch(
            proposal, target, beta_vec, [alpha], no_iters
        )[0]
    else:
        # if use moment or geometric averaging
        beta_dists = find_average_batch(proposal, target, beta_vec, option)

    logZ_lower, logZ_upper = run_ais(beta_dists, no_samples)
    path = "../res/gaussian_1d/"
    fname = (
        path
        + "option_{}_mean1_{}_mean2_{}_var1_{}_var2_{}".format(
            option, mean1, mean2, var1, var2
        )
        + "_no_betas_{}_no_samples_{}_seed_{}_alpha_{}".format(
            no_betas, no_samples, seed, alpha
        )
    )
    np.savetxt(fname + "_lower_bound.txt", [logZ_lower])
    np.savetxt(fname + "_upper_bound.txt", [logZ_upper])
    return logZ_lower, logZ_upper


def compare_bounds():
    no_betass = [3, 5, 7, 9, 11, 15, 21, 31, 41, 51, 101]
    no_samples = 1000
    no_seeds = 20
    alphas = [0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.95, 1.0]
    # mean1, var1 = -4.0, 1.0
    mean1, var1 = -4.0, 0.2
    # mean2, var2 = 4.0, 0.2
    mean2, var2 = 4.0, 1.0

    # run alpha
    for i, alpha in tqdm(enumerate(alphas)):
        if alpha == 0.01 or alpha == 0.05:
            no_iters = 10000
        else:
            no_iters = 100
        for seed in tqdm(range(no_seeds)):
            for no_betas in no_betass:
                compute_bounds_ais(
                    mean1,
                    mean2,
                    var1,
                    var2,
                    no_betas,
                    no_samples,
                    seed,
                    "alpha",
                    alpha,
                    no_iters,
                )

    # run geometric
    for seed in tqdm(range(no_seeds)):
        for no_betas in no_betass:
            compute_bounds_ais(
                mean1,
                mean2,
                var1,
                var2,
                no_betas,
                no_samples,
                seed,
                "geometric",
            )

    # run moment
    for seed in tqdm(range(no_seeds)):
        for no_betas in no_betass:
            compute_bounds_ais(
                mean1, mean2, var1, var2, no_betas, no_samples, seed, "moment",
            )


def plot_bounds():
    no_betass = [3, 5, 7, 9, 11, 15, 21, 31, 41, 51, 101]
    no_samples = 1000
    no_seeds = 20
    alphas = [0.01, 0.05, 0.1, 0.2, 0.5, 0.8]
    # mean1, var1 = -4.0, 1.0
    mean1, var1 = -4.0, 0.2
    # mean2, var2 = 4.0, 0.2
    mean2, var2 = 4.0, 1.0

    path = "../../res/gaussian_1d/"
    alpha_res = np.zeros([no_seeds, len(alphas), len(no_betass), 2])
    for i, seed in enumerate(range(no_seeds)):
        for j, alpha in enumerate(alphas):
            for k, no_betas in enumerate(no_betass):
                fname = (
                    path
                    + "option_{}_mean1_{}_mean2_{}_var1_{}_var2_{}".format(
                        "alpha", mean1, mean2, var1, var2
                    )
                    + "_no_betas_{}_no_samples_{}_seed_{}_alpha_{}".format(
                        no_betas, no_samples, seed, alpha
                    )
                )
                alpha_res[i, j, k, 0] = np.loadtxt(fname + "_lower_bound.txt")
                alpha_res[i, j, k, 1] = np.loadtxt(fname + "_upper_bound.txt")
    alpha_mean = np.mean(alpha_res, 0)
    alpha_ste = np.std(alpha_res, 0) / np.sqrt(no_seeds)

    geometric_res = np.zeros([no_seeds, len(no_betass), 2])
    for i, seed in enumerate(range(no_seeds)):
        for k, no_betas in enumerate(no_betass):
            fname = (
                path
                + "option_{}_mean1_{}_mean2_{}_var1_{}_var2_{}".format(
                    "geometric", mean1, mean2, var1, var2
                )
                + "_no_betas_{}_no_samples_{}_seed_{}_alpha_{}".format(
                    no_betas, no_samples, seed, 1.0
                )
            )
            geometric_res[i, k, 0] = np.loadtxt(fname + "_lower_bound.txt")
            geometric_res[i, k, 1] = np.loadtxt(fname + "_upper_bound.txt")
    geometric_mean = np.mean(geometric_res, 0)
    geometric_ste = np.std(geometric_res, 0) / np.sqrt(no_seeds)

    moment_res = np.zeros([no_seeds, len(no_betass), 2])
    for i, seed in enumerate(range(no_seeds)):
        for k, no_betas in enumerate(no_betass):
            fname = (
                path
                + "option_{}_mean1_{}_mean2_{}_var1_{}_var2_{}".format(
                    "moment", mean1, mean2, var1, var2
                )
                + "_no_betas_{}_no_samples_{}_seed_{}_alpha_{}".format(
                    no_betas, no_samples, seed, 1.0
                )
            )
            moment_res[i, k, 0] = np.loadtxt(fname + "_lower_bound.txt")
            moment_res[i, k, 1] = np.loadtxt(fname + "_upper_bound.txt")
    moment_mean = np.mean(moment_res, 0)
    moment_ste = np.std(moment_res, 0) / np.sqrt(no_seeds)

    plt.figure(figsize=(5, 3))
    # colors = plt.get_cmap("tab10").colors[::-1]
    # colors = cm.viridis(np.linspace(0, 1, len(alphas) + 2))
    colors = cm.rainbow(np.linspace(0, 1, len(alphas) + 2))
    # colors =  plt.cm.Vega20c( (4./3*np.arange(20*3/4)).astype(int) )

    plt.axhline(0, color="k", linewidth=2, label="truth")

    plt.plot(
        no_betass, geometric_mean[:, 0], color=colors[0], label="geometric",
    )
    plt.fill_between(
        no_betass,
        geometric_mean[:, 0] + 3 * geometric_ste[:, 0],
        geometric_mean[:, 0] - 3 * geometric_ste[:, 0],
        color=colors[0],
        alpha=0.2,
    )

    plt.plot(
        no_betass, geometric_mean[:, 1], "--", color=colors[0],
    )
    plt.fill_between(
        no_betass,
        geometric_mean[:, 1] + 3 * geometric_ste[:, 1],
        geometric_mean[:, 1] - 3 * geometric_ste[:, 1],
        color=colors[0],
        alpha=0.2,
    )

    for i, alpha in enumerate(alphas):
        plt.plot(
            no_betass,
            alpha_mean[i, :, 0],
            color=colors[i + 1],
            label=r"$\alpha = {:.2f}$".format(alpha),
        )
        plt.fill_between(
            no_betass,
            alpha_mean[i, :, 0] + 3 * alpha_ste[i, :, 0],
            alpha_mean[i, :, 0] - 3 * alpha_ste[i, :, 0],
            color=colors[i + 1],
            alpha=0.2,
        )
        plt.plot(
            no_betass, alpha_mean[i, :, 1], "--", color=colors[i + 1],
        )
        plt.fill_between(
            no_betass,
            alpha_mean[i, :, 1] + 3 * alpha_ste[i, :, 1],
            alpha_mean[i, :, 1] - 3 * alpha_ste[i, :, 1],
            color=colors[i + 1],
            alpha=0.2,
        )

    plt.plot(
        no_betass, moment_mean[:, 0], color=colors[i + 2], label="moment",
    )
    plt.fill_between(
        no_betass,
        moment_mean[:, 0] + 3 * moment_ste[:, 0],
        moment_mean[:, 0] - 3 * moment_ste[:, 0],
        color=colors[i + 2],
        alpha=0.2,
    )

    plt.plot(no_betass, moment_mean[:, 1], "--", color=colors[i + 2])
    plt.fill_between(
        no_betass,
        moment_mean[:, 1] + 3 * moment_ste[:, 1],
        moment_mean[:, 1] - 3 * moment_ste[:, 1],
        color=colors[i + 2],
        alpha=0.2,
    )

    plt.xlabel("K")
    plt.ylabel(r"$\log \mathcal{Z}$ estimate")
    plt.xscale("log")
    plt.legend(ncol=2)
    plt.title(
        r"$N({:.2f},{:.2f}) \rightarrow N({:.2f},{:.2f})$".format(
            mean1, var1, mean2, var2
        )
    )
    plt.savefig(
        "/tmp/gaussian_1d_logZ_mean1_{}_mean2_{}_var1_{}_var2_{}.pdf".format(
            mean1, mean2, var1, var2,
        ),
        bbox_inches="tight",
        pad_inches=0,
    )


if __name__ == "__main__":
    # compare_bounds()
    plot_bounds()
