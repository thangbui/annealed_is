import numpy as np
import pdb
from utils import compute_log_marginal_likelihood


no_seeds = 10
dims = [2, 5, 10]
no_data = 100
sigma = 1.0
for dim in dims:
    for seed in range(no_seeds):
        # create random data
        x = np.random.randn(no_data, dim)
        true_weights = np.random.randn(dim)
        y = np.dot(x, true_weights) + sigma * np.random.randn(no_data)
        # compute exact marginal likelihood
        lm = compute_log_marginal_likelihood(x, y, sigma)
        # save
        fname = "./data/linreg_dim_{}_seed_{}".format(dim, seed)
        np.savetxt(fname + "_input.txt", x)
        np.savetxt(fname + "_output.txt", y.T)
        np.savetxt(fname + "_log_ml.txt", [lm])
