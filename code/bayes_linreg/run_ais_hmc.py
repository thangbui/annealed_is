import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tqdm import tqdm

from ais_mcmc import sample_annealed_importance_chain
from utils import make_prior, make_likelihood, target_log_prob_fn

tfd = tfp.distributions

np.random.seed(42)
tf.random.set_seed(42)


def run_ais(seed, dims, num_chains, num_steps, hmc_step):

    # load data
    dtype = np.float32
    fname = "data/linreg_dim_{}_seed_{}".format(dims, seed)
    x = np.loadtxt(fname + "_input.txt").astype(dtype)
    y = np.loadtxt(fname + "_output.txt").astype(dtype)
    true_lm = np.loadtxt(fname + "_log_ml.txt").astype(dtype)

    def target_log_prob(weights):
        return target_log_prob_fn(weights, x, y)

    proposal = tfd.MultivariateNormalDiag(loc=tf.zeros(dims, dtype))

    (
        weight_samples,
        ais_weights_lower,
        ais_weights_upper,
        kernel_results,
    ) = sample_annealed_importance_chain(
        num_steps=num_steps,
        proposal_log_prob_fn=proposal.log_prob,
        target_log_prob_fn=target_log_prob,
        current_state=proposal.sample(num_chains),
        make_kernel_fn=lambda tlp_fn: tfp.mcmc.NoUTurnSampler(
            target_log_prob_fn=tlp_fn, step_size=hmc_step,
        ),
        seed=42,
    )
    log_normalizer_estimate_lower = tf.reduce_logsumexp(
        ais_weights_lower
    ) - np.log(num_chains)
    log_normalizer_estimate_upper = tf.reduce_logsumexp(
        ais_weights_upper
    ) - np.log(num_chains)
    log_normalizer_lowerbound = tf.reduce_mean(ais_weights_lower)
    log_normalizer_upperbound = tf.reduce_mean(ais_weights_upper)
    log_normalizer_truth = true_lm

    return [
        log_normalizer_estimate_lower,
        log_normalizer_estimate_upper,
        log_normalizer_lowerbound,
        log_normalizer_upperbound,
        log_normalizer_truth,
    ]


def run_loop():
    num_chainss = [1000]
    for num_chains in tqdm(num_chainss):
        for dims in tqdm([2, 5, 10]):
            for num_steps in tqdm([5, 10, 20, 50, 100, 1000]):
                for seed in tqdm(range(10)):
                    for hmc_step in [0.05, 0.1]:
                        res = run_ais(
                            seed, dims, num_chains, num_steps, hmc_step
                        )
                        fname = "res/linreg_nuts_dim_{}_seed_{}_num_chains_{}_num_steps_{}_hmc_step_{}_result.txt".format(
                            dims, seed, num_chains, num_steps, hmc_step
                        )
                        np.savetxt(fname, res, fmt="%.4f")


if __name__ == "__main__":
    run_loop()
