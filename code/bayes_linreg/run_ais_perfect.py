import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tqdm import tqdm


from ais_gauss_perfect import sample_ais_perfect_transition
from utils import target_log_prob_fn, compute_exact_posterior

tfd = tfp.distributions

np.random.seed(42)
tf.random.set_seed(42)


def run_ais(seed, dims, num_chains, num_steps):
    # load data
    dtype = np.float32
    fname = "data/linreg_dim_{}_seed_{}".format(dims, seed)
    x = np.loadtxt(fname + "_input.txt").astype(dtype)
    y = np.loadtxt(fname + "_output.txt").astype(dtype)
    true_lm = np.loadtxt(fname + "_log_ml.txt").astype(dtype)

    # configure ais
    mean, cov, _, _ = compute_exact_posterior(x, y)

    def target_log_prob(weights):
        return target_log_prob_fn(weights, x, y)

    proposal = tfd.MultivariateNormalDiag(loc=tf.zeros(dims, dtype))
    target = tfd.MultivariateNormalFullCovariance(
        loc=mean.astype(dtype), covariance_matrix=cov.astype(dtype)
    )

    (
        weight_samples,
        ais_weights_lower,
        ais_weights_upper,
    ) = sample_ais_perfect_transition(
        num_steps=num_steps,
        proposal_log_prob_fn=proposal.log_prob,
        target_log_prob_fn=target_log_prob,
        current_state=proposal.sample(num_chains),
        proposal_density=proposal,
        target_density=target,
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
    num_chains = 1000
    for dims in tqdm([2, 5, 10]):
        for seed in tqdm(range(10)):
            for num_steps in [5, 10, 20, 50, 100]:
                res = run_ais(seed, dims, num_chains, num_steps)
                fname = "res/linreg_dim_{}_seed_{}_num_chains_{}_num_steps_{}_result.txt".format(
                    dims, seed, num_chains, num_steps
                )
                np.savetxt(fname, res, fmt="%.4f")


if __name__ == "__main__":
    run_loop()
