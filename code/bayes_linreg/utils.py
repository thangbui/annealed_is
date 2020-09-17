import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np

tfd = tfp.distributions


def compute_log_marginal_likelihood(x, y, sigma=1.0):
    dtype = np.float32
    x, y = x.astype(dtype), y.astype(dtype)
    dist = tfd.MultivariateNormalFullCovariance(
        covariance_matrix=tf.matmul(x, tf.transpose(x)) + tf.eye(x.shape[0])
    )
    return dist.log_prob(y)


def make_prior(dims, dtype=np.float32):
    return tfd.MultivariateNormalDiag(loc=tf.zeros(dims, dtype=np.float32))


def make_likelihood(weights, x):
    return tfd.MultivariateNormalDiag(
        loc=tf.tensordot(weights, x, axes=[[-1], [-1]])
    )


def target_log_prob_fn(weights, x, y):
    a = make_prior(x.shape[1]).log_prob(weights)
    b = make_likelihood(weights, x).log_prob(y)
    return a + b


def compute_exact_posterior(x, y, sigma=1.0):
    precxmean = np.dot(x.T, y) / sigma ** 2
    prec = np.eye(x.shape[1]) + np.dot(x.T, x) / sigma ** 2
    cov = np.linalg.inv(prec)
    mean = np.dot(cov, precxmean)
    return mean, cov, prec, precxmean
