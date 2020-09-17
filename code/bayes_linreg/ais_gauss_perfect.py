import pdb
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.mcmc.internal import util as mcmc_util

tfd = tfp.distributions


__all__ = [
    "sample_ais_perfect_transition",
]


def sample_ais_perfect_transition(
    num_steps,
    proposal_log_prob_fn,
    target_log_prob_fn,
    current_state,
    proposal_density,
    target_density,
    parallel_iterations=1,
    name=None,
):

    with tf.name_scope(name or "sample_ais_perfect_transition"):
        num_steps = tf.convert_to_tensor(
            value=num_steps, dtype=tf.int32, name="num_steps"
        )
        if mcmc_util.is_list_like(current_state):
            current_state = [
                tf.convert_to_tensor(s, name="current_state")
                for s in current_state
            ]
        else:
            current_state = tf.convert_to_tensor(
                value=current_state, name="current_state"
            )

        def _make_convex_combined_log_prob_fn(iter_):
            def _fn(*args):
                p = tf.identity(
                    proposal_log_prob_fn(*args), name="proposal_log_prob"
                )
                t = tf.identity(
                    target_log_prob_fn(*args), name="target_log_prob"
                )
                dtype = dtype_util.base_dtype(p.dtype)
                beta = tf.cast(iter_ + 1, dtype) / tf.cast(num_steps, dtype)
                return tf.identity(
                    beta * t + (1.0 - beta) * p,
                    name="convex_combined_log_prob",
                )

            return _fn

        def _find_exact_intermediate_density(iter_):
            dtype = dtype_util.base_dtype(proposal_density.dtype)
            beta = tf.cast(iter_ + 1, dtype) / tf.cast(num_steps, dtype)
            q_mean, q_covar = (
                proposal_density.mean(),
                proposal_density.covariance(),
            )
            p_mean, p_covar = (
                target_density.mean(),
                target_density.covariance(),
            )
            q_prec = np.linalg.inv(q_covar)
            p_prec = np.linalg.inv(p_covar)
            q_precxmean = np.linalg.solve(q_covar, q_mean)
            p_precxmean = np.linalg.solve(p_covar, p_mean)

            qt_precxmean = (1.0 - beta) * q_precxmean + beta * p_precxmean
            qt_prec = (1.0 - beta) * q_prec + beta * p_prec
            qt_cov = np.linalg.inv(qt_prec)
            qt_mean = np.dot(qt_cov, qt_precxmean)
            return tfd.MultivariateNormalFullCovariance(
                loc=qt_mean.astype(np.float32),
                covariance_matrix=qt_cov.astype(np.float32),
            )

        def _loop_body(
            iter_, ais_weights_lower, ais_weights_upper, current_state
        ):
            """Closure which implements `tf.while_loop` body."""
            fcurrent = _make_convex_combined_log_prob_fn(iter_ - 1)
            fnext = _make_convex_combined_log_prob_fn(iter_)

            x = (
                current_state
                if mcmc_util.is_list_like(current_state)
                else [current_state]
            )
            fcurrent_log_prob = fcurrent(*x)
            fnext_log_prob = fnext(*x)
            ais_weights_lower += fnext_log_prob - fcurrent_log_prob

            q_inter = _find_exact_intermediate_density(iter_)
            next_state = q_inter.sample(x[0].shape[0])

            x = (
                next_state
                if mcmc_util.is_list_like(next_state)
                else [next_state]
            )
            fcurrent_log_prob = fcurrent(*x)
            fnext_log_prob = fnext(*x)
            ais_weights_upper += fnext_log_prob - fcurrent_log_prob

            return [
                iter_ + 1,
                ais_weights_lower,
                ais_weights_upper,
                next_state,
            ]

        num_chains = current_state.shape[0]
        ais_weights_lower = tf.zeros(shape=[num_chains], dtype=tf.float32)
        ais_weights_upper = tf.zeros(shape=[num_chains], dtype=tf.float32)

        [
            _,
            ais_weights_lower,
            ais_weights_upper,
            current_state,
        ] = tf.while_loop(
            cond=lambda iter_, *args: iter_ < num_steps,
            body=_loop_body,
            loop_vars=[
                np.int32(0),
                ais_weights_lower,
                ais_weights_upper,
                current_state,
            ],
            parallel_iterations=parallel_iterations,
        )

        return [current_state, ais_weights_lower, ais_weights_upper]

