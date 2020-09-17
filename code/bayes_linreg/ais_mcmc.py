# Copyright 2018 The TensorFlow Probability Authors.
# Thang Bui, Sep 2020:
#   + change the weight updates
#   + return the upper bound of the log partition fn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Markov chain Monte Carlo driver, `sample_chain_annealed_importance_chain`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import pdb

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
from tensorflow_probability.python.mcmc.sample_annealed_importance import (
    _find_inner_mh_results,
    AISResults,
)


__all__ = [
    "sample_annealed_importance_chain",
]


def sample_annealed_importance_chain(
    num_steps,
    proposal_log_prob_fn,
    target_log_prob_fn,
    current_state,
    make_kernel_fn,
    parallel_iterations=1,
    seed=None,
    name=None,
):
    is_seeded = seed is not None
    seed = samplers.sanitize_seed(seed, salt="mcmc.sample_ais_chain")

    with tf.name_scope(name or "sample_annealed_importance_chain"):
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

        def _loop_body(
            iter_,
            seed,
            ais_weights_lower,
            ais_weights_upper,
            current_state,
            kernel_results,
        ):
            """Closure which implements `tf.while_loop` body."""
            iter_seed, next_seed = (
                samplers.split_seed(seed, salt="ais_chain.seeded_one_step")
                if is_seeded
                else (seed, seed)
            )
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

            kernel = make_kernel_fn(fnext)
            one_step_kwargs = dict(seed=iter_seed) if is_seeded else {}
            next_state, inner_results = kernel.one_step(
                current_state, kernel_results.inner_results, **one_step_kwargs
            )
            kernel_results = AISResults(
                proposal_log_prob=fcurrent_log_prob,
                target_log_prob=fnext_log_prob,
                inner_results=inner_results,
            )
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
                next_seed,
                ais_weights_lower,
                ais_weights_upper,
                next_state,
                kernel_results,
            ]

        def _bootstrap_results(init_state):
            """Creates first version of `previous_kernel_results`."""
            kernel = make_kernel_fn(_make_convex_combined_log_prob_fn(iter_=0))
            inner_results = kernel.bootstrap_results(init_state)
            num_chains = current_state.shape[0]
            proposal_log_prob = tf.fill(
                [num_chains], np.nan, name="bootstrap_proposal_log_prob"
            )
            target_log_prob = tf.fill(
                [num_chains], np.nan, name="target_target_log_prob"
            )

            return AISResults(
                proposal_log_prob=proposal_log_prob,
                target_log_prob=target_log_prob,
                inner_results=inner_results,
            )

        previous_kernel_results = _bootstrap_results(current_state)
        num_chains = current_state.shape[0]
        ais_weights_lower = tf.zeros(shape=[num_chains], dtype=tf.float32)
        ais_weights_upper = tf.zeros(shape=[num_chains], dtype=tf.float32)

        [
            _,
            _,
            ais_weights_lower,
            ais_weights_upper,
            current_state,
            kernel_results,
        ] = tf.while_loop(
            cond=lambda iter_, *args: iter_ < num_steps,
            body=_loop_body,
            loop_vars=[
                np.int32(0),  # iter_
                seed,
                ais_weights_lower,
                ais_weights_upper,
                current_state,
                previous_kernel_results,
            ],
            parallel_iterations=parallel_iterations,
        )

        return [
            current_state,
            ais_weights_lower,
            ais_weights_upper,
            kernel_results,
        ]

