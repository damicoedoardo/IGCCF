#!/usr/bin/env python
__author__ = "XXX"
__email__ = "XXX"

import numpy as np
import tensorflow as tf
import scipy.sparse as sps
from constants import *

def to_tf_sparse_tensor(matrix, type=np.float32):
    """Convert a sparse matrix to a tensorflow sparse tensor

    Args:
        matrix (sps.spmatrix): sparse matrix
        type (np.dtype): type to use for the tensorflow conversion

    Returns:
        tf.SparseTensor: sparse tensor
    """
    if not isinstance(matrix, sps.coo_matrix):
        matrix = matrix.tocoo()
    matrix = matrix.astype(type)
    indices = np.mat([matrix.row, matrix.col]).transpose()
    return tf.SparseTensor(indices, matrix.data, matrix.shape)


def tf_dropout_sparse(X, keep_prob, n_nonzero_elems):
    """Dropout for sparse tensors.

    Args:
        X: sparse tensor
        keep_prob: 1 - prob_dropout
        n_nonzero_elems: number of non zero elements

    Returns:
        tf.SparseTensor
    """
    mask = tf.keras.backend.random_bernoulli((n_nonzero_elems,), p=keep_prob, seed=SEED)
    mask = tf.cast(mask, dtype=tf.bool)
    x_masked = tf.sparse.retain(X, mask)
    # normalize so that the expected value is the same
    # x_out = x_masked * tf.math.divide(1.0, keep_prob)
    x_masked._values = x_masked._values * tf.math.divide(1.0, keep_prob)
    return x_masked
