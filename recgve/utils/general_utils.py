#!/usr/bin/env python
__author__ = "XXX"
__email__ = "XXX"

import numpy as np
import logging
import os
import scipy.sparse as sps

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL"))


def get_top_k_scored_items(scores, top_k, sort_top_k=True):
    """Extract top K items from a matrix of scores for each user-item pair, optionally sort results per user.

    Args:
        scores (np.array): score matrix (users x items).
        top_k (int): number of top items to recommend.
        sort_top_k (bool): flag to sort top k results.

    Returns:
        np.array, np.array: indices into score matrix for each users top items, scores corresponding to top items.
    """

    logger.info(f"Sort_top_k:{sort_top_k}")
    # ensure we're working with a dense ndarray
    if isinstance(scores, sps.spmatrix):
        logger.warning("Scores are in a sparse format, densify them")
        scores = scores.todense()

    if scores.shape[1] < top_k:
        logger.warning(
            "Number of items is less than top_k, limiting top_k to number of items"
        )
    k = min(top_k, scores.shape[1])

    test_user_idx = np.arange(scores.shape[0])[:, None]

    # get top K items and scores
    # this determines the un-ordered top-k item indices for each user
    top_items = np.argpartition(scores, -k, axis=1)[:, -k:]
    top_scores = scores[test_user_idx, top_items]

    if sort_top_k:
        sort_ind = np.argsort(-top_scores)
        top_items = top_items[test_user_idx, sort_ind]
        top_scores = top_scores[test_user_idx, sort_ind]

    return np.array(top_items), np.array(top_scores)


def truncate_top_k(x, k):
    """Keep top_k highest values elements for each row of a numpy array

    Args:
        x (np.Array): numpy array
        k (int): number of elements to keep for each row

    Returns:
        np.Array: processed array
    """
    # todo: can be optimized to work on sparse matrices
    m, n = x.shape
    # get (unsorted) indices of top-k values
    topk_indices = np.argpartition(x, -k, axis=1)[:, -k:]
    # get k-th value
    rows, _ = np.indices((m, k))
    kth_vals = x[rows, topk_indices].min(axis=1)
    return np.where(x.T < kth_vals, 0, x.T).T


def truncate_top_k_2(x, k):
    """Keep top_k highest values elements for each row of a numpy array

    Args:
        x (np.Array): numpy array
        k (int): number of elements to keep for each row

    Returns:
        np.Array: processed array
    """
    s = x.shape
    # ind = np.argsort(x)[:, : s[1] - k]
    ind = np.argpartition(x, -k, axis=1)[:, :-k]
    rows = np.arange(s[0])[:, None]
    x[rows, ind] = 0
    return x


def normalize_csr_sparse_matrix(x, axis=1):
    """Normalize a csr_sparse matrix row-wise, sum of each row will be one

    Args:
        matrix (sps.csr_matrix): sparse matrix to normalize

    Returns:
        sps_csr_matrix: normalized sparse matrix
    """
    if not isinstance(x, sps.csr_matrix):
        raise ValueError("Matrix is not sps.csr_matrix")

    s = sps.csr_matrix(1 / x.sum(axis=axis))
    norm_x = x.multiply(s)

    # x.data = x.data / np.repeat(
    #     np.add.reduceat(x.data, x.indptr[:-1]), np.diff(x.indptr)
    # )
    return norm_x


def print_dict(dictionary):
    """Print a dictionary in a proper format

    Args:
        dictionary (dict):
    """
    for k, v in dictionary.items():
        print(f"{k}: {v}")


def threshold_sparse_matrix(X, keep_perc):
    """
    threshold a sparse matrix to a given sparsity level

    Args:
        X(sps.csr_matrix): sparse matrix
        sparsity_level(float): target sparsity level
    Returns:
        sps.csr_matrix: thresholded matrix
    """
    n = X.shape[0] * X.shape[1]
    non_zero_elem = len(X.data)
    sparsity = non_zero_elem / n
    print(f"Sparsity: {sparsity}")
    n_tokeep = round(non_zero_elem * keep_perc)
    induced_sparsity = n_tokeep / n
    print(f"Induced Sparsity: {induced_sparsity}")

    data = X.data.copy()
    idx = np.argpartition(data, -n_tokeep)[-n_tokeep:]
    min = sorted(data[idx])[0]
    X.data[data < min] = 0
    X.eliminate_zeros()
    return X, induced_sparsity, sparsity
