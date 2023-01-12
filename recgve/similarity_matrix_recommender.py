#!/usr/bin/env python
__author__ = "XXX"
__email__ = "XXX"

from abc import ABC, abstractmethod
from recommender_interface import Recommender
from utils.general_utils import truncate_top_k_2, normalize_csr_sparse_matrix, threshold_sparse_matrix
from utils.pandas_utils import remap_column_consecutive
from constants import *
import numpy as np
import scipy.sparse as sps
import tensorflow as tf
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import similaripy
from utils.tensorflow_utils import to_tf_sparse_tensor
from sklearn.metrics.pairwise import cosine_similarity
from numpy.linalg import matrix_power


class ItemSimilarityMatrixRecommender(Recommender, ABC):
    def __init__(self, train_data):
        super().__init__(train_data=train_data)
        # will be set by compute similarity matrix
        self.sim_matrix = None

    def compute_items_scores(self, user_data):
        sim = self.compute_similarity_matrix()

        user_interactions = user_data["interactions"].copy()
        # todo convert interaction df into sps matrix
        # remap to consecutive user idxs
        mapping_dict = remap_column_consecutive(
            user_interactions, DEFAULT_USER_COL, mapping_dict=True
        )
        user_idxs = list(mapping_dict.keys())

        rows = user_interactions[DEFAULT_USER_COL].values
        cols = user_interactions[DEFAULT_ITEM_COL].values
        data = np.ones(len(rows))

        urm = sps.coo_matrix(
            (data, (rows, cols)), shape=(len(user_idxs), self.item_count)
        )
        # urm = to_tf_sparse_tensor(urm)
        # scores = tf.sparse.sparse_dense_matmul(urm, sim)
        scores = urm * sim
        scores_df = pd.DataFrame(scores, index=user_idxs)
        return scores_df

    @abstractmethod
    def compute_similarity_matrix(self):
        pass


class ItemKnn(ItemSimilarityMatrixRecommender):
    def __init__(self, train_data, topk):
        super().__init__(train_data=train_data)
        self.topk = topk

    def compute_similarity_matrix(self):
        rows = self.train_data[DEFAULT_USER_COL].values
        cols = self.train_data[DEFAULT_ITEM_COL].values
        data = np.ones(len(rows))
        urm = sps.coo_matrix(
            (data, (rows, cols)), shape=(self.user_count, self.item_count)
        )

        # degree_user = np.array(urm.sum(axis=1)).squeeze()
        # D_user = sps.diags(degree_user, format="csr")
        # D_user = D_user.power(-1)
        #
        # degree_item = np.array(urm.sum(axis=0)).squeeze()
        # D_item = sps.diags(degree_item, format="csr")
        # D_item = D_item.power(-1 / 2)
        # urm_tilda = D_user * urm  # * D_item
        # sim = urm_tilda.T * urm
        sim = cosine_similarity(urm.T, urm.T, dense_output=False)

        if self.topk is not None:
            sim = truncate_top_k_2(sim.todense(), k=self.topk)

        sparse_sim = sps.csr_matrix(sim)
        return sparse_sim


# class EASE(ItemSimilarityMatrixRecommender):
#     def __init__(self, train_data, l2):
#         super().__init__(train_data=train_data)
#         self.l2 = l2
#
#     def compute_similarity_matrix(self):
#         rows = self.train_data[DEFAULT_USER_COL].values
#         cols = self.train_data[DEFAULT_ITEM_COL].values
#         data = np.ones(len(rows))
#         urm = sps.coo_matrix(
#             (data, (rows, cols)), shape=(self.user_count, self.item_count)
#         )
#         alpha = 0.75
#         user_count = urm.shape[0]
#         G = ((urm.T * urm).toarray())/urm.shape[0]
#
#         mu = np.diag(G) / user_count
#         variance_times_userCount = np.diag(G) - mu * mu * user_count
#
#         # standardizing the data-matrix G (if alpha=1, then G becomes the correlation matrix)
#         G -= mu[:, None] * (mu * user_count)
#         rescaling = np.power(variance_times_userCount, alpha / 2.0)
#         scaling = 1.0 / rescaling
#         G = scaling[:, None] * G * scaling
#
#         diagIndices = np.diag_indices(G.shape[0])
#         G[diagIndices] += self.l2
#         P = np.linalg.inv(G)#.toarray())
#         B = P / (-np.diag(P))
#         B[diagIndices] = 0
#
#         B = scaling[:, None] * B * rescaling
#         return B
