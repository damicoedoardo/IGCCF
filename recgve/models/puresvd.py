#!/usr/bin/env python
__author__ = "XXX"
__email__ = "XXX"

from similarity_matrix_recommender import ItemSimilarityMatrixRecommender
from constants import *
import numpy as np
import scipy.sparse as sps
from sklearn.utils.extmath import randomized_svd
from constants import *


class PureSVD(ItemSimilarityMatrixRecommender):
    """PureSVD
    """

    def __init__(self, train_data, n_components=10, full_data=None):
        """PureSVD
        """
        super().__init__(train_data=train_data)
        if full_data is not None:
            self.user_count = len(full_data["userID"].unique())
            self.item_count = len(full_data["itemID"].unique())
        self.n_components = n_components

    def compute_similarity_matrix(self):
        rows = self.train_data[DEFAULT_USER_COL].values
        cols = self.train_data[DEFAULT_ITEM_COL].values
        data = np.ones(len(rows))
        urm = sps.coo_matrix(
            (data, (rows, cols)), shape=(self.user_count, self.item_count)
        )

        U, Sigma, Q_T = randomized_svd(
            urm, n_components=self.n_components, random_state=SEED
        )

        sim = np.matmul(np.transpose(Q_T), Q_T)
        return sim
