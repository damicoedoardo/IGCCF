#!/usr/bin/env python
__author__ = "XXX"
__email__ = "XXX"

from tensorflow import keras

import numpy as np
import scipy.sparse as sps
import logging
import os
import tensorflow as tf
from constants import *
from recommender_interface import Recommender
from representations_based_recommender import RepresentationsBasedRecommender
from utils.decorator_utils import timing
from utils.graph_utils import (
    nxgraph_from_user_item_interaction_df,
    symmetric_normalized_laplacian_matrix,
)
from utils.pandas_utils import remap_column_consecutive
from utils.tensorflow_utils import to_tf_sparse_tensor
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL"))


class FISM(keras.Model, Recommender):
    """FISM

    Note:
        paper: https://dl.acm.org/doi/abs/10.1145/2487575.2487589?casa_token=gXExHeRMlqEAAAAA:4wJMJP7xuPxR_crj_tCVxC7NyfbOfKUt9ZaE56bU2X2WtNA-D4onSiZhFh1HURUoQ8Ae2fncw-ZiCw

    Attributes:
        train_data (pd.DataFrame): dataframe containing user-item interactions
        embeddings_size (int): dimension of user-item embeddings
        alpha (float): algorithm-specific parameter
    """

    def __init__(self, train_data, embeddings_size, alpha, full_data=None):
        """FISM

        Note:
            paper: https://dl.acm.org/doi/abs/10.1145/2487575.2487589?casa_token=gXExHeRMlqEAAAAA:4wJMJP7xuPxR_crj_tCVxC7NyfbOfKUt9ZaE56bU2X2WtNA-D4onSiZhFh1HURUoQ8Ae2fncw-ZiCw

        Attributes:
            train_data (pd.DataFrame): dataframe containing user-item interactions
            embeddings_size (int): dimension of user-item embeddings
            alpha (float): algorithm-specific parameter
        """
        keras.Model.__init__(self)
        if full_data is not None:
            Recommender.__init__(self, full_data)
        else:
            Recommender.__init__(self, train_data)

        self.embeddings_size = embeddings_size
        self.alpha = alpha

        rows = train_data[DEFAULT_USER_COL].values
        cols = train_data[DEFAULT_ITEM_COL].values
        data = np.ones(len(rows))
        urm = sps.coo_matrix(
            (data, (rows, cols)), shape=(self.user_count, self.item_count)
        )
        self.urm = to_tf_sparse_tensor(urm)

        diag = np.array(urm.sum(axis=1)).squeeze() ** self.alpha
        self.diag = tf.convert_to_tensor(diag, dtype=tf.float32)

        d_diag = sps.diags(diag, format="csr")
        self.d_diag = to_tf_sparse_tensor(d_diag)

        # create embeddings
        initializer = tf.initializers.GlorotUniform()
        self.u_item_emb = tf.Variable(
            initializer(shape=[self.item_count, embeddings_size]), trainable=True,
        )
        self.item_emb = tf.Variable(
            initializer(shape=[self.item_count, embeddings_size]), trainable=True,
        )
        self.user_biases = tf.Variable(
            initializer(shape=[self.user_count, 1]), trainable=True,
        )
        self.item_biases = tf.Variable(
            initializer(shape=[self.item_count, 1]), trainable=True,
        )

    def __call__(self, urm):
        """Return users and items embeddings

        Returns:
            tf.Variable: embeddings of users and items
        """
        user_repr = tf.sparse.sparse_dense_matmul(urm, self.u_item_emb)

        # sim = tf.matmul(self.u_item_emb, tf.transpose(self.item_emb))
        # scores = tf.sparse.sparse_dense_matmul(self.urm, sim)
        # scores = tf.sparse.sparse_dense_matmul(self.d_diag, scores)
        # scores = scores + self.user_biases + tf.transpose(self.item_biases)
        return user_repr, self.item_emb, self.user_biases, self.item_biases, self.diag

    @tf.function
    def tf_compute_scores(self, urm, d_diag, user_ids):
        user_emb, item_emb, users_biases, items_biases, diag = self(urm)
        # filter user_bias
        filtered_users_biases = tf.gather(users_biases, user_ids)
        a_user_emb = tf.sparse.sparse_dense_matmul(d_diag, user_emb)
        dot = tf.matmul(a_user_emb, tf.transpose(item_emb))
        scores = dot + filtered_users_biases + tf.transpose(items_biases)
        return scores

    @timing
    def compute_items_scores(self, user_data):
        user_interactions = user_data["interactions"]
        # remap to consecutive user idxs
        user_interactions, mapping_dict = remap_column_consecutive(
            user_interactions, DEFAULT_USER_COL, mapping_dict=True, inplace=False
        )
        user_idxs = list(mapping_dict.keys())
        tf_user_idxs = tf.constant(np.array(user_idxs))

        rows = user_interactions[DEFAULT_USER_COL].values
        cols = user_interactions[DEFAULT_ITEM_COL].values
        data = np.ones(len(rows))

        urm = sps.coo_matrix(
            (data, (rows, cols)), shape=(len(user_idxs), self.item_count)
        )

        # create d_diag alpha term
        diag = np.array(urm.sum(axis=1)).squeeze() ** self.alpha
        d_diag = sps.diags(diag, format="csr")
        d_diag = to_tf_sparse_tensor(d_diag)

        urm = to_tf_sparse_tensor(urm)

        scores = self.tf_compute_scores(urm, d_diag, tf_user_idxs).numpy()

        scores = pd.DataFrame(scores, index=user_idxs)
        return scores
