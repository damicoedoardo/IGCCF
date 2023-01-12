#!/usr/bin/env python
__author__ = "XXX"
__email__ = "XXX"

from tensorflow import keras

import logging
import os
import tensorflow as tf
from constants import *
from representations_based_recommender import RepresentationsBasedRecommender
from utils.decorator_utils import timing
from utils.general_utils import normalize_csr_sparse_matrix, truncate_top_k, truncate_top_k_2
from utils.graph_utils import (
    nxgraph_from_user_item_interaction_df,
    symmetric_normalized_laplacian_matrix,
    urm_from_nxgraph,
)
from sklearn.metrics.pairwise import cosine_similarity

from utils.pandas_utils import remap_column_consecutive
from utils.tensorflow_utils import to_tf_sparse_tensor, tf_dropout_sparse
import pandas as pd
import numpy as np
import scipy.sparse as sps

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL"))


class IGCCF(keras.Model, RepresentationsBasedRecommender):
    """Inductive Graph Convolutional Matrix Factorization

    Attributes:
        train_data (pd.DataFrame): dataframe containing user-item interactions
        embeddings_size (int): dimension of user-item embeddings
        convolution_depth (int): number of convolution step to perform
        edge_dropout (flaot): percentage of edges to drop in the item-item graph
        user_profile_dropout (float): percentage of user profile to drop
        top_k (int): top_k similar items to keep in the item-item graph
    """

    def __init__(
        self,
        train_data,
        embeddings_size,
        convolution_depth,
        user_profile_dropout=0.0,
        top_k=15,
        full_data=None
    ):
        """Inductive Graph Convolutional Matrix Factorization

        Attributes:
            train_data (pd.DataFrame): dataframe containing user-item interactions
            embeddings_size (int): dimension of user-item embeddings
            convolution_depth (int): number of convolution step to perform
            edge_dropout (flaot): percentage of edges to drop in the item-item graph
            user_profile_dropout (float): percentage of user profile to drop
        """
        keras.Model.__init__(self)
        if full_data is not None:
            RepresentationsBasedRecommender.__init__(self, full_data)
        else:
            RepresentationsBasedRecommender.__init__(self, train_data)

        self.embeddings_size = embeddings_size
        self.user_profile_dropout = user_profile_dropout
        self.top_k = top_k
        self.k = convolution_depth

        # create embeddings
        initializer = tf.initializers.GlorotUniform(seed=SEED)
        self.item_embeddings = tf.Variable(
            initializer(shape=[self.item_count, embeddings_size]), trainable=True,
        )

        rows = train_data[DEFAULT_USER_COL].values
        cols = train_data[DEFAULT_ITEM_COL].values
        data = np.ones(len(rows))
        urm = sps.coo_matrix((data, (rows, cols)), shape=(self.user_count, self.item_count))
        self.urm = to_tf_sparse_tensor(urm)

        # create projected adjacency matrix
        proj_item_adjacency = self._build_item_graph_adjacency(urm, top_k, convolution_depth)
        self.S = to_tf_sparse_tensor(proj_item_adjacency)

    @timing
    def _build_item_graph_adjacency(self, urm, top_k, convolution_depth):
        adj = cosine_similarity(urm.T, urm.T, dense_output=True)
        S = truncate_top_k_2(adj, top_k)
        S = sps.csr_matrix(S)
        return S

    def __call__(self, urm, training=True):
        """Return users and items embeddings

        Args:
            urm (tf.SparseTensor): urm as tf sparse tensor
            training (bool): if training

        Returns:
            tf.Variable, tf.Variable: embeddings of users and items
        """
        x = self.item_embeddings
        S = self.S
        # propagation step
        if training:
            urm = tf_dropout_sparse(
                urm, 1 - self.user_profile_dropout, urm.values.get_shape()[0]
            )

        for i in range(self.k):
            x = tf.sparse.sparse_dense_matmul(S, x)

        user_emb = tf.sparse.sparse_dense_matmul(urm, x)

        # Return user and item embeddings
        return user_emb, x

    def compute_representations(self, user_data):
        user_interactions = user_data["interactions"].copy()
        # remap to consecutive user idxs
        mapping_dict = remap_column_consecutive(
            user_interactions, DEFAULT_USER_COL, mapping_dict=True
        )
        user_idxs = list(mapping_dict.keys())

        rows = user_interactions[DEFAULT_USER_COL].values
        cols = user_interactions[DEFAULT_ITEM_COL].values
        data = np.ones(len(rows))

        urm = sps.coo_matrix((data, (rows, cols)), shape=(len(user_idxs), self.item_count))

        urm = to_tf_sparse_tensor(urm)

        logger.info("Computing representations")
        users_emb, items_emb = self(training=False, urm=urm)
        users_repr_df = pd.DataFrame(users_emb.numpy(), index=user_idxs)
        items_repr_df = pd.DataFrame(items_emb.numpy(), index=self.item_idxs)
        logger.info("Representation computed")
        return users_repr_df, items_repr_df
