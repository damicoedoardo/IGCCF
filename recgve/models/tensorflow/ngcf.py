#!/usr/bin/env python
__author__ = "XXX"
__email__ = "XXX"

from tensorflow import keras
import scipy.sparse as sps
import logging
import os
import tensorflow as tf
from constants import *
from representations_based_recommender import (
    RepresentationsBasedRecommender,
)
from utils.graph_utils import (
    nxgraph_from_user_item_interaction_df,
    symmetric_normalized_laplacian_matrix,
)
from utils.tensorflow_utils import to_tf_sparse_tensor, tf_dropout_sparse
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL"))


class NGCF(keras.Model, RepresentationsBasedRecommender):
    """Neural Graph Collaborative Filtering

    Note:
        paper: https://arxiv.org/abs/1905.08108

    Attributes:
        train_data (pd.DataFrame): dataframe containing user-item interactions
        embeddings_size (int): dimension of user-item embeddings
        convolution_depth (int): number of convolution step to perform
        mess_dropout (float): message dropout percentage
        node_dropout (float): node dropout percentage
    """

    def __init__(
        self,
        train_data,
        embeddings_size,
        convolution_depth,
        mess_dropout=0.0,
        node_dropout=0.0,
    ):
        """Neural Graph Collaborative Filtering

        Note:
            paper: https://arxiv.org/abs/1905.08108

        Args:
            train_data (pd.DataFrame): dataframe containing user-item interactions
            embeddings_size (int): dimension of user-item embeddings
            convolution_depth (int): number of convolution step to perform
            mess_dropout (float): message dropout percentage
        node_dropout (float): node dropout percentage
        """
        keras.Model.__init__(self)
        RepresentationsBasedRecommender.__init__(self, train_data)

        self.embeddings_size = embeddings_size

        # create embeddings
        initializer = tf.initializers.GlorotUniform()
        self.embeddings = tf.Variable(
            initializer(shape=[self.user_count + self.item_count, embeddings_size]),
            trainable=True,
        )

        self.k = convolution_depth
        self.mess_dropout = mess_dropout
        self.node_dropout = node_dropout

        # Compute propagation matrix
        # S will be (L + I)
        graph = nxgraph_from_user_item_interaction_df(
            train_data, user_col=DEFAULT_USER_COL, item_col=DEFAULT_ITEM_COL
        )
        L = symmetric_normalized_laplacian_matrix(graph, self_loop=False)

        S = L.copy()
        S = S.tolil()
        S.setdiag(1)
        S = S.tocsr()

        self.L = to_tf_sparse_tensor(L)
        self.S = to_tf_sparse_tensor(S)

        # create weight matrices
        self.all_weights = self._create_weights_matrices()

    def _create_weights_matrices(self):
        """ Create weights matrices"""
        weights = dict()
        initializer = tf.initializers.GlorotUniform()
        for i in range(self.k):
            weights["W_gc_{}".format(i)] = tf.Variable(
                initializer(shape=[self.embeddings_size, self.embeddings_size])
            )
            weights["b_gc_{}".format(i)] = tf.Variable(initializer(shape=[1, self.embeddings_size]))

            weights["W_bi_{}".format(i)] = tf.Variable(
                initializer(shape=[self.embeddings_size, self.embeddings_size])
            )
            weights["b_bi_{}".format(i)] = tf.Variable(initializer(shape=[1, self.embeddings_size]))

        return weights

    def __call__(self, training=True):
        x = self.embeddings
        depth_embeddings = [self.embeddings]

        if training:
            # apply node dropout
            S = tf_dropout_sparse(
                self.S, 1 - self.node_dropout, self.S.values.get_shape()[0]
            )
        else:
            S = self.S

        if training:
            # apply node dropout for L_only
            L = tf_dropout_sparse(
                self.L, 1 - self.node_dropout, self.L.values.get_shape()[0]
            )
        else:
            L = self.L

        # propagate the embeddings
        for i in range(self.k):
            # propagate the embeddings
            prop_emb = tf.sparse.sparse_dense_matmul(S, x)
            L_prop_emb = tf.sparse.sparse_dense_matmul(L, x)

            # apply dense layer and non-linearity
            embeddings = (
                tf.matmul(prop_emb, self.all_weights["W_gc_%d" % i])
                + self.all_weights["b_gc_%d" % i]
            )

            # compute dot-product
            bi_embeddings = tf.multiply(x, L_prop_emb)

            # apply dense layer and non-linearity to the dot-product
            bi_embeddings = (
                tf.matmul(bi_embeddings, self.all_weights["W_bi_%d" % i])
                + self.all_weights["b_bi_%d" % i]
            )

            x = tf.nn.leaky_relu(embeddings + bi_embeddings)

            # apply mess_dropout
            if training:
                x = tf.nn.dropout(x, rate=self.mess_dropout)
            norm_embeddings = tf.math.l2_normalize(x, axis=1)

            depth_embeddings += [norm_embeddings]

        # concatenate embeddings at each depth
        all_embeddings = tf.concat(depth_embeddings, 1)
        return all_embeddings

    def compute_representations(self, user_data):
        user_interactions = user_data["interactions"]
        user_id = user_interactions[DEFAULT_USER_COL].unique()
        logger.info("Computing representations")
        embeddings = self(training=False)
        users_emb = tf.gather(embeddings, tf.constant(user_id)).numpy()
        items_emb = tf.gather(
            embeddings, tf.constant(self.items_after_users_idxs)
        ).numpy()
        users_repr_df = pd.DataFrame(users_emb, index=user_id)
        items_repr_df = pd.DataFrame(items_emb, index=self.item_idxs)
        logger.info("Representation computed")
        return users_repr_df, items_repr_df
